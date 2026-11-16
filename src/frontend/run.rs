//! `macgpu run -- CMD ARGS...`: like `time`, for the GPU.
//!
//! Spawns the command with inherited stdio, tracks its process tree
//! (descendants via their parent PID, including processes started later),
//! samples at a short interval, and accumulates the tree's GPU time, share,
//! energy, memory, and the time the GPU spent under a limiter state. The
//! summary goes to stderr so stdout stays the command's; the exit status is
//! the command's.
//!
//! GPU time is the driver's `AppUsage` time, which apportions busy time among
//! work running at the same moment: all processes together reach about one
//! second per second when the GPU is saturated. The command's GPU time is its
//! share of the GPU's busy time, compared with wall time in the summary; when
//! other apps use the GPU concurrently it is less than the time the command had
//! work in flight.
//!
//! Accuracy: per-process GPU time comes from the driver's per-client counters,
//! which vanish when a process exits. GPU time between a process's last sample
//! and its exit is therefore not measured. To keep that small, `run` samples
//! every 100 ms by default and takes a final sample as soon as the command
//! exits. The command shares macgpu's resource coalition (an unprivileged
//! process cannot create one: `coalition_create` and
//! `posix_spawnattr_setcoalition_np` both return EPERM), so the coalition's
//! unattributed GPU time in windows where a tracked process exited is
//! reported separately as an estimate of the lost time. That time is the
//! coalition's in-flight time, not an apportioned share, so it is an upper
//! estimate.

use super::signals;

mod summary;
use crate::analyze::{AnalyzeOptions, Analyzer};
use crate::cli::{SamplingArgs, SourceName, parse_interval};
use crate::model::Report;
use crate::snapshot::Sampler;
use crate::sys::process;
use anyhow::{Context, Result};
use clap::Args;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{OsString, c_void};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, ExitStatus};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Args, Debug, Clone, Default)]
#[command(after_help = "\
The summary goes to stderr; the command's stdout is untouched. macgpu exits with the
command's status (128 + N if a signal killed it) and forwards SIGINT, SIGTERM, SIGHUP,
and SIGQUIT sent to macgpu (a Ctrl-C on the terminal already reaches the command).

GPU time of a process between its last sample and its exit is not measured, because
the driver's per-client counters disappear with the process; a shorter --interval
loses less. The command shares the terminal's app coalition, so energy is the app's
measured GPU energy apportioned by GPU time (or whole-GPU energy apportioned by GPU
time when per-app energy is unavailable).

Examples:
  macgpu run -- python train.py --epochs 1
  macgpu run --json -o gpu.json -- ./benchmark")]
pub struct RunArgs {
    /// Sampling window in seconds. Shorter windows lose less GPU time of
    /// processes that exit between samples.
    #[arg(short, long, value_name = "SECONDS", default_value_t = 0.1, value_parser = parse_interval)]
    pub interval: f64,

    /// Data sources to turn off, comma-separated.
    #[arg(long, value_delimiter = ',', value_name = "SOURCE")]
    pub disable: Vec<SourceName>,

    /// Print the result as JSON on stderr instead of the summary.
    #[arg(short, long)]
    pub json: bool,

    /// Write the JSON result to FILE (the summary still goes to stderr).
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Do not print the summary.
    #[arg(short, long)]
    pub quiet: bool,

    /// The command to run, after `--`.
    #[arg(
        required = true,
        trailing_var_arg = true,
        allow_hyphen_values = true,
        value_name = "COMMAND"
    )]
    pub command: Vec<OsString>,
}

/// Signals forwarded to the command.
const FORWARDED: [i32; 4] = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT];

/// How often to look for new descendants and signals between samples.
const POLL: Duration = Duration::from_millis(20);

pub fn run(args: &RunArgs) -> Result<i32> {
    let sources = SamplingArgs {
        interval: Some(args.interval),
        disable: args.disable.clone(),
    }
    .sources();
    let interval = Duration::from_secs_f64(args.interval);
    let mut analyzer = Analyzer::new(AnalyzeOptions { include_idle: true });

    signals::install(&FORWARDED);
    let mut sampler = Sampler::new(sources);
    let mut previous = sampler.snapshot();
    let own_coalition = coalition_of(std::process::id() as i32);

    let program = &args.command[0];
    let started = Instant::now();
    let mut child = match Command::new(program).args(&args.command[1..]).spawn() {
        Ok(child) => child,
        Err(error) => {
            eprintln!("macgpu: cannot run {}: {error}", program.to_string_lossy());
            return Ok(match error.kind() {
                io::ErrorKind::NotFound => 127,
                _ => 126,
            });
        }
    };
    let root = child.id() as i32;
    // The child cannot be reaped before the waiter thread starts, so its start
    // time is readable even if it already exited.
    let mut tree = ProcessTree::new(root);
    let (exited_sender, exited) = mpsc::channel();
    thread::Builder::new()
        .name("macgpu-run-wait".into())
        .spawn(move || {
            let _ = exited_sender.send(child.wait());
        })
        .context("spawn wait thread")?;

    let mut measurement = Measurement::new(own_coalition);
    let mut due = previous.at + interval;
    let status: io::Result<ExitStatus> = loop {
        if let Some(received) = signals::take()
            && received.sender != 0
            && tree.is_alive(root)
        {
            // Signals from the terminal (sender 0) already reach the command,
            // which is in the same foreground process group.
            // SAFETY: `root` was started by this process and is checked alive;
            // any received signal number is valid for kill(2).
            unsafe { libc::kill(root, received.signal) };
        }
        tree.refresh();
        let now = Instant::now();
        let wait = due.saturating_duration_since(now).min(POLL);
        match exited.recv_timeout(wait) {
            Ok(status) => {
                let wall = started.elapsed();
                tree.refresh();
                let current = sampler.snapshot();
                let report = analyzer.analyze(&previous, &current);
                measurement.add(&report, &mut tree);
                measurement.wall_seconds = wall.as_secs_f64();
                break status;
            }
            Err(RecvTimeoutError::Timeout) => {
                if Instant::now() >= due {
                    let current = sampler.snapshot();
                    let report = analyzer.analyze(&previous, &current);
                    previous = current;
                    measurement.add(&report, &mut tree);
                    due += interval;
                    let now = Instant::now();
                    if due <= now {
                        due = now + interval;
                    }
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                break Err(io::Error::other("lost track of the command"));
            }
        }
    };
    let status = status.context("waiting for the command")?;
    let code = exit_code(status);

    let summary = measurement.summary(args, interval, status, tree.len(), tree.alive_descendants());
    if let Some(path) = &args.output {
        let json = serde_json::to_string_pretty(&summary)?;
        std::fs::write(path, json + "\n")
            .with_context(|| format!("cannot write {}", path.display()))?;
    }
    if !args.quiet {
        let text = if args.json {
            serde_json::to_string_pretty(&summary)? + "\n"
        } else {
            summary.render()
        };
        let _ = io::stderr().write_all(text.as_bytes());
    }
    Ok(code)
}

/// The shell convention: the exit code, or 128 + the signal that killed it.
fn exit_code(status: ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(code), _) => code,
        (None, Some(signal)) => 128 + signal,
        (None, None) => 1,
    }
}

/// The resource coalition of `pid` (`PROC_PIDCOALITIONINFO`, index 0).
fn coalition_of(pid: i32) -> Option<u64> {
    #[repr(C)]
    struct CoalitionInfo {
        ids: [u64; 2],
        reserved: [u64; 3],
    }
    const PROC_PIDCOALITIONINFO: i32 = 20;
    let mut info = CoalitionInfo {
        ids: [0; 2],
        reserved: [0; 3],
    };
    let size = size_of::<CoalitionInfo>() as i32;
    // SAFETY: `info` is a correctly sized, writable C-compatible buffer and
    // `size` is its byte length, which is what proc_pidinfo expects.
    let written = unsafe {
        libc::proc_pidinfo(
            pid,
            PROC_PIDCOALITIONINFO,
            0,
            (&mut info as *mut CoalitionInfo).cast::<c_void>(),
            size,
        )
    };
    (written == size && info.ids[0] != 0).then_some(info.ids[0])
}

/// Direct children of `pid`.
fn child_pids(pid: i32) -> Vec<i32> {
    let mut capacity = 64usize;
    loop {
        let mut pids = vec![0i32; capacity];
        // SAFETY: `pids` has room for `capacity` i32s, matching the byte size
        // passed to proc_listchildpids.
        let count = unsafe {
            libc::proc_listchildpids(
                pid,
                pids.as_mut_ptr().cast::<c_void>(),
                (capacity * size_of::<i32>()) as i32,
            )
        };
        if count <= 0 {
            return Vec::new();
        }
        let count = count as usize;
        if count < capacity {
            pids.truncate(count);
            pids.retain(|pid| *pid > 0);
            return pids;
        }
        capacity *= 4;
    }
}

/// The command's process tree: every process seen in it, keyed by PID and
/// start time so a reused PID is not mistaken for a member.
///
/// When a process exits, its children are reparented to launchd, so the tree
/// is refreshed every [`POLL`]: a grandchild is tracked as long as it was seen
/// before its parent exited.
pub struct ProcessTree {
    members: HashMap<i32, u64>,
    alive: Vec<i32>,
    root: i32,
    /// Members that exited since the last [`ProcessTree::take_exits`].
    exits: usize,
}

impl ProcessTree {
    pub fn new(root: i32) -> Self {
        let mut tree = Self {
            members: HashMap::new(),
            alive: Vec::new(),
            root,
            exits: 0,
        };
        if let Some(start_us) = process::start_time_us(root) {
            tree.members.insert(root, start_us);
            tree.alive.push(root);
        }
        tree.refresh();
        tree
    }

    /// Notes exited members and adds new descendants.
    pub fn refresh(&mut self) {
        let members = &self.members;
        let before = self.alive.len();
        self.alive
            .retain(|pid| process::start_time_us(*pid) == members.get(pid).copied());
        self.exits += before - self.alive.len();

        let mut queue = self.alive.clone();
        while let Some(parent) = queue.pop() {
            for child in child_pids(parent) {
                let Some(start_us) = process::start_time_us(child) else {
                    continue;
                };
                if self.members.get(&child) != Some(&start_us) {
                    self.members.insert(child, start_us);
                    self.alive.push(child);
                    queue.push(child);
                }
            }
        }
    }

    pub fn contains(&self, pid: i32, start_us: u64) -> bool {
        self.members.get(&pid) == Some(&start_us)
    }

    /// Adds `pid` if its parent is a live member (a GPU process the poll missed).
    pub fn adopt(&mut self, pid: i32, start_us: u64, ppid: i32) {
        if !self.contains(pid, start_us) && self.alive.contains(&ppid) {
            self.members.insert(pid, start_us);
            self.alive.push(pid);
        }
    }

    /// True while `pid` is a member and still the same process.
    pub fn is_alive(&self, pid: i32) -> bool {
        self.members
            .get(&pid)
            .is_some_and(|start_us| process::start_time_us(pid) == Some(*start_us))
    }

    pub fn take_exits(&mut self) -> usize {
        std::mem::take(&mut self.exits)
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Live members other than the root.
    pub fn alive_descendants(&self) -> usize {
        self.alive.iter().filter(|pid| **pid != self.root).count()
    }
}

/// Accumulates the tracked processes' usage over the run's windows.
#[derive(Debug, Default)]
pub struct Measurement {
    own_coalition: Option<u64>,
    pub wall_seconds: f64,
    windows: usize,
    gpu_ns: u64,
    peak_ms_per_s: f64,
    share_seconds: f64,
    peak_share_pct: f64,
    device_seconds: Option<f64>,
    peak_device_pct: Option<f64>,
    energy_j: Option<f64>,
    energy_from_app: bool,
    energy_from_device: bool,
    /// Windows with tracked GPU time but no way to estimate its energy.
    energy_missing_windows: usize,
    device_energy_j: Option<f64>,
    peak_memory_bytes: Option<u64>,
    limited_seconds: Option<f64>,
    thermal_limited_seconds: Option<f64>,
    exit_estimate_ns: u64,
    /// Windows with an exit whose lost time the coalition could not separate
    /// from overlapping GPU work.
    exit_estimate_unknown: usize,
    processes: BTreeMap<(i32, u64), ProcessTotals>,
}

#[derive(Debug, Clone, Default)]
struct ProcessTotals {
    name: String,
    gpu_ns: u64,
    peak_memory_bytes: Option<u64>,
    energy_j: Option<f64>,
}

impl Measurement {
    pub fn new(own_coalition: Option<u64>) -> Self {
        Self {
            own_coalition,
            ..Self::default()
        }
    }

    pub fn add(&mut self, report: &Report, tree: &mut ProcessTree) {
        for process in &report.processes {
            if let Some(ppid) = process.ppid {
                tree.adopt(process.pid, process.start_us, ppid);
            }
        }
        self.windows += 1;
        let window = report.window_seconds.max(0.0);
        let window_ns = (window * 1e9) as u64;

        let mut gpu_ns = 0u64;
        let mut share = 0.0;
        let mut device: Option<f64> = None;
        let mut memory: Option<u64> = None;
        let mut energy = 0.0;
        let mut energy_known = false;
        let mut energy_missing = false;
        for process in report
            .processes
            .iter()
            .filter(|process| tree.contains(process.pid, process.start_us))
        {
            gpu_ns += process.gpu_ns;
            share += process.share_pct;
            if let Some(pct) = process.device_pct {
                *device.get_or_insert(0.0) += pct;
            }
            if let Some(bytes) = process.graphics_memory_bytes {
                *memory.get_or_insert(0) += bytes;
            }

            let process_energy = if let Some(power) = process.est_power_w {
                self.energy_from_app = true;
                Some(power * window)
            } else if process.gpu_ns == 0 {
                None
            } else if let (Some(power), true) =
                (report.device.power_w, report.totals.attributed_gpu_ns > 0)
            {
                self.energy_from_device = true;
                Some(
                    power * window * process.gpu_ns as f64 / report.totals.attributed_gpu_ns as f64,
                )
            } else {
                energy_missing = true;
                None
            };
            if let Some(joules) = process_energy {
                energy += joules;
                energy_known = true;
            }

            let totals = self
                .processes
                .entry((process.pid, process.start_us))
                .or_default();
            totals.name.clone_from(&process.name);
            totals.gpu_ns += process.gpu_ns;
            if let Some(bytes) = process.graphics_memory_bytes {
                totals.peak_memory_bytes = Some(totals.peak_memory_bytes.unwrap_or(0).max(bytes));
            }
            if let Some(joules) = process_energy.filter(|joules| *joules > 0.0) {
                *totals.energy_j.get_or_insert(0.0) += joules;
            }
        }

        self.gpu_ns += gpu_ns;
        if window > 0.0 {
            self.peak_ms_per_s = self.peak_ms_per_s.max(gpu_ns as f64 / 1e6 / window);
        }
        self.share_seconds += share / 100.0 * window;
        self.peak_share_pct = self.peak_share_pct.max(share);
        if report.device.active_pct.is_some() {
            let pct = device.unwrap_or(0.0);
            *self.device_seconds.get_or_insert(0.0) += pct / 100.0 * window;
            self.peak_device_pct = Some(self.peak_device_pct.unwrap_or(0.0).max(pct));
        }
        if energy_known {
            *self.energy_j.get_or_insert(0.0) += energy;
        }
        if energy_missing {
            self.energy_missing_windows += 1;
        }
        if let Some(power) = report.device.power_w {
            *self.device_energy_j.get_or_insert(0.0) += power * window;
        }
        if let Some(bytes) = memory {
            self.peak_memory_bytes = Some(self.peak_memory_bytes.unwrap_or(0).max(bytes));
        }
        if let Some(pct) = report.device.limited_pct() {
            *self.limited_seconds.get_or_insert(0.0) += pct / 100.0 * window;
        }
        if let Some(pct) = report.device.thermal_limited_pct {
            *self.thermal_limited_seconds.get_or_insert(0.0) += pct / 100.0 * window;
        }

        // A member exited in this window: GPU time it used after the previous
        // sample shows up only as the coalition's unattributed (in-flight)
        // time. The coalition has work in flight at most the whole window.
        let exits = tree.take_exits();
        if exits > 0
            && let Some(app) = self
                .own_coalition
                .and_then(|id| report.apps.iter().find(|app| app.coalition_id == id))
        {
            match app.unattributed_gpu_ns {
                Some(unattributed) => self.exit_estimate_ns += unattributed.min(window_ns),
                None => self.exit_estimate_unknown += 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AppView, DeviceView, ProcessView, StateShare, Totals};
    use std::os::unix::process::ExitStatusExt;

    #[test]
    fn tree_tracks_descendants_started_later() {
        // sh starts a grandchild after a delay; the tree must find it.
        let mut child = Command::new("/bin/sh")
            .args(["-c", "sleep 0.3; /bin/sleep 1 & wait"])
            .spawn()
            .unwrap();
        let root = child.id() as i32;
        let mut tree = ProcessTree::new(root);
        assert_eq!(tree.len(), 1);
        let deadline = Instant::now() + Duration::from_secs(3);
        while tree.len() < 3 && Instant::now() < deadline {
            thread::sleep(POLL);
            tree.refresh();
        }
        assert!(tree.len() >= 3, "members {}", tree.len());
        assert!(tree.alive_descendants() >= 1);
        child.wait().unwrap();
        tree.refresh();
        assert!(tree.take_exits() >= 1);
        assert_eq!(tree.take_exits(), 0);
        assert!(!tree.contains(root, 0));
        assert!(!tree.is_empty());
    }

    #[test]
    fn reads_own_coalition_and_children() {
        assert!(coalition_of(std::process::id() as i32).is_some());
        let mut child = Command::new("/bin/sleep").arg("1").spawn().unwrap();
        let children = child_pids(std::process::id() as i32);
        assert!(children.contains(&(child.id() as i32)), "{children:?}");
        child.kill().unwrap();
        child.wait().unwrap();
    }

    #[test]
    fn exit_codes_follow_the_shell() {
        assert_eq!(exit_code(ExitStatus::from_raw(0)), 0);
        assert_eq!(exit_code(ExitStatus::from_raw(3 << 8)), 3);
        assert_eq!(
            exit_code(ExitStatus::from_raw(libc::SIGKILL)),
            128 + libc::SIGKILL
        );
    }

    fn tree_with(pid: i32, start_us: u64) -> ProcessTree {
        let mut tree = ProcessTree {
            members: HashMap::new(),
            alive: Vec::new(),
            root: pid,
            exits: 0,
        };
        tree.members.insert(pid, start_us);
        tree.alive.push(pid);
        tree
    }

    fn process(pid: i32, gpu_ns: u64) -> ProcessView {
        ProcessView {
            pid,
            start_us: 1,
            ppid: Some(1),
            name: format!("p{pid}"),
            gpu_ns,
            share_pct: 50.0,
            queues: 1,
            ..ProcessView::default()
        }
    }

    fn report(processes: Vec<ProcessView>) -> Report {
        Report {
            window_seconds: 0.5,
            device: DeviceView {
                active_pct: Some(80.0),
                power_w: Some(10.0),
                limiters: vec![StateShare {
                    name: "PWR_ZONE".into(),
                    pct: 50.0,
                }],
                thermal_limited_pct: Some(10.0),
                ..DeviceView::default()
            },
            totals: Totals {
                attributed_gpu_ns: processes.iter().map(|p| p.gpu_ns).sum(),
                ..Totals::default()
            },
            processes,
            ..Report::default()
        }
    }

    fn args() -> RunArgs {
        RunArgs {
            interval: 0.5,
            command: vec!["bench".into(), "--fast".into()],
            ..RunArgs::default()
        }
    }

    #[test]
    fn accumulates_only_the_tree() {
        let mut tree = tree_with(100, 1);
        let mut measurement = Measurement::new(Some(9));

        // The tree's process with estimated power, an outsider, and a child
        // the poll missed (adopted via its ppid).
        let mut ours = process(100, 250_000_000);
        ours.est_power_w = Some(6.0);
        ours.device_pct = Some(40.0);
        ours.graphics_memory_bytes = Some(1 << 30);
        let outsider = process(200, 250_000_000);
        let mut missed = process(101, 0);
        missed.ppid = Some(100);
        missed.graphics_memory_bytes = Some(1 << 20);
        measurement.add(
            &report(vec![ours.clone(), outsider.clone(), missed]),
            &mut tree,
        );
        assert!(tree.contains(101, 1));

        // Second window: no per-app estimate, so whole-GPU power apportioned.
        let mut second = ours.clone();
        second.est_power_w = None;
        second.gpu_ns = 500_000_000;
        second.device_pct = Some(80.0);
        measurement.add(&report(vec![second, outsider]), &mut tree);
        measurement.wall_seconds = 1.0;

        let summary = measurement.summary(
            &args(),
            Duration::from_millis(500),
            ExitStatus::from_raw(0),
            2,
            0,
        );
        assert!((summary.gpu_seconds - 0.75).abs() < 1e-9);
        assert!((summary.avg_busy_ms_per_s - 750.0).abs() < 1e-9);
        assert!((summary.peak_busy_ms_per_s - 1000.0).abs() < 1e-9);
        // 40% then 80% of the device over two half-second windows.
        assert!((summary.avg_device_pct.unwrap() - 60.0).abs() < 1e-9);
        assert_eq!(summary.peak_device_pct, Some(80.0));
        // 6 W * 0.5 s + 10 W * 0.5 s * (500 / 750).
        let expected = 3.0 + 10.0 * 0.5 * (500.0 / 750.0);
        assert!(
            (summary.energy_j.unwrap() - expected).abs() < 1e-9,
            "{:?}",
            summary.energy_j
        );
        assert_eq!(summary.energy_method, Some("mixed"));
        assert!(summary.energy_complete);
        assert_eq!(summary.device_energy_j, Some(10.0));
        assert_eq!(
            summary.peak_graphics_memory_bytes,
            Some((1 << 30) + (1 << 20))
        );
        // PWR_ZONE 50% of powered time, powered 80% of each 0.5 s window.
        assert!((summary.limited_seconds.unwrap() - 0.4).abs() < 1e-9);
        assert_eq!(summary.thermal_limited_seconds, Some(0.1));
        assert_eq!(summary.gpu_processes.len(), 2);
        assert_eq!(summary.gpu_processes[0].pid, 100);
        assert_eq!(summary.exit_code, Some(0));
        assert_eq!(summary.exit_estimate_gpu_seconds, None);

        let text = summary.render();
        assert!(
            text.contains("`bench --fast` exited with status 0"),
            "{text}"
        );
        assert!(
            text.contains(
                "GPU time      0.750 s apportioned share = 75.0% of wall time  (avg 750.0 ms/s, peak 1000.0 ms/s)"
            ),
            "{text}"
        );
        assert!(
            text.contains("device share  avg 60.0%, peak 80.0%"),
            "{text}"
        );
        assert!(text.contains("GPU memory    peak 1.0 GiB"), "{text}");
        assert!(text.contains("p100 (100) 0.750 s"), "{text}");
        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["gpu_processes"][0]["name"], "p100");
        assert!(json.get("exit_estimate_gpu_seconds").is_none());
    }

    #[test]
    fn estimates_time_lost_at_exit_from_the_coalition() {
        let mut tree = tree_with(100, 1);
        let mut measurement = Measurement::new(Some(9));
        measurement.add(&report(vec![process(100, 100_000_000)]), &mut tree);
        // The process exits; its coalition shows unattributed time.
        tree.exits = 1;
        let mut last = report(vec![]);
        last.apps = vec![AppView {
            coalition_id: 9,
            unattributed_gpu_ns: Some(80_000_000),
            ..AppView::default()
        }];
        last.device.power_w = None;
        measurement.add(&last, &mut tree);
        measurement.wall_seconds = 1.0;
        let summary = measurement.summary(
            &args(),
            Duration::from_millis(500),
            ExitStatus::from_raw(libc::SIGTERM),
            1,
            2,
        );
        assert_eq!(summary.exit_estimate_gpu_seconds, Some(0.08));
        assert_eq!(summary.signal, Some(libc::SIGTERM));
        let text = summary.render();
        assert!(text.contains("was killed by signal 15"), "{text}");
        assert!(
            text.contains("+ up to 0.080 s in flight after a process's last sample"),
            "{text}"
        );
        assert!(
            text.contains("2 processes started by the command were still running"),
            "{text}"
        );

        // Unattributed time inseparable from overlap: noted, not guessed.
        let mut tree = tree_with(100, 1);
        let mut measurement = Measurement::new(Some(9));
        tree.exits = 1;
        last.apps[0].unattributed_gpu_ns = None;
        measurement.add(&last, &mut tree);
        let summary = measurement.summary(
            &args(),
            Duration::from_millis(500),
            ExitStatus::from_raw(0),
            1,
            0,
        );
        assert_eq!(summary.exit_estimate_gpu_seconds, None);
        assert!(
            summary
                .notes
                .iter()
                .any(|note| note.contains("could not be estimated"))
        );
    }

    #[test]
    fn reports_unavailable_sources() {
        let mut tree = tree_with(100, 1);
        let mut measurement = Measurement::new(None);
        let mut bare = report(vec![process(100, 100_000_000)]);
        bare.device = DeviceView::default();
        measurement.add(&bare, &mut tree);
        measurement.wall_seconds = 0.5;
        let summary = measurement.summary(
            &args(),
            Duration::from_millis(500),
            ExitStatus::from_raw(0),
            2,
            0,
        );
        assert_eq!(summary.energy_j, None);
        assert!(!summary.energy_complete);
        assert_eq!(summary.avg_device_pct, None);
        let text = summary.render();
        assert!(text.contains("GPU share     avg 50.0%"), "{text}");
        assert!(text.contains("GPU energy    unavailable"), "{text}");
        assert!(text.contains("GPU memory    unavailable"), "{text}");
        assert!(text.contains("limiter       unavailable"), "{text}");
        assert!(text.contains("energy is incomplete: 1 windows"), "{text}");
    }
}
