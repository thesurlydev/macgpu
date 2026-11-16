//! Controlled-load checks: runs a Metal load in a child process and verifies
//! macgpu's readings against its own measurements.

use super::{
    BUSY_BASELINE_PCT, COALITION_SKEW_NS, Check, DoctorArgs, ENERGY_SETTLE, ENERGY_TOLERANCE_PCT,
    IDLE_WINDOW, LOAD_ALLOC_MIB, LoadReport, RESIDENCY_TOLERANCE_PTS, Report, Sampler, Snapshot,
    Status, WindowSummary, analyze, source, source_ok,
};
use crate::analyze::AnalyzeOptions;
use crate::format;
use crate::metal_load::{
    AttributionBounds, LoadConfig, LoadProcess, LoadResult, OVER_TOLERANCE_PCT, QUIET_SHARE,
    SLACK_NS, TOLERANCE_PCT, other_gpu_ns,
};
use crate::model::ProcessView;
use crate::sys::powermetrics::{self, PowerStats};
use anyhow::{Context, Result};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::time::Duration;

/// Set by a caller that already holds the [`GpuLoadLock`] when it runs
/// `macgpu doctor --load`, so the doctor does not wait for itself.
pub const GPU_LOCK_HELD_ENV: &str = "MACGPU_GPU_LOCK_HELD";

/// An exclusive advisory lock (`flock`) that serializes controlled GPU loads
/// across processes: `doctor --load` and the end-to-end tests. It lives in the
/// per-user temporary directory, and the kernel releases it when the holder
/// exits, however it exits.
pub struct GpuLoadLock {
    /// Holding the open file holds the lock; closing it releases the lock.
    _file: File,
}

impl GpuLoadLock {
    pub fn path() -> PathBuf {
        std::env::temp_dir().join("macgpu-gpu-load.lock")
    }

    /// Blocks until the lock is free.
    pub fn acquire() -> io::Result<Self> {
        Self::lock(libc::LOCK_EX).map(|lock| lock.expect("blocking flock returned"))
    }

    /// Like [`Self::acquire`], but says on stderr when it has to wait.
    pub fn acquire_or_wait() -> io::Result<Self> {
        if let Some(lock) = Self::lock(libc::LOCK_EX | libc::LOCK_NB)? {
            return Ok(lock);
        }
        eprintln!(
            "waiting for another macgpu GPU load to finish (lock {})",
            Self::path().display()
        );
        Self::acquire()
    }

    /// `None` when `LOCK_NB` is set and another process holds the lock.
    fn lock(operation: i32) -> io::Result<Option<Self>> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(Self::path())?;
        loop {
            // SAFETY: `file` is open, so its file descriptor is valid for flock.
            if unsafe { libc::flock(file.as_raw_fd(), operation) } == 0 {
                return Ok(Some(Self { _file: file }));
            }
            let error = io::Error::last_os_error();
            match error.kind() {
                io::ErrorKind::Interrupted => continue,
                io::ErrorKind::WouldBlock => return Ok(None),
                _ => return Err(error),
            }
        }
    }
}

pub(crate) fn run_load(
    args: &DoctorArgs,
    sampler: &mut Sampler,
    checks: &mut Vec<Check>,
) -> Result<LoadReport> {
    let exe = std::env::current_exe().context("cannot find the macgpu executable")?;
    let config = LoadConfig {
        seconds: args.load_seconds,
        alloc_mib: LOAD_ALLOC_MIB,
        wait: true,
        hold: true,
        ..LoadConfig::default()
    };
    let options = AnalyzeOptions { include_idle: true };
    let mut child = LoadProcess::spawn(&exe, &config)?;
    let pid = child.pid();

    let idle_start = sampler.snapshot();
    std::thread::sleep(IDLE_WINDOW);
    let load_start = sampler.snapshot();
    child.start()?;

    // powermetrics samples most of the load, ending before it does.
    let powermetrics_window = Duration::from_secs_f64((args.load_seconds - 1.0).max(0.5));
    let powermetrics = if powermetrics::sudo_available() {
        Some(
            powermetrics::start(powermetrics_window).and_then(|sampling| {
                let started = sampler.snapshot();
                let stats = sampling.finish()?;
                Ok((stats, started, sampler.snapshot()))
            }),
        )
    } else {
        None
    };

    let measured = child.wait_done()?;
    let load_end = sampler.snapshot();
    let idle = analyze(&idle_start, &load_start, &options);
    // Coalition energy is credited late; give it time to catch up.
    let settled = if source_ok(&idle, source::COALITIONS) {
        std::thread::sleep(ENERGY_SETTLE);
        Some(sampler.snapshot())
    } else {
        None
    };
    child.finish()?;

    let load = analyze(&load_start, &load_end, &options);
    // The app rows' energy is scaled to IOReport's when it is known, so the
    // check reads the raw coalition counters.
    let settled = settled.map(|settled| {
        (
            analyze(&load_start, &settled, &options),
            coalition_gpu_energy_j(&load_start, &settled),
        )
    });
    let process = load.processes.iter().find(|process| process.pid == pid);

    checks.push(attribution_check(
        process.map(|process| process.gpu_ns),
        &measured,
        other_gpu_ns(&load, pid),
        (load.window_seconds * 1e9) as u64,
    ));
    checks.push(conservation_check(&load));
    device_checks(&idle, &load, &measured, checks);
    load_coalition_checks(
        &idle,
        &load,
        settled.as_ref().map(|(report, energy)| (report, *energy)),
        process,
        &measured,
        checks,
    );
    checks.push(graphics_memory_check(&idle, pid, measured.alloc_bytes));
    checks.push(temperature_change_check(&idle, &load));
    let powermetrics = match powermetrics {
        None => {
            checks.push(Check::new(
                "load",
                "powermetrics",
                Status::Skip,
                "sudo needs a password; run `sudo -v` first to include this cross-check",
            ));
            None
        }
        Some(Err(error)) => {
            checks.push(Check::new(
                "load",
                "powermetrics",
                Status::Warn,
                format!("{error:#}"),
            ));
            None
        }
        Some(Ok((stats, started, ended))) => {
            let window = analyze(&started, &ended, &options);
            checks.extend(powermetrics_checks(&stats, &window));
            Some(stats)
        }
    };

    Ok(LoadReport {
        pid,
        config,
        idle: WindowSummary::of(&idle, pid),
        load: WindowSummary::of(&load, pid),
        measured,
        powermetrics,
    })
}

/// Compares macgpu's per-process GPU time for the load with the load's own
/// command-buffer timing, allowing for other processes' GPU work as
/// [`AttributionBounds`] describes.
pub(crate) fn attribution_check(
    measured_ns: Option<u64>,
    truth: &LoadResult,
    others_ns: u64,
    window_ns: u64,
) -> Check {
    let scope = "load";
    let Some(measured_ns) = measured_ns else {
        return Check::new(
            scope,
            "attribution",
            Status::Fail,
            "the load process had no GPU time in macgpu's report",
        );
    };
    let bounds = AttributionBounds::new(truth.gpu_ns, truth.busy_fraction(), others_ns, window_ns);
    let error = (measured_ns as f64 - truth.gpu_ns as f64) / truth.gpu_ns.max(1) as f64 * 100.0;
    let mut detail = format!(
        "macgpu {:.1} ms vs load's own {:.1} ms ({error:+.2}%)",
        measured_ns as f64 / 1e6,
        truth.gpu_ns as f64 / 1e6,
    );
    if !bounds.quiet {
        let _ = write!(
            detail,
            "; other processes used {:.1} ms, which stretches the load's own timing",
            others_ns as f64 / 1e6
        );
    }
    let status = if !bounds.contains(measured_ns) {
        let _ = write!(
            detail,
            "; expected {:.1}-{:.1} ms",
            bounds.lower_ns as f64 / 1e6,
            bounds.upper_ns as f64 / 1e6
        );
        Status::Fail
    } else if bounds.quiet && error.abs() > TOLERANCE_PCT {
        let _ = write!(detail, "; more than {TOLERANCE_PCT}% off on a quiet GPU");
        Status::Warn
    } else {
        Status::Pass
    };
    Check::new(scope, "attribution", status, detail)
}

/// `AppUsage` apportions the GPU among concurrent contexts, so summed over
/// every process it cannot exceed the window, nor (with IOReport) the GPU's
/// active time. On an M4 Max the saturated sum was 956-982 ms/s.
fn conservation_check(load: &Report) -> Check {
    let window_ns = load.window_seconds * 1e9;
    let total = load.totals.attributed_gpu_ns as f64;
    let ceiling = match load.device.active_pct {
        Some(active) => window_ns * active / 100.0,
        None => window_ns,
    };
    let allowed = ceiling * (1.0 + TOLERANCE_PCT / 100.0) + SLACK_NS as f64;
    Check::new(
        "load",
        "AppUsage total",
        if total <= allowed {
            Status::Pass
        } else {
            Status::Fail
        },
        format!(
            "all processes {:.0} ms/s; GPU active {:.0} ms/s",
            total / load.window_seconds / 1e6,
            ceiling / load.window_seconds / 1e6
        ),
    )
}

fn device_checks(idle: &Report, load: &Report, measured: &LoadResult, checks: &mut Vec<Check>) {
    let scope = "load";
    if !source_ok(load, source::IOREPORT) {
        checks.push(Check::new(
            scope,
            "device activity",
            Status::Skip,
            "IOReport is not available",
        ));
        return;
    }
    // The load alone kept the GPU busy for this share of the window.
    let busy_pct = measured.gpu_ns as f64 / (load.window_seconds * 1e9) * 100.0;
    // Another process loading the GPU during the baseline hides the rise.
    let busy_baseline = idle
        .device
        .active_pct
        .is_some_and(|active| active >= BUSY_BASELINE_PCT);
    let rise_failed = if busy_baseline {
        Status::Warn
    } else {
        Status::Fail
    };
    checks.push(match (idle.device.active_pct, load.device.active_pct) {
        (Some(before), Some(after)) => {
            let expected = (busy_pct * 0.9 - 5.0).clamp(0.0, 95.0);
            Check::new(
                scope,
                "device activity",
                if after < expected {
                    Status::Fail
                } else if after > before {
                    Status::Pass
                } else {
                    rise_failed
                },
                format!(
                    "active {before:.1}% idle -> {after:.1}% under load (load alone busy {busy_pct:.1}%)"
                ),
            )
        }
        _ => Check::new(scope, "device activity", Status::Fail, "no active residency"),
    });
    checks.push(match (idle.device.power_w, load.device.power_w) {
        (Some(before), Some(after)) => Check::new(
            scope,
            "power",
            if after >= before + 1.0 {
                Status::Pass
            } else {
                rise_failed
            },
            format!(
                "{} idle -> {} under load",
                format::watts(before),
                format::watts(after)
            ),
        ),
        _ => Check::new(scope, "power", Status::Fail, "no GPU power reading"),
    });
    checks.push(full_load_clock_check(
        load.device.avg_mhz,
        load.device.max_mhz,
        busy_pct,
        load.device.thermal_limited_pct,
    ));
}

/// Load share above which the GPU should run at its top P-state.
const SATURATED_PCT: f64 = 80.0;

/// Under a saturating load the GPU runs at its top P-state (an M4 Max averaged
/// 1577.9 MHz against a 1578 MHz table maximum), so the table's highest entry
/// should match the observed average clock. Power or thermal limiting can hold
/// the clock lower, which is a warning, but an average above the maximum means
/// the table or the residency is wrong.
pub(crate) fn full_load_clock_check(
    avg_mhz: Option<f64>,
    max_mhz: Option<f64>,
    busy_pct: f64,
    thermal_limited_pct: Option<f64>,
) -> Check {
    let scope = "load";
    let name = "full-load clock";
    let (Some(avg), Some(max)) = (avg_mhz, max_mhz) else {
        return Check::new(scope, name, Status::Skip, "no clock or frequency table");
    };
    if busy_pct < SATURATED_PCT {
        return Check::new(
            scope,
            name,
            Status::Skip,
            format!("load busy {busy_pct:.1}% did not saturate the GPU"),
        );
    }
    let detail = format!(
        "average {avg:.0} MHz under load, table max {max:.0} MHz ({:.1}%)",
        avg / max * 100.0
    );
    if avg > max + 0.5 {
        Check::new(scope, name, Status::Fail, detail)
    } else if avg >= max * 0.9 {
        Check::new(scope, name, Status::Pass, detail)
    } else {
        let thermal = thermal_limited_pct
            .map(|pct| format!("; thermal limited {pct:.0}%"))
            .unwrap_or_default();
        Check::new(scope, name, Status::Warn, format!("{detail}{thermal}"))
    }
}

/// The coalition counts work while in flight, so it never reads below the
/// members' `AppUsage` (which apportions overlapping work) nor below the load's
/// own spans; on a quiet GPU it matches `AppUsage`.
fn load_coalition_checks(
    idle: &Report,
    load: &Report,
    settled: Option<(&Report, Option<f64>)>,
    process: Option<&ProcessView>,
    measured: &LoadResult,
    checks: &mut Vec<Check>,
) {
    let scope = "load";
    if !source_ok(load, source::COALITIONS) {
        checks.push(Check::new(
            scope,
            "coalition",
            Status::Skip,
            "coalitions are not available",
        ));
        return;
    }
    let Some(process) = process else {
        checks.push(Check::new(
            scope,
            "coalition",
            Status::Fail,
            "the load process is missing from the report",
        ));
        return;
    };
    let app = load.apps.iter().find(|app| {
        Some(app.coalition_id) == process.coalition_id || app.gpu_pids.contains(&process.pid)
    });
    let Some(app) = app else {
        checks.push(Check::new(
            scope,
            "coalition",
            Status::Fail,
            "no app row contains the load process",
        ));
        return;
    };
    let Some(in_flight) = app.in_flight_gpu_ns else {
        checks.push(Check::new(
            scope,
            "coalition time",
            Status::Fail,
            format!("app {:?} has no coalition GPU time", app.name),
        ));
        return;
    };
    let others = load.totals.attributed_gpu_ns.saturating_sub(process.gpu_ns);
    let quiet = is_quiet(others, load);
    let coalition = in_flight as f64;
    let members = app.attributed_gpu_ns as f64;
    let truth = measured.gpu_ns as f64;
    let mut detail = format!(
        "app {:?}: coalition (in flight) {:.1} ms vs members' AppUsage {:.1} ms; load's own {:.1} ms",
        app.name,
        coalition / 1e6,
        members / 1e6,
        truth / 1e6
    );
    // The app row is apportioned like the process rows: its members plus its
    // exited time when known.
    let apportioned = app.attributed_gpu_ns + app.unattributed_gpu_ns.unwrap_or(0);
    // `AppUsage` carries a per-command-buffer overhead the coalition counter
    // does not, so on a quiet GPU it can read a few percent above it.
    let status = if members
        > coalition * (1.0 + OVER_TOLERANCE_PCT / 100.0) + COALITION_SKEW_NS as f64
        || coalition < truth * (1.0 - TOLERANCE_PCT / 100.0) - SLACK_NS as f64
        || app.gpu_ns != apportioned
        || app.gpu_ns < process.gpu_ns
    {
        if app.gpu_ns != apportioned || app.gpu_ns < process.gpu_ns {
            detail.push_str(&format!(
                "; app row {:.1} ms is not its members plus exited time",
                app.gpu_ns as f64 / 1e6
            ));
        }
        Status::Fail
    } else if !quiet {
        detail.push_str("; other GPU work overlapped, which the coalition counts in full");
        Status::Pass
    } else if (coalition - members).abs()
        <= coalition * TOLERANCE_PCT / 100.0 + COALITION_SKEW_NS as f64
    {
        Status::Pass
    } else {
        Status::Warn
    };
    checks.push(Check::new(scope, "coalition time", status, detail));

    let Some((settled, apps_energy)) = settled else {
        return;
    };
    let device_energy = settled
        .device
        .power_w
        .map(|power| power * settled.window_seconds);
    // Another app's energy lags in or out across the window's edges when it
    // used the GPU too, so only a quiet GPU is held to the tolerance.
    let settled_others = settled
        .totals
        .attributed_gpu_ns
        .saturating_sub(pid_gpu_ns(settled, process.pid));
    let quiet = is_quiet(idle.totals.attributed_gpu_ns, idle) && is_quiet(settled_others, settled);
    checks.push(match (apps_energy, device_energy) {
        (Some(apps), Some(device)) if device > 0.0 => coalition_energy_check(apps, device, quiet),
        _ => Check::new(
            scope,
            "coalition energy",
            Status::Skip,
            "needs coalition GPU energy and IOReport GPU power",
        ),
    });
}

/// GPU energy summed over every coalition between two snapshots, from the raw
/// kernel counters. A coalition new in the window counts from 0. `None` when
/// coalitions or their GPU energy are unavailable.
pub fn coalition_gpu_energy_j(before: &Snapshot, after: &Snapshot) -> Option<f64> {
    let (Some(Ok(before)), Some(Ok(after))) = (&before.coalitions, &after.coalitions) else {
        return None;
    };
    if !after.has_gpu_energy {
        return None;
    }
    let nj: u64 = after
        .iter()
        .map(|coalition| match before.get(coalition.id) {
            Some(old) => coalition.usage.since(&old.usage).gpu_energy_nj,
            None => coalition.usage.gpu_energy_nj,
        })
        .sum();
    Some(nj as f64 / 1e9)
}

/// All coalitions' settled GPU energy against IOReport GPU energy over the
/// same window.
pub(crate) fn coalition_energy_check(apps_j: f64, device_j: f64, quiet: bool) -> Check {
    let error = (apps_j - device_j) / device_j * 100.0;
    let mut detail = format!("apps {apps_j:.2} J vs IOReport GPU {device_j:.2} J ({error:+.1}%)");
    let status = if error.abs() <= ENERGY_TOLERANCE_PCT {
        Status::Pass
    } else if !quiet {
        detail.push_str("; other processes used the GPU, and coalition energy lags by 1-1.5 s");
        Status::Warn
    } else if error.abs() <= ENERGY_TOLERANCE_PCT * 3.0 {
        Status::Warn
    } else {
        Status::Fail
    };
    Check::new("load", "coalition energy", status, detail)
}

/// Whether `others_ns` of GPU time is a negligible share of `report`'s window.
fn is_quiet(others_ns: u64, report: &Report) -> bool {
    (others_ns as f64) < report.window_seconds * 1e9 * QUIET_SHARE
}

fn pid_gpu_ns(report: &Report, pid: i32) -> u64 {
    report
        .processes
        .iter()
        .filter(|process| process.pid == pid)
        .map(|process| process.gpu_ns)
        .sum()
}

/// Measured on the idle window, before the load submits work: right after GPU
/// work the footprint carries about 160 MiB more for 1-2 s.
fn graphics_memory_check(idle: &Report, pid: i32, alloc_bytes: u64) -> Check {
    let scope = "load";
    if !source_ok(idle, source::MEMORY) {
        return Check::new(
            scope,
            "graphics memory",
            Status::Skip,
            "per-process memory is not available",
        );
    }
    let process = idle.processes.iter().find(|process| process.pid == pid);
    match process.and_then(|process| process.graphics_memory_bytes) {
        None => Check::new(
            scope,
            "graphics memory",
            Status::Fail,
            "not readable for the load process, which this user owns",
        ),
        Some(bytes) => {
            // The pipeline and small buffers add well under a MiB.
            let ceiling = alloc_bytes + alloc_bytes / 100 + (16 << 20);
            Check::new(
                scope,
                "graphics memory",
                if (alloc_bytes..=ceiling).contains(&bytes) {
                    Status::Pass
                } else {
                    Status::Fail
                },
                format!(
                    "{} for a {} buffer",
                    format::bytes(bytes),
                    format::bytes(alloc_bytes)
                ),
            )
        }
    }
}

fn temperature_change_check(idle: &Report, load: &Report) -> Check {
    let scope = "load";
    if !source_ok(load, source::TEMPERATURE) {
        return Check::new(
            scope,
            "temperature",
            Status::Skip,
            "temperature is not available",
        );
    }
    match (&idle.device.temperature, &load.device.temperature) {
        (Some(before), Some(after)) => Check::new(
            scope,
            "temperature",
            if after.max_c >= before.max_c + 0.5 {
                Status::Pass
            } else {
                Status::Warn
            },
            format!(
                "max {:.1} C idle -> {:.1} C under load",
                before.max_c, after.max_c
            ),
        ),
        _ => Check::new(scope, "temperature", Status::Warn, "no GPU temperature"),
    }
}

/// Cross-checks IOReport against powermetrics over the same stretch of load.
pub(crate) fn powermetrics_checks(stats: &PowerStats, window: &Report) -> Vec<Check> {
    let scope = "load";
    let device = &window.device;
    let mut checks = Vec::new();
    checks.push(match (stats.active_residency_pct, device.active_pct) {
        (Some(theirs), Some(ours)) => Check::new(
            scope,
            "powermetrics residency",
            if (theirs - ours).abs() <= RESIDENCY_TOLERANCE_PTS {
                Status::Pass
            } else {
                Status::Fail
            },
            format!("powermetrics {theirs:.1}% vs macgpu {ours:.1}% active"),
        ),
        _ => Check::new(
            scope,
            "powermetrics residency",
            Status::Skip,
            "missing on one side",
        ),
    });
    checks.push(match (stats.power_mw, device.power_w) {
        (Some(theirs), Some(ours)) => {
            let theirs = f64::from(theirs) / 1000.0;
            let close = (theirs - ours).abs() <= (theirs * 0.15).max(0.5);
            Check::new(
                scope,
                "powermetrics power",
                if close { Status::Pass } else { Status::Fail },
                format!(
                    "powermetrics {} vs macgpu {}",
                    format::watts(theirs),
                    format::watts(ours)
                ),
            )
        }
        _ => Check::new(
            scope,
            "powermetrics power",
            Status::Skip,
            "missing on one side",
        ),
    });
    let theirs: Vec<f64> = stats
        .frequency_residency
        .iter()
        .map(|entry| f64::from(entry.mhz))
        .collect();
    let ours: Vec<f64> = device
        .pstates
        .iter()
        .filter_map(|state| state.mhz)
        .collect();
    checks.push(if theirs.is_empty() || ours.is_empty() {
        Check::new(
            scope,
            "powermetrics frequencies",
            Status::Skip,
            "missing on one side",
        )
    } else {
        let same = theirs.len() == ours.len()
            && theirs
                .iter()
                .zip(&ours)
                .all(|(theirs, ours)| (theirs - ours).abs() <= 1.0);
        let list = |table: &[f64]| {
            table
                .iter()
                .map(|mhz| format!("{mhz:.0}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        Check::new(
            scope,
            "powermetrics frequencies",
            if same { Status::Pass } else { Status::Fail },
            if same {
                format!("tables match: MHz {}", list(&ours))
            } else {
                format!(
                    "powermetrics MHz {} vs macgpu MHz {}",
                    list(&theirs),
                    list(&ours)
                )
            },
        )
    });
    checks
}
