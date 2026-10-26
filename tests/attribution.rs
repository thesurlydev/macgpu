//! End-to-end attribution tests against real Metal loads in child processes.
//!
//! These depend only on the GPU registry, so they pass without the IOReport,
//! coalition, and memory readers (with IOReport they also use GPUPH).
//!
//! `AppUsage` apportions the GPU among contexts running at the same time,
//! while the load's own timing (command-buffer `GPUEndTime - GPUStartTime`)
//! is wall time in flight, which also covers other processes' overlapping
//! work. So the tests check the invariants that hold under any background
//! load, and the load-versus-own-timing comparison only through
//! [`AttributionBounds`]:
//! - one process is never credited more than its own spans, nor more than the
//!   window per live queue (also under queue churn);
//! - summed over every process, `AppUsage` never exceeds the window: with two
//!   saturating loads whose spans add up to 1.8-1.9x the window, the sum
//!   stayed at 0.96-0.98x;
//! - a load loses at most what every other process was credited: other
//!   processes' `AppUsage`, or GPU active time minus the load's when IOReport
//!   is available, which also covers processes that exited in the window;
//! - on a quiet GPU (other processes under [`QUIET_SHARE`] of the window) a
//!   full-duty load's `AppUsage` matches its own timing within 4% (measured on
//!   an M4 Max: -0.20% to +2.5%); with other processes under
//!   [`UNCONTENDED_SHARE`] a saturating load dominates and a busier load gets
//!   a larger share. With more GPU work these checks are skipped, and the test
//!   says so: a competing saturating load took 54-64% of the GPU from a
//!   full-duty test load.

mod common;

use common::*;
use macgpu::analyze::{AnalyzeOptions, analyze};
use macgpu::engine::{Engine, EngineConfig};
use macgpu::frontend::doctor::GPU_LOCK_HELD_ENV;
use macgpu::metal_load::{
    AttributionBounds, LoadConfig, LoadResult, OVER_TOLERANCE_PCT, QUIET_SHARE, SLACK_NS,
    TOLERANCE_PCT, other_gpu_ns,
};
use macgpu::model::{Report, SourceState, source};
use macgpu::snapshot::{Sampler, Snapshot, SourceSet};
use macgpu::sys::iokit::{GpuClient, RegistryReader};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Attempts for checks that a process exiting inside the window can upset.
const ATTEMPTS: usize = 4;

/// How far below its own timing a load's `AppUsage` reads with nothing else
/// on the GPU: 0.907x at duty 0.1 and 0.933-0.954x at duty 0.25.
const LOW_DUTY_DEFICIT_PCT: f64 = 12.0;

/// Other processes' work, as a share of the window, below which the share
/// and duty-step checks must hold. A load loses at most what other processes
/// are credited, so with 10% the duty steps of 1.0, 0.5, and 0.2 stay apart by
/// more than the 0.85x checked. The idle desktop measured 47-63 ms/s here
/// (GPU active time less the load's), above the 5% of [`QUIET_SHARE`].
const UNCONTENDED_SHARE: f64 = 0.10;

fn registry_only() -> SourceSet {
    SourceSet {
        ioreport: false,
        coalitions: false,
        memory: false,
        temperature: false,
        system: false,
    }
}

fn gpu_ns_of(report: &Report, pid: i32) -> u64 {
    report
        .processes
        .iter()
        .find(|process| process.pid == pid)
        .map(|process| process.gpu_ns)
        .unwrap_or(0)
}

fn window_ns(report: &Report) -> u64 {
    (report.window_seconds * 1e9) as u64
}

fn ms_per_s(ns: u64, report: &Report) -> f64 {
    ns as f64 / 1e6 / report.window_seconds
}

/// Asserts what holds whatever else uses the GPU: the load is never credited
/// more than its own spans (plus the latency `AppUsage` also counts), and all
/// processes together never more than the window.
fn assert_physical_limits(label: &str, report: &Report, measured: u64, truth: &LoadResult) {
    let total = report.totals.attributed_gpu_ns;
    let upper = truth.gpu_ns as f64 * (1.0 + OVER_TOLERANCE_PCT / 100.0) + SLACK_NS as f64;
    assert!(
        measured as f64 <= upper,
        "{label}: credited {:.1} ms, more than the load's own {:.1} ms",
        ms(measured),
        ms(truth.gpu_ns)
    );
    let ceiling = window_ns(report) as f64 * (1.0 + TOLERANCE_PCT / 100.0) + SLACK_NS as f64;
    assert!(
        total as f64 <= ceiling,
        "{label}: all processes {:.0} ms/s, more than one GPU",
        ms_per_s(total, report)
    );
}

/// Whether other processes' work in the window is below [`QUIET_SHARE`].
fn is_quiet(others_ns: u64, report: &Report) -> bool {
    (others_ns as f64) < window_ns(report) as f64 * QUIET_SHARE
}

/// Whether other processes' work in the window is below [`UNCONTENDED_SHARE`].
fn is_uncontended(others_ns: u64, report: &Report) -> bool {
    (others_ns as f64) < window_ns(report) as f64 * UNCONTENDED_SHARE
}

/// The least a load can be credited whatever else runs: its own timing, less
/// the low-duty deficit, less everything other processes were credited.
fn contention_floor_ns(truth: &LoadResult, others_ns: u64) -> f64 {
    truth.gpu_ns as f64 * (1.0 - LOW_DUTY_DEFICIT_PCT / 100.0) - others_ns as f64 - SLACK_NS as f64
}

/// Checks the load's `AppUsage` on a quiet GPU against [`TOLERANCE_PCT`] (for
/// full-duty loads only), and otherwise against the contention floor and the
/// ceiling of [`AttributionBounds`]. Returns `Some(quiet)` when inside, `None`
/// when outside.
fn within_bounds(label: &str, report: &Report, pid: i32, truth: &LoadResult) -> Option<bool> {
    let measured = gpu_ns_of(report, pid);
    let others = other_gpu_ns(report, pid);
    let bounds = AttributionBounds::new(
        truth.gpu_ns,
        truth.busy_fraction(),
        others,
        window_ns(report),
    );
    let floor = contention_floor_ns(truth, others);
    eprintln!(
        "{label}: macgpu {:.1} ms, load's own {:.1} ms ({:.0}% busy), error {:+.2}%, other processes {:.1} ms, all {:.0} ms/s, {}",
        ms(measured),
        ms(truth.gpu_ns),
        truth.busy_fraction() * 100.0,
        error_pct(measured, truth.gpu_ns),
        ms(others),
        ms_per_s(report.totals.attributed_gpu_ns, report),
        if bounds.quiet { "quiet" } else { "busy GPU" }
    );
    let error = error_pct(measured, truth.gpu_ns);
    if bounds.quiet && error.abs() > TOLERANCE_PCT {
        // Work from a process that exited inside the window is invisible to
        // AppUsage, so a window can look quiet without being quiet.
        eprintln!("{label}: {error:+.2}% on a GPU that looked quiet");
        None
    } else if bounds.quiet {
        Some(true)
    } else if measured as f64 >= floor && measured <= bounds.upper_ns {
        eprintln!("{label}: other GPU work, so only the contention range was checked");
        Some(false)
    } else {
        eprintln!(
            "{label}: outside {:.1}-{:.1} ms",
            floor / 1e6,
            ms(bounds.upper_ns)
        );
        None
    }
}

/// Per-process GPU time for the load, summed over the engine's windows and
/// over one window spanning the whole load, matches the load's own timing as
/// far as other GPU work allows.
#[test]
#[ignore]
    fn per_process_gpu_time_matches_load_measurement() {
    let _lock = gpu_lock();
    let config = LoadConfig {
        seconds: 3.0,
        ..LoadConfig::default()
    };
    let mut verdicts = Vec::new();
    for attempt in 1..=ATTEMPTS {
        let mut child = spawn_held(&config);
        let pid = child.pid();
        let engine = Engine::start(EngineConfig {
            interval: Duration::from_millis(500),
            sources: SourceSet::default(),
            options: AnalyzeOptions { include_idle: true },
        });
        let mut sampler = Sampler::new(SourceSet::default());
        // The first engine window ends before the load starts.
        let first = engine.recv().expect("first report");
        assert_eq!(first.sources[source::REGISTRY].state, SourceState::Ok);
        let before = sampler.snapshot();
        child.start().unwrap();
        let truth = child.wait_done().unwrap();
        let after = sampler.snapshot();
        let done_ms = macgpu::sys::clock::unix_ms();

        let mut windows = Vec::new();
        loop {
            let report = engine.recv().expect("report");
            let ended_after_done = report.timestamp_ms > done_ms;
            windows.push(report);
            if ended_after_done {
                break;
            }
        }
        drop(engine);
        child.finish().unwrap();

        let single = analyze(&before, &after, &AnalyzeOptions { include_idle: true });
        let single_ns = gpu_ns_of(&single, pid);
        let summed_ns: u64 = windows.iter().map(|report| gpu_ns_of(report, pid)).sum();

        // The counter is frozen before go and after done, so the windows must
        // add up to exactly the single window: nothing lost, nothing doubled.
        assert!(
            summed_ns.abs_diff(single_ns) <= 2_000_000,
            "{} engine windows sum to {:.1} ms, one window says {:.1} ms",
            windows.len(),
            ms(summed_ns),
            ms(single_ns)
        );
        for report in &windows {
            assert!(
                gpu_ns_of(report, pid) <= window_ns(report) + SLACK_NS,
                "one queue credited more than its window"
            );
        }
        let label = format!("attempt {attempt}");
        assert_physical_limits(&label, &single, single_ns, &truth);
        let verdict = within_bounds(&label, &single, pid, &truth);
        verdicts.push(verdict);
        if verdict == Some(true) {
            break;
        }
    }
    // A process exiting inside the window takes its share out of view (and,
    // without IOReport, out of the contention estimate), so one attempt in
    // range is enough. Measured: 1 in about 10 attempts is affected.
    assert!(
        verdicts.iter().any(Option::is_some),
        "the load's AppUsage was outside the expected range in all {ATTEMPTS} attempts"
    );
    if !verdicts.contains(&Some(true)) {
        eprintln!(
            "warning: the GPU was never quiet in {ATTEMPTS} attempts; only the contention-aware range was checked"
        );
    }
}

/// Two saturating loads at once: each one's own timing covers nearly the
/// whole window, but `AppUsage` splits the GPU between them, so the sum over
/// every process stays within one GPU.
#[test]
#[ignore]
    fn concurrent_loads_share_one_gpu() {
    let _lock = gpu_lock();
    let config = LoadConfig {
        seconds: 3.0,
        ..LoadConfig::default()
    };
    let mut loads = [spawn_held(&config), spawn_held(&config)];
    let mut sampler = Sampler::new(SourceSet::default());
    let before = sampler.snapshot();
    for load in &mut loads {
        load.start().unwrap();
    }
    let truths: Vec<LoadResult> = loads
        .iter_mut()
        .map(|load| load.wait_done().unwrap())
        .collect();
    let after = sampler.snapshot();
    let report = analyze(&before, &after, &AnalyzeOptions { include_idle: true });
    let pids: Vec<i32> = loads.iter().map(|load| load.pid()).collect();
    for load in loads {
        load.finish().unwrap();
    }

    let own: u64 = truths.iter().map(|truth| truth.gpu_ns).sum();
    let shares: Vec<u64> = pids.iter().map(|pid| gpu_ns_of(&report, *pid)).collect();
    let total = report.totals.attributed_gpu_ns;
    // Work by processes other than the two loads.
    let outside = other_gpu_ns(&report, pids[0]).saturating_sub(shares[1]);
    eprintln!(
        "own timing {:.0} + {:.0} ms/s; AppUsage {:.0} + {:.0} ms/s; all processes {:.0} ms/s; other processes {:.0} ms/s",
        ms_per_s(truths[0].gpu_ns, &report),
        ms_per_s(truths[1].gpu_ns, &report),
        ms_per_s(shares[0], &report),
        ms_per_s(shares[1], &report),
        ms_per_s(total, &report),
        ms_per_s(outside, &report)
    );
    assert!(
        own as f64 >= window_ns(&report) as f64 * 1.4,
        "the loads did not overlap: own timing {:.0} ms/s",
        ms_per_s(own, &report)
    );
    let quiet = is_uncontended(outside, &report);
    if !quiet {
        eprintln!("other GPU work: skipped the check that each load gets 15% of the GPU");
    }
    for (index, (share, truth)) in shares.iter().zip(&truths).enumerate() {
        assert_physical_limits("concurrent", &report, *share, truth);
        let others = outside + shares[1 - index];
        assert!(
            *share as f64 >= contention_floor_ns(truth, others),
            "a load got {:.0} ms/s while everything else got {:.0} ms/s",
            ms_per_s(*share, &report),
            ms_per_s(others, &report)
        );
        // With only the two loads, each gets a real share of the GPU.
        if quiet {
            assert!(
                *share as f64 >= window_ns(&report) as f64 * 0.15,
                "a load got only {:.0} ms/s",
                ms_per_s(*share, &report)
            );
        }
    }
}

/// A load's share of `AppUsage` follows its duty cycle, and at full duty it
/// dominates the GPU. Both hold only while other processes use under
/// [`UNCONTENDED_SHARE`] of the GPU; beyond that each duty is checked against
/// the contention floor instead.
#[test]
#[ignore]
    fn load_share_scales_with_duty_cycle() {
    let _lock = gpu_lock();
    // The previous quiet step's duty and rate.
    let mut previous: Option<(f64, f64)> = None;
    for duty in [1.0, 0.5, 0.2] {
        let config = LoadConfig {
            seconds: 2.0,
            duty,
            ..LoadConfig::default()
        };
        let mut child = spawn_held(&config);
        let pid = child.pid();
        let mut sampler = Sampler::new(SourceSet::default());
        let before = sampler.snapshot();
        child.start().unwrap();
        let truth = child.wait_done().unwrap();
        let after = sampler.snapshot();
        child.finish().unwrap();
        let report = analyze(&before, &after, &AnalyzeOptions { include_idle: true });

        let measured = gpu_ns_of(&report, pid);
        let rate = ms_per_s(measured, &report);
        let share = measured as f64 / report.totals.attributed_gpu_ns.max(1) as f64;
        let others = other_gpu_ns(&report, pid);
        eprintln!(
            "duty {duty:.1}: own {:.0} ms/s, AppUsage {rate:.0} ms/s (ratio {:.3}), {:.0}% of all AppUsage, other processes {:.0} ms/s",
            ms_per_s(truth.gpu_ns, &report),
            measured as f64 / truth.gpu_ns.max(1) as f64,
            share * 100.0,
            ms_per_s(others, &report)
        );
        assert_physical_limits(&format!("duty {duty}"), &report, measured, &truth);
        assert!(
            measured as f64 >= contention_floor_ns(&truth, others),
            "duty {duty}: {rate:.0} ms/s while other processes got {:.0} ms/s",
            ms_per_s(others, &report)
        );
        if !is_uncontended(others, &report) {
            eprintln!("duty {duty:.1}: other GPU work, so the share and step checks were skipped");
            previous = None;
            continue;
        }
        if duty == 1.0 {
            assert!(share > 0.5, "a saturating load got {:.0}%", share * 100.0);
        }
        if let Some((previous_duty, previous_rate)) = previous {
            assert!(
                rate < previous_rate * 0.85,
                "duty {duty} got {rate:.0} ms/s, duty {previous_duty} {previous_rate:.0} ms/s"
            );
        }
        previous = Some((duty, rate));
    }
}

/// The physical ceiling for one process in one window: every live queue busy
/// for the whole window.
fn ceiling_ns(window_ns: u64, queues: usize) -> u64 {
    window_ns * queues as u64 + 2_000_000
}

/// The pre-fix diff from section 4.1 of the plan: per client, summed over
/// queues, and a decrease credited the client's whole remaining lifetime.
fn old_client_delta(previous: Option<&GpuClient>, current: &GpuClient) -> u64 {
    match previous {
        Some(previous) if current.accumulated_ns() >= previous.accumulated_ns() => {
            current.accumulated_ns() - previous.accumulated_ns()
        }
        Some(_) => current.accumulated_ns(),
        None => current.accumulated_ns(),
    }
}

fn client_of(snapshot: &Snapshot, pid: i32) -> Option<&GpuClient> {
    snapshot
        .registry
        .as_ref()
        .ok()?
        .clients()
        .find(|client| client.pid == pid)
}

/// Section 4.1, end to end. Releasing a Metal command queue removes its
/// `AppUsage` entry while the client stays open, so the client's summed
/// counter goes down. The old diff then credited the client's remaining
/// lifetime; the queue-aligned diff never credits more than is possible and
/// its total tracks the load's own measurement.
#[test]
#[ignore]
    fn queue_churn_never_over_credits() {
    let _lock = gpu_lock();
    const QUEUES: usize = 2;
    const SAMPLE: Duration = Duration::from_millis(50);
    let config = LoadConfig {
        seconds: 4.0,
        queues: QUEUES,
        churn_ms: 800,
        // Long enough that a sample sees the released entry gone before its
        // replacement appears.
        churn_gap_ms: 150,
        ..LoadConfig::default()
    };
    let mut child = spawn_held(&config);
    let pid = child.pid();
    // GPU active time over the whole run, to bound other processes' work
    // including any that exit during it. Kept out of the 50 ms samples.
    let mut device = Sampler::new(SourceSet {
        ioreport: true,
        ..registry_only()
    });
    let device_before = device.snapshot();
    let mut sampler = Sampler::new(registry_only());
    let mut snapshots = vec![sampler.snapshot()];
    child.start().unwrap();
    let started = Instant::now();
    let run = Duration::from_secs_f64(config.seconds) + Duration::from_millis(300);
    while started.elapsed() < run {
        std::thread::sleep(SAMPLE);
        snapshots.push(sampler.snapshot());
    }
    let truth = child.wait_done().unwrap();
    snapshots.push(sampler.snapshot());
    let device_after = device.snapshot();
    child.finish().unwrap();
    assert!(truth.queues_destroyed >= 3, "{truth:?}");

    // Queue lifecycle, from raw registry reads.
    let clients: Vec<&GpuClient> = snapshots
        .iter()
        .map(|snapshot| client_of(snapshot, pid).expect("load client present in every sample"))
        .collect();
    assert!(
        clients
            .iter()
            .all(|client| client.registry_id == clients[0].registry_id),
        "the load's GPU client was replaced; expected one client for the whole run"
    );
    let fewer_entries = clients
        .iter()
        .filter(|client| client.queues.len() < QUEUES)
        .count();
    let decreases = clients
        .windows(2)
        .filter(|pair| pair[1].accumulated_ns() < pair[0].accumulated_ns())
        .count();
    eprintln!(
        "{} samples: {fewer_entries} with fewer than {QUEUES} AppUsage entries, {decreases} where the client's summed counter went down; {} queues released",
        clients.len(),
        truth.queues_destroyed
    );
    assert!(
        fewer_entries > 0,
        "a released queue's AppUsage entry never disappeared"
    );
    assert!(decreases > 0, "the client's summed counter never went down");

    let mut old_over = 0;
    let mut old_total = 0u64;
    let mut new_total = 0u64;
    let mut worst_old = 0.0f64;
    for pair in snapshots.windows(2) {
        let report = analyze(&pair[0], &pair[1], &AnalyzeOptions { include_idle: true });
        let window_ns = (report.window_seconds * 1e9) as u64;
        let ceiling = ceiling_ns(window_ns, QUEUES);
        let new = gpu_ns_of(&report, pid);
        let old = old_client_delta(client_of(&pair[0], pid), client_of(&pair[1], pid).unwrap());
        assert!(
            new <= ceiling,
            "macgpu credited {:.1} ms in a {:.1} ms window with {QUEUES} queues",
            ms(new),
            ms(window_ns)
        );
        if old > ceiling {
            old_over += 1;
            worst_old = worst_old.max(old as f64 / window_ns as f64);
        }
        old_total += old;
        new_total += new;
    }
    eprintln!(
        "old diff: {old_over} impossible windows (worst {worst_old:.1}x the window), total {:.1} ms; macgpu total {:.1} ms; load's own {:.1} ms",
        ms(old_total),
        ms(new_total),
        ms(truth.gpu_ns)
    );
    assert!(
        old_over > 0,
        "the old diff never over-credited, so this test no longer reproduces section 4.1"
    );

    // A released queue loses at most its last partial window: up to one
    // sample interval per release.
    let lost_bound = truth.queues_destroyed * (SAMPLE.as_nanos() as u64 + 20_000_000);
    // Other processes' work: their `AppUsage`, or GPU active time less the
    // load's credited total when IOReport has it. Active time keeps the work
    // of a competitor that exits during the run, which `AppUsage` drops. The
    // load's lost releases only raise this estimate, which keeps the floor
    // below sound under any other GPU work.
    let whole = analyze(
        &device_before,
        &device_after,
        &AnalyzeOptions { include_idle: true },
    );
    let by_app_usage = whole
        .totals
        .attributed_gpu_ns
        .saturating_sub(gpu_ns_of(&whole, pid));
    let by_residency = whole
        .device
        .active_pct
        .map(|active| (active / 100.0 * window_ns(&whole) as f64) as u64)
        .unwrap_or(0)
        .saturating_sub(new_total);
    let others = by_app_usage.max(by_residency);
    let upper = truth.gpu_ns as f64 * (1.0 + OVER_TOLERANCE_PCT / 100.0) + SLACK_NS as f64;
    // The load runs at full duty, so its quiet `AppUsage` is within
    // TOLERANCE_PCT of its own timing; it loses at most `others` to other work.
    let lower =
        truth.gpu_ns as f64 * (1.0 - TOLERANCE_PCT / 100.0) - others as f64 - SLACK_NS as f64;
    eprintln!(
        "other processes {:.1} ms ({}); lower bound {:.1} ms, releases may lose up to {:.1} ms",
        ms(others),
        if is_quiet(others, &whole) {
            "quiet"
        } else {
            "busy GPU"
        },
        lower / 1e6,
        ms(lost_bound)
    );
    assert!(
        new_total as f64 <= upper,
        "macgpu total {:.1} ms exceeds the load's own {:.1} ms",
        ms(new_total),
        ms(truth.gpu_ns)
    );
    assert!(
        (new_total + lost_bound) as f64 >= lower,
        "macgpu total {:.1} ms is too far below the load's own {:.1} ms (others {:.1} ms)",
        ms(new_total),
        ms(truth.gpu_ns),
        ms(others)
    );
}

/// The observing macgpu process must not become a GPU client itself.
#[test]
#[ignore]
    fn observer_holds_no_gpu_client() {
    let mut child = Command::new(MACGPU)
        .args(["--json", "--count", "3", "--interval", "0.5"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn macgpu");
    let pid = child.id() as i32;
    let lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut reader = RegistryReader::new();
    let mut reports = 0;
    for line in lines {
        let report: Report = serde_json::from_str(&line.expect("read")).expect("NDJSON report");
        assert_eq!(report.schema_version, macgpu::model::SCHEMA_VERSION);
        assert!(report.totals.gpu_clients > 0);
        assert!(
            clients_of(&mut reader, pid).is_empty(),
            "macgpu (pid {pid}) opened a GPU client while sampling"
        );
        reports += 1;
    }
    assert!(child.wait().expect("wait").success());
    assert_eq!(reports, 3);
}

#[test]
#[ignore]
    fn doctor_json_reports_every_source() {
    let output = Command::new(MACGPU)
        .args(["doctor", "--json"])
        .output()
        .expect("run doctor");
    assert!(output.status.success(), "{output:?}");
    let doctor: serde_json::Value = serde_json::from_slice(&output.stdout).expect("doctor JSON");
    let sources = doctor["sources"].as_array().expect("sources");
    for name in [
        source::REGISTRY,
        source::IOREPORT,
        source::COALITIONS,
        source::MEMORY,
        source::TEMPERATURE,
        source::SYSTEM,
    ] {
        assert!(
            sources.iter().any(|source| source["name"] == name),
            "{name} missing"
        );
    }
    let registry = sources
        .iter()
        .find(|source| source["name"] == source::REGISTRY)
        .unwrap();
    assert_eq!(registry["state"], "ok");
    assert_eq!(registry["readable_without_root"], true);
    let observer = doctor["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|check| check["name"] == "observer")
        .expect("observer check");
    assert_eq!(observer["status"], "pass", "{observer}");
    assert!(doctor["gpus"][0]["cores"].as_u64().unwrap_or(0) > 0);
    assert_eq!(doctor["ok"], true);
}

/// `doctor --load` passes on this machine. Checks whose source is not
/// available are skipped, so this holds before and after every reader lands;
/// once they do, it verifies power, coalitions, and memory too.
#[test]
#[ignore]
    fn doctor_load_passes() {
    let _lock = gpu_lock();
    let output = Command::new(MACGPU)
        .args(["doctor", "--load", "--load-seconds", "2", "--json"])
        // This test holds the lock the doctor would wait for.
        .env(GPU_LOCK_HELD_ENV, "1")
        .output()
        .expect("run doctor --load");
    let doctor: serde_json::Value = serde_json::from_slice(&output.stdout).expect("doctor JSON");
    let checks = doctor["checks"].as_array().expect("checks");
    for check in checks {
        eprintln!(
            "{} {} {}: {}",
            check["status"], check["scope"], check["name"], check["detail"]
        );
    }
    let attribution = checks
        .iter()
        .find(|check| check["name"] == "attribution")
        .expect("attribution check");
    assert_ne!(attribution["status"], "fail", "{attribution}");
    assert!(output.status.success(), "doctor --load failed: {output:?}");
    assert_eq!(doctor["ok"], true);
    assert!(doctor["load"]["measured"]["gpu_ns"].as_u64().unwrap() > 0);
}
