//! End-to-end tests for per-app (coalition) attribution and per-process
//! graphics memory, against a real Metal load in a child process.
//!
//! DEPENDS ON the coalition and task-memory readers (`sys::coalition`,
//! `sys::task_memory`, `analyze::apps`), and `coalition_energy_matches_ioreport`
//! also on the IOReport reader. Each test first checks that its sources report
//! `ok` and fails with a message naming the missing source otherwise.

mod common;

use common::*;
use macgpu::analyze::{AnalyzeOptions, analyze};
use macgpu::engine::{Engine, EngineConfig};
use macgpu::frontend::doctor::coalition_gpu_energy_j;
use macgpu::metal_load::{
    LoadConfig, LoadProcess, OVER_TOLERANCE_PCT, QUIET_SHARE, SLACK_NS, TOLERANCE_PCT,
};
use macgpu::model::{AppView, ProcessView, Report, source};
use macgpu::snapshot::{Sampler, SourceSet};
use std::path::Path;
use std::time::Duration;

/// Coalition and `AppUsage` counters are credited at slightly different
/// moments; the apps reader measured up to 15 ms per window.
const SKEW_NS: u64 = 15_000_000;
/// How long coalition energy takes to catch up with IOReport after a load.
const ENERGY_SETTLE: Duration = Duration::from_secs(2);
/// The longest delay measured between IOReport GPU energy and coalition energy.
const ENERGY_LAG: Duration = Duration::from_millis(1500);
/// Settled coalition energy against IOReport, with quiet window edges.
const ENERGY_TOLERANCE_PCT: f64 = 2.0;

fn process(report: &Report, pid: i32) -> Option<&ProcessView> {
    report.processes.iter().find(|process| process.pid == pid)
}

/// An app row is apportioned like the process rows: its members' `AppUsage`
/// plus its exited time when known.
fn assert_apportioned(app: &AppView) {
    assert_eq!(
        app.gpu_ns,
        app.attributed_gpu_ns + app.unattributed_gpu_ns.unwrap_or(0),
        "app {:?}",
        app.name
    );
}

fn app_of(report: &Report, coalition_id: u64) -> Option<&AppView> {
    report
        .apps
        .iter()
        .find(|app| app.coalition_id == coalition_id)
}

/// A load process that exits in the middle of a window loses that window's
/// GPU time from `AppUsage` (its client is gone by the next sample), but the
/// resource coalition keeps it: in flight, and in the app row as apportioned
/// unattributed time when it can be separated from overlapping work.
#[test]
#[ignore = "needs real GPU hardware and a Metal load; not available on virtual machines"]
fn exited_process_time_is_kept_by_coalition() {
    let _lock = gpu_lock();
    const INTERVAL: Duration = Duration::from_secs(1);
    // Starts right after a window boundary and ends half way through the
    // third window, so about a fifth of the load lands in the exit window.
    let config = LoadConfig {
        seconds: 2.5,
        wait: true,
        hold: false,
        ..LoadConfig::default()
    };
    let mut child = LoadProcess::spawn(Path::new(MACGPU), &config).expect("spawn load");
    let pid = child.pid();
    let engine = Engine::start(EngineConfig {
        interval: INTERVAL,
        sources: SourceSet::default(),
        options: AnalyzeOptions { include_idle: true },
    });
    let first = engine.recv().expect("first report");
    require(&first, source::COALITIONS);
    let started_ms = macgpu::sys::clock::unix_ms();
    child.start().unwrap();
    let truth = child.wait_done().unwrap();
    let done_ms = macgpu::sys::clock::unix_ms();
    child.finish().unwrap();
    let exited_ms = macgpu::sys::clock::unix_ms();

    let mut windows = Vec::new();
    loop {
        let report = engine.recv().expect("report");
        let after_exit = report.timestamp_ms > exited_ms;
        windows.push(report);
        if after_exit {
            break;
        }
    }
    drop(engine);

    let coalition_id = windows
        .iter()
        .find_map(|report| process(report, pid).and_then(|process| process.coalition_id))
        .expect("the load process's coalition_id");
    let exit_window = windows.last().unwrap();
    assert!(
        process(exit_window, pid).is_none(),
        "the exited load still has a GPU client in the exit window"
    );

    let mut attributed = 0u64;
    let mut coalition = 0u64;
    let mut members = 0u64;
    for report in &windows {
        attributed += process(report, pid)
            .map(|process| process.gpu_ns)
            .unwrap_or(0);
        if let Some(app) = app_of(report, coalition_id) {
            assert_apportioned(app);
            coalition += app.in_flight_gpu_ns.expect("coalition in-flight time");
            members += app.attributed_gpu_ns;
        }
    }
    let exit_app =
        app_of(exit_window, coalition_id).expect("the load's app row in the exit window");
    let exit_gap = exit_app
        .in_flight_gpu_ns
        .unwrap_or(0)
        .saturating_sub(exit_app.attributed_gpu_ns);
    let others = exit_window.totals.attributed_gpu_ns;
    for app in &exit_window.apps {
        if app.in_flight_gpu_ns.unwrap_or(0) > 0 || app.attributed_gpu_ns > 0 {
            eprintln!(
                "exit window app {:?}: in flight {:.1} ms, AppUsage {:.1} ms, exited {:?} ms",
                app.name,
                ms(app.in_flight_gpu_ns.unwrap_or(0)),
                ms(app.attributed_gpu_ns),
                app.unattributed_gpu_ns.map(ms)
            );
        }
    }
    eprintln!(
        "exit window GPU active {:.1}% of {:.2} s",
        exit_window.device.active_pct.unwrap_or(f64::NAN),
        exit_window.window_seconds
    );
    eprintln!(
        "load's own {:.1} ms; its AppUsage {:.1} ms; coalition {:.1} ms vs members' AppUsage {:.1} ms; exit window: {:.1} ms only the coalition saw in flight, {} apportioned as exited, other processes {:.1} ms",
        ms(truth.gpu_ns),
        ms(attributed),
        ms(coalition),
        ms(members),
        ms(exit_gap),
        exit_app
            .unattributed_gpu_ns
            .map_or_else(|| "none".into(), |exited| format!("{:.1} ms", ms(exited))),
        ms(others)
    );

    // The part of the load that ran in the exit window, where the load no
    // longer has a client. The coalition counts work in flight, which is never
    // less than the load's own spans, so it must show at least this much
    // beyond what the live members' AppUsage explains.
    let exit_window_start_ms = windows[windows.len() - 2].timestamp_ms;
    let in_exit_window = done_ms.saturating_sub(exit_window_start_ms) as f64
        / done_ms.saturating_sub(started_ms).max(1) as f64;
    let lost = truth.gpu_ns as f64 * in_exit_window.clamp(0.0, 1.0);
    eprintln!(
        "{:.0}% of the load ran in the exit window: about {:.1} ms",
        in_exit_window * 100.0,
        lost / 1e6
    );
    assert!(lost >= 100e6, "the load barely reached the exit window");
    assert!(
        exit_gap as f64 >= lost * 0.8 - SKEW_NS as f64,
        "the coalition did not keep the exited load's time: {:.1} ms of about {:.1} ms",
        ms(exit_gap),
        lost / 1e6
    );
    // Apportioned exited time never exceeds the in-flight time it comes from.
    // Live processes can take at most their own AppUsage away from the exited
    // load's share, so the app row keeps the rest. An overlap ratio measured
    // on other apps does not carry over: small apps that overlapped the load
    // read 1.5-2x in flight, while the load itself barely overlapped them.
    match exit_app.unattributed_gpu_ns {
        Some(exited) => {
            assert!(
                exited <= exit_gap + SKEW_NS,
                "apportioned exited time {:.1} ms exceeds the in-flight gap {:.1} ms",
                ms(exited),
                ms(exit_gap)
            );
            assert!(
                exited as f64 >= lost * 0.8 - others as f64 - SKEW_NS as f64,
                "the app row kept {:.1} ms of about {:.1} ms exited work while live processes used {:.1} ms",
                ms(exited),
                lost / 1e6,
                ms(others)
            );
        }
        None => eprintln!("exited time overlapped other GPU work, so the app row leaves it out"),
    }
    // The coalition counts the load's work while in flight, like the load's
    // own command-buffer spans, so over the whole run it covers them.
    assert!(
        coalition as f64 >= truth.gpu_ns as f64 * (1.0 - TOLERANCE_PCT / 100.0) - SLACK_NS as f64,
        "coalition {:.1} ms is below the load's own {:.1} ms",
        ms(coalition),
        ms(truth.gpu_ns)
    );
}

/// While the load runs, its coalition's GPU time covers the load's own
/// measurement. `AppUsage` apportions GPU time among overlapping contexts
/// while the coalition counts work in flight, so the two agree only on a
/// quiet GPU, and there `AppUsage` reads a few percent above the coalition.
#[test]
#[ignore = "needs real GPU hardware and a Metal load; not available on virtual machines"]
fn coalition_time_covers_load() {
    let _lock = gpu_lock();
    let mut child = spawn_held(&LoadConfig {
        seconds: 3.0,
        ..LoadConfig::default()
    });
    let pid = child.pid();
    let mut sampler = Sampler::new(SourceSet::default());
    let before = sampler.snapshot();
    child.start().unwrap();
    let truth = child.wait_done().unwrap();
    let after = sampler.snapshot();
    child.finish().unwrap();

    let report = analyze(&before, &after, &AnalyzeOptions { include_idle: true });
    require(&report, source::COALITIONS);
    let load = process(&report, pid).expect("load process");
    let coalition_id = load.coalition_id.expect("the load process's coalition_id");
    let app = app_of(&report, coalition_id).expect("the load's app row");
    assert_apportioned(app);
    assert!(app.gpu_ns >= load.gpu_ns);
    let in_flight = app.in_flight_gpu_ns.expect("coalition in-flight time");
    let others = report.totals.attributed_gpu_ns - load.gpu_ns;
    let window_ns = (report.window_seconds * 1e9) as u64;
    eprintln!(
        "app {:?}: coalition {:.1} ms, members' AppUsage {:.1} ms, load's AppUsage {:.1} ms, load's own {:.1} ms, other processes {:.1} ms",
        app.name,
        ms(in_flight),
        ms(app.attributed_gpu_ns),
        ms(load.gpu_ns),
        ms(truth.gpu_ns),
        ms(others)
    );
    assert!(app.gpu_pids.contains(&pid), "{:?}", app.gpu_pids);
    assert!(app.attributed_gpu_ns >= load.gpu_ns);
    // `AppUsage` carries a per-command-buffer overhead the coalition counter
    // does not: on a quiet GPU it read 2.1-2.4% above it.
    assert!(
        app.attributed_gpu_ns as f64
            <= in_flight as f64 * (1.0 + OVER_TOLERANCE_PCT / 100.0) + SKEW_NS as f64,
        "members' AppUsage {:.1} ms is far above the coalition's {:.1} ms",
        ms(app.attributed_gpu_ns),
        ms(in_flight)
    );
    assert!(
        in_flight as f64 >= truth.gpu_ns as f64 * (1.0 - TOLERANCE_PCT / 100.0) - SLACK_NS as f64,
        "coalition {:.1} ms below the load's own {:.1} ms",
        ms(in_flight),
        ms(truth.gpu_ns)
    );
    if (others as f64) < window_ns as f64 * QUIET_SHARE {
        let gap = in_flight.abs_diff(app.attributed_gpu_ns);
        assert!(
            gap as f64 <= in_flight as f64 * TOLERANCE_PCT / 100.0 + SKEW_NS as f64,
            "quiet GPU: coalition {:.1} ms vs AppUsage {:.1} ms",
            ms(in_flight),
            ms(app.attributed_gpu_ns)
        );
    }
}

/// Coalition GPU energy, summed over every coalition from the raw counters,
/// matches IOReport whole-GPU energy over a load. Needs IOReport as well.
///
/// Coalition energy is credited 1-1.5 s late: at the end of a 3 s load it read
/// 35-51% below IOReport, and 2 s later within 0.2%. So the window ends 2 s
/// after the load. The lag also moves energy across both edges of the window:
/// in from the [`ENERGY_LAG`] before it starts, out from the last
/// [`ENERGY_LAG`] of it. On a quiet GPU both are negligible; while other
/// processes start or stop GPU work they are not, so the tolerance adds the
/// difference between the two edges' IOReport energy.
#[test]
#[ignore = "needs real IOReport and coalition hardware readings; not available on virtual machines"]
fn coalition_energy_matches_ioreport() {
    let _lock = gpu_lock();
    let mut misses = Vec::new();
    for _ in 0..3 {
        let mut child = spawn_held(&LoadConfig {
            seconds: 3.0,
            ..LoadConfig::default()
        });
        let mut sampler = Sampler::new(SourceSet::default());
        let lead_in = sampler.snapshot();
        std::thread::sleep(ENERGY_LAG);
        let before = sampler.snapshot();
        child.start().unwrap();
        child.wait_done().unwrap();
        std::thread::sleep(ENERGY_SETTLE - ENERGY_LAG);
        let tail = sampler.snapshot();
        std::thread::sleep(ENERGY_LAG);
        let after = sampler.snapshot();
        child.finish().unwrap();

        let report = analyze(&before, &after, &AnalyzeOptions { include_idle: true });
        require(&report, source::COALITIONS);
        require(&report, source::IOREPORT);
        // The app rows' energy is scaled to IOReport's when it is known, so
        // compare the raw coalition counters.
        let apps = coalition_gpu_energy_j(&before, &after).expect("coalition GPU energy");
        let device = gpu_energy_j(&report);
        let lead_in = gpu_energy_j(&analyze(&lead_in, &before, &AnalyzeOptions::default()));
        let tail = gpu_energy_j(&analyze(&tail, &after, &AnalyzeOptions::default()));
        let tolerance = device * ENERGY_TOLERANCE_PCT / 100.0 + (lead_in - tail).abs();
        eprintln!(
            "apps {apps:.2} J vs IOReport {device:.2} J ({:+.2}%); edges {lead_in:.2} J in, {tail:.2} J out; tolerance {tolerance:.2} J",
            (apps - device) / device * 100.0
        );
        assert!(device > 1.0, "the load used only {device:.2} J");
        if (apps - device).abs() <= tolerance {
            return;
        }
        misses.push(format!(
            "{apps:.2} J vs {device:.2} J (tolerance {tolerance:.2} J)"
        ));
    }
    panic!("coalition energy never matched IOReport: {misses:?}");
}

fn gpu_energy_j(report: &Report) -> f64 {
    report.device.power_w.expect("GPU power") * report.window_seconds
}

/// Sampling intervals for [`app_energy_matches_device_energy_at_every_interval`]:
/// shorter than, equal to, and longer than coalition energy's 0.5 s step.
const ENERGY_INTERVALS_MS: [u64; 4] = [100, 250, 500, 1000];

/// The apps' energy sums to IOReport's GPU energy over a load at every
/// sampling interval, and in no window does it exceed the device's.
///
/// The kernel credits coalition GPU energy in steps about 0.5 s apart, so most
/// windows shorter than that see none for any app. Splitting each window by
/// its own coalition energy lost those windows' energy: at 0.25 s the apps
/// summed to about half of IOReport's. Every window's device energy is split
/// among the apps, so the sum holds whatever else uses the GPU; only windows
/// in which nothing used the GPU keep their (idle) energy unattributed.
#[test]
#[ignore = "needs real IOReport and coalition hardware readings; not available on virtual machines"]
fn app_energy_matches_device_energy_at_every_interval() {
    let _lock = gpu_lock();
    let mut child = spawn_held(&LoadConfig {
        seconds: 4.0,
        duty: 0.6,
        ..LoadConfig::default()
    });
    let engines: Vec<Engine> = ENERGY_INTERVALS_MS
        .iter()
        .map(|ms| {
            Engine::start(EngineConfig {
                interval: Duration::from_millis(*ms),
                ..EngineConfig::default()
            })
        })
        .collect();
    std::thread::sleep(Duration::from_millis(500));
    child.start().unwrap();
    child.wait_done().unwrap();
    std::thread::sleep(Duration::from_secs(1));
    let stop_ms = macgpu::sys::clock::unix_ms();
    child.finish().unwrap();

    for (engine, interval_ms) in engines.iter().zip(ENERGY_INTERVALS_MS) {
        let (mut apps_j, mut device_j, mut windows) = (0.0, 0.0, 0);
        loop {
            let report = engine.recv().expect("report");
            require(&report, source::COALITIONS);
            require(&report, source::IOREPORT);
            let device = gpu_energy_j(&report);
            let apps: f64 = report.apps.iter().filter_map(|app| app.energy_j).sum();
            assert!(
                report
                    .apps
                    .iter()
                    .all(|app| app.energy_j.unwrap_or(0.0) >= 0.0),
                "negative app energy at {interval_ms} ms"
            );
            assert!(
                apps <= device * (1.0 + 1e-9) + 1e-9,
                "at {interval_ms} ms one window's apps used {apps:.4} J of the device's {device:.4} J"
            );
            apps_j += apps;
            device_j += device;
            windows += 1;
            if report.timestamp_ms >= stop_ms {
                break;
            }
        }
        eprintln!(
            "{interval_ms} ms: apps {apps_j:.2} J vs IOReport {device_j:.2} J ({:.3}) over {windows} windows",
            apps_j / device_j
        );
        assert!(device_j > 2.0, "the load used only {device_j:.2} J");
        assert!(
            apps_j >= device_j * 0.98,
            "at {interval_ms} ms the apps' energy {apps_j:.2} J is {:.1}% of IOReport's {device_j:.2} J",
            apps_j / device_j * 100.0
        );
    }
}

/// Largest relative difference allowed between `run` energy at 0.1 s and 1 s.
const RUN_ENERGY_TOLERANCE: f64 = 0.2;

/// `macgpu run` reports about the same energy for a load at its default 0.1 s
/// interval as at 1 s. Both measure the same load at the same time: an outer
/// `run -i 0.1` runs an inner `run -i 1`, which runs the load, so other GPU work
/// affects both alike. The outer tree also holds the inner macgpu, which never
/// opens a Metal device. Before energy was split over a trailing span, the
/// 0.1 s run reported about a fifth of the 1 s run's energy.
///
/// At any interval a run misses about a second of the load's energy, because
/// coalition energy trails it: the load's first second is split by energy
/// credited before it started, and its last second is credited after it
/// exited. How much depends on where the 0.5 s energy steps and the window
/// edges fall, so the two figures differ by up to about 1 s of 8 (4 s loads
/// differed by up to 13%), hence [`RUN_ENERGY_TOLERANCE`].
#[test]
#[ignore = "needs real GPU energy readings and a Metal load; not available on virtual machines"]
fn run_energy_does_not_depend_on_interval() {
    let _lock = gpu_lock();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("apps-run-energy");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let (outer, inner) = (dir.join("outer.json"), dir.join("inner.json"));
    let status = std::process::Command::new(MACGPU)
        .args(["run", "-i", "0.1", "-q", "-o"])
        .arg(&outer)
        .args(["--", MACGPU, "run", "-i", "1", "-q", "-o"])
        .arg(&inner)
        .args(["--", MACGPU, "__load", "--seconds", "8"])
        .stdout(std::process::Stdio::null())
        .status()
        .expect("run macgpu");
    assert!(status.success(), "{status}");
    let read = |path: &Path| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).expect("summary")).expect("json")
    };
    let (outer, inner) = (read(&outer), read(&inner));
    let energy = |summary: &serde_json::Value| summary["energy_j"].as_f64().expect("energy_j");
    let device = |summary: &serde_json::Value| {
        summary["device_energy_j"]
            .as_f64()
            .expect("device_energy_j")
    };
    let (fast, slow) = (energy(&outer), energy(&inner));
    eprintln!(
        "run energy: {fast:.2} J at 0.1 s, {slow:.2} J at 1 s ({:.3}); device {:.2} J and {:.2} J",
        fast / slow,
        device(&outer),
        device(&inner)
    );
    assert!(slow > 1.0, "the load used only {slow:.2} J at 1 s");
    assert!(fast <= device(&outer) * (1.0 + 1e-9));
    assert!(
        (fast / slow - 1.0).abs() <= RUN_ENERGY_TOLERANCE,
        "{fast:.2} J at 0.1 s against {slow:.2} J at 1 s"
    );
}

/// A child holding a touched 1 GiB `MTLBuffer` shows 1 GiB of graphics
/// memory (the plan measured 1028 MB).
///
/// Measured before the child submits any GPU work: right after a load the
/// footprint carries about 160 MiB more, freed 1-2 s after the last submission.
#[test]
#[ignore = "needs real GPU hardware and a Metal buffer; not available on virtual machines"]
fn graphics_memory_of_one_gib_buffer() {
    const GIB: u64 = 1 << 30;
    let child = spawn_held(&LoadConfig {
        alloc_mib: 1024,
        ..LoadConfig::default()
    });
    let pid = child.pid();
    assert_eq!(child.ready.alloc_bytes, GIB);
    let mut sampler = Sampler::new(SourceSet::default());
    let before = sampler.snapshot();
    std::thread::sleep(Duration::from_millis(200));
    let after = sampler.snapshot();
    let report = analyze(&before, &after, &AnalyzeOptions { include_idle: true });
    // Dropping the child without starting it kills it by PID.
    drop(child);

    require(&report, source::MEMORY);
    let bytes = process(&report, pid)
        .expect("load process")
        .graphics_memory_bytes
        .expect("graphics memory for a process this user owns");
    eprintln!(
        "graphics memory {:.2} MiB for a 1 GiB buffer",
        bytes as f64 / (1 << 20) as f64
    );
    assert!(
        (GIB..=GIB + 16 * (1 << 20)).contains(&bytes),
        "{bytes} bytes for a 1 GiB buffer"
    );
}
