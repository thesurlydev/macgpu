//! Rendering tests with ratatui's `TestBackend`, driven by the synthetic feed.

use super::app::{Overlay, Screen};
use super::app_tests::{SECOND, app, fed, feed, press, process_key, render};
use super::synthetic::{self};
use crate::model::Report;
use crate::view::{Group, RowKey};
use crossterm::event::KeyCode;

// ---- rendering --------------------------------------------------------------

#[test]
fn full_layout_shows_every_section() {
    let mut app = fed(60);
    let screen = render(&mut app, 120, 40);
    for expected in [
        "macgpu",
        "Apple M4 Max 40-core GPU",
        "LIVE 1s",
        " GPU ",
        "active",
        "norm",
        "MHz",
        "°C",
        " MEM ",
        "wired limit 96.0 GiB",
        "thermal",
        " USE ",
        "ms/s attributed",
        "P-states",
        "memory partial (other users' processes need root)",
        "history",
        "who",
        "power",
        "clock",
        "RECOVERY",
        "GPU recovered from a hang",
        "processes",
        "sort GPU ms/s",
        "GPU ms/s",
        "SHARE",
        "DEV%",
        "WATTS",
        "GPU MEM",
        "LAST",
        "HISTORY",
        "ollama",
        "python3.12",
        "WindowServer",
        "(exited or unattributed)",
        "q quit",
    ] {
        assert!(
            screen.contains(expected),
            "missing {expected:?} in\n{screen}"
        );
    }
    // Estimated process power is marked; WindowServer has no readable memory.
    assert!(
        screen
            .lines()
            .any(|line| line.contains("ollama") && line.contains('~')),
        "estimated process watts are not marked\n{screen}"
    );
    let window_server = screen
        .lines()
        .find(|line| line.contains("WindowServer") && line.contains("412"))
        .expect("WindowServer row");
    assert!(window_server.contains(" - "), "{window_server}");
}

#[test]
fn compact_layout_fits_small_terminals() {
    let mut app = fed(60);
    let screen = render(&mut app, 80, 24);
    assert!(screen.contains("LIVE 1s"), "{screen}");
    assert!(screen.contains("GPU ms/s"), "{screen}");
    assert!(screen.contains("ollama"), "{screen}");
    assert!(screen.lines().all(|line| line.chars().count() <= 80));
    // No full-size header meters in compact mode.
    assert!(
        !screen.lines().any(|line| line.starts_with(" MEM ")),
        "{screen}"
    );

    let tiny = render(&mut app, 50, 12);
    assert!(tiny.contains("ollama"), "{tiny}");
    let too_small = render(&mut app, 20, 4);
    assert!(too_small.contains("terminal too"), "{too_small}");
}

#[test]
fn renders_before_any_report_and_with_empty_reports() {
    let mut app = app();
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("waiting for the first sample"), "{screen}");

    // A report with nothing but a timestamp: every optional field missing.
    app.push_report(
        Report {
            timestamp_ms: 1_000,
            window_seconds: 1.0,
            ..Report::default()
        },
        false,
    );
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("nothing used the GPU"), "{screen}");
    assert!(screen.contains("GPU ms/s"), "{screen}");
    assert!(screen.contains("active -"), "{screen}");
    assert!(
        !screen.contains("WATTS"),
        "no power column without data\n{screen}"
    );
    assert!(
        !screen.contains("GPU MEM"),
        "no memory column without data\n{screen}"
    );
    let compact = render(&mut app, 80, 24);
    assert!(compact.contains("nothing used the GPU"), "{compact}");

    // The app level says why it is empty when coalitions are unavailable.
    let mut report = synthetic::report(5, SECOND, false);
    report.apps.clear();
    report.sources.insert(
        crate::model::source::COALITIONS.into(),
        crate::model::SourceStatus::with(crate::model::SourceState::Denied, "sandboxed"),
    );
    app.push_report(report, false);
    press(&mut app, &[KeyCode::Char('g')]);
    let screen = render(&mut app, 120, 40);
    assert!(
        screen.contains("no per-app data: coalitions denied, sandboxed"),
        "{screen}"
    );
}

#[test]
fn limiter_and_memory_limit_follow_the_model() {
    let mut app = app();
    let mut report = synthetic::report(5, SECOND, false);
    // Demand-following states are not limits.
    report.device.limiters = vec![
        crate::model::StateShare {
            name: "PERF".into(),
            pct: 80.0,
        },
        crate::model::StateShare {
            name: "PRFBOOST".into(),
            pct: 20.0,
        },
    ];
    report.device.thermal_limited_pct = Some(0.0);
    report.device.counters.clear();
    report.device.memory.wired_limit_bytes = None;
    app.push_report(report.clone(), false);
    let screen = render(&mut app, 120, 40);
    assert!(!screen.contains("limiter"), "{screen}");
    assert!(!screen.contains("LIMITER"), "{screen}");
    // The memory line shortens to fit: the long form on a wide terminal.
    assert!(screen.contains("allocated · of 128.0 GiB RAM"), "{screen}");
    let wide = render(&mut app, 160, 40);
    assert!(
        wide.contains("wired limit default · 128.0 GiB RAM"),
        "{wide}"
    );

    let mut limited = synthetic::report(6, SECOND, false);
    limited.device.limiters = vec![crate::model::StateShare {
        name: "FAST_DIE".into(),
        pct: 100.0,
    }];
    for (key, count) in [
        ("GPU Throttler Counters/Throttle Counter DPE", 2),
        ("GPU Throttler Counters/Throttle Counter Total", 2),
        ("THM Counters/THM Activation Count", 1),
        ("Issue Counts/Total Restarts", 1),
    ] {
        limited.device.counters.insert(key.to_string(), count);
    }
    app.push_report(limited, false);
    let screen = render(&mut app, 160, 40);
    assert!(screen.contains("limiter FAST_DIE 100%"), "{screen}");
    assert!(screen.contains("LIMITER"), "{screen}");
    // Unofficial readings are marked as guesses.
    assert!(
        screen.contains("LIMITER   FAST_DIE (die temp?) 100% of powered time · limited about"),
        "{screen}"
    );
    // The total is used, not added to its parts; issue counters are events.
    assert!(
        screen.contains("throttle counter +2 · thermal activations +1"),
        "{screen}"
    );
    assert!(!screen.contains("Restarts"), "{screen}");
}

#[test]
fn falls_back_to_attributed_time_without_ioreport() {
    let mut app = app();
    for step in 1..=5 {
        let mut report = synthetic::report(step, SECOND, false);
        report.device = Default::default();
        report.apps.clear();
        for process in &mut report.processes {
            process.device_pct = None;
            process.est_power_w = None;
            process.coalition_id = None;
        }
        app.push_report(report, false);
    }
    let screen = render(&mut app, 120, 40);
    // Without IOReport the stack is in GPU ms/s, on a round scale.
    assert!(screen.contains("who · processes · 0-1000"), "{screen}");
    assert!(screen.contains("scale 1000"), "{screen}");
    assert!(!screen.contains("DEV%"), "{screen}");
    assert!(!screen.contains("WATTS"), "{screen}");
}

#[test]
fn detail_help_and_confirm_overlays_render() {
    let mut app = fed(60);
    app.selected = Some(process_key(3310));
    app.rebuild_rows();
    press(&mut app, &[KeyCode::Enter]);
    assert!(matches!(app.screen, Screen::Detail(_)));
    let screen = render(&mut app, 120, 40);
    for expected in [
        "process",
        "ollama (3310)",
        "/Applications/Ollama.app",
        "Ollama",
        "client 0x10003e02",
        "queue 1",
        "GPU ms/s",
        "watts (est.)",
        "GPU memory",
    ] {
        assert!(
            screen.contains(expected),
            "missing {expected:?} in\n{screen}"
        );
    }

    press(&mut app, &[KeyCode::Char('k')]);
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("send signal"), "{screen}");
    assert!(screen.contains("SIGTERM"), "{screen}");
    press(&mut app, &[KeyCode::Esc, KeyCode::Esc]);
    assert_eq!(app.screen, Screen::Main);

    app.group = Group::App;
    app.rebuild_rows();
    app.selected = Some(RowKey::App { coalition_id: 1201 });
    app.rebuild_rows();
    press(&mut app, &[KeyCode::Enter]);
    let screen = render(&mut app, 120, 40);
    for expected in [
        "Google Chrome",
        "measured",
        "billed to it by others",
        "GPU PROCESSES IN THIS APP",
        "Google Chrome Helper (GPU)",
    ] {
        assert!(
            screen.contains(expected),
            "missing {expected:?} in\n{screen}"
        );
    }
    let compact = render(&mut app, 80, 24);
    assert!(compact.contains("Google Chrome"), "{compact}");
    press(&mut app, &[KeyCode::Esc]);

    press(&mut app, &[KeyCode::Char('?')]);
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("macgpu help"), "{screen}");
    assert!(screen.contains("apportions"), "{screen}");
    press(&mut app, &[KeyCode::Esc]);
    assert_eq!(app.overlay, Overlay::None);
}

#[test]
fn app_detail_explains_in_flight_and_inseparable_exited_time() {
    let mut app = fed(30);
    app.group = Group::App;
    app.selected = Some(RowKey::App { coalition_id: 1201 });
    app.rebuild_rows();
    press(&mut app, &[KeyCode::Enter]);
    let screen = render(&mut app, 160, 40);
    let line = screen
        .lines()
        .find(|line| line.contains("in flight"))
        .unwrap_or_else(|| panic!("no in-flight line in\n{screen}"));
    assert!(line.contains("each count in full"), "{line}");
    assert!(screen.contains("exited    1.5 ms/s"), "{screen}");
    press(&mut app, &[KeyCode::Esc]);
    // In flight is only in the app detail, never in the table.
    assert!(!render(&mut app, 160, 40).contains("in flight"));

    // Sample 50: a member exited while Ollama's work overlapped.
    feed(&mut app, 31..=50, false);
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("unattributed ≥0.0 ms/s"), "{screen}");
    let chrome = screen
        .lines()
        .rev()
        .find(|line| line.contains("Google Chrome"))
        .unwrap_or_else(|| panic!("no Chrome row in\n{screen}"));
    assert!(chrome.contains("     - "), "{chrome}");
    press(&mut app, &[KeyCode::Enter]);
    let screen = render(&mut app, 160, 40);
    assert!(screen.contains("exited    -"), "{screen}");
    assert!(screen.contains("cannot be separated"), "{screen}");
}

#[test]
fn tree_view_nests_processes_under_apps() {
    let mut app = fed(30);
    press(&mut app, &[KeyCode::Char('t')]);
    let names: Vec<(u8, String)> = app
        .rows
        .iter()
        .map(|row| (row.depth, row.row.name.clone()))
        .collect();
    let ghostty = names
        .iter()
        .position(|(depth, name)| *depth == 0 && name == "Ghostty")
        .expect("Ghostty app header");
    assert!(
        names[ghostty + 1..]
            .iter()
            .take_while(|(depth, _)| *depth > 0)
            .any(|(_, name)| name == "python3.12"),
        "{names:?}"
    );
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("└─"), "{screen}");
    assert!(screen.contains("tree"), "{screen}");

    // Queue level: process, client, queue.
    app.group = Group::Queue;
    app.rebuild_rows();
    let depths: Vec<u8> = app.rows.iter().map(|row| row.depth).collect();
    assert!(depths.contains(&2), "{depths:?}");
}

#[test]
fn tree_falls_back_to_parents_without_coalitions() {
    let mut app = app();
    let mut report = synthetic::report(30, SECOND, true);
    report.apps.clear();
    for process in &mut report.processes {
        process.coalition_id = None;
    }
    app.push_report(report, false);
    press(&mut app, &[KeyCode::Char('t')]);
    let chrome = app
        .rows
        .iter()
        .position(|row| row.row.name == "Google Chrome")
        .expect("chrome");
    assert_eq!(app.rows[chrome].depth, 0);
    assert_eq!(app.rows[chrome + 1].row.name, "Google Chrome Helper (GPU)");
    assert_eq!(app.rows[chrome + 1].depth, 1);
}

/// With bundle grouping (coalitions disabled) the report gives every process
/// its group's synthetic coalition id, so the tree nests processes under their
/// app header instead of listing each app header with no children and every
/// process again as a flat row. Mirrors what `analyze::apps::from_bundles`
/// emits: app rows keyed by bundle, and each process pointing at its group.
#[test]
fn tree_nests_processes_under_bundle_apps() {
    let mut app = app();
    let mut report = synthetic::report(30, SECOND, true);
    let mut groups: std::collections::BTreeMap<String, Vec<i32>> = Default::default();
    for process in &report.processes {
        let key = process
            .path
            .as_deref()
            .and_then(crate::sys::process::bundle_path)
            .map(str::to_string)
            .unwrap_or_else(|| process.name.clone());
        groups.entry(key).or_default().push(process.pid);
    }
    report.apps.clear();
    for process in &mut report.processes {
        process.coalition_id = None;
    }
    for (index, (bundle, pids)) in groups.iter().enumerate() {
        let coalition_id = 1 << 63 | index as u64;
        let members: Vec<_> = report
            .processes
            .iter()
            .filter(|process| pids.contains(&process.pid))
            .collect();
        let gpu_ns = members.iter().map(|process| process.gpu_ns).sum();
        report.apps.push(crate::model::AppView {
            coalition_id,
            name: crate::sys::process::bundle_name(bundle).to_string(),
            bundle_path: Some(bundle.clone()),
            gpu_ns,
            attributed_gpu_ns: gpu_ns,
            members: pids.len(),
            gpu_pids: pids.clone(),
            ..crate::model::AppView::default()
        });
        for process in &mut report.processes {
            if pids.contains(&process.pid) {
                process.coalition_id = Some(coalition_id);
            }
        }
    }
    app.push_report(report, false);
    press(&mut app, &[KeyCode::Char('t')]);
    let rows = &app.rows;
    let header = |name: &str| {
        rows.iter()
            .position(|row| row.row.name == name && row.depth == 0)
            .unwrap_or_else(|| panic!("no header {name}"))
    };
    let chrome = header("Google Chrome");
    assert_eq!(rows[chrome + 1].row.name, "Google Chrome Helper (GPU)");
    assert_eq!(rows[chrome + 1].depth, 1);
    // No GPU process is orphaned at depth 0 beside its own app header.
    let orphans = rows
        .iter()
        .filter(|row| row.depth == 0 && row.row.pid.is_some())
        .count();
    assert_eq!(orphans, 0, "{rows:?}");
}
