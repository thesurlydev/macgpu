//! Rendering tests with ratatui's `TestBackend` and state-machine tests, all
//! driven by the synthetic feed.

use super::app::{App, Column, Effect, Overlay, Screen, SortKey};
use super::synthetic::{self, SyntheticFeed};
use super::{pump, ui};
use crate::engine::ReportFeed;
use crate::frontend::alerts::{AlertArgs, Alerts};
use crate::model::Report;
use crate::view::{Filter, Group, RowKey};
use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use std::time::Duration;

const SECOND: Duration = Duration::from_secs(1);

fn app() -> App {
    App::new(Group::Process, Filter::default(), false, false).with_feed(true, SECOND)
}

/// An app fed `steps` synthetic reports.
fn fed(steps: u64) -> App {
    let mut app = app();
    feed(&mut app, 1..=steps, false);
    app
}

fn feed(app: &mut App, steps: impl Iterator<Item = u64>, include_idle: bool) {
    for step in steps {
        app.push_report(synthetic::report(step, SECOND, include_idle), false);
    }
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

fn press(app: &mut App, codes: &[KeyCode]) {
    for code in codes {
        app.handle_key(key(*code));
    }
}

fn type_text(app: &mut App, text: &str) {
    for character in text.chars() {
        app.handle_key(key(KeyCode::Char(character)));
    }
}

fn render(app: &mut App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
    terminal.draw(|frame| ui::draw(frame, app)).expect("draw");
    let buffer = terminal.backend().buffer();
    let mut out = String::new();
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            line.push_str(buffer[(x, y)].symbol());
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

fn process_key(pid: i32) -> RowKey {
    let report = synthetic::report(1, SECOND, true);
    let process = report
        .processes
        .iter()
        .find(|process| process.pid == pid)
        .expect("synthetic process");
    RowKey::Process {
        pid,
        start_us: process.start_us,
    }
}

fn selected_pid(app: &App) -> Option<i32> {
    app.selected_row().and_then(|row| row.row.pid)
}

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
    assert!(screen.contains("~"), "{screen}");
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

// ---- state machine ----------------------------------------------------------

#[test]
fn selection_follows_the_row_across_refreshes() {
    let mut app = fed(10);
    // At step 10 python3.12 is idle and ollama leads.
    app.selected = Some(process_key(412));
    app.rebuild_rows();
    let before = app.selected_index;
    // python3.12 starts at 20 s and ranks above WindowServer.
    feed(&mut app, 11..=21, false);
    assert_eq!(app.selected, Some(process_key(412)));
    assert_eq!(selected_pid(&app), Some(412));
    assert_ne!(
        app.selected_index, before,
        "row moved but selection followed it"
    );

    press(&mut app, &[KeyCode::Down]);
    let moved = app.selected;
    feed(&mut app, 31..=32, false);
    assert_eq!(app.selected, moved);
    press(&mut app, &[KeyCode::Home]);
    assert_eq!(app.selected_index, 0);
    press(&mut app, &[KeyCode::End]);
    assert_eq!(app.selected_index, app.rows.len() - 1);
}

#[test]
fn recently_active_rows_linger_instead_of_blinking() {
    // ollama bursts for 24 s of every 40: at 25 s it just went idle.
    let mut app = fed(25);
    let ollama = app
        .rows
        .iter()
        .find(|row| row.row.pid == Some(3310))
        .expect("ollama lingers");
    assert_eq!(ollama.row.busy_ms_per_s, 0.0);
    assert_eq!(ollama.row.activity, crate::model::Activity::Recent);
    // Ten seconds later it is gone.
    feed(&mut app, 26..=36, false);
    assert!(app.rows.iter().all(|row| row.row.pid != Some(3310)));

    // On a live feed, a process that exited does not linger at all.
    let mut live = self::app().with_liveness(|pid, _| pid != 3310);
    feed(&mut live, 1..=25, false);
    assert!(live.rows.iter().all(|row| row.row.pid != Some(3310)));
}

#[test]
fn filter_applies_as_you_type_and_escape_restores() {
    let mut app = fed(30);
    press(&mut app, &[KeyCode::Char('/')]);
    type_text(&mut app, "chrome, !gpu");
    assert!(matches!(app.overlay, Overlay::Filter { .. }));
    let names: Vec<&str> = app.rows.iter().map(|row| row.row.name.as_str()).collect();
    assert!(
        names.is_empty()
            || names
                .iter()
                .all(|name| name.to_lowercase().contains("chrome")),
        "{names:?}"
    );
    press(&mut app, &[KeyCode::Esc]);
    assert!(app.filter.is_empty());

    press(&mut app, &[KeyCode::Char('/')]);
    type_text(&mut app, "python");
    press(&mut app, &[KeyCode::Enter]);
    assert_eq!(app.filter_text, "python");
    assert_eq!(app.rows.len(), 1);
    assert_eq!(selected_pid(&app), Some(6120));
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("filter python"), "{screen}");

    // Esc on the main screen clears the filter.
    press(&mut app, &[KeyCode::Esc]);
    assert!(app.filter.is_empty());
    assert!(app.rows.len() > 1);
}

#[test]
fn sort_cycles_and_reverses_keeping_unattributed_last() {
    let mut app = fed(30);
    assert_eq!(app.sort, SortKey::Gpu);
    press(&mut app, &[KeyCode::Char('s')]);
    assert_eq!(app.sort, SortKey::Power);
    press(&mut app, &[KeyCode::Char(','), KeyCode::Char(',')]);
    assert_eq!(app.sort, SortKey::Name);
    let names: Vec<String> = app
        .rows
        .iter()
        .filter(|row| row.row.key != RowKey::Unattributed)
        .map(|row| row.row.name.to_lowercase())
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
    assert_eq!(
        app.rows.last().map(|row| row.row.key),
        Some(RowKey::Unattributed)
    );

    press(&mut app, &[KeyCode::Char('S')]);
    assert!(app.reverse);
    let first = app.rows[0].row.name.to_lowercase();
    assert!(names.iter().all(|name| *name <= first));
    assert_eq!(
        app.rows.last().map(|row| row.row.key),
        Some(RowKey::Unattributed)
    );

    press(&mut app, &[KeyCode::Char('.')]);
    assert_eq!(app.sort, SortKey::Gpu);
}

#[test]
fn memory_sort_puts_missing_values_last() {
    let mut app = fed(30);
    app.sort = SortKey::Memory;
    app.rebuild_rows();
    assert_eq!(
        selected_pid(&app),
        Some(3310),
        "ollama has the most GPU memory"
    );
    let last_real = app
        .rows
        .iter()
        .rev()
        .find(|row| row.row.key != RowKey::Unattributed)
        .expect("rows");
    assert_eq!(last_real.row.memory_bytes, None);
}

#[test]
fn grouping_cycles_and_keeps_the_place() {
    let mut app = fed(30);
    app.selected = Some(process_key(6120));
    app.rebuild_rows();
    press(&mut app, &[KeyCode::Char('g')]);
    assert_eq!(app.group, Group::App);
    assert_eq!(app.selected, Some(RowKey::App { coalition_id: 1100 }));
    press(&mut app, &[KeyCode::Char('g')]);
    assert_eq!(app.group, Group::Queue);
    assert!(matches!(
        app.selected,
        Some(RowKey::Queue {
            registry_id: 0x1000_6c00,
            ..
        })
    ));
    press(&mut app, &[KeyCode::Char('g')]);
    assert_eq!(app.group, Group::Client);
    assert_eq!(
        app.selected,
        Some(RowKey::Client {
            registry_id: 0x1000_6c00
        })
    );
    press(&mut app, &[KeyCode::Char('G')]);
    assert_eq!(app.group, Group::Queue);
    press(&mut app, &[KeyCode::Char('g'), KeyCode::Char('g')]);
    assert_eq!(app.group, Group::Process);
    assert_eq!(selected_pid(&app), Some(6120));
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("processes"), "{screen}");
}

#[test]
fn freeze_holds_the_display_and_catches_up() {
    let mut app = fed(10);
    let shown = app.report.as_ref().map(|report| report.timestamp_ms);
    press(&mut app, &[KeyCode::Char('f')]);
    assert!(app.frozen);
    feed(&mut app, 11..=15, false);
    assert_eq!(app.report.as_ref().map(|report| report.timestamp_ms), shown);
    assert_eq!(app.pending_reports(), 5);
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("FROZEN +5"), "{screen}");

    press(&mut app, &[KeyCode::Char('f')]);
    assert!(!app.frozen);
    assert_eq!(app.pending_reports(), 0);
    assert_eq!(app.history.device.len(), 15, "no samples lost while frozen");
    assert_ne!(app.report.as_ref().map(|report| report.timestamp_ms), shown);
}

#[test]
fn signal_needs_confirmation_and_respects_readonly() {
    let mut app = fed(30);
    app.selected = Some(process_key(6120));
    app.rebuild_rows();
    press(&mut app, &[KeyCode::Char('k')]);
    let Overlay::Confirm(confirm) = &app.overlay else {
        panic!("expected a confirmation, got {:?}", app.overlay);
    };
    assert_eq!(confirm.pid, 6120);
    assert_eq!(confirm.signal, 0, "SIGTERM is the default");
    assert!(
        app.take_effects().is_empty(),
        "nothing is sent before confirming"
    );

    press(&mut app, &[KeyCode::Right, KeyCode::Enter]);
    let effects = app.take_effects();
    assert_eq!(
        effects,
        vec![Effect::Signal {
            pid: 6120,
            start_us: Some(process_key(6120).start_us()),
            signal: libc::SIGKILL,
            name: "python3.12".into(),
        }]
    );

    press(&mut app, &[KeyCode::Char('k'), KeyCode::Esc]);
    assert!(app.take_effects().is_empty());
    assert_eq!(app.overlay, Overlay::None);

    let mut readonly =
        App::new(Group::Process, Filter::default(), false, true).with_feed(true, SECOND);
    feed(&mut readonly, 1..=30, false);
    press(&mut readonly, &[KeyCode::Char('k')]);
    assert_eq!(readonly.overlay, Overlay::None);
    assert!(
        readonly
            .status
            .as_ref()
            .is_some_and(|status| status.text.contains("read-only"))
    );
    assert!(!render(&mut readonly, 120, 40).contains("k signal"));

    // The unattributed row has no process.
    app.selected = Some(RowKey::Unattributed);
    app.rebuild_rows();
    press(&mut app, &[KeyCode::Char('k')]);
    assert_eq!(app.overlay, Overlay::None);
}

trait StartUs {
    fn start_us(&self) -> u64;
}

impl StartUs for RowKey {
    fn start_us(&self) -> u64 {
        match self {
            RowKey::Process { start_us, .. } => *start_us,
            _ => 0,
        }
    }
}

#[test]
fn interval_and_idle_keys_emit_effects() {
    let mut app = fed(3);
    press(&mut app, &[KeyCode::Char('+')]);
    assert_eq!(
        app.take_effects(),
        vec![Effect::SetInterval(Duration::from_secs(2))]
    );
    press(&mut app, &[KeyCode::Char('-')]);
    assert_eq!(
        app.take_effects(),
        vec![Effect::SetInterval(Duration::from_millis(500))]
    );
    app.interval_applied(Duration::from_millis(100), true);
    press(&mut app, &[KeyCode::Char('-')]);
    assert!(
        app.take_effects().is_empty(),
        "already at the shortest interval"
    );

    press(&mut app, &[KeyCode::Char('a')]);
    assert_eq!(app.take_effects(), vec![Effect::SetIncludeIdle(true)]);
    assert!(app.include_idle);

    press(&mut app, &[KeyCode::Char('q')]);
    assert_eq!(app.take_effects(), vec![Effect::Quit]);
    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
    assert_eq!(app.take_effects(), vec![Effect::Quit]);
}

#[test]
fn zoom_changes_the_history_span() {
    let mut app = fed(3);
    assert_eq!(app.zoom_span_ms(), 60_000);
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("last 1m"), "{screen}");
    press(&mut app, &[KeyCode::Char('z'), KeyCode::Char('z')]);
    assert_eq!(app.zoom_span_ms(), 900_000);
    press(
        &mut app,
        &[KeyCode::Char('Z'), KeyCode::Char('Z'), KeyCode::Char('Z')],
    );
    assert_eq!(app.zoom_span_ms(), 3_600_000);
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("last 1h"), "{screen}");
}

#[test]
fn replay_feed_is_labeled_and_cannot_change_interval_or_signal() {
    let mut feed = SyntheticFeed::replay(3);
    let mut app = App::new(Group::Process, Filter::default(), false, false)
        .with_feed(feed.is_live(), feed.interval());
    let mut alerts = Alerts::from_args(&AlertArgs::default()).expect("alerts");
    pump(&mut feed, &mut app, &mut alerts);
    assert_eq!(app.reports_seen, 3);
    assert!(app.feed_ended);
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("REPLAY ENDED"), "{screen}");

    press(&mut app, &[KeyCode::Char('+')]);
    assert!(app.take_effects().is_empty());
    assert!(
        app.status
            .as_ref()
            .is_some_and(|status| status.text.contains("replay"))
    );
    press(&mut app, &[KeyCode::Char('k')]);
    assert_eq!(app.overlay, Overlay::None);
}

#[test]
fn pump_applies_live_reports_and_alert_state() {
    let mut feed = SyntheticFeed::unpaced();
    let mut app = app();
    let mut alerts = Alerts::from_args(&AlertArgs::default()).expect("alerts");
    pump(&mut feed, &mut app, &mut alerts);
    // An unpaced feed never times out; the turn is capped.
    assert_eq!(app.reports_seen, 64);
    assert!(!app.feed_ended);

    app.push_report(synthetic::report(100, SECOND, false), true);
    assert!(app.alert_firing);
    assert!(render(&mut app, 120, 40).contains("ALERT"));
}

#[test]
fn mouse_selects_rows_and_sorts_by_header() {
    let mut app = fed(30);
    render(&mut app, 120, 40);
    let table = app.hitboxes.table;
    let click = |column, row| MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::NONE,
    };
    app.handle_mouse(click(table.x + 10, table.y + 2));
    assert_eq!(app.selected_index, 2);

    // Double click opens the detail screen.
    app.handle_mouse(click(table.x + 10, table.y + 2));
    assert!(matches!(app.screen, Screen::Detail(_)));
    press(&mut app, &[KeyCode::Esc]);

    render(&mut app, 120, 40);
    let (x, _, _) = *app
        .hitboxes
        .header
        .iter()
        .find(|(_, _, column)| *column == Column::Memory)
        .expect("memory column");
    app.handle_mouse(click(x + 1, app.hitboxes.header_y));
    assert_eq!(app.sort, SortKey::Memory);
    app.handle_mouse(click(x + 1, app.hitboxes.header_y));
    assert!(app.reverse);

    let before = app.selected_index;
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: table.x,
        row: table.y,
        modifiers: KeyModifiers::NONE,
    });
    assert_eq!(app.selected_index, (before + 3).min(app.rows.len() - 1));
}

#[test]
fn history_charts_cover_the_who_series() {
    let mut app = fed(120);
    let who = super::graphs::WhoChart::rank(&app);
    let names: Vec<&str> = who
        .series
        .iter()
        .map(|(_, name, ..)| name.as_str())
        .collect();
    // Over the last minute the MLX script outweighs Ollama's bursts.
    assert_eq!(names.first(), Some(&"Ghostty"), "{names:?}");
    assert!(names.contains(&"Ollama"), "{names:?}");
    // Device % of the series never exceeds the device's active %.
    let total: f64 = who.series.iter().map(|(.., average)| average).sum();
    assert!(total <= 100.0, "{total}");
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains("■ Ollama"), "{screen}");
}

/// Prints the main screens for review: `cargo test dump_screens -- --ignored --nocapture`.
#[test]
#[ignore]
fn dump_screens() {
    let mut app = fed(60);
    println!("{}", render(&mut app, 120, 40));
    println!("{}", render(&mut app, 80, 24));
    app.selected = Some(process_key(3310));
    app.rebuild_rows();
    press(&mut app, &[KeyCode::Enter]);
    println!("{}", render(&mut app, 120, 40));
    press(
        &mut app,
        &[KeyCode::Esc, KeyCode::Char('t'), KeyCode::Char('g')],
    );
    println!("{}", render(&mut app, 120, 40));
}

#[test]
#[ignore]
fn bench_draw() {
    let started = std::time::Instant::now();
    let mut app = app();
    feed(&mut app, 1..=3600, true);
    let ingest = started.elapsed();
    for zoom in [0, 4] {
        app.zoom = zoom;
        let started = std::time::Instant::now();
        for _ in 0..20 {
            render(&mut app, 200, 60);
        }
        println!("zoom {zoom}: {:?} per draw", started.elapsed() / 20);
    }
    println!("ingest 3600 reports: {ingest:?} ({:?} each)", ingest / 3600);
}
