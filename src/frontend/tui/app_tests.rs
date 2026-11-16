//! State-machine tests and the helpers shared with the rendering tests.

use super::app::{App, Column, Effect, Overlay, Screen, SortKey};
use super::synthetic::{self, SyntheticFeed};
use super::{pump, ui};
use crate::engine::ReportFeed;
use crate::frontend::alerts::{AlertArgs, Alerts};
use crate::view::{Filter, Group, RowKey};
use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use std::time::Duration;

pub(super) const SECOND: Duration = Duration::from_secs(1);

pub(super) fn app() -> App {
    App::new(Group::Process, Filter::default(), false, false).with_feed(true, SECOND)
}

/// An app fed `steps` synthetic reports.
pub(super) fn fed(steps: u64) -> App {
    let mut app = app();
    feed(&mut app, 1..=steps, false);
    app
}

pub(super) fn feed(app: &mut App, steps: impl Iterator<Item = u64>, include_idle: bool) {
    for step in steps {
        app.push_report(synthetic::report(step, SECOND, include_idle), false);
    }
}

pub(super) fn key(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

pub(super) fn press(app: &mut App, codes: &[KeyCode]) {
    for code in codes {
        app.handle_key(key(*code));
    }
}

pub(super) fn type_text(app: &mut App, text: &str) {
    for character in text.chars() {
        app.handle_key(key(KeyCode::Char(character)));
    }
}

pub(super) fn render(app: &mut App, width: u16, height: u16) -> String {
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

pub(super) fn process_key(pid: i32) -> RowKey {
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

pub(super) fn selected_pid(app: &App) -> Option<i32> {
    app.selected_row().and_then(|row| row.row.pid)
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
    assert_eq!(app.reports_seen, super::MAX_REPORTS_PER_TURN as u64);
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
    let averages: Vec<f64> = who.series.iter().map(|series| series.average).collect();
    assert!(!averages.is_empty(), "no ranked series");
    assert!(
        averages.windows(2).all(|pair| pair[0] >= pair[1]),
        "{averages:?}"
    );
    // Device % of the series never exceeds the device's active %.
    let total: f64 = averages.iter().sum();
    assert!(total <= 100.0, "{total}");
    let screen = render(&mut app, 120, 40);
    assert!(screen.contains('■'), "{screen}");
}

#[test]
fn end_on_detail_scrolls_without_moving_the_hidden_selection() {
    let mut app = fed(30);
    press(&mut app, &[KeyCode::Enter]);
    assert!(matches!(app.screen, Screen::Detail(_)));
    let selected = app.selected;
    let selected_index = app.selected_index;
    press(&mut app, &[KeyCode::End]);
    assert_eq!(app.selected, selected);
    assert_eq!(app.selected_index, selected_index);
    let Screen::Detail(detail) = &app.screen else {
        panic!("detail screen");
    };
    assert_eq!(detail.scroll, u16::MAX);
}

#[test]
fn control_chords_do_not_type_into_the_filter() {
    let mut app = fed(30);
    press(&mut app, &[KeyCode::Char('/')]);
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL));
    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL));
    // AltGr reports as CONTROL|ALT on some terminals; it must still type.
    app.handle_key(KeyEvent::new(
        KeyCode::Char('x'),
        KeyModifiers::CONTROL | KeyModifiers::ALT,
    ));
    let Overlay::Filter { input, .. } = &app.overlay else {
        panic!("filter prompt");
    };
    assert_eq!(input, "x");
}
