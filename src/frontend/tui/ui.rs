//! Draws the main screen: header meters, history charts, the dynamic events
//! strip, the table, and the footer. Layout adapts to the terminal: `Full`
//! from 100x30, `Compact` for small panes (80x24 tmux, SSH), `Tiny` below that.

use super::app::{App, Confirm, Overlay, SIGNALS, Screen, StatusLevel};
use super::graphs::{self, WhoChart};
use super::header::{event_lines, header_lines};
use super::style::{self, GPU, LABEL};
use super::table::draw_table;
use super::{detail, help};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Widget};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Density {
    Full,
    Compact,
    Tiny,
}

const FULL_MIN_WIDTH: u16 = 100;
const FULL_MIN_HEIGHT: u16 = 30;
const COMPACT_MIN_WIDTH: u16 = 60;
const COMPACT_MIN_HEIGHT: u16 = 16;

pub(super) fn density(area: Rect) -> Density {
    if area.width >= FULL_MIN_WIDTH && area.height >= FULL_MIN_HEIGHT {
        Density::Full
    } else if area.width >= COMPACT_MIN_WIDTH && area.height >= COMPACT_MIN_HEIGHT {
        Density::Compact
    } else {
        Density::Tiny
    }
}

/// Events within this long of the latest report keep the events strip open.
pub(super) const EVENT_WINDOW_MS: u64 = 10 * 60 * 1000;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.width < 30 || area.height < 6 {
        frame.render_widget(
            Paragraph::new("macgpu: terminal too small").style(style::dim()),
            area,
        );
        return;
    }
    let density = density(area);
    let who = WhoChart::rank(app);

    let header = header_lines(app, density, area.width);
    let header_height = header.len() as u16;
    let footer_y = area.y + area.height - 1;
    let mut y = area.y;
    frame.render_widget(
        Paragraph::new(header),
        Rect::new(area.x, y, area.width, header_height),
    );
    y += header_height;

    let body = Rect::new(area.x, y, area.width, footer_y.saturating_sub(y));
    match &app.screen {
        Screen::Detail(detail) => detail::draw(frame, body, app, detail),
        Screen::Main => draw_main(frame, body, app, density, &who),
    }

    draw_footer(frame, Rect::new(area.x, footer_y, area.width, 1), app);

    match &app.overlay {
        Overlay::Help { scroll } => help::draw(frame, area, *scroll),
        Overlay::Confirm(confirm) => draw_confirm(frame, area, confirm),
        Overlay::Filter { .. } | Overlay::None => {}
    }
}

fn draw_main(frame: &mut Frame, area: Rect, app: &mut App, density: Density, who: &WhoChart) {
    let events = event_lines(app, if density == Density::Full { 3 } else { 1 });
    let events_height = events.len() as u16;
    let history_height = match density {
        Density::Full => (area.height.saturating_sub(events_height) * 2 / 5).clamp(8, 14),
        Density::Compact if area.height >= 18 => 5,
        _ => 0,
    };
    // The table gets at least six rows before the charts get any.
    let history_height = if area.height.saturating_sub(history_height + events_height) < 6 {
        0
    } else {
        history_height
    };

    let mut y = area.y;
    if history_height > 0 && app.report.is_some() {
        let history = Rect::new(area.x, y, area.width, history_height);
        match density {
            Density::Full => graphs::draw_history_full(frame, history, app, who),
            _ => graphs::draw_history_compact(frame, history, app, who),
        }
        y += history_height;
    }
    if events_height > 0 {
        frame.render_widget(
            Paragraph::new(events),
            Rect::new(area.x, y, area.width, events_height),
        );
        y += events_height;
    }
    let table = Rect::new(
        area.x,
        y,
        area.width,
        (area.y + area.height).saturating_sub(y),
    );
    draw_table(frame, table, app, who, density);
}

// ---- footer and overlays ------------------------------------------------

/// Key, description, and whether the hint is only shown when signalling is on.
const HINTS: &[(&str, &str, bool)] = &[
    ("q", "quit", false),
    ("?", "help", false),
    ("/", "filter", false),
    ("Esc", "clear", false),
    ("g G", "group", false),
    ("t", "tree", false),
    ("s . ,", "sort", false),
    ("S r", "reverse", false),
    ("⏎", "detail", false),
    ("↑↓", "move", false),
    ("PgUp PgDn", "page", false),
    ("Home End", "ends", false),
    ("k", "signal", true),
    ("f space", "freeze", false),
    ("z Z", "zoom", false),
    ("+ -", "interval", false),
    ("a", "idle", false),
];

fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let line = if let Overlay::Filter { input, .. } = &app.overlay {
        Line::from(vec![
            Span::styled(" filter ", Style::new().fg(Color::Black).bg(style::WARNING)),
            Span::raw(" "),
            Span::styled(input.clone(), style::title()),
            Span::styled("▏", style::accent(style::WARNING)),
            Span::styled(
                "  name, path or PID; comma separates, ! excludes · Enter keep · Esc cancel · Ctrl-U clear · Ctrl-W word · Backspace",
                style::dim(),
            ),
        ])
    } else if let Some(status) = &app.status {
        let color = match status.level {
            StatusLevel::Info => LABEL,
            StatusLevel::Error => style::ALERT,
        };
        Line::from(Span::styled(
            format!(" {}", status.text),
            style::accent(color),
        ))
    } else {
        let mut spans = vec![Span::raw(" ")];
        let detail = matches!(app.screen, Screen::Detail(_));
        let hints: &[(&str, &str, bool)] = if detail {
            &[
                ("Esc ⏎ ← Backspace", "back", false),
                ("↑↓ PgUp PgDn", "scroll", false),
                ("Home End", "ends", false),
                ("k", "signal", true),
                ("z Z", "zoom", false),
                ("f space", "freeze", false),
                ("?", "help", false),
                ("q", "quit", false),
            ]
        } else {
            HINTS
        };
        for &(key, text, signal) in hints {
            if signal && app.readonly {
                continue;
            }
            spans.push(Span::styled(key, style::accent(GPU)));
            spans.push(Span::styled(format!(" {text}  "), style::dim()));
        }
        Line::from(spans)
    };
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_confirm(frame: &mut Frame, area: Rect, confirm: &Confirm) {
    let width = 60.min(area.width.saturating_sub(4));
    let height = 7.min(area.height);
    let popup = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    );
    let mut choices = vec![Span::raw(" ")];
    for (index, (_, name)) in SIGNALS.iter().enumerate() {
        let style = if index == confirm.signal {
            Style::new()
                .fg(Color::Black)
                .bg(style::ALERT)
                .add_modifier(Modifier::BOLD)
        } else {
            style::label()
        };
        choices.push(Span::styled(format!(" {name} "), style));
        choices.push(Span::raw(" "));
    }
    let lines = vec![
        Line::from(vec![
            Span::raw(" Send "),
            Span::styled(SIGNALS[confirm.signal].1, style::strong(style::ALERT)),
            Span::raw(" to "),
            Span::styled(confirm.name.clone(), style::title()),
            Span::styled(format!(" (pid {})?", confirm.pid), style::label()),
        ]),
        Line::raw(""),
        Line::from(choices),
        Line::raw(""),
        Line::from(Span::styled(
            " ←→ choose · Enter send · Esc cancel",
            style::dim(),
        )),
    ];
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(style::accent(style::ALERT))
        .title(Span::styled(" send signal ", style::strong(style::ALERT)));
    Clear.render(popup, frame.buffer_mut());
    frame.render_widget(Paragraph::new(lines).block(block), popup);
}
