//! The `?` overlay: every key, and what every number means.

use super::style::{self, GPU};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Widget, Wrap};

const KEYS: &[(&str, &str)] = &[
    ("q  Ctrl-C", "quit"),
    ("?  F1", "this help (Esc closes)"),
    (
        "↑ ↓ PgUp PgDn Home End",
        "move the selection; mouse click selects, wheel scrolls",
    ),
    (
        "Enter  double-click",
        "detail screen for the selected row; Esc goes back",
    ),
    (
        "/",
        "filter: comma-separated name, path or PID substrings; !term excludes (chrome, !helper)",
    ),
    ("Esc", "clear the filter"),
    (
        "s  .  >",
        "next sort column (click a column header to sort by it)",
    ),
    (",  <", "previous sort column"),
    ("S  r", "reverse the sort"),
    (
        "g  G",
        "group by queue, client, process, or app (next, previous)",
    ),
    (
        "t",
        "tree: processes under their app, or clients and queues under their process",
    ),
    (
        "f  space",
        "freeze the display; samples are kept and applied when you resume",
    ),
    ("z  Z", "history span: 1m, 5m, 15m, 30m, 1h"),
    ("+  -", "sampling interval longer or shorter (live only)"),
    ("a", "show or hide GPU clients that were idle in the window"),
    (
        "k",
        "send a signal to the selected process (asks first; off with --readonly)",
    ),
];

const METRICS: &[(&str, &str)] = &[
    (
        "GPU ms/s",
        "share of GPU busy time, in ms per second. The driver apportions busy time among contexts that run at once, so rows at one level sum to about 1000 when the GPU is saturated. Never clamped or rescaled.",
    ),
    (
        "SHARE",
        "the row's part of all attributed GPU time in the window. Sums to 100% across rows at one level.",
    ),
    (
        "DEV%",
        "device-normalized: SHARE times the GPU's active %, so rows sum to the device's active time. Needs IOReport.",
    ),
    (
        "active %",
        "share of the window the GPU spent in any active P-state (IOReport GPUPH), integrated over the whole window, not a point sample.",
    ),
    (
        "norm %",
        "frequency-normalized: active % times average clock / max clock. Low when the GPU is busy at a low clock.",
    ),
    (
        "WATTS",
        "app rows: GPU power measured from the kernel's per-app GPU energy counter. ~ marks an estimate: process rows split their app's measured energy by GPU time.",
    ),
    (
        "unattributed",
        "GPU time the per-app counters saw that no live client accounts for, usually from clients that exited inside the window (the EXITED column for apps). An app row is its processes plus this time. When members exited while the app's work overlapped other GPU work it cannot be separated: EXITED shows - and the total is a lower bound (shown with ≥).",
    ),
    (
        "in flight",
        "app detail only: wall time the app had GPU work queued, from the kernel's per-app counter. Apps whose work overlaps each count in full, so it can exceed the app's GPU ms/s and the apps together can exceed 1000.",
    ),
    (
        "GPU MEM",
        "the process's graphics memory footprint. - for other users' processes, which need root.",
    ),
    (
        "LAST",
        "time since the last submission: ● used the GPU this window, ◦ within the last 10 s, blank idle.",
    ),
    (
        "HISTORY",
        "GPU ms/s per sample, newest on the right, on a fixed 0-1000 scale; any use shows at least one block.",
    ),
    (
        "who chart",
        "the top apps (or processes) stacked by device % (ms/s without IOReport), averaged per column; legend shows the average over the span. ■ in the table marks the same rows.",
    ),
    (
        "limiter",
        "the GPU power controller's states (IOReport PWRCTRL) as a share of powered time, shown when a state other than the demand-following PERF, PRFBOOST, DEADLINE, or DEADAWAR is in effect, for example PWR_ZONE or FAST_DIE. A limiter state does not always lower the clock. Apple documents none of these names; readings such as \"FAST_DIE (die temp?)\" are guesses. The strip also shows thermal (CLTM) limiting, the share of the window with a thermal cap in force, and throttle and thermal activation counters.",
    ),
    (
        "cpu",
        "macgpu's own CPU use, in percent of one core, sampler and display included.",
    ),
];

/// Help lines for a popup `width` cells wide, with descriptions wrapped and
/// indented under their own column.
pub fn lines(width: u16) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        "Keys",
        style::strong(GPU).add_modifier(Modifier::UNDERLINED),
    ))];
    for (key, text) in KEYS {
        push_wrapped(
            &mut lines,
            Span::styled(format!("  {key:<24}"), style::accent(GPU)),
            text,
            width,
        );
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "What the numbers mean",
        style::strong(GPU).add_modifier(Modifier::UNDERLINED),
    )));
    for (name, text) in METRICS {
        push_wrapped(
            &mut lines,
            Span::styled(format!("  {name:<14}"), style::title()),
            text,
            width,
        );
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "A value shown as - means its source was unavailable, never zero. Source problems appear under the header.",
        style::dim(),
    )));
    lines
}

fn push_wrapped(lines: &mut Vec<Line<'static>>, head: Span<'static>, text: &str, width: u16) {
    let indent = head.width();
    let room = (width as usize).saturating_sub(indent).max(20);
    let mut current = String::new();
    let mut first = Some(head);
    let mut flush = |current: &mut String, lines: &mut Vec<Line<'static>>| {
        let head = first
            .take()
            .unwrap_or_else(|| Span::raw(" ".repeat(indent)));
        lines.push(Line::from(vec![
            head,
            Span::styled(std::mem::take(current), style::label()),
        ]));
    };
    for word in text.split(' ') {
        if !current.is_empty() && current.chars().count() + 1 + word.chars().count() > room {
            flush(&mut current, lines);
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    flush(&mut current, lines);
}

pub fn draw(frame: &mut Frame, area: Rect, scroll: u16) {
    let width = area.width.saturating_sub(4).min(110);
    let height = area.height.saturating_sub(2);
    let popup = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(style::accent(GPU))
        .title(Span::styled(" macgpu help ", style::strong(GPU)))
        .title(Line::from(Span::styled(" ↑↓ scroll · Esc close ", style::dim())).right_aligned());
    Clear.render(popup, frame.buffer_mut());
    frame.render_widget(
        Paragraph::new(lines(width.saturating_sub(2)))
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        popup,
    );
}
