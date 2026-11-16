//! The detail screen for one process, app, or the unattributed row: identity,
//! current numbers, history charts, and a per-client, per-queue, or per-member
//! breakdown.
use super::fmt;

use super::app::{App, Detail, span_label};
use super::charts::{self, BrailleSeries};
use super::history::{RowMetric, bucketize};
use super::style::{self, DIM, GPU, LABEL};
use crate::format;
use crate::model::{Activity, AppView, ProcessView, Report, SourceState};
use crate::view::RowKey;
use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

pub fn draw(frame: &mut Frame, area: Rect, app: &App, detail: &Detail) {
    let Some(report) = &app.report else {
        return;
    };
    let (kind, name) = match detail.key {
        RowKey::Process { pid, .. } => (
            "process",
            detail
                .process
                .as_ref()
                .map(|process| format!("{} ({pid})", process.name))
                .unwrap_or_else(|| pid.to_string()),
        ),
        RowKey::App { coalition_id } => (
            "app",
            detail
                .app
                .as_ref()
                .map(|app| app.name.clone())
                .unwrap_or_else(|| format!("coalition {coalition_id}")),
        ),
        _ => (
            "unattributed",
            "exited or unattributed GPU time".to_string(),
        ),
    };
    let mut title = vec![
        Span::styled(format!(" {kind} "), style::dim()),
        Span::styled(format!("{name} "), style::strong(GPU)),
    ];
    if detail.exited {
        title.push(Span::styled(
            " EXITED ",
            Style::new().fg(Color::Black).bg(style::ALERT),
        ));
        title.push(Span::raw(" "));
    }
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(style::dim())
        .title(Line::from(title))
        .title(
            Line::from(Span::styled(
                format!(
                    " history last {} · Esc/Backspace/Left back ",
                    span_label(app.zoom_span_ms())
                ),
                style::dim(),
            ))
            .right_aligned(),
        );
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 3 {
        return;
    }

    let info = match detail.key {
        RowKey::Process { .. } => process_info(report, detail),
        RowKey::App { .. } => app_info(report, detail),
        _ => unattributed_info(report),
    };
    let info_height = (info.len() as u16).min(inner.height / 2);
    frame.render_widget(
        Paragraph::new(info),
        Rect::new(
            inner.x + 1,
            inner.y,
            inner.width.saturating_sub(2),
            info_height,
        ),
    );

    let remaining = inner.height - info_height;
    let plot_width = inner.width.saturating_sub(2 + CHART_LABEL_WIDTH);
    let charts = chart_specs(app, detail, plot_width as usize * 2);
    let per_chart = if remaining >= 18 { 3 } else { 2 };
    let charts_height = (charts.len() as u16 * per_chart).min(remaining.saturating_sub(4));
    let chart_area = Rect::new(
        inner.x + 1,
        inner.y + info_height,
        inner.width.saturating_sub(2),
        charts_height,
    );
    draw_charts(frame.buffer_mut(), chart_area, charts, per_chart);

    let breakdown = match detail.key {
        RowKey::Process { .. } => process_breakdown(detail),
        RowKey::App { .. } => app_breakdown(report, detail),
        _ => Vec::new(),
    };
    let table_y = chart_area.y + chart_area.height;
    let table_height = (inner.y + inner.height).saturating_sub(table_y);
    if table_height > 0 && !breakdown.is_empty() {
        let max_scroll = breakdown.len().saturating_sub(table_height as usize) as u16;
        frame.render_widget(
            Paragraph::new(breakdown).scroll((detail.scroll.min(max_scroll), 0)),
            Rect::new(
                inner.x + 1,
                table_y,
                inner.width.saturating_sub(2),
                table_height,
            ),
        );
    }
}

fn field(name: &str) -> Span<'static> {
    Span::styled(format!("{name:<10}"), style::dim())
}

fn sep() -> Span<'static> {
    Span::styled(" · ", style::dim())
}

fn activity_text(activity: Activity, age: Option<u64>) -> Vec<Span<'static>> {
    let (text, color) = match activity {
        Activity::Active => ("active", GPU),
        Activity::Recent => ("recent", LABEL),
        Activity::Idle => ("idle", DIM),
    };
    let mut spans = vec![Span::styled(text, style::accent(color))];
    if let Some(age) = age {
        spans.push(Span::styled(
            format!(", last submit {}", format::age_ms(age)),
            style::dim(),
        ));
    }
    spans
}

fn process_info(report: &Report, detail: &Detail) -> Vec<Line<'static>> {
    let Some(process) = &detail.process else {
        return vec![Line::from(Span::styled(
            "no data for this process",
            style::dim(),
        ))];
    };
    let mut lines = Vec::new();
    lines.push(process_now(report, process));
    lines.push(process_activity(process));
    lines.extend(process_identity(report, detail, process));
    lines
}

fn process_now(report: &Report, process: &ProcessView) -> Line<'static> {
    let mut now = vec![field("now")];
    now.push(Span::styled(
        format!("{} ms/s", fmt::ms(process.busy_ms_per_s)),
        style::strong(GPU),
    ));
    now.push(sep());
    now.push(Span::styled(
        format!("share {}", fmt::pct(process.share_pct)),
        style::label(),
    ));
    if let Some(device) = process.device_pct {
        now.push(sep());
        now.push(Span::styled(
            format!("dev {}", fmt::pct(device)),
            style::label(),
        ));
    }
    if let Some(power) = process.est_power_w {
        now.push(sep());
        now.push(Span::styled(
            format!("~{} W", fmt::watts_digits(power)),
            style::accent(style::POWER),
        ));
        now.push(Span::styled(" estimated from app energy", style::dim()));
    }
    now.push(sep());
    match process.graphics_memory_bytes {
        Some(bytes) => now.push(Span::styled(
            format!("GPU mem {}", format::bytes(bytes)),
            style::accent(style::MEMORY),
        )),
        None => now.push(Span::styled(
            format!("GPU mem - ({})", memory_gap_reason(report)),
            style::dim(),
        )),
    }
    Line::from(now)
}

fn process_activity(process: &ProcessView) -> Line<'static> {
    let mut state = vec![field("activity")];
    state.extend(activity_text(process.activity, process.last_submit_age_ms));
    state.push(sep());
    state.push(Span::styled(
        format!(
            "{} {} on {} {}",
            process.queues,
            plural(process.queues, "queue", "queues"),
            process.clients.len(),
            plural(process.clients.len(), "client", "clients")
        ),
        style::label(),
    ));
    Line::from(state)
}

fn process_identity(report: &Report, detail: &Detail, process: &ProcessView) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some(path) = &process.path {
        lines.push(Line::from(vec![
            field("path"),
            Span::styled(path.clone(), style::label()),
        ]));
    }
    if let Some(app) = &detail.app {
        let mut spans = vec![
            field("app"),
            Span::styled(app.name.clone(), style::title()),
            Span::styled(format!(" (coalition {})", app.coalition_id), style::dim()),
        ];
        if let Some(bundle) = &app.bundle_path {
            spans.push(sep());
            spans.push(Span::styled(bundle.clone(), style::label()));
        }
        lines.push(Line::from(spans));
    } else if let Some(coalition_id) = process.coalition_id {
        lines.push(Line::from(vec![
            field("app"),
            Span::styled(format!("coalition {coalition_id}"), style::label()),
        ]));
    }

    if !detail.parents.is_empty() {
        let mut spans = vec![field("parents")];
        for (index, (pid, name)) in detail.parents.iter().enumerate() {
            if index > 0 {
                spans.push(Span::styled(" < ", style::dim()));
            }
            spans.push(Span::styled(name.clone(), style::label()));
            spans.push(Span::styled(format!(" ({pid})"), style::dim()));
        }
        lines.push(Line::from(spans));
    } else if let Some(ppid) = process.ppid {
        let name = report
            .processes
            .iter()
            .find(|candidate| candidate.pid == ppid)
            .map(|parent| format!("{} ", parent.name))
            .unwrap_or_default();
        lines.push(Line::from(vec![
            field("parent"),
            Span::styled(format!("{name}({ppid})"), style::label()),
        ]));
    }

    let started_ms = process.start_us / 1000;
    let mut identity = vec![field("lifetime")];
    if let Some(uid) = process.uid {
        identity.push(Span::styled(format!("uid {uid}"), style::label()));
        identity.push(sep());
    }
    identity.push(Span::styled(
        format!("started {}", fmt::clock(started_ms)),
        style::label(),
    ));
    if report.timestamp_ms > started_ms {
        identity.push(Span::styled(
            format!(
                " ({} ago)",
                fmt::process_age((report.timestamp_ms - started_ms) / 1000)
            ),
            style::dim(),
        ));
    }
    identity.push(sep());
    identity.push(Span::styled(
        format!(
            "GPU time {} since its clients opened",
            format::seconds(process.lifetime_gpu_ns as f64 / 1e9)
        ),
        style::label(),
    ));
    lines.push(Line::from(identity));
    lines
}

fn app_info(report: &Report, detail: &Detail) -> Vec<Line<'static>> {
    let Some(app) = &detail.app else {
        return vec![Line::from(Span::styled(
            "no data for this app",
            style::dim(),
        ))];
    };
    let mut lines = vec![app_now(app)];
    if let Some(energy) = app_billing(app) {
        lines.push(energy);
    }
    lines.extend(app_in_flight(app));
    lines.push(app_exited(report, app));
    lines.extend(app_identity(report, app));
    lines
}

fn app_now(app: &AppView) -> Line<'static> {
    let mut now = vec![
        field("now"),
        Span::styled(
            format!("{} ms/s", fmt::ms(app.busy_ms_per_s)),
            style::strong(GPU),
        ),
        sep(),
        Span::styled(format!("share {}", fmt::pct(app.share_pct)), style::label()),
    ];
    if let Some(device) = app.device_pct {
        now.push(sep());
        now.push(Span::styled(
            format!("dev {}", fmt::pct(device)),
            style::label(),
        ));
    }
    if let Some(power) = app.power_w {
        now.push(sep());
        now.push(Span::styled(
            format!("{} W", fmt::watts_digits(power)),
            style::strong(style::POWER),
        ));
        now.push(Span::styled(" measured", style::dim()));
    }
    if let Some(bytes) = app.graphics_memory_bytes {
        now.push(sep());
        now.push(Span::styled(
            format!("GPU mem {}", format::bytes(bytes)),
            style::accent(style::MEMORY),
        ));
    }
    Line::from(now)
}

fn app_billing(app: &AppView) -> Option<Line<'static>> {
    let mut energy = vec![field("billing")];
    let mut any = false;
    for (label, value) in [
        ("billed to it by others ", app.billed_to_me_w),
        ("spent for others ", app.billed_to_others_w),
    ] {
        // The driver may not bill across apps at all (0 on macOS 27).
        if let Some(value) = value
            && value > 0.0
        {
            if any {
                energy.push(sep());
            }
            any = true;
            energy.push(Span::styled(label, style::dim()));
            energy.push(Span::styled(
                format!("{} W", fmt::watts_digits(value)),
                style::accent(style::POWER),
            ));
        }
    }
    if let (Some(ane), Some(ane_power)) = (app.ane_ms_per_s, app.ane_power_w)
        && (ane > 0.0 || ane_power > 0.0)
    {
        if any {
            energy.push(sep());
        }
        any = true;
        energy.push(Span::styled(
            format!(
                "ANE {} ms/s, {} W",
                fmt::ms(ane),
                fmt::watts_digits(ane_power)
            ),
            style::label(),
        ));
    }
    if any {
        return Some(Line::from(energy));
    }
    None
}

fn app_in_flight(app: &AppView) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some(in_flight) = app.in_flight_ms_per_s {
        lines.push(Line::from(vec![
            field("in flight"),
            Span::styled(format!("{} ms/s", fmt::ms(in_flight)), style::accent(GPU)),
            Span::styled(
                "  wall time with GPU work queued; apps that overlap each count in full",
                style::dim(),
            ),
        ]));
    }
    lines
}

fn app_exited(report: &Report, app: &AppView) -> Line<'static> {
    let window_s = report.window_seconds.max(1e-9);
    let exited = match app.unattributed_gpu_ns {
        Some(gpu_ns) => vec![
            field("exited"),
            Span::styled(
                format!("{} ms/s", fmt::ms(gpu_ns as f64 / 1e6 / window_s)),
                style::label(),
            ),
            Span::styled(
                "  from clients that exited or that no live client accounts for",
                style::dim(),
            ),
        ],
        None => vec![
            field("exited"),
            Span::styled("-", style::dim()),
            Span::styled(
                "  members exited during overlapping GPU work, so it cannot be separated",
                style::dim(),
            ),
        ],
    };
    Line::from(exited)
}

fn app_identity(report: &Report, app: &AppView) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some(bundle) = &app.bundle_path {
        lines.push(Line::from(vec![
            field("bundle"),
            Span::styled(bundle.clone(), style::label()),
        ]));
    }
    if let Some(pid) = app.responsible_pid {
        let name = report
            .processes
            .iter()
            .find(|process| process.pid == pid)
            .map(|process| format!("{} ", process.name))
            .unwrap_or_default();
        lines.push(Line::from(vec![
            field("leader"),
            Span::styled(format!("{name}({pid})"), style::label()),
            sep(),
            Span::styled(format!("coalition {}", app.coalition_id), style::dim()),
        ]));
    }
    let mut lifetime = vec![
        field("lifetime"),
        Span::styled(
            format!(
                "GPU time {}",
                format::seconds(app.lifetime_gpu_ns as f64 / 1e9)
            ),
            style::label(),
        ),
    ];
    if let Some(energy) = app.lifetime_energy_j {
        lifetime.push(sep());
        lifetime.push(Span::styled(
            format!("GPU energy {}", joules(energy)),
            style::label(),
        ));
    }
    lifetime.push(sep());
    lifetime.push(Span::styled(
        format!(
            "{} members, {} using the GPU",
            app.members,
            app.gpu_pids.len()
        ),
        style::label(),
    ));
    lines.push(Line::from(lifetime));
    lines
}

/// Why a process has no GPU memory figure.
fn memory_gap_reason(report: &Report) -> String {
    match report.sources.get(crate::model::source::MEMORY) {
        Some(status) if matches!(status.state, SourceState::Ok | SourceState::Partial) => {
            "another user's process; root can read it".into()
        }
        Some(status) => {
            let state = format!("{:?}", status.state).to_lowercase();
            match &status.detail {
                Some(detail) => format!("memory source {state}: {detail}"),
                None => format!("memory source {state}"),
            }
        }
        None => "memory source not reported".into(),
    }
}

fn plural(count: usize, one: &'static str, many: &'static str) -> &'static str {
    if count == 1 { one } else { many }
}

fn joules(value: f64) -> String {
    if value >= 1e6 {
        format!("{:.1} MJ", value / 1e6)
    } else if value >= 1e3 {
        format!("{:.1} kJ", value / 1e3)
    } else {
        format!("{value:.1} J")
    }
}

fn unattributed_info(report: &Report) -> Vec<Line<'static>> {
    let now = report
        .unattributed
        .as_ref()
        .map(|unattributed| unattributed.busy_ms_per_s)
        .unwrap_or(0.0);
    let partial = report
        .unattributed
        .as_ref()
        .is_some_and(|unattributed| unattributed.partial);
    let mut now_line = vec![
        field("now"),
        Span::styled(format!("{} ms/s", fmt::ms(now)), style::strong(GPU)),
    ];
    if partial {
        now_line.push(Span::styled(
            " at least: some apps' exited time overlapped other GPU work and could not be separated",
            style::accent(style::WARNING),
        ));
    }
    vec![
        Line::from(now_line),
        Line::from(Span::styled(
            "GPU time the kernel's per-app (coalition) counters saw that no live GPU client accounts for.",
            style::label(),
        )),
        Line::from(Span::styled(
            "Usually work from clients or processes that exited inside the window; it is shown, not hidden or spread over other rows.",
            style::dim(),
        )),
    ]
}

struct ChartSpec {
    name: &'static str,
    value: String,
    color: Color,
    values: Vec<Option<f64>>,
    max: f64,
    scale: String,
}

const CHART_LABEL_WIDTH: u16 = 16;

/// The charts for the detail subject, with `dots` values each (two per cell).
fn chart_specs(app: &App, detail: &Detail, dots: usize) -> Vec<ChartSpec> {
    let Some(end) = app.history.latest_ms() else {
        return Vec::new();
    };
    if dots == 0 {
        return Vec::new();
    }
    let span = app.zoom_span_ms();
    let coverage = app.history.coverage(end, span, dots);
    let key = detail.key;
    let peak = |values: &[Option<f64>]| values.iter().flatten().copied().fold(0.0, f64::max);
    let mut specs = Vec::new();

    let busy = app
        .history
        .row_buckets(&key, RowMetric::Busy, end, span, &coverage);
    let current_busy = match key {
        RowKey::Process { .. } => detail.process.as_ref().map(|process| process.busy_ms_per_s),
        RowKey::App { .. } => detail.app.as_ref().map(|app| app.busy_ms_per_s),
        _ => app
            .report
            .as_ref()
            .and_then(|report| report.unattributed.as_ref())
            .map(|unattributed| unattributed.busy_ms_per_s),
    };
    let max = fmt::nice_ceiling(peak(&busy).max(10.0));
    specs.push(ChartSpec {
        name: "GPU ms/s",
        value: current_busy.map(fmt::ms).unwrap_or_else(|| "-".into()),
        color: GPU,
        values: busy,
        max,
        scale: format!("scale {max:.0}"),
    });

    let (power_now, estimated) = match key {
        RowKey::Process { .. } => (
            detail
                .process
                .as_ref()
                .and_then(|process| process.est_power_w),
            true,
        ),
        RowKey::App { .. } => (detail.app.as_ref().and_then(|app| app.power_w), false),
        _ => (None, false),
    };
    if let Some(power) = power_now {
        let values = app
            .history
            .row_buckets(&key, RowMetric::Power, end, span, &coverage);
        let max = fmt::nice_ceiling(peak(&values).max(0.5));
        specs.push(ChartSpec {
            name: if estimated { "watts (est.)" } else { "watts" },
            value: format!(
                "{}{} W",
                if estimated { "~" } else { "" },
                fmt::watts_digits(power)
            ),
            color: style::POWER,
            values,
            max,
            scale: format!("scale {max} W"),
        });
    }

    let memory_now = match key {
        RowKey::Process { .. } => detail
            .process
            .as_ref()
            .and_then(|process| process.graphics_memory_bytes),
        RowKey::App { .. } => detail
            .app
            .as_ref()
            .and_then(|app| app.graphics_memory_bytes),
        _ => None,
    };
    if let (Some(bytes), Some(series)) = (memory_now, app.history.rows.get(&key)) {
        let raw = series.memory_buckets(end, span, dots);
        // Only where reports covered the bucket, so the start of history stays blank.
        let covered = bucketize(
            app.history
                .device_samples(|_| Some(1.0), end.saturating_sub(span)),
            end,
            span,
            dots,
        );
        let values: Vec<Option<f64>> = raw
            .into_iter()
            .zip(covered)
            .map(|(value, covered)| covered.and(value))
            .collect();
        let max = fmt::bytes_ceiling(peak(&values));
        specs.push(ChartSpec {
            name: "GPU memory",
            value: format::bytes(bytes),
            color: style::MEMORY,
            values,
            max,
            scale: format!("scale {}", fmt::bytes_short(max as u64)),
        });
    }
    specs
}

fn draw_charts(buffer: &mut Buffer, area: Rect, specs: Vec<ChartSpec>, per_chart: u16) {
    if area.height < 2 || area.width < 30 {
        return;
    }
    let label_width = CHART_LABEL_WIDTH;
    let plot_width = area.width.saturating_sub(label_width);
    let mut y = area.y;
    for spec in specs {
        if y + per_chart > area.y + area.height {
            break;
        }
        buffer.set_stringn(area.x, y, spec.name, label_width as usize - 1, style::dim());
        buffer.set_stringn(
            area.x,
            y + 1,
            &spec.value,
            label_width as usize - 1,
            style::strong(spec.color),
        );
        if per_chart >= 3 {
            buffer.set_stringn(
                area.x,
                y + 2,
                &spec.scale,
                label_width as usize - 1,
                style::dim(),
            );
        }
        let plot = Rect::new(area.x + label_width, y, plot_width, per_chart);
        charts::braille(
            buffer,
            plot,
            &[BrailleSeries {
                values: &spec.values,
                color: spec.color,
                fill: true,
            }],
            spec.max,
        );
        y += per_chart;
    }
}

fn process_breakdown(detail: &Detail) -> Vec<Line<'static>> {
    let Some(process) = &detail.process else {
        return Vec::new();
    };
    let mut lines = vec![Line::from(Span::styled(
        format!(
            "{:<22} {:<8} {:>9} {:>9} {:>7}  {}",
            "CLIENT / QUEUE", "API", "GPU ms/s", "LIFETIME", "LAST", "QUEUES (now/max)"
        ),
        style::label().add_modifier(Modifier::BOLD),
    ))];
    for client in &process.clients {
        let focused = detail.focus
            == Some(RowKey::Client {
                registry_id: client.registry_id,
            });
        let queues = match (client.command_queue_count, client.command_queue_count_max) {
            (Some(now), Some(max)) => format!("{now}/{max}"),
            (Some(now), None) => now.to_string(),
            _ => "-".into(),
        };
        let text = format!(
            "{:<22} {:<8} {:>9} {:>9} {:>7}  {}",
            format!("client {:#x}", client.registry_id),
            "",
            fmt::ms(client.busy_ms_per_s),
            format::seconds(client.lifetime_gpu_ns as f64 / 1e9),
            client
                .last_submit_age_ms
                .map(format::age_ms)
                .unwrap_or_else(|| "-".into()),
            queues
        );
        lines.push(highlight(
            Line::from(Span::styled(text, style::title())),
            focused,
        ));
        for (index, queue) in client.queues.iter().enumerate() {
            let focused = detail.focus
                == Some(RowKey::Queue {
                    registry_id: client.registry_id,
                    index: queue.index,
                });
            let branch = if index + 1 == client.queues.len() {
                "└─"
            } else {
                "├─"
            };
            let glyph = match queue.activity {
                Activity::Active => "●",
                Activity::Recent => "◦",
                Activity::Idle => " ",
            };
            let line = Line::from(vec![
                Span::styled(format!("  {branch} "), style::dim()),
                Span::styled(
                    format!("{:<17} ", format!("queue {}", queue.index)),
                    style::label(),
                ),
                Span::styled(
                    format!("{:<8} ", format::truncate(&queue.api, 8)),
                    style::dim(),
                ),
                Span::styled(
                    format!("{:>9} ", fmt::ms(queue.busy_ms_per_s)),
                    if queue.busy_ms_per_s > 0.0 {
                        style::accent(GPU)
                    } else {
                        style::dim()
                    },
                ),
                Span::styled(
                    format!(
                        "{:>9} ",
                        format::seconds(queue.lifetime_gpu_ns as f64 / 1e9)
                    ),
                    style::dim(),
                ),
                Span::styled(
                    format!(
                        "{glyph} {:>5}",
                        queue
                            .last_submit_age_ms
                            .map(format::age_ms)
                            .unwrap_or_else(|| "-".into())
                    ),
                    style::label(),
                ),
            ]);
            lines.push(highlight(line, focused));
        }
    }
    lines
}

fn app_breakdown(report: &Report, detail: &Detail) -> Vec<Line<'static>> {
    let Some(app) = &detail.app else {
        return Vec::new();
    };
    let members: Vec<&ProcessView> = report
        .processes
        .iter()
        .filter(|process| process.coalition_id == Some(app.coalition_id))
        .collect();
    let mut lines = vec![Line::from(Span::styled(
        format!(
            "{:>7}  {:<32} {:>9} {:>7} {:>7} {:>8} {:>7}",
            "PID", "GPU PROCESSES IN THIS APP", "GPU ms/s", "SHARE", "WATTS", "GPU MEM", "LAST"
        ),
        style::label().add_modifier(Modifier::BOLD),
    ))];
    if members.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no member process has a GPU client right now (a shows idle ones)",
            style::dim(),
        )));
    }
    for process in members {
        lines.push(member_line(process));
    }
    lines
}

fn member_line(process: &ProcessView) -> Line<'static> {
    let dash = || "-".to_string();
    Line::from(vec![
        Span::styled(format!("{:>7}  ", process.pid), style::label()),
        Span::styled(
            format!("{:<32} ", format::truncate(&process.name, 32)),
            style::title(),
        ),
        Span::styled(
            format!("{:>9} ", fmt::ms(process.busy_ms_per_s)),
            if process.busy_ms_per_s > 0.0 {
                style::accent(GPU)
            } else {
                style::dim()
            },
        ),
        Span::styled(
            format!("{:>7} ", fmt::pct(process.share_pct)),
            style::label(),
        ),
        Span::styled(
            format!(
                "{:>7} ",
                process
                    .est_power_w
                    .map(|watts| format!("~{}", fmt::watts_digits(watts)))
                    .unwrap_or_else(dash)
            ),
            style::accent(style::POWER),
        ),
        Span::styled(
            format!(
                "{:>8} ",
                process
                    .graphics_memory_bytes
                    .map(fmt::bytes_short)
                    .unwrap_or_else(dash)
            ),
            style::accent(style::MEMORY),
        ),
        Span::styled(
            format!(
                "{:>7}",
                process
                    .last_submit_age_ms
                    .map(format::age_ms)
                    .unwrap_or_else(dash)
            ),
            style::dim(),
        ),
    ])
}

fn highlight(line: Line<'static>, focused: bool) -> Line<'static> {
    if focused {
        line.style(Style::new().bg(style::SELECTED_BG))
    } else {
        line
    }
}
