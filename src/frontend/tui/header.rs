//! The header meters and the dynamic events strip.
use super::fmt;

use super::app::{App, interval_label};
use super::charts::{self};
use super::style::{self, GPU};
use crate::format;
use crate::model::{EventKind, Report, SourceState};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::ui::Density;

use super::ui::EVENT_WINDOW_MS;
pub(super) fn sep() -> Span<'static> {
    Span::styled(" · ", style::dim())
}

pub(super) fn badge(text: &str, fg: Color, bg: Color) -> Span<'static> {
    Span::styled(
        format!(" {text} "),
        Style::new().fg(fg).bg(bg).add_modifier(Modifier::BOLD),
    )
}

fn feed_badge(app: &App) -> Span<'static> {
    let interval = interval_label(app.interval);
    match (app.live, app.feed_ended) {
        (true, false) => badge(&format!("LIVE {interval}"), Color::Black, Color::Green),
        (true, true) => badge("SAMPLER STOPPED", Color::White, style::ALERT),
        (false, false) => badge(&format!("REPLAY {interval}"), Color::Black, Color::Magenta),
        (false, true) => badge("REPLAY ENDED", Color::Black, Color::Magenta),
    }
}

pub(super) fn header_lines(app: &App, density: Density, width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let report = app.report.as_ref();

    // Title line, with the time and macgpu's own cost on the right.
    let mut left = vec![Span::styled(" macgpu ", style::strong(GPU))];
    if let Some(device) = report.map(|report| &report.device)
        && density != Density::Tiny
        && let Some(model) = &device.model
    {
        let cores = device
            .gpu_cores
            .map(|cores| format!(" {cores}-core GPU"))
            .unwrap_or_default();
        left.push(Span::styled(format!("{model}{cores}  "), style::label()));
    }
    left.push(feed_badge(app));
    if app.frozen {
        let pending = app.pending_reports();
        let text = if pending > 0 {
            format!("FROZEN +{pending}")
        } else {
            "FROZEN".into()
        };
        left.push(Span::raw(" "));
        left.push(badge(&text, Color::Black, Color::Yellow));
    }
    if app.alert_firing {
        left.push(Span::raw(" "));
        left.push(badge("ALERT", Color::White, style::ALERT));
    }
    if density != Density::Full {
        left.extend(compact_gpu_spans(report));
    }
    let right = match report {
        Some(report) => format!(
            "cpu {:.1}%  {} ",
            report.overhead.process_cpu_pct,
            fmt::clock(report.timestamp_ms)
        ),
        None => "waiting for the first sample ".into(),
    };
    lines.push(right_aligned(
        left,
        Span::styled(right, style::dim()),
        width,
    ));

    let Some(report) = report else {
        return lines;
    };
    match density {
        Density::Full => {
            lines.push(gpu_line(report));
            lines.push(memory_line(report, width));
            lines.push(usage_line(report, width));
        }
        Density::Compact => lines.push(compact_context_line(report)),
        Density::Tiny => {}
    }
    if density != Density::Tiny
        && let Some(problems) = problems_line(report)
    {
        lines.push(problems);
    }
    lines
}

/// `left` and `right` on one line, `right` dropped when it does not fit.
fn right_aligned(mut left: Vec<Span<'static>>, right: Span<'static>, width: u16) -> Line<'static> {
    let used: usize = left.iter().map(Span::width).sum();
    let room = width as usize;
    if used + right.width() < room {
        left.push(Span::raw(" ".repeat(room - used - right.width())));
        left.push(right);
    }
    Line::from(left)
}

fn gpu_line(report: &Report) -> Line<'static> {
    let device = &report.device;
    let mut spans = vec![Span::styled(" GPU ", style::label())];
    match device.active_pct {
        Some(active) => {
            spans.extend(charts::gauge(active / 100.0, 20, GPU));
            spans.push(Span::styled(
                format!(" {:>5}", fmt::pct(active)),
                style::strong(GPU),
            ));
            spans.push(Span::styled(" active", style::dim()));
        }
        None => spans.push(Span::styled("active -", style::dim())),
    }
    if let Some(normalized) = device.freq_normalized_pct {
        spans.push(Span::styled("  norm ", style::dim()));
        spans.push(Span::styled(
            fmt::pct(normalized),
            style::accent(style::NORMALIZED),
        ));
    }
    if let Some(mhz) = device.avg_mhz {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!("{mhz:.0}"),
            style::accent(style::CLOCK),
        ));
        let max = device
            .max_mhz
            .map(|max| format!("/{max:.0}"))
            .unwrap_or_default();
        spans.push(Span::styled(format!("{max} MHz"), style::dim()));
    }
    if let Some(power) = device.power_w {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format::watts(power),
            style::accent(style::POWER),
        ));
    }
    if let Some(temperature) = &device.temperature {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!("{:.0}°C", temperature.avg_c),
            style::accent(style::TEMPERATURE),
        ));
        spans.push(Span::styled(
            format!(" max {:.0}", temperature.max_c),
            style::dim(),
        ));
    }
    if let Some(limit) = limiter_text(report) {
        spans.push(Span::styled("  limiter ", style::dim()));
        spans.push(Span::styled(limit, style::accent(style::ALERT)));
    }
    Line::from(spans)
}

fn memory_line(report: &Report, width: u16) -> Line<'static> {
    let device = &report.device;
    let memory = &device.memory;
    let mut spans = vec![Span::styled(" MEM ", style::label())];
    // The gauge is against the wired limit, or physical memory when the OS
    // default limit applies (its value is not exposed).
    let ceiling = memory.wired_limit_bytes.or(memory.total_physical_bytes);
    match (memory.in_use_bytes, ceiling) {
        (Some(in_use), Some(limit)) if limit > 0 => {
            spans.extend(charts::gauge(
                in_use as f64 / limit as f64,
                20,
                style::MEMORY,
            ));
            spans.push(Span::raw(" "));
        }
        _ => {}
    }

    let in_use = memory.in_use_bytes.map(|bytes| {
        vec![
            Span::styled(format::bytes(bytes), style::strong(style::MEMORY)),
            Span::styled(" in use", style::dim()),
        ]
    });
    let allocated = memory.allocated_bytes.map(|bytes| {
        vec![
            Span::styled(format::bytes(bytes), style::accent(style::MEMORY)),
            Span::styled(" allocated", style::dim()),
        ]
    });
    let limit = |short: bool| match (memory.wired_limit_bytes, memory.total_physical_bytes) {
        (Some(limit), _) => Some(vec![
            Span::styled("wired limit ", style::dim()),
            Span::styled(format::bytes(limit), style::label()),
        ]),
        (None, Some(total)) if short => Some(vec![
            Span::styled("of ", style::dim()),
            Span::styled(format::bytes(total), style::label()),
            Span::styled(" RAM", style::dim()),
        ]),
        (None, Some(total)) => Some(vec![
            Span::styled("wired limit default · ", style::dim()),
            Span::styled(format::bytes(total), style::label()),
            Span::styled(" RAM", style::dim()),
        ]),
        (None, None) => None,
    };

    let mut pressure = Vec::new();
    let system = &device.system;
    if let Some(thermal) = &system.thermal_pressure {
        pressure.push(Span::styled("   thermal ", style::dim()));
        pressure.push(Span::styled(thermal.clone(), pressure_style(thermal)));
    }
    if let Some(level) = &system.memory_pressure {
        pressure.push(Span::styled("  memory ", style::dim()));
        pressure.push(Span::styled(level.clone(), pressure_style(level)));
    }

    // Drop the least useful parts until the line fits.
    let candidates = [
        vec![in_use.clone(), allocated.clone(), limit(false)],
        vec![in_use.clone(), allocated.clone(), limit(true)],
        vec![in_use.clone(), allocated.clone()],
        vec![in_use.clone()],
    ];
    let width_of = |spans: &[Span]| spans.iter().map(Span::width).sum::<usize>();
    let fixed = width_of(&spans) + width_of(&pressure);
    let mut chosen = Vec::new();
    for candidate in candidates {
        let mut parts = Vec::new();
        for (index, part) in candidate.into_iter().flatten().enumerate() {
            if index > 0 {
                parts.push(sep());
            }
            parts.extend(part);
        }
        let fits = fixed + width_of(&parts) <= width as usize;
        chosen = parts;
        if fits {
            break;
        }
    }
    if chosen.is_empty() {
        chosen.push(Span::styled("-", style::dim()));
    }
    spans.extend(chosen);
    spans.extend(pressure);
    Line::from(spans)
}

fn usage_line(report: &Report, width: u16) -> Line<'static> {
    let totals = &report.totals;
    let mut spans = vec![
        Span::styled(" USE ", style::label()),
        Span::styled(fmt::ms(totals.attributed_ms_per_s), style::strong(GPU)),
        Span::styled(" ms/s attributed", style::dim()),
        sep(),
        Span::styled(totals.active_processes.to_string(), style::label()),
        Span::styled(
            format!("/{} processes active", totals.gpu_processes),
            style::dim(),
        ),
    ];
    if totals.active_apps > 0 {
        spans.push(sep());
        spans.push(Span::styled(totals.active_apps.to_string(), style::label()));
        spans.push(Span::styled(" apps", style::dim()));
    }
    if let Some(unattributed) = &report.unattributed
        && (unattributed.busy_ms_per_s >= 0.05 || unattributed.partial)
    {
        spans.push(sep());
        spans.push(Span::styled("unattributed ", style::dim()));
        // Partial: some apps' share could not be separated, so this is a floor.
        let floor = if unattributed.partial { "≥" } else { "" };
        spans.push(Span::styled(
            format!("{floor}{}", fmt::ms(unattributed.busy_ms_per_s)),
            style::label(),
        ));
        spans.push(Span::styled(" ms/s", style::dim()));
    }
    let mut right = Vec::new();
    let pstates = &report.device.pstates;
    if !pstates.is_empty() {
        let off = pstates
            .iter()
            .find(|state| state.name.eq_ignore_ascii_case("OFF"))
            .map(|state| state.pct);
        let active: Vec<f64> = pstates
            .iter()
            .filter(|state| !state.name.eq_ignore_ascii_case("OFF"))
            .map(|state| state.pct)
            .collect();
        let max = active.iter().copied().fold(0.0, f64::max);
        right.push(Span::styled("P-states ", style::dim()));
        right.push(Span::styled(
            charts::histogram(&active, max),
            style::accent(style::CLOCK),
        ));
        if let Some(off) = off {
            right.push(Span::styled(format!(" off {off:.0}%"), style::dim()));
        }
        right.push(Span::raw(" "));
    }
    let right_width: usize = right.iter().map(Span::width).sum();
    if let Some(firmware) = &report.device.firmware_utilization {
        let text = format!(
            "fw dev {:.0}% ren {:.0}% tiler {:.0}%",
            firmware.device_pct, firmware.renderer_pct, firmware.tiler_pct
        );
        let used: usize = spans.iter().map(Span::width).sum();
        if used + 3 + text.chars().count() + right_width + 2 <= width as usize {
            spans.push(sep());
            spans.push(Span::styled(text, style::dim()));
        }
    }
    let used: usize = spans.iter().map(Span::width).sum();
    if !right.is_empty() && used + right_width + 2 <= width as usize {
        spans.push(Span::raw(" ".repeat(width as usize - used - right_width)));
        spans.extend(right);
    }
    Line::from(spans)
}

fn compact_gpu_spans(report: Option<&Report>) -> Vec<Span<'static>> {
    let Some(report) = report else {
        return Vec::new();
    };
    let device = &report.device;
    let mut spans = Vec::new();
    if let Some(active) = device.active_pct {
        spans.push(Span::styled("  GPU ", style::label()));
        spans.push(Span::styled(fmt::pct(active), style::strong(GPU)));
    } else {
        spans.push(Span::styled("  ", style::label()));
        spans.push(Span::styled(
            format!("{} ms/s", fmt::ms(report.totals.attributed_ms_per_s)),
            style::strong(GPU),
        ));
    }
    if let Some(mhz) = device.avg_mhz {
        spans.push(Span::styled(
            format!(" {mhz:.0}MHz"),
            style::accent(style::CLOCK),
        ));
    }
    if let Some(power) = device.power_w {
        spans.push(Span::styled(
            format!(" {}W", fmt::watts_digits(power)),
            style::accent(style::POWER),
        ));
    }
    if let Some(temperature) = &device.temperature {
        spans.push(Span::styled(
            format!(" {:.0}°C", temperature.avg_c),
            style::accent(style::TEMPERATURE),
        ));
    }
    spans
}

fn compact_context_line(report: &Report) -> Line<'static> {
    let device = &report.device;
    let mut spans = vec![Span::raw(" ")];
    if let Some(in_use) = device.memory.in_use_bytes {
        spans.push(Span::styled("mem ", style::dim()));
        spans.push(Span::styled(
            fmt::bytes_short(in_use),
            style::accent(style::MEMORY),
        ));
        if let Some(limit) = device.memory.wired_limit_bytes {
            spans.push(Span::styled(
                format!("/{}", fmt::bytes_short(limit)),
                style::dim(),
            ));
        }
        spans.push(sep());
    }
    if device.active_pct.is_some() {
        spans.push(Span::styled(
            fmt::ms(report.totals.attributed_ms_per_s),
            style::accent(GPU),
        ));
        spans.push(Span::styled(" ms/s", style::dim()));
        spans.push(sep());
    }
    spans.push(Span::styled(
        format!(
            "{}/{} active",
            report.totals.active_processes, report.totals.gpu_processes
        ),
        style::label(),
    ));
    if let Some(thermal) = &device.system.thermal_pressure
        && thermal != "nominal"
    {
        spans.push(sep());
        spans.push(Span::styled(
            format!("thermal {thermal}"),
            pressure_style(thermal),
        ));
    }
    if let Some(pressure) = &device.system.memory_pressure
        && pressure != "normal"
    {
        spans.push(sep());
        spans.push(Span::styled(
            format!("memory {pressure}"),
            pressure_style(pressure),
        ));
    }
    if let Some(limit) = limiter_text(report) {
        spans.push(sep());
        spans.push(Span::styled(
            format!("limiter {limit}"),
            style::accent(style::ALERT),
        ));
    }
    Line::from(spans)
}

fn problems_line(report: &Report) -> Option<Line<'static>> {
    let problems: Vec<String> = report
        .sources
        .iter()
        .filter(|(_, status)| !matches!(status.state, SourceState::Ok | SourceState::Disabled))
        .map(|(name, status)| {
            let state = match status.state {
                SourceState::Partial => "partial",
                SourceState::Stale => "stale",
                SourceState::Denied => "denied",
                SourceState::Unavailable => "unavailable",
                SourceState::Ok | SourceState::Disabled => "",
            };
            match &status.detail {
                Some(detail) => format!("{name} {state} ({detail})"),
                None => format!("{name} {state}"),
            }
        })
        .collect();
    if problems.is_empty() {
        return None;
    }
    Some(Line::from(vec![
        Span::styled(" ! ", style::strong(style::WARNING)),
        Span::styled(problems.join(" · "), style::accent(style::WARNING)),
    ]))
}

fn pressure_style(level: &str) -> Style {
    match level {
        "nominal" | "normal" => style::label(),
        "moderate" | "warning" => style::accent(style::WARNING),
        _ => style::strong(style::ALERT),
    }
}

const THROTTLE_PREFIX: &str = "GPU Throttler Counters/";
const THROTTLE_TOTAL: &str = "GPU Throttler Counters/Throttle Counter Total";
const THM_ACTIVATIONS: &str = "THM Counters/THM Activation Count";

/// A plain-words reading of a `PWRCTRL` state name. Apple documents none of
/// these, so they are guesses from the names and are shown with a "?".
fn limiter_guess(name: &str) -> Option<&'static str> {
    Some(match name {
        "PWR_ZONE" => "power zone",
        "POWER" => "power",
        "FAST_DIE" => "die temp",
        "CLTMLIMI" => "thermal (CLTM)",
        "PPMLIMIT" | "PPMTARGE" => "power mgmt",
        "PRFLIMIT" => "perf limit",
        _ => return None,
    })
}

/// The power controller's limiting reason, if anything other than normal
/// performance control is in effect.
fn limiter_text(report: &Report) -> Option<String> {
    let limits: Vec<String> = report
        .device
        .limiting_states()
        .filter(|limiter| limiter.pct >= 1.0)
        .take(2)
        .map(|limiter| format!("{} {:.0}%", limiter.name, limiter.pct))
        .collect();
    (!limits.is_empty()).then(|| limits.join(", "))
}

// ---- events strip -------------------------------------------------------

pub(super) fn event_lines(app: &App, max: usize) -> Vec<Line<'static>> {
    let Some(report) = &app.report else {
        return Vec::new();
    };
    let mut lines = Vec::new();

    let device = &report.device;
    let mut throttle = Vec::new();
    let states: Vec<String> = device
        .limiting_states()
        .filter(|limiter| limiter.pct >= 1.0)
        .take(2)
        .map(|limiter| match limiter_guess(&limiter.name) {
            Some(guess) => format!("{} ({guess}?) {:.0}%", limiter.name, limiter.pct),
            None => format!("{} {:.0}%", limiter.name, limiter.pct),
        })
        .collect();
    if !states.is_empty() {
        throttle.push(format!("{} of powered time", states.join(", ")));
        if let Some(limited) = device.limited_pct() {
            throttle.push(format!("limited about {limited:.0}% of the window"));
        }
    }
    if let Some(thermal) = device.thermal_limited_pct
        && thermal >= 0.5
    {
        throttle.push(format!("thermal (CLTM) limited {thermal:.0}%"));
    }
    // Issue counters are left out: they also arrive as GPU issue events.
    let counter = |key: &str| device.counters.get(key).copied().unwrap_or(0);
    let throttles = match device.counters.get(THROTTLE_TOTAL) {
        Some(total) => *total,
        None => device
            .counters
            .iter()
            .filter(|(key, _)| key.starts_with(THROTTLE_PREFIX))
            .map(|(_, count)| *count)
            .sum(),
    };
    if throttles > 0 {
        throttle.push(format!("throttle counter +{throttles}"));
    }
    let activations = counter(THM_ACTIVATIONS);
    if activations > 0 {
        throttle.push(format!("thermal activations +{activations}"));
    }
    if !throttle.is_empty() {
        lines.push(Line::from(vec![
            Span::styled(
                " LIMITER  ",
                Style::new().fg(Color::Black).bg(style::WARNING),
            ),
            Span::styled(
                format!(" {}", throttle.join(" · ")),
                style::accent(style::ALERT),
            ),
        ]));
    }

    let latest = report.timestamp_ms;
    let recent: Vec<_> = app
        .history
        .events
        .iter()
        .rev()
        .filter(|event| latest.saturating_sub(event.timestamp_ms) <= EVENT_WINDOW_MS)
        .collect();
    let room = max.saturating_sub(lines.len());
    for (index, event) in recent.iter().take(room).enumerate() {
        let (label, color) = event_label(event.kind);
        let mut spans = vec![
            Span::styled(
                format!(" {label:<8} "),
                Style::new().fg(Color::Black).bg(color),
            ),
            Span::styled(
                format!(" {} ", fmt::clock(event.timestamp_ms)),
                style::dim(),
            ),
            Span::styled(event.message.clone(), style::accent(color)),
        ];
        if index + 1 == room && recent.len() > room {
            spans.push(Span::styled(
                format!("  (+{} more in the last 10m)", recent.len() - room),
                style::dim(),
            ));
        }
        lines.push(Line::from(spans));
    }
    lines.truncate(max);
    lines
}

fn event_label(kind: EventKind) -> (&'static str, Color) {
    match kind {
        EventKind::GpuRecovery => ("RECOVERY", style::ALERT),
        EventKind::GpuIssue => ("ISSUE", style::ALERT),
        EventKind::ThermalPressure => ("THERMAL", style::TEMPERATURE),
        EventKind::Alert => ("ALERT", style::WARNING),
        EventKind::AlertCleared => ("CLEARED", Color::Green),
    }
}
