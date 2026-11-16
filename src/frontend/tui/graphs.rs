//! The history panel: the "who used the GPU" stack and the device line charts.
use super::fmt;

use super::app::{App, span_label};
use super::charts::{self, BrailleSeries};
use super::history::RowMetric;
use super::style::{self, DIM, GPU};
use crate::format;
use crate::view::RowKey;
use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::Color;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders};
use std::collections::HashMap;

/// The "who used the GPU" series: the top apps (or processes, without
/// coalition data) by average use over the visible span.
pub struct WhoSeries {
    pub key: RowKey,
    pub name: String,
    pub color: Color,
    pub average: f64,
}

pub struct WhoChart {
    /// Stack device % (rows sum to the device's active %) when the device's
    /// active % is known, otherwise GPU ms/s.
    pub metric: RowMetric,
    pub level_label: &'static str,
    /// Largest average first.
    pub series: Vec<WhoSeries>,
    /// Every other row of the same level.
    pub others: Vec<RowKey>,
}

impl WhoChart {
    pub fn rank(app: &App) -> Self {
        let report = app.report.as_ref();
        let by_app = report.is_some_and(|report| !report.apps.is_empty());
        let metric = if report.is_some_and(|report| report.device.active_pct.is_some()) {
            RowMetric::DevicePct
        } else {
            RowMetric::Busy
        };
        let mut chart = WhoChart {
            metric,
            level_label: if by_app { "apps" } else { "processes" },
            series: Vec::new(),
            others: Vec::new(),
        };
        let Some(end) = app.history.latest_ms() else {
            return chart;
        };
        let span = app.zoom_span_ms();
        let coverage: f64 = app.history.coverage(end, span, 1).iter().sum();
        if coverage <= 0.0 {
            return chart;
        }
        let mut ranked: Vec<(RowKey, String, f64)> = app
            .history
            .rows
            .iter()
            .filter(|(key, _)| match key {
                RowKey::App { .. } => by_app,
                RowKey::Process { .. } => !by_app,
                _ => false,
            })
            .filter_map(|(key, series)| {
                let average = app.history.row_buckets(key, metric, end, span, &[coverage])[0]?;
                (average > 0.0).then(|| (*key, series.name.clone(), average))
            })
            .collect();
        ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
        // Two processes with one name (helpers, several instances) get their PID.
        let mut counts: HashMap<String, usize> = HashMap::new();
        for (_, name, _) in ranked.iter().take(style::SERIES.len()) {
            *counts.entry(name.clone()).or_default() += 1;
        }
        for (key, name, _) in ranked.iter_mut().take(style::SERIES.len()) {
            if counts[name.as_str()] > 1
                && let RowKey::Process { pid, .. } = key
            {
                name.push_str(&format!(" {pid}"));
            }
        }
        for (index, (key, name, average)) in ranked.into_iter().enumerate() {
            if index < style::SERIES.len() {
                chart.series.push(WhoSeries {
                    key,
                    name,
                    color: style::SERIES[index],
                    average,
                });
            } else {
                chart.others.push(key);
            }
        }
        chart
    }

    pub fn color_of(&self, key: &RowKey) -> Option<Color> {
        self.series
            .iter()
            .find(|series| series.key == *key)
            .map(|series| series.color)
    }

    fn unit(&self) -> &'static str {
        match self.metric {
            RowMetric::DevicePct => "% of GPU time",
            _ => "GPU ms/s",
        }
    }

    /// Draws the stack into `area` and returns its scale.
    fn draw(&self, buffer: &mut Buffer, area: Rect, app: &App, series_count: usize) -> f64 {
        let Some(end) = app.history.latest_ms() else {
            return 0.0;
        };
        let span = app.zoom_span_ms();
        let buckets = area.width as usize;
        let coverage = app.history.coverage(end, span, buckets);
        let mut layers: Vec<(Color, Vec<Option<f64>>)> = self
            .series
            .iter()
            .take(series_count)
            .map(|series| {
                (
                    series.color,
                    app.history
                        .row_buckets(&series.key, self.metric, end, span, &coverage),
                )
            })
            .collect();
        let mut rest: Vec<RowKey> = self.others.clone();
        rest.extend(
            self.series
                .iter()
                .skip(series_count)
                .map(|series| series.key),
        );
        if !rest.is_empty() {
            let mut other = vec![None; buckets];
            for key in &rest {
                let values = app
                    .history
                    .row_buckets(key, self.metric, end, span, &coverage);
                for (slot, value) in other.iter_mut().zip(values) {
                    if let Some(value) = value {
                        *slot = Some(slot.unwrap_or(0.0) + value);
                    }
                }
            }
            layers.push((style::SERIES_OTHER, other));
        }
        let peak = (0..buckets)
            .map(|column| {
                layers
                    .iter()
                    .filter_map(|(_, values)| values[column])
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let max = match self.metric {
            RowMetric::DevicePct => fmt::nice_ceiling(peak.max(5.0)).min(100.0),
            _ => fmt::nice_ceiling(peak.max(50.0)),
        };
        charts::stacked(buffer, area, &layers, max);
        max
    }
}

pub(super) fn draw_history_full(frame: &mut Frame, area: Rect, app: &App, who: &WhoChart) {
    let span = span_label(app.zoom_span_ms());
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(style::dim())
        .title(Line::from(vec![
            Span::styled(" history ", style::title()),
            Span::styled(format!("last {span} "), style::dim()),
        ]))
        .title(Line::from(Span::styled(" z zoom ", style::dim())).alignment(Alignment::Right));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width < 20 || inner.height < 3 {
        return;
    }
    let left_width = inner.width * 11 / 20;
    let left = Rect::new(inner.x, inner.y, left_width, inner.height);
    let divider_x = inner.x + left_width;
    let right = Rect::new(
        divider_x + 2,
        inner.y,
        inner.width.saturating_sub(left_width + 2),
        inner.height,
    );
    let buffer = frame.buffer_mut();
    for y in inner.y..inner.y + inner.height {
        if let Some(cell) = buffer.cell_mut((divider_x, y)) {
            cell.set_symbol("│").set_fg(DIM);
        }
    }
    draw_who(buffer, left, app, who, true);
    draw_device_charts(buffer, right, app);
}

pub(super) fn draw_history_compact(frame: &mut Frame, area: Rect, app: &App, who: &WhoChart) {
    let left_width = area.width / 2;
    let left = Rect::new(area.x, area.y, left_width.saturating_sub(1), area.height);
    let right = Rect::new(
        area.x + left_width + 1,
        area.y,
        area.width.saturating_sub(left_width + 1),
        area.height,
    );
    let buffer = frame.buffer_mut();
    draw_who(buffer, left, app, who, false);
    draw_device_charts(buffer, right, app);
}

/// The stacked chart with its title and legend (beside it when there is
/// room, otherwise on the last line).
fn draw_who(buffer: &mut Buffer, area: Rect, app: &App, who: &WhoChart, roomy: bool) {
    if area.height < 2 || area.width < 10 {
        return;
    }
    let side_legend = roomy && area.width >= 56;
    let legend_width = if side_legend { 26 } else { 0 };
    let title_y = area.y;
    let chart = Rect::new(
        area.x,
        area.y + 1,
        area.width - legend_width,
        area.height - 1 - u16::from(!side_legend),
    );
    let series_count = if side_legend {
        (area.height as usize - 1).min(style::SERIES.len())
    } else {
        style::SERIES.len()
    };
    let max = who.draw(buffer, chart, app, series_count.saturating_sub(1).max(1));
    let detail = if chart.width >= 40 {
        format!(
            "· top {} by {} · scale {}",
            who.level_label,
            who.unit(),
            scale_label(who, max)
        )
    } else {
        format!("· {} · {}", who.level_label, scale_label(who, max))
    };
    let title = Line::from(vec![
        Span::styled("who ", style::title()),
        Span::styled(detail, style::dim()),
    ]);
    buffer.set_line(area.x, title_y, &title, chart.width);

    if who.series.is_empty() {
        let text = if app.history.device.len() < 2 {
            "collecting history..."
        } else {
            "no GPU use in this span"
        };
        buffer.set_stringn(
            chart.x + 1,
            chart.y + chart.height / 2,
            text,
            chart.width as usize,
            style::dim(),
        );
        return;
    }
    let value = |value: f64| match who.metric {
        RowMetric::DevicePct => format!("{value:>5.1}%"),
        _ => format!("{:>6}", fmt::ms(value)),
    };
    let shown = series_count.saturating_sub(1).max(1);
    if side_legend {
        let x = area.x + area.width - legend_width + 1;
        for (row, series) in who.series.iter().take(shown).enumerate() {
            let line = Line::from(vec![
                Span::styled("■ ", style::accent(series.color)),
                Span::styled(format::truncate(&series.name, 15), style::label()),
            ]);
            let y = area.y + 1 + row as u16;
            buffer.set_line(x, y, &line, legend_width - 8);
            buffer.set_stringn(
                x + legend_width - 8,
                y,
                value(series.average),
                7,
                style::label(),
            );
        }
        let others = who.series.len().saturating_sub(shown) + who.others.len();
        if others > 0 && (shown as u16) + 1 < area.height {
            let y = area.y + 1 + shown as u16;
            let line = Line::from(vec![
                Span::styled("■ ", style::accent(style::SERIES_OTHER)),
                Span::styled(format!("{others} more"), style::dim()),
            ]);
            buffer.set_line(x, y, &line, legend_width - 1);
        }
        buffer.set_stringn(
            x,
            area.y,
            format!("avg {}", span_label(app.zoom_span_ms())),
            legend_width as usize,
            style::dim(),
        );
    } else {
        let mut spans = Vec::new();
        for series in who.series.iter().take(shown) {
            spans.push(Span::styled("■", style::accent(series.color)));
            spans.push(Span::styled(
                format!(
                    "{} {} ",
                    format::truncate(&series.name, 12),
                    value(series.average).trim()
                ),
                style::label(),
            ));
        }
        buffer.set_line(
            area.x,
            area.y + area.height - 1,
            &Line::from(spans),
            area.width,
        );
    }
}

fn scale_label(who: &WhoChart, max: f64) -> String {
    match who.metric {
        RowMetric::DevicePct => format!("0-{max:.0}%"),
        _ => format!("0-{max:.0}"),
    }
}

struct MetricChart {
    name: &'static str,
    value: String,
    note: String,
    color: Color,
    values: Vec<Option<f64>>,
    overlay: Option<(Color, Vec<Option<f64>>)>,
    max: f64,
}

/// Device line charts: active % (with frequency-normalized % as a line),
/// power, clock, and temperature, as many as fit. Falls back to attributed
/// GPU ms/s when IOReport has no data.
fn draw_device_charts(buffer: &mut Buffer, area: Rect, app: &App) {
    let (Some(report), Some(end)) = (&app.report, app.history.latest_ms()) else {
        return;
    };
    let label_width = 13u16;
    if area.width <= label_width + 4 || area.height == 0 {
        return;
    }
    let chart_width = area.width - label_width;
    let span = app.zoom_span_ms();
    let dots = chart_width as usize * 2;
    let history = &app.history;
    let series = |metric: fn(&super::history::DevicePoint) -> Option<f32>| {
        super::history::bucketize(
            history.device_samples(metric, end.saturating_sub(span)),
            end,
            span,
            dots,
        )
    };
    let peak = |values: &[Option<f64>]| values.iter().flatten().copied().fold(0.0, f64::max);
    let device = &report.device;

    let mut charts = Vec::new();
    if device.active_pct.is_some() {
        let normalized = device
            .freq_normalized_pct
            .is_some()
            .then(|| (style::NORMALIZED, series(|point| point.normalized_pct)));
        charts.push(MetricChart {
            name: "active",
            value: device.active_pct.map(fmt::pct).unwrap_or_default(),
            note: device
                .freq_normalized_pct
                .map(|value| format!("norm {}", fmt::pct(value)))
                .unwrap_or_default(),
            color: GPU,
            values: series(|point| point.active_pct),
            overlay: normalized,
            max: 100.0,
        });
    } else {
        let values = series(|point| Some(point.attributed_ms_per_s));
        let max = fmt::nice_ceiling(peak(&values).max(100.0));
        charts.push(MetricChart {
            name: "GPU ms/s",
            value: fmt::ms(report.totals.attributed_ms_per_s),
            note: format!("scale {max:.0}"),
            color: GPU,
            values,
            overlay: None,
            max,
        });
    }
    if let Some(power) = device.power_w {
        let values = series(|point| point.power_w);
        let max = fmt::nice_ceiling(peak(&values).max(1.0));
        charts.push(MetricChart {
            name: "power",
            value: format::watts(power),
            note: format!("scale {max:.0} W"),
            color: style::POWER,
            values,
            overlay: None,
            max,
        });
    }
    if let Some(mhz) = device.avg_mhz {
        let values = series(|point| point.mhz);
        let max = device
            .max_mhz
            .unwrap_or_else(|| fmt::nice_ceiling(peak(&values).max(100.0)));
        charts.push(MetricChart {
            name: "clock",
            value: format!("{mhz:.0} MHz"),
            note: format!("max {max:.0}"),
            color: style::CLOCK,
            values,
            overlay: None,
            max,
        });
    }
    if let Some(temperature) = &device.temperature {
        let values = series(|point| point.temperature_c);
        charts.push(MetricChart {
            name: "temp",
            value: format!("{:.0}°C", temperature.avg_c),
            note: "scale 110°C".into(),
            color: style::TEMPERATURE,
            values,
            overlay: None,
            max: 110.0,
        });
    }

    // Each chart needs at least two rows; drop from the end until they fit.
    let fit = (area.height as usize / 2).max(1);
    charts.truncate(fit);
    let count = charts.len() as u16;
    let base = area.height / count;
    let mut extra = area.height % count;
    let mut y = area.y;
    for chart in charts {
        let height = base + u16::from(extra > 0);
        extra = extra.saturating_sub(1);
        buffer.set_stringn(area.x, y, chart.name, label_width as usize, style::dim());
        buffer.set_stringn(
            area.x,
            y + 1,
            &chart.value,
            label_width as usize - 1,
            style::strong(chart.color),
        );
        if height >= 3 {
            let note_color = chart.overlay.as_ref().map_or(DIM, |(color, _)| *color);
            buffer.set_stringn(
                area.x,
                y + 2,
                &chart.note,
                label_width as usize - 1,
                style::accent(note_color),
            );
        }
        let plot = Rect::new(area.x + label_width, y, chart_width, height);
        let mut lines = vec![BrailleSeries {
            values: &chart.values,
            color: chart.color,
            fill: true,
        }];
        if let Some((color, values)) = &chart.overlay {
            lines.push(BrailleSeries {
                values,
                color: *color,
                fill: false,
            });
        }
        // A faint baseline keeps empty charts readable.
        for x in plot.x..plot.x + plot.width {
            if let Some(cell) = buffer.cell_mut((x, plot.y + plot.height - 1)) {
                cell.set_symbol("⠀");
            }
        }
        charts::braille(buffer, plot, &lines, chart.max);
        y += height;
    }
}
