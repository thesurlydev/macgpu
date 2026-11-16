//! The process, app, client, or queue table with adaptive columns.
use super::fmt;

use super::app::{App, Column, DisplayRow, SortKey};
use super::charts;
use super::history::RowMetric;
use super::style::{self, DIM, GPU, LABEL};
use crate::format;
use crate::model::{Activity, Report, SourceState};
use crate::view::{Group, Row, RowKey};
use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders};

use super::graphs::WhoChart;
use super::ui::Density;

/// Full scale of the GPU ms/s sparkline.
const SPARKLINE_MAX_MSPS: f64 = 1000.0;

struct Spark {
    end: u64,
    span: Option<(usize, u64)>,
    coverage: Vec<f64>,
}

struct PlannedColumn {
    column: Column,
    header: &'static str,
    width: u16,
    right: bool,
}

fn plan_columns(app: &App, width: u16, density: Density) -> Vec<PlannedColumn> {
    let rows = &app.rows;
    let any = |test: fn(&DisplayRow) -> bool| rows.iter().any(test);
    let group = app.group;
    let mut columns: Vec<Column> = vec![Column::Pid, Column::Name, Column::Gpu, Column::Share];
    if any(|row| row.row.device_pct.is_some()) {
        columns.push(Column::DevicePct);
    }
    if any(|row| row.row.power_w.is_some()) {
        columns.push(Column::Power);
    }
    if any(|row| row.row.memory_bytes.is_some()) {
        columns.push(Column::Memory);
    }
    if group == Group::App || (app.tree && any(|row| row.row.unattributed_ms_per_s.is_some())) {
        columns.push(Column::Exited);
    }
    if group != Group::Queue {
        columns.push(Column::Children);
    }
    if any(|row| row.row.last_submit_age_ms.is_some()) {
        columns.push(Column::Last);
    }
    columns.push(Column::Lifetime);
    columns.push(Column::History);

    let children_header = match (group, app.tree) {
        (Group::App, false) => "PROCS",
        (_, true) => "N",
        _ => "QUEUES",
    };
    let header = |column: Column| -> (&'static str, u16, bool) {
        match column {
            Column::Pid => ("PID", 7, true),
            Column::Name => (
                match group {
                    Group::App if !app.tree => "APP",
                    Group::Queue => "PROCESS / QUEUE",
                    Group::Client => "PROCESS / CLIENT",
                    _ => "NAME",
                },
                0,
                false,
            ),
            Column::Gpu => ("GPU ms/s", 8, true),
            Column::Share => ("SHARE", 6, true),
            Column::DevicePct => ("DEV%", 6, true),
            Column::Power => ("WATTS", 6, true),
            Column::Memory => ("GPU MEM", 7, true),
            Column::Exited => ("EXITED", 6, true),
            Column::Children => (children_header, children_header.len().max(2) as u16, true),
            Column::Last => ("LAST", 7, false),
            Column::Lifetime => ("LIFETIME", 8, true),
            Column::History => ("HISTORY", 0, false),
        }
    };

    let name_min: u16 = if app.tree { 22 } else { 18 };
    let mut history_width: u16 = if density == Density::Full { 20 } else { 10 };
    let drop_order = [
        Column::Lifetime,
        Column::Exited,
        Column::DevicePct,
        Column::Children,
        Column::Share,
        Column::History,
        Column::Memory,
        Column::Power,
        Column::Last,
    ];
    let fixed = |columns: &[Column], history_width: u16| -> u16 {
        columns
            .iter()
            .map(|column| match column {
                Column::Name => 0,
                Column::History => history_width + 1,
                other => header(*other).1 + 1,
            })
            .sum::<u16>()
            + 1
    };
    let mut drops = drop_order.iter();
    while width < fixed(&columns, history_width) + name_min {
        if columns.contains(&Column::History) && history_width > 8 {
            history_width = 8;
            continue;
        }
        match drops.next() {
            Some(column) => columns.retain(|candidate| candidate != column),
            None => break,
        }
    }
    let mut name_width = width.saturating_sub(fixed(&columns, history_width));
    // Past a comfortable name width, spare room goes to the sparkline.
    if columns.contains(&Column::History) && name_width > 44 {
        let grow = (name_width - 44).min(40u16.saturating_sub(history_width));
        history_width += grow;
        name_width -= grow;
    }

    columns
        .into_iter()
        .map(|column| {
            let (title, width, right) = header(column);
            PlannedColumn {
                column,
                header: title,
                width: match column {
                    Column::Name => name_width.max(4),
                    Column::History => history_width,
                    _ => width,
                },
                right,
            }
        })
        .collect()
}

pub(super) fn draw_table(
    frame: &mut Frame,
    area: Rect,
    app: &mut App,
    who: &WhoChart,
    density: Density,
) {
    if area.height < 3 {
        app.hitboxes.table = Rect::default();
        return;
    }
    draw_title(frame, area, app);

    let columns = plan_columns(app, area.width, density);
    let header_y = area.y + 1;
    let body = Rect::new(area.x, area.y + 2, area.width, area.height - 2);
    app.hitboxes.table = body;
    app.hitboxes.header_y = header_y;
    app.hitboxes.header.clear();
    draw_header(
        frame.buffer_mut(),
        &mut app.hitboxes,
        &columns,
        header_y,
        app.sort,
    );

    if app.rows.is_empty() {
        empty_message(frame.buffer_mut(), body, app);
        return;
    }

    let end = app.history.latest_ms().unwrap_or(0);
    let history_column = columns
        .iter()
        .find(|planned| planned.column == Column::History);
    let spark_span = history_column.map(|planned| {
        (
            planned.width as usize,
            planned.width as u64 * app.interval.as_millis().max(1) as u64,
        )
    });
    let coverage = spark_span
        .map(|(width, span)| app.history.coverage(end, span, width))
        .unwrap_or_default();
    let spark = Spark {
        end,
        span: spark_span,
        coverage,
    };

    draw_rows(frame, area, body, app, &columns, who, &spark);
}

fn draw_title(frame: &mut Frame, area: Rect, app: &App) {
    let level = match app.group {
        Group::Queue => "queues",
        Group::Client => "clients",
        Group::Process => "processes",
        Group::App => "apps",
    };
    let mut title = vec![
        Span::styled(format!(" {level} "), style::title()),
        Span::styled(format!("{} ", app.rows.len()), style::dim()),
        Span::styled("─ sort ", style::dim()),
        Span::styled(
            format!(
                "{} {} ",
                app.sort.label(),
                if app.reverse { "▲" } else { "▼" }
            ),
            style::label(),
        ),
    ];
    if app.tree {
        title.push(Span::styled("─ tree ", style::label()));
    }
    if !app.filter.is_empty() {
        title.push(Span::styled("─ filter ", style::dim()));
        title.push(Span::styled(
            format!("{} ", app.filter),
            style::accent(style::WARNING),
        ));
    }
    if app.include_idle {
        title.push(Span::styled("─ idle shown ", style::dim()));
    }
    let rule = Block::new()
        .borders(Borders::TOP)
        .border_style(style::dim())
        .title(Line::from(title));
    frame.render_widget(rule, Rect::new(area.x, area.y, area.width, 1));
}

fn draw_header(
    buffer: &mut Buffer,
    hitboxes: &mut super::app::Hitboxes,
    columns: &[PlannedColumn],
    header_y: u16,
    sort: SortKey,
) {
    let mut x = hitboxes.table.x + 1;
    for planned in columns {
        let sorted = planned.column.sort_key() == Some(sort)
            && !matches!(
                planned.column,
                Column::Share | Column::DevicePct | Column::History
            );
        let text = if planned.right {
            format!("{:>width$}", planned.header, width = planned.width as usize)
        } else {
            format!("{:<width$}", planned.header, width = planned.width as usize)
        };
        let style = if sorted {
            style::strong(GPU).add_modifier(Modifier::UNDERLINED)
        } else {
            style::label().add_modifier(Modifier::BOLD)
        };
        buffer.set_stringn(x, header_y, text, planned.width as usize, style);
        hitboxes.header.push((x, planned.width, planned.column));
        x += planned.width + 1;
    }
}

fn empty_message(buffer: &mut Buffer, body: Rect, app: &App) {
    let text = if app.report.is_none() {
        "waiting for the first sample...".to_string()
    } else if !app.filter.is_empty() {
        format!(
            "nothing matches the filter \"{}\" (Esc clears it)",
            app.filter
        )
    } else if let Some(reason) = app
        .report
        .as_ref()
        .filter(|_| app.group == Group::App)
        .and_then(apps_unavailable)
    {
        format!("no per-app data: {reason} (g switches the grouping level)")
    } else {
        "nothing used the GPU in this window (a shows idle GPU clients)".to_string()
    };
    buffer.set_stringn(
        body.x + 2,
        body.y,
        text,
        body.width as usize - 2,
        style::dim(),
    );
}

fn draw_rows(
    frame: &mut Frame,
    area: Rect,
    body: Rect,
    app: &mut App,
    columns: &[PlannedColumn],
    who: &WhoChart,
    spark: &Spark,
) {
    let buffer = frame.buffer_mut();
    // Keep the selection visible.
    let visible = body.height as usize;
    if app.selected_index < app.scroll {
        app.scroll = app.selected_index;
    } else if app.selected_index >= app.scroll + visible {
        app.scroll = app.selected_index + 1 - visible;
    }
    app.scroll = app.scroll.min(app.rows.len().saturating_sub(visible));

    for (offset, display) in app.rows.iter().skip(app.scroll).take(visible).enumerate() {
        let y = body.y + offset as u16;
        let index = app.scroll + offset;
        let selected = index == app.selected_index;
        let row = &display.row;
        let active = row.activity == Activity::Active || row.busy_ms_per_s > 0.0;
        let mut x = area.x + 1;
        for planned in columns {
            let width = planned.width as usize;
            if planned.column == Column::Name {
                let line = name_line(display, who, width, active);
                buffer.set_line(x, y, &line, planned.width);
                x += planned.width + 1;
                continue;
            }
            if planned.column == Column::Last {
                let (glyph, color) = match row.activity {
                    Activity::Active => ("●", GPU),
                    Activity::Recent => ("◦", LABEL),
                    Activity::Idle => (" ", DIM),
                };
                let age = match (row.last_submit_age_ms, row.activity) {
                    (Some(age), _) => format::age_ms(age),
                    (None, Activity::Idle) => "-".into(),
                    (None, _) => String::new(),
                };
                buffer.set_line(
                    x,
                    y,
                    &Line::from(vec![
                        Span::styled(format!("{glyph} "), style::accent(color)),
                        Span::styled(age, if active { style::label() } else { style::dim() }),
                    ]),
                    planned.width,
                );
                x += planned.width + 1;
                continue;
            }
            let (text, text_style) = cell_for(planned, row, app, spark);
            let text = if planned.right {
                format!("{text:>width$}")
            } else {
                text
            };
            buffer.set_stringn(x, y, text, width, text_style);
            x += planned.width + 1;
        }
        if selected {
            buffer.set_style(
                Rect::new(area.x, y, area.width, 1),
                Style::new().bg(style::SELECTED_BG),
            );
        }
    }

    // Scroll position.
    if app.rows.len() > visible {
        let text = format!(
            " {}-{} of {} ",
            app.scroll + 1,
            (app.scroll + visible).min(app.rows.len()),
            app.rows.len()
        );
        let width = text.chars().count() as u16;
        buffer.set_stringn(
            area.x + area.width.saturating_sub(width + 1),
            area.y,
            text,
            width as usize,
            style::dim(),
        );
    }
}

fn cell_for(planned: &PlannedColumn, row: &Row, app: &App, spark: &Spark) -> (String, Style) {
    let width = planned.width as usize;
    let cell =
        |text: String, style: Style| -> (String, Style) { (format::truncate(&text, width), style) };
    let dash = || ("-".to_string(), style::dim());
    match planned.column {
        Column::Pid => match row.pid {
            Some(pid) => cell(pid.to_string(), style::label()),
            None => dash(),
        },
        Column::Gpu => cell(
            fmt::ms(row.busy_ms_per_s),
            if row.busy_ms_per_s > 0.0 {
                style::accent(GPU)
            } else {
                style::dim()
            },
        ),
        Column::Share => match row.share() {
            Some(value) => cell(fmt::pct(value), share_style(value)),
            None => dash(),
        },
        Column::DevicePct => match row.device_pct {
            Some(value) => cell(fmt::pct(value), share_style(value)),
            None => dash(),
        },
        Column::Power => match row.power_w {
            Some(value) => {
                let estimated = if row.power_estimated { "~" } else { "" };
                cell(
                    format!("{estimated}{}", fmt::watts_digits(value)),
                    if value > 0.0 {
                        style::accent(style::POWER)
                    } else {
                        style::dim()
                    },
                )
            }
            None => dash(),
        },
        Column::Memory => match row.memory_bytes {
            Some(bytes) => cell(fmt::bytes_short(bytes), style::accent(style::MEMORY)),
            None => dash(),
        },
        // "-" when it could not be separated from overlapping work.
        Column::Exited => match row.unattributed_ms_per_s {
            Some(value) if value >= 0.05 => cell(fmt::ms(value), style::label()),
            Some(_) => cell("0".into(), style::dim()),
            None => dash(),
        },
        Column::Children => {
            if row.key == RowKey::Unattributed {
                dash()
            } else {
                cell(row.children.to_string(), style::label())
            }
        }
        Column::Lifetime if row.key == RowKey::Unattributed => dash(),
        Column::Lifetime => cell(
            format::seconds(row.lifetime_gpu_ns as f64 / 1e9),
            style::dim(),
        ),
        Column::History => {
            let (_, span) = spark.span.unwrap_or_default();
            let values = app.history.row_buckets(
                &row.key,
                RowMetric::Busy,
                spark.end,
                span,
                &spark.coverage,
            );
            cell(
                charts::sparkline(&values, SPARKLINE_MAX_MSPS),
                style::accent(GPU),
            )
        }
        Column::Name | Column::Last => unreachable!("rendered directly in draw_rows"),
    }
}

/// Why the app level is empty, when the coalition source failed.
fn apps_unavailable(report: &Report) -> Option<String> {
    let status = report.sources.get(crate::model::source::COALITIONS)?;
    if matches!(status.state, SourceState::Ok | SourceState::Partial) {
        return None;
    }
    let state = format!("{:?}", status.state).to_lowercase();
    Some(match &status.detail {
        Some(detail) => format!("coalitions {state}, {detail}"),
        None => format!("coalitions {state}"),
    })
}

fn share_style(value: f64) -> Style {
    if value >= 0.05 {
        style::label()
    } else {
        style::dim()
    }
}

fn name_line(display: &DisplayRow, who: &WhoChart, width: usize, active: bool) -> Line<'static> {
    let row = &display.row;
    let mut spans = Vec::new();
    let mut used = 0;
    if !display.prefix.is_empty() {
        used += display.prefix.chars().count();
        spans.push(Span::styled(display.prefix.clone(), style::dim()));
    }
    if let Some(color) = who.color_of(&row.key) {
        spans.push(Span::styled("■ ", style::accent(color)));
        used += 2;
    }
    let mut name_style = if active { Style::new() } else { style::label() };
    if display.depth == 0 && display.prefix.is_empty() && !matches!(row.key, RowKey::Unattributed) {
        name_style = name_style.add_modifier(if active {
            Modifier::BOLD
        } else {
            Modifier::empty()
        });
    }
    if display.synthesized || row.key == RowKey::Unattributed {
        name_style = style::label().add_modifier(Modifier::ITALIC);
    }
    let room = width.saturating_sub(used);
    let name = format::truncate(&row.name, room);
    used += name.chars().count();
    spans.push(Span::styled(name, name_style));
    let detail = match row.key {
        RowKey::Client { .. } | RowKey::Queue { .. } => row.detail.clone(),
        _ => None,
    };
    if let Some(detail) = detail
        && used + 2 < width
    {
        spans.push(Span::styled(
            format!(" {}", format::truncate(&detail, width - used - 1)),
            style::dim(),
        ));
    }
    Line::from(spans)
}
