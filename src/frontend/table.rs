//! Printed output: one table (or JSON object) per sample.

use super::alerts::Alerts;
use crate::cli::ViewArgs;
use crate::engine::Engine;
use crate::format;
use crate::model::{Report, SourceState};
use crate::view::{self, Filter, Group, Row, RowKey};
use anyhow::{Context, Result};
use std::fmt::Write as _;
use std::io::{self, IsTerminal, Write};

/// Exit status when `--exit-on-alert` is set and an alert fired.
pub const ALERT_EXIT_CODE: i32 = 3;

pub struct TableOptions {
    pub group: Group,
    pub filter: Filter,
    pub top: usize,
    pub all: bool,
    pub verbose: bool,
}

/// Runs the printed modes (`--once`, `--watch`, `--count`, `--json`, or piped).
pub fn run(args: &ViewArgs) -> Result<i32> {
    let count = match args.count {
        Some(0) => None,
        Some(count) => Some(count),
        None if args.watch => None,
        None => Some(1),
    };
    let default_interval = if args.watch || count != Some(1) {
        1.0
    } else {
        2.0
    };
    let engine = Engine::start(args.sampling.engine_config(default_interval, args.all));
    let mut alerts = Alerts::from_args(&args.alerts)?;
    let options = TableOptions {
        group: args.selection.group,
        filter: args.selection.filter(),
        top: args.top,
        all: args.all,
        verbose: args.verbose,
    };
    let clear = args.watch && !args.json && io::stdout().is_terminal();
    let mut stdout = io::stdout().lock();
    let mut printed = 0usize;
    let mut alert_fired = false;

    let result = (|| -> io::Result<()> {
        while let Some(mut report) = engine.recv() {
            alert_fired |= alerts.evaluate(&mut report);
            if args.json {
                let text = if count == Some(1) {
                    serde_json::to_string_pretty(&report).map_err(io::Error::other)?
                } else {
                    serde_json::to_string(&report).map_err(io::Error::other)?
                };
                writeln!(stdout, "{text}")?;
            } else {
                if clear {
                    write!(stdout, "\x1b[2J\x1b[H")?;
                }
                write!(stdout, "{}", render(&report, &options))?;
            }
            stdout.flush()?;

            printed += 1;
            if count.is_some_and(|count| printed >= count) {
                break;
            }
            if args.alerts.exit_on_alert && alert_fired {
                break;
            }
        }
        Ok(())
    })();

    match result {
        Ok(()) => {}
        // `macgpu --watch | head` closes the pipe: a normal way to stop.
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => {}
        Err(error) => return Err(error).context("failed to write output"),
    }

    Ok(if args.alerts.exit_on_alert && alert_fired {
        ALERT_EXIT_CODE
    } else {
        0
    })
}

/// Renders the header and table for one report.
pub fn render(report: &Report, options: &TableOptions) -> String {
    let mut out = String::new();
    for line in header_lines(report) {
        let _ = writeln!(out, "{line}");
    }
    let _ = writeln!(out);

    let mut rows = view::rows(report, options.group, &options.filter);
    if !options.all {
        // Coalition energy lags GPU time by about a window, so an app can show
        // measured power after its GPU time stopped. Keep it, so the power
        // column still accounts for the device's power.
        rows.retain(|row| {
            row.gpu_ns > 0
                || (!row.power_estimated && row.power_w.is_some_and(|watts| watts >= 0.05))
        });
        if options.top > 0 {
            rows.truncate(options.top);
        }
    }
    out.push_str(&render_rows(&rows, options, report));
    out
}

pub fn header_lines(report: &Report) -> Vec<String> {
    let device = &report.device;
    let mut lines = Vec::new();

    let mut gpu = vec![format!("window {:.2}s", report.window_seconds)];
    if let Some(model) = &device.model {
        let cores = device
            .gpu_cores
            .map(|cores| format!(" ({cores} cores)"))
            .unwrap_or_default();
        gpu.push(format!("{model}{cores}"));
    }
    if let Some(active) = device.active_pct {
        gpu.push(format!("active {active:.1}%"));
    }
    if let Some(mhz) = device.avg_mhz {
        let normalized = device
            .freq_normalized_pct
            .map(|value| format!(" (normalized {value:.1}%)"))
            .unwrap_or_default();
        gpu.push(format!("{mhz:.0} MHz{normalized}"));
    }
    if let Some(power) = device.power_w {
        gpu.push(format::watts(power));
    }
    if let Some(temperature) = &device.temperature {
        gpu.push(format!(
            "{:.0} C (max {:.0} C)",
            temperature.avg_c, temperature.max_c
        ));
    }
    lines.push(gpu.join("  "));

    let mut usage = vec![format!(
        "attributed {:.1} ms/s across {} active of {} GPU processes",
        report.totals.attributed_ms_per_s,
        report.totals.active_processes,
        report.totals.gpu_processes
    )];
    if report.totals.active_apps > 0 {
        usage.push(format!("{} active apps", report.totals.active_apps));
    }
    if let Some(firmware) = &device.firmware_utilization {
        usage.push(format!(
            "firmware device {:.0}% renderer {:.0}% tiler {:.0}%",
            firmware.device_pct, firmware.renderer_pct, firmware.tiler_pct
        ));
    }
    lines.push(usage.join("  "));

    let mut context = Vec::new();
    if let (Some(in_use), Some(allocated)) =
        (device.memory.in_use_bytes, device.memory.allocated_bytes)
    {
        let limit = device
            .memory
            .wired_limit_bytes
            .map(|limit| format!(", wired limit {}", format::bytes(limit)))
            .unwrap_or_default();
        context.push(format!(
            "gpu memory {} in use / {} allocated{limit}",
            format::bytes(in_use),
            format::bytes(allocated)
        ));
    }
    let limiters: Vec<String> = device
        .limiting_states()
        .filter(|limiter| limiter.pct >= 1.0)
        .take(3)
        .map(|limiter| format!("{} {:.0}%", limiter.name, limiter.pct))
        .collect();
    if !limiters.is_empty() {
        context.push(format!("limiter {}", limiters.join(", ")));
    }
    if let Some(thermal) = &device.system.thermal_pressure {
        context.push(format!("thermal {thermal}"));
    }
    if let Some(memory) = &device.system.memory_pressure {
        context.push(format!("memory pressure {memory}"));
    }
    context.push(format!(
        "macgpu cpu {:.1}%",
        report.overhead.process_cpu_pct
    ));
    lines.push(context.join("  "));

    let problems: Vec<String> = report
        .sources
        .iter()
        .filter(|(_, status)| !matches!(status.state, SourceState::Ok | SourceState::Disabled))
        .map(|(name, status)| {
            let state = match status.state {
                SourceState::Partial => "partial",
                SourceState::Stale => "stale",
                SourceState::Denied => "denied",
                _ => "unavailable",
            };
            match &status.detail {
                Some(detail) => format!("{name} {state}: {detail}"),
                None => format!("{name} {state}"),
            }
        })
        .collect();
    if !problems.is_empty() {
        lines.push(format!("sources  {}", problems.join("; ")));
    }
    for event in &report.events {
        lines.push(format!("event  {}", event.message));
    }
    lines
}

fn render_rows(rows: &[Row], options: &TableOptions, report: &Report) -> String {
    let mut out = String::new();
    let name_width = if options.verbose { 80 } else { 32 };
    let show_device = rows.iter().any(|row| row.device_pct.is_some());
    let show_power = rows.iter().any(|row| row.power_w.is_some());
    let show_memory = rows.iter().any(|row| row.memory_bytes.is_some());
    let power_label = if options.group == Group::App {
        "POWER"
    } else {
        "~POWER"
    };
    let children_label = match options.group {
        Group::App => "PROCS",
        Group::Process | Group::Client => "QUEUES",
        Group::Queue => "",
    };

    let mut header = format!(
        "{:>7}  {:<name_width$} {:>9} {:>6}",
        "PID",
        options.group.label().to_uppercase(),
        "GPU ms/s",
        "SHARE"
    );
    if show_device {
        let _ = write!(header, " {:>6}", "DEV%");
    }
    if show_power {
        let _ = write!(header, " {power_label:>8}");
    }
    if show_memory {
        let _ = write!(header, " {:>9}", "GPU MEM");
    }
    if options.group == Group::App {
        let _ = write!(header, " {:>8}", "EXITED");
    }
    if !children_label.is_empty() {
        let _ = write!(header, " {children_label:>6}");
    }
    let _ = write!(header, " {:>6}", "LAST");
    if options.all {
        let _ = write!(header, " {:>9}", "LIFETIME");
    }
    let _ = writeln!(out, "{header}");

    if rows.is_empty() {
        let what = if options.filter.is_empty() {
            ""
        } else {
            " matching the filter"
        };
        let _ = writeln!(out, "  nothing{what} used the GPU in this window");
        return out;
    }

    for row in rows {
        let name = if options.verbose {
            row.detail.as_deref().unwrap_or(&row.name)
        } else {
            &row.name
        };
        let pid = row
            .pid
            .map(|pid| pid.to_string())
            .unwrap_or_else(|| "-".into());
        let mut line = format!(
            "{:>7}  {:<name_width$} {:>9.1} {:>6}",
            pid,
            format::truncate(name, name_width),
            row.busy_ms_per_s,
            row.share()
                .map(|share| format!("{share:.1}%"))
                .unwrap_or_else(|| "-".into())
        );
        if show_device {
            let _ = write!(
                line,
                " {:>6}",
                row.device_pct
                    .map(|value| format!("{value:.1}"))
                    .unwrap_or_else(|| "-".into())
            );
        }
        if show_power {
            let _ = write!(
                line,
                " {:>8}",
                row.power_w.map(format::watts).unwrap_or_else(|| "-".into())
            );
        }
        if show_memory {
            let _ = write!(
                line,
                " {:>9}",
                row.memory_bytes
                    .map(format::bytes)
                    .unwrap_or_else(|| "-".into())
            );
        }
        if options.group == Group::App {
            let _ = write!(
                line,
                " {:>8}",
                row.unattributed_ms_per_s
                    .filter(|value| *value >= 0.05)
                    .map(|value| format!("{value:.1}"))
                    .unwrap_or_else(|| "-".into())
            );
        }
        if !children_label.is_empty() {
            if row.key == RowKey::Unattributed {
                let _ = write!(line, " {:>6}", "-");
            } else {
                let _ = write!(line, " {:>6}", row.children);
            }
        }
        let _ = write!(
            line,
            " {:>6}",
            row.last_submit_age_ms
                .map(format::age_ms)
                .unwrap_or_else(|| "-".into())
        );
        if options.all {
            let _ = write!(
                line,
                " {:>9}",
                format::seconds(row.lifetime_gpu_ns as f64 / 1e9)
            );
        }
        let _ = writeln!(out, "{line}");
    }

    if options.group == Group::Process
        && rows.iter().any(|row| row.memory_bytes.is_none())
        && show_memory
    {
        let other_users = report
            .processes
            .iter()
            .any(|process| process.graphics_memory_bytes.is_none());
        if other_users {
            let _ = writeln!(
                out,
                "\nGPU MEM is - for other users' processes; run as root to see them."
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceView, ProcessView, Totals};

    #[test]
    fn renders_header_and_rows() {
        let report = Report {
            window_seconds: 2.0,
            device: DeviceView {
                model: Some("Apple M4 Max".into()),
                gpu_cores: Some(40),
                active_pct: Some(42.0),
                avg_mhz: Some(1182.0),
                freq_normalized_pct: Some(31.5),
                power_w: Some(12.3),
                ..DeviceView::default()
            },
            totals: Totals {
                attributed_ms_per_s: 400.0,
                active_processes: 1,
                gpu_processes: 3,
                ..Totals::default()
            },
            processes: vec![ProcessView {
                pid: 431,
                name: "WindowServer".into(),
                gpu_ns: 800_000_000,
                busy_ms_per_s: 400.0,
                share_pct: 100.0,
                device_pct: Some(42.0),
                queues: 3,
                last_submit_age_ms: Some(5),
                ..ProcessView::default()
            }],
            ..Report::default()
        };
        let options = TableOptions {
            group: Group::Process,
            filter: Filter::default(),
            top: 15,
            all: false,
            verbose: false,
        };
        let text = render(&report, &options);
        assert!(text.contains("Apple M4 Max (40 cores)"), "{text}");
        assert!(text.contains("active 42.0%"));
        assert!(text.contains("1182 MHz (normalized 31.5%)"));
        assert!(text.contains("12.3 W"));
        assert!(text.contains("WindowServer"));
        assert!(text.contains("400.0"));
        assert!(text.contains("DEV%"));
    }

    #[test]
    fn says_when_nothing_used_the_gpu() {
        let options = TableOptions {
            group: Group::App,
            filter: Filter::parse("zzz"),
            top: 15,
            all: false,
            verbose: false,
        };
        let text = render(&Report::default(), &options);
        assert!(
            text.contains("nothing matching the filter used the GPU"),
            "{text}"
        );
    }
}
