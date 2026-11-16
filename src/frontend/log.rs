//! `macgpu log`: stable, column-aligned streaming lines for long-running
//! logging and grepping, in the style of `nvidia-smi dmon` and `pmon`.
//!
//! Every sample prints one `device` line, one line per row at the chosen
//! level (`process`, `app`, `client`, or `queue`), and one `event` line per
//! event. Each line starts with a timestamp and its type, so `grep process`
//! or `awk '$2 == "device"'` pick out one kind. Text lines keep fixed columns
//! with the name last (names contain spaces); unavailable values print as `-`.
//! With `--csv`, every line has the same columns and unavailable values are
//! empty.

use super::alerts::{self, AlertArgs, Alerts};
use super::signals;
use super::table::{alert_exit_code, is_broken_pipe};
use crate::cli::{SamplingArgs, SelectionArgs};
use crate::engine::Engine;
use crate::model::{EventKind, Report};
use crate::view::{self, Filter, Group, Row, RowKey};
use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Args, Debug, Clone, Default)]
#[command(after_help = "\
Line types: device (whole GPU), process|app|client|queue (one per row at the --group level), event.
Text columns, device:  ACT% NORM% MHz W TEMP TMAX MEM_MiB LIM% BUSY PROCS
Text columns, rows:    PID [ID] BUSY [INFLIGHT] SHARE% DEV% W MEM_MiB N LAST_s NAME
BUSY is apportioned GPU ms per second (the driver splits busy time among work running at
the same moment, so all rows sum to about 1000 when the GPU is saturated), W is watts
(~W: estimated), N is queues (or GPU processes for apps), LAST_s is seconds since the last
submission, LIM% is time under a power or thermal limiter state. App lines add ID (the
coalition) and INFLIGHT: ms per second the app had GPU work in flight, where work that
overlaps another app's counts fully for each.

Examples:
  macgpu log -g app -i 2
  macgpu log --csv -o gpu.csv -d 8h
  macgpu log -s device --time unix | awk '{print $1, $5}'
  macgpu log -f chrome --alert 'process:chrome>500ms/s for 5s' --exit-on-alert")]
pub struct LogArgs {
    #[command(flatten)]
    pub sampling: SamplingArgs,

    #[command(flatten)]
    pub selection: SelectionArgs,

    /// Line types to print, comma-separated.
    #[arg(
        short = 's',
        long = "select",
        value_enum,
        value_delimiter = ',',
        default_values_t = [LineKind::Device, LineKind::Rows, LineKind::Events],
        value_name = "TYPES"
    )]
    pub select: Vec<LineKind>,

    /// Write CSV with a header row instead of aligned text.
    #[arg(long)]
    pub csv: bool,

    /// Write to FILE instead of standard output.
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Append to the --output file instead of truncating it. The CSV header is
    /// skipped when the file is not empty.
    #[arg(long, requires = "output")]
    pub append: bool,

    /// Stop after this many samples. 0 or unset means until Ctrl-C.
    #[arg(short, long)]
    pub count: Option<usize>,

    /// Stop after this long, for example 30s, 10m, 8h.
    #[arg(short, long, value_parser = parse_duration_arg, value_name = "DURATION")]
    pub duration: Option<f64>,

    /// Timestamp format.
    #[arg(short, long, value_enum, default_value_t = TimeFormat::Iso)]
    pub time: TimeFormat,

    /// Include rows that did not use the GPU in the window.
    #[arg(short, long)]
    pub all: bool,

    /// Print only the top N rows per sample. 0 prints all.
    #[arg(short = 'n', long, default_value_t = 0)]
    pub top: usize,

    /// Print executable or bundle paths instead of names.
    #[arg(short, long)]
    pub verbose: bool,

    /// Do not print the header.
    #[arg(long)]
    pub no_header: bool,

    #[command(flatten)]
    pub alerts: AlertArgs,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    /// One whole-GPU line per sample.
    Device,
    /// One line per process (or app, client, queue) per sample.
    Rows,
    /// GPU recoveries, issues, thermal changes, and alerts.
    Events,
}

#[derive(ValueEnum, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TimeFormat {
    /// Local time, ISO 8601 with milliseconds and UTC offset.
    #[default]
    Iso,
    /// UTC, ISO 8601 with milliseconds.
    Utc,
    /// Seconds since the Unix epoch, with milliseconds.
    Unix,
    /// Milliseconds since the Unix epoch.
    UnixMs,
    /// Local wall-clock time of day, HH:MM:SS.
    Clock,
    /// Seconds since the first sample.
    Elapsed,
}

pub(crate) fn parse_duration_arg(text: &str) -> Result<f64, String> {
    let seconds = alerts::parse_duration(text)?;
    if seconds <= 0.0 {
        return Err("duration must be positive".into());
    }
    Ok(seconds)
}

/// How many samples to take for `count` and `duration` at `interval`.
/// `None` means unlimited.
pub(crate) fn sample_limit(
    count: Option<usize>,
    duration: Option<f64>,
    interval: Duration,
) -> Option<usize> {
    let count = count.filter(|count| *count > 0);
    let from_duration =
        duration.map(|seconds| ((seconds / interval.as_secs_f64()).round() as usize).max(1));
    match (count, from_duration) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

pub fn run(args: &LogArgs) -> Result<i32> {
    let mut alerts = Alerts::from_args(&args.alerts)?;
    let config = args.sampling.engine_config(1.0, args.all);
    let limit = sample_limit(
        args.count,
        args.duration,
        crate::engine::clamp_interval(config.interval),
    );

    let (mut out, write_header): (Box<dyn Write>, bool) = match &args.output {
        Some(path) => {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(args.append)
                .truncate(!args.append)
                .open(path)
                .with_context(|| format!("cannot open {}", path.display()))?;
            let empty = file.metadata().map(|meta| meta.len() == 0).unwrap_or(true);
            (Box::new(BufWriter::new(file)), empty || !args.csv)
        }
        None => (Box::new(BufWriter::new(io::stdout().lock())), true),
    };

    let mut formatter = LineFormatter::new(args);
    signals::install(&signals::STOP);
    let engine = Engine::start(config);

    let result = (|| -> io::Result<i32> {
        if write_header && !args.no_header {
            for line in formatter.header() {
                writeln!(out, "{line}")?;
            }
            out.flush()?;
        }
        let mut printed = 0usize;
        let mut alert_fired = false;
        loop {
            if signals::take().is_some() {
                break;
            }
            let mut report = match signals::recv_or_stop(|timeout| engine.recv_timeout(timeout)) {
                signals::Recv::Item(report) => report,
                signals::Recv::Timeout => continue,
                signals::Recv::Stop | signals::Recv::Disconnected => break,
            };
            alert_fired |= alerts.evaluate(&mut report);
            for line in formatter.lines(&report) {
                writeln!(out, "{line}")?;
            }
            out.flush()?;
            printed += 1;
            if limit.is_some_and(|limit| printed >= limit)
                || (args.alerts.exit_on_alert && alert_fired)
            {
                break;
            }
        }
        Ok(alert_exit_code(args.alerts.exit_on_alert, alert_fired))
    })();

    match result {
        Ok(code) => Ok(code),
        // `macgpu log | head` closes the pipe: a normal way to stop.
        Err(error) if is_broken_pipe(&error) => Ok(0),
        Err(error) => Err(error).context("failed to write log output"),
    }
}

/// Formats reports into log lines. Pure, so the format is unit tested.
pub struct LineFormatter {
    group: Group,
    filter: Filter,
    kinds: Vec<LineKind>,
    csv: bool,
    time: TimeFormat,
    all: bool,
    top: usize,
    verbose: bool,
    first_timestamp_ms: Option<u64>,
}

macro_rules! csv_columns {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        /// CSV columns, the same for every line type.
        pub const CSV_COLUMNS: [&str; 25] = [$($name),+];

        /// Typed column index: a set call cannot name a column that does not exist.
        #[repr(usize)]
        enum Column {
            $($variant),+
        }
    };
}

csv_columns! {
    Time => "time",
    Type => "type",
    Pid => "pid",
    Id => "id",
    BusyMsPerS => "busy_ms_per_s",
    InFlightMsPerS => "in_flight_ms_per_s",
    SharePct => "share_pct",
    DevicePct => "device_pct",
    PowerW => "power_w",
    PowerEstimated => "power_estimated",
    GpuMemoryBytes => "gpu_memory_bytes",
    Children => "children",
    LastSubmitS => "last_submit_s",
    ActivePct => "active_pct",
    NormalizedPct => "normalized_pct",
    Mhz => "mhz",
    TempC => "temp_c",
    TempMaxC => "temp_max_c",
    LimitedPct => "limited_pct",
    ActiveProcesses => "active_processes",
    WindowS => "window_s",
    Event => "event",
    Name => "name",
    Path => "path",
    Message => "message",
}

const DASH: &str = "-";
const MIB: f64 = 1024.0 * 1024.0;

impl LineFormatter {
    pub fn new(args: &LogArgs) -> Self {
        Self {
            group: args.selection.group,
            filter: args.selection.filter(),
            kinds: args.select.clone(),
            csv: args.csv,
            time: args.time,
            all: args.all,
            top: args.top,
            verbose: args.verbose,
            first_timestamp_ms: None,
        }
    }

    fn time_width(&self) -> usize {
        match self.time {
            TimeFormat::Iso => 29,
            TimeFormat::Utc => 24,
            TimeFormat::Unix => 14,
            TimeFormat::UnixMs => 13,
            TimeFormat::Clock => 8,
            TimeFormat::Elapsed => 10,
        }
    }

    fn has_id(&self) -> bool {
        self.group != Group::Process
    }

    pub fn header(&self) -> Vec<String> {
        if self.csv {
            return vec![CSV_COLUMNS.join(",")];
        }
        let time = format!("{:<width$}", "#TIME", width = self.time_width());
        let mut lines = Vec::new();
        if self.kinds.contains(&LineKind::Device) {
            lines.push(format!(
                "{time} {:<7} {:>6} {:>6} {:>5} {:>6} {:>5} {:>5} {:>7} {:>5} {:>8} {:>5}",
                "TYPE",
                "ACT%",
                "NORM%",
                "MHz",
                "W",
                "TEMP",
                "TMAX",
                "MEM_MiB",
                "LIM%",
                "BUSY",
                "PROCS"
            ));
        }
        if self.kinds.contains(&LineKind::Rows) {
            let mut line = format!("{time} {:<7} {:>7}", "TYPE", "PID");
            if self.has_id() {
                let _ = write!(line, " {:>18}", "ID");
            }
            let power = if self.group == Group::App { "W" } else { "~W" };
            let _ = write!(line, " {:>8}", "BUSY");
            if self.group == Group::App {
                let _ = write!(line, " {:>8}", "INFLIGHT");
            }
            let _ = write!(
                line,
                " {:>6} {:>6} {power:>6} {:>7} {:>4} {:>7} {}",
                "SHARE%",
                "DEV%",
                "MEM_MiB",
                "N",
                "LAST_s",
                if self.verbose { "PATH" } else { "NAME" }
            );
            lines.push(line);
        }
        if self.kinds.contains(&LineKind::Events) {
            lines.push(format!("{time} {:<7} {} {}", "TYPE", "KIND", "MESSAGE"));
        }
        lines
    }

    pub fn lines(&mut self, report: &Report) -> Vec<String> {
        let first = *self.first_timestamp_ms.get_or_insert(report.timestamp_ms);
        let time = format_time(self.time, report.timestamp_ms, first);
        let mut lines = Vec::new();
        if self.kinds.contains(&LineKind::Device) {
            lines.push(self.device_line(&time, report));
        }
        if self.kinds.contains(&LineKind::Rows) {
            let mut rows = view::rows(report, self.group, &self.filter);
            if !self.all {
                rows.retain(|row| row.gpu_ns > 0);
            }
            if self.top > 0 {
                rows.truncate(self.top);
            }
            for row in &rows {
                lines.push(self.row_line(&time, report, row));
            }
        }
        if self.kinds.contains(&LineKind::Events) {
            for event in &report.events {
                lines.push(self.event_line(&time, report, event.kind, &event.message));
            }
        }
        lines
    }

    fn device_line(&self, time: &str, report: &Report) -> String {
        let device = &report.device;
        let limited = device.limited_pct();
        if self.csv {
            let mut cells = Csv::new(time, "device");
            cells.set(
                Column::BusyMsPerS,
                num(Some(report.totals.attributed_ms_per_s), 1),
            );
            cells.set(Column::PowerW, num(device.power_w, 3));
            cells.set(Column::PowerEstimated, "false".into());
            cells.set(
                Column::GpuMemoryBytes,
                device
                    .memory
                    .in_use_bytes
                    .map(|b| b.to_string())
                    .unwrap_or_default(),
            );
            cells.set(Column::ActivePct, num(device.active_pct, 2));
            cells.set(Column::NormalizedPct, num(device.freq_normalized_pct, 2));
            cells.set(Column::Mhz, num(device.avg_mhz, 0));
            cells.set(
                Column::TempC,
                num(device.temperature.as_ref().map(|t| t.avg_c), 1),
            );
            cells.set(
                Column::TempMaxC,
                num(device.temperature.as_ref().map(|t| t.max_c), 1),
            );
            cells.set(Column::LimitedPct, num(limited, 1));
            cells.set(
                Column::ActiveProcesses,
                report.totals.active_processes.to_string(),
            );
            cells.set(Column::WindowS, num(Some(report.window_seconds), 3));
            return cells.finish();
        }
        format!(
            "{time:<width$} {:<7} {:>6} {:>6} {:>5} {:>6} {:>5} {:>5} {:>7} {:>5} {:>8} {:>5}",
            "device",
            dash(device.active_pct, 1),
            dash(device.freq_normalized_pct, 1),
            dash(device.avg_mhz, 0),
            dash(device.power_w, 2),
            dash(device.temperature.as_ref().map(|t| t.avg_c), 1),
            dash(device.temperature.as_ref().map(|t| t.max_c), 1),
            dash(device.memory.in_use_bytes.map(|b| b as f64 / MIB), 0),
            dash(limited, 1),
            format!("{:.1}", report.totals.attributed_ms_per_s),
            report.totals.active_processes,
            width = self.time_width(),
        )
    }

    fn row_line(&self, time: &str, report: &Report, row: &Row) -> String {
        let kind = match row.key {
            RowKey::Unattributed => "process",
            _ => self.group.label(),
        };
        let id = row_id(row.key);
        let name = if self.verbose {
            row.detail.as_deref().unwrap_or(&row.name)
        } else {
            &row.name
        };
        let last = row.last_submit_age_ms.map(|ms| ms as f64 / 1000.0);
        let in_flight = match row.key {
            RowKey::App { coalition_id } => report
                .apps
                .iter()
                .find(|app| app.coalition_id == coalition_id)
                .and_then(|app| app.in_flight_ms_per_s),
            _ => None,
        };
        if self.csv {
            let mut cells = Csv::new(time, kind);
            cells.set(
                Column::Pid,
                row.pid.map(|pid| pid.to_string()).unwrap_or_default(),
            );
            cells.set(Column::Id, id.unwrap_or_default());
            cells.set(Column::BusyMsPerS, num(Some(row.busy_ms_per_s), 3));
            cells.set(Column::InFlightMsPerS, num(in_flight, 3));
            cells.set(Column::SharePct, num(row.share(), 2));
            cells.set(Column::DevicePct, num(row.device_pct, 2));
            cells.set(Column::PowerW, num(row.power_w, 3));
            if row.power_w.is_some() {
                cells.set(Column::PowerEstimated, row.power_estimated.to_string());
            }
            cells.set(
                Column::GpuMemoryBytes,
                row.memory_bytes.map(|b| b.to_string()).unwrap_or_default(),
            );
            if row.key != RowKey::Unattributed {
                cells.set(Column::Children, row.children.to_string());
            }
            cells.set(Column::LastSubmitS, num(last, 1));
            cells.set(Column::WindowS, num(Some(report.window_seconds), 3));
            cells.set(Column::Name, row.name.clone());
            cells.set(Column::Path, row.detail.clone().unwrap_or_default());
            return cells.finish();
        }
        let mut line = format!(
            "{time:<width$} {kind:<7} {:>7}",
            row.pid
                .map(|pid| pid.to_string())
                .unwrap_or_else(|| DASH.into()),
            width = self.time_width()
        );
        if self.has_id() {
            let _ = write!(line, " {:>18}", id.as_deref().unwrap_or(DASH));
        }
        let children = if row.key == RowKey::Unattributed {
            DASH.to_string()
        } else {
            row.children.to_string()
        };
        let _ = write!(line, " {:>8.1}", row.busy_ms_per_s);
        if self.group == Group::App {
            let _ = write!(line, " {:>8}", dash(in_flight, 1));
        }
        let _ = write!(
            line,
            " {:>6} {:>6} {:>6} {:>7} {:>4} {:>7} {name}",
            dash(row.share(), 1),
            dash(row.device_pct, 1),
            dash(row.power_w, 2),
            dash(row.memory_bytes.map(|b| b as f64 / MIB), 0),
            children,
            dash(last, 1),
        );
        line
    }

    fn event_line(&self, time: &str, report: &Report, kind: EventKind, message: &str) -> String {
        let kind = event_kind_name(kind);
        if self.csv {
            let mut cells = Csv::new(time, "event");
            cells.set(Column::Event, kind.to_string());
            cells.set(Column::WindowS, num(Some(report.window_seconds), 3));
            cells.set(Column::Message, message.to_string());
            return cells.finish();
        }
        format!(
            "{time:<width$} {:<7} {kind} {message}",
            "event",
            width = self.time_width()
        )
    }
}

fn row_id(key: RowKey) -> Option<String> {
    match key {
        RowKey::App { coalition_id } => Some(coalition_id.to_string()),
        RowKey::Client { registry_id } => Some(format!("{registry_id:#x}")),
        RowKey::Queue { registry_id, index } => Some(format!("{registry_id:#x}.{index}")),
        RowKey::Process { .. } | RowKey::Unattributed => None,
    }
}

pub fn event_kind_name(kind: EventKind) -> &'static str {
    match kind {
        EventKind::GpuRecovery => "gpu_recovery",
        EventKind::GpuIssue => "gpu_issue",
        EventKind::ThermalPressure => "thermal_pressure",
        EventKind::Alert => "alert",
        EventKind::AlertCleared => "alert_cleared",
    }
}

/// A CSV line with [`CSV_COLUMNS`], filled by typed index.
struct Csv {
    cells: Vec<String>,
}

impl Csv {
    fn new(time: &str, kind: &str) -> Self {
        let mut csv = Self {
            cells: vec![String::new(); CSV_COLUMNS.len()],
        };
        csv.set(Column::Time, time.to_string());
        csv.set(Column::Type, kind.to_string());
        csv
    }

    fn set(&mut self, column: Column, value: String) {
        self.cells[column as usize] = value;
    }

    fn finish(self) -> String {
        self.cells
            .iter()
            .map(|cell| csv_escape(cell))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Quotes a CSV cell when needed (RFC 4180).
pub fn csv_escape(cell: &str) -> String {
    if cell.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", cell.replace('"', "\"\""))
    } else {
        cell.to_string()
    }
}

fn format_value(value: Option<f64>, decimals: usize, fallback: &str) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.decimals$}"))
        .unwrap_or_else(|| fallback.into())
}

fn num(value: Option<f64>, decimals: usize) -> String {
    format_value(value, decimals, "")
}

fn dash(value: Option<f64>, decimals: usize) -> String {
    format_value(value, decimals, DASH)
}

/// Formats a Unix-epoch millisecond timestamp.
pub fn format_time(format: TimeFormat, timestamp_ms: u64, first_ms: u64) -> String {
    let seconds = (timestamp_ms / 1000) as i64;
    let millis = timestamp_ms % 1000;
    match format {
        TimeFormat::Unix => format!("{seconds}.{millis:03}"),
        TimeFormat::UnixMs => timestamp_ms.to_string(),
        TimeFormat::Elapsed => format!(
            "{:.3}",
            timestamp_ms.saturating_sub(first_ms) as f64 / 1000.0
        ),
        TimeFormat::Utc => {
            let Some(tm) = broken_down(seconds, false) else {
                return "unavailable".into();
            };
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{millis:03}Z",
                tm.tm_year + 1900,
                tm.tm_mon + 1,
                tm.tm_mday,
                tm.tm_hour,
                tm.tm_min,
                tm.tm_sec
            )
        }
        TimeFormat::Iso => {
            let Some(tm) = broken_down(seconds, true) else {
                return "unavailable".into();
            };
            let offset_minutes = tm.tm_gmtoff / 60;
            let sign = if offset_minutes < 0 { '-' } else { '+' };
            let offset_minutes = offset_minutes.abs();
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{millis:03}{sign}{:02}:{:02}",
                tm.tm_year + 1900,
                tm.tm_mon + 1,
                tm.tm_mday,
                tm.tm_hour,
                tm.tm_min,
                tm.tm_sec,
                offset_minutes / 60,
                offset_minutes % 60
            )
        }
        TimeFormat::Clock => {
            let Some(tm) = broken_down(seconds, true) else {
                return "--:--:--".into();
            };
            format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
        }
    }
}

fn broken_down(seconds: i64, local: bool) -> Option<libc::tm> {
    let time: libc::time_t = seconds;
    // SAFETY: `tm` is plain data; localtime_r/gmtime_r write into it and
    // report failure through a null return, which the caller handles.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let result = unsafe {
        if local {
            libc::localtime_r(&time, &mut tm)
        } else {
            libc::gmtime_r(&time, &mut tm)
        }
    };
    (!result.is_null()).then_some(tm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use crate::model::{
        AppView, DeviceMemory, DeviceView, Event, ProcessView, Temperature, Totals, Unattributed,
    };
    use clap::Parser;

    fn args(extra: &[&str]) -> LogArgs {
        let mut argv = vec!["macgpu", "log"];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv).expect("parse").command {
            Some(crate::cli::Command::Log(args)) => args,
            other => panic!("{other:?}"),
        }
    }

    fn report() -> Report {
        Report {
            timestamp_ms: 1_798_300_801_123,
            window_seconds: 1.0,
            device: DeviceView {
                active_pct: Some(42.25),
                avg_mhz: Some(1182.0),
                power_w: Some(12.5),
                temperature: Some(Temperature {
                    avg_c: 61.04,
                    max_c: 66.0,
                    sensors: 4,
                }),
                memory: DeviceMemory {
                    in_use_bytes: Some(2 << 30),
                    ..DeviceMemory::default()
                },
                ..DeviceView::default()
            },
            totals: Totals {
                attributed_ms_per_s: 450.0,
                active_processes: 2,
                ..Totals::default()
            },
            processes: vec![
                ProcessView {
                    pid: 431,
                    start_us: 1,
                    name: "WindowServer".into(),
                    path: Some("/System/Library/WindowServer".into()),
                    gpu_ns: 300_000_000,
                    busy_ms_per_s: 300.0,
                    share_pct: 66.7,
                    device_pct: Some(28.2),
                    est_power_w: Some(8.0),
                    graphics_memory_bytes: None,
                    queues: 3,
                    last_submit_age_ms: Some(5),
                    ..ProcessView::default()
                },
                ProcessView {
                    pid: 900,
                    start_us: 1,
                    name: "Google Chrome Helper (GPU), \"beta\"".into(),
                    gpu_ns: 150_000_000,
                    busy_ms_per_s: 150.0,
                    share_pct: 33.3,
                    graphics_memory_bytes: Some(512 << 20),
                    queues: 1,
                    ..ProcessView::default()
                },
                ProcessView {
                    pid: 901,
                    start_us: 1,
                    name: "idle".into(),
                    ..ProcessView::default()
                },
            ],
            apps: vec![AppView {
                coalition_id: 77,
                name: "Google Chrome".into(),
                responsible_pid: Some(899),
                gpu_ns: 150_000_000,
                busy_ms_per_s: 150.0,
                share_pct: 33.3,
                power_w: Some(4.0),
                in_flight_ms_per_s: Some(210.0),
                gpu_pids: vec![900],
                ..AppView::default()
            }],
            unattributed: Some(Unattributed {
                gpu_ns: 1_000_000,
                busy_ms_per_s: 1.0,
                partial: false,
            }),
            events: vec![Event {
                timestamp_ms: 1,
                kind: EventKind::Alert,
                message: "alert `active>40%` firing: active 42.2%".into(),
            }],
            ..Report::default()
        }
    }

    #[test]
    fn text_lines_are_aligned_with_name_last() {
        let mut formatter = LineFormatter::new(&args(&["--time", "unix"]));
        let header = formatter.header();
        assert_eq!(header.len(), 3);
        let lines = formatter.lines(&report());
        assert_eq!(
            lines,
            vec![
                "1798300801.123 device    42.2      -  1182  12.50  61.0  66.0    2048   0.0    450.0     2",
                "1798300801.123 process     431    300.0   66.7   28.2   8.00       -    3     0.0 WindowServer",
                "1798300801.123 process     900    150.0   33.3      -      -     512    1       - Google Chrome Helper (GPU), \"beta\"",
                "1798300801.123 process       -      1.0      -      -      -       -    -       - (exited or unattributed)",
                "1798300801.123 event   alert alert `active>40%` firing: active 42.2%",
            ]
        );
        // Header columns line up with the values.
        let device_header = &header[0];
        assert_eq!(
            device_header.find("ACT%").map(|i| i + 4),
            lines[0].find("42.2").map(|i| i + 4)
        );
        let row_header = &header[1];
        assert_eq!(row_header.find("NAME"), lines[1].find("WindowServer"));
        // Awk-style splitting finds the same fields on every row line.
        for line in &lines[1..4] {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(fields[1], "process");
            assert!(fields.len() >= 11, "{line}");
        }
    }

    #[test]
    fn app_level_has_id_and_measured_power() {
        let mut formatter =
            LineFormatter::new(&args(&["-g", "app", "-s", "rows", "--time", "elapsed"]));
        assert!(formatter.header()[0].contains("ID"));
        assert!(formatter.header()[0].contains("      W"));
        let lines = formatter.lines(&report());
        assert_eq!(
            formatter.header()[0].find("INFLIGHT").map(|i| i + 8),
            lines[0].find("210.0").map(|i| i + 5)
        );
        assert_eq!(
            lines,
            vec![
                "0.000      app         899                 77    150.0    210.0   33.3      -   4.00       -    1       - Google Chrome"
            ]
        );
        let mut next = report();
        next.timestamp_ms += 2_500;
        assert!(formatter.lines(&next)[0].starts_with("2.500 "));
    }

    #[test]
    fn all_and_top_and_filter() {
        let mut formatter = LineFormatter::new(&args(&["-s", "rows", "--all"]));
        assert_eq!(formatter.lines(&report()).len(), 4);
        let mut formatter = LineFormatter::new(&args(&["-s", "rows", "-n", "1"]));
        assert_eq!(formatter.lines(&report()).len(), 1);
        let mut formatter = LineFormatter::new(&args(&["-s", "rows", "-f", "chrome"]));
        let lines = formatter.lines(&report());
        assert_eq!(lines.len(), 1);
        assert!(lines[0].ends_with("Google Chrome Helper (GPU), \"beta\""));
        let mut formatter = LineFormatter::new(&args(&["-s", "rows", "-v", "-f", "window"]));
        assert!(formatter.lines(&report())[0].ends_with("/System/Library/WindowServer"));
    }

    #[test]
    fn csv_has_constant_columns() {
        let mut formatter = LineFormatter::new(&args(&["--csv", "--time", "unix-ms"]));
        let header = formatter.header();
        assert_eq!(header, vec![CSV_COLUMNS.join(",")]);
        let lines = formatter.lines(&report());
        assert_eq!(lines.len(), 5);
        for line in &lines {
            assert_eq!(parse_csv_line(line).len(), CSV_COLUMNS.len(), "{line}");
        }
        let device = parse_csv_line(&lines[0]);
        let column = |name: &str| CSV_COLUMNS.iter().position(|c| *c == name).unwrap();
        assert_eq!(device[column("time")], "1798300801123");
        assert_eq!(device[column("type")], "device");
        assert_eq!(device[column("active_pct")], "42.25");
        assert_eq!(device[column("normalized_pct")], "");
        assert_eq!(device[column("gpu_memory_bytes")], (2u64 << 30).to_string());
        let chrome = parse_csv_line(&lines[2]);
        assert_eq!(
            chrome[column("name")],
            "Google Chrome Helper (GPU), \"beta\""
        );
        assert_eq!(chrome[column("pid")], "900");
        assert_eq!(chrome[column("power_w")], "");
        assert_eq!(chrome[column("power_estimated")], "");
        let server = parse_csv_line(&lines[1]);
        assert_eq!(server[column("power_estimated")], "true");
        assert_eq!(server[column("children")], "3");
        let event = parse_csv_line(&lines[4]);
        assert_eq!(event[column("event")], "alert");
        assert!(event[column("message")].contains("firing"));
    }

    /// Minimal RFC 4180 line parser for tests.
    pub(crate) fn parse_csv_line(line: &str) -> Vec<String> {
        let mut cells = vec![String::new()];
        let mut quoted = false;
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            match (c, quoted) {
                ('"', true) if chars.peek() == Some(&'"') => {
                    chars.next();
                    cells.last_mut().unwrap().push('"');
                }
                ('"', _) => quoted = !quoted,
                (',', false) => cells.push(String::new()),
                (c, _) => cells.last_mut().unwrap().push(c),
            }
        }
        cells
    }

    #[test]
    fn formats_timestamps() {
        let ms = 1_798_300_801_123;
        assert_eq!(format_time(TimeFormat::Unix, ms, 0), "1798300801.123");
        assert_eq!(format_time(TimeFormat::UnixMs, ms, 0), "1798300801123");
        assert_eq!(
            format_time(TimeFormat::Utc, ms, 0),
            "2026-12-26T16:00:01.123Z"
        );
        assert_eq!(format_time(TimeFormat::Elapsed, ms, ms - 1500), "1.500");
        let iso = format_time(TimeFormat::Iso, ms, 0);
        assert_eq!(iso.len(), 29, "{iso}");
        assert!(
            iso.ends_with(":00") || iso.ends_with(":30") || iso.ends_with(":45"),
            "{iso}"
        );
        assert_eq!(format_time(TimeFormat::Clock, ms, 0).len(), 8);
    }

    #[test]
    fn sample_limits() {
        let second = Duration::from_secs(1);
        assert_eq!(sample_limit(None, None, second), None);
        assert_eq!(sample_limit(Some(0), None, second), None);
        assert_eq!(sample_limit(Some(5), None, second), Some(5));
        assert_eq!(sample_limit(None, Some(10.0), second), Some(10));
        assert_eq!(
            sample_limit(None, Some(10.0), Duration::from_secs(3)),
            Some(3)
        );
        assert_eq!(sample_limit(None, Some(0.2), second), Some(1));
        assert_eq!(sample_limit(Some(4), Some(10.0), second), Some(4));
        assert!(parse_duration_arg("0s").is_err());
        assert_eq!(parse_duration_arg("8h"), Ok(28_800.0));
    }

    #[test]
    fn escapes_csv() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
    }
}
