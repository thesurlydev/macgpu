//! `macgpu record` writes one [`Report`] JSON object per line (NDJSON);
//! `macgpu replay` plays a recording back through [`ReplayFeed`], in the
//! interactive view or as printed tables or JSON.
//!
//! Recordings are the same schema as `--json` output. Unknown fields are
//! ignored, so recordings from newer versions that only added fields still
//! load; a newer `schema_version` (a field changed meaning) is refused with a
//! clear error.

use super::alerts::{AlertArgs, Alerts};
use super::log::{parse_duration_arg, sample_limit};
use super::signals;
use super::table::{self, ALERT_EXIT_CODE, TableOptions};
use super::tui;
use crate::cli::{SamplingArgs, SelectionArgs};
use crate::engine::{Engine, FeedEvent, ReportFeed, clamp_interval};
use crate::model::{Report, SCHEMA_VERSION};
use anyhow::{Context, Result, bail};
use clap::Args;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Args, Debug, Clone, Default)]
#[command(after_help = "\
Writes one JSON report per line (the --json schema), flushed after every sample, so
an interrupted recording stays readable. Stop with Ctrl-C, --count, or --duration;
the summary on stderr gives the size per sample.

Examples:
  macgpu record -o overnight.ndjson -i 5 -d 10h
  macgpu replay overnight.ndjson --speed 60")]
pub struct RecordArgs {
    /// File to write, or - for standard output.
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    #[command(flatten)]
    pub sampling: SamplingArgs,

    /// Stop after this many samples. 0 or unset means until Ctrl-C.
    #[arg(short, long)]
    pub count: Option<usize>,

    /// Stop after this long, for example 30s, 10m, 8h.
    #[arg(short, long, value_parser = parse_duration_arg, value_name = "DURATION")]
    pub duration: Option<f64>,

    /// Include idle GPU processes, clients, and queues in every sample (about
    /// ten times larger). By default only processes, clients, and queues that
    /// used the GPU in the window are recorded.
    #[arg(short, long)]
    pub all: bool,
}

#[derive(Args, Debug, Clone, Default)]
#[command(after_help = "\
On a terminal the recording opens in the interactive view, paced by the recorded
timestamps. Otherwise, or with --print or --json, each sample is printed as a table
or a JSON line, as fast as possible unless --speed is given.")]
pub struct ReplayArgs {
    /// Recording made with `macgpu record` (or `macgpu --json --watch`), or - for standard input.
    #[arg(value_name = "FILE")]
    pub file: PathBuf,

    /// Playback speed relative to the recording, for example 10 for ten times faster.
    #[arg(long, value_parser = parse_speed, conflicts_with = "fast")]
    pub speed: Option<f64>,

    /// Play back as fast as possible, ignoring timestamps.
    #[arg(long)]
    pub fast: bool,

    /// Print tables instead of opening the interactive view.
    #[arg(long)]
    pub print: bool,

    /// Print one JSON report per line.
    #[arg(short, long)]
    pub json: bool,

    #[command(flatten)]
    pub selection: SelectionArgs,

    /// Show only the top N active rows in printed tables. 0 shows all.
    #[arg(short = 'n', long, default_value_t = 15)]
    pub top: usize,

    /// Include idle rows (if the recording has them).
    #[arg(short, long)]
    pub all: bool,

    /// Show full executable paths instead of names.
    #[arg(short, long)]
    pub verbose: bool,

    #[command(flatten)]
    pub alerts: AlertArgs,
}

fn parse_speed(text: &str) -> Result<f64, String> {
    let speed: f64 = text
        .parse()
        .map_err(|_| "speed must be a number".to_string())?;
    if !speed.is_finite() || speed <= 0.0 {
        return Err("speed must be greater than 0".into());
    }
    Ok(speed)
}

pub fn record(args: &RecordArgs) -> Result<i32> {
    let config = args.sampling.engine_config(1.0, args.all);
    let interval = clamp_interval(config.interval);
    let limit = sample_limit(args.count, args.duration, interval);
    let to_stdout = args.output.as_os_str() == "-";
    let mut out: Box<dyn Write> = if to_stdout {
        Box::new(BufWriter::new(io::stdout().lock()))
    } else {
        let file = File::create(&args.output)
            .with_context(|| format!("cannot create {}", args.output.display()))?;
        Box::new(BufWriter::new(file))
    };
    let target = if to_stdout {
        "standard output".to_string()
    } else {
        args.output.display().to_string()
    };

    signals::install(&signals::STOP);
    let engine = Engine::start(config);
    eprintln!(
        "macgpu: recording to {target} every {:.1} s{}; Ctrl-C stops",
        interval.as_secs_f64(),
        match limit {
            Some(limit) => format!(" for {limit} samples"),
            None => String::new(),
        }
    );

    let mut samples = 0usize;
    let mut bytes = 0u64;
    let result = (|| -> io::Result<()> {
        loop {
            if signals::take().is_some() {
                return Ok(());
            }
            let mut report = match engine.recv_timeout(Duration::from_millis(200)) {
                Ok(report) => report,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            };
            if !args.all {
                prune_idle(&mut report);
            }
            let mut line = serde_json::to_string(&report).map_err(io::Error::other)?;
            line.push('\n');
            out.write_all(line.as_bytes())?;
            out.flush()?;
            samples += 1;
            bytes += line.len() as u64;
            if limit.is_some_and(|limit| samples >= limit) {
                return Ok(());
            }
        }
    })();
    match result {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => {}
        Err(error) => return Err(error).context("failed to write the recording"),
    }

    let per_sample = if samples == 0 {
        0
    } else {
        bytes / samples as u64
    };
    eprintln!(
        "macgpu: recorded {samples} samples, {bytes} bytes ({per_sample} bytes per sample, {} per hour at this interval) to {target}",
        crate::format::bytes((per_sample as f64 * 3600.0 / interval.as_secs_f64()) as u64)
    );
    Ok(0)
}

/// Drops clients and queues that did not use the GPU in the window. Idle
/// queues of busy processes (WindowServer has dozens) dominate a report's size.
fn prune_idle(report: &mut Report) {
    for process in &mut report.processes {
        process.clients.retain(|client| client.gpu_ns > 0);
        for client in &mut process.clients {
            client.queues.retain(|queue| queue.gpu_ns > 0);
        }
    }
}

pub fn replay(args: &ReplayArgs) -> Result<i32> {
    let interactive = !args.print
        && !args.json
        && io::stdout().is_terminal()
        && io::stdin().is_terminal()
        && args.file.as_os_str() != "-";
    let speed = if args.fast {
        None
    } else if interactive {
        Some(args.speed.unwrap_or(1.0))
    } else {
        args.speed
    };
    let feed = ReplayFeed::open(&args.file, speed)?;
    let problem = feed.problem_handle();

    if interactive {
        let options = tui::TuiOptions {
            group: args.selection.group,
            filter: args.selection.filter(),
            include_idle: args.all,
            readonly: true,
            alerts: Alerts::from_args(&args.alerts)?,
        };
        tui::run(Box::new(feed), options)?;
        return finish(&problem).map(|()| 0);
    }

    let mut feed = feed;
    let mut alerts = Alerts::from_args(&args.alerts)?;
    let options = TableOptions {
        group: args.selection.group,
        filter: args.selection.filter(),
        top: args.top,
        all: args.all,
        verbose: args.verbose,
    };
    let mut stdout = io::stdout().lock();
    let mut alert_fired = false;
    let mut first = true;
    let result = (|| -> Result<()> {
        loop {
            let mut report = match feed.next_report(Duration::from_secs(3600)) {
                FeedEvent::Report(report) => report,
                FeedEvent::Timeout => continue,
                FeedEvent::End => return Ok(()),
            };
            alert_fired |= alerts.evaluate(&mut report);
            if args.json {
                writeln!(stdout, "{}", serde_json::to_string(&report)?)?;
            } else {
                if !first {
                    writeln!(stdout)?;
                }
                writeln!(
                    stdout,
                    "== {}",
                    super::log::format_time(super::log::TimeFormat::Iso, report.timestamp_ms, 0)
                )?;
                write!(stdout, "{}", table::render(&report, &options))?;
            }
            stdout.flush()?;
            first = false;
            if args.alerts.exit_on_alert && alert_fired {
                return Ok(());
            }
        }
    })();
    if let Err(error) = result {
        let broken_pipe = error
            .downcast_ref::<io::Error>()
            .is_some_and(|error| error.kind() == io::ErrorKind::BrokenPipe);
        if !broken_pipe {
            return Err(error);
        }
    }
    finish(&problem)?;
    Ok(if args.alerts.exit_on_alert && alert_fired {
        ALERT_EXIT_CODE
    } else {
        0
    })
}

/// Reports what went wrong while reading, after playback ended.
fn finish(problem: &Arc<Mutex<Problem>>) -> Result<()> {
    let problem = problem.lock().expect("replay problem lock");
    if let Some(warning) = &problem.warning {
        eprintln!("macgpu: {warning}");
    }
    match &problem.error {
        Some(error) => bail!("{error}"),
        None => Ok(()),
    }
}

/// Problems found while streaming a recording. Playback stops at the first
/// error; the caller reports it once the view closes.
#[derive(Debug, Default)]
pub struct Problem {
    pub error: Option<String>,
    pub warning: Option<String>,
}

/// Plays back a recording as a [`ReportFeed`], streaming it line by line so
/// overnight recordings do not need to fit in memory.
pub struct ReplayFeed {
    reader: Box<dyn BufRead>,
    name: String,
    line_number: usize,
    /// The next report, already read, and when it is due.
    pending: Option<(Report, Instant)>,
    /// `None` plays as fast as possible.
    speed: Option<f64>,
    last_timestamp_ms: u64,
    interval: Duration,
    problem: Arc<Mutex<Problem>>,
}

impl ReplayFeed {
    /// Opens a recording and reads its first report, so an empty, corrupt, or
    /// too-new file fails before playback starts. `-` reads standard input.
    pub fn open(path: &Path, speed: Option<f64>) -> Result<Self> {
        let (reader, name): (Box<dyn BufRead>, String) = if path.as_os_str() == "-" {
            (
                Box::new(BufReader::new(io::stdin())),
                "standard input".into(),
            )
        } else {
            let file =
                File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
            (Box::new(BufReader::new(file)), path.display().to_string())
        };
        Self::from_reader(reader, name, speed)
    }

    pub fn from_reader(reader: Box<dyn BufRead>, name: String, speed: Option<f64>) -> Result<Self> {
        let mut feed = Self {
            reader,
            name,
            line_number: 0,
            pending: None,
            speed,
            last_timestamp_ms: 0,
            interval: Duration::from_secs(1),
            problem: Arc::new(Mutex::new(Problem::default())),
        };
        let first = match feed.read_report() {
            Ok(Some(report)) => report,
            Ok(None) => bail!("{} has no reports", feed.name),
            Err(error) => bail!("{error}"),
        };
        if first.window_seconds.is_finite() && first.window_seconds > 0.0 {
            feed.interval = Duration::from_secs_f64(first.window_seconds);
        }
        feed.last_timestamp_ms = first.timestamp_ms;
        feed.pending = Some((first, Instant::now()));
        Ok(feed)
    }

    /// Shared with the caller so problems can be reported after the feed was
    /// handed to the interactive view.
    pub fn problem_handle(&self) -> Arc<Mutex<Problem>> {
        Arc::clone(&self.problem)
    }

    /// Longest pause played back between two reports; longer gaps (the Mac
    /// slept, recording paused) are shortened to this.
    fn max_gap(&self) -> Duration {
        (self.interval * 5).max(Duration::from_secs(5))
    }

    /// Reads the next report. `Ok(None)` at the end of the recording.
    fn read_report(&mut self) -> Result<Option<Report>, String> {
        let mut line = String::new();
        loop {
            line.clear();
            let read = self
                .reader
                .read_line(&mut line)
                .map_err(|error| format!("{}: read failed: {error}", self.name))?;
            if read == 0 {
                return Ok(None);
            }
            self.line_number += 1;
            if line.trim().is_empty() {
                continue;
            }
            return match parse_report(&line) {
                Ok(report) => Ok(Some(report)),
                // A recording interrupted mid-write ends with a partial line.
                Err(_) if !line.ends_with('\n') => {
                    self.problem.lock().expect("replay problem lock").warning = Some(format!(
                        "{}: ignored the incomplete last line {}",
                        self.name, self.line_number
                    ));
                    Ok(None)
                }
                Err(error) => Err(format!("{} line {}: {error}", self.name, self.line_number)),
            };
        }
    }

    fn load_next(&mut self, previous_due: Instant) {
        match self.read_report() {
            Ok(Some(report)) => {
                let gap_ms = report.timestamp_ms.saturating_sub(self.last_timestamp_ms);
                self.last_timestamp_ms = report.timestamp_ms.max(self.last_timestamp_ms);
                let due = match self.speed {
                    Some(speed) => {
                        let gap = Duration::from_millis(gap_ms).min(self.max_gap());
                        previous_due + gap.div_f64(speed)
                    }
                    None => previous_due,
                };
                self.pending = Some((report, due));
            }
            Ok(None) => self.pending = None,
            Err(error) => {
                self.problem.lock().expect("replay problem lock").error = Some(error);
                self.pending = None;
            }
        }
    }
}

/// Parses one recorded line, checking the schema version first.
pub fn parse_report(line: &str) -> Result<Report> {
    let value: serde_json::Value = serde_json::from_str(line).context("not valid JSON")?;
    let Some(version) = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
    else {
        bail!("not a macgpu report (no schema_version)");
    };
    if version > u64::from(SCHEMA_VERSION) {
        bail!(
            "recorded with report schema version {version}, newer than this macgpu understands \
             ({SCHEMA_VERSION}); upgrade macgpu to replay it"
        );
    }
    serde_json::from_value(value).context("not a macgpu report")
}

impl ReportFeed for ReplayFeed {
    fn next_report(&mut self, timeout: Duration) -> FeedEvent {
        let Some((_, due)) = &self.pending else {
            return FeedEvent::End;
        };
        let due = *due;
        if self.speed.is_some() {
            let now = Instant::now();
            if due > now {
                let wait = due - now;
                if wait > timeout {
                    thread::sleep(timeout);
                    return FeedEvent::Timeout;
                }
                thread::sleep(wait);
            }
        }
        let (report, due) = self.pending.take().expect("pending report");
        // Pace from when the report was due, so small delays in the consumer
        // do not stretch the timeline; after a long stall (the view was busy),
        // continue from now rather than bursting to catch up.
        let now = Instant::now();
        let base = if now > due + Duration::from_millis(100) {
            now
        } else {
            due
        };
        self.load_next(base);
        FeedEvent::Report(Box::new(report))
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn set_interval(&mut self, _interval: Duration) -> bool {
        false
    }

    fn is_live(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceView, ProcessView};
    use std::io::Cursor;

    fn report(timestamp_ms: u64) -> Report {
        Report {
            schema_version: SCHEMA_VERSION,
            timestamp_ms,
            window_seconds: 0.5,
            device: DeviceView {
                power_w: Some(1.5),
                ..DeviceView::default()
            },
            processes: vec![ProcessView {
                pid: 7,
                name: "proc".into(),
                gpu_ns: 1,
                ..ProcessView::default()
            }],
            ..Report::default()
        }
    }

    fn feed(text: &str, speed: Option<f64>) -> Result<ReplayFeed> {
        ReplayFeed::from_reader(
            Box::new(Cursor::new(text.as_bytes().to_vec())),
            "test".into(),
            speed,
        )
    }

    fn lines(reports: &[Report]) -> String {
        reports
            .iter()
            .map(|report| serde_json::to_string(report).unwrap() + "\n")
            .collect()
    }

    fn drain(feed: &mut ReplayFeed) -> Vec<Report> {
        let mut reports = Vec::new();
        loop {
            match feed.next_report(Duration::from_secs(10)) {
                FeedEvent::Report(report) => reports.push(*report),
                FeedEvent::Timeout => {}
                FeedEvent::End => return reports,
            }
        }
    }

    #[test]
    fn round_trips_reports() {
        let reports = vec![report(1_000), report(1_500), report(2_000)];
        let mut feed = feed(&lines(&reports), None).unwrap();
        assert_eq!(feed.interval(), Duration::from_millis(500));
        assert!(!feed.is_live());
        assert!(!feed.set_interval(Duration::from_secs(1)));
        assert_eq!(drain(&mut feed), reports);
        let problem = feed.problem_handle();
        assert!(problem.lock().unwrap().error.is_none());
    }

    #[test]
    fn tolerates_unknown_fields_and_blank_lines() {
        let mut value = serde_json::to_value(report(1_000)).unwrap();
        value["future_field"] = serde_json::json!({"x": 1});
        value["device"]["future_metric"] = serde_json::json!(3.5);
        let text = format!("\n{}\n\n", value);
        let mut feed = feed(&text, None).unwrap();
        assert_eq!(drain(&mut feed), vec![report(1_000)]);
    }

    #[test]
    fn rejects_newer_schema() {
        let mut value = serde_json::to_value(report(1_000)).unwrap();
        value["schema_version"] = serde_json::json!(SCHEMA_VERSION + 1);
        let error = feed(&format!("{value}\n"), None).err().unwrap().to_string();
        assert!(
            error.contains("newer than this macgpu understands"),
            "{error}"
        );
        let error = feed("{\"a\":1}\n", None).err().unwrap().to_string();
        assert!(error.contains("no schema_version"), "{error}");
        let error = feed("", None).err().unwrap().to_string();
        assert!(error.contains("has no reports"), "{error}");
    }

    #[test]
    fn stops_at_corrupt_line_and_ignores_partial_last_line() {
        let text = lines(&[report(1_000)]) + "garbage\n" + &lines(&[report(2_000)]);
        let mut feed = feed(&text, None).unwrap();
        assert_eq!(drain(&mut feed).len(), 1);
        let problem = feed.problem_handle();
        let error = problem.lock().unwrap().error.clone().unwrap();
        assert!(error.contains("test line 2"), "{error}");

        let full = lines(&[report(1_000), report(2_000)]);
        let truncated = &full[..full.len() - 10];
        let mut feed = super::tests::feed(truncated, None).unwrap();
        assert_eq!(drain(&mut feed).len(), 1);
        let problem = feed.problem_handle();
        assert!(problem.lock().unwrap().error.is_none());
        assert!(
            problem
                .lock()
                .unwrap()
                .warning
                .as_deref()
                .unwrap()
                .contains("incomplete last line 2")
        );
    }

    #[test]
    fn paces_by_timestamps_with_speed_and_caps_gaps() {
        // 200 ms apart at 2x: about 100 ms between reports.
        let reports = vec![report(0), report(200), report(400)];
        let mut feed = feed(&lines(&reports), Some(2.0)).unwrap();
        let started = Instant::now();
        assert_eq!(drain(&mut feed).len(), 3);
        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_millis(190), "{elapsed:?}");
        assert!(elapsed < Duration::from_millis(600), "{elapsed:?}");

        // A short timeout returns Timeout while the next report is not due.
        let mut feed = super::tests::feed(&lines(&[report(0), report(1_000)]), Some(1.0)).unwrap();
        assert!(matches!(
            feed.next_report(Duration::ZERO),
            FeedEvent::Report(_)
        ));
        assert!(matches!(
            feed.next_report(Duration::from_millis(10)),
            FeedEvent::Timeout
        ));

        // A ten-hour gap plays back as the 5 s cap, at 100x speed: 50 ms.
        let mut feed =
            super::tests::feed(&lines(&[report(0), report(36_000_000)]), Some(100.0)).unwrap();
        let started = Instant::now();
        assert_eq!(drain(&mut feed).len(), 2);
        assert!(started.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn prunes_idle_clients_and_queues() {
        use crate::model::{ClientView, QueueView};
        let queue = |gpu_ns| QueueView {
            gpu_ns,
            ..QueueView::default()
        };
        let mut pruned = report(1);
        pruned.processes[0].queues = 3;
        pruned.processes[0].clients = vec![
            ClientView {
                gpu_ns: 5,
                queues: vec![queue(5), queue(0)],
                ..ClientView::default()
            },
            ClientView {
                gpu_ns: 0,
                queues: vec![queue(0)],
                ..ClientView::default()
            },
        ];
        prune_idle(&mut pruned);
        assert_eq!(pruned.processes[0].clients.len(), 1);
        assert_eq!(pruned.processes[0].clients[0].queues.len(), 1);
        assert_eq!(pruned.processes[0].queues, 3);
    }

    #[test]
    fn parses_speed() {
        assert_eq!(parse_speed("10"), Ok(10.0));
        assert!(parse_speed("0").is_err());
        assert!(parse_speed("-1").is_err());
        assert!(parse_speed("x").is_err());
    }
}
