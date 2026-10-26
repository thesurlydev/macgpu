//! `macgpu serve`: a Prometheus exporter.
//!
//! The [`Engine`] samples on its own thread; a collector thread turns each
//! report into the exposition text once, and HTTP requests only copy that
//! text, so a scrape never waits for (or triggers) a sample.
//!
//! Counters versus gauges: cumulative GPU time and energy are counters
//! (`_total`), so `rate()` and `increase()` work at any scrape interval. Process
//! counters accumulate from when the exporter first saw the process; app
//! counters are the kernel's coalition counters, which include exited helpers.
//! Busy, share, power, temperature, and memory are gauges for the latest
//! window.

use super::alerts;
use super::signals;
use crate::cli::SamplingArgs;
use crate::engine::{Engine, clamp_interval};
use crate::model::{Report, SourceState};
use crate::view::Filter;
use anyhow::{Context, Result, anyhow};
use clap::{Args, ValueEnum};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::io;
use std::net::{SocketAddr, TcpListener, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Not allocated in the Prometheus default port registry
/// (github.com/prometheus/prometheus/wiki/Default-port-allocations) and not a
/// well-known service port.
pub const DEFAULT_PORT: u16 = 10060;

#[derive(Args, Debug, Clone, Default)]
#[command(after_help = "\
Endpoints:
  /metrics       Prometheus text format
  /report.json   the latest report (the --json schema)
  /healthz       200 when samples are fresh, 503 otherwise

Process series carry pid, process, app, and coalition labels; app series carry app
and coalition. By default only processes and apps that used the GPU since the
exporter started are exported; --top limits them further.

Example scrape config:
  - job_name: macgpu
    static_configs: [{targets: ['127.0.0.1:10060']}]")]
pub struct ServeArgs {
    /// Address to listen on, HOST:PORT or PORT. Use 0.0.0.0:PORT to accept
    /// scrapes from other machines.
    #[arg(short, long, default_value_t = format!("127.0.0.1:{DEFAULT_PORT}"), value_name = "ADDR")]
    pub listen: String,

    #[command(flatten)]
    pub sampling: SamplingArgs,

    /// Export at most N process series and N app series, ranked by GPU time in
    /// the latest window, then since start. 0 means no limit.
    #[arg(short = 'n', long, default_value_t = 0)]
    pub top: usize,

    /// Also export processes and apps that have not used the GPU since the
    /// exporter started.
    #[arg(short, long)]
    pub all: bool,

    /// Per-row series to export, comma-separated.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [Series::Process, Series::App],
        value_name = "LEVELS"
    )]
    pub series: Vec<Series>,

    /// Only export processes and apps matching this filter (the --filter syntax).
    #[arg(short, long, value_name = "EXPR")]
    pub filter: Option<String>,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Series {
    Process,
    App,
}

pub fn run(args: &ServeArgs) -> Result<i32> {
    let address = parse_listen(&args.listen)?;
    let listener = TcpListener::bind(address).map_err(|error| bind_error(address, &error))?;
    let local = listener.local_addr().context("listener address")?;
    let server = tiny_http::Server::from_listener(listener, None)
        .map_err(|error| anyhow!("cannot serve on {local}: {error}"))?;

    let config = args.sampling.engine_config(1.0, true);
    let interval = clamp_interval(config.interval);
    let exporter = Exporter::new(ExportOptions {
        top: args.top,
        all: args.all,
        processes: args.series.contains(&Series::Process),
        apps: args.series.contains(&Series::App),
        filter: Filter::parse(args.filter.as_deref().unwrap_or_default()),
    });
    let shared = Arc::new(RwLock::new(Shared::new(exporter.render_empty())));
    let stop = Arc::new(AtomicBool::new(false));

    signals::install(&signals::STOP);
    let collector = {
        let shared = Arc::clone(&shared);
        let stop = Arc::clone(&stop);
        thread::Builder::new()
            .name("macgpu-collector".into())
            .spawn(move || collect(Engine::start(config), exporter, &shared, &stop))
            .context("spawn collector thread")?
    };

    eprintln!(
        "macgpu: serving Prometheus metrics on http://{local}/metrics (sampling every {:.1} s); Ctrl-C stops",
        interval.as_secs_f64()
    );
    let stale_after = (interval * 3).max(Duration::from_secs(5));
    while !stop.load(Ordering::SeqCst) {
        if signals::take().is_some() {
            break;
        }
        match server.recv_timeout(Duration::from_millis(200)) {
            Ok(Some(request)) => respond(request, &shared, stale_after),
            Ok(None) => {}
            Err(error) => eprintln!("macgpu: http: {error}"),
        }
    }
    stop.store(true, Ordering::SeqCst);
    let _ = collector.join();
    Ok(0)
}

/// Accepts `HOST:PORT`, `:PORT`, or `PORT`.
fn parse_listen(text: &str) -> Result<SocketAddr> {
    let text = text.trim();
    let candidate = if let Ok(port) = text.parse::<u16>() {
        format!("127.0.0.1:{port}")
    } else if let Some(port) = text.strip_prefix(':') {
        format!("127.0.0.1:{port}")
    } else {
        text.to_string()
    };
    candidate
        .to_socket_addrs()
        .ok()
        .and_then(|mut addresses| addresses.next())
        .ok_or_else(|| anyhow!("invalid --listen address {text:?}; expected HOST:PORT or PORT"))
}

fn bind_error(address: SocketAddr, error: &io::Error) -> anyhow::Error {
    match error.kind() {
        io::ErrorKind::AddrInUse => anyhow!(
            "port {} on {} is already in use (another macgpu serve?); choose another with --listen {}:PORT",
            address.port(),
            address.ip(),
            address.ip()
        ),
        io::ErrorKind::PermissionDenied => anyhow!(
            "not allowed to listen on {address}; ports below 1024 need root, choose another with --listen"
        ),
        _ => anyhow!("cannot listen on {address}: {error}"),
    }
}

/// What HTTP handlers read. Replaced wholesale after each sample.
struct Shared {
    metrics: Arc<String>,
    report_json: Option<Arc<String>>,
    last_sample: Option<Instant>,
}

impl Shared {
    fn new(metrics: String) -> Self {
        Self {
            metrics: Arc::new(metrics),
            report_json: None,
            last_sample: None,
        }
    }
}

fn collect(engine: Engine, mut exporter: Exporter, shared: &RwLock<Shared>, stop: &AtomicBool) {
    while !stop.load(Ordering::SeqCst) {
        let report = match engine.recv_timeout(Duration::from_millis(200)) {
            Ok(report) => report,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let metrics = exporter.update(&report);
        let json = serde_json::to_string(&report).unwrap_or_default();
        let mut shared = shared.write().expect("exporter state lock");
        shared.metrics = Arc::new(metrics);
        shared.report_json = Some(Arc::new(json));
        shared.last_sample = Some(Instant::now());
    }
    // The sampler stopped: let the HTTP loop end too.
    stop.store(true, Ordering::SeqCst);
}

fn respond(request: tiny_http::Request, shared: &RwLock<Shared>, stale_after: Duration) {
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_string();
    let method_ok = matches!(
        request.method(),
        tiny_http::Method::Get | tiny_http::Method::Head
    );
    let (status, content_type, body): (u16, &str, Arc<String>) = if !method_ok {
        (
            405,
            "text/plain; charset=utf-8",
            Arc::new("method not allowed\n".into()),
        )
    } else {
        let shared = shared.read().expect("exporter state lock");
        match path.as_str() {
            "/metrics" => (
                200,
                "text/plain; version=0.0.4; charset=utf-8",
                Arc::clone(&shared.metrics),
            ),
            "/report.json" => match &shared.report_json {
                Some(json) => (200, "application/json", Arc::clone(json)),
                None => (
                    503,
                    "text/plain; charset=utf-8",
                    Arc::new("no sample yet\n".into()),
                ),
            },
            "/healthz" => {
                let (status, text) = match shared.last_sample {
                    None => (503, "no sample yet\n".to_string()),
                    Some(at) if at.elapsed() > stale_after => (
                        503,
                        format!(
                            "stale: last sample {:.1} s ago\n",
                            at.elapsed().as_secs_f64()
                        ),
                    ),
                    Some(_) => (200, "ok\n".to_string()),
                };
                (status, "text/plain; charset=utf-8", Arc::new(text))
            }
            "/" => (
                200,
                "text/plain; charset=utf-8",
                Arc::new("macgpu exporter\n/metrics\n/report.json\n/healthz\n".into()),
            ),
            _ => (
                404,
                "text/plain; charset=utf-8",
                Arc::new("not found\n".into()),
            ),
        }
    };
    let header = tiny_http::Header::from_bytes("Content-Type", content_type)
        .expect("valid content type header");
    let response = tiny_http::Response::from_string(body.as_str())
        .with_status_code(status)
        .with_header(header);
    // A client that went away is not an error worth reporting.
    let _ = request.respond(response);
}

#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    pub top: usize,
    pub all: bool,
    pub processes: bool,
    pub apps: bool,
    pub filter: Filter,
}

/// Cumulative state across reports, and rendering of the exposition text.
#[derive(Debug, Default)]
pub struct Exporter {
    options: ExportOptions,
    samples: u64,
    device_energy_j: Option<f64>,
    attributed_gpu_s: f64,
    unattributed_gpu_s: f64,
    device_counters: BTreeMap<String, u64>,
    /// Keyed by PID and start time.
    processes: HashMap<(i32, u64), ProcessTotals>,
    /// Keyed by coalition ID (or the bundle key when coalitions are unavailable).
    apps: HashMap<u64, AppTotals>,
}

/// An app that is idle may be left out of reports; its totals are kept this
/// long so the counter continues when it is active again.
const APP_RETENTION_MS: u64 = 3_600_000;

#[derive(Debug, Clone, Copy, Default)]
struct AppTotals {
    gpu_s: f64,
    in_flight_s: Option<f64>,
    last_seen_ms: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ProcessTotals {
    gpu_s: f64,
    energy_j: Option<f64>,
}

impl Exporter {
    pub fn new(options: ExportOptions) -> Self {
        Self {
            options,
            ..Self::default()
        }
    }

    /// Metrics before the first sample: only the exporter's own.
    pub fn render_empty(&self) -> String {
        let mut out = Families::default();
        self.exporter_families(&mut out, None);
        out.render()
    }

    /// Folds `report` into the counters and renders the exposition text.
    pub fn update(&mut self, report: &Report) -> String {
        self.samples += 1;
        let window = report.window_seconds.max(0.0);
        if let Some(power) = report.device.power_w {
            *self.device_energy_j.get_or_insert(0.0) += power * window;
        }
        self.attributed_gpu_s += report.totals.attributed_gpu_ns as f64 / 1e9;
        if let Some(unattributed) = &report.unattributed {
            self.unattributed_gpu_s += unattributed.gpu_ns as f64 / 1e9;
        }
        for (name, increment) in &report.device.counters {
            *self.device_counters.entry(name.clone()).or_default() += increment;
        }

        // The engine runs with idle processes included, so a process missing
        // from the report no longer has GPU clients and its series end.
        let live: HashSet<(i32, u64)> = report
            .processes
            .iter()
            .map(|process| (process.pid, process.start_us))
            .collect();
        self.processes.retain(|key, _| live.contains(key));
        for process in &report.processes {
            let totals = self
                .processes
                .entry((process.pid, process.start_us))
                .or_default();
            totals.gpu_s += process.gpu_ns as f64 / 1e9;
            if let Some(power) = process.est_power_w {
                *totals.energy_j.get_or_insert(0.0) += power * window;
            }
        }
        for app in &report.apps {
            let totals = self.apps.entry(app.coalition_id).or_default();
            totals.gpu_s += app.gpu_ns as f64 / 1e9;
            if let Some(in_flight) = app.in_flight_gpu_ns {
                *totals.in_flight_s.get_or_insert(0.0) += in_flight as f64 / 1e9;
            }
            totals.last_seen_ms = report.timestamp_ms;
        }
        self.apps.retain(|_, totals| {
            report.timestamp_ms.saturating_sub(totals.last_seen_ms) <= APP_RETENTION_MS
        });

        let mut out = Families::default();
        self.exporter_families(&mut out, Some(report));
        device_families(&mut out, report, self);
        if self.options.processes {
            self.process_families(&mut out, report);
        }
        if self.options.apps {
            self.app_families(&mut out, report);
        }
        out.render()
    }

    fn exporter_families(&self, out: &mut Families, report: Option<&Report>) {
        out.gauge(
            "macgpu_exporter_build_info",
            "macgpu version; always 1.",
            labels(&[("version", env!("CARGO_PKG_VERSION"))]),
            1.0,
        );
        out.counter(
            "macgpu_exporter_samples_total",
            "Samples taken since the exporter started.",
            String::new(),
            self.samples as f64,
        );
        let Some(report) = report else {
            return;
        };
        out.gauge(
            "macgpu_exporter_window_seconds",
            "Length of the latest sampling window.",
            String::new(),
            report.window_seconds,
        );
        out.gauge(
            "macgpu_exporter_last_sample_timestamp_seconds",
            "End of the latest sampling window, in seconds since the Unix epoch.",
            String::new(),
            report.timestamp_ms as f64 / 1000.0,
        );
        out.gauge(
            "macgpu_exporter_sample_duration_seconds",
            "Wall time to take the latest sample.",
            String::new(),
            report.overhead.sample_wall_ms / 1000.0,
        );
        out.gauge(
            "macgpu_exporter_sample_cpu_seconds",
            "Sampler CPU time for the latest sample.",
            String::new(),
            report.overhead.sample_cpu_ms / 1000.0,
        );
        out.gauge(
            "macgpu_exporter_cpu_ratio",
            "macgpu's whole-process CPU use over the latest window, in cores.",
            String::new(),
            report.overhead.process_cpu_pct / 100.0,
        );
        for (source, status) in &report.sources {
            let state = serde_json::to_value(status.state)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_default();
            let up = matches!(status.state, SourceState::Ok | SourceState::Partial);
            out.gauge(
                "macgpu_source_up",
                "1 when a data source produced data for the latest window; state gives the detail.",
                labels(&[("source", source), ("state", &state)]),
                f64::from(u8::from(up)),
            );
        }
    }

    fn process_families(&self, out: &mut Families, report: &Report) {
        let app_names: HashMap<u64, &str> = report
            .apps
            .iter()
            .map(|app| (app.coalition_id, app.name.as_str()))
            .collect();
        let mut selected: Vec<_> = report
            .processes
            .iter()
            .filter(|process| {
                let totals = self.processes.get(&(process.pid, process.start_us));
                self.options.all || totals.is_some_and(|totals| totals.gpu_s > 0.0)
            })
            .filter(|process| {
                self.options.filter.is_empty()
                    || self.options.filter.matches(&crate::view::Row {
                        pid: Some(process.pid),
                        name: process.name.clone(),
                        detail: process.path.clone(),
                        ..row_template()
                    })
            })
            .collect();
        let total_gpu = |process: &crate::model::ProcessView| {
            self.processes
                .get(&(process.pid, process.start_us))
                .map(|totals| totals.gpu_s)
                .unwrap_or(0.0)
        };
        selected.sort_by(|a, b| {
            b.gpu_ns
                .cmp(&a.gpu_ns)
                .then_with(|| total_gpu(b).total_cmp(&total_gpu(a)))
                .then_with(|| a.pid.cmp(&b.pid))
        });
        let omitted = if self.options.top > 0 && selected.len() > self.options.top {
            let omitted = selected.len() - self.options.top;
            selected.truncate(self.options.top);
            omitted
        } else {
            0
        };
        out.gauge(
            "macgpu_exporter_omitted_processes",
            "GPU processes left out of the per-process series by --top.",
            String::new(),
            omitted as f64,
        );

        for process in selected {
            let coalition = process
                .coalition_id
                .map(|id| id.to_string())
                .unwrap_or_default();
            let app = process
                .coalition_id
                .and_then(|id| app_names.get(&id).copied())
                .unwrap_or_default();
            let pid = process.pid.to_string();
            let set = labels(&[
                ("pid", &pid),
                ("process", &process.name),
                ("app", app),
                ("coalition", &coalition),
            ]);
            let totals = self
                .processes
                .get(&(process.pid, process.start_us))
                .copied()
                .unwrap_or_default();
            out.counter(
                "macgpu_process_gpu_seconds_total",
                "GPU time the driver apportioned to the process (AppUsage) since the exporter first saw it. Busy time is split among work running at the same moment, so all processes together grow by at most about 1 per second per GPU.",
                set.clone(),
                totals.gpu_s,
            );
            out.gauge(
                "macgpu_process_gpu_busy_ratio",
                "Apportioned GPU time per wall-clock second over the latest window. All processes sum to at most about 1 when the GPU is saturated.",
                set.clone(),
                process.busy_ms_per_s / 1000.0,
            );
            out.gauge(
                "macgpu_process_gpu_share_ratio",
                "Share of all processes' apportioned GPU time in the latest window.",
                set.clone(),
                process.share_pct / 100.0,
            );
            if let Some(device) = process.device_pct {
                out.gauge(
                    "macgpu_process_gpu_device_ratio",
                    "The process's slice of the device's active time (share times device active ratio).",
                    set.clone(),
                    device / 100.0,
                );
            }
            if let Some(power) = process.est_power_w {
                out.gauge(
                    "macgpu_process_power_watts",
                    "Estimated GPU power: the app's measured GPU power apportioned by this process's share of the app's GPU time.",
                    set.clone(),
                    power,
                );
            }
            if let Some(energy) = totals.energy_j {
                out.counter(
                    "macgpu_process_energy_joules_total",
                    "Estimated GPU energy since the exporter first saw the process.",
                    set.clone(),
                    energy,
                );
            }
            if let Some(bytes) = process.graphics_memory_bytes {
                out.gauge(
                    "macgpu_process_graphics_memory_bytes",
                    "Graphics memory footprint (TASK_VM_INFO ledger_tag_graphics_footprint); absent for other users' processes without root.",
                    set.clone(),
                    bytes as f64,
                );
            }
            out.gauge(
                "macgpu_process_queues",
                "Metal command queues (AppUsage entries) across the process's GPU clients.",
                set.clone(),
                process.queues as f64,
            );
            if let Some(age) = process.last_submit_age_ms {
                out.gauge(
                    "macgpu_process_last_submit_age_seconds",
                    "Time since the process last submitted GPU work.",
                    set,
                    age as f64 / 1000.0,
                );
            }
        }
    }

    fn app_families(&self, out: &mut Families, report: &Report) {
        let mut selected: Vec<_> = report
            .apps
            .iter()
            .filter(|app| {
                self.options.all
                    || self
                        .apps
                        .get(&app.coalition_id)
                        .is_some_and(|totals| totals.gpu_s > 0.0)
            })
            .filter(|app| {
                self.options.filter.is_empty()
                    || self.options.filter.matches(&crate::view::Row {
                        pid: app.responsible_pid,
                        name: app.name.clone(),
                        detail: app.bundle_path.clone(),
                        ..row_template()
                    })
            })
            .collect();
        selected.sort_by(|a, b| {
            b.gpu_ns
                .cmp(&a.gpu_ns)
                .then_with(|| b.lifetime_gpu_ns.cmp(&a.lifetime_gpu_ns))
                .then_with(|| a.coalition_id.cmp(&b.coalition_id))
        });
        let omitted = if self.options.top > 0 && selected.len() > self.options.top {
            let omitted = selected.len() - self.options.top;
            selected.truncate(self.options.top);
            omitted
        } else {
            0
        };
        out.gauge(
            "macgpu_exporter_omitted_apps",
            "Apps left out of the per-app series by --top.",
            String::new(),
            omitted as f64,
        );

        for app in selected {
            let coalition = app.coalition_id.to_string();
            let set = labels(&[("app", &app.name), ("coalition", &coalition)]);
            let totals = self
                .apps
                .get(&app.coalition_id)
                .copied()
                .unwrap_or_default();
            out.counter(
                "macgpu_app_gpu_seconds_total",
                "App GPU time (its processes' apportioned time plus known unattributed time of exited helpers) since the exporter first saw the app.",
                set.clone(),
                totals.gpu_s,
            );
            if let Some(in_flight) = totals.in_flight_s {
                out.counter(
                    "macgpu_app_gpu_in_flight_seconds_total",
                    "Wall time the app's coalition had GPU work in flight since the exporter first saw it. Work overlapping another app's counts fully for each, so apps together can exceed 1 per second.",
                    set.clone(),
                    in_flight,
                );
            }
            if let Some(ratio) = app.in_flight_ms_per_s {
                out.gauge(
                    "macgpu_app_gpu_in_flight_ratio",
                    "Share of the latest window the app had GPU work in flight (overlap with other apps counts fully).",
                    set.clone(),
                    ratio / 1000.0,
                );
            }
            if let Some(energy) = app.lifetime_energy_j {
                out.counter(
                    "macgpu_app_gpu_energy_joules_total",
                    "Measured GPU energy of the app's resource coalition since the coalition was created (kernel gpu_energy_nj counter). A relaunched app gets a new coalition label.",
                    set.clone(),
                    energy,
                );
            }
            out.gauge(
                "macgpu_app_gpu_busy_ratio",
                "App GPU time (its processes' apportioned time plus known unattributed time) per wall-clock second over the latest window.",
                set.clone(),
                app.busy_ms_per_s / 1000.0,
            );
            out.gauge(
                "macgpu_app_gpu_share_ratio",
                "Share of all apps' GPU time in the latest window.",
                set.clone(),
                app.share_pct / 100.0,
            );
            if let Some(device) = app.device_pct {
                out.gauge(
                    "macgpu_app_gpu_device_ratio",
                    "The app's slice of the device's active time.",
                    set.clone(),
                    device / 100.0,
                );
            }
            if let Some(unattributed) = app.unattributed_gpu_ns {
                out.gauge(
                    "macgpu_app_unattributed_gpu_busy_ratio",
                    "Coalition GPU time per second of tasks without a live GPU client (exited helpers). Absent when it cannot be separated from overlapping GPU work.",
                    set.clone(),
                    unattributed as f64 / 1e9 / report.window_seconds.max(1e-9),
                );
            }
            for (name, help, value) in [
                (
                    "macgpu_app_power_watts",
                    "Measured GPU power of the app's coalition over the latest window.",
                    app.power_w,
                ),
                (
                    "macgpu_app_billed_to_me_watts",
                    "GPU power other coalitions (such as WindowServer) spent on this app's behalf.",
                    app.billed_to_me_w,
                ),
                (
                    "macgpu_app_billed_to_others_watts",
                    "GPU power this app spent on behalf of others.",
                    app.billed_to_others_w,
                ),
                (
                    "macgpu_app_ane_power_watts",
                    "Neural Engine power of the app's coalition.",
                    app.ane_power_w,
                ),
            ] {
                if let Some(value) = value {
                    out.gauge(name, help, set.clone(), value);
                }
            }
            if let Some(ane) = app.ane_ms_per_s {
                out.gauge(
                    "macgpu_app_ane_busy_ratio",
                    "Neural Engine time per wall-clock second.",
                    set.clone(),
                    ane / 1000.0,
                );
            }
            if let Some(bytes) = app.graphics_memory_bytes {
                out.gauge(
                    "macgpu_app_graphics_memory_bytes",
                    "Sum of the known graphics memory of the app's processes.",
                    set.clone(),
                    bytes as f64,
                );
            }
            out.gauge(
                "macgpu_app_processes",
                "Live processes in the app's coalition.",
                set,
                app.members as f64,
            );
        }
    }
}

fn device_families(out: &mut Families, report: &Report, exporter: &Exporter) {
    let device = &report.device;
    if device.model.is_some() || device.gpu_cores.is_some() {
        let cores = device.gpu_cores.map(|c| c.to_string()).unwrap_or_default();
        out.gauge(
            "macgpu_gpu_info",
            "GPU identity; always 1.",
            labels(&[
                ("model", device.model.as_deref().unwrap_or_default()),
                ("cores", &cores),
            ]),
            1.0,
        );
    }
    let ratio = |value: Option<f64>| value.map(|pct| pct / 100.0);
    for (name, help, value) in [
        (
            "macgpu_gpu_active_ratio",
            "Share of the window the GPU spent in an active P-state (IOReport GPUPH).",
            ratio(device.active_pct),
        ),
        (
            "macgpu_gpu_freq_normalized_ratio",
            "Active ratio weighted by clock: active * average clock / max clock.",
            ratio(device.freq_normalized_pct),
        ),
        (
            "macgpu_gpu_frequency_hertz",
            "Residency-weighted average GPU clock while active.",
            device.avg_mhz.map(|mhz| mhz * 1e6),
        ),
        (
            "macgpu_gpu_max_frequency_hertz",
            "Highest GPU clock in the frequency table.",
            device.max_mhz.map(|mhz| mhz * 1e6),
        ),
        (
            "macgpu_gpu_power_watts",
            "Average GPU power over the window (IOReport GPU Energy).",
            device.power_w,
        ),
        (
            "macgpu_gpu_limited_ratio",
            "Share of the window the GPU spent under a power or thermal limiter state (PWRCTRL limiting states or a CLTM cap); clocks are not necessarily reduced.",
            ratio(alerts::limited_pct(device)),
        ),
        (
            "macgpu_gpu_thermal_limited_ratio",
            "Share of the window under thermal (CLTM) limiting.",
            ratio(device.thermal_limited_pct),
        ),
        (
            "macgpu_gpu_memory_in_use_bytes",
            "GPU memory in use (PerformanceStatistics In use system memory).",
            device.memory.in_use_bytes.map(|b| b as f64),
        ),
        (
            "macgpu_gpu_memory_allocated_bytes",
            "GPU memory allocated (PerformanceStatistics Alloc system memory).",
            device.memory.allocated_bytes.map(|b| b as f64),
        ),
        (
            "macgpu_gpu_wired_limit_bytes",
            "Effective GPU wired-memory limit (iogpu.wired_limit_mb or the default).",
            device.memory.wired_limit_bytes.map(|b| b as f64),
        ),
        (
            "macgpu_gpu_recommended_working_set_bytes",
            "MTLDevice.recommendedMaxWorkingSetSize.",
            device
                .memory
                .recommended_working_set_bytes
                .map(|b| b as f64),
        ),
        (
            "macgpu_system_memory_bytes",
            "Physical memory.",
            device.memory.total_physical_bytes.map(|b| b as f64),
        ),
    ] {
        if let Some(value) = value {
            out.gauge(name, help, String::new(), value);
        }
    }
    if let Some(energy) = exporter.device_energy_j {
        out.counter(
            "macgpu_gpu_energy_joules_total",
            "GPU energy since the exporter started (sum of window power times window length).",
            String::new(),
            energy,
        );
    }
    if let Some(temperature) = &device.temperature {
        for (stat, value) in [("average", temperature.avg_c), ("max", temperature.max_c)] {
            out.gauge(
                "macgpu_gpu_temperature_celsius",
                "GPU die temperature from the SMC: mean or hottest sensor.",
                labels(&[("stat", stat)]),
                value,
            );
        }
    }
    if let Some(firmware) = &device.firmware_utilization {
        for (engine, value) in [
            ("device", firmware.device_pct),
            ("renderer", firmware.renderer_pct),
            ("tiler", firmware.tiler_pct),
        ] {
            out.gauge(
                "macgpu_gpu_firmware_utilization_ratio",
                "Firmware PerformanceStatistics utilization, averaged over readings in the window.",
                labels(&[("engine", engine)]),
                value / 100.0,
            );
        }
    }
    for pstate in &device.pstates {
        let mhz = pstate
            .mhz
            .map(|mhz| format!("{mhz:.0}"))
            .unwrap_or_default();
        out.gauge(
            "macgpu_gpu_pstate_residency_ratio",
            "Share of the window spent in each GPU P-state, including OFF.",
            labels(&[("state", &pstate.name), ("mhz", &mhz)]),
            pstate.pct / 100.0,
        );
    }
    for limiter in &device.limiters {
        out.gauge(
            "macgpu_gpu_limiter_residency_ratio",
            "Share of the window in each power-controller limiter state (IOReport PWRCTRL).",
            labels(&[("state", &limiter.name)]),
            limiter.pct / 100.0,
        );
    }
    for (name, total) in &exporter.device_counters {
        out.counter(
            "macgpu_gpu_counter_events_total",
            "Increments of IOReport throttle, thermal, and issue counters since the exporter started.",
            labels(&[("counter", name)]),
            *total as f64,
        );
    }
    if let Some(recoveries) = device.recovery_count {
        out.counter(
            "macgpu_gpu_recoveries_total",
            "GPU hang recoveries since boot.",
            String::new(),
            recoveries as f64,
        );
    }
    for (name, help, current, levels) in [
        (
            "macgpu_thermal_pressure",
            "System thermal pressure level; 1 for the current level.",
            device.system.thermal_pressure.as_deref(),
            &["nominal", "moderate", "heavy", "trapping", "sleeping"][..],
        ),
        (
            "macgpu_memory_pressure",
            "System memory pressure level; 1 for the current level.",
            device.system.memory_pressure.as_deref(),
            &["normal", "warning", "critical"][..],
        ),
    ] {
        let Some(current) = current else {
            continue;
        };
        let mut levels: Vec<&str> = levels.to_vec();
        if !levels.contains(&current) {
            levels.push(current);
        }
        for level in levels {
            out.gauge(
                name,
                help,
                labels(&[("level", level)]),
                f64::from(u8::from(level == current)),
            );
        }
    }

    let totals = &report.totals;
    out.gauge(
        "macgpu_gpu_attributed_busy_ratio",
        "Apportioned GPU time of all processes per wall-clock second over the latest window (at most about 1 per GPU).",
        String::new(),
        totals.attributed_ms_per_s / 1000.0,
    );
    out.counter(
        "macgpu_gpu_attributed_seconds_total",
        "Apportioned GPU time of all processes since the exporter started.",
        String::new(),
        exporter.attributed_gpu_s,
    );
    if let Some(unattributed) = &report.unattributed {
        out.gauge(
            "macgpu_gpu_unattributed_busy_ratio",
            "Coalition GPU time per second that no live client accounts for (exited clients); a lower bound when macgpu_gpu_unattributed_partial is 1.",
            String::new(),
            unattributed.busy_ms_per_s / 1000.0,
        );
        out.gauge(
            "macgpu_gpu_unattributed_partial",
            "1 when some apps' unattributed time could not be separated from overlapping GPU work, so the unattributed series are lower bounds.",
            String::new(),
            f64::from(u8::from(unattributed.partial)),
        );
    }
    if report.unattributed.is_some() || exporter.unattributed_gpu_s > 0.0 {
        out.counter(
            "macgpu_gpu_unattributed_seconds_total",
            "Coalition GPU time no live client accounted for, since the exporter started (a lower bound; see macgpu_gpu_unattributed_partial).",
            String::new(),
            exporter.unattributed_gpu_s,
        );
    }
    for (name, help, value) in [
        (
            "macgpu_gpu_clients",
            "GPU client connections.",
            totals.gpu_clients,
        ),
        (
            "macgpu_gpu_processes",
            "Processes with GPU clients.",
            totals.gpu_processes,
        ),
        (
            "macgpu_gpu_active_processes",
            "Processes that used the GPU in the latest window.",
            totals.active_processes,
        ),
        (
            "macgpu_gpu_active_apps",
            "Apps that used the GPU in the latest window.",
            totals.active_apps,
        ),
    ] {
        out.gauge(name, help, String::new(), value as f64);
    }
}

fn row_template() -> crate::view::Row {
    crate::view::Row {
        key: crate::view::RowKey::Unattributed,
        pid: None,
        name: String::new(),
        detail: None,
        gpu_ns: 0,
        busy_ms_per_s: 0.0,
        share_pct: 0.0,
        device_pct: None,
        power_w: None,
        power_estimated: false,
        memory_bytes: None,
        lifetime_gpu_ns: 0,
        activity: crate::model::Activity::Idle,
        last_submit_age_ms: None,
        children: 0,
        unattributed_ms_per_s: None,
        uid: None,
    }
}

/// Formats a label set: `{a="x",b="y"}`, escaping per the text format.
fn labels(pairs: &[(&str, &str)]) -> String {
    if pairs.is_empty() {
        return String::new();
    }
    let inner: Vec<String> = pairs
        .iter()
        .map(|(name, value)| format!("{name}=\"{}\"", escape_label(value)))
        .collect();
    format!("{{{}}}", inner.join(","))
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Gauge,
    Counter,
}

/// One metric family: its type, help, and samples as (labels, value).
struct Family {
    kind: Kind,
    help: &'static str,
    samples: Vec<(String, f64)>,
}

/// Metric families in first-seen order, so every sample of a family is
/// emitted together under one HELP and TYPE.
#[derive(Default)]
struct Families {
    order: Vec<&'static str>,
    families: HashMap<&'static str, Family>,
}

impl Families {
    fn gauge(&mut self, name: &'static str, help: &'static str, labels: String, value: f64) {
        self.add(name, Kind::Gauge, help, labels, value);
    }

    fn counter(&mut self, name: &'static str, help: &'static str, labels: String, value: f64) {
        debug_assert!(name.ends_with("_total"));
        self.add(name, Kind::Counter, help, labels, value);
    }

    fn add(
        &mut self,
        name: &'static str,
        kind: Kind,
        help: &'static str,
        labels: String,
        value: f64,
    ) {
        if !value.is_finite() {
            return;
        }
        let family = self.families.entry(name).or_insert_with(|| {
            self.order.push(name);
            Family {
                kind,
                help,
                samples: Vec::new(),
            }
        });
        family.samples.push((labels, value));
    }

    fn render(&self) -> String {
        let mut out = String::new();
        for name in &self.order {
            let Family {
                kind,
                help,
                samples,
            } = &self.families[name];
            let kind = match kind {
                Kind::Gauge => "gauge",
                Kind::Counter => "counter",
            };
            let _ = writeln!(out, "# HELP {name} {help}");
            // Text format 0.0.4: HELP, TYPE, and samples share the full name,
            // `_total` included.
            let _ = writeln!(out, "# TYPE {name} {kind}");
            for (labels, value) in samples {
                let _ = writeln!(out, "{name}{labels} {value}");
            }
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::{
        AppView, DeviceMemory, DeviceView, FirmwareUtilization, PStateShare, ProcessView,
        SourceStatus, StateShare, SystemView, Temperature, Totals, Unattributed,
    };

    /// A report with every field the exporter reads.
    pub(crate) fn full_report() -> Report {
        Report {
            schema_version: 1,
            timestamp_ms: 1_798_300_801_000,
            window_seconds: 2.0,
            device: DeviceView {
                model: Some("Apple M4 Max".into()),
                gpu_cores: Some(40),
                firmware_utilization: Some(FirmwareUtilization {
                    device_pct: 50.0,
                    renderer_pct: 40.0,
                    tiler_pct: 10.0,
                    readings: 8,
                }),
                active_pct: Some(60.0),
                avg_mhz: Some(1300.0),
                max_mhz: Some(1578.0),
                freq_normalized_pct: Some(49.4),
                power_w: Some(20.0),
                pstates: vec![
                    PStateShare {
                        name: "OFF".into(),
                        mhz: None,
                        pct: 40.0,
                    },
                    PStateShare {
                        name: "P15".into(),
                        mhz: Some(1578.0),
                        pct: 60.0,
                    },
                ],
                limiters: vec![StateShare {
                    name: "PWR_ZONE".into(),
                    pct: 25.0,
                }],
                thermal_limited_pct: Some(5.0),
                counters: [("GPU_THROTTLE".to_string(), 3u64)].into_iter().collect(),
                temperature: Some(Temperature {
                    avg_c: 70.0,
                    max_c: 80.5,
                    sensors: 44,
                }),
                memory: DeviceMemory {
                    in_use_bytes: Some(4 << 30),
                    allocated_bytes: Some(6 << 30),
                    wired_limit_bytes: Some(96 << 30),
                    recommended_working_set_bytes: Some(100 << 30),
                    total_physical_bytes: Some(128 << 30),
                },
                system: SystemView {
                    thermal_pressure: Some("nominal".into()),
                    memory_pressure: Some("warning".into()),
                },
                recovery_count: Some(2),
            },
            totals: Totals {
                attributed_gpu_ns: 1_500_000_000,
                attributed_ms_per_s: 750.0,
                gpu_clients: 9,
                gpu_processes: 3,
                active_processes: 2,
                active_apps: 1,
            },
            processes: vec![
                ProcessView {
                    pid: 900,
                    start_us: 5,
                    name: "Google \"Chrome\" Helper".into(),
                    path: Some("/Applications/Google Chrome.app/x".into()),
                    coalition_id: Some(77),
                    gpu_ns: 1_000_000_000,
                    busy_ms_per_s: 500.0,
                    share_pct: 66.7,
                    device_pct: Some(40.0),
                    est_power_w: Some(12.0),
                    lifetime_gpu_ns: 50_000_000_000,
                    graphics_memory_bytes: Some(1 << 30),
                    queues: 2,
                    last_submit_age_ms: Some(20),
                    ..ProcessView::default()
                },
                ProcessView {
                    pid: 431,
                    start_us: 6,
                    name: "WindowServer".into(),
                    coalition_id: Some(10),
                    gpu_ns: 500_000_000,
                    busy_ms_per_s: 250.0,
                    share_pct: 33.3,
                    queues: 3,
                    ..ProcessView::default()
                },
                ProcessView {
                    pid: 777,
                    start_us: 7,
                    name: "idle".into(),
                    ..ProcessView::default()
                },
            ],
            apps: vec![AppView {
                coalition_id: 77,
                name: "Google Chrome".into(),
                bundle_path: Some("/Applications/Google Chrome.app".into()),
                responsible_pid: Some(899),
                gpu_ns: 1_000_000_000,
                busy_ms_per_s: 500.0,
                share_pct: 100.0,
                device_pct: Some(40.0),
                attributed_gpu_ns: 900_000_000,
                in_flight_gpu_ns: Some(1_600_000_000),
                in_flight_ms_per_s: Some(800.0),
                unattributed_gpu_ns: Some(100_000_000),
                energy_j: Some(24.0),
                power_w: Some(12.0),
                billed_to_me_w: Some(0.5),
                billed_to_others_w: Some(0.0),
                ane_ms_per_s: Some(3.0),
                ane_power_w: Some(0.1),
                lifetime_gpu_ns: 80_000_000_000,
                lifetime_energy_j: Some(1500.0),
                graphics_memory_bytes: Some(1 << 30),
                members: 12,
                gpu_pids: vec![900],
            }],
            unattributed: Some(Unattributed {
                gpu_ns: 100_000_000,
                busy_ms_per_s: 50.0,
                partial: true,
            }),
            sources: [
                ("registry".to_string(), SourceStatus::ok()),
                (
                    "temperature".to_string(),
                    SourceStatus::with(SourceState::Disabled, "disabled"),
                ),
            ]
            .into_iter()
            .collect(),
            ..Report::default()
        }
    }

    /// Parses exposition text into (name, labels, value), checking that every
    /// sample follows a TYPE line for its family.
    pub(crate) fn parse_exposition(text: &str) -> Vec<(String, BTreeMap<String, String>, f64)> {
        let mut types: HashMap<String, String> = HashMap::new();
        let mut helps: HashSet<String> = HashSet::new();
        let mut samples = Vec::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("# HELP ") {
                let name = rest.split(' ').next().unwrap().to_string();
                assert!(helps.insert(name.clone()), "duplicate HELP for {name}");
                continue;
            }
            if let Some(rest) = line.strip_prefix("# TYPE ") {
                let mut parts = rest.split(' ');
                let family = parts.next().unwrap().to_string();
                let kind = parts.next().unwrap().to_string();
                assert!(kind == "gauge" || kind == "counter", "{line}");
                assert!(
                    types.insert(family.clone(), kind).is_none(),
                    "duplicate TYPE {family}"
                );
                continue;
            }
            assert!(!line.starts_with('#'), "{line}");
            let (series, value) = line.rsplit_once(' ').expect("sample line");
            let value: f64 = value.parse().unwrap_or_else(|_| panic!("value in {line}"));
            let (name, labels) = match series.split_once('{') {
                Some((name, rest)) => (name, rest.strip_suffix('}').expect("closing brace")),
                None => (series, ""),
            };
            assert!(
                name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "metric name {name}"
            );
            let kind = types
                .get(name)
                .unwrap_or_else(|| panic!("no TYPE before {name}"));
            if kind == "counter" {
                assert!(
                    name.ends_with("_total"),
                    "counter {name} must end in _total"
                );
            }
            let mut parsed = BTreeMap::new();
            let mut rest = labels;
            while !rest.is_empty() {
                let (key, after) = rest.split_once("=\"").expect("label");
                let mut value = String::new();
                let mut chars = after.char_indices();
                let end = loop {
                    let (index, c) = chars.next().expect("closing quote");
                    match c {
                        '\\' => {
                            let (_, escaped) = chars.next().unwrap();
                            value.push(if escaped == 'n' { '\n' } else { escaped });
                        }
                        '"' => break index,
                        c => value.push(c),
                    }
                };
                parsed.insert(key.to_string(), value);
                rest = after[end + 1..].trim_start_matches(',');
            }
            samples.push((name.to_string(), parsed, value));
        }
        samples
    }

    fn value(
        samples: &[(String, BTreeMap<String, String>, f64)],
        name: &str,
        label: Option<(&str, &str)>,
    ) -> Option<f64> {
        samples
            .iter()
            .find(|(n, labels, _)| {
                n == name && label.is_none_or(|(k, v)| labels.get(k).map(String::as_str) == Some(v))
            })
            .map(|(_, _, value)| *value)
    }

    fn options() -> ExportOptions {
        ExportOptions {
            processes: true,
            apps: true,
            ..ExportOptions::default()
        }
    }

    #[test]
    fn exports_every_field_with_correct_types() {
        let mut exporter = Exporter::new(options());
        let text = exporter.update(&full_report());
        let samples = parse_exposition(&text);
        let get = |name: &str| value(&samples, name, None);

        assert_eq!(get("macgpu_gpu_active_ratio"), Some(0.6));
        assert_eq!(get("macgpu_gpu_frequency_hertz"), Some(1.3e9));
        assert_eq!(get("macgpu_gpu_power_watts"), Some(20.0));
        assert_eq!(get("macgpu_gpu_energy_joules_total"), Some(40.0));
        // PWR_ZONE 25% of powered time, powered 60% of the window; CLTM 5%.
        assert_eq!(get("macgpu_gpu_limited_ratio"), Some(0.15));
        assert_eq!(get("macgpu_gpu_unattributed_partial"), Some(1.0));
        assert_eq!(get("macgpu_gpu_recoveries_total"), Some(2.0));
        assert_eq!(
            get("macgpu_gpu_memory_in_use_bytes"),
            Some((4u64 << 30) as f64)
        );
        assert_eq!(get("macgpu_gpu_attributed_seconds_total"), Some(1.5));
        assert_eq!(get("macgpu_gpu_unattributed_seconds_total"), Some(0.1));
        assert_eq!(get("macgpu_exporter_samples_total"), Some(1.0));
        assert_eq!(
            value(
                &samples,
                "macgpu_gpu_temperature_celsius",
                Some(("stat", "max"))
            ),
            Some(80.5)
        );
        assert_eq!(
            value(
                &samples,
                "macgpu_gpu_pstate_residency_ratio",
                Some(("mhz", "1578"))
            ),
            Some(0.6)
        );
        assert_eq!(
            value(
                &samples,
                "macgpu_gpu_counter_events_total",
                Some(("counter", "GPU_THROTTLE"))
            ),
            Some(3.0)
        );
        assert_eq!(
            value(
                &samples,
                "macgpu_memory_pressure",
                Some(("level", "warning"))
            ),
            Some(1.0)
        );
        assert_eq!(
            value(
                &samples,
                "macgpu_memory_pressure",
                Some(("level", "normal"))
            ),
            Some(0.0)
        );
        assert_eq!(
            value(
                &samples,
                "macgpu_gpu_firmware_utilization_ratio",
                Some(("engine", "tiler"))
            ),
            Some(0.1)
        );
        assert_eq!(
            value(
                &samples,
                "macgpu_source_up",
                Some(("source", "temperature"))
            ),
            Some(0.0)
        );
        assert_eq!(
            value(&samples, "macgpu_gpu_info", Some(("model", "Apple M4 Max"))),
            Some(1.0)
        );

        // Process labels, escaping, and app name lookup.
        let chrome = samples
            .iter()
            .find(|(name, labels, _)| {
                name == "macgpu_process_gpu_seconds_total" && labels["pid"] == "900"
            })
            .expect("chrome series");
        assert_eq!(chrome.1["process"], "Google \"Chrome\" Helper");
        assert_eq!(chrome.1["app"], "Google Chrome");
        assert_eq!(chrome.1["coalition"], "77");
        assert_eq!(chrome.2, 1.0);
        assert!(text.contains(r#"process="Google \"Chrome\" Helper""#));
        assert_eq!(
            value(
                &samples,
                "macgpu_process_energy_joules_total",
                Some(("pid", "900"))
            ),
            Some(24.0)
        );
        assert_eq!(
            value(
                &samples,
                "macgpu_process_graphics_memory_bytes",
                Some(("pid", "900"))
            ),
            Some((1u64 << 30) as f64)
        );
        assert_eq!(
            value(
                &samples,
                "macgpu_process_gpu_busy_ratio",
                Some(("pid", "431"))
            ),
            Some(0.25)
        );
        // Idle since start: not exported by default.
        assert_eq!(
            value(&samples, "macgpu_process_queues", Some(("pid", "777"))),
            None
        );
        // No estimate for WindowServer: no power series rather than zero.
        assert_eq!(
            value(&samples, "macgpu_process_power_watts", Some(("pid", "431"))),
            None
        );

        assert_eq!(
            value(
                &samples,
                "macgpu_app_gpu_seconds_total",
                Some(("app", "Google Chrome"))
            ),
            Some(1.0)
        );
        assert_eq!(get("macgpu_app_gpu_energy_joules_total"), Some(1500.0));
        assert_eq!(get("macgpu_app_unattributed_gpu_busy_ratio"), Some(0.05));
        assert_eq!(get("macgpu_app_billed_to_me_watts"), Some(0.5));
        assert_eq!(get("macgpu_app_gpu_in_flight_seconds_total"), Some(1.6));
        assert_eq!(get("macgpu_app_gpu_in_flight_ratio"), Some(0.8));
        assert_eq!(get("macgpu_app_processes"), Some(12.0));
    }

    #[test]
    fn counters_accumulate_and_end_with_the_process() {
        let mut exporter = Exporter::new(options());
        exporter.update(&full_report());
        let mut second = full_report();
        second.processes[0].gpu_ns = 2_000_000_000;
        second.processes[0].est_power_w = None;
        let samples = parse_exposition(&exporter.update(&second));
        assert_eq!(
            value(
                &samples,
                "macgpu_process_gpu_seconds_total",
                Some(("pid", "900"))
            ),
            Some(3.0)
        );
        assert_eq!(
            value(
                &samples,
                "macgpu_process_energy_joules_total",
                Some(("pid", "900"))
            ),
            Some(24.0)
        );
        assert_eq!(
            value(&samples, "macgpu_gpu_energy_joules_total", None),
            Some(80.0)
        );
        assert_eq!(
            value(&samples, "macgpu_exporter_samples_total", None),
            Some(2.0)
        );
        // Idle in this window but used the GPU earlier: still exported.
        let mut third = full_report();
        third.processes[1].gpu_ns = 0;
        let samples = parse_exposition(&exporter.update(&third));
        assert_eq!(
            value(
                &samples,
                "macgpu_process_gpu_seconds_total",
                Some(("pid", "431"))
            ),
            Some(1.0)
        );

        // The process exits; a new process reusing the PID starts from zero.
        let mut fourth = full_report();
        fourth.processes[0].start_us = 99;
        let samples = parse_exposition(&exporter.update(&fourth));
        assert_eq!(
            value(
                &samples,
                "macgpu_process_gpu_seconds_total",
                Some(("pid", "900"))
            ),
            Some(1.0)
        );
    }

    #[test]
    fn app_counters_continue_while_the_app_is_absent() {
        let mut exporter = Exporter::new(options());
        exporter.update(&full_report());
        let mut idle = full_report();
        idle.apps.clear();
        idle.timestamp_ms += 60_000;
        let samples = parse_exposition(&exporter.update(&idle));
        assert_eq!(value(&samples, "macgpu_app_gpu_seconds_total", None), None);
        let mut back = full_report();
        back.timestamp_ms += 120_000;
        let samples = parse_exposition(&exporter.update(&back));
        assert_eq!(
            value(&samples, "macgpu_app_gpu_seconds_total", None),
            Some(2.0)
        );
        // Gone for more than an hour: starts over.
        let mut later = full_report();
        later.apps.clear();
        later.timestamp_ms += 120_000 + APP_RETENTION_MS + 1;
        exporter.update(&later);
        let mut again = full_report();
        again.timestamp_ms = later.timestamp_ms + 1;
        let samples = parse_exposition(&exporter.update(&again));
        assert_eq!(
            value(&samples, "macgpu_app_gpu_seconds_total", None),
            Some(1.0)
        );
    }

    #[test]
    fn cardinality_controls() {
        let mut exporter = Exporter::new(ExportOptions {
            top: 1,
            ..options()
        });
        let samples = parse_exposition(&exporter.update(&full_report()));
        let processes: Vec<_> = samples
            .iter()
            .filter(|(name, _, _)| name == "macgpu_process_gpu_seconds_total")
            .collect();
        assert_eq!(processes.len(), 1);
        assert_eq!(processes[0].1["pid"], "900");
        assert_eq!(
            value(&samples, "macgpu_exporter_omitted_processes", None),
            Some(1.0)
        );

        let mut exporter = Exporter::new(ExportOptions {
            all: true,
            ..options()
        });
        let samples = parse_exposition(&exporter.update(&full_report()));
        assert_eq!(
            value(&samples, "macgpu_process_queues", Some(("pid", "777"))),
            Some(0.0)
        );

        let mut exporter = Exporter::new(ExportOptions {
            filter: Filter::parse("window"),
            apps: false,
            ..options()
        });
        let samples = parse_exposition(&exporter.update(&full_report()));
        assert!(samples.iter().all(
            |(name, labels, _)| !name.starts_with("macgpu_process_") || labels["pid"] == "431"
        ));
        assert!(
            samples
                .iter()
                .all(|(name, _, _)| !name.starts_with("macgpu_app_"))
        );
    }

    #[test]
    fn empty_and_minimal_reports_render() {
        let exporter = Exporter::new(options());
        let samples = parse_exposition(&exporter.render_empty());
        assert_eq!(
            value(&samples, "macgpu_exporter_samples_total", None),
            Some(0.0)
        );

        let mut exporter = Exporter::new(options());
        let samples = parse_exposition(&exporter.update(&Report::default()));
        assert_eq!(value(&samples, "macgpu_gpu_power_watts", None), None);
        assert_eq!(
            value(&samples, "macgpu_gpu_energy_joules_total", None),
            None
        );
        assert_eq!(
            value(&samples, "macgpu_gpu_attributed_busy_ratio", None),
            Some(0.0)
        );
    }

    #[test]
    fn parses_listen_addresses() {
        assert_eq!(
            parse_listen("9999").unwrap(),
            "127.0.0.1:9999".parse().unwrap()
        );
        assert_eq!(
            parse_listen(":9999").unwrap(),
            "127.0.0.1:9999".parse().unwrap()
        );
        assert_eq!(
            parse_listen("0.0.0.0:80").unwrap(),
            "0.0.0.0:80".parse().unwrap()
        );
        assert!(parse_listen("nonsense").is_err());
        let error = bind_error(
            "127.0.0.1:10060".parse().unwrap(),
            &io::Error::from(io::ErrorKind::AddrInUse),
        );
        assert!(
            error.to_string().contains("--listen 127.0.0.1:PORT"),
            "{error}"
        );
    }
}
