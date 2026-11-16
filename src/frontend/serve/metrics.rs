use crate::model::{Report, SourceState};
use crate::view::Filter;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

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
pub(super) const APP_RETENTION_MS: u64 = 3_600_000;

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
        emit(
            out,
            "",
            &[
                gauge(
                    "macgpu_exporter_window_seconds",
                    "Length of the latest sampling window.",
                    Some(report.window_seconds),
                ),
                gauge(
                    "macgpu_exporter_last_sample_timestamp_seconds",
                    "End of the latest sampling window, in seconds since the Unix epoch.",
                    Some(report.timestamp_ms as f64 / 1000.0),
                ),
                gauge(
                    "macgpu_exporter_sample_duration_seconds",
                    "Wall time to take the latest sample.",
                    Some(report.overhead.sample_wall_ms / 1000.0),
                ),
                gauge(
                    "macgpu_exporter_sample_cpu_seconds",
                    "Sampler CPU time for the latest sample.",
                    Some(report.overhead.sample_cpu_ms / 1000.0),
                ),
                gauge(
                    "macgpu_exporter_cpu_ratio",
                    "macgpu's whole-process CPU use over the latest window, in cores.",
                    Some(report.overhead.process_cpu_pct / 100.0),
                ),
            ],
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
                        detail: process.path.clone(),
                        ..crate::view::Row::blank(
                            crate::view::RowKey::Unattributed,
                            process.name.clone(),
                        )
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
            emit(
                out,
                &set,
                &[
                    counter(
                        "macgpu_process_gpu_seconds_total",
                        "GPU time the driver apportioned to the process (AppUsage) since the exporter first saw it. Busy time is split among work running at the same moment, so all processes together grow by at most about 1 per second per GPU.",
                        Some(totals.gpu_s),
                    ),
                    gauge(
                        "macgpu_process_gpu_busy_ratio",
                        "Apportioned GPU time per wall-clock second over the latest window. All processes sum to at most about 1 when the GPU is saturated.",
                        Some(process.busy_ms_per_s / 1000.0),
                    ),
                    gauge(
                        "macgpu_process_gpu_share_ratio",
                        "Share of all processes' apportioned GPU time in the latest window.",
                        Some(process.share_pct / 100.0),
                    ),
                    gauge(
                        "macgpu_process_gpu_device_ratio",
                        "The process's slice of the device's active time (share times device active ratio).",
                        process.device_pct.map(|device| device / 100.0),
                    ),
                    gauge(
                        "macgpu_process_power_watts",
                        "Estimated GPU power: the app's measured GPU power apportioned by this process's share of the app's GPU time.",
                        process.est_power_w,
                    ),
                    counter(
                        "macgpu_process_energy_joules_total",
                        "Estimated GPU energy since the exporter first saw the process.",
                        totals.energy_j,
                    ),
                    gauge(
                        "macgpu_process_graphics_memory_bytes",
                        "Graphics memory footprint (TASK_VM_INFO ledger_tag_graphics_footprint); absent for other users' processes without root.",
                        process.graphics_memory_bytes.map(|bytes| bytes as f64),
                    ),
                    gauge(
                        "macgpu_process_queues",
                        "Metal command queues (AppUsage entries) across the process's GPU clients.",
                        Some(process.queues as f64),
                    ),
                    gauge(
                        "macgpu_process_last_submit_age_seconds",
                        "Time since the process last submitted GPU work.",
                        process.last_submit_age_ms.map(|age| age as f64 / 1000.0),
                    ),
                ],
            );
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
                        detail: app.bundle_path.clone(),
                        ..crate::view::Row::blank(
                            crate::view::RowKey::Unattributed,
                            app.name.clone(),
                        )
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
            emit(
                out,
                &set,
                &[
                    counter(
                        "macgpu_app_gpu_seconds_total",
                        "App GPU time (its processes' apportioned time plus known unattributed time of exited helpers) since the exporter first saw the app.",
                        Some(totals.gpu_s),
                    ),
                    counter(
                        "macgpu_app_gpu_in_flight_seconds_total",
                        "Wall time the app's coalition had GPU work in flight since the exporter first saw it. Work overlapping another app's counts fully for each, so apps together can exceed 1 per second.",
                        totals.in_flight_s,
                    ),
                    gauge(
                        "macgpu_app_gpu_in_flight_ratio",
                        "Share of the latest window the app had GPU work in flight (overlap with other apps counts fully).",
                        app.in_flight_ms_per_s.map(|ratio| ratio / 1000.0),
                    ),
                    counter(
                        "macgpu_app_gpu_energy_joules_total",
                        "Measured GPU energy of the app's resource coalition since the coalition was created (kernel gpu_energy_nj counter). A relaunched app gets a new coalition label.",
                        app.lifetime_energy_j,
                    ),
                    gauge(
                        "macgpu_app_gpu_busy_ratio",
                        "App GPU time (its processes' apportioned time plus known unattributed time) per wall-clock second over the latest window.",
                        Some(app.busy_ms_per_s / 1000.0),
                    ),
                    gauge(
                        "macgpu_app_gpu_share_ratio",
                        "Share of all apps' GPU time in the latest window.",
                        Some(app.share_pct / 100.0),
                    ),
                    gauge(
                        "macgpu_app_gpu_device_ratio",
                        "The app's slice of the device's active time.",
                        app.device_pct.map(|device| device / 100.0),
                    ),
                    gauge(
                        "macgpu_app_unattributed_gpu_busy_ratio",
                        "Coalition GPU time per second of tasks without a live GPU client (exited helpers). Absent when it cannot be separated from overlapping GPU work.",
                        app.unattributed_gpu_ns
                            .map(|ns| ns as f64 / 1e9 / report.window_seconds.max(1e-9)),
                    ),
                    gauge(
                        "macgpu_app_power_watts",
                        "Measured GPU power of the app's coalition over the latest window.",
                        app.power_w,
                    ),
                    gauge(
                        "macgpu_app_billed_to_me_watts",
                        "GPU power other coalitions (such as WindowServer) spent on this app's behalf.",
                        app.billed_to_me_w,
                    ),
                    gauge(
                        "macgpu_app_billed_to_others_watts",
                        "GPU power this app spent on behalf of others.",
                        app.billed_to_others_w,
                    ),
                    gauge(
                        "macgpu_app_ane_power_watts",
                        "Neural Engine power of the app's coalition.",
                        app.ane_power_w,
                    ),
                    gauge(
                        "macgpu_app_ane_busy_ratio",
                        "Neural Engine time per wall-clock second.",
                        app.ane_ms_per_s.map(|ane| ane / 1000.0),
                    ),
                    gauge(
                        "macgpu_app_graphics_memory_bytes",
                        "Sum of the known graphics memory of the app's processes.",
                        app.graphics_memory_bytes.map(|bytes| bytes as f64),
                    ),
                    gauge(
                        "macgpu_app_processes",
                        "Live processes in the app's coalition.",
                        Some(app.members as f64),
                    ),
                ],
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
    emit(
        out,
        "",
        &[
            gauge(
                "macgpu_gpu_active_ratio",
                "Share of the window the GPU spent in an active P-state (IOReport GPUPH).",
                ratio(device.active_pct),
            ),
            gauge(
                "macgpu_gpu_freq_normalized_ratio",
                "Active ratio weighted by clock: active * average clock / max clock.",
                ratio(device.freq_normalized_pct),
            ),
            gauge(
                "macgpu_gpu_frequency_hertz",
                "Residency-weighted average GPU clock while active.",
                device.avg_mhz.map(|mhz| mhz * 1e6),
            ),
            gauge(
                "macgpu_gpu_max_frequency_hertz",
                "Highest GPU clock in the frequency table.",
                device.max_mhz.map(|mhz| mhz * 1e6),
            ),
            gauge(
                "macgpu_gpu_power_watts",
                "Average GPU power over the window (IOReport GPU Energy).",
                device.power_w,
            ),
            gauge(
                "macgpu_gpu_limited_ratio",
                "Share of the window the GPU spent under a power or thermal limiter state (PWRCTRL limiting states or a CLTM cap); clocks are not necessarily reduced.",
                ratio(device.limited_pct()),
            ),
            gauge(
                "macgpu_gpu_thermal_limited_ratio",
                "Share of the window under thermal (CLTM) limiting.",
                ratio(device.thermal_limited_pct),
            ),
            gauge(
                "macgpu_gpu_memory_in_use_bytes",
                "GPU memory in use (PerformanceStatistics In use system memory).",
                device.memory.in_use_bytes.map(|b| b as f64),
            ),
            gauge(
                "macgpu_gpu_memory_allocated_bytes",
                "GPU memory allocated (PerformanceStatistics Alloc system memory).",
                device.memory.allocated_bytes.map(|b| b as f64),
            ),
            gauge(
                "macgpu_gpu_wired_limit_bytes",
                "Effective GPU wired-memory limit (iogpu.wired_limit_mb or the default).",
                device.memory.wired_limit_bytes.map(|b| b as f64),
            ),
            gauge(
                "macgpu_system_memory_bytes",
                "Physical memory.",
                device.memory.total_physical_bytes.map(|b| b as f64),
            ),
        ],
    );
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
            "Share of powered GPU time (every state but IDLE_OFF) in each power-controller state \
             (IOReport PWRCTRL), not of the window; a state does not always lower the clock.",
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
    emit(
        out,
        "",
        &[
            gauge(
                "macgpu_gpu_attributed_busy_ratio",
                "Apportioned GPU time of all processes per wall-clock second over the latest window (at most about 1 per GPU).",
                Some(totals.attributed_ms_per_s / 1000.0),
            ),
            counter(
                "macgpu_gpu_attributed_seconds_total",
                "Apportioned GPU time of all processes since the exporter started.",
                Some(exporter.attributed_gpu_s),
            ),
            gauge(
                "macgpu_gpu_unattributed_busy_ratio",
                "Coalition GPU time per second that no live client accounts for (exited clients); a lower bound when macgpu_gpu_unattributed_partial is 1.",
                report
                    .unattributed
                    .as_ref()
                    .map(|unattributed| unattributed.busy_ms_per_s / 1000.0),
            ),
            gauge(
                "macgpu_gpu_unattributed_partial",
                "1 when some apps' unattributed time could not be separated from overlapping GPU work, so the unattributed series are lower bounds.",
                report
                    .unattributed
                    .as_ref()
                    .map(|unattributed| f64::from(u8::from(unattributed.partial))),
            ),
            counter(
                "macgpu_gpu_unattributed_seconds_total",
                "Coalition GPU time no live client accounted for, since the exporter started (a lower bound; see macgpu_gpu_unattributed_partial).",
                (report.unattributed.is_some() || exporter.unattributed_gpu_s > 0.0)
                    .then_some(exporter.unattributed_gpu_s),
            ),
            gauge(
                "macgpu_gpu_clients",
                "GPU client connections.",
                Some(totals.gpu_clients as f64),
            ),
            gauge(
                "macgpu_gpu_processes",
                "Processes with GPU clients.",
                Some(totals.gpu_processes as f64),
            ),
            gauge(
                "macgpu_gpu_active_processes",
                "Processes that used the GPU in the latest window.",
                Some(totals.active_processes as f64),
            ),
            gauge(
                "macgpu_gpu_active_apps",
                "Apps that used the GPU in the latest window.",
                Some(totals.active_apps as f64),
            ),
        ],
    );
}

/// One row of a metric table: a sample emitted only when the value is present.
struct Sample {
    name: &'static str,
    help: &'static str,
    kind: Kind,
    value: Option<f64>,
}

fn gauge(name: &'static str, help: &'static str, value: Option<f64>) -> Sample {
    Sample {
        name,
        help,
        kind: Kind::Gauge,
        value,
    }
}

fn counter(name: &'static str, help: &'static str, value: Option<f64>) -> Sample {
    Sample {
        name,
        help,
        kind: Kind::Counter,
        value,
    }
}

fn emit(out: &mut Families, set: &str, samples: &[Sample]) {
    for sample in samples {
        let Some(value) = sample.value else {
            continue;
        };
        let labels = set.to_string();
        match sample.kind {
            Kind::Gauge => out.gauge(sample.name, sample.help, labels, value),
            Kind::Counter => out.counter(sample.name, sample.help, labels, value),
        }
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
