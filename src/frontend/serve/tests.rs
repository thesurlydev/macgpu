use super::http::{
    QUEUE, Responders, WORKERS, bind_error, dispatch, parse_listen, select_response,
};
use super::metrics::{APP_RETENTION_MS, ExportOptions, Exporter};
use super::state::Shared;
use crate::model::{
    AppView, DeviceMemory, DeviceView, FirmwareUtilization, PStateShare, ProcessView, Report,
    SourceState, SourceStatus, StateShare, SystemView, Temperature, Totals, Unattributed,
};
use crate::view::Filter;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

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
/// Byte-identical output for a fixed report; guards the table-driven
/// refactor of the three family builders.
const FULL_REPORT_SNAPSHOT: &str = r#"# HELP macgpu_exporter_build_info macgpu version; always 1.
# TYPE macgpu_exporter_build_info gauge
macgpu_exporter_build_info{version="0.1.0"} 1
# HELP macgpu_exporter_samples_total Samples taken since the exporter started.
# TYPE macgpu_exporter_samples_total counter
macgpu_exporter_samples_total 1
# HELP macgpu_exporter_window_seconds Length of the latest sampling window.
# TYPE macgpu_exporter_window_seconds gauge
macgpu_exporter_window_seconds 2
# HELP macgpu_exporter_last_sample_timestamp_seconds End of the latest sampling window, in seconds since the Unix epoch.
# TYPE macgpu_exporter_last_sample_timestamp_seconds gauge
macgpu_exporter_last_sample_timestamp_seconds 1798300801
# HELP macgpu_exporter_sample_duration_seconds Wall time to take the latest sample.
# TYPE macgpu_exporter_sample_duration_seconds gauge
macgpu_exporter_sample_duration_seconds 0
# HELP macgpu_exporter_sample_cpu_seconds Sampler CPU time for the latest sample.
# TYPE macgpu_exporter_sample_cpu_seconds gauge
macgpu_exporter_sample_cpu_seconds 0
# HELP macgpu_exporter_cpu_ratio macgpu's whole-process CPU use over the latest window, in cores.
# TYPE macgpu_exporter_cpu_ratio gauge
macgpu_exporter_cpu_ratio 0
# HELP macgpu_source_up 1 when a data source produced data for the latest window; state gives the detail.
# TYPE macgpu_source_up gauge
macgpu_source_up{source="registry",state="ok"} 1
macgpu_source_up{source="temperature",state="disabled"} 0
# HELP macgpu_gpu_info GPU identity; always 1.
# TYPE macgpu_gpu_info gauge
macgpu_gpu_info{model="Apple M4 Max",cores="40"} 1
# HELP macgpu_gpu_active_ratio Share of the window the GPU spent in an active P-state (IOReport GPUPH).
# TYPE macgpu_gpu_active_ratio gauge
macgpu_gpu_active_ratio 0.6
# HELP macgpu_gpu_freq_normalized_ratio Active ratio weighted by clock: active * average clock / max clock.
# TYPE macgpu_gpu_freq_normalized_ratio gauge
macgpu_gpu_freq_normalized_ratio 0.494
# HELP macgpu_gpu_frequency_hertz Residency-weighted average GPU clock while active.
# TYPE macgpu_gpu_frequency_hertz gauge
macgpu_gpu_frequency_hertz 1300000000
# HELP macgpu_gpu_max_frequency_hertz Highest GPU clock in the frequency table.
# TYPE macgpu_gpu_max_frequency_hertz gauge
macgpu_gpu_max_frequency_hertz 1578000000
# HELP macgpu_gpu_power_watts Average GPU power over the window (IOReport GPU Energy).
# TYPE macgpu_gpu_power_watts gauge
macgpu_gpu_power_watts 20
# HELP macgpu_gpu_limited_ratio Share of the window the GPU spent under a power or thermal limiter state (PWRCTRL limiting states or a CLTM cap); clocks are not necessarily reduced.
# TYPE macgpu_gpu_limited_ratio gauge
macgpu_gpu_limited_ratio 0.15
# HELP macgpu_gpu_thermal_limited_ratio Share of the window under thermal (CLTM) limiting.
# TYPE macgpu_gpu_thermal_limited_ratio gauge
macgpu_gpu_thermal_limited_ratio 0.05
# HELP macgpu_gpu_memory_in_use_bytes GPU memory in use (PerformanceStatistics In use system memory).
# TYPE macgpu_gpu_memory_in_use_bytes gauge
macgpu_gpu_memory_in_use_bytes 4294967296
# HELP macgpu_gpu_memory_allocated_bytes GPU memory allocated (PerformanceStatistics Alloc system memory).
# TYPE macgpu_gpu_memory_allocated_bytes gauge
macgpu_gpu_memory_allocated_bytes 6442450944
# HELP macgpu_gpu_wired_limit_bytes Effective GPU wired-memory limit (iogpu.wired_limit_mb or the default).
# TYPE macgpu_gpu_wired_limit_bytes gauge
macgpu_gpu_wired_limit_bytes 103079215104
# HELP macgpu_system_memory_bytes Physical memory.
# TYPE macgpu_system_memory_bytes gauge
macgpu_system_memory_bytes 137438953472
# HELP macgpu_gpu_energy_joules_total GPU energy since the exporter started (sum of window power times window length).
# TYPE macgpu_gpu_energy_joules_total counter
macgpu_gpu_energy_joules_total 40
# HELP macgpu_gpu_temperature_celsius GPU die temperature from the SMC: mean or hottest sensor.
# TYPE macgpu_gpu_temperature_celsius gauge
macgpu_gpu_temperature_celsius{stat="average"} 70
macgpu_gpu_temperature_celsius{stat="max"} 80.5
# HELP macgpu_gpu_firmware_utilization_ratio Firmware PerformanceStatistics utilization, averaged over readings in the window.
# TYPE macgpu_gpu_firmware_utilization_ratio gauge
macgpu_gpu_firmware_utilization_ratio{engine="device"} 0.5
macgpu_gpu_firmware_utilization_ratio{engine="renderer"} 0.4
macgpu_gpu_firmware_utilization_ratio{engine="tiler"} 0.1
# HELP macgpu_gpu_pstate_residency_ratio Share of the window spent in each GPU P-state, including OFF.
# TYPE macgpu_gpu_pstate_residency_ratio gauge
macgpu_gpu_pstate_residency_ratio{state="OFF",mhz=""} 0.4
macgpu_gpu_pstate_residency_ratio{state="P15",mhz="1578"} 0.6
# HELP macgpu_gpu_limiter_residency_ratio Share of powered GPU time (every state but IDLE_OFF) in each power-controller state (IOReport PWRCTRL), not of the window; a state does not always lower the clock.
# TYPE macgpu_gpu_limiter_residency_ratio gauge
macgpu_gpu_limiter_residency_ratio{state="PWR_ZONE"} 0.25
# HELP macgpu_gpu_counter_events_total Increments of IOReport throttle, thermal, and issue counters since the exporter started.
# TYPE macgpu_gpu_counter_events_total counter
macgpu_gpu_counter_events_total{counter="GPU_THROTTLE"} 3
# HELP macgpu_gpu_recoveries_total GPU hang recoveries since boot.
# TYPE macgpu_gpu_recoveries_total counter
macgpu_gpu_recoveries_total 2
# HELP macgpu_thermal_pressure System thermal pressure level; 1 for the current level.
# TYPE macgpu_thermal_pressure gauge
macgpu_thermal_pressure{level="nominal"} 1
macgpu_thermal_pressure{level="moderate"} 0
macgpu_thermal_pressure{level="heavy"} 0
macgpu_thermal_pressure{level="trapping"} 0
macgpu_thermal_pressure{level="sleeping"} 0
# HELP macgpu_memory_pressure System memory pressure level; 1 for the current level.
# TYPE macgpu_memory_pressure gauge
macgpu_memory_pressure{level="normal"} 0
macgpu_memory_pressure{level="warning"} 1
macgpu_memory_pressure{level="critical"} 0
# HELP macgpu_gpu_attributed_busy_ratio Apportioned GPU time of all processes per wall-clock second over the latest window (at most about 1 per GPU).
# TYPE macgpu_gpu_attributed_busy_ratio gauge
macgpu_gpu_attributed_busy_ratio 0.75
# HELP macgpu_gpu_attributed_seconds_total Apportioned GPU time of all processes since the exporter started.
# TYPE macgpu_gpu_attributed_seconds_total counter
macgpu_gpu_attributed_seconds_total 1.5
# HELP macgpu_gpu_unattributed_busy_ratio Coalition GPU time per second that no live client accounts for (exited clients); a lower bound when macgpu_gpu_unattributed_partial is 1.
# TYPE macgpu_gpu_unattributed_busy_ratio gauge
macgpu_gpu_unattributed_busy_ratio 0.05
# HELP macgpu_gpu_unattributed_partial 1 when some apps' unattributed time could not be separated from overlapping GPU work, so the unattributed series are lower bounds.
# TYPE macgpu_gpu_unattributed_partial gauge
macgpu_gpu_unattributed_partial 1
# HELP macgpu_gpu_unattributed_seconds_total Coalition GPU time no live client accounted for, since the exporter started (a lower bound; see macgpu_gpu_unattributed_partial).
# TYPE macgpu_gpu_unattributed_seconds_total counter
macgpu_gpu_unattributed_seconds_total 0.1
# HELP macgpu_gpu_clients GPU client connections.
# TYPE macgpu_gpu_clients gauge
macgpu_gpu_clients 9
# HELP macgpu_gpu_processes Processes with GPU clients.
# TYPE macgpu_gpu_processes gauge
macgpu_gpu_processes 3
# HELP macgpu_gpu_active_processes Processes that used the GPU in the latest window.
# TYPE macgpu_gpu_active_processes gauge
macgpu_gpu_active_processes 2
# HELP macgpu_gpu_active_apps Apps that used the GPU in the latest window.
# TYPE macgpu_gpu_active_apps gauge
macgpu_gpu_active_apps 1
# HELP macgpu_exporter_omitted_processes GPU processes left out of the per-process series by --top.
# TYPE macgpu_exporter_omitted_processes gauge
macgpu_exporter_omitted_processes 0
# HELP macgpu_process_gpu_seconds_total GPU time the driver apportioned to the process (AppUsage) since the exporter first saw it. Busy time is split among work running at the same moment, so all processes together grow by at most about 1 per second per GPU.
# TYPE macgpu_process_gpu_seconds_total counter
macgpu_process_gpu_seconds_total{pid="900",process="Google \"Chrome\" Helper",app="Google Chrome",coalition="77"} 1
macgpu_process_gpu_seconds_total{pid="431",process="WindowServer",app="",coalition="10"} 0.5
# HELP macgpu_process_gpu_busy_ratio Apportioned GPU time per wall-clock second over the latest window. All processes sum to at most about 1 when the GPU is saturated.
# TYPE macgpu_process_gpu_busy_ratio gauge
macgpu_process_gpu_busy_ratio{pid="900",process="Google \"Chrome\" Helper",app="Google Chrome",coalition="77"} 0.5
macgpu_process_gpu_busy_ratio{pid="431",process="WindowServer",app="",coalition="10"} 0.25
# HELP macgpu_process_gpu_share_ratio Share of all processes' apportioned GPU time in the latest window.
# TYPE macgpu_process_gpu_share_ratio gauge
macgpu_process_gpu_share_ratio{pid="900",process="Google \"Chrome\" Helper",app="Google Chrome",coalition="77"} 0.667
macgpu_process_gpu_share_ratio{pid="431",process="WindowServer",app="",coalition="10"} 0.33299999999999996
# HELP macgpu_process_gpu_device_ratio The process's slice of the device's active time (share times device active ratio).
# TYPE macgpu_process_gpu_device_ratio gauge
macgpu_process_gpu_device_ratio{pid="900",process="Google \"Chrome\" Helper",app="Google Chrome",coalition="77"} 0.4
# HELP macgpu_process_power_watts Estimated GPU power: the app's measured GPU power apportioned by this process's share of the app's GPU time.
# TYPE macgpu_process_power_watts gauge
macgpu_process_power_watts{pid="900",process="Google \"Chrome\" Helper",app="Google Chrome",coalition="77"} 12
# HELP macgpu_process_energy_joules_total Estimated GPU energy since the exporter first saw the process.
# TYPE macgpu_process_energy_joules_total counter
macgpu_process_energy_joules_total{pid="900",process="Google \"Chrome\" Helper",app="Google Chrome",coalition="77"} 24
# HELP macgpu_process_graphics_memory_bytes Graphics memory footprint (TASK_VM_INFO ledger_tag_graphics_footprint); absent for other users' processes without root.
# TYPE macgpu_process_graphics_memory_bytes gauge
macgpu_process_graphics_memory_bytes{pid="900",process="Google \"Chrome\" Helper",app="Google Chrome",coalition="77"} 1073741824
# HELP macgpu_process_queues Metal command queues (AppUsage entries) across the process's GPU clients.
# TYPE macgpu_process_queues gauge
macgpu_process_queues{pid="900",process="Google \"Chrome\" Helper",app="Google Chrome",coalition="77"} 2
macgpu_process_queues{pid="431",process="WindowServer",app="",coalition="10"} 3
# HELP macgpu_process_last_submit_age_seconds Time since the process last submitted GPU work.
# TYPE macgpu_process_last_submit_age_seconds gauge
macgpu_process_last_submit_age_seconds{pid="900",process="Google \"Chrome\" Helper",app="Google Chrome",coalition="77"} 0.02
# HELP macgpu_exporter_omitted_apps Apps left out of the per-app series by --top.
# TYPE macgpu_exporter_omitted_apps gauge
macgpu_exporter_omitted_apps 0
# HELP macgpu_app_gpu_seconds_total App GPU time (its processes' apportioned time plus known unattributed time of exited helpers) since the exporter first saw the app.
# TYPE macgpu_app_gpu_seconds_total counter
macgpu_app_gpu_seconds_total{app="Google Chrome",coalition="77"} 1
# HELP macgpu_app_gpu_in_flight_seconds_total Wall time the app's coalition had GPU work in flight since the exporter first saw it. Work overlapping another app's counts fully for each, so apps together can exceed 1 per second.
# TYPE macgpu_app_gpu_in_flight_seconds_total counter
macgpu_app_gpu_in_flight_seconds_total{app="Google Chrome",coalition="77"} 1.6
# HELP macgpu_app_gpu_in_flight_ratio Share of the latest window the app had GPU work in flight (overlap with other apps counts fully).
# TYPE macgpu_app_gpu_in_flight_ratio gauge
macgpu_app_gpu_in_flight_ratio{app="Google Chrome",coalition="77"} 0.8
# HELP macgpu_app_gpu_energy_joules_total Measured GPU energy of the app's resource coalition since the coalition was created (kernel gpu_energy_nj counter). A relaunched app gets a new coalition label.
# TYPE macgpu_app_gpu_energy_joules_total counter
macgpu_app_gpu_energy_joules_total{app="Google Chrome",coalition="77"} 1500
# HELP macgpu_app_gpu_busy_ratio App GPU time (its processes' apportioned time plus known unattributed time) per wall-clock second over the latest window.
# TYPE macgpu_app_gpu_busy_ratio gauge
macgpu_app_gpu_busy_ratio{app="Google Chrome",coalition="77"} 0.5
# HELP macgpu_app_gpu_share_ratio Share of all apps' GPU time in the latest window.
# TYPE macgpu_app_gpu_share_ratio gauge
macgpu_app_gpu_share_ratio{app="Google Chrome",coalition="77"} 1
# HELP macgpu_app_gpu_device_ratio The app's slice of the device's active time.
# TYPE macgpu_app_gpu_device_ratio gauge
macgpu_app_gpu_device_ratio{app="Google Chrome",coalition="77"} 0.4
# HELP macgpu_app_unattributed_gpu_busy_ratio Coalition GPU time per second of tasks without a live GPU client (exited helpers). Absent when it cannot be separated from overlapping GPU work.
# TYPE macgpu_app_unattributed_gpu_busy_ratio gauge
macgpu_app_unattributed_gpu_busy_ratio{app="Google Chrome",coalition="77"} 0.05
# HELP macgpu_app_power_watts Measured GPU power of the app's coalition over the latest window.
# TYPE macgpu_app_power_watts gauge
macgpu_app_power_watts{app="Google Chrome",coalition="77"} 12
# HELP macgpu_app_billed_to_me_watts GPU power other coalitions (such as WindowServer) spent on this app's behalf.
# TYPE macgpu_app_billed_to_me_watts gauge
macgpu_app_billed_to_me_watts{app="Google Chrome",coalition="77"} 0.5
# HELP macgpu_app_billed_to_others_watts GPU power this app spent on behalf of others.
# TYPE macgpu_app_billed_to_others_watts gauge
macgpu_app_billed_to_others_watts{app="Google Chrome",coalition="77"} 0
# HELP macgpu_app_ane_power_watts Neural Engine power of the app's coalition.
# TYPE macgpu_app_ane_power_watts gauge
macgpu_app_ane_power_watts{app="Google Chrome",coalition="77"} 0.1
# HELP macgpu_app_ane_busy_ratio Neural Engine time per wall-clock second.
# TYPE macgpu_app_ane_busy_ratio gauge
macgpu_app_ane_busy_ratio{app="Google Chrome",coalition="77"} 0.003
# HELP macgpu_app_graphics_memory_bytes Sum of the known graphics memory of the app's processes.
# TYPE macgpu_app_graphics_memory_bytes gauge
macgpu_app_graphics_memory_bytes{app="Google Chrome",coalition="77"} 1073741824
# HELP macgpu_app_processes Live processes in the app's coalition.
# TYPE macgpu_app_processes gauge
macgpu_app_processes{app="Google Chrome",coalition="77"} 12
"#;

#[test]
fn full_report_snapshot_is_byte_identical() {
    let mut exporter = Exporter::new(options());
    assert_eq!(exporter.update(&full_report()), FULL_REPORT_SNAPSHOT);
}

#[test]
fn json_failure_and_missing_sample_are_not_empty_successes() {
    let mut shared = Shared::new("metrics\n".into());
    let stale_after = Duration::from_secs(5);
    let (status, _, body) = select_response(
        &shared,
        "/report.json",
        &tiny_http::Method::Get,
        stale_after,
    );
    assert_eq!(status, 503);
    assert_eq!(&*body, "no sample yet\n");
    shared.last_sample = Some(Instant::now());
    let (status, _, body) = select_response(
        &shared,
        "/report.json",
        &tiny_http::Method::Get,
        stale_after,
    );
    assert_eq!(status, 503);
    assert!(body.contains("JSON encoding failed"));
    let (status, _, body) =
        select_response(&shared, "/healthz", &tiny_http::Method::Get, stale_after);
    assert_eq!((status, &*body), (200, "ok\n"));
    let (status, _, body) =
        select_response(&shared, "/healthz", &tiny_http::Method::Get, Duration::ZERO);
    assert_eq!(status, 503);
    assert!(body.contains("stale:"));
    let (status, _, _) =
        select_response(&shared, "/metrics", &tiny_http::Method::Post, stale_after);
    assert_eq!(status, 405);
}

#[test]
fn response_pool_bounds_stalled_clients() {
    let responders = Responders::new();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let executed = Arc::new(AtomicUsize::new(0));
    for index in 0..WORKERS + QUEUE {
        let release_rx = Arc::clone(&release_rx);
        let started_tx = started_tx.clone();
        let executed = Arc::clone(&executed);
        assert!(responders.submit(move || {
            let _ = started_tx.send(());
            let _ = release_rx
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10));
            executed.fetch_add(1, Ordering::SeqCst);
        }));
        if index < WORKERS {
            // Every worker takes a job before the queue starts to fill.
            started_rx
                .recv_timeout(Duration::from_secs(3))
                .expect("worker starts its job");
        }
    }
    assert!(
        !responders.submit(|| {}),
        "a full queue refuses more work instead of growing"
    );

    for _ in 0..WORKERS + QUEUE {
        release_tx.send(()).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while executed.load(Ordering::SeqCst) < WORKERS + QUEUE && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(executed.load(Ordering::SeqCst), WORKERS + QUEUE);
}

#[test]
fn stalled_client_does_not_block_another_scrape_or_shutdown() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::thread;

    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let address = server.server_addr().to_ip().unwrap();
    let mut initial = Shared::new("x".repeat(64 << 20));
    initial.last_sample = Some(Instant::now());
    let shared = Arc::new(RwLock::new(initial));
    let responders = Responders::new();
    let stop = Arc::new(AtomicBool::new(false));
    let (exited_tx, exited_rx) = mpsc::channel::<()>();
    let accept = {
        let stop = Arc::clone(&stop);
        let shared = Arc::clone(&shared);
        thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                match server.recv_timeout(Duration::from_millis(50)) {
                    Ok(Some(request)) => {
                        dispatch(
                            &responders,
                            request,
                            Arc::clone(&shared),
                            Duration::from_secs(5),
                        );
                    }
                    Ok(None) => {}
                    Err(_) => break,
                }
            }
            exited_tx.send(()).unwrap();
        })
    };

    let mut stalled = TcpStream::connect(address).unwrap();
    stalled
        .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .unwrap();
    thread::sleep(Duration::from_millis(100));

    let mut scrape = TcpStream::connect(address).unwrap();
    scrape
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    scrape
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut answer = String::new();
    scrape.read_to_string(&mut answer).unwrap();
    assert!(answer.contains("200 OK"), "{answer}");
    assert!(answer.contains("ok"), "{answer}");

    stop.store(true, Ordering::SeqCst);
    exited_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("accept loop must stop while a client is stalled");
    drop(stalled);
    let _ = accept.join();
}

#[test]
fn full_queue_answers_503_instead_of_dropping_the_request() {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let address = server.server_addr().to_ip().unwrap();
    let shared = Arc::new(RwLock::new(Shared::new(String::new())));
    let responders = Responders::new();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let (started_tx, started_rx) = mpsc::channel::<()>();
    for index in 0..WORKERS + QUEUE {
        let release_rx = Arc::clone(&release_rx);
        let started_tx = started_tx.clone();
        assert!(responders.submit(move || {
            let _ = started_tx.send(());
            let _ = release_rx
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10));
        }));
        if index < WORKERS {
            started_rx
                .recv_timeout(Duration::from_secs(3))
                .expect("worker starts its job");
        }
    }

    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    client
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .unwrap();
    let request = server
        .recv_timeout(Duration::from_secs(3))
        .unwrap()
        .expect("request arrives");
    dispatch(&responders, request, shared, Duration::from_secs(5));
    let mut answer = String::new();
    client.read_to_string(&mut answer).unwrap();
    assert!(answer.contains("503"), "{answer}");
    for _ in 0..WORKERS + QUEUE {
        let _ = release_tx.send(());
    }
}
