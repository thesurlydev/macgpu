//! A deterministic, synthetic [`ReportFeed`] with every report field filled
//! in, for developing and testing the interactive view without real GPU load.
//!
//! The scenario on an "Apple M4 Max": WindowServer composites steadily, Chrome's
//! GPU helper and Safari's WebKit GPU process draw pages, an Ollama runner
//! bursts on two queues, an MLX script started from a Ghostty terminal starts
//! at 20 s, a short-lived Metal compiler exits at 12 samples and leaves
//! unattributed time behind, a GPU recovery happens at sample 45, and an alert
//! fires at 30 and clears at 40. Like the driver, it apportions busy time so
//! the processes sum to at most about 1000 ms/s.

use crate::analyze::RECENT_SECONDS;
use crate::engine::{FeedEvent, ReportFeed};
use crate::model::{
    Activity, AppView, ClientView, DeviceMemory, DeviceView, Event, EventKind, FirmwareUtilization,
    Overhead, PStateShare, ProcessView, QueueView, Report, SCHEMA_VERSION, SourceState,
    SourceStatus, StateShare, SystemView, Temperature, Totals, Unattributed, source,
};
use std::collections::BTreeMap;
use std::time::Duration;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
/// 2025-10-25 14:00:00 UTC.
const BASE_MS: u64 = 1_761_400_800_000;
const MAX_MHZ: f64 = 1578.0;
const PSTATE_MHZ: [f64; 15] = [
    338.0, 618.0, 796.0, 924.0, 952.0, 1056.0, 1062.0, 1182.0, 1182.0, 1312.0, 1242.0, 1380.0,
    1326.0, 1470.0, 1578.0,
];

/// A synthetic feed that returns a report on every call (for tests).
pub struct SyntheticFeed {
    step: u64,
    interval: Duration,
    include_idle: bool,
    live: bool,
    limit: Option<u64>,
}

impl SyntheticFeed {
    /// Returns a report on every call.
    pub fn unpaced() -> Self {
        Self {
            step: 0,
            interval: Duration::from_secs(1),
            include_idle: false,
            live: true,
            limit: None,
        }
    }

    /// Behaves like a recording: fixed interval, ends after `reports`.
    pub fn replay(reports: u64) -> Self {
        Self {
            live: false,
            limit: Some(reports),
            ..Self::unpaced()
        }
    }
}

impl ReportFeed for SyntheticFeed {
    fn next_report(&mut self, _timeout: Duration) -> FeedEvent {
        if self.limit.is_some_and(|limit| self.step >= limit) {
            return FeedEvent::End;
        }
        self.step += 1;
        FeedEvent::Report(Box::new(report(
            self.step,
            self.interval,
            self.include_idle,
        )))
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn set_interval(&mut self, interval: Duration) -> bool {
        if !self.live {
            return false;
        }
        self.interval = interval;
        true
    }

    fn is_live(&self) -> bool {
        self.live
    }

    fn set_include_idle(&mut self, include_idle: bool) {
        self.include_idle = include_idle;
    }
}

struct Spec {
    pid: i32,
    ppid: i32,
    uid: u32,
    name: &'static str,
    path: &'static str,
    coalition: u64,
    start_s: u64,
    memory: Option<u64>,
    /// `(registry_id, [(api, fraction of the process's time)])`.
    clients: &'static [(u64, &'static [(&'static str, f64)])],
    busy: fn(f64) -> f64,
    /// Steps during which the process exists, if limited.
    alive: Option<(u64, u64)>,
}

const SPECS: &[Spec] = &[
    Spec {
        pid: 412,
        ppid: 1,
        uid: 88,
        name: "WindowServer",
        path: "/System/Library/PrivateFrameworks/SkyLight.framework/Versions/A/Resources/WindowServer",
        coalition: 300,
        start_s: 86_400 * 3,
        memory: None,
        clients: &[
            (
                0x1000_0a1c,
                &[("Metal", 0.55), ("Metal", 0.2), ("Metal", 0.1)],
            ),
            (0x1000_0b3f, &[("Metal", 0.15)]),
        ],
        busy: |t| {
            35.0 + 15.0 * (t / 7.0).sin()
                + if (t as u64).is_multiple_of(17) {
                    40.0
                } else {
                    0.0
                }
        },
        alive: None,
    },
    Spec {
        pid: 2201,
        ppid: 2190,
        uid: 501,
        name: "Google Chrome Helper (GPU)",
        path: "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Versions/141.0.7390.55/Helpers/Google Chrome Helper (GPU).app/Contents/MacOS/Google Chrome Helper (GPU)",
        coalition: 1201,
        start_s: 7_200,
        memory: Some(642 * MIB),
        clients: &[(0x1000_2c40, &[("Metal", 0.8), ("Metal", 0.2)])],
        busy: |t| 70.0 + 40.0 * (t / 11.0).sin(),
        alive: None,
    },
    Spec {
        pid: 2190,
        ppid: 1,
        uid: 501,
        name: "Google Chrome",
        path: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        coalition: 1201,
        start_s: 7_260,
        memory: Some(38 * MIB),
        clients: &[(0x1000_2b10, &[("Metal", 1.0)])],
        busy: |_| 0.0,
        alive: None,
    },
    Spec {
        pid: 7001,
        ppid: 1,
        uid: 501,
        name: "MTLCompilerService",
        path: "/System/Library/Frameworks/Metal.framework/Versions/A/XPCServices/MTLCompilerService.xpc/Contents/MacOS/MTLCompilerService",
        coalition: 1201,
        start_s: 30,
        memory: Some(3 * MIB),
        clients: &[(0x1000_7a01, &[("Metal", 1.0)])],
        busy: |_| 30.0,
        alive: Some((8, 12)),
    },
    Spec {
        pid: 3310,
        ppid: 3301,
        uid: 501,
        name: "ollama",
        path: "/Applications/Ollama.app/Contents/Resources/ollama",
        coalition: 1400,
        start_s: 1_800,
        memory: Some(18 * GIB + 400 * MIB),
        clients: &[(0x1000_3e02, &[("Metal", 0.52), ("Metal", 0.48)])],
        busy: |t| {
            if t % 40.0 < 24.0 {
                450.0 + 60.0 * t.sin()
            } else {
                0.0
            }
        },
        alive: None,
    },
    Spec {
        pid: 2800,
        ppid: 1,
        uid: 501,
        name: "com.apple.WebKit.GPU",
        path: "/System/Library/Frameworks/WebKit.framework/Versions/A/XPCServices/com.apple.WebKit.GPU.xpc/Contents/MacOS/com.apple.WebKit.GPU",
        coalition: 1300,
        start_s: 4_000,
        memory: Some(210 * MIB),
        clients: &[(0x1000_2f88, &[("Metal", 1.0)])],
        busy: |t| 8.0 + 5.0 * (t / 3.0).sin(),
        alive: None,
    },
    Spec {
        pid: 1800,
        ppid: 1,
        uid: 501,
        name: "ghostty",
        path: "/Applications/Ghostty.app/Contents/MacOS/ghostty",
        coalition: 1100,
        start_s: 20_000,
        memory: Some(96 * MIB),
        clients: &[(0x1000_1a00, &[("Metal", 1.0)])],
        busy: |t| 3.0 + 2.0 * (t / 2.0).sin(),
        alive: None,
    },
    Spec {
        pid: 6120,
        ppid: 1805,
        uid: 501,
        name: "python3.12",
        path: "/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/bin/python3.12",
        coalition: 1100,
        start_s: 10,
        memory: Some(9 * GIB + 200 * MIB),
        clients: &[(0x1000_6c00, &[("Metal", 1.0)])],
        busy: |t| {
            if t >= 20.0 {
                240.0 + 50.0 * (t / 5.0).sin()
            } else {
                0.0
            }
        },
        alive: None,
    },
    Spec {
        pid: 640,
        ppid: 1,
        uid: 501,
        name: "Finder",
        path: "/System/Library/CoreServices/Finder.app/Contents/MacOS/Finder",
        coalition: 600,
        start_s: 86_000,
        memory: Some(12 * MIB),
        clients: &[(0x1000_0640, &[("Metal", 1.0)])],
        busy: |_| 0.0,
        alive: None,
    },
];

const APPS: &[(u64, &str, Option<&str>, i32)] = &[
    (300, "WindowServer", None, 412),
    (
        1201,
        "Google Chrome",
        Some("/Applications/Google Chrome.app"),
        2190,
    ),
    (1300, "Safari", Some("/Applications/Safari.app"), 2700),
    (1400, "Ollama", Some("/Applications/Ollama.app"), 3301),
    (1100, "Ghostty", Some("/Applications/Ghostty.app"), 1800),
    (
        600,
        "Finder",
        Some("/System/Library/CoreServices/Finder.app"),
        640,
    ),
];

fn alive(spec: &Spec, step: u64) -> bool {
    spec.alive
        .is_none_or(|(from, to)| step >= from && step <= to)
}

fn busy_at(spec: &Spec, step: u64, seconds_per_step: f64) -> f64 {
    if !alive(spec, step) {
        return 0.0;
    }
    (spec.busy)(step as f64 * seconds_per_step).max(0.0)
}

/// The synthetic report for `step` (1-based).
pub fn report(step: u64, interval: Duration, include_idle: bool) -> Report {
    let dt = interval.as_secs_f64().max(0.001);
    let timestamp_ms = BASE_MS + (step as f64 * dt * 1000.0) as u64;
    let recent_steps = ((RECENT_SECONDS as f64) / dt).ceil() as u64;

    // The driver apportions busy time among concurrent contexts.
    let demand: f64 = SPECS.iter().map(|spec| busy_at(spec, step, dt)).sum();
    let apportion = if demand > 990.0 { 990.0 / demand } else { 1.0 };
    let mut processes = Vec::new();
    for spec in SPECS.iter().filter(|spec| alive(spec, step)) {
        let busy = busy_at(spec, step, dt) * apportion;
        let gpu_ns = (busy * 1e6 * dt) as u64;
        let last_active = (0..=step.min(10_000))
            .find(|back| busy_at(spec, step - back, dt) > 0.0)
            .map(|back| back as f64 * dt);
        let activity = match last_active {
            Some(0.0) => Activity::Active,
            Some(age) if age <= recent_steps as f64 * dt => Activity::Recent,
            _ => Activity::Idle,
        };
        let age_ms = match last_active {
            Some(age) => (age * 1000.0) as u64 + 4 + (step % 7),
            None => 3_600_000 + spec.pid as u64 * 1000,
        };
        let lifetime_base = spec.start_s * 1_000_000 * (spec.pid as u64 % 7 + 1);
        let lifetime_gpu_ns = lifetime_base + (busy * 1e6 * step as f64 * dt * 0.8) as u64;
        let clients: Vec<ClientView> = spec
            .clients
            .iter()
            .map(|(registry_id, queues)| {
                let queues: Vec<QueueView> = queues
                    .iter()
                    .enumerate()
                    .map(|(index, (api, fraction))| QueueView {
                        index,
                        api: (*api).into(),
                        gpu_ns: (gpu_ns as f64 * fraction) as u64,
                        busy_ms_per_s: busy * fraction,
                        lifetime_gpu_ns: (lifetime_gpu_ns as f64 * fraction) as u64,
                        last_submit_age_ms: Some(age_ms + index as u64 * 3),
                        activity,
                    })
                    .collect();
                ClientView {
                    registry_id: *registry_id,
                    gpu_ns: queues.iter().map(|queue| queue.gpu_ns).sum(),
                    busy_ms_per_s: queues.iter().map(|queue| queue.busy_ms_per_s).sum(),
                    lifetime_gpu_ns: queues.iter().map(|queue| queue.lifetime_gpu_ns).sum(),
                    command_queue_count: Some(queues.len() as u32),
                    command_queue_count_max: Some(queues.len() as u32 + 1),
                    last_submit_age_ms: Some(age_ms),
                    queues,
                }
            })
            .collect();
        processes.push(ProcessView {
            pid: spec.pid,
            start_us: (BASE_MS / 1000 - spec.start_s) * 1_000_000,
            name: spec.name.into(),
            path: Some(spec.path.into()),
            uid: Some(spec.uid),
            ppid: Some(spec.ppid),
            coalition_id: Some(spec.coalition),
            gpu_ns,
            busy_ms_per_s: busy,
            share_pct: 0.0,
            device_pct: None,
            est_power_w: None,
            lifetime_gpu_ns,
            graphics_memory_bytes: spec.memory.map(|bytes| {
                let wobble = (step % 5) * MIB / 4;
                if busy > 0.0 { bytes + wobble } else { bytes }
            }),
            activity,
            last_submit_age_ms: Some(age_ms),
            queues: clients.iter().map(|client| client.queues.len()).sum(),
            clients,
        });
    }

    // The compiler service exited at step 12; its last window's time shows up
    // only in Chrome's coalition counter.
    let unattributed_chrome_ns = if step == 13 {
        (30.0 * 1e6 * dt) as u64
    } else {
        (1.5 * 1e6 * dt) as u64
    };

    let attributed_ns: u64 = processes.iter().map(|process| process.gpu_ns).sum();
    let attributed_ms_per_s = attributed_ns as f64 / 1e6 / dt;
    let active_pct = (2.0 + attributed_ms_per_s / 10.2).clamp(0.5, 99.6);
    let avg_mhz = 338.0 + (MAX_MHZ - 338.0) * (active_pct / 100.0).powf(0.7);
    let normalized = active_pct * avg_mhz / MAX_MHZ;
    let power_w = 0.08 + 41.0 * (normalized / 100.0).powf(1.25);
    let temperature = 51.0 + 40.0 * (active_pct / 100.0).powf(1.4);

    for process in &mut processes {
        process.share_pct = pct(process.gpu_ns, attributed_ns);
        process.device_pct = Some(process.share_pct * active_pct / 100.0);
    }

    // At sample 50 a Chrome helper exits while Ollama's work overlaps, so its
    // time cannot be separated from the overlap.
    let overlap_unknown = step == 50;
    let coalition_total_ns: u64 = attributed_ns + unattributed_chrome_ns;
    let window_ns = (dt * 1e9) as u64;
    let mut apps = Vec::new();
    for (coalition_id, name, bundle, responsible) in APPS {
        let members: Vec<&ProcessView> = processes
            .iter()
            .filter(|process| process.coalition_id == Some(*coalition_id))
            .collect();
        let attributed: u64 = members.iter().map(|process| process.gpu_ns).sum();
        let unattributed = if *coalition_id == 1201 {
            unattributed_chrome_ns
        } else {
            0
        };
        let gpu_ns = attributed + unattributed;
        // In flight counts overlapped work in full: undo the apportioning, and
        // stretch by how busy everyone else kept the GPU at the same time.
        let others = (attributed_ns - attributed) as f64 / window_ns as f64;
        let in_flight_ns =
            ((attributed as f64 / apportion * (1.0 + others)) as u64 + unattributed).min(window_ns);
        let share = pct(gpu_ns, coalition_total_ns);
        let app_power = power_w * share / 100.0;
        apps.push(AppView {
            coalition_id: *coalition_id,
            name: (*name).into(),
            bundle_path: bundle.map(str::to_string),
            responsible_pid: Some(*responsible),
            gpu_ns,
            busy_ms_per_s: gpu_ns as f64 / 1e6 / dt,
            share_pct: share,
            device_pct: Some(share * active_pct / 100.0),
            attributed_gpu_ns: attributed,
            in_flight_gpu_ns: Some(in_flight_ns),
            in_flight_ms_per_s: Some(in_flight_ns as f64 / 1e6 / dt),
            unattributed_gpu_ns: (!(overlap_unknown && *coalition_id == 1201))
                .then_some(unattributed),
            energy_j: Some(app_power * dt),
            power_w: Some(app_power),
            billed_to_me_w: Some(if *coalition_id == 1201 { 0.31 } else { 0.0 }),
            billed_to_others_w: Some(if *coalition_id == 300 { 0.42 } else { 0.0 }),
            ane_ms_per_s: Some(if *coalition_id == 1100 { 3.5 } else { 0.0 }),
            ane_power_w: Some(if *coalition_id == 1100 { 0.12 } else { 0.0 }),
            lifetime_gpu_ns: members.iter().map(|process| process.lifetime_gpu_ns).sum(),
            lifetime_energy_j: Some(members.len() as f64 * 812.5 + step as f64 * app_power),
            graphics_memory_bytes: members
                .iter()
                .filter_map(|process| process.graphics_memory_bytes)
                .fold(None, |total, bytes| Some(total.unwrap_or(0) + bytes)),
            members: members.len() + 3,
            gpu_pids: members.iter().map(|process| process.pid).collect(),
        });
        let app = apps.last().expect("just pushed");
        for process in processes
            .iter_mut()
            .filter(|process| process.coalition_id == Some(*coalition_id))
        {
            process.est_power_w = Some(if attributed == 0 {
                0.0
            } else {
                app.power_w.unwrap_or(0.0) * process.gpu_ns as f64 / gpu_ns as f64
            });
        }
    }
    apps.retain(|app| app.gpu_ns > 0);
    apps.sort_by_key(|app| std::cmp::Reverse(app.gpu_ns));

    let off = 100.0 - active_pct;
    let center = (avg_mhz - 338.0) / (MAX_MHZ - 338.0) * 14.0;
    let weights: Vec<f64> = (0..15)
        .map(|index| (-((index as f64 - center).powi(2)) / 4.0).exp())
        .collect();
    let weight_sum: f64 = weights.iter().sum();
    let mut pstates = vec![PStateShare {
        name: "OFF".into(),
        mhz: None,
        pct: off,
    }];
    pstates.extend(
        weights
            .iter()
            .enumerate()
            .map(|(index, weight)| PStateShare {
                name: format!("P{}", index + 1),
                mhz: Some(PSTATE_MHZ[index]),
                pct: active_pct * weight / weight_sum,
            }),
    );

    let heavy = active_pct > 85.0;
    let hot = temperature > 85.0;
    let mut limiters = vec![StateShare {
        name: "PERF".into(),
        pct: if heavy { 0.0 } else { 100.0 },
    }];
    if heavy {
        limiters.push(StateShare {
            name: "PWR_ZONE".into(),
            pct: 100.0,
        });
    }
    limiters.retain(|limiter| limiter.pct > 0.0);
    // Keys as the sampler emits them: every channel, zeros included.
    let throttles = if hot { 3 } else { 0 };
    let counters: BTreeMap<String, u64> = [
        ("GPU Throttler Counters/Throttle Counter DPE", throttles),
        ("GPU Throttler Counters/Throttle Counter Misc", 0),
        ("GPU Throttler Counters/Throttle Counter Total", throttles),
        ("THM Counters/SFE Tier3 Count", 0),
        ("THM Counters/THM Activation Count", u64::from(hot)),
        ("Issue Counts/Total Restarts", u64::from(step == 45)),
        ("Issue Counts/DM Kill Count", 0),
    ]
    .into_iter()
    .map(|(key, count)| (key.to_string(), count))
    .collect();

    let mut events = Vec::new();
    if step == 30 {
        events.push(Event {
            timestamp_ms,
            kind: EventKind::Alert,
            message: "alert active>90 firing: active 99.6% for 10s".into(),
        });
    }
    if step == 40 {
        events.push(Event {
            timestamp_ms,
            kind: EventKind::AlertCleared,
            message: "alert active>90 cleared".into(),
        });
    }
    if step == 45 {
        events.push(Event {
            timestamp_ms,
            kind: EventKind::GpuRecovery,
            message: "GPU recovered from a hang (recovery count 1)".into(),
        });
    }

    let in_use: u64 = processes
        .iter()
        .filter_map(|process| process.graphics_memory_bytes)
        .sum::<u64>()
        + 2 * GIB
        + 300 * MIB;
    let active_processes = processes
        .iter()
        .filter(|process| process.gpu_ns > 0)
        .count();
    let gpu_processes = processes.len();
    let gpu_clients = processes.iter().map(|process| process.clients.len()).sum();
    let active_apps = apps.len();
    if !include_idle {
        processes.retain(|process| process.gpu_ns > 0);
    }
    processes.sort_by(|a, b| b.gpu_ns.cmp(&a.gpu_ns).then(a.pid.cmp(&b.pid)));

    let mut sources = BTreeMap::new();
    for name in [
        source::REGISTRY,
        source::IOREPORT,
        source::COALITIONS,
        source::TEMPERATURE,
        source::SYSTEM,
    ] {
        sources.insert(name.to_string(), SourceStatus::ok());
    }
    sources.insert(
        source::MEMORY.to_string(),
        SourceStatus::with(SourceState::Partial, "other users' processes need root"),
    );

    Report {
        schema_version: SCHEMA_VERSION,
        timestamp_ms,
        window_seconds: dt,
        device: DeviceView {
            model: Some("Apple M4 Max".into()),
            gpu_cores: Some(40),
            firmware_utilization: Some(FirmwareUtilization {
                device_pct: (active_pct * 0.94).min(100.0),
                renderer_pct: (active_pct * 0.81).min(100.0),
                tiler_pct: (active_pct * 0.12).min(100.0),
                readings: 4,
            }),
            active_pct: Some(active_pct),
            avg_mhz: Some(avg_mhz),
            max_mhz: Some(MAX_MHZ),
            freq_normalized_pct: Some(normalized),
            power_w: Some(power_w),
            pstates,
            limiters,
            thermal_limited_pct: Some(if hot { (temperature - 85.0) * 4.0 } else { 0.0 }),
            counters,
            temperature: Some(Temperature {
                avg_c: temperature,
                max_c: temperature + 5.6,
                sensors: 44,
            }),
            memory: DeviceMemory {
                in_use_bytes: Some(in_use),
                allocated_bytes: Some(in_use + in_use / 7),
                wired_limit_bytes: Some(96 * GIB),
                total_physical_bytes: Some(128 * GIB),
            },
            system: SystemView {
                thermal_pressure: Some(if hot { "moderate" } else { "nominal" }.into()),
                memory_pressure: Some("normal".into()),
            },
            recovery_count: Some(u64::from(step >= 45)),
        },
        totals: Totals {
            attributed_gpu_ns: attributed_ns,
            attributed_ms_per_s,
            gpu_clients,
            gpu_processes,
            active_processes,
            active_apps,
        },
        processes,
        apps,
        unattributed: Some(Unattributed {
            gpu_ns: if overlap_unknown {
                0
            } else {
                unattributed_chrome_ns
            },
            busy_ms_per_s: if overlap_unknown {
                0.0
            } else {
                unattributed_chrome_ns as f64 / 1e6 / dt
            },
            partial: overlap_unknown,
        }),
        events,
        sources,
        overhead: Overhead {
            sample_wall_ms: 1.9,
            sample_cpu_ms: 1.1,
            process_cpu_pct: 0.4,
        },
    }
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_are_consistent() {
        for step in [1, 13, 30, 45] {
            let report = report(step, Duration::from_secs(1), true);
            let share: f64 = report
                .processes
                .iter()
                .map(|process| process.share_pct)
                .sum();
            assert!((share - 100.0).abs() < 1e-6, "{share}");
            let device: f64 = report
                .processes
                .iter()
                .filter_map(|process| process.device_pct)
                .sum();
            assert!((device - report.device.active_pct.unwrap()).abs() < 1e-6);
            assert!(
                report
                    .apps
                    .iter()
                    .all(|app| app.gpu_ns >= app.attributed_gpu_ns)
            );
        }
        assert_eq!(report(45, Duration::from_secs(1), false).events.len(), 1);
    }

    #[test]
    fn replay_feed_ends_and_refuses_interval_changes() {
        let mut feed = SyntheticFeed::replay(2);
        assert!(!feed.is_live());
        assert!(!feed.set_interval(Duration::from_secs(2)));
        assert!(matches!(
            feed.next_report(Duration::ZERO),
            FeedEvent::Report(_)
        ));
        assert!(matches!(
            feed.next_report(Duration::ZERO),
            FeedEvent::Report(_)
        ));
        assert!(matches!(feed.next_report(Duration::ZERO), FeedEvent::End));
    }
}
