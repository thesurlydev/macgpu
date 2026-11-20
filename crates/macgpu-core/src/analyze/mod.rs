//! Turns two [`Snapshot`]s into a [`Report`].
//!
//! Everything here is a pure function of the snapshots, so accuracy rules are
//! unit tested with hand-built inputs. The one exception is app energy over
//! consecutive windows: coalition energy moves in steps about 0.5 s apart, so
//! an [`Analyzer`] carries just under a second of it from window to window
//! (see the `apps` module).

mod apps;
mod device;
mod processes;

use crate::snapshot::Snapshot;
use macgpu_model::{Overhead, Report, SCHEMA_VERSION, SourceState, SourceStatus, source};

pub use apps::ENERGY_SPAN_NS;

/// Conversion factors between nanoseconds and seconds or milliseconds.
pub(crate) const NS_PER_S: f64 = 1_000_000_000.0;
pub(crate) const NS_PER_MS: f64 = 1_000_000.0;

/// Saturating sum of GPU or energy counters.
pub(crate) fn saturating_sum(values: impl Iterator<Item = u64>) -> u64 {
    values.fold(0u64, u64::saturating_add)
}

/// A process or queue that submitted work within this many seconds, but not in
/// the current window, is shown as recent rather than idle.
pub const RECENT_SECONDS: u64 = 10;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnalyzeOptions {
    /// Keep processes with no GPU time in the window.
    pub include_idle: bool,
}

/// Analyzes consecutive windows: each call's `previous` is the last call's
/// `current`. Sampling loops use this rather than [`analyze`], so app energy
/// in windows shorter than [`ENERGY_SPAN_NS`] is split by the coalition energy
/// of that trailing span rather than lost.
#[derive(Debug, Clone, Default)]
pub struct Analyzer {
    options: AnalyzeOptions,
    energy: apps::EnergyHistory,
}

impl Analyzer {
    pub fn new(options: AnalyzeOptions) -> Self {
        Self {
            options,
            energy: apps::EnergyHistory::default(),
        }
    }

    pub fn set_options(&mut self, options: AnalyzeOptions) {
        self.options = options;
    }

    /// Builds the report for the window from `previous` to `current`.
    pub fn analyze(&mut self, previous: &Snapshot, current: &Snapshot) -> Report {
        build(previous, current, self.options, &mut self.energy)
    }
}

/// Builds the report for one window from `previous` to `current` on its own.
/// App energy in a window shorter than [`ENERGY_SPAN_NS`] then sees only that
/// window's coalition energy; use an [`Analyzer`] for consecutive windows.
pub fn analyze(previous: &Snapshot, current: &Snapshot, options: &AnalyzeOptions) -> Report {
    Analyzer::new(*options).analyze(previous, current)
}

fn build(
    previous: &Snapshot,
    current: &Snapshot,
    options: AnalyzeOptions,
    energy: &mut apps::EnergyHistory,
) -> Report {
    let window_ns = current
        .at
        .saturating_duration_since(previous.at)
        .as_nanos()
        .max(1) as u64;
    let window_seconds = window_ns as f64 / NS_PER_S;

    let mut report = Report {
        schema_version: SCHEMA_VERSION,
        timestamp_ms: current.unix_ms,
        window_seconds,
        ..Report::default()
    };

    report.sources.insert(
        source::REGISTRY.to_string(),
        match (&previous.registry, &current.registry) {
            (Ok(_), Ok(_)) => SourceStatus::ok(),
            (Err(_), Ok(_)) => {
                SourceStatus::with(SourceState::Partial, "previous sample failed; no deltas")
            }
            (_, Err(error)) => SourceStatus::with(SourceState::Unavailable, error.clone()),
        },
    );

    let built = processes::build(previous, current, window_ns);
    report.processes = built.processes;
    report.totals.gpu_clients = built.clients;

    device::fill(previous, current, window_ns, &mut report);
    processes::apply_device_share(&mut report.processes, report.device.active_pct);
    apps::fill(
        previous,
        current,
        window_ns,
        options.include_idle,
        energy,
        &mut report,
    );

    report.totals.gpu_processes = report.processes.len();
    report.totals.active_processes = report
        .processes
        .iter()
        .filter(|process| process.gpu_ns > 0)
        .count();
    report.totals.active_apps = report.apps.iter().filter(|app| app.gpu_ns > 0).count();
    report.totals.attributed_gpu_ns =
        saturating_sum(report.processes.iter().map(|process| process.gpu_ns));
    report.totals.attributed_ms_per_s = ms_per_s(report.totals.attributed_gpu_ns, window_ns);

    if !options.include_idle {
        report.processes.retain(|process| process.gpu_ns > 0);
    }

    report.overhead = Overhead {
        sample_wall_ms: current.cost_wall_ns as f64 / NS_PER_MS,
        sample_cpu_ms: current.cost_cpu_ns as f64 / NS_PER_MS,
        process_cpu_pct: current
            .process_cpu_ns
            .saturating_sub(previous.process_cpu_ns) as f64
            / window_ns as f64
            * 100.0,
    };

    report
}

/// GPU time in milliseconds per wall-clock second.
pub(crate) fn ms_per_s(gpu_ns: u64, window_ns: u64) -> f64 {
    if window_ns == 0 {
        0.0
    } else {
        gpu_ns as f64 / NS_PER_MS / (window_ns as f64 / NS_PER_S)
    }
}

/// `part / whole` as a percentage, 0 when `whole` is 0.
pub(crate) fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use crate::snapshot::Snapshot;
    use crate::sys::iokit::{Accelerator, GpuClient, QueueUsage, Registry};
    use crate::sys::process::ProcessInfo;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    pub fn queue(accumulated_ns: u64, last_submitted_ns: u64) -> QueueUsage {
        QueueUsage {
            api: "Metal".into(),
            accumulated_ns,
            last_submitted_ns,
        }
    }

    pub fn client(registry_id: u64, pid: i32, queues: Vec<QueueUsage>) -> GpuClient {
        GpuClient {
            registry_id,
            pid,
            creator_name: format!("proc{pid}"),
            queues,
            command_queue_count: None,
            command_queue_count_max: None,
        }
    }

    /// A snapshot `offset` after `base` with the given clients.
    pub fn snapshot(
        base: Instant,
        offset: Duration,
        mach_now_ns: u64,
        clients: Vec<GpuClient>,
    ) -> Snapshot {
        let mut processes = HashMap::new();
        for client in &clients {
            processes.insert(
                client.pid,
                ProcessInfo {
                    pid: client.pid,
                    ppid: 1,
                    uid: 501,
                    start_us: 1_000_000 + client.pid as u64,
                    path: Some(format!("/usr/bin/proc{}", client.pid)),
                    name: format!("proc{}", client.pid),
                },
            );
        }
        Snapshot {
            at: base + offset,
            unix_ms: 1_700_000_000_000 + offset.as_millis() as u64,
            mach_now_ns,
            registry: Ok(Registry {
                accelerators: vec![Accelerator {
                    registry_id: 1,
                    clients,
                    ..Accelerator::default()
                }],
            }),
            firmware_readings: Vec::new(),
            processes,
            ioreport: None,
            coalitions: None,
            memory: None,
            temperatures: None,
            system: None,
            frequencies: None,
            cost_wall_ns: 500_000,
            cost_cpu_ns: 400_000,
            process_cpu_ns: offset.as_nanos() as u64 / 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn report_totals_and_overhead() {
        let base = Instant::now();
        let previous = snapshot(
            base,
            Duration::ZERO,
            0,
            vec![client(10, 100, vec![queue(1_000, 0)])],
        );
        let current = snapshot(
            base,
            Duration::from_secs(2),
            2_000_000_000,
            vec![client(10, 100, vec![queue(500_001_000, 1_900_000_000)])],
        );
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        assert_eq!(report.schema_version, SCHEMA_VERSION);
        assert!((report.window_seconds - 2.0).abs() < 1e-9);
        assert_eq!(report.totals.attributed_gpu_ns, 500_000_000);
        assert!((report.totals.attributed_ms_per_s - 250.0).abs() < 1e-9);
        assert_eq!(report.totals.active_processes, 1);
        assert!((report.overhead.process_cpu_pct - 1.0).abs() < 1e-9);
        assert_eq!(report.sources[source::REGISTRY].state, SourceState::Ok);
    }

    #[test]
    fn idle_processes_are_filtered_unless_requested() {
        let base = Instant::now();
        let clients = vec![
            client(10, 100, vec![queue(1_000, 0)]),
            client(11, 101, vec![queue(5, 0)]),
        ];
        let previous = snapshot(base, Duration::ZERO, 0, clients.clone());
        let mut moved = clients;
        moved[0].queues[0].accumulated_ns += 1_000_000;
        let current = snapshot(base, Duration::from_secs(1), 1_000_000_000, moved);

        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        assert_eq!(report.processes.len(), 1);
        assert_eq!(report.totals.gpu_processes, 2);

        let report = analyze(&previous, &current, &AnalyzeOptions { include_idle: true });
        assert_eq!(report.processes.len(), 2);
    }
}
