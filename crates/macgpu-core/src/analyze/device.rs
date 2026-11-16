//! Whole-GPU view: identity, firmware utilization, memory, recoveries, and
//! (from IOReport, SMC, and system sources) residency, clock, power, limiters,
//! temperature, and pressure.

use super::NS_PER_S;
use crate::snapshot::Snapshot;
use crate::sys::gpu_freq::FrequencyTable;
use crate::sys::iokit::{Accelerator, Registry};
use crate::sys::ioreport::{IoReportSample, channel, energy_to_joules};
use crate::sys::smc::{self, GpuTemperatures};
use crate::sys::system::{self, SystemContext};
use macgpu_model::{
    DeviceView, Event, EventKind, FirmwareUtilization, PStateShare, Report, SourceState,
    SourceStatus, StateShare, Temperature, source,
};
use std::collections::BTreeMap;

/// IOReport state names that mean the GPU is powered down rather than clocked.
const INACTIVE_STATES: [&str; 3] = ["OFF", "IDLE", "DOWN"];
/// `PWRCTRL` state for a powered-down GPU; limiter shares exclude it.
const POWER_CONTROLLER_IDLE: &str = "IDLE_OFF";
/// `GPU_CLTM` state when no thermal cap is in force.
const CLTM_UNCAPPED: &str = "NO_CLTM";
/// Throttle counters whose names contain this are gauges, not counters.
const GAUGE_MARKER: &str = "Percent";
/// Active residency above which a GPU Energy counter that did not move is
/// reported as stale.
const STALE_ENERGY_ACTIVE_PCT: f64 = 1.0;

pub(super) fn fill(previous: &Snapshot, current: &Snapshot, window_ns: u64, report: &mut Report) {
    let device = &mut report.device;

    if let Ok(registry) = &current.registry {
        if let Some(accelerator) = registry.accelerators.first() {
            device.model = accelerator.info.model.clone();
            device.gpu_cores = accelerator.info.core_count;
        }
        device.memory.in_use_bytes = registry_sum(registry, |accelerator| {
            accelerator.performance.in_use_system_memory
        });
        device.memory.allocated_bytes = registry_sum(registry, |accelerator| {
            accelerator.performance.alloc_system_memory
        });
        device.recovery_count = registry_sum(registry, |accelerator| {
            accelerator.performance.recovery_count
        });
    }

    device.firmware_utilization = firmware_utilization(current);

    let previous_recoveries = previous.registry.as_ref().ok().and_then(|registry| {
        registry_sum(registry, |accelerator| {
            accelerator.performance.recovery_count
        })
    });
    if let (Some(before), Some(after)) = (previous_recoveries, device.recovery_count)
        && after > before
    {
        report.events.push(Event {
            timestamp_ms: current.unix_ms,
            kind: EventKind::GpuRecovery,
            message: format!(
                "GPU recovered from {} hang(s); {after} since boot",
                after - before
            ),
        });
    }

    let ioreport = match (&previous.ioreport, &current.ioreport) {
        (_, None) => SourceStatus::with(SourceState::Disabled, "disabled"),
        (_, Some(Err(error))) => SourceStatus::with(SourceState::Unavailable, error.clone()),
        (Some(Ok(before)), Some(Ok(after))) => {
            let mut health = Health::default();
            fill_ioreport(
                before,
                after,
                current.frequencies.as_ref(),
                window_ns,
                device,
                &mut health,
            );
            if let Some(event) = issue_event(before, after, current.unix_ms) {
                report.events.push(event);
            }
            health.status()
        }
        (_, Some(Ok(_))) => {
            SourceStatus::with(SourceState::Partial, "previous sample failed; no deltas")
        }
    };
    report
        .sources
        .insert(source::IOREPORT.to_string(), ioreport);

    let temperature = match &current.temperatures {
        None => SourceStatus::with(SourceState::Disabled, "disabled"),
        Some(Err(error)) => SourceStatus::with(SourceState::Unavailable, error.clone()),
        Some(Ok(temperatures)) => fill_temperature(temperatures, device),
    };
    report
        .sources
        .insert(source::TEMPERATURE.to_string(), temperature);

    let system = match &current.system {
        None => SourceStatus::with(SourceState::Disabled, "disabled"),
        Some(context) => fill_system(context, device),
    };
    report.sources.insert(source::SYSTEM.to_string(), system);
    if let (Some(before), Some(after)) = (&previous.system, &current.system)
        && let Some(event) = thermal_event(before, after, current.unix_ms)
    {
        report.events.push(event);
    }
}

/// Problems found while reading one source, turned into its status.
#[derive(Debug, Default)]
struct Health {
    /// Data that is missing or unusable.
    missing: Vec<String>,
    /// Counters that exist but did not advance when they should have.
    stale: Vec<String>,
}

impl Health {
    fn status(self) -> SourceStatus {
        let state = if !self.stale.is_empty() {
            SourceState::Stale
        } else if !self.missing.is_empty() {
            SourceState::Partial
        } else {
            return SourceStatus::ok();
        };
        let details: Vec<String> = self.stale.into_iter().chain(self.missing).collect();
        SourceStatus::with(state, details.join("; "))
    }
}

/// Residency per state of every channel named `(subgroup, name)` between two
/// samples, summed by state name (a chip with several GPU dies reports one
/// channel per die). `None` when the channel is missing from either sample or
/// any of its counters reset.
fn residency(
    before: &IoReportSample,
    after: &IoReportSample,
    (subgroup, name): (&str, &str),
) -> Option<Vec<(String, u64)>> {
    let mut total: Option<Vec<(String, u64)>> = None;
    for channel in after.find(subgroup, name) {
        let delta = channel.state_delta(before.matching(channel)?)?;
        match &mut total {
            None => total = Some(delta),
            Some(total) => {
                for (state, ticks) in delta {
                    match total.iter_mut().find(|(name, _)| *name == state) {
                        Some((_, sum)) => *sum = sum.saturating_add(ticks),
                        None => total.push((state, ticks)),
                    }
                }
            }
        }
    }
    total
}

fn fill_ioreport(
    before: &IoReportSample,
    after: &IoReportSample,
    table: Option<&FrequencyTable>,
    window_ns: u64,
    device: &mut DeviceView,
    health: &mut Health,
) {
    match residency(before, after, channel::PERFORMANCE_STATES) {
        None => health
            .missing
            .push("GPU performance states (GPUPH) unavailable".into()),
        Some(states) => fill_performance_states(&states, table, device, health),
    }

    let seconds = sample_seconds(before, after, window_ns);
    match energy_joules(before, after) {
        Err(problem) => health.missing.push(problem),
        Ok(joules) => {
            let active_pct = device.active_pct.unwrap_or(0.0);
            if joules == 0.0 && active_pct > STALE_ENERGY_ACTIVE_PCT {
                health.stale.push(format!(
                    "GPU Energy did not advance although the GPU was {active_pct:.0}% active"
                ));
            } else {
                device.power_w = Some(joules / seconds);
            }
        }
    }

    match residency(before, after, channel::POWER_CONTROLLER) {
        None => health
            .missing
            .push("power controller states (PWRCTRL) unavailable".into()),
        Some(states) => device.limiters = limiter_shares(&states),
    }

    match residency(before, after, channel::CLTM) {
        None => health
            .missing
            .push("thermal limiting (GPU_CLTM) unavailable".into()),
        Some(states) => device.thermal_limited_pct = thermal_limited_pct(&states),
    }

    device.counters = counter_deltas(before, after);
}

/// Seconds between the two IOReport samples, falling back to the window.
fn sample_seconds(before: &IoReportSample, after: &IoReportSample, window_ns: u64) -> f64 {
    let ns = after.taken_at_ns.saturating_sub(before.taken_at_ns);
    let ns = if ns == 0 { window_ns.max(1) } else { ns };
    ns as f64 / NS_PER_S
}

/// Fills active residency, P-state shares, and clock from `GPUPH` residency.
fn fill_performance_states(
    states: &[(String, u64)],
    table: Option<&FrequencyTable>,
    device: &mut DeviceView,
    health: &mut Health,
) {
    let total: u64 = states.iter().map(|(_, ticks)| ticks).sum();
    if total == 0 {
        health
            .stale
            .push("GPU performance states (GPUPH) did not advance".into());
        return;
    }
    let active_states: Vec<&(String, u64)> = states
        .iter()
        .filter(|(name, _)| !INACTIVE_STATES.contains(&name.as_str()))
        .collect();
    let active: u64 = active_states.iter().map(|(_, ticks)| ticks).sum();
    let active_pct = active as f64 / total as f64 * 100.0;
    device.active_pct = Some(active_pct);

    let table = match table {
        None => {
            health
                .missing
                .push("no GPU frequency table in the device tree; MHz unavailable".into());
            None
        }
        Some(table) if !table_fits(table, &active_states) => {
            health.missing.push(format!(
                "GPU frequency table has {} states but GPUPH has {}; MHz unavailable",
                table.mhz.len(),
                active_states.len()
            ));
            None
        }
        Some(table) => Some(table),
    };

    device.pstates = states
        .iter()
        .map(|(name, ticks)| PStateShare {
            name: name.clone(),
            mhz: table.and_then(|table| table.mhz_for_state(name)),
            pct: *ticks as f64 / total as f64 * 100.0,
        })
        .collect();

    let Some(table) = table else {
        return;
    };
    let Some(max_mhz) = table.max_mhz() else {
        return;
    };
    device.max_mhz = Some(max_mhz);
    if active == 0 {
        // Idle all window: there is no clock to average, but the normalized
        // utilization is a real zero.
        device.freq_normalized_pct = Some(0.0);
        return;
    }
    let weighted: f64 = active_states
        .iter()
        .filter_map(|(name, ticks)| Some(table.mhz_for_state(name)? * *ticks as f64))
        .sum();
    let avg_mhz = weighted / active as f64;
    device.avg_mhz = Some(avg_mhz);
    device.freq_normalized_pct = Some(active_pct * avg_mhz / max_mhz);
}

/// True when every active state has a clock in the table and the table has no
/// extra entries, so the index-to-MHz mapping is the one the table defines.
fn table_fits(table: &FrequencyTable, active_states: &[&(String, u64)]) -> bool {
    table.mhz.len() == active_states.len()
        && active_states
            .iter()
            .all(|(name, _)| table.mhz_for_state(name).is_some())
}

/// GPU energy over the window in joules, summed over every `GPU Energy` channel.
fn energy_joules(before: &IoReportSample, after: &IoReportSample) -> Result<f64, String> {
    let mut joules = None;
    for channel in after.find("", channel::GPU_ENERGY) {
        let Some(earlier) = before.matching(channel) else {
            continue;
        };
        let delta = channel
            .simple_delta(earlier)
            .ok_or_else(|| "GPU Energy counter reset or invalid".to_string())?;
        let value = energy_to_joules(delta, &channel.unit)
            .ok_or_else(|| format!("GPU Energy has unknown unit {:?}", channel.unit))?;
        joules = Some(joules.unwrap_or(0.0) + value);
    }
    joules.ok_or_else(|| "GPU Energy channel unavailable".to_string())
}

/// `PWRCTRL` states with nonzero residency, as a share of the time the GPU was
/// powered (every state but `IDLE_OFF`), largest first.
fn limiter_shares(states: &[(String, u64)]) -> Vec<StateShare> {
    let powered: u64 = states
        .iter()
        .filter(|(name, _)| name != POWER_CONTROLLER_IDLE)
        .map(|(_, ticks)| ticks)
        .sum();
    if powered == 0 {
        return Vec::new();
    }
    let mut shares: Vec<StateShare> = states
        .iter()
        .filter(|(name, ticks)| name != POWER_CONTROLLER_IDLE && *ticks > 0)
        .map(|(name, ticks)| StateShare {
            name: name.clone(),
            pct: *ticks as f64 / powered as f64 * 100.0,
        })
        .collect();
    shares.sort_by(|a, b| b.pct.total_cmp(&a.pct));
    shares
}

/// Share of the window with a thermal (CLTM) cap in force: every `GPU_CLTM`
/// state except `NO_CLTM`. `None` when the channel has no `NO_CLTM` state or
/// did not advance.
fn thermal_limited_pct(states: &[(String, u64)]) -> Option<f64> {
    let total: u64 = states.iter().map(|(_, ticks)| ticks).sum();
    let uncapped = states
        .iter()
        .find(|(name, _)| name == CLTM_UNCAPPED)
        .map(|(_, ticks)| *ticks)?;
    (total > 0).then(|| total.saturating_sub(uncapped) as f64 / total as f64 * 100.0)
}

/// Increments of the throttle, THM, and issue counters, keyed
/// `"<subgroup>/<channel>"`. Gauges and counters that reset are skipped.
fn counter_deltas(before: &IoReportSample, after: &IoReportSample) -> BTreeMap<String, u64> {
    let mut counters = BTreeMap::new();
    for subgroup in [
        channel::THROTTLE_COUNTERS,
        channel::THM_COUNTERS,
        channel::ISSUE_COUNTS,
    ] {
        for channel in after.subgroup(subgroup) {
            if channel.name.contains(GAUGE_MARKER) {
                continue;
            }
            if let Some(delta) = before
                .matching(channel)
                .and_then(|earlier| channel.simple_delta(earlier))
            {
                counters.insert(format!("{subgroup}/{}", channel.name), delta);
            }
        }
    }
    counters
}

/// An event when any IOReport issue counter (restarts, kills) increased.
fn issue_event(
    before: &IoReportSample,
    after: &IoReportSample,
    timestamp_ms: u64,
) -> Option<Event> {
    let increases: Vec<String> = after
        .subgroup(channel::ISSUE_COUNTS)
        .filter_map(|channel| {
            let delta = channel.simple_delta(before.matching(channel)?)?;
            (delta > 0).then(|| format!("{} +{delta}", channel.name))
        })
        .collect();
    (!increases.is_empty()).then(|| Event {
        timestamp_ms,
        kind: EventKind::GpuIssue,
        message: format!("GPU issue counters increased: {}", increases.join(", ")),
    })
}

fn fill_temperature(temperatures: &GpuTemperatures, device: &mut DeviceView) -> SourceStatus {
    match temperatures.summary() {
        Some((avg_c, max_c, sensors)) => {
            device.temperature = Some(Temperature {
                avg_c,
                max_c,
                sensors,
            });
            if (sensors as usize) < temperatures.sensors.len() {
                SourceStatus::with(
                    SourceState::Partial,
                    format!(
                        "{} of {} GPU sensors gave implausible readings",
                        temperatures.sensors.len() - sensors as usize,
                        temperatures.sensors.len()
                    ),
                )
            } else {
                SourceStatus::ok()
            }
        }
        None if temperatures.not_reporting() => SourceStatus::with(
            SourceState::Stale,
            format!(
                "GPU temperature sensors are not reporting (all {} read below {} C, as they \
                 do for about a second at a time while the GPU idles); temperature left out",
                temperatures.sensors.len(),
                smc::MIN_PLAUSIBLE_C
            ),
        ),
        None => SourceStatus::with(
            SourceState::Unavailable,
            "no GPU temperature sensor gave a plausible reading",
        ),
    }
}

fn fill_system(context: &SystemContext, device: &mut DeviceView) -> SourceStatus {
    let mut health = Health::default();
    device.system.thermal_pressure = context
        .thermal_pressure_level
        .map(system::thermal_pressure_name);
    if device.system.thermal_pressure.is_none() {
        health.missing.push("thermal pressure unreadable".into());
    }
    device.system.memory_pressure = context
        .memory_pressure_level
        .map(system::memory_pressure_name);
    if device.system.memory_pressure.is_none() {
        health.missing.push("memory pressure unreadable".into());
    }
    device.memory.total_physical_bytes = context.total_physical_bytes;
    if context.total_physical_bytes.is_none() {
        health
            .missing
            .push("physical memory size unreadable".into());
    }
    device.memory.wired_limit_bytes = context.wired_limit_bytes();
    if context.wired_limit_mb.is_none() {
        health
            .missing
            .push("iogpu.wired_limit_mb unreadable".into());
    }
    let status = health.status();
    if status.state == SourceState::Ok && context.wired_limit_is_default() {
        // Not a failure: the OS default applies and its value is not exposed.
        return SourceStatus::with(
            SourceState::Ok,
            "GPU wired limit is the OS default (iogpu.wired_limit_mb=0)",
        );
    }
    status
}

/// An event when the thermal pressure level changed between snapshots.
fn thermal_event(
    before: &SystemContext,
    after: &SystemContext,
    timestamp_ms: u64,
) -> Option<Event> {
    let (Some(from), Some(to)) = (before.thermal_pressure_level, after.thermal_pressure_level)
    else {
        return None;
    };
    (from != to).then(|| Event {
        timestamp_ms,
        kind: EventKind::ThermalPressure,
        message: format!(
            "thermal pressure changed from {} to {}",
            system::thermal_pressure_name(from),
            system::thermal_pressure_name(to)
        ),
    })
}

/// Averages every firmware reading taken during the window.
fn firmware_utilization(current: &Snapshot) -> Option<FirmwareUtilization> {
    let readings: Vec<(u8, u8, u8)> = current
        .firmware_readings
        .iter()
        .filter_map(|reading| {
            Some((
                reading.device_utilization_pct?,
                reading.renderer_utilization_pct.unwrap_or(0),
                reading.tiler_utilization_pct.unwrap_or(0),
            ))
        })
        .collect();
    if readings.is_empty() {
        return None;
    }
    let count = readings.len() as f64;
    let mean = |pick: fn(&(u8, u8, u8)) -> u8| {
        readings
            .iter()
            .map(|reading| f64::from(pick(reading)))
            .sum::<f64>()
            / count
    };
    Some(FirmwareUtilization {
        device_pct: mean(|reading| reading.0),
        renderer_pct: mean(|reading| reading.1),
        tiler_pct: mean(|reading| reading.2),
        readings: readings.len() as u32,
    })
}

/// Sums an optional counter over every GPU accelerator.
fn registry_sum(registry: &Registry, value: impl Fn(&Accelerator) -> Option<u64>) -> Option<u64> {
    sum_some(registry.accelerators.iter().map(value))
}

fn sum_some(values: impl Iterator<Item = Option<u64>>) -> Option<u64> {
    values.fold(None, |total, value| match (total, value) {
        (Some(total), Some(value)) => Some(total.saturating_add(value)),
        (total, value) => total.or(value),
    })
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::sys::iokit::PerformanceStatistics;
    use crate::sys::ioreport::testing::{energy, simple_channel, state_channel};
    use crate::sys::ioreport::{ChannelValue, GPU_STATS};
    use crate::sys::smc::SensorReading;
    use std::time::{Duration, Instant};

    fn reading(device: u8) -> PerformanceStatistics {
        PerformanceStatistics {
            device_utilization_pct: Some(device),
            renderer_utilization_pct: Some(device / 2),
            tiler_utilization_pct: Some(1),
            ..PerformanceStatistics::default()
        }
    }

    #[test]
    fn averages_firmware_readings() {
        let base = Instant::now();
        let mut current = snapshot(base, Duration::from_secs(1), 0, vec![]);
        current.firmware_readings = vec![reading(0), reading(100), reading(20)];
        let utilization = firmware_utilization(&current).expect("utilization");
        assert_eq!(utilization.readings, 3);
        assert!((utilization.device_pct - 40.0).abs() < 1e-9);
        assert!((utilization.renderer_pct - 20.0).abs() < 1e-9);
    }

    #[test]
    fn emits_recovery_event() {
        let base = Instant::now();
        let mut previous = snapshot(base, Duration::ZERO, 0, vec![]);
        let mut current = snapshot(base, Duration::from_secs(1), 0, vec![]);
        if let Ok(registry) = &mut previous.registry {
            registry.accelerators[0].performance.recovery_count = Some(1);
        }
        if let Ok(registry) = &mut current.registry {
            registry.accelerators[0].performance.recovery_count = Some(3);
        }
        let mut report = Report::default();
        fill(&previous, &current, 1_000_000_000, &mut report);
        assert_eq!(report.device.recovery_count, Some(3));
        assert_eq!(report.events.len(), 1);
        assert_eq!(report.events[0].kind, EventKind::GpuRecovery);
    }

    /// A table with three active states whose clocks are not sorted.
    fn table() -> FrequencyTable {
        FrequencyTable {
            mhz: vec![400.0, 1200.0, 1000.0],
            source: "test".into(),
        }
    }

    const GPUPH: (&str, &str) = channel::PERFORMANCE_STATES;
    const PWRCTRL: (&str, &str) = channel::POWER_CONTROLLER;
    const CLTM: (&str, &str) = channel::CLTM;

    /// One IOReport sample with cumulative GPUPH ticks for (OFF, P1, P2, P3),
    /// GPU energy in nJ, PWRCTRL (IDLE_OFF, PERF, PWR_ZONE), GPU_CLTM
    /// (P1 cap, NO_CLTM), and a restart counter.
    fn sample(
        at_ns: u64,
        gpuph: [u64; 4],
        energy_nj: u64,
        pwrctrl: [u64; 3],
        cltm: [u64; 2],
        restarts: u64,
    ) -> IoReportSample {
        IoReportSample {
            taken_at_ns: at_ns,
            channels: vec![
                state_channel(
                    GPUPH.0,
                    GPUPH.1,
                    &[
                        ("OFF", gpuph[0]),
                        ("P1", gpuph[1]),
                        ("P2", gpuph[2]),
                        ("P3", gpuph[3]),
                    ],
                ),
                energy(energy_nj),
                state_channel(
                    PWRCTRL.0,
                    PWRCTRL.1,
                    &[
                        ("IDLE_OFF", pwrctrl[0]),
                        ("PERF", pwrctrl[1]),
                        ("PWR_ZONE", pwrctrl[2]),
                    ],
                ),
                state_channel(CLTM.0, CLTM.1, &[("P1", cltm[0]), ("NO_CLTM", cltm[1])]),
                simple_channel(
                    GPU_STATS,
                    channel::ISSUE_COUNTS,
                    "Total Restarts",
                    "events",
                    restarts,
                ),
                simple_channel(
                    GPU_STATS,
                    channel::THROTTLE_COUNTERS,
                    "IVDM: Max Throttle Percent",
                    "",
                    100,
                ),
                simple_channel(
                    GPU_STATS,
                    channel::THROTTLE_COUNTERS,
                    "Throttle Counter Total",
                    "",
                    restarts * 7,
                ),
            ],
        }
    }

    fn pair(before: IoReportSample, after: IoReportSample) -> (Snapshot, Snapshot) {
        let base = Instant::now();
        let mut previous = snapshot(base, Duration::ZERO, 0, vec![]);
        let mut current = snapshot(base, Duration::from_secs(2), 0, vec![]);
        previous.ioreport = Some(Ok(before));
        current.ioreport = Some(Ok(after));
        previous.frequencies = Some(table());
        current.frequencies = Some(table());
        (previous, current)
    }

    fn analyze_pair(before: IoReportSample, after: IoReportSample) -> Report {
        let (previous, current) = pair(before, after);
        let mut report = Report::default();
        fill(&previous, &current, 2_000_000_000, &mut report);
        report
    }

    fn close(actual: Option<f64>, expected: f64) -> bool {
        actual.is_some_and(|actual| (actual - expected).abs() < 1e-9)
    }

    #[test]
    fn residency_frequency_and_power() {
        // Window: OFF 50, P1 10, P2 30, P3 10 ticks; 20 J over 2 s.
        let report = analyze_pair(
            sample(
                1_000_000_000,
                [100, 0, 0, 0],
                5_000_000_000,
                [100, 0, 0],
                [0, 100],
                0,
            ),
            sample(
                3_000_000_000,
                [150, 10, 30, 10],
                25_000_000_000,
                [150, 20, 30],
                [10, 190],
                0,
            ),
        );
        let device = &report.device;
        assert!(close(device.active_pct, 50.0));
        // (10 * 400 + 30 * 1200 + 10 * 1000) / 50
        assert!(close(device.avg_mhz, 1000.0));
        assert!(close(device.max_mhz, 1200.0));
        assert!(close(device.freq_normalized_pct, 50.0 * 1000.0 / 1200.0));
        assert!(close(device.power_w, 10.0));
        assert_eq!(device.pstates.len(), 4);
        assert_eq!(device.pstates[0].name, "OFF");
        assert_eq!(device.pstates[0].mhz, None);
        assert!((device.pstates[0].pct - 50.0).abs() < 1e-9);
        assert_eq!(device.pstates[2].mhz, Some(1200.0));
        assert!((device.pstates[2].pct - 30.0).abs() < 1e-9);

        // Limiters exclude IDLE_OFF and are sorted: PWR_ZONE 30 of 50 powered ticks.
        assert_eq!(device.limiters.len(), 2);
        assert_eq!(device.limiters[0].name, "PWR_ZONE");
        assert!((device.limiters[0].pct - 60.0).abs() < 1e-9);
        assert_eq!(device.limiters[1].name, "PERF");
        assert!((device.limiters[1].pct - 40.0).abs() < 1e-9);

        // CLTM: 10 capped of 100 ticks.
        assert!(close(device.thermal_limited_pct, 10.0));
        assert_eq!(device.counters.get("Issue Counts/Total Restarts"), Some(&0));
        assert!(!device.counters.keys().any(|key| key.contains("Percent")));
        assert_eq!(report.sources[source::IOREPORT], SourceStatus::ok());
        assert!(report.events.is_empty());
    }

    #[test]
    fn idle_window_reports_zero_normalized_and_no_clock() {
        let report = analyze_pair(
            sample(0, [100, 0, 0, 0], 1_000, [100, 0, 0], [0, 100], 0),
            sample(
                1_000_000_000,
                [200, 0, 0, 0],
                1_000,
                [200, 0, 0],
                [0, 200],
                0,
            ),
        );
        let device = &report.device;
        assert!(close(device.active_pct, 0.0));
        assert_eq!(device.avg_mhz, None);
        assert!(close(device.freq_normalized_pct, 0.0));
        assert!(close(device.power_w, 0.0));
        assert!(device.limiters.is_empty());
        assert_eq!(report.sources[source::IOREPORT].state, SourceState::Ok);
    }

    #[test]
    fn frozen_energy_is_stale_not_zero() {
        let report = analyze_pair(
            sample(0, [100, 0, 0, 0], 7_000, [100, 0, 0], [0, 100], 0),
            sample(
                1_000_000_000,
                [100, 0, 0, 100],
                7_000,
                [100, 100, 0],
                [0, 200],
                0,
            ),
        );
        assert!(close(report.device.active_pct, 100.0));
        assert_eq!(report.device.power_w, None);
        let status = &report.sources[source::IOREPORT];
        assert_eq!(status.state, SourceState::Stale);
        assert!(
            status
                .detail
                .as_deref()
                .unwrap_or("")
                .contains("GPU Energy")
        );
    }

    #[test]
    fn counter_reset_drops_the_metric() {
        let report = analyze_pair(
            sample(0, [100, 50, 0, 0], 9_000, [100, 0, 0], [0, 100], 0),
            sample(
                1_000_000_000,
                [10, 5, 0, 0],
                4_000,
                [200, 0, 0],
                [0, 200],
                0,
            ),
        );
        assert_eq!(report.device.active_pct, None);
        assert_eq!(report.device.power_w, None);
        assert_eq!(report.sources[source::IOREPORT].state, SourceState::Partial);
    }

    #[test]
    fn mismatched_table_hides_mhz() {
        let (previous, mut current) = pair(
            sample(0, [100, 0, 0, 0], 0, [100, 0, 0], [0, 100], 0),
            sample(
                1_000_000_000,
                [100, 50, 50, 0],
                1_000,
                [100, 100, 0],
                [0, 200],
                0,
            ),
        );
        current.frequencies = Some(FrequencyTable {
            mhz: vec![400.0, 1200.0],
            source: "test".into(),
        });
        let mut report = Report::default();
        fill(&previous, &current, 1_000_000_000, &mut report);
        assert!(close(report.device.active_pct, 100.0));
        assert_eq!(report.device.avg_mhz, None);
        assert_eq!(report.device.max_mhz, None);
        assert!(
            report
                .device
                .pstates
                .iter()
                .all(|state| state.mhz.is_none())
        );
        assert_eq!(report.sources[source::IOREPORT].state, SourceState::Partial);

        current.frequencies = None;
        let mut report = Report::default();
        fill(&previous, &current, 1_000_000_000, &mut report);
        assert_eq!(report.device.avg_mhz, None);
        assert_eq!(report.sources[source::IOREPORT].state, SourceState::Partial);
    }

    #[test]
    fn sums_channels_from_several_dies() {
        let mut before = sample(0, [100, 0, 0, 0], 0, [0, 0, 0], [0, 0], 0);
        let mut after = sample(
            1_000_000_000,
            [150, 50, 0, 0],
            2_000_000_000,
            [0, 0, 0],
            [0, 0],
            0,
        );
        let die = |sample: &mut IoReportSample, off: u64, p3: u64, nj: u64| {
            let mut gpuph = state_channel(
                GPUPH.0,
                GPUPH.1,
                &[("OFF", off), ("P1", 0), ("P2", 0), ("P3", p3)],
            );
            gpuph.driver_id = 8;
            let mut energy = energy(nj);
            energy.driver_id = 8;
            sample.channels.push(gpuph);
            sample.channels.push(energy);
        };
        die(&mut before, 0, 0, 0);
        die(&mut after, 0, 100, 1_000_000_000);
        let report = analyze_pair(before, after);
        // 150 active of 200 ticks, at (50 * 400 + 100 * 1000) / 150 MHz.
        assert!(close(report.device.active_pct, 75.0));
        assert!(close(report.device.avg_mhz, 800.0));
        assert!(close(report.device.power_w, 3.0));
    }

    #[test]
    fn issue_counters_emit_event() {
        let report = analyze_pair(
            sample(0, [100, 0, 0, 0], 0, [100, 0, 0], [0, 100], 1),
            sample(1_000_000_000, [200, 0, 0, 0], 0, [200, 0, 0], [0, 200], 3),
        );
        assert_eq!(
            report.device.counters.get("Issue Counts/Total Restarts"),
            Some(&2)
        );
        assert_eq!(
            report
                .device
                .counters
                .get("GPU Throttler Counters/Throttle Counter Total"),
            Some(&14)
        );
        assert_eq!(report.events.len(), 1);
        assert_eq!(report.events[0].kind, EventKind::GpuIssue);
        assert!(report.events[0].message.contains("Total Restarts +2"));
    }

    #[test]
    fn sentinel_counter_is_skipped() {
        let before = sample(0, [100, 0, 0, 0], 0, [100, 0, 0], [0, 100], 1);
        let mut after = sample(1_000_000_000, [200, 0, 0, 0], 0, [200, 0, 0], [0, 200], 3);
        for channel in &mut after.channels {
            if channel.name == "Total Restarts" {
                channel.value = ChannelValue::Simple(None);
            }
        }
        let report = analyze_pair(before, after);
        assert!(
            !report
                .device
                .counters
                .contains_key("Issue Counts/Total Restarts")
        );
        assert!(report.events.is_empty());
    }

    #[test]
    fn source_states_for_missing_and_disabled() {
        let base = Instant::now();
        let previous = snapshot(base, Duration::ZERO, 0, vec![]);
        let mut current = snapshot(base, Duration::from_secs(1), 0, vec![]);
        let mut report = Report::default();
        fill(&previous, &current, 1_000_000_000, &mut report);
        for name in [source::IOREPORT, source::TEMPERATURE, source::SYSTEM] {
            assert_eq!(report.sources[name].state, SourceState::Disabled);
        }

        current.ioreport = Some(Err("no IOReport".into()));
        current.temperatures = Some(Err("no SMC".into()));
        let mut report = Report::default();
        fill(&previous, &current, 1_000_000_000, &mut report);
        assert_eq!(
            report.sources[source::IOREPORT].state,
            SourceState::Unavailable
        );
        assert_eq!(
            report.sources[source::TEMPERATURE].state,
            SourceState::Unavailable
        );

        current.ioreport = Some(Ok(IoReportSample::default()));
        let mut report = Report::default();
        fill(&previous, &current, 1_000_000_000, &mut report);
        assert_eq!(report.sources[source::IOREPORT].state, SourceState::Partial);
        assert_eq!(report.device.active_pct, None);
    }

    #[test]
    fn temperature_summary_and_partial() {
        let mut device = DeviceView::default();
        let temperatures = GpuTemperatures {
            sensors: vec![
                SensorReading {
                    key: "Tg04".into(),
                    celsius: 60.0,
                },
                SensorReading {
                    key: "Tg05".into(),
                    celsius: 70.0,
                },
            ],
        };
        assert_eq!(
            fill_temperature(&temperatures, &mut device),
            SourceStatus::ok()
        );
        assert_eq!(
            device.temperature,
            Some(Temperature {
                avg_c: 65.0,
                max_c: 70.0,
                sensors: 2
            })
        );
        let mut broken = temperatures.clone();
        broken.sensors[0].celsius = -1.0;
        assert_eq!(
            fill_temperature(&broken, &mut device).state,
            SourceState::Partial
        );
        broken.sensors[1].celsius = 400.0;
        assert_eq!(
            fill_temperature(&broken, &mut device).state,
            SourceState::Unavailable
        );
    }

    #[test]
    fn idle_sensors_that_stop_reporting_leave_temperature_out() {
        // Idle GPU sensors can read placeholders of 1.6-1.8 C, below the
        // 2 C minimum for a plausible die temperature.
        let mut device = DeviceView::default();
        let temperatures = GpuTemperatures {
            sensors: [-4.0, 1.6, 0.0, 1.8]
                .iter()
                .cycle()
                .take(44)
                .enumerate()
                .map(|(index, celsius)| SensorReading {
                    key: format!("Tg{index:02}"),
                    celsius: *celsius,
                })
                .collect(),
        };
        let status = fill_temperature(&temperatures, &mut device);
        assert_eq!(status.state, SourceState::Stale);
        assert!(
            status
                .detail
                .as_deref()
                .unwrap_or_default()
                .contains("not reporting"),
            "{status:?}"
        );
        assert_eq!(device.temperature, None);
    }

    #[test]
    fn system_context_and_thermal_event() {
        let base = Instant::now();
        let mut previous = snapshot(base, Duration::ZERO, 0, vec![]);
        let mut current = snapshot(base, Duration::from_secs(1), 0, vec![]);
        let context = SystemContext {
            thermal_pressure_level: Some(0),
            memory_pressure_level: Some(2),
            wired_limit_mb: Some(0),
            total_physical_bytes: Some(48 << 30),
        };
        previous.system = Some(context.clone());
        current.system = Some(SystemContext {
            thermal_pressure_level: Some(2),
            ..context
        });
        let mut report = Report::default();
        fill(&previous, &current, 1_000_000_000, &mut report);
        let device = &report.device;
        assert_eq!(device.system.thermal_pressure.as_deref(), Some("heavy"));
        assert_eq!(device.system.memory_pressure.as_deref(), Some("warning"));
        assert_eq!(device.memory.total_physical_bytes, Some(48 << 30));
        assert_eq!(device.memory.wired_limit_bytes, None);
        assert_eq!(report.sources[source::SYSTEM].state, SourceState::Ok);
        assert_eq!(report.events.len(), 1);
        assert_eq!(report.events[0].kind, EventKind::ThermalPressure);
        assert!(report.events[0].message.contains("nominal to heavy"));

        current.system = Some(SystemContext {
            wired_limit_mb: Some(1024),
            thermal_pressure_level: None,
            ..SystemContext::default()
        });
        let mut report = Report::default();
        fill(&previous, &current, 1_000_000_000, &mut report);
        assert_eq!(report.device.memory.wired_limit_bytes, Some(1 << 30));
        assert_eq!(report.sources[source::SYSTEM].state, SourceState::Partial);
        assert!(report.events.is_empty());
    }
}
