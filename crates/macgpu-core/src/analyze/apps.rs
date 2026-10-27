//! Per-app view from resource coalitions, per-process graphics memory, and
//! estimated per-process GPU power.
//!
//! Accuracy rules:
//! - An app's GPU time is apportioned like the process rows: its members'
//!   `AppUsage` time plus its unattributed (exited) time when known, so app
//!   rows add up to process rows and shares sum to 100.
//! - The coalition counter is in-flight time (see [`AppView`]): `AppUsage`
//!   apportions GPU time among overlapping contexts, the coalition counter
//!   does not. Measured on an M4 Max: with no overlapping work the two agree
//!   within ±15 ms per window; with a second app busy on the GPU, an app's
//!   coalition time was 1.3-3.4x its `AppUsage` time. It is reported as
//!   `in_flight_gpu_ns`, includes tasks that exited inside the window, and
//!   never goes negative. A coalition created inside the window is credited
//!   with its whole counter.
//! - Unattributed (exited) time is the device's GPU busy time that live
//!   clients' `AppUsage` does not account for, split among the apps that can
//!   hold it, see [`apportion_exited`]. An overlap ratio measured on one app
//!   does not carry over to another (a small app that ran during a load reads
//!   1.75x in flight while the load itself reads about 1.05x), so the
//!   in-flight counter only bounds and weights each app's share. Without
//!   IOReport the overlap-ratio estimate in [`unattributed`] is the fallback.
//! - Coalition GPU energy needs no apportioning for overlap: with two loads
//!   and the desktop overlapping (about 2300 ms/s of in-flight time summed
//!   over apps), the apps' energy summed to IOReport's GPU energy within 0.1%
//!   over the run, and within 1-12% per steady 1 s window. But it trails GPU
//!   time by about a second, so single windows during ramps were off by up to
//!   a window's worth (31.8 W for one app against 17.2 W device). When
//!   IOReport's GPU power is known, the apps' power is therefore scaled to sum
//!   to it, and the coalition energies decide only the split.
//! - The kernel credits coalition GPU energy in steps about 0.5 s apart, for
//!   every coalition at once (M4 Max, macOS 27: 60 steps in 30 s, 0.49-0.51 s
//!   apart), so most windows shorter than that see no energy for any app.
//!   Splitting each window by its own energy lost that window's device energy:
//!   at 0.25 s windows the apps summed to 49% of IOReport's energy, and
//!   `macgpu run` at 0.1 s reported a fifth of a load's energy. The split
//!   therefore uses the coalition energy of a trailing span of at least
//!   [`ENERGY_SPAN_NS`] (see [`EnergyHistory`]), which always contains a step;
//!   a window that long is split by its own energy alone. Before the first
//!   step is seen the device energy is split by apportioned GPU time.
//! - A process's GPU power is an estimate: the app's measured GPU power split
//!   by the process's share of the app's apportioned time (exited tasks keep
//!   theirs when known).

use super::{ms_per_s, pct};
use crate::snapshot::Snapshot;
use crate::sys::coalition::{Coalition, CoalitionTable, CoalitionUsage};
use crate::sys::process::{ProcessInfo, bundle_name};
use crate::sys::task_memory::{GraphicsMemory, MemoryTable};
use macgpu_model::{AppView, ProcessView, Report, SourceState, SourceStatus, Unattributed, source};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

/// Largest difference between coalition and `AppUsage` time seen in windows
/// without exits or overlap: the two counters are credited at slightly
/// different moments. A smaller gap is not reported as exited time.
pub const SKEW_TOLERANCE_NS: u64 = 15_000_000;

/// Without the device's busy time, the overlap-ratio estimate of unattributed
/// time is reported only when it can be off by at most this fraction of it
/// (or [`SKEW_TOLERANCE_NS`]).
const OVERLAP_UNCERTAINTY: f64 = 0.25;

/// Marks synthetic app keys used when coalitions are unavailable.
const SYNTHETIC_ID_BIT: u64 = 1 << 63;

/// Shortest trailing span whose coalition GPU energy splits a window's energy
/// among apps. Coalition energy moves in steps about 0.5 s apart, so a span
/// this long always contains one. It is a little under a second so a 1 s
/// window, whose length jitters by a few ms, is split by its own energy alone.
pub const ENERGY_SPAN_NS: u64 = 900_000_000;

/// Coalition GPU energy of the most recent consecutive windows, carried from
/// one window to the next by [`super::Analyzer`]: as few windows, ending with
/// the current one, as cover [`ENERGY_SPAN_NS`].
#[derive(Debug, Clone, Default)]
pub struct EnergyHistory {
    /// Oldest first: each window's length and its coalitions' nonzero GPU
    /// energy, in nanojoules.
    windows: VecDeque<(u64, Vec<(u64, u64)>)>,
}

impl EnergyHistory {
    pub fn clear(&mut self) {
        self.windows.clear();
    }

    /// Adds the current window and drops older windows the span no longer needs.
    fn push(&mut self, window_ns: u64, energy_nj: Vec<(u64, u64)>) {
        self.windows.push_back((window_ns, energy_nj));
        let mut span_ns = self.span_ns();
        while self.windows.len() > 1 {
            let oldest = self.windows[0].0;
            if span_ns.saturating_sub(oldest) < ENERGY_SPAN_NS {
                break;
            }
            span_ns -= oldest;
            self.windows.pop_front();
        }
    }

    fn span_ns(&self) -> u64 {
        self.windows.iter().fold(0u64, |total, (window_ns, _)| {
            total.saturating_add(*window_ns)
        })
    }

    /// Each coalition's GPU energy over the span.
    fn energy_nj(&self) -> HashMap<u64, u64> {
        let mut totals: HashMap<u64, u64> = HashMap::new();
        for (id, nj) in self.windows.iter().flat_map(|(_, energy)| energy) {
            let total = totals.entry(*id).or_default();
            *total = total.saturating_add(*nj);
        }
        totals
    }
}

pub(super) fn fill(
    previous: &Snapshot,
    current: &Snapshot,
    window_ns: u64,
    include_idle: bool,
    history: &mut EnergyHistory,
    report: &mut Report,
) {
    fill_memory(current.memory.as_ref(), report);

    match &current.coalitions {
        Some(Ok(table)) => {
            let status = if coalitions(previous).is_some() {
                SourceStatus::ok()
            } else {
                SourceStatus::with(SourceState::Partial, "previous sample failed; no deltas")
            };
            report
                .sources
                .insert(source::COALITIONS.to_string(), status);
            from_coalitions(
                previous,
                current,
                table,
                window_ns,
                include_idle,
                history,
                report,
            );
        }
        Some(Err(error)) => {
            history.clear();
            report.sources.insert(
                source::COALITIONS.to_string(),
                SourceStatus::with(
                    SourceState::Unavailable,
                    format!("{error}; apps grouped by bundle"),
                ),
            );
            from_bundles(window_ns, include_idle, report);
        }
        None => {
            history.clear();
            report.sources.insert(
                source::COALITIONS.to_string(),
                SourceStatus::with(SourceState::Disabled, "disabled; apps grouped by bundle"),
            );
            from_bundles(window_ns, include_idle, report);
        }
    }

    report.apps.sort_by(|a, b| {
        b.gpu_ns
            .cmp(&a.gpu_ns)
            .then_with(|| b.attributed_gpu_ns.cmp(&a.attributed_gpu_ns))
            .then_with(|| b.lifetime_gpu_ns.cmp(&a.lifetime_gpu_ns))
            .then_with(|| a.coalition_id.cmp(&b.coalition_id))
    });
}

/// Sets each process's graphics memory and the memory source status.
fn fill_memory(memory: Option<&MemoryTable>, report: &mut Report) {
    let Some(memory) = memory else {
        report.sources.insert(
            source::MEMORY.to_string(),
            SourceStatus::with(SourceState::Disabled, "disabled"),
        );
        return;
    };
    let (mut readable, mut denied) = (0, 0);
    for process in &mut report.processes {
        process.graphics_memory_bytes = memory.charged_bytes(process.pid);
        match memory.get(process.pid) {
            Some(GraphicsMemory::Bytes { .. }) => readable += 1,
            Some(GraphicsMemory::Denied) => denied += 1,
            Some(GraphicsMemory::Unavailable) | None => {}
        }
    }
    let detail = || {
        format!(
            "{denied} of {} GPU processes belong to other users; reading them needs root",
            readable + denied
        )
    };
    let status = match (readable, denied) {
        (_, 0) => SourceStatus::ok(),
        (0, _) => SourceStatus::with(SourceState::Denied, detail()),
        _ => SourceStatus::with(SourceState::Partial, detail()),
    };
    report.sources.insert(source::MEMORY.to_string(), status);
}

/// Live-client facts about one app's member processes.
#[derive(Default)]
struct Members {
    /// Indexes into `report.processes`.
    processes: Vec<usize>,
    attributed_ns: u64,
    /// Clients that used the GPU in the window.
    active_clients: usize,
    /// A member's GPU client from the previous sample is gone.
    client_vanished: bool,
}

/// An app chosen for the report, with its apportioned time.
struct Included<'a> {
    coalition: &'a Coalition,
    delta: CoalitionUsage,
    app: &'a Members,
    /// In-flight time beyond the live members' `AppUsage` that may be exited
    /// work, or `None` when the app cannot hold exited time.
    exited_gap_ns: Option<u64>,
    /// The overlap-ratio estimate, used when the device's busy time is unknown.
    overlap_estimate_ns: Option<u64>,
    unattributed_ns: Option<u64>,
    /// `app.attributed_ns` plus `unattributed_ns` when known.
    gpu_ns: u64,
}

/// The snapshot's coalition table, when it was read.
fn coalitions(snapshot: &Snapshot) -> Option<&CoalitionTable> {
    match &snapshot.coalitions {
        Some(Ok(table)) => Some(table),
        _ => None,
    }
}

/// App rows from `table`, the current coalitions, with window deltas from the
/// previous sample's coalitions when it has them.
fn from_coalitions(
    previous: &Snapshot,
    current: &Snapshot,
    table: &CoalitionTable,
    window_ns: u64,
    include_idle: bool,
    history: &mut EnergyHistory,
    report: &mut Report,
) {
    let baseline = coalitions(previous);
    let mut members: HashMap<u64, Members> = HashMap::new();
    for (index, process) in report.processes.iter_mut().enumerate() {
        process.coalition_id = table.coalition_of(process.pid);
        let Some(id) = process.coalition_id else {
            continue;
        };
        let app = members.entry(id).or_default();
        app.processes.push(index);
        app.attributed_ns = app.attributed_ns.saturating_add(process.gpu_ns);
        app.active_clients += process
            .clients
            .iter()
            .filter(|client| client.gpu_ns > 0)
            .count();
    }
    if let (Some(baseline), Ok(previous_registry), Ok(current_registry)) =
        (baseline, &previous.registry, &current.registry)
    {
        let live: HashSet<u64> = current_registry
            .clients()
            .map(|client| client.registry_id)
            .collect();
        for client in previous_registry.clients() {
            if !live.contains(&client.registry_id)
                && let Some(id) = baseline
                    .coalition_of(client.pid)
                    .or_else(|| table.coalition_of(client.pid))
            {
                members.entry(id).or_default().client_vanished = true;
            }
        }
    }

    // Window deltas. Without a baseline there are none.
    let deltas: HashMap<u64, CoalitionUsage> = table
        .iter()
        .map(|coalition| {
            let delta = match baseline {
                None => CoalitionUsage::default(),
                Some(baseline) => match baseline.get(coalition.id) {
                    Some(old) => coalition.usage.since(&old.usage),
                    None => coalition.usage,
                },
            };
            (coalition.id, delta)
        })
        .collect();

    // The energy that splits this window: the trailing span's.
    let energy = table.has_gpu_energy && baseline.is_some();
    if energy {
        let moved = deltas
            .iter()
            .filter(|(_, delta)| delta.gpu_energy_nj > 0)
            .map(|(id, delta)| (*id, delta.gpu_energy_nj))
            .collect();
        history.push(window_ns, moved);
    } else {
        history.clear();
    }
    let span_energy_nj = history.energy_nj();
    let span_ns = history.span_ns();
    let span_energy = |id: u64| span_energy_nj.get(&id).copied().unwrap_or(0);

    let empty = Members::default();
    let evidence = |id: u64| {
        let app = members.get(&id).unwrap_or(&empty);
        deltas.get(&id).is_some_and(|delta| delta.tasks_exited > 0) || app.client_vanished
    };
    // Overlap measured on apps whose gap cannot contain exited work, for the
    // fallback estimate.
    let (mut reference_coalition, mut reference_usage) = (0u64, 0u64);
    for (id, app) in &members {
        if app.attributed_ns > 0 && !evidence(*id) {
            reference_coalition += deltas.get(id).map(|delta| delta.gpu_ns).unwrap_or(0);
            reference_usage += app.attributed_ns;
        }
    }
    let overlap = (reference_usage > 0)
        .then(|| (reference_coalition as f64 / reference_usage as f64).max(1.0));
    let device_active_clients: usize = members.values().map(|app| app.active_clients).sum();

    // Pass 1: which apps to show, and their apportioned time.
    let mut included = Vec::new();
    for coalition in table.iter() {
        let delta = deltas.get(&coalition.id).copied().unwrap_or_default();
        let app = members.get(&coalition.id).unwrap_or(&empty);
        let active = delta.gpu_ns > 0
            || delta.gpu_energy_nj > 0
            || delta.gpu_energy_billed_to_me_nj > 0
            || span_energy(coalition.id) > 0
            || app.attributed_ns > 0;
        if !active && !(include_idle && !app.processes.is_empty()) {
            continue;
        }
        // Only apps with an exited task or vanished client, or with no live
        // client at all, can hold exited time; any other gap is overlap.
        let gap = delta.gpu_ns.saturating_sub(app.attributed_ns);
        let exited_gap_ns =
            (baseline.is_some() && gap > 0 && (app.attributed_ns == 0 || evidence(coalition.id)))
                .then_some(gap);
        let overlap_estimate_ns = baseline.and_then(|_| {
            // Without a reference, any device-wide overlap is possible.
            let (estimate, bound) = match overlap {
                Some(overlap) => (overlap, overlap),
                None => (1.0, device_active_clients.max(1) as f64),
            };
            unattributed(
                delta.gpu_ns,
                app.attributed_ns,
                evidence(coalition.id),
                estimate,
                bound.max(app.active_clients as f64),
            )
        });
        included.push(Included {
            coalition,
            delta,
            app,
            exited_gap_ns,
            overlap_estimate_ns,
            unattributed_ns: baseline.map(|_| 0),
            gpu_ns: app.attributed_ns,
        });
    }
    let active_ns = report
        .device
        .active_pct
        .map(|active| (active / 100.0 * window_ns as f64).clamp(0.0, window_ns as f64) as u64);
    let live_ns = report
        .processes
        .iter()
        .fold(0u64, |total, process| total.saturating_add(process.gpu_ns));
    match active_ns {
        Some(active_ns) => apportion_exited(&mut included, active_ns, live_ns),
        None => {
            for entry in &mut included {
                if entry.exited_gap_ns.is_some() {
                    entry.unattributed_ns = entry.overlap_estimate_ns;
                    entry.gpu_ns = entry
                        .app
                        .attributed_ns
                        .saturating_add(entry.unattributed_ns.unwrap_or(0));
                }
            }
        }
    }
    cap_unattributed(&mut included, window_ns);
    let total_ns = included
        .iter()
        .fold(0u64, |total, entry| total.saturating_add(entry.gpu_ns));

    // Pass 2: the views.
    let window_seconds = window_ns as f64 / 1e9;
    let watts = |nj: u64| nj as f64 / 1e9 / window_seconds;
    let ane = table.has_ane && baseline.is_some();
    // Coalition energy over the trailing span decides the split; IOReport's GPU
    // energy, when known, the total (coalition energy trails it by about a
    // second). Before any coalition energy is seen, apportioned GPU time splits
    // it. Without IOReport, the span's coalition energy at its average rate.
    let span_total_nj = included.iter().fold(0u64, |total, entry| {
        total.saturating_add(span_energy(entry.coalition.id))
    });
    let device_nj = report
        .device
        .power_w
        .map(|watts| watts * window_seconds * 1e9);
    let energy_nj_of = |id: u64, gpu_ns: u64| match device_nj {
        Some(device_nj) if span_total_nj > 0 => {
            device_nj * span_energy(id) as f64 / span_total_nj as f64
        }
        Some(device_nj) if total_ns > 0 => device_nj * gpu_ns as f64 / total_ns as f64,
        Some(_) => 0.0,
        None if span_ns > 0 => span_energy(id) as f64 * window_ns as f64 / span_ns as f64,
        None => 0.0,
    };
    let mut unattributed_total = 0u64;
    let mut partial = false;
    let mut estimates: Vec<(usize, f64)> = Vec::new();
    for Included {
        coalition,
        delta,
        app,
        unattributed_ns,
        gpu_ns,
        ..
    } in included
    {
        match unattributed_ns {
            Some(exited) => unattributed_total = unattributed_total.saturating_add(exited),
            None => partial = baseline.is_some(),
        }
        let energy_nj = energy_nj_of(coalition.id, gpu_ns);
        let power_w = energy.then(|| energy_nj / 1e9 / window_seconds);
        if let Some(power_w) = power_w {
            // Split by apportioned time; exited tasks keep their known share.
            for index in &app.processes {
                let share = if gpu_ns == 0 {
                    0.0
                } else {
                    report.processes[*index].gpu_ns as f64 / gpu_ns as f64
                };
                estimates.push((*index, power_w * share));
            }
        }
        let gpu_pids: Vec<i32> = app
            .processes
            .iter()
            .map(|index| report.processes[*index].pid)
            .collect();
        let graphics_memory_bytes = app
            .processes
            .iter()
            .filter_map(|index| report.processes[*index].graphics_memory_bytes)
            .reduce(u64::saturating_add);
        let leader = coalition.leader.as_ref().or_else(|| {
            gpu_pids
                .iter()
                .min()
                .and_then(|pid| current.processes.get(pid))
        });
        let (name, bundle_path) = app_name(coalition, leader);
        let share_pct = pct(gpu_ns, total_ns);

        report.apps.push(AppView {
            coalition_id: coalition.id,
            name,
            bundle_path,
            responsible_pid: coalition.responsible_pid,
            gpu_ns,
            busy_ms_per_s: ms_per_s(gpu_ns, window_ns),
            share_pct,
            device_pct: report
                .device
                .active_pct
                .map(|active| share_pct * active / 100.0),
            in_flight_gpu_ns: baseline.map(|_| delta.gpu_ns),
            in_flight_ms_per_s: baseline.map(|_| ms_per_s(delta.gpu_ns, window_ns)),
            attributed_gpu_ns: app.attributed_ns,
            unattributed_gpu_ns: unattributed_ns,
            energy_j: energy.then(|| energy_nj / 1e9),
            power_w,
            billed_to_me_w: energy.then(|| watts(delta.gpu_energy_billed_to_me_nj)),
            billed_to_others_w: energy.then(|| watts(delta.gpu_energy_billed_to_others_nj)),
            ane_ms_per_s: ane.then(|| ms_per_s(delta.ane_ns, window_ns)),
            ane_power_w: ane.then(|| watts(delta.ane_energy_nj)),
            lifetime_gpu_ns: coalition.usage.gpu_ns,
            lifetime_energy_j: table
                .has_gpu_energy
                .then(|| coalition.usage.gpu_energy_nj as f64 / 1e9),
            graphics_memory_bytes,
            members: coalition.members.len(),
            gpu_pids,
        });
    }

    for (index, power_w) in estimates {
        report.processes[index].est_power_w = Some(power_w);
    }
    report.unattributed = baseline.map(|_| Unattributed {
        gpu_ns: unattributed_total,
        busy_ms_per_s: ms_per_s(unattributed_total, window_ns),
        partial,
    });
}

/// Splits exited GPU time among the apps that can hold it.
///
/// Live clients' `AppUsage` tracks the device's busy time: in windows without
/// exits it fell short of IOReport's active time by 0-77 ms/s on an M4 Max,
/// typically 8-38 ms/s (84-100% of it). So the busy time live
/// clients do not account for, `active - live`, is the exited work. It goes to
/// the apps with a gap that may hold exited work (see [`Included`]), in
/// proportion to the gaps and never more than an app's own gap: exited work
/// apportioned among overlapping contexts is never more than its time in
/// flight, and the coalition counter sums its tasks' in-flight time.
///
/// Without the device's active time (IOReport unavailable) the caller falls
/// back to [`unattributed`]. Shares within [`SKEW_TOLERANCE_NS`] for apps with
/// live clients are counter skew and count as 0.
fn apportion_exited(entries: &mut [Included], active_ns: u64, live_ns: u64) {
    let gaps = entries
        .iter()
        .filter_map(|entry| entry.exited_gap_ns)
        .fold(0u64, u64::saturating_add);
    if gaps == 0 {
        return;
    }
    let scale = (active_ns.saturating_sub(live_ns) as f64 / gaps as f64).min(1.0);
    for entry in entries.iter_mut() {
        let Some(gap) = entry.exited_gap_ns else {
            continue;
        };
        let share = (gap as f64 * scale) as u64;
        let skew = entry.app.attributed_ns > 0 && share <= SKEW_TOLERANCE_NS;
        entry.unattributed_ns = Some(if skew { 0 } else { share });
        entry.gpu_ns = entry
            .app
            .attributed_ns
            .saturating_add(entry.unattributed_ns.unwrap_or(0));
    }
}

/// Scales known unattributed time down so the apportioned total fits the
/// window: the GPU cannot be busy longer than the window, however much work
/// overlapped. The per-app estimates assume the exited work overlapped like the
/// reference apps' work did; when several apps' exited tasks overlapped each
/// other more than that (two loads ending together, say), the estimates sum to
/// more than the free GPU time. The free time is `window - attributed` (at
/// least 0; `AppUsage` can slightly exceed the window when saturated), and it
/// is shared in proportion to the estimates.
fn cap_unattributed(entries: &mut [Included], window_ns: u64) {
    let (mut attributed, mut unattributed) = (0u64, 0u64);
    for entry in entries.iter() {
        attributed = attributed.saturating_add(entry.app.attributed_ns);
        unattributed = unattributed.saturating_add(entry.unattributed_ns.unwrap_or(0));
    }
    let free = window_ns.saturating_sub(attributed);
    if unattributed <= free {
        return;
    }
    let scale = free as f64 / unattributed as f64;
    for entry in entries.iter_mut() {
        if let Some(exited) = &mut entry.unattributed_ns {
            *exited = (*exited as f64 * scale) as u64;
            entry.gpu_ns = entry.app.attributed_ns.saturating_add(*exited);
        }
    }
}

/// GPU time of tasks without a live client, in `AppUsage` (apportioned)
/// units, or `None` when it cannot be separated from overlap. This is the
/// fallback for when the device's busy time is unknown (no IOReport); see
/// [`apportion_exited`] for why an overlap ratio measured on other apps can
/// under-count by about 2x.
///
/// Every live nanosecond of `AppUsage` time counts as 1 or more nanoseconds of
/// in-flight (coalition) time, depending on how much other work ran at the
/// same time. `overlap` is that factor measured in the same window on apps
/// without exits, and it is assumed to apply to the exited work as well. So:
///
/// - `attributed_ns == 0`: every in-flight nanosecond came from tasks without
///   a live client, so `in_flight / overlap`.
/// - No exit evidence (no member task exited, no member client vanished): 0;
///   any gap is overlap or counter skew.
/// - Otherwise `(in_flight - overlap * attributed) / overlap`. The app's own
///   live work may have overlapped by anywhere from 1 to `overlap_bound` (more
///   than `overlap` when several of its clients ran at once), which moves the
///   result by up to `(overlap_bound - 1) * attributed / overlap`; the
///   estimate is reported only when that is at most [`OVERLAP_UNCERTAINTY`]
///   of it. Estimates within [`SKEW_TOLERANCE_NS`] are 0.
fn unattributed(
    in_flight_ns: u64,
    attributed_ns: u64,
    exit_evidence: bool,
    overlap: f64,
    overlap_bound: f64,
) -> Option<u64> {
    let overlap = overlap.max(1.0);
    if attributed_ns == 0 {
        return Some((in_flight_ns as f64 / overlap) as u64);
    }
    if !exit_evidence {
        return Some(0);
    }
    let attributed = attributed_ns as f64;
    let estimate = ((in_flight_ns as f64 - overlap * attributed) / overlap).max(0.0);
    if estimate <= SKEW_TOLERANCE_NS as f64 {
        return Some(0);
    }
    let spread = (overlap_bound.max(overlap) - 1.0) * attributed / overlap;
    (spread <= (OVERLAP_UNCERTAINTY * estimate).max(SKEW_TOLERANCE_NS as f64))
        .then_some(estimate as u64)
}

/// The app's display name and bundle: the leader's `.app` bundle name, else
/// its executable name.
fn app_name(coalition: &Coalition, leader: Option<&ProcessInfo>) -> (String, Option<String>) {
    match leader {
        Some(leader) => match leader.bundle_path() {
            Some(bundle) => (bundle_name(bundle).to_string(), Some(bundle.to_string())),
            None => (leader.name.clone(), None),
        },
        None => (format!("coalition {}", coalition.id), None),
    }
}

/// Groups processes by `.app` bundle when coalitions are unavailable, so
/// helpers still roll up into their app. Energy and exited time are unknown.
fn from_bundles(window_ns: u64, include_idle: bool, report: &mut Report) {
    let mut groups: BTreeMap<String, Vec<&ProcessView>> = BTreeMap::new();
    for process in &report.processes {
        let key = match process
            .path
            .as_deref()
            .and_then(crate::sys::process::bundle_path)
        {
            Some(bundle) => bundle.to_string(),
            None => format!("pid {} {}", process.pid, process.start_us),
        };
        groups.entry(key).or_default().push(process);
    }
    let total_ns = report
        .processes
        .iter()
        .fold(0u64, |total, process| total.saturating_add(process.gpu_ns));

    for (key, processes) in groups {
        let gpu_ns = processes
            .iter()
            .fold(0u64, |total, process| total.saturating_add(process.gpu_ns));
        if gpu_ns == 0 && !include_idle {
            continue;
        }
        let bundle = processes[0]
            .path
            .as_deref()
            .and_then(crate::sys::process::bundle_path);
        let name = match bundle {
            Some(bundle) => bundle_name(bundle).to_string(),
            None => processes[0].name.clone(),
        };
        report.apps.push(AppView {
            coalition_id: synthetic_id(&key),
            name,
            bundle_path: bundle.map(str::to_string),
            responsible_pid: None,
            gpu_ns,
            busy_ms_per_s: ms_per_s(gpu_ns, window_ns),
            share_pct: pct(gpu_ns, total_ns),
            device_pct: report
                .device
                .active_pct
                .map(|active| pct(gpu_ns, total_ns) * active / 100.0),
            attributed_gpu_ns: gpu_ns,
            lifetime_gpu_ns: processes.iter().fold(0u64, |total, process| {
                total.saturating_add(process.lifetime_gpu_ns)
            }),
            graphics_memory_bytes: processes
                .iter()
                .filter_map(|process| process.graphics_memory_bytes)
                .reduce(u64::saturating_add),
            members: processes.len(),
            gpu_pids: processes.iter().map(|process| process.pid).collect(),
            ..AppView::default()
        });
    }
    report.unattributed = None;
}

/// A stable key for a bundle group: FNV-1a of `key` with the high bit set, so
/// it cannot collide with a kernel coalition ID.
fn synthetic_id(key: &str) -> u64 {
    let hash = key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    hash | SYNTHETIC_ID_BIT
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::super::{AnalyzeOptions, analyze};
    use super::*;
    use crate::sys::coalition::CoalitionUsage;
    use crate::sys::iokit::GpuClient;
    use std::time::{Duration, Instant};

    const MS: u64 = 1_000_000;
    const SECOND: u64 = 1_000_000_000;

    fn info(pid: i32, path: &str) -> ProcessInfo {
        ProcessInfo {
            pid,
            ppid: 1,
            uid: 501,
            start_us: 1_000_000 + pid as u64,
            path: Some(path.to_string()),
            name: path.rsplit('/').next().unwrap_or(path).to_string(),
        }
    }

    fn usage(gpu_ns: u64, energy_nj: u64, tasks_exited: u64) -> CoalitionUsage {
        CoalitionUsage {
            tasks_started: 10,
            tasks_exited,
            gpu_ns,
            gpu_energy_nj: energy_nj,
            ane_ns: gpu_ns / 10,
            ane_energy_nj: energy_nj / 10,
            ..CoalitionUsage::default()
        }
    }

    fn coalition(
        id: u64,
        usage: CoalitionUsage,
        members: Vec<i32>,
        leader: Option<ProcessInfo>,
    ) -> Coalition {
        Coalition {
            id,
            usage,
            responsible_pid: leader.as_ref().map(|leader| leader.pid),
            members,
            leader,
        }
    }

    const CHROME: &str = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
    const CHROME_GPU: &str = "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Helpers/Google Chrome Helper (GPU).app/Contents/MacOS/Google Chrome Helper (GPU)";
    const RENDERER: &str = "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Helpers/Google Chrome Helper (Renderer).app/Contents/MacOS/Google Chrome Helper (Renderer)";
    const WINDOW_SERVER: &str =
        "/System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer";

    /// Two snapshots one second apart: Chrome (coalition 100: main 10, GPU
    /// helper 11, renderer 12) and WindowServer (coalition 395, pid 431, no
    /// responsible process).
    fn pair(
        chrome: (CoalitionUsage, CoalitionUsage),
        window_server: (CoalitionUsage, CoalitionUsage),
        clients: (Vec<GpuClient>, Vec<GpuClient>),
    ) -> (Snapshot, Snapshot) {
        let base = Instant::now();
        let mut snapshots = Vec::new();
        for (offset, (chrome_usage, ws_usage, clients)) in [
            (0, (chrome.0, window_server.0, clients.0)),
            (1, (chrome.1, window_server.1, clients.1)),
        ] {
            let mut snapshot =
                snapshot(base, Duration::from_secs(offset), offset * SECOND, clients);
            for (pid, path) in [(11, CHROME_GPU), (12, RENDERER), (431, WINDOW_SERVER)] {
                if snapshot.processes.contains_key(&pid) {
                    snapshot.processes.insert(pid, info(pid, path));
                }
            }
            snapshot.coalitions = Some(Ok(CoalitionTable::new(
                vec![
                    coalition(100, chrome_usage, vec![10, 11, 12], Some(info(10, CHROME))),
                    coalition(395, ws_usage, vec![431, 818], None),
                    coalition(7, usage(0, 0, 0), vec![99], None),
                ],
                true,
                true,
            )));
            let mut memory = MemoryTable::default();
            memory.readings.insert(
                11,
                GraphicsMemory::Bytes {
                    footprint: 300 * MS,
                    compressed: 20 * MS,
                    nofootprint: 5,
                },
            );
            memory.readings.insert(12, GraphicsMemory::Unavailable);
            memory.readings.insert(431, GraphicsMemory::Denied);
            snapshot.memory = Some(memory);
            snapshots.push(snapshot);
        }
        let current = snapshots.pop().expect("current");
        let previous = snapshots.pop().expect("previous");
        (previous, current)
    }

    fn steady_clients(
        gpu_ms: u64,
        renderer_ms: u64,
        ws_ms: u64,
    ) -> (Vec<GpuClient>, Vec<GpuClient>) {
        (
            vec![
                client(1, 11, vec![queue(SECOND, 1)]),
                client(2, 12, vec![queue(SECOND, 1)]),
                client(3, 431, vec![queue(SECOND, 1)]),
            ],
            vec![
                client(1, 11, vec![queue(SECOND + gpu_ms * MS, SECOND)]),
                client(2, 12, vec![queue(SECOND + renderer_ms * MS, SECOND)]),
                client(3, 431, vec![queue(SECOND + ws_ms * MS, SECOND)]),
            ],
        )
    }

    fn app(report: &Report, id: u64) -> &AppView {
        report
            .apps
            .iter()
            .find(|app| app.coalition_id == id)
            .expect("app")
    }

    /// Analyzes the pair as if IOReport had measured the GPU `active_pct` busy.
    fn with_active(previous: &Snapshot, current: &Snapshot, active_pct: f64) -> Report {
        with_device(previous, current, Some(active_pct), None)
    }

    /// Analyzes the pair as if IOReport had measured these device figures.
    fn with_device(
        previous: &Snapshot,
        current: &Snapshot,
        active_pct: Option<f64>,
        power_w: Option<f64>,
    ) -> Report {
        let mut report = analyze(previous, current, &AnalyzeOptions::default());
        report.apps.clear();
        report.unattributed = None;
        report.device.active_pct = active_pct;
        report.device.power_w = power_w;
        fill(
            previous,
            current,
            SECOND,
            false,
            &mut EnergyHistory::default(),
            &mut report,
        );
        report
    }

    fn set_usage(snapshot: &mut Snapshot, id: u64, usage: CoalitionUsage) {
        let Some(Ok(table)) = &snapshot.coalitions else {
            panic!("table")
        };
        let coalitions = table
            .iter()
            .cloned()
            .map(|mut coalition| {
                if coalition.id == id {
                    coalition.usage = usage;
                }
                coalition
            })
            .collect();
        snapshot.coalitions = Some(Ok(CoalitionTable::new(coalitions, true, true)));
    }

    #[test]
    fn groups_names_and_apportions() {
        let (previous, current) = pair(
            (
                usage(5 * SECOND, 9 * SECOND, 3),
                usage(5 * SECOND + 400 * MS, 11 * SECOND, 3),
            ),
            (
                usage(SECOND, SECOND, 0),
                usage(SECOND + 100 * MS, SECOND + 200_000_000, 0),
            ),
            steady_clients(300, 100, 100),
        );
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        assert_eq!(report.sources[source::COALITIONS].state, SourceState::Ok);

        let chrome = app(&report, 100);
        assert_eq!(chrome.name, "Google Chrome");
        assert_eq!(
            chrome.bundle_path.as_deref(),
            Some("/Applications/Google Chrome.app")
        );
        assert_eq!(chrome.responsible_pid, Some(10));
        assert_eq!(chrome.gpu_ns, 400 * MS);
        assert_eq!(chrome.in_flight_gpu_ns, Some(400 * MS));
        assert_eq!(chrome.in_flight_ms_per_s, Some(400.0));
        assert_eq!(chrome.attributed_gpu_ns, 400 * MS);
        assert_eq!(chrome.unattributed_gpu_ns, Some(0));
        assert!((chrome.share_pct - 80.0).abs() < 1e-9);
        assert!((chrome.busy_ms_per_s - 400.0).abs() < 1e-6);
        assert!((chrome.power_w.expect("power") - 2.0).abs() < 1e-9);
        assert!((chrome.energy_j.expect("energy") - 2.0).abs() < 1e-9);
        assert!((chrome.ane_ms_per_s.expect("ane") - 40.0).abs() < 1e-6);
        assert!((chrome.ane_power_w.expect("ane power") - 0.2).abs() < 1e-9);
        assert_eq!(chrome.billed_to_me_w, Some(0.0));
        assert_eq!(chrome.lifetime_gpu_ns, 5 * SECOND + 400 * MS);
        assert_eq!(chrome.members, 3);
        assert_eq!(chrome.gpu_pids.len(), 2);
        // Only the GPU helper's memory is known.
        assert_eq!(chrome.graphics_memory_bytes, Some(320 * MS));

        // WindowServer has no responsible process: named after its GPU member.
        let window_server = app(&report, 395);
        assert_eq!(window_server.name, "WindowServer");
        assert_eq!(window_server.responsible_pid, None);
        assert_eq!(window_server.graphics_memory_bytes, None);

        // Coalition 7 never used the GPU.
        assert_eq!(report.apps.len(), 2);
        assert_eq!(report.apps[0].coalition_id, 100);

        // Per-process power: 2 W split 3:1.
        let helper = report.process(11, 1_000_011).expect("helper");
        assert_eq!(helper.coalition_id, Some(100));
        assert!((helper.est_power_w.expect("estimate") - 1.5).abs() < 1e-9);
        let renderer = report.process(12, 1_000_012).expect("renderer");
        assert!((renderer.est_power_w.expect("estimate") - 0.5).abs() < 1e-9);
        assert_eq!(helper.graphics_memory_bytes, Some(320 * MS));
        assert_eq!(renderer.graphics_memory_bytes, None);

        assert_eq!(report.sources[source::MEMORY].state, SourceState::Partial);
        let unattributed = report.unattributed.clone().expect("unattributed");
        assert_eq!(unattributed.gpu_ns, 0);
        assert!(!unattributed.partial);
    }

    #[test]
    fn app_rows_add_up_to_process_rows() {
        // Both apps overlap: in-flight time is 2.5x and 1.5x their AppUsage
        // time, and the in-flight total exceeds the window.
        let (previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(6 * SECOND, 4 * SECOND, 3)),
            (usage(SECOND, 0, 0), usage(SECOND + 900 * MS, SECOND, 0)),
            steady_clients(300, 100, 600),
        );
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        let mut shares = 0.0;
        for app in &report.apps {
            let processes: u64 = report
                .processes
                .iter()
                .filter(|process| process.coalition_id == Some(app.coalition_id))
                .map(|process| process.gpu_ns)
                .sum();
            assert_eq!(app.gpu_ns, processes + app.unattributed_gpu_ns.unwrap_or(0));
            shares += app.share_pct;
            // Process power estimates add up to the app's measured power.
            let estimates: f64 = report
                .processes
                .iter()
                .filter(|process| process.coalition_id == Some(app.coalition_id))
                .filter_map(|process| process.est_power_w)
                .sum();
            assert!((estimates - app.power_w.expect("power")).abs() < 1e-9);
        }
        assert!((shares - 100.0).abs() < 1e-9);
        let chrome = app(&report, 100);
        assert_eq!(chrome.gpu_ns, 400 * MS);
        assert!((chrome.share_pct - 40.0).abs() < 1e-9);
        assert_eq!(chrome.in_flight_gpu_ns, Some(SECOND));
        assert_eq!(app(&report, 395).in_flight_gpu_ns, Some(900 * MS));
        assert_eq!(
            report.apps[0].coalition_id, 395,
            "sorted by apportioned time"
        );
    }

    #[test]
    fn exited_time_without_live_clients_is_unattributed() {
        // A short-lived helper ran 250 ms and exited; no live member used the GPU.
        let (previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + 250 * MS, 0, 4)),
            (usage(SECOND, 0, 0), usage(SECOND + 100 * MS, 0, 0)),
            steady_clients(0, 0, 100),
        );
        let report = with_active(&previous, &current, 40.0);
        let chrome = app(&report, 100);
        assert_eq!(chrome.attributed_gpu_ns, 0);
        assert_eq!(chrome.unattributed_gpu_ns, Some(250 * MS));
        let unattributed = report.unattributed.clone().expect("unattributed");
        assert_eq!(unattributed.gpu_ns, 250 * MS);
        assert!((unattributed.busy_ms_per_s - 250.0).abs() < 1e-6);
        // Idle members are filtered with their processes.
        assert!(report.process(11, 1_000_011).is_none());
    }

    #[test]
    fn exited_load_gets_the_busy_time_live_clients_miss() {
        // Measured by the metal teammate: a load in Chrome's coalition exited
        // mid-window after about 479 ms. The GPU was 57.9% busy; WindowServer
        // (no exits) read 1.75x in flight because all its work ran during the
        // load, while the load barely overlapped it. Device busy time not in
        // live AppUsage, 579 - 58 = 521 ms, is capped at Chrome's 494 ms gap.
        let (previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + 510 * MS, 0, 4)),
            (usage(SECOND, 0, 0), usage(SECOND + 73 * MS, 0, 0)),
            steady_clients(16, 0, 42),
        );
        let report = with_active(&previous, &current, 57.9);
        let chrome = app(&report, 100);
        assert_eq!(chrome.unattributed_gpu_ns, Some(494 * MS));
        assert_eq!(chrome.gpu_ns, 510 * MS);
        // WindowServer's gap is overlap: no exits, so none of the pool.
        assert_eq!(app(&report, 395).unattributed_gpu_ns, Some(0));
        let unattributed = report.unattributed.clone().expect("unattributed");
        assert_eq!(unattributed.gpu_ns, 494 * MS);
        assert!(!unattributed.partial);
    }

    #[test]
    fn exited_time_is_limited_by_device_busy_time() {
        // Chrome's live work overlapped WindowServer's, so its 600 ms gap is
        // mostly overlap: the GPU was only 700 ms busy and live clients used
        // 600 ms of it, so its exited task gets 100 ms.
        let (previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + SECOND, 0, 4)),
            (usage(SECOND, 0, 0), usage(SECOND + 300 * MS, 0, 0)),
            steady_clients(300, 100, 200),
        );
        let report = with_active(&previous, &current, 70.0);
        let chrome = app(&report, 100);
        assert_eq!(chrome.unattributed_gpu_ns, Some(100 * MS));
        assert_eq!(chrome.gpu_ns, 500 * MS);
        assert_eq!(chrome.in_flight_gpu_ns, Some(SECOND));
    }

    #[test]
    fn exited_time_is_unknown_without_device_busy_time() {
        let (previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + SECOND, 0, 4)),
            (usage(SECOND, 0, 0), usage(SECOND + 300 * MS, 0, 0)),
            steady_clients(300, 100, 200),
        );
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        assert_eq!(report.device.active_pct, None);
        let chrome = app(&report, 100);
        assert_eq!(chrome.gpu_ns, 400 * MS, "unknown exited time is left out");
        assert_eq!(chrome.in_flight_gpu_ns, Some(SECOND), "but stays in flight");
        assert_eq!(chrome.unattributed_gpu_ns, None);
        // Without exit evidence the gap is overlap, not exited time.
        assert_eq!(app(&report, 395).unattributed_gpu_ns, Some(0));
        let unattributed = report.unattributed.clone().expect("unattributed");
        assert!(unattributed.partial);
        assert_eq!(unattributed.gpu_ns, 0);
    }

    #[test]
    fn busy_time_is_split_by_gap() {
        // Two apps without live clients share 200 ms of missing busy time in
        // proportion to their 300 ms and 100 ms of in-flight time.
        let (previous, mut current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + 300 * MS, 0, 4)),
            (usage(SECOND, 0, 0), usage(SECOND + 200 * MS, 0, 0)),
            steady_clients(0, 0, 200),
        );
        set_usage(&mut current, 7, usage(100 * MS, 0, 0));
        let report = with_active(&previous, &current, 40.0);
        assert_eq!(app(&report, 100).unattributed_gpu_ns, Some(150 * MS));
        assert_eq!(app(&report, 7).unattributed_gpu_ns, Some(50 * MS));
        assert_eq!(app(&report, 7).gpu_ns, 50 * MS);
    }

    #[test]
    fn vanished_client_is_exit_evidence() {
        let (mut previous, mut current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + 500 * MS, 0, 3)),
            (usage(SECOND, 0, 0), usage(SECOND + 100 * MS, 0, 0)),
            steady_clients(100, 0, 100),
        );
        // The renderer closed its GPU client during the window.
        if let (Ok(old), Ok(new)) = (&mut previous.registry, &mut current.registry) {
            assert_eq!(old.accelerators[0].clients.len(), 3);
            new.accelerators[0]
                .clients
                .retain(|client| client.pid != 12);
        }
        let report = with_active(&previous, &current, 60.0);
        assert_eq!(app(&report, 100).unattributed_gpu_ns, Some(400 * MS));
    }

    #[test]
    fn small_gaps_are_skew() {
        // A 10 ms gap with an exit, on a saturated GPU, is counter skew.
        let (previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + 110 * MS, 0, 4)),
            (usage(SECOND, 0, 0), usage(SECOND + 100 * MS, 0, 0)),
            steady_clients(100, 0, 100),
        );
        let report = with_active(&previous, &current, 100.0);
        assert_eq!(app(&report, 100).unattributed_gpu_ns, Some(0));
        assert!(!report.unattributed.clone().expect("unattributed").partial);
    }

    #[test]
    fn app_power_sums_to_device_power() {
        // Coalition energy says Chrome 2 W and WindowServer 0.2 W, but it lags:
        // IOReport measured 4.4 W, so both scale by 2 and keep their split.
        let (previous, current) = pair(
            (
                usage(5 * SECOND, 9 * SECOND, 3),
                usage(5 * SECOND + 400 * MS, 11 * SECOND, 3),
            ),
            (
                usage(SECOND, SECOND, 0),
                usage(SECOND + 100 * MS, SECOND + 200 * MS, 0),
            ),
            steady_clients(300, 100, 100),
        );
        let report = with_device(&previous, &current, None, Some(4.4));
        let chrome = app(&report, 100);
        assert!((chrome.power_w.expect("power") - 4.0).abs() < 1e-9);
        assert!((chrome.energy_j.expect("energy") - 4.0).abs() < 1e-9);
        assert!((chrome.lifetime_energy_j.expect("lifetime") - 11.0).abs() < 1e-9);
        assert!((app(&report, 395).power_w.expect("power") - 0.4).abs() < 1e-9);
        let total: f64 = report.apps.iter().filter_map(|app| app.power_w).sum();
        assert!((total - 4.4).abs() < 1e-9);
        // Process estimates follow the app's power: 4 W split 3:1.
        let helper = report.process(11, 1_000_011).expect("helper");
        assert!((helper.est_power_w.expect("estimate") - 3.0).abs() < 1e-9);

        // Without IOReport power the coalition figure stands.
        let report = with_device(&previous, &current, None, None);
        assert!((app(&report, 100).power_w.expect("power") - 2.0).abs() < 1e-9);
    }

    /// One sample of a sequence: Chrome's GPU helper (pid 11) and WindowServer
    /// (pid 431) with these cumulative queue times and coalition energies.
    fn sample(
        base: Instant,
        at_ms: u64,
        (chrome_ns, chrome_nj): (u64, u64),
        (ws_ns, ws_nj): (u64, u64),
    ) -> Snapshot {
        let clients = vec![
            client(1, 11, vec![queue(SECOND + chrome_ns, at_ms * MS)]),
            client(3, 431, vec![queue(SECOND + ws_ns, at_ms * MS)]),
        ];
        let mut snapshot = snapshot(base, Duration::from_millis(at_ms), at_ms * MS, clients);
        snapshot.processes.insert(11, info(11, CHROME_GPU));
        snapshot.processes.insert(431, info(431, WINDOW_SERVER));
        snapshot.coalitions = Some(Ok(CoalitionTable::new(
            vec![
                coalition(
                    100,
                    usage(5 * SECOND + chrome_ns, chrome_nj, 3),
                    vec![10, 11, 12],
                    Some(info(10, CHROME)),
                ),
                coalition(395, usage(SECOND + ws_ns, ws_nj, 0), vec![431, 818], None),
            ],
            true,
            true,
        )));
        snapshot
    }

    /// Analyzes consecutive windows with one carried history, as if IOReport
    /// had measured `power_w` in each.
    fn windows(samples: &[Snapshot], power_w: Option<f64>) -> Vec<Report> {
        let mut history = EnergyHistory::default();
        samples
            .windows(2)
            .map(|pair| {
                let mut report = analyze(&pair[0], &pair[1], &AnalyzeOptions::default());
                report.apps.clear();
                report.unattributed = None;
                report.device.power_w = power_w;
                let window_ns = pair[1].at.duration_since(pair[0].at).as_nanos() as u64;
                fill(
                    &pair[0],
                    &pair[1],
                    window_ns,
                    false,
                    &mut history,
                    &mut report,
                );
                report
            })
            .collect()
    }

    fn power_of(report: &Report, id: u64) -> f64 {
        report
            .apps
            .iter()
            .find(|app| app.coalition_id == id)
            .and_then(|app| app.power_w)
            .unwrap_or(0.0)
    }

    #[test]
    fn short_windows_split_device_energy_over_the_trailing_span() {
        // 30 windows of 100 ms at a steady 10 W. Chrome uses 60 ms/s of GPU
        // time and WindowServer 20 ms/s, but their coalition energy moves only
        // every 500 ms (windows 3, 8, 13, ...): 4:1 at first, 1:1 from window 13.
        let base = Instant::now();
        let (mut chrome_nj, mut ws_nj) = (0, 0);
        let mut samples = vec![sample(base, 0, (0, 0), (0, 0))];
        for window in 0..30u64 {
            if window % 5 == 3 {
                let (chrome, ws) = if window < 13 { (4, 1) } else { (1, 1) };
                chrome_nj += chrome * SECOND;
                ws_nj += ws * SECOND;
            }
            let at_ms = (window + 1) * 100;
            samples.push(sample(
                base,
                at_ms,
                (at_ms * 60 / 1000 * MS, chrome_nj),
                (at_ms * 20 / 1000 * MS, ws_nj),
            ));
        }
        let reports = windows(&samples, Some(10.0));

        // Every window's device energy goes to the apps: none is lost and none
        // is moved into another window.
        for (index, report) in reports.iter().enumerate() {
            let total: f64 = report.apps.iter().filter_map(|app| app.power_w).sum();
            assert!((total - 10.0).abs() < 1e-9, "window {index}: {total} W");
        }
        let energy: f64 = reports
            .iter()
            .flat_map(|report| &report.apps)
            .filter_map(|app| app.energy_j)
            .sum();
        assert!((energy - 30.0).abs() < 1e-9, "{energy} J");

        // Before the first step, apportioned GPU time splits it: 60:20.
        assert!((power_of(&reports[0], 100) - 7.5).abs() < 1e-9);
        // Then the coalition energy of the last 900 ms: windows 3 to 12 see
        // only 4:1 steps, window 13 the steps of windows 8 and 13 (5:2), and
        // from window 17 only 1:1 steps.
        for (index, report) in reports.iter().enumerate().take(13).skip(3) {
            assert!((power_of(report, 100) - 8.0).abs() < 1e-9, "{index}");
        }
        assert!((power_of(&reports[13], 100) - 50.0 / 7.0).abs() < 1e-9);
        for (index, report) in reports.iter().enumerate().skip(17) {
            assert!((power_of(report, 100) - 5.0).abs() < 1e-9, "{index}");
            assert!((power_of(report, 395) - 5.0).abs() < 1e-9, "{index}");
        }
        // Process estimates follow the app's power.
        let helper = reports[20].process(11, 1_000_011).expect("helper");
        assert!((helper.est_power_w.expect("estimate") - 5.0).abs() < 1e-9);

        // A single window on its own sees no step between two steps, so only
        // GPU time can split it.
        let alone = with_device(&samples[5], &samples[6], None, Some(10.0));
        assert!((power_of(&alone, 100) - 7.5).abs() < 1e-9);
    }

    #[test]
    fn short_windows_without_device_power_spread_each_step_over_the_span() {
        // No IOReport: one 0.9 J step of Chrome's coalition energy, in window
        // 12, after Chrome stopped using the GPU. It is spread at 1 W over the
        // 900 ms span, keeping its total, and Chrome stays a row while the
        // step is in the span even though it used no GPU time.
        let base = Instant::now();
        let mut samples = vec![sample(base, 0, (0, 0), (0, 0))];
        for window in 0..30u64 {
            let chrome_nj = if window >= 12 { 900 * MS } else { 0 };
            let at_ms = (window + 1) * 100;
            samples.push(sample(
                base,
                at_ms,
                (0, chrome_nj),
                (at_ms * 20 / 1000 * MS, 0),
            ));
        }
        let reports = windows(&samples, None);
        for (index, report) in reports.iter().enumerate() {
            let chrome = report.apps.iter().find(|app| app.coalition_id == 100);
            if (12..21).contains(&index) {
                let chrome = chrome.expect("chrome row");
                assert!(
                    (chrome.power_w.expect("power") - 1.0).abs() < 1e-9,
                    "{index}"
                );
                assert_eq!(chrome.gpu_ns, 0);
            } else {
                assert!(chrome.is_none(), "window {index}");
            }
        }
        let energy: f64 = reports
            .iter()
            .flat_map(|report| &report.apps)
            .filter_map(|app| app.energy_j)
            .sum();
        assert!((energy - 0.9).abs() < 1e-9, "{energy} J");
    }

    #[test]
    fn energy_span_covers_at_least_the_span() {
        let mut history = EnergyHistory::default();
        // 1 s windows, even a little short, are split by their own energy.
        for window_ns in [SECOND, 998 * MS, 1_002 * MS, 998 * MS] {
            history.push(window_ns, vec![(1, 5)]);
            assert_eq!(history.windows.len(), 1);
            assert_eq!(history.energy_nj()[&1], 5);
        }
        // Short windows keep the fewest that cover ENERGY_SPAN_NS.
        for _ in 0..20 {
            history.push(100 * MS, vec![(1, 1), (2, 3)]);
        }
        assert_eq!(history.windows.len(), 9);
        assert_eq!(history.span_ns(), ENERGY_SPAN_NS);
        assert_eq!(history.energy_nj()[&2], 27);
        history.push(450 * MS, Vec::new());
        history.push(450 * MS, Vec::new());
        assert_eq!(history.windows.len(), 2);
        history.push(2 * SECOND, Vec::new());
        assert_eq!(history.windows.len(), 1);
        history.clear();
        assert_eq!(history.span_ns(), 0);
    }

    #[test]
    fn overlap_ratio_is_the_fallback_without_device_busy_time() {
        // The renderer exited after 300 ms of GPU work; the GPU helper is live.
        // WindowServer (no exits) shows no overlap.
        let (previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + 320 * MS, 0, 4)),
            (usage(SECOND, 0, 0), usage(SECOND + 100 * MS, 0, 0)),
            steady_clients(20, 0, 100),
        );
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        assert_eq!(report.device.active_pct, None);
        assert_eq!(app(&report, 100).unattributed_gpu_ns, Some(300 * MS));
    }

    #[test]
    fn fallback_is_capped_to_the_free_window() {
        // Chrome's exited tasks report 900 ms in flight while WindowServer's
        // live clients used 600 ms of the 1 s window: at most 400 ms is free.
        let (previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + 900 * MS, 0, 4)),
            (usage(SECOND, 0, 0), usage(SECOND + 600 * MS, 0, 0)),
            steady_clients(0, 0, 600),
        );
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        let chrome = app(&report, 100);
        assert_eq!(chrome.unattributed_gpu_ns, Some(400 * MS));
        assert_eq!(chrome.gpu_ns, 400 * MS);
        assert_eq!(chrome.in_flight_gpu_ns, Some(900 * MS));
    }

    #[test]
    fn overlap_ratio_estimate() {
        assert_eq!(unattributed(110 * MS, 100 * MS, true, 1.0, 1.0), Some(0));
        assert_eq!(unattributed(400 * MS, 100 * MS, false, 1.0, 1.0), Some(0));
        assert_eq!(unattributed(400 * MS, 0, false, 1.0, 9.0), Some(400 * MS));
        // In-flight time converts to apportioned time at the measured overlap.
        assert_eq!(unattributed(400 * MS, 0, false, 2.0, 2.0), Some(200 * MS));
        // 1.5x overlap measured elsewhere: (600 - 1.5 * 140) / 1.5 = 260, and
        // the app's own overlap of at most 1.5x moves it by 47, within 25%.
        assert_eq!(
            unattributed(600 * MS, 140 * MS, true, 1.5, 1.5),
            Some(260 * MS)
        );
        // Two of the app's own clients ran: its overlap may reach 2x.
        assert_eq!(unattributed(600 * MS, 140 * MS, true, 1.5, 2.0), None);
        assert_eq!(unattributed(400 * MS, 100 * MS, true, 1.0, 3.0), None);
    }

    #[test]
    fn idle_apps_follow_include_idle() {
        let (previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND, 0, 3)),
            (usage(SECOND, 0, 0), usage(SECOND + 100 * MS, 0, 0)),
            steady_clients(0, 0, 100),
        );
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        assert_eq!(report.apps.len(), 1);
        assert_eq!(report.totals.active_apps, 1);
        let report = analyze(&previous, &current, &AnalyzeOptions { include_idle: true });
        assert_eq!(report.apps.len(), 2);
        assert_eq!(app(&report, 100).gpu_ns, 0);
        assert_eq!(report.totals.active_apps, 1);
    }

    #[test]
    fn new_coalition_is_credited_from_zero() {
        let (mut previous, current) = pair(
            (usage(0, 0, 0), usage(200 * MS, 1_000_000_000, 0)),
            (usage(SECOND, 0, 0), usage(SECOND + 100 * MS, 0, 0)),
            steady_clients(200, 0, 100),
        );
        let Some(Ok(table)) = &previous.coalitions else {
            panic!("table")
        };
        let only_window_server: Vec<Coalition> = table
            .iter()
            .filter(|coalition| coalition.id == 395)
            .cloned()
            .collect();
        previous.coalitions = Some(Ok(CoalitionTable::new(only_window_server, true, true)));
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        let chrome = app(&report, 100);
        assert_eq!(chrome.gpu_ns, 200 * MS);
        assert!((chrome.power_w.expect("power") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn no_baseline_reports_lifetimes_only() {
        let (mut previous, current) = pair(
            (usage(5 * SECOND, 0, 3), usage(5 * SECOND + 400 * MS, 0, 3)),
            (usage(SECOND, 0, 0), usage(SECOND + 100 * MS, 0, 0)),
            steady_clients(300, 100, 100),
        );
        previous.coalitions = Some(Err("failed".into()));
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        assert_eq!(
            report.sources[source::COALITIONS].state,
            SourceState::Partial
        );
        let chrome = app(&report, 100);
        assert_eq!(chrome.gpu_ns, 400 * MS);
        assert_eq!(chrome.in_flight_gpu_ns, None);
        assert_eq!(chrome.attributed_gpu_ns, 400 * MS);
        assert_eq!(chrome.lifetime_gpu_ns, 5 * SECOND + 400 * MS);
        assert_eq!(chrome.power_w, None);
        assert_eq!(chrome.unattributed_gpu_ns, None);
        assert!(report.unattributed.is_none());
    }

    #[test]
    fn falls_back_to_bundles() {
        let (previous, mut current) = pair(
            (usage(0, 0, 0), usage(0, 0, 0)),
            (usage(0, 0, 0), usage(0, 0, 0)),
            steady_clients(300, 100, 50),
        );
        current.coalitions = Some(Err("coalition syscall failed".into()));
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        let status = &report.sources[source::COALITIONS];
        assert_eq!(status.state, SourceState::Unavailable);
        assert!(
            status
                .detail
                .as_deref()
                .unwrap_or_default()
                .contains("bundle")
        );
        assert_eq!(report.apps.len(), 2);
        let chrome = &report.apps[0];
        assert_eq!(chrome.name, "Google Chrome");
        assert_eq!(chrome.gpu_ns, 400 * MS);
        assert_eq!(chrome.gpu_pids, vec![11, 12]);
        assert!(chrome.coalition_id & SYNTHETIC_ID_BIT != 0);
        assert_eq!(chrome.power_w, None);
        assert_eq!(chrome.unattributed_gpu_ns, None);
        assert_eq!(report.apps[1].name, "WindowServer");
        assert!(report.unattributed.is_none());
        assert!(
            report
                .processes
                .iter()
                .all(|process| process.coalition_id.is_none())
        );

        current.coalitions = None;
        let report = analyze(&previous, &current, &AnalyzeOptions::default());
        assert_eq!(
            report.sources[source::COALITIONS].state,
            SourceState::Disabled
        );
        assert_eq!(report.apps.len(), 2);
    }

    #[test]
    fn memory_status() {
        let mut report = Report {
            processes: vec![
                ProcessView {
                    pid: 431,
                    ..ProcessView::default()
                },
                ProcessView {
                    pid: 5,
                    ..ProcessView::default()
                },
            ],
            ..Report::default()
        };
        let mut memory = MemoryTable::default();
        memory.readings.insert(431, GraphicsMemory::Denied);
        fill_memory(Some(&memory), &mut report);
        assert_eq!(report.sources[source::MEMORY].state, SourceState::Denied);
        memory.readings.insert(
            5,
            GraphicsMemory::Bytes {
                footprint: 0,
                compressed: 0,
                nofootprint: 0,
            },
        );
        fill_memory(Some(&memory), &mut report);
        assert_eq!(report.sources[source::MEMORY].state, SourceState::Partial);
        assert_eq!(report.processes[1].graphics_memory_bytes, Some(0));
        assert_eq!(report.processes[0].graphics_memory_bytes, None);
        fill_memory(None, &mut report);
        assert_eq!(report.sources[source::MEMORY].state, SourceState::Disabled);
    }
}
