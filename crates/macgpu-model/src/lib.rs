//! The report data model: one [`Report`] per sampling window.
//!
//! Every front end (table, JSON, TUI, log, Prometheus, record and replay)
//! consumes `Report`. It is the stable, versioned JSON schema: new fields are
//! added as optional so older recordings still load.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bumped when a field changes meaning or is removed. Adding optional fields
/// does not bump it.
pub const SCHEMA_VERSION: u32 = 1;

/// Measurements for one sampling window, from the previous sample to this one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    /// End of the window, in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    pub window_seconds: f64,
    pub device: DeviceView,
    pub totals: Totals,
    /// Processes that own GPU clients, sorted by window GPU time, descending.
    /// Idle processes are included only when the report was built with
    /// `include_idle`.
    #[serde(default)]
    pub processes: Vec<ProcessView>,
    /// Apps (resource coalitions) with GPU activity, sorted by window GPU time.
    #[serde(default)]
    pub apps: Vec<AppView>,
    /// Apportioned GPU time no live GPU client accounts for, typically clients
    /// or processes that exited inside the window: the device's busy time
    /// (IOReport) beyond live clients' `AppUsage`, or without IOReport an
    /// overlap-ratio estimate from the coalition counters.
    /// `None` when coalitions are unavailable or there is no baseline yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unattributed: Option<Unattributed>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<Event>,
    #[serde(default)]
    pub sources: BTreeMap<String, SourceStatus>,
    #[serde(default)]
    pub overhead: Overhead,
}

/// Whole-GPU measurements.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeviceView {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_cores: Option<u32>,

    /// Firmware `PerformanceStatistics` readings. These are instantaneous; when
    /// several readings fall inside the window they are averaged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firmware_utilization: Option<FirmwareUtilization>,

    /// Share of the window the GPU spent in any active P-state (IOReport
    /// `GPUPH`), integrated over the whole window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_pct: Option<f64>,
    /// Residency-weighted average clock while active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avg_mhz: Option<f64>,
    /// Highest clock in the frequency table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_mhz: Option<f64>,
    /// `active_pct * avg_mhz / max_mhz`: active time weighted by clock speed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freq_normalized_pct: Option<f64>,
    /// Average GPU power over the window (IOReport `Energy Model / GPU Energy`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_w: Option<f64>,
    /// Residency per P-state over the window, including `OFF`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pstates: Vec<PStateShare>,
    /// Power-controller states (IOReport `PWRCTRL`) with nonzero residency, as a
    /// share of the time the GPU was powered (every state except `IDLE_OFF`),
    /// largest first. `PERF` means the clock followed demand, unconstrained.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub limiters: Vec<StateShare>,
    /// Share of the window spent under thermal (CLTM) limiting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_limited_pct: Option<f64>,
    /// Counter increments inside the window (throttle, THM, issue counters),
    /// keyed `"<IOReport subgroup>/<channel>"`, for example
    /// `"Issue Counts/Total Restarts"`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub counters: BTreeMap<String, u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Temperature>,
    #[serde(default)]
    pub memory: DeviceMemory,
    #[serde(default)]
    pub system: SystemView,
    /// Cumulative GPU recoveries (hangs) since boot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_count: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FirmwareUtilization {
    pub device_pct: f64,
    pub renderer_pct: f64,
    pub tiler_pct: f64,
    /// How many readings were averaged.
    pub readings: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PStateShare {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mhz: Option<f64>,
    pub pct: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StateShare {
    pub name: String,
    pub pct: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Temperature {
    /// Mean of the GPU die sensors.
    pub avg_c: f64,
    /// Hottest GPU die sensor.
    pub max_c: f64,
    pub sensors: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeviceMemory {
    /// `PerformanceStatistics` "In use system memory".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_use_bytes: Option<u64>,
    /// `PerformanceStatistics` "Alloc system memory".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocated_bytes: Option<u64>,
    /// GPU wired-memory limit set with `iogpu.wired_limit_mb`. `None` when the
    /// sysctl is 0: the OS default applies and its value is not exposed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wired_limit_bytes: Option<u64>,
    /// `MTLDevice.recommendedMaxWorkingSetSize`. Not filled by the sampler:
    /// creating an `MTLDevice` would make macgpu a GPU client of its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_working_set_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_physical_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SystemView {
    /// `nominal`, `moderate`, `heavy`, `trapping`, or `sleeping`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_pressure: Option<String>,
    /// `normal`, `warning`, or `critical`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_pressure: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Totals {
    /// Sum of every client's GPU time in the window. The driver apportions busy
    /// time among contexts that run concurrently, so this is at most about the
    /// window length.
    pub attributed_gpu_ns: u64,
    /// `attributed_gpu_ns` per wall-clock second, in milliseconds.
    pub attributed_ms_per_s: f64,
    pub gpu_clients: usize,
    pub gpu_processes: usize,
    pub active_processes: usize,
    pub active_apps: usize,
}

/// How recently a process or queue submitted GPU work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activity {
    /// Used the GPU inside this window.
    Active,
    /// Submitted work within the last `RECENT_SECONDS` (see `macgpu_core::analyze`).
    Recent,
    #[default]
    Idle,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessView {
    pub pid: i32,
    /// Process start time (µs since the Unix epoch); with `pid`, a unique key.
    pub start_us: u64,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ppid: Option<i32>,
    /// Resource coalition this process belongs to (the app). When resource
    /// coalitions are unavailable it is the synthetic bundle key instead (see
    /// [`AppView`]), so it still links the process to its app row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coalition_id: Option<u64>,

    /// GPU time in the window, summed over all of the process's queues.
    pub gpu_ns: u64,
    /// `gpu_ns` per wall-clock second, in milliseconds: this process's share of
    /// GPU busy time. The driver apportions time among contexts that run
    /// concurrently, so the sum across processes is at most about 1000.
    pub busy_ms_per_s: f64,
    /// Share of all attributed GPU time in the window. Sums to 100 across processes.
    pub share_pct: f64,
    /// `share_pct * device.active_pct / 100`: the process's slice of the device's
    /// active time. Sums to `device.active_pct` across processes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_pct: Option<f64>,
    /// GPU power apportioned from the app's measured coalition GPU energy by
    /// this process's share of the app's GPU time. An estimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub est_power_w: Option<f64>,

    /// Total GPU time of the process's live clients since they were created.
    pub lifetime_gpu_ns: u64,
    /// Graphics memory (`TASK_VM_INFO` `ledger_tag_graphics_footprint`). `None`
    /// when unreadable, which is the case for other users' processes without root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphics_memory_bytes: Option<u64>,

    pub activity: Activity,
    /// Time since the most recent submission on any queue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_submit_age_ms: Option<u64>,
    /// Number of command queues (`AppUsage` entries) across clients.
    pub queues: usize,
    #[serde(default)]
    pub clients: Vec<ClientView>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClientView {
    pub registry_id: u64,
    pub gpu_ns: u64,
    pub busy_ms_per_s: f64,
    pub lifetime_gpu_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_queue_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_queue_count_max: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_submit_age_ms: Option<u64>,
    #[serde(default)]
    pub queues: Vec<QueueView>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueueView {
    /// Position in the client's `AppUsage` array.
    pub index: usize,
    pub api: String,
    pub gpu_ns: u64,
    pub busy_ms_per_s: f64,
    pub lifetime_gpu_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_submit_age_ms: Option<u64>,
    pub activity: Activity,
}

/// One app: a resource coalition (an app plus its helpers and XPC services).
///
/// `gpu_ns`, `busy_ms_per_s`, and `share_pct` are apportioned time, in the same
/// units as the process rows: the members' `AppUsage` time plus the app's
/// unattributed (exited) time when known. So an app row equals the sum of its
/// process rows plus its known unattributed time, and the shares sum to 100.
///
/// The `in_flight_*` fields are the coalition's kernel counter instead: wall
/// time the app had GPU work in flight. Work that overlaps another app's counts
/// in full for each, so it can exceed `gpu_ns` and the apps' total can exceed
/// the window. Without overlapping work the two agree within about 15 ms per
/// window; with a second app busy on the GPU, in-flight time was 1.3-3.4x the
/// apportioned time on an M4 Max.
///
/// GPU energy (`energy_j`, `power_w`) comes from the coalition too, but it is
/// already apportioned: over a run the apps' energy sums to the device's GPU
/// energy, with or without overlapping work. It trails the GPU time by about a
/// second, so when IOReport's GPU power is known the apps' power is scaled to
/// sum to `device.power_w` and the coalition energies decide only the split.
/// The kernel credits coalition energy in steps about 0.5 s apart, so a window
/// shorter than 0.9 s is split by the coalition energy of the trailing 0.9 s
/// of windows. The split still lags by about a second while a load ramps up or
/// stops.
///
/// When resource coalitions are unavailable, apps are grouped by `.app` bundle
/// from the process rows instead: `coalition_id` is then a synthetic key (high
/// bit set), unattributed time is unknown, and the in-flight and energy fields
/// are absent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AppView {
    pub coalition_id: u64,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub responsible_pid: Option<i32>,

    /// Apportioned GPU time in the window: `attributed_gpu_ns` plus
    /// `unattributed_gpu_ns` when known.
    pub gpu_ns: u64,
    pub busy_ms_per_s: f64,
    /// Share of all apps' `gpu_ns` in the window. Sums to 100 across apps.
    pub share_pct: f64,
    /// `share_pct * device.active_pct / 100`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_pct: Option<f64>,
    /// Coalition GPU time in the window (kernel counter, includes exited
    /// tasks): wall time the app had GPU work in flight, overlapping apps each
    /// counted in full. `None` without coalitions or a baseline sample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_flight_gpu_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_flight_ms_per_s: Option<f64>,
    /// Sum of member processes' client (`AppUsage`) GPU time in the window.
    pub attributed_gpu_ns: u64,
    /// Apportioned GPU time of tasks without a live GPU client, typically
    /// processes or clients that exited inside the window: the app's part of
    /// the device's busy time (IOReport) that live clients' `AppUsage` does not
    /// account for, split among apps with exits (or no live client) by their
    /// in-flight gap (`in_flight_gpu_ns - attributed_gpu_ns`) and never more
    /// than that gap. 0 for apps without exits: their gap is overlap. Without
    /// IOReport it is estimated from the overlap ratio of apps without exits,
    /// which can under-count, and is `None` when that cannot separate it from
    /// overlap; it is then left out of `gpu_ns`, and `in_flight_gpu_ns` still
    /// covers it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unattributed_gpu_ns: Option<u64>,

    /// GPU energy in the window, in joules: `power_w` times the window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_j: Option<f64>,
    /// GPU power in the window. When IOReport's GPU power is known it is
    /// `device.power_w` split by the coalitions' `gpu_energy_nj` over the
    /// window, or over the trailing 0.9 s for shorter windows, so the apps sum
    /// to the device (by apportioned GPU time until any coalition energy has
    /// moved); otherwise the coalition's own `gpu_energy_nj` at its average
    /// rate over that span, which trails real power by about a second.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_w: Option<f64>,
    /// GPU power other coalitions spent on this app's behalf (for example
    /// WindowServer compositing), billed to this app. 0 on macOS 27: the GPU
    /// driver does not bill on behalf of other coalitions there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub billed_to_me_w: Option<f64>,
    /// GPU power this app spent on behalf of others.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub billed_to_others_w: Option<f64>,
    /// Apple Neural Engine time per wall-clock second, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ane_ms_per_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ane_power_w: Option<f64>,

    /// In-flight (coalition counter) GPU time since the coalition was created,
    /// including exited members. Monotonic for a given `coalition_id`. When
    /// grouped by bundle it is the live clients' `AppUsage` lifetime instead,
    /// which drops when a client exits.
    pub lifetime_gpu_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifetime_energy_j: Option<f64>,
    /// Sum of known member graphics memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphics_memory_bytes: Option<u64>,
    /// Number of live member processes.
    pub members: usize,
    /// PIDs of members that own GPU clients.
    #[serde(default)]
    pub gpu_pids: Vec<i32>,
}

/// Apportioned GPU time no live GPU client accounts for, summed over apps.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Unattributed {
    pub gpu_ns: u64,
    pub busy_ms_per_s: f64,
    /// True when some apps' unattributed time is unknown (no device busy time
    /// from IOReport, and overlap hides it), so `gpu_ns` is a lower bound.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub partial: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub timestamp_ms: u64,
    pub kind: EventKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// The GPU recovered from a hang (`recoveryCount` increased).
    GpuRecovery,
    /// IOReport issue counters (restarts, kills) increased.
    GpuIssue,
    ThermalPressure,
    /// A user-defined alert started firing.
    Alert,
    /// A user-defined alert stopped firing.
    AlertCleared,
}

/// Whether a data source produced usable data for this window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceStatus {
    pub state: SourceState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl SourceStatus {
    pub fn ok() -> Self {
        Self {
            state: SourceState::Ok,
            detail: None,
        }
    }

    pub fn with(state: SourceState, detail: impl Into<String>) -> Self {
        Self {
            state,
            detail: Some(detail.into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceState {
    Ok,
    /// Some data was available, some was not (for example memory for other users).
    Partial,
    /// The source exists but its counters did not advance when they should have.
    Stale,
    /// The OS denied access.
    Denied,
    /// The source does not exist on this machine or failed.
    Unavailable,
    /// Turned off by the user.
    Disabled,
}

/// What macgpu itself cost.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Overhead {
    /// Wall time to take the latest sample.
    pub sample_wall_ms: f64,
    /// Sampler-thread CPU time to take the latest sample.
    pub sample_cpu_ms: f64,
    /// macgpu's whole-process CPU usage over the window, in percent of one core.
    pub process_cpu_pct: f64,
}

/// Names of the data sources in [`Report::sources`].
pub mod source {
    pub const REGISTRY: &str = "registry";
    pub const IOREPORT: &str = "ioreport";
    pub const COALITIONS: &str = "coalitions";
    pub const MEMORY: &str = "memory";
    pub const TEMPERATURE: &str = "temperature";
    pub const SYSTEM: &str = "system";
}

/// `PWRCTRL` states that cap the GPU for power or temperature. Apple documents
/// none of them. Observed on M4 Max, macOS 27: `PERF`, `PRFBOOST`, `DEADLINE`,
/// and `DEADAWAR` follow demand (`PRFBOOST` at the top clock), while
/// `FAST_DIE` coincided with a 100 C die and clocks below the maximum. Every
/// state not listed here is treated as not limiting. A limiter state does not
/// always lower the clock (`PWR_ZONE` has been seen at 100% at the top
/// P-state), so report "time under a limiter state", never "throttled".
pub const LIMITING_STATES: &[&str] = &[
    "POWER", "PWR_ZONE", "FAST_DIE", "PRFLIMIT", "PPMLIMIT", "CLTMLIMI", "PPMTARGE",
];

/// Whether a `PWRCTRL` state name is in [`LIMITING_STATES`].
pub fn is_limiting_state(name: &str) -> bool {
    LIMITING_STATES
        .iter()
        .any(|state| name.trim().eq_ignore_ascii_case(state))
}

impl DeviceView {
    /// Power-controller states that impose a limit, largest share first.
    pub fn limiting_states(&self) -> impl Iterator<Item = &StateShare> {
        self.limiters
            .iter()
            .filter(|state| is_limiting_state(&state.name))
    }

    /// Share of the window the GPU spent under a power or thermal limiter
    /// state: the larger of the CLTM thermal-cap share and the power
    /// controller's limiting-state residency. `None` when neither is known.
    ///
    /// Limiter residency is a share of powered time, which is about the GPUPH
    /// active time, so it is scaled by `active_pct` to a share of the window
    /// before it is compared with the CLTM share (already a share of the window).
    pub fn limited_pct(&self) -> Option<f64> {
        let controller = self.active_pct.map(|active| {
            // fold from +0.0: an empty float `sum()` is -0.0, which prints as "-0.0".
            let limiting = self
                .limiting_states()
                .fold(0.0, |total, state| total + state.pct);
            limiting.min(100.0) * active / 100.0
        });
        match (controller, self.thermal_limited_pct) {
            (Some(controller), Some(thermal)) => Some(controller.max(thermal).min(100.0)),
            (Some(value), None) | (None, Some(value)) => Some(value.min(100.0)),
            (None, None) => None,
        }
    }
}

impl Report {
    /// Process key that survives PID reuse.
    pub fn process(&self, pid: i32, start_us: u64) -> Option<&ProcessView> {
        self.processes
            .iter()
            .find(|process| process.pid == pid && process.start_us == start_us)
    }
}
