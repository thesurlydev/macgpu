//! `macgpu doctor`: checks every data source on this machine and, with
//! `--load`, verifies macgpu's numbers against a controlled Metal load.
//!
//! Sources are checked generically from [`Report::sources`] (does the source
//! exist, is it readable without root, is it updating) and then with specific
//! value checks. The load runs in a child process (`macgpu __load`), so the
//! doctor itself never opens a Metal device, and attribution is verified on a
//! PID other than the observer's.
//!
//! Exit status: 0 when nothing failed, 1 when any check failed. A broken GPU
//! registry (the core source) is always a failure; other sources being
//! unavailable are warnings, since macgpu degrades gracefully without them.

use crate::analyze::{AnalyzeOptions, analyze};
use crate::format;
use crate::metal_load::{LoadConfig, LoadResult};
use crate::model::{Report, SourceState, SourceStatus, source};
use crate::snapshot::{Sampler, Snapshot, SourceSet};
#[cfg(test)]
use crate::sys::powermetrics;
use crate::sys::powermetrics::PowerStats;
use anyhow::Result;
use clap::Args;
use serde::Serialize;
use std::ffi::{CString, c_void};
use std::fmt::Write as _;
use std::time::Duration;

/// Window used to check the sources while the machine is idle.
const SOURCE_WINDOW: Duration = Duration::from_secs(1);
/// Idle baseline taken with the load process open but not yet running.
const IDLE_WINDOW: Duration = Duration::from_secs(1);
/// Buffer the load allocates, to verify per-process graphics memory.
const LOAD_ALLOC_MIB: u64 = 256;
/// Default length of `--load`, shared by the flag and `DoctorArgs::default`.
const DEFAULT_LOAD_SECONDS: f64 = 3.0;

/// Idle-baseline activity above which other processes were loading the GPU,
/// so a rise under the load cannot be expected.
const BUSY_BASELINE_PCT: f64 = 50.0;
/// Coalition and `AppUsage` counters are credited at slightly different
/// moments: up to 15 ms per window.
const COALITION_SKEW_NS: u64 = 15_000_000;
/// Coalition GPU energy against IOReport whole-GPU energy, once settled
/// (measured within 0.2%).
const ENERGY_TOLERANCE_PCT: f64 = 2.0;
/// Coalition energy is credited 1-1.5 s late: a 3 s load read 35-51% low at
/// its end and within 0.2% two seconds later.
const ENERGY_SETTLE: Duration = Duration::from_secs(2);
/// powermetrics' active residency against IOReport's, in percentage points.
const RESIDENCY_TOLERANCE_PTS: f64 = 5.0;

#[derive(Args, Debug, Clone)]
pub struct DoctorArgs {
    /// Also run a short Metal load in a child process and verify per-process,
    /// per-app, power, memory, and temperature readings against it. Uses the GPU.
    /// With cached sudo credentials, also runs `sudo -n powermetrics`; it never
    /// prompts for a password.
    #[arg(long)]
    pub load: bool,

    /// Length of the load, in seconds.
    #[arg(long, value_name = "SECONDS", default_value_t = DEFAULT_LOAD_SECONDS, value_parser = parse_load_seconds)]
    pub load_seconds: f64,

    /// Emit one JSON object instead of text.
    #[arg(short, long)]
    pub json: bool,
}

impl Default for DoctorArgs {
    fn default() -> Self {
        Self {
            load: false,
            load_seconds: DEFAULT_LOAD_SECONDS,
            json: false,
        }
    }
}

fn parse_load_seconds(value: &str) -> Result<f64, String> {
    let seconds: f64 = value
        .parse()
        .map_err(|_| "load seconds must be a number".to_string())?;
    if !(1.0..=60.0).contains(&seconds) {
        return Err("load seconds must be between 1 and 60".to_string());
    }
    Ok(seconds)
}

#[derive(Debug, Clone, Serialize)]
pub struct DoctorReport {
    pub macgpu_version: String,
    pub machine: Machine,
    pub gpus: Vec<GpuInfo>,
    pub sources: Vec<SourceCheck>,
    pub checks: Vec<Check>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load: Option<LoadReport>,
    /// True when no check failed.
    pub ok: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Machine {
    /// Model identifier, such as `Mac16,6`.
    pub model: Option<String>,
    pub chip: Option<String>,
    pub os_version: Option<String>,
    pub os_build: Option<String>,
    /// Running as root, so "readable without root" could not be verified.
    pub root: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct GpuInfo {
    pub model: Option<String>,
    pub cores: Option<u32>,
    pub generation: Option<u32>,
    pub variant: Option<String>,
    pub gps: Option<u32>,
    pub mgpus: Option<u32>,
    pub driver_version: Option<String>,
    pub clients: usize,
}

/// Generic health of one source, from [`Report::sources`].
#[derive(Debug, Clone, Serialize)]
pub struct SourceCheck {
    pub name: String,
    pub state: Option<SourceState>,
    /// `None` when unknown (for example a disabled source).
    pub exists: Option<bool>,
    pub readable_without_root: Option<bool>,
    pub updating: Option<bool>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Pass,
    Warn,
    Fail,
    Skip,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Warn => "warn",
            Status::Fail => "FAIL",
            Status::Skip => "skip",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    /// The source the check is about, or `load` for the load verification.
    pub scope: String,
    pub name: String,
    pub status: Status,
    pub detail: String,
}

impl Check {
    fn new(scope: &str, name: &str, status: Status, detail: impl Into<String>) -> Self {
        Self {
            scope: scope.to_string(),
            name: name.to_string(),
            status,
            detail: detail.into(),
        }
    }
}

/// What the load verification measured.
#[derive(Debug, Clone, Serialize)]
pub struct LoadReport {
    pub pid: i32,
    pub config: LoadConfig,
    /// The load's own measurements: the ground truth.
    pub measured: LoadResult,
    pub idle: WindowSummary,
    pub load: WindowSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub powermetrics: Option<PowerStats>,
}

/// The numbers the load checks compare, for one window.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WindowSummary {
    pub window_seconds: f64,
    /// GPU time macgpu attributed to the load process.
    pub load_gpu_ns: u64,
    /// GPU time of every other process in the window.
    pub other_gpu_ns: u64,
    pub active_pct: Option<f64>,
    pub power_w: Option<f64>,
    pub temperature_max_c: Option<f64>,
    pub load_graphics_memory_bytes: Option<u64>,
}

impl WindowSummary {
    fn of(report: &Report, pid: i32) -> Self {
        let load = report.processes.iter().find(|process| process.pid == pid);
        let load_gpu_ns = load.map(|process| process.gpu_ns).unwrap_or(0);
        Self {
            window_seconds: report.window_seconds,
            load_gpu_ns,
            other_gpu_ns: report.totals.attributed_gpu_ns.saturating_sub(load_gpu_ns),
            active_pct: report.device.active_pct,
            power_w: report.device.power_w,
            temperature_max_c: report
                .device
                .temperature
                .as_ref()
                .map(|temperature| temperature.max_c),
            load_graphics_memory_bytes: load.and_then(|process| process.graphics_memory_bytes),
        }
    }
}

pub fn run(args: &DoctorArgs) -> Result<i32> {
    let report = diagnose(args)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render(&report));
    }
    Ok(if report.ok { 0 } else { 1 })
}

/// Runs every check. The load, when requested, runs `current_exe() __load`.
pub fn diagnose(args: &DoctorArgs) -> Result<DoctorReport> {
    let options = AnalyzeOptions { include_idle: true };
    let mut sampler = Sampler::new(SourceSet::default());
    let first = sampler.snapshot();
    std::thread::sleep(SOURCE_WINDOW);
    let second = sampler.snapshot();
    let report = analyze(&first, &second, &options);

    let mut checks = Vec::new();
    let sources = source_checks(&report, &first, &second);
    registry_checks(&report, &second, &mut checks);
    ioreport_checks(&report, &mut checks);
    coalition_checks(&report, &mut checks);
    memory_checks(&report, &mut checks);
    temperature_checks(&report, &mut checks);
    system_checks(&report, &mut checks);

    let registry_ok = state_of(&report, source::REGISTRY) == Some(SourceState::Ok);
    let load = if !args.load {
        None
    } else if !registry_ok {
        checks.push(Check::new(
            "load",
            "load",
            Status::Skip,
            "the GPU registry is not readable",
        ));
        None
    } else {
        // Another controlled load (a test, or a second doctor) would skew
        // every check, so wait for it unless the caller holds the lock.
        let _lock = match std::env::var_os(GPU_LOCK_HELD_ENV) {
            Some(_) => None,
            None => Some(GpuLoadLock::acquire_or_wait()?),
        };
        Some(run_load(args, &mut sampler, &mut checks)?)
    };

    let ok = !checks.iter().any(|check| check.status == Status::Fail);
    Ok(DoctorReport {
        macgpu_version: env!("CARGO_PKG_VERSION").to_string(),
        machine: machine(),
        gpus: gpus(&second),
        sources,
        checks,
        load,
        ok,
    })
}

fn machine() -> Machine {
    Machine {
        model: sysctl_string("hw.model"),
        chip: sysctl_string("machdep.cpu.brand_string"),
        os_version: sysctl_string("kern.osproductversion"),
        os_build: sysctl_string("kern.osversion"),
        // SAFETY: geteuid takes no arguments and cannot fail.
        root: unsafe { libc::geteuid() } == 0,
    }
}

fn sysctl_string(name: &str) -> Option<String> {
    let name = CString::new(name).ok()?;
    let mut size = 0usize;
    // SAFETY: a null buffer asks sysctlbyname for the value's size.
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 || size == 0 {
        return None;
    }
    let mut buffer = vec![0u8; size];
    // SAFETY: `buffer` holds `size` bytes, as sysctlbyname reported.
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buffer.as_mut_ptr().cast::<c_void>(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 {
        return None;
    }
    buffer.truncate(size);
    let end = buffer.iter().position(|byte| *byte == 0).unwrap_or(size);
    let text = String::from_utf8_lossy(&buffer[..end]).trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn gpus(snapshot: &Snapshot) -> Vec<GpuInfo> {
    let Ok(registry) = &snapshot.registry else {
        return Vec::new();
    };
    registry
        .accelerators
        .iter()
        .map(|accelerator| GpuInfo {
            model: accelerator.info.model.clone(),
            cores: accelerator.info.core_count,
            generation: accelerator.info.generation,
            variant: accelerator.info.variant.clone(),
            gps: accelerator.info.num_gps,
            mgpus: accelerator.info.num_mgpus,
            driver_version: accelerator.info.driver_version.clone(),
            clients: accelerator.clients.len(),
        })
        .collect()
}

fn state_of(report: &Report, name: &str) -> Option<SourceState> {
    report.sources.get(name).map(|status| status.state)
}

fn source_ok(report: &Report, name: &str) -> bool {
    matches!(
        state_of(report, name),
        Some(SourceState::Ok | SourceState::Partial)
    )
}

const SOURCE_ORDER: [&str; 6] = [
    source::REGISTRY,
    source::IOREPORT,
    source::COALITIONS,
    source::MEMORY,
    source::TEMPERATURE,
    source::SYSTEM,
];

fn source_checks(report: &Report, first: &Snapshot, second: &Snapshot) -> Vec<SourceCheck> {
    let mut names: Vec<String> = SOURCE_ORDER.iter().map(|name| name.to_string()).collect();
    names.extend(
        report
            .sources
            .keys()
            .filter(|name| !SOURCE_ORDER.contains(&name.as_str()))
            .cloned(),
    );
    names
        .iter()
        .map(|name| {
            let mut check = source_check(name, report.sources.get(name));
            if check.updating == Some(true) {
                check.updating = updating(name, report, first, second);
            }
            check
        })
        .collect()
}

/// Maps a reported [`SourceStatus`] to exists / readable / updating.
pub fn source_check(name: &str, status: Option<&SourceStatus>) -> SourceCheck {
    let (exists, readable, updating) = match status.map(|status| status.state) {
        Some(SourceState::Ok | SourceState::Partial) => (Some(true), Some(true), Some(true)),
        Some(SourceState::Stale) => (Some(true), Some(true), Some(false)),
        Some(SourceState::Denied) => (Some(true), Some(false), None),
        Some(SourceState::Unavailable) => (Some(false), None, None),
        Some(SourceState::Disabled) | None => (None, None, None),
    };
    SourceCheck {
        name: name.to_string(),
        state: status.map(|status| status.state),
        exists,
        readable_without_root: readable,
        updating,
        detail: match status {
            Some(status) => status.detail.clone(),
            None => Some("not reported".to_string()),
        },
    }
}

/// Source-specific evidence that counters advanced over the window. `None`
/// when the idle machine gives no evidence either way.
fn updating(name: &str, report: &Report, first: &Snapshot, second: &Snapshot) -> Option<bool> {
    match name {
        source::REGISTRY => {
            let lifetime = |snapshot: &Snapshot| -> u64 {
                snapshot
                    .registry
                    .as_ref()
                    .map(|registry| {
                        registry
                            .clients()
                            .map(|client| client.accumulated_ns())
                            .fold(0u64, u64::saturating_add)
                    })
                    .unwrap_or(0)
            };
            (lifetime(first) != lifetime(second) || report.totals.attributed_gpu_ns > 0)
                .then_some(true)
        }
        source::IOREPORT => {
            Some(!report.device.pstates.is_empty() || report.device.power_w.is_some())
        }
        source::COALITIONS => (!report.apps.is_empty()).then_some(true),
        source::TEMPERATURE => Some(report.device.temperature.is_some()),
        _ => Some(true),
    }
}

/// Generic warning for a non-core source that is not fully working.
fn degraded_source(report: &Report, name: &str, checks: &mut Vec<Check>) -> bool {
    let Some(status) = report.sources.get(name) else {
        checks.push(Check::new(name, "reported", Status::Warn, "not reported"));
        return true;
    };
    let detail = status.detail.clone().unwrap_or_default();
    let degraded = match status.state {
        SourceState::Ok | SourceState::Partial => return false,
        SourceState::Disabled => Check::new(name, "available", Status::Skip, "disabled"),
        SourceState::Stale => Check::new(name, "updating", Status::Warn, detail),
        SourceState::Denied => Check::new(name, "readable", Status::Warn, detail),
        SourceState::Unavailable => Check::new(name, "available", Status::Warn, detail),
    };
    checks.push(degraded);
    true
}

fn registry_checks(report: &Report, snapshot: &Snapshot, checks: &mut Vec<Check>) {
    let scope = source::REGISTRY;
    let registry = match &snapshot.registry {
        Ok(registry) => registry,
        Err(error) => {
            checks.push(Check::new(scope, "readable", Status::Fail, error.clone()));
            return;
        }
    };
    if registry.accelerators.is_empty() {
        checks.push(Check::new(
            scope,
            "accelerator",
            Status::Fail,
            "no IOAccelerator service found",
        ));
        return;
    }
    let clients = registry.clients().count();
    let with_queues = registry
        .clients()
        .filter(|client| !client.queues.is_empty())
        .count();
    checks.push(Check::new(
        scope,
        "clients",
        if clients > 0 {
            Status::Pass
        } else {
            Status::Fail
        },
        format!(
            "{clients} GPU clients, {with_queues} with AppUsage entries; one sample of every source took {:.2} ms",
            report.overhead.sample_wall_ms
        ),
    ));
    let own_pid = std::process::id() as i32;
    let own = registry
        .clients()
        .filter(|client| client.pid == own_pid)
        .count();
    checks.push(if own == 0 {
        Check::new(
            scope,
            "observer",
            Status::Pass,
            "macgpu itself holds no GPU client",
        )
    } else {
        Check::new(
            scope,
            "observer",
            Status::Warn,
            format!("macgpu itself holds {own} GPU client(s); it should not open a Metal device"),
        )
    });
}

fn ioreport_checks(report: &Report, checks: &mut Vec<Check>) {
    let scope = source::IOREPORT;
    if degraded_source(report, scope, checks) {
        return;
    }
    let device = &report.device;
    if device.pstates.is_empty() {
        checks.push(Check::new(
            scope,
            "residency",
            Status::Warn,
            "no GPU P-state residency",
        ));
    } else {
        let total: f64 = device.pstates.iter().map(|state| state.pct).sum();
        checks.push(Check::new(
            scope,
            "residency",
            if (total - 100.0).abs() <= 2.0 {
                Status::Pass
            } else {
                Status::Warn
            },
            format!(
                "{} P-states sum to {total:.1}% of the window; active {}",
                device.pstates.len(),
                optional_pct(device.active_pct)
            ),
        ));
    }
    checks.push(match device.power_w {
        Some(power) if power >= 0.0 => Check::new(
            scope,
            "power",
            Status::Pass,
            format!("{} idle", format::watts(power)),
        ),
        Some(power) => Check::new(scope, "power", Status::Warn, format!("negative: {power} W")),
        None => Check::new(scope, "power", Status::Warn, "no GPU energy counter"),
    });
    let active: Vec<Option<f64>> = device
        .pstates
        .iter()
        .filter(|state| !INACTIVE_PSTATES.contains(&state.name.as_str()))
        .map(|state| state.mhz)
        .collect();
    checks.push(frequency_table_check(&active, device.max_mhz));
}

/// GPUPH states that are not active P-states.
const INACTIVE_PSTATES: [&str; 3] = ["OFF", "IDLE", "DOWN"];

/// Sanity checks on the clock of each active P-state. The table is the
/// hardware DVFS table sorted by voltage, so it is not increasing (on an M4
/// Max: 1182 then 1312 MHz share a voltage, then 1242 and 1380); that is
/// expected. Every active state needs a clock in a plausible range, and the
/// reported maximum must be the table's.
pub fn frequency_table_check(active: &[Option<f64>], max_mhz: Option<f64>) -> Check {
    let scope = source::IOREPORT;
    if active.is_empty() {
        return Check::new(scope, "frequencies", Status::Warn, "no active P-states");
    }
    let table: Vec<f64> = active.iter().flatten().copied().collect();
    if table.len() != active.len() {
        return Check::new(
            scope,
            "frequencies",
            Status::Warn,
            format!(
                "clocks for {} of {} active P-states; the frequency table does not match GPUPH, so MHz are hidden",
                table.len(),
                active.len()
            ),
        );
    }
    let listed = table
        .iter()
        .map(|mhz| format!("{mhz:.0}"))
        .collect::<Vec<_>>()
        .join(" ");
    let plausible = table.iter().all(|mhz| (50.0..=5000.0).contains(mhz));
    let table_max = table.iter().copied().fold(f64::MIN, f64::max);
    let max_matches = max_mhz.is_some_and(|max| (max - table_max).abs() <= 0.5);
    if plausible && max_matches {
        Check::new(
            scope,
            "frequencies",
            Status::Pass,
            format!("{} P-states, MHz {listed}", table.len()),
        )
    } else {
        Check::new(
            scope,
            "frequencies",
            Status::Warn,
            format!("implausible table (max {max_mhz:?}): MHz {listed}"),
        )
    }
}

fn coalition_checks(report: &Report, checks: &mut Vec<Check>) {
    let scope = source::COALITIONS;
    if degraded_source(report, scope, checks) {
        return;
    }
    let with_energy = report
        .apps
        .iter()
        .filter(|app| app.energy_j.is_some())
        .count();
    checks.push(Check::new(
        scope,
        "apps",
        if report.apps.is_empty() {
            Status::Warn
        } else {
            Status::Pass
        },
        format!(
            "{} apps with GPU activity, {with_energy} with GPU energy",
            report.apps.len()
        ),
    ));
}

fn memory_checks(report: &Report, checks: &mut Vec<Check>) {
    let scope = source::MEMORY;
    if degraded_source(report, scope, checks) {
        return;
    }
    let readable = report
        .processes
        .iter()
        .filter(|process| process.graphics_memory_bytes.is_some())
        .count();
    checks.push(Check::new(
        scope,
        "processes",
        if readable > 0 {
            Status::Pass
        } else {
            Status::Warn
        },
        format!(
            "graphics memory readable for {readable} of {} GPU processes (others need root)",
            report.processes.len()
        ),
    ));
}

fn temperature_checks(report: &Report, checks: &mut Vec<Check>) {
    let scope = source::TEMPERATURE;
    if degraded_source(report, scope, checks) {
        return;
    }
    checks.push(match &report.device.temperature {
        Some(temperature)
            if temperature.sensors > 0 && (5.0..=130.0).contains(&temperature.avg_c) =>
        {
            Check::new(
                scope,
                "sensors",
                Status::Pass,
                format!(
                    "{} GPU sensors, average {:.1} C, max {:.1} C",
                    temperature.sensors, temperature.avg_c, temperature.max_c
                ),
            )
        }
        Some(temperature) => Check::new(
            scope,
            "sensors",
            Status::Warn,
            format!(
                "implausible: {} sensors, average {:.1} C",
                temperature.sensors, temperature.avg_c
            ),
        ),
        None => Check::new(scope, "sensors", Status::Warn, "no GPU temperature"),
    });
}

fn system_checks(report: &Report, checks: &mut Vec<Check>) {
    let scope = source::SYSTEM;
    if degraded_source(report, scope, checks) {
        return;
    }
    let system = &report.device.system;
    let memory = &report.device.memory;
    let mut found = Vec::new();
    if let Some(level) = &system.thermal_pressure {
        found.push(format!("thermal pressure {level}"));
    }
    if let Some(level) = &system.memory_pressure {
        found.push(format!("memory pressure {level}"));
    }
    if let Some(limit) = memory.wired_limit_bytes {
        found.push(format!("wired limit {}", format::bytes(limit)));
    }
    checks.push(Check::new(
        scope,
        "context",
        if found.is_empty() {
            Status::Warn
        } else {
            Status::Pass
        },
        if found.is_empty() {
            "no thermal, memory pressure, or wired-limit readings".to_string()
        } else {
            found.join(", ")
        },
    ));
}

mod load;

use load::run_load;
pub use load::{GPU_LOCK_HELD_ENV, GpuLoadLock, coalition_gpu_energy_j};
#[cfg(test)]
pub(crate) use load::{
    attribution_check, coalition_energy_check, full_load_clock_check, powermetrics_checks,
};

fn optional_pct(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.1}%"))
}

fn yes_no(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "-",
    }
}

fn state_label(state: Option<SourceState>) -> &'static str {
    match state {
        Some(SourceState::Ok) => "ok",
        Some(SourceState::Partial) => "partial",
        Some(SourceState::Stale) => "stale",
        Some(SourceState::Denied) => "denied",
        Some(SourceState::Unavailable) => "unavailable",
        Some(SourceState::Disabled) => "disabled",
        None => "missing",
    }
}

pub fn render(report: &DoctorReport) -> String {
    let mut out = String::new();
    let machine = &report.machine;
    let unknown = || "?".to_string();
    let _ = writeln!(out, "macgpu {} doctor", report.macgpu_version);
    let _ = writeln!(
        out,
        "machine  {}, {}, macOS {} ({}){}",
        machine.model.clone().unwrap_or_else(unknown),
        machine.chip.clone().unwrap_or_else(unknown),
        machine.os_version.clone().unwrap_or_else(unknown),
        machine.os_build.clone().unwrap_or_else(unknown),
        if machine.root {
            ", running as root: no-root access not verified"
        } else {
            ""
        }
    );
    if report.gpus.is_empty() {
        let _ = writeln!(out, "gpu      not found");
    }
    for gpu in &report.gpus {
        let mut parts = vec![gpu.model.clone().unwrap_or_else(unknown)];
        if let Some(cores) = gpu.cores {
            parts.push(format!("{cores} cores"));
        }
        if let (Some(generation), Some(variant)) = (gpu.generation, &gpu.variant) {
            parts.push(format!("gen {generation}{variant}"));
        }
        if let (Some(gps), Some(mgpus)) = (gpu.gps, gpu.mgpus) {
            parts.push(format!("{gps} GPs, {mgpus} mGPUs"));
        }
        if let Some(driver) = &gpu.driver_version {
            parts.push(format!("driver {driver}"));
        }
        parts.push(format!("{} clients", gpu.clients));
        let _ = writeln!(out, "gpu      {}", parts.join(", "));
    }

    let _ = writeln!(
        out,
        "\n{:<12} {:<12} {:<7} {:<8} {:<9} DETAIL",
        "SOURCE", "STATE", "EXISTS", "NO-ROOT", "UPDATING"
    );
    for source in &report.sources {
        let _ = writeln!(
            out,
            "{:<12} {:<12} {:<7} {:<8} {:<9} {}",
            source.name,
            state_label(source.state),
            yes_no(source.exists),
            yes_no(source.readable_without_root),
            yes_no(source.updating),
            source.detail.as_deref().unwrap_or("")
        );
    }

    let _ = writeln!(out);
    let mut in_load = false;
    for check in &report.checks {
        if check.scope == "load" && !in_load {
            in_load = true;
            match &report.load {
                Some(load) => {
                    let _ = writeln!(
                        out,
                        "\nload: {:.1} s in pid {}, {} buffer, load's own GPU time {:.1} ms ({:.0}% busy)",
                        load.config.seconds,
                        load.pid,
                        format::bytes(load.measured.alloc_bytes),
                        load.measured.gpu_ns as f64 / 1e6,
                        load.measured.busy_fraction() * 100.0
                    );
                }
                None => {
                    let _ = writeln!(out, "\nload:");
                }
            }
        }
        let scope = if check.scope == "load" {
            ""
        } else {
            check.scope.as_str()
        };
        let _ = writeln!(
            out,
            "{}  {:<11} {:<24} {}",
            check.status.label(),
            scope,
            check.name,
            check.detail
        );
    }

    let count = |status: Status| {
        report
            .checks
            .iter()
            .filter(|check| check.status == status)
            .count()
    };
    let _ = writeln!(
        out,
        "\n{}: {} passed, {} warnings, {} failed, {} skipped{}",
        if report.ok { "ok" } else { "FAILED" },
        count(Status::Pass),
        count(Status::Warn),
        count(Status::Fail),
        count(Status::Skip),
        if report.load.is_none() {
            "; run `macgpu doctor --load` to verify readings against a GPU load"
        } else {
            ""
        }
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{PStateShare, Temperature};

    const MS: u64 = 1_000_000;

    fn truth(gpu_ms: u64, wall_ms: u64) -> LoadResult {
        LoadResult {
            gpu_ns: gpu_ms * MS,
            wall_ns: wall_ms * MS,
            ..LoadResult::default()
        }
    }

    #[test]
    fn attribution_passes_within_tolerance() {
        let check = attribution_check(Some(2_716 * MS), &truth(2_722, 3_000), 130 * MS, 3_000 * MS);
        assert_eq!(check.status, Status::Pass, "{}", check.detail);
    }

    #[test]
    fn attribution_allows_for_other_gpu_work() {
        // 1.1 s of other GPU work stretched the load's command buffers.
        let check = attribution_check(
            Some(1_668 * MS),
            &truth(2_682, 3_000),
            1_132 * MS,
            3_010 * MS,
        );
        assert_eq!(check.status, Status::Pass, "{}", check.detail);
        assert!(
            check.detail.contains("other processes used"),
            "{}",
            check.detail
        );
    }

    #[test]
    fn attribution_fails_when_unexplained() {
        let load = truth(2_760, 3_000);
        assert_eq!(
            attribution_check(Some(1_000 * MS), &load, 100 * MS, 3_000 * MS).status,
            Status::Fail
        );
        // More than physically possible: the lifetime-sized delta bug.
        assert_eq!(
            attribution_check(Some(9_000 * MS), &load, 0, 3_000 * MS).status,
            Status::Fail
        );
        assert_eq!(
            attribution_check(None, &load, 0, 3_000 * MS).status,
            Status::Fail
        );
    }

    #[test]
    fn maps_source_states() {
        let ok = source_check("registry", Some(&SourceStatus::ok()));
        assert_eq!(
            (ok.exists, ok.readable_without_root, ok.updating),
            (Some(true), Some(true), Some(true))
        );
        let stale = source_check(
            "ioreport",
            Some(&SourceStatus::with(SourceState::Stale, "frozen")),
        );
        assert_eq!(stale.updating, Some(false));
        assert_eq!(stale.detail.as_deref(), Some("frozen"));
        let denied = source_check(
            "memory",
            Some(&SourceStatus::with(SourceState::Denied, "root")),
        );
        assert_eq!(
            (denied.exists, denied.readable_without_root),
            (Some(true), Some(false))
        );
        let missing = source_check("smc", None);
        assert_eq!(missing.exists, None);
    }

    #[test]
    fn frequency_table_sanity() {
        let m4_max = [338.0, 618.0, 1182.0, 1312.0, 1242.0, 1578.0].map(Some);
        assert_eq!(
            frequency_table_check(&m4_max, Some(1578.0)).status,
            Status::Pass
        );
        assert_eq!(
            frequency_table_check(&[Some(338.0), None], Some(338.0)).status,
            Status::Warn
        );
        assert_eq!(
            frequency_table_check(&[Some(0.0), Some(1578.0)], Some(1578.0)).status,
            Status::Warn
        );
        assert_eq!(
            frequency_table_check(&[Some(338.0), Some(1578.0)], Some(1600.0)).status,
            Status::Warn
        );
        assert_eq!(frequency_table_check(&[], None).status, Status::Warn);
    }

    #[test]
    fn coalition_energy_tolerates_other_gpu_work() {
        assert_eq!(
            coalition_energy_check(129.5, 130.1, true).status,
            Status::Pass
        );
        assert_eq!(
            coalition_energy_check(125.0, 130.0, true).status,
            Status::Warn
        );
        assert_eq!(
            coalition_energy_check(152.8, 168.3, true).status,
            Status::Fail
        );
        assert_eq!(
            coalition_energy_check(152.8, 168.3, false).status,
            Status::Warn
        );
    }

    #[test]
    fn full_load_clock_reaches_table_max() {
        let check = |avg, busy| full_load_clock_check(avg, Some(1578.0), busy, None).status;
        assert_eq!(check(Some(1577.9), 99.0), Status::Pass);
        assert_eq!(check(Some(1200.0), 99.0), Status::Warn);
        assert_eq!(check(Some(1700.0), 99.0), Status::Fail);
        assert_eq!(check(Some(600.0), 20.0), Status::Skip);
        assert_eq!(check(None, 99.0), Status::Skip);
    }

    #[test]
    fn powermetrics_cross_check() {
        let stats = powermetrics::parse(
            "GPU HW active residency:  98.00% (338 MHz: 0% 618 MHz: 1% 1578 MHz: 97%)\nGPU Power: 40000 mW\n",
        );
        let mut window = Report::default();
        window.device.active_pct = Some(97.0);
        window.device.power_w = Some(41.0);
        window.device.pstates = [None, Some(338.0), Some(618.0), Some(1578.0)]
            .into_iter()
            .map(|mhz| PStateShare {
                name: String::new(),
                mhz,
                pct: 0.0,
            })
            .collect();
        let checks = powermetrics_checks(&stats, &window);
        assert!(
            checks.iter().all(|check| check.status == Status::Pass),
            "{checks:?}"
        );

        window.device.pstates.pop();
        let checks = powermetrics_checks(&stats, &window);
        assert_eq!(checks[2].status, Status::Fail);
    }

    #[test]
    fn renders_compact_text() {
        let mut report = Report::default();
        report.device.temperature = Some(Temperature {
            avg_c: 60.0,
            max_c: 64.0,
            sensors: 44,
        });
        let doctor = DoctorReport {
            macgpu_version: "0.0.0".into(),
            machine: Machine::default(),
            gpus: vec![GpuInfo {
                model: Some("Apple M4 Max".into()),
                cores: Some(40),
                clients: 70,
                ..GpuInfo::default()
            }],
            sources: vec![source_check("registry", Some(&SourceStatus::ok()))],
            checks: vec![
                Check::new("registry", "clients", Status::Pass, "70 GPU clients"),
                attribution_check(Some(990 * MS), &truth(1_000, 1_100), 0, 1_100 * MS),
            ],
            load: None,
            ok: true,
        };
        let text = render(&doctor);
        assert!(
            text.contains("gpu      Apple M4 Max, 40 cores, 70 clients"),
            "{text}"
        );
        assert!(text.contains("registry     ok"), "{text}");
        assert!(text.contains("pass  registry"), "{text}");
        assert!(text.contains("\nload:\n"), "{text}");
        assert!(text.contains("ok: 2 passed"), "{text}");
    }
}
