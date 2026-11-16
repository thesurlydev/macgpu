//! One raw sample of every source, and the [`Sampler`] that takes it.
//!
//! A [`Snapshot`] holds cumulative counters as plain Rust data. Reports are
//! computed from two snapshots by [`crate::analyze`], which is pure and unit
//! testable.

use crate::sys::clock;
use crate::sys::coalition::{CoalitionReader, CoalitionTable};
use crate::sys::gpu_freq::{self, FrequencyTable};
use crate::sys::iokit::{PerformanceStatistics, Registry, RegistryReader};
use crate::sys::ioreport::{IoReportReader, IoReportSample};
use crate::sys::process::{ProcessCache, ProcessInfo};
use crate::sys::smc::{GpuTemperatures, SmcReader};
use crate::sys::system::{self, SystemContext};
use crate::sys::task_memory::{MemoryReader, MemoryTable};
use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

/// Which optional sources to read. The GPU registry is always read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSet {
    /// IOReport: P-states, frequency, power, limiters.
    pub ioreport: bool,
    /// Resource coalitions: per-app GPU time and energy.
    pub coalitions: bool,
    /// Per-process graphics memory.
    pub memory: bool,
    /// SMC GPU temperature.
    pub temperature: bool,
    /// Thermal pressure, memory pressure, wired-memory limits.
    pub system: bool,
}

impl Default for SourceSet {
    fn default() -> Self {
        Self {
            ioreport: true,
            coalitions: true,
            memory: true,
            temperature: true,
            system: true,
        }
    }
}

/// A source result. Errors are kept as strings so snapshots stay `Clone`.
pub type SourceResult<T> = Result<T, String>;

/// Keeps the full error chain in a source's detail string.
fn error_detail(error: impl std::fmt::Display) -> String {
    format!("{error:#}")
}

/// Takes a sample from a cached reader, preserving its construction error.
fn read_cached<T, R>(
    reader: &mut SourceResult<T>,
    read: impl FnOnce(&mut T) -> SourceResult<R>,
) -> SourceResult<R> {
    match reader {
        Ok(reader) => read(reader),
        Err(error) => Err(error.clone()),
    }
}

/// Cumulative counters from every enabled source at one instant.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub at: Instant,
    pub unix_ms: u64,
    /// `mach_absolute_time` in ns, to age `lastSubmittedTime` values.
    pub mach_now_ns: u64,
    pub registry: SourceResult<Registry>,
    /// Firmware utilization readings polled since the previous snapshot,
    /// ending with this snapshot's own reading.
    pub firmware_readings: Vec<PerformanceStatistics>,
    /// Process metadata for every PID that owns a GPU client.
    pub processes: HashMap<i32, ProcessInfo>,
    /// `None` when the source is disabled.
    pub ioreport: Option<SourceResult<IoReportSample>>,
    pub coalitions: Option<SourceResult<CoalitionTable>>,
    pub memory: Option<MemoryTable>,
    pub temperatures: Option<SourceResult<GpuTemperatures>>,
    pub system: Option<SystemContext>,
    /// Static GPU frequency table, read once.
    pub frequencies: Option<FrequencyTable>,
    /// Cost of taking this snapshot.
    pub cost_wall_ns: u64,
    pub cost_cpu_ns: u64,
    /// Whole-process CPU time of macgpu when the snapshot finished.
    pub process_cpu_ns: u64,
}

/// Holds every reader and its long-lived state (subscriptions, connections,
/// caches). Readers hold CoreFoundation objects, so a `Sampler` stays on the
/// thread that created it.
pub struct Sampler {
    sources: SourceSet,
    registry: RegistryReader,
    processes: ProcessCache,
    ioreport: Option<SourceResult<IoReportReader>>,
    coalitions: Option<CoalitionReader>,
    memory: Option<MemoryReader>,
    smc: Option<SourceResult<SmcReader>>,
    frequencies: Option<FrequencyTable>,
    firmware_readings: Vec<PerformanceStatistics>,
}

impl Sampler {
    pub fn new(sources: SourceSet) -> Self {
        Self {
            sources,
            registry: RegistryReader::new(),
            processes: ProcessCache::new(),
            ioreport: sources
                .ioreport
                .then(|| IoReportReader::new().map_err(error_detail)),
            coalitions: sources.coalitions.then(CoalitionReader::new),
            memory: sources.memory.then(MemoryReader::new),
            smc: sources
                .temperature
                .then(|| SmcReader::new().map_err(error_detail)),
            frequencies: if sources.ioreport {
                gpu_freq::read()
            } else {
                None
            },
            firmware_readings: Vec::new(),
        }
    }

    /// Takes one cheap firmware utilization reading between snapshots. The
    /// readings are averaged into the next report. A failed poll is recorded
    /// by the next snapshot's full registry read, in `Snapshot::registry`.
    pub fn poll_firmware(&mut self) {
        if let Ok(readings) = self.registry.read_performance() {
            self.firmware_readings.extend(readings);
        }
    }

    pub fn snapshot(&mut self) -> Snapshot {
        let started = Instant::now();
        let started_cpu = clock::thread_cpu_ns();

        let registry = self.registry.read().map_err(error_detail);
        // The window boundary is the registry read: AppUsage is the primary
        // counter, and later sources (notably SMC) can take several ms.
        let at = Instant::now();
        let unix_ms = clock::unix_ms();
        let mach_now_ns = clock::mach_now_ns();
        let mut firmware_readings = std::mem::take(&mut self.firmware_readings);
        if let Ok(registry) = &registry {
            firmware_readings.extend(
                registry
                    .accelerators
                    .iter()
                    .map(|accelerator| accelerator.performance.clone()),
            );
        }

        let gpu_pids: Vec<i32> = match &registry {
            Ok(registry) => {
                let pids: BTreeSet<i32> = registry.clients().map(|client| client.pid).collect();
                pids.into_iter().collect()
            }
            Err(_) => Vec::new(),
        };
        let mut processes = HashMap::with_capacity(gpu_pids.len());
        for pid in &gpu_pids {
            if let Some(info) = self.processes.get(*pid) {
                processes.insert(*pid, info);
            }
        }
        self.processes.retain(&|pid| processes.contains_key(&pid));

        let ioreport = self
            .ioreport
            .as_mut()
            .map(|reader| read_cached(reader, |reader| reader.sample().map_err(error_detail)));
        let coalitions = self
            .coalitions
            .as_mut()
            .map(|reader| reader.read(&gpu_pids).map_err(error_detail));
        let memory = self.memory.as_mut().map(|reader| reader.read(&gpu_pids));
        let temperatures = self
            .smc
            .as_mut()
            .map(|reader| read_cached(reader, |reader| reader.read().map_err(error_detail)));
        let system = self.sources.system.then(system::read);

        Snapshot {
            at,
            unix_ms,
            mach_now_ns,
            registry,
            firmware_readings,
            processes,
            ioreport,
            coalitions,
            memory,
            temperatures,
            system,
            frequencies: self.frequencies.clone(),
            cost_wall_ns: started.elapsed().as_nanos() as u64,
            cost_cpu_ns: clock::thread_cpu_ns().saturating_sub(started_cpu),
            process_cpu_ns: clock::process_cpu_ns(),
        }
    }
}
