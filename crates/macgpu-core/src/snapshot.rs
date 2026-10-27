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

impl Snapshot {
    /// GPU-client PIDs in this snapshot.
    pub fn gpu_pids(&self) -> BTreeSet<i32> {
        match &self.registry {
            Ok(registry) => registry.clients().map(|client| client.pid).collect(),
            Err(_) => BTreeSet::new(),
        }
    }
}

/// Holds every reader and its long-lived state (subscriptions, connections,
/// caches). Readers hold CoreFoundation objects, so a `Sampler` stays on the
/// thread that created it.
pub struct Sampler {
    sources: SourceSet,
    registry: RegistryReader,
    processes: ProcessCache,
    ioreport: Option<Result<IoReportReader, String>>,
    coalitions: Option<CoalitionReader>,
    memory: Option<MemoryReader>,
    smc: Option<Result<SmcReader, String>>,
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
                .then(|| IoReportReader::new().map_err(|error| format!("{error:#}"))),
            coalitions: sources.coalitions.then(CoalitionReader::new),
            memory: sources.memory.then(MemoryReader::new),
            smc: sources
                .temperature
                .then(|| SmcReader::new().map_err(|error| format!("{error:#}"))),
            frequencies: if sources.ioreport {
                gpu_freq::read()
            } else {
                None
            },
            firmware_readings: Vec::new(),
        }
    }

    /// Takes one cheap firmware utilization reading between snapshots. The
    /// readings are averaged into the next report.
    pub fn poll_firmware(&mut self) {
        if let Ok(readings) = self.registry.read_performance() {
            self.firmware_readings.extend(readings);
        }
    }

    pub fn sources(&self) -> SourceSet {
        self.sources
    }

    pub fn snapshot(&mut self) -> Snapshot {
        let started = Instant::now();
        let started_cpu = clock::thread_cpu_ns();

        let registry = self.registry.read().map_err(|error| format!("{error:#}"));
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

        let ioreport = self.ioreport.as_mut().map(|reader| match reader {
            Ok(reader) => reader.sample().map_err(|error| format!("{error:#}")),
            Err(error) => Err(error.clone()),
        });
        let coalitions = self
            .coalitions
            .as_mut()
            .map(|reader| reader.read(&gpu_pids).map_err(|error| format!("{error:#}")));
        let memory = self.memory.as_mut().map(|reader| reader.read(&gpu_pids));
        let temperatures = self.smc.as_mut().map(|reader| match reader {
            Ok(reader) => reader.read().map_err(|error| format!("{error:#}")),
            Err(error) => Err(error.clone()),
        });
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
