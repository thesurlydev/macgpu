//! Resource coalitions: per-app GPU time and GPU energy for every process,
//! without root, including tasks that have exited.
//!
//! A resource coalition is an app plus its helpers, XPC services, and child
//! processes; Activity Monitor's Energy tab groups by it. The kernel keeps
//! cumulative counters per coalition (`struct coalition_resource_usage`) that
//! only ever increase and keep the time of tasks that have exited. Every
//! interface used here works unprivileged for every process:
//!
//! - `proc_listcoalitions` lists every coalition and its task count in one call.
//! - `proc_pidinfo(PROC_PIDCOALITIONINFO)` maps a PID to its coalition.
//! - `coalition_info_resource_usage` reads one coalition's counters.
//! - `responsibility_get_pid_responsible_for_pid` names the app a process
//!   belongs to (private, but exported by libSystem).
//!
//! Cost: a read sums every member's ledgers, so the reader reads on every
//! sample only the coalitions that have a member owning a GPU client or whose
//! GPU, ANE or energy counters moved at their last read.
//! Other coalitions with tasks are read every [`COLD_REFRESH`], staggered
//! across samples, and an empty coalition once more after its last task exits
//! (nothing can charge it after that). PID to
//! coalition and PID to responsible PID are cached while the PID stays in
//! consecutive listings: a process never changes coalition, and xnu allocates
//! PIDs sequentially up to 99,999, so a PID seen in two consecutive listings is
//! the same process.

use super::clock;
use super::process::{self, ProcessInfo};
use anyhow::{Result, bail};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_int, c_void};
use std::time::{Duration, Instant};

/// How often a coalition is re-read when it has no member that owns a GPU
/// client and its counters did not move at its last read.
///
/// GPU time such a coalition starts accumulating without owning a GPU client
/// at sample time (a short-lived helper that exits inside a window) is
/// credited to the window in which it is next read, at most this much later,
/// so totals are conserved.
pub const COLD_REFRESH: Duration = Duration::from_secs(5);

/// `PROC_PIDCOALITIONINFO` from `<sys/proc_info.h>`.
const PROC_PIDCOALITIONINFO: c_int = 20;
/// `LISTCOALITIONS_SINGLE_TYPE` from `<sys/proc_info_private.h>`.
const LISTCOALITIONS_SINGLE_TYPE: c_int = 2;
/// `COALITION_TYPE_RESOURCE` from `<mach/coalition.h>`.
const COALITION_TYPE_RESOURCE: c_int = 0;

/// `struct proc_pidcoalitioninfo`: index 0 is the resource coalition, 1 the
/// jetsam coalition.
#[repr(C)]
#[derive(Default)]
struct ProcPidCoalitionInfo {
    coalition_id: [u64; 2],
    reserved: [u64; 3],
}

/// `struct procinfo_coalinfo` from `<mach/coalition.h>` (private).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcInfoCoalInfo {
    coalition_id: u64,
    coalition_type: u32,
    coalition_tasks: u32,
}

/// `struct coalition_resource_usage` from xnu `osfmk/mach/coalition.h`, as
/// u64 slot indexes. The struct is private and only grows at the end; the
/// kernel copies `min(buffer size, its size)`, so the buffer is larger than
/// any known layout and pre-filled with [`UNWRITTEN`] to detect which slots a
/// given kernel provides.
mod slot {
    pub const TASKS_STARTED: usize = 0;
    pub const TASKS_EXITED: usize = 1;
    /// Nanoseconds.
    pub const GPU_TIME: usize = 8;
    /// `mach_absolute_time` units.
    pub const ANE_MACH_TIME: usize = 38;
    pub const ANE_ENERGY_NJ: usize = 39;
    /// GPU energy the coalition's own tasks used.
    pub const GPU_ENERGY_NJ: usize = 41;
    /// GPU energy other coalitions used on this coalition's behalf.
    pub const GPU_ENERGY_NJ_BILLED_TO_ME: usize = 42;
    /// GPU energy this coalition used on others' behalf.
    pub const GPU_ENERGY_NJ_BILLED_TO_OTHERS: usize = 43;
    /// Buffer size in slots; the kernel copies at most its own struct size,
    /// which may grow, so the buffer is larger than any known layout.
    pub const BUFFER: usize = 64;
}

/// Marks buffer slots the kernel did not write.
const UNWRITTEN: u64 = u64::MAX;

/// Number of phases used to spread cold coalition reads across samples.
const COLD_PHASES: u64 = 16;

unsafe extern "C" {
    fn proc_listcoalitions(
        flavor: c_int,
        coalition_type: c_int,
        buffer: *mut c_void,
        buffer_size: c_int,
    ) -> c_int;
    fn coalition_info_resource_usage(coalition_id: u64, usage: *mut c_void, size: usize) -> c_int;
    fn responsibility_get_pid_responsible_for_pid(pid: libc::pid_t) -> libc::pid_t;
}

/// Cumulative counters of one coalition. Every field only increases.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoalitionUsage {
    pub tasks_started: u64,
    pub tasks_exited: u64,
    /// GPU time of every task that ever ran in the coalition.
    pub gpu_ns: u64,
    /// GPU energy the coalition's tasks used.
    pub gpu_energy_nj: u64,
    /// GPU energy other coalitions used on this coalition's behalf, for
    /// example WindowServer compositing its windows.
    pub gpu_energy_billed_to_me_nj: u64,
    /// GPU energy this coalition used on behalf of others.
    pub gpu_energy_billed_to_others_nj: u64,
    /// Apple Neural Engine time, converted from mach ticks.
    pub ane_ns: u64,
    pub ane_energy_nj: u64,
}

impl CoalitionUsage {
    /// Counter increase from `previous` to `self`, never negative.
    pub fn since(&self, previous: &CoalitionUsage) -> CoalitionUsage {
        CoalitionUsage {
            tasks_started: self.tasks_started.saturating_sub(previous.tasks_started),
            tasks_exited: self.tasks_exited.saturating_sub(previous.tasks_exited),
            gpu_ns: self.gpu_ns.saturating_sub(previous.gpu_ns),
            gpu_energy_nj: self.gpu_energy_nj.saturating_sub(previous.gpu_energy_nj),
            gpu_energy_billed_to_me_nj: self
                .gpu_energy_billed_to_me_nj
                .saturating_sub(previous.gpu_energy_billed_to_me_nj),
            gpu_energy_billed_to_others_nj: self
                .gpu_energy_billed_to_others_nj
                .saturating_sub(previous.gpu_energy_billed_to_others_nj),
            ane_ns: self.ane_ns.saturating_sub(previous.ane_ns),
            ane_energy_nj: self.ane_energy_nj.saturating_sub(previous.ane_energy_nj),
        }
    }

    /// Whether the GPU, ANE or GPU energy counters differ from `other`.
    fn activity_differs(&self, other: &CoalitionUsage) -> bool {
        self.gpu_ns != other.gpu_ns
            || self.gpu_energy_nj != other.gpu_energy_nj
            || self.gpu_energy_billed_to_me_nj != other.gpu_energy_billed_to_me_nj
            || self.gpu_energy_billed_to_others_nj != other.gpu_energy_billed_to_others_nj
            || self.ane_ns != other.ane_ns
            || self.ane_energy_nj != other.ane_energy_nj
    }

    /// Whether the coalition has ever used the GPU or the ANE, or been billed
    /// for GPU energy.
    fn ever_used_gpu(&self) -> bool {
        self.gpu_ns > 0
            || self.gpu_energy_nj > 0
            || self.gpu_energy_billed_to_me_nj > 0
            || self.gpu_energy_billed_to_others_nj > 0
            || self.ane_ns > 0
            || self.ane_energy_nj > 0
    }

    fn from_slots(slots: &[u64; slot::BUFFER]) -> Self {
        let get = |index: usize| match slots[index] {
            UNWRITTEN => 0,
            value => value,
        };
        CoalitionUsage {
            tasks_started: get(slot::TASKS_STARTED),
            tasks_exited: get(slot::TASKS_EXITED),
            gpu_ns: get(slot::GPU_TIME),
            gpu_energy_nj: get(slot::GPU_ENERGY_NJ),
            gpu_energy_billed_to_me_nj: get(slot::GPU_ENERGY_NJ_BILLED_TO_ME),
            gpu_energy_billed_to_others_nj: get(slot::GPU_ENERGY_NJ_BILLED_TO_OTHERS),
            ane_ns: clock::mach_ticks_to_ns(get(slot::ANE_MACH_TIME)),
            ane_energy_nj: get(slot::ANE_ENERGY_NJ),
        }
    }
}

/// One coalition at one instant.
#[derive(Debug, Clone, PartialEq)]
pub struct Coalition {
    pub id: u64,
    /// The latest reading. For a cold coalition that was not re-read in this
    /// sample, the reading from an earlier sample.
    pub usage: CoalitionUsage,
    /// Live member PIDs, ascending.
    pub members: Vec<i32>,
    /// The process the members are responsible to (the app), when the kernel
    /// knows one. WindowServer's processes have none.
    pub responsible_pid: Option<i32>,
    /// The process that names the app: the responsible process, else the
    /// lowest-PID member that owns a GPU client, else the lowest-PID member.
    /// Resolved only for coalitions that have used the GPU or own a client.
    pub leader: Option<ProcessInfo>,
}

/// Every resource coalition and its counters at one instant.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CoalitionTable {
    coalitions: HashMap<u64, Coalition>,
    /// Resource coalition of every live PID.
    pids: HashMap<i32, u64>,
    /// Whether this kernel provides the GPU energy slots (41-43).
    pub has_gpu_energy: bool,
    /// Whether this kernel provides the ANE slots (38-39).
    pub has_ane: bool,
    /// Coalitions whose counters were read in this sample.
    pub read_count: usize,
}

impl CoalitionTable {
    /// Builds a table from coalitions; the PID index comes from their members.
    pub fn new(coalitions: Vec<Coalition>, has_gpu_energy: bool, has_ane: bool) -> Self {
        let read_count = coalitions.len();
        Self::new_with_read_count(coalitions, has_gpu_energy, has_ane, read_count)
    }

    /// Builds a table with the number of coalitions whose counters this read.
    fn new_with_read_count(
        coalitions: Vec<Coalition>,
        has_gpu_energy: bool,
        has_ane: bool,
        read_count: usize,
    ) -> Self {
        let mut pids = HashMap::new();
        let coalitions = coalitions
            .into_iter()
            .map(|coalition| {
                for pid in &coalition.members {
                    pids.insert(*pid, coalition.id);
                }
                (coalition.id, coalition)
            })
            .collect();
        CoalitionTable {
            coalitions,
            pids,
            has_gpu_energy,
            has_ane,
            read_count,
        }
    }

    pub fn get(&self, id: u64) -> Option<&Coalition> {
        self.coalitions.get(&id)
    }

    /// The resource coalition `pid` belongs to.
    pub fn coalition_of(&self, pid: i32) -> Option<u64> {
        self.pids.get(&pid).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Coalition> {
        self.coalitions.values()
    }

    pub fn len(&self) -> usize {
        self.coalitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coalitions.is_empty()
    }
}

/// Member PIDs per coalition and the coalitions that own GPU clients.
type Memberships = (HashMap<u64, Vec<i32>>, HashSet<u64>);

/// What the reader remembers about a live PID.
#[derive(Clone, Copy)]
struct PidEntry {
    coalition: u64,
    responsible: Option<i32>,
    /// The read that last listed this PID.
    seen: u64,
}

/// What the reader remembers about a coalition.
struct Tracked {
    usage: CoalitionUsage,
    /// When the counters are next due for a cold read.
    due: Instant,
    /// Task count when the counters were last read.
    tasks: u32,
    /// The GPU, ANE or energy counters moved at the last read.
    moved: bool,
    /// The read that last listed this coalition.
    seen: u64,
}

/// Reads the coalition table, keeping caches between samples.
#[derive(Default)]
pub struct CoalitionReader {
    pids: HashMap<i32, PidEntry>,
    tracked: HashMap<u64, Tracked>,
    /// Process info for leaders. Entries are dropped as soon as the PID leaves
    /// a listing, so a reused PID cannot inherit a stale leader without the
    /// extra start-time sysctl a `ProcessCache` would run.
    leaders: HashMap<i32, Option<ProcessInfo>>,
    /// Highest slot index the kernel writes, learned on the first read.
    written_slots: Option<usize>,
    /// Counts reads; marks which cache entries the latest listing contained.
    generation: u64,
}

impl CoalitionReader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads every coalition's membership, and the counters of coalitions that
    /// are active or due (see the module docs). `gpu_pids` are the PIDs that
    /// own GPU clients; their coalitions are always read.
    pub fn read(&mut self, gpu_pids: &[i32]) -> Result<CoalitionTable> {
        let now = Instant::now();
        self.generation += 1;
        let generation = self.generation;
        let listed = list_coalitions()?;

        let (members, gpu_coalitions) = self.read_membership(generation)?;
        let read_count = self.read_counters(&listed, &gpu_coalitions, generation, now)?;
        Ok(self.assemble_table(members, &gpu_coalitions, gpu_pids, read_count))
    }

    /// Resolves every live PID to its coalition, from cached or fresh lookups.
    fn read_membership(&mut self, generation: u64) -> Result<Memberships> {
        let live_pids = process::try_all_pids()?;
        let mut members: HashMap<u64, Vec<i32>> = HashMap::new();
        for pid in live_pids {
            let coalition = match self.pids.get_mut(&pid) {
                Some(entry) => {
                    entry.seen = generation;
                    entry.coalition
                }
                None => {
                    let Some(coalition) = coalition_of_pid(pid) else {
                        continue;
                    };
                    self.pids.insert(
                        pid,
                        PidEntry {
                            coalition,
                            responsible: responsible_pid(pid),
                            seen: generation,
                        },
                    );
                    coalition
                }
            };
            members.entry(coalition).or_default().push(pid);
        }
        self.pids.retain(|_, entry| entry.seen == generation);
        let pids = &self.pids;
        self.leaders.retain(|pid, _| pids.contains_key(pid));
        let gpu_coalitions = pids
            .values()
            .filter(|entry| members.contains_key(&entry.coalition))
            .map(|entry| entry.coalition)
            .collect();
        Ok((members, gpu_coalitions))
    }

    /// Reads the counters of coalitions that are active, own a GPU client or
    /// are due for a cold refresh. Returns how many coalitions were read.
    fn read_counters(
        &mut self,
        listed: &[ProcInfoCoalInfo],
        gpu_coalitions: &HashSet<u64>,
        generation: u64,
        now: Instant,
    ) -> Result<usize> {
        let mut buffer = [UNWRITTEN; slot::BUFFER];
        let mut read_count = 0;
        let mut failures = 0;
        let mut last_error = None;
        for info in listed {
            let id = info.coalition_id;
            let due = match self.tracked.get_mut(&id) {
                None => true,
                Some(tracked) => {
                    tracked.seen = generation;
                    let frozen = info.coalition_tasks == 0 && tracked.tasks == 0;
                    !frozen && (tracked.moved || gpu_coalitions.contains(&id) || now >= tracked.due)
                }
            };
            if !due {
                continue;
            }
            buffer.fill(UNWRITTEN);
            // SAFETY: `buffer` is writable for `size_of_val` bytes, the size passed to the kernel.
            let result = unsafe {
                coalition_info_resource_usage(
                    id,
                    buffer.as_mut_ptr().cast::<c_void>(),
                    size_of_val(&buffer),
                )
            };
            if result != 0 {
                // Usually the coalition was reaped after it was listed.
                failures += 1;
                last_error = Some(std::io::Error::last_os_error());
                self.tracked.remove(&id);
                continue;
            }
            read_count += 1;
            if self.written_slots.is_none() {
                self.written_slots = buffer.iter().rposition(|value| *value != UNWRITTEN);
            }
            let first_read = !self.tracked.contains_key(&id);
            let tracked = self.tracked.entry(id).or_insert(Tracked {
                usage: CoalitionUsage::default(),
                due: now,
                tasks: 0,
                moved: false,
                seen: generation,
            });
            let usage = CoalitionUsage::from_slots(&buffer);
            // A coalition seen for the first time is re-read next sample if it
            // ever used the GPU, so it settles into the right group quickly.
            tracked.moved = if first_read {
                usage.ever_used_gpu()
            } else {
                usage.activity_differs(&tracked.usage)
            };
            tracked.usage = usage;
            tracked.tasks = info.coalition_tasks;
            tracked.due = now + COLD_REFRESH;
            if first_read {
                // Stagger cold reads so they do not all land on one sample.
                tracked.due -= COLD_REFRESH * (id % COLD_PHASES) as u32 / COLD_PHASES as u32;
            }
        }
        self.tracked.retain(|_, tracked| tracked.seen == generation);
        if read_count == 0 && failures > 0 {
            bail!(
                "coalition_info_resource_usage failed for all {failures} coalitions: {}",
                last_error
                    .map(|error| error.to_string())
                    .unwrap_or_default()
            );
        }
        Ok(read_count)
    }

    /// Assembles membership, counters and leaders into the sample's table.
    fn assemble_table(
        &mut self,
        mut members: HashMap<u64, Vec<i32>>,
        gpu_coalitions: &HashSet<u64>,
        gpu_pids: &[i32],
        read_count: usize,
    ) -> CoalitionTable {
        let gpu_pid_set: HashSet<i32> = gpu_pids.iter().copied().collect();
        let mut coalitions = Vec::with_capacity(self.tracked.len());
        for (id, tracked) in &self.tracked {
            let mut member_pids = members.remove(id).unwrap_or_default();
            member_pids.sort_unstable();
            let responsible = most_common(
                member_pids
                    .iter()
                    .filter_map(|pid| self.pids.get(pid).and_then(|entry| entry.responsible)),
            );
            let leader = if tracked.usage.ever_used_gpu() || gpu_coalitions.contains(id) {
                leader_candidates(&member_pids, responsible, &gpu_pid_set)
                    .into_iter()
                    .find_map(|pid| {
                        self.leaders
                            .entry(pid)
                            .or_insert_with(|| process::info(pid))
                            .clone()
                    })
            } else {
                None
            };
            coalitions.push(Coalition {
                id: *id,
                usage: tracked.usage,
                members: member_pids,
                responsible_pid: responsible,
                leader,
            });
        }

        let written = self.written_slots.unwrap_or(0);
        CoalitionTable::new_with_read_count(
            coalitions,
            written >= slot::GPU_ENERGY_NJ_BILLED_TO_OTHERS,
            written >= slot::ANE_ENERGY_NJ,
            read_count,
        )
    }
}

/// Every live resource coalition, in one call.
fn list_coalitions() -> Result<Vec<ProcInfoCoalInfo>> {
    let element = size_of::<ProcInfoCoalInfo>();
    let mut capacity = 1024usize;
    loop {
        let mut list = vec![ProcInfoCoalInfo::default(); capacity];
        // SAFETY: `list` is writable for `(capacity * element)` bytes and the return is only read as a count.
        let bytes = unsafe {
            proc_listcoalitions(
                LISTCOALITIONS_SINGLE_TYPE,
                COALITION_TYPE_RESOURCE,
                list.as_mut_ptr().cast::<c_void>(),
                (capacity * element) as c_int,
            )
        };
        // The libproc wrapper turns errors into 0.
        if bytes <= 0 {
            bail!(
                "proc_listcoalitions failed: {}",
                std::io::Error::last_os_error()
            );
        }
        let count = bytes as usize / element;
        if count < capacity {
            list.truncate(count);
            list.retain(|info| info.coalition_type == COALITION_TYPE_RESOURCE as u32);
            return Ok(list);
        }
        capacity *= 2;
    }
}

/// The resource coalition `pid` belongs to. Works for every user's processes.
fn coalition_of_pid(pid: i32) -> Option<u64> {
    let mut info = ProcPidCoalitionInfo::default();
    let size = size_of::<ProcPidCoalitionInfo>() as c_int;
    // SAFETY: `info` is a zeroed, correctly sized buffer for `PROC_PIDCOALITIONINFO` output.
    let written = unsafe {
        libc::proc_pidinfo(
            pid,
            PROC_PIDCOALITIONINFO,
            0,
            (&mut info as *mut ProcPidCoalitionInfo).cast::<c_void>(),
            size,
        )
    };
    (written == size && info.coalition_id[0] != 0).then_some(info.coalition_id[0])
}

/// The PID `pid` is responsible to, for example the app of an XPC service.
/// `None` when the kernel has none (WindowServer) or the process is gone.
fn responsible_pid(pid: i32) -> Option<i32> {
    // SAFETY: the function takes a pid by value and returns a pid or a non-positive sentinel.
    let responsible = unsafe { responsibility_get_pid_responsible_for_pid(pid) };
    (responsible > 0).then_some(responsible)
}

/// PIDs to try, in order, for the process that names a coalition.
fn leader_candidates(
    members: &[i32],
    responsible: Option<i32>,
    gpu_pids: &HashSet<i32>,
) -> Vec<i32> {
    let mut candidates: Vec<i32> = responsible.into_iter().collect();
    candidates.extend(members.iter().filter(|pid| gpu_pids.contains(pid)));
    candidates.extend(members.first());
    candidates.dedup();
    candidates
}

/// The most frequent value; ties go to the smallest.
fn most_common(mut values: impl Iterator<Item = i32>) -> Option<i32> {
    let first = values.next()?;
    let Some(second) = values.next() else {
        // Most coalitions have one member: skip the map.
        return Some(first);
    };
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for value in [first, second].into_iter().chain(values) {
        *counts.entry(value).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(value, _)| value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_the_kernel() {
        assert_eq!(size_of::<ProcPidCoalitionInfo>(), 40);
        assert_eq!(size_of::<ProcInfoCoalInfo>(), 16);
    }

    #[test]
    fn usage_delta_never_goes_negative() {
        let previous = CoalitionUsage {
            gpu_ns: 100,
            gpu_energy_nj: 50,
            ..CoalitionUsage::default()
        };
        let current = CoalitionUsage {
            gpu_ns: 350,
            gpu_energy_nj: 40,
            ..CoalitionUsage::default()
        };
        let delta = current.since(&previous);
        assert_eq!(delta.gpu_ns, 250);
        assert_eq!(delta.gpu_energy_nj, 0);
    }

    #[test]
    fn unwritten_slots_read_as_zero() {
        let mut slots = [UNWRITTEN; slot::BUFFER];
        slots[slot::GPU_TIME] = 7;
        let usage = CoalitionUsage::from_slots(&slots);
        assert_eq!(usage.gpu_ns, 7);
        assert_eq!(usage.gpu_energy_nj, 0);
        assert!(usage.ever_used_gpu());
    }

    #[test]
    fn leader_prefers_responsible_then_gpu_member_then_lowest() {
        let gpu: HashSet<i32> = [818].into_iter().collect();
        assert_eq!(
            leader_candidates(&[431, 818], Some(56), &gpu),
            vec![56, 818, 431]
        );
        assert_eq!(leader_candidates(&[431, 818], None, &gpu), vec![818, 431]);
        assert_eq!(
            leader_candidates(&[431, 818], None, &HashSet::new()),
            vec![431]
        );
        assert_eq!(most_common([5, 7, 7, 5, 3].into_iter()), Some(5));
        assert_eq!(most_common(std::iter::empty()), None);
    }

    #[test]
    fn table_indexes_members() {
        let table = CoalitionTable::new(
            vec![Coalition {
                id: 9,
                usage: CoalitionUsage::default(),
                members: vec![1, 2],
                responsible_pid: None,
                leader: None,
            }],
            true,
            true,
        );
        assert_eq!(table.coalition_of(2), Some(9));
        assert_eq!(table.coalition_of(3), None);
        assert_eq!(table.len(), 1);
    }

    /// Live: every process's coalition is readable without root, including
    /// WindowServer's, which another user owns.
    #[test]
    #[ignore = "needs Apple Silicon hardware and a live coalition table"]
    fn reads_live_coalitions() {
        let own = std::process::id() as i32;
        let mut reader = CoalitionReader::new();
        let table = reader.read(&[own]).expect("coalition table");
        assert!(table.len() > 10, "only {} coalitions", table.len());
        assert!(table.has_gpu_energy && table.has_ane);

        let own_coalition = table.coalition_of(own).expect("own coalition");
        let coalition = table.get(own_coalition).expect("own coalition entry");
        assert!(coalition.members.contains(&own));
        assert!(coalition.usage.tasks_started >= 1);
        assert!(coalition.leader.is_some());

        let window_server = process::all_pids()
            .into_iter()
            .find(|pid| process::info(*pid).is_some_and(|info| info.name == "WindowServer"));
        if let Some(pid) = window_server {
            let id = table.coalition_of(pid).expect("WindowServer coalition");
            let coalition = table.get(id).expect("WindowServer entry");
            assert!(
                coalition.usage.gpu_ns > 0,
                "WindowServer never used the GPU?"
            );
            assert_eq!(coalition.responsible_pid, None);
            let leader = coalition.leader.as_ref().expect("WindowServer leader");
            assert_eq!(leader.name, "WindowServer");
        }

        // A second read uses the caches and still sees everything.
        let again = reader.read(&[own]).expect("second read");
        assert_eq!(again.coalition_of(own), Some(own_coalition));
        assert!(
            again.read_count < again.len(),
            "cold coalitions were re-read"
        );
    }

    /// Runs a Metal compute kernel in this process for about `duration` and
    /// returns the summed `GPUEndTime - GPUStartTime` of its command buffers.
    fn metal_work(duration: std::time::Duration) -> u64 {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
            MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
        };
        const SOURCE: &str = r"
            #include <metal_stdlib>
            using namespace metal;
            kernel void spin(device float *out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
                float x = float(id) * 0.001f;
                for (uint i = 0; i < 8192; i++) { x = fma(x, 1.000001f, 0.5f); x = sin(x) * 0.99f + cos(x); }
                out[id] = x;
            }";
        let threads = 1usize << 18;
        let device = MTLCreateSystemDefaultDevice().expect("Metal device");
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .expect("kernel compiles");
        let function = library
            .newFunctionWithName(&NSString::from_str("spin"))
            .expect("kernel function");
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .expect("pipeline");
        let output = device
            .newBufferWithLength_options(threads * 4, MTLResourceOptions::StorageModePrivate)
            .expect("buffer");
        let queue = device.newCommandQueue().expect("queue");
        let started = std::time::Instant::now();
        let mut gpu_seconds = 0.0;
        while started.elapsed() < duration {
            let buffer = queue.commandBuffer().expect("command buffer");
            let encoder = buffer.computeCommandEncoder().expect("encoder");
            encoder.setComputePipelineState(&pipeline);
            // SAFETY: the buffer holds `threads` floats, one per thread.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&output), 0, 0) };
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            buffer.commit();
            buffer.waitUntilCompleted();
            gpu_seconds += buffer.GPUEndTime() - buffer.GPUStartTime();
        }
        (gpu_seconds * 1e9) as u64
    }

    /// Live: pins the struct layout. Metal work in this process must show up in
    /// its coalition's `gpu_time` (slot 8, ns) at least as fully as the work's
    /// own command-buffer spans (the coalition counts in-flight time, and
    /// other members only add to it), and `gpu_energy_nj` (slot 41), once it
    /// has settled, must imply a plausible power. The process's `AppUsage`
    /// time must be close: it also counts per-command-buffer overhead outside
    /// the spans, and reads lower when other GPU work overlaps.
    #[test]
    #[ignore = "needs Apple Silicon hardware and a Metal device"]
    fn gpu_slots_track_metal_work() {
        use crate::sys::iokit::RegistryReader;
        let own = std::process::id() as i32;
        let own_usage = |registry: &crate::sys::iokit::Registry| {
            registry
                .clients()
                .filter(|client| client.pid == own)
                .map(|client| (client.registry_id, client.accumulated_ns()))
                .collect::<HashMap<u64, u64>>()
        };
        // Open the device first so the client exists in both samples.
        metal_work(std::time::Duration::from_millis(50));

        let mut registry = RegistryReader::new();
        let mut reader = CoalitionReader::new();
        let before_registry = registry.read().expect("registry");
        let before = reader.read(&[own]).expect("coalitions");
        let started = std::time::Instant::now();
        let spans_ns = metal_work(std::time::Duration::from_millis(1500));
        let after = reader.read(&[own]).expect("coalitions");
        let after_registry = registry.read().expect("registry");
        let elapsed = started.elapsed().as_secs_f64();

        let id = after.coalition_of(own).expect("own coalition");
        let before_usage = before.get(id).expect("own entry before").usage;
        let mut delta = after.get(id).expect("own entry").usage.since(&before_usage);

        // The kernel charges coalition GPU energy about 1-1.5 s after the
        // work, so the first post-load read can hold almost none of it. Read
        // until the counter stops catching up, or give up after a few seconds
        // and let the assertion below report what did arrive. Idle accounting
        // keeps adding a few millijoules every second, so only wait for
        // increments above 1% of what has arrived. GPU time is charged
        // promptly and stays with the first post-load read.
        let mut energy_nj = after.get(id).expect("own entry").usage.gpu_energy_nj;
        let mut changed = Instant::now();
        let deadline = Instant::now() + Duration::from_secs(4);
        while Instant::now() < changed + Duration::from_secs(1) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(100));
            let reading = reader.read(&[own]).expect("coalitions");
            let energy = reading.get(id).expect("own entry").usage.gpu_energy_nj;
            let threshold = energy_nj.saturating_sub(before_usage.gpu_energy_nj) / 100;
            if energy.saturating_sub(energy_nj) > threshold {
                energy_nj = energy;
                changed = Instant::now();
            }
        }
        delta.gpu_energy_nj = energy_nj.saturating_sub(before_usage.gpu_energy_nj);
        let old = own_usage(&before_registry);
        let app_usage_ns: u64 = own_usage(&after_registry)
            .iter()
            .map(|(client, now)| now.saturating_sub(old.get(client).copied().unwrap_or(0)))
            .sum();

        assert!(spans_ns > 500_000_000, "the load barely ran: {spans_ns} ns");
        assert!(
            delta.gpu_ns as f64 >= spans_ns as f64 * 0.9,
            "coalition gpu_time {} ns < command-buffer spans {spans_ns} ns",
            delta.gpu_ns
        );
        assert!(app_usage_ns > 0, "no AppUsage time for this process");
        assert!(
            app_usage_ns as f64 <= delta.gpu_ns as f64 * 1.15 + 50e6,
            "AppUsage {app_usage_ns} ns far above coalition {} ns",
            delta.gpu_ns
        );
        let watts = delta.gpu_energy_nj as f64 / 1e9 / elapsed;
        assert!(
            (0.2..300.0).contains(&watts),
            "implausible GPU power {watts} W"
        );
    }
}
