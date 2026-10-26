//! Per-process GPU memory from the `TASK_VM_INFO` graphics ledgers.
//!
//! `task_name_for_pid` returns a task name port for processes of the same user
//! (or any process for root). `task_info(TASK_VM_INFO)` through that port
//! returns the kernel's per-task ledgers, including the memory tagged as
//! graphics: Metal buffers and textures, IOSurfaces the task owns, and so on.
//! For other users' processes, such as WindowServer, the port is denied without
//! root; those readings are recorded as [`GraphicsMemory::Denied`], distinct
//! from a real 0.

use mach2::kern_return::{KERN_SUCCESS, kern_return_t};
use mach2::mach_types::task_name_t;
use mach2::message::mach_msg_type_number_t;
use mach2::port::{MACH_PORT_NULL, mach_port_name_t};
use mach2::task_info::{TASK_VM_INFO, task_vm_info};
use mach2::traps::mach_task_self;
use std::collections::HashMap;
use std::mem::{offset_of, size_of};

unsafe extern "C" {
    fn task_name_for_pid(
        target_task: mach2::mach_types::task_t,
        pid: i32,
        task_name: *mut task_name_t,
    ) -> kern_return_t;
}

/// One process's graphics memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsMemory {
    Bytes {
        /// `ledger_tag_graphics_footprint`: resident graphics memory charged
        /// to the task.
        footprint: u64,
        /// `ledger_tag_graphics_footprint_compressed`: the part of the
        /// task's graphics memory that is in the compressor.
        compressed: u64,
        /// `ledger_tag_graphics_nofootprint`: graphics memory the task maps
        /// but is not charged for, such as shared surfaces.
        nofootprint: u64,
    },
    /// The task port was denied: another user's process without root.
    Denied,
    /// The process exited, or `task_info` failed.
    Unavailable,
}

impl GraphicsMemory {
    /// Graphics memory charged to the process: resident plus compressed.
    pub fn charged_bytes(&self) -> Option<u64> {
        match self {
            GraphicsMemory::Bytes {
                footprint,
                compressed,
                ..
            } => Some(footprint.saturating_add(*compressed)),
            _ => None,
        }
    }
}

/// Graphics memory readings for the processes that own GPU clients.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryTable {
    pub readings: HashMap<i32, GraphicsMemory>,
}

impl MemoryTable {
    pub fn get(&self, pid: i32) -> Option<GraphicsMemory> {
        self.readings.get(&pid).copied()
    }

    /// Charged graphics memory for `pid`, when it was readable.
    pub fn charged_bytes(&self, pid: i32) -> Option<u64> {
        self.get(pid).and_then(|reading| reading.charged_bytes())
    }
}

#[derive(Default)]
pub struct MemoryReader {}

impl MemoryReader {
    pub fn new() -> Self {
        Self {}
    }

    /// Reads graphics memory for each of `pids`.
    pub fn read(&mut self, pids: &[i32]) -> MemoryTable {
        MemoryTable {
            readings: pids.iter().map(|pid| (*pid, read_pid(*pid))).collect(),
        }
    }
}

/// A task name port, deallocated on drop.
struct NamePort(mach_port_name_t);

impl Drop for NamePort {
    fn drop(&mut self) {
        unsafe { mach2::mach_port::mach_port_deallocate(mach_task_self(), self.0) };
    }
}

/// `task_vm_info` count up to and including the graphics ledgers.
const GRAPHICS_COUNT: mach_msg_type_number_t =
    ((offset_of!(task_vm_info, ledger_tag_graphics_nofootprint_compressed) + size_of::<i64>())
        / size_of::<u32>()) as mach_msg_type_number_t;

const _: () = assert!(GRAPHICS_COUNT as usize * size_of::<u32>() <= size_of::<task_vm_info>());

fn read_pid(pid: i32) -> GraphicsMemory {
    if pid <= 0 {
        return GraphicsMemory::Unavailable;
    }
    let mut name: task_name_t = MACH_PORT_NULL;
    let result = unsafe { task_name_for_pid(mach_task_self(), pid, &mut name) };
    if result != KERN_SUCCESS || name == MACH_PORT_NULL {
        // The process may also have exited; tell the two apart.
        let alive = unsafe { libc::kill(pid, 0) } == 0
            || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
        return if alive {
            GraphicsMemory::Denied
        } else {
            GraphicsMemory::Unavailable
        };
    }
    let port = NamePort(name);

    let mut info = task_vm_info::default();
    let mut count = (size_of::<task_vm_info>() / size_of::<u32>()) as mach_msg_type_number_t;
    let result = unsafe {
        mach2::task::task_info(
            port.0,
            TASK_VM_INFO,
            (&mut info as *mut task_vm_info).cast(),
            &mut count,
        )
    };
    if result != KERN_SUCCESS || count < GRAPHICS_COUNT {
        return GraphicsMemory::Unavailable;
    }
    let bytes = |value: i64| u64::try_from(value).unwrap_or(0);
    GraphicsMemory::Bytes {
        footprint: bytes(info.ledger_tag_graphics_footprint),
        compressed: bytes(info.ledger_tag_graphics_footprint_compressed),
        nofootprint: bytes(info.ledger_tag_graphics_nofootprint),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn mach_port_get_refs(
            task: mach2::mach_types::ipc_space_t,
            name: mach_port_name_t,
            right: mach2::port::mach_port_right_t,
            refs: *mut mach2::port::mach_port_urefs_t,
        ) -> kern_return_t;
    }

    /// User references on the send right `task_name_for_pid(pid)` returns,
    /// not counting the one this call takes.
    fn name_port_refs(pid: i32) -> u32 {
        let mut name: task_name_t = MACH_PORT_NULL;
        let result = unsafe { task_name_for_pid(mach_task_self(), pid, &mut name) };
        assert_eq!(result, KERN_SUCCESS);
        let port = NamePort(name);
        let mut refs = 0;
        let result = unsafe {
            mach_port_get_refs(
                mach_task_self(),
                port.0,
                mach2::port::MACH_PORT_RIGHT_SEND,
                &mut refs,
            )
        };
        assert_eq!(result, KERN_SUCCESS);
        refs - 1
    }

    #[test]
    fn reads_own_graphics_memory_without_leaking_ports() {
        let own = std::process::id() as i32;
        let before = name_port_refs(own);
        let mut reader = MemoryReader::new();
        for _ in 0..200 {
            let table = reader.read(&[own]);
            assert!(matches!(table.get(own), Some(GraphicsMemory::Bytes { .. })));
            assert!(table.charged_bytes(own).is_some());
        }
        assert_eq!(name_port_refs(own), before, "task name port leaked");
    }

    #[test]
    fn other_users_processes_are_denied_not_zero() {
        // launchd is root-owned.
        let table = MemoryReader::new().read(&[1]);
        assert_eq!(table.get(1), Some(GraphicsMemory::Denied));
        assert_eq!(table.charged_bytes(1), None);
    }

    #[test]
    fn exited_processes_are_unavailable() {
        let mut child = std::process::Command::new("/usr/bin/true")
            .spawn()
            .expect("spawn");
        let pid = child.id() as i32;
        child.wait().expect("wait");
        let table = MemoryReader::new().read(&[pid, -1]);
        assert_eq!(table.get(pid), Some(GraphicsMemory::Unavailable));
        assert_eq!(table.get(-1), Some(GraphicsMemory::Unavailable));
    }
}
