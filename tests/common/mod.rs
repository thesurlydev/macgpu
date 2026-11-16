//! Shared helpers for the end-to-end tests.
//!
//! Every test that generates GPU load takes [`gpu_lock`], so loads from these
//! tests never overlap: not within one test binary, not across concurrent
//! `cargo test` runs, and not with a `macgpu doctor --load`. Loads that do not
//! take the lock can still run, so every assertion allows for other GPU work.

#![allow(dead_code)]

use macgpu::frontend::doctor::GpuLoadLock;
use macgpu::metal_load::{LoadConfig, LoadProcess};
use macgpu::model::{Report, SourceState};
use macgpu::sys::iokit::{GpuClient, RegistryReader};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

pub const MACGPU: &str = env!("CARGO_BIN_EXE_macgpu");

/// Fails when a report's named source is not usable for a test's assertions.
pub fn require(report: &Report, name: &str) {
    let status = report
        .sources
        .get(name)
        .unwrap_or_else(|| panic!("source {name} not reported"));
    assert!(
        matches!(status.state, SourceState::Ok | SourceState::Partial),
        "source {name} is {:?} ({}); this test needs its reader",
        status.state,
        status.detail.as_deref().unwrap_or("")
    );
}

/// Held while a test generates GPU load.
pub struct GpuLock {
    _process: GpuLoadLock,
    _thread: MutexGuard<'static, ()>,
}

/// Serializes GPU-heavy tests: across threads of this test binary, then
/// across processes through the same `flock` that `doctor --load` takes.
pub fn gpu_lock() -> GpuLock {
    static LOCK: Mutex<()> = Mutex::new(());
    let thread = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let process = GpuLoadLock::acquire_or_wait().expect("take the GPU load lock");
    GpuLock {
        _process: process,
        _thread: thread,
    }
}

/// Spawns `macgpu __load` that waits for the go line and holds its GPU client
/// open until finished.
pub fn spawn_held(config: &LoadConfig) -> LoadProcess {
    let config = LoadConfig {
        wait: true,
        hold: true,
        ..config.clone()
    };
    LoadProcess::spawn(Path::new(MACGPU), &config).expect("spawn macgpu __load")
}

/// The GPU clients a PID owns right now, read directly from the registry.
pub fn clients_of(reader: &mut RegistryReader, pid: i32) -> Vec<GpuClient> {
    reader
        .read()
        .expect("read GPU registry")
        .clients()
        .filter(|client| client.pid == pid)
        .cloned()
        .collect()
}

/// Relative error of `measured` against `truth`, in percent.
pub fn error_pct(measured: u64, truth: u64) -> f64 {
    (measured as f64 - truth as f64) / truth.max(1) as f64 * 100.0
}

pub fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}
