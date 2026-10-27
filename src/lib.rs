//! macgpu: per-process and per-app GPU usage, power, and memory for Apple
//! Silicon Macs, without sudo.
//!
//! This crate holds the front ends: the CLI, table, TUI, log, record and
//! replay, Prometheus server, and the `run` and `doctor` commands. Sampling
//! and analysis live in `macgpu-core` and the report schema in
//! `macgpu-model`; both are re-exported here under their old paths.

#[cfg(not(target_os = "macos"))]
compile_error!("macgpu supports macOS only");

pub use macgpu_core::{analyze, engine, snapshot, sys};
pub use macgpu_model as model;

pub mod cli;
pub mod format;
pub mod frontend;
pub mod metal_load;
pub mod view;
