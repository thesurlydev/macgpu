//! macgpu: per-process and per-app GPU usage, power, and memory for Apple
//! Silicon Macs, without sudo.
//!
//! The library exposes the sampler ([`snapshot::Sampler`]), the analysis that
//! turns two snapshots into a [`model::Report`] ([`analyze::analyze`]), and the
//! background [`engine::Engine`], so other front ends (a menu bar app, a GUI)
//! can reuse the core.

#[cfg(not(target_os = "macos"))]
compile_error!("macgpu supports macOS only");

pub mod analyze;
pub mod cli;
pub mod engine;
pub mod format;
pub mod frontend;
pub mod metal_load;
pub mod model;
pub mod snapshot;
pub mod sys;
pub mod view;
