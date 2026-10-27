//! The macgpu sampler ([`snapshot::Sampler`]), the analysis that turns two
//! snapshots into a [`macgpu_model::Report`] ([`analyze::Analyzer`]), and the
//! background [`engine::Engine`]. No sudo, and no Metal device is opened.
//!
//! Front ends (the `macgpu` binary, a GUI) depend on this crate for sampling.
//! Readers that only consume reports, possibly on another platform, depend on
//! `macgpu-model` alone.

#[cfg(not(target_os = "macos"))]
compile_error!("macgpu-core supports macOS only");

pub mod analyze;
pub mod engine;
pub mod snapshot;
pub mod sys;
