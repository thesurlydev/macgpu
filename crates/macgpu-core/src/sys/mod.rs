//! Platform readers. Each converts OS data into plain, `Send` Rust values.

#![deny(clippy::undocumented_unsafe_blocks)]

pub mod cf;
pub mod clock;
pub mod coalition;
pub mod gpu_freq;
pub mod iokit;
pub mod ioreport;
pub mod powermetrics;
pub mod process;
pub mod smc;
pub mod system;
pub mod task_memory;
