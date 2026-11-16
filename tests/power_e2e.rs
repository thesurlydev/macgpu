//! End-to-end tests for whole-GPU readings (IOReport residency, clock, power,
//! and SMC temperature) under a real Metal load in a child process.
//!
//! DEPENDS ON the IOReport and SMC readers (`sys::ioreport`, `sys::gpu_freq`,
//! `sys::smc`, and their part of `analyze::device`). Each test first checks
//! that its sources report `ok` and fails with a message naming the missing
//! source otherwise.

mod common;

use common::*;
use macgpu::analyze::{AnalyzeOptions, analyze};
use macgpu::metal_load::LoadConfig;
use macgpu::model::{Report, source};
use macgpu::snapshot::{Sampler, SourceSet};
use std::time::Duration;

/// Idle-baseline activity above which another process was loading the GPU.
const BUSY_BASELINE_PCT: f64 = 50.0;

/// An idle window with the load process open, then a window covering the load.
/// Retried (up to three times) while another process keeps the GPU busy during
/// the idle window, since a rise cannot be measured from a busy baseline.
fn idle_and_load(config: &LoadConfig) -> (Report, Report, f64) {
    for _ in 0..2 {
        let (idle, load, busy_pct) = idle_and_load_once(config);
        if idle.device.active_pct.unwrap_or(0.0) < BUSY_BASELINE_PCT {
            return (idle, load, busy_pct);
        }
        eprintln!(
            "another process kept the GPU {:.0}% active during the idle window; retrying",
            idle.device.active_pct.unwrap_or(0.0)
        );
    }
    idle_and_load_once(config)
}

fn idle_and_load_once(config: &LoadConfig) -> (Report, Report, f64) {
    let mut child = spawn_held(config);
    let mut sampler = Sampler::new(SourceSet::default());
    let idle_start = sampler.snapshot();
    std::thread::sleep(Duration::from_secs(1));
    let load_start = sampler.snapshot();
    child.start().unwrap();
    let truth = child.wait_done().unwrap();
    let load_end = sampler.snapshot();
    child.finish().unwrap();
    let options = AnalyzeOptions { include_idle: true };
    let idle = analyze(&idle_start, &load_start, &options);
    let load = analyze(&load_start, &load_end, &options);
    let busy_pct = truth.gpu_ns as f64 / (load.window_seconds * 1e9) * 100.0;
    (idle, load, busy_pct)
}

/// Active residency and power rise under load; residency is integrated over
/// the window, so it covers at least the time the load kept the GPU busy.
#[test]
#[ignore = "needs real IOReport power readings and a Metal load; not available on CI VMs"]
fn active_residency_and_power_rise_under_load() {
    let _lock = gpu_lock();
    let (idle, load, busy_pct) = idle_and_load(&LoadConfig {
        seconds: 3.0,
        ..LoadConfig::default()
    });
    require(&load, source::IOREPORT);
    let before = idle.device.active_pct.expect("idle active_pct");
    let after = load.device.active_pct.expect("load active_pct");
    let idle_w = idle.device.power_w.expect("idle power_w");
    let load_w = load.device.power_w.expect("load power_w");
    eprintln!(
        "active {before:.1}% -> {after:.1}% (load alone {busy_pct:.1}% busy), power {idle_w:.2} W -> {load_w:.2} W, avg {:?} MHz",
        load.device.avg_mhz
    );
    assert!(after > before, "active residency did not rise");
    assert!(
        after >= busy_pct * 0.9 - 5.0,
        "active {after:.1}% is below the {busy_pct:.1}% the load alone kept the GPU busy"
    );
    assert!(after <= 100.0 + 1e-6);
    assert!(load_w >= idle_w + 2.0, "{idle_w:.2} W -> {load_w:.2} W");

    // AppUsage apportions active GPU time, so summed over every process it
    // stays within GPUPH active time (M4 Max: 956-970 ms/s at 99.5% active).
    let active_ns = after / 100.0 * load.window_seconds * 1e9;
    let total = load.totals.attributed_gpu_ns as f64;
    eprintln!(
        "all AppUsage {:.0} ms/s vs GPUPH active {:.0} ms/s",
        total / load.window_seconds / 1e6,
        active_ns / load.window_seconds / 1e6
    );
    assert!(
        total <= active_ns * 1.03 + 5e6,
        "AppUsage {:.0} ms exceeds active time {:.0} ms",
        total / 1e6,
        active_ns / 1e6
    );

    let total: f64 = load.device.pstates.iter().map(|state| state.pct).sum();
    assert!((total - 100.0).abs() <= 2.0, "P-states sum to {total:.1}%");
    let avg = load.device.avg_mhz.expect("avg_mhz");
    let max = load.device.max_mhz.expect("max_mhz");
    assert!(
        avg > 0.0 && avg <= max + 0.5,
        "avg {avg} MHz, max {max} MHz"
    );
    if let Some(normalized) = load.device.freq_normalized_pct {
        assert!(normalized <= after + 1e-6, "{normalized} > {after}");
    }
}

/// GPU temperature is reported, plausible, and does not fall under load.
#[test]
#[ignore = "needs real SMC temperature sensors and a Metal load; not available on CI VMs"]
fn temperature_is_reported_under_load() {
    let _lock = gpu_lock();
    let (idle, load, _) = idle_and_load(&LoadConfig {
        seconds: 3.0,
        ..LoadConfig::default()
    });
    require(&load, source::TEMPERATURE);
    let before = idle.device.temperature.clone().expect("idle temperature");
    let after = load.device.temperature.clone().expect("load temperature");
    eprintln!(
        "GPU max {:.1} C -> {:.1} C, average {:.1} C -> {:.1} C, {} sensors",
        before.max_c, after.max_c, before.avg_c, after.avg_c, after.sensors
    );
    assert!(after.sensors > 0);
    assert!((5.0..=130.0).contains(&after.avg_c), "{}", after.avg_c);
    assert!(after.max_c >= after.avg_c);
    assert!(
        after.max_c >= before.max_c - 2.0,
        "the GPU cooled down under load: {:.1} C -> {:.1} C",
        before.max_c,
        after.max_c
    );
}
