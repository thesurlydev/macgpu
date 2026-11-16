//! A controlled Metal compute workload, used by `macgpu doctor` and the
//! end-to-end tests to generate known GPU load.
//!
//! The load runs in a child process (`macgpu __load ...`) so attribution is
//! always tested on a different PID than the observer, and so the observing
//! macgpu process never opens a Metal device (which would make it a GPU client
//! itself). The child reports what it did as JSON lines on stdout, including
//! its own GPU time measured from command-buffer timestamps: the ground truth
//! macgpu's numbers are compared against.
//!
//! Protocol, one JSON object per line:
//! 1. `{"event":"ready",...}` once the device, queues, pipeline, and buffer exist.
//! 2. With `--wait`, the child then blocks until it reads a line on stdin.
//! 3. `{"event":"done",...}` when the load finishes.
//! 4. With `--hold`, the child keeps its device, queues, and buffer open until
//!    stdin reaches end of file, so an observer can take a closing sample while
//!    the GPU client still exists. Without it the child exits right away.
//!
//! How the ground truth relates to `AppUsage` (measured on an M4 Max, macOS 27):
//! - `AppUsage` apportions the GPU among contexts that run at the same time:
//!   summed over every process it stays at or below the window (956-982 ms/s
//!   with the GPU saturated) and tracks GPUPH active residency.
//! - `GPUEndTime - GPUStartTime` spans are wall time in flight, so they also
//!   cover other processes' work that overlapped (WindowServer frames, other
//!   loads). Under contention the span sum is an upper bound of the process's
//!   `AppUsage`, which reads lower by roughly the overlapping work.
//! - With very short command buffers (under about 1 ms) `AppUsage` also counts
//!   60-100 µs of per-buffer overhead that falls outside the span.
//! - The default shape (one dispatch of 2-3 ms per command buffer) keeps both
//!   effects small: within 0.7% when the rest of the GPU is quiet.
//!
//! [`AttributionBounds`] turns this into the range macgpu's number must fall in.

use crate::model::Report;
use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, BufReader, Lines, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

/// Name of the hidden subcommand that runs the load.
pub const SUBCOMMAND: &str = "__load";

/// Each thread runs `iterations` rounds of dependent transcendental math, so
/// the kernel is compute bound and its duration scales with `iterations`.
const KERNEL_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void spin(device float *out [[buffer(0)]],
                 constant uint &iterations [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    float x = float(id) * 0.001f;
    for (uint i = 0; i < iterations; i++) {
        x = fma(x, 1.000001f, 0.5f);
        x = sin(x) * 0.99f + cos(x);
    }
    out[id] = x;
}
";

const THREADGROUP_WIDTH: usize = 256;

/// Options for one load run. Also the arguments of `macgpu __load`.
#[derive(Args, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadConfig {
    /// How long to generate load, in seconds.
    #[arg(long, default_value_t = 3.0)]
    pub seconds: f64,

    /// Fraction of the time the GPU is kept busy, in (0, 1]. After each round
    /// of command buffers the load sleeps so busy time is this share of wall time.
    #[arg(long, default_value_t = 1.0)]
    pub duty: f64,

    /// Number of live command queues. Each round commits one command buffer on
    /// every queue, so the queues run concurrently.
    #[arg(long, default_value_t = 1)]
    pub queues: usize,

    /// Allocate a shared buffer of this many MiB and write every page, so it is
    /// resident and counted in the process's graphics footprint.
    #[arg(long, default_value_t = 0)]
    pub alloc_mib: u64,

    /// Destroy the oldest command queue and create a new one this often, in
    /// milliseconds. 0 turns churn off.
    #[arg(long, default_value_t = 0)]
    pub churn_ms: u64,

    /// With churn, wait this long between releasing a queue and creating its
    /// replacement, in milliseconds, so observers can see the gap.
    #[arg(long, default_value_t = 0)]
    pub churn_gap_ms: u64,

    /// Kernel loop iterations per thread; sets the length of one command buffer.
    #[arg(long, default_value_t = 4096)]
    pub iterations: u32,

    /// Threads per dispatch.
    #[arg(long, default_value_t = 1 << 18)]
    pub threads: usize,

    /// Wait for a line on stdin after printing `ready` before starting the load.
    #[arg(long)]
    pub wait: bool,

    /// After printing `done`, keep the device open until stdin is closed.
    #[arg(long)]
    pub hold: bool,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            seconds: 3.0,
            duty: 1.0,
            queues: 1,
            alloc_mib: 0,
            churn_ms: 0,
            churn_gap_ms: 0,
            iterations: 4096,
            threads: 1 << 18,
            wait: false,
            hold: false,
        }
    }
}

impl LoadConfig {
    /// Arguments for `macgpu __load` that reproduce this config.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = vec![
            SUBCOMMAND.to_string(),
            format!("--seconds={}", self.seconds),
            format!("--duty={}", self.duty),
            format!("--queues={}", self.queues),
            format!("--alloc-mib={}", self.alloc_mib),
            format!("--churn-ms={}", self.churn_ms),
            format!("--churn-gap-ms={}", self.churn_gap_ms),
            format!("--iterations={}", self.iterations),
            format!("--threads={}", self.threads),
        ];
        if self.wait {
            args.push("--wait".into());
        }
        if self.hold {
            args.push("--hold".into());
        }
        args
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.seconds.is_finite() && self.seconds >= 0.0,
            "--seconds must be a non-negative number"
        );
        ensure!(
            self.duty > 0.0 && self.duty <= 1.0,
            "--duty must be in (0, 1]"
        );
        ensure!(
            (1..=64).contains(&self.queues),
            "--queues must be between 1 and 64"
        );
        ensure!(self.iterations > 0, "--iterations must be positive");
        ensure!(
            self.threads >= THREADGROUP_WIDTH,
            "--threads must be at least {THREADGROUP_WIDTH}"
        );
        Ok(())
    }
}

/// One line of the load's output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum LoadEvent {
    Ready(Ready),
    Done(LoadResult),
}

/// The load is set up and about to start (or waiting for the go line).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Ready {
    pub pid: i32,
    pub device: String,
    pub queues: usize,
    /// Bytes of the touched buffer, 0 when none was requested.
    pub alloc_bytes: u64,
    pub recommended_working_set_bytes: u64,
    pub current_allocated_bytes: u64,
}

/// What the load did, measured by the load itself.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoadResult {
    pub pid: i32,
    pub wall_ns: u64,
    /// Sum over every command buffer of `GPUEndTime - GPUStartTime`: the time
    /// the GPU spent executing this process's work, per queue, summed.
    pub gpu_ns: u64,
    pub command_buffers: u64,
    /// Queues created over the run, including the initial ones.
    pub queues_created: u64,
    /// Queues released while the load ran (churn).
    pub queues_destroyed: u64,
    /// Per queue, in creation order.
    pub per_queue: Vec<QueueResult>,
    /// `mach_absolute_time` in ns at the first submission and after the last
    /// completion, comparable with `AppUsage` `lastSubmittedTime`.
    pub start_mach_ns: u64,
    pub end_mach_ns: u64,
    pub alloc_bytes: u64,
}

impl LoadResult {
    pub fn busy_fraction(&self) -> f64 {
        if self.wall_ns == 0 {
            0.0
        } else {
            self.gpu_ns as f64 / self.wall_ns as f64
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueueResult {
    pub gpu_ns: u64,
    pub command_buffers: u64,
    /// True when the queue was released before the load ended.
    pub destroyed: bool,
}

type Device = ProtocolObject<dyn MTLDevice>;
type Queue = ProtocolObject<dyn MTLCommandQueue>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
type Buffer = ProtocolObject<dyn MTLBuffer>;

struct LiveQueue {
    queue: Retained<Queue>,
    /// Index into `LoadResult::per_queue`.
    slot: usize,
}

/// Runs `macgpu __load`: prints JSON lines on stdout and returns the exit code.
pub fn run(config: &LoadConfig) -> Result<i32> {
    let stdout = io::stdout();
    let emit = |event: &LoadEvent| -> Result<()> {
        let mut out = stdout.lock();
        writeln!(out, "{}", serde_json::to_string(event)?)?;
        out.flush().context("failed to flush stdout")
    };
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    let mut load = Load::new(config)?;
    emit(&LoadEvent::Ready(load.ready()))?;
    if config.wait {
        // A closed stdin also starts the load; the observer may have exited.
        let _ = lines.next();
    }
    let result = load.run()?;
    emit(&LoadEvent::Done(result))?;
    if config.hold {
        for line in lines {
            if line.is_err() {
                break;
            }
        }
    }
    Ok(0)
}

struct Load {
    config: LoadConfig,
    device: Retained<Device>,
    pipeline: Retained<Pipeline>,
    output: Retained<Buffer>,
    iterations: Retained<Buffer>,
    queues: Vec<LiveQueue>,
    per_queue: Vec<QueueResult>,
    /// Kept alive (and resident) for the whole run.
    _allocation: Option<Retained<Buffer>>,
    alloc_bytes: u64,
}

impl Load {
    fn new(config: &LoadConfig) -> Result<Self> {
        config.validate()?;
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(KERNEL_SOURCE), None)
            .map_err(|error| anyhow::anyhow!("compiling the load kernel failed: {error}"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("spin"))
            .context("load kernel function missing")?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| anyhow::anyhow!("creating the load pipeline failed: {error}"))?;
        let output = device
            .newBufferWithLength_options(
                config.threads * size_of::<f32>(),
                MTLResourceOptions::StorageModePrivate,
            )
            .context("allocating the output buffer failed")?;
        let iterations = device
            .newBufferWithLength_options(size_of::<u32>(), MTLResourceOptions::StorageModeShared)
            .context("allocating the iterations buffer failed")?;
        // SAFETY: the buffer is shared, at least 4 bytes, and not yet in use by the GPU.
        unsafe {
            iterations
                .contents()
                .cast::<u32>()
                .as_ptr()
                .write(config.iterations)
        };

        let alloc_bytes = config.alloc_mib << 20;
        let allocation = if alloc_bytes > 0 {
            let buffer = device
                .newBufferWithLength_options(
                    alloc_bytes as usize,
                    MTLResourceOptions::StorageModeShared,
                )
                .with_context(|| format!("allocating {} MiB failed", config.alloc_mib))?;
            // SAFETY: the shared buffer's contents are `alloc_bytes` long and only
            // this thread touches them. Writing every page makes it resident.
            unsafe {
                std::ptr::write_bytes(
                    buffer.contents().cast::<u8>().as_ptr(),
                    1,
                    alloc_bytes as usize,
                )
            };
            Some(buffer)
        } else {
            None
        };

        let mut load = Self {
            config: config.clone(),
            device,
            pipeline,
            output,
            iterations,
            queues: Vec::new(),
            per_queue: Vec::new(),
            _allocation: allocation,
            alloc_bytes,
        };
        for _ in 0..config.queues {
            load.add_queue()?;
        }
        Ok(load)
    }

    fn ready(&self) -> Ready {
        Ready {
            pid: std::process::id() as i32,
            device: self.device.name().to_string(),
            queues: self.queues.len(),
            alloc_bytes: self.alloc_bytes,
            recommended_working_set_bytes: self.device.recommendedMaxWorkingSetSize(),
            current_allocated_bytes: self.device.currentAllocatedSize() as u64,
        }
    }

    fn add_queue(&mut self) -> Result<()> {
        let queue = self
            .device
            .newCommandQueue()
            .context("creating a command queue failed")?;
        self.queues.push(LiveQueue {
            queue,
            slot: self.per_queue.len(),
        });
        self.per_queue.push(QueueResult::default());
        Ok(())
    }

    fn run(&mut self) -> Result<LoadResult> {
        let threads = MTLSize {
            width: self.config.threads,
            height: 1,
            depth: 1,
        };
        let group = MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        let churn = (self.config.churn_ms > 0).then(|| Duration::from_millis(self.config.churn_ms));
        let mut destroyed = 0u64;
        let mut gpu_s = 0.0f64;
        let mut command_buffers = 0u64;

        let started = Instant::now();
        let start_mach_ns = crate::sys::clock::mach_now_ns();
        let deadline = started + Duration::from_secs_f64(self.config.seconds);
        let mut next_churn = churn.map(|period| started + period);

        while Instant::now() < deadline {
            if let (Some(period), Some(due)) = (churn, next_churn)
                && Instant::now() >= due
            {
                // Release the oldest queue; its command buffers have all completed.
                let oldest = self.queues.remove(0);
                self.per_queue[oldest.slot].destroyed = true;
                drop(oldest);
                destroyed += 1;
                if self.config.churn_gap_ms > 0 {
                    std::thread::sleep(Duration::from_millis(self.config.churn_gap_ms));
                }
                self.add_queue()?;
                next_churn = Some(due + period);
            }

            let round_started = Instant::now();
            // Command buffers are autoreleased; drain them every round so a
            // released queue is really freed.
            autoreleasepool(|_| -> Result<()> {
                let mut committed = Vec::with_capacity(self.queues.len());
                for live in &self.queues {
                    let buffer = live
                        .queue
                        .commandBuffer()
                        .context("creating a command buffer failed")?;
                    let encoder = buffer
                        .computeCommandEncoder()
                        .context("creating a compute encoder failed")?;
                    encoder.setComputePipelineState(&self.pipeline);
                    // SAFETY: both buffers outlive the command buffer, which is
                    // waited on below, and match the kernel's argument types.
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(&self.output), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&self.iterations), 0, 1);
                    }
                    encoder.dispatchThreads_threadsPerThreadgroup(threads, group);
                    encoder.endEncoding();
                    buffer.commit();
                    committed.push((live.slot, buffer));
                }
                for (slot, buffer) in committed {
                    buffer.waitUntilCompleted();
                    if buffer.status() != MTLCommandBufferStatus::Completed {
                        let reason = buffer
                            .error()
                            .map(|error| error.to_string())
                            .unwrap_or_else(|| format!("status {:?}", buffer.status()));
                        bail!("command buffer failed: {reason}");
                    }
                    let gpu = (buffer.GPUEndTime() - buffer.GPUStartTime()).max(0.0);
                    gpu_s += gpu;
                    let queue = &mut self.per_queue[slot];
                    queue.gpu_ns += (gpu * 1e9) as u64;
                    queue.command_buffers += 1;
                    command_buffers += 1;
                }
                Ok(())
            })?;

            if self.config.duty < 1.0 {
                let busy = round_started.elapsed();
                std::thread::sleep(busy.mul_f64((1.0 - self.config.duty) / self.config.duty));
            }
        }

        Ok(LoadResult {
            pid: std::process::id() as i32,
            wall_ns: started.elapsed().as_nanos() as u64,
            gpu_ns: (gpu_s * 1e9) as u64,
            command_buffers,
            queues_created: self.per_queue.len() as u64,
            queues_destroyed: destroyed,
            per_queue: self.per_queue.clone(),
            start_mach_ns,
            end_mach_ns: crate::sys::clock::mach_now_ns(),
            alloc_bytes: self.alloc_bytes,
        })
    }
}

/// Accepted error of the default load shape on a quiet GPU. Measured on an
/// M4 Max: -0.20% to +2.5%.
pub const TOLERANCE_PCT: f64 = 4.0;
/// How far `AppUsage` may read above the load's own timing. It also counts
/// latency the spans do not, such as the GPU waking between command buffers:
/// up to +5% at 50% duty.
pub const OVER_TOLERANCE_PCT: f64 = 6.0;
/// Accepted error of the contention model.
const CONTENTION_TOLERANCE_PCT: f64 = 6.0;
/// Absolute slack for timestamp granularity and sample alignment.
pub const SLACK_NS: u64 = 5_000_000;
/// The GPU counts as quiet when other processes used less than this share of
/// the window.
pub const QUIET_SHARE: f64 = 0.05;

/// The range macgpu's `AppUsage` time for a load process must fall in, given
/// the load's own measurement and the GPU time other processes used meanwhile.
///
/// - Upper bound: the load's span sum plus [`OVER_TOLERANCE_PCT`]. Other work
///   only ever stretches the spans, so more than this is over-crediting.
/// - Lower bound: other processes' work lands inside the load's spans in
///   proportion to how busy the load kept the GPU, and `AppUsage` apportions
///   it away, so expect `truth - busy_fraction * others`, minus
///   [`CONTENTION_TOLERANCE_PCT`] (and never demand more than `truth` minus
///   [`TOLERANCE_PCT`]). Measured on an M4 Max at duty 0.1-1.0 with 0.1-2.1 s
///   of other work in 3 s windows, macgpu read between -0.4% and +5% of that
///   expectation. `others` must include work `AppUsage` no longer shows
///   (processes that exited inside the window); see [`other_gpu_ns`].
/// - `quiet`: other processes used under [`QUIET_SHARE`] of the window, so for
///   the default (full duty) shape the error should be within [`TOLERANCE_PCT`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttributionBounds {
    pub lower_ns: u64,
    pub upper_ns: u64,
    pub quiet: bool,
}

impl AttributionBounds {
    pub fn new(truth_ns: u64, busy_fraction: f64, others_ns: u64, window_ns: u64) -> Self {
        let truth = truth_ns as f64;
        let slack = SLACK_NS as f64;
        let upper = truth * (1.0 + OVER_TOLERANCE_PCT / 100.0) + slack;
        let expected = truth - busy_fraction.clamp(0.0, 1.0) * others_ns as f64;
        let lower = (truth * (1.0 - TOLERANCE_PCT / 100.0))
            .min(expected - truth * CONTENTION_TOLERANCE_PCT / 100.0)
            - slack;
        Self {
            lower_ns: lower.max(0.0) as u64,
            upper_ns: upper as u64,
            quiet: (others_ns as f64) < window_ns as f64 * QUIET_SHARE,
        }
    }

    pub fn contains(&self, measured_ns: u64) -> bool {
        (self.lower_ns..=self.upper_ns).contains(&measured_ns)
    }
}

/// GPU time other processes used in the window, for [`AttributionBounds`]:
/// their `AppUsage`, or, when IOReport is available and larger, the GPU's
/// active time minus the load's share. The second catches work from processes
/// that exited inside the window, whose `AppUsage` is gone.
pub fn other_gpu_ns(report: &Report, pid: i32) -> u64 {
    let load = report
        .processes
        .iter()
        .find(|process| process.pid == pid)
        .map(|process| process.gpu_ns)
        .unwrap_or(0);
    let by_app_usage = report.totals.attributed_gpu_ns.saturating_sub(load);
    let by_residency = report
        .device
        .active_pct
        .map(|active| (active / 100.0 * report.window_seconds * 1e9) as u64)
        .unwrap_or(0)
        .saturating_sub(load);
    by_app_usage.max(by_residency)
}

/// A `macgpu __load` child process driven over the JSON-lines protocol.
/// Killed (by its own PID) on drop if it is still running.
pub struct LoadProcess {
    pub ready: Ready,
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Lines<BufReader<ChildStdout>>,
}

impl LoadProcess {
    /// Spawns `exe __load` with `config` and waits for its `ready` line.
    /// `wait` and `hold` are used as given: with `wait`, the load starts only on
    /// [`LoadProcess::start`]; with `hold`, its GPU client stays open until
    /// [`LoadProcess::finish`].
    pub fn spawn(exe: &Path, config: &LoadConfig) -> Result<Self> {
        let mut child = Command::new(exe)
            .args(config.to_args())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to start {}", exe.display()))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().context("load stdout missing")?;
        let mut process = Self {
            ready: Ready::default(),
            child,
            stdin,
            lines: BufReader::new(stdout).lines(),
        };
        match process.next_event()? {
            LoadEvent::Ready(ready) => process.ready = ready,
            other => bail!("expected ready from the load, got {other:?}"),
        }
        Ok(process)
    }

    pub fn pid(&self) -> i32 {
        self.ready.pid
    }

    /// Sends the go line to a load spawned with `wait`.
    pub fn start(&mut self) -> Result<()> {
        let stdin = self.stdin.as_mut().context("load stdin closed")?;
        stdin.write_all(b"go\n")?;
        stdin.flush().context("failed to start the load")
    }

    /// Waits for the `done` line.
    pub fn wait_done(&mut self) -> Result<LoadResult> {
        match self.next_event()? {
            LoadEvent::Done(result) => Ok(result),
            other => bail!("expected done from the load, got {other:?}"),
        }
    }

    /// Closes stdin and waits for the child to exit.
    pub fn finish(mut self) -> Result<()> {
        self.stdin.take();
        let status = self.child.wait().context("waiting for the load failed")?;
        ensure!(status.success(), "the load exited with {status}");
        Ok(())
    }

    fn next_event(&mut self) -> Result<LoadEvent> {
        let line = self
            .lines
            .next()
            .context("the load exited without reporting")?
            .context("reading the load's output failed")?;
        serde_json::from_str(&line).with_context(|| format!("unexpected load output {line:?}"))
    }
}

impl Drop for LoadProcess {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        config: LoadConfig,
    }

    #[test]
    fn args_round_trip() {
        let config = LoadConfig {
            seconds: 1.5,
            duty: 0.25,
            queues: 3,
            alloc_mib: 64,
            churn_ms: 200,
            churn_gap_ms: 50,
            iterations: 77,
            threads: 4096,
            wait: true,
            hold: true,
        };
        let args = config.to_args();
        assert_eq!(args[0], SUBCOMMAND);
        let parsed = Wrapper::try_parse_from(
            std::iter::once("load".to_string()).chain(args.into_iter().skip(1)),
        )
        .expect("parse");
        assert_eq!(parsed.config, config);
    }

    #[test]
    fn rejects_bad_config() {
        let bad = [
            LoadConfig {
                duty: 0.0,
                ..LoadConfig::default()
            },
            LoadConfig {
                queues: 0,
                ..LoadConfig::default()
            },
            LoadConfig {
                seconds: f64::NAN,
                ..LoadConfig::default()
            },
        ];
        for config in bad {
            assert!(config.validate().is_err(), "{config:?}");
        }
    }

    #[test]
    fn bounds_on_a_quiet_gpu() {
        let ms = 1_000_000;
        // Others used 1% of the window.
        let bounds = AttributionBounds::new(2_000 * ms, 0.9, 30 * ms, 3_000 * ms);
        assert!(bounds.quiet);
        assert!(bounds.contains(2_000 * ms + 100 * ms), "{bounds:?}");
        assert!(bounds.contains(2_000 * ms - 100 * ms), "{bounds:?}");
        assert!(!bounds.contains(2_000 * ms + 150 * ms), "{bounds:?}");
        assert!(!bounds.contains(2_000 * ms - 160 * ms), "{bounds:?}");
    }

    #[test]
    fn bounds_allow_for_contention() {
        // Measured: spans 2682 ms at 89% busy with 1132 ms of other work in a
        // 3 s window; macgpu read 1668 ms.
        let ms = 1_000_000;
        let bounds = AttributionBounds::new(2_682 * ms, 0.89, 1_132 * ms, 3_010 * ms);
        assert!(!bounds.quiet);
        assert!(bounds.contains(1_668 * ms), "{bounds:?}");
        assert!(!bounds.contains(1_300 * ms), "{bounds:?}");
        assert!(!bounds.contains(2_900 * ms), "{bounds:?}");
    }

    #[test]
    fn events_are_tagged_json_lines() {
        let line = serde_json::to_string(&LoadEvent::Ready(Ready {
            pid: 7,
            ..Ready::default()
        }))
        .expect("serialize");
        assert!(line.starts_with(r#"{"event":"ready","pid":7"#), "{line}");
        let back: LoadEvent = serde_json::from_str(&line).expect("parse");
        assert!(matches!(back, LoadEvent::Ready(ready) if ready.pid == 7));
    }
}
