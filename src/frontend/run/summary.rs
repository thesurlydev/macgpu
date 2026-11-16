//! Builds and renders the `macgpu run` summary.

use super::Measurement;
use crate::cli::RunArgs;
use crate::format;
use serde::Serialize;
use std::fmt::Write as _;
use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;
use std::time::Duration;

impl Measurement {
    fn process_summaries(&self) -> Vec<ProcessSummary> {
        let mut processes: Vec<ProcessSummary> = self
            .processes
            .iter()
            .filter(|(_, totals)| totals.gpu_ns > 0 || totals.peak_memory_bytes.is_some())
            .map(|((pid, _), totals)| ProcessSummary {
                pid: *pid,
                name: totals.name.clone(),
                gpu_seconds: totals.gpu_ns as f64 / 1e9,
                peak_graphics_memory_bytes: totals.peak_memory_bytes,
                energy_j: totals.energy_j,
            })
            .collect();
        processes.sort_by(|a, b| b.gpu_seconds.total_cmp(&a.gpu_seconds));
        processes
    }

    fn energy_method(&self) -> Option<&'static str> {
        match (self.energy_from_app, self.energy_from_device) {
            (true, true) => Some("mixed"),
            (true, false) => Some("app_energy_by_gpu_time"),
            (false, true) => Some("device_energy_by_gpu_time"),
            (false, false) => None,
        }
    }

    fn notes(&self, interval: Duration, still_running: usize) -> Vec<String> {
        let mut notes = vec![format!(
            "GPU time a process used between its last sample and its exit is not measured; \
         samples were {:.0} ms apart",
            interval.as_secs_f64() * 1000.0
        )];
        if self.energy_missing_windows > 0 {
            notes.push(format!(
                "energy is incomplete: {} windows had GPU time but no power data",
                self.energy_missing_windows
            ));
        }
        if self.exit_estimate_unknown > 0 {
            notes.push(format!(
            "in {} windows a process exited while other GPU work overlapped, so its time after the last sample could not be estimated",
            self.exit_estimate_unknown
        ));
        }
        if still_running > 0 {
            notes.push(format!(
            "{still_running} processes started by the command were still running when it exited and were not measured after that"
        ));
        }
        notes
    }

    pub fn summary(
        &self,
        args: &RunArgs,
        interval: Duration,
        status: ExitStatus,
        tracked: usize,
        still_running: usize,
    ) -> RunSummary {
        let wall = self.wall_seconds.max(1e-9);
        let gpu_seconds = self.gpu_ns as f64 / 1e9;
        let processes = self.process_summaries();
        let energy_method = self.energy_method();
        let notes = self.notes(interval, still_running);
        RunSummary {
            schema_version: 1,
            command: args
                .command
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect(),
            exit_code: status.code(),
            signal: status.signal(),
            wall_seconds: self.wall_seconds,
            interval_seconds: interval.as_secs_f64(),
            samples: self.windows,
            gpu_seconds,
            exit_estimate_gpu_seconds: (self.own_coalition.is_some() && self.exit_estimate_ns > 0)
                .then_some(self.exit_estimate_ns as f64 / 1e9),
            gpu_to_wall_pct: gpu_seconds / wall * 100.0,
            avg_busy_ms_per_s: gpu_seconds * 1000.0 / wall,
            peak_busy_ms_per_s: self.peak_ms_per_s,
            avg_share_pct: self.share_seconds / wall * 100.0,
            peak_share_pct: self.peak_share_pct,
            avg_device_pct: self.device_seconds.map(|seconds| seconds / wall * 100.0),
            peak_device_pct: self.peak_device_pct,
            energy_j: self.energy_j,
            avg_power_w: self.energy_j.map(|joules| joules / wall),
            energy_method,
            energy_complete: self.energy_j.is_some() && self.energy_missing_windows == 0,
            device_energy_j: self.device_energy_j,
            peak_graphics_memory_bytes: self.peak_memory_bytes,
            limited_seconds: self.limited_seconds,
            thermal_limited_seconds: self.thermal_limited_seconds,
            processes_tracked: tracked,
            gpu_processes: processes,
            still_running,
            notes,
        }
    }
}

/// The result of `macgpu run`, also its `--json` schema.
#[derive(Debug, Clone, Serialize)]
pub struct RunSummary {
    pub schema_version: u32,
    pub command: Vec<String>,
    /// `None` when a signal ended the command.
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    /// From spawning the command to noticing its exit.
    pub wall_seconds: f64,
    pub interval_seconds: f64,
    pub samples: usize,
    /// GPU time of the command's process tree: the driver's apportioned share
    /// of GPU busy time (`AppUsage`). Work running at the same time as other
    /// apps' work is split between them, so this is at most about wall time
    /// per GPU, and less than the time the command had work in flight when
    /// the GPU was shared.
    pub gpu_seconds: f64,
    /// `gpu_seconds` as a percentage of `wall_seconds`: 100 means the command
    /// had the whole GPU to itself for its whole run.
    pub gpu_to_wall_pct: f64,
    /// Estimated GPU time used after a process's last sample, before it exited,
    /// from the shared coalition's unattributed in-flight time (an upper
    /// estimate, not apportioned). Not included in `gpu_seconds`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_estimate_gpu_seconds: Option<f64>,
    pub avg_busy_ms_per_s: f64,
    /// Highest single-window rate.
    pub peak_busy_ms_per_s: f64,
    /// Share of all processes' GPU time.
    pub avg_share_pct: f64,
    pub peak_share_pct: f64,
    /// Share of the device's active time (needs GPU residency data).
    pub avg_device_pct: Option<f64>,
    pub peak_device_pct: Option<f64>,
    /// Estimated GPU energy of the tree.
    pub energy_j: Option<f64>,
    pub avg_power_w: Option<f64>,
    /// `app_energy_by_gpu_time`, `device_energy_by_gpu_time`, or `mixed`.
    pub energy_method: Option<&'static str>,
    /// False when some windows had GPU time but no power data.
    pub energy_complete: bool,
    /// Whole-GPU energy during the run, all processes included.
    pub device_energy_j: Option<f64>,
    pub peak_graphics_memory_bytes: Option<u64>,
    /// Time the GPU spent under a power or thermal limiter state during the
    /// run (clocks are not necessarily reduced).
    pub limited_seconds: Option<f64>,
    /// Time a thermal (CLTM) cap was in force.
    pub thermal_limited_seconds: Option<f64>,
    /// Processes seen in the command's tree.
    pub processes_tracked: usize,
    /// Tree members that used the GPU or held graphics memory.
    pub gpu_processes: Vec<ProcessSummary>,
    pub still_running: usize,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessSummary {
    pub pid: i32,
    pub name: String,
    pub gpu_seconds: f64,
    pub peak_graphics_memory_bytes: Option<u64>,
    pub energy_j: Option<f64>,
}

impl RunSummary {
    pub fn render(&self) -> String {
        let mut out = String::new();
        let command = self.command.join(" ");
        let ended = match (self.exit_code, self.signal) {
            (Some(code), _) => format!("exited with status {code}"),
            (None, Some(signal)) => format!("was killed by signal {signal}"),
            (None, None) => "ended".into(),
        };
        let _ = writeln!(
            out,
            "macgpu run: `{}` {ended} after {}",
            format::truncate(&command, 60),
            seconds(self.wall_seconds)
        );
        let _ = writeln!(
            out,
            "  GPU time      {} apportioned share = {:.1}% of wall time  (avg {:.1} ms/s, peak {:.1} ms/s)",
            seconds(self.gpu_seconds),
            self.gpu_to_wall_pct,
            self.avg_busy_ms_per_s,
            self.peak_busy_ms_per_s,
        );
        if let Some(extra) = self.exit_estimate_gpu_seconds {
            let _ = writeln!(
                out,
                "                + up to {} in flight after a process's last sample (coalition estimate)",
                seconds(extra)
            );
        }
        match (self.avg_device_pct, self.peak_device_pct) {
            (Some(avg), Some(peak)) => {
                let _ = writeln!(
                    out,
                    "  device share  avg {avg:.1}%, peak {peak:.1}% of GPU active time"
                );
            }
            _ => {
                let _ = writeln!(
                    out,
                    "  GPU share     avg {:.1}%, peak {:.1}% of all processes' GPU time",
                    self.avg_share_pct, self.peak_share_pct
                );
            }
        }
        let whole = self
            .device_energy_j
            .map(|joules| format!("; whole GPU {joules:.1} J"))
            .unwrap_or_default();
        match (self.energy_j, self.avg_power_w) {
            (Some(joules), Some(watts)) => {
                let method = match self.energy_method {
                    Some("device_energy_by_gpu_time") => "whole-GPU energy by GPU time",
                    Some("mixed") => "app or whole-GPU energy by GPU time",
                    _ => "app energy by GPU time",
                };
                let partial = if self.energy_complete {
                    ""
                } else {
                    ", incomplete"
                };
                let _ = writeln!(
                    out,
                    "  GPU energy    {joules:.1} J, avg {}  (estimate: {method}{partial}{whole})",
                    format::watts(watts)
                );
            }
            _ => {
                let _ = writeln!(
                    out,
                    "  GPU energy    unavailable (no GPU power data{whole})"
                );
            }
        }
        let _ = writeln!(
            out,
            "  GPU memory    {}",
            self.peak_graphics_memory_bytes
                .map(|bytes| format!("peak {}", format::bytes(bytes)))
                .unwrap_or_else(|| "unavailable".into())
        );
        match self.limited_seconds {
            Some(limited) => {
                let thermal = self
                    .thermal_limited_seconds
                    .map(|thermal| format!(", thermal cap {}", seconds(thermal)))
                    .unwrap_or_default();
                let _ = writeln!(
                    out,
                    "  limiter       {} under a power or thermal limiter state{thermal}",
                    seconds(limited)
                );
            }
            None => {
                let _ = writeln!(out, "  limiter       unavailable");
            }
        }
        let busiest: Vec<String> = self
            .gpu_processes
            .iter()
            .filter(|process| process.gpu_seconds > 0.0)
            .take(3)
            .map(|process| {
                format!(
                    "{} ({}) {}",
                    process.name,
                    process.pid,
                    seconds(process.gpu_seconds)
                )
            })
            .collect();
        let used = self
            .gpu_processes
            .iter()
            .filter(|process| process.gpu_seconds > 0.0)
            .count();
        let _ = writeln!(
            out,
            "  processes     {} tracked, {used} used the GPU{}{}",
            self.processes_tracked,
            if busiest.is_empty() { "" } else { ": " },
            busiest.join(", ")
        );
        let _ = writeln!(
            out,
            "  sampling      {} windows of {:.0} ms",
            self.samples,
            self.interval_seconds * 1000.0
        );
        for note in &self.notes {
            let _ = writeln!(out, "  note: {note}");
        }
        out
    }
}

fn seconds(value: f64) -> String {
    if value < 10.0 {
        format!("{value:.3} s")
    } else {
        format::seconds(value)
    }
}
