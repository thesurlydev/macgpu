//! `powermetrics` GPU section, used only by `macgpu doctor` as an independent
//! cross-check of the IOReport numbers.
//!
//! powermetrics needs root, and macgpu never prompts for a password: the
//! cross-check runs only when `sudo -n` already works (cached credentials or a
//! passwordless rule). Note that powermetrics exits with status 0 even when it
//! refuses to run unprivileged, so success is judged from its output.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct PowerStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_frequency_mhz: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_residency_pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_residency_pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_mw: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_state: Option<String>,
    /// Residency per active frequency, in powermetrics' order: the GPU's DVFS
    /// frequency table, from the `GPU HW active residency` line.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub frequency_residency: Vec<FrequencyResidency>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct FrequencyResidency {
    pub mhz: u32,
    pub pct: f64,
}

/// True when `sudo -n` works without a password. Never prompts.
pub fn sudo_available() -> bool {
    Command::new("sudo")
        .args(["-n", "true"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// A running one-sample `sudo -n /usr/bin/powermetrics` over `interval`.
pub struct Sampling {
    child: Option<Child>,
}

/// Starts one powermetrics GPU sample covering the next `interval`. Use
/// [`Sampling::finish`] to collect it, so the caller can take its own samples
/// over the same window meanwhile.
pub fn start(interval: Duration) -> Result<Sampling> {
    // A zero or very short interval samples no useful GPU work; 100 ms is the
    // smallest window that still gives powermetrics time to collect counters.
    let millis = interval.as_millis().clamp(100, u32::MAX as u128) as u32;
    let child = Command::new("sudo")
        .args([
            "-n",
            "/usr/bin/powermetrics",
            "-s",
            "gpu_power",
            "-i",
            &millis.to_string(),
            "-n",
            "1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to run powermetrics")?;
    Ok(Sampling { child: Some(child) })
}

impl Drop for Sampling {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // SIGTERM, not `kill()`: `sudo` relays it to the root-owned
            // powermetrics, whereas SIGKILL would end only `sudo`.
            // SAFETY: `kill` only sends a signal; the pid is a child we have not yet reaped.
            unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
            let _ = child.wait();
        }
    }
}

impl Sampling {
    /// Waits for the sample and parses it.
    pub fn finish(mut self) -> Result<PowerStats> {
        // Taking the child leaves nothing for `Drop` to kill or reap.
        let output = self
            .child
            .take()
            .context("powermetrics sample already finished")?
            .wait_with_output()
            .context("failed to run powermetrics")?;
        let text = String::from_utf8_lossy(&output.stdout);
        let stats = parse(&text);
        if stats.active_residency_pct.is_none()
            && stats.active_frequency_mhz.is_none()
            && stats.power_mw.is_none()
        {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let reason = stderr.trim();
            if reason.is_empty() {
                bail!("could not parse powermetrics GPU output");
            }
            bail!("powermetrics failed: {reason}");
        }
        Ok(stats)
    }
}

pub fn parse(text: &str) -> PowerStats {
    let mut stats = PowerStats::default();

    for line in text.lines() {
        let line = line.trim();

        if let Some(value) = line.strip_prefix("GPU HW active frequency:") {
            stats.active_frequency_mhz = value.trim().trim_end_matches("MHz").trim().parse().ok();
        } else if let Some(value) = line.strip_prefix("GPU HW active residency:") {
            stats.active_residency_pct = value
                .trim()
                .split('%')
                .next()
                .and_then(|value| value.trim().parse().ok());
            stats.frequency_residency = parse_frequency_residency(value);
        } else if let Some(value) = line.strip_prefix("GPU idle residency:") {
            stats.idle_residency_pct = value
                .trim()
                .split('%')
                .next()
                .and_then(|value| value.trim().parse().ok());
        } else if let Some(value) = line.strip_prefix("GPU Power:") {
            stats.power_mw = value.trim().trim_end_matches("mW").trim().parse().ok();
        } else if let Some(value) = line.strip_prefix("GPU SW requested state:") {
            stats.requested_state = value
                .split('(')
                .nth(1)
                .and_then(|value| value.split(':').next())
                .map(|value| value.trim().to_string());
        }
    }

    stats
}

/// Parses the parenthesized list in `3.07% (338 MHz: 3.1% 618 MHz:   0%)`.
fn parse_frequency_residency(value: &str) -> Vec<FrequencyResidency> {
    let Some(list) = value
        .split_once('(')
        .map(|(_, rest)| rest.split(')').next().unwrap_or(rest))
    else {
        return Vec::new();
    };
    let tokens: Vec<&str> = list.split_whitespace().collect();
    tokens
        .windows(3)
        .filter(|window| window[1] == "MHz:")
        .filter_map(|window| {
            Some(FrequencyResidency {
                mhz: window[0].parse().ok()?,
                pct: window[2].trim_end_matches('%').parse().ok()?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "\
*** Sampled system activity (Sun Oct 25 14:54:49 2026 -0700) (2015.39ms elapsed) ***

**** GPU usage ****

GPU HW active frequency: 338 MHz
GPU HW active residency:   3.07% (338 MHz: 3.1% 618 MHz:   0% 796 MHz:   0% 1578 MHz: .02%)
GPU SW requested state: (P1 : 100% P2 :   0% P3 :   0%)
GPU SW state: (SW_P1 : 3.1% SW_P2 :   0%)
GPU idle residency:  96.93%
GPU Power: 28 mW
";

    #[test]
    fn parses_gpu_power_section() {
        let stats = parse(FIXTURE);
        assert_eq!(stats.active_frequency_mhz, Some(338));
        assert_eq!(stats.active_residency_pct, Some(3.07));
        assert_eq!(stats.idle_residency_pct, Some(96.93));
        assert_eq!(stats.power_mw, Some(28));
        assert_eq!(stats.requested_state.as_deref(), Some("P1"));
    }

    #[test]
    fn parses_frequency_residency() {
        let stats = parse(FIXTURE);
        let table: Vec<(u32, f64)> = stats
            .frequency_residency
            .iter()
            .map(|entry| (entry.mhz, entry.pct))
            .collect();
        assert_eq!(
            table,
            vec![(338, 3.1), (618, 0.0), (796, 0.0), (1578, 0.02)]
        );
    }

    #[test]
    fn unprivileged_output_parses_to_nothing() {
        let stats = parse("powermetrics must be invoked as the superuser\n");
        assert_eq!(stats, PowerStats::default());
    }
}
