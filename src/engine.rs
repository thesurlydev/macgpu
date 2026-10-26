//! The sampling engine: a background thread that owns the [`Sampler`], takes a
//! snapshot every interval, and sends one [`Report`] per window.
//!
//! Front ends read reports from the engine (or from a recording) through
//! [`ReportFeed`], so sampling cadence never depends on rendering.

use crate::analyze::{AnalyzeOptions, analyze};
use crate::model::Report;
use crate::snapshot::{Sampler, SourceSet};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub const MIN_INTERVAL: Duration = Duration::from_millis(100);
pub const MAX_INTERVAL: Duration = Duration::from_secs(3600);

/// How often to take a cheap firmware utilization reading between snapshots.
pub const FIRMWARE_POLL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngineConfig {
    pub interval: Duration,
    pub sources: SourceSet,
    pub options: AnalyzeOptions,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            sources: SourceSet::default(),
            options: AnalyzeOptions::default(),
        }
    }
}

enum Control {
    SetInterval(Duration),
    SetOptions(AnalyzeOptions),
}

/// A running sampler thread. Dropping it stops the thread.
pub struct Engine {
    reports: Receiver<Report>,
    control: Option<Sender<Control>>,
    thread: Option<JoinHandle<()>>,
    interval: Duration,
}

impl Engine {
    pub fn start(config: EngineConfig) -> Self {
        let (report_sender, reports) = mpsc::sync_channel(64);
        let (control, control_receiver) = mpsc::channel();
        let interval = clamp_interval(config.interval);
        let thread = thread::Builder::new()
            .name("macgpu-sampler".into())
            .spawn(move || run(config, interval, report_sender, control_receiver))
            .expect("spawn sampler thread");
        Self {
            reports,
            control: Some(control),
            thread: Some(thread),
            interval,
        }
    }

    /// Waits for the next report. `None` if the sampler thread stopped.
    pub fn recv(&self) -> Option<Report> {
        self.reports.recv().ok()
    }

    pub fn recv_timeout(&self, timeout: Duration) -> Result<Report, RecvTimeoutError> {
        self.reports.recv_timeout(timeout)
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Changes the sampling interval. Takes effect from the current window.
    pub fn set_interval(&mut self, interval: Duration) {
        self.interval = clamp_interval(interval);
        self.send(Control::SetInterval(self.interval));
    }

    pub fn set_options(&mut self, options: AnalyzeOptions) {
        self.send(Control::SetOptions(options));
    }

    fn send(&self, control: Control) {
        if let Some(sender) = &self.control {
            let _ = sender.send(control);
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Closing the control channel stops the thread at its next wait.
        self.control.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub fn clamp_interval(interval: Duration) -> Duration {
    interval.clamp(MIN_INTERVAL, MAX_INTERVAL)
}

fn run(
    config: EngineConfig,
    mut interval: Duration,
    reports: mpsc::SyncSender<Report>,
    control: Receiver<Control>,
) {
    let mut options = config.options;
    let mut sampler = Sampler::new(config.sources);
    let mut previous = sampler.snapshot();
    let mut due = previous.at + interval;

    loop {
        // Wait for the window to end, polling firmware utilization and control.
        loop {
            let now = Instant::now();
            if now >= due {
                break;
            }
            let wait = (due - now).min(FIRMWARE_POLL);
            match control.recv_timeout(wait) {
                Ok(Control::SetInterval(new_interval)) => {
                    interval = new_interval;
                    due = previous.at + interval;
                }
                Ok(Control::SetOptions(new_options)) => options = new_options,
                Err(RecvTimeoutError::Timeout) => {
                    if due.saturating_duration_since(Instant::now()) > FIRMWARE_POLL / 2 {
                        sampler.poll_firmware();
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }

        let current = sampler.snapshot();
        let report = analyze(&previous, &current, &options);
        previous = current;
        if reports.send(report).is_err() {
            return;
        }

        // Keep a fixed cadence, but never try to catch up after a stall.
        due += interval;
        let now = Instant::now();
        if due <= now {
            due = now + interval;
        }
    }
}

/// What a [`ReportFeed`] produced.
#[derive(Debug)]
pub enum FeedEvent {
    Report(Box<Report>),
    /// No report arrived before the timeout.
    Timeout,
    /// The feed is exhausted (end of a recording, or the sampler stopped).
    End,
}

/// A stream of reports: live from an [`Engine`], or replayed from a recording.
pub trait ReportFeed {
    /// Waits up to `timeout` for the next report.
    fn next_report(&mut self, timeout: Duration) -> FeedEvent;
    /// The feed's sampling interval (live) or the recording's typical interval.
    fn interval(&self) -> Duration;
    /// Changes the interval. Returns false if the feed cannot (recordings).
    fn set_interval(&mut self, interval: Duration) -> bool;
    /// True for live sampling, false for recordings.
    fn is_live(&self) -> bool;
    /// Changes whether idle processes are included, where supported.
    fn set_include_idle(&mut self, _include_idle: bool) {}
}

impl ReportFeed for Engine {
    fn next_report(&mut self, timeout: Duration) -> FeedEvent {
        match self.recv_timeout(timeout) {
            Ok(report) => FeedEvent::Report(Box::new(report)),
            Err(RecvTimeoutError::Timeout) => FeedEvent::Timeout,
            Err(RecvTimeoutError::Disconnected) => FeedEvent::End,
        }
    }

    fn interval(&self) -> Duration {
        Engine::interval(self)
    }

    fn set_interval(&mut self, interval: Duration) -> bool {
        Engine::set_interval(self, interval);
        true
    }

    fn is_live(&self) -> bool {
        true
    }

    fn set_include_idle(&mut self, include_idle: bool) {
        self.set_options(AnalyzeOptions { include_idle });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_emits_reports_at_interval() {
        let config = EngineConfig {
            interval: Duration::from_millis(200),
            ..EngineConfig::default()
        };
        let started = Instant::now();
        let engine = Engine::start(config);
        let first = engine.recv().expect("first report");
        let second = engine.recv().expect("second report");
        let elapsed = started.elapsed();
        assert!(
            first.window_seconds > 0.15 && first.window_seconds < 0.5,
            "{}",
            first.window_seconds
        );
        assert!(second.timestamp_ms >= first.timestamp_ms);
        assert!(elapsed >= Duration::from_millis(380), "{elapsed:?}");
        assert!(first.totals.gpu_clients > 0);
        drop(engine);
    }

    #[test]
    fn interval_is_clamped() {
        assert_eq!(clamp_interval(Duration::from_millis(1)), MIN_INTERVAL);
        assert_eq!(clamp_interval(Duration::from_secs(10_000)), MAX_INTERVAL);
    }
}
