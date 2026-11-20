use super::metrics::Exporter;
use crate::engine::Engine;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, PoisonError, RwLock};
use std::time::{Duration, Instant};

/// What HTTP handlers read. Replaced wholesale after each sample.
pub(super) struct Shared {
    pub(super) metrics: Arc<str>,
    pub(super) report_json: Option<Arc<str>>,
    pub(super) last_sample: Option<Instant>,
}

impl Shared {
    pub(super) fn new(metrics: String) -> Self {
        Self {
            metrics: Arc::from(metrics),
            report_json: None,
            last_sample: None,
        }
    }
}

pub(super) fn collect(
    engine: &Engine,
    mut exporter: Exporter,
    shared: &RwLock<Shared>,
    stop: &AtomicBool,
) {
    while !stop.load(Ordering::SeqCst) {
        let report = match engine.recv_timeout(Duration::from_millis(200)) {
            Ok(report) => report,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let metrics = exporter.update(&report);
        let json = match serde_json::to_string(&report) {
            Ok(json) => Some(Arc::from(json)),
            Err(error) => {
                eprintln!("macgpu: cannot encode report as JSON: {error}");
                None
            }
        };
        let mut shared = shared.write().unwrap_or_else(PoisonError::into_inner);
        shared.metrics = Arc::from(metrics);
        shared.report_json = json;
        shared.last_sample = Some(Instant::now());
    }
    stop.store(true, Ordering::SeqCst);
}
