//! Time series kept by the interactive view: whole-device metrics per report,
//! sparse per-row series keyed by [`RowKey`], and recent events.
//!
//! Charts read the series through [`bucketize`], which averages samples into
//! equal time buckets weighted by how much of each bucket a sample's window
//! covers. Every report describes a window rather than an instant, so this is
//! exact for any zoom level and for recordings with gaps.

use crate::model::{Event, Report};
use crate::view::{Row, RowKey};
use std::collections::{HashMap, VecDeque};

/// Whole-device samples kept: one hour at a 0.5 s interval.
pub const DEVICE_CAPACITY: usize = 7200;
/// Per-row samples kept. Rows only store windows where they did something.
pub const ROW_CAPACITY: usize = 3600;
/// Memory changes kept per row.
pub const MEMORY_CAPACITY: usize = 512;
pub const EVENT_CAPACITY: usize = 200;
/// Rows not seen for this long are dropped.
pub const ROW_RETENTION_MS: u64 = 3_600_000;

/// A fixed-capacity FIFO: pushing onto a full ring drops the oldest entry.
#[derive(Debug, Clone)]
pub struct Ring<T> {
    items: VecDeque<T>,
    capacity: usize,
}

impl<T> Ring<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(capacity.min(256)),
            capacity: capacity.max(1),
        }
    }

    pub fn push(&mut self, item: T) {
        if self.items.len() == self.capacity {
            self.items.pop_front();
        }
        self.items.push_back(item);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn last(&self) -> Option<&T> {
        self.items.back()
    }

    /// Oldest first.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        self.items.iter()
    }
}

/// Whole-device values for one report window. `None` means the source had no value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DevicePoint {
    pub end_ms: u64,
    pub window_ms: u32,
    pub active_pct: Option<f32>,
    pub normalized_pct: Option<f32>,
    pub mhz: Option<f32>,
    pub power_w: Option<f32>,
    pub temperature_c: Option<f32>,
    pub attributed_ms_per_s: f32,
}

impl DevicePoint {
    pub fn from_report(report: &Report) -> Self {
        let device = &report.device;
        Self {
            end_ms: report.timestamp_ms,
            window_ms: window_ms(report),
            active_pct: device.active_pct.map(|value| value as f32),
            normalized_pct: device.freq_normalized_pct.map(|value| value as f32),
            mhz: device.avg_mhz.map(|value| value as f32),
            power_w: device.power_w.map(|value| value as f32),
            temperature_c: device.temperature.as_ref().map(|value| value.avg_c as f32),
            attributed_ms_per_s: report.totals.attributed_ms_per_s as f32,
        }
    }
}

/// One window in which a row used the GPU or drew power.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowPoint {
    pub end_ms: u64,
    pub window_ms: u32,
    pub busy_ms_per_s: f32,
    /// NaN when unknown.
    pub device_pct: f32,
    /// NaN when unknown.
    pub power_w: f32,
}

/// The history of one row.
#[derive(Debug, Clone)]
pub struct RowSeries {
    pub name: String,
    pub points: Ring<RowPoint>,
    /// `(end_ms, bytes)` whenever the row's GPU memory changed.
    pub memory: Ring<(u64, u64)>,
    pub last_seen_ms: u64,
}

impl RowSeries {
    fn new(name: String) -> Self {
        Self {
            name,
            points: Ring::new(ROW_CAPACITY),
            memory: Ring::new(MEMORY_CAPACITY),
            last_seen_ms: 0,
        }
    }

    /// `(end_ms, window_ms, value)` samples for `metric` that end after
    /// `since_ms`, newest first.
    pub fn samples(
        &self,
        metric: RowMetric,
        since_ms: u64,
    ) -> impl Iterator<Item = (u64, u32, f64)> + '_ {
        let recent = self
            .points
            .iter()
            .rev()
            .take_while(move |point| point.end_ms > since_ms);
        recent.filter_map(move |point| {
            let value = match metric {
                RowMetric::Busy => point.busy_ms_per_s,
                RowMetric::DevicePct => point.device_pct,
                RowMetric::Power => point.power_w,
            };
            (!value.is_nan()).then_some((point.end_ms, point.window_ms, value as f64))
        })
    }

    /// GPU memory as a step function sampled at the end of each bucket.
    pub fn memory_buckets(&self, end_ms: u64, span_ms: u64, buckets: usize) -> Vec<Option<f64>> {
        let start = end_ms.saturating_sub(span_ms);
        (0..buckets)
            .map(|index| {
                let bucket_end = start + span_ms * (index as u64 + 1) / buckets.max(1) as u64;
                self.memory
                    .iter()
                    .rev()
                    .find(|(at, _)| *at <= bucket_end)
                    .map(|(_, bytes)| *bytes as f64)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowMetric {
    Busy,
    DevicePct,
    Power,
}

/// Everything the view remembers between reports.
#[derive(Debug)]
pub struct History {
    pub device: Ring<DevicePoint>,
    pub rows: HashMap<RowKey, RowSeries>,
    pub events: Ring<Event>,
}

impl Default for History {
    fn default() -> Self {
        Self {
            device: Ring::new(DEVICE_CAPACITY),
            rows: HashMap::new(),
            events: Ring::new(EVENT_CAPACITY),
        }
    }
}

impl History {
    /// Records one report. `rows` are the report's rows at every grouping level.
    pub fn record(&mut self, report: &Report, rows: &[Row]) {
        let end_ms = report.timestamp_ms;
        let window_ms = window_ms(report);
        self.device.push(DevicePoint::from_report(report));
        for event in &report.events {
            self.events.push(event.clone());
        }

        for row in rows {
            let series = self
                .rows
                .entry(row.key)
                .or_insert_with(|| RowSeries::new(row.name.clone()));
            series.last_seen_ms = end_ms;
            if series.name != row.name {
                series.name.clone_from(&row.name);
            }
            let power = row.power_w.unwrap_or(f64::NAN);
            if row.busy_ms_per_s > 0.0 || power > 0.0 {
                series.points.push(RowPoint {
                    end_ms,
                    window_ms,
                    busy_ms_per_s: row.busy_ms_per_s as f32,
                    device_pct: row.device_pct.map_or(f32::NAN, |value| value as f32),
                    power_w: power as f32,
                });
            }
            if let Some(bytes) = row.memory_bytes
                && series.memory.last().is_none_or(|(_, last)| *last != bytes)
            {
                series.memory.push((end_ms, bytes));
            }
        }

        self.rows
            .retain(|_, series| end_ms.saturating_sub(series.last_seen_ms) <= ROW_RETENTION_MS);
    }

    pub fn latest_ms(&self) -> Option<u64> {
        self.device.last().map(|point| point.end_ms)
    }

    /// `(end_ms, window_ms, value)` for a device metric for windows ending
    /// after `since_ms`, newest first, skipping missing values.
    pub fn device_samples(
        &self,
        metric: fn(&DevicePoint) -> Option<f32>,
        since_ms: u64,
    ) -> impl Iterator<Item = (u64, u32, f64)> + '_ {
        let recent = self
            .device
            .iter()
            .rev()
            .take_while(move |point| point.end_ms > since_ms);
        recent.filter_map(move |point| {
            metric(point).map(|value| (point.end_ms, point.window_ms, value as f64))
        })
    }

    /// Per bucket, how much of it any report covered (in milliseconds). Rows
    /// store only nonzero windows, so this is the denominator for row series.
    pub fn coverage(&self, end_ms: u64, span_ms: u64, buckets: usize) -> Vec<f64> {
        accumulate(
            self.device_samples(|_| Some(1.0), end_ms.saturating_sub(span_ms)),
            end_ms,
            span_ms,
            buckets,
        )
        .into_iter()
        .map(|(_, covered)| covered)
        .collect()
    }

    /// Averages a sparse row series into buckets: windows without a point count as 0.
    pub fn row_buckets(
        &self,
        key: &RowKey,
        metric: RowMetric,
        end_ms: u64,
        span_ms: u64,
        coverage: &[f64],
    ) -> Vec<Option<f64>> {
        let sums = match self.rows.get(key) {
            Some(series) => accumulate(
                series.samples(metric, end_ms.saturating_sub(span_ms)),
                end_ms,
                span_ms,
                coverage.len(),
            ),
            None => vec![(0.0, 0.0); coverage.len()],
        };
        sums.iter()
            .zip(coverage)
            .map(|((sum, _), covered)| (*covered > 0.0).then(|| sum / covered))
            .collect()
    }
}

fn window_ms(report: &Report) -> u32 {
    (report.window_seconds * 1000.0)
        .round()
        .clamp(1.0, u32::MAX as f64) as u32
}

/// Averages `(end_ms, window_ms, value)` samples into `buckets` equal buckets
/// covering `(end_ms - span_ms, end_ms]`. A bucket no sample covers is `None`.
pub fn bucketize(
    samples: impl Iterator<Item = (u64, u32, f64)>,
    end_ms: u64,
    span_ms: u64,
    buckets: usize,
) -> Vec<Option<f64>> {
    accumulate(samples, end_ms, span_ms, buckets)
        .into_iter()
        .map(|(sum, covered)| (covered > 0.0).then(|| sum / covered))
        .collect()
}

/// Per bucket: (sum of value x overlap, total overlap), overlap in milliseconds.
fn accumulate(
    samples: impl Iterator<Item = (u64, u32, f64)>,
    end_ms: u64,
    span_ms: u64,
    buckets: usize,
) -> Vec<(f64, f64)> {
    let mut sums = vec![(0.0, 0.0); buckets];
    if buckets == 0 || span_ms == 0 {
        return sums;
    }
    let start = end_ms as f64 - span_ms as f64;
    let width = span_ms as f64 / buckets as f64;
    for (sample_end, window, value) in samples {
        let sample_end = sample_end as f64;
        let sample_start = sample_end - window as f64;
        if sample_end <= start || sample_start >= end_ms as f64 || !value.is_finite() {
            continue;
        }
        let from = ((sample_start - start) / width).floor().max(0.0) as usize;
        let to = (((sample_end - start) / width).ceil() as usize).min(buckets);
        for (index, sum) in sums.iter_mut().enumerate().take(to).skip(from) {
            let bucket_start = start + index as f64 * width;
            let overlap = sample_end.min(bucket_start + width) - sample_start.max(bucket_start);
            if overlap > 0.0 {
                sum.0 += value * overlap;
                sum.1 += overlap;
            }
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceView, EventKind, Totals};
    use crate::view::Row;

    #[test]
    fn ring_drops_oldest_when_full() {
        let mut ring = Ring::new(3);
        assert_eq!(ring.len(), 0);
        for value in 1..=5 {
            ring.push(value);
        }
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.iter().copied().collect::<Vec<_>>(), vec![3, 4, 5]);
        assert_eq!(ring.last(), Some(&5));
    }

    #[test]
    fn ring_capacity_is_at_least_one() {
        let mut ring = Ring::new(0);
        ring.push(1);
        ring.push(2);
        assert_eq!(ring.iter().copied().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn buckets_are_time_weighted() {
        // Two 1 s windows: 0..1000 at 10, 1000..2000 at 30. Four buckets of 500 ms.
        let samples = vec![(1000u64, 1000u32, 10.0), (2000, 1000, 30.0)];
        let buckets = bucketize(samples.into_iter(), 2000, 2000, 4);
        assert_eq!(
            buckets,
            vec![Some(10.0), Some(10.0), Some(30.0), Some(30.0)]
        );

        // One bucket over both windows averages them.
        let samples = vec![(1000u64, 1000u32, 10.0), (2000, 1000, 30.0)];
        assert_eq!(
            bucketize(samples.into_iter(), 2000, 2000, 1),
            vec![Some(20.0)]
        );

        // A bucket split unevenly between windows is weighted by overlap.
        let samples = vec![(1000u64, 1000u32, 0.0), (4000, 3000, 100.0)];
        let buckets = bucketize(samples.into_iter(), 4000, 4000, 2);
        assert_eq!(buckets, vec![Some(50.0), Some(100.0)]);
    }

    #[test]
    fn uncovered_buckets_are_none() {
        let samples = vec![(4000u64, 1000u32, 5.0)];
        let buckets = bucketize(samples.into_iter(), 4000, 4000, 4);
        assert_eq!(buckets, vec![None, None, None, Some(5.0)]);
        assert!(
            bucketize(std::iter::empty(), 4000, 0, 4)
                .iter()
                .all(Option::is_none)
        );
    }

    fn report(at_ms: u64, active: Option<f64>) -> Report {
        Report {
            timestamp_ms: at_ms,
            window_seconds: 1.0,
            device: DeviceView {
                active_pct: active,
                ..DeviceView::default()
            },
            totals: Totals {
                attributed_ms_per_s: 100.0,
                ..Totals::default()
            },
            ..Report::default()
        }
    }

    fn row(key: RowKey, busy: f64, memory: Option<u64>) -> Row {
        Row {
            key,
            pid: None,
            name: "proc".into(),
            detail: None,
            gpu_ns: (busy * 1e6) as u64,
            busy_ms_per_s: busy,
            share_pct: 0.0,
            device_pct: None,
            power_w: None,
            power_estimated: true,
            memory_bytes: memory,
            lifetime_gpu_ns: 0,
            activity: crate::model::Activity::Idle,
            last_submit_age_ms: None,
            children: 0,
            unattributed_ms_per_s: None,
            uid: None,
        }
    }

    #[test]
    fn rows_are_sparse_and_idle_windows_count_as_zero() {
        let key = RowKey::Process {
            pid: 1,
            start_us: 1,
        };
        let mut history = History::default();
        history.record(&report(1000, Some(10.0)), &[row(key, 200.0, Some(64))]);
        history.record(&report(2000, Some(10.0)), &[row(key, 0.0, Some(64))]);
        history.record(&report(3000, None), &[row(key, 400.0, Some(128))]);

        let series = &history.rows[&key];
        assert_eq!(series.points.len(), 2, "idle window stores no point");
        assert_eq!(series.memory.len(), 2, "unchanged memory stores no point");

        let coverage = history.coverage(3000, 3000, 3);
        assert_eq!(coverage, vec![1000.0, 1000.0, 1000.0]);
        let busy = history.row_buckets(&key, RowMetric::Busy, 3000, 3000, &coverage);
        assert_eq!(busy, vec![Some(200.0), Some(0.0), Some(400.0)]);
        let averaged = history.row_buckets(&key, RowMetric::Busy, 3000, 3000, &[3000.0]);
        assert_eq!(averaged, vec![Some(200.0)]);

        let memory = series.memory_buckets(3000, 3000, 3);
        assert_eq!(memory, vec![Some(64.0), Some(64.0), Some(128.0)]);

        let active: Vec<f64> = history
            .device_samples(|point| point.active_pct, 0)
            .map(|(_, _, value)| value)
            .collect();
        assert_eq!(active, vec![10.0, 10.0]);
        let recent: Vec<f64> = history
            .device_samples(|point| point.active_pct, 1000)
            .map(|(_, _, value)| value)
            .collect();
        assert_eq!(recent, vec![10.0], "only windows ending after 1000 ms");
    }

    #[test]
    fn stale_rows_are_dropped_and_events_kept() {
        let key = RowKey::App { coalition_id: 9 };
        let mut history = History::default();
        history.record(&report(1000, None), &[row(key, 1.0, None)]);
        let mut later = report(1000 + ROW_RETENTION_MS + 1, None);
        later.events.push(Event {
            timestamp_ms: later.timestamp_ms,
            kind: EventKind::GpuRecovery,
            message: "GPU recovered".into(),
        });
        history.record(&later, &[]);
        assert!(history.rows.is_empty());
        assert_eq!(history.events.len(), 1);
        assert_eq!(history.latest_ms(), Some(later.timestamp_ms));
    }
}
