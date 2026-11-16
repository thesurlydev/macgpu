//! Per-queue, per-client, and per-process GPU time from `AppUsage` counters.
//!
//! Accuracy rules:
//! - Clients are diffed by IORegistry entry ID, never by PID, so PID reuse and
//!   multiple clients per process are handled.
//! - Each `AppUsage` entry is one command queue with its own monotonic counter.
//!   Queues are aligned between samples so that a queue disappearing (its
//!   counter leaving the client's sum) never produces a negative or
//!   lifetime-sized delta.
//! - A queue that did not exist in the previous sample was created inside the
//!   window. It is credited with its full counter, capped at the window length
//!   because one queue cannot have run longer than the window it was created in.
//! - Raw values are never clamped or rescaled across processes; normalized
//!   values are separate fields.

use super::{NS_PER_MS, RECENT_SECONDS, ms_per_s, pct, saturating_sum};
use crate::snapshot::Snapshot;
use crate::sys::iokit::{GpuClient, QueueUsage};
use macgpu_model::{Activity, ClientView, ProcessView, QueueView};
use std::collections::HashMap;

pub(super) struct Built {
    pub processes: Vec<ProcessView>,
    pub clients: usize,
}

pub(super) fn build(previous: &Snapshot, current: &Snapshot, window_ns: u64) -> Built {
    let Ok(current_registry) = &current.registry else {
        return Built {
            processes: Vec::new(),
            clients: 0,
        };
    };
    // Without a previous registry there is no baseline: report lifetimes only.
    let previous_clients: Option<HashMap<u64, &GpuClient>> =
        previous.registry.as_ref().ok().map(|registry| {
            registry
                .clients()
                .map(|client| (client.registry_id, client))
                .collect()
        });

    let mut by_pid: HashMap<i32, ProcessView> = HashMap::new();
    let mut client_count = 0;

    for client in current_registry.clients() {
        client_count += 1;
        let previous_client = previous_clients
            .as_ref()
            .and_then(|clients| clients.get(&client.registry_id).copied());
        let view = client_view(
            client,
            previous_client,
            previous_clients.is_some(),
            current.mach_now_ns,
            window_ns,
        );

        let process = by_pid
            .entry(client.pid)
            .or_insert_with(|| new_process(client, current));
        process.gpu_ns = process.gpu_ns.saturating_add(view.gpu_ns);
        process.lifetime_gpu_ns = process.lifetime_gpu_ns.saturating_add(view.lifetime_gpu_ns);
        process.queues += view.queues.len();
        process.last_submit_age_ms =
            min_option(process.last_submit_age_ms, view.last_submit_age_ms);
        process.clients.push(view);
    }

    let total_ns = saturating_sum(by_pid.values().map(|process| process.gpu_ns));
    let mut processes: Vec<ProcessView> = by_pid.into_values().collect();
    for process in &mut processes {
        process.busy_ms_per_s = ms_per_s(process.gpu_ns, window_ns);
        process.share_pct = pct(process.gpu_ns, total_ns);
        process.activity = activity(process.gpu_ns, process.last_submit_age_ms);
        process.clients.sort_by(|a, b| {
            b.gpu_ns
                .cmp(&a.gpu_ns)
                .then(a.registry_id.cmp(&b.registry_id))
        });
    }
    processes.sort_by(|a, b| {
        b.gpu_ns
            .cmp(&a.gpu_ns)
            .then_with(|| b.lifetime_gpu_ns.cmp(&a.lifetime_gpu_ns))
            .then_with(|| a.pid.cmp(&b.pid))
    });

    Built {
        processes,
        clients: client_count,
    }
}

/// Sets each process's slice of device active time.
pub(super) fn apply_device_share(processes: &mut [ProcessView], device_active_pct: Option<f64>) {
    for process in processes {
        process.device_pct = device_active_pct.map(|active| process.share_pct * active / 100.0);
    }
}

fn new_process(client: &GpuClient, current: &Snapshot) -> ProcessView {
    let info = current.processes.get(&client.pid);
    ProcessView {
        pid: client.pid,
        start_us: info.map(|info| info.start_us).unwrap_or(0),
        name: info
            .map(|info| info.name.clone())
            .unwrap_or_else(|| client.creator_name.clone()),
        path: info.and_then(|info| info.path.clone()),
        uid: info.map(|info| info.uid),
        ppid: info.map(|info| info.ppid),
        ..ProcessView::default()
    }
}

fn client_view(
    client: &GpuClient,
    previous: Option<&GpuClient>,
    have_baseline: bool,
    mach_now_ns: u64,
    window_ns: u64,
) -> ClientView {
    let pairing: Vec<Option<usize>> = match previous {
        Some(previous) => align_queues(&previous.queues, &client.queues),
        None => vec![None; client.queues.len()],
    };

    let mut queues = Vec::with_capacity(client.queues.len());
    for (index, (queue, matched)) in client.queues.iter().zip(pairing).enumerate() {
        let gpu_ns = if !have_baseline {
            0
        } else {
            match matched.and_then(|matched| previous.map(|previous| &previous.queues[matched])) {
                Some(old) => queue.accumulated_ns - old.accumulated_ns,
                None => queue.accumulated_ns.min(window_ns),
            }
        };
        let last_submit_age_ms = submit_age_ms(queue.last_submitted_ns, mach_now_ns);
        queues.push(QueueView {
            index,
            api: queue.api.clone(),
            gpu_ns,
            busy_ms_per_s: ms_per_s(gpu_ns, window_ns),
            lifetime_gpu_ns: queue.accumulated_ns,
            last_submit_age_ms,
            activity: activity(gpu_ns, last_submit_age_ms),
        });
    }

    let gpu_ns = saturating_sum(queues.iter().map(|queue| queue.gpu_ns));
    ClientView {
        registry_id: client.registry_id,
        gpu_ns,
        busy_ms_per_s: ms_per_s(gpu_ns, window_ns),
        lifetime_gpu_ns: client.accumulated_ns(),
        command_queue_count: client.command_queue_count,
        command_queue_count_max: client.command_queue_count_max,
        last_submit_age_ms: queues
            .iter()
            .filter_map(|queue| queue.last_submit_age_ms)
            .min(),
        queues,
    }
}

/// Whether `current` can be the same queue as `previous` one sample later.
fn continues(previous: &QueueUsage, current: &QueueUsage) -> bool {
    current.api == previous.api
        && current.accumulated_ns >= previous.accumulated_ns
        && current.last_submitted_ns >= previous.last_submitted_ns
}

/// Pairs each current queue with the index of the previous queue it continues.
///
/// `AppUsage` entries carry no identifier. Entries keep their relative order,
/// so queues that persist form an order-preserving subsequence of both arrays,
/// and each persisting queue's counters never go down. This finds the longest
/// such pairing (an LCS under the [`continues`] relation). Unpaired current
/// entries are new queues; unpaired previous entries were destroyed.
pub(super) fn align_queues(previous: &[QueueUsage], current: &[QueueUsage]) -> Vec<Option<usize>> {
    if previous.len() == current.len()
        && previous
            .iter()
            .zip(current)
            .all(|(old, new)| continues(old, new))
    {
        return (0..current.len()).map(Some).collect();
    }

    let rows = previous.len();
    let columns = current.len();
    // lengths[i][j]: best pairing of previous[i..] with current[j..].
    let mut lengths = vec![vec![0u32; columns + 1]; rows + 1];
    for i in (0..rows).rev() {
        for j in (0..columns).rev() {
            let skip = lengths[i + 1][j].max(lengths[i][j + 1]);
            lengths[i][j] = if continues(&previous[i], &current[j]) {
                skip.max(lengths[i + 1][j + 1] + 1)
            } else {
                skip
            };
        }
    }

    let mut pairing = vec![None; columns];
    let (mut i, mut j) = (0, 0);
    while i < rows && j < columns {
        if continues(&previous[i], &current[j]) && lengths[i][j] == lengths[i + 1][j + 1] + 1 {
            pairing[j] = Some(i);
            i += 1;
            j += 1;
        } else if lengths[i + 1][j] >= lengths[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    pairing
}

fn submit_age_ms(last_submitted_ns: u64, mach_now_ns: u64) -> Option<u64> {
    (last_submitted_ns > 0)
        .then(|| mach_now_ns.saturating_sub(last_submitted_ns) / NS_PER_MS as u64)
}

fn activity(gpu_ns: u64, last_submit_age_ms: Option<u64>) -> Activity {
    if gpu_ns > 0 {
        Activity::Active
    } else if last_submit_age_ms.is_some_and(|age| age < RECENT_SECONDS * 1000) {
        Activity::Recent
    } else {
        Activity::Idle
    }
}

fn min_option(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use std::time::{Duration, Instant};

    const SECOND: u64 = 1_000_000_000;

    #[test]
    fn aligns_unchanged_queues_positionally() {
        let previous = [queue(10, 1), queue(20, 2)];
        let current = [queue(15, 3), queue(20, 2)];
        assert_eq!(align_queues(&previous, &current), vec![Some(0), Some(1)]);
    }

    #[test]
    fn aligns_when_a_queue_is_removed() {
        // The middle queue (300 s lifetime) was destroyed; the others advanced.
        let previous = [
            queue(100 * SECOND, 5),
            queue(300 * SECOND, 6),
            queue(7 * SECOND, 7),
        ];
        let current = [queue(100 * SECOND + 10, 8), queue(7 * SECOND + 20, 9)];
        assert_eq!(align_queues(&previous, &current), vec![Some(0), Some(2)]);
    }

    #[test]
    fn aligns_when_a_queue_is_added() {
        let previous = [queue(50, 1)];
        let current = [queue(60, 2), queue(5, 3)];
        assert_eq!(align_queues(&previous, &current), vec![Some(0), None]);
    }

    #[test]
    fn aligns_when_one_queue_is_replaced() {
        let previous = [queue(1_000, 10), queue(2_000, 20)];
        let current = [queue(1_500, 30), queue(40, 31)];
        assert_eq!(align_queues(&previous, &current), vec![Some(0), None]);
    }

    /// A destroyed queue contributes no delta: only the surviving queue's own
    /// counter increase is credited to the window.
    #[test]
    fn destroyed_queue_does_not_inflate_window() {
        let base = Instant::now();
        let previous = snapshot(
            base,
            Duration::ZERO,
            10 * SECOND,
            vec![client(
                7,
                42,
                vec![
                    queue(3_600 * SECOND, 9 * SECOND),
                    queue(900 * SECOND, 9 * SECOND),
                ],
            )],
        );
        let current = snapshot(
            base,
            Duration::from_secs(2),
            12 * SECOND,
            vec![client(
                7,
                42,
                vec![queue(3_600 * SECOND + 250_000_000, 11 * SECOND)],
            )],
        );
        let built = build(&previous, &current, 2 * SECOND);
        assert_eq!(built.processes.len(), 1);
        assert_eq!(built.processes[0].gpu_ns, 250_000_000);
        assert_eq!(
            built.processes[0].lifetime_gpu_ns,
            3_600 * SECOND + 250_000_000
        );
    }

    #[test]
    fn new_client_is_credited_capped_at_window() {
        let base = Instant::now();
        let previous = snapshot(base, Duration::ZERO, 0, vec![]);
        let current = snapshot(
            base,
            Duration::from_secs(1),
            SECOND,
            vec![client(
                9,
                77,
                vec![queue(400_000_000, SECOND), queue(50 * SECOND, SECOND)],
            )],
        );
        let built = build(&previous, &current, SECOND);
        assert_eq!(built.processes[0].gpu_ns, 400_000_000 + SECOND);
    }

    #[test]
    fn no_baseline_means_no_deltas() {
        let base = Instant::now();
        let mut previous = snapshot(base, Duration::ZERO, 0, vec![]);
        previous.registry = Err("failed".into());
        let current = snapshot(
            base,
            Duration::from_secs(1),
            SECOND,
            vec![client(9, 77, vec![queue(5 * SECOND, 1)])],
        );
        let built = build(&previous, &current, SECOND);
        assert_eq!(built.processes[0].gpu_ns, 0);
        assert_eq!(built.processes[0].lifetime_gpu_ns, 5 * SECOND);
    }

    #[test]
    fn aggregates_clients_by_process_and_sorts() {
        let base = Instant::now();
        let previous = snapshot(
            base,
            Duration::ZERO,
            0,
            vec![
                client(1, 431, vec![queue(1_000, 1)]),
                client(2, 431, vec![queue(2_000, 1)]),
                client(3, 3159, vec![queue(47 * SECOND, 1)]),
            ],
        );
        let current = snapshot(
            base,
            Duration::from_secs(1),
            100 * SECOND,
            vec![
                client(1, 431, vec![queue(1_000 + 300_000_000, 99 * SECOND)]),
                client(2, 431, vec![queue(2_000 + 100_000_000, 99 * SECOND)]),
                client(3, 3159, vec![queue(47 * SECOND, 1)]),
                client(4, 1234, vec![queue(500_000_000, 99 * SECOND)]),
            ],
        );
        let built = build(&previous, &current, SECOND);
        assert_eq!(built.clients, 4);
        let pids: Vec<i32> = built.processes.iter().map(|process| process.pid).collect();
        assert_eq!(pids, vec![1234, 431, 3159]);

        let window_server = &built.processes[1];
        assert_eq!(window_server.gpu_ns, 400_000_000);
        assert_eq!(window_server.clients.len(), 2);
        assert!((window_server.share_pct - 400.0 / 900.0 * 100.0).abs() < 1e-9);
        assert_eq!(window_server.activity, Activity::Active);
        assert_eq!(window_server.start_us, 1_000_431);

        let idle = &built.processes[2];
        assert_eq!(idle.gpu_ns, 0);
        assert_eq!(idle.activity, Activity::Idle);
        assert_eq!(idle.last_submit_age_ms, Some(99_999));
    }

    #[test]
    fn recent_activity_from_last_submission() {
        let base = Instant::now();
        let clients = vec![client(1, 5, vec![queue(1_000, 95 * SECOND)])];
        let previous = snapshot(base, Duration::ZERO, 99 * SECOND, clients.clone());
        let current = snapshot(base, Duration::from_secs(1), 100 * SECOND, clients);
        let built = build(&previous, &current, SECOND);
        assert_eq!(built.processes[0].activity, Activity::Recent);
        assert_eq!(built.processes[0].last_submit_age_ms, Some(5_000));
    }

    #[test]
    fn device_share_scales_by_active_residency() {
        let mut processes = vec![
            ProcessView {
                share_pct: 75.0,
                ..ProcessView::default()
            },
            ProcessView {
                share_pct: 25.0,
                ..ProcessView::default()
            },
        ];
        apply_device_share(&mut processes, Some(40.0));
        assert_eq!(processes[0].device_pct, Some(30.0));
        assert_eq!(processes[1].device_pct, Some(10.0));
        apply_device_share(&mut processes, None);
        assert_eq!(processes[0].device_pct, None);
    }
}
