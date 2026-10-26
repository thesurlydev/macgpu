//! Keeps rows that just went idle on screen for a few seconds.

use crate::analyze::RECENT_SECONDS;
use crate::model::{Activity, AppView, ProcessView, Report};
use std::collections::{HashMap, HashSet};

/// Whether the process with this PID and start time still exists.
pub type ProcessAlive = fn(i32, u64) -> bool;

/// Rows that just went idle stay listed this long, so they do not flicker.
const LINGER_MS: u64 = RECENT_SECONDS * 1000;

/// Re-adds processes and apps that used the GPU within the last few seconds
/// but are missing from a report built without idle rows, with zero usage, so
/// rows do not blink in and out between bursts.
#[derive(Default)]
pub(super) struct Linger {
    processes: HashMap<(i32, u64), (ProcessView, u64)>,
    apps: HashMap<u64, (AppView, u64)>,
}

impl Linger {
    pub(super) fn merge(&mut self, report: &mut Report, alive: Option<ProcessAlive>) {
        let now = report.timestamp_ms;
        for process in &report.processes {
            if process.gpu_ns > 0 {
                self.processes
                    .insert((process.pid, process.start_us), (process.clone(), now));
            }
        }
        for app in &report.apps {
            if app.gpu_ns > 0 {
                self.apps.insert(app.coalition_id, (app.clone(), now));
            }
        }
        let fresh = |seen: u64| now >= seen && now - seen <= LINGER_MS;
        self.processes.retain(|_, (_, seen)| fresh(*seen));
        self.apps.retain(|_, (_, seen)| fresh(*seen));

        let present: HashSet<(i32, u64)> = report
            .processes
            .iter()
            .map(|process| (process.pid, process.start_us))
            .collect();
        let mut added = false;
        let mut exited = Vec::new();
        for (key, (process, seen)) in &self.processes {
            if present.contains(key) {
                continue;
            }
            if alive.is_some_and(|alive| !alive(key.0, key.1)) {
                exited.push(*key);
                continue;
            }
            report.processes.push(idle_process(process, now - seen));
            added = true;
        }
        for key in exited {
            self.processes.remove(&key);
        }
        let present: HashSet<u64> = report.apps.iter().map(|app| app.coalition_id).collect();
        for (coalition_id, (app, _)) in &self.apps {
            if !present.contains(coalition_id) {
                report.apps.push(idle_app(app));
                added = true;
            }
        }
        if added {
            report.processes.sort_by(|a, b| {
                b.gpu_ns
                    .cmp(&a.gpu_ns)
                    .then_with(|| (a.pid, a.start_us).cmp(&(b.pid, b.start_us)))
            });
            report.apps.sort_by(|a, b| {
                b.gpu_ns
                    .cmp(&a.gpu_ns)
                    .then(a.coalition_id.cmp(&b.coalition_id))
            });
        }
    }
}

fn idle_process(process: &ProcessView, elapsed_ms: u64) -> ProcessView {
    let mut idle = process.clone();
    idle.gpu_ns = 0;
    idle.busy_ms_per_s = 0.0;
    idle.share_pct = 0.0;
    idle.device_pct = idle.device_pct.map(|_| 0.0);
    idle.est_power_w = idle.est_power_w.map(|_| 0.0);
    idle.activity = Activity::Recent;
    idle.last_submit_age_ms = idle.last_submit_age_ms.map(|age| age + elapsed_ms);
    for client in &mut idle.clients {
        client.gpu_ns = 0;
        client.busy_ms_per_s = 0.0;
        client.last_submit_age_ms = client.last_submit_age_ms.map(|age| age + elapsed_ms);
        for queue in &mut client.queues {
            queue.gpu_ns = 0;
            queue.busy_ms_per_s = 0.0;
            queue.last_submit_age_ms = queue.last_submit_age_ms.map(|age| age + elapsed_ms);
            if queue.activity == Activity::Active {
                queue.activity = Activity::Recent;
            }
        }
    }
    idle
}

fn idle_app(app: &AppView) -> AppView {
    let mut idle = app.clone();
    idle.gpu_ns = 0;
    idle.busy_ms_per_s = 0.0;
    idle.share_pct = 0.0;
    idle.device_pct = idle.device_pct.map(|_| 0.0);
    idle.attributed_gpu_ns = 0;
    idle.in_flight_gpu_ns = idle.in_flight_gpu_ns.map(|_| 0);
    idle.in_flight_ms_per_s = idle.in_flight_ms_per_s.map(|_| 0.0);
    idle.unattributed_gpu_ns = Some(0);
    idle.energy_j = idle.energy_j.map(|_| 0.0);
    idle.power_w = idle.power_w.map(|_| 0.0);
    idle.billed_to_me_w = None;
    idle.billed_to_others_w = None;
    idle.ane_ms_per_s = idle.ane_ms_per_s.map(|_| 0.0);
    idle.ane_power_w = idle.ane_power_w.map(|_| 0.0);
    idle
}
