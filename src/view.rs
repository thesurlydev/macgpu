//! Flattens a [`Report`] into display rows at a chosen grouping level, and
//! filters them. Shared by the table, TUI, and log front ends.

use crate::model::{Activity, Report};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// The level at which GPU usage is aggregated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    /// One row per Metal command queue (`AppUsage` entry).
    Queue,
    /// One row per GPU client connection (`AGXDeviceUserClient`).
    Client,
    /// One row per process.
    #[default]
    Process,
    /// One row per app (resource coalition: an app plus its helpers).
    App,
}

impl Group {
    /// The next level in the cycle queue, client, process, app.
    pub fn next(self) -> Self {
        match self {
            Group::Queue => Group::Client,
            Group::Client => Group::Process,
            Group::Process => Group::App,
            Group::App => Group::Queue,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Group::Queue => "queue",
            Group::Client => "client",
            Group::Process => "process",
            Group::App => "app",
        }
    }
}

/// Identifies a row across refreshes so selection and history follow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RowKey {
    Process { pid: i32, start_us: u64 },
    App { coalition_id: u64 },
    Client { registry_id: u64 },
    Queue { registry_id: u64, index: usize },
    Unattributed,
}

/// One display row.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub key: RowKey,
    pub pid: Option<i32>,
    pub name: String,
    /// Executable path, bundle path, or API, depending on level.
    pub detail: Option<String>,
    pub gpu_ns: u64,
    pub busy_ms_per_s: f64,
    pub share_pct: f64,
    pub device_pct: Option<f64>,
    pub power_w: Option<f64>,
    /// True when `power_w` is apportioned rather than measured.
    pub power_estimated: bool,
    pub memory_bytes: Option<u64>,
    pub lifetime_gpu_ns: u64,
    pub activity: Activity,
    pub last_submit_age_ms: Option<u64>,
    /// Queues (process, client) or GPU processes (app).
    pub children: usize,
    /// App rows: GPU time from exited or unattributed clients, in ms/s.
    pub unattributed_ms_per_s: Option<f64>,
    pub uid: Option<u32>,
}

impl Row {
    /// Share of attributed GPU time. `None` for the unattributed row, which is
    /// outside the attributed total that shares are taken of.
    pub fn share(&self) -> Option<f64> {
        (self.key != RowKey::Unattributed).then_some(self.share_pct)
    }

    fn blank(key: RowKey, name: String) -> Self {
        Self {
            key,
            pid: None,
            name,
            detail: None,
            gpu_ns: 0,
            busy_ms_per_s: 0.0,
            share_pct: 0.0,
            device_pct: None,
            power_w: None,
            power_estimated: false,
            memory_bytes: None,
            lifetime_gpu_ns: 0,
            activity: Activity::Idle,
            last_submit_age_ms: None,
            children: 0,
            unattributed_ms_per_s: None,
            uid: None,
        }
    }
}

/// Builds rows for `group`, applying `filter`, sorted by GPU time.
pub fn rows(report: &Report, group: Group, filter: &Filter) -> Vec<Row> {
    let mut rows = match group {
        Group::Process => process_rows(report),
        Group::App => app_rows(report),
        Group::Client => client_rows(report),
        Group::Queue => queue_rows(report),
    };
    rows.retain(|row| filter.matches(row));
    rows.sort_by(|a, b| {
        b.gpu_ns
            .cmp(&a.gpu_ns)
            .then_with(|| b.lifetime_gpu_ns.cmp(&a.lifetime_gpu_ns))
            .then_with(|| a.key.cmp(&b.key))
    });
    if group == Group::Process
        && let Some(unattributed) = &report.unattributed
        && unattributed.gpu_ns > 0
        && filter.is_empty()
    {
        let mut row = Row::blank(RowKey::Unattributed, "(exited or unattributed)".into());
        row.gpu_ns = unattributed.gpu_ns;
        row.busy_ms_per_s = unattributed.busy_ms_per_s;
        rows.push(row);
    }
    rows
}

fn process_rows(report: &Report) -> Vec<Row> {
    report
        .processes
        .iter()
        .map(|process| Row {
            key: RowKey::Process {
                pid: process.pid,
                start_us: process.start_us,
            },
            pid: Some(process.pid),
            name: process.name.clone(),
            detail: process.path.clone(),
            gpu_ns: process.gpu_ns,
            busy_ms_per_s: process.busy_ms_per_s,
            share_pct: process.share_pct,
            device_pct: process.device_pct,
            power_w: process.est_power_w,
            power_estimated: true,
            memory_bytes: process.graphics_memory_bytes,
            lifetime_gpu_ns: process.lifetime_gpu_ns,
            activity: process.activity,
            last_submit_age_ms: process.last_submit_age_ms,
            children: process.queues,
            unattributed_ms_per_s: None,
            uid: process.uid,
        })
        .collect()
}

fn app_rows(report: &Report) -> Vec<Row> {
    report
        .apps
        .iter()
        .map(|app| {
            let window_ns = (report.window_seconds * 1e9).max(1.0);
            Row {
                key: RowKey::App {
                    coalition_id: app.coalition_id,
                },
                pid: app.responsible_pid,
                name: app.name.clone(),
                detail: app.bundle_path.clone(),
                gpu_ns: app.gpu_ns,
                busy_ms_per_s: app.busy_ms_per_s,
                share_pct: app.share_pct,
                device_pct: app.device_pct,
                power_w: app.power_w,
                power_estimated: false,
                memory_bytes: app.graphics_memory_bytes,
                lifetime_gpu_ns: app.lifetime_gpu_ns,
                activity: if app.gpu_ns > 0 {
                    Activity::Active
                } else {
                    Activity::Idle
                },
                last_submit_age_ms: None,
                children: app.gpu_pids.len(),
                unattributed_ms_per_s: app
                    .unattributed_gpu_ns
                    .map(|gpu_ns| gpu_ns as f64 / 1e6 / (window_ns / 1e9)),
                uid: None,
            }
        })
        .collect()
}

fn client_rows(report: &Report) -> Vec<Row> {
    let total: u64 = report.processes.iter().map(|process| process.gpu_ns).sum();
    report
        .processes
        .iter()
        .flat_map(|process| {
            process.clients.iter().map(move |client| Row {
                key: RowKey::Client {
                    registry_id: client.registry_id,
                },
                pid: Some(process.pid),
                name: process.name.clone(),
                detail: Some(format!("client {:#x}", client.registry_id)),
                gpu_ns: client.gpu_ns,
                busy_ms_per_s: client.busy_ms_per_s,
                share_pct: share(client.gpu_ns, total),
                device_pct: None,
                power_w: None,
                power_estimated: true,
                memory_bytes: None,
                lifetime_gpu_ns: client.lifetime_gpu_ns,
                activity: if client.gpu_ns > 0 {
                    Activity::Active
                } else {
                    process.activity.min_recent()
                },
                last_submit_age_ms: client.last_submit_age_ms,
                children: client.queues.len(),
                unattributed_ms_per_s: None,
                uid: process.uid,
            })
        })
        .collect()
}

fn queue_rows(report: &Report) -> Vec<Row> {
    let total: u64 = report.processes.iter().map(|process| process.gpu_ns).sum();
    report
        .processes
        .iter()
        .flat_map(|process| {
            process.clients.iter().flat_map(move |client| {
                client.queues.iter().map(move |queue| Row {
                    key: RowKey::Queue {
                        registry_id: client.registry_id,
                        index: queue.index,
                    },
                    pid: Some(process.pid),
                    name: process.name.clone(),
                    detail: Some(format!(
                        "{} queue {} on client {:#x}",
                        queue.api, queue.index, client.registry_id
                    )),
                    gpu_ns: queue.gpu_ns,
                    busy_ms_per_s: queue.busy_ms_per_s,
                    share_pct: share(queue.gpu_ns, total),
                    device_pct: None,
                    power_w: None,
                    power_estimated: true,
                    memory_bytes: None,
                    lifetime_gpu_ns: queue.lifetime_gpu_ns,
                    activity: queue.activity,
                    last_submit_age_ms: queue.last_submit_age_ms,
                    children: 0,
                    unattributed_ms_per_s: None,
                    uid: process.uid,
                })
            })
        })
        .collect()
}

fn share(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 / total as f64 * 100.0
    }
}

impl Activity {
    /// A client of an active process is not necessarily active itself.
    fn min_recent(self) -> Activity {
        match self {
            Activity::Active => Activity::Recent,
            other => other,
        }
    }
}

/// A row filter: comma-separated, case-insensitive substrings matched against
/// the name, detail (path), and PID. Terms prefixed with `!` exclude. A row
/// passes when it matches no exclusion and, if any inclusions exist, at least
/// one inclusion. Example: `chrome, !helper`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Filter {
    include: Vec<String>,
    exclude: Vec<String>,
    pids: Vec<i32>,
}

impl Filter {
    pub fn parse(expression: &str) -> Self {
        let mut filter = Filter::default();
        for term in expression
            .split(',')
            .map(str::trim)
            .filter(|term| !term.is_empty())
        {
            match term.strip_prefix('!') {
                Some(excluded) if !excluded.trim().is_empty() => {
                    filter.exclude.push(excluded.trim().to_lowercase())
                }
                Some(_) => {}
                None => filter.include.push(term.to_lowercase()),
            }
        }
        filter
    }

    /// Restricts rows to these PIDs (in addition to the text terms).
    pub fn with_pids(mut self, pids: &[i32]) -> Self {
        self.pids = pids.to_vec();
        self
    }

    pub fn is_empty(&self) -> bool {
        self.include.is_empty() && self.exclude.is_empty() && self.pids.is_empty()
    }

    pub fn matches(&self, row: &Row) -> bool {
        if !self.pids.is_empty() && !row.pid.is_some_and(|pid| self.pids.contains(&pid)) {
            return false;
        }
        let haystack = format!(
            "{} {} {}",
            row.name.to_lowercase(),
            row.detail.as_deref().unwrap_or_default().to_lowercase(),
            row.pid.map(|pid| pid.to_string()).unwrap_or_default()
        );
        if self
            .exclude
            .iter()
            .any(|term| haystack.contains(term.as_str()))
        {
            return false;
        }
        self.include.is_empty()
            || self
                .include
                .iter()
                .any(|term| haystack.contains(term.as_str()))
    }
}

impl std::fmt::Display for Filter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let terms: Vec<String> = self
            .include
            .iter()
            .cloned()
            .chain(self.exclude.iter().map(|term| format!("!{term}")))
            .collect();
        write!(formatter, "{}", terms.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AppView, ClientView, ProcessView, QueueView, Unattributed};

    fn report() -> Report {
        let process = |pid: i32, name: &str, gpu_ns: u64| ProcessView {
            pid,
            start_us: 1,
            name: name.into(),
            path: Some(format!("/Applications/{name}.app/Contents/MacOS/{name}")),
            gpu_ns,
            busy_ms_per_s: gpu_ns as f64 / 1e6,
            share_pct: 0.0,
            queues: 1,
            clients: vec![ClientView {
                registry_id: pid as u64 * 10,
                gpu_ns,
                queues: vec![QueueView {
                    index: 0,
                    api: "Metal".into(),
                    gpu_ns,
                    ..QueueView::default()
                }],
                ..ClientView::default()
            }],
            ..ProcessView::default()
        };
        Report {
            window_seconds: 1.0,
            processes: vec![
                process(431, "WindowServer", 300),
                process(900, "Google Chrome Helper (GPU)", 500),
                process(901, "Google Chrome Helper (Renderer)", 0),
            ],
            apps: vec![AppView {
                coalition_id: 7,
                name: "Google Chrome".into(),
                gpu_ns: 500,
                ..AppView::default()
            }],
            unattributed: Some(Unattributed {
                gpu_ns: 20,
                busy_ms_per_s: 0.00002,
                partial: false,
            }),
            ..Report::default()
        }
    }

    #[test]
    fn process_rows_sorted_with_unattributed_last() {
        let rows = rows(&report(), Group::Process, &Filter::default());
        let keys: Vec<RowKey> = rows.iter().map(|row| row.key).collect();
        assert_eq!(
            keys,
            vec![
                RowKey::Process {
                    pid: 900,
                    start_us: 1
                },
                RowKey::Process {
                    pid: 431,
                    start_us: 1
                },
                RowKey::Process {
                    pid: 901,
                    start_us: 1
                },
                RowKey::Unattributed,
            ]
        );
    }

    #[test]
    fn filter_includes_and_excludes() {
        let filter = Filter::parse("chrome, !renderer");
        let rows = rows(&report(), Group::Process, &filter);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pid, Some(900));

        let filter = Filter::parse("!helper");
        let names: Vec<String> = super::rows(&report(), Group::Process, &filter)
            .into_iter()
            .map(|row| row.name)
            .collect();
        assert_eq!(names, vec!["WindowServer".to_string()]);

        let filter = Filter::default().with_pids(&[431]);
        assert_eq!(super::rows(&report(), Group::Process, &filter).len(), 1);
        assert_eq!(filter.to_string(), "");
        assert_eq!(Filter::parse("a, !b").to_string(), "a, !b");
    }

    #[test]
    fn client_queue_and_app_levels() {
        let report = report();
        assert_eq!(rows(&report, Group::Client, &Filter::default()).len(), 3);
        let queues = rows(&report, Group::Queue, &Filter::default());
        assert_eq!(queues.len(), 3);
        assert_eq!(
            queues[0].key,
            RowKey::Queue {
                registry_id: 9000,
                index: 0
            }
        );
        let apps = rows(&report, Group::App, &Filter::default());
        assert_eq!(apps.len(), 1);
        assert_eq!(apps[0].name, "Google Chrome");
    }

    #[test]
    fn group_cycles() {
        assert_eq!(Group::Queue.next(), Group::Client);
        assert_eq!(Group::App.next(), Group::Queue);
    }
}
