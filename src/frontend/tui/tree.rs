//! Tree view rows: apps over their processes (or parent processes without
//! coalition data), and processes over their clients and queues.

use super::app::{DisplayRow, SortKey};
use crate::model::{Activity, Report};
use crate::view::{self, Filter, Group, Row, RowKey};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub(super) fn sort_rows(rows: &mut [Row], sort: SortKey, reverse: bool) {
    rows.sort_by(|a, b| {
        // The unattributed row always stays last.
        match (a.key == RowKey::Unattributed, b.key == RowKey::Unattributed) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => sort.compare(a, b, reverse),
        }
    });
}

/// Builds the tree: apps over their processes (process and app levels; parent
/// processes when there is no coalition data), processes over clients (client
/// level), or processes over clients over queues (queue level).
pub(super) fn tree_rows(
    report: &Report,
    group: Group,
    filter: &Filter,
    sort: SortKey,
    reverse: bool,
) -> Vec<DisplayRow> {
    let all = Filter::default();
    let mut processes = view::rows(report, Group::Process, &all);
    let unattributed = processes
        .iter()
        .position(|row| row.key == RowKey::Unattributed)
        .map(|index| processes.remove(index));
    sort_rows(&mut processes, sort, reverse);

    let mut nodes: Vec<Node> = match group {
        Group::Process | Group::App => {
            if report.apps.is_empty()
                && report
                    .processes
                    .iter()
                    .all(|process| process.coalition_id.is_none())
            {
                parent_tree(report, processes)
            } else {
                app_tree(report, processes, sort, reverse)
            }
        }
        Group::Client | Group::Queue => {
            let mut clients = view::rows(report, Group::Client, &all);
            sort_rows(&mut clients, sort, reverse);
            let mut queues = view::rows(report, Group::Queue, &all);
            sort_rows(&mut queues, sort, reverse);
            processes
                .into_iter()
                .map(|process| {
                    let pid = process.pid;
                    let children = clients
                        .iter()
                        .filter(|client| client.pid == pid)
                        .map(|client| {
                            let RowKey::Client { registry_id } = client.key else {
                                unreachable!("client rows have client keys")
                            };
                            let children = if group == Group::Queue {
                                queues
                                    .iter()
                                    .filter(|queue| {
                                        matches!(queue.key, RowKey::Queue { registry_id: id, .. } if id == registry_id)
                                    })
                                    .map(|queue| Node::leaf(queue.clone()))
                                    .collect()
                            } else {
                                Vec::new()
                            };
                            Node {
                                row: client.clone(),
                                synthesized: false,
                                children,
                            }
                        })
                        .collect();
                    Node {
                        row: process,
                        synthesized: false,
                        children,
                    }
                })
                .collect()
        }
    };
    nodes.retain_mut(|node| node.filter(filter));
    if let Some(row) = unattributed
        && filter.is_empty()
    {
        nodes.push(Node::leaf(row));
    }

    let mut out = Vec::new();
    for node in &nodes {
        node.flatten(0, "", true, &mut out);
    }
    out
}

struct Node {
    row: Row,
    synthesized: bool,
    children: Vec<Node>,
}

impl Node {
    fn leaf(row: Row) -> Self {
        Self {
            row,
            synthesized: false,
            children: Vec::new(),
        }
    }

    /// Keeps a node if it matches, with all its children, or if any child does,
    /// with only the matching children.
    fn filter(&mut self, filter: &Filter) -> bool {
        if filter.is_empty() || filter.matches(&self.row) {
            return true;
        }
        self.children.retain_mut(|child| child.filter(filter));
        !self.children.is_empty()
    }

    fn flatten(&self, depth: u8, indent: &str, last: bool, out: &mut Vec<DisplayRow>) {
        let prefix = if depth == 0 {
            String::new()
        } else {
            format!("{indent}{}", if last { "└─" } else { "├─" })
        };
        out.push(DisplayRow {
            row: self.row.clone(),
            depth,
            prefix,
            synthesized: self.synthesized,
        });
        let child_indent = if depth == 0 {
            String::new()
        } else {
            format!("{indent}{}", if last { "  " } else { "│ " })
        };
        for (index, child) in self.children.iter().enumerate() {
            child.flatten(
                depth.saturating_add(1),
                &child_indent,
                index + 1 == self.children.len(),
                out,
            );
        }
    }
}

fn app_tree(report: &Report, processes: Vec<Row>, sort: SortKey, reverse: bool) -> Vec<Node> {
    let coalition_of: HashMap<(i32, u64), Option<u64>> = report
        .processes
        .iter()
        .map(|process| ((process.pid, process.start_us), process.coalition_id))
        .collect();
    let mut members: HashMap<u64, Vec<Row>> = HashMap::new();
    let mut loose = Vec::new();
    for row in processes {
        let coalition = match row.key {
            RowKey::Process { pid, start_us } => {
                coalition_of.get(&(pid, start_us)).copied().flatten()
            }
            _ => None,
        };
        match coalition {
            Some(coalition_id) => members.entry(coalition_id).or_default().push(row),
            None => loose.push(row),
        }
    }

    let mut headers = view::rows(report, Group::App, &Filter::default());
    let known: HashSet<u64> = report.apps.iter().map(|app| app.coalition_id).collect();
    for (coalition_id, rows) in &members {
        if !known.contains(coalition_id) {
            headers.push(synthesized_app(*coalition_id, rows));
        }
    }
    sort_rows(&mut headers, sort, reverse);

    let mut nodes: Vec<Node> = headers
        .into_iter()
        .map(|header| {
            let RowKey::App { coalition_id } = header.key else {
                unreachable!("app rows have app keys")
            };
            Node {
                synthesized: !known.contains(&coalition_id),
                children: members
                    .remove(&coalition_id)
                    .unwrap_or_default()
                    .into_iter()
                    .map(Node::leaf)
                    .collect(),
                row: header,
            }
        })
        .collect();
    nodes.extend(loose.into_iter().map(Node::leaf));
    nodes
}

/// An app header for a coalition the report has no app row for.
fn synthesized_app(coalition_id: u64, members: &[Row]) -> Row {
    let first = &members[0];
    let bundle = first
        .detail
        .as_deref()
        .and_then(crate::sys::process::bundle_path);
    let name = bundle
        .map(|path| crate::sys::process::bundle_name(path).to_string())
        .unwrap_or_else(|| first.name.clone());
    let sum_opt = |value: fn(&Row) -> Option<f64>| {
        members
            .iter()
            .filter_map(value)
            .fold(None, |total: Option<f64>, value| {
                Some(total.unwrap_or(0.0) + value)
            })
    };
    Row {
        key: RowKey::App { coalition_id },
        pid: None,
        name,
        detail: bundle.map(str::to_string),
        gpu_ns: members.iter().map(|row| row.gpu_ns).sum(),
        busy_ms_per_s: members.iter().map(|row| row.busy_ms_per_s).sum(),
        share_pct: members.iter().map(|row| row.share_pct).sum(),
        device_pct: sum_opt(|row| row.device_pct),
        power_w: sum_opt(|row| row.power_w),
        power_estimated: true,
        memory_bytes: members
            .iter()
            .filter_map(|row| row.memory_bytes)
            .fold(None, |total: Option<u64>, value| {
                Some(total.unwrap_or(0) + value)
            }),
        lifetime_gpu_ns: members.iter().map(|row| row.lifetime_gpu_ns).sum(),
        activity: members
            .iter()
            .map(|row| row.activity)
            .min_by_key(|activity| match activity {
                Activity::Active => 0,
                Activity::Recent => 1,
                Activity::Idle => 2,
            })
            .unwrap_or_default(),
        last_submit_age_ms: members
            .iter()
            .filter_map(|row| row.last_submit_age_ms)
            .min(),
        children: members.len(),
        unattributed_ms_per_s: None,
        uid: None,
    }
}

/// Processes under their parent process, when there is no coalition data.
fn parent_tree(report: &Report, processes: Vec<Row>) -> Vec<Node> {
    let ppid_of: HashMap<i32, i32> = report
        .processes
        .iter()
        .filter_map(|process| Some((process.pid, process.ppid?)))
        .collect();
    let pids: HashSet<i32> = processes.iter().filter_map(|row| row.pid).collect();
    let parent = |row: &Row| {
        row.pid
            .and_then(|pid| ppid_of.get(&pid).copied())
            .filter(|ppid| pids.contains(ppid) && Some(*ppid) != row.pid)
    };
    let mut children: HashMap<i32, Vec<Row>> = HashMap::new();
    let mut roots = Vec::new();
    for row in processes {
        match parent(&row) {
            Some(ppid) => children.entry(ppid).or_default().push(row),
            None => roots.push(row),
        }
    }
    fn build(row: Row, children: &mut HashMap<i32, Vec<Row>>, depth: usize) -> Node {
        let kids = match row.pid {
            Some(pid) if depth < 16 => children.remove(&pid).unwrap_or_default(),
            _ => Vec::new(),
        };
        Node {
            row,
            synthesized: false,
            children: kids
                .into_iter()
                .map(|child| build(child, children, depth + 1))
                .collect(),
        }
    }
    let mut nodes: Vec<Node> = roots
        .into_iter()
        .map(|row| build(row, &mut children, 0))
        .collect();
    // Cycles leave rows unreachable from any root; list them flat.
    let mut rest: Vec<Row> = children.into_values().flatten().collect();
    rest.sort_by_key(|row| row.key);
    nodes.extend(rest.into_iter().map(Node::leaf));
    nodes
}
