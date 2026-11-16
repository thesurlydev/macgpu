//! The interactive view's state machine. It holds no terminal: keys, mouse
//! events, and reports go in; [`Effect`]s for the feed and the OS come out, so
//! everything here is unit tested without a TTY.

use super::history::History;
use super::linger::{Linger, ProcessAlive};
pub(super) use super::sort::SortKey;
use super::tree::{sort_rows, tree_rows};
use crate::model::{AppView, ProcessView, Report};
use crate::view::{self, Filter, Group, Row, RowKey};
use ratatui::layout::Rect;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Sampling intervals `+` and `-` step through, in milliseconds.
pub const INTERVAL_STEPS_MS: [u64; 12] = [
    100, 200, 250, 500, 1_000, 2_000, 3_000, 5_000, 10_000, 15_000, 30_000, 60_000,
];
/// History spans `z` and `Z` step through, in seconds.
pub const ZOOM_SPANS_S: [u64; 5] = [60, 300, 900, 1800, 3600];
const DEFAULT_ZOOM: usize = 0;
/// Reports buffered while frozen; older ones are applied anyway.
const MAX_PENDING: usize = 3600;
const STATUS_DURATION: Duration = Duration::from_secs(4);
pub(super) const DOUBLE_CLICK: Duration = Duration::from_millis(400);
/// Rows per PageUp/PageDown in the table and per PageUp/PageDown of help.
pub(super) const PAGE_ROWS: u16 = 10;
/// Rows per mouse wheel notch.
pub(super) const WHEEL_ROWS: isize = 3;
/// Depth limit for the parent chain shown in process detail.
const MAX_PARENT_DEPTH: usize = 16;

/// Signals offered by the confirmation prompt, the default first.
pub const SIGNALS: [(i32, &str); 4] = [
    (libc::SIGTERM, "SIGTERM"),
    (libc::SIGKILL, "SIGKILL"),
    (libc::SIGINT, "SIGINT"),
    (libc::SIGHUP, "SIGHUP"),
];

/// What the run loop must do on the state machine's behalf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    Quit,
    SetInterval(Duration),
    SetIncludeIdle(bool),
    /// Send `signal` to `pid`, but only if it still started at `start_us`.
    Signal {
        pid: i32,
        start_us: Option<u64>,
        signal: i32,
        name: String,
    },
}

/// A row as displayed: flat, or indented under its parent in tree view.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayRow {
    pub row: Row,
    pub depth: u8,
    /// Tree glyphs drawn before the name, empty in flat view.
    pub prefix: String,
    /// A tree header whose numbers are summed from its children because the
    /// report has no row for it (an app with no coalition data).
    pub synthesized: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Screen {
    Main,
    Detail(Box<Detail>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detail {
    /// A process, app, or the unattributed row.
    pub key: RowKey,
    /// The client or queue the user opened the detail from, highlighted.
    pub focus: Option<RowKey>,
    pub scroll: u16,
    /// Latest view of the subject, kept after it exits.
    pub process: Option<ProcessView>,
    pub app: Option<AppView>,
    pub exited: bool,
    /// Parent chain `(pid, name)` from the parent up, live feeds only.
    pub parents: Vec<(i32, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Overlay {
    None,
    Help { scroll: u16 },
    Filter { input: String, previous: String },
    Confirm(Confirm),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Confirm {
    pub pid: i32,
    pub start_us: Option<u64>,
    pub name: String,
    /// Index into [`SIGNALS`].
    pub signal: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusLevel {
    Info,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    pub text: String,
    pub level: StatusLevel,
    pub until: Instant,
}

/// Table columns, used for header clicks and adaptive layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Column {
    Pid,
    Name,
    Gpu,
    Share,
    DevicePct,
    Power,
    Memory,
    Exited,
    Children,
    Last,
    Lifetime,
    History,
}

impl Column {
    pub fn sort_key(self) -> Option<SortKey> {
        match self {
            Column::Pid => Some(SortKey::Pid),
            Column::Name => Some(SortKey::Name),
            Column::Gpu | Column::Share | Column::DevicePct | Column::History => Some(SortKey::Gpu),
            Column::Power => Some(SortKey::Power),
            Column::Memory => Some(SortKey::Memory),
            Column::Children => Some(SortKey::Children),
            Column::Last => Some(SortKey::Last),
            Column::Lifetime => Some(SortKey::Lifetime),
            Column::Exited => None,
        }
    }
}

/// Screen positions from the last draw, for mouse handling.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Hitboxes {
    /// Table body (rows only).
    pub table: Rect,
    /// Header cells: `(x, width, column)` on row `header_y`.
    pub header_y: u16,
    pub header: Vec<(u16, u16, Column)>,
}

/// Looks up a live process's parent PID and name.
pub type ProcessLookup = fn(i32) -> Option<(i32, String)>;

pub struct App {
    pub live: bool,
    pub readonly: bool,
    pub interval: Duration,
    pub include_idle: bool,
    pub feed_ended: bool,
    pub alert_firing: bool,

    /// The report on screen (with recently idle rows merged back in).
    pub report: Option<Report>,
    pub history: History,
    pub(super) pending: VecDeque<(Report, bool)>,
    pub(super) linger: Linger,
    #[cfg(test)]
    pub reports_seen: u64,

    pub group: Group,
    pub sort: SortKey,
    pub reverse: bool,
    pub filter: Filter,
    pub filter_text: String,
    pub tree: bool,
    pub frozen: bool,
    pub zoom: usize,

    pub rows: Vec<DisplayRow>,
    pub selected: Option<RowKey>,
    pub selected_index: usize,
    pub scroll: usize,

    pub screen: Screen,
    pub overlay: Overlay,
    pub status: Option<Status>,
    pub hitboxes: Hitboxes,
    pub(super) last_click: Option<(Instant, RowKey)>,
    pub(super) effects: Vec<Effect>,
    pub(super) dirty: bool,
    pub(super) lookup: Option<ProcessLookup>,
    pub(super) alive: Option<ProcessAlive>,
}

impl App {
    pub fn new(group: Group, filter: Filter, include_idle: bool, readonly: bool) -> Self {
        let filter_text = filter.to_string();
        Self {
            live: true,
            readonly,
            interval: Duration::from_secs(1),
            include_idle,
            feed_ended: false,
            alert_firing: false,
            report: None,
            history: History::default(),
            pending: VecDeque::new(),
            linger: Linger::default(),
            #[cfg(test)]
            reports_seen: 0,
            group,
            sort: SortKey::Gpu,
            reverse: false,
            filter,
            filter_text,
            tree: false,
            frozen: false,
            zoom: DEFAULT_ZOOM,
            rows: Vec::new(),
            selected: None,
            selected_index: 0,
            scroll: 0,
            screen: Screen::Main,
            overlay: Overlay::None,
            status: None,
            hitboxes: Hitboxes::default(),
            last_click: None,
            effects: Vec::new(),
            dirty: true,
            lookup: None,
            alive: None,
        }
    }

    /// Describes the feed: live or replay, and its interval.
    pub fn with_feed(mut self, live: bool, interval: Duration) -> Self {
        self.live = live;
        self.interval = interval;
        self
    }

    /// Enables parent-chain lookups in the detail screen (live feeds only).
    pub fn with_lookup(mut self, lookup: ProcessLookup) -> Self {
        self.lookup = Some(lookup);
        self
    }

    /// Stops recently idle processes from lingering once they have exited
    /// (live feeds only; a recording cannot be checked).
    pub fn with_liveness(mut self, alive: ProcessAlive) -> Self {
        self.alive = Some(alive);
        self
    }

    pub fn take_effects(&mut self) -> Vec<Effect> {
        std::mem::take(&mut self.effects)
    }

    /// True once since the last call if anything visible changed.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::replace(&mut self.dirty, false)
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn zoom_span_ms(&self) -> u64 {
        ZOOM_SPANS_S[self.zoom.min(ZOOM_SPANS_S.len() - 1)] * 1000
    }

    /// Expires the status message. Call periodically.
    pub fn tick(&mut self, now: Instant) {
        if self
            .status
            .as_ref()
            .is_some_and(|status| now >= status.until)
        {
            self.status = None;
            self.dirty = true;
        }
    }

    pub fn set_status(&mut self, text: impl Into<String>, level: StatusLevel) {
        self.status = Some(Status {
            text: text.into(),
            level,
            until: Instant::now() + STATUS_DURATION,
        });
        self.dirty = true;
    }

    // ---- reports ------------------------------------------------------------

    /// Takes a new report. `alert_firing` is what the alert rules said about it.
    pub fn push_report(&mut self, report: Report, alert_firing: bool) {
        self.count_report();
        if self.frozen {
            if self.pending.len() == MAX_PENDING
                && let Some((oldest, firing)) = self.pending.pop_front()
            {
                self.apply_report(oldest, firing);
            }
            self.pending.push_back((report, alert_firing));
            self.dirty = true;
            return;
        }
        self.apply_report(report, alert_firing);
    }

    #[cfg(test)]
    fn count_report(&mut self) {
        self.reports_seen += 1;
    }

    #[cfg(not(test))]
    fn count_report(&mut self) {}

    pub fn end_feed(&mut self) {
        if !self.feed_ended {
            self.feed_ended = true;
            self.dirty = true;
        }
    }

    pub fn pending_reports(&self) -> usize {
        self.pending.len()
    }

    fn apply_report(&mut self, mut report: Report, alert_firing: bool) {
        self.linger.merge(&mut report, self.alive);
        let mut all_rows = Vec::new();
        for group in [Group::Process, Group::App, Group::Client, Group::Queue] {
            all_rows.extend(view::rows(&report, group, &Filter::default()));
        }
        self.history.record(&report, &all_rows);
        self.alert_firing = alert_firing;
        if let Screen::Detail(detail) = &mut self.screen {
            detail.refresh(&report);
        }
        self.report = Some(report);
        self.rebuild_rows();
    }

    /// Recomputes the displayed rows and keeps the selection on the same row.
    pub fn rebuild_rows(&mut self) {
        self.dirty = true;
        let Some(report) = &self.report else {
            self.rows.clear();
            return;
        };
        self.rows = if self.tree {
            tree_rows(report, self.group, &self.filter, self.sort, self.reverse)
        } else {
            let mut rows = view::rows(report, self.group, &self.filter);
            sort_rows(&mut rows, self.sort, self.reverse);
            rows.into_iter()
                .map(|row| DisplayRow {
                    row,
                    depth: 0,
                    prefix: String::new(),
                    synthesized: false,
                })
                .collect()
        };
        self.reselect();
    }

    fn reselect(&mut self) {
        if self.rows.is_empty() {
            self.selected = None;
            self.selected_index = 0;
            return;
        }
        if let Some(index) = self
            .selected
            .and_then(|key| self.rows.iter().position(|row| row.row.key == key))
        {
            self.selected_index = index;
        } else {
            self.selected_index = self.selected_index.min(self.rows.len() - 1);
            self.selected = Some(self.rows[self.selected_index].row.key);
        }
    }

    pub(super) fn select_index(&mut self, index: usize) {
        if self.rows.is_empty() {
            return;
        }
        self.selected_index = index.min(self.rows.len() - 1);
        self.selected = Some(self.rows[self.selected_index].row.key);
        self.dirty = true;
    }

    pub(super) fn move_selection(&mut self, delta: isize) {
        if self.rows.is_empty() {
            return;
        }
        let index = (self.selected_index as isize + delta).clamp(0, self.rows.len() as isize - 1);
        self.select_index(index as usize);
    }

    pub fn selected_row(&self) -> Option<&DisplayRow> {
        self.rows.get(self.selected_index)
    }

    // ---- input --------------------------------------------------------------

    // ---- actions ------------------------------------------------------------

    pub(super) fn apply_filter(&mut self, text: String) {
        self.filter = Filter::parse(&text);
        self.filter_text = self.filter.to_string();
        self.rebuild_rows();
        if self.filter.is_empty() {
            self.set_status("filter cleared", StatusLevel::Info);
        }
    }

    pub(super) fn set_sort(&mut self, sort: SortKey, reset_reverse: bool) {
        self.sort = sort;
        if reset_reverse {
            self.reverse = false;
        }
        self.rebuild_rows();
        self.sort_status();
    }

    pub(super) fn sort_status(&mut self) {
        let direction = if self.reverse { ", reversed" } else { "" };
        self.set_status(
            format!("sort by {}{direction}", self.sort.label()),
            StatusLevel::Info,
        );
    }

    pub(super) fn set_group(&mut self, group: Group) {
        let related = match (self.selected, &self.report) {
            (Some(key), Some(report)) => related_key(report, key, group),
            _ => None,
        };
        self.group = group;
        if related.is_some() {
            self.selected = related;
        }
        self.scroll = 0;
        self.rebuild_rows();
        self.set_status(format!("grouped by {}", group.label()), StatusLevel::Info);
    }

    pub(super) fn toggle_freeze(&mut self) {
        self.frozen = !self.frozen;
        if self.frozen {
            self.set_status("frozen: f resumes", StatusLevel::Info);
        } else {
            let pending = std::mem::take(&mut self.pending);
            let count = pending.len();
            for (report, firing) in pending {
                self.apply_report(report, firing);
            }
            self.set_status(
                format!("resumed, caught up {count} samples"),
                StatusLevel::Info,
            );
        }
    }

    pub(super) fn step_zoom(&mut self, longer: bool) {
        self.zoom = if longer {
            (self.zoom + 1) % ZOOM_SPANS_S.len()
        } else {
            (self.zoom + ZOOM_SPANS_S.len() - 1) % ZOOM_SPANS_S.len()
        };
        self.set_status(
            format!("history span {}", span_label(self.zoom_span_ms())),
            StatusLevel::Info,
        );
    }

    pub(super) fn step_interval(&mut self, longer: bool) {
        if !self.live {
            self.set_status(
                "replay: the recording fixes the interval",
                StatusLevel::Error,
            );
            return;
        }
        let current = self.interval.as_millis() as u64;
        let next = if longer {
            INTERVAL_STEPS_MS.iter().find(|step| **step > current)
        } else {
            INTERVAL_STEPS_MS.iter().rev().find(|step| **step < current)
        };
        match next {
            Some(next) => self
                .effects
                .push(Effect::SetInterval(Duration::from_millis(*next))),
            None => self.set_status(
                format!("interval already at {}", interval_label(self.interval)),
                StatusLevel::Info,
            ),
        }
    }

    /// The run loop reports whether the feed accepted a new interval.
    pub fn interval_applied(&mut self, interval: Duration, accepted: bool) {
        if accepted {
            self.interval = interval;
            self.set_status(
                format!("sampling every {}", interval_label(interval)),
                StatusLevel::Info,
            );
        } else {
            self.set_status("this feed cannot change its interval", StatusLevel::Error);
        }
    }

    /// The run loop reports the outcome of a signal.
    pub fn signal_sent(&mut self, result: Result<String, String>) {
        match result {
            Ok(text) => self.set_status(text, StatusLevel::Info),
            Err(text) => self.set_status(text, StatusLevel::Error),
        }
    }

    /// The process `k` would signal: `(pid, start_us, name)`.
    fn signal_target(&self) -> Option<(i32, Option<u64>, String)> {
        let report = self.report.as_ref()?;
        let key = match &self.screen {
            Screen::Detail(detail) => detail.key,
            Screen::Main => self.selected?,
        };
        let process_target = |pid: i32| {
            report
                .processes
                .iter()
                .find(|process| process.pid == pid)
                .map(|process| (pid, Some(process.start_us), process.name.clone()))
        };
        match key {
            RowKey::Process { pid, start_us } => {
                let name = report
                    .process(pid, start_us)
                    .map(|process| process.name.clone())
                    .or_else(|| match &self.screen {
                        Screen::Detail(detail) => {
                            detail.process.as_ref().map(|process| process.name.clone())
                        }
                        Screen::Main => None,
                    })?;
                Some((pid, Some(start_us), name))
            }
            RowKey::Client { .. } | RowKey::Queue { .. } => {
                let pid = self.selected_row()?.row.pid?;
                process_target(pid)
            }
            RowKey::App { coalition_id } => {
                let app = report.app(coalition_id)?;
                let pid = app.responsible_pid?;
                process_target(pid).or(Some((pid, None, app.name.clone())))
            }
            RowKey::Unattributed => None,
        }
    }

    pub(super) fn prompt_signal(&mut self) {
        if self.readonly {
            self.set_status("read-only: signals are disabled", StatusLevel::Error);
            return;
        }
        if !self.live {
            self.set_status(
                "replay: cannot signal processes from a recording",
                StatusLevel::Error,
            );
            return;
        }
        let Some((pid, start_us, name)) = self.signal_target() else {
            self.set_status("no process to signal here", StatusLevel::Error);
            return;
        };
        if pid <= 1 || pid == std::process::id() as i32 {
            self.set_status(
                format!("refusing to signal {name} ({pid})"),
                StatusLevel::Error,
            );
            return;
        }
        self.overlay = Overlay::Confirm(Confirm {
            pid,
            start_us,
            name,
            signal: 0,
        });
    }

    pub(super) fn open_detail(&mut self) {
        let (Some(row), Some(report)) = (self.selected_row(), &self.report) else {
            return;
        };
        let (key, focus) = match row.row.key {
            key @ (RowKey::Process { .. } | RowKey::App { .. } | RowKey::Unattributed) => {
                (key, None)
            }
            focus @ (RowKey::Client { .. } | RowKey::Queue { .. }) => {
                let Some(process) = row
                    .row
                    .pid
                    .and_then(|pid| report.processes.iter().find(|process| process.pid == pid))
                else {
                    return;
                };
                (
                    RowKey::Process {
                        pid: process.pid,
                        start_us: process.start_us,
                    },
                    Some(focus),
                )
            }
        };
        let mut detail = Detail {
            key,
            focus,
            scroll: 0,
            process: None,
            app: None,
            exited: false,
            parents: Vec::new(),
        };
        detail.refresh(report);
        if let (Some(lookup), Some(process)) = (self.lookup, &detail.process)
            && self.live
        {
            detail.parents = parent_chain(process.ppid, lookup);
        }
        self.screen = Screen::Detail(Box::new(detail));
    }
}

impl Detail {
    fn refresh(&mut self, report: &Report) {
        match self.key {
            RowKey::Process { pid, start_us } => match report.process(pid, start_us) {
                Some(process) => {
                    self.process = Some(process.clone());
                    self.exited = false;
                }
                None => self.exited = self.process.is_some(),
            },
            RowKey::App { coalition_id } => {
                match report
                    .apps
                    .iter()
                    .find(|app| app.coalition_id == coalition_id)
                {
                    Some(app) => {
                        self.app = Some(app.clone());
                        self.exited = false;
                    }
                    None => self.exited = self.app.is_some(),
                }
            }
            _ => {}
        }
        // An app's process: show its app too.
        if let Some(coalition_id) = self
            .process
            .as_ref()
            .and_then(|process| process.coalition_id)
            && let Some(app) = report
                .apps
                .iter()
                .find(|app| app.coalition_id == coalition_id)
        {
            self.app = Some(app.clone());
        }
    }
}

fn parent_chain(mut ppid: Option<i32>, lookup: ProcessLookup) -> Vec<(i32, String)> {
    let mut chain = Vec::new();
    while let Some(pid) = ppid {
        if pid <= 0 || chain.len() >= MAX_PARENT_DEPTH || chain.iter().any(|(seen, _)| *seen == pid)
        {
            break;
        }
        match lookup(pid) {
            Some((parent, name)) => {
                chain.push((pid, name));
                ppid = (pid != 1).then_some(parent);
            }
            None => {
                chain.push((pid, "?".into()));
                break;
            }
        }
    }
    chain
}

/// Checks the system process table.
pub fn system_alive(pid: i32, start_us: u64) -> bool {
    crate::sys::process::start_time_us(pid) == Some(start_us)
}

/// Looks up a live process with the system process table.
pub fn system_lookup(pid: i32) -> Option<(i32, String)> {
    crate::sys::process::info(pid).map(|info| (info.ppid, info.name))
}

pub fn interval_label(interval: Duration) -> String {
    let seconds = interval.as_secs_f64();
    if seconds < 1.0 {
        format!("{}ms", interval.as_millis())
    } else if seconds.fract() == 0.0 {
        format!("{seconds:.0}s")
    } else {
        format!("{seconds:.1}s")
    }
}

pub fn span_label(span_ms: u64) -> String {
    let seconds = span_ms / 1000;
    if seconds >= 3600 && seconds.is_multiple_of(3600) {
        format!("{}h", seconds / 3600)
    } else if seconds >= 60 {
        format!("{}m", seconds / 60)
    } else {
        format!("{seconds}s")
    }
}

/// The row at `group` that best corresponds to `key`, so switching levels keeps
/// the user's place: a process maps to its app, its first client, or first queue.
fn related_key(report: &Report, key: RowKey, group: Group) -> Option<RowKey> {
    let process = match key {
        RowKey::Process { pid, start_us } => report.process(pid, start_us),
        RowKey::Client { registry_id } | RowKey::Queue { registry_id, .. } => report
            .processes
            .iter()
            .find(|process| process.clients.iter().any(|c| c.registry_id == registry_id)),
        RowKey::App { coalition_id } => {
            let app = report
                .apps
                .iter()
                .find(|app| app.coalition_id == coalition_id)?;
            report
                .processes
                .iter()
                .filter(|process| process.coalition_id == Some(coalition_id))
                .max_by_key(|process| process.gpu_ns)
                .or_else(|| {
                    let pid = app.responsible_pid?;
                    report.processes.iter().find(|process| process.pid == pid)
                })
        }
        RowKey::Unattributed => None,
    }?;
    match group {
        Group::Process => Some(RowKey::Process {
            pid: process.pid,
            start_us: process.start_us,
        }),
        Group::App => process
            .coalition_id
            .map(|coalition_id| RowKey::App { coalition_id }),
        Group::Client => {
            if let RowKey::Queue { registry_id, .. } = key {
                return Some(RowKey::Client { registry_id });
            }
            process
                .clients
                .iter()
                .max_by_key(|c| c.gpu_ns)
                .map(|client| RowKey::Client {
                    registry_id: client.registry_id,
                })
        }
        Group::Queue => process
            .clients
            .iter()
            .filter(|client| match key {
                RowKey::Client { registry_id } => client.registry_id == registry_id,
                _ => true,
            })
            .flat_map(|client| client.queues.iter().map(move |queue| (client, queue)))
            .max_by_key(|(_, queue)| queue.gpu_ns)
            .map(|(client, queue)| RowKey::Queue {
                registry_id: client.registry_id,
                index: queue.index,
            }),
    }
}
