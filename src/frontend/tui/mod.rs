//! The interactive view: header meters, history charts, a sortable, filterable
//! table at any grouping level, a tree view, per-row detail, and events.
//!
//! The loop never blocks on the feed: it waits for input with a short timeout,
//! then takes whatever reports are ready, and redraws only when something
//! changed. The state machine lives in [`app`] and has no terminal, so it is
//! tested directly; rendering is tested with ratatui's `TestBackend`.

mod app;
#[cfg(test)]
mod app_tests;
mod charts;
mod detail;
mod fmt;
mod graphs;
mod header;
mod help;
mod history;
mod input;
mod linger;
#[cfg(test)]
mod render_tests;
mod sort;
mod style;
#[cfg(test)]
mod synthetic;
mod table;
mod tree;
mod ui;

use crate::engine::{FeedEvent, ReportFeed};
use crate::frontend::alerts::Alerts;
use crate::frontend::signals;
use crate::view::{Filter, Group};
use anyhow::{Context, Result};
use app::{App, Effect};
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{cursor, execute};
use ratatui::Terminal;
use ratatui::backend::{Backend, CrosstermBackend};
use std::io::{self, Stdout};
use std::sync::{Mutex, Once};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

pub struct TuiOptions {
    pub group: Group,
    pub filter: Filter,
    pub include_idle: bool,
    pub readonly: bool,
    pub alerts: Alerts,
}

/// How long to wait for input before checking the feed again. Reports reach
/// the screen at most this late; idle wakeups cost microseconds.
const INPUT_POLL: Duration = Duration::from_millis(50);
/// After the feed ends there is nothing to check; wake only for input.
const IDLE_POLL: Duration = Duration::from_millis(500);
/// Reports applied per loop turn, so a recording that returns reports
/// without pacing cannot starve input handling.
const MAX_REPORTS_PER_TURN: usize = 64;

pub fn run(feed: Box<dyn ReportFeed>, options: TuiOptions) -> Result<()> {
    // SIGTERM, SIGHUP, or SIGINT from `kill` would otherwise end the process
    // with the terminal still in raw mode on the alternate screen. The loop
    // sees the recorded signal within one input poll and quits normally.
    signals::install(&signals::STOP);
    let mut terminal = TerminalGuard::enter().context("failed to set up the terminal")?;
    let result = run_loop(&mut terminal.terminal, feed, options);
    drop(terminal);
    result
}

fn run_loop<B: Backend>(
    terminal: &mut Terminal<B>,
    mut feed: Box<dyn ReportFeed>,
    options: TuiOptions,
) -> Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let TuiOptions {
        group,
        filter,
        include_idle,
        readonly,
        mut alerts,
    } = options;
    let live = feed.is_live();
    let mut app = App::new(group, filter, include_idle, readonly).with_feed(live, feed.interval());
    if live {
        app = app
            .with_lookup(app::system_lookup)
            .with_liveness(app::system_alive);
    }

    loop {
        if signals::take().is_some() {
            return Ok(());
        }
        if app.take_dirty() {
            terminal.draw(|frame| ui::draw(frame, &mut app))?;
        }

        let timeout = if app.feed_ended {
            IDLE_POLL
        } else {
            INPUT_POLL
        };
        if event::poll(timeout)? {
            loop {
                match event::read()? {
                    Event::Key(key) => app.handle_key(key),
                    Event::Mouse(mouse) => app.handle_mouse(mouse),
                    Event::Resize(..) => app.mark_dirty(),
                    _ => {}
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }

        if !app.feed_ended {
            pump(feed.as_mut(), &mut app, &mut alerts);
        }

        for effect in app.take_effects() {
            match effect {
                Effect::Quit => return Ok(()),
                Effect::SetInterval(interval) => {
                    let accepted = feed.set_interval(interval);
                    app.interval_applied(feed.interval(), accepted);
                }
                Effect::SetIncludeIdle(include_idle) => feed.set_include_idle(include_idle),
                Effect::Signal {
                    pid,
                    start_us,
                    signal,
                    name,
                } => app.signal_sent(send_signal(pid, start_us, signal, &name)),
            }
        }
        app.tick(Instant::now());
    }
}

/// Moves ready reports from the feed into the app without waiting.
fn pump(feed: &mut dyn ReportFeed, app: &mut App, alerts: &mut Alerts) {
    for _ in 0..MAX_REPORTS_PER_TURN {
        match feed.next_report(Duration::ZERO) {
            FeedEvent::Report(mut report) => {
                let firing = alerts.evaluate(&mut report);
                app.push_report(*report, firing);
            }
            FeedEvent::Timeout => break,
            FeedEvent::End => {
                app.end_feed();
                break;
            }
        }
    }
}

/// Sends `signal`, after checking the PID still belongs to the same process.
fn send_signal(pid: i32, start_us: Option<u64>, signal: i32, name: &str) -> Result<String, String> {
    use crate::sys::process;
    let signal_name = app::SIGNALS
        .iter()
        .find(|(number, _)| *number == signal)
        .map_or("signal", |(_, name)| *name);
    if let Some(start_us) = start_us
        && process::start_time_us(pid) != Some(start_us)
    {
        return Err(format!("{name} ({pid}) already exited; nothing sent"));
    }
    match process::send_signal(pid, signal) {
        Ok(()) => Ok(format!("sent {signal_name} to {name} ({pid})")),
        Err(error) if error.raw_os_error() == Some(libc::EPERM) => Err(format!(
            "not permitted to signal {name} ({pid}); it belongs to another user"
        )),
        Err(error) => Err(format!("could not signal {name} ({pid}): {error}")),
    }
}

/// Raw mode, the alternate screen, and mouse capture for as long as it lives.
/// Restores the terminal on drop and on panic.
struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    fn enter() -> Result<Self> {
        install_panic_hook();
        terminal::enable_raw_mode()?;
        *UI_THREAD
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(thread::current().id());
        let setup = execute!(
            io::stdout(),
            EnterAlternateScreen,
            EnableMouseCapture,
            cursor::Hide
        );
        if let Err(error) = setup {
            restore_terminal();
            return Err(error.into());
        }
        match Terminal::new(CrosstermBackend::new(io::stdout())) {
            Ok(terminal) => Ok(Self { terminal }),
            Err(error) => {
                restore_terminal();
                Err(error.into())
            }
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        *UI_THREAD
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = None;
        restore_terminal();
    }
}

/// The thread that owns the terminal while the view is open.
static UI_THREAD: Mutex<Option<ThreadId>> = Mutex::new(None);

fn restore_terminal() {
    let _ = execute!(
        io::stdout(),
        DisableMouseCapture,
        LeaveAlternateScreen,
        cursor::Show
    );
    let _ = terminal::disable_raw_mode();
}

/// Restores the terminal before the default panic message prints, so the
/// message is readable and the shell is usable afterwards. Only a panic on the
/// thread that owns the terminal restores it; the view keeps running (and
/// shows the feed as stopped) if the sampler thread panics.
fn install_panic_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let owner = *UI_THREAD
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if owner == Some(thread::current().id()) {
                restore_terminal();
            }
            previous(info);
        }));
    });
}
