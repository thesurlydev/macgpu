//! Termination signals for the long-running front ends (`log`, `serve`,
//! `record`, `run`).
//!
//! The handler only records the signal in atomics; the main loop polls
//! [`take`] and stops cleanly (flushing output, printing a summary) or, for
//! `run`, forwards the signal to the child.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

/// How long a front end waits for a sample before checking for a signal again.
pub const POLL_INTERVAL: Duration = Duration::from_millis(200);

/// What a signal-aware receive produced.
#[derive(Debug)]
pub enum Recv<T> {
    Item(T),
    Timeout,
    Stop,
    Disconnected,
}

/// Receives one item, checking for a termination signal first.
pub fn recv_or_stop<T>(
    mut receive: impl FnMut(Duration) -> Result<T, RecvTimeoutError>,
) -> Recv<T> {
    if take().is_some() {
        return Recv::Stop;
    }
    match receive(POLL_INTERVAL) {
        Ok(item) => Recv::Item(item),
        Err(RecvTimeoutError::Timeout) => Recv::Timeout,
        Err(RecvTimeoutError::Disconnected) => Recv::Disconnected,
    }
}

static SIGNAL: AtomicI32 = AtomicI32::new(0);
static SENDER: AtomicI32 = AtomicI32::new(0);

/// A signal that arrived since the last [`take`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Received {
    pub signal: i32,
    /// PID of the sending process, or 0 when the kernel sent it (for example
    /// Ctrl-C on a terminal, which the terminal also delivers to every process
    /// in the foreground process group).
    pub sender: i32,
}

/// The signals that stop a front end: SIGINT, SIGTERM, SIGHUP.
pub const STOP: [i32; 3] = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP];

extern "C" fn record(signal: libc::c_int, info: *mut libc::siginfo_t, _context: *mut libc::c_void) {
    let sender = if info.is_null() {
        0
    } else {
        // SAFETY: the kernel supplied a valid `siginfo_t` for this SA_SIGINFO
        // delivery; `si_pid` is plain data read in the handler.
        unsafe { (*info).si_pid }
    };
    SENDER.store(sender, Ordering::SeqCst);
    SIGNAL.store(signal, Ordering::SeqCst);
}

/// Installs the recording handler for `signals`, replacing the default action
/// (termination) so the caller decides what to do.
pub fn install(signals: &[i32]) {
    for &signal in signals {
        // SAFETY: `action` is zero-initialized, its mask is initialized by
        // sigemptyset, and `record` is a valid handler. These signals are
        // always valid, so sigaction cannot fail; the old action is unneeded.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = record as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(signal, &action, std::ptr::null_mut());
        }
    }
}

/// Returns and clears the most recent signal.
pub fn take() -> Option<Received> {
    let signal = SIGNAL.swap(0, Ordering::SeqCst);
    (signal != 0).then(|| Received {
        signal,
        sender: SENDER.load(Ordering::SeqCst),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The signal is always recorded; the sender is this process.
    ///
    /// Delivery races with any concurrently reaped child in this process: on
    /// macOS a signal delivered while a child-status change is pending can be
    /// handed that change's `siginfo` (`si_code` a `CLD_*` value, `si_pid` the
    /// child), so `sender` can name a process that never sent anything. Only
    /// the sender is affected. `run` treats any nonzero sender as "a process
    /// sent this, forward it", and a stray child PID is nonzero too, so a
    /// clobbered terminal signal is at worst forwarded redundantly (the child
    /// is in the same process group and has it already). The test therefore
    /// retries until it sees a delivery carrying its own PID.
    #[test]
    fn records_signal_and_sender() {
        // SIGUSR2 is otherwise unused by the test harness.
        install(&[libc::SIGUSR2]);
        assert_eq!(take(), None);
        let own = std::process::id() as i32;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "no delivery carried this process's PID"
            );
            // SAFETY: SIGUSR2 is a valid signal and the target is this process.
            unsafe { libc::kill(libc::getpid(), libc::SIGUSR2) };
            // Another test thread may take delivery, asynchronously.
            let poll_until = std::time::Instant::now() + std::time::Duration::from_millis(100);
            while std::time::Instant::now() < poll_until {
                if let Some(received) = take() {
                    assert_eq!(received.signal, libc::SIGUSR2);
                    if received.sender == own {
                        assert_eq!(take(), None);
                        return;
                    }
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}
