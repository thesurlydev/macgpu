//! Termination signals for the long-running front ends (`log`, `serve`,
//! `record`, `run`).
//!
//! The handler only records the signal in atomics; the main loop polls
//! [`take`] and stops cleanly (flushing output, printing a summary) or, for
//! `run`, forwards the signal to the child.

use std::sync::atomic::{AtomicI32, Ordering};

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
        unsafe { (*info).si_pid }
    };
    SENDER.store(sender, Ordering::SeqCst);
    SIGNAL.store(signal, Ordering::SeqCst);
}

/// Installs the recording handler for `signals`, replacing the default action
/// (termination) so the caller decides what to do.
pub fn install(signals: &[i32]) {
    for &signal in signals {
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

    #[test]
    fn records_signal_and_sender() {
        // SIGUSR2 is otherwise unused by the test harness.
        install(&[libc::SIGUSR2]);
        assert_eq!(take(), None);
        unsafe { libc::kill(libc::getpid(), libc::SIGUSR2) };
        // Another test thread may take delivery, asynchronously.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let received = loop {
            if let Some(received) = take() {
                break received;
            }
            assert!(std::time::Instant::now() < deadline, "signal not recorded");
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert_eq!(received.signal, libc::SIGUSR2);
        assert_eq!(received.sender, std::process::id() as i32);
        assert_eq!(take(), None);
    }
}
