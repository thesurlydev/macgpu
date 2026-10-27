//! Clocks: mach absolute time in nanoseconds, CPU time, and wall time.

use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

fn timebase() -> (u64, u64) {
    static TIMEBASE: OnceLock<(u64, u64)> = OnceLock::new();
    *TIMEBASE.get_or_init(|| {
        let mut info = mach2::mach_time::mach_timebase_info { numer: 0, denom: 0 };
        unsafe { mach2::mach_time::mach_timebase_info(&mut info) };
        if info.denom == 0 {
            (1, 1)
        } else {
            (u64::from(info.numer), u64::from(info.denom))
        }
    })
}

/// Converts mach absolute time ticks to nanoseconds.
pub fn mach_ticks_to_ns(ticks: u64) -> u64 {
    let (numer, denom) = timebase();
    ((u128::from(ticks) * u128::from(numer)) / u128::from(denom)) as u64
}

/// `mach_absolute_time` in nanoseconds. This is the clock `AppUsage`
/// `lastSubmittedTime` uses: it does not advance during sleep.
pub fn mach_now_ns() -> u64 {
    mach_ticks_to_ns(unsafe { mach2::mach_time::mach_absolute_time() })
}

/// CPU time consumed by the calling thread, in nanoseconds.
pub fn thread_cpu_ns() -> u64 {
    clock_ns(libc::CLOCK_THREAD_CPUTIME_ID)
}

/// CPU time consumed by the whole process (user plus system), in nanoseconds.
pub fn process_cpu_ns() -> u64 {
    clock_ns(libc::CLOCK_PROCESS_CPUTIME_ID)
}

fn clock_ns(clock: libc::clockid_t) -> u64 {
    let mut spec = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_gettime(clock, &mut spec) } != 0 {
        return 0;
    }
    (spec.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(spec.tv_nsec as u64)
}

/// Milliseconds since the Unix epoch.
pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clocks_advance() {
        let mach = mach_now_ns();
        let cpu = thread_cpu_ns();
        let mut x = 0u64;
        for i in 0..200_000u64 {
            x = x.wrapping_add(i * i);
        }
        assert!(x > 0);
        assert!(mach_now_ns() > mach);
        assert!(thread_cpu_ns() > cpu);
        assert!(process_cpu_ns() > 0);
        assert!(unix_ms() > 1_700_000_000_000);
    }
}
