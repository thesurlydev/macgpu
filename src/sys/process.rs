//! Process metadata without spawning `ps`.

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::io;
use std::mem::MaybeUninit;

/// Identity and names for one process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessInfo {
    pub pid: i32,
    pub ppid: i32,
    pub uid: u32,
    /// Process start time in microseconds since the Unix epoch. Together with
    /// the PID this identifies a process across PID reuse.
    pub start_us: u64,
    /// Full executable path, when readable.
    pub path: Option<String>,
    /// Short name: the executable's file name, or the kernel's `p_comm`.
    pub name: String,
}

impl ProcessInfo {
    /// The `.app` bundle the executable lives in, if any, for example
    /// `/Applications/Google Chrome.app`.
    pub fn bundle_path(&self) -> Option<&str> {
        bundle_path(self.path.as_deref()?)
    }
}

/// Returns the outermost `.app` bundle containing `path`.
pub fn bundle_path(path: &str) -> Option<&str> {
    let index = path.find(".app/")?;
    Some(&path[..index + ".app".len()])
}

/// Returns the bundle's display name, for example `Google Chrome`.
pub fn bundle_name(bundle_path: &str) -> &str {
    let file = bundle_path.rsplit('/').next().unwrap_or(bundle_path);
    file.strip_suffix(".app").unwrap_or(file)
}

/// Reads identity and names for `pid`. Works for every user's processes.
pub fn info(pid: i32) -> Option<ProcessInfo> {
    let (ppid, uid, start_us, comm) = bsd_info(pid)?;
    let path = executable_path(pid);
    let name = path
        .as_deref()
        .and_then(|path| path.rsplit('/').next())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .unwrap_or(comm);
    Some(ProcessInfo {
        pid,
        ppid,
        uid,
        start_us,
        path,
        name,
    })
}

/// A cache of [`ProcessInfo`] keyed by PID and validated by start time, so the
/// executable path is looked up once per process rather than once per sample.
#[derive(Default)]
pub struct ProcessCache {
    entries: HashMap<i32, ProcessInfo>,
}

impl ProcessCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&mut self, pid: i32) -> Option<ProcessInfo> {
        let start_us = start_time_us(pid);
        if let (Some(cached), Some(start_us)) = (self.entries.get(&pid), start_us)
            && cached.start_us == start_us
        {
            return Some(cached.clone());
        }
        match info(pid) {
            Some(fresh) => {
                self.entries.insert(pid, fresh.clone());
                Some(fresh)
            }
            None => {
                self.entries.remove(&pid);
                None
            }
        }
    }

    /// Drops entries for PIDs not in `live`.
    pub fn retain(&mut self, live: &dyn Fn(i32) -> bool) {
        self.entries.retain(|pid, _| live(*pid));
    }
}

/// Parent, uid, start time, and `p_comm`, for any user's process.
///
/// `PROC_PIDTBSDINFO` is restricted to the caller's own user, so this combines
/// `PROC_PIDT_SHORTBSDINFO` (allowed for every process) with the start time
/// from `sysctl(KERN_PROC_PID)`.
fn bsd_info(pid: i32) -> Option<(i32, u32, u64, String)> {
    let mut info = MaybeUninit::<libc::proc_bsdshortinfo>::zeroed();
    let size = size_of::<libc::proc_bsdshortinfo>() as c_int;
    let written = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDT_SHORTBSDINFO,
            0,
            info.as_mut_ptr().cast::<c_void>(),
            size,
        )
    };
    if written != size {
        return None;
    }
    let info = unsafe { info.assume_init() };
    let start_us = start_time_us(pid)?;
    Some((
        info.pbsi_ppid as i32,
        info.pbsi_uid,
        start_us,
        c_chars_to_string(&info.pbsi_comm),
    ))
}

/// `sizeof(struct kinfo_proc)` on arm64 macOS.
const KINFO_PROC_SIZE: usize = 648;

/// Process start time in µs since the Unix epoch, from `sysctl(KERN_PROC_PID)`.
///
/// `struct kinfo_proc` begins with `struct extern_proc`, whose first member is
/// the union holding `struct timeval p_starttime` (an `i64` seconds field and an
/// `i32` microseconds field), so only those first 12 bytes are read.
pub fn start_time_us(pid: i32) -> Option<u64> {
    let mut mib = [libc::CTL_KERN, libc::KERN_PROC, libc::KERN_PROC_PID, pid];
    // u64 elements keep the buffer 8-byte aligned.
    let mut buffer = [0u64; KINFO_PROC_SIZE / 8];
    let mut size = KINFO_PROC_SIZE;
    let result = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as u32,
            buffer.as_mut_ptr().cast::<c_void>(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    // A missing PID returns success with size 0.
    if result != 0 || size != KINFO_PROC_SIZE {
        return None;
    }
    let seconds = buffer[0];
    let microseconds = buffer[1] & 0xffff_ffff;
    (seconds > 0).then(|| {
        seconds
            .saturating_mul(1_000_000)
            .saturating_add(microseconds)
    })
}

/// Full executable path via `proc_pidpath`.
pub fn executable_path(pid: i32) -> Option<String> {
    let mut buffer = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    let length = unsafe {
        libc::proc_pidpath(
            pid,
            buffer.as_mut_ptr().cast::<c_void>(),
            buffer.len() as u32,
        )
    };
    if length <= 0 {
        return None;
    }
    buffer.truncate(length as usize);
    String::from_utf8(buffer)
        .ok()
        .filter(|path| !path.is_empty())
}

/// Every PID on the system.
pub fn all_pids() -> Vec<i32> {
    let mut capacity = 4096usize;
    loop {
        let mut pids = vec![0i32; capacity];
        let bytes = (capacity * size_of::<i32>()) as c_int;
        let count = unsafe { libc::proc_listallpids(pids.as_mut_ptr().cast::<c_void>(), bytes) };
        if count < 0 {
            return Vec::new();
        }
        let count = count as usize;
        if count < capacity {
            pids.truncate(count);
            pids.retain(|pid| *pid > 0);
            return pids;
        }
        capacity *= 2;
    }
}

/// Sends `signal` to `pid`.
pub fn send_signal(pid: i32, signal: i32) -> io::Result<()> {
    if pid <= 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid pid"));
    }
    if unsafe { libc::kill(pid, signal) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn c_chars_to_string(chars: &[c_char]) -> String {
    let bytes: Vec<u8> = chars
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Converts a NUL-terminated C string pointer. Returns an empty string for null.
///
/// # Safety
/// `pointer` must be null or point to a NUL-terminated string.
pub unsafe fn c_str(pointer: *const c_char) -> String {
    if pointer.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_own_process() {
        let pid = std::process::id() as i32;
        let info = info(pid).expect("own process info");
        assert_eq!(info.pid, pid);
        assert!(info.start_us > 0);
        assert!(info.path.is_some());
        assert!(!info.name.is_empty());
    }

    #[test]
    fn reads_other_users_process() {
        // launchd is root-owned and always PID 1.
        let info = info(1).expect("launchd info");
        assert_eq!(info.uid, 0);
        assert_eq!(info.name, "launchd");
    }

    #[test]
    fn cache_survives_repeat_lookups() {
        let pid = std::process::id() as i32;
        let mut cache = ProcessCache::new();
        let first = cache.get(pid).expect("first");
        let second = cache.get(pid).expect("second");
        assert_eq!(first, second);
    }

    #[test]
    fn finds_bundle() {
        let path = "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Helpers/Google Chrome Helper (GPU).app/Contents/MacOS/Google Chrome Helper (GPU)";
        let bundle = bundle_path(path).expect("bundle");
        assert_eq!(bundle, "/Applications/Google Chrome.app");
        assert_eq!(bundle_name(bundle), "Google Chrome");
        assert_eq!(bundle_path("/usr/bin/zsh"), None);
    }

    #[test]
    fn lists_pids() {
        let pids = all_pids();
        assert!(pids.contains(&1));
        assert!(pids.contains(&(std::process::id() as i32)));
    }
}
