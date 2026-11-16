//! System context: thermal pressure, memory pressure, GPU wired-memory limits.
//!
//! Every value here is readable without root and costs a few microseconds.
//!
//! Creating an `MTLDevice` to read `recommendedMaxWorkingSetSize` would open
//! an `AGXDeviceUserClient` for macgpu's own PID, so macgpu would list itself
//! as a GPU client. The default wired limit (when `iogpu.wired_limit_mb` is 0) is
//! not exposed through any sysctl or registry property, so it is left unknown
//! rather than estimated.

use std::ffi::{CStr, c_char, c_int};
use std::sync::Mutex;

unsafe extern "C" {
    fn notify_register_check(name: *const c_char, out_token: *mut c_int) -> u32;
    fn notify_get_state(token: c_int, state: *mut u64) -> u32;
}

const NOTIFY_STATUS_OK: u32 = 0;
/// `kOSThermalNotificationPressureLevelName`.
const THERMAL_PRESSURE_NOTIFICATION: &CStr = c"com.apple.system.thermalpressurelevel";

/// System state at one instant.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemContext {
    /// `OSThermalPressureLevel` from the thermal pressure notification.
    pub thermal_pressure_level: Option<u64>,
    /// `kern.memorystatus_vm_pressure_level`: 1 normal, 2 warning, 4 critical.
    pub memory_pressure_level: Option<u32>,
    /// `iogpu.wired_limit_mb`. 0 means the OS default applies.
    pub wired_limit_mb: Option<u64>,
    /// `hw.memsize`.
    pub total_physical_bytes: Option<u64>,
}

impl SystemContext {
    /// The GPU wired-memory limit in bytes when the user has set one. `None`
    /// when the OS default applies (its value is not exposed) or the sysctl is
    /// missing.
    pub fn wired_limit_bytes(&self) -> Option<u64> {
        self.wired_limit_mb
            .filter(|mb| *mb > 0)
            .map(|mb| mb.saturating_mul(1024 * 1024))
    }

    /// True when `iogpu.wired_limit_mb` is 0, so the OS default limit applies.
    pub fn wired_limit_is_default(&self) -> bool {
        self.wired_limit_mb == Some(0)
    }
}

pub fn read() -> SystemContext {
    SystemContext {
        thermal_pressure_level: thermal_pressure_level(),
        memory_pressure_level: sysctl_u64("kern.memorystatus_vm_pressure_level")
            .and_then(|level| u32::try_from(level).ok()),
        wired_limit_mb: sysctl_u64("iogpu.wired_limit_mb"),
        total_physical_bytes: sysctl_u64("hw.memsize"),
    }
}

/// Name of an `OSThermalPressureLevel` (macOS values 0 to 4).
pub fn thermal_pressure_name(level: u64) -> String {
    match level {
        0 => "nominal".into(),
        1 => "moderate".into(),
        2 => "heavy".into(),
        3 => "trapping".into(),
        4 => "sleeping".into(),
        other => format!("level {other}"),
    }
}

/// Name of a `kern.memorystatus_vm_pressure_level` value.
pub fn memory_pressure_name(level: u32) -> String {
    match level {
        1 => "normal".into(),
        2 => "warning".into(),
        4 => "critical".into(),
        other => format!("level {other}"),
    }
}

fn thermal_pressure_level() -> Option<u64> {
    static TOKEN: Mutex<Option<c_int>> = Mutex::new(None);
    let mut token = TOKEN.lock().ok()?;
    if token.is_none() {
        let mut registered: c_int = 0;
        // SAFETY: the notification name is NUL-terminated and `registered` is a valid out-parameter.
        let status = unsafe {
            notify_register_check(THERMAL_PRESSURE_NOTIFICATION.as_ptr(), &mut registered)
        };
        if status != NOTIFY_STATUS_OK {
            // Leave the cache empty so registration is retried on the next read.
            return None;
        }
        *token = Some(registered);
    }
    let registered = (*token).expect("registered thermal pressure token");
    let mut state = 0u64;
    // SAFETY: `registered` came from `notify_register_check` and `state` is a valid out-parameter.
    let status = unsafe { notify_get_state(registered, &mut state) };
    (status == NOTIFY_STATUS_OK).then_some(state)
}

/// Reads an integer sysctl of 4 or 8 bytes.
fn sysctl_u64(name: &str) -> Option<u64> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut value = [0u8; 8];
    let mut size = value.len();
    // SAFETY: `name` is NUL-terminated, `value` is writable for `size` bytes, and `size` is a valid in/out parameter.
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 {
        return None;
    }
    match size {
        4 => Some(u64::from(u32::from_ne_bytes(value[..4].try_into().ok()?))),
        8 => Some(u64::from_ne_bytes(value)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_levels() {
        assert_eq!(thermal_pressure_name(0), "nominal");
        assert_eq!(thermal_pressure_name(2), "heavy");
        assert_eq!(thermal_pressure_name(9), "level 9");
        assert_eq!(memory_pressure_name(1), "normal");
        assert_eq!(memory_pressure_name(4), "critical");
        assert_eq!(memory_pressure_name(3), "level 3");
    }

    #[test]
    fn wired_limit_only_when_set() {
        let mut context = SystemContext {
            wired_limit_mb: Some(0),
            ..SystemContext::default()
        };
        assert_eq!(context.wired_limit_bytes(), None);
        assert!(context.wired_limit_is_default());
        context.wired_limit_mb = Some(40_960);
        assert_eq!(context.wired_limit_bytes(), Some(40_960 * 1024 * 1024));
        assert!(!context.wired_limit_is_default());
    }

    #[test]
    fn reads_live_context() {
        let started = std::time::Instant::now();
        let context = read();
        let elapsed = started.elapsed();
        eprintln!(
            "system context {context:?} in {:.3} ms",
            elapsed.as_secs_f64() * 1e3
        );
        assert!(context.thermal_pressure_level.is_some());
        assert!(context.memory_pressure_level.is_some());
        assert!(context.total_physical_bytes.unwrap_or(0) > 1 << 30);
        assert!(context.wired_limit_mb.is_some());
    }
}
