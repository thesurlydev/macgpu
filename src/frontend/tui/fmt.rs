//! Numeric formats shared by the interactive view.

use crate::format::{GIB, KIB, MIB};

/// GPU ms/s: one decimal below 10 000.
pub fn ms(value: f64) -> String {
    if value >= 9_999.5 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

pub fn pct(value: f64) -> String {
    format!("{value:.1}%")
}

/// Bare watts for table cells (no unit): two decimals below 10 W, one above.
pub fn watts_digits(value: f64) -> String {
    if value >= 9.995 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

/// Bytes as `18.4G`, `642M`, `12K` (binary units) for narrow cells.
pub fn bytes_short(bytes: u64) -> String {
    let value = bytes as f64;
    if value >= 100.0 * GIB {
        format!("{:.0}G", value / GIB)
    } else if value >= GIB {
        format!("{:.1}G", value / GIB)
    } else if value >= MIB {
        format!("{:.0}M", value / MIB)
    } else if value >= KIB {
        format!("{:.0}K", value / KIB)
    } else {
        format!("{bytes}B")
    }
}

/// Local wall-clock time `HH:MM:SS` for a Unix timestamp in milliseconds.
pub fn clock(timestamp_ms: u64) -> String {
    let seconds = (timestamp_ms / 1000) as libc::time_t;
    // SAFETY: `tm` is plain data and `localtime_r` only writes into it.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&seconds, &mut tm) }.is_null() {
        return "--:--:--".into();
    }
    format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

/// The smallest of 1, 2, 2.5, 5 times a power of ten that is at least `value`.
pub fn nice_ceiling(value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        return 1.0;
    }
    let magnitude = 10f64.powf(value.log10().floor());
    for step in [1.0, 2.0, 2.5, 5.0, 10.0] {
        let candidate = step * magnitude;
        if candidate >= value * (1.0 - 1e-9) {
            return candidate;
        }
    }
    10.0 * magnitude
}

/// A process age: `42s`, `17m`, `5.2h`, `8.9d`.
pub fn process_age(seconds: u64) -> String {
    match seconds {
        0..=59 => format!("{seconds}s"),
        60..=3_599 => format!("{}m", seconds / 60),
        3_600..=172_799 => format!("{:.1}h", seconds as f64 / 3600.0),
        _ => format!("{:.1}d", seconds as f64 / 86_400.0),
    }
}

/// A round chart ceiling in binary units: 20 GiB, 750 MiB.
pub fn bytes_ceiling(peak: f64) -> f64 {
    let unit = if peak >= GIB { GIB } else { MIB };
    nice_ceiling((peak / unit).max(1.0)) * unit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_compact_values() {
        assert_eq!(ms(1234.56), "1234.6");
        assert_eq!(ms(12_345.0), "12345");
        assert_eq!(watts_digits(0.314), "0.31");
        assert_eq!(watts_digits(23.44), "23.4");
        assert_eq!(bytes_short(18 * (1 << 30) + 400 * (1 << 20)), "18.4G");
        assert_eq!(bytes_short(642 * (1 << 20)), "642M");
        assert_eq!(bytes_short(128 * (1 << 30)), "128G");
        assert_eq!(bytes_short(512), "512B");
        assert_eq!(clock(0).len(), 8);
    }

    #[test]
    fn nice_ceilings() {
        assert_eq!(nice_ceiling(0.0), 1.0);
        assert_eq!(nice_ceiling(7.0), 10.0);
        assert_eq!(nice_ceiling(10.0), 10.0);
        assert_eq!(nice_ceiling(21.0), 25.0);
        assert_eq!(nice_ceiling(1300.0), 2000.0);
        assert_eq!(nice_ceiling(0.3), 0.5);
    }
}
