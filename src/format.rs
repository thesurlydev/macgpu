//! Human-readable formatting shared by the text front ends.

/// Truncates to `max` characters, ending with `...` when shortened.
pub fn truncate(value: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if value.chars().count() <= max {
        return value.to_string();
    }
    if max <= 3 {
        return value.chars().take(max).collect();
    }
    let mut result: String = value.chars().take(max - 3).collect();
    result.push_str("...");
    result
}

pub const KIB: f64 = 1024.0;
pub const MIB: f64 = KIB * 1024.0;
pub const GIB: f64 = MIB * 1024.0;

pub fn bytes(bytes: u64) -> String {
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.0} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.0} KiB", bytes / KIB)
    } else {
        format!("{bytes:.0} B")
    }
}

/// Seconds as `12.3s`, `6.2m`, or `6.95h`.
pub fn seconds(seconds: f64) -> String {
    if seconds >= 3600.0 {
        format!("{:.2}h", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1}m", seconds / 60.0)
    } else {
        format!("{seconds:.1}s")
    }
}

/// Age of a last submission: `now`, `850ms`, `12s`, `5m`, `3h`, `2d`.
pub fn age_ms(age_ms: u64) -> String {
    match age_ms {
        0..=99 => "now".into(),
        100..=999 => format!("{age_ms}ms"),
        1_000..=59_999 => format!("{}s", age_ms / 1_000),
        60_000..=3_599_999 => format!("{}m", age_ms / 60_000),
        3_600_000..=86_399_999 => format!("{}h", age_ms / 3_600_000),
        _ => format!("{}d", age_ms / 86_400_000),
    }
}

/// Watts with precision that suits the magnitude: `12.3 W`, `850 mW`.
pub fn watts(watts: f64) -> String {
    if watts >= 1.0 {
        format!("{watts:.1} W")
    } else {
        format!("{:.0} mW", watts * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_respects_width() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdefghij", 10), "abcdefghij");
        assert_eq!(truncate("abcdefghijk", 10), "abcdefg...");
        assert_eq!(truncate("abcdef", 3), "abc");
    }

    #[test]
    fn formats_units() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(9_315_614_720), "8.7 GiB");
        assert_eq!(bytes(2048), "2 KiB");
        assert_eq!(seconds(12.34), "12.3s");
        assert_eq!(seconds(373.0), "6.2m");
        assert_eq!(seconds(25_002.5), "6.95h");
        assert_eq!(age_ms(5), "now");
        assert_eq!(age_ms(850), "850ms");
        assert_eq!(age_ms(12_500), "12s");
        assert_eq!(age_ms(7_200_000), "2h");
        assert_eq!(watts(12.34), "12.3 W");
        assert_eq!(watts(0.85), "850 mW");
    }
}
