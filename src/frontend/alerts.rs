//! Threshold alerts over reports.
//!
//! Rules are given with `--alert` (see [`RULE_HELP`] for the grammar). Each
//! report is evaluated in order; a rule fires once its condition has held for
//! its duration and clears once the condition has been false for the same
//! duration. Firing and clearing append [`EventKind::Alert`] and
//! [`EventKind::AlertCleared`] events to the report, so every front end shows
//! and exports them.

use crate::format;
use crate::model::{DeviceView, EventKind, Report};
use crate::view::{self, Filter, Group};
use anyhow::{Result, anyhow};
use clap::Args;
use std::fmt;

/// The `--alert` grammar, shown in `--help`.
pub const RULE_HELP: &str = "\
Alert rule (repeatable).

Grammar:  [SCOPE:PATTERN | METRIC] OP VALUE[UNIT] [for DURATION]   or   recovery | issue

Device metrics:  active %, normalized % (clock-weighted active), firmware %,
  mhz, power W, temp C (GPU die average), temp_max C (hottest sensor),
  memory (GPU memory in use: B, KiB, MiB, GiB, or KB, MB, GB), busy ms/s
  (apportioned GPU time of all processes; about 1000 when saturated),
  limited % (time under a power or thermal limiter state; clocks are not
  necessarily reduced).
Scopes:  process:PATTERN or app:PATTERN. PATTERN uses the --filter syntax
  (comma-separated substrings of name, path, or PID; `!` excludes), and the
  values of all matching rows are summed. The unit picks the metric: ms/s
  apportioned GPU time (the driver splits busy time among concurrently running
  work), % share of all GPU time, W power, bytes graphics memory.
OP:  >, >=, <, <=.   DURATION:  500ms, 10s, 5m, 1h.
recovery fires when the GPU recovers from a hang; issue when the IOReport
  issue counters move.

A rule fires once its condition has held for DURATION (at once without `for`)
and clears once it has been false for the same DURATION. A rule whose value is
unavailable (for example power with the power source disabled) never fires.

Examples:  active>80%   'power>20W for 10s'   temp>95C   memory>40GiB
  'process:chrome>500ms/s for 5s'   app:Xcode>30%   recovery";

#[derive(Args, Debug, Clone, Default)]
pub struct AlertArgs {
    /// Alert rule (repeatable), for example 'power>20W for 10s'. See --help.
    #[arg(long = "alert", value_name = "RULE", long_help = RULE_HELP, value_parser = validate_rule)]
    pub rules: Vec<String>,

    /// Exit with status 3 once any alert fires (printed and log modes).
    #[arg(long)]
    pub exit_on_alert: bool,
}

fn validate_rule(text: &str) -> Result<String, String> {
    Rule::parse(text).map(|_| text.to_string())
}

/// Evaluates every rule against each report in turn.
#[derive(Debug, Clone, Default)]
pub struct Alerts {
    rules: Vec<(Rule, RuleState)>,
    last_recovery_count: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RuleState {
    /// Seconds the condition has held continuously.
    true_seconds: f64,
    /// Seconds the condition has been false continuously.
    false_seconds: f64,
    firing: bool,
}

impl Alerts {
    pub fn from_args(args: &AlertArgs) -> Result<Self> {
        Self::from_rules(&args.rules)
    }

    pub fn from_rules(rules: &[String]) -> Result<Self> {
        let rules = rules
            .iter()
            .map(|text| {
                Rule::parse(text)
                    .map(|rule| (rule, RuleState::default()))
                    .map_err(|error| anyhow!("invalid --alert {text:?}: {error}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rules,
            last_recovery_count: None,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The rules, as written, that are firing now.
    pub fn firing(&self) -> Vec<String> {
        self.rules
            .iter()
            .filter(|(_, state)| state.firing)
            .map(|(rule, _)| rule.to_string())
            .collect()
    }

    /// Evaluates rules against `report`, appending alert events to it.
    /// Returns true if any alert is firing after this report.
    pub fn evaluate(&mut self, report: &mut Report) -> bool {
        let recovered = report
            .events
            .iter()
            .any(|event| event.kind == EventKind::GpuRecovery)
            || matches!(
                (self.last_recovery_count, report.device.recovery_count),
                (Some(before), Some(after)) if after > before
            );
        let issue = report
            .events
            .iter()
            .any(|event| event.kind == EventKind::GpuIssue);
        if report.device.recovery_count.is_some() {
            self.last_recovery_count = report.device.recovery_count;
        }

        let window = report.window_seconds.max(0.0);
        let mut events = Vec::new();
        for (rule, state) in &mut self.rules {
            let observed = match &rule.condition {
                Condition::Recovery => Some(Observed::flag(recovered, "GPU recovered from a hang")),
                Condition::Issue => Some(Observed::flag(issue, "GPU issue counters increased")),
                Condition::Threshold(threshold) => threshold.observe(report),
            };
            let Some(observed) = observed else {
                continue;
            };
            // Round to the nearest window, so `for 10s` with 1 s windows fires
            // on the tenth window even when measured windows run slightly short.
            let tolerance = window / 2.0;
            if observed.holds {
                state.true_seconds += window;
                state.false_seconds = 0.0;
                if !state.firing && state.true_seconds + tolerance >= rule.hold_seconds {
                    state.firing = true;
                    events.push((
                        EventKind::Alert,
                        format!("alert {rule} firing: {}", observed.text),
                    ));
                }
            } else {
                state.false_seconds += window;
                state.true_seconds = 0.0;
                if state.firing && state.false_seconds + tolerance >= rule.hold_seconds {
                    state.firing = false;
                    events.push((
                        EventKind::AlertCleared,
                        format!("alert {rule} cleared: {}", observed.text),
                    ));
                }
            }
        }
        for (kind, message) in events {
            report.events.push(crate::model::Event {
                timestamp_ms: report.timestamp_ms,
                kind,
                message,
            });
        }
        self.rules.iter().any(|(_, state)| state.firing)
    }
}

/// One parsed `--alert` rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    text: String,
    condition: Condition,
    hold_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum Condition {
    Threshold(Threshold),
    Recovery,
    Issue,
}

#[derive(Debug, Clone, PartialEq)]
struct Threshold {
    subject: Subject,
    comparison: Comparison,
    /// In the metric's unit: percent, watts, Celsius, MHz, bytes, or ms/s.
    value: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum Subject {
    Device(DeviceMetric),
    Rows {
        group: Group,
        filter: Filter,
        metric: RowMetric,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceMetric {
    Active,
    Normalized,
    Firmware,
    Mhz,
    Power,
    Temp,
    TempMax,
    Memory,
    Busy,
    Limited,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowMetric {
    Busy,
    Share,
    Power,
    Memory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Comparison {
    Above,
    AtLeast,
    Below,
    AtMost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    Percent,
    Watts,
    Celsius,
    Mhz,
    Bytes,
    MsPerS,
}

impl Unit {
    fn format(self, value: f64) -> String {
        match self {
            Unit::Percent => format!("{value:.1}%"),
            Unit::Watts => format::watts(value),
            Unit::Celsius => format!("{value:.1} C"),
            Unit::Mhz => format!("{value:.0} MHz"),
            Unit::Bytes => format::bytes(value.max(0.0) as u64),
            Unit::MsPerS => format!("{value:.1} ms/s"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Unit::Percent => "%",
            Unit::Watts => "W",
            Unit::Celsius => "C",
            Unit::Mhz => "MHz",
            Unit::Bytes => "bytes",
            Unit::MsPerS => "ms/s",
        }
    }
}

impl DeviceMetric {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "active" => Self::Active,
            "normalized" | "norm" => Self::Normalized,
            "firmware" => Self::Firmware,
            "mhz" | "freq" | "frequency" => Self::Mhz,
            "power" => Self::Power,
            "temp" | "temperature" => Self::Temp,
            "temp_max" | "tempmax" => Self::TempMax,
            "memory" | "mem" => Self::Memory,
            "busy" => Self::Busy,
            "limited" | "throttled" => Self::Limited,
            _ => return None,
        })
    }

    fn unit(self) -> Unit {
        match self {
            Self::Active | Self::Normalized | Self::Firmware | Self::Limited => Unit::Percent,
            Self::Mhz => Unit::Mhz,
            Self::Power => Unit::Watts,
            Self::Temp | Self::TempMax => Unit::Celsius,
            Self::Memory => Unit::Bytes,
            Self::Busy => Unit::MsPerS,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Normalized => "normalized",
            Self::Firmware => "firmware",
            Self::Mhz => "clock",
            Self::Power => "power",
            Self::Temp => "temp",
            Self::TempMax => "temp_max",
            Self::Memory => "memory",
            Self::Busy => "busy",
            Self::Limited => "limited",
        }
    }

    fn value(self, report: &Report) -> Option<f64> {
        let device = &report.device;
        match self {
            Self::Active => device.active_pct,
            Self::Normalized => device.freq_normalized_pct,
            Self::Firmware => device
                .firmware_utilization
                .as_ref()
                .map(|firmware| firmware.device_pct),
            Self::Mhz => device.avg_mhz,
            Self::Power => device.power_w,
            Self::Temp => device.temperature.as_ref().map(|t| t.avg_c),
            Self::TempMax => device.temperature.as_ref().map(|t| t.max_c),
            Self::Memory => device.memory.in_use_bytes.map(|bytes| bytes as f64),
            Self::Busy => Some(report.totals.attributed_ms_per_s),
            Self::Limited => limited_pct(device),
        }
    }
}

impl RowMetric {
    fn from_unit(unit: Unit) -> Option<Self> {
        Some(match unit {
            Unit::MsPerS => Self::Busy,
            Unit::Percent => Self::Share,
            Unit::Watts => Self::Power,
            Unit::Bytes => Self::Memory,
            Unit::Celsius | Unit::Mhz => return None,
        })
    }

    fn unit(self) -> Unit {
        match self {
            Self::Busy => Unit::MsPerS,
            Self::Share => Unit::Percent,
            Self::Power => Unit::Watts,
            Self::Memory => Unit::Bytes,
        }
    }
}

/// Share of the window under a power or thermal limiter state. See
/// [`DeviceView::limited_pct`].
pub fn limited_pct(device: &DeviceView) -> Option<f64> {
    device.limited_pct()
}

struct Observed {
    holds: bool,
    text: String,
}

impl Observed {
    fn flag(holds: bool, text: &str) -> Self {
        Self {
            holds,
            text: if holds { text.into() } else { "none".into() },
        }
    }
}

impl Threshold {
    fn observe(&self, report: &Report) -> Option<Observed> {
        let (value, unit, label) = match &self.subject {
            Subject::Device(metric) => (
                metric.value(report)?,
                metric.unit(),
                metric.label().to_string(),
            ),
            Subject::Rows {
                group,
                filter,
                metric,
            } => {
                let rows = view::rows(report, *group, filter);
                let value = match metric {
                    RowMetric::Busy => Some(
                        rows.iter()
                            .fold(0.0, |total, row| total + row.busy_ms_per_s),
                    ),
                    RowMetric::Share => {
                        Some(rows.iter().fold(0.0, |total, row| total + row.share_pct))
                    }
                    RowMetric::Power => sum_known(rows.iter().map(|row| row.power_w)),
                    RowMetric::Memory => sum_known(
                        rows.iter()
                            .map(|row| row.memory_bytes.map(|bytes| bytes as f64)),
                    ),
                }?;
                let label = format!(
                    "{} {} matching {filter}",
                    rows.len(),
                    group_label(*group, rows.len())
                );
                (value, metric.unit(), label)
            }
        };
        let holds = match self.comparison {
            Comparison::Above => value > self.value,
            Comparison::AtLeast => value >= self.value,
            Comparison::Below => value < self.value,
            Comparison::AtMost => value <= self.value,
        };
        Some(Observed {
            holds,
            text: format!("{label} {}", unit.format(value)),
        })
    }
}

fn group_label(group: Group, count: usize) -> &'static str {
    match (group, count == 1) {
        (Group::App, true) => "app",
        (Group::App, false) => "apps",
        (_, true) => "process",
        (_, false) => "processes",
    }
}

/// Sums the known values; 0 for no rows, `None` when rows exist but none is known.
fn sum_known(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let mut any_row = false;
    let mut total = None;
    for value in values {
        any_row = true;
        if let Some(value) = value {
            total = Some(total.unwrap_or(0.0) + value);
        }
    }
    if any_row { total } else { Some(0.0) }
}

impl fmt::Display for Rule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "`{}`", self.text)
    }
}

impl Rule {
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let lower = normalized.to_lowercase();
        match lower.as_str() {
            "recovery" => {
                return Ok(Self {
                    text: normalized,
                    condition: Condition::Recovery,
                    hold_seconds: 0.0,
                });
            }
            "issue" => {
                return Ok(Self {
                    text: normalized,
                    condition: Condition::Issue,
                    hold_seconds: 0.0,
                });
            }
            "" => return Err("empty rule".into()),
            _ => {}
        }

        let (condition, hold_seconds) = match lower.rfind(" for ") {
            Some(index) => {
                let duration = &normalized[index + " for ".len()..];
                let seconds = parse_duration(duration)?;
                (&normalized[..index], seconds)
            }
            None => (normalized.as_str(), 0.0),
        };

        let operator_at = condition
            .find(['<', '>'])
            .ok_or("expected a comparison: >, >=, <, or <=")?;
        let (subject, rest) = condition.split_at(operator_at);
        let (comparison, value) = if let Some(value) = rest.strip_prefix(">=") {
            (Comparison::AtLeast, value)
        } else if let Some(value) = rest.strip_prefix("<=") {
            (Comparison::AtMost, value)
        } else if let Some(value) = rest.strip_prefix('>') {
            (Comparison::Above, value)
        } else {
            (Comparison::Below, &rest[1..])
        };
        let (value, unit) = parse_value(value.trim())?;
        let subject = subject.trim();

        let subject = match subject.split_once(':') {
            Some((scope, pattern)) => {
                let group = match scope.trim().to_lowercase().as_str() {
                    "process" | "proc" => Group::Process,
                    "app" => Group::App,
                    other => {
                        return Err(format!("unknown scope {other:?}; use process: or app:"));
                    }
                };
                let filter = Filter::parse(pattern);
                if filter.is_empty() {
                    return Err(format!(
                        "{scope}: needs a pattern, for example {scope}:chrome"
                    ));
                }
                let unit = unit.ok_or(
                    "a process or app rule needs a unit: ms/s, %, W, or bytes (for example GiB)",
                )?;
                let metric = RowMetric::from_unit(unit).ok_or(format!(
                    "{} is not a per-row metric; use ms/s, %, W, or bytes",
                    unit.name()
                ))?;
                Subject::Rows {
                    group,
                    filter,
                    metric,
                }
            }
            None => {
                let metric = DeviceMetric::parse(&subject.to_lowercase()).ok_or(format!(
                    "unknown metric {subject:?}; expected active, normalized, firmware, mhz, \
                     power, temp, temp_max, memory, busy, limited, or a process: or app: scope"
                ))?;
                if let Some(unit) = unit
                    && unit != metric.unit()
                {
                    return Err(format!(
                        "{subject} is measured in {}, not {}",
                        metric.unit().name(),
                        unit.name()
                    ));
                }
                Subject::Device(metric)
            }
        };

        Ok(Self {
            text: normalized.clone(),
            condition: Condition::Threshold(Threshold {
                subject,
                comparison,
                value,
            }),
            hold_seconds,
        })
    }
}

/// Parses `20`, `20W`, `850mW`, `95C`, `30%`, `40GiB`, `500ms/s`, `1.2GHz`,
/// returning the value in the unit's base (W, C, %, bytes, ms/s, MHz).
fn parse_value(text: &str) -> Result<(f64, Option<Unit>), String> {
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let number: f64 = number
        .parse()
        .map_err(|_| format!("expected a number, got {text:?}"))?;
    let unit_text = unit.trim().to_lowercase();
    let (scale, unit) = match unit_text.as_str() {
        "" => (1.0, None),
        "%" => (1.0, Some(Unit::Percent)),
        "w" => (1.0, Some(Unit::Watts)),
        "mw" => (1e-3, Some(Unit::Watts)),
        "c" | "°c" => (1.0, Some(Unit::Celsius)),
        "mhz" => (1.0, Some(Unit::Mhz)),
        "ghz" => (1e3, Some(Unit::Mhz)),
        "ms/s" => (1.0, Some(Unit::MsPerS)),
        "b" => (1.0, Some(Unit::Bytes)),
        "kib" => (1024.0, Some(Unit::Bytes)),
        "mib" => (1024.0 * 1024.0, Some(Unit::Bytes)),
        "gib" => (1024.0 * 1024.0 * 1024.0, Some(Unit::Bytes)),
        "tib" => (1024.0 * 1024.0 * 1024.0 * 1024.0, Some(Unit::Bytes)),
        "kb" => (1e3, Some(Unit::Bytes)),
        "mb" => (1e6, Some(Unit::Bytes)),
        "gb" => (1e9, Some(Unit::Bytes)),
        "tb" => (1e12, Some(Unit::Bytes)),
        other => return Err(format!("unknown unit {other:?}")),
    };
    Ok((number * scale, unit))
}

/// Parses a duration: `500ms`, `10s`, `1.5m`, `2h`, or bare seconds. Returns seconds.
pub fn parse_duration(text: &str) -> Result<f64, String> {
    let text = text.trim().to_lowercase();
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let number: f64 = number
        .parse()
        .map_err(|_| format!("expected a duration like 10s, got {text:?}"))?;
    let scale = match unit.trim() {
        "ms" => 1e-3,
        "" | "s" | "sec" => 1.0,
        "m" | "min" => 60.0,
        "h" => 3600.0,
        other => {
            return Err(format!(
                "unknown duration unit {other:?}; use ms, s, m, or h"
            ));
        }
    };
    let seconds = number * scale;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(format!("invalid duration {text:?}"));
    }
    Ok(seconds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        AppView, Event, FirmwareUtilization, ProcessView, StateShare, Temperature, Totals,
    };

    fn rule(text: &str) -> Rule {
        Rule::parse(text).unwrap_or_else(|error| panic!("{text}: {error}"))
    }

    fn threshold(text: &str) -> Threshold {
        match rule(text).condition {
            Condition::Threshold(threshold) => threshold,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parses_device_rules() {
        let parsed = rule("active>80%");
        assert_eq!(parsed.hold_seconds, 0.0);
        assert_eq!(
            parsed.condition,
            Condition::Threshold(Threshold {
                subject: Subject::Device(DeviceMetric::Active),
                comparison: Comparison::Above,
                value: 80.0
            })
        );
        let parsed = rule("  power >= 20W   for 10s ");
        assert_eq!(parsed.hold_seconds, 10.0);
        assert_eq!(parsed.to_string(), "`power >= 20W for 10s`");
        assert_eq!(
            threshold("power >= 20W for 10s").comparison,
            Comparison::AtLeast
        );
        assert_eq!(threshold("temp>95C").value, 95.0);
        assert_eq!(threshold("temp_max<=100").comparison, Comparison::AtMost);
        assert_eq!(
            threshold("memory>40GiB").value,
            40.0 * 1024.0 * 1024.0 * 1024.0
        );
        assert_eq!(threshold("memory>40GB").value, 40e9);
        assert_eq!(threshold("power>850mW").value, 0.85);
        assert_eq!(threshold("mhz<1.2GHz").value, 1200.0);
        assert_eq!(threshold("busy>500ms/s").comparison, Comparison::Above);
        assert_eq!(
            threshold("Active<5% for 1m").subject,
            Subject::Device(DeviceMetric::Active)
        );
        assert_eq!(rule("Active<5% for 1m").hold_seconds, 60.0);
        assert_eq!(rule("RECOVERY").condition, Condition::Recovery);
        assert_eq!(rule("issue").condition, Condition::Issue);
    }

    #[test]
    fn parses_scoped_rules() {
        let parsed = threshold("process:chrome>500ms/s");
        assert_eq!(
            parsed.subject,
            Subject::Rows {
                group: Group::Process,
                filter: Filter::parse("chrome"),
                metric: RowMetric::Busy
            }
        );
        let parsed = threshold("app:Google Chrome, !canary > 30%");
        assert_eq!(
            parsed.subject,
            Subject::Rows {
                group: Group::App,
                filter: Filter::parse("google chrome, !canary"),
                metric: RowMetric::Share
            }
        );
        assert!(matches!(
            threshold("app:Xcode>2W").subject,
            Subject::Rows {
                metric: RowMetric::Power,
                ..
            }
        ));
        assert!(matches!(
            threshold("process:ollama>=10GiB").subject,
            Subject::Rows {
                metric: RowMetric::Memory,
                ..
            }
        ));
        assert_eq!(rule("process:chrome>500ms/s for 5s").hold_seconds, 5.0);
    }

    #[test]
    fn rejects_bad_rules() {
        for (text, needle) in [
            ("", "empty"),
            ("active", "comparison"),
            ("activ>80%", "unknown metric"),
            ("active>80W", "measured in %"),
            ("active>abc", "expected a number"),
            ("active>80 parsecs", "unknown unit"),
            ("power>20W for ever", "duration"),
            ("power>20W for 10y", "duration unit"),
            ("process:>5ms/s", "needs a pattern"),
            ("process:chrome>5", "needs a unit"),
            ("process:chrome>5C", "not a per-row metric"),
            ("node:chrome>5%", "unknown scope"),
        ] {
            let error = Rule::parse(text).expect_err(text);
            assert!(error.contains(needle), "{text}: {error}");
        }
        assert!(Alerts::from_rules(&["bogus".into()]).is_err());
        assert!(validate_rule("active>80%").is_ok());
    }

    #[test]
    fn parses_durations() {
        assert_eq!(parse_duration("500ms"), Ok(0.5));
        assert_eq!(parse_duration("10"), Ok(10.0));
        assert_eq!(parse_duration("1.5m"), Ok(90.0));
        assert_eq!(parse_duration("2h"), Ok(7200.0));
        assert!(parse_duration("-1s").is_err());
        assert!(parse_duration("s").is_err());
    }

    fn report(active: Option<f64>) -> Report {
        Report {
            timestamp_ms: 1_000,
            window_seconds: 1.0,
            device: DeviceView {
                active_pct: active,
                power_w: Some(25.0),
                temperature: Some(Temperature {
                    avg_c: 90.0,
                    max_c: 97.0,
                    sensors: 4,
                }),
                firmware_utilization: Some(FirmwareUtilization {
                    device_pct: 50.0,
                    ..FirmwareUtilization::default()
                }),
                ..DeviceView::default()
            },
            totals: Totals {
                attributed_ms_per_s: 900.0,
                ..Totals::default()
            },
            processes: vec![
                ProcessView {
                    pid: 10,
                    start_us: 1,
                    name: "Google Chrome Helper (GPU)".into(),
                    gpu_ns: 400_000_000,
                    busy_ms_per_s: 400.0,
                    share_pct: 40.0,
                    est_power_w: Some(4.0),
                    graphics_memory_bytes: Some(1 << 30),
                    ..ProcessView::default()
                },
                ProcessView {
                    pid: 11,
                    start_us: 1,
                    name: "Google Chrome Helper (Renderer)".into(),
                    gpu_ns: 200_000_000,
                    busy_ms_per_s: 200.0,
                    share_pct: 20.0,
                    est_power_w: None,
                    graphics_memory_bytes: None,
                    ..ProcessView::default()
                },
                ProcessView {
                    pid: 12,
                    start_us: 1,
                    name: "WindowServer".into(),
                    gpu_ns: 300_000_000,
                    busy_ms_per_s: 300.0,
                    share_pct: 30.0,
                    ..ProcessView::default()
                },
            ],
            apps: vec![AppView {
                coalition_id: 5,
                name: "Xcode".into(),
                gpu_ns: 100,
                share_pct: 35.0,
                power_w: Some(3.0),
                ..AppView::default()
            }],
            ..Report::default()
        }
    }

    fn observe(text: &str, report: &Report) -> Option<(bool, String)> {
        threshold(text)
            .observe(report)
            .map(|observed| (observed.holds, observed.text))
    }

    #[test]
    fn observes_values() {
        let report = report(Some(85.0));
        assert_eq!(
            observe("active>80%", &report),
            Some((true, "active 85.0%".into()))
        );
        assert_eq!(observe("active<=85", &report).map(|o| o.0), Some(true));
        assert_eq!(observe("active<85", &report).map(|o| o.0), Some(false));
        assert_eq!(
            observe("power>20W", &report).map(|o| o.1),
            Some("power 25.0 W".into())
        );
        assert_eq!(observe("temp>95C", &report).map(|o| o.0), Some(false));
        assert_eq!(observe("temp_max>95C", &report).map(|o| o.0), Some(true));
        assert_eq!(observe("firmware>=50%", &report).map(|o| o.0), Some(true));
        assert_eq!(observe("busy>800ms/s", &report).map(|o| o.0), Some(true));
        // Unavailable sources never hold.
        assert_eq!(observe("memory>1GiB", &report), None);
        assert_eq!(observe("mhz>1", &report), None);
        // IOReport active time known, no limiter residency: a real 0.
        assert_eq!(observe("limited>1%", &report).map(|o| o.0), Some(false));
        assert_eq!(observe("limited>1%", &super::tests::report(None)), None);
        assert_eq!(observe("active>1%", &super::tests::report(None)), None);

        // Rows are summed across matches.
        assert_eq!(
            observe("process:chrome>550ms/s", &report),
            Some((true, "2 processes matching chrome 600.0 ms/s".into()))
        );
        assert_eq!(
            observe("process:chrome,!renderer>550ms/s", &report).map(|o| o.0),
            Some(false)
        );
        assert_eq!(
            observe("process:chrome>=60%", &report).map(|o| o.0),
            Some(true)
        );
        // Unknown power on one member: the known values are summed.
        assert_eq!(
            observe("process:chrome>3W", &report).map(|o| o.1),
            Some("2 processes matching chrome 4.0 W".into())
        );
        // Every matching row unknown: unavailable.
        assert_eq!(observe("process:renderer>1MiB", &report), None);
        assert_eq!(observe("process:windowserver>1W", &report), None);
        // No matching row: zero.
        assert_eq!(
            observe("process:nothing>1ms/s", &report),
            Some((false, "0 processes matching nothing 0.0 ms/s".into()))
        );
        assert_eq!(
            observe("process:12>299ms/s", &report).map(|o| o.0),
            Some(true)
        );
        assert_eq!(
            observe("app:xcode>30%", &report),
            Some((true, "1 app matching xcode 35.0%".into()))
        );
        assert_eq!(observe("app:xcode>2W", &report).map(|o| o.0), Some(true));
    }

    #[test]
    fn limited_combines_controller_and_thermal() {
        let mut device = DeviceView::default();
        assert_eq!(limited_pct(&device), None);
        let state = |name: &str, pct: f64| StateShare {
            name: name.into(),
            pct,
        };
        device.limiters = vec![
            state("PERF", 40.0),
            state("PWR_ZONE", 30.0),
            state("PRFBOOST", 10.0),
            state("DEADLINE", 5.0),
            state("CLTMLIMI", 10.0),
            state("FAST_DIE", 5.0),
        ];
        // Limiter residency without GPUPH active time cannot be placed in the window.
        assert_eq!(limited_pct(&device), None);
        // 45% of powered time, powered half the window.
        device.active_pct = Some(50.0);
        assert_eq!(limited_pct(&device), Some(22.5));
        device.thermal_limited_pct = Some(30.0);
        assert_eq!(limited_pct(&device), Some(30.0));
        // IOReport working, GPU off the whole window: a real 0.
        device.limiters.clear();
        device.active_pct = Some(0.0);
        device.thermal_limited_pct = None;
        assert_eq!(limited_pct(&device), Some(0.0));
        assert!(crate::model::is_limiting_state(" ppmlimit "));
        assert!(!crate::model::is_limiting_state("IDLE_OFF"));
    }

    fn evaluate(
        alerts: &mut Alerts,
        active: Option<f64>,
        window: f64,
    ) -> (bool, Vec<(EventKind, String)>) {
        let mut report = report(active);
        report.window_seconds = window;
        let firing = alerts.evaluate(&mut report);
        (
            firing,
            report
                .events
                .into_iter()
                .map(|event| (event.kind, event.message))
                .collect(),
        )
    }

    #[test]
    fn fires_immediately_without_duration() {
        let mut alerts = Alerts::from_rules(&["active>80%".into()]).unwrap();
        assert!(!alerts.is_empty());
        let (firing, events) = evaluate(&mut alerts, Some(50.0), 1.0);
        assert!(!firing && events.is_empty());
        let (firing, events) = evaluate(&mut alerts, Some(90.0), 1.0);
        assert!(firing);
        assert_eq!(
            events,
            vec![(
                EventKind::Alert,
                "alert `active>80%` firing: active 90.0%".into()
            )]
        );
        assert_eq!(alerts.firing(), vec!["`active>80%`".to_string()]);
        // Still firing: no new event.
        let (firing, events) = evaluate(&mut alerts, Some(95.0), 1.0);
        assert!(firing && events.is_empty());
        let (firing, events) = evaluate(&mut alerts, Some(10.0), 1.0);
        assert!(!firing);
        assert_eq!(
            events,
            vec![(
                EventKind::AlertCleared,
                "alert `active>80%` cleared: active 10.0%".into()
            )]
        );
    }

    #[test]
    fn duration_hysteresis() {
        let mut alerts = Alerts::from_rules(&["active>80% for 3s".into()]).unwrap();
        // Two true windows, then a dip: the timer restarts.
        assert!(!evaluate(&mut alerts, Some(90.0), 1.0).0);
        assert!(!evaluate(&mut alerts, Some(90.0), 1.0).0);
        assert!(!evaluate(&mut alerts, Some(10.0), 1.0).0);
        assert!(!evaluate(&mut alerts, Some(90.0), 1.0).0);
        assert!(!evaluate(&mut alerts, Some(90.0), 1.0).0);
        // Slightly short windows still fire on the third.
        let (firing, events) = evaluate(&mut alerts, Some(90.0), 0.98);
        assert!(firing);
        assert_eq!(events.len(), 1);
        // Clearing needs 3 s of false too; a brief dip does not clear.
        assert!(evaluate(&mut alerts, Some(10.0), 1.0).0);
        assert!(evaluate(&mut alerts, Some(90.0), 1.0).0);
        assert!(evaluate(&mut alerts, Some(10.0), 1.0).0);
        assert!(evaluate(&mut alerts, Some(10.0), 1.0).0);
        let (firing, events) = evaluate(&mut alerts, Some(10.0), 1.0);
        assert!(!firing);
        assert_eq!(events[0].0, EventKind::AlertCleared);
    }

    #[test]
    fn unavailable_values_keep_state() {
        let mut alerts = Alerts::from_rules(&["active>80% for 2s".into()]).unwrap();
        assert!(!evaluate(&mut alerts, Some(90.0), 1.0).0);
        assert!(!evaluate(&mut alerts, None, 1.0).0);
        assert!(evaluate(&mut alerts, Some(90.0), 1.0).0);
        // Unavailable does not clear a firing alert.
        assert!(evaluate(&mut alerts, None, 1.0).0);
        assert!(evaluate(&mut alerts, None, 1.0).0);
    }

    #[test]
    fn recovery_from_event_or_counter() {
        let mut alerts = Alerts::from_rules(&["recovery".into()]).unwrap();
        let mut quiet = report(None);
        quiet.device.recovery_count = Some(2);
        assert!(!alerts.evaluate(&mut quiet));

        let mut recovered = report(None);
        recovered.device.recovery_count = Some(3);
        assert!(alerts.evaluate(&mut recovered));
        assert_eq!(recovered.events[0].kind, EventKind::Alert);
        assert!(
            recovered.events[0]
                .message
                .contains("recovered from a hang")
        );

        let mut after = report(None);
        after.device.recovery_count = Some(3);
        assert!(!alerts.evaluate(&mut after));
        assert_eq!(after.events[0].kind, EventKind::AlertCleared);

        let mut evented = report(None);
        evented.events.push(Event {
            timestamp_ms: 1,
            kind: EventKind::GpuRecovery,
            message: "GPU recovered".into(),
        });
        assert!(alerts.evaluate(&mut evented));

        let mut issues = Alerts::from_rules(&["issue".into()]).unwrap();
        let mut issue = report(None);
        issue.events.push(Event {
            timestamp_ms: 1,
            kind: EventKind::GpuIssue,
            message: "issue".into(),
        });
        assert!(issues.evaluate(&mut issue));
    }

    #[test]
    fn no_rules_never_fire() {
        let mut alerts = Alerts::from_args(&AlertArgs::default()).unwrap();
        assert!(alerts.is_empty());
        let mut report = report(Some(100.0));
        assert!(!alerts.evaluate(&mut report));
        assert!(report.events.is_empty());
    }
}
