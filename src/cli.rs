//! Command-line interface.

use crate::analyze::AnalyzeOptions;
use crate::engine::{EngineConfig, MAX_INTERVAL, MIN_INTERVAL};
pub(crate) use crate::frontend::run::RunArgs;
use crate::frontend::{
    alerts::{AlertArgs, Alerts},
    doctor::DoctorArgs,
    log::LogArgs,
    record::{RecordArgs, ReplayArgs},
    serve::ServeArgs,
    tui::TuiOptions,
};
use crate::metal_load::{LoadConfig, SUBCOMMAND};
use crate::snapshot::SourceSet;
use crate::view::{Filter, Group};
use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "macgpu",
    version,
    about = "See how the Apple Silicon GPU is used, and by which processes and apps",
    long_about = "See how the Apple Silicon GPU is used, and by which processes and apps.\n\n\
                  On a terminal, `macgpu` opens an interactive view. Piped, or with --once, \
                  --watch, --count, or --json, it prints a table or JSON instead.\n\n\
                  Everything works without sudo: per-process GPU time from the GPU driver's \
                  per-client counters, per-app GPU time and measured GPU energy from kernel \
                  resource coalitions, GPU clock, residency, and power from IOReport, and GPU \
                  temperature from the SMC.",
    args_conflicts_with_subcommands = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    #[command(flatten)]
    pub view: ViewArgs,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Open the interactive view (the default on a terminal).
    Top(TopArgs),
    /// Stream one aligned line per process (or app) per sample, like `nvidia-smi pmon`.
    Log(LogArgs),
    /// Serve Prometheus metrics over HTTP.
    Serve(ServeArgs),
    /// Record samples to an NDJSON file for later replay.
    Record(RecordArgs),
    /// Replay a recording in the interactive view, or print it.
    Replay(ReplayArgs),
    /// Run a command and report the GPU time, energy, and memory it used.
    Run(RunArgs),
    /// Check every data source on this machine and report what works.
    Doctor(DoctorArgs),
    /// Internal: run a controlled Metal load in this process (used by doctor and tests).
    #[command(name = SUBCOMMAND, hide = true)]
    Load(LoadConfig),
}

/// Options shared by every live-sampling mode.
#[derive(Args, Debug, Clone, Default)]
pub struct SamplingArgs {
    /// Sampling window in seconds. Default 1, or 2 for a single printed table.
    #[arg(short, long, value_name = "SECONDS", value_parser = parse_interval)]
    pub interval: Option<f64>,

    /// Data sources to turn off, comma-separated.
    #[arg(long, value_delimiter = ',', value_name = "SOURCE")]
    pub disable: Vec<SourceName>,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceName {
    /// IOReport GPU residency, clock, power, and limiters.
    Power,
    /// Per-app coalitions (GPU time, energy).
    Apps,
    /// Per-process graphics memory.
    Memory,
    /// SMC GPU temperature.
    Temperature,
    /// Thermal and memory pressure, wired-memory limit.
    System,
}

impl SamplingArgs {
    pub fn interval_or(&self, default_seconds: f64) -> Duration {
        Duration::from_secs_f64(self.interval.unwrap_or(default_seconds))
    }

    pub fn sources(&self) -> SourceSet {
        let mut sources = SourceSet::default();
        for name in &self.disable {
            match name {
                SourceName::Power => sources.ioreport = false,
                SourceName::Apps => sources.coalitions = false,
                SourceName::Memory => sources.memory = false,
                SourceName::Temperature => sources.temperature = false,
                SourceName::System => sources.system = false,
            }
        }
        sources
    }

    pub fn engine_config(&self, default_seconds: f64, include_idle: bool) -> EngineConfig {
        EngineConfig {
            interval: self.interval_or(default_seconds),
            sources: self.sources(),
            options: AnalyzeOptions { include_idle },
        }
    }
}

/// Which rows to show and how to group them.
#[derive(Args, Debug, Clone, Default)]
pub struct SelectionArgs {
    /// Aggregation level.
    #[arg(short, long, value_enum, default_value_t = Group::Process)]
    pub group: Group,

    /// Filter rows: comma-separated substrings of name, path, or PID; prefix `!` to exclude.
    /// Example: "chrome, !helper".
    #[arg(short, long, value_name = "EXPR")]
    pub filter: Option<String>,

    /// Only show these PIDs (repeatable, or comma-separated).
    #[arg(short, long, value_delimiter = ',', value_name = "PID")]
    pub pid: Vec<i32>,
}

impl SelectionArgs {
    pub fn filter(&self) -> Filter {
        Filter::parse(self.filter.as_deref().unwrap_or_default()).with_pids(&self.pid)
    }
}

/// Default mode: table, JSON, or (on a terminal) the interactive view.
#[derive(Args, Debug, Clone, Default)]
pub struct ViewArgs {
    #[command(flatten)]
    pub sampling: SamplingArgs,

    #[command(flatten)]
    pub selection: SelectionArgs,

    /// Print one table and exit, even on a terminal.
    #[arg(short = '1', long)]
    pub once: bool,

    /// Refresh the table continuously until Ctrl-C.
    #[arg(short, long)]
    pub watch: bool,

    /// Number of samples. Defaults to 1, or unlimited with --watch. 0 means unlimited.
    #[arg(short, long)]
    pub count: Option<usize>,

    /// Show only the top N active rows. 0 shows all.
    #[arg(short = 'n', long, default_value_t = 15)]
    pub top: usize,

    /// Include idle processes and add a lifetime column. Ignores --top.
    #[arg(short, long)]
    pub all: bool,

    /// Show full executable paths instead of names.
    #[arg(short, long)]
    pub verbose: bool,

    /// Emit JSON. With --watch or --count, one object per line (NDJSON).
    #[arg(short, long)]
    pub json: bool,

    /// Interactive view only: disable sending signals to processes.
    #[arg(long)]
    pub readonly: bool,

    #[command(flatten)]
    pub alerts: AlertArgs,
}

impl ViewArgs {
    /// True when the user asked for printed output rather than the interactive view.
    pub fn wants_printed_output(&self) -> bool {
        self.once || self.watch || self.count.is_some() || self.json || self.alerts.exit_on_alert
    }

    pub fn tui_options(&self) -> Result<TuiOptions> {
        tui_options(
            self.selection.group,
            self.selection.filter(),
            self.all,
            self.readonly,
            &self.alerts,
        )
    }
}

/// Interactive view options.
#[derive(Args, Debug, Clone, Default)]
pub struct TopArgs {
    #[command(flatten)]
    pub sampling: SamplingArgs,

    #[command(flatten)]
    pub selection: SelectionArgs,

    /// Include idle processes.
    #[arg(short, long)]
    pub all: bool,

    /// Disable sending signals to processes.
    #[arg(long)]
    pub readonly: bool,

    #[command(flatten)]
    pub alerts: AlertArgs,
}

impl TopArgs {
    pub fn tui_options(&self) -> Result<TuiOptions> {
        tui_options(
            self.selection.group,
            self.selection.filter(),
            self.all,
            self.readonly,
            &self.alerts,
        )
    }
}

/// Builds interactive-view options once for every command that opens the TUI.
pub(crate) fn tui_options(
    group: Group,
    filter: Filter,
    include_idle: bool,
    readonly: bool,
    alerts: &AlertArgs,
) -> Result<TuiOptions> {
    Ok(TuiOptions {
        group,
        filter,
        include_idle,
        readonly,
        alerts: Alerts::from_args(alerts)?,
    })
}

pub fn parse_interval(value: &str) -> Result<f64, String> {
    let interval: f64 = value
        .parse()
        .map_err(|_| "interval must be a number".to_string())?;
    let minimum = MIN_INTERVAL.as_secs_f64();
    let maximum = MAX_INTERVAL.as_secs_f64();
    if !interval.is_finite() || interval < minimum {
        return Err(format!("interval must be at least {minimum} seconds"));
    }
    if interval > maximum {
        return Err(format!("interval must be at most {maximum} seconds"));
    }
    Ok(interval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_default_mode_flags() {
        let cli = Cli::try_parse_from([
            "macgpu",
            "--once",
            "-g",
            "app",
            "--disable",
            "power,temperature",
            "-f",
            "chrome",
        ])
        .expect("parse");
        assert!(cli.command.is_none());
        assert!(cli.view.once);
        assert_eq!(cli.view.selection.group, Group::App);
        let sources = cli.view.sampling.sources();
        assert!(!sources.ioreport && !sources.temperature && sources.coalitions);
        assert!(cli.view.wants_printed_output());
    }

    #[test]
    fn rejects_bad_interval() {
        assert!(parse_interval("0.01").is_err());
        assert!(parse_interval("abc").is_err());
        assert_eq!(parse_interval("2"), Ok(2.0));
    }
}
