//! `macgpu` entry point: parses the CLI and dispatches to a front end.

use clap::Parser;
use macgpu::cli::{Cli, Command, ViewArgs};
use macgpu::engine::Engine;
use macgpu::frontend::{self, tui};
use std::io::{self, IsTerminal};
use std::process::ExitCode;

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
        Err(error) => {
            eprintln!("macgpu: {error:#}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<i32> {
    match cli.command {
        None => run_default(&cli.view),
        Some(Command::Top(args)) => {
            if args.alerts.exit_on_alert {
                anyhow::bail!("--exit-on-alert is not supported by top; use a printed mode");
            }
            let engine = Engine::start(args.sampling.engine_config(1.0, args.all));
            let options = args.tui_options()?;
            tui::run(Box::new(engine), options)?;
            Ok(0)
        }
        Some(Command::Log(args)) => frontend::log::run(&args),
        Some(Command::Serve(args)) => frontend::serve::run(&args),
        Some(Command::Record(args)) => frontend::record::record(&args),
        Some(Command::Replay(args)) => frontend::record::replay(&args),
        Some(Command::Run(args)) => frontend::run::run(&args),
        Some(Command::Doctor(args)) => frontend::doctor::run(&args),
        Some(Command::Load(config)) => macgpu::metal_load::run(&config),
    }
}

/// Bare `macgpu`: the interactive view on a terminal, printed output otherwise.
fn run_default(args: &ViewArgs) -> anyhow::Result<i32> {
    let interactive =
        !args.wants_printed_output() && io::stdout().is_terminal() && io::stdin().is_terminal();
    if !interactive {
        return frontend::table::run(args);
    }
    let engine = Engine::start(args.sampling.engine_config(1.0, args.all));
    let options = args.tui_options()?;
    tui::run(Box::new(engine), options)?;
    Ok(0)
}
