//! `macgpu serve`: a Prometheus exporter.
//!
//! The [`Engine`] samples on its own thread; a collector thread turns each
//! report into the exposition text once, and HTTP requests only copy that
//! text, so a scrape never waits for (or triggers) a sample.
//!
//! Counters versus gauges: cumulative GPU time and energy are counters
//! (`_total`), so `rate()` and `increase()` work at any scrape interval. Process
//! counters accumulate from when the exporter first saw the process; app
//! counters are the kernel's coalition counters, which include exited helpers.
//! Busy, share, power, temperature, and memory are gauges for the latest
//! window.

mod http;
mod metrics;
mod state;

#[cfg(test)]
mod tests;

pub use metrics::{ExportOptions, Exporter};

use super::signals;
use crate::cli::SamplingArgs;
use crate::engine::{Engine, clamp_interval};
use crate::view::Filter;
use anyhow::{Context, Result, anyhow};
use clap::{Args, ValueEnum};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

/// Not allocated in the Prometheus default port registry
/// (github.com/prometheus/prometheus/wiki/Default-port-allocations) and not a
/// well-known service port.
pub const DEFAULT_PORT: u16 = 10060;

#[derive(Args, Debug, Clone, Default)]
#[command(after_help = "\
Endpoints:
  /metrics       Prometheus text format
  /report.json   the latest report (the --json schema)
  /healthz       200 when samples are fresh, 503 otherwise

Process series carry pid, process, app, and coalition labels; app series carry app
and coalition. By default only processes and apps that used the GPU since the
exporter started are exported; --top limits them further.

Example scrape config:
  - job_name: macgpu
    static_configs: [{targets: ['127.0.0.1:10060']}]")]
pub struct ServeArgs {
    /// Address to listen on, HOST:PORT or PORT. Defaults to 127.0.0.1. The
    /// endpoint is unauthenticated and exposes process names and executable
    /// paths, so binding beyond loopback should be a deliberate choice. Use
    /// 0.0.0.0:PORT to accept scrapes from other machines.
    #[arg(short, long, default_value_t = format!("127.0.0.1:{DEFAULT_PORT}"), value_name = "ADDR")]
    pub listen: String,

    #[command(flatten)]
    pub sampling: SamplingArgs,

    /// Export at most N process series and N app series, ranked by GPU time in
    /// the latest window, then since start. 0 means no limit.
    #[arg(short = 'n', long, default_value_t = 0)]
    pub top: usize,

    /// Also export processes and apps that have not used the GPU since the
    /// exporter started.
    #[arg(short, long)]
    pub all: bool,

    /// Per-row series to export, comma-separated.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [Series::Process, Series::App],
        value_name = "LEVELS"
    )]
    pub series: Vec<Series>,

    /// Only export processes and apps matching this filter (the --filter syntax).
    #[arg(short, long, value_name = "EXPR")]
    pub filter: Option<String>,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Series {
    Process,
    App,
}

pub fn run(args: &ServeArgs) -> Result<i32> {
    let address = http::parse_listen(&args.listen)?;
    let listener = TcpListener::bind(address).map_err(|error| http::bind_error(address, &error))?;
    let local = listener.local_addr().context("listener address")?;
    let server = tiny_http::Server::from_listener(listener, None)
        .map_err(|error| anyhow!("cannot serve on {local}: {error}"))?;

    let config = args.sampling.engine_config(1.0, true);
    let interval = clamp_interval(config.interval);
    let exporter = Exporter::new(ExportOptions {
        top: args.top,
        all: args.all,
        processes: args.series.contains(&Series::Process),
        apps: args.series.contains(&Series::App),
        filter: Filter::parse(args.filter.as_deref().unwrap_or_default()),
    });
    let shared = Arc::new(RwLock::new(state::Shared::new(exporter.render_empty())));
    let stop = Arc::new(AtomicBool::new(false));

    signals::install(&signals::STOP);
    let collector = {
        let shared = Arc::clone(&shared);
        let stop = Arc::clone(&stop);
        thread::Builder::new()
            .name("macgpu-collector".into())
            .spawn(move || state::collect(Engine::start(config), exporter, &shared, &stop))
            .context("spawn collector thread")?
    };

    eprintln!(
        "macgpu: serving Prometheus metrics on http://{local}/metrics (sampling every {:.1} s); Ctrl-C stops",
        interval.as_secs_f64()
    );
    let stale_after = (interval * 3).max(Duration::from_secs(5));
    let responders = http::Responders::new();
    while !stop.load(Ordering::SeqCst) {
        if signals::take().is_some() {
            break;
        }
        match server.recv_timeout(Duration::from_millis(200)) {
            Ok(Some(request)) => {
                http::dispatch(&responders, request, Arc::clone(&shared), stale_after);
            }
            Ok(None) => {}
            Err(error) => eprintln!("macgpu: http: {error}"),
        }
    }
    stop.store(true, Ordering::SeqCst);
    let _ = collector.join();
    Ok(0)
}
