use super::state::Shared;
use anyhow::{Result, anyhow};
use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::thread;
use std::time::Duration;

pub(super) const WORKERS: usize = 4;
pub(super) const QUEUE: usize = 16;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A fixed set of threads that write responses. A client that stops reading
/// occupies one worker; the accept loop and shutdown never wait on it.
pub(super) struct Responders {
    tx: SyncSender<Job>,
    warned: AtomicBool,
}

impl Responders {
    pub(super) fn new() -> Self {
        let (tx, rx) = mpsc::sync_channel::<Job>(QUEUE);
        let rx = Arc::new(Mutex::new(rx));
        for id in 0..WORKERS {
            let rx = Arc::clone(&rx);
            thread::Builder::new()
                .name(format!("macgpu-http-{id}"))
                .spawn(move || {
                    loop {
                        let job = {
                            let guard = rx.lock().unwrap_or_else(PoisonError::into_inner);
                            guard.recv_timeout(Duration::from_millis(200))
                        };
                        match job {
                            Ok(job) => job(),
                            Err(RecvTimeoutError::Timeout) => continue,
                            Err(RecvTimeoutError::Disconnected) => break,
                        }
                    }
                })
                .expect("spawn response worker");
        }
        Self {
            tx,
            warned: AtomicBool::new(false),
        }
    }

    /// False means the queue is full and the job was dropped.
    pub(super) fn submit(&self, job: impl FnOnce() + Send + 'static) -> bool {
        if self.tx.try_send(Box::new(job)).is_ok() {
            self.warned.store(false, Ordering::Relaxed);
            true
        } else {
            if !self.warned.swap(true, Ordering::Relaxed) {
                eprintln!("macgpu: http: response queue full, dropping request");
            }
            false
        }
    }
}

/// Accepts `HOST:PORT`, `:PORT`, or `PORT`.
pub(super) fn parse_listen(text: &str) -> Result<SocketAddr> {
    let text = text.trim();
    let candidate = if let Ok(port) = text.parse::<u16>() {
        format!("127.0.0.1:{port}")
    } else if let Some(port) = text.strip_prefix(':') {
        format!("127.0.0.1:{port}")
    } else {
        text.to_string()
    };
    candidate
        .to_socket_addrs()
        .ok()
        .and_then(|mut addresses| addresses.next())
        .ok_or_else(|| anyhow!("invalid --listen address {text:?}; expected HOST:PORT or PORT"))
}

pub(super) fn bind_error(address: SocketAddr, error: &io::Error) -> anyhow::Error {
    match error.kind() {
        io::ErrorKind::AddrInUse => anyhow!(
            "port {} on {} is already in use (another macgpu serve?); choose another with --listen {}:PORT",
            address.port(),
            address.ip(),
            address.ip()
        ),
        io::ErrorKind::PermissionDenied => anyhow!(
            "not allowed to listen on {address}; ports below 1024 need root, choose another with --listen"
        ),
        _ => anyhow!("cannot listen on {address}: {error}"),
    }
}

fn plain(status: u16, body: &str) -> (u16, &'static str, Arc<str>) {
    (status, "text/plain; charset=utf-8", Arc::from(body))
}

pub(super) fn select_response(
    shared: &Shared,
    path: &str,
    method: &tiny_http::Method,
    stale_after: Duration,
) -> (u16, &'static str, Arc<str>) {
    if !matches!(method, tiny_http::Method::Get | tiny_http::Method::Head) {
        return (
            405,
            "text/plain; charset=utf-8",
            Arc::from("method not allowed\n"),
        );
    }
    match path {
        "/metrics" => (
            200,
            "text/plain; version=0.0.4; charset=utf-8",
            Arc::clone(&shared.metrics),
        ),
        "/report.json" => match &shared.report_json {
            Some(json) => (200, "application/json", Arc::clone(json)),
            None if shared.last_sample.is_some() => (
                503,
                "text/plain; charset=utf-8",
                Arc::from("report unavailable: JSON encoding failed\n"),
            ),
            None => (
                503,
                "text/plain; charset=utf-8",
                Arc::from("no sample yet\n"),
            ),
        },
        "/healthz" => {
            let (status, text) = match shared.last_sample {
                None => (503, "no sample yet\n".to_string()),
                Some(at) => {
                    let age = at.elapsed().as_secs_f64();
                    if age > stale_after.as_secs_f64() {
                        (503, format!("stale: last sample {age:.1} s ago\n"))
                    } else {
                        (200, "ok\n".to_string())
                    }
                }
            };
            (status, "text/plain; charset=utf-8", Arc::from(text))
        }
        "/" => plain(200, "macgpu exporter\n/metrics\n/report.json\n/healthz\n"),
        _ => plain(404, "not found\n"),
    }
}

/// Hands a request to the workers, answering 503 here when the queue is full.
pub(super) fn dispatch(
    responders: &Responders,
    request: tiny_http::Request,
    shared: Arc<RwLock<Shared>>,
    stale_after: Duration,
) {
    // The request stays reachable after a refused submit, so it can be
    // answered here instead of being dropped.
    let slot = Arc::new(Mutex::new(Some(request)));
    let job_slot = Arc::clone(&slot);
    let accepted = responders.submit(move || {
        let request = job_slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(request) = request {
            respond(request, &shared, stale_after);
        }
    });
    if !accepted {
        let request = slot.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(request) = request {
            reject_busy(request);
        }
    }
}

/// Answers a request the worker queue had no room for.
fn reject_busy(request: tiny_http::Request) {
    let response = tiny_http::Response::from_string("server busy\n").with_status_code(503);
    let _ = request.respond(response);
}

pub(super) fn respond(request: tiny_http::Request, shared: &RwLock<Shared>, stale_after: Duration) {
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_string();
    let method = request.method().clone();
    let shared = shared.read().unwrap_or_else(PoisonError::into_inner);
    let (status, content_type, body) = select_response(&shared, &path, &method, stale_after);
    drop(shared);
    let header = tiny_http::Header::from_bytes("Content-Type", content_type)
        .expect("valid content type header");
    let response = tiny_http::Response::from_string(&*body)
        .with_status_code(status)
        .with_header(header);
    // A client that went away is not an error worth reporting.
    let _ = request.respond(response);
}
