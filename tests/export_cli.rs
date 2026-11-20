//! End-to-end tests of `macgpu log`, `serve`, `record`, `replay`, and `run`:
//! each runs the real binary for a short time and validates its output.

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn macgpu() -> Command {
    Command::new(env!("CARGO_BIN_EXE_macgpu"))
}

fn run(args: &[&str]) -> Output {
    macgpu().args(args).output().expect("run macgpu")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A fresh directory for one test's files.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("export-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Minimal RFC 4180 line parser.
fn csv_cells(line: &str) -> Vec<String> {
    let mut cells = vec![String::new()];
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                chars.next();
                cells.last_mut().unwrap().push('"');
            }
            ('"', _) => quoted = !quoted,
            (',', false) => cells.push(String::new()),
            (c, _) => cells.last_mut().unwrap().push(c),
        }
    }
    cells
}

#[test]
#[ignore = "needs visible GPU clients; not available on virtual machines"]
fn log_text_lines_parse() {
    let output = run(&["log", "-c", "3", "-i", "0.2", "--time", "unix", "-a"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    let mut devices = 0;
    let mut rows = 0;
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let time: f64 = fields[0]
            .parse()
            .unwrap_or_else(|_| panic!("time in {line}"));
        assert!(time > 1.7e9, "{line}");
        match fields[1] {
            "device" => {
                devices += 1;
                assert_eq!(fields.len(), 12, "{line}");
                // BUSY is always known; the rest are numbers or `-`.
                let _: f64 = fields[10].parse().expect("busy");
                for field in &fields[2..] {
                    assert!(
                        *field == "-" || field.parse::<f64>().is_ok(),
                        "{field} in {line}"
                    );
                }
            }
            "process" => {
                rows += 1;
                assert!(fields.len() >= 11, "{line}");
                // BUSY is always known; the rest are numbers or `-`.
                let _: f64 = fields[3].parse().expect("busy");
                for field in &fields[4..10] {
                    assert!(
                        *field == "-" || field.parse::<f64>().is_ok(),
                        "{field} in {line}"
                    );
                }
                // SHARE is unknown only on the exited or unattributed row,
                // which has no PID.
                assert_eq!(fields[2] == "-", fields[4] == "-", "{line}");
            }
            "event" => {}
            other => panic!("unexpected line type {other}: {line}"),
        }
    }
    assert_eq!(devices, 3, "{text}");
    assert!(
        rows >= 3,
        "every sample has GPU processes with --all: {text}"
    );
}

#[test]
#[ignore = "needs visible GPU clients; not available on virtual machines"]
fn log_csv_columns_are_consistent_and_append_skips_header() {
    let dir = scratch("log-csv");
    let path = dir.join("gpu.csv");
    let path_text = path.to_str().unwrap();
    for _ in 0..2 {
        let output = run(&[
            "log", "--csv", "-c", "2", "-i", "0.2", "-a", "-g", "client", "-o", path_text,
            "--append",
        ]);
        assert!(output.status.success(), "{}", stderr(&output));
        assert!(output.stdout.is_empty());
    }
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines = text.lines();
    let header = csv_cells(lines.next().unwrap());
    assert_eq!(header[0], "time");
    let mut devices = 0;
    let mut clients = 0;
    for line in lines {
        let cells = csv_cells(line);
        assert_eq!(cells.len(), header.len(), "{line}");
        assert_ne!(cells[0], "time", "header repeated on append");
        match cells[1].as_str() {
            "device" => devices += 1,
            "client" => {
                clients += 1;
                assert!(cells[3].starts_with("0x"), "{line}");
                cells[2].parse::<i32>().expect("pid");
            }
            "event" => {}
            other => panic!("unexpected type {other}"),
        }
    }
    assert_eq!(devices, 4);
    assert!(clients >= 4);
}

#[test]
fn log_exit_on_alert() {
    let started = Instant::now();
    let output = run(&[
        "log",
        "-c",
        "50",
        "-i",
        "0.2",
        "-s",
        "events",
        "--alert",
        "busy>=0ms/s",
        "--exit-on-alert",
    ]);
    assert_eq!(output.status.code(), Some(3), "{}", stderr(&output));
    assert!(started.elapsed() < Duration::from_secs(5));
    let text = stdout(&output);
    assert!(
        text.contains(" event   alert alert `busy>=0ms/s` firing"),
        "{text}"
    );

    let output = run(&["log", "--alert", "bogus>1"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        stderr(&output).contains("unknown metric"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn log_stops_cleanly_on_sigterm() {
    let mut child = macgpu()
        .args(["log", "-i", "0.2", "-s", "device"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    // Signal only after two samples arrived, so the test does not depend on
    // how quickly the sampler starts on a loaded machine.
    let mut reader = BufReader::new(child.stdout.take().unwrap());
    let mut text = String::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    while text
        .lines()
        .filter(|line| line.contains(" device "))
        .count()
        < 2
    {
        assert!(Instant::now() < deadline, "no samples: {text}");
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap() == 0 {
            panic!("log exited early: {text}");
        }
        text.push_str(&line);
    }
    // SAFETY: `kill` takes plain integers and touches no memory; the pid is that
    // of a child this test spawned and has not yet waited on, so it cannot have
    // been reused.
    unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
    let status = wait_timeout(&mut child, Duration::from_secs(5)).expect("log stopped");
    assert_eq!(status.code(), Some(0));
    reader.read_to_string(&mut text).unwrap();
    assert!(
        text.lines()
            .filter(|line| line.contains(" device "))
            .count()
            >= 2,
        "{text}"
    );
}

/// `macgpu --watch | head` closes the pipe on the reader's side; macgpu must
/// treat the resulting EPIPE as a normal stop and exit 0, not report it as an
/// error the way `log` and `record` already do.
#[test]
fn printed_table_stops_cleanly_on_broken_pipe() {
    for args in [
        vec!["--watch", "-i", "0.2"],
        vec!["--json", "--watch", "-i", "0.2"],
    ] {
        let mut child = macgpu()
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        // Read one sample, then close the read end so the next write fails.
        let mut reader = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).expect("first sample");
        assert!(!line.is_empty(), "no output for {args:?}");
        drop(reader);
        let status = wait_timeout(&mut child, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("macgpu {args:?} did not stop after the pipe closed"));
        assert_eq!(status.code(), Some(0), "macgpu {args:?} exited on EPIPE");
    }
}

/// Waits until `parent` has a child process named `name`.
fn wait_for_child_named(parent: u32, name: &str, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let found = macgpu::sys::process::all_pids().into_iter().any(|pid| {
            macgpu::sys::process::info(pid)
                .is_some_and(|info| info.ppid == parent as i32 && info.name == name)
        });
        if found {
            return;
        }
        thread::sleep(Duration::from_millis(20));
    }
    panic!("{name} never started under {parent}");
}

fn wait_timeout(child: &mut Child, timeout: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().unwrap() {
            return Some(status);
        }
        thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();
    None
}

/// GET `path`; returns (status, headers, body).
fn http_get(port: u16, path: &str) -> (u16, HashMap<String, String>, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    write!(stream, "GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n").unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let (head, body) = response.split_once("\r\n\r\n").expect("http response");
    let mut lines = head.lines();
    let status = lines
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let headers = lines
        .filter_map(|line| line.split_once(": "))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.to_string()))
        .collect();
    (status, headers, body.to_string())
}

/// Stops a server started by this test by its PID.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn start_server(extra: &[&str]) -> (Server, u16) {
    let mut child = macgpu()
        .args(["serve", "--listen", "127.0.0.1:0", "-i", "0.2"])
        .args(extra)
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut reader = BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let port = line
        .split("http://127.0.0.1:")
        .nth(1)
        .and_then(|rest| rest.split('/').next())
        .and_then(|port| port.parse().ok())
        .unwrap_or_else(|| panic!("no port in {line:?}"));
    (Server(child), port)
}

type Sample = (String, BTreeMap<String, String>, f64);

/// Parses Prometheus text format 0.0.4 strictly enough to catch format bugs.
fn parse_metrics(text: &str) -> Vec<Sample> {
    let mut types = HashMap::new();
    let mut samples = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, kind) = rest.split_once(' ').unwrap();
            assert!(matches!(kind, "gauge" | "counter"), "{line}");
            assert!(
                types.insert(name.to_string(), kind.to_string()).is_none(),
                "{line}"
            );
            continue;
        }
        if line.starts_with("# HELP ") {
            continue;
        }
        let (series, value) = line.rsplit_once(' ').expect("sample");
        let value: f64 = value.parse().unwrap_or_else(|_| panic!("{line}"));
        let (name, labels) = series.split_once('{').unwrap_or((series, ""));
        let kind = types
            .get(name)
            .unwrap_or_else(|| panic!("no TYPE for {name}"));
        if kind == "counter" {
            assert!(name.ends_with("_total"), "{name}");
            assert!(value >= 0.0, "{line}");
        }
        let mut parsed = BTreeMap::new();
        for pair in labels
            .trim_end_matches('}')
            .split("\",")
            .filter(|pair| !pair.is_empty())
        {
            let (key, value) = pair.split_once("=\"").unwrap_or_else(|| panic!("{line}"));
            parsed.insert(key.to_string(), value.trim_end_matches('"').to_string());
        }
        samples.push((name.to_string(), parsed, value));
    }
    samples
}

#[test]
#[ignore = "needs visible GPU clients for process series; not available on virtual machines"]
fn serve_exposes_metrics_report_and_health() {
    let (mut server, port) = start_server(&["--all"]);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (status, _, body) = http_get(port, "/healthz");
        if status == 200 {
            assert_eq!(body, "ok\n");
            break;
        }
        assert_eq!(status, 503);
        assert!(Instant::now() < deadline, "never healthy: {body}");
        thread::sleep(Duration::from_millis(100));
    }
    // Wait for the exporter to count samples, rather than guessing at a delay.
    let deadline = Instant::now() + Duration::from_secs(10);
    let (status, headers, body) = loop {
        let (status, headers, body) = http_get(port, "/metrics");
        let samples = parse_metrics(&body);
        if samples
            .iter()
            .any(|(name, _, value)| name == "macgpu_exporter_samples_total" && *value >= 3.0)
        {
            break (status, headers, body);
        }
        assert!(
            Instant::now() < deadline,
            "exporter never counted 3 samples: {body}"
        );
        thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(status, 200);
    assert!(headers["content-type"].starts_with("text/plain; version=0.0.4"));
    let samples = parse_metrics(&body);
    let names: std::collections::HashSet<&str> =
        samples.iter().map(|(name, _, _)| name.as_str()).collect();
    for expected in [
        "macgpu_exporter_samples_total",
        "macgpu_exporter_build_info",
        "macgpu_gpu_attributed_busy_ratio",
        "macgpu_gpu_attributed_seconds_total",
        "macgpu_gpu_processes",
        "macgpu_process_gpu_seconds_total",
        "macgpu_process_gpu_busy_ratio",
        "macgpu_process_queues",
        "macgpu_source_up",
    ] {
        assert!(names.contains(expected), "missing {expected} in\n{body}");
    }
    let samples_total = samples
        .iter()
        .find(|(name, _, _)| name == "macgpu_exporter_samples_total")
        .unwrap()
        .2;
    assert!(samples_total >= 3.0, "{samples_total}");
    let process = samples
        .iter()
        .find(|(name, _, _)| name == "macgpu_process_gpu_seconds_total")
        .unwrap();
    for label in ["pid", "process", "app", "coalition"] {
        assert!(process.1.contains_key(label), "{process:?}");
    }

    let (status, headers, body) = http_get(port, "/report.json");
    assert_eq!(status, 200);
    assert_eq!(headers["content-type"], "application/json");
    let report: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert!(report["processes"].as_array().is_some());

    assert_eq!(http_get(port, "/nope").0, 404);

    // SIGTERM stops it cleanly.
    // SAFETY: `kill` takes plain integers and touches no memory; the pid is that
    // of a child this test spawned and has not yet waited on, so it cannot have
    // been reused.
    unsafe { libc::kill(server.0.id() as i32, libc::SIGTERM) };
    let status = wait_timeout(&mut server.0, Duration::from_secs(5)).expect("stopped");
    assert_eq!(status.code(), Some(0));
}

#[test]
fn serve_reports_port_in_use() {
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = taken.local_addr().unwrap().port();
    let output = run(&["serve", "--listen", &format!("127.0.0.1:{port}")]);
    assert_eq!(output.status.code(), Some(1));
    let message = stderr(&output);
    assert!(message.contains("already in use"), "{message}");
    assert!(message.contains("--listen"), "{message}");
}

#[test]
fn record_then_replay_round_trips() {
    let dir = scratch("record");
    let path = dir.join("gpu.ndjson");
    let path_text = path.to_str().unwrap();
    let output = run(&["record", "-o", path_text, "-c", "3", "-i", "0.2", "--all"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let summary = stderr(&output);
    assert!(summary.contains("recorded 3 samples"), "{summary}");
    assert!(summary.contains("bytes per sample"), "{summary}");

    let recorded: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("report line"))
        .collect();
    assert_eq!(recorded.len(), 3);
    for report in &recorded {
        assert_eq!(report["schema_version"], 1);
        assert!(report["window_seconds"].as_f64().unwrap() > 0.1);
    }

    let output = run(&["replay", path_text, "--json"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let replayed: Vec<serde_json::Value> = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(replayed, recorded);

    // Paced at 2x: three 0.2 s windows take about 0.2 s after the first.
    let started = Instant::now();
    let output = run(&["replay", path_text, "--print", "--speed", "2", "-a"]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(started.elapsed() >= Duration::from_millis(150));
    let text = stdout(&output);
    assert_eq!(text.matches("== ").count(), 3, "{text}");
    assert!(text.contains("GPU ms/s"), "{text}");

    // Replay through stdin.
    let mut child = macgpu()
        .args(["replay", "-", "--json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    // Feed stdin from a thread: the child's stdout can fill its pipe before
    // stdin is fully written, and writing inline would then deadlock.
    let mut stdin = child.stdin.take().unwrap();
    let recording = std::fs::read(&path).unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&recording));
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap().unwrap();
    assert_eq!(stdout(&output).lines().count(), 3);
}

#[test]
fn replay_rejects_future_schema() {
    let dir = scratch("future");
    let path = dir.join("future.ndjson");
    std::fs::write(&path, "{\"schema_version\": 99, \"timestamp_ms\": 1}\n").unwrap();
    let output = run(&["replay", path.to_str().unwrap(), "--print"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("schema version 99"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn run_passes_through_stdio_and_exit_status() {
    let output = run(&["run", "--", "/bin/sh", "-c", "echo hello; exit 7"]);
    assert_eq!(output.status.code(), Some(7));
    assert_eq!(stdout(&output), "hello\n");
    let summary = stderr(&output);
    assert!(summary.contains("exited with status 7"), "{summary}");
    assert!(summary.contains("GPU time"), "{summary}");

    let output = run(&["run", "--", "/no/such/command"]);
    assert_eq!(output.status.code(), Some(127));
}

#[test]
fn run_writes_json_and_forwards_sigterm() {
    let dir = scratch("run");
    let path = dir.join("run.json");
    let mut child = macgpu()
        .args([
            "run",
            "-q",
            "-o",
            path.to_str().unwrap(),
            "--",
            "/bin/sleep",
            "30",
        ])
        .spawn()
        .unwrap();
    // macgpu installs its handlers before spawning the command, so once the
    // command exists the signal is forwarded rather than fatal. No fixed delay
    // after the poll is needed.
    wait_for_child_named(child.id(), "sleep", Duration::from_secs(30));
    // SAFETY: `kill` takes plain integers and touches no memory; the pid is that
    // of a child this test spawned and has not yet waited on, so it cannot have
    // been reused.
    unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
    let status = wait_timeout(&mut child, Duration::from_secs(5)).expect("forwarded SIGTERM");
    assert_eq!(status.code(), Some(128 + libc::SIGTERM));
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(summary["signal"], libc::SIGTERM);
    assert_eq!(summary["command"][0], "/bin/sleep");
    assert!(summary["wall_seconds"].as_f64().unwrap() > 0.0);
    assert_eq!(summary["gpu_seconds"], 0.0);
}

/// Measures a real Metal workload: `macgpu __load` run under `macgpu run`.
#[test]
#[ignore = "needs a real GPU and a visible Metal load; not available on virtual machines"]
fn run_measures_gpu_time_of_a_metal_load() {
    let load = env!("CARGO_BIN_EXE_macgpu");
    let output = run(&[
        "run",
        "--json",
        "--",
        "/bin/sh",
        "-c",
        &format!("'{load}' __load --seconds 1.5; true"),
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stderr(&output);
    let json_start = text.find("{\n").expect("json summary");
    let summary: serde_json::Value = serde_json::from_str(&text[json_start..]).unwrap();
    let gpu_seconds = summary["gpu_seconds"].as_f64().unwrap();
    // Other GPU work (WindowServer, concurrent loads) time-slices the GPU, so
    // only require a clearly nonzero share of the 1.5 s load.
    assert!(gpu_seconds > 0.05, "{summary}");
    assert!(
        summary["peak_busy_ms_per_s"].as_f64().unwrap() > 20.0,
        "{summary}"
    );
    // sh and the load process (sh forks because of the `; true`).
    assert!(
        summary["processes_tracked"].as_u64().unwrap() >= 2,
        "{summary}"
    );
    assert_eq!(summary["gpu_processes"][0]["name"], "macgpu", "{summary}");
}
