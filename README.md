# macgpu

Apple Silicon GPU monitor with per-process and per-app usage, power attribution, memory, thermals, and Prometheus export.

## Why macgpu?

macOS's Activity Monitor provides a basic GPU usage percentage, but it lacks the depth required for performance tuning and power analysis. `macgpu` fills this gap by providing:

- **Detailed Power & Thermals**: Track actual GPU wattage and die temperature in real-time.
- **App-Level Attribution**: Group GPU usage by resource coalitions, so you see the total impact of an app (including its helpers) rather than a fragmented list of processes.
- **Developer Tooling**: Measure the GPU impact of specific commands with `run`, record sessions for later analysis, and export metrics to Prometheus.
- **Architectural Insight**: See how long the GPU spent under power and thermal limiter states, and the breakdown of firmware utilization (renderer vs. tiler).
- **Zero Interference**: The sampler does not open a Metal device, so it does not appear as a GPU client and skew the results.


## Use Cases

`macgpu` is useful for a variety of scenarios where standard monitors fall short:

- **Performance Profiling**: Use `macgpu run` to get a detailed GPU time and energy estimate for a specific script or binary, which suits benchmarking AI models or rendering pipelines. Energy is apportioned by GPU time, not metered per process.
- **Battery Life Optimization**: Identify "hidden" GPU drains by grouping usage at the app level, helping you find which background process is killing your laptop's endurance.
- **Thermal Analysis**: Correlate spikes in GPU power with die temperature and limiter states to see whether your hardware spends heavy loads under power or thermal limits.
- **Automated Regressions**: Run `macgpu run --json` on a self-hosted Apple Silicon machine with real GPU access to track GPU time and energy across commits. Hosted macOS CI runners are virtual machines with no GPU power, temperature, or client data, and energy is apportioned, so expect noise and compare against a baseline.
- **Infrastructure Monitoring**: Use `macgpu serve` to feed real-time GPU metrics into a Prometheus/Grafana dashboard for monitoring a remote Mac workstation.
- **Metal App Debugging**: Drill down to the `queue` level to see how your application's command queues are distributed across the GPU.

`macgpu` reads the GPU driver's per-client counters, kernel resource
coalitions, IOReport, and the SMC directly. The sampling path needs no sudo,
spawns no subprocesses, and does not open a Metal device, so it never shows up
as a GPU client itself. `macgpu doctor --load` is the one exception: with sudo
credentials already cached, it also runs `sudo -n powermetrics` as a
cross-check, and never prompts for a password. Measured on an M4 Max, a full
sample of every source takes about 5 ms of CPU; the interactive view at a 1 s interval uses about 0.6%
of one core.

## Quick start

```sh
macgpu                 # interactive view on a terminal
macgpu --once          # one printed table (2 s window)
macgpu -g app --once   # one row per app instead of per process
macgpu --watch         # refreshing table
macgpu --json          # one JSON report
macgpu doctor          # check which data sources work on this machine
```

On a terminal, bare `macgpu` opens the interactive view. When output or input
is piped, or with `--once`, `--watch`, `--count`, `--json`, or
`--exit-on-alert`, it prints a table or JSON instead.

## Install

```sh
cargo install --git https://github.com/thesurlydev/macgpu
```

To build from a checkout:

```sh
cargo build --release   # the binary is target/release/macgpu
mise run install        # or: build it and install it to ~/.local/bin
```

## Requirements

- macOS on Apple Silicon only. Tested on an M4 Max with macOS 27; macOS 26 and
  other chips are not verified.
- The readers use private IOReport, SMC, and kernel coalition interfaces that
  Apple does not document, so an OS or chip change can affect them.
- Rust 1.88 or newer to build (edition 2024).

## What it shows

```text
window 2.01s  Apple M4 Max (40 cores)  active 67.4%  1472 MHz (normalized 62.9%)  26.5 W  99 C (max 106 C)
attributed 653.3 ms/s across 3 active of 74 GPU processes  2 active apps  firmware device 73% renderer 12% tiler 2%
gpu memory 2.8 GiB in use / 7.7 GiB allocated  limiter FAST_DIE 32%  thermal nominal  memory pressure normal  macgpu cpu 0.3%
sources  memory partial: 5 of 74 GPU processes belong to other users; reading them needs root

    PID  PROCESS                           GPU ms/s  SHARE   DEV%   ~POWER   GPU MEM QUEUES   LAST
  89538  macgpu                               583.0  89.2%   60.1   22.7 W   162 MiB      1    now
    431  WindowServer                          59.1   9.1%    6.1    2.6 W         -      3    now
  56132  ghostty                               11.2   1.7%    1.2   435 mW   372 MiB      5    now
      -  (exited or unattributed)              20.4      -      -        -         -      -      -

GPU MEM is - for other users' processes; run as root to see them.
```

Here the top row is `macgpu __load`, the built-in Metal test workload, running
in another terminal.

Device header:

| Field | Meaning | Source |
|---|---|---|
| `active` | Share of the window the GPU was powered and running (P-state residency). | IOReport `GPUPH` |
| `MHz` | Residency-weighted average clock while active. | IOReport, P-state table from the registry |
| `normalized` | Active time weighted by clock relative to the top clock: how much of the GPU's full capacity was used. | derived |
| `W` | GPU power. | IOReport `Energy Model` |
| `C` | Average and hottest GPU die temperature. | SMC `Tg*` keys |
| `firmware` | Device, renderer, and tiler utilization reported by the GPU firmware, averaged over the window. | `PerformanceStatistics` |
| `gpu memory` | GPU memory in use and allocated. | `PerformanceStatistics` |
| `limiter` | Time the power controller spent in a limiting state. A limiter state does not always mean lower clocks, so it is reported as time under a limiter, not as throttling. | IOReport `PWRCTRL`, `GPU_CLTM` |
| `thermal`, `memory pressure` | System thermal and memory pressure. | notify, sysctl |

Rows:

| Column | Meaning |
|---|---|
| `GPU ms/s` | The process's share of GPU busy time, in milliseconds per second. See "What GPU ms/s means" below. |
| `SHARE` | Percentage of all attributed GPU time in the window. Sums to 100%. |
| `DEV%` | `SHARE` times the GPU's active %, so rows sum to the device's active time. Needs IOReport. |
| `POWER` | App rows: measured GPU energy of the app's coalition, scaled so apps sum to device power. Process rows (`~POWER`): the app's power split by GPU time, an estimate. |
| `GPU MEM` | Graphics memory footprint (the `graphics` ledger). `-` for other users' processes, which need root. |
| `QUEUES` | Metal command queues the process or client holds. App rows show `PROCS` instead: the number of GPU processes in the app. |
| `LAST` | Time since the last command buffer was submitted. |
| `EXITED` | App rows: GPU time of member processes that exited during the window. `-` when overlapping GPU work makes it impossible to separate; the app total is then a lower bound. |
| `LIFETIME` | With `--all`: total GPU time since the client was created. |

### Grouping

`-g` (or `g` in the interactive view) changes the row level:

- `queue`: one row per Metal command queue.
- `client`: one row per GPU connection (`AGXDeviceUserClient`).
- `process`: one row per process (default).
- `app`: one row per app, using the kernel's resource coalitions. Chrome, its
  GPU helper, and its renderers form one row; so do a terminal and the
  commands run inside it. When coalitions are unavailable, apps fall back to
  the outermost `.app` bundle of each process.

### What GPU ms/s means

The driver keeps a cumulative GPU time counter for every Metal command queue
(`accumulatedGPUTime` in the `AppUsage` entries). When several processes run
on the GPU at the same moment, the driver splits the busy time among them. So:

- The rows at one level sum to at most about 1000 ms/s, which is reached when
  the GPU is saturated.
- Two processes that each keep the GPU fully busy show about 500 ms/s each.
- On a quiet GPU, a process's ms/s matches its own command-buffer timing
  within a few percent.

App rows add a second figure, `in_flight_ms_per_s` (JSON and the app detail
in the interactive view). This is the coalition's kernel counter: wall time
with GPU work queued. Apps that overlap each count in full, so the total can
exceed 1000. It also covers processes that exited during the window, whose
per-queue counters disappear with them; `macgpu` uses it to estimate the
`EXITED` time.

Queues are matched across samples by identity, not by position, so a queue
that is released and replaced is never credited with its whole lifetime.

## Interactive view

```sh
macgpu
macgpu top -g app -f chrome
```

The view has header meters, history charts (device activity, normalized
activity, power, clock, temperature, and a stacked chart of who used the GPU),
an events strip, and a sortable table at any grouping level. It adapts to
terminals down to 60x16.

| Key | Action |
|---|---|
| `q`, `Ctrl-C`, `Ctrl-Q` | Quit |
| `?`, `F1` | Help, including what every metric means |
| arrows, `PgUp`, `PgDn`, `Home`, `End` | Move the selection |
| `Enter` | Detail for the selected process, app, or the unattributed row; `Esc` goes back |
| `/` | Filter (the `--filter` syntax); `Enter` applies it and `Esc` cancels the edit. On the main screen, `Esc` clears an applied filter |
| `s` `.` `>`, `,` `<` | Next, previous sort column |
| `S`, `r` | Reverse the sort |
| `g`, `G` | Next, previous grouping level |
| `t` | Tree view: apps over processes, processes over clients and queues |
| `f`, space | Freeze the display; samples are buffered and applied on resume |
| `z`, `Z` | History span: 1m, 5m, 15m, 30m, 1h |
| `+`, `-` (also `=`, `_`) | Sampling interval, 100 ms to 60 s; live view only |
| `a` | Include idle clients |
| `k` | Send a signal to the selected process, after confirmation |

The mouse also works: click selects, double-click opens the detail, the wheel
scrolls, and clicking a column header sorts by it. `--readonly` disables
signals. Before signaling, `macgpu` checks the process start time, so a reused
PID is never signaled.

## Printed output

| Flag | Meaning |
|---|---|
| `-i, --interval <SECONDS>` | Sampling window. Default 1, or 2 for a single printed table. |
| `-1, --once` | Print one table and exit, even on a terminal. |
| `-w, --watch` | Refresh the table until Ctrl-C. |
| `-c, --count <N>` | Number of samples. Defaults to 1, or unlimited with `--watch`. 0 means unlimited. |
| `-n, --top <N>` | Show the top N active rows. Default 15; 0 shows all. |
| `-a, --all` | Include idle processes and add a lifetime column. Ignores `--top`. |
| `-v, --verbose` | Show full executable paths. |
| `-j, --json` | Emit JSON: one pretty-printed object for a single sample, otherwise one compact object per line. |
| `-g, --group <LEVEL>` | `queue`, `client`, `process`, or `app`. |
| `-f, --filter <EXPR>` | Comma-separated substrings of name, path, or PID; prefix `!` to exclude. Example: `"chrome, !helper"`. |
| `-p, --pid <PID>` | Only these PIDs, repeatable or comma-separated. |
| `--disable <SOURCE>` | Turn off `power`, `apps`, `memory`, `temperature`, or `system`. |
| `--readonly` | Disable sending signals in the interactive view. |
| `--alert <RULE>` | Repeatable alert rule; see "Alerts" below. |
| `--exit-on-alert` | Exit with status 3 when an alert fires (printed, `log`, and `replay` modes; `top` rejects it). |
| `-h`, `-V` | Help and version. |

The JSON schema carries a `schema_version`, every device and row field above,
per-source status (`ok`, `partial`, `stale`, `denied`, `unavailable`,
`disabled`, with a reason), events, and `macgpu`'s own overhead.

## Subcommands

Each subcommand has more flags than the shared ones above. `macgpu <subcommand>
--help` is authoritative; the main omissions are:

| Subcommand | Additional flags |
|---|---|
| `log` | `-s, --select <TYPES>` (default `device rows events`), `--csv`, `-o, --output <FILE>`, `--append`, `-c, --count <N>`, `-d, --duration <DURATION>`, `-t, --time <FORMAT>` (default `iso`), `-a, --all`, `-n, --top <N>` (default 0, all), `-v, --verbose`, `--no-header`, `--alert`, `--exit-on-alert` |
| `serve` | `-l, --listen <ADDR>` (default `127.0.0.1:10060`), `-n, --top <N>` (default 0, all), `-a, --all`, `--series <LEVELS>` (default `process app`), `-f, --filter <EXPR>` |
| `record` | `-o, --output <FILE>` (required; `-` for stdout), `-c, --count <N>`, `-d, --duration <DURATION>`, `-a, --all` (about ten times larger) |
| `replay` | `--speed <SPEED>`, `--fast`, `--print`, `-j, --json`, `-g, --group`, `-f, --filter`, `-p, --pid`, `-n, --top` (default 15), `-a, --all`, `-v, --verbose`, `--alert`, `--exit-on-alert` |
| `run` | `-i, --interval <SECONDS>` (default 0.1), `-j, --json` (on stderr), `-o, --output <FILE>`, `-q, --quiet` |
| `doctor` | `--load`, `--load-seconds <SECONDS>` (default 3), `-j, --json` |
| `top` | The shared sampling and selection flags, plus `--all`, `--readonly`, and `--alert` (`--exit-on-alert` is rejected) |

`-i, --interval <SECONDS>` and `--disable <SOURCE>` are accepted by the default
view, `top`, `log`, `serve`, `record`, and `run`. `-g, --group`, `-f, --filter`,
and `-p, --pid` are accepted by the default view, `top`, `log`, and `replay`;
`serve` has `--filter` and `--series` instead, and `record` and `run` have none
of them.

### `log`: one line per row per sample

```sh
macgpu log -i 1 -g app
macgpu log --csv -o gpu.csv --append -d 8h
macgpu log -s device,events
```

Aligned text like `nvidia-smi pmon`, or CSV with `--csv`. `-s` selects device
lines, row lines, and event lines.

### `serve`: Prometheus metrics

```sh
macgpu serve                       # 127.0.0.1:10060
macgpu serve --listen 0.0.0.0:10060 --top 20
```

The endpoint is unauthenticated and exposes process names and executable paths.
Only bind beyond loopback on a trusted network.

Endpoints: `/metrics` (Prometheus text format), `/report.json` (the latest
report), and `/healthz` (200 when samples are fresh, 503 otherwise). Process
series carry `pid`, `process`, `app`, and `coalition` labels; app series carry
`app` and `coalition`. Only processes and apps that used the GPU since the
exporter started are exported; `--all` includes idle ones. `--top` caps how many
series are exported (default: no cap); use it to bound label cardinality.

### `record` and `replay`

```sh
macgpu record -o overnight.ndjson -i 5 -d 10h
macgpu replay overnight.ndjson --speed 60
macgpu replay overnight.ndjson --print -g app
```

A recording is one JSON report per line, flushed after every sample, so an
interrupted recording stays readable. A sample is about 3-6 KB by default.
`replay` opens the interactive view (or prints with `--print` or `--json`)
and accepts `-` for standard input.

### `run`: measure one command

```sh
macgpu run -- python train.py --epochs 1
macgpu run --json -o gpu.json -- ./benchmark
```

Runs the command and reports its GPU time, energy, peak GPU memory, share of
the device, and time under a limiter. The summary goes to stderr; the
command's stdout is untouched. `macgpu` exits with the command's status (128
plus the signal number if a signal killed it; 127 if the command was not found,
126 if it could not be run) and forwards SIGINT, SIGTERM, SIGHUP, and SIGQUIT to
it.

The command runs in the terminal's app coalition (creating a separate
coalition needs privileges), so its energy is the app's measured GPU energy
apportioned by GPU time, or whole-GPU energy apportioned by GPU time when
per-app energy is unavailable.

### `doctor`: check the data sources

```sh
macgpu doctor
macgpu doctor --load
```

Reports whether each source exists, works without root, and is updating, then
runs sanity checks. `--load` also runs a short Metal workload in a child
process (length set by `--load-seconds`, default 3) and checks per-process,
per-app, power, memory, frequency, and temperature readings against it. If
sudo credentials are already cached, it also uses `sudo -n powermetrics` to
cross-check residency, power, and clocks; it never prompts for a password.
`doctor` exits with status 1 when a check fails.

## Alerts

`--alert` works in the printed table, the interactive view, `log`, and
`replay`. A rule fires once its condition has held for its duration and clears
once it has been false for the same duration.

```sh
macgpu log --alert 'power>20W for 10s' --alert temp>95C
macgpu --watch --alert 'process:chrome>500ms/s for 5s' --exit-on-alert
macgpu log --alert recovery
```

Device metrics: `active`, `normalized`, `firmware`, `mhz`, `power`, `temp`,
`temp_max`, `memory`, `busy`, and `limited`. Scoped rules (`process:PATTERN`,
`app:PATTERN`) sum the matching rows; the unit picks the metric (`ms/s`, `%`,
`W`, or bytes). `recovery` fires when the GPU recovers from a hang, and
`issue` when the IOReport issue counters move. `--exit-on-alert` exits with
status 3.

## Data sources

No source needs root to run. Graphics memory is partial without it: other
users' processes cannot be read.

| Source | Provides | Notes |
|---|---|---|
| IOKit registry (`IOAccelerator`) | Per-client and per-queue GPU time, client creator, queue counts, firmware utilization, GPU memory, recovery count | Read natively in about 0.5 ms. |
| Resource coalitions | Per-app in-flight GPU time and GPU energy, responsible process | Only coalitions with GPU activity are re-read each sample. |
| `TASK_VM_INFO` graphics ledger | Per-process graphics memory | Same-user processes only; reads about 160 MiB high for 1-2 s after a submission. |
| IOReport | P-state residency, clock, GPU energy, limiter states, thermal limiting, throttle and issue counters | Private framework; about 1 ms of CPU per sample. |
| SMC | GPU temperature sensors | |
| sysctl and notify | Thermal pressure, memory pressure, GPU wired-memory limit | |

Timings in this table are from an M4 Max. `macgpu doctor` shows the state of each
source on the current machine.

IOReport channel metadata is read from each sample because subscriptions can
drop or reorder channels, and coalition time and energy come from undocumented
`coalition_info_resource_usage` counter slots. These private interfaces are why
a new chip or macOS release may need new source checks.

## Accuracy and limitations

- **GPU time is apportioned.** Under contention, each process's ms/s is its
  share of GPU busy time, not its own wall time on the GPU. The app
  `in_flight_ms_per_s` figure gives the wall-time view.
- **Exited processes.** A process's per-queue counters disappear when it
  exits, so time it used between the last sample and its exit is not in its
  own row. At app level, the coalition still counts it, and `macgpu` shows it
  as `EXITED` time. Where overlapping GPU work makes that time impossible to
  separate, `EXITED` shows `-` and the app total is a lower bound. Short-lived
  processes are best measured with `macgpu run`, which samples every 100 ms.
- **Very short command buffers** (under about 1 ms) read about 60-100 µs high
  each.
- **App power lags real power** by about a second after a load starts or
  stops, because coalition energy is updated late. The kernel updates it only
  every 0.5 s, so windows shorter than 0.9 s split the GPU's power by the apps'
  coalition energy over the last 0.9 s. When IOReport reports GPU power, the
  apps' power is scaled to sum to it, and the coalition energies decide only
  the split. In a test with overlapping loads, summed app energy matched
  IOReport's GPU energy within 0.1% over a run and within 1-12% per steady 1 s
  window; single windows during load ramps can be off by much more.
- **Graphics memory** of other users' processes needs root.
- **GPU temperature gaps.** While the GPU idles, its sensors sometimes stop
  reporting for about a second and read placeholder values near 0 C. Those
  windows leave the temperature out, with the temperature source `stale`.
- **Limiter names** (`FAST_DIE`, `PWR_ZONE`, and others) are undocumented.
  `macgpu` reports time spent in them, not their cause.
- **Not counted:** the media engine (VideoToolbox), the Neural Engine, and the
  display controller are separate from the GPU cores.
- **One GPU.** Accelerators are aggregated.

## Development

```sh
mise run gate   # fmt, clippy, the Linux schema check for macgpu-model, and tests
```

or the pieces by hand:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The workspace has three crates:

| Crate | Contents | Platforms |
|---|---|---|
| `macgpu-model` (`crates/macgpu-model`) | The versioned `Report` schema that `--json`, `record`, and `serve` emit | any |
| `macgpu-core` (`crates/macgpu-core`) | System readers, the sampler, the analysis, and the background `Engine` | macOS |
| `macgpu` (repository root) | The CLI and every front end | macOS |

Another program can link `macgpu-core` to sample in-process, or depend on
`macgpu-model` alone to read reports from `macgpu serve` or a recording on any
platform.

The hardware end-to-end tests in `tests/` generate real GPU load through the
hidden `macgpu __load` subcommand and check readings against invariants that
hold even when other work shares the GPU. On an M4 Max they take about a minute. Tests that
need real GPU hardware or a visible GPU client are marked `#[ignore]`, because
macOS virtual machines have no IOReport power, SMC sensors, or visible GPU
clients. The default test suite covers the CLI paths without
loading the GPU. On an Apple Silicon Mac, run the ignored tests too:

```sh
cargo test --workspace -- --include-ignored
```
