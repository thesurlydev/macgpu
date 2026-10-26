# macgpu

See how the Apple Silicon GPU is being used, and by which processes and apps.

`macgpu` reads the GPU driver's per-client counters, kernel resource
coalitions, IOReport, and the SMC directly. It needs no sudo, does not spawn
`ioreg` or `powermetrics`, and does not open a Metal device, so it never shows
up as a GPU client itself. A full sample of every source takes about 5 ms of
CPU; the interactive view at a 1 s interval uses about 0.6% of one core.

## Quick start

```sh
macgpu                 # interactive view on a terminal
macgpu --once          # one printed table (2 s window)
macgpu -g app --once   # one row per app instead of per process
macgpu --watch         # refreshing table
macgpu --json          # one JSON report
macgpu doctor          # check which data sources work on this machine
```

On a terminal, bare `macgpu` opens the interactive view. When output is piped,
or with `--once`, `--watch`, `--count`, or `--json`, it prints a table or JSON
instead.

## Requirements

- macOS on Apple Silicon.
- Rust 1.85 or newer to build (edition 2024).

```sh
cargo build --release
```

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
| `DEV%` | GPU ms/s as a share of the window, scaled to the device's active time. |
| `POWER` | App rows: measured GPU energy of the app's coalition, scaled so apps sum to device power. Process rows (`~POWER`): the app's power split by GPU time, an estimate. |
| `GPU MEM` | Graphics memory footprint (the `graphics` ledger). `-` for other users' processes, which need root. |
| `QUEUES` | Metal command queues with GPU time. |
| `LAST` | Time since the last command buffer was submitted. |
| `EXITED` | App rows: GPU time of member processes that exited during the window. |
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
| `q`, `Ctrl-C` | Quit |
| `?`, `F1` | Help, including what every metric means |
| arrows, `PgUp`, `PgDn`, `Home`, `End` | Move the selection |
| `Enter` | Detail for the selected process, app, or the unattributed row; `Esc` goes back |
| `/` | Filter (the `--filter` syntax); `Esc` clears it |
| `s` `.` `>`, `,` `<` | Next, previous sort column |
| `S`, `r` | Reverse the sort |
| `g`, `G` | Next, previous grouping level |
| `t` | Tree view: apps over processes, processes over clients and queues |
| `f`, space | Freeze the display; samples are buffered and applied on resume |
| `z`, `Z` | History span: 1m, 5m, 15m, 30m, 1h |
| `+`, `-` | Sampling interval, 100 ms to 60 s |
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
| `-c, --count <N>` | Number of samples. 0 means unlimited. |
| `-n, --top <N>` | Show the top N active rows. Default 15; 0 shows all. |
| `-a, --all` | Include idle processes and add a lifetime column. |
| `-v, --verbose` | Show full executable paths. |
| `-j, --json` | Emit JSON; one object per line with `--watch` or `--count`. |
| `-g, --group <LEVEL>` | `queue`, `client`, `process`, or `app`. |
| `-f, --filter <EXPR>` | Comma-separated substrings of name, path, or PID; prefix `!` to exclude. Example: `"chrome, !helper"`. |
| `-p, --pid <PID>` | Only these PIDs. |
| `--disable <SOURCE>` | Turn off `power`, `apps`, `memory`, `temperature`, or `system`. |

The JSON schema carries a `schema_version`, every device and row field above,
per-source status (`ok`, `partial`, `stale`, `denied`, `unavailable`,
`disabled`, with a reason), events, and `macgpu`'s own overhead.

## Subcommands

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

Endpoints: `/metrics` (Prometheus text format), `/report.json` (the latest
report), and `/healthz` (200 when samples are fresh, 503 otherwise). Process
series carry `pid`, `process`, `app`, and `coalition` labels; app series carry
`app` and `coalition`. Only processes and apps that used the GPU since the
exporter started are exported, so label cardinality stays bounded.

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
command's stdout is untouched. `macgpu` exits with the command's status and
forwards SIGINT, SIGTERM, SIGHUP, and SIGQUIT to it.

The command runs in the terminal's app coalition (creating a separate
coalition needs privileges), so its energy is the app's measured GPU energy
split by GPU time.

### `doctor`: check the data sources

```sh
macgpu doctor
macgpu doctor --load
```

Reports whether each source exists, works without root, and is updating, then
runs sanity checks. `--load` also runs a short Metal workload in a child
process and checks per-process, per-app, power, memory, frequency, and
temperature readings against it.

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

Every source works without root.

| Source | Provides | Notes |
|---|---|---|
| IOKit registry (`IOAccelerator`) | Per-client and per-queue GPU time, client creator, queue counts, firmware utilization, GPU memory, recovery count | Read natively in about 0.5 ms. |
| Resource coalitions | Per-app in-flight GPU time and GPU energy, responsible process | Only coalitions with GPU activity are re-read each sample. |
| `TASK_VM_INFO` graphics ledger | Per-process graphics memory | Same-user processes only; reads about 160 MiB high for 1-2 s after a submission. |
| IOReport | P-state residency, clock, GPU energy, limiter states, thermal limiting, throttle and issue counters | Private framework; about 1 ms of CPU per sample. |
| SMC | GPU temperature sensors | |
| sysctl and notify | Thermal pressure, memory pressure, GPU wired-memory limit | |

`macgpu doctor` shows the state of each source on the current machine.

## Accuracy and limitations

- **GPU time is apportioned.** Under contention, each process's ms/s is its
  share of GPU busy time, not its own wall time on the GPU. The app
  `in_flight_ms_per_s` figure gives the wall-time view.
- **Exited processes.** A process's per-queue counters disappear when it
  exits, so time it used between the last sample and its exit is not in its
  own row. At app level, the coalition still counts it, and `macgpu` shows it
  as `EXITED` time. Short-lived processes are best measured with `macgpu run`,
  which samples every 100 ms.
- **Very short command buffers** (under about 1 ms) read about 60-100 µs high
  each.
- **App power lags real power** by about one window after a load starts or
  stops, because coalition energy is updated late. Totals over a run match
  IOReport within 0.1%.
- **Graphics memory** of other users' processes needs root.
- **Limiter names** (`FAST_DIE`, `PWR_ZONE`, and others) are undocumented.
  `macgpu` reports time spent in them, not their cause.
- **Not counted:** the media engine (VideoToolbox), the Neural Engine, and the
  display controller are separate from the GPU cores.
- **One GPU.** Accelerators are aggregated.

## Development

```sh
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

The end-to-end tests in `tests/` generate real GPU load through the hidden
`macgpu __load` subcommand and check readings against invariants that hold
even when other work shares the GPU. They take about a minute.
