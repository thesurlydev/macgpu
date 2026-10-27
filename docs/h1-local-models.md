# Using macgpu to improve h1 on local models

[h1](https://github.com/thesurlydev/h1) is a coding-agent harness that can run
against local model servers such as omlx, Ollama, and LM Studio. When it does,
every turn spends GPU time and energy on the Mac serving the model. macgpu can
measure that, so changes to h1 can be judged by what they cost on the GPU, not
only by speed or pass rate.

This guide covers four uses, in the order to try them:

1. Measure prompt-cache effectiveness.
2. Add energy and throttling figures to benchmark runs.
3. Report energy per turn in h1 (a proposal).
4. Route on GPU thermal state (deferred).

The first two need no changes to h1. Run them first: if the numbers turn out
interesting, they justify the third.

[experiments/h1](../experiments/h1/README.md) turns this guide into
ready-made experiments: `./lab.py run <experiment>` runs h1 against a local
model server on small coding tasks, records each run with macgpu, and
compares variants of h1's setup by pass rate, tokens, and GPU energy.

## Scope

This only applies when h1 talks to a model server on Apple Silicon. With
OpenRouter or another remote provider, the GPU work happens in someone else's
data center and macgpu has nothing to measure.

## What to measure, and where

**Measure the model server, not h1.** h1 itself does almost no GPU work. The
inference happens in the server process, which is long-running, separate from
h1, and in its own app coalition. So `macgpu run -- h1 -p "..."` reports close
to zero. Instead, run `macgpu record` (or `log`) on the side and filter to the
server's app row.

**Run macgpu on the Mac that hosts the model.** For the SWE-bench Pro setup in
h1's `bench/h1-bench-config.toml`, that is the omlx server at `<internal-ip>`
(<internal-host>), not the machine running Harbor and the task containers.

**Find the server's app row.** While h1 has a turn in flight, run:

```sh
macgpu -g app --once
```

The server appears as one app row: omlx, Ollama, or LM Studio, including any
runner or helper processes. Use a substring of that name as the filter in the
commands below (for example `-f omlx`). Keep the server Mac otherwise idle
while measuring. When other apps use the GPU at the same time, the driver
splits busy time among them, which blurs the numbers.

## Summarizing a recording

`macgpu record` writes one JSON report per line. Each report's `apps` array
carries per-window `gpu_ns`, `energy_j`, and `graphics_memory_bytes` for every
app that used the GPU, and `device.thermal_limited_pct` gives the time under a
power or thermal limiter. This script totals them for one app:

```python
# sum_app.py: total GPU time and energy for one app across a macgpu recording.
# Usage: python3 sum_app.py <app-name-substring> <recording.ndjson>
import json, sys

pattern = sys.argv[1].lower()
gpu_ns = energy_j = peak_mem = 0
limited = window = 0.0
for line in open(sys.argv[2]):
    r = json.loads(line)
    w = r["window_seconds"]
    window += w
    limited += (r["device"].get("thermal_limited_pct") or 0) * w
    for app in r["apps"]:
        if pattern in app["name"].lower():
            gpu_ns += app["gpu_ns"]
            energy_j += app["energy_j"] or 0
            peak_mem = max(peak_mem, app.get("graphics_memory_bytes") or 0)
print(f"samples window {window:.1f} s")
print(f"gpu time       {gpu_ns / 1e9:.2f} s")
print(f"gpu energy     {energy_j:.1f} J")
print(f"peak gpu mem   {peak_mem / 2**30:.2f} GiB")
print(f"thermal limit  {limited / window if window else 0:.1f}% of the time")
```

The sections below refer to it as `sum_app.py`.

## 1. Measure prompt-cache effectiveness

A local server that caches the KV state of the request prefix (omlx does, and
persists it across restarts) only skips prefill when the system prompt and
tools block are byte-identical to an earlier request. A cache hit shows up
directly as less GPU time and fewer joules per request. That makes macgpu a
direct check on any h1 change that touches the request prefix: tool ordering,
system prompt assembly, skill listings, or compaction.

### Worked example: plugin tool ordering

h1's omlx analysis (`docs/analysis/202609020000-omlx-integration-and-efficiency.md`,
item 1) found that WASM plugin tools were sent in `HashMap` order, which
changes in every new process. That reshuffled the tools block on every `-p`
call and every fresh `--resume`, defeating the prefix cache. Commit `42f73e02`
fixed it by sorting the tool definitions. The fix has landed, so this
measurement confirms its effect and serves as a regression check for later
changes.

Setup:

- Build h1 with plugins (`--features full`) at `42f73e02^` (before) and at
  `42f73e02` or `main` (after). A worktree per build keeps them separate.
- Load at least two plugin tools. With one, the order cannot change.
- Configure an omlx provider as in `bench/h1-bench-config.toml`, pointed at
  your server.
- Use a fixed prompt that needs no tools, so each request is almost all
  prefill: for example `h1 -p "reply with the word ok"`.

For each build, on the server Mac:

```sh
macgpu record -o before.ndjson -i 0.5     # Ctrl-C when the runs finish
```

and on the h1 machine, run the prompt once to warm the cache, then ten more
times, each a fresh process:

```sh
for i in $(seq 1 11); do h1 -p "reply with the word ok" >/dev/null; done
```

Then compare:

```sh
python3 sum_app.py omlx before.ndjson
python3 sum_app.py omlx after.ndjson
```

Expected result: with the fix, every run after the first hits the cache, so
GPU time and energy per run fall to roughly the cost of the short answer.
Before the fix, any run whose tool order differs from the cached one pays for
a full prefill again. If the "after" numbers do not drop, something else in
the prefix is still changing between processes.

## 2. Energy and throttling for benchmark runs

The SWE-bench Pro bench (`docs/bench.md`) reports pass rate. Recording the
server alongside it adds what the run cost and whether the hardware held up.

On the server Mac, start a recording before the bench and stop it after:

```sh
macgpu record -o bench-$(date +%Y%m%d-%H%M).ndjson -i 5
```

On the bench machine:

```sh
mise run bench --tasks 10
```

From the recording, `sum_app.py` gives:

- **GPU energy for the run**, and divided by the task count, joules per task.
  This compares models, quantizations, and h1 settings
  (`max_output_bytes`, compaction thresholds, loop-guard limits) on cost as
  well as score.
- **Peak GPU memory**, which shows how close a model and context size come to
  the GPU wired-memory limit.
- **Time under a limiter.** A long run that spends much of its time
  thermally limited slows down partway through, so later tasks are slower
  than earlier ones. Timing comparisons between runs are only fair when both
  ran under similar limiting.

To get warnings while the run is going, run the logger with an alert instead
of, or next to, the recording:

```sh
macgpu log -s device,events -i 5 --alert 'limited>50% for 60s' --alert recovery
```

`recovery` fires if the GPU recovers from a hang, which would also explain a
cluster of failed tasks.

Per-task figures need the task start and end times from the Harbor job, matched
against the `timestamp_ms` of each sample. This is clean at concurrency 1.
<internal-host> serves requests serially, and with `--concurrent 2` the tasks overlap in
time and cannot be separated this way.

For a server that runs all the time, `macgpu serve --listen 0.0.0.0:10060` on
the server Mac exposes the same figures as Prometheus metrics.

## 3. Energy per turn in h1 (proposal)

h1 already ends each model round trip with a cost line (tokens, cost, context
use, time to first token), and `/usage` totals the session's cost. For local
providers the cost is $0, which hides what a turn actually spends. Energy is
the local-model equivalent: joules per turn in the cost line, and tokens per
joule in `/usage`. This also covers the existing idea of showing a rate in the
turn info line (`docs/ideas/202606141841-tokens-per-second-turn-info.md`).

### How h1 would get the number

Each app row in macgpu's JSON carries `lifetime_energy_j`, the coalition's
cumulative GPU energy. So h1 does not need a sampler running during the turn.
It reads the server's `lifetime_energy_j` when the turn starts and when it
ends, and the difference is the turn's energy.

Two ways to read it:

- **Server on the same Mac:** run `macgpu --json -g app -f <server> -i 0.1`
  as a subprocess at each turn boundary.
- **Server on another machine** (such as <internal-host>): run `macgpu serve` on the
  server Mac and read `/report.json` over HTTP.

The HTTP form covers both cases, and h1 can then run on Linux or Windows
against a Mac model server. A provider profile could name the source, for
example:

```toml
[providers.omlx]
base_url = "http://<internal-ip>:8000/v1"
energy_report = "http://<internal-ip>:10060/report.json"
energy_app = "omlx"
```

These keys are hypothetical. h1 has no such setting today.

### Fit with h1's principles

- **Provider-agnostic:** the setting goes on any provider profile and names an
  app, so it works for omlx, Ollama, and LM Studio, and does not tie h1 to
  one vendor.
- **Intent layer:** the energy figure becomes another field on the existing
  usage data, next to tokens and cost. Nothing macOS-specific reaches
  `h1-core`.
- **KISS:** reading JSON over HTTP adds no Apple-specific dependencies to h1.
  Do not link `macgpu-core`: it pulls in Apple system crates and would make
  part of h1 macOS-only. If h1 wants typed reports, `macgpu-model` holds only
  the serde types and builds on every platform.

### Limits to state in the UI

- If another client uses the same server during the turn, the difference
  includes its work. Mark the figure as an estimate (`~`), the way h1 already
  marks estimated prices.
- macgpu's app energy lags real power by about one sampling window, so a
  very short turn can read low. Session totals are accurate.
- This is GPU energy only, not CPU, memory, or whole-system power.

## 4. Route on GPU thermal state (deferred)

h1 can route prompts between cheap, strong, and local models. In principle,
the router could avoid the local model while the server's GPU is thermally
limited (`device.thermal_limited_pct` or `device.system.thermal_pressure`). macgpu does not report battery or AC power, so "avoid local
on battery" would need another source.

This adds a live dependency to the routing decision for a benefit that is not
measured yet. Revisit it only if use 2 shows sustained throttling in practice.

## Accuracy notes

- Under contention, an app's GPU time is its apportioned share of busy time,
  not its own wall time. Keep the server Mac otherwise idle.
- MLX and llama.cpp run on the GPU, so their work is counted. Work on the
  Neural Engine or the media engine is not part of GPU energy.
- App energy totals over a run match IOReport within 0.1%.
- Loading a model from disk is mostly I/O and shows little GPU time. Warm the
  model before measuring, so the first run does not include it.
- `macgpu doctor` checks that every data source works on the server Mac
  before a long recording.
