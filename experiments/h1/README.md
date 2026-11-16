# h1 local-model lab

Ready-made experiments for running [h1](https://github.com/thesurlydev/h1)
against a local model server and measuring every run with macgpu. Each
experiment compares a few variants of h1's setup on small coding tasks and
reports pass rate, tokens, prefill and decode speed, and the GPU energy the
model server spent. h1 itself does almost no GPU work, so the lab measures
the model server's app row, not h1.

## Quick start

```sh
cargo build --release          # in the macgpu repo root, once
cd experiments/h1
./lab.py check                 # h1, macgpu, the server, and its context length
./lab.py run baseline          # every task with h1's defaults
./lab.py list                  # all experiments and tasks
```

Each run prints one progress line, and the experiment ends with a summary
table. Everything is saved under `results/<experiment>-<timestamp>/`:

| File | Contents |
|---|---|
| `summary.md` | The summary table |
| `runs.csv`, `runs.jsonl` | One row per run with every metric |
| `meta.json` | Server, model, server context length, h1 version, variants |
| `runs/<run>/` | The generated `h1.toml`, h1's stdout and stderr, the macgpu recording, h1's session and event log, and the check output |

`./lab.py report <results dir>` prints a summary again. `./lab.py run <name>
--dry-run` prints the run order without running anything.

## Requirements

- An Apple Silicon Mac running the model server. **Run the lab on that Mac.**
  macgpu measures the GPU of the machine it runs on, so a lab pointed at a
  server on another machine would measure nothing.
- h1 on PATH (or set `h1` in `server.toml`).
- Python 3.11 or newer. The lab uses only the standard library.
- A model server with an OpenAI-compatible API: Ollama, LM Studio, or omlx.

## Pointing it at your server

`server.toml` defaults to Ollama on localhost with the first chat model it
lists. Override any key in `server.local.toml`, which is not tracked:

```toml
# omlx
provider = "omlx"
base_url = "http://localhost:8000/v1"
model = "Qwen3.5-9B-MLX-4bit"
app = "omlx"
```

`app` is a substring of the server's row in `macgpu -g app --once`. `check`
confirms it matches the GPU work of a warm-up request.

## How a run works

Each run starts from a clean slate, so variants differ only in what the
experiment changes:

1. The task's files are copied into a new working directory, or into a fixed
   one for `workdir = "stable"`.
2. The lab writes an h1 config for the variant and passes it with `--config`,
   so `~/.h1/config.toml` and project configs do not apply. `HOME` points at
   an empty directory, so your plugins, skills, and memory under `~/.h1` and
   `~/.claude` stay out unless the variant sets `home = "user"`.
3. `macgpu record` starts, then `h1 --yolo -p "<task prompt>"` runs with
   stdin closed or fed from the task's `stdin` file.
4. After h1 exits, the recorder keeps going for about a second so the late
   app-energy sample lands, then stops.
5. The task's check command decides pass or fail. `h1 -p` exits 0 even when
   the task failed, so its exit status is not used.
6. h1's flight recorder gives the start and end of every model round trip,
   its tokens, cached tokens, and time to first token. The lab lines those up
   with the macgpu samples to split the server's GPU energy into prefill
   (start of the round trip to first token) and decode (the rest).

`--yolo` lets the model run any command without asking, as your user, in the
run's working directory. Use tasks you would let an agent run unattended.

Before the first run, the lab sends each model a one-token request so model
loading is not part of any measurement.

## Experiments

| Experiment | Question | Look at | What it can change in h1 |
|---|---|---|---|
| `baseline` | How does h1 do on every task with its defaults? | pass, GPU J, new prompt tok | A reference to compare later runs against |
| `prefix-cache` | Does the server reuse the prompt prefix across fresh h1 processes, and what does a new working directory cost? | cached, 1st ttft, prefill J | Where `{{date}}` and `{{cwd}}` sit in the system prompt |
| `tool-surface` | What do the built-in tools, and your plugins, skills, and memory, cost a local model per request? | new prompt tok, 1st ttft, GPU J | Defaults for local providers, such as a smaller tool set |
| `tool-output-cap` | Does a smaller cap on tool output save energy, or cost extra round trips? | pass, trips, new prompt tok, GPU J | `[tools] max_output_bytes` defaults for local models, and which part of long output to keep |
| `context-window` | What happens when h1's context window differs from the server's? | pass, CONTEXT-LIMITED notes, compactions in runs.csv | Detecting the server's real context length instead of assuming one |
| `models` | Which local model gives the best pass rate per joule? | pass, decode tok/s, J/ktok | Default local model and routing tiers |
| `h1-builds` | Does a change to h1 help on local models? | every column, against `installed` | Evidence for or against the change |

Run the models experiment with the models to compare, or with none to try
every chat model the server lists:

```sh
./lab.py run models --model qwen3:8b --model qwen3.8:27b-mlx
```

For `h1-builds`, build the change first (`cargo build --release` in a sibling
`h1` checkout). The `dev` variant uses `../h1/target/release/h1`, relative to
the macgpu repo.

Useful options for any experiment:

```sh
./lab.py run tool-output-cap --repeats 5              # more repeats
./lab.py run baseline --tasks fix-bug                 # one task only
./lab.py run baseline --model qwen3:8b                # another model
./lab.py run baseline --h1 ../h1/target/release/h1    # another h1 build
```

## Reading the summary

Values are medians over the runs of each task and variant.

| Column | Meaning |
|---|---|
| pass | Runs whose check passed |
| wall s | Time for the whole `h1 -p` run |
| trips | Model round trips (requests to the server) |
| new prompt tok | Prompt tokens the server had to process, summed over round trips: prompt tokens minus cached tokens. This is the prefill work. |
| cached | Share of prompt tokens the server reported as cached, over all runs of the row |
| out tok | Completion tokens, summed over round trips |
| 1st ttft s | Time to first token of the first round trip, which includes prefilling the system prompt and tools |
| decode tok/s | Completion tokens per second after the first token |
| GPU J | GPU energy of the server's app over the run, from macgpu |
| vs first | GPU J against the first variant of the same task |
| prefill J/ktok | Prefill energy per thousand new prompt tokens |
| decode J/ktok | Decode energy per thousand completion tokens |
| peak GiB | Peak GPU memory of the server's app: model weights plus KV cache |
| limited % | Time the GPU spent under a power or thermal limiter |

Notes under the table flag runs that came within 5% of the server's context
length (the server probably truncated them), runs that timed out, and a
server app filter that matched nothing.

Small tasks have small token counts, so J/ktok figures from them are noisy.
Use more repeats, or the longer tasks, before reading much into a difference
of a few percent. Interleaved order (the default) spreads thermal drift
across variants; keep the Mac otherwise idle while an experiment runs.

## Writing your own

An experiment is a TOML file in `experiments/`:

| Key | Meaning |
|---|---|
| `description` | Shown by `list` and at the top of the summary |
| `tasks` | Task names from `tasks/` |
| `repeats` | Measured runs per task and variant. Default 3. |
| `order` | `interleaved` (default) or `blocked` (all runs of one variant together, which keeps a server-side cache warm) |
| `warmup_repeats` | Extra runs before the measured ones, left out of the summary. Most useful with `blocked`. |
| `variants_from` | `"models"`: one variant per `--model`, instead of `[[variant]]` tables |
| `[[variant]]` | One table per variant, keys below |

| Variant key | Meaning |
|---|---|
| `name` | Required |
| `args` | Extra h1 flags, such as `["--no-tools"]` |
| `config` | A table merged into the generated h1 config, such as `{ tools = { max_output_bytes = 8000 } }` |
| `env` | Extra environment variables for h1 |
| `home` | `isolated` (default) or `user` |
| `workdir` | `fresh` (default) or `stable` |
| `model` | Model id for this variant |
| `h1` | h1 binary for this variant (a bare name is looked up on PATH; a path is relative to the macgpu repo) |
| `context_window` | Tokens h1 assumes, or `"server"` for the length the server reports. `"server"` also scales compaction to a quarter of it. |
| `system_prompt` | `builtin` (default), `env-last` (h1's built-in prompt with the date and directory lines moved to the end), or a file path relative to this directory |

A task is a directory in `tasks/` with a `task.toml` and a `files/` directory
that is copied into the working directory:

| Key | Meaning |
|---|---|
| `description` | Shown by `list` |
| `prompt` | The prompt given to `h1 -p` |
| `check` | Shell command run in the working directory after h1 exits; exit 0 means pass. `$H1_OUTPUT` is h1's stdout and `$TASK_DIR` the task directory, for comparing against the original files. |
| `stdin` | Optional file from `files/` piped to h1, which appends it to the prompt |
| `timeout` | Seconds before the run is killed and counted as failed. Default 900. |

## Findings so far

The first runs on an M4 Max, with Ollama 0.40.11 serving `qwen3.8:27b-mlx`
(MLX runner), h1 0.6.0, on 2026-10-27, measured with the fixed macgpu
described at the end. These are small samples (one to three runs per row),
so they show where to look, not settled numbers.

1. **Prompt size decides the wait for a new session, and your own setup
   doubles it.** In `tool-surface`, h1's defaults send 2,838 prompt tokens
   on the first request. Running with the real `HOME` adds the plugins,
   skills, and memory index under `~/.h1` and `~/.claude`, for 6,986 tokens.
   On runs that missed the server's prompt cache, the first token took
   32-46 s instead of 11-17 s, and the run used 906-1,463 J of GPU energy
   instead of 423-765 J. With `--no-tools` (173 tokens) the first token came
   in about a second, for 37-59 J. Once the prompt is cached, a repeat
   request takes under a second either way, so the cost lands on every new
   session and every change to the prompt. Worth trying in h1: a leaner
   default surface for local providers, such as no imported
   `~/.claude/skills` and fewer plugin tools unless asked for.
2. **The server's reported context length is not a hard limit here.** Ollama
   reports 4096 tokens for this model in `/api/ps`, yet it processed
   requests of up to almost 8,000 prompt tokens, and every task passed.
   Giving h1 that 4096 (`context-window`, variant `server`) makes it compact
   when the conversation grows, which adds summarization requests and gained
   nothing in pass rate. Whether it costs more overall is not settled: with
   one run per row, run-to-run differences in how many round trips the model
   takes are as large as the differences between variants. Run it with
   `--repeats 5` before drawing a conclusion. Either way, h1 should not
   adopt a server-reported length without checking that the server enforces
   it.
3. **A new working directory barely costs anything on this server.** h1 puts
   `{{date}}` and `{{cwd}}` early in its system prompt, so a new directory
   could have forced a full prefill. In `prefix-cache`, Ollama still served
   93-97% of the prompt from its cache. A new directory added about 50
   prompt tokens and 0.3 s to the first token, and no measurable energy.
   Moving those lines to the end (`env-last`) did not help. No h1 change is
   needed for this server. Repeat the experiment on omlx before
   generalizing.
4. **Sustained runs hit the GPU's limiter.** During the agentic tasks the
   GPU spent 80-95% of the time under a power or thermal limiter, and
   decode ran at 35-47 tokens per second, against 56-60 in the short tasks.
   Compare variants only when their `limited %` is similar, and prefer
   interleaved order so drift spreads across variants.

Measurement note: building this lab exposed a macgpu bug. macgpu dropped
part of each app's GPU energy at short sampling intervals: about half at
0.25 s, 30-40% on these Ollama runs even at 0.5 s, and `macgpu run` at its
default 0.1 s reported about a fifth. The kernel credits app energy in steps
about 0.5 s apart, and windows without a step lost their energy. It is fixed
in the working tree alongside this lab. Results recorded with an older
macgpu undercount GPU J; runs recorded with the fix put 90-94% of the
device's GPU energy on the model server.
