#!/usr/bin/env python3
"""Run h1 against a local model server and measure every run with macgpu.

  ./lab.py check                      check h1, macgpu, the server, and its context length
  ./lab.py list                       list the experiments and tasks
  ./lab.py run <experiment> [opts]    run an experiment and print its summary
  ./lab.py report <results dir>       print the summary of an earlier run again

See README.md for what each experiment shows. Needs Python 3.11 or newer and
nothing outside the standard library.
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parent
REPO = KIT.parent.parent
RESULTS = KIT / "results"

# A request this close to the server's reported context length reached it.
CONTEXT_LIMIT_FRACTION = 0.95
# h1's system prompt and tools alone take about 3k tokens; an agentic task
# needs room for file contents and tool output on top of that.
SMALL_CONTEXT = 16384


# --- settings ---------------------------------------------------------------


def load_settings():
    settings = tomllib.loads((KIT / "server.toml").read_text())
    local = KIT / "server.local.toml"
    if local.exists():
        settings.update(tomllib.loads(local.read_text()))
    settings["app"] = settings["app"].lower()
    return settings


def resolve_bin(value):
    """A bare name is looked up on PATH; a path is relative to the macgpu repo."""
    value = os.path.expanduser(os.path.expandvars(value))
    if "/" not in value:
        found = shutil.which(value)
        if not found:
            sys.exit(f"lab: {value} is not on PATH")
        return found
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        sys.exit(f"lab: {path} does not exist")
    return str(path.resolve())


def macgpu_bin(settings):
    built = REPO / settings["macgpu"]
    if built.exists():
        return str(built)
    found = shutil.which("macgpu")
    if not found:
        sys.exit(f"lab: no macgpu at {built} or on PATH; run `cargo build --release` in {REPO}")
    return found


# --- model server -----------------------------------------------------------


def http_json(url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def server_root(base_url):
    return base_url.rstrip("/").removesuffix("/v1")


def server_models(base_url):
    try:
        listing = http_json(base_url.rstrip("/") + "/models")
    except (urllib.error.URLError, OSError) as err:
        sys.exit(f"lab: cannot reach the model server at {base_url}: {err}")
    return listing.get("data", [])


def chat_models(base_url):
    ids = [m["id"] for m in server_models(base_url)]
    return [i for i in ids if not re.search(r"embed|rerank", i, re.I)]


def pick_model(settings, base_url):
    if settings.get("model"):
        return settings["model"]
    models = chat_models(base_url)
    if not models:
        sys.exit("lab: the server lists no chat models; set model in server.toml")
    return models[0]


def server_context(base_url, model):
    """The context length the server is actually running the model with, if it says.

    Ollama reports it for loaded models in /api/ps, omlx in /v1/models, and LM
    Studio in /api/v0/models. Call after warmup() so the model is loaded.
    """
    root = server_root(base_url)
    try:
        for m in http_json(root + "/api/ps").get("models", []):
            if model in (m.get("name"), m.get("model")) and m.get("context_length"):
                return int(m["context_length"])
    except (urllib.error.URLError, OSError, ValueError):
        pass
    try:
        for m in server_models(base_url):
            if m.get("id") == model:
                for key in ("context_length", "max_model_len", "loaded_context_length"):
                    if m.get(key):
                        return int(m[key])
    except SystemExit:
        pass
    try:
        m = http_json(f"{root}/api/v0/models/{model}")
        if m.get("loaded_context_length"):
            return int(m["loaded_context_length"])
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return None


def warmup(base_url, model):
    """Load the model so the first measured run does not pay for it."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "stream": False,
    }
    http_json(base_url.rstrip("/") + "/chat/completions", body, timeout=900)


# --- macgpu -----------------------------------------------------------------


class Recorder:
    """`macgpu record` running for the length of one h1 run."""

    def __init__(self, macgpu, path, interval):
        self.path = path
        self.err = open(path.with_suffix(".err"), "w")
        self.proc = subprocess.Popen(
            [macgpu, "record", "-o", str(path), "-i", str(interval)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self.err,
        )
        deadline = time.time() + 15
        while not (path.exists() and path.stat().st_size > 0):
            if self.proc.poll() is not None:
                self.err.close()
                sys.exit(f"lab: macgpu record exited early: {path.with_suffix('.err').read_text()}")
            if time.time() > deadline:
                self.stop()
                sys.exit("lab: macgpu record wrote no sample in 15 s")
            time.sleep(0.05)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.err.close()


def load_samples(path, app):
    """Per-window figures for the server's app rows and the whole device."""
    samples = []
    for line in path.read_text().splitlines():
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue  # an interrupted last line
        window = report["window_seconds"]
        end = report["timestamp_ms"] / 1000
        apps = [
            a
            for a in report.get("apps", [])
            if app in a["name"].lower() or app in (a.get("bundle_path") or "").lower()
        ]
        device = report["device"]
        samples.append(
            {
                "t0": end - window,
                "t1": end,
                "w": window,
                "energy": sum(a.get("energy_j") or 0 for a in apps),
                "gpu_ns": sum(a.get("gpu_ns") or 0 for a in apps),
                "mem": sum(a.get("graphics_memory_bytes") or 0 for a in apps),
                "device_j": (device.get("power_w") or 0) * window,
                "limited": device.get("thermal_limited_pct") or 0,
                "apps": [a["name"] for a in apps],
            }
        )
    return samples


def between(samples, start, end, key):
    """Sum a per-window figure over [start, end], prorating partial windows.

    macgpu scales each window's app energy to that window's device energy, so
    the figure is in time with the GPU work, to within about a second around
    the start and end of a load.
    """
    total = 0.0
    for s in samples:
        lo, hi = max(s["t0"], start), min(s["t1"], end)
        if hi > lo:
            total += s[key] * (hi - lo) / s["w"]
    return total


# --- h1 ---------------------------------------------------------------------


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {v!r} as TOML")


def toml_key(k):
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def toml_dumps(table):
    lines = []

    def emit(t, prefix):
        scalars = [(k, v) for k, v in t.items() if not isinstance(v, dict)]
        if prefix and scalars:
            lines.append(f"\n[{prefix}]")
        lines.extend(f"{toml_key(k)} = {toml_value(v)}" for k, v in scalars)
        for k, v in t.items():
            if isinstance(v, dict):
                emit(v, f"{prefix}.{toml_key(k)}" if prefix else toml_key(k))

    emit(table, "")
    return "\n".join(lines).lstrip() + "\n"


def deep_merge(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def h1_version(h1):
    out = subprocess.run([h1, "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return out.stdout.strip() or out.stderr.strip()


def builtin_prompt(h1, cache_dir):
    """h1's built-in system prompt, exported with `h1 setup --system-prompt`."""
    cached = cache_dir / "builtin-system.md"
    if not cached.exists():
        with tempfile.TemporaryDirectory() as home:
            subprocess.run(
                [h1, "setup", "--system-prompt"],
                env=os.environ | {"HOME": home},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=True,
            )
            shutil.copy(Path(home) / ".h1" / "system.md", cached)
    return cached.read_text()


def system_prompt_file(spec, h1, cache_dir):
    """Path for [engine] system_prompt_path, or None for h1's built-in prompt."""
    if spec in (None, "builtin"):
        return None
    if spec == "env-last":
        # Move the per-day and per-directory lines to the end, so everything
        # before them is the same for every project and every day.
        path = cache_dir / "env-last-system.md"
        if not path.exists():
            lines = builtin_prompt(h1, cache_dir).splitlines()
            env = [line for line in lines if line.startswith(("Current date:", "Current working directory:"))]
            rest = [line for line in lines if line not in env]
            path.write_text("\n".join(rest + [""] + env) + "\n")
        return path
    path = KIT / spec
    if not path.exists():
        sys.exit(f"lab: system prompt {path} does not exist")
    return path


# --- experiments and tasks --------------------------------------------------


class Task:
    def __init__(self, name):
        self.name = name
        self.dir = KIT / "tasks" / name
        if not (self.dir / "task.toml").exists():
            sys.exit(f"lab: no task named {name} (tasks/{name}/task.toml)")
        spec = tomllib.loads((self.dir / "task.toml").read_text())
        self.description = spec.get("description", "")
        self.prompt = spec["prompt"]
        self.check = spec["check"]
        self.stdin = spec.get("stdin")
        self.timeout = spec.get("timeout", 900)


def load_experiment(name):
    path = KIT / "experiments" / f"{name}.toml"
    if not path.exists():
        names = ", ".join(sorted(p.stem for p in (KIT / "experiments").glob("*.toml")))
        sys.exit(f"lab: no experiment named {name}; choose from {names}")
    return tomllib.loads(path.read_text())


def expand_variants(spec, args, base_url):
    if spec.get("variants_from") == "models":
        models = args.model or chat_models(base_url)
        return [{"name": m, "model": m} for m in models]
    variants = spec.get("variant", [])
    if not variants:
        sys.exit("lab: the experiment has no [[variant]] tables")
    return variants


def schedule(variants, tasks, repeats, order):
    """Interleaved order spreads thermal drift across variants; blocked order
    keeps a server-side cache warm for one variant at a time."""
    if order == "blocked":
        return [(v, t, r) for v in variants for r in range(repeats) for t in tasks]
    return [(v, t, r) for r in range(repeats) for v in variants for t in tasks]


# --- one run ----------------------------------------------------------------


def load_trips(session_dir):
    """One entry per model round trip, from h1's flight recorder."""
    trips, compactions, start = [], 0, None
    for path in sorted((session_dir / "events").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type", "")
            if kind == "turn_start":
                start = event["ts_ms"] / 1000
            elif kind == "turn_end" and start is not None:
                usage = event.get("usage") or {}
                trips.append(
                    {
                        "t0": start,
                        "t1": event["ts_ms"] / 1000,
                        "ttft": (event.get("ttft_ms") or 0) / 1000,
                        "prompt": usage.get("prompt_tokens") or 0,
                        "cached": usage.get("cache_read_tokens") or 0,
                        "completion": usage.get("completion_tokens") or 0,
                    }
                )
                start = None
            elif kind == "compaction_start":
                compactions += 1
    return trips, compactions


def run_one(ctx, index, variant, task, rep):
    name = f"{index:03d}-{variant['name']}-{task.name}-r{rep + 1}"
    run_dir = ctx["out"] / "runs" / re.sub(r"[^A-Za-z0-9._-]", "_", name)
    run_dir.mkdir(parents=True)

    if variant.get("workdir", "fresh") == "stable":
        work = ctx["out"] / "stable-work" / task.name
        shutil.rmtree(work, ignore_errors=True)
    else:
        work = run_dir / "work"
    shutil.copytree(task.dir / "files", work)

    if variant.get("home", "isolated") == "isolated":
        home = run_dir / "home"
        home.mkdir()
    else:
        home = Path.home()

    h1 = resolve_bin(variant.get("h1", ctx["h1"]))
    model = variant.get("model", ctx["model"])
    provider = ctx["provider"]
    config = {
        "engine": {"provider": provider, "model": model},
        "providers": {provider: {"base_url": ctx["base_url"], "kind": "openai"}},
    }
    server_window = ctx["server_ctx"].get(model)
    window = variant.get("context_window")
    if window == "server":
        if not server_window:
            sys.exit("lab: context_window = \"server\" but the server did not report its context length")
        window = server_window
        # h1's default reserve (16384) is larger than many local contexts.
        config["engine"]["compaction"] = {
            "reserve_tokens": max(1024, window // 4),
            "keep_recent_tokens": window // 4,
        }
    if window:
        config["providers"][provider]["context_window"] = int(window)
    prompt_path = system_prompt_file(variant.get("system_prompt"), h1, ctx["out"])
    if prompt_path:
        config["engine"]["system_prompt_path"] = str(prompt_path)
    deep_merge(config, variant.get("config", {}))
    config_path = run_dir / "h1.toml"
    config_path.write_text(toml_dumps(config))

    sessions = run_dir / "sessions"
    cmd = [h1, "--config", str(config_path), "--session-dir", str(sessions), "--yolo"]
    cmd += [str(a) for a in variant.get("args", [])]
    cmd += ["-p", task.prompt]
    env = os.environ | {"HOME": str(home)} | {k: str(v) for k, v in variant.get("env", {}).items()}

    stdout_path = run_dir / "stdout.txt"
    recorder = Recorder(ctx["macgpu"], run_dir / "gpu.ndjson", ctx["interval"])
    proc = None
    try:
        with open(stdout_path, "w") as out, open(run_dir / "stderr.txt", "w") as err:
            stdin = open(work / task.stdin) if task.stdin else subprocess.DEVNULL
            started = time.time()
            proc = subprocess.Popen(cmd, cwd=work, stdin=stdin, stdout=out, stderr=err, env=env)
            try:
                code, timed_out = proc.wait(timeout=task.timeout), False
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                code, timed_out = None, True
            finished = time.time()
            if stdin is not subprocess.DEVNULL:
                stdin.close()
        # Let the late app-energy sample land in the recording.
        time.sleep(max(1.0, 3 * ctx["interval"]))
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        recorder.stop()

    check = subprocess.run(
        task.check,
        shell=True,
        cwd=work,
        env=os.environ | {"H1_OUTPUT": str(stdout_path), "TASK_DIR": str(task.dir)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    (run_dir / "check.txt").write_text(check.stdout + check.stderr)

    trips, compactions = load_trips(sessions)
    samples = load_samples(run_dir / "gpu.ndjson", ctx["app"])
    prefill_j = sum(between(samples, t["t0"], t["t0"] + t["ttft"], "energy") for t in trips)
    decode_j = sum(between(samples, t["t0"] + t["ttft"], t["t1"], "energy") for t in trips)
    decode_s = sum(max(0.0, t["t1"] - t["t0"] - t["ttft"]) for t in trips)
    new_prompt = sum(t["prompt"] - t["cached"] for t in trips)
    completion = sum(t["completion"] for t in trips)
    max_prompt = max((t["prompt"] for t in trips), default=0)
    duration = sum(s["w"] for s in samples)
    near_limit = bool(server_window) and max_prompt >= CONTEXT_LIMIT_FRACTION * server_window
    over_limit = bool(server_window) and max_prompt > server_window

    return {
        "run": run_dir.name,
        "variant": variant["name"],
        "task": task.name,
        "repeat": rep + 1,
        "warmup": rep < ctx["warmup_repeats"],
        "model": model,
        "passed": check.returncode == 0 and not timed_out,
        "exit_code": code,
        "timed_out": timed_out,
        "wall_s": finished - started,
        "round_trips": len(trips),
        "compactions": compactions,
        "prompt_tokens": sum(t["prompt"] for t in trips),
        "cached_tokens": sum(t["cached"] for t in trips),
        "new_prompt_tokens": new_prompt,
        "completion_tokens": completion,
        "max_prompt_tokens": max_prompt,
        "near_server_context": near_limit,
        "over_server_context": over_limit,
        "first_ttft_s": trips[0]["ttft"] if trips else None,
        "prefill_s": sum(t["ttft"] for t in trips),
        "decode_tok_per_s": completion / decode_s if decode_s else None,
        "gpu_s": sum(s["gpu_ns"] for s in samples) / 1e9,
        "gpu_j": sum(s["energy"] for s in samples),
        "device_j": sum(s["device_j"] for s in samples),
        "prefill_j": prefill_j,
        "decode_j": decode_j,
        "prefill_j_per_ktok": prefill_j / new_prompt * 1000 if new_prompt else None,
        "decode_j_per_ktok": decode_j / completion * 1000 if completion else None,
        "peak_gpu_mem_gib": max((s["mem"] for s in samples), default=0) / 2**30,
        "limited_pct": sum(s["limited"] * s["w"] for s in samples) / duration if duration else 0,
        "server_apps": sorted({a for s in samples for a in s["apps"]}),
    }


# --- summary ----------------------------------------------------------------


COLUMNS = [
    ("pass", None),
    ("wall s", "wall_s"),
    ("trips", "round_trips"),
    ("new prompt tok", "new_prompt_tokens"),
    ("cached", None),
    ("out tok", "completion_tokens"),
    ("1st ttft s", "first_ttft_s"),
    ("decode tok/s", "decode_tok_per_s"),
    ("GPU J", "gpu_j"),
    ("vs first", None),
    ("prefill J/ktok", "prefill_j_per_ktok"),
    ("decode J/ktok", "decode_j_per_ktok"),
    ("peak GiB", "peak_gpu_mem_gib"),
    ("limited %", "limited_pct"),
]


def median(runs, key):
    values = [r[key] for r in runs if r.get(key) is not None]
    return statistics.median(values) if values else None


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if v.is_integer() or abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def summarize(out):
    meta = json.loads((out / "meta.json").read_text())
    recorded = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines() if line]
    runs = [r for r in recorded if not r.get("warmup")]
    if not runs:
        return "no measured runs recorded\n"
    warmups = len(recorded) - len(runs)
    lines = [
        f"# {meta['experiment']}",
        "",
        meta.get("description", "").strip(),
        "",
        f"- model server: {meta['provider']} at {meta['base_url']}, app filter `{meta['app']}`",
        "- server context length: "
        + ", ".join(f"{m} {c or 'not reported'}" for m, c in meta["server_ctx"].items()),
        f"- h1: {meta['h1_version']}",
        f"- runs: {len(runs)}, medians per task and variant; `vs first` compares GPU J with the first variant",
        *([f"- {warmups} warm-up runs are left out of the table and kept in runs.csv"] if warmups else []),
        "",
        "| task | variant | runs | " + " | ".join(c for c, _ in COLUMNS) + " |",
        "|" + "---|" * (3 + len(COLUMNS)),
    ]
    tasks = list(dict.fromkeys(r["task"] for r in runs))
    variants = list(dict.fromkeys(r["variant"] for r in runs))
    for task in tasks:
        first_j = None
        for variant in variants:
            group = [r for r in runs if r["task"] == task and r["variant"] == variant]
            if not group:
                continue
            gpu_j = median(group, "gpu_j")
            if first_j is None:
                first_j = gpu_j
            cells = []
            for label, key in COLUMNS:
                if label == "pass":
                    cells.append(f"{sum(r['passed'] for r in group)}/{len(group)}")
                elif label == "cached":
                    total = sum(r["prompt_tokens"] for r in group)
                    cached = sum(r["cached_tokens"] for r in group)
                    cells.append(f"{cached / total:.0%}" if total else "-")
                elif label == "vs first":
                    if first_j and gpu_j is not None and variant != variants[0]:
                        cells.append(f"{(gpu_j - first_j) / first_j:+.0%}")
                    else:
                        cells.append("-")
                else:
                    cells.append(fmt(median(group, key)))
            lines.append(f"| {task} | {variant} | {len(group)} | " + " | ".join(cells) + " |")

    notes = []
    over = [r for r in runs if r.get("over_server_context")]
    near = [r for r in runs if r.get("near_server_context") and not r.get("over_server_context")]
    if over:
        notes.append(
            f"- {len(over)} of {len(runs)} runs had a request with more prompt tokens than the context "
            "length the server reports, so the server does not enforce that length as a hard limit. "
            "Compare pass rates before trusting either figure."
        )
    if near:
        notes.append(
            f"- {len(near)} of {len(runs)} runs had a request within {1 - CONTEXT_LIMIT_FRACTION:.0%} of the "
            "server's reported context length. A server that enforces it truncates such prompts without an error."
        )
    compacted = [r for r in runs if r.get("compactions")]
    if compacted:
        notes.append(
            f"- {len(compacted)} runs compacted their context. Compaction requests count toward GPU J "
            "but not toward trips, tokens, or the prefill and decode split."
        )
    if not any(r["server_apps"] for r in runs):
        notes.append(
            f"- No app matching `{meta['app']}` used the GPU, so GPU figures are zero. "
            "Check `app` in server.toml against `macgpu -g app --once` during a run."
        )
    timed_out = [r for r in runs if r["timed_out"]]
    if timed_out:
        notes.append(f"- {len(timed_out)} runs hit the task timeout.")
    if notes:
        lines += ["", "Notes:", *notes]
    lines += ["", f"Per-run data: {out / 'runs.csv'}", ""]
    return "\n".join(lines)


def write_reports(out):
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines() if line]
    if runs:
        with open(out / "runs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
    text = summarize(out)
    (out / "summary.md").write_text(text)
    return text


# --- commands ---------------------------------------------------------------


def cmd_check(args):
    settings = load_settings()
    ok = True
    h1 = resolve_bin(args.h1 or settings["h1"])
    print(f"h1       {h1} ({h1_version(h1)})")
    macgpu = macgpu_bin(settings)
    print(f"macgpu   {macgpu}")
    base_url = settings["base_url"]
    models = chat_models(base_url)
    print(f"server   {settings['provider']} at {base_url}: {len(models)} chat models")
    model = args.model[0] if args.model else pick_model(settings, base_url)
    print(f"model    {model}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "warmup.ndjson"
        recorder = Recorder(macgpu, path, 0.25)
        try:
            started = time.time()
            warmup(base_url, model)
            print(f"warmup   {time.time() - started:.1f} s (includes loading the model if it was not loaded)")
            time.sleep(1.0)
        finally:
            recorder.stop()
        samples = load_samples(path, settings["app"])
    apps = sorted({a for s in samples for a in s["apps"]})
    if apps and sum(s["gpu_ns"] for s in samples):
        print(f"gpu      app filter `{settings['app']}` matched {', '.join(apps)}")
    else:
        ok = False
        print(
            f"gpu      app filter `{settings['app']}` matched no GPU work during warmup. "
            "Run `macgpu -g app --once` while the server generates, and set app in server.local.toml."
        )

    window = server_context(base_url, model)
    if window is None:
        print("context  the server did not report its context length")
    elif window < SMALL_CONTEXT:
        ok = False
        print(
            f"context  {window} tokens: too small for agentic tasks. h1's prompt and tools take about 3k,\n"
            "         and a server that enforces this length truncates longer requests without an error.\n"
            "         For Ollama, raise it with the Context length setting in the app, or\n"
            "         OLLAMA_CONTEXT_LENGTH=32768 (launchctl setenv) and a restart. The context-window\n"
            "         experiment shows whether this server enforces it."
        )
    else:
        print(f"context  {window} tokens")
    return 0 if ok else 1


def cmd_list(args):
    print("experiments:")
    for path in sorted((KIT / "experiments").glob("*.toml")):
        spec = tomllib.loads(path.read_text())
        first = spec.get("description", "").strip().splitlines()[:1]
        print(f"  {path.stem:<18} {first[0] if first else ''}")
    print("\ntasks:")
    for path in sorted((KIT / "tasks").glob("*/task.toml")):
        spec = tomllib.loads(path.read_text())
        print(f"  {path.parent.name:<18} {spec.get('description', '')}")
    return 0


def cmd_run(args):
    settings = load_settings()
    spec = load_experiment(args.experiment)
    base_url = settings["base_url"]
    h1 = args.h1 or settings["h1"]
    variants = expand_variants(spec, args, base_url)
    for variant in variants:
        resolve_bin(variant.get("h1", h1))
    model = pick_model(settings, base_url) if spec.get("variants_from") == "models" or not args.model else args.model[0]
    tasks = [Task(t) for t in (args.tasks.split(",") if args.tasks else spec["tasks"])]
    warmup_repeats = spec.get("warmup_repeats", 0)
    repeats = args.repeats or spec.get("repeats", 3)
    plan = schedule(variants, tasks, warmup_repeats + repeats, spec.get("order", "interleaved"))

    if args.dry_run:
        for i, (v, t, r) in enumerate(plan, 1):
            print(f"  {i:3d}  {v['name']:<22} {t.name:<18} r{r + 1}")
        return 0

    out = RESULTS / f"{args.experiment}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    print(f"lab: {len(plan)} runs of {args.experiment} -> {out}")

    server_ctx = {}
    for m in dict.fromkeys(v.get("model", model) for v in variants):
        print(f"lab: warming up {m}")
        warmup(base_url, m)
        server_ctx[m] = server_context(base_url, m)
        if server_ctx[m] and server_ctx[m] < SMALL_CONTEXT:
            print(f"lab: warning: the server runs {m} with a {server_ctx[m]}-token context; see ./lab.py check")

    ctx = {
        "out": out,
        "h1": h1,
        "macgpu": macgpu_bin(settings),
        "provider": settings["provider"],
        "base_url": base_url,
        "model": model,
        "app": settings["app"],
        "interval": settings["interval"],
        "server_ctx": server_ctx,
        "warmup_repeats": warmup_repeats,
    }
    meta = {
        "experiment": args.experiment,
        "description": spec.get("description", ""),
        "provider": settings["provider"],
        "base_url": base_url,
        "model": model,
        "app": settings["app"],
        "server_ctx": server_ctx,
        "h1_version": h1_version(resolve_bin(h1)),
        "variants": variants,
        "tasks": [t.name for t in tasks],
        "repeats": repeats,
        "warmup_repeats": warmup_repeats,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    try:
        with open(out / "runs.jsonl", "a") as log:
            for i, (variant, task, rep) in enumerate(plan, 1):
                if i > 1:
                    time.sleep(settings["settle"])
                result = run_one(ctx, i, variant, task, rep)
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(
                    f"[{i}/{len(plan)}] {variant['name']} {task.name} r{rep + 1}: "
                    f"{'pass' if result['passed'] else 'FAIL'}  {result['wall_s']:.1f} s  "
                    f"{result['round_trips']} trips  {result['new_prompt_tokens']:,} new prompt tok  "
                    f"{result['gpu_j']:.0f} J"
                    + (f"  {result['compactions']} compactions" if result["compactions"] else "")
                    + ("  OVER-SERVER-CONTEXT" if result["over_server_context"] else "")
                )
    except KeyboardInterrupt:
        print("\nlab: interrupted; summarizing the runs that finished")
    print()
    print(write_reports(out))
    return 0


def cmd_report(args):
    out = Path(args.results)
    if not out.is_absolute() and not out.exists():
        out = RESULTS / args.results
    print(write_reports(out))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check h1, macgpu, the server, and its context length")
    check.add_argument("--model", action="append", help="model to check (default: from server.toml)")
    check.add_argument("--h1", help="h1 binary (default: from server.toml)")
    check.set_defaults(func=cmd_check)

    sub.add_parser("list", help="list experiments and tasks").set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="run an experiment")
    run.add_argument("experiment", help="name of a file in experiments/, without .toml")
    run.add_argument("--repeats", type=int, help="runs per variant and task (default: from the experiment)")
    run.add_argument("--tasks", help="comma-separated tasks instead of the experiment's")
    run.add_argument(
        "--model",
        action="append",
        help="model for every variant; repeatable for the models experiment (default: server.toml)",
    )
    run.add_argument("--h1", help="h1 binary for variants that do not name one")
    run.add_argument("--dry-run", action="store_true", help="print the run order and exit")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="print the summary of an earlier run")
    report.add_argument("results", help="results directory, or its name under results/")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
