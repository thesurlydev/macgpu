# What would make macgpu stand out

Research date: 2026-10-25. Test machine: Apple M4 Max (40-core GPU), macOS 27.0.1 (26A434).

Goal: a fast and accurate way to see how the Mac GPU is being used, and by which processes.

This analysis combines three strands:
- a survey of existing tools and their issue trackers
- reading the source of macmon, mactop, Stats, nvtop and xnu
- hands-on probes on the test machine, run without root, including a controlled Metal compute load

Evidence tags:
- **[measured]**: observed on the test machine.
- **[source]**: read in source code or official docs.
- **[inferred]**: reasoning that still needs validation.

## 1. Summary

The per-process GPU time number macgpu shows today is no longer a differentiator.

- Activity Monitor reads the same `AGXDeviceUserClient` `AppUsage` counters.
- macmon added a per-process GPU column in v0.9.0 (2026-10-07).
- mactop has an experimental, rescaled version.
- At least five small tools also diff `accumulatedGPUTime`.
- Stats is now looking at the same approach (#3710, opened 2026-10-24).

Having the number is not enough to stand out. macgpu can lead on four things no current tool combines:

1. **Accuracy you can trust and explain.** Diff per client, never clamp or rescale, integrate over the whole window, and handle exited processes. Document every metric's semantics, and say when a source is unavailable rather than reporting zero.
2. **Answering "who" at every level.** Show the queue, the client, the process, and the app (via coalitions), and let the user toggle between those levels. No macOS tool groups Chrome helpers, WebKit GPU processes or Electron helpers into their parent app.
3. **Measured per-app GPU power, without sudo.** The kernel's coalition counters carry GPU energy in nanojoules. On the test machine they matched the IOReport whole-GPU energy within 0.7%. No tool shows this today.
4. **A fast, visual TUI with near-zero overhead.** Read the IORegistry in-process: 0.5 ms per sample, against 16 ms for the current `ioreg` subprocess. Add history graphs, a per-process detail view and an app tree view. Keep first-class scriptable output, which macmon explicitly does not offer for per-process data.

Everything in this list works without root on macOS 27, which the probes verified. The one exception is GPU memory for processes owned by other users.

## 2. Landscape

Full comparison and links are in section 10.

| Tool | Per-process GPU | Per-app grouping | GPU power without sudo | Per-process GPU memory | Per-process data in scriptable output | Notes |
|---|---|---|---|---|---|---|
| Activity Monitor | Yes (`AppUsage`) | Partial: the process hierarchy, not GPU aggregation | No | No | No | GUI only. GPU % ignores clock speed. |
| macmon (Rust) | Yes, since v0.9. Sums before diffing and clamps to 100% | No | Yes. Uses AppleCLPC IDs specific to macOS 27, extracted from kernelcaches | No | No: TUI only (issue #21) | The strongest competitor. Active. |
| mactop (Go) | Experimental. Rescaled so the total equals active residency | No | Partly. The README says sudo; mJ counters are frozen on 27 | No | No | 22 layouts. Its overhead draws complaints. |
| asitop and forks | No | No | No (powermetrics) | No | No | asitop is abandoned and crashes on M5. |
| Stats | No (asked repeatedly, #1163, #1364, #3710) | No | n/a | No | No | Menu bar. Device utilization flips between 0 and 100% on M4. |
| btop | No | No | Yes (IOReport) | No | No | Source builds only. |
| nvtop | Lists GPU PIDs, no utilization | No | No | No | No | Its Apple backend is "limited". |
| Instruments | After the fact | No | n/a | Own app only | Export | Heavy. Its trace attribution is noisy. |

Recurring complaints from issue trackers and forums, each of which macgpu can answer:
- **"Which process is using my GPU?"** This is the most requested Stats feature, declined several times.
- **"Why does a monitor need sudo?"**
- **The monitor's own CPU use.** For example, mactop "pegs an entire CPU", and macmon raised the CPU clock just by running.
- **Utilization that ignores frequency.** "67% GPU at 338 MHz and 0.27 W" (btop #1833).
- **Breakage on new macOS versions or chips.** asitop's KeyError on M5. Energy counters frozen on macOS 27.
- **No GPU memory per process.**
- **Broken over SSH or on headless machines.**

## 3. What the platform actually exposes (no root, macOS 27)

| Signal | Source | Root | Status on test machine | Cost |
|---|---|---|---|---|
| Per-client GPU time, one entry per command queue | `AGXDeviceUserClient` `AppUsage[].accumulatedGPUTime` (ns) | No | Works [measured]. For a 3 s window: 3000.2 ms for the load process, matching its own command-buffer timing | 0.5 ms via IOKit vs 15.8-17.5 ms via `ioreg` subprocess [measured] |
| Last submit time per queue | `AppUsage[].lastSubmittedTime` (mach absolute time in ns) | No | Works [measured] | Included above |
| Queue counts | `CommandQueueCount`, `CommandQueueCountMax` | No | On 40 of 85 clients [measured] | Included above |
| Device, renderer and tiler utilization; GPU memory in use and allocated | `PerformanceStatistics` on `IOAccelerator` | No | Works, but instantaneous and jumpy on M4 [measured, Stats #2399] | Included above |
| GPU hang recoveries | `PerformanceStatistics` `recoveryCount`, `lastRecoveryTime`; IOReport `Issue Counts` | No | Present, 0 during probes [measured] | Included above |
| GPU model, core count, topology | `model`, `gpu-core-count`, `GPUConfigurationVariable` | No | Works [measured] | Once at startup |
| GPU P-state residency, giving active % and average frequency | IOReport `GPU Stats / GPU Performance States / GPUPH` (`OFF`, `P1`..`P15`) | No | Works and updates continuously [measured] | A few ms per sample; less with a GPU-only subscription [measured, inferred] |
| Whole-GPU power | IOReport `Energy Model / GPU Energy` (nJ, AGX driver) | No | Works: 8 mW idle, 43.9 W at P15 [measured] | Included above |
| Per-app GPU time and GPU energy | `coalition_info_resource_usage` `gpu_time` (slot 8, ns) and `gpu_energy_nj` (slot 41), plus billed-to-me and billed-to-others | No, for every coalition including WindowServer | Works [measured]. Time within 1.5% of `AppUsage`; energy within 0.7% of IOReport. Survives process exit | 2-11 ms for about 650 coalitions [measured] |
| PID to coalition | `proc_pidinfo(pid, PROC_PIDCOALITIONINFO)` | No | Worked for all 1,091 PIDs [measured] | Included above |
| Responsible app for a PID | `responsibility_get_pid_responsible_for_pid` (private) | No | Works. Returns -1 for WindowServer [measured] | Cheap |
| Per-process GPU memory | `task_name_for_pid` + `TASK_VM_INFO` `ledger_tag_graphics_footprint` (also `media_*` and `neural_*`) | Same user only; root for others | Works. A 1 GiB `MTLBuffer` read as 1028 MB [measured] | Cheap |
| Why the GPU is limited | IOReport `PWRCTRL` (`PERF`, `PWR_ZONE`, `CLTMLIMI`, `PPMLIMIT`...), `GPU_CLTM`, `GPU Throttler Counters`, `THM Counters` | No | Works; 100% `PWR_ZONE` under load [measured] | Included in IOReport sample |
| GPU temperature | SMC `Tg*` keys (44 float sensors on M4 Max) | No | Works: average 60 C idle, 93 C under load [measured] | About 10-20 ms for the `Tg*` set [inferred] |
| Thermal pressure | `notify_get_state("com.apple.system.thermalpressurelevel")` | No | Readable [measured] | Cheap |
| GPU wired memory limit (relevant to LLM users) | `sysctl iogpu.wired_limit_mb`, `iogpu.wired_lwm_mb`, `iogpu.dynamic_lwm` | Read without root | Works [measured] | Cheap |

Dead ends, verified:
- `task_gpu_utilisation` and `TASK_POWER_INFO_V2` GPU fields are always 0 on arm64 (xnu compiles them for x86_64 only).
- `proc_pid_rusage` has no GPU field.
- `top` has no GPU stat.
- `powermetrics --show-process-gpu` reports 0, and needs root anyway.
- IOReport `GPU Stats / Temperature` reads 0.
- The HID temperature sensors do not track the GPU.
- The PMGR `Energy Model` mJ channels (`GPU`, `ANE`, `DRAM`, `CPU Energy`) update only every 1-34 s on macOS 27, so short-interval deltas read 0.
- There is no per-process ANE utilization and no per-process media engine counter.

## 4. Issues in the current implementation

These affect accuracy today and should be fixed whatever else is built.

1. **Counter decrease is counted as a full-lifetime delta.**
   - In `src/ioreg.rs`, the `diff` match arm `Some(_) => client.accumulated_ns` fires when a client's summed `accumulatedGPUTime` goes down.
   - That can happen when one of its `AppUsage` queue entries disappears [inferred]. The arm then credits the client's whole remaining lifetime to the window, which can be hours of GPU time.
   - A decrease should contribute 0 from that client, or be diffed per queue entry.
2. **Device utilization is one instantaneous reading taken at the end of the window.**
   - On M4 it flips between 0 and 100% (Stats #2399, #2748).
   - Replace it with GPUPH active residency integrated over the window, or at least average several in-window readings.
3. **Two subprocesses per sample.** `ioreg -l -w0` costs about 16 ms wall and 14.7 ms CPU [measured], and `ps` is spawned every sample as well. This is the observer effect users complain about.
4. **Time from processes that exit inside the window is lost.** Coalition counters keep it (section 5.3).
5. **No protection against PID reuse.** Key processes by PID plus start time.
6. **`--power` depends on sudo and powermetrics.** IOReport now provides the same frequency, residency and power data without root.

## 5. Standout features

The features are ordered by how much they set macgpu apart and how much later work depends on them.

### 5.1 Native, low-overhead sampling core (foundation)

- **Read the GPU registry directly.**
  - Replace the `ioreg` and `ps` subprocesses with an in-process IOKit walk: `IOServiceGetMatchingServices("IOAccelerator")`, then `IORegistryEntryGetChildIterator`.
  - On each client, read only `IOUserClientCreator`, `AppUsage` and `CommandQueueCount` with `IORegistryEntryCreateCFProperty`.
  - Use `IOIteratorIsValid` to detect a registry change mid-walk, and retry.
- **Get process names without `ps`.** Use `proc_name`, falling back to `sysctl KERN_PROC` for other users' PIDs.
- **Call IOReport over FFI.** Declare the roughly 15 `extern "C"` functions by hand, linked with `-lIOReport`, rather than depending on the macmon crate. Subscribe only to the GPU channels.
- **One sampler thread feeding ring buffers.** Each front end (table, JSON, TUI, exporter) reads from them, so the UI stays responsive and the sample rate does not depend on the render rate.
- **Report macgpu's own cost.** Show its CPU time per sample in the header and in JSON. No competitor does this, and it is the answer to the overhead complaints.
- **Target overhead:** under 1 ms CPU per sample for the GPU path, which allows 4-10 Hz sampling. The measured costs support this.

### 5.2 Sudoless GPU power, frequency and residency

- **Header metrics from IOReport, no powermetrics:**
  - `GPUPH` gives active residency and residency-weighted average frequency.
  - `Energy Model / GPU Energy` (nJ) gives watts.
  - This removes the sudo prompt entirely; `--power` becomes on by default.
- **Frequency-normalized utilization.** Report `active% x avg_freq / max_freq` as well as plain active %. btop added this after #1833, and Activity Monitor's GPU % is criticized for lacking it.
- **P-state distribution.** Show a small histogram of time spent in each state, with MHz labels. The frequency table comes from `voltage-states9` on the `pmgr` node, or `perf-states` on `sgx`. On the test machine the decoded table is not monotonic, so the index-to-MHz mapping needs to be checked against powermetrics' residency line before shipping (section 8).

### 5.3 App-level attribution through coalitions

This is the strongest differentiator and is verified end to end.

- **A coalition is one app plus its XPC services and helpers.** For example, Chrome with all its helpers, or a terminal with everything launched from it.
- **Read the counters for every coalition, without root.** `coalition_info_resource_usage` returns:
  - `gpu_time` (ns)
  - `gpu_energy_nj`
  - `gpu_energy_nj_billed_to_me`
  - `gpu_energy_nj_billed_to_others`
  - `ane_mach_time` and `ane_energy_nj`
- **The counters only ever increase, and include exited tasks.** This fixes "exited clients are not attributed" at app level.
- **Name each app.** Use `responsibility_get_pid_responsible_for_pid` (private) or the coalition's leader, and fall back to the bundle name derived from the executable path.
- **Toggle the grouping level**, as intel_gpu_top's `H` key does: queue, client, process, app. Each level has its own sort and totals.
- **Measured per-app GPU watts.** This is unique. Energy Impact in Activity Monitor ignores GPU power entirely (Eclectic Light, 2026-06-29).
- **Show billed energy.** For example, WindowServer compositing charged back to the app that caused it, which explains why an idle-looking app costs power.
- **Reconcile and label the gap.**
  - Coalition time and the sum of its processes' `AppUsage` time differ slightly. WindowServer's gap was about 2% [measured].
  - Show the difference as an "exited or unattributed" row instead of hiding it.

### 5.4 Interactive TUI (the "visual" in the goal)

Use ratatui with crossterm. The layout follows patterns proven in nvtop, nvitop and amdgpu_top.

- **Header meters:**
  - active % (raw and frequency-normalized)
  - average MHz
  - watts
  - temperature
  - limiter state, the `PWRCTRL` reason (for example "power zone" or "thermal")
  - unified memory: in use, allocated, wired limit
  - thermal pressure
- **History panel:**
  - Braille line charts of GPU active %, watts and frequency.
  - Stacked area of the top N apps' GPU share over time, so you can see who used the GPU over the last few minutes, not just right now.
  - Zoomable time scale, as in zenith.
- **Process and app table:**
  - columns for GPU ms/s, share, watts (app level), GPU memory, queues, and an activity state from `lastSubmittedTime` (Active, Recent or Idle)
  - an inline sparkline per row
- **Keys:**
  - `/` filter (macmon syntax: `chrome, !helper`)
  - `s`/`S` or `,`/`.` cycle the sort
  - `t` tree view
  - `g` cycle the grouping level
  - `f` freeze
  - `Enter` opens the detail screen
  - `k` sends a signal, with confirmation; `--readonly` disables it
  - `?` help
  - mouse support
- **Detail screen** for one process or app:
  - its GPU ms/s, watts and memory history
  - per-queue breakdown
  - executable path, bundle ID, coalition members, parent chain
- **Dynamic panels**, like nvtop's encode and decode bars: the throttle, hang and recovery indicators appear only when something is happening.
- **Adaptive layout** (nvitop's auto, full and compact modes) so it works in small tmux panes and over SSH.

### 5.5 An explicit accuracy model

Make accuracy a visible feature, not a footnote.

- **Clear metric definitions**, carried over from the README into `?` help and JSON field docs:
  - **busy ms/s:** per queue, can exceed 1000.
  - **share:** of all attributed time.
  - **device-normalized %:** share multiplied by GPUPH active %, so the rows sum to the device's active time.
  - **frequency-normalized %.**
- **No clamping and no rescaling of raw values.** Normalized values are separate, labeled columns. Contrast macmon (clamps to 100%) and mactop (rescales every row whenever any process changes).
- **Diff per client, keyed on PID plus start time.**
  - A decrease is treated as 0 (see section 4.1).
  - Clients created inside the window are credited from 0.
- **Source health.** Every field is either a value or explicitly unavailable, never a silent 0. This matters on macOS 27, where some IOReport counters are visible but no longer update.
- **Smoothing and window integration.** The table defaults to a 1-2 s window, while graphs can use faster samples with EMA smoothing. All whole-device numbers are integrated over the window, not point samples.

### 5.6 `macgpu doctor`: source validation

- **Check every source.** For each data source:
  - Does it exist?
  - Is it readable without root?
  - Is it updating? Sample it twice, during a short self-generated Metal load if the user agrees.
  - Do the sources agree with each other? For example, coalition energy against IOReport energy, and `AppUsage` time against coalition time.
- **Print a compact report.** Include the machine model, OS build and GPU config.

This turns the macOS 27 breakage pattern (counters frozen, entitlements added) from silent wrong numbers into a clear diagnosis. It also produces bug reports good enough to support new chips and OS releases quickly. No competitor has this.

### 5.7 Per-process GPU memory

- **Add a `GPU MEM` column.** Use `ledger_tag_graphics_footprint` via `task_name_for_pid` + `TASK_VM_INFO`. It is exact for the user's own processes without root (the 1 GiB buffer read as 1028 MB).
  - Show `-` for other users' processes, with a footnote that root fills them in.
  - Optionally show `graphics_nofootprint`, which covers shared surfaces.
- **Show the device-wide context:**
  - `In use system memory` / `Alloc system memory`
  - the `iogpu.wired_limit_mb` setting
  - `MTLDevice.recommendedMaxWorkingSetSize`
  - memory pressure level
- **Why it matters for LLM users.** Running local models with MLX, llama.cpp or Ollama is a large share of people who watch the Mac GPU closely, and "how close am I to the wired limit" is their real question.

### 5.8 Measure a command: `macgpu run -- <cmd>`

Like `time`, but for the GPU:

- Spawn the command and track its process tree.
- On exit, print:
  - total GPU seconds
  - average and peak GPU %
  - GPU energy in joules (from the tree's coalition, or apportioned by `AppUsage` share)
  - peak graphics memory
  - throttle time
- Optionally emit JSON for CI and benchmark harnesses.

This fits the README's stated use case of benchmark harnesses and CI. No Mac tool does it, and nvidia-smi users build it by hand.

Note: a child launched from a terminal shares the terminal's coalition. So per-command energy needs either a fresh coalition, which is not possible unprivileged [inferred], or `AppUsage`-weighted apportioning of whole-GPU energy. Validate which is more accurate.

### 5.9 Recording, export and integration

- **`macgpu log`.** A stable, column-aligned streaming format modeled on `nvidia-smi pmon` and `dmon`: timestamp prefix, chosen metric groups, CSV mode.
- **NDJSON** (exists today): add schema versioning and the new fields.
- **`macgpu serve`.** A Prometheus exporter with `app`, `pid` and `process` labels, for per-process and per-app GPU time, watts and memory. macmon's exporter has no per-process data.
- **`record` and `replay`.** Write samples to a file and replay them in the TUI. Useful for "what was using the GPU overnight" and for attaching to bug reports.
- **Library crate** (exists today). `macgpu-core` exposes the sampler, analysis, and engine so a menu bar app or GUI can reuse them, and `macgpu-model` holds the report schema for readers on any platform.

### 5.10 Events and alerts

- **GPU hang recovery events.** Detect them from `recoveryCount` and `lastRecoveryTime` and the IOReport `Issue Counts`. Show them in the TUI timeline and log them. No tool shows these today.
- **Threshold alerts.** Highlight rows, or exit with a status code in non-interactive mode, when GPU % or watts stay above a threshold for a duration. Useful in scripts ("fail the benchmark if something else used the GPU").

## 6. Non-goals

- **A general system dashboard** (CPU per core, fans, network, disk). macmon, mactop and btop already cover this. macgpu should stay GPU-first and only show context that explains GPU behavior: power, temperature, throttling, memory pressure.
- **Per-process ANE or media engine usage.** No source exists. Coalition `ane_mach_time` and `ane_energy_nj` give per-app ANE figures as a bonus column, but whole-device ANE power is fragile on macOS 27.
- **Per-process GPU memory for other users without root.**
- **Kernel-level command buffer tracing.** That is Instruments' job.

## 7. Suggested order of work

1. **Fix the accuracy issues in section 4** and add PID plus start-time keying.
2. **Build the native IOKit sampler and the IOReport GPU reader** (5.1, 5.2). Drop the `ioreg`, `ps` and powermetrics dependencies; keep powermetrics only as an optional cross-check in `doctor`.
3. **Add the coalition reader and the app grouping level** (5.3), including measured per-app watts and the unattributed row.
4. **Write `macgpu doctor`** (5.6). It validates everything built so far on each chip and OS combination.
5. **Build the TUI** (5.4) on the shared sampler, with history, tree view, detail screen and filter.
6. **Add GPU memory** (5.7) and the throttle, temperature and thermal context.
7. **Add `run`, `log`, `serve`, `record` and `replay`** (5.8, 5.9) and the events (5.10).
8. **Update the README.** Its "Compared with Activity Monitor" section, which currently says macgpu exposes no metric Activity Monitor lacks, will be out of date once 5.2-5.7 land.

## 8. Open questions to validate

- **Frequency mapping.** The GPU frequency table decoded from `perf-states` / `voltage-states9` is not monotonic: 1182, 1182, 1312, 1242, 1380, 1326... Compare against powermetrics' `GPU HW active residency` line, which lists the MHz for each state, before showing MHz.
- **Queue entry lifecycle.** Do `AppUsage` queue entries disappear while their client lives (the trigger for section 4.1)? Test by creating and releasing `MTLCommandQueue`s in one process.
- **The coalition gap.** How large is the gap between coalition `gpu_time` and the coalition's summed `AppUsage` over many windows and workloads? Is it explained by exited clients alone?
- **GPUPH against the firmware figure.** How does GPUPH active % compare with `Device Utilization %`? A clocked but idle GPU likely counts as active in GPUPH.
- **Energy across chips.** Does `Energy Model / GPU Energy` (nJ) keep working without root on M1-M3 and M5, and on macOS 26? It is the AGX driver channel, not the PMGR channel that macOS 27 froze.
- **Temperature sensors.** Which SMC `Tg*` keys exist on other chips? Is the average of the hotter key in each pair the right summary?
- **Billed energy.** Is `gpu_energy_nj_billed_to_me` populated for apps whose content WindowServer composites?
- **Per-command energy.** For `macgpu run`, is there an unprivileged way to put the child in its own coalition, or is apportioning the only option?

### Answers from the implementation (M4 Max, macOS 27, October 2026)

All proposals in section 5 are implemented. Measuring them answered most of the questions above:

- **Frequency mapping.** The table is ordered by voltage, not frequency, and is correct as read. Under a saturating load the residency-weighted clock reaches 1557-1578 MHz against a 1578 MHz table maximum. `doctor` checks that every active P-state has a clock and that the maximum matches.
- **Queue entry lifecycle.** Confirmed: releasing an `MTLCommandQueue` removes its `AppUsage` entry while the client lives, and a new queue takes the lowest free slot. The old index-based diff produced windows 4.2-6.8 times the window length under queue churn. Matching queues by identity never exceeded `window x queues`.
- **AppUsage is apportioned.** This was not anticipated in section 3. When several contexts run at once, the driver splits busy time among them: two concurrent full loads read about 450-490 ms/s each, and the sum over all processes never exceeded the window. Per-process `GPU ms/s` is therefore a share of busy time, and the rows sum to about 1000 at saturation.
- **The coalition gap.** Coalition `gpu_time` is wall time with work in flight, so overlapping apps each count in full (in-flight totals of 2200-2650 ms/s against an apportioned total of 988-1008 ms/s). The gap is overlap plus exited clients, not jitter. macgpu keeps the apportioned figure as the app's `GPU ms/s` and exposes the coalition counter separately as `in_flight_ms_per_s`.
- **Coalition energy.** It does not over-count under overlap: over a run with two concurrent loads, the coalition sum matched IOReport within 0.0% (640.15 J against 640.17 J). It lags by about one second, so single windows at the start or end of a load are off; app power is scaled to IOReport device power each window.
- **Billed energy.** Always 0 on macOS 27, so it is not shown.
- **Per-command energy.** `coalition_create` and `posix_spawnattr_setcoalition_np` return EPERM unprivileged. `launchctl submit` would give the command its own coalition at the cost of no longer being a child; `run` apportions the app's energy by GPU time instead.
- **Accuracy on a quiet GPU.** Per-process GPU time matches the workload's own command-buffer timing within 1-5%. Command buffers shorter than about 1 ms read 60-100 µs high each.

Still open: energy and temperature keys on other chips and macOS versions, and a live powermetrics cross-check (it needs sudo).

## 9. Technical reference

### IOReport

- **Functions.** Link `-lIOReport` and declare by hand:
  - `IOReportCopyChannelsInGroup`, `IOReportMergeChannels`
  - `IOReportCreateSubscription`, `IOReportCreateSamples`, `IOReportCreateSamplesDelta`
  - `IOReportChannelGetGroup`, `IOReportChannelGetSubGroup`, `IOReportChannelGetChannelName`, `IOReportChannelGetUnitLabel`, `IOReportChannelGetFormat`
  - `IOReportSimpleGetIntegerValue`
  - `IOReportStateGetCount`, `IOReportStateGetNameForIndex`, `IOReportStateGetResidency`
- **Read channel metadata from the returned sample,** not the request; subscriptions can drop or reorder channels.
- **Frequency and residency** from `GPU Stats / GPU Performance States / GPUPH` (unit `24Mticks`; states `OFF`, `P1`..`P15`):
  - active = sum(P*) / sum(all)
  - avg_freq = sum(res_n x f_n) / sum(P*)
- **Power** from `Energy Model / GPU Energy` (nJ): W = delta_nJ / 1e9 / dt. Drop negative deltas.
- **Limiter** from `GPU Stats / GPU Power Controller States / PWRCTRL`. Also useful: `GPU_CLTM`, `GPU Throttler Counters`, `THM Counters`, `Issue Counts`, `UMA Pool Memory Usage`.

### Coalitions

- **PID to coalition.** `proc_pidinfo(pid, 20 /* PROC_PIDCOALITIONINFO */, 0, &info, 40)` fills `struct { uint64_t coalition_id[2]; uint64_t reserved[3]; }`. Index 0 is the resource coalition.
- **Usage.** `coalition_info_resource_usage(cid, buf, size)` is exported from libSystem. The kernel copies `min(size, kernel size)`; the live struct has at least 50 u64 slots.
- **Slots (u64 index)** used here, from xnu `osfmk/mach/coalition.h`:

| Slot | Field |
|---|---|
| 0 | `tasks_started` |
| 1 | `tasks_exited` |
| 3 | `cpu_time` |
| 8 | `gpu_time` (ns) |
| 11 | `energy` (nJ) |
| 38 | `ane_mach_time` |
| 39 | `ane_energy_nj` |
| 40 | `phys_footprint` |
| 41 | `gpu_energy_nj` |
| 42 | `gpu_energy_nj_billed_to_me` |
| 43 | `gpu_energy_nj_billed_to_others` |

- **Copy the layout.** The struct is not in the public SDK. Pin the layout with a test that checks slot 8 against `AppUsage` under load.

### Per-process memory

- `task_name_for_pid(mach_task_self(), pid, &port)`, then `task_info(port, TASK_VM_INFO, ...)`.
- Read `ledger_tag_graphics_footprint` and its `_compressed` variant, plus `ledger_tag_graphics_nofootprint`.
- This works for the same uid only.

### Registry

- `IOAccelerator` node:
  - `model`, `gpu-core-count`
  - `GPUConfigurationVariable` (`num_cores`, `num_gps`, `num_mgpus`, `gpu_gen`, `gpu_var`)
  - `PerformanceStatistics`
  - `AGCInfo` (`fLastSubmissionPID`, `fBusyCount`)
- `AGXDeviceUserClient` keys: `IOUserClientCreator` (`"pid N, name"`), `AppUsage` (`API`, `accumulatedGPUTime`, `lastSubmittedTime`), `CommandQueueCount`, `CommandQueueCountMax`. There is no per-client memory key.

### Temperature

- AppleSMC user client, `IOConnectCallStructMethod` selector 2.
- Read the 4-byte float keys starting with `Tg`. They come in pairs about 5.6 C apart.
- Cache the key info, because each key-info lookup costs about 0.25 ms.

### Measured costs (M4 Max)

| Operation | Cost |
|---|---|
| `ioreg` subprocess | 15.8-17.5 ms wall, 14.7 ms CPU |
| IOKit, `PerformanceStatistics` plus all clients | 0.51 ms |
| IOKit, all properties | 1.62 ms (pulls in the 48 KB `IOReportLegend`) |
| IOReport, 678 channels | 5.4 ms per sample, 0.6-2.4 ms per delta |
| Coalition sweep, 1,091 PIDs and about 650 coalitions | 3-13 ms (cache the PID-to-coalition map to cut this) |

## 10. Sources

**Mac tools:**
- [macmon](https://github.com/vladkens/macmon): `src/app/procs.rs`, `src/sources.rs`, `src/clpc.rs`, README notes on macOS 27
- [mactop](https://github.com/metaspartan/mactop)
- [asitop](https://github.com/tlkh/asitop)
- [Stats](https://github.com/exelban/stats)
- [btop](https://github.com/aristocratos/btop)
- [nvtop](https://github.com/Syllo/nvtop)
- [NeoAsitop](https://github.com/op06072/NeoAsitop)
- [socpowerbud](https://github.com/dehydratedpotato/socpowerbud)
- [metop](https://github.com/FanBB2333/metop)
- [macgtop](https://github.com/Acelogic/macgtop)
- [MacUnburden](https://github.com/LMU-AI/MacUnburden)

**UX references:**
- [nvitop](https://github.com/XuehaiPan/nvitop)
- [amdgpu_top](https://github.com/Umio-Yasuno/amdgpu_top)
- [intel_gpu_top](https://manpages.debian.org/testing/intel-gpu-tools/intel_gpu_top.1.en.html)
- [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [gpustat](https://github.com/wookayin/gpustat)
- [zenith](https://github.com/bvaisvil/zenith)

**Issues cited:**
- Stats [#1163](https://github.com/exelban/stats/issues/1163), [#1364](https://github.com/exelban/stats/issues/1364), [#2399](https://github.com/exelban/stats/issues/2399), [#2748](https://github.com/exelban/stats/issues/2748), [#3710](https://github.com/exelban/stats/issues/3710)
- macmon [#21](https://github.com/vladkens/macmon/issues/21), [#69](https://github.com/vladkens/macmon/issues/69), [#76](https://github.com/vladkens/macmon/issues/76)
- btop [#1833](https://github.com/aristocratos/btop/issues/1833)
- asitop [#91](https://github.com/tlkh/asitop/issues/91)
- context-labs/mactop [#8](https://github.com/context-labs/mactop/issues/8), [#65](https://github.com/context-labs/mactop/issues/65)

**Kernel:**
- [xnu-12377.1.9](https://github.com/apple-oss-distributions/xnu/tree/xnu-12377.1.9): `osfmk/mach/coalition.h`, `osfmk/kern/coalition.c`, `osfmk/arm64/machine_routines.c`, `bsd/kern/sys_coalition.c`, `bsd/kern/proc_info.c`

**Articles:**
- [Eclectic Light: what does Activity Monitor measure](https://eclecticlight.co/2026/06/29/what-does-activity-monitor-measure/)
- [Apple GPU trace attribution](https://siddhantkhare.com/writing/apple-gpu-trace-attribution)
