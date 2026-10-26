# `macgpu` Capability Comparison

While general-purpose monitors like macOS Activity Monitor and `btop` provide a basic overview of GPU utilization, `macgpu` is designed as a professional-grade diagnostic and benchmarking tool for Apple Silicon GPUs. This document outlines the differences in capabilities and highlights why `macgpu` is the superior choice for developers, power users, and system analysts.

## Feature Comparison Matrix

| Feature | Activity Monitor | `btop` | `macgpu` |
| :--- | :---: | :---: | :---: |
| **Overall GPU % Usage** | ✅ | ✅ | ✅ |
| **Per-Process % Usage** | ✅ | ❌ | ✅ |
| **Per-App Aggregated Usage** | ❌ | ❌ | ✅ |
| **GPU Power Consumption (Watts)** | ❌ | ❌ | ✅ |
| **Per-App Power Attribution** | ❌ | ❌ | ✅ |
| **GPU Die Temperature** | ❌ | ❌ | ✅ |
| **Real-time Clock Speeds (MHz)** | ❌ | ❌ | ✅ |
| **Normalized Capacity** | ❌ | ❌ | ✅ |
| **Firmware Utilization (Renderer/Tiler)** | ❌ | ❌ | ✅ |
| **Power Limiter/Throttling State** | ❌ | ❌ | ✅ |
| **Per-Process Graphics Memory** | ✅ (Limited) | ❌ | ✅ (Detailed) |
| **Metal Queue Tracking** | ❌ | ❌ | ✅ |
| **Command-line Benchmarking (`run`)** | ❌ | ❌ | ✅ |
| **Session Recording & Replay** | ❌ | ❌ | ✅ |
| **Prometheus Metrics Exporter** | ❌ | ❌ | ✅ |
| **Custom Alerting System** | ❌ | ❌ | ✅ |
| **Low-Overhead Sampling** | ✅ | ✅ | ✅ |

---

## Why `macgpu` Stands Out

`macgpu` doesn't just repeat what Activity Monitor or `btop` do; it provides an entirely different level of visibility into the Apple Silicon GPU architecture.

### 1. True Power and Thermal Analysis
Activity Monitor tells you if the GPU is "busy," but not how much energy it is consuming or how hot it is getting. `macgpu` provides:
*   **Actual Wattage**: See the exact power draw of the GPU.
*   **Thermal Visibility**: Monitor the GPU die temperature directly from the SMC.
*   **Energy Attribution**: Estimate how much power a specific application is consuming on the GPU, which is critical for battery life optimization.

### 2. Developer-First Tooling
`macgpu` transforms GPU monitoring from a passive observation task into an active development workflow:
*   **`macgpu run`**: Instead of manually watching a graph while running a script, you can wrap your command in `macgpu run` to get a precise summary of total GPU time, peak memory, and energy used.
*   **Recording & Replay**: Capture GPU activity during a bug or a performance spike and replay it later in the interactive view for frame-by-frame analysis.

### 3. Infrastructure and Automation
For those managing fleets of Macs or running long-term experiments:
*   **Prometheus Integration**: The `serve` command allows you to integrate GPU metrics into a professional monitoring stack (like Grafana), enabling long-term trend analysis.
*   **Programmable Alerts**: Set specific triggers (e.g., `power > 20W for 10s`) to automatically notify you or exit a process when certain thresholds are met.

### 4. Low-Level Architectural Insight
`macgpu` exposes "under-the-hood" data that is hidden from the average user:
*   **Limiter States**: Identify exactly when and why the GPU is being throttled (e.g., `FAST_DIE` or `PWR_ZONE` limits).
*   **Firmware Breakdown**: See the split between the device, renderer, and tiler utilization to understand where the bottleneck lies in the graphics pipeline.
*   **Queue-Level Granularity**: Drilled-down views that show usage per Metal command queue, allowing developers to see how their specific GPU work is being scheduled.
