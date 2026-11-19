# AGENTS.md

macgpu is a Rust workspace (root `macgpu` binary, `macgpu-core` sampler, `macgpu-model` report schema) for Apple Silicon.
The README documents commands and metrics.

- Default branch `main`; remotes `origin` (GitHub) and `forgejo`. Forgejo CI (`.forgejo/workflows/ci.yml`) runs on Linux only: rustfmt for the workspace plus check, clippy and tests for `macgpu-model`. Everything else is verified locally with the gate.
- It builds and tests on macOS on Apple Silicon only; the sampler reads IOKit, IOReport, SMC and kernel coalitions.
  Only `macgpu-model` is portable.
- Verify with `mise run gate`: rustfmt, clippy with warnings denied, `check-model-linux` (`cargo check -p macgpu-model
  --target x86_64-unknown-linux-gnu`, which needs that rustup target installed), then `cargo test --workspace`.

## Tests

- Tests that generate real GPU load, or need a live GPU, are `#[ignore]`d. `mise run test` and the gate skip them;
  `mise run test-hardware` runs everything with `--include-ignored`. Run it when you change sampling or attribution,
  and say so if you did not.
- `mise run test-unit` is the fast loop; `mise run test-e2e` runs the CLI test binaries without GPU load.
- `mise run install` copies the release binary to `~/.local/bin/macgpu`. Other tools on the machine run that installed
  binary, so an install changes what they see.

## Contracts with other repos

- `macgpu-model`'s `Report` is the versioned JSON schema behind `--json`, `record` and `serve`. Add fields as optional;
  removing or renaming one, or changing its meaning, bumps `SCHEMA_VERSION`. On 2026-11-17 a removed working-set
  field broke a downstream consumer of the report, so check its consumers before removing anything.
- Local tools may depend on `macgpu-core` and `macgpu-model` by path (`../macgpu/crates/...`), so they build against
  whatever is checked out in the primary `~/projects/macgpu`, uncommitted changes included. A broken or half-finished
  edit there breaks them.
- The sampler must not open a Metal device: that would make macgpu a GPU client and skew its own numbers. Anything
  that needs `MTLDevice` data (such as the recommended working set) is read by the caller instead.
