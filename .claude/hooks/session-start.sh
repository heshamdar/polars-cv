#!/bin/bash
# SessionStart hook: prepare the environment so tests, linters, and the
# pre-commit hook all work in Claude Code on the web.
#
# Why the build, not just `pre-commit install`: the committed
# `.pre-commit-config.yaml` runs `cargo fmt`/`clippy` and a `pytest -m structural`
# lane that needs the compiled `_lib.abi3.so`. Installing the git hook without a
# built, MSRV-correct toolchain would make every commit fail the hook rather than
# pass it. So this brings the whole environment up, then wires the hook in.
#
# ASYNC: the first stdout line hands the work to the harness's background runner
# so a cold full compile of the polars stack never blocks the session prompt.
# The trade-off is a race window: for the first few minutes the extension may not
# be importable and @plugin_required tests will skip. Progress streams to
# `.claude/hooks/.session-start.log`, and `.claude/hooks/.session-start.done`
# is written once the environment is ready — check it before relying on the
# build. If you need to run tests immediately, run `uv run --directory polars-cv
# maturin develop` yourself and wait for it.
echo '{"async": true, "asyncTimeout": 900000}'

# Idempotent and safe to re-run: `uv sync` and `maturin develop` are no-ops when
# already current, and `pre-commit install` just rewrites `.git/hooks/pre-commit`.
set -euo pipefail

# Only run in Claude Code on the web / remote containers. Local checkouts are set
# up by their developer (who has already run `pre-commit install`).
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

LOG=".claude/hooks/.session-start.log"
DONE=".claude/hooks/.session-start.done"
rm -f "$DONE"
# Mirror everything below to the log as well as the harness's async output.
exec > >(tee "$LOG") 2>&1

# 0. Reclaim disk from any stale `--release` tree. The dev loop builds *only*
#    debug (`maturin develop`, below) — release artifacts come from benchmarks
#    or an accidental project-install build, are never used by tests or the
#    prompt-facing loop, and a full release build of this stack is ~2 GB. In an
#    ephemeral remote container that is pure storage pressure, so clear it. (Run
#    `cargo clean --release` yourself later if a benchmark rebuilds it.)
if [ -d target/release ]; then
  echo "session-start: reclaiming stale target/release ($(du -sh target/release 2>/dev/null | cut -f1))"
  cargo clean --release 2>/dev/null || rm -rf target/release
fi

# 1. Rust toolchain. `rust-toolchain.toml` pins `channel = "stable"`, and both
#    crates require MSRV 1.96 — but a fresh container commonly ships an older
#    cached `stable` (e.g. 1.94) that fails to compile the polars 0.54 stack
#    (E0658 on now-stable APIs), which then makes the whole plugin-dependent
#    suite self-skip silently. `rustup update stable` upgrades the installed
#    stable that the pin resolves to (and self-updates rustup itself), so the
#    pinned channel becomes >= 1.96. Best-effort — never hard-block the session
#    on a missing rustup or a transient network failure; a still-too-old
#    toolchain then surfaces as a clear MSRV error at `maturin develop` below.
if command -v rustup >/dev/null 2>&1; then
  rustup update stable \
    || echo "session-start: 'rustup update stable' failed; using the installed toolchain" >&2
  echo "session-start: rust toolchain -> $(rustc --version 2>/dev/null || echo 'unknown')"
fi

# 2. Python dev dependencies (includes pre-commit and maturin). The uv project
#    lives under polars-cv/.
#
#    `--no-install-project` is load-bearing, not a micro-optimization. Because
#    polars-cv uses the maturin build backend, a plain `uv sync` *builds and
#    installs the project itself* as an editable wheel — and that build runs
#    under the workspace `[profile.release]` (lto = "fat", codegen-units = 1,
#    opt-level = 3), the slowest possible full-LTO compile of the whole polars
#    stack. Step 3's debug `maturin develop` then overwrites its `_lib.abi3.so`,
#    so the release build is pure waste. Sync only the dependency groups here
#    and let step 3 own the single (debug) build — the one way this repo builds
#    for development.
uv sync --group dev --no-install-project --directory polars-cv

# 3. Build the plugin (debug — the one canonical dev build, what CI and
#    scripts/verify.sh use). Needed for @plugin_required tests and the
#    structural pre-commit hook.
uv run --no-sync --directory polars-cv maturin develop

# 4. Install the pre-commit git hook so commits are gated by ruff / cargo fmt /
#    clippy / the structural guards before they land. Hook installation lives in
#    `.git/hooks` (not tracked), so it must be re-done in every fresh container.
uv run --no-sync --directory polars-cv pre-commit install

echo "session-start: environment ready"
touch "$DONE"
