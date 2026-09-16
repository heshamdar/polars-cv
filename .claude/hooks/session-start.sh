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
# Idempotent and safe to re-run: `uv sync` and `maturin develop` are no-ops when
# already current, and `pre-commit install` just rewrites `.git/hooks/pre-commit`.
set -euo pipefail

# Only run in Claude Code on the web / remote containers. Local checkouts are set
# up by their developer (who has already run `pre-commit install`).
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

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
#    stack. On a fresh container (cold cache, every web session) that is a
#    10+ minute blocking compile that step 3 then *throws away*, because the
#    debug `maturin develop` build overwrites the release `_lib.abi3.so`. A
#    synchronous SessionStart hook doing that reads to the user as "the session
#    hangs after initialization". So sync only the dependency groups here and
#    let step 3 own the single (debug) build — matching the repo rule that the
#    dev loop never builds `--release`.
uv sync --group dev --no-install-project --directory polars-cv

# 3. Build the plugin (debug — what CI and scripts/verify.sh use). Needed for
#    @plugin_required tests and the structural pre-commit hook.
uv run --directory polars-cv maturin develop

# 4. Install the pre-commit git hook so commits are gated by ruff / cargo fmt /
#    clippy / the structural guards before they land. Hook installation lives in
#    `.git/hooks` (not tracked), so it must be re-done in every fresh container.
uv run --directory polars-cv pre-commit install
