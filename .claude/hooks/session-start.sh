#!/bin/bash
# SessionStart hook: prepare the environment so tests, linters, and the
# pre-commit hook all work in Claude Code on the web.
#
# It does NOT build the extension. The session runs `maturin develop` itself
# (from polars-cv/, debug) when it needs the plugin: a build started here raced
# the session's own, and until one finishes @plugin_required tests self-skip and
# the pre-commit hook's `pytest -m structural` lane cannot import `_lib.abi3.so`.
# Build before testing or committing.
#
# ASYNC: the first stdout line hands the work to the harness's background runner
# so the toolchain update and dependency sync never block the session prompt.
# Progress streams to `.claude/hooks/.session-start.log`, and
# `.claude/hooks/.session-start.done` is written once the environment is ready.
echo '{"async": true, "asyncTimeout": 900000}'

# Idempotent and safe to re-run: `uv sync` is a no-op when already current, and `pre-commit install` just rewrites `.git/hooks/pre-commit`.
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

# 0. Reclaim disk from any stale `--release` (or `--profile benchmark`) tree. The dev loop builds *only*
#    debug (`maturin develop`, run by the session) — release artifacts come from benchmarks
#    or an accidental project-install build, are never used by tests or the
#    prompt-facing loop, and a full release build of this stack is ~2 GB. In an
#    ephemeral remote container that is pure storage pressure, so clear it. (Run
#    `cargo clean --release` yourself later if a benchmark rebuilds it.)
if [ -d target/release ]; then
  echo "session-start: reclaiming stale target/release ($(du -sh target/release 2>/dev/null | cut -f1))"
  cargo clean --release 2>/dev/null || rm -rf target/release
fi
if [ -d target/benchmark ]; then
  echo "session-start: reclaiming stale target/benchmark ($(du -sh target/benchmark 2>/dev/null | cut -f1))"
  cargo clean --profile benchmark 2>/dev/null || rm -rf target/benchmark
fi

# 1. Rust toolchain. `rust-toolchain.toml` pins `channel = "stable"`, and both
#    crates require MSRV 1.96 — but a fresh container commonly ships an older
#    cached `stable` (e.g. 1.94) that fails to compile the polars 0.54+ stack
#    (E0658 on now-stable APIs), which then makes the whole plugin-dependent
#    suite self-skip silently. `rustup update stable` upgrades the installed
#    stable that the pin resolves to (and self-updates rustup itself), so the
#    pinned channel becomes >= 1.96. Best-effort — never hard-block the session
#    on a missing rustup or a transient network failure; a still-too-old
#    toolchain then surfaces as a clear MSRV error at the session's `maturin develop`.
if command -v rustup >/dev/null 2>&1; then
  rustup update stable \
    || echo "session-start: 'rustup update stable' failed; using the installed toolchain" >&2
  echo "session-start: rust toolchain -> $(rustc --version 2>/dev/null || echo 'unknown')"
fi

# 1b. A new toolchain orphans everything in target/: every artifact's hash
#     includes the compiler, so the next build recompiles the whole polars stack
#     under new names and the old build (~5 GB in debug alone) is never read or
#     deleted again. `rustup update stable` above moves to each new stable, so
#     remember the compiler target/ was built with and clean it when that
#     changes. (No stamp yet, as on the first run with this step: keep target/.)
RUSTC_STAMP="target/.rustc-vV"
rustc_now="$(rustc -vV 2>/dev/null || true)"
if [ -n "$rustc_now" ]; then
  if [ -f "$RUSTC_STAMP" ] && [ "$(cat "$RUSTC_STAMP")" != "$rustc_now" ]; then
    echo "session-start: toolchain changed; reclaiming target/ ($(du -sh target 2>/dev/null | cut -f1))"
    cargo clean
  fi
  mkdir -p target
  printf '%s' "$rustc_now" > "$RUSTC_STAMP"
fi

# 2. Python dependencies: the dev group (includes pre-commit and maturin) and
#    the docs group (mkdocs, for verify.sh's `mkdocs build --strict` lane). The
#    uv project lives under polars-cv/.
#
#    `--no-install-project` is load-bearing, not a micro-optimization. Because
#    polars-cv uses the maturin build backend, a plain `uv sync` *builds and
#    installs the project itself* as an editable wheel — and that build runs
#    under the workspace `[profile.release]` (lto = "fat", codegen-units = 1,
#    opt-level = 3), the slowest possible full-LTO compile of the whole polars
#    stack, which the session's debug `maturin develop` then overwrites, so the
#    release build is pure waste. Sync only the dependency groups here and leave
#    the single (debug) build to the session — the one way this repo builds for
#    development. (This sync also uninstalls the editable project: build after.)
uv sync --group dev --group docs --no-install-project --directory polars-cv

# 2b. Give every later shell in this session the PyO3 build environment
#     `maturin develop` sets, so an ad-hoc `cargo test`/`cargo clippy` does not
#     invalidate the extension build (the next `maturin develop` would rebuild
#     the polars stack, ~80s). Values come from scripts/with-pyo3-env.sh, the
#     single authority; it needs the venv interpreter, hence after the sync.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  (
    source scripts/with-pyo3-env.sh
    printf 'export PYO3_PYTHON=%q\nexport PYO3_ENVIRONMENT_SIGNATURE=%q\n' \
      "$PYO3_PYTHON" "$PYO3_ENVIRONMENT_SIGNATURE"
  ) >> "$CLAUDE_ENV_FILE"
fi

# 3. Install the pre-commit git hook so commits are gated by ruff / cargo fmt /
#    clippy / the structural guards before they land. Hook installation lives in
#    `.git/hooks` (not tracked), so it must be re-done in every fresh container.
uv run --no-sync --directory polars-cv pre-commit install

# 4. cargo-deny, for verify.sh's supply-chain lane (CI installs it with an
#    action; there is no crate dependency on it). Prebuilt release binaries live
#    on GitHub, which the web container's network policy may block, so build it
#    from crates.io (~2.5 min, cached with the container afterwards). Last,
#    because nothing but that one lane needs it; best-effort, since a failure
#    here shows up as a clear `cargo deny` FAIL in verify.sh anyway.
if ! command -v cargo-deny >/dev/null 2>&1; then
  cargo install --locked cargo-deny \
    || echo "session-start: 'cargo install cargo-deny' failed; verify.sh's cargo deny lane will fail" >&2
fi

# 5. cargo-machete, for verify.sh's unused-dependency lane (CI installs it with
#    an action). Same reasoning as cargo-deny: built from crates.io, last, and a
#    failure shows as a clear `cargo machete` FAIL in verify.sh.
if ! command -v cargo-machete >/dev/null 2>&1; then
  cargo install --locked cargo-machete \
    || echo "session-start: 'cargo install cargo-machete' failed; verify.sh's cargo machete lane will fail" >&2
fi

echo "session-start: environment ready"
touch "$DONE"
