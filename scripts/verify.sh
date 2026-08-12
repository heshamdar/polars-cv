#!/usr/bin/env bash
#
# Run every check CI runs, report each one's exit code, and exit non-zero if
# any failed.
#
# This exists because reading a *filtered view* of a check's output has
# repeatedly produced false "all green" reports on this repo: a `grep | head`
# that cut off the failing suite below the fold, and a `maturin ... | tail`
# whose reported exit code was tail's, not maturin's. Both looked like
# success. Every command below has its own exit code captured directly and
# printed, and the summary is computed from those codes rather than from
# anything a human or an agent read off the screen.
#
# Usage:
#   scripts/verify.sh            # everything
#   scripts/verify.sh --fast     # skip the slow lane
#
# Run from anywhere; paths are resolved relative to the repo root.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# No toolchain is pinned here on purpose. `rust-toolchain.toml` at the repo
# root is the single authority and rustup honours it automatically; CI uses
# `dtolnay/rust-toolchain@stable`, which resolves to the same channel. Naming a
# version here made this a third declaration that could disagree with both --
# and it did: it pinned 1.96 while the manifest says `stable`, so a local run
# silently checked a different compiler than CI, downloading it to do so.

FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1

FAILED=0
declare -a RESULTS

run_check() {
    local label="$1"; shift
    local log
    log="$(mktemp)"
    "$@" >"$log" 2>&1
    local code=$?
    if [[ $code -eq 0 ]]; then
        RESULTS+=("  ok    (exit 0)   $label")
    else
        RESULTS+=("  FAIL  (exit $code)   $label")
        FAILED=1
        echo "===== FAILED: $label (exit $code) ====="
        tail -40 "$log"
        echo "===== end $label ====="
    fi
    rm -f "$log"
}

echo "Verifying at $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown') ..."

run_check "cargo fmt --check"        cargo fmt --all -- --check
run_check "cargo clippy -D warnings" cargo clippy --workspace --all-targets --all-features -- -D warnings
run_check "cargo test view-buffer"   cargo test -p view-buffer --all-features
run_check "cargo test polars-cv"     cargo test -p polars-cv

# The Python lanes need the compiled extension to match the working tree. The
# install is editable, so Python sources are always current while the .so stays
# at its last build -- a stale .so silently turns plugin tests into skips.
run_check "maturin develop (debug)"  uv run --no-sync --directory polars-cv maturin develop

# The structural lane runs first and on its own: it is what pre-commit runs,
# so when it fails the local hook would have caught this before the push, and
# saying so up front beats finding it under the full suite's output. It is a
# subset of the fast lane below, which still runs it -- the point is the
# separate exit code, not skipping it later.
run_check "pytest (structural lane)" \
    uv run --no-sync --directory polars-cv pytest tests/ -q -m "structural and not slow"

run_check "pytest (fast lane)" \
    uv run --no-sync --directory polars-cv pytest tests/ -q -m "not network and not slow"
if [[ $FAST -eq 0 ]]; then
    run_check "pytest (slow lane)" \
        uv run --no-sync --directory polars-cv pytest tests/ -q -m "slow and not network"
fi

run_check "ruff check"  uvx ruff check polars-cv/python polars-cv/tests polars-cv/benchmarks
run_check "ruff format" uvx ruff format --check polars-cv/python polars-cv/tests polars-cv/benchmarks

# `--strict` matches the `docs` job in ci.yml: a warning fails the build instead
# of scrolling past.
#
# `--no-sync`, like the pytest lanes above, and for a sharper reason than
# consistency: letting `uv run` sync makes it rebuild the extension through the
# PEP 517 backend, which maturin runs at `--profile release`. That is a
# multi-minute re-optimisation of the whole polars stack on every verification,
# for a documentation build that does not use the extension at all.
#
# The docs dependencies are a separate group, so a checkout that has not run
# `uv sync --group docs` fails this check with a missing-module error. That is
# the same contract the pytest lanes already have with `--group dev`.
run_check "mkdocs build --strict" \
    uv run --no-sync --directory polars-cv mkdocs build --strict

echo
echo "Summary:"
printf '%s\n' "${RESULTS[@]}"
echo

if [[ $FAILED -eq 0 ]]; then
    echo "PASS"
else
    echo "FAIL — at least one check above exited non-zero"
fi
exit $FAILED
