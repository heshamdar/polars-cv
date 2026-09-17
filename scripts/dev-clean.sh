#!/usr/bin/env bash
#
# Reclaim build-artifact disk without throwing away the dev cache.
#
# The dev loop builds one thing: the debug extension (`maturin develop`), whose
# artifacts live in `target/debug`. Everything else in `target/` is reclaimable:
# a `--release` tree (benchmarks, or an accidental project-install build) is
# ~2 GB of the whole polars stack the dev loop never touches, and `target/wheels`
# holds distributables. Deleting `target/debug` too just forces the next
# `maturin develop` to recompile from scratch — so by default we keep it.
#
# Usage:
#   scripts/dev-clean.sh            # remove target/release + target/wheels (keep debug)
#   scripts/dev-clean.sh --all      # also remove target/debug (full cold rebuild next time)
#
# Run from anywhere; paths resolve relative to the repo root.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ALL=0
[[ "${1:-}" == "--all" ]] && ALL=1

before="$(du -sh target 2>/dev/null | cut -f1 || echo '?')"
echo "target/ before: ${before}"

# `cargo clean --release` is the precise, cargo-aware way to drop just the
# release profile; fall back to rm if cargo is unavailable.
if [ -d target/release ]; then
    cargo clean --release 2>/dev/null || rm -rf target/release
    echo "removed target/release"
fi

if [ -d target/wheels ]; then
    rm -rf target/wheels
    echo "removed target/wheels"
fi

if [[ $ALL -eq 1 ]]; then
    cargo clean 2>/dev/null || rm -rf target/debug
    echo "removed target/debug (next build is a full cold rebuild)"
fi

after="$(du -sh target 2>/dev/null | cut -f1 || echo '0')"
echo "target/ after:  ${after}"
