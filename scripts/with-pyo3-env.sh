#!/usr/bin/env bash
#
# The PyO3 build environment `maturin develop` uses -- the single authority for
# it, shared by every cargo invocation in the dev loop (scripts/verify.sh, the
# pre-commit clippy hook).
#
# Why this exists: pyo3-build-config's build script declares
# `rerun-if-env-changed` on PYO3_ENVIRONMENT_SIGNATURE and PYO3_PYTHON. maturin
# sets both; a bare `cargo clippy` / `cargo test` sets neither. So every switch
# between the two re-runs that build script, which rebuilds pyo3 and everything
# above it -- the whole polars stack. Measured: `maturin develop` ~78s after a
# clippy run, clippy ~34s after `maturin develop`, paid on every verify.sh run
# and after every pre-commit clippy, with no code change at all. With both sides
# seeing the same values each is a no-op (~1-2s) when nothing changed.
#
# The values are derived, not written down: the interpreter is the project venv
# maturin builds against, and the signature is maturin's
# `<implementation>-<major>.<minor>-<pointer width>bit` spelling of it. That
# spelling is maturin's, not ours, so `--check` (run by verify.sh right after
# `maturin develop`) proves the two still agree: if they drift, cargo finds
# maturin's fresh build stale and the check fails instead of the rebuild
# quietly coming back.
#
# Usage (from anywhere; paths resolve against the repo root):
#   source scripts/with-pyo3-env.sh           # export into the current shell
#   scripts/with-pyo3-env.sh cargo clippy ... # run one command under it
#   scripts/with-pyo3-env.sh --check          # after `maturin develop`: fail
#                                             # unless cargo sees its build fresh

_pyo3_env_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_pyo3_env_python="$_pyo3_env_root/polars-cv/.venv/bin/python"

if [[ ! -x "$_pyo3_env_python" ]]; then
    echo "with-pyo3-env: no project venv interpreter at $_pyo3_env_python." >&2
    echo "  Run: uv sync --directory polars-cv --group dev --no-install-project" >&2
    # `return` when sourced, `exit` when executed.
    return 1 2>/dev/null || exit 1
fi

export PYO3_PYTHON="$_pyo3_env_python"
PYO3_ENVIRONMENT_SIGNATURE="$("$_pyo3_env_python" -c '
import sys
bits = 64 if sys.maxsize > 2**32 else 32
v = sys.version_info
print(f"{sys.implementation.name}-{v.major}.{v.minor}-{bits}bit")
')"
export PYO3_ENVIRONMENT_SIGNATURE

# Sourced: the exports above are the whole job.
[[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0

set -uo pipefail

if [[ "${1:-}" == "--check" ]]; then
    # Build the exact unit graph `maturin develop` builds -- the plugin lib with
    # the features `[tool.maturin]` declares (read from pyproject, not restated)
    # -- and require every artifact fresh. Run right after `maturin develop`, a
    # stale artifact means cargo and maturin disagree about the build
    # environment, i.e. this script's derivation has drifted from maturin's.
    cd "$_pyo3_env_root" || exit 1
    features="$("$_pyo3_env_python" -c '
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
with open("polars-cv/pyproject.toml", "rb") as f:
    print(",".join(tomllib.load(f)["tool"]["maturin"].get("features", [])))
')" || exit 1
    cargo build -p polars-cv --lib --features "$features" --message-format=json 2>/dev/null \
        | "$_pyo3_env_python" -c '
import json, sys
stale = total = 0
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("reason") != "compiler-artifact":
        continue
    total += 1
    if not msg["fresh"]:
        stale += 1
        print("stale:", msg["package_id"], file=sys.stderr)
if total == 0:
    sys.exit("with-pyo3-env --check: cargo reported no artifacts; the probe is checking nothing")
if stale:
    sys.exit(
        f"with-pyo3-env --check: {stale}/{total} artifacts were rebuilt right after "
        "`maturin develop` -- cargo and maturin disagree about the PyO3 build "
        "environment, so each rebuilds the other. Compare the values this script "
        "derives with what maturin sets (maturin develop -v)."
    )
print(f"with-pyo3-env --check: all {total} artifacts fresh")
'
    exit $?
fi

if [[ $# -eq 0 ]]; then
    echo "usage: source $0 | $0 <command...> | $0 --check" >&2
    exit 2
fi
exec "$@"
