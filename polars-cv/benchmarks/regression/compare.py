"""Compare two regression-suite result files and gate on regressions.

Loads a baseline and a candidate JSON (each produced by ``run_suite``), keys
results by ``(framework, operation, image_size, image_count, gpu_mode)``,
computes the signed percent change in throughput / latency / peak memory, and
classifies each as IMPROVED / REGRESSED / NEUTRAL (plus MISSING / NEW for keys
present in only one file).

Exits non-zero if any result REGRESSED or went MISSING, so it can gate a perf
change. Memory is advisory by default (pass ``--gate-memory`` to include it).

This module deliberately depends only on the standard library so it can run on
a machine where polars-cv is not built.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Defaults mirror benchmarks.regression.config.Thresholds, duplicated here so
# this module stays import-light (no package side effects when used standalone).
DEFAULT_THROUGHPUT_PCT = 7.0
DEFAULT_LATENCY_PCT = 7.0
DEFAULT_MEMORY_PCT = 20.0

ResultKey = tuple[str, str, tuple[int, ...], int, Any]


class UnmeasuredMemory(RuntimeError):
    """`--gate-memory` was requested for a result that carries no measurement."""


class Status(str, Enum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    NEUTRAL = "NEUTRAL"
    MISSING = "MISSING"  # in baseline, gone in candidate (could hide a regression)
    NEW = "NEW"  # only in candidate


@dataclass
class Delta:
    key: ResultKey
    throughput_pct: float
    latency_pct: float
    #: None when either run did not measure memory. Never 0.0 for "unknown".
    memory_pct: float | None
    status: Status
    reason: str


def _key(d: dict[str, Any]) -> ResultKey:
    # image_size serializes to a JSON array -> loads as list; normalize to a
    # tuple so it is hashable and keys line up between runs.
    return (
        d["framework"],
        d["operation"],
        tuple(d["image_size"]),
        d["image_count"],
        d.get("gpu_mode"),
    )


@dataclass
class LoadedRun:
    """One results file: its records, its provenance, and its format version.

    v1 files are a bare JSON array with no provenance at all (it lived in a
    `.meta.json` sidecar this module never read). They still load, because the
    committed `reports/**` baselines are v1 and remain useful as archives —
    but `meta` is None for them, and :func:`check_comparable` treats comparing
    across versions as a hard mismatch rather than guessing.
    """

    results: dict[ResultKey, dict[str, Any]]
    meta: dict[str, Any] | None
    schema_version: int
    path: str


def load_results(path: str | Path) -> dict[ResultKey, dict[str, Any]]:
    """Load a results file and return just its records, keyed by result."""
    return load_run(path).results


def load_run(path: str | Path) -> LoadedRun:
    """Load a results file in either format."""
    raw = json.loads(Path(path).read_text())

    if isinstance(raw, list):
        # v1: a bare array, no provenance.
        return LoadedRun({_key(d): d for d in raw}, None, 1, str(path))

    if not isinstance(raw, dict) or "results" not in raw:
        msg = (
            f"{path}: expected a JSON array (v1) or an object with a 'results' "
            f"key (v2), got {type(raw).__name__}"
        )
        raise ValueError(msg)

    version = int(raw.get("schema_version", 2))
    return LoadedRun(
        {_key(d): d for d in raw["results"]},
        raw.get("meta"),
        version,
        str(path),
    )


#: Fields whose disagreement makes two runs incomparable, with why.
#:
#: Each of these has silently produced a meaningless PASS or FAIL: a debug
#: extension is several times slower than a release one, a four-thread
#: streaming run is several times faster than a one-thread one, and
#: `POLARS_CV_OPTIMIZATIONS` changes the physical graph outright.
HARD_META_FIELDS: dict[str, str] = {
    "build_profile": "a debug build is several times slower than a release one",
    "polars_cv_optimizations": "changes the physical graph and the cache key",
    "polars_version": "the engine under test is different",
}


def check_comparable(base: LoadedRun, cand: LoadedRun) -> list[str]:
    """Reasons these two runs must not be compared, if any.

    Returns an empty list when they may be. This is the check that was missing
    entirely: `run_suite` wrote provenance to a sidecar and this module never
    opened it, so nothing stopped a one-thread baseline being compared against
    a four-thread candidate, or a debug build against a release one.
    """
    problems: list[str] = []

    if base.schema_version != cand.schema_version:
        problems.append(
            f"schema_version: {base.path} is v{base.schema_version}, "
            f"{cand.path} is v{cand.schema_version}. A v1 file carries no "
            f"provenance, so there is no way to establish the two runs are "
            f"comparable — and v1 baselines came from machines that no longer "
            f"exist. Re-measure the baseline rather than comparing across."
        )
        return problems

    if base.meta is None or cand.meta is None:
        problems.append(
            "provenance is absent from at least one run (v1 format), so "
            "comparability cannot be established."
        )
        return problems

    for field, why in HARD_META_FIELDS.items():
        a, b = base.meta.get(field), cand.meta.get(field)
        if a != b:
            problems.append(f"{field}: {a!r} vs {b!r} — {why}")

    # Cells are compared by the pool each one actually got, not by what it
    # requested: `streaming@auto` is 4 threads on this runner and 2 on another.
    a_cells = {
        k: v.get("thread_pool_size") for k, v in (base.meta.get("cells") or {}).items()
    }
    b_cells = {
        k: v.get("thread_pool_size") for k, v in (cand.meta.get("cells") or {}).items()
    }
    if a_cells != b_cells:
        problems.append(
            f"cells: {a_cells} vs {b_cells} — the two runs did not measure the "
            f"same engine/thread points, so per-key deltas mix different "
            f"execution modes."
        )

    a_cfg = base.meta.get("config") or {}
    b_cfg = cand.meta.get("config") or {}
    for field in ("image_counts", "image_sizes", "scenarios"):
        if a_cfg.get(field) != b_cfg.get(field):
            problems.append(
                f"config.{field}: {a_cfg.get(field)!r} vs {b_cfg.get(field)!r} "
                f"— a different matrix was measured."
            )

    return problems


def _pct(base: float | None, cand: float | None) -> float | None:
    """Signed percent change from base to cand, or None if either is unmeasured.

    ``None`` propagates rather than collapsing to ``0.0``. An unmeasured value
    used to arrive as ``0.0``, and ``0 -> 0`` is a 0% change, so a whole column
    that was never measured classified as NEUTRAL — a hole in the results
    reading as a clean bill of health.
    """
    if base is None or cand is None:
        return None
    if base == 0:
        return 0.0 if cand == 0 else float("inf")
    return (cand - base) / base * 100.0


def classify(
    base: dict[str, Any],
    cand: dict[str, Any],
    *,
    throughput_pct: float,
    latency_pct: float,
    memory_pct: float,
    gate_memory: bool,
) -> tuple[Status, str, float, float, float | None]:
    tp = _pct(
        base["throughput_images_per_second"], cand["throughput_images_per_second"]
    )
    # Latency is the reciprocal of throughput (both derive from the same wall
    # time), so it is shown for context but NOT used to gate — doing so would
    # just double-count the same signal. ``latency_pct`` is accepted for API
    # symmetry / possible future divergent metrics.
    _ = latency_pct
    lat = _pct(base["latency_ms_per_image"], cand["latency_ms_per_image"])
    mem = _pct(base["peak_memory_mb"], cand["peak_memory_mb"])

    # Regression gate: throughput dropped past the band.
    if tp <= -throughput_pct:
        return (Status.REGRESSED, "throughput regression", tp, lat, mem)
    if gate_memory:
        if mem is None:
            # Refuse rather than pass. `--gate-memory` on an unmeasured column
            # would otherwise report "no memory regressions" about nothing.
            msg = (
                f"--gate-memory was requested but this result has no memory "
                f"measurement (base status="
                f"{base.get('memory_status', 'unknown')!r}, cand status="
                f"{cand.get('memory_status', 'unknown')!r}). Install psutil or "
                f"drop --gate-memory; do not gate on an absent number."
            )
            raise UnmeasuredMemory(msg)
        if mem >= memory_pct:
            return (Status.REGRESSED, "memory regression", tp, lat, mem)
    if tp >= throughput_pct:
        return (Status.IMPROVED, "throughput improvement", tp, lat, mem)
    return (Status.NEUTRAL, "within threshold", tp, lat, mem)


def compare(
    baseline: dict[ResultKey, dict[str, Any]],
    candidate: dict[ResultKey, dict[str, Any]],
    *,
    throughput_pct: float = DEFAULT_THROUGHPUT_PCT,
    latency_pct: float = DEFAULT_LATENCY_PCT,
    memory_pct: float = DEFAULT_MEMORY_PCT,
    gate_memory: bool = False,
) -> list[Delta]:
    deltas: list[Delta] = []
    for key in sorted(set(baseline) | set(candidate), key=lambda k: tuple(map(str, k))):
        base = baseline.get(key)
        cand = candidate.get(key)
        if base is None:
            deltas.append(Delta(key, 0.0, 0.0, None, Status.NEW, "new in candidate"))
            continue
        if cand is None:
            deltas.append(
                Delta(key, 0.0, 0.0, None, Status.MISSING, "missing from candidate")
            )
            continue
        status, reason, tp, lat, mem = classify(
            base,
            cand,
            throughput_pct=throughput_pct,
            latency_pct=latency_pct,
            memory_pct=memory_pct,
            gate_memory=gate_memory,
        )
        deltas.append(Delta(key, tp, lat, mem, status, reason))
    return deltas


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "   n/a"
    if v == float("inf"):
        return "  +inf"
    if v == float("-inf"):
        return "  -inf"
    return f"{v:+6.1f}"


def _fmt_key(key: ResultKey) -> str:
    framework, operation, size, count, gpu = key
    size_s = "x".join(str(d) for d in size)
    gpu_s = f"/{gpu}" if gpu else ""
    return f"{framework:<22} {operation:<20} {size_s:>9} n={count:<5}{gpu_s}"


def print_table(deltas: list[Delta]) -> None:
    header = (
        f"{'framework / operation / size / count':<60} "
        f"{'thru%':>7} {'lat%':>7} {'mem%':>7}  status"
    )
    print(header)
    print("-" * len(header))
    for d in deltas:
        print(
            f"{_fmt_key(d.key):<60} "
            f"{_fmt_pct(d.throughput_pct):>7} {_fmt_pct(d.latency_pct):>7} "
            f"{_fmt_pct(d.memory_pct):>7}  {d.status.value}"
        )


def summarize(deltas: list[Delta]) -> dict[str, Any]:
    counts: dict[str, int] = {s.value: 0 for s in Status}
    for d in deltas:
        counts[d.status.value] += 1
    regressions = [
        {
            "key": list(map(_jsonable, d.key)),
            "throughput_pct": d.throughput_pct,
            "latency_pct": d.latency_pct,
            "reason": d.reason,
        }
        for d in deltas
        if d.status in (Status.REGRESSED, Status.MISSING)
    ]
    return {"counts": counts, "regressions": regressions}


def _jsonable(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two regression-suite result files and gate on regressions."
    )
    parser.add_argument("baseline", help="baseline results JSON")
    parser.add_argument("candidate", help="candidate results JSON")
    parser.add_argument(
        "--throughput-threshold",
        type=float,
        default=DEFAULT_THROUGHPUT_PCT,
        help="percent throughput drop that counts as a regression (default: 5)",
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        default=DEFAULT_LATENCY_PCT,
        help="percent latency rise that counts as a regression (default: 5)",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=DEFAULT_MEMORY_PCT,
        help="percent peak-memory rise that counts as a regression (default: 15)",
    )
    parser.add_argument(
        "--gate-memory",
        action="store_true",
        help="treat memory regressions as failures (advisory by default)",
    )
    parser.add_argument(
        "--allow-mismatch",
        help=(
            "comma-separated provenance fields to compare across anyway "
            "(e.g. 'polars_version'). Each one you name is a way the numbers "
            "can differ for a reason that is not the change under test."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="also print a machine-readable summary"
    )
    args = parser.parse_args(argv)

    base_run = load_run(args.baseline)
    cand_run = load_run(args.candidate)

    problems = check_comparable(base_run, cand_run)
    allowed = {f.strip() for f in (args.allow_mismatch or "").split(",") if f.strip()}
    blocking = [p for p in problems if p.split(":", 1)[0] not in allowed]
    if blocking:
        print("ERROR: these runs are not comparable:", file=sys.stderr)
        for problem in blocking:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nRe-measure so both runs share these, or pass "
            "--allow-mismatch FIELD[,FIELD] to compare anyway and own the "
            "result.",
            file=sys.stderr,
        )
        return 2
    for problem in problems:
        print(f"WARNING (allowed): {problem}", file=sys.stderr)

    baseline = base_run.results
    candidate = cand_run.results
    try:
        deltas = compare(
            baseline,
            candidate,
            throughput_pct=args.throughput_threshold,
            latency_pct=args.latency_threshold,
            memory_pct=args.memory_threshold,
            gate_memory=args.gate_memory,
        )
    except UnmeasuredMemory as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_table(deltas)
    summary = summarize(deltas)
    print()
    print("Summary: " + ", ".join(f"{k}={v}" for k, v in summary["counts"].items()))
    if args.json:
        print(json.dumps(summary, indent=2))

    failed = (
        summary["counts"][Status.REGRESSED.value]
        + summary["counts"][Status.MISSING.value]
    )
    if failed:
        print(f"\nFAIL: {failed} regressed/missing result(s).")
        return 1
    print("\nPASS: no regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
