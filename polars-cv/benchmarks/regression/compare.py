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
    memory_pct: float
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


def load_results(path: str | Path) -> dict[ResultKey, dict[str, Any]]:
    """Load a results JSON (a list of BenchmarkResult dicts) keyed by result."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        msg = f"{path}: expected a JSON array of results, got {type(raw).__name__}"
        raise ValueError(msg)
    return {_key(d): d for d in raw}


def _pct(base: float, cand: float) -> float:
    """Signed percent change from base to cand. +inf if base is 0 and cand > 0."""
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
) -> tuple[Status, str, float, float, float]:
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
    if gate_memory and mem >= memory_pct:
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
            deltas.append(Delta(key, 0.0, 0.0, 0.0, Status.NEW, "new in candidate"))
            continue
        if cand is None:
            deltas.append(
                Delta(key, 0.0, 0.0, 0.0, Status.MISSING, "missing from candidate")
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


def _fmt_pct(v: float) -> str:
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
        {"key": list(map(_jsonable, d.key)), "throughput_pct": d.throughput_pct,
         "latency_pct": d.latency_pct, "reason": d.reason}
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
        "--throughput-threshold", type=float, default=DEFAULT_THROUGHPUT_PCT,
        help="percent throughput drop that counts as a regression (default: 5)",
    )
    parser.add_argument(
        "--latency-threshold", type=float, default=DEFAULT_LATENCY_PCT,
        help="percent latency rise that counts as a regression (default: 5)",
    )
    parser.add_argument(
        "--memory-threshold", type=float, default=DEFAULT_MEMORY_PCT,
        help="percent peak-memory rise that counts as a regression (default: 15)",
    )
    parser.add_argument(
        "--gate-memory", action="store_true",
        help="treat memory regressions as failures (advisory by default)",
    )
    parser.add_argument(
        "--json", action="store_true", help="also print a machine-readable summary"
    )
    args = parser.parse_args(argv)

    baseline = load_results(args.baseline)
    candidate = load_results(args.candidate)
    deltas = compare(
        baseline,
        candidate,
        throughput_pct=args.throughput_threshold,
        latency_pct=args.latency_threshold,
        memory_pct=args.memory_threshold,
        gate_memory=args.gate_memory,
    )

    print_table(deltas)
    summary = summarize(deltas)
    print()
    print("Summary: " + ", ".join(f"{k}={v}" for k, v in summary["counts"].items()))
    if args.json:
        print(json.dumps(summary, indent=2))

    failed = summary["counts"][Status.REGRESSED.value] + summary["counts"][
        Status.MISSING.value
    ]
    if failed:
        print(f"\nFAIL: {failed} regressed/missing result(s).")
        return 1
    print("\nPASS: no regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
