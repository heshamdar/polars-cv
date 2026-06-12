"""Run the fixed regression matrix and write a results JSON.

This wraps the existing ``benchmarks.scenarios`` run_all_* functions for the
two polars-cv adapters only, repeats the whole suite ``suite_repeats`` times,
and keeps the best-of per result (the underlying scenarios only expose the mean
over iterations, so whole-suite repeats are how we reject noise).

Thread pinning MUST happen before polars is imported, so this module sets the
env vars at import time, before importing anything that pulls in polars.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.frameworks import BaseFrameworkAdapter, BenchmarkResult

from benchmarks.regression.config import (
    ALL_SCENARIOS,
    DEFAULT,
    POLARS_CV_ADAPTERS,
    SuiteConfig,
)


def _pin_threads(n: int) -> None:
    """Pin the thread count for reproducibility.

    Must run before polars/polars_cv are first imported, otherwise the pool is
    already sized. ``setdefault`` is used so an explicit env override wins.
    """
    os.environ["POLARS_MAX_THREADS"] = str(n)
    os.environ["RAYON_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


def build_adapters(names: list[str]) -> list[BaseFrameworkAdapter]:
    """Construct the named adapters, failing loud if any is unavailable.

    A missing/unbuilt polars-cv would otherwise yield an empty results file
    that silently compares as all-NEUTRAL.
    """
    from benchmarks.frameworks import get_adapter

    adapters: list[BaseFrameworkAdapter] = []
    for name in names:
        adapter = get_adapter(name)  # raises ValueError on an unknown name
        if not adapter.is_available():
            msg = (
                f"adapter {name!r} is not available — is polars-cv built? "
                f"Run `maturin develop --release` first."
            )
            raise RuntimeError(msg)
        adapters.append(adapter)
    return adapters


def _run_once(
    adapters: list[BaseFrameworkAdapter], cfg: SuiteConfig, *, quiet: bool
) -> list[BenchmarkResult]:
    """One full pass over the configured scenarios."""
    from benchmarks.scenarios.e2e_workflow import run_all_e2e_workflows
    from benchmarks.scenarios.pipelines import run_all_pipelines
    from benchmarks.scenarios.single_ops import run_all_single_ops

    verbose = not quiet
    results: list[BenchmarkResult] = []
    if "single_ops" in cfg.scenarios:
        results += run_all_single_ops(
            adapters,
            cfg.image_counts,
            cfg.image_sizes,
            cfg.warmup_iterations,
            cfg.benchmark_iterations,
            verbose=verbose,
        )
    if "pipelines" in cfg.scenarios:
        results += run_all_pipelines(
            adapters,
            cfg.image_counts,
            cfg.image_sizes,
            cfg.warmup_iterations,
            cfg.benchmark_iterations,
            complexity_filter=None,
            verbose=verbose,
        )
    if "e2e" in cfg.scenarios:
        results += run_all_e2e_workflows(
            adapters,
            cfg.image_counts,
            cfg.image_sizes,
            cfg.warmup_iterations,
            cfg.benchmark_iterations,
            verbose=verbose,
        )
    if "zero_copy" in cfg.scenarios:
        # zero_copy has its own hardcoded matrix and no adapter arg; its results
        # are polars-cv only, which is exactly what we want.
        from benchmarks.scenarios.zero_copy_ingestion import (
            run_benchmarks as run_zero_copy,
        )

        results += run_zero_copy()
    return results


def _result_key(r: BenchmarkResult) -> tuple:
    return (r.framework, r.operation, tuple(r.image_size), r.image_count, r.gpu_mode)


def _aggregate_best(runs: list[list[BenchmarkResult]]) -> list[BenchmarkResult]:
    """Reduce repeated runs to one best-of result per key.

    Best-of = the repeat with the highest throughput (and its matching latency
    / time); peak memory is the median across repeats (less order-sensitive
    than the single best run's RSS).
    """
    by_key: dict[tuple, list[BenchmarkResult]] = {}
    for run in runs:
        for r in run:
            by_key.setdefault(_result_key(r), []).append(r)

    aggregated: list[BenchmarkResult] = []
    for results in by_key.values():
        best = max(results, key=lambda r: r.throughput_images_per_second)
        median_mem = statistics.median(r.peak_memory_mb for r in results)
        aggregated.append(replace(best, peak_memory_mb=median_mem))
    aggregated.sort(key=lambda r: tuple(map(str, _result_key(r))))
    return aggregated


def run_suite(cfg: SuiteConfig, *, quiet: bool = True) -> list[BenchmarkResult]:
    adapters = build_adapters(POLARS_CV_ADAPTERS)
    runs = [_run_once(adapters, cfg, quiet=quiet) for _ in range(cfg.suite_repeats)]
    return _aggregate_best(runs)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _write_meta(out_path: Path, cfg: SuiteConfig, num_threads: int) -> None:
    meta = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_threads": num_threads,
        "adapters": POLARS_CV_ADAPTERS,
        "config": {
            "image_counts": cfg.image_counts,
            "image_sizes": [list(s) for s in cfg.image_sizes],
            "warmup_iterations": cfg.warmup_iterations,
            "benchmark_iterations": cfg.benchmark_iterations,
            "suite_repeats": cfg.suite_repeats,
            "scenarios": list(cfg.scenarios),
        },
    }
    out_path.with_suffix(out_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2)
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the polars-cv regression suite.")
    parser.add_argument("--out", required=True, help="output results JSON path")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT.scenarios),
        help=f"comma-separated subset of {ALL_SCENARIOS} (default: %(default)s)",
    )
    parser.add_argument("--counts", help="comma-separated image counts override")
    parser.add_argument("--sizes", help="comma-separated square sizes override")
    parser.add_argument("--threads", type=int, default=DEFAULT.num_threads)
    parser.add_argument("--repeats", type=int, default=DEFAULT.suite_repeats)
    parser.add_argument("--warmup", type=int, default=DEFAULT.warmup_iterations)
    parser.add_argument("--iterations", type=int, default=DEFAULT.benchmark_iterations)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace) -> SuiteConfig:
    scenarios = tuple(s.strip() for s in args.scenarios.split(",") if s.strip())
    unknown = [s for s in scenarios if s not in ALL_SCENARIOS]
    if unknown:
        msg = f"unknown scenario(s): {unknown}; valid: {ALL_SCENARIOS}"
        raise SystemExit(msg)
    counts = (
        [int(c) for c in args.counts.split(",")]
        if args.counts
        else DEFAULT.image_counts
    )
    sizes = (
        [(int(s), int(s)) for s in args.sizes.split(",")]
        if args.sizes
        else DEFAULT.image_sizes
    )
    return SuiteConfig(
        image_counts=counts,
        image_sizes=sizes,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        suite_repeats=args.repeats,
        scenarios=scenarios,
        num_threads=args.threads,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _pin_threads(args.threads)
    cfg = _cfg_from_args(args)

    # Imported here, after _pin_threads, so the thread env is set first.
    from benchmarks.utils.results import ResultsCollector

    results = run_suite(cfg, quiet=args.quiet)
    if not results:
        print("ERROR: suite produced no results.", file=sys.stderr)
        return 2

    collector = ResultsCollector()
    collector.add_many(results)
    out_path = Path(args.out)
    out_path.write_text(collector.to_json(indent=2))
    _write_meta(out_path, cfg, args.threads)
    print(f"Wrote {len(results)} results to {out_path} (threads={args.threads}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
