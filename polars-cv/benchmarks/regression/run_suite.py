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
    from collections.abc import Collection

    from benchmarks.frameworks import BaseFrameworkAdapter, BenchmarkResult

from benchmarks.regression.config import (
    ALL_SCENARIOS,
    DEFAULT,
    DEFAULT_SELECTION,
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
                f"Run `maturin develop --profile benchmark` first."
            )
            raise RuntimeError(msg)
        adapters.append(adapter)
    return adapters


# One runner per scenario: (adapters, cfg, selected case names or None,
# verbose) -> results. Keyed by `config.ALL_SCENARIOS`, checked below, so a
# scenario the selection accepts cannot silently run nothing here.
def _single_ops(
    adapters: list[BaseFrameworkAdapter],
    cfg: SuiteConfig,
    names: Collection[str] | None,
    verbose: bool,
) -> list[BenchmarkResult]:
    from benchmarks.scenarios.single_ops import run_all_single_ops

    return run_all_single_ops(
        adapters,
        cfg.image_counts,
        cfg.image_sizes,
        cfg.warmup_iterations,
        cfg.benchmark_iterations,
        verbose=verbose,
        names=names,
    )


def _pipelines(
    adapters: list[BaseFrameworkAdapter],
    cfg: SuiteConfig,
    names: Collection[str] | None,
    verbose: bool,
) -> list[BenchmarkResult]:
    from benchmarks.scenarios.pipelines import run_all_pipelines

    return run_all_pipelines(
        adapters,
        cfg.image_counts,
        cfg.image_sizes,
        cfg.warmup_iterations,
        cfg.benchmark_iterations,
        complexity_filter=None,
        verbose=verbose,
        names=names,
    )


def _e2e(
    adapters: list[BaseFrameworkAdapter],
    cfg: SuiteConfig,
    names: Collection[str] | None,
    verbose: bool,
) -> list[BenchmarkResult]:
    from benchmarks.scenarios.e2e_workflow import run_all_e2e_workflows

    return run_all_e2e_workflows(
        adapters,
        cfg.image_counts,
        cfg.image_sizes,
        cfg.warmup_iterations,
        cfg.benchmark_iterations,
        verbose=verbose,
        names=names,
    )


def _targeted(
    adapters: list[BaseFrameworkAdapter],
    cfg: SuiteConfig,
    names: Collection[str] | None,
    verbose: bool,
) -> list[BenchmarkResult]:
    # Eager-only, direct calls: no adapter, so no streaming twin.
    from benchmarks.scenarios.targeted import run_all_targeted

    return run_all_targeted(
        cfg.image_counts,
        cfg.image_sizes,
        cfg.warmup_iterations,
        cfg.benchmark_iterations,
        names=names,
        verbose=verbose,
    )


def _zero_copy(
    adapters: list[BaseFrameworkAdapter],
    cfg: SuiteConfig,
    names: Collection[str] | None,
    verbose: bool,
) -> list[BenchmarkResult]:
    # zero_copy has its own hardcoded matrix and no adapter arg; its results
    # are polars-cv only, which is exactly what we want. They are also its
    # own record type, so they are converted here rather than appended raw —
    # appending them raw is what made this scenario die in `_aggregate_best`
    # every time it was selected.
    from benchmarks.scenarios.zero_copy_ingestion import (
        run_benchmarks as run_zero_copy,
    )
    from benchmarks.scenarios.zero_copy_ingestion import to_suite_results

    return to_suite_results(run_zero_copy())


def _remote(
    adapters: list[BaseFrameworkAdapter],
    cfg: SuiteConfig,
    names: Collection[str] | None,
    verbose: bool,
) -> list[BenchmarkResult]:
    # The remote fetch path (`file_path` over http/s3/gs/az). Also its own
    # matrix: it serves a corpus over loopback HTTP, so it has no adapter
    # arg and no external dependency. Opt-in, like zero_copy.
    from benchmarks.scenarios.remote_source import (
        run_benchmarks as run_remote_source,
    )

    return run_remote_source(
        cfg.image_counts,
        cfg.image_sizes,
        warmup_iterations=cfg.warmup_iterations,
        # The remote path is dominated by fetch, not by the ops, so it needs
        # far fewer repetitions than a compute scenario to settle — and each
        # one costs a full round of connections.
        benchmark_iterations=max(3, cfg.benchmark_iterations // 3),
        verbose=verbose,
    )


_RUNNERS = {
    "single_ops": _single_ops,
    "pipelines": _pipelines,
    "e2e": _e2e,
    "targeted": _targeted,
    "zero_copy": _zero_copy,
    "remote": _remote,
}
if set(_RUNNERS) != set(ALL_SCENARIOS):
    _msg = f"runners {sorted(_RUNNERS)} != scenarios {sorted(ALL_SCENARIOS)}"
    raise RuntimeError(_msg)


def _run_once(
    adapters: list[BaseFrameworkAdapter], cfg: SuiteConfig, *, quiet: bool
) -> list[BenchmarkResult]:
    """One full pass over the selected cases, in suite order."""
    sel = cfg.selection
    results: list[BenchmarkResult] = []
    for scenario in sel.scenarios():
        results += _RUNNERS[scenario](adapters, cfg, sel.cases(scenario), not quiet)

    # Every scenario must hand back the suite's own record. A scenario with its
    # own result type that skips the conversion above would otherwise fail in
    # `_aggregate_best` with an AttributeError naming a field, several frames
    # from the scenario that produced it — which is exactly how the `zero_copy`
    # breakage read for as long as it existed.
    from benchmarks.frameworks import BenchmarkResult as _SuiteResult

    foreign = {type(r).__name__ for r in results if not isinstance(r, _SuiteResult)}
    if foreign:
        msg = (
            f"scenario(s) returned {sorted(foreign)} rather than "
            f"benchmarks.frameworks.BenchmarkResult; convert at the call site "
            f"in its _RUNNERS entry (see zero_copy_ingestion.to_suite_results)"
        )
        raise TypeError(msg)
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


def is_debug_build() -> bool:
    """Whether the imported extension is unoptimised (``debug_assertions``)."""
    from polars_cv import _lib

    return bool(_lib.__debug_assertions__)


def run_suite(
    cfg: SuiteConfig, *, quiet: bool = True, allow_debug_build: bool = False
) -> list[BenchmarkResult]:
    """Run ``cfg.selection`` ``cfg.suite_repeats`` times; best-of per result.

    A debug extension is refused unless ``allow_debug_build`` (a smoke test of
    the harness itself): its numbers look plausible and measure nothing, and
    the results are marked so ``compare`` refuses them too.
    """
    if is_debug_build() and not allow_debug_build:
        msg = (
            "the imported polars-cv extension is a debug build; benchmark an "
            "optimised one: `maturin develop --profile benchmark` (thin LTO) or "
            "`--release`. Pass --allow-debug-build only to smoke-test the harness."
        )
        raise SystemExit(msg)
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
        # `compare` refuses results whose build was unoptimised.
        "debug_build": is_debug_build(),
        # Pass this back as `--select` to run the same cases on the other side.
        "selection": cfg.selection.render(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_threads": num_threads,
        "adapters": POLARS_CV_ADAPTERS,
        "config": {
            "image_counts": cfg.image_counts,
            "image_sizes": [list(s) for s in cfg.image_sizes],
            "warmup_iterations": cfg.warmup_iterations,
            "benchmark_iterations": cfg.benchmark_iterations,
            "suite_repeats": cfg.suite_repeats,
        },
    }
    out_path.with_suffix(out_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2)
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the polars-cv regression suite.")
    parser.add_argument("--out", required=True, help="output results JSON path")
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--select",
        help=(
            "comma-separated `scenario[:glob]` selectors, e.g. "
            "`pipelines,single_ops:rotate_*,targeted:geom_*` "
            f"(default: {DEFAULT_SELECTION})"
        ),
    )
    which.add_argument(
        "--changed",
        metavar="REF",
        help=(
            "select the cases the files changed since REF's merge base can move "
            "(benchmarks/regression/relevance.py); run the base side with the "
            "`selection` this writes to the .meta.json, via --select"
        ),
    )
    parser.add_argument("--counts", help="comma-separated image counts override")
    parser.add_argument("--sizes", help="comma-separated square sizes override")
    parser.add_argument("--threads", type=int, default=DEFAULT.num_threads)
    parser.add_argument("--repeats", type=int, default=DEFAULT.suite_repeats)
    parser.add_argument("--warmup", type=int, default=DEFAULT.warmup_iterations)
    parser.add_argument("--iterations", type=int, default=DEFAULT.benchmark_iterations)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--allow-debug-build",
        action="store_true",
        help="run against a debug extension (harness smoke test; compare refuses it)",
    )
    return parser.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace) -> SuiteConfig:
    from benchmarks.regression import relevance, selection

    try:
        if args.changed:
            changed = relevance.changed_files(args.changed)
            sel, unbenchmarked = relevance.select_for(changed)
            if unbenchmarked:
                print(f"not measured by any scenario: {unbenchmarked}", file=sys.stderr)
            print(f"--changed {args.changed} selects: {sel.render() or '(nothing)'}")
        else:
            sel = selection.parse(args.select or DEFAULT_SELECTION)
    except ValueError as e:
        raise SystemExit(str(e)) from e
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
        selection=sel,
        num_threads=args.threads,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _pin_threads(args.threads)
    cfg = _cfg_from_args(args)
    if not cfg.selection:
        print("Nothing to benchmark: no selected case executes the changed code.")
        return 0

    # Imported here, after _pin_threads, so the thread env is set first.
    from benchmarks.utils.results import ResultsCollector

    results = run_suite(cfg, quiet=args.quiet, allow_debug_build=args.allow_debug_build)
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
