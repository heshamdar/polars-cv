"""Run the regression matrix and write a results JSON with its provenance.

Drives `benchmarks.regression.config.DEFAULT_CELLS` — eager@1, streaming@1 and
streaming@auto — each in **its own subprocess** (`benchmarks.regression.cell`),
because `POLARS_MAX_THREADS` sizes the pool at the first polars import and is
inert afterwards. One interpreter cannot host two thread counts, so the matrix
cannot be a loop in here.

That also buys exact per-cell peak memory: a process that runs one cell and
exits reports a kernel-maintained `ru_maxrss` high-water mark, with no sampler
thread and no psutil.

This module no longer pins threads itself. It used to, from `main()`, while its
own docstring claimed it happened at import time and that `setdefault` let an
explicit env override win — neither of which the code did. Pinning now lives in
the child's first statement, where the ordering is a property of the design
rather than a comment someone has to honour, and the child *verifies* the pool
it actually got before measuring anything.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.frameworks import BaseFrameworkAdapter, BenchmarkResult

from benchmarks.regression.config import (
    ALL_SCENARIOS,
    AUTO,
    DEFAULT,
    SCHEMA_VERSION,
    Cell,
    SuiteConfig,
    result_key,
)


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


def run_scenarios(
    adapters: list[BaseFrameworkAdapter], cfg: SuiteConfig, *, quiet: bool
) -> list[BenchmarkResult]:
    """One full pass over the configured scenarios, for the given adapters.

    Called by `benchmarks.regression.cell` inside a thread-pinned subprocess;
    it is the only entry point that actually runs a scenario.
    """
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
        # are polars-cv only, which is exactly what we want. They are also its
        # own record type, so they are converted here rather than appended raw —
        # appending them raw is what made this scenario die in `_aggregate_best`
        # every time it was selected.
        from benchmarks.scenarios.zero_copy_ingestion import (
            run_benchmarks as run_zero_copy,
        )
        from benchmarks.scenarios.zero_copy_ingestion import to_suite_results

        results += to_suite_results(run_zero_copy())
    if "remote" in cfg.scenarios:
        # The remote fetch path (`file_path` over http/s3/gs/az). Also its own
        # matrix: it serves a corpus over loopback HTTP, so it has no adapter
        # arg and no external dependency. Opt-in, like zero_copy.
        from benchmarks.scenarios.remote_source import (
            run_benchmarks as run_remote_source,
        )

        results += run_remote_source(
            cfg.image_counts,
            cfg.image_sizes,
            warmup_iterations=cfg.warmup_iterations,
            # The remote path is dominated by fetch, not by the ops, so it needs
            # far fewer repetitions than a compute scenario to settle — and each
            # one costs a full round of connections.
            benchmark_iterations=max(3, cfg.benchmark_iterations // 3),
            verbose=verbose,
        )

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
            f"in _run_once (see zero_copy_ingestion.to_suite_results)"
        )
        raise TypeError(msg)
    return results


def _result_key(r: BenchmarkResult) -> tuple:
    """The identity of a measurement, via the one definition in `config`."""
    from dataclasses import asdict

    return result_key(asdict(r))


def _aggregate_best(runs: list[list[BenchmarkResult]]) -> list[BenchmarkResult]:
    """Reduce repeated runs to one best-of result per key.

    Best-of = the repeat with the highest throughput (and its matching latency
    / time); peak memory is the median across repeats, and `None` when no
    repeat measured it — never 0.0, which would read as a measurement of
    nothing.
    """
    by_key: dict[tuple, list[BenchmarkResult]] = {}
    for run in runs:
        for r in run:
            by_key.setdefault(_result_key(r), []).append(r)

    aggregated: list[BenchmarkResult] = []
    for results in by_key.values():
        best = max(results, key=lambda r: r.throughput_images_per_second)
        measured = [r.peak_memory_mb for r in results if r.peak_memory_mb is not None]
        deltas = [
            r.peak_memory_delta_mb
            for r in results
            if r.peak_memory_delta_mb is not None
        ]
        aggregated.append(
            replace(
                best,
                peak_memory_mb=statistics.median(measured) if measured else None,
                peak_memory_delta_mb=statistics.median(deltas) if deltas else None,
            )
        )
    aggregated.sort(key=lambda r: tuple(map(str, _result_key(r))))
    return aggregated


def _run_cell(cell: Cell, cfg: SuiteConfig, *, quiet: bool) -> dict:
    """Run one cell in a fresh, thread-pinned subprocess and return its payload.

    Failure is loud. A cell that dies, or that got a different thread pool than
    it asked for, must not silently contribute nothing: an absent cell compares
    as MISSING at worst and as "no regression here" at best, and both are
    worse than a failed run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{cell.engine}-{cell.threads}.json"
        argv = [
            sys.executable,
            "-m",
            "benchmarks.regression.cell",
            "--engine",
            cell.engine,
            "--threads",
            str(cell.threads),
            "--out",
            str(out),
            "--scenarios",
            ",".join(cfg.scenarios),
            "--counts",
            ",".join(str(c) for c in cfg.image_counts),
            "--sizes",
            ",".join(str(w) for w, _ in cfg.image_sizes),
            "--warmup",
            str(cfg.warmup_iterations),
            "--iterations",
            str(cfg.benchmark_iterations),
        ]
        if quiet:
            argv.append("--quiet")

        proc = subprocess.run(argv, text=True, capture_output=quiet)
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip() or f"exit {proc.returncode}"
            msg = f"cell {cell.name} failed: {detail}"
            raise RuntimeError(msg)
        return json.loads(out.read_text())


def run_suite(
    cfg: SuiteConfig, *, quiet: bool = True
) -> tuple[list[BenchmarkResult], dict]:
    """Run every cell `suite_repeats` times; return results and cell provenance."""
    from benchmarks.utils.timing import result_from_dict

    runs: list[list[BenchmarkResult]] = []
    cell_info: dict[str, dict] = {}

    for _ in range(cfg.suite_repeats):
        this_run: list[BenchmarkResult] = []
        for cell in cfg.cells:
            payload = _run_cell(cell, cfg, quiet=quiet)
            this_run.extend(result_from_dict(r) for r in payload["results"])
            # Peak RSS is the max across repeats: it is a high-water mark, so
            # the largest is the true one rather than an average of watermarks.
            prior = cell_info.get(cell.name, {}).get("peak_rss_mb", 0.0)
            cell_info[cell.name] = {
                "engine": payload["engine"],
                "threads_requested": payload["threads_requested"],
                "thread_pool_size": payload["thread_pool_size"],
                "peak_rss_mb": max(prior, payload["peak_rss_mb"]),
            }
        runs.append(this_run)

    return _aggregate_best(runs), cell_info


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


def _cpu_model() -> str | None:
    """CPU model from /proc/cpuinfo, or None off Linux."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def build_meta(cfg: SuiteConfig, cell_info: dict) -> dict:
    """Everything needed to decide whether two runs may be compared.

    This used to be a `.meta.json` sidecar that `compare.py` never read, and it
    recorded neither the build profile nor `POLARS_CV_OPTIMIZATIONS` — so a
    debug-vs-release comparison, or one across different optimization settings,
    printed PASS or FAIL without noticing. It now rides inside the results file,
    because a sidecar can be renamed, lost, or dropped by a partial copy, which
    is one more way for provenance to go missing quietly.
    """
    import polars as pl

    import polars_cv

    build = polars_cv.build_info()
    return {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_info": build,
        # A debug extension is several times slower than a release one; mixing
        # the two reads as a catastrophic regression rather than as a mistake.
        "build_profile": _build_profile(),
        "polars_version": pl.__version__,
        # Silently changes the physical graph *and* the compiled-graph cache
        # key, and was recorded nowhere.
        "polars_cv_optimizations": os.environ.get("POLARS_CV_OPTIMIZATIONS"),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        # `sched_getaffinity` is the number this process may actually use;
        # `cpu_count` is the machine's. Under a cgroup they differ, and the
        # first is the one a throughput number should be read against.
        "available_parallelism": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "cells": cell_info,
        "config": {
            "image_counts": cfg.image_counts,
            "image_sizes": [list(s) for s in cfg.image_sizes],
            "warmup_iterations": cfg.warmup_iterations,
            "benchmark_iterations": cfg.benchmark_iterations,
            "suite_repeats": cfg.suite_repeats,
            "scenarios": list(cfg.scenarios),
            "cells": [[c.engine, c.threads] for c in cfg.cells],
        },
    }


def _build_profile() -> str:
    """ "release", "debug", or "unknown" for the loaded extension.

    Read from the compiled artifact's path rather than asserted: `maturin
    develop` writes the same filename either way, so the only honest source is
    where cargo put the object it copied.
    """
    import polars_cv

    lib = Path(polars_cv.__file__).parent / "_lib.abi3.so"
    if not lib.exists():
        return "unknown"
    # maturin copies the artifact in, so mtime/size cannot distinguish. Fall
    # back to the workspace target dir, which is what `--release` changes.
    root = Path(__file__).resolve().parents[3]
    release = root / "target" / "release"
    debug = root / "target" / "debug"
    stamp = lib.stat().st_mtime
    candidates = {
        "release": (release / "libpolars_cv.so"),
        "debug": (debug / "libpolars_cv.so"),
    }
    best, best_delta = "unknown", None
    for profile, path in candidates.items():
        if not path.exists():
            continue
        delta = abs(path.stat().st_mtime - stamp)
        if best_delta is None or delta < best_delta:
            best, best_delta = profile, delta
    # Within a couple of seconds means maturin copied that one.
    return best if best_delta is not None and best_delta < 5 else "unknown"


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
    parser.add_argument(
        "--cells",
        help=(
            "comma-separated engine@threads cells, e.g. "
            "'eager@1,streaming@1,streaming@auto' (default: all three)"
        ),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT.suite_repeats)
    parser.add_argument("--warmup", type=int, default=DEFAULT.warmup_iterations)
    parser.add_argument("--iterations", type=int, default=DEFAULT.benchmark_iterations)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _parse_cells(spec: str) -> tuple[Cell, ...]:
    cells = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "@" not in item:
            msg = f"cell {item!r} must be engine@threads, e.g. streaming@auto"
            raise SystemExit(msg)
        engine, _, threads = item.partition("@")
        if engine not in {"eager", "streaming"}:
            msg = f"cell {item!r}: engine must be 'eager' or 'streaming'"
            raise SystemExit(msg)
        if engine == "eager" and threads != "1":
            # Not a style rule. The row loop is sequential, so eager@N is
            # byte-for-byte eager@1; accepting it would invite a duplicate
            # measurement to be read as a scaling result.
            msg = (
                f"cell {item!r}: eager execution is single-threaded by "
                f"construction (one plugin call, sequential row loop, no "
                f"rayon), so only eager@1 is meaningful. Use streaming@{threads} "
                f"to measure scaling."
            )
            raise SystemExit(msg)
        cells.append(Cell(engine, threads if threads == AUTO else int(threads)))
    return tuple(cells)


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
        cells=_parse_cells(args.cells) if args.cells else DEFAULT.cells,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _cfg_from_args(args)

    results, cell_info = run_suite(cfg, quiet=args.quiet)
    if not results:
        print("ERROR: suite produced no results.", file=sys.stderr)
        return 2

    from dataclasses import asdict

    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": build_meta(cfg, cell_info),
        "results": [asdict(r) for r in results],
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2))

    cells = ", ".join(
        f"{name} (pool={info['thread_pool_size']})"
        for name, info in sorted(cell_info.items())
    )
    print(f"Wrote {len(results)} results to {out_path}. Cells: {cells}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
