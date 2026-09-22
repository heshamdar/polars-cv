"""One (engine, threads) cell of the matrix, run in its own process.

Why a subprocess
----------------

Two independent reasons, and either alone would force it:

1. **Thread pinning must precede the first polars import.** `POLARS_MAX_THREADS`
   sizes the pool when polars is first imported and is inert afterwards. One
   process therefore cannot host two pool sizes, so a matrix with more than one
   thread count cannot be a loop inside one interpreter.
2. **Exact peak memory.** `resource.getrusage(RUSAGE_SELF).ru_maxrss` is a
   kernel-maintained high-water mark — no sampler thread, no GIL contention, no
   10 ms aliasing, and no dependency on psutil. It never decreases within a
   process, which is exactly why it has to be scoped to a process that runs one
   cell and then exits.

What the cells are, and why these three
---------------------------------------

The plugin does not parallelise *within* a call: `CompiledGraph::execute` is a
plain sequential `for row_idx in 0..len` loop, and there is no rayon anywhere in
either Rust crate. All multi-core execution comes from the polars **streaming**
engine slicing the input into morsels and invoking the plugin concurrently.

That single fact sets the matrix:

- **eager@1** — one plugin call, the whole column, one thread. Measures the row
  loop, the per-row `ViewExpr` re-planning, decode/encode, and the
  `Vec::with_capacity(n_rows)` buffering of every row result before the output
  column is built. Blind to anything concurrent.
- **streaming@1** — per-call fixed cost with parallelism held out: the
  compiled-graph cache lookup, `prefetch_remote_sources`, and
  `resolve_auto_source_formats`, all of which run once per morsel rather than
  once per column. Eager amortises them over the whole column and hides them.
- **streaming@N** — morsel scaling, and the only cell that exercises the
  concurrent path through the global `Mutex` around the compiled-graph cache.

**eager@N is deliberately absent.** Given the sequential row loop it is
byte-for-byte eager@1; running it would spend time to produce a duplicate. Its
absence is a consequence of the execution model, not an oversight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: Thread specification meaning "whatever the machine offers".
AUTO = "auto"


def pin_threads(threads: int | str) -> None:
    """Pin the thread pool, or leave it to the machine for :data:`AUTO`.

    MUST run before polars is imported. The caller is a fresh process whose
    first statement is this, which is the only way to make that ordering a
    property of the design rather than a comment someone has to honour.

    Note this pins the pool the **streaming** engine spreads morsels across.
    Eager throughput does not move with it — the row loop is sequential — which
    is why the eager cell is pinned to 1 and never re-run at other widths.
    """
    if threads == AUTO:
        # Deliberately unset rather than set to a detected value: polars' own
        # default is the fact we want to measure, and a detected value we
        # computed differently would be a second authority on it.
        for var in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ.pop(var, None)
        return

    for var in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[var] = str(threads)


def peak_rss_mb() -> float:
    """This process's peak RSS, from the kernel.

    `ru_maxrss` is in kilobytes on Linux and bytes on macOS — a platform
    difference that silently produces a 1024x error if assumed either way.
    """
    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024 if sys.platform != "darwin" else raw / (1024 * 1024)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=["eager", "streaming"])
    parser.add_argument("--threads", required=True, help=f"an integer or {AUTO!r}")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--scenarios", required=True, help="comma-separated")
    parser.add_argument("--counts", required=True, help="comma-separated")
    parser.add_argument("--sizes", required=True, help="comma-separated square sizes")
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    threads: int | str = AUTO if args.threads == AUTO else int(args.threads)

    # Before any import that reaches polars. Everything below is deferred.
    pin_threads(threads)

    import polars as pl

    from benchmarks.frameworks import get_adapter
    from benchmarks.regression.config import SuiteConfig
    from benchmarks.regression.run_suite import run_scenarios

    adapter = get_adapter(f"polars-cv-{args.engine}")
    if not adapter.is_available():
        print(
            f"adapter polars-cv-{args.engine} is not available — is polars-cv "
            f"built? Run `maturin develop --release` first.",
            file=sys.stderr,
        )
        return 2

    pool_size = pl.thread_pool_size()
    if threads != AUTO and pool_size != threads:
        # The env var is a *request*; the pool size is the fact. They diverge
        # whenever pinning happens after the first polars import, and a run
        # that silently used a different width than it reports is worse than
        # no run.
        print(
            f"requested {threads} thread(s) but polars built a pool of "
            f"{pool_size}. Pinning did not take effect — something imported "
            f"polars before pin_threads ran.",
            file=sys.stderr,
        )
        return 3

    cfg = SuiteConfig(
        image_counts=[int(c) for c in args.counts.split(",")],
        image_sizes=[(int(s), int(s)) for s in args.sizes.split(",")],
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        scenarios=tuple(s.strip() for s in args.scenarios.split(",") if s.strip()),
    )
    results = run_scenarios([adapter], cfg, quiet=args.quiet)

    payload = {
        "engine": args.engine,
        "threads_requested": args.threads,
        "thread_pool_size": pool_size,
        "peak_rss_mb": peak_rss_mb(),
        "results": [_as_dict(r) for r in results],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    return 0


def _as_dict(result: object) -> dict:
    from dataclasses import asdict

    return asdict(result)  # type: ignore[call-overload]


if __name__ == "__main__":
    raise SystemExit(main())
