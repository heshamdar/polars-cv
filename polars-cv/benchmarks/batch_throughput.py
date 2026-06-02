"""Batch-throughput microbenchmark for the per-row execution hot loop.

This isolates the *orchestration* cost of running a pipeline over many rows
(decode -> ops -> encode), as opposed to the cost of the image kernels
themselves. It is the artifact used to demonstrate the per-row decode
optimizations and to compare eager vs streaming execution.

It deliberately uses the ``blob`` (VIEW-protocol) source so that the in-memory
zero-copy decode path is exercised on every row, rather than PNG decode time
dominating the measurement.

Usage:
    python -m benchmarks.batch_throughput
    python -m benchmarks.batch_throughput --count 5000 --size 128 --ops resize,grayscale
    python -m benchmarks.batch_throughput --count 2000 --size 256 --repeats 5
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import numpy as np
import polars as pl

import polars_cv.expressions  # noqa: F401  (registers the `.cv` namespace)
from polars_cv import Pipeline


def _peak_rss_mb() -> float:
    """Best-effort peak resident-set size in MiB."""
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        import sys

        return kb / 1024.0 if sys.platform != "darwin" else kb / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def _make_blobs(count: int, size: int) -> list[bytes]:
    """Create ``count`` random RGB images encoded as VIEW-protocol blobs."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(count):
        arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        rows.append(arr)
    df = pl.DataFrame({"img": rows}, schema={"img": pl.Array(pl.UInt8, (size, size, 3))})
    blobs = df.select(
        blob=pl.col("img").cv.pipe(Pipeline().source("array")).sink("blob")
    )["blob"].to_list()
    return blobs


def _build_pipeline(ops: list[str], size: int) -> Pipeline:
    pipe = Pipeline().source("blob")
    target = max(1, size // 2)
    for op in ops:
        if op == "resize":
            pipe = pipe.resize(height=target, width=target)
        elif op == "grayscale":
            pipe = pipe.grayscale()
        elif op == "blur":
            pipe = pipe.blur(sigma=1.5)
        elif op == "threshold":
            pipe = pipe.grayscale().threshold(128)
        else:
            raise ValueError(f"Unknown op '{op}'")
    return pipe


def _run(df: pl.DataFrame, expr: pl.Expr, streaming: bool) -> float:
    """Return wall-clock seconds for one full collect over the batch."""
    gc.collect()
    start = time.perf_counter()
    if streaming:
        df.lazy().select(out=expr).collect(engine="streaming")
    else:
        df.select(out=expr)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000, help="number of rows")
    parser.add_argument("--size", type=int, default=256, help="image side length")
    parser.add_argument("--ops", type=str, default="resize,grayscale")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    print(
        f"batch_throughput: count={args.count} size={args.size}x{args.size} "
        f"ops={'+'.join(ops)} repeats={args.repeats}"
    )

    blobs = _make_blobs(args.count, args.size)
    df = pl.DataFrame({"blob": blobs})
    pipe = _build_pipeline(ops, args.size)

    rss_before = _peak_rss_mb()
    for streaming in (False, True):
        expr = pl.col("blob").cv.pipe(pipe).sink("blob")
        # Warm-up (build/registration, allocator warm).
        _run(df, expr, streaming)
        times = [_run(df, expr, streaming) for _ in range(args.repeats)]
        best = min(times)
        med = statistics.median(times)
        mode = "streaming" if streaming else "eager"
        print(
            f"  {mode:>9}: {args.count / best:10.1f} rows/s "
            f"(best {best * 1e3:7.2f} ms, median {med * 1e3:7.2f} ms)"
        )
    rss_after = _peak_rss_mb()
    print(f"  peak RSS: {rss_after:.1f} MiB (Δ {rss_after - rss_before:+.1f} MiB)")


if __name__ == "__main__":
    main()
