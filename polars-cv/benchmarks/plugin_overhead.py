"""
Plugin-overhead micro-benchmark: small buffers, many rows.

Measures the fixed per-call and per-row costs of the `vb_graph` plugin
(graph compilation, parameter resolution, op dispatch) rather than kernel
time: 64x64 u8 buffers are small enough that interpreter overhead is a
meaningful fraction of the work. This is the benchmark that motivated and
gates the compiled-graph cache + parameter slot binding work.

Run directly:

    uv run python benchmarks/plugin_overhead.py [--rows 100000] [--repeat 5]

Reports eager and streaming-engine timings for:
  - static:  blob source -> scale -> clamp -> relu     (all-literal params)
  - dynamic: blob source -> scale(pl.col) -> clamp -> relu (per-row param)
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import polars as pl

from polars_cv import Pipeline


def _make_df(rows: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    # VIEW-protocol blob round-trips without decode cost.
    blob = pl.DataFrame({"arr": [img.tolist()]}).with_columns(
        blob=pl.col("arr").cv.pipe(Pipeline().source("array", dtype="u8")).sink("blob")
    )["blob"][0]
    return pl.DataFrame(
        {
            "img": [blob] * rows,
            "factor": rng.uniform(0.5, 2.0, size=rows),
        }
    )


def _time(fn, repeat: int) -> tuple[float, float]:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    df = _make_df(args.rows)

    static_pipe = (
        Pipeline().source("blob").scale(1.5).clamp(min_val=0.0, max_val=255.0).relu()
    )
    dynamic_pipe = (
        Pipeline()
        .source("blob")
        .scale(pl.col("factor"))
        .clamp(min_val=0.0, max_val=255.0)
        .relu()
    )

    cases = {
        "static ": static_pipe,
        "dynamic": dynamic_pipe,
    }

    print(f"rows={args.rows} buffer=64x64 u8 repeat={args.repeat}")
    print(f"{'case':<22} {'best (s)':>10} {'median (s)':>11} {'rows/s':>12}")
    # Typed Array sink: exercises the tensor-output path (per-element
    # construction vs flat reshape) on top of the same kernel chain. The
    # source dtype assertion makes the f32 output plannable.
    cases["static->array"] = (
        Pipeline()
        .source("blob", dtype="u8")
        .scale(1.5)
        .clamp(min_val=0.0, max_val=255.0)
        .relu()
    )
    for name, pipe in cases.items():
        sink_args = ("array",) if name.endswith("array") else ("blob",)
        sink_kwargs = {"shape": [64, 64]} if name.endswith("array") else {}
        expr = pl.col("img").cv.pipe(pipe).sink(*sink_args, **sink_kwargs)

        def eager(expr=expr):
            df.with_columns(out=expr)

        def streaming(expr=expr):
            df.lazy().with_columns(out=expr).collect(engine="streaming")

        for mode, fn in (("eager", eager), ("streaming", streaming)):
            best, median = _time(fn, args.repeat)
            print(
                f"{name} {mode:<13} {best:>10.3f} {median:>11.3f} "
                f"{args.rows / best:>12.0f}"
            )


if __name__ == "__main__":
    main()
