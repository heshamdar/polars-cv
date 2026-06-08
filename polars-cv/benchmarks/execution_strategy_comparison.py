"""Benchmark comparing FullImage vs Tiled execution strategies.

Measures the throughput and latency of running image pipelines under
``set_execution_strategy("full")`` vs ``set_execution_strategy("tiled")``
across different image sizes and pipeline configurations.

Uses the blob/VIEW-protocol source to isolate compute time from PNG decode
overhead (same approach as batch_throughput.py).

Key questions being answered:
  1. At what image size does tiling start to win?
  2. Which pipeline shapes benefit most (single op vs multi-op segments)?
  3. Does a global barrier (resize) diminish the tiling benefit?

Usage:
    cd polars-cv
    uv run python -m benchmarks.execution_strategy_comparison
    uv run python -m benchmarks.execution_strategy_comparison --sizes 512,1024,2048
    uv run python -m benchmarks.execution_strategy_comparison --count 30 --repeats 5 --quiet
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from dataclasses import dataclass

import numpy as np
import polars as pl

import polars_cv
import polars_cv.expressions  # noqa: F401  (registers .cv namespace)
from polars_cv import Pipeline


# ── Data preparation ──────────────────────────────────────────────────────────

def _make_blobs(count: int, h: int, w: int) -> list[bytes]:
    """Encode ``count`` random RGB images as VIEW-protocol blobs."""
    rng = np.random.default_rng(42)
    arrays = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(count)]
    df = pl.DataFrame(
        {"img": arrays}, schema={"img": pl.Array(pl.UInt8, (h, w, 3))}
    )
    return (
        df.select(blob=pl.col("img").cv.pipe(Pipeline().source("array")).sink("blob"))
        ["blob"]
        .to_list()
    )


# ── Pipeline definitions ──────────────────────────────────────────────────────

PIPELINES: dict[str, Pipeline] = {}  # populated below per-size


def _pipelines(h: int, w: int) -> dict[str, Pipeline]:
    """Return pipeline configs for images of size h×w."""
    base = Pipeline().source("blob")
    target = max(64, min(h, w) // 2)
    return {
        "single_op:grayscale": base.grayscale(),
        "single_op:blur_sigma2": base.blur(sigma=2.0),
        "2op_pointwise:scale+clamp": base.scale(0.5).clamp(0.0, 1.0),
        "3op_segment:gray+blur+threshold": base.grayscale().blur(sigma=2.0).threshold(128.0),
        "5op_segment:scale+relu+gray+blur+clamp": (
            base.scale(0.5).grayscale().blur(sigma=1.5).clamp(0.0, 1.0)
        ),
        "global_barrier:gray+resize+scale": (
            base.grayscale()
            .resize(height=target, width=target)
            .scale(0.5)
        ),
    }


# ── Timing ────────────────────────────────────────────────────────────────────

def _run(df: pl.DataFrame, expr: pl.Expr, streaming: bool) -> float:
    gc.collect()
    t0 = time.perf_counter()
    if streaming:
        df.lazy().select(out=expr).collect(engine="streaming")
    else:
        df.select(out=expr)
    return time.perf_counter() - t0


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Result:
    size: str
    pipeline: str
    strategy: str
    mode: str          # "eager" | "streaming"
    best_s: float
    median_s: float
    count: int

    @property
    def rows_per_s(self) -> float:
        return self.count / self.best_s

    @property
    def ms_per_image(self) -> float:
        return self.best_s / self.count * 1000.0


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_comparison(
    sizes: list[int],
    count: int,
    repeats: int,
    warmup: int,
    streaming_only: bool,
    quiet: bool,
) -> list[Result]:
    strategies = [("full", "full_image"), ("tiled", "tiled")]
    modes = ["streaming"] if streaming_only else ["eager", "streaming"]
    results: list[Result] = []

    for sz in sizes:
        size_label = f"{sz}x{sz}"
        size_bytes = sz * sz * 3  # RGB u8
        size_kb = size_bytes / 1024

        if not quiet:
            print(f"\n{'─'*70}")
            print(f"  Image size: {size_label}  ({size_kb:.0f} KB)  count={count}")
            print(f"{'─'*70}")

        blobs = _make_blobs(count, sz, sz)
        df = pl.DataFrame({"blob": blobs})
        pipes = _pipelines(sz, sz)

        for pipe_name, pipe in pipes.items():
            expr = pl.col("blob").cv.pipe(pipe).sink("blob")

            for strategy_name, strategy_label in strategies:
                polars_cv.set_execution_strategy(strategy_name)

                for mode in modes:
                    streaming = mode == "streaming"
                    # Warm-up
                    for _ in range(warmup):
                        _run(df, expr, streaming)

                    times = [_run(df, expr, streaming) for _ in range(repeats)]
                    best = min(times)
                    med = statistics.median(times)

                    results.append(Result(
                        size=size_label,
                        pipeline=pipe_name,
                        strategy=strategy_label,
                        mode=mode,
                        best_s=best,
                        median_s=med,
                        count=count,
                    ))

                    if not quiet:
                        print(
                            f"  {pipe_name:<42}  {strategy_label:<11}  {mode:<9}"
                            f"  {count/best:8.1f} rows/s  "
                            f"({best*1e3:.1f}ms best)"
                        )

    # Restore default
    polars_cv.set_execution_strategy("adaptive")
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_speedup_table(results: list[Result], mode: str) -> None:
    """Print a speedup matrix: rows = pipelines × sizes, col = tiled/full ratio."""
    import itertools

    subset = [r for r in results if r.mode == mode]
    if not subset:
        return

    sizes = list(dict.fromkeys(r.size for r in subset))
    pipes = list(dict.fromkeys(r.pipeline for r in subset))

    # Map (size, pipe, strategy) → best_s
    lookup: dict[tuple[str, str, str], float] = {
        (r.size, r.pipeline, r.strategy): r.best_s for r in subset
    }

    # Header
    col_w = 12
    header = f"{'Pipeline':<44}" + "".join(f"{s:>{col_w}}" for s in sizes)
    print(f"\n  Speedup: tiled / full_image  [{mode}]  (>1.0 = tiling wins)")
    print("  " + "─" * len(header))
    print("  " + header)
    print("  " + "─" * len(header))

    for pipe in pipes:
        row = f"{pipe:<44}"
        for sz in sizes:
            full = lookup.get((sz, pipe, "full_image"))
            tiled = lookup.get((sz, pipe, "tiled"))
            if full and tiled:
                ratio = full / tiled  # >1 means tiled was faster
                marker = " ▲" if ratio > 1.05 else (" ▼" if ratio < 0.95 else "  ")
                row += f"{ratio:>{col_w-2}.2f}×{marker}"
            else:
                row += f"{'n/a':>{col_w}}"
        print("  " + row)

    print("  " + "─" * len(header))
    print("  ▲ tiling >5% faster   ▼ tiling >5% slower")


def _print_throughput_table(results: list[Result], mode: str) -> None:
    """Print rows/s for each (size, pipeline, strategy)."""
    subset = [r for r in results if r.mode == mode]
    if not subset:
        return

    sizes = list(dict.fromkeys(r.size for r in subset))
    pipes = list(dict.fromkeys(r.pipeline for r in subset))
    strategies = list(dict.fromkeys(r.strategy for r in subset))

    lookup: dict[tuple[str, str, str], float] = {
        (r.size, r.pipeline, r.strategy): r.rows_per_s for r in subset
    }

    col_w = 14
    strat_cols = len(strategies) * col_w
    header = f"{'Pipeline':<44}"
    for sz in sizes:
        header += f"{sz:^{strat_cols}}"
    print(f"\n  Throughput (rows/s)  [{mode}]")
    sub = "  " + " " * 44
    for _ in sizes:
        for s in strategies:
            sub += f"{s:>{col_w}}"
    print(sub)
    print("  " + "─" * len(sub.lstrip()))

    for pipe in pipes:
        row = f"{pipe:<44}"
        for sz in sizes:
            for strat in strategies:
                v = lookup.get((sz, pipe, strat))
                row += f"{v:>{col_w}.0f}" if v else f"{'n/a':>{col_w}}"
        print("  " + row)

    print("  " + "─" * len(sub.lstrip()))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="256,512,1024,2048",
                        help="Comma-separated image side lengths (default: 256,512,1024,2048)")
    parser.add_argument("--count", type=int, default=50,
                        help="Images per benchmark run (default: 50)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Timing repetitions per configuration (default: 5)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warm-up iterations before timing (default: 2)")
    parser.add_argument("--streaming-only", action="store_true",
                        help="Only test streaming engine (skip eager)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-run progress lines")
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    print("=" * 70)
    print("  Execution Strategy Comparison: FullImage vs Tiled (segment-level)")
    print("=" * 70)
    print(f"  Sizes:    {sizes}")
    print(f"  Count:    {args.count} images per run")
    print(f"  Repeats:  {args.repeats}  Warmup: {args.warmup}")
    print(f"  Note:     'tiled' = Tiled{{tile_size=256, threshold_bytes=0}}")
    print(f"            (always tiles, regardless of image size)")
    print()

    results = run_comparison(
        sizes=sizes,
        count=args.count,
        repeats=args.repeats,
        warmup=args.warmup,
        streaming_only=args.streaming_only,
        quiet=args.quiet,
    )

    modes = ["streaming"] if args.streaming_only else ["eager", "streaming"]
    for mode in modes:
        _print_speedup_table(results, mode)
        _print_throughput_table(results, mode)

    print()
    print("  Interpretation guide:")
    print("   • Speedup > 1.0 at 1024×1024+ on multi-op segments → tiling wins")
    print("   • Speedup ≈ 1.0 for single ops → overhead ≈ benefit")
    print("   • Speedup < 1.0 at small sizes → tile overhead > cache benefit")
    print("   • global_barrier pipeline: smaller gain (resize dominates, breaks segment)")


if __name__ == "__main__":
    main()
