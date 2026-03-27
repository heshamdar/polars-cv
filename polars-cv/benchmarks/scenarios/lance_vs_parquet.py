"""
Lance vs Parquet storage backend benchmarks for polars-cv CV workflows.

This module compares two data storage patterns for computer vision datasets:

  Parquet pattern
    ``metadata.parquet`` (file paths + labels) + separate image files on disk.
    polars-cv reads via ``Pipeline().source("file_path")``.

  Lance pattern
    ``dataset.lance`` — image bytes + metadata stored together in one file.
    polars-cv reads via ``Pipeline().source("image_bytes")`` after loading
    the binary column from Lance with ``pl.from_arrow(ds.to_table(...))``.

Three scenarios are measured:

  sequential_scan
    Simulate one full training epoch: load ALL images in order and run a
    resize pipeline on each.  Measures sustained read + processing throughput.

  random_access
    Simulate a training step: load a random batch of N images and process
    them.  This is where Lance's repetition-index-based random access
    (O(1-2 IO ops) per sample) should show the clearest advantage over
    Parquet's pattern of: read metadata → seek + read N individual files.

  e2e_pipeline
    Full ImageNet-style preprocessing (resize 256 → centre-crop 224 →
    normalise) with a ``"numpy"`` sink, matching real inference workloads.

Run standalone::

    cd polars-cv
    python -m benchmarks.scenarios.lance_vs_parquet

Or via the main CLI::

    python -m benchmarks.run_benchmarks --scenario lance_vs_parquet \\
        --counts 100,500 --sizes 224

Lance is an optional dependency — the module imports cleanly without it and
every public function raises a clear ``ImportError`` when Lance is absent.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from benchmarks.utils.data_gen import (
    LanceDataset,
    generate_imagefolder_dataset,
    generate_lance_dataset,
)
from benchmarks.utils.memory import run_timed_with_memory

if TYPE_CHECKING:
    pass

LANCE_AVAILABLE: bool = importlib.util.find_spec("lance") is not None

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StorageBackendResult:
    """Benchmark result for a storage backend comparison."""

    backend: str              # "parquet" | "lance"
    scenario: str             # "sequential_scan" | "random_access" | "e2e_pipeline"
    image_count: int          # dataset size (sequential/e2e) or dataset size (random)
    batch_size: int | None    # None for sequential/e2e; N for random access
    image_size: tuple[int, int]  # (width, height)
    total_time_seconds: float
    throughput_images_per_second: float
    latency_ms_per_image: float
    peak_memory_mb: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_lance() -> None:
    if not LANCE_AVAILABLE:
        msg = (
            "lance is not installed. "
            "Install it with: pip install lance pyarrow"
        )
        raise ImportError(msg)


def _build_resize_pipeline(source: str) -> "polars_cv.Pipeline":  # type: ignore[name-defined]  # noqa: F821
    """Return a minimal resize pipeline for the given source type."""
    import polars_cv
    return polars_cv.Pipeline().source(source).resize(height=224, width=224)


def _build_imagenet_pipeline(source: str) -> "polars_cv.Pipeline":  # type: ignore[name-defined]  # noqa: F821
    """Return an ImageNet-style preprocess pipeline (resize→crop→normalise)."""
    import polars_cv
    return (
        polars_cv.Pipeline()
        .source(source)
        .resize(height=256, width=256)
        .crop(top=16, left=16, height=224, width=224)
        .normalize(method="minmax")
    )


# ---------------------------------------------------------------------------
# Sequential scan benchmarks
# ---------------------------------------------------------------------------

def _run_parquet_sequential(
    dataset_path: Path,
    metadata_path: Path,
    warmup: int,
    iterations: int,
) -> StorageBackendResult:
    """Benchmark: Parquet metadata + file_path source — sequential scan."""
    pipe = _build_resize_pipeline("file_path")

    def run() -> None:
        df = pl.read_parquet(metadata_path)
        result = df.with_columns(
            processed=pl.col("path").cv.pipe(pipe).sink("blob")
        )
        _ = result["processed"].to_list()

    image_count = pl.read_parquet(metadata_path).height

    for _ in range(warmup):
        run()

    total_time = 0.0
    peak_mem = 0.0
    for _ in range(iterations):
        _, elapsed, mem = run_timed_with_memory(run)
        total_time += elapsed
        peak_mem = max(peak_mem, mem.peak_memory_mb)

    avg = total_time / iterations
    return StorageBackendResult(
        backend="parquet",
        scenario="sequential_scan",
        image_count=image_count,
        batch_size=None,
        image_size=(224, 224),
        total_time_seconds=avg,
        throughput_images_per_second=image_count / avg,
        latency_ms_per_image=avg / image_count * 1000,
        peak_memory_mb=peak_mem,
    )


def _run_lance_sequential(
    lance_dataset: LanceDataset,
    warmup: int,
    iterations: int,
) -> StorageBackendResult:
    """Benchmark: Lance inline bytes — sequential scan."""
    _require_lance()
    import lance

    pipe = _build_resize_pipeline("image_bytes")
    ds_path = str(lance_dataset.dataset_path)

    def run() -> None:
        ds = lance.dataset(ds_path)
        df = pl.from_arrow(ds.to_table(columns=["image_bytes", "label"]))
        result = df.with_columns(
            processed=pl.col("image_bytes").cv.pipe(pipe).sink("blob")
        )
        _ = result["processed"].to_list()

    image_count = lance_dataset.image_count

    for _ in range(warmup):
        run()

    total_time = 0.0
    peak_mem = 0.0
    for _ in range(iterations):
        _, elapsed, mem = run_timed_with_memory(run)
        total_time += elapsed
        peak_mem = max(peak_mem, mem.peak_memory_mb)

    avg = total_time / iterations
    return StorageBackendResult(
        backend="lance",
        scenario="sequential_scan",
        image_count=image_count,
        batch_size=None,
        image_size=(224, 224),
        total_time_seconds=avg,
        throughput_images_per_second=image_count / avg,
        latency_ms_per_image=avg / image_count * 1000,
        peak_memory_mb=peak_mem,
    )


# ---------------------------------------------------------------------------
# Random access benchmarks
# ---------------------------------------------------------------------------

def _run_parquet_random_access(
    metadata_path: Path,
    batch_size: int,
    image_count: int,
    warmup: int,
    iterations: int,
    rng: np.random.Generator,
) -> StorageBackendResult:
    """Benchmark: Parquet + file_path — random batch access."""
    pipe = _build_resize_pipeline("image_bytes")
    df_meta = pl.read_parquet(metadata_path)

    def run() -> None:
        indices = rng.choice(image_count, size=batch_size, replace=False)
        paths = df_meta["path"][indices.tolist()].to_list()
        # Read each file individually — this is the Parquet random-access pattern
        img_bytes = [Path(p).read_bytes() for p in paths]
        batch_df = pl.DataFrame({"image_bytes": img_bytes})
        result = batch_df.with_columns(
            processed=pl.col("image_bytes").cv.pipe(pipe).sink("blob")
        )
        _ = result["processed"].to_list()

    for _ in range(warmup):
        run()

    total_time = 0.0
    peak_mem = 0.0
    for _ in range(iterations):
        _, elapsed, mem = run_timed_with_memory(run)
        total_time += elapsed
        peak_mem = max(peak_mem, mem.peak_memory_mb)

    avg = total_time / iterations
    return StorageBackendResult(
        backend="parquet",
        scenario="random_access",
        image_count=image_count,
        batch_size=batch_size,
        image_size=(224, 224),
        total_time_seconds=avg,
        throughput_images_per_second=batch_size / avg,
        latency_ms_per_image=avg / batch_size * 1000,
        peak_memory_mb=peak_mem,
    )


def _run_lance_random_access(
    lance_dataset: LanceDataset,
    batch_size: int,
    warmup: int,
    iterations: int,
    rng: np.random.Generator,
) -> StorageBackendResult:
    """Benchmark: Lance ds.take() — random batch access.

    ``ds.take(indices)`` uses Lance's repetition index to seek directly to
    the required row offsets — O(1-2 IO ops) per sample regardless of
    dataset size.  This is the core Lance random-access advantage.
    """
    _require_lance()
    import lance

    pipe = _build_resize_pipeline("image_bytes")
    ds_path = str(lance_dataset.dataset_path)
    image_count = lance_dataset.image_count

    def run() -> None:
        ds = lance.dataset(ds_path)
        # indices must be a Python list of ints (not numpy array)
        indices = rng.choice(image_count, size=batch_size, replace=False).tolist()
        batch_table = ds.take(indices, columns=["image_bytes"])
        batch_df = pl.from_arrow(batch_table)
        result = batch_df.with_columns(
            processed=pl.col("image_bytes").cv.pipe(pipe).sink("blob")
        )
        _ = result["processed"].to_list()

    for _ in range(warmup):
        run()

    total_time = 0.0
    peak_mem = 0.0
    for _ in range(iterations):
        _, elapsed, mem = run_timed_with_memory(run)
        total_time += elapsed
        peak_mem = max(peak_mem, mem.peak_memory_mb)

    avg = total_time / iterations
    return StorageBackendResult(
        backend="lance",
        scenario="random_access",
        image_count=image_count,
        batch_size=batch_size,
        image_size=(224, 224),
        total_time_seconds=avg,
        throughput_images_per_second=batch_size / avg,
        latency_ms_per_image=avg / batch_size * 1000,
        peak_memory_mb=peak_mem,
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline benchmarks
# ---------------------------------------------------------------------------

def _run_parquet_e2e(
    metadata_path: Path,
    warmup: int,
    iterations: int,
) -> StorageBackendResult:
    """Benchmark: Parquet + file_path — ImageNet-style e2e pipeline."""
    pipe = _build_imagenet_pipeline("file_path")

    def run() -> None:
        df = pl.read_parquet(metadata_path)
        result = df.with_columns(
            processed=pl.col("path").cv.pipe(pipe).sink("numpy")
        )
        _ = result["processed"].to_list()

    image_count = pl.read_parquet(metadata_path).height

    for _ in range(warmup):
        run()

    total_time = 0.0
    peak_mem = 0.0
    for _ in range(iterations):
        _, elapsed, mem = run_timed_with_memory(run)
        total_time += elapsed
        peak_mem = max(peak_mem, mem.peak_memory_mb)

    avg = total_time / iterations
    return StorageBackendResult(
        backend="parquet",
        scenario="e2e_pipeline",
        image_count=image_count,
        batch_size=None,
        image_size=(224, 224),
        total_time_seconds=avg,
        throughput_images_per_second=image_count / avg,
        latency_ms_per_image=avg / image_count * 1000,
        peak_memory_mb=peak_mem,
    )


def _run_lance_e2e(
    lance_dataset: LanceDataset,
    warmup: int,
    iterations: int,
) -> StorageBackendResult:
    """Benchmark: Lance inline bytes — ImageNet-style e2e pipeline."""
    _require_lance()
    import lance

    pipe = _build_imagenet_pipeline("image_bytes")
    ds_path = str(lance_dataset.dataset_path)

    def run() -> None:
        ds = lance.dataset(ds_path)
        df = pl.from_arrow(ds.to_table(columns=["image_bytes", "label"]))
        result = df.with_columns(
            processed=pl.col("image_bytes").cv.pipe(pipe).sink("numpy")
        )
        _ = result["processed"].to_list()

    image_count = lance_dataset.image_count

    for _ in range(warmup):
        run()

    total_time = 0.0
    peak_mem = 0.0
    for _ in range(iterations):
        _, elapsed, mem = run_timed_with_memory(run)
        total_time += elapsed
        peak_mem = max(peak_mem, mem.peak_memory_mb)

    avg = total_time / iterations
    return StorageBackendResult(
        backend="lance",
        scenario="e2e_pipeline",
        image_count=image_count,
        batch_size=None,
        image_size=(224, 224),
        total_time_seconds=avg,
        throughput_images_per_second=image_count / avg,
        latency_ms_per_image=avg / image_count * 1000,
        peak_memory_mb=peak_mem,
    )


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

# Batch sizes used for the random-access scenario
RANDOM_ACCESS_BATCH_SIZES: list[int] = [16, 64, 256]


def run_lance_vs_parquet(
    image_counts: list[int] | None = None,
    image_sizes: list[tuple[int, int]] | None = None,
    warmup_iterations: int = 3,
    benchmark_iterations: int = 5,
    verbose: bool = True,
) -> list[StorageBackendResult]:
    """
    Run all Lance vs Parquet storage benchmark scenarios.

    Creates paired datasets (same images, same seed) in temporary directories,
    runs sequential scan, random access, and e2e pipeline benchmarks for both
    backends, then cleans up.

    Args:
        image_counts: Dataset sizes to test (default: ``[100, 500]``).
        image_sizes: Image ``(width, height)`` pairs (default: ``[(224, 224)]``).
        warmup_iterations: Runs discarded before timing.
        benchmark_iterations: Timed runs per scenario.
        verbose: Whether to print per-scenario progress.

    Returns:
        List of :class:`StorageBackendResult` for every
        scenario × image_count × image_size combination.
    """
    if not LANCE_AVAILABLE:
        print(
            "  SKIPPED: lance not installed. "
            "Install with: pip install lance pyarrow",
            file=sys.stderr,
        )
        return []

    if image_counts is None:
        image_counts = [100, 500]
    if image_sizes is None:
        image_sizes = [(224, 224)]

    results: list[StorageBackendResult] = []
    rng = np.random.default_rng(0)

    for width, height in image_sizes:
        for n_images in image_counts:
            if verbose:
                print(
                    f"\n  {n_images} images @ {width}x{height}",
                    flush=True,
                )

            with tempfile.TemporaryDirectory(prefix="polars_cv_bench_") as tmp:
                tmp_path = Path(tmp)

                # --- dataset creation ---
                if verbose:
                    print("    Generating datasets...", end="", flush=True)

                parquet_ds = generate_imagefolder_dataset(
                    output_dir=tmp_path / "parquet",
                    num_images=n_images,
                    num_classes=5,
                    height=height,
                    width=width,
                    pattern="mixed",
                    base_seed=42,
                )
                lance_ds = generate_lance_dataset(
                    output_dir=tmp_path / "lance",
                    num_images=n_images,
                    num_classes=5,
                    height=height,
                    width=width,
                    pattern="mixed",
                    base_seed=42,
                )
                if verbose:
                    print(" done", flush=True)

                # --- sequential scan ---
                if verbose:
                    print("    sequential_scan: parquet...", end="", flush=True)
                r = _run_parquet_sequential(
                    tmp_path / "parquet",
                    parquet_ds.metadata_path,
                    warmup_iterations,
                    benchmark_iterations,
                )
                results.append(r)
                if verbose:
                    print(
                        f" {r.throughput_images_per_second:.1f} img/s  "
                        f"lance...",
                        end="",
                        flush=True,
                    )

                r = _run_lance_sequential(lance_ds, warmup_iterations, benchmark_iterations)
                results.append(r)
                if verbose:
                    print(f" {r.throughput_images_per_second:.1f} img/s", flush=True)

                # --- random access ---
                for batch_size in RANDOM_ACCESS_BATCH_SIZES:
                    if batch_size > n_images:
                        continue
                    if verbose:
                        print(
                            f"    random_access batch={batch_size}: "
                            f"parquet...",
                            end="",
                            flush=True,
                        )
                    r = _run_parquet_random_access(
                        parquet_ds.metadata_path,
                        batch_size,
                        n_images,
                        warmup_iterations,
                        benchmark_iterations,
                        rng,
                    )
                    results.append(r)
                    if verbose:
                        print(
                            f" {r.throughput_images_per_second:.1f} img/s  "
                            f"lance...",
                            end="",
                            flush=True,
                        )

                    r = _run_lance_random_access(
                        lance_ds,
                        batch_size,
                        warmup_iterations,
                        benchmark_iterations,
                        rng,
                    )
                    results.append(r)
                    if verbose:
                        print(f" {r.throughput_images_per_second:.1f} img/s", flush=True)

                # --- e2e pipeline ---
                if verbose:
                    print("    e2e_pipeline:    parquet...", end="", flush=True)
                r = _run_parquet_e2e(
                    parquet_ds.metadata_path,
                    warmup_iterations,
                    benchmark_iterations,
                )
                results.append(r)
                if verbose:
                    print(
                        f" {r.throughput_images_per_second:.1f} img/s  "
                        f"lance...",
                        end="",
                        flush=True,
                    )

                r = _run_lance_e2e(lance_ds, warmup_iterations, benchmark_iterations)
                results.append(r)
                if verbose:
                    print(f" {r.throughput_images_per_second:.1f} img/s", flush=True)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_storage_results(results: list[StorageBackendResult]) -> None:
    """Print a formatted comparison table for storage backend results."""
    if not results:
        print("No storage benchmark results to display.")
        return

    scenarios = sorted({r.scenario for r in results})
    configs = sorted(
        {(r.image_count, r.batch_size, r.image_size) for r in results},
        key=lambda t: (t[0], t[1] if t[1] is not None else -1, t[2]),
    )

    print("\n" + "=" * 72)
    print("LANCE vs PARQUET — Storage Backend Comparison")
    print("=" * 72)

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario}")
        print(
            f"  {'Config':<28} {'Backend':<10} "
            f"{'Throughput':>14} {'Latency':>12} {'Mem (MB)':>10}"
        )
        print("  " + "-" * 68)

        for image_count, batch_size, image_size in configs:
            batch_rows = [
                r
                for r in results
                if r.scenario == scenario
                and r.image_count == image_count
                and r.batch_size == batch_size
                and r.image_size == image_size
            ]
            if not batch_rows:
                continue

            label = (
                f"{image_count} imgs {image_size[0]}x{image_size[1]}"
                + (f" batch={batch_size}" if batch_size else "")
            )

            parquet_rows = [r for r in batch_rows if r.backend == "parquet"]
            lance_rows = [r for r in batch_rows if r.backend == "lance"]

            for row in parquet_rows + lance_rows:
                print(
                    f"  {label:<28} {row.backend:<10} "
                    f"{row.throughput_images_per_second:>13.1f}/s "
                    f"{row.latency_ms_per_image:>11.2f}ms "
                    f"{row.peak_memory_mb:>9.1f}"
                )

            # speedup annotation
            if parquet_rows and lance_rows:
                p = parquet_rows[0].throughput_images_per_second
                l = lance_rows[0].throughput_images_per_second
                speedup = l / p if p > 0 else float("nan")
                marker = "  <-- faster" if l > p else "  <-- faster" if p > l else ""
                direction = "Lance" if l > p else "Parquet" if p > l else "tie"
                print(
                    f"  {'':28} {'speedup':<10} "
                    f"  {direction} {abs(speedup):.2f}x"
                )

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Standalone CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Lance vs Parquet storage backends for polars-cv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--counts",
        default="100,500",
        help="Comma-separated image counts (default: 100,500)",
    )
    parser.add_argument(
        "--sizes",
        default="224",
        help="Comma-separated image sizes in pixels (default: 224)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Timed iterations (default: 5)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step progress output",
    )
    args = parser.parse_args()

    counts = [int(c.strip()) for c in args.counts.split(",")]
    sizes = [(int(s.strip()), int(s.strip())) for s in args.sizes.split(",")]

    if not args.quiet:
        print("Lance vs Parquet Storage Backend Benchmark", flush=True)
        print(f"Image counts : {counts}", flush=True)
        print(f"Image sizes  : {sizes}", flush=True)
        print(f"Warmup       : {args.warmup}", flush=True)
        print(f"Iterations   : {args.iterations}", flush=True)

    results = run_lance_vs_parquet(
        image_counts=counts,
        image_sizes=sizes,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        verbose=not args.quiet,
    )

    print_storage_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
