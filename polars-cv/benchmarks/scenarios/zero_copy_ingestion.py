"""
Benchmark comparing zero-copy vs copy-based data ingestion.

This benchmark measures the performance difference between:
1. Zero-copy blob source (direct buffer reference)
2. Copy-based image_bytes source (requires decoding)
3. List/Array source with dtype auto-inference vs explicit dtype

Run with:
    uv run python -m benchmarks.scenarios.zero_copy_ingestion
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from PIL import Image

from polars_cv import Pipeline

if TYPE_CHECKING:
    pass


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    rows: int
    total_time_ms: float
    per_row_us: float
    throughput_rows_per_sec: float


def create_test_images(n_images: int, size: tuple[int, int] = (256, 256)) -> list[bytes]:
    """Create n test images as PNG bytes."""
    images = []
    for i in range(n_images):
        # Create slightly different images
        arr = np.random.randint(0, 256, size, dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())
    return images


def create_blob_data(n_rows: int, shape: tuple[int, int] = (256, 256)) -> list[bytes]:
    """Create n VIEW protocol blob bytes."""
    # First create images, then convert to blob
    images = create_test_images(n_rows, shape)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes").sink("blob")
    result = df.select(pl.col("img").cv.pipe(pipeline))

    return result["img"].to_list()


def create_list_data(n_rows: int, shape: tuple[int, int] = (64, 64)) -> pl.DataFrame:
    """Create n rows of nested list data."""
    rows = []
    for _ in range(n_rows):
        # Create 2D array as nested list
        arr = np.random.randint(0, 256, shape, dtype=np.uint8).tolist()
        rows.append(arr)

    df = pl.DataFrame({"arr": rows})
    return df.cast({"arr": pl.List(pl.List(pl.UInt8))})


def benchmark_image_bytes_source(n_rows: int = 100, size: tuple[int, int] = (256, 256)) -> BenchmarkResult:
    """Benchmark image_bytes source (requires PNG decoding)."""
    images = create_test_images(n_rows, size)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes").sink("numpy")

    # Warmup
    _ = df.head(5).select(pl.col("img").cv.pipe(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(pl.col("img").cv.pipe(pipeline))
    _ = result["img"].to_list()  # Force evaluation
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    per_row_us = (elapsed * 1_000_000) / n_rows
    throughput = n_rows / elapsed

    return BenchmarkResult(
        name="image_bytes",
        rows=n_rows,
        total_time_ms=elapsed_ms,
        per_row_us=per_row_us,
        throughput_rows_per_sec=throughput,
    )


def benchmark_blob_source(n_rows: int = 100, size: tuple[int, int] = (256, 256)) -> BenchmarkResult:
    """Benchmark blob source (zero-copy path)."""
    blobs = create_blob_data(n_rows, size)
    df = pl.DataFrame({"blob": blobs})

    pipeline = Pipeline().source("blob").sink("numpy")

    # Warmup
    _ = df.head(5).select(pl.col("blob").cv.pipe(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(pl.col("blob").cv.pipe(pipeline))
    _ = result["blob"].to_list()  # Force evaluation
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    per_row_us = (elapsed * 1_000_000) / n_rows
    throughput = n_rows / elapsed

    return BenchmarkResult(
        name="blob",
        rows=n_rows,
        total_time_ms=elapsed_ms,
        per_row_us=per_row_us,
        throughput_rows_per_sec=throughput,
    )


def benchmark_list_source_explicit_dtype(n_rows: int = 100, size: tuple[int, int] = (64, 64)) -> BenchmarkResult:
    """Benchmark list source with explicit dtype."""
    df = create_list_data(n_rows, size)

    pipeline = Pipeline().source("list", dtype="u8").sink("numpy")

    # Warmup
    _ = df.head(5).select(pl.col("arr").cv.pipe(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(pl.col("arr").cv.pipe(pipeline))
    _ = result["arr"].to_list()  # Force evaluation
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    per_row_us = (elapsed * 1_000_000) / n_rows
    throughput = n_rows / elapsed

    return BenchmarkResult(
        name="list_explicit_dtype",
        rows=n_rows,
        total_time_ms=elapsed_ms,
        per_row_us=per_row_us,
        throughput_rows_per_sec=throughput,
    )


def benchmark_list_source_auto_dtype(n_rows: int = 100, size: tuple[int, int] = (64, 64)) -> BenchmarkResult:
    """Benchmark list source with auto dtype inference."""
    df = create_list_data(n_rows, size)

    # No explicit dtype - will be inferred
    pipeline = Pipeline().source("list").sink("numpy")

    # Warmup
    _ = df.head(5).select(pl.col("arr").cv.pipe(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(pl.col("arr").cv.pipe(pipeline))
    _ = result["arr"].to_list()  # Force evaluation
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    per_row_us = (elapsed * 1_000_000) / n_rows
    throughput = n_rows / elapsed

    return BenchmarkResult(
        name="list_auto_dtype",
        rows=n_rows,
        total_time_ms=elapsed_ms,
        per_row_us=per_row_us,
        throughput_rows_per_sec=throughput,
    )


def run_benchmarks() -> list[BenchmarkResult]:
    """Run all ingestion benchmarks."""
    print("=" * 60)
    print("Zero-Copy Ingestion Benchmarks")
    print("=" * 60)

    results = []

    # Run each benchmark
    print("\nRunning image_bytes benchmark (baseline)...")
    results.append(benchmark_image_bytes_source(n_rows=100, size=(256, 256)))

    print("Running blob benchmark (zero-copy path)...")
    results.append(benchmark_blob_source(n_rows=100, size=(256, 256)))

    print("Running list source with explicit dtype...")
    results.append(benchmark_list_source_explicit_dtype(n_rows=100, size=(64, 64)))

    print("Running list source with auto dtype inference...")
    results.append(benchmark_list_source_auto_dtype(n_rows=100, size=(64, 64)))

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # Header
    print(f"{'Source':<25} {'Rows':<8} {'Total (ms)':<12} {'Per Row (µs)':<15} {'Throughput':<15}")
    print("-" * 75)

    for r in results:
        print(
            f"{r.name:<25} {r.rows:<8} {r.total_time_ms:<12.2f} "
            f"{r.per_row_us:<15.2f} {r.throughput_rows_per_sec:<15.1f}"
        )

    print("-" * 75)

    # Comparison
    if len(results) >= 2:
        baseline = results[0]
        print("\nSpeedup vs image_bytes baseline:")
        for r in results[1:]:
            if r.total_time_ms > 0:
                speedup = baseline.total_time_ms / r.total_time_ms
                print(f"  {r.name}: {speedup:.2f}x")


if __name__ == "__main__":
    results = run_benchmarks()
    print_results(results)

