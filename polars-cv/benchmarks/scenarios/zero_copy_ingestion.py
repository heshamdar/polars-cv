"""
Benchmark comparing zero-copy vs copy-based data handling.

This benchmark measures the performance difference between:

Ingestion benchmarks:
1. Zero-copy blob source (direct buffer reference)
2. Copy-based image_bytes source (requires decoding)
3. List/Array source with dtype auto-inference vs explicit dtype

Output benchmarks:
1. Numpy struct output (zero-copy ownership transfer)
2. PNG/JPEG encoding (requires copy + compression)
3. Blob output (VIEW protocol serialization)

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


def create_test_images(
    n_images: int, size: tuple[int, int] = (256, 256)
) -> list[bytes]:
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
    result = df.select(pl.col("img").cv.pipeline(pipeline))

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


def benchmark_image_bytes_source(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> BenchmarkResult:
    """Benchmark image_bytes source (requires PNG decoding)."""
    images = create_test_images(n_rows, size)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes").sink("numpy")

    # Warmup
    _ = df.head(5).select(pl.col("img").cv.pipeline(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(pl.col("img").cv.pipeline(pipeline))
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


def benchmark_blob_source(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> BenchmarkResult:
    """Benchmark blob source (zero-copy path)."""
    blobs = create_blob_data(n_rows, size)
    df = pl.DataFrame({"blob": blobs})

    pipeline = Pipeline().source("blob").sink("numpy")

    # Warmup
    _ = df.head(5).select(pl.col("blob").cv.pipeline(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(pl.col("blob").cv.pipeline(pipeline))
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


def benchmark_list_source_explicit_dtype(
    n_rows: int = 100, size: tuple[int, int] = (64, 64)
) -> BenchmarkResult:
    """Benchmark list source with explicit dtype."""
    df = create_list_data(n_rows, size)

    pipeline = Pipeline().source("list", dtype="u8").sink("numpy")

    # Warmup
    _ = df.head(5).select(pl.col("arr").cv.pipeline(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(pl.col("arr").cv.pipeline(pipeline))
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


def benchmark_list_source_auto_dtype(
    n_rows: int = 100, size: tuple[int, int] = (64, 64)
) -> BenchmarkResult:
    """Benchmark list source with auto dtype inference."""
    df = create_list_data(n_rows, size)

    # No explicit dtype - will be inferred
    pipeline = Pipeline().source("list").sink("numpy")

    # Warmup
    _ = df.head(5).select(pl.col("arr").cv.pipeline(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(pl.col("arr").cv.pipeline(pipeline))
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


def benchmark_numpy_output(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> BenchmarkResult:
    """Benchmark numpy sink output (zero-copy struct format)."""
    images = create_test_images(n_rows, size)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes").sink("numpy")

    # Warmup
    _ = df.head(5).select(pl.col("img").cv.pipeline(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(output=pl.col("img").cv.pipeline(pipeline))
    # Access all struct values to force evaluation
    _ = result["output"].to_list()
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    per_row_us = (elapsed * 1_000_000) / n_rows
    throughput = n_rows / elapsed

    return BenchmarkResult(
        name="numpy_output",
        rows=n_rows,
        total_time_ms=elapsed_ms,
        per_row_us=per_row_us,
        throughput_rows_per_sec=throughput,
    )


def benchmark_png_output(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> BenchmarkResult:
    """Benchmark PNG sink output (requires encoding/compression)."""
    images = create_test_images(n_rows, size)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes").sink("png")

    # Warmup
    _ = df.head(5).select(pl.col("img").cv.pipeline(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(output=pl.col("img").cv.pipeline(pipeline))
    _ = result["output"].to_list()  # Force evaluation
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    per_row_us = (elapsed * 1_000_000) / n_rows
    throughput = n_rows / elapsed

    return BenchmarkResult(
        name="png_output",
        rows=n_rows,
        total_time_ms=elapsed_ms,
        per_row_us=per_row_us,
        throughput_rows_per_sec=throughput,
    )


def benchmark_blob_output(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> BenchmarkResult:
    """Benchmark blob sink output (VIEW protocol)."""
    images = create_test_images(n_rows, size)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes").sink("blob")

    # Warmup
    _ = df.head(5).select(pl.col("img").cv.pipeline(pipeline))

    # Benchmark
    start = time.perf_counter()
    result = df.select(output=pl.col("img").cv.pipeline(pipeline))
    _ = result["output"].to_list()  # Force evaluation
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    per_row_us = (elapsed * 1_000_000) / n_rows
    throughput = n_rows / elapsed

    return BenchmarkResult(
        name="blob_output",
        rows=n_rows,
        total_time_ms=elapsed_ms,
        per_row_us=per_row_us,
        throughput_rows_per_sec=throughput,
    )


def run_ingestion_benchmarks() -> list[BenchmarkResult]:
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


def run_output_benchmarks() -> list[BenchmarkResult]:
    """Run all output benchmarks."""
    print("\n" + "=" * 60)
    print("Zero-Copy Output Benchmarks")
    print("=" * 60)

    results = []

    print("\nRunning numpy output benchmark (zero-copy struct)...")
    results.append(benchmark_numpy_output(n_rows=100, size=(256, 256)))

    print("Running blob output benchmark (VIEW protocol)...")
    results.append(benchmark_blob_output(n_rows=100, size=(256, 256)))

    print("Running PNG output benchmark (encoding required)...")
    results.append(benchmark_png_output(n_rows=100, size=(256, 256)))

    return results


def run_benchmarks() -> list[BenchmarkResult]:
    """Run all benchmarks."""
    ingestion_results = run_ingestion_benchmarks()
    output_results = run_output_benchmarks()
    return ingestion_results + output_results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    # Header
    print(
        f"{'Benchmark':<25} {'Rows':<8} {'Total (ms)':<12} {'Per Row (µs)':<15} {'Throughput':<15}"
    )
    print("-" * 75)

    for r in results:
        print(
            f"{r.name:<25} {r.rows:<8} {r.total_time_ms:<12.2f} "
            f"{r.per_row_us:<15.2f} {r.throughput_rows_per_sec:<15.1f}"
        )

    print("-" * 75)

    # Ingestion comparison
    ingestion_results = [
        r
        for r in results
        if r.name in ["image_bytes", "blob", "list_explicit_dtype", "list_auto_dtype"]
    ]
    if len(ingestion_results) >= 2:
        baseline = ingestion_results[0]
        print("\nIngestion speedup vs image_bytes baseline:")
        for r in ingestion_results[1:]:
            if r.total_time_ms > 0:
                speedup = baseline.total_time_ms / r.total_time_ms
                print(f"  {r.name}: {speedup:.2f}x")

    # Output comparison
    output_results = [
        r for r in results if r.name in ["numpy_output", "png_output", "blob_output"]
    ]
    if len(output_results) >= 2:
        # numpy_output is the zero-copy baseline for output
        baseline = next(
            (r for r in output_results if r.name == "numpy_output"), output_results[0]
        )
        print("\nOutput comparison (numpy_output as baseline):")
        for r in output_results:
            if r.total_time_ms > 0:
                ratio = r.total_time_ms / baseline.total_time_ms
                print(f"  {r.name}: {ratio:.2f}x (relative time)")


if __name__ == "__main__":
    results = run_benchmarks()
    print_results(results)
