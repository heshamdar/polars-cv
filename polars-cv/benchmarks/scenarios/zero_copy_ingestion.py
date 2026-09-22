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

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
from PIL import Image

from benchmarks.utils.timing import TimingStats, measure, to_result
from polars_cv import Pipeline

if TYPE_CHECKING:
    from benchmarks.frameworks import BenchmarkResult


@dataclass
class IngestionResult:
    """One measurement from this module's own, row-oriented matrix.

    Deliberately *not* named ``BenchmarkResult``. It used to be, shadowing
    ``benchmarks.frameworks.BenchmarkResult`` — a different record with
    different fields — and `benchmarks.regression.run_suite` appended these to a
    list of those without converting. Selecting the ``zero_copy`` scenario
    therefore died in aggregation with ``'BenchmarkResult' object has no
    attribute 'framework'``. Two records that mean different things now read
    differently; :func:`to_suite_results` is the one conversion between them.
    """

    name: str
    rows: int
    #: (width, height) of the images measured; carried so the conversion below
    #: does not have to invent one for the suite's result key.
    size: tuple[int, int]
    total_time_ms: float
    per_row_us: float
    throughput_rows_per_sec: float
    #: The samples these three were derived from. Carried so
    #: :func:`to_suite_results` can hand the suite a real statistic instead of
    #: re-deriving one, and so the comparator sees this scenario's dispersion
    #: like every other scenario's.
    stats: TimingStats


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

    pipeline = Pipeline().source("image_bytes")
    result = df.select(pl.col("img").cv.pipe(pipeline).sink("blob"))

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


DEFAULT_ITERATIONS = 7
"""Timed runs per ingestion benchmark.

Each of these used to be a *single* un-repeated span after a 5-row warmup —
one sample, no dispersion, and no way to tell a real change from a scheduler
hiccup. Seven is enough for a median that one outlier cannot move, and these
benchmarks are cheap enough to afford it.
"""


def _measure_ingestion(
    name: str,
    call: Callable[[], object],
    *,
    rows: int,
    size: tuple[int, int],
    iterations: int = DEFAULT_ITERATIONS,
) -> IngestionResult:
    """Time *call* through the timing authority and shape this module's record.

    The warmup is the same callable over the same data, not a 5-row head: a
    head-warmed run pays first-touch allocation on the rest of the column
    inside iteration one.
    """
    stats = measure(call, warmup_fn=call, warmup=1, iterations=iterations, label=name)
    median = stats.median_s
    return IngestionResult(
        name=name,
        rows=rows,
        size=size,
        total_time_ms=median * 1000,
        per_row_us=(median * 1_000_000) / rows,
        throughput_rows_per_sec=rows / median,
        stats=stats,
    )


def benchmark_image_bytes_source(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> IngestionResult:
    """Benchmark image_bytes source (requires PNG decoding)."""
    images = create_test_images(n_rows, size)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes")

    def run() -> object:
        result = df.select(pl.col("img").cv.pipe(pipeline).sink("numpy"))
        return result["img"].to_list()  # Force evaluation

    return _measure_ingestion("image_bytes", run, rows=n_rows, size=size)


def benchmark_blob_source(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> IngestionResult:
    """Benchmark blob source (zero-copy path)."""
    blobs = create_blob_data(n_rows, size)
    df = pl.DataFrame({"blob": blobs})

    pipeline = Pipeline().source("blob")

    def run() -> object:
        result = df.select(pl.col("blob").cv.pipe(pipeline).sink("numpy"))
        return result["blob"].to_list()  # Force evaluation

    return _measure_ingestion("blob", run, rows=n_rows, size=size)


def benchmark_list_source_explicit_dtype(
    n_rows: int = 100, size: tuple[int, int] = (64, 64)
) -> IngestionResult:
    """Benchmark list source with explicit dtype."""
    df = create_list_data(n_rows, size)

    pipeline = Pipeline().source("list", dtype="u8")

    def run() -> object:
        result = df.select(pl.col("arr").cv.pipe(pipeline).sink("numpy"))
        return result["arr"].to_list()  # Force evaluation

    return _measure_ingestion("list_explicit_dtype", run, rows=n_rows, size=size)


def benchmark_list_source_auto_dtype(
    n_rows: int = 100, size: tuple[int, int] = (64, 64)
) -> IngestionResult:
    """Benchmark list source with auto dtype inference."""
    df = create_list_data(n_rows, size)

    # No explicit dtype - will be inferred
    pipeline = Pipeline().source("list")

    def run() -> object:
        result = df.select(pl.col("arr").cv.pipe(pipeline).sink("numpy"))
        return result["arr"].to_list()  # Force evaluation

    return _measure_ingestion("list_auto_dtype", run, rows=n_rows, size=size)


def benchmark_numpy_output(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> IngestionResult:
    """Benchmark numpy sink output (zero-copy struct format)."""
    images = create_test_images(n_rows, size)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes")

    def run() -> object:
        result = df.select(output=pl.col("img").cv.pipe(pipeline).sink("numpy"))
        # Access all struct values to force evaluation
        return result["output"].to_list()

    return _measure_ingestion("numpy_output", run, rows=n_rows, size=size)


def benchmark_png_output(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> IngestionResult:
    """Benchmark PNG sink output (requires encoding/compression)."""
    images = create_test_images(n_rows, size)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes")

    def run() -> object:
        result = df.select(output=pl.col("img").cv.pipe(pipeline).sink("png"))
        return result["output"].to_list()  # Force evaluation

    return _measure_ingestion("png_output", run, rows=n_rows, size=size)


def benchmark_blob_output(
    n_rows: int = 100, size: tuple[int, int] = (256, 256)
) -> IngestionResult:
    """Benchmark blob sink output (VIEW protocol)."""
    images = create_test_images(n_rows, size)
    df = pl.DataFrame({"img": images})

    pipeline = Pipeline().source("image_bytes")

    def run() -> object:
        result = df.select(output=pl.col("img").cv.pipe(pipeline).sink("blob"))
        return result["output"].to_list()  # Force evaluation

    return _measure_ingestion("blob_output", run, rows=n_rows, size=size)


def run_ingestion_benchmarks() -> list[IngestionResult]:
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


def run_output_benchmarks() -> list[IngestionResult]:
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


def run_benchmarks() -> list[IngestionResult]:
    """Run all benchmarks."""
    ingestion_results = run_ingestion_benchmarks()
    output_results = run_output_benchmarks()
    return ingestion_results + output_results


def to_suite_results(results: list[IngestionResult]) -> "list[BenchmarkResult]":
    """Convert to the record the regression suite aggregates on.

    The suite keys results on ``(framework, operation, image_size,
    image_count, gpu_mode)`` and gates on throughput, none of which an
    :class:`IngestionResult` carries under those names. Both engines are
    polars-cv here, and these benchmarks all run eager, so ``framework`` is
    fixed rather than measured.

    Memory is reported as unmeasured rather than as ``0.0``. It used to be the
    latter, which ``compare._pct`` mapped 0 → 0 to 0% and classified NEUTRAL —
    a hole in the results that read as a clean bill of health. ``memory=None``
    now carries a status string saying so.
    """
    import polars as pl

    return [
        to_result(
            r.stats,
            framework="polars-cv-eager",
            engine="eager",
            operation=f"zero_copy_{r.name}",
            image_count=r.rows,
            image_size=r.size,
            thread_pool_size=pl.thread_pool_size(),
            memory=None,
        )
        for r in results
    ]


def print_results(results: list[IngestionResult]) -> None:
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
