"""
Single operation benchmark scenarios.

This module provides benchmarks for individual image processing operations
across all framework adapters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchmarks.frameworks import (
    BaseFrameworkAdapter,
    BenchmarkResult,
    OperationParams,
    OperationType,
)
from benchmarks.utils.data_gen import generate_image_set
from benchmarks.utils.memory import run_timed_with_memory

if TYPE_CHECKING:
    pass


@dataclass
class SingleOpBenchmarkConfig:
    """Configuration for a single operation benchmark."""

    operation: OperationType
    name: str
    params: OperationParams
    description: str


def get_single_op_benchmarks(
    source_height: int = 256,
    source_width: int = 256,
) -> list[SingleOpBenchmarkConfig]:
    """
    Get the list of single operation benchmarks to run.

    Args:
        source_height: Source image height.
        source_width: Source image width.

    Returns:
        List of benchmark configurations.
    """
    return [
        SingleOpBenchmarkConfig(
            operation=OperationType.RESIZE,
            name="resize",
            params=OperationParams(
                operation=OperationType.RESIZE,
                height=224,
                width=224,
            ),
            description="Resize from {source_height}x{source_width} to 224x224",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.GRAYSCALE,
            name="grayscale",
            params=OperationParams(operation=OperationType.GRAYSCALE),
            description="Convert RGB to grayscale",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.NORMALIZE,
            name="normalize",
            params=OperationParams(operation=OperationType.NORMALIZE),
            description="Min-max normalization to [0, 1]",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.FLIP_H,
            name="flip_horizontal",
            params=OperationParams(operation=OperationType.FLIP_H),
            description="Horizontal flip",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.FLIP_V,
            name="flip_vertical",
            params=OperationParams(operation=OperationType.FLIP_V),
            description="Vertical flip",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.CROP,
            name="crop_center",
            params=OperationParams(
                operation=OperationType.CROP,
                # Center crop - use smaller of source or 128 for crop size
                # This ensures we don't try to crop larger than the source
                crop_top=max(0, (source_height - min(128, source_height)) // 2),
                crop_left=max(0, (source_width - min(128, source_width)) // 2),
                crop_height=min(128, source_height),
                crop_width=min(128, source_width),
            ),
            description="Center crop to 128x128 (or smaller if source is smaller)",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.BLUR,
            name="blur",
            params=OperationParams(
                operation=OperationType.BLUR,
                sigma=2.0,
            ),
            description="Gaussian blur with sigma=2.0",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.THRESHOLD,
            name="threshold",
            params=OperationParams(
                operation=OperationType.THRESHOLD,
                # Use 127 instead of 128 to avoid boundary issues between
                # integer (OpenCV) and floating-point (Torchvision) implementations
                threshold_value=127,
            ),
            description="Binary threshold at 128",
        ),
    ]


def run_single_op_benchmark(
    adapter: BaseFrameworkAdapter,
    benchmark: SingleOpBenchmarkConfig,
    image_count: int,
    image_size: tuple[int, int],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> BenchmarkResult:
    """
    Run a single operation benchmark on an adapter.

    This benchmarks pure operation performance by pre-decoding images
    before timing. This removes PNG decode overhead for fair comparison
    across frameworks.

    Args:
        adapter: Framework adapter to benchmark.
        benchmark: Benchmark configuration.
        image_count: Number of images to process.
        image_size: Image dimensions (width, height).
        warmup_iterations: Number of warmup runs.
        benchmark_iterations: Number of timed runs.

    Returns:
        Benchmark result with timing and memory statistics.
    """
    width, height = image_size

    # Generate test images
    image_set = generate_image_set(
        count=image_count,
        height=height,
        width=width,
        channels=3,
        pattern="gradient",
    )

    operations = [benchmark.params]

    # Pre-decode images to native format (removes decode overhead from timing)
    decoded_images = adapter.prepare_decoded_images(image_set.image_bytes)
    warmup_decoded = adapter.prepare_decoded_images(image_set.image_bytes[:10])

    # Warmup
    for _ in range(warmup_iterations):
        adapter.run_pipeline_on_decoded(warmup_decoded, operations)

    # Benchmark (using pre-decoded images for fair comparison)
    total_time = 0.0
    peak_memory = 0.0

    for _ in range(benchmark_iterations):
        _, elapsed, mem_stats = run_timed_with_memory(
            lambda: adapter.run_pipeline_on_decoded(decoded_images, operations)
        )
        total_time += elapsed
        peak_memory = max(peak_memory, mem_stats.peak_memory_mb)

    avg_time = total_time / benchmark_iterations
    throughput = image_count / avg_time
    latency_ms = (avg_time / image_count) * 1000

    return BenchmarkResult(
        framework=adapter.name,
        operation=benchmark.name,
        image_count=image_count,
        image_size=image_size,
        total_time_seconds=avg_time,
        throughput_images_per_second=throughput,
        latency_ms_per_image=latency_ms,
        peak_memory_mb=peak_memory,
    )


def run_single_op_benchmark_gpu(
    adapter: Any,  # TorchvisionAdapter with GPU support
    benchmark: SingleOpBenchmarkConfig,
    image_count: int,
    image_size: tuple[int, int],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run a single operation benchmark on a GPU adapter with cold and warm starts.

    Cold start: Uses pre-decoded images (decode overhead removed, but includes
    transfer to GPU). This is comparable to other frameworks' pre-decoded benchmarks.

    Warm start: Data already resident on GPU (pure operation performance).

    Args:
        adapter: GPU-capable framework adapter.
        benchmark: Benchmark configuration.
        image_count: Number of images to process.
        image_size: Image dimensions (width, height).
        warmup_iterations: Number of warmup runs.
        benchmark_iterations: Number of timed runs.

    Returns:
        Tuple of (cold_start_result, warm_start_result).
    """
    width, height = image_size

    # Generate test images
    image_set = generate_image_set(
        count=image_count,
        height=height,
        width=width,
        channels=3,
        pattern="gradient",
    )

    operations = [benchmark.params]

    # Pre-decode images to native format (removes PNG decode overhead)
    decoded_images = adapter.prepare_decoded_images(image_set.image_bytes)
    warmup_decoded = adapter.prepare_decoded_images(image_set.image_bytes[:10])

    # Warmup
    for _ in range(warmup_iterations):
        adapter.run_pipeline_on_decoded(warmup_decoded, operations)
        adapter.synchronize()

    # Cold start benchmark (pre-decoded but includes transfer to GPU)
    cold_total_time = 0.0
    cold_peak_memory = 0.0

    for _ in range(benchmark_iterations):
        start = time.perf_counter()
        adapter.run_pipeline_on_decoded(decoded_images, operations)
        adapter.synchronize()
        elapsed = time.perf_counter() - start

        cold_total_time += elapsed
        # Memory tracking less accurate for GPU

    cold_avg_time = cold_total_time / benchmark_iterations
    cold_throughput = image_count / cold_avg_time
    cold_latency_ms = (cold_avg_time / image_count) * 1000

    cold_result = BenchmarkResult(
        framework=adapter.name,
        operation=benchmark.name,
        image_count=image_count,
        image_size=image_size,
        total_time_seconds=cold_avg_time,
        throughput_images_per_second=cold_throughput,
        latency_ms_per_image=cold_latency_ms,
        peak_memory_mb=cold_peak_memory,
        gpu_mode="cold",
    )

    # Warm start benchmark (data already on GPU)
    preloaded = adapter.preload_to_device(image_set.image_bytes)
    adapter.synchronize()

    warm_total_time = 0.0
    warm_peak_memory = 0.0

    for _ in range(benchmark_iterations):
        start = time.perf_counter()
        adapter.run_pipeline_batch_warm(preloaded, operations)
        adapter.synchronize()
        elapsed = time.perf_counter() - start

        warm_total_time += elapsed

    warm_avg_time = warm_total_time / benchmark_iterations
    warm_throughput = image_count / warm_avg_time
    warm_latency_ms = (warm_avg_time / image_count) * 1000

    warm_result = BenchmarkResult(
        framework=adapter.name,
        operation=benchmark.name,
        image_count=image_count,
        image_size=image_size,
        total_time_seconds=warm_avg_time,
        throughput_images_per_second=warm_throughput,
        latency_ms_per_image=warm_latency_ms,
        peak_memory_mb=warm_peak_memory,
        gpu_mode="warm",
    )

    return cold_result, warm_result


def run_all_single_ops(
    adapters: list[BaseFrameworkAdapter],
    image_counts: list[int],
    image_sizes: list[tuple[int, int]],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Run all single operation benchmarks across all adapters and configurations.

    Args:
        adapters: List of framework adapters to benchmark.
        image_counts: List of image counts to test.
        image_sizes: List of image sizes to test.
        warmup_iterations: Number of warmup runs.
        benchmark_iterations: Number of timed runs.
        verbose: Whether to print progress output.

    Returns:
        List of all benchmark results.
    """
    results: list[BenchmarkResult] = []

    total_combinations = len(image_sizes) * len(image_counts) * 8 * len(adapters)
    current = 0

    for size_idx, size in enumerate(image_sizes):
        benchmarks = get_single_op_benchmarks(size[1], size[0])

        if verbose:
            print(
                f"\n  Size {size_idx + 1}/{len(image_sizes)}: {size[0]}x{size[1]}",
                flush=True,
            )

        for count_idx, count in enumerate(image_counts):
            if verbose:
                print(
                    f"    Count {count_idx + 1}/{len(image_counts)}: {count} images",
                    flush=True,
                )

            for benchmark in benchmarks:
                for adapter in adapters:
                    current += 1

                    if not adapter.is_available():
                        if verbose:
                            print(
                                f"      [{current}/{total_combinations}] "
                                f"{adapter.name}/{benchmark.name}: SKIPPED (unavailable)",
                                flush=True,
                            )
                        continue

                    if verbose:
                        print(
                            f"      [{current}/{total_combinations}] "
                            f"{adapter.name}/{benchmark.name}...",
                            end="",
                            flush=True,
                        )

                    try:
                        if adapter.supports_gpu and hasattr(adapter, "synchronize"):
                            # GPU adapter - run both cold and warm
                            cold, warm = run_single_op_benchmark_gpu(
                                adapter,
                                benchmark,
                                count,
                                size,
                                warmup_iterations,
                                benchmark_iterations,
                            )
                            results.append(cold)
                            results.append(warm)
                            if verbose:
                                print(
                                    f" {cold.throughput_images_per_second:.1f} img/s "
                                    f"(cold), {warm.throughput_images_per_second:.1f} "
                                    f"img/s (warm)",
                                    flush=True,
                                )
                        else:
                            # CPU adapter
                            result = run_single_op_benchmark(
                                adapter,
                                benchmark,
                                count,
                                size,
                                warmup_iterations,
                                benchmark_iterations,
                            )
                            results.append(result)
                            if verbose:
                                print(
                                    f" {result.throughput_images_per_second:.1f} img/s",
                                    flush=True,
                                )
                    except Exception as e:
                        if verbose:
                            print(f" ERROR: {e}", flush=True)
                        else:
                            print(
                                f"Error benchmarking {adapter.name}/{benchmark.name}: "
                                f"{e}",
                                flush=True,
                            )

    return results
