"""
Chained pipeline benchmark scenarios.

This module provides benchmarks for chained image processing operations
(multiple operations applied sequentially) across all framework adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchmarks.frameworks import (
    BaseFrameworkAdapter,
    BenchmarkResult,
    OperationParams,
    OperationType,
)
from benchmarks.utils.data_gen import generate_image_set
from benchmarks.utils.timing import measure, measure_memory, to_result

if TYPE_CHECKING:
    pass


@dataclass
class PipelineBenchmarkConfig:
    """Configuration for a pipeline benchmark."""

    name: str
    operations: list[OperationParams]
    description: str
    complexity: str  # "light", "medium", "heavy"


def get_pipeline_benchmarks(
    source_height: int = 256,
    source_width: int = 256,
) -> list[PipelineBenchmarkConfig]:
    """
    Get the list of pipeline benchmarks to run.

    Args:
        source_height: Source image height.
        source_width: Source image width.

    Returns:
        List of pipeline benchmark configurations.
    """
    return [
        # Light pipeline (2 ops)
        PipelineBenchmarkConfig(
            name="light_pipeline",
            operations=[
                OperationParams(
                    operation=OperationType.RESIZE,
                    height=224,
                    width=224,
                ),
                OperationParams(operation=OperationType.NORMALIZE),
            ],
            description="Resize to 224x224 + normalize",
            complexity="light",
        ),
        # Medium pipeline (4 ops)
        PipelineBenchmarkConfig(
            name="medium_pipeline",
            operations=[
                OperationParams(
                    operation=OperationType.RESIZE,
                    height=256,
                    width=256,
                ),
                OperationParams(
                    operation=OperationType.CROP,
                    crop_top=16,
                    crop_left=16,
                    crop_height=224,
                    crop_width=224,
                ),
                OperationParams(operation=OperationType.NORMALIZE),
                OperationParams(operation=OperationType.FLIP_H),
            ],
            description="Resize + center crop + normalize + flip",
            complexity="medium",
        ),
        # Heavy pipeline (6 ops)
        PipelineBenchmarkConfig(
            name="heavy_pipeline",
            operations=[
                OperationParams(
                    operation=OperationType.RESIZE,
                    height=256,
                    width=256,
                ),
                OperationParams(operation=OperationType.FLIP_H),
                OperationParams(operation=OperationType.GRAYSCALE),
                OperationParams(
                    operation=OperationType.BLUR,
                    sigma=1.5,
                ),
                OperationParams(operation=OperationType.NORMALIZE),
                OperationParams(
                    operation=OperationType.THRESHOLD,
                    # Use 127 instead of 128 to avoid boundary issues between
                    # integer (OpenCV) and floating-point (Torchvision) implementations
                    threshold_value=127,
                ),
            ],
            description="Resize + flip + grayscale + blur + normalize + threshold",
            complexity="heavy",
        ),
        # ImageNet preprocessing pipeline
        PipelineBenchmarkConfig(
            name="imagenet_preprocess",
            operations=[
                OperationParams(
                    operation=OperationType.RESIZE,
                    height=256,
                    width=256,
                ),
                OperationParams(
                    operation=OperationType.CROP,
                    crop_top=16,
                    crop_left=16,
                    crop_height=224,
                    crop_width=224,
                ),
                OperationParams(operation=OperationType.NORMALIZE),
            ],
            description="Standard ImageNet preprocessing",
            complexity="medium",
        ),
        # Medical imaging style pipeline
        PipelineBenchmarkConfig(
            name="medical_pipeline",
            operations=[
                OperationParams(operation=OperationType.GRAYSCALE),
                OperationParams(operation=OperationType.NORMALIZE),
                OperationParams(
                    operation=OperationType.RESIZE,
                    height=512,
                    width=512,
                ),
            ],
            description="Grayscale + normalize + resize (medical imaging style)",
            complexity="medium",
        ),
    ]


def run_pipeline_benchmark(
    adapter: BaseFrameworkAdapter,
    benchmark: PipelineBenchmarkConfig,
    image_count: int,
    image_size: tuple[int, int],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> BenchmarkResult:
    """
    Run a pipeline benchmark on an adapter.

    This benchmarks pure pipeline performance by pre-decoding images
    before timing. This removes PNG decode overhead for fair comparison
    across frameworks.

    Args:
        adapter: Framework adapter to benchmark.
        benchmark: Pipeline benchmark configuration.
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

    # Pre-decode images to native format (removes decode overhead from timing)
    decoded_images = adapter.prepare_decoded_images(image_set.image_bytes)

    def run() -> object:
        return adapter.run_pipeline_on_decoded(decoded_images, benchmark.operations)

    stats = measure(
        run,
        warmup_fn=run,
        warmup=warmup_iterations,
        iterations=benchmark_iterations,
        label=benchmark.name,
    )
    memory = measure_memory(run)

    return to_result(
        stats,
        framework=adapter.name,
        engine=adapter.engine,
        operation=benchmark.name,
        image_count=image_count,
        image_size=image_size,
        thread_pool_size=adapter.thread_pool_size,
        memory=memory,
    )


def run_pipeline_benchmark_gpu(
    adapter: Any,  # TorchvisionAdapter with GPU support
    benchmark: PipelineBenchmarkConfig,
    image_count: int,
    image_size: tuple[int, int],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run a pipeline benchmark on a GPU adapter with cold and warm starts.

    Cold start: Uses pre-decoded images (decode overhead removed, but includes
    transfer to GPU). This is comparable to other frameworks' pre-decoded benchmarks.

    Warm start: Data already resident on GPU (pure operation performance).

    Args:
        adapter: GPU-capable framework adapter.
        benchmark: Pipeline benchmark configuration.
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

    # Pre-decode images to native format (removes PNG decode overhead)
    decoded_images = adapter.prepare_decoded_images(image_set.image_bytes)

    def run_cold() -> object:
        result = adapter.run_pipeline_on_decoded(decoded_images, benchmark.operations)
        # Inside the timed region deliberately: without the sync the GPU call
        # is asynchronous and the measurement is of queue submission.
        adapter.synchronize()
        return result

    cold_stats = measure(
        run_cold,
        warmup_fn=run_cold,
        warmup=warmup_iterations,
        iterations=benchmark_iterations,
        label=f"{benchmark.name}[cold]",
    )
    cold_result = to_result(
        cold_stats,
        framework=adapter.name,
        engine=adapter.engine,
        operation=benchmark.name,
        image_count=image_count,
        image_size=image_size,
        thread_pool_size=adapter.thread_pool_size,
        # Device memory is not tracked; `None` says so, where 0.0 claimed a
        # measurement of nothing.
        memory=None,
        gpu_mode="cold",
    )

    # Warm start benchmark (data already on GPU)
    preloaded = adapter.preload_to_device(image_set.image_bytes)
    adapter.synchronize()

    def run_warm() -> object:
        result = adapter.run_pipeline_batch_warm(preloaded, benchmark.operations)
        adapter.synchronize()
        return result

    warm_stats = measure(
        run_warm,
        warmup_fn=run_warm,
        warmup=warmup_iterations,
        iterations=benchmark_iterations,
        label=f"{benchmark.name}[warm]",
    )
    warm_result = to_result(
        warm_stats,
        framework=adapter.name,
        engine=adapter.engine,
        operation=benchmark.name,
        image_count=image_count,
        image_size=image_size,
        thread_pool_size=adapter.thread_pool_size,
        memory=None,
        gpu_mode="warm",
    )

    return cold_result, warm_result


def run_all_pipelines(
    adapters: list[BaseFrameworkAdapter],
    image_counts: list[int],
    image_sizes: list[tuple[int, int]],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    complexity_filter: str | None = None,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Run all pipeline benchmarks across all adapters and configurations.

    Args:
        adapters: List of framework adapters to benchmark.
        image_counts: List of image counts to test.
        image_sizes: List of image sizes to test.
        warmup_iterations: Number of warmup runs.
        benchmark_iterations: Number of timed runs.
        complexity_filter: If set, only run pipelines of this complexity.
        verbose: Whether to print progress output.

    Returns:
        List of all benchmark results.
    """
    results: list[BenchmarkResult] = []

    # Count total combinations for progress
    sample_benchmarks = get_pipeline_benchmarks()
    if complexity_filter:
        sample_benchmarks = [
            b for b in sample_benchmarks if b.complexity == complexity_filter
        ]
    total_combinations = (
        len(image_sizes) * len(image_counts) * len(sample_benchmarks) * len(adapters)
    )
    current = 0

    for size_idx, size in enumerate(image_sizes):
        benchmarks = get_pipeline_benchmarks(size[1], size[0])

        # Filter by complexity if requested
        if complexity_filter:
            benchmarks = [b for b in benchmarks if b.complexity == complexity_filter]

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
                            cold, warm = run_pipeline_benchmark_gpu(
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
                            result = run_pipeline_benchmark(
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
