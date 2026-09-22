"""
Single operation benchmark scenarios.

This module provides benchmarks for individual image processing operations
across all framework adapters.
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
        SingleOpBenchmarkConfig(
            operation=OperationType.ROTATE,
            name="rotate_90",
            params=OperationParams(
                operation=OperationType.ROTATE,
                angle=90.0,
                expand=False,
            ),
            description="Rotate 90 degrees (zero-copy fast path in polars-cv)",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.ROTATE,
            name="rotate_45",
            params=OperationParams(
                operation=OperationType.ROTATE,
                angle=45.0,
                expand=True,
            ),
            description="Rotate 45 degrees with canvas expansion",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.INVERT,
            name="invert",
            params=OperationParams(operation=OperationType.INVERT),
            description="Invert pixel values (255 - pixel)",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.ADJUST_CONTRAST,
            name="adjust_contrast",
            params=OperationParams(
                operation=OperationType.ADJUST_CONTRAST,
                contrast_factor=1.5,
            ),
            description="Adjust contrast (factor=1.5)",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.ADJUST_BRIGHTNESS,
            name="adjust_brightness",
            params=OperationParams(
                operation=OperationType.ADJUST_BRIGHTNESS,
                brightness_factor=1.3,
            ),
            description="Adjust brightness (factor=1.3)",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.SHARPEN,
            name="sharpen",
            params=OperationParams(
                operation=OperationType.SHARPEN,
                sharpen_strength=1.5,
            ),
            description="Sharpen image (strength=1.5)",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.PAD,
            name="pad",
            params=OperationParams(
                operation=OperationType.PAD,
                pad_top=20,
                pad_bottom=20,
                pad_left=20,
                pad_right=20,
                pad_value=0,
            ),
            description="Pad 20px on all edges",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.ERODE,
            name="erode",
            params=OperationParams(
                operation=OperationType.ERODE,
                ksize=3,
                iterations=1,
            ),
            description="Morphological erosion (3x3 kernel)",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.DILATE,
            name="dilate",
            params=OperationParams(
                operation=OperationType.DILATE,
                ksize=3,
                iterations=1,
            ),
            description="Morphological dilation (3x3 kernel)",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.HISTOGRAM_EQUALIZE,
            name="histogram_equalize",
            params=OperationParams(operation=OperationType.HISTOGRAM_EQUALIZE),
            description="Histogram equalization for contrast enhancement",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.CANNY,
            name="canny",
            params=OperationParams(
                operation=OperationType.CANNY,
                low_threshold=50.0,
                high_threshold=150.0,
            ),
            description="Canny edge detection (low=50, high=150)",
        ),
        SingleOpBenchmarkConfig(
            operation=OperationType.SOBEL,
            name="sobel_x",
            params=OperationParams(
                operation=OperationType.SOBEL,
                sobel_axis="x",
            ),
            description="Sobel gradient (X axis)",
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

    def run() -> object:
        return adapter.run_pipeline_on_decoded(decoded_images, operations)

    # Warm on the full working set, not a 10-image slice: the slice warmed the
    # allocator and the compiled-graph cache for a different shape than the one
    # being timed.
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

    def run_cold() -> object:
        result = adapter.run_pipeline_on_decoded(decoded_images, operations)
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
        # Host RSS says nothing about device memory, and reporting 0.0 would
        # claim it did. `memory=None` records "not requested" instead.
        memory=None,
        gpu_mode="cold",
    )

    # Warm start benchmark (data already on GPU)
    preloaded = adapter.preload_to_device(image_set.image_bytes)
    adapter.synchronize()

    def run_warm() -> object:
        result = adapter.run_pipeline_batch_warm(preloaded, operations)
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

    sample_benchmarks = get_single_op_benchmarks()
    total_combinations = (
        len(image_sizes) * len(image_counts) * len(sample_benchmarks) * len(adapters)
    )
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
