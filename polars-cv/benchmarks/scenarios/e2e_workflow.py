"""
End-to-end workflow benchmark scenarios.

This module provides benchmarks for complete workflows from file loading
to processed output in memory, simulating real-world usage patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.frameworks import (
    BaseFrameworkAdapter,
    BenchmarkResult,
    OperationParams,
    OperationType,
)
from benchmarks.utils.data_gen import temporary_image_set
from benchmarks.utils.timing import measure, measure_memory, to_result

if TYPE_CHECKING:
    pass


@dataclass
class E2EWorkflowConfig:
    """Configuration for an end-to-end workflow benchmark."""

    name: str
    operations: list[OperationParams]
    description: str


def get_e2e_workflows() -> list[E2EWorkflowConfig]:
    """
    Get the list of end-to-end workflow benchmarks to run.

    Returns:
        List of workflow configurations.
    """
    return [
        E2EWorkflowConfig(
            name="basic_preprocess",
            operations=[
                OperationParams(
                    operation=OperationType.RESIZE,
                    height=224,
                    width=224,
                ),
                OperationParams(operation=OperationType.NORMALIZE),
            ],
            description="Load files → resize → normalize → memory",
        ),
        E2EWorkflowConfig(
            name="imagenet_workflow",
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
            description="Load files → resize → center crop → normalize → memory",
        ),
        E2EWorkflowConfig(
            name="augmentation_workflow",
            operations=[
                OperationParams(
                    operation=OperationType.RESIZE,
                    height=256,
                    width=256,
                ),
                OperationParams(operation=OperationType.FLIP_H),
                OperationParams(operation=OperationType.NORMALIZE),
            ],
            description="Load files → resize → flip → normalize → memory",
        ),
    ]


def run_e2e_workflow_standard(
    adapter: BaseFrameworkAdapter,
    workflow: E2EWorkflowConfig,
    file_paths: list[Path],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> BenchmarkResult:
    """
    Run an end-to-end workflow benchmark using standard file loading.

    Args:
        adapter: Framework adapter to benchmark.
        workflow: Workflow configuration.
        file_paths: List of image file paths.
        warmup_iterations: Number of warmup runs.
        benchmark_iterations: Number of timed runs.

    Returns:
        Benchmark result.
    """
    image_count = len(file_paths)

    def run_workflow() -> list[Any]:
        results = []
        for path in file_paths:
            img = adapter.load_from_file(path)
            for op in workflow.operations:
                img = adapter.apply_operation(img, op)
            results.append(adapter.to_numpy(img))
        return results

    stats = measure(
        run_workflow,
        warmup_fn=run_workflow,
        warmup=warmup_iterations,
        iterations=benchmark_iterations,
        label=f"e2e_{workflow.name}",
    )
    memory = measure_memory(run_workflow)

    # Get image size from first file
    first_img = adapter.load_from_file(file_paths[0])
    arr = adapter.to_numpy(first_img)
    image_size = (arr.shape[1], arr.shape[0])  # (width, height)

    return to_result(
        stats,
        framework=adapter.name,
        engine=adapter.engine,
        operation=f"e2e_{workflow.name}",
        image_count=image_count,
        image_size=image_size,
        thread_pool_size=adapter.thread_pool_size,
        memory=memory,
    )


def run_e2e_workflow_polars(
    adapter: Any,  # PolarsCVAdapter
    workflow: E2EWorkflowConfig,
    file_paths: list[Path],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> BenchmarkResult:
    """
    Run an end-to-end workflow benchmark using Polars file loading.

    This uses Polars' native file reading capabilities for the polars-cv
    adapter, which may provide different performance characteristics.

    Args:
        adapter: polars-cv adapter.
        workflow: Workflow configuration.
        file_paths: List of image file paths.
        warmup_iterations: Number of warmup runs.
        benchmark_iterations: Number of timed runs.

    Returns:
        Benchmark result.
    """
    image_count = len(file_paths)

    # Read all image bytes into a list
    image_bytes = [p.read_bytes() for p in file_paths]

    def run_workflow() -> list[Any]:
        return adapter.run_pipeline_batch(image_bytes, workflow.operations)

    stats = measure(
        run_workflow,
        warmup_fn=run_workflow,
        warmup=warmup_iterations,
        iterations=benchmark_iterations,
        label=f"e2e_{workflow.name}",
    )
    memory = measure_memory(run_workflow)

    # Determine image size from file
    from PIL import Image

    with Image.open(file_paths[0]) as img:
        image_size = img.size  # (width, height)

    return to_result(
        stats,
        framework=adapter.name,
        engine=adapter.engine,
        operation=f"e2e_{workflow.name}",
        image_count=image_count,
        image_size=image_size,
        thread_pool_size=adapter.thread_pool_size,
        memory=memory,
    )


def run_e2e_workflow_gpu(
    adapter: Any,  # TorchvisionAdapter with GPU
    workflow: E2EWorkflowConfig,
    file_paths: list[Path],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run an end-to-end workflow benchmark on GPU with cold and warm starts.

    Args:
        adapter: GPU-capable torchvision adapter.
        workflow: Workflow configuration.
        file_paths: List of image file paths.
        warmup_iterations: Number of warmup runs.
        benchmark_iterations: Number of timed runs.

    Returns:
        Tuple of (cold_start_result, warm_start_result).
    """
    image_count = len(file_paths)

    # Read image bytes
    image_bytes = [p.read_bytes() for p in file_paths]

    def run_cold() -> object:
        result = adapter.run_pipeline_batch(image_bytes, workflow.operations)
        # Inside the timed region deliberately: without the sync the GPU call
        # is asynchronous and the measurement is of queue submission.
        adapter.synchronize()
        return result

    cold_stats = measure(
        run_cold,
        warmup_fn=run_cold,
        warmup=warmup_iterations,
        iterations=benchmark_iterations,
        label=f"e2e_{workflow.name}[cold]",
    )

    # Determine image size
    from PIL import Image

    with Image.open(file_paths[0]) as img:
        image_size = img.size

    cold_result = to_result(
        cold_stats,
        framework=adapter.name,
        engine=adapter.engine,
        operation=f"e2e_{workflow.name}",
        image_count=image_count,
        image_size=image_size,
        thread_pool_size=adapter.thread_pool_size,
        # Device memory is not tracked; `None` says so, where 0.0 claimed a
        # measurement of nothing.
        memory=None,
        gpu_mode="cold",
    )

    # Warm start (data already on GPU)
    preloaded = adapter.preload_to_device(image_bytes)
    adapter.synchronize()

    def run_warm() -> object:
        result = adapter.run_pipeline_batch_warm(preloaded, workflow.operations)
        adapter.synchronize()
        return result

    warm_stats = measure(
        run_warm,
        warmup_fn=run_warm,
        warmup=warmup_iterations,
        iterations=benchmark_iterations,
        label=f"e2e_{workflow.name}[warm]",
    )
    warm_result = to_result(
        warm_stats,
        framework=adapter.name,
        engine=adapter.engine,
        operation=f"e2e_{workflow.name}",
        image_count=image_count,
        image_size=image_size,
        thread_pool_size=adapter.thread_pool_size,
        memory=None,
        gpu_mode="warm",
    )

    return cold_result, warm_result


def run_all_e2e_workflows(
    adapters: list[BaseFrameworkAdapter],
    image_counts: list[int],
    image_sizes: list[tuple[int, int]],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Run all end-to-end workflow benchmarks.

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
    workflows = get_e2e_workflows()

    # Count total combinations for progress
    total_combinations = (
        len(image_sizes) * len(image_counts) * len(workflows) * len(adapters)
    )
    current = 0

    for size_idx, size in enumerate(image_sizes):
        width, height = size

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

            # Create temporary image files
            if verbose:
                print("      Generating temporary image files...", end="", flush=True)

            with temporary_image_set(
                count=count,
                height=height,
                width=width,
                channels=3,
                pattern="gradient",
            ) as image_set:
                if image_set.file_paths is None:
                    if verbose:
                        print(" FAILED", flush=True)
                    continue

                if verbose:
                    print(" done", flush=True)

                file_paths = image_set.file_paths

                for workflow in workflows:
                    for adapter in adapters:
                        current += 1

                        if not adapter.is_available():
                            if verbose:
                                print(
                                    f"      [{current}/{total_combinations}] "
                                    f"{adapter.name}/{workflow.name}: "
                                    f"SKIPPED (unavailable)",
                                    flush=True,
                                )
                            continue

                        if verbose:
                            print(
                                f"      [{current}/{total_combinations}] "
                                f"{adapter.name}/{workflow.name}...",
                                end="",
                                flush=True,
                            )

                        try:
                            if adapter.supports_gpu and hasattr(adapter, "synchronize"):
                                # GPU adapter
                                cold, warm = run_e2e_workflow_gpu(
                                    adapter,
                                    workflow,
                                    file_paths,
                                    warmup_iterations,
                                    benchmark_iterations,
                                )
                                results.append(cold)
                                results.append(warm)
                                if verbose:
                                    print(
                                        f" {cold.throughput_images_per_second:.1f} "
                                        f"img/s (cold), "
                                        f"{warm.throughput_images_per_second:.1f} "
                                        f"img/s (warm)",
                                        flush=True,
                                    )
                            elif "polars" in adapter.name.lower():
                                # polars-cv adapter
                                result = run_e2e_workflow_polars(
                                    adapter,
                                    workflow,
                                    file_paths,
                                    warmup_iterations,
                                    benchmark_iterations,
                                )
                                results.append(result)
                                if verbose:
                                    print(
                                        f" {result.throughput_images_per_second:.1f} "
                                        f"img/s",
                                        flush=True,
                                    )
                            else:
                                # Standard adapter
                                result = run_e2e_workflow_standard(
                                    adapter,
                                    workflow,
                                    file_paths,
                                    warmup_iterations,
                                    benchmark_iterations,
                                )
                                results.append(result)
                                if verbose:
                                    print(
                                        f" {result.throughput_images_per_second:.1f} "
                                        f"img/s",
                                        flush=True,
                                    )
                        except Exception as e:
                            if verbose:
                                print(f" ERROR: {e}", flush=True)
                            else:
                                print(
                                    f"Error in E2E benchmark {adapter.name}/"
                                    f"{workflow.name}: {e}",
                                    flush=True,
                                )

    return results
