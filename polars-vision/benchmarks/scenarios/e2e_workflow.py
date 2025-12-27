"""
End-to-end workflow benchmark scenarios.

This module provides benchmarks for complete workflows from file loading
to processed output in memory, simulating real-world usage patterns.
"""

from __future__ import annotations

import time
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
from benchmarks.utils.memory import run_timed_with_memory

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

    # Warmup
    for _ in range(warmup_iterations):
        run_workflow()

    # Benchmark
    total_time = 0.0
    peak_memory = 0.0

    for _ in range(benchmark_iterations):
        _, elapsed, mem_stats = run_timed_with_memory(run_workflow)
        total_time += elapsed
        peak_memory = max(peak_memory, mem_stats.peak_memory_mb)

    avg_time = total_time / benchmark_iterations
    throughput = image_count / avg_time
    latency_ms = (avg_time / image_count) * 1000

    # Get image size from first file
    first_img = adapter.load_from_file(file_paths[0])
    arr = adapter.to_numpy(first_img)
    image_size = (arr.shape[1], arr.shape[0])  # (width, height)

    return BenchmarkResult(
        framework=adapter.name,
        operation=f"e2e_{workflow.name}",
        image_count=image_count,
        image_size=image_size,
        total_time_seconds=avg_time,
        throughput_images_per_second=throughput,
        latency_ms_per_image=latency_ms,
        peak_memory_mb=peak_memory,
    )


def run_e2e_workflow_polars(
    adapter: Any,  # PolarsVisionAdapter
    workflow: E2EWorkflowConfig,
    file_paths: list[Path],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> BenchmarkResult:
    """
    Run an end-to-end workflow benchmark using Polars file loading.

    This uses Polars' native file reading capabilities for the polars-vision
    adapter, which may provide different performance characteristics.

    Args:
        adapter: Polars-vision adapter.
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

    # Warmup
    for _ in range(warmup_iterations):
        run_workflow()

    # Benchmark
    total_time = 0.0
    peak_memory = 0.0

    for _ in range(benchmark_iterations):
        _, elapsed, mem_stats = run_timed_with_memory(run_workflow)
        total_time += elapsed
        peak_memory = max(peak_memory, mem_stats.peak_memory_mb)

    avg_time = total_time / benchmark_iterations
    throughput = image_count / avg_time
    latency_ms = (avg_time / image_count) * 1000

    # Determine image size from file
    from PIL import Image

    with Image.open(file_paths[0]) as img:
        image_size = img.size  # (width, height)

    return BenchmarkResult(
        framework=adapter.name,
        operation=f"e2e_{workflow.name}",
        image_count=image_count,
        image_size=image_size,
        total_time_seconds=avg_time,
        throughput_images_per_second=throughput,
        latency_ms_per_image=latency_ms,
        peak_memory_mb=peak_memory,
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

    # Warmup
    for _ in range(warmup_iterations):
        adapter.run_pipeline_batch(image_bytes[:10], workflow.operations)
        adapter.synchronize()

    # Cold start (load + process)
    cold_total_time = 0.0

    for _ in range(benchmark_iterations):
        start = time.perf_counter()
        adapter.run_pipeline_batch(image_bytes, workflow.operations)
        adapter.synchronize()
        elapsed = time.perf_counter() - start
        cold_total_time += elapsed

    cold_avg_time = cold_total_time / benchmark_iterations
    cold_throughput = image_count / cold_avg_time
    cold_latency_ms = (cold_avg_time / image_count) * 1000

    # Determine image size
    from PIL import Image

    with Image.open(file_paths[0]) as img:
        image_size = img.size

    cold_result = BenchmarkResult(
        framework=adapter.name,
        operation=f"e2e_{workflow.name}",
        image_count=image_count,
        image_size=image_size,
        total_time_seconds=cold_avg_time,
        throughput_images_per_second=cold_throughput,
        latency_ms_per_image=cold_latency_ms,
        peak_memory_mb=0.0,
        gpu_mode="cold",
    )

    # Warm start (data already on GPU)
    preloaded = adapter.preload_to_device(image_bytes)
    adapter.synchronize()

    warm_total_time = 0.0

    for _ in range(benchmark_iterations):
        start = time.perf_counter()
        adapter.run_pipeline_batch_warm(preloaded, workflow.operations)
        adapter.synchronize()
        elapsed = time.perf_counter() - start
        warm_total_time += elapsed

    warm_avg_time = warm_total_time / benchmark_iterations
    warm_throughput = image_count / warm_avg_time
    warm_latency_ms = (warm_avg_time / image_count) * 1000

    warm_result = BenchmarkResult(
        framework=adapter.name,
        operation=f"e2e_{workflow.name}",
        image_count=image_count,
        image_size=image_size,
        total_time_seconds=warm_avg_time,
        throughput_images_per_second=warm_throughput,
        latency_ms_per_image=warm_latency_ms,
        peak_memory_mb=0.0,
        gpu_mode="warm",
    )

    return cold_result, warm_result


def run_all_e2e_workflows(
    adapters: list[BaseFrameworkAdapter],
    image_counts: list[int],
    image_sizes: list[tuple[int, int]],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
) -> list[BenchmarkResult]:
    """
    Run all end-to-end workflow benchmarks.

    Args:
        adapters: List of framework adapters to benchmark.
        image_counts: List of image counts to test.
        image_sizes: List of image sizes to test.
        warmup_iterations: Number of warmup runs.
        benchmark_iterations: Number of timed runs.

    Returns:
        List of all benchmark results.
    """
    results: list[BenchmarkResult] = []
    workflows = get_e2e_workflows()

    for size in image_sizes:
        width, height = size

        for count in image_counts:
            # Create temporary image files
            with temporary_image_set(
                count=count,
                height=height,
                width=width,
                channels=3,
                pattern="gradient",
            ) as image_set:
                if image_set.file_paths is None:
                    continue

                file_paths = image_set.file_paths

                for workflow in workflows:
                    for adapter in adapters:
                        if not adapter.is_available():
                            continue

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
                            elif "polars" in adapter.name.lower():
                                # Polars-vision adapter
                                result = run_e2e_workflow_polars(
                                    adapter,
                                    workflow,
                                    file_paths,
                                    warmup_iterations,
                                    benchmark_iterations,
                                )
                                results.append(result)
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
                        except Exception as e:
                            print(
                                f"Error in E2E benchmark {adapter.name}/"
                                f"{workflow.name}: {e}"
                            )

    return results
