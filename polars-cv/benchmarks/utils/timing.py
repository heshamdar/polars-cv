"""Time one polars-cv call the way every regression scenario reports it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarks.utils.memory import run_timed_with_memory

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarks.frameworks import BenchmarkResult


def timed_result(
    label: str,
    call: Callable[[], Any],
    item_count: int,
    image_size: tuple[int, int],
    warmup_iterations: int,
    benchmark_iterations: int,
) -> BenchmarkResult:
    """Run *call* warmup + timed times; report the mean as a suite result.

    ``item_count`` is what one call processes (images, or rows for geometry),
    so ``throughput_images_per_second`` is items per second. Every case here
    runs eager, so ``framework`` is fixed rather than measured.
    """
    from benchmarks.frameworks import BenchmarkResult

    for _ in range(warmup_iterations):
        call()

    total_time = 0.0
    peak_memory = 0.0
    for _ in range(benchmark_iterations):
        _, elapsed, mem_stats = run_timed_with_memory(call)
        total_time += elapsed
        peak_memory = max(peak_memory, mem_stats.peak_memory_mb)

    avg_time = total_time / benchmark_iterations
    return BenchmarkResult(
        framework="polars-cv-eager",
        operation=label,
        image_count=item_count,
        image_size=image_size,
        total_time_seconds=avg_time,
        throughput_images_per_second=item_count / avg_time,
        latency_ms_per_image=(avg_time / item_count) * 1000,
        peak_memory_mb=peak_memory,
    )
