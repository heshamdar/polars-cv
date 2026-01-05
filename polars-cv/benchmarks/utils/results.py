"""
Results collection and formatting utilities.

This module provides functions for collecting, aggregating, and displaying
benchmark results in various formats (tables, JSON, CSV).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from benchmarks.frameworks import BenchmarkResult

if TYPE_CHECKING:
    pass


def group_results_by_operation(
    results: list[BenchmarkResult],
) -> dict[str, list[BenchmarkResult]]:
    """
    Group results by operation name.

    Args:
        results: List of benchmark results.

    Returns:
        Dictionary mapping operation names to results.
    """
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        key = result.operation
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    return grouped


def group_results_by_framework(
    results: list[BenchmarkResult],
) -> dict[str, list[BenchmarkResult]]:
    """
    Group results by framework name.

    Args:
        results: List of benchmark results.

    Returns:
        Dictionary mapping framework names to results.
    """
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        key = result.framework
        if result.gpu_mode:
            key = f"{key} [{result.gpu_mode}]"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    return grouped


def results_to_dict_list(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    """
    Convert results to a list of dictionaries.

    Args:
        results: List of benchmark results.

    Returns:
        List of dictionaries.
    """
    return [asdict(r) for r in results]


def results_to_json(results: list[BenchmarkResult], indent: int = 2) -> str:
    """
    Convert results to JSON string.

    Args:
        results: List of benchmark results.
        indent: JSON indentation level.

    Returns:
        JSON string.
    """
    data = results_to_dict_list(results)
    return json.dumps(data, indent=indent)


def results_to_csv(results: list[BenchmarkResult]) -> str:
    """
    Convert results to CSV string.

    Args:
        results: List of benchmark results.

    Returns:
        CSV string.
    """
    if not results:
        return ""

    # Get headers from first result
    headers = [
        "framework",
        "operation",
        "image_count",
        "image_size",
        "total_time_seconds",
        "throughput_images_per_second",
        "latency_ms_per_image",
        "peak_memory_mb",
        "gpu_mode",
    ]

    lines = [",".join(headers)]

    for r in results:
        row = [
            r.framework,
            r.operation,
            str(r.image_count),
            f"{r.image_size[0]}x{r.image_size[1]}",
            f"{r.total_time_seconds:.4f}",
            f"{r.throughput_images_per_second:.2f}",
            f"{r.latency_ms_per_image:.4f}",
            f"{r.peak_memory_mb:.2f}",
            r.gpu_mode or "",
        ]
        lines.append(",".join(row))

    return "\n".join(lines)


def format_table_rich(
    results: list[BenchmarkResult],
    title: str = "Benchmark Results",
) -> str:
    """
    Format results as a rich table (requires rich library).

    Args:
        results: List of benchmark results.
        title: Table title.

    Returns:
        Formatted table string.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        # Create table
        table = Table(title=title)

        # Add columns
        table.add_column("Framework", style="cyan", no_wrap=True)
        table.add_column("Throughput (img/s)", justify="right", style="green")
        table.add_column("Latency (ms/img)", justify="right", style="yellow")
        table.add_column("Memory (MB)", justify="right", style="red")

        # Sort by throughput (descending)
        sorted_results = sorted(
            results,
            key=lambda r: r.throughput_images_per_second,
            reverse=True,
        )

        # Add rows
        for r in sorted_results:
            framework = r.framework
            if r.gpu_mode:
                framework = f"{framework} [{r.gpu_mode}]"

            table.add_row(
                framework,
                f"{r.throughput_images_per_second:,.1f}",
                f"{r.latency_ms_per_image:.3f}",
                f"{r.peak_memory_mb:.1f}",
            )

        # Render to string
        console = Console(record=True, width=100)
        console.print(table)
        return console.export_text()

    except ImportError:
        # Fallback to simple format
        return format_table_simple(results, title)


def format_table_simple(
    results: list[BenchmarkResult],
    title: str = "Benchmark Results",
) -> str:
    """
    Format results as a simple ASCII table.

    Args:
        results: List of benchmark results.
        title: Table title.

    Returns:
        Formatted table string.
    """
    if not results:
        return "No results to display."

    # Column widths
    fw_width = max(len(r.framework) + (8 if r.gpu_mode else 0) for r in results)
    fw_width = max(fw_width, len("Framework"))

    lines = []
    lines.append(f"\n=== {title} ===\n")

    # Header
    header = (
        f"| {'Framework':<{fw_width}} | "
        f"{'Throughput (img/s)':>18} | "
        f"{'Latency (ms/img)':>16} | "
        f"{'Memory (MB)':>12} |"
    )
    separator = "-" * len(header)

    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    # Sort by throughput
    sorted_results = sorted(
        results,
        key=lambda r: r.throughput_images_per_second,
        reverse=True,
    )

    # Rows
    for r in sorted_results:
        framework = r.framework
        if r.gpu_mode:
            framework = f"{framework} [{r.gpu_mode}]"

        row = (
            f"| {framework:<{fw_width}} | "
            f"{r.throughput_images_per_second:>18,.1f} | "
            f"{r.latency_ms_per_image:>16.3f} | "
            f"{r.peak_memory_mb:>12.1f} |"
        )
        lines.append(row)

    lines.append(separator)
    return "\n".join(lines)


def format_comparison_table(
    results: list[BenchmarkResult],
    operation: str,
    image_count: int,
    image_size: tuple[int, int],
) -> str:
    """
    Format a comparison table for a specific operation and configuration.

    Args:
        results: All benchmark results.
        operation: Operation to filter by.
        image_count: Image count to filter by.
        image_size: Image size to filter by.

    Returns:
        Formatted comparison table.
    """
    # Filter results
    filtered = [
        r
        for r in results
        if r.operation == operation
        and r.image_count == image_count
        and r.image_size == image_size
    ]

    if not filtered:
        return f"No results for {operation} with {image_count} images at {image_size}"

    title = f"{operation} | {image_count} images | {image_size[0]}x{image_size[1]}"

    return format_table_rich(filtered, title)


def print_summary(results: list[BenchmarkResult]) -> None:
    """
    Print a summary of benchmark results.

    Args:
        results: List of benchmark results.
    """
    if not results:
        print("No benchmark results.")
        return

    # Group by operation
    by_operation = group_results_by_operation(results)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for op_name, op_results in by_operation.items():
        print(f"\n--- {op_name} ---")

        # Find unique configurations
        configs = set((r.image_count, r.image_size) for r in op_results)

        for count, size in sorted(configs):
            config_results = [
                r for r in op_results if r.image_count == count and r.image_size == size
            ]

            print(f"\n  {count} images @ {size[0]}x{size[1]}:")

            # Sort by throughput
            sorted_results = sorted(
                config_results,
                key=lambda r: r.throughput_images_per_second,
                reverse=True,
            )

            for r in sorted_results:
                framework = r.framework
                if r.gpu_mode:
                    framework = f"{framework} [{r.gpu_mode}]"
                print(
                    f"    {framework:30s} "
                    f"{r.throughput_images_per_second:>10,.1f} img/s "
                    f"{r.latency_ms_per_image:>8.2f} ms/img"
                )

    print("\n" + "=" * 60)


class ResultsCollector:
    """
    Collector for benchmark results with aggregation and export capabilities.
    """

    def __init__(self) -> None:
        """Initialize an empty results collector."""
        self.results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult) -> None:
        """
        Add a result to the collector.

        Args:
            result: Benchmark result to add.
        """
        self.results.append(result)

    def add_many(self, results: list[BenchmarkResult]) -> None:
        """
        Add multiple results to the collector.

        Args:
            results: List of benchmark results to add.
        """
        self.results.extend(results)

    def to_json(self, indent: int = 2) -> str:
        """
        Export results to JSON.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        return results_to_json(self.results, indent)

    def to_csv(self) -> str:
        """
        Export results to CSV.

        Returns:
            CSV string.
        """
        return results_to_csv(self.results)

    def print_tables(self) -> None:
        """Print formatted tables for all results."""
        by_operation = group_results_by_operation(self.results)

        for op_name, op_results in by_operation.items():
            # Get unique configurations
            configs = set((r.image_count, r.image_size) for r in op_results)

            for count, size in sorted(configs):
                config_results = [
                    r
                    for r in op_results
                    if r.image_count == count and r.image_size == size
                ]
                title = f"{op_name} | {count} images | {size[0]}x{size[1]}"
                print(format_table_rich(config_results, title))
                print()

    def print_summary(self) -> None:
        """Print a summary of all results."""
        print_summary(self.results)

    def clear(self) -> None:
        """Clear all results."""
        self.results = []
