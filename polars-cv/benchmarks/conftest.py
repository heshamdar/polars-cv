"""
Pytest configuration and fixtures for benchmarks.

This module provides shared fixtures and configuration options for
running the benchmarking suite with pytest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.config.argparsing import Parser


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    image_counts: list[int] = field(default_factory=lambda: [10, 100, 1000])
    image_sizes: list[tuple[int, int]] = field(
        default_factory=lambda: [(256, 256), (512, 512), (1024, 1024)]
    )
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    output_format: str = "csv"  # or "json", "csv"
    frameworks: list[str] = field(
        default_factory=lambda: [
            "polars-cv-eager",
            "polars-cv-streaming",
            "opencv",
            "pillow",
            "torchvision-cpu",
            "torchvision-mps",
        ]
    )


def pytest_addoption(parser: "Parser") -> None:
    """Add custom command-line options for benchmarks."""
    parser.addoption(
        "--benchmark-counts",
        action="store",
        default="10,100,1000",
        help="Comma-separated list of image counts to benchmark",
    )
    parser.addoption(
        "--benchmark-sizes",
        action="store",
        default="256,512,1024",
        help="Comma-separated list of image sizes to benchmark",
    )
    parser.addoption(
        "--benchmark-warmup",
        action="store",
        default=3,
        type=int,
        help="Number of warmup iterations",
    )
    parser.addoption(
        "--benchmark-iterations",
        action="store",
        default=10,
        type=int,
        help="Number of benchmark iterations",
    )
    parser.addoption(
        "--benchmark-frameworks",
        action="store",
        default=None,
        help="Comma-separated list of frameworks to benchmark",
    )
    parser.addoption(
        "--benchmark-output",
        action="store",
        default="table",
        choices=["table", "json", "csv"],
        help="Output format for benchmark results",
    )


@pytest.fixture
def benchmark_config(request: pytest.FixtureRequest) -> BenchmarkConfig:
    """
    Create benchmark configuration from command-line options.

    Args:
        request: Pytest fixture request.

    Returns:
        BenchmarkConfig instance with parsed options.
    """
    config = request.config

    # Parse image counts
    counts_str = config.getoption("--benchmark-counts")
    counts = [int(c.strip()) for c in counts_str.split(",")]

    # Parse image sizes
    sizes_str = config.getoption("--benchmark-sizes")
    sizes = [(int(s.strip()), int(s.strip())) for s in sizes_str.split(",")]

    # Parse frameworks
    frameworks_str = config.getoption("--benchmark-frameworks")
    if frameworks_str:
        frameworks = [f.strip() for f in frameworks_str.split(",")]
    else:
        frameworks = BenchmarkConfig().frameworks

    return BenchmarkConfig(
        image_counts=counts,
        image_sizes=sizes,
        warmup_iterations=config.getoption("--benchmark-warmup"),
        benchmark_iterations=config.getoption("--benchmark-iterations"),
        output_format=config.getoption("--benchmark-output"),
        frameworks=frameworks,
    )
