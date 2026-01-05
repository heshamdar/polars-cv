"""
Framework adapters for benchmarking.

Each adapter provides a consistent interface for image processing operations
across different frameworks (polars-cv, OpenCV, PIL, torchvision).
"""

from __future__ import annotations

from .base import BaseFrameworkAdapter, BenchmarkResult, OperationParams, OperationType
from .opencv_adapter import OpenCVAdapter
from .pillow_adapter import PillowAdapter
from .polars_cv_adapter import (
    PolarsCVAdapter,
    PolarsCVEagerAdapter,
    PolarsCVStreamingAdapter,
)
from .torchvision_adapter import (
    TorchvisionAdapter,
    TorchvisionCPUAdapter,
    TorchvisionCUDAAdapter,
    TorchvisionMPSAdapter,
)

__all__ = [
    "BaseFrameworkAdapter",
    "BenchmarkResult",
    "OpenCVAdapter",
    "OperationParams",
    "OperationType",
    "PillowAdapter",
    "PolarsCVAdapter",
    "PolarsCVEagerAdapter",
    "PolarsCVStreamingAdapter",
    "TorchvisionAdapter",
    "TorchvisionCPUAdapter",
    "TorchvisionCUDAAdapter",
    "TorchvisionMPSAdapter",
]


def get_adapter(name: str) -> BaseFrameworkAdapter:
    """
    Get a framework adapter by name.

    Args:
        name: Adapter name (e.g., "opencv", "polars-cv-eager").

    Returns:
        Framework adapter instance.

    Raises:
        ValueError: If adapter name is not recognized.
    """
    adapters: dict[str, type[BaseFrameworkAdapter]] = {
        "opencv": OpenCVAdapter,
        "pillow": PillowAdapter,
        "polars-cv-eager": PolarsCVEagerAdapter,
        "polars-cv-streaming": PolarsCVStreamingAdapter,
        "torchvision-cpu": TorchvisionCPUAdapter,
        "torchvision-mps": TorchvisionMPSAdapter,
        "torchvision-cuda": TorchvisionCUDAAdapter,
    }

    if name not in adapters:
        valid = list(adapters.keys())
        msg = f"Unknown adapter: {name}. Valid: {valid}"
        raise ValueError(msg)

    return adapters[name]()


def get_available_adapters() -> list[BaseFrameworkAdapter]:
    """
    Get all available framework adapters.

    Returns:
        List of adapters that are available (dependencies installed).
    """
    all_adapters = [
        PolarsCVEagerAdapter(),
        PolarsCVStreamingAdapter(),
        OpenCVAdapter(),
        PillowAdapter(),
        TorchvisionCPUAdapter(),
        TorchvisionMPSAdapter(),
    ]

    return [a for a in all_adapters if a.is_available()]
