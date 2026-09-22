"""
Utility modules for benchmarking.

Includes:
- data_gen: Test image generation
- timing: THE timing authority (measure / measure_memory / to_result)
- memory: Memory sampling primitives, used only by `timing`
- results: Results collection and formatting
- validation: Output equality verification
"""

from __future__ import annotations

from .data_gen import (
    GeneratedImageSet,
    generate_image_bytes,
    generate_image_set,
    temporary_image_set,
)
from .memory import (
    MemoryStats,
    MemoryTracker,
    get_current_memory_mb,
    track_memory,
)
from .results import ResultsCollector, format_table_rich, print_summary
from .timing import MemoryMeasurement, TimingStats, measure, measure_memory, to_result
from .validation import OutputValidator, ValidationResult, validate_outputs

__all__ = [
    "GeneratedImageSet",
    "MemoryMeasurement",
    "MemoryStats",
    "MemoryTracker",
    "OutputValidator",
    "ResultsCollector",
    "TimingStats",
    "ValidationResult",
    "format_table_rich",
    "generate_image_bytes",
    "generate_image_set",
    "get_current_memory_mb",
    "measure",
    "measure_memory",
    "print_summary",
    "temporary_image_set",
    "to_result",
    "track_memory",
    "validate_outputs",
]
