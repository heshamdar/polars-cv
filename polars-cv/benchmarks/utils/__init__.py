"""
Utility modules for benchmarking.

Includes:
- data_gen: Test image generation
- memory: Memory profiling utilities
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
    run_timed_with_memory,
    track_memory,
)
from .results import ResultsCollector, format_table_rich, print_summary
from .validation import OutputValidator, ValidationResult, validate_outputs

__all__ = [
    "GeneratedImageSet",
    "MemoryStats",
    "MemoryTracker",
    "OutputValidator",
    "ResultsCollector",
    "ValidationResult",
    "format_table_rich",
    "generate_image_bytes",
    "generate_image_set",
    "get_current_memory_mb",
    "print_summary",
    "run_timed_with_memory",
    "temporary_image_set",
    "track_memory",
    "validate_outputs",
]
