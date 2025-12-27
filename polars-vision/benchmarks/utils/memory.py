"""
Memory profiling utilities for benchmarking.

This module provides functions for measuring peak memory usage during
benchmark runs using psutil for accurate RSS tracking.
"""

from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, TypeVar

if TYPE_CHECKING:
    pass

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

T = TypeVar("T")


@dataclass
class MemoryStats:
    """Memory statistics from a measurement run."""

    peak_memory_mb: float
    start_memory_mb: float
    end_memory_mb: float
    delta_mb: float

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MemoryStats(peak={self.peak_memory_mb:.1f}MB, "
            f"delta={self.delta_mb:+.1f}MB)"
        )


def get_current_memory_mb() -> float:
    """
    Get current process memory usage in MB.

    Returns:
        Current RSS memory in megabytes.
    """
    if not PSUTIL_AVAILABLE:
        return 0.0

    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def force_gc() -> None:
    """
    Force garbage collection to get accurate memory readings.

    Runs multiple GC passes to ensure all garbage is collected.
    """
    for _ in range(3):
        gc.collect()


class MemoryTracker:
    """
    Tracks peak memory usage during a code block.

    Uses a background thread to sample memory at regular intervals.

    Attributes:
        sample_interval_ms: Time between memory samples in milliseconds.
    """

    def __init__(self, sample_interval_ms: float = 10.0) -> None:
        """
        Initialize the memory tracker.

        Args:
            sample_interval_ms: Time between memory samples in milliseconds.
        """
        self.sample_interval_ms = sample_interval_ms
        self._peak_memory_mb: float = 0.0
        self._start_memory_mb: float = 0.0
        self._end_memory_mb: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _sample_loop(self) -> None:
        """Background thread that samples memory usage."""
        while self._running:
            current = get_current_memory_mb()
            with self._lock:
                if current > self._peak_memory_mb:
                    self._peak_memory_mb = current
            time.sleep(self.sample_interval_ms / 1000.0)

    def start(self) -> None:
        """Start tracking memory usage."""
        force_gc()
        self._start_memory_mb = get_current_memory_mb()
        self._peak_memory_mb = self._start_memory_mb
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> MemoryStats:
        """
        Stop tracking and return memory statistics.

        Returns:
            MemoryStats with peak and delta measurements.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Final sample
        self._end_memory_mb = get_current_memory_mb()
        with self._lock:
            peak = max(self._peak_memory_mb, self._end_memory_mb)

        return MemoryStats(
            peak_memory_mb=peak,
            start_memory_mb=self._start_memory_mb,
            end_memory_mb=self._end_memory_mb,
            delta_mb=self._end_memory_mb - self._start_memory_mb,
        )


@contextmanager
def track_memory(sample_interval_ms: float = 10.0) -> Iterator[MemoryTracker]:
    """
    Context manager for tracking memory usage during a code block.

    Args:
        sample_interval_ms: Time between memory samples in milliseconds.

    Yields:
        MemoryTracker instance. Call tracker.stop() to get stats after the block.

    Example:
        >>> with track_memory() as tracker:
        ...     # Code that uses memory
        ...     result = process_images(data)
        >>> stats = tracker.stop()
        >>> print(f"Peak memory: {stats.peak_memory_mb:.1f} MB")
    """
    tracker = MemoryTracker(sample_interval_ms)
    tracker.start()
    try:
        yield tracker
    finally:
        if tracker._running:
            tracker.stop()


@dataclass
class TimedResult(MemoryStats):
    """Result of a timed and memory-tracked operation."""

    elapsed_seconds: float = 0.0
    result: object = None

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TimedResult(time={self.elapsed_seconds:.3f}s, "
            f"peak={self.peak_memory_mb:.1f}MB)"
        )


def run_with_memory_tracking(
    func: Callable[[], T],
    sample_interval_ms: float = 10.0,
) -> tuple[T, MemoryStats]:
    """
    Run a function while tracking memory usage.

    Args:
        func: Function to run (takes no arguments).
        sample_interval_ms: Time between memory samples.

    Returns:
        Tuple of (function result, memory stats).
    """
    tracker = MemoryTracker(sample_interval_ms)
    tracker.start()
    try:
        result = func()
    finally:
        stats = tracker.stop()

    return result, stats


def run_timed_with_memory(
    func: Callable[[], T],
    sample_interval_ms: float = 10.0,
) -> tuple[T, float, MemoryStats]:
    """
    Run a function while tracking both time and memory.

    Args:
        func: Function to run (takes no arguments).
        sample_interval_ms: Time between memory samples.

    Returns:
        Tuple of (function result, elapsed seconds, memory stats).
    """
    tracker = MemoryTracker(sample_interval_ms)
    tracker.start()
    start_time = time.perf_counter()
    try:
        result = func()
    finally:
        elapsed = time.perf_counter() - start_time
        stats = tracker.stop()

    return result, elapsed, stats


def measure_baseline_memory() -> float:
    """
    Measure baseline memory after garbage collection.

    Returns:
        Baseline memory in MB.
    """
    force_gc()
    return get_current_memory_mb()
