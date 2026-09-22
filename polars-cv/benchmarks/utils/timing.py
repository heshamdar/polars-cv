"""The single authority for how long something took, and what gets reported.

Every timed span in ``benchmarks/`` goes through :func:`measure` or
:func:`measure_phase`, and every result record is built by :func:`to_result`.
``tests/test_timing_authority.py`` rejects any module that reads a clock or
constructs a ``BenchmarkResult`` itself, so this is not a convention to
remember — it is the only way in.

Why both halves are guarded
---------------------------

Before this module the suite had six structurally-identical timing loops, plus
fourteen un-repeated spans in ``zero_copy_ingestion``, thirty in
``inference_pipeline_comparison``, and two further modules
(``plugin_overhead``, ``batch_throughput``) that reported *min and median*
while every scenario reported the *mean*. One suite, two answers to "what is
the number", and nothing able to notice the difference.

Banning the clock alone would not have fixed that. Each scenario also reduced
its own samples — four independent chances to divide by the wrong thing. So the
record constructor is the authority too: you cannot emit a result without
having gone through the code that computed the statistic in it.

What statistic, and why
-----------------------

The headline is the **median**, and the gate uses it.

- The *mean* — what every scenario used — is the worst of the three here.
  Benchmark noise is right-tailed: a co-tenant scheduling hiccup moves the mean
  and never moves it back, and on a shared CI runner that is most rounds.
- The *minimum* is right for a pure microbenchmark and systematically
  optimistic for the streaming engine, where it rewards thread-scheduling luck.
  It is recorded because it is the honest statistic for the protocol-floor
  measurement, but it does not gate.
- Dispersion (:attr:`TimingStats.mad_s`, :attr:`TimingStats.iqr_s`) and the raw
  samples travel into the record because the comparator needs per-key noise to
  classify honestly. Without it a threshold is a guess applied uniformly to
  keys with wildly different variance.

Garbage collection is **left enabled** during the timed region. Disabling it
would lower variance, but these workloads hold multi-megabyte image buffers and
a benchmark that suppresses collection is measuring a memory regime no user
runs. A full collection is forced *before* the loop instead, so a collection
owed by the setup does not land inside iteration one.

Memory is measured in a separate, **untimed** pass (:func:`measure_memory`).
The previous instrument sampled RSS from a background thread *inside* every
timed region — which held the GIL, and so taxed each framework in proportion to
how much Python it ran. That is a confound that tracks exactly the thing a
cross-framework comparison is trying to isolate.
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from benchmarks.frameworks.base import BenchmarkResult

T = TypeVar("T")

# Memory status strings. `OK` is the only one with numbers attached; the others
# carry `None`, never `0.0` — a zero reads as "measured, and it was nothing",
# which is how an unmeasured memory column previously compared as NEUTRAL.
MEMORY_OK = "ok"
MEMORY_UNAVAILABLE = "unavailable: psutil not installed"
MEMORY_NOT_REQUESTED = "not requested"


@dataclass(frozen=True)
class MemoryMeasurement:
    """Peak RSS over one untimed pass, relative to a post-GC baseline.

    ``delta_mb`` is the number to read. ``peak_mb`` is whole-process RSS, which
    is effectively monotonic within a process (the allocator's high-water mark
    does not come back down), so across a suite it tracks *run order* at least
    as much as it tracks the framework under test.
    """

    status: str
    peak_mb: float | None = None
    baseline_mb: float | None = None
    delta_mb: float | None = None

    @property
    def measured(self) -> bool:
        return self.status == MEMORY_OK


@dataclass(frozen=True)
class TimingStats:
    """Per-iteration wall times for one span, and the statistics over them.

    ``samples_s`` is retained in full. It is what lets the comparator compute a
    noise band per key rather than apply one invented threshold to every key,
    and it is small — ten floats per result.
    """

    label: str
    iterations: int
    warmup_iterations: int
    samples_s: tuple[float, ...]

    @property
    def min_s(self) -> float:
        return min(self.samples_s)

    @property
    def median_s(self) -> float:
        return statistics.median(self.samples_s)

    @property
    def mean_s(self) -> float:
        return statistics.fmean(self.samples_s)

    @property
    def mad_s(self) -> float:
        """Median absolute deviation — dispersion that one outlier cannot move."""
        med = self.median_s
        return statistics.median([abs(s - med) for s in self.samples_s])

    @property
    def iqr_s(self) -> float:
        """Interquartile range; 0.0 for fewer than four samples."""
        if len(self.samples_s) < 4:
            return 0.0
        quartiles = statistics.quantiles(self.samples_s, n=4, method="inclusive")
        return quartiles[2] - quartiles[0]

    @property
    def relative_mad(self) -> float:
        """MAD as a fraction of the median — the unit a threshold compares to."""
        med = self.median_s
        return self.mad_s / med if med > 0 else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "samples_s": list(self.samples_s),
            "min_s": self.min_s,
            "median_s": self.median_s,
            "mean_s": self.mean_s,
            "mad_s": self.mad_s,
            "iqr_s": self.iqr_s,
            "relative_mad": self.relative_mad,
        }


def measure(
    fn: Callable[[], object],
    *,
    warmup_fn: Callable[[], object],
    warmup: int,
    iterations: int,
    label: str,
) -> TimingStats:
    """Time *fn* ``iterations`` times after ``warmup`` runs of *warmup_fn*.

    ``warmup_fn`` is required, and callers pass a callable over the **full**
    working set. Every scenario previously warmed on ``image_bytes[:10]`` and
    measured the whole set, so the allocator and the compiled-graph cache were
    warmed for a different shape than the one being timed. Making the parameter
    required is what stops the next scenario forgetting; making it a separate
    callable is what lets a scenario warm a cheaper equivalent when the full set
    genuinely cannot be afforded — deliberately, and visibly at the call site.

    Args:
        fn: The callable to time. Takes no arguments; return value is discarded.
        warmup_fn: Callable run ``warmup`` times before timing starts.
        warmup: Number of untimed warmup runs. Must be >= 1.
        iterations: Number of timed runs. Must be >= 1.
        label: Name for this span, carried into the result for attribution.

    Returns:
        TimingStats over the timed runs.

    Raises:
        ValueError: If ``warmup`` or ``iterations`` is below 1. A zero-warmup
            run measures the compiled-graph cache miss in iteration one, and a
            zero-iteration run has no statistic at all; both used to be
            expressible and neither is meaningful.
    """
    if warmup < 1:
        raise ValueError(
            f"warmup must be >= 1, got {warmup}: the first call to a polars-cv "
            f"pipeline pays graph compilation, so an unwarmed iteration one "
            f"measures the cache miss rather than the op."
        )
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    for _ in range(warmup):
        warmup_fn()

    # Settle anything the setup and warmup left owed, so it is not collected
    # inside iteration one. GC stays enabled during the loop — see module
    # docstring.
    gc.collect()

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)

    return TimingStats(
        label=label,
        iterations=iterations,
        warmup_iterations=warmup,
        samples_s=tuple(samples),
    )


def measure_phase(fn: Callable[[], T], *, label: str) -> tuple[T, float]:
    """Time a single un-repeated span, returning its value and elapsed seconds.

    For phases that genuinely happen once — a corpus build, a model load, a
    one-shot ingestion — where repeating would measure a warm cache rather than
    the phase. It returns a bare float rather than :class:`TimingStats` so that
    a single sample can never be mistaken for a distribution: there is no
    median to report and no dispersion to gate on.

    A span that *can* be repeated should use :func:`measure` instead.
    """
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


class Stopwatch:
    """A running clock for one-shot phases in straight-line script code.

    :func:`measure` needs a callable, which is right for anything repeatable.
    Some phases are not: a dataset load, a model download, the first-batch
    latency of a freshly built DataLoader. Repeating those measures a warm
    cache rather than the phase. Wrapping each in a closure only to call it
    once obscures the script more than it clarifies it.

    So this exists, and it still routes through the authority — which is the
    point. It reports a single sample, deliberately with no median and no
    dispersion, so a one-shot number can never be mistaken for a distribution.
    Anything repeatable uses :func:`measure`.
    """

    __slots__ = ("label", "_start")

    def __init__(self, label: str) -> None:
        self.label = label
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        """Seconds since construction. Safe to call more than once."""
        return time.perf_counter() - self._start


def stopwatch(label: str = "") -> Stopwatch:
    """Start a :class:`Stopwatch`. See its docstring for when this is right."""
    return Stopwatch(label)


def measure_memory(
    fn: Callable[[], object],
    *,
    sample_interval_ms: float = 10.0,
) -> MemoryMeasurement:
    """Run *fn* once, untimed, sampling peak RSS.

    Deliberately separate from :func:`measure`: the sampler thread holds the
    GIL every ``sample_interval_ms``, which perturbs wall time in proportion to
    how much Python the callable runs — a confound that tracks framework choice.

    Returns a measurement whose ``status`` says why the numbers are ``None``
    when they are. It never reports ``0.0`` for an unmeasured value.
    """
    from benchmarks.utils.memory import (
        PSUTIL_AVAILABLE,
        MemoryTracker,
        force_gc,
        get_current_memory_mb,
    )

    if not PSUTIL_AVAILABLE:
        fn()
        return MemoryMeasurement(status=MEMORY_UNAVAILABLE)

    force_gc()
    baseline = get_current_memory_mb()
    tracker = MemoryTracker(sample_interval_ms)
    tracker.start()
    try:
        fn()
    finally:
        stats = tracker.stop()

    peak = max(stats.peak_memory_mb, baseline)
    return MemoryMeasurement(
        status=MEMORY_OK,
        peak_mb=peak,
        baseline_mb=baseline,
        delta_mb=peak - baseline,
    )


def result_from_dict(payload: dict) -> BenchmarkResult:
    """Rebuild a record from its own serialised form.

    The authority owns both directions. Deserialising does not compute a
    statistic, so it is safe — but routing it through here keeps
    `BenchmarkResult(...)` un-callable elsewhere, which is what stops a module
    hand-building a dict and "deserialising" it into a record that never went
    through :func:`to_result`.

    `image_size` round-trips through JSON as a list; it is restored to the
    tuple the result key hashes on.
    """
    from benchmarks.frameworks.base import BenchmarkResult

    fields = dict(payload)
    if "image_size" in fields and fields["image_size"] is not None:
        fields["image_size"] = tuple(fields["image_size"])
    return BenchmarkResult(**fields)


def to_result(
    stats: TimingStats,
    *,
    framework: str,
    engine: str,
    operation: str,
    image_count: int,
    image_size: tuple[int, int],
    thread_pool_size: int,
    memory: MemoryMeasurement | None = None,
    gpu_mode: str | None = None,
) -> BenchmarkResult:
    """Build the one result record, from the one statistic.

    Throughput and latency are derived here, from ``median_s``, so the formula
    exists once. Four scenarios previously each wrote their own
    ``image_count / avg_time`` and ``(avg_time / image_count) * 1000``.

    ``engine`` and ``thread_pool_size`` are explicit parameters rather than
    inferred from ``framework``: they used to live only inside the adapter's
    *name string*, which is why nothing downstream could normalise per core or
    refuse to compare a one-thread run against a four-thread one.
    """
    from benchmarks.frameworks.base import BenchmarkResult

    if image_count < 1:
        raise ValueError(f"image_count must be >= 1, got {image_count}")

    median = stats.median_s
    if median <= 0:
        raise ValueError(
            f"{stats.label}: median wall time was {median}s — the callable did "
            f"no measurable work, so throughput would be infinite rather than "
            f"fast. Check the operation actually ran."
        )

    mem = memory or MemoryMeasurement(status=MEMORY_NOT_REQUESTED)
    return BenchmarkResult(
        framework=framework,
        engine=engine,
        operation=operation,
        image_count=image_count,
        image_size=image_size,
        total_time_seconds=median,
        throughput_images_per_second=image_count / median,
        latency_ms_per_image=(median / image_count) * 1000,
        thread_pool_size=thread_pool_size,
        timing=stats.as_dict(),
        peak_memory_mb=mem.peak_mb,
        peak_memory_delta_mb=mem.delta_mb,
        memory_status=mem.status,
        gpu_mode=gpu_mode,
    )
