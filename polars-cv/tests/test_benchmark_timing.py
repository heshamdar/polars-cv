"""The timing authority's own behaviour.

`test_timing_authority.py` proves nothing else in `benchmarks/` reads a clock.
This proves the one that does computes the right thing — otherwise the ratchet
has merely centralised an error.

No compiled extension needed: every test here drives `measure` with a plain
callable, so these run in the structural lane.
"""

from __future__ import annotations

import pytest
from benchmarks.utils.timing import (
    MEMORY_NOT_REQUESTED,
    MEMORY_OK,
    MemoryMeasurement,
    TimingStats,
    measure,
    measure_phase,
    stopwatch,
    to_result,
)

pytestmark = pytest.mark.structural


def _stats(samples: tuple[float, ...]) -> TimingStats:
    return TimingStats(
        label="t", iterations=len(samples), warmup_iterations=1, samples_s=samples
    )


class TestStatistics:
    """The statistic is the whole point of the authority."""

    def test_median_ignores_a_single_outlier_that_moves_the_mean(self) -> None:
        """This is why the headline changed from mean to median.

        Nine clean samples and one co-tenant hiccup. The mean moves by ~50%,
        which on a 7% gate is a false regression; the median does not move at
        all.
        """
        clean = (0.10,) * 9
        with_hiccup = clean + (5.0,)

        assert _stats(clean).median_s == pytest.approx(0.10)
        assert _stats(with_hiccup).median_s == pytest.approx(0.10)

        mean_shift = _stats(with_hiccup).mean_s / _stats(clean).mean_s
        assert mean_shift > 1.4, (
            f"fixture no longer demonstrates the problem: the outlier moves the "
            f"mean by only {mean_shift:.2f}x"
        )

    def test_mad_is_zero_for_identical_samples_and_positive_otherwise(self) -> None:
        assert _stats((0.1, 0.1, 0.1)).mad_s == 0.0
        assert _stats((0.1, 0.2, 0.3)).mad_s > 0.0

    def test_relative_mad_is_a_fraction_of_the_median(self) -> None:
        # median 0.10, deviations 0.01/0/0.01 -> MAD 0.01 -> 10%.
        stats = _stats((0.09, 0.10, 0.11))
        assert stats.relative_mad == pytest.approx(0.1, rel=1e-6)

    def test_relative_mad_is_zero_rather_than_dividing_by_zero(self) -> None:
        """A degenerate all-zero sample set must not raise here.

        `to_result` is where a zero median is rejected, with a message about
        the callable doing no work. This property just must not blow up first.
        """
        assert _stats((0.0, 0.0)).relative_mad == 0.0

    def test_iqr_needs_four_samples(self) -> None:
        assert _stats((0.1, 0.2, 0.3)).iqr_s == 0.0
        assert _stats((0.1, 0.2, 0.3, 0.4)).iqr_s > 0.0

    def test_samples_survive_into_the_dict(self) -> None:
        """The comparator's noise band is computed from these."""
        payload = _stats((0.1, 0.2)).as_dict()
        assert payload["samples_s"] == [0.1, 0.2]
        assert set(payload) >= {"min_s", "median_s", "mad_s", "relative_mad"}


class TestMeasure:
    def test_runs_warmup_then_iterations(self) -> None:
        calls = {"warm": 0, "timed": 0}

        def warm() -> None:
            calls["warm"] += 1

        def timed() -> None:
            calls["timed"] += 1

        stats = measure(timed, warmup_fn=warm, warmup=3, iterations=5, label="x")

        assert calls == {"warm": 3, "timed": 5}
        assert stats.iterations == 5
        assert stats.warmup_iterations == 3
        assert len(stats.samples_s) == 5

    def test_zero_warmup_is_refused(self) -> None:
        """An unwarmed first iteration measures graph compilation, not the op."""
        with pytest.raises(ValueError, match="warmup must be >= 1"):
            measure(
                lambda: None, warmup_fn=lambda: None, warmup=0, iterations=1, label="x"
            )

    def test_zero_iterations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            measure(
                lambda: None, warmup_fn=lambda: None, warmup=1, iterations=0, label="x"
            )

    def test_measure_phase_returns_the_value_and_a_single_sample(self) -> None:
        value, elapsed = measure_phase(lambda: "loaded", label="dataset")
        assert value == "loaded"
        assert elapsed >= 0.0

    def test_stopwatch_reads_are_monotonic_and_repeatable(self) -> None:
        sw = stopwatch("phase")
        first = sw.elapsed()
        second = sw.elapsed()
        assert 0.0 <= first <= second


class TestToResult:
    def _call(self, **overrides: object) -> object:
        kwargs: dict[str, object] = {
            "framework": "polars-cv-eager",
            "engine": "eager",
            "operation": "resize",
            "image_count": 100,
            "image_size": (256, 256),
            "thread_pool_size": 4,
        }
        kwargs.update(overrides)
        return to_result(_stats((0.4, 0.5, 0.6)), **kwargs)  # type: ignore[arg-type]

    def test_throughput_and_latency_derive_from_the_median(self) -> None:
        """One formula, in one place. Four scenarios each had their own."""
        result = self._call()
        assert result.total_time_seconds == pytest.approx(0.5)
        assert result.throughput_images_per_second == pytest.approx(200.0)
        assert result.latency_ms_per_image == pytest.approx(5.0)

    def test_engine_and_pool_size_are_recorded_not_inferred(self) -> None:
        result = self._call(engine="streaming", thread_pool_size=8)
        assert result.engine == "streaming"
        assert result.thread_pool_size == 8

    def test_unmeasured_memory_is_none_with_a_status_never_zero(self) -> None:
        """A 0.0 here is what made an unmeasured column compare as NEUTRAL."""
        result = self._call(memory=None)
        assert result.peak_memory_mb is None
        assert result.peak_memory_delta_mb is None
        assert result.memory_status == MEMORY_NOT_REQUESTED

    def test_measured_memory_is_carried_through(self) -> None:
        mem = MemoryMeasurement(
            status=MEMORY_OK, peak_mb=120.0, baseline_mb=100.0, delta_mb=20.0
        )
        result = self._call(memory=mem)
        assert result.peak_memory_delta_mb == pytest.approx(20.0)
        assert result.memory_status == MEMORY_OK

    def test_a_zero_median_is_refused_rather_than_reported_as_infinite(self) -> None:
        """Dividing by a zero median would publish infinite throughput."""
        with pytest.raises(ValueError, match="no measurable work"):
            to_result(
                _stats((0.0, 0.0)),
                framework="f",
                engine="eager",
                operation="op",
                image_count=1,
                image_size=(1, 1),
                thread_pool_size=1,
            )

    def test_zero_image_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="image_count must be >= 1"):
            self._call(image_count=0)
