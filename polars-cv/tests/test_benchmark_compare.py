"""`benchmarks/regression/compare.py`'s classification and its refusals.

This is the module that decides whether a change shipped a regression, and
until now nothing exercised it. The cases below are the ones where getting it
wrong is silent: an unmeasured column that reads as NEUTRAL, a zero baseline
that reads as an improvement, a dropped key that reads as nothing at all.

`compare.py` is stdlib-only by design — it has to run on a machine where
polars-cv is not built — so these tests import it directly and need no
extension.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from benchmarks.regression.compare import (
    LoadedRun,
    Status,
    UnmeasuredMemory,
    _pct,
    check_comparable,
    classify,
    compare,
    load_results,
    load_run,
)

pytestmark = pytest.mark.structural

THRESHOLDS: dict[str, Any] = {
    "throughput_pct": 7.0,
    "latency_pct": 7.0,
    "memory_pct": 20.0,
    "gate_memory": False,
}


def _record(
    *,
    throughput: float,
    latency: float | None = None,
    memory: float | None = 100.0,
    memory_status: str = "ok",
    operation: str = "resize",
) -> dict[str, Any]:
    return {
        "framework": "polars-cv-eager",
        "engine": "eager",
        "operation": operation,
        "image_count": 300,
        "image_size": [256, 256],
        "total_time_seconds": 300 / throughput,
        "throughput_images_per_second": throughput,
        "latency_ms_per_image": latency if latency is not None else 1000 / throughput,
        "thread_pool_size": 1,
        "timing": {"median_s": 300 / throughput, "relative_mad": 0.01},
        "peak_memory_mb": memory,
        "peak_memory_delta_mb": memory,
        "memory_status": memory_status,
        "gpu_mode": None,
    }


class TestPct:
    def test_none_propagates_rather_than_collapsing_to_zero(self) -> None:
        """The defect this replaced: unmeasured arrived as 0.0, and 0 -> 0 is 0%.

        A whole column that was never measured therefore classified NEUTRAL —
        a hole in the results reading as a clean bill of health.
        """
        assert _pct(None, None) is None
        assert _pct(1.0, None) is None
        assert _pct(None, 1.0) is None

    def test_zero_to_zero_is_no_change(self) -> None:
        assert _pct(0.0, 0.0) == 0.0

    def test_zero_baseline_with_a_real_candidate_is_infinite(self) -> None:
        assert _pct(0.0, 5.0) == float("inf")

    def test_ordinary_change(self) -> None:
        assert _pct(100.0, 90.0) == pytest.approx(-10.0)
        assert _pct(100.0, 110.0) == pytest.approx(10.0)


class TestClassify:
    def test_a_drop_past_the_band_regresses(self) -> None:
        status, _, tp, _, _ = classify(
            _record(throughput=100.0), _record(throughput=90.0), **THRESHOLDS
        )
        assert status is Status.REGRESSED
        assert tp == pytest.approx(-10.0)

    def test_exactly_the_threshold_regresses(self) -> None:
        """The band is inclusive on the regression side (`tp <= -threshold`).

        Pinned because a boundary that silently flips is the difference between
        a gate that fires and one that does not.
        """
        status, *_ = classify(
            _record(throughput=100.0), _record(throughput=93.0), **THRESHOLDS
        )
        assert status is Status.REGRESSED

    def test_exactly_the_threshold_improves_on_the_other_side(self) -> None:
        status, *_ = classify(
            _record(throughput=100.0), _record(throughput=107.0), **THRESHOLDS
        )
        assert status is Status.IMPROVED

    def test_inside_the_band_is_neutral(self) -> None:
        status, *_ = classify(
            _record(throughput=100.0), _record(throughput=96.0), **THRESHOLDS
        )
        assert status is Status.NEUTRAL

    def test_memory_is_advisory_unless_gated(self) -> None:
        base = _record(throughput=100.0, memory=100.0)
        cand = _record(throughput=100.0, memory=200.0)

        assert classify(base, cand, **THRESHOLDS)[0] is Status.NEUTRAL
        gated = {**THRESHOLDS, "gate_memory": True}
        assert classify(base, cand, **gated)[0] is Status.REGRESSED

    def test_gating_on_an_unmeasured_column_refuses_rather_than_passes(self) -> None:
        """`--gate-memory` over `None` must not report "no memory regressions".

        This is the whole point of carrying `None` instead of `0.0`: the
        absence is now loud where it used to classify as NEUTRAL.
        """
        base = _record(throughput=100.0, memory=None, memory_status="not requested")
        cand = _record(throughput=100.0, memory=None, memory_status="not requested")

        assert classify(base, cand, **THRESHOLDS)[0] is Status.NEUTRAL

        gated = {**THRESHOLDS, "gate_memory": True}
        with pytest.raises(UnmeasuredMemory, match="no memory measurement"):
            classify(base, cand, **gated)

    def test_unmeasured_memory_reports_none_not_zero_percent(self) -> None:
        base = _record(throughput=100.0, memory=None)
        cand = _record(throughput=100.0, memory=None)
        _, _, _, _, mem = classify(base, cand, **THRESHOLDS)
        assert mem is None


class TestCompare:
    def test_a_key_missing_from_the_candidate_fails_rather_than_vanishing(
        self,
    ) -> None:
        """A dropped result could hide the regression that dropped it."""
        base = {("f", "resize", (256, 256), 300, None): _record(throughput=100.0)}
        deltas = compare(base, {}, **THRESHOLDS)
        assert [d.status for d in deltas] == [Status.MISSING]

    def test_a_new_key_is_reported_but_does_not_fail(self) -> None:
        cand = {("f", "blur", (256, 256), 300, None): _record(throughput=100.0)}
        deltas = compare({}, cand, **THRESHOLDS)
        assert [d.status for d in deltas] == [Status.NEW]

    def test_identical_runs_are_all_neutral(self) -> None:
        key = ("f", "resize", (256, 256), 300, None)
        rec = _record(throughput=100.0)
        deltas = compare({key: rec}, {key: rec}, **THRESHOLDS)
        assert [d.status for d in deltas] == [Status.NEUTRAL]


class TestLoadResults:
    """v1 and v2 both load; anything else is refused.

    The previous version of this test asserted `{"results": []}` was rejected.
    That was correct for v1, where a results file *was* a bare array — and it
    is now the v2 envelope, so the assertion was updated rather than kept.
    """

    def test_a_bare_array_loads_as_v1(self, tmp_path: Any) -> None:
        path = tmp_path / "v1.json"
        path.write_text(json.dumps([_record(throughput=100.0)]))
        run = load_run(path)
        assert run.schema_version == 1
        assert run.meta is None
        assert len(run.results) == 1

    def test_an_envelope_loads_as_v2_with_its_meta(self, tmp_path: Any) -> None:
        path = tmp_path / "v2.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "meta": {"build_profile": "release"},
                    "results": [_record(throughput=100.0)],
                }
            )
        )
        run = load_run(path)
        assert run.schema_version == 2
        assert run.meta == {"build_profile": "release"}
        assert len(run.results) == 1

    def test_a_payload_that_is_neither_is_refused(self, tmp_path: Any) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"rows": []}')
        with pytest.raises(ValueError, match="expected a JSON array"):
            load_results(path)


class TestComparability:
    """The check that was missing entirely, and what it refuses."""

    def _run(self, **meta: Any) -> LoadedRun:
        base = {
            "build_profile": "release",
            "polars_cv_optimizations": None,
            "polars_version": "1.42.0",
            "cells": {"streaming@auto": {"thread_pool_size": 4}},
            "config": {
                "image_counts": [300],
                "image_sizes": [[256, 256]],
                "scenarios": ["pipelines"],
            },
        }
        base.update(meta)
        return LoadedRun({}, base, 2, "run.json")

    def test_identical_provenance_is_comparable(self) -> None:
        assert check_comparable(self._run(), self._run()) == []

    def test_a_debug_build_is_not_comparable_to_a_release_one(self) -> None:
        problems = check_comparable(self._run(build_profile="debug"), self._run())
        assert any("build_profile" in p for p in problems)

    def test_different_thread_pools_are_not_comparable(self) -> None:
        """The case that silently printed PASS before.

        A four-thread streaming run is several times faster than a one-thread
        one, so comparing them reports a spectacular improvement that is
        entirely an artefact of the machine.
        """
        two_core = self._run(cells={"streaming@auto": {"thread_pool_size": 2}})
        problems = check_comparable(two_core, self._run())
        assert any("cells" in p for p in problems)

    def test_different_optimizations_are_not_comparable(self) -> None:
        problems = check_comparable(
            self._run(polars_cv_optimizations="scalar_fusion=0"), self._run()
        )
        assert any("polars_cv_optimizations" in p for p in problems)

    def test_a_different_matrix_is_not_comparable(self) -> None:
        other = self._run(
            config={
                "image_counts": [50],
                "image_sizes": [[256, 256]],
                "scenarios": ["pipelines"],
            }
        )
        problems = check_comparable(other, self._run())
        assert any("image_counts" in p for p in problems)

    def test_v1_against_v2_is_refused_before_anything_else(self) -> None:
        v1 = LoadedRun({}, None, 1, "old.json")
        problems = check_comparable(v1, self._run())
        assert len(problems) == 1
        assert "schema_version" in problems[0]
