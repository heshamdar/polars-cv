"""Committed fixtures for the parity framework's own logic.

The framework is a set of guards, and a guard that cannot fail is worse than
none (``tests/AGENTS.md``, "Writing a Sanitation Guard"). So each piece with
non-trivial logic is pinned here with inputs it must reject and inputs it
must accept, calling the *same* helper the sweeps call:

* ``tolerance.compare`` and ``tolerance.propagate`` — the arithmetic every
  reference comparison rests on;
* the known-divergence predicates — which decide what the sweeps withhold;
* ``cases._merge_per_row`` — which decides what may ride as an expression;
* ``budget.property_lanes`` — which decides what runs in which lane;
* and, against the running engine, the three checks themselves: each is fed
  a deliberately wrong reference or a perturbed execution and must fail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import plugin_required
from tests.parity.framework import known
from tests.parity.framework.budget import DEEP_FACTOR, examples, lane_settings
from tests.parity.framework.cases import _merge_per_row
from tests.parity.framework.run import PerRow, Step
from tests.parity.framework.tolerance import (
    EXACT,
    Tol,
    close,
    compare,
    lsb,
    propagate,
    sparse,
)

pytestmark = pytest.mark.structural

u8 = np.uint8


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    """``compare`` must reject every kind of disagreement it claims to."""

    @pytest.mark.parametrize(
        ("actual", "expected", "tol", "reason"),
        [
            (np.array([1, 2], u8), np.array([1, 3], u8), EXACT, "exceed"),
            (np.array([1, 2], u8), np.array([1, 2], np.uint16), EXACT, "dtype"),
            (np.array([1, 2], u8), np.array([[1, 2]], u8), EXACT, "shape"),
            (np.array([np.nan, 1.0]), np.array([1.0, np.nan]), EXACT, "NaN"),
            (np.array([np.inf]), np.array([-np.inf]), EXACT, "infinity"),
            (np.array([10, 20], u8), np.array([12, 20], u8), lsb(1), "exceed"),
            # Two outliers where the sparse budget (ceil(0.2 * 5) = 1) is one.
            (
                np.array([0, 0, 0, 9, 9], u8),
                np.zeros(5, u8),
                sparse(0, 0.2, 99),
                "exceed",
            ),
            # One outlier within budget but beyond the sparse bound.
            (
                np.array([0, 0, 0, 0, 50], u8),
                np.zeros(5, u8),
                sparse(0, 0.2, 10),
                "sparse bound",
            ),
            (np.array([1.0 + 1e-3]), np.array([1.0]), close(0, 1e-6), "exceed"),
        ],
    )
    def test_rejects(self, actual, expected, tol: Tol, reason: str) -> None:
        mismatch = compare(actual, expected, tol)
        assert mismatch is not None, f"compare accepted {actual} vs {expected} at {tol}"
        assert reason in str(mismatch), f"rejected for the wrong reason: {mismatch}"

    @pytest.mark.parametrize(
        ("actual", "expected", "tol"),
        [
            (np.array([1, 2], u8), np.array([1, 2], u8), EXACT),
            (np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), EXACT),
            (np.array([np.inf, -np.inf]), np.array([np.inf, -np.inf]), EXACT),
            (np.array([10, 21], u8), np.array([11, 20], u8), lsb(1)),
            (np.array([0, 0, 0, 0, 9], u8), np.zeros(5, u8), sparse(0, 0.2, 99)),
            # ceil: one outlier is allowed even when frac * size < 1.
            (np.array([0, 7], u8), np.zeros(2, u8), sparse(0, 0.01, 99)),
            (np.array([1.0 + 1e-7]), np.array([1.0]), close(0, 1e-6)),
            (np.zeros((0, 3), u8), np.zeros((0, 3), u8), EXACT),
        ],
    )
    def test_accepts(self, actual, expected, tol: Tol) -> None:
        assert compare(actual, expected, tol) is None

    def test_64_bit_integers_compare_exactly(self) -> None:
        """float64 cannot tell 2**63-1 from 2**63-2; the exact path must."""
        big = np.array([2**63 - 1], np.int64)
        assert compare(big, big - 1, EXACT) is not None


# ---------------------------------------------------------------------------
# propagate
# ---------------------------------------------------------------------------


class TestPropagate:
    """The end-to-end bound's rules, each pinned."""

    def test_exact_input_carries_only_the_step_error(self) -> None:
        assert propagate(EXACT, lsb(1), 5.0, kind="spatial", integer_out=True) == lsb(1)

    def test_unbounded_stays_unbounded(self) -> None:
        assert propagate(None, EXACT, 1.0, kind="movement", integer_out=False) is None

    @pytest.mark.parametrize(
        ("gain", "kind"), [(math.inf, "pointwise"), (1.0, "global")]
    )
    def test_discontinuity_loses_the_bound(self, gain: float, kind: str) -> None:
        assert propagate(lsb(1), EXACT, gain, kind=kind, integer_out=True) is None

    def test_sparse_error_survives_only_pointwise_and_movement(self) -> None:
        incoming = sparse(1, 0.1, 50)
        assert propagate(incoming, EXACT, 1.0, kind="spatial", integer_out=True) is None
        kept = propagate(incoming, EXACT, 1.0, kind="movement", integer_out=True)
        assert kept is not None and kept.frac == pytest.approx(0.1)

    def test_errors_add_with_gain_and_requantization(self) -> None:
        out = propagate(lsb(2), lsb(1), 3.0, kind="spatial", integer_out=True)
        # 3 * 2 (gain * incoming) + 1 (own) + 1 (re-rounding an integer)
        assert out == Tol(atol=8.0)

    def test_movement_does_not_requantize(self) -> None:
        assert propagate(lsb(2), EXACT, 1.0, kind="movement", integer_out=True) == lsb(
            2
        )

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown step kind"):
            propagate(EXACT, EXACT, 1.0, kind="fused", integer_out=False)


# ---------------------------------------------------------------------------
# Known-divergence predicates
# ---------------------------------------------------------------------------

_CROP_BAND = Step("crop", {"top": 1, "left": 0, "height": 2, "width": 3})
_CROP_ORIGIN = Step("crop", {"top": 0, "left": 0, "height": 2, "width": 3})
_SELECT = Step("channel_select", {"index": 0})
_RESHAPE = Step("reshape", {"shape": [6, 1]})
_FLIP = Step("flip", {"axes": [0]})
_BLUR = Step("blur", {"sigma": 1.0})


def _divergence(repro, *, raises=AssertionError, match="the defect"):
    return known.Divergence(
        key="fixture",
        summary="Fixed: never.",
        repro=repro,
        affects_step=lambda step, x: False,
        raises=raises,
        match=match,
    )


def _fails_with(exc: BaseException):
    def repro() -> None:
        raise exc

    return repro


class TestStillReproduces:
    """A registered repro counts only when it fails with its own defect."""

    @pytest.mark.parametrize(
        ("divergence", "reason"),
        [
            (_divergence(lambda: None), "the repro passes"),
            (
                _divergence(_fails_with(TypeError("renamed helper"))),
                "failed with TypeError",
            ),
            (
                _divergence(_fails_with(AssertionError("another assertion"))),
                "not with 'the defect'",
            ),
            (
                _divergence(
                    _fails_with(AssertionError("the defect")), raises=ValueError
                ),
                "failed with AssertionError, not ValueError",
            ),
        ],
    )
    def test_rejects(self, divergence: known.Divergence, reason: str) -> None:
        with pytest.raises(AssertionError, match=reason):
            known.still_reproduces(divergence)

    def test_accepts_its_own_failure(self) -> None:
        known.still_reproduces(
            _divergence(_fails_with(AssertionError("row 1: the defect, again")))
        )


class TestDivergencePredicates:
    """Each predicate flags its defect's cases and nothing else."""

    @pytest.mark.parametrize(
        ("chain", "key"),
        [
            ([_CROP_BAND, _SELECT], "view-offset-lost"),
            ([_CROP_BAND, _FLIP, _SELECT], "view-offset-lost"),
            ([_FLIP, _RESHAPE], "reshape-after-view"),
            ([_CROP_ORIGIN, _RESHAPE], "reshape-after-view"),
        ],
    )
    def test_chain_predicates_flag(self, chain: list[Step], key: str) -> None:
        divergence = known.chain_divergence(chain)
        assert divergence is not None and divergence.key == key

    @pytest.mark.parametrize(
        "chain",
        [
            [_CROP_ORIGIN, _SELECT],  # no offset
            [_CROP_BAND, _BLUR, _SELECT],  # blur materializes the crop
            [_BLUR, _RESHAPE],  # reshape of a fresh buffer
            [_RESHAPE],
            [_SELECT],
        ],
    )
    def test_chain_predicates_pass(self, chain: list[Step]) -> None:
        assert known.chain_divergence(chain) is None

    def test_through_f32_flags_only_unrepresentable_values(self) -> None:
        flagged = np.full((1, 1, 3), 2**24 + 1, np.uint32)
        fine = np.full((1, 1, 3), 2**24, np.uint32)
        f64_exact = np.full((1, 1, 3), 0.5)
        f64_inexact = np.full((1, 1, 3), 0.1)
        step = Step("to_bgr")
        assert known.step_divergence(step, flagged).key == "through-f32"
        assert known.step_divergence(step, f64_inexact).key == "through-f32"
        assert known.step_divergence(step, fine) is None
        assert known.step_divergence(step, f64_exact) is None
        assert known.step_divergence(Step("flip", {"axes": [0]}), flagged) is None

    def test_an_avoid_entry_wins_over_a_value_only_one(self) -> None:
        """A large i32 image resized to zero width is both ``through-f32``
        (value-only) and ``derived-extent-zero`` (avoid); the lookup must
        report the one that stops the case being built."""
        image = np.full((14, 1, 4), 2**30, np.int32)
        step = Step("resize_to_height", {"height": 1, "filter": "nearest"})
        assert known.step_divergence(step, image).key == "derived-extent-zero"
        assert known.append_divergence([], step, [image]).key == "derived-extent-zero"

    @pytest.mark.parametrize(
        ("step", "shape", "flagged"),
        [
            (Step("resize_max", {"max_size": 1, "filter": "nearest"}), (3, 1, 1), True),
            (
                Step("resize_max", {"max_size": 2, "filter": "nearest"}),
                (3, 1, 1),
                False,
            ),
            (
                Step(
                    "resize_scale",
                    {"scale_x": 0.25, "scale_y": 0.25, "filter": "nearest"},
                ),
                (1, 1, 1),
                True,
            ),
            (
                Step(
                    "resize_scale",
                    {"scale_x": 0.5, "scale_y": 0.5, "filter": "nearest"},
                ),
                (1, 1, 1),
                False,
            ),
            (
                Step(
                    "letterbox",
                    {"height": 1, "width": 1, "value": 0.0, "filter": "nearest"},
                ),
                (1, 3, 1),
                True,
            ),
            (
                Step(
                    "letterbox",
                    {"height": 1, "width": 1, "value": 0.0, "filter": "nearest"},
                ),
                (1, 2, 1),
                False,
            ),
        ],
    )
    def test_derived_extent_zero(self, step: Step, shape: tuple, flagged: bool) -> None:
        divergence = known.step_divergence(step, np.zeros(shape, u8))
        assert (
            divergence is not None and divergence.key == "derived-extent-zero"
        ) is flagged


# ---------------------------------------------------------------------------
# Per-row merging
# ---------------------------------------------------------------------------


class TestMergePerRow:
    """What may vary per row is exactly what the signature lets be an expression."""

    def test_equal_rows_stay_literal(self) -> None:
        rows = [{"sigma": 1.0}, {"sigma": 1.0}]
        assert _merge_per_row("blur", rows) == {"sigma": 1.0}

    def test_an_eligible_difference_becomes_per_row(self) -> None:
        rows = [{"sigma": 1.0}, {"sigma": 2.0}]
        assert _merge_per_row("blur", rows) == {"sigma": PerRow((1.0, 2.0))}

    def test_an_ineligible_difference_cannot_merge(self) -> None:
        rows = [{"axes": [0]}, {"axes": [1]}]
        assert _merge_per_row("flip", rows) is None

    def test_list_lengths_cannot_vary(self) -> None:
        rows = [{"shape": [1, 1]}, {"shape": [1, 1, 1]}]
        assert _merge_per_row("reshape", rows) is None


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------


class TestLanes:
    """The fast lane is small and repeatable; the deep lane searches."""

    def test_fast_is_derandomized_and_deep_is_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POLARS_CV_PARITY_RANDOM", raising=False)
        monkeypatch.delenv("POLARS_CV_PARITY_SCALE", raising=False)
        assert lane_settings(1, deep=False).derandomize is True
        assert lane_settings(1, deep=True).derandomize is False

    def test_deep_runs_more(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("POLARS_CV_PARITY_SCALE", raising=False)
        assert examples(2, deep=True) == DEEP_FACTOR * examples(2, deep=False)

    def test_scale_is_validated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POLARS_CV_PARITY_SCALE", "-1")
        with pytest.raises(ValueError, match="positive"):
            examples(1, deep=False)

    def test_every_property_has_a_deep_twin_in_the_slow_lane(self) -> None:
        """``property_lanes`` wrote ``<name>_deep`` next to every property."""
        from tests.parity.oracle import test_parity_single_ops as module

        deep = module.test_op_matches_its_reference_deep
        marks = {m.name for m in getattr(deep, "pytestmark", [])}
        assert "slow" in marks and "parametrize" in marks


# ---------------------------------------------------------------------------
# The checks, watched failing against the running engine
# ---------------------------------------------------------------------------


@plugin_required
class TestChecksCatchDefects:
    """Each check must fail when handed a wrong reference or a perturbation."""

    _IMAGE = np.arange(4 * 5 * 3, dtype=u8).reshape(4, 5, 3)

    def test_check_step_rejects_a_wrong_reference(self) -> None:
        from dataclasses import replace

        from tests.parity.framework.checks import ParityFailure, check_step
        from tests.parity.framework.oracle import OPS

        right = OPS["flip_h"]
        wrong = replace(right, ref=OPS["flip_v"].ref)
        step = Step("flip_h")
        check_step([self._IMAGE], step, spec=right)  # the real entry passes
        with pytest.raises(ParityFailure, match="disagrees with its reference"):
            check_step([self._IMAGE], step, spec=wrong)

    def test_chain_checker_rejects_a_wrong_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import replace

        from tests.parity.framework.checks import ChainChecker, ParityFailure
        from tests.parity.framework.oracle import OPS

        monkeypatch.setitem(
            OPS, "invert", replace(OPS["invert"], ref=lambda x, p: x.copy())
        )
        checker = ChainChecker(self._IMAGE)
        checker.push(Step("flip_v"))
        with pytest.raises(ParityFailure, match="step 2"):
            checker.push(Step("invert"))

    def test_check_invariant_rejects_a_perturbed_axis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import replace

        from tests.parity.framework import io
        from tests.parity.framework.checks import ParityFailure, check_invariant
        from tests.parity.framework.run import Axes

        honest = io.SINKS["list"]

        def off_by_one(series, info):
            return [None if r is None else r + 1 for r in honest.decode(series, info)]

        monkeypatch.setitem(io.SINKS, "list", replace(honest, decode=off_by_one))
        with pytest.raises(ParityFailure, match="differs from the baseline"):
            check_invariant(
                [self._IMAGE], [Step("flip_h")], Axes(), [Axes(sink="list")]
            )


class TestOptimizationAxis:
    """The optimize axis is the pass registry, both ways round."""

    def test_every_pass_is_swept_alone_and_removed(self) -> None:
        from polars_cv._optimize import PASS_NAMES
        from tests.parity.framework.run import OPTIMIZATION

        for name in PASS_NAMES:
            assert f"only:{name}" in OPTIMIZATION
            assert f"without:{name}" in OPTIMIZATION

    def test_settings_map_to_the_flags_they_name(self) -> None:
        from polars_cv._optimize import PASS_NAMES
        from tests.parity.framework.run import opt_flags_for

        name = PASS_NAMES[0]
        only = opt_flags_for(f"only:{name}")
        without = opt_flags_for(f"without:{name}")
        assert getattr(only, name) and not any(getattr(only, n) for n in PASS_NAMES[1:])
        assert not getattr(without, name) and all(
            getattr(without, n) for n in PASS_NAMES[1:]
        )

    def test_an_unknown_setting_is_refused(self) -> None:
        from tests.parity.framework.run import Axes, opt_flags_for

        with pytest.raises(ValueError, match="unknown"):
            opt_flags_for("only:no_such_pass")
        with pytest.raises(ValueError, match="unknown axis value"):
            Axes(optimize="some")
