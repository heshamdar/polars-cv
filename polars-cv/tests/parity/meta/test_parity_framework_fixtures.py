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
from typing import ClassVar

import numpy as np
import pytest

from tests.conftest import plugin_required
from tests.parity.framework import known
from tests.parity.framework.budget import DEEP_FACTOR, examples, lane_settings
from tests.parity.framework.cases import Batch, _merge_per_row
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


class TestRoundHalfAway:
    """The oracle's float -> integer rounding is the engine's (Rust's
    ``f64::round``), exactly: ``floor(|v| + 0.5)`` rounds the largest double
    below 0.5 up, because the addition itself rounds to 1.0."""

    def test_just_below_a_half_rounds_down(self) -> None:
        from tests.parity.framework.oracle import round_half_away

        below = np.nextafter(0.5, 0.0)
        assert round_half_away(np.array([below, -below])).tolist() == [0.0, -0.0]

    def test_halves_round_away_from_zero(self) -> None:
        from tests.parity.framework.oracle import round_half_away

        v = np.array([0.5, 1.5, 2.5, -0.5, -2.5, 0.49, 0.51, 7.0])
        assert round_half_away(v).tolist() == [1.0, 2.0, 3.0, -1.0, -3.0, 0.0, 1.0, 7.0]

    def test_large_odd_integers_stay_put(self) -> None:
        # Above 2**52 every double is an integer, and |v| + 0.5 rounds to the
        # next even one.
        from tests.parity.framework.oracle import round_half_away

        v = np.array([2.0**52 + 1, -(2.0**52 + 1)])
        assert round_half_away(v).tolist() == v.tolist()


class TestCastGain:
    """A narrowing or sign-changing integer cast wraps, so a one-level error
    on its input (127 vs 128 into ``i8``) is a 255-level one on its output:
    it is discontinuous there, and the chain's bound must not survive it.
    A widening cast, or a float one (which saturates), is a plain 1."""

    def test_a_wrapping_cast_is_discontinuous(self) -> None:
        from tests.parity.framework.oracle import spec_for

        cast = spec_for("cast")
        u8 = np.zeros((1, 1, 1), np.uint8)
        assert math.isinf(cast.gain_for(u8, {"dtype": "i8"}))
        assert math.isinf(cast.gain_for(np.zeros((1, 1, 1), np.int16), {"dtype": "u8"}))
        assert math.isinf(cast.gain_for(np.zeros((1, 1, 1), np.int8), {"dtype": "u16"}))

    def test_a_cast_that_cannot_wrap_has_unit_gain(self) -> None:
        from tests.parity.framework.oracle import spec_for

        cast = spec_for("cast")
        assert cast.gain_for(np.zeros((1, 1, 1), np.uint8), {"dtype": "i16"}) == 1.0
        assert cast.gain_for(np.zeros((1, 1, 1), np.float32), {"dtype": "u8"}) == 1.0
        assert cast.gain_for(np.zeros((1, 1, 1), np.uint8), {"dtype": "f32"}) == 1.0


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


class TestComparisonSpace:
    """A bound can hold in another representation than the output's.

    u8 colour with alpha is resampled premultiplied, in 8 bits, by the engine
    and by Pillow alike. Un-premultiplying multiplies each side's rounding by
    ``MAX / alpha``, so in straight colour no dense bound holds where the
    output is translucent. In premultiplied space both sides agree within a
    few units: that is where the bound is stated.
    """

    #: The F10 case: alpha alternating 46/102 down a 10x1 RGBA column,
    #: bilinear to 1x1. Exact colour 84.6; the engine gave 86, Pillow 82.
    COLUMN = np.repeat(np.array([46, 102] * 5, u8)[:, None, None], 4, axis=2)
    PARAMS: ClassVar[dict] = {"height": 1, "width": 1, "filter": "bilinear"}

    def test_a_translucent_resample_holds_in_premultiplied_space(self) -> None:
        from tests.parity.framework.oracle import _resize_tol

        tol = _resize_tol(self.COLUMN, self.PARAMS)
        engine = np.array([[[86, 86, 86, 74]]], u8)
        pillow = np.array([[[82, 82, 82, 74]]], u8)
        assert compare(engine, pillow, tol) is None

    def test_a_wrong_kernel_is_still_caught(self) -> None:
        from tests.parity.framework.oracle import _resize_tol

        tol = _resize_tol(self.COLUMN, self.PARAMS)
        opaque = np.full((1, 1, 4), 255, u8)
        off = opaque.copy()
        off[..., :3] -= 6
        mismatch = compare(off, opaque, tol)
        assert mismatch is not None and "exceed" in str(mismatch)

    def test_a_bound_in_another_space_does_not_propagate(self) -> None:
        spaced = Tol(atol=2, space="premultiplied")
        # Over an exact input the step's own bound stands, space and all...
        assert propagate(EXACT, spaced, 1.0, kind="spatial", integer_out=True) == spaced
        # ...but it does not compose with straight-space errors either way.
        assert propagate(spaced, lsb(1), 1.0, kind="spatial", integer_out=True) is None
        assert propagate(lsb(1), spaced, 1.0, kind="spatial", integer_out=True) is None

    def test_an_unknown_space_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown comparison space"):
            Tol(atol=1, space="linear-light")


# ---------------------------------------------------------------------------
# propagate
# ---------------------------------------------------------------------------


class TestOpaqueResampleBound:
    """Two correct u8 fixed-point resamplers can differ by 2.

    Each rounds within 1 of the exact resample, on opposite sides of it, so
    the bound between them is 2 for every filter. F11, a bilinear
    ``resize_min(14)`` of a 16x16 RGB image: engine 34, Pillow 36, exact 35.0.
    """

    RGB = np.zeros((16, 16, 3), u8)
    PARAMS: ClassVar[dict] = {"min_size": 14, "filter": "bilinear"}

    def test_opposite_roundings_of_one_resample_agree(self) -> None:
        from tests.parity.framework.oracle import _resize_tol

        tol = _resize_tol(self.RGB, self.PARAMS)
        assert compare(np.array([34], u8), np.array([36], u8), tol) is None

    def test_a_wrong_kernel_is_still_caught(self) -> None:
        from tests.parity.framework.oracle import _resize_tol

        tol = _resize_tol(self.RGB, self.PARAMS)
        assert compare(np.array([33], u8), np.array([36], u8), tol) is not None


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

    def test_an_avoid_entry_wins_over_a_value_only_one(self) -> None:
        """When several entries cover one case, the lookup reports the one
        that stops the case being built, wherever it sits in the registry."""
        value_only = _divergence(lambda: None)
        avoid = known.Divergence(**{**value_only.__dict__, "key": "a", "avoid": True})
        assert known._first([value_only, avoid]) is avoid
        assert known._first([avoid, value_only]) is avoid
        assert known._first([value_only]) is value_only
        assert known._first([]) is None


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

    def test_a_scalar_or_list_parameter_varies_only_as_a_scalar(self) -> None:
        """``histogram(bins=)`` is a per-row count or literal edges."""
        counts = [{"bins": 4}, {"bins": 8}]
        assert _merge_per_row("histogram", counts) == {"bins": PerRow((4, 8))}
        edges = [{"bins": [0.0, 1.0]}, {"bins": [0.0, 2.0]}]
        assert _merge_per_row("histogram", edges) is None
        assert _merge_per_row("histogram", [{"bins": 2}, {"bins": [0.0, 1.0]}]) is None

    def test_a_per_row_enum_varies_per_row(self) -> None:
        """``str | pl.Expr`` (a per-row enum) admits an expression too."""
        rows = [{"mode": "edge"}, {"mode": "reflect"}]
        merged = _merge_per_row("pad", rows)
        assert merged == {"mode": PerRow(("edge", "reflect"))}

    def test_a_list_of_expressions_varies_per_row(self) -> None:
        rows = [{"kernel": [1.0, 2.0]}, {"kernel": [1.0, 3.0]}]
        merged = _merge_per_row("convolve2d", rows)
        assert merged == {"kernel": PerRow(([1.0, 2.0], [1.0, 3.0]))}


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


class TestReferenceClaims:
    """A step that claims a reference must be able to compute it.

    ``has_reference`` gates which rows are compared; a claim the reference
    cannot honour turns a skip into a harness error deep in a chain hunt
    (``rotate_and_scale`` at a quarter turn on i8 was claimed as data
    movement, then sent to ``cv2.warpAffine``, which has no i8).
    """

    @pytest.mark.parametrize(
        ("method", "params", "claimed"),
        [
            # A bare quarter turn is data movement: every dtype.
            (
                "rotate",
                {"angle": 90.0, "interpolation": "bilinear", "border_value": 0.0},
                True,
            ),
            # About a centre, scaled, it is a warp: OpenCV's dtypes only.
            (
                "rotate_and_scale",
                {
                    "angle": 0.0,
                    "center": (0.0137, 0.0137),
                    "output_size": (1, 1),
                    "scale": 0.5,
                },
                False,
            ),
            # Which op it is decides, not which arguments are present: a
            # quarter turn about the default centre is still a warp.
            (
                "rotate_and_scale",
                {"angle": 90.0, "output_size": (2, 2), "scale": 1.0},
                False,
            ),
        ],
    )
    def test_a_claimed_reference_runs(
        self, method: str, params: dict, claimed: bool
    ) -> None:
        from tests.parity.framework.oracle import spec_for

        spec = spec_for(method)
        x = np.arange(8, dtype=np.int8).reshape(2, 2, 2)
        assert spec.has_reference(x, params) is claimed
        if claimed:
            assert spec.ref is not None
            spec.ref(x, params)


class TestBatchProxy:
    """``Batch.proxy`` stands in for every row when one literal argument set
    is drawn, so what a strategy reads off it must hold for every row."""

    def test_a_non_finite_row_reaches_the_proxy(self) -> None:
        # The deep lane drew `histogram(bins=1)` with an auto range for a
        # finite proxy while another row held `inf`: the engine refused that
        # row (numpy's rule: no equal-width bins over infinity) and the
        # oracle reported a mismatch. The strategy was right; the proxy hid
        # the row.
        finite = np.array([[[0.5]]])
        infinite = np.array([[[np.inf]]])
        batch = Batch(specs=(None, None), images=(finite, infinite))
        proxy = batch.proxy()
        assert not np.isfinite(proxy).all()
        assert proxy.shape == finite.shape
        assert np.isfinite(finite).all(), "the row itself is not modified"

    def test_a_finite_batch_keeps_its_smallest_crop(self) -> None:
        small = np.zeros((2, 2, 1))
        big = np.ones((4, 3, 1))
        proxy = Batch(specs=(None, None), images=(big, small)).proxy()
        assert proxy.shape == (2, 2, 1) and (proxy == 0).all()
