"""Parity tests: the AUC expressions reproduce the eager `_auc.py` integrals.

These are pure-expression tests (no compiled plugin) — they build curves as
DataFrames and assert :mod:`polars_cv.metrics._auc_expr` matches the scalar
functions in :mod:`polars_cv.metrics._auc` that it replaces. While both exist
this is a direct cross-check; once the eager path is removed the same numbers
are pinned by these curves.
"""

from __future__ import annotations

import random

import polars as pl
import pytest

from polars_cv.metrics._auc import (
    partial_auc,
    trapz_auc,
)
from polars_cv.metrics._auc_expr import (
    collapse_curve,
    collapse_scores,
    mann_whitney_auc_expr,
    partial_auc_expr,
    trapz_auc_expr,
)
from tests._metric_refs import ref_mann_whitney, ref_weighted_mann_whitney

_TOL = 1e-9


def _random_curve(rng: random.Random, n: int) -> tuple[list[float], list[float]]:
    """Strictly-increasing x in [0, 10], y in [0, 1]."""
    xs = sorted({round(rng.uniform(0.0, 10.0), 4) for _ in range(n)})
    ys = [round(rng.uniform(0.0, 1.0), 4) for _ in xs]
    return xs, ys


def _scalar_trapz(xs: list[float], ys: list[float], correction=None) -> float:
    return trapz_auc(pl.Series("x", xs), pl.Series("y", ys), correction)


def _expr_trapz(xs: list[float], ys: list[float], correction=None) -> float:
    df = pl.DataFrame({"x": xs, "y": ys})
    return df.select(auc=trapz_auc_expr(x="x", y="y", correction=correction)).item()


class TestTrapzParity:
    def test_matches_eager_trapz_over_random_curves(self) -> None:
        rng = random.Random(0)
        for _ in range(200):
            xs, ys = _random_curve(rng, rng.randint(2, 12))
            assert _expr_trapz(xs, ys) == pytest.approx(_scalar_trapz(xs, ys), abs=_TOL)

    def test_normalize_correction(self) -> None:
        rng = random.Random(1)
        for _ in range(100):
            xs, ys = _random_curve(rng, rng.randint(2, 12))
            assert _expr_trapz(xs, ys, "normalize") == pytest.approx(
                _scalar_trapz(xs, ys, "normalize"), abs=_TOL
            )

    def test_single_point_is_zero(self) -> None:
        assert _expr_trapz([1.0], [0.5]) == pytest.approx(0.0, abs=_TOL)


class TestPartialParity:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_matches_eager_partial_over_random_curves_and_ranges(self) -> None:
        # ``lo`` is bounded by the curve's max x. Beyond that the two definitions
        # deliberately diverge — see ``test_out_of_range_right_extends_last_y``.
        # Everywhere a real FROC range lands (lo >= xmin = 0 at the origin, hi
        # possibly past xmax) they agree exactly, including left- and right-fill.
        rng = random.Random(2)
        for _ in range(300):
            xs, ys = _random_curve(rng, rng.randint(2, 12))
            lo = round(rng.uniform(-1.0, xs[-1]), 3)
            # Keep the range overlapping the curve's x-span (lo <= xmax already;
            # force hi >= xmin) — the only regime where the two definitions
            # agree, and the only one a FROC range reaches.
            hi = round(max(lo + rng.uniform(0.5, 6.0), xs[0] + 0.1), 3)
            df = pl.DataFrame({"x": xs, "y": ys})
            got = df.select(auc=partial_auc_expr(x="x", y="y", lo=lo, hi=hi)).item()
            want = partial_auc(pl.Series("x", xs), pl.Series("y", ys), lo, hi)
            assert got == pytest.approx(want, abs=1e-7), (xs, ys, lo, hi)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("correction", ["normalize", "mcclish"])
    def test_corrections(self, correction: str) -> None:
        rng = random.Random(3)
        for _ in range(100):
            xs, ys = _random_curve(rng, rng.randint(2, 12))
            lo = round(rng.uniform(0.0, xs[-1]), 3)
            hi = round(max(lo + rng.uniform(0.5, 5.0), xs[0] + 0.1), 3)
            df = pl.DataFrame({"x": xs, "y": ys})
            got = df.select(
                auc=partial_auc_expr(x="x", y="y", lo=lo, hi=hi, correction=correction)
            ).item()
            want = partial_auc(
                pl.Series("x", xs), pl.Series("y", ys), lo, hi, correction
            )
            assert got == pytest.approx(want, abs=1e-7)

    def test_degenerate_range_is_zero(self) -> None:
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.5, 1.0]})
        assert (
            df.select(auc=partial_auc_expr(x="x", y="y", lo=2.0, hi=1.0)).item() == 0.0
        )

    def test_out_of_range_right_extends_last_y(self) -> None:
        # A range entirely to the right of the curve integrates the flat
        # extension of the rightmost operating point (sensitivity stays at its
        # max beyond the observed FP range). This is the consistent
        # generalization of the eager path's right-fill; the eager
        # ``partial_auc`` instead fell back to y[0] at ``lo`` there, an
        # asymmetry not worth reproducing.
        df = pl.DataFrame({"x": [0.0, 2.0], "y": [0.3, 0.9]})
        got = df.select(auc=partial_auc_expr(x="x", y="y", lo=3.0, hi=5.0)).item()
        assert got == pytest.approx((5.0 - 3.0) * 0.9, abs=1e-9)

    def test_out_of_range_left_extends_first_y(self) -> None:
        # A range entirely to the left of the curve integrates the flat
        # extension of the leftmost operating point (the symmetric counterpart
        # of the right case).
        df = pl.DataFrame({"x": [2.0, 4.0], "y": [0.3, 0.9]})
        got = df.select(auc=partial_auc_expr(x="x", y="y", lo=0.0, hi=1.0)).item()
        assert got == pytest.approx((1.0 - 0.0) * 0.3, abs=1e-9)


def _mw_auc(scores, labels, weights=None) -> float:
    """AUC through the two-stage weighted path (collapse_scores → reduction)."""
    data = {"score": scores, "label": labels}
    if weights is not None:
        data["w"] = weights
    lf = pl.DataFrame(data).lazy()
    bucketed = collapse_scores(
        lf, score="score", label="label", weight="w" if weights is not None else None
    )
    return bucketed.select(auc=mann_whitney_auc_expr()).collect().item()


class TestMannWhitneyParity:
    def test_unit_weight_matches_unweighted_reference(self) -> None:
        """weight=1 (and weight=None) equals the brute-force unweighted MW."""
        rng = random.Random(4)
        for _ in range(200):
            n = rng.randint(2, 20)
            scores = [round(rng.uniform(0.0, 1.0), 2) for _ in range(n)]  # ties likely
            labels = [rng.random() < 0.5 for _ in range(n)]
            want = ref_mann_whitney(scores, labels)
            assert _mw_auc(scores, labels) == pytest.approx(want, abs=1e-9)
            assert _mw_auc(scores, labels, [1.0] * n) == pytest.approx(want, abs=1e-9)

    def test_matches_weighted_reference(self) -> None:
        """Random non-unit weights match the brute-force weighted MW oracle."""
        rng = random.Random(11)
        for _ in range(300):
            n = rng.randint(2, 12)
            scores = [round(rng.uniform(0.0, 1.0), 1) for _ in range(n)]  # heavy ties
            labels = [rng.random() < 0.5 for _ in range(n)]
            weights = [rng.choice([0.5, 1.0, 2.0, 3.0]) for _ in range(n)]
            got = _mw_auc(scores, labels, weights)
            want = ref_weighted_mann_whitney(scores, labels, weights)
            assert got == pytest.approx(want, abs=1e-9), (scores, labels, weights)

    def test_weighted_per_group_equals_individual(self) -> None:
        """Grouped weighted MW equals each group computed on its own."""
        rng = random.Random(12)
        frames = []
        expected: dict[str, float] = {}
        for gid in ("a", "b", "c"):
            n = rng.randint(3, 10)
            scores = [round(rng.uniform(0.0, 1.0), 1) for _ in range(n)]
            labels = [rng.random() < 0.5 for _ in range(n)]
            weights = [rng.choice([0.5, 1.0, 2.0]) for _ in range(n)]
            frames.append(
                pl.DataFrame({"g": gid, "score": scores, "label": labels, "w": weights})
            )
            expected[gid] = ref_weighted_mann_whitney(scores, labels, weights)
        df = pl.concat(frames)
        got = (
            collapse_scores(
                df.lazy(), score="score", label="label", weight="w", group_keys=["g"]
            )
            .group_by("g")
            .agg(auc=mann_whitney_auc_expr())
            .collect()
        )
        got_map = dict(zip(got["g"].to_list(), got["auc"].to_list()))
        for gid, want in expected.items():
            assert got_map[gid] == pytest.approx(want, abs=_TOL)


class TestGroupAwareness:
    def test_trapz_per_group_equals_individual(self) -> None:
        rng = random.Random(5)
        frames = []
        expected: dict[str, float] = {}
        for gid in ("a", "b", "c"):
            xs, ys = _random_curve(rng, rng.randint(3, 10))
            frames.append(pl.DataFrame({"g": gid, "x": xs, "y": ys}))
            expected[gid] = _scalar_trapz(xs, ys)
        df = pl.concat(frames)
        got = (
            df.lazy()
            .pipe(collapse_curve, x_col="x", y_col="y", group_keys=["g"])
            .group_by("g")
            .agg(auc=trapz_auc_expr(x="x", y="y"))
            .collect()
        )
        got_map = dict(zip(got["g"].to_list(), got["auc"].to_list()))
        for gid, want in expected.items():
            assert got_map[gid] == pytest.approx(want, abs=_TOL)
