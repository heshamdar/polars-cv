"""Tests for FROC/LROC metrics utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars_cv.metrics import ContourMatcher, froc_curve, lroc_curve
from polars_cv.metrics._metrics._lroc import _build_lroc_curve

from tests.conftest import plugin_required

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


def _mask(
    width: int, height: int, x0: int, y0: int, x1: int, y1: int
) -> list[list[float]]:
    """Create a binary rectangle mask."""
    rows: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            row.append(1.0 if (x0 <= x < x1 and y0 <= y < y1) else 0.0)
        rows.append(row)
    return rows


def _heatmap_with_peak(
    width: int,
    height: int,
    cx: int,
    cy: int,
    radius: int,
    peak: float,
) -> list[list[float]]:
    """Create a radial heatmap blob."""
    rows: list[list[float]] = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            dist2 = (x - cx) ** 2 + (y - cy) ** 2
            if dist2 <= radius**2:
                rows[y][x] = max(
                    rows[y][x], peak * (1.0 - dist2 / float(max(radius**2, 1)))
                )
    return rows


def _dataset() -> pl.DataFrame:
    """Build a compact test dataset."""
    return pl.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "pred_heatmap": [
                _heatmap_with_peak(16, 16, 5, 5, 4, 0.95),  # likely TP
                _heatmap_with_peak(16, 16, 12, 12, 3, 0.85),  # likely FP
                _heatmap_with_peak(16, 16, 4, 4, 4, 0.9),  # negative with prediction
                [[0.0 for _ in range(16)] for _ in range(16)],  # empty
            ],
            "gt_mask": [
                _mask(16, 16, 2, 2, 8, 8),
                _mask(16, 16, 2, 2, 8, 8),
                [[0.0 for _ in range(16)] for _ in range(16)],
                [[0.0 for _ in range(16)] for _ in range(16)],
            ],
            "gt_label": [True, True, False, False],
            "sample_weight": [1.0, 1.2, 0.8, 1.0],
        },
        schema={
            "image_id": pl.String,
            "pred_heatmap": pl.List(pl.List(pl.Float64)),
            "gt_mask": pl.List(pl.List(pl.Float64)),
            "gt_label": pl.Boolean,
            "sample_weight": pl.Float64,
        },
    )


@plugin_required
class TestFrocMetrics:
    """Integration tests for FROC metric computation."""

    def test_froc_compute_returns_expected_columns(self) -> None:
        """FROC compute returns dense curve columns and valid ranges."""
        matcher = ContourMatcher(iou_threshold=0.5)
        table = matcher.match(
            _dataset().lazy(),
            pred_col="pred_heatmap",
            gt_col="gt_mask",
            image_id_col="image_id",
            weight_col="sample_weight",
        )
        result = froc_curve(table)
        assert set(result.curve.columns) == {
            "threshold",
            "tp",
            "fp",
            "fn",
            "total_gts",
            "fp_per_image",
            "sensitivity",
        }
        assert result.curve.height >= 1
        assert 0.0 <= result.auc() <= 10.0
        assert 0.0 <= result.sensitivity_at_fp(1.0) <= 1.0

    def test_froc_shape_mismatch_validation(self) -> None:
        """Disabling auto-resize validates shape mismatches."""
        df = _dataset().with_columns(
            pred_heatmap=pl.when(pl.col("image_id") == "a")
            .then(pl.lit([[0.0 for _ in range(8)] for _ in range(8)]))
            .otherwise(pl.col("pred_heatmap"))
        )
        matcher = ContourMatcher(auto_resize=False)
        with pytest.raises(ValueError, match="shapes differ"):
            matcher.match(
                df.lazy(),
                pred_col="pred_heatmap",
                gt_col="gt_mask",
                image_id_col="image_id",
            )

    def test_froc_shape_mismatch_auto_resize(self) -> None:
        """Auto-resize handles shape mismatches without eager preprocessing."""
        df = _dataset().with_columns(
            pred_heatmap=pl.when(pl.col("image_id") == "a")
            .then(pl.lit([[0.0 for _ in range(8)] for _ in range(8)]))
            .otherwise(pl.col("pred_heatmap"))
        )
        matcher = ContourMatcher(auto_resize=True)
        table = matcher.match(
            df.lazy(),
            pred_col="pred_heatmap",
            gt_col="gt_mask",
            image_id_col="image_id",
        )
        result = froc_curve(table)
        assert result.curve.height >= 1

    def test_froc_bootstrap_ci_runs(self) -> None:
        """Bootstrap CI returns bounded interval and expected sample count."""
        matcher = ContourMatcher()
        table = matcher.match(
            _dataset(),
            pred_col="pred_heatmap",
            gt_col="gt_mask",
            image_id_col="image_id",
        )
        result = froc_curve(table)
        ci = result.bootstrap_ci(n_bootstrap=20, seed=42)
        assert len(ci.distribution) == 20
        assert ci.ci_lower <= ci.ci_upper


@plugin_required
class TestLrocMetrics:
    """Integration tests for LROC metric computation."""

    def test_lroc_compute_returns_expected_columns(self) -> None:
        """LROC compute returns FPF/sensitivity points."""
        matcher = ContourMatcher(gt_min_contour_area=1.0)
        table = matcher.match(
            _dataset().lazy(),
            pred_col="pred_heatmap",
            gt_col="gt_mask",
            image_id_col="image_id",
            weight_col="sample_weight",
        )
        result = lroc_curve(table)
        assert set(result.curve.columns) == {"threshold", "fpf", "sensitivity"}
        assert result.curve.height >= 1
        assert 0.0 <= result.auc() <= 1.0
        assert 0.0 <= result.sensitivity_at_fpf(0.25) <= 1.0

    def test_lroc_allows_multiple_targets_per_positive(self) -> None:
        """LROC computes image-level localization when positives have multiple GTs."""
        multi_mask = _mask(16, 16, 2, 2, 6, 6)
        for y in range(10, 14):
            for x in range(10, 14):
                multi_mask[y][x] = 1.0
        df = _dataset().with_columns(
            gt_mask=pl.when(pl.col("image_id") == "a")
            .then(pl.lit(multi_mask))
            .otherwise(pl.col("gt_mask"))
        )
        matcher = ContourMatcher()
        table = matcher.match(
            df.lazy(),
            pred_col="pred_heatmap",
            gt_col="gt_mask",
            image_id_col="image_id",
        )
        result = lroc_curve(table)
        assert set(result.curve.columns) == {"threshold", "fpf", "sensitivity"}
        assert result.curve.height >= 1
        assert 0.0 <= result.auc() <= 1.0


def test_lroc_curve_builder_expected_points() -> None:
    """LROC curve builder computes expected sensitivity and FPF values."""
    per_image = pl.DataFrame(
        {
            "image_id": ["p_tp", "p_fn", "n_fp", "n_none"],
            "gt_label": [True, True, False, False],
            "stratify": ["1", "1", "0", "0"],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "max_score": [0.9, 0.6, 0.8, None],
            "top_is_tp": [True, False, False, False],
        }
    )
    curve = _build_lroc_curve(per_image).sort("threshold")

    by_threshold = {
        float(row["threshold"]): (float(row["fpf"]), float(row["sensitivity"]))
        for row in curve.iter_rows(named=True)
    }
    assert by_threshold[0.6] == (0.5, 0.5)
    assert by_threshold[0.8] == (0.5, 0.5)
    assert by_threshold[0.9] == (0.0, 0.5)
    assert by_threshold[float("inf")] == (0.0, 0.0)


def test_lroc_curve_builder_weighted_points() -> None:
    """Weighted LROC curve uses weighted positive/negative totals."""
    per_image = pl.DataFrame(
        {
            "image_id": ["p_tp", "p_fn", "n_fp", "n_none"],
            "gt_label": [True, True, False, False],
            "stratify": ["1", "1", "0", "0"],
            "weight": [2.0, 1.0, 1.0, 3.0],
            "max_score": [0.9, 0.6, 0.8, None],
            "top_is_tp": [True, False, False, False],
        }
    )
    curve = _build_lroc_curve(per_image).sort("threshold")
    row_06 = curve.filter(pl.col("threshold") == 0.6).to_dicts()[0]
    row_09 = curve.filter(pl.col("threshold") == 0.9).to_dicts()[0]

    assert row_06["sensitivity"] == pytest.approx(2.0 / 3.0)
    assert row_06["fpf"] == pytest.approx(1.0 / 4.0)
    assert row_09["sensitivity"] == pytest.approx(2.0 / 3.0)
    assert row_09["fpf"] == pytest.approx(0.0)
