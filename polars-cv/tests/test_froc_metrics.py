"""Tests for FROC/LROC metrics utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars_cv.metrics import FROCAnalyzer, LROCAnalyzer

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
    """Integration tests for FROC metric analyzer."""

    def test_froc_compute_returns_expected_columns(self) -> None:
        """FROC compute returns dense curve columns and valid ranges."""
        result = FROCAnalyzer(iou_threshold=0.5).compute(
            _dataset().lazy(),
            pred_col="pred_heatmap",
            gt_mask_col="gt_mask",
            gt_label_col="gt_label",
            image_id_col="image_id",
            weight_col="sample_weight",
            stratify_col="gt_label",
        )
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
        analyzer = FROCAnalyzer(auto_resize=False)
        with pytest.raises(ValueError, match="shapes differ"):
            analyzer.compute(
                df.lazy(),
                pred_col="pred_heatmap",
                gt_mask_col="gt_mask",
                gt_label_col="gt_label",
                image_id_col="image_id",
            )

    def test_froc_bootstrap_ci_runs(self) -> None:
        """Bootstrap CI returns bounded interval and expected sample count."""
        result = FROCAnalyzer().compute(
            _dataset(),
            pred_col="pred_heatmap",
            gt_mask_col="gt_mask",
            gt_label_col="gt_label",
            image_id_col="image_id",
            stratify_col="gt_label",
        )
        ci = result.bootstrap_ci(n_bootstrap=20, seed=42)
        assert len(ci.distribution) == 20
        assert ci.ci_lower <= ci.ci_upper


@plugin_required
class TestLrocMetrics:
    """Integration tests for LROC metric analyzer."""

    def test_lroc_compute_returns_expected_columns(self) -> None:
        """LROC compute returns FPF/sensitivity points."""
        result = LROCAnalyzer().compute(
            _dataset().lazy(),
            pred_col="pred_heatmap",
            gt_mask_col="gt_mask",
            gt_label_col="gt_label",
            image_id_col="image_id",
            weight_col="sample_weight",
            stratify_col="gt_label",
        )
        assert set(result.curve.columns) == {"threshold", "fpf", "sensitivity"}
        assert result.curve.height >= 1
        assert 0.0 <= result.auc() <= 1.0
        assert 0.0 <= result.sensitivity_at_fpf(0.25) <= 1.0

    def test_lroc_requires_single_target_per_positive(self) -> None:
        """LROC raises when positive samples have multiple extracted targets."""
        multi_mask = _mask(16, 16, 2, 2, 6, 6)
        for y in range(10, 14):
            for x in range(10, 14):
                multi_mask[y][x] = 1.0
        df = _dataset().with_columns(
            gt_mask=pl.when(pl.col("image_id") == "a")
            .then(pl.lit(multi_mask))
            .otherwise(pl.col("gt_mask"))
        )
        with pytest.raises(ValueError, match="expects <= 1 target contour"):
            LROCAnalyzer().compute(
                df.lazy(),
                pred_col="pred_heatmap",
                gt_mask_col="gt_mask",
                gt_label_col="gt_label",
                image_id_col="image_id",
            )
