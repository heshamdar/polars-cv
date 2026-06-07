"""Tests for vectorized bootstrap CI computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_cv.metrics import DetectionTable, average_precision
from polars_cv.metrics._bootstrap import (
    BootstrapResult,
    bootstrap_metric_sequential,
    bootstrap_pr_auc,
)
from polars_cv.metrics._types import (
    COL_CLASS_ID,
    COL_DET_IDX,
    COL_GT_IDX,
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IOU,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
    DEFAULT_CLASS,
)

if TYPE_CHECKING:
    pass


@pytest.fixture()
def bootstrap_table() -> DetectionTable:
    """Create a table for bootstrap testing."""
    det_df = pl.DataFrame(
        {
            COL_IMAGE_ID: ["a", "a", "b", "b", "c"],
            COL_CLASS_ID: [DEFAULT_CLASS] * 5,
            COL_SCORE: [0.95, 0.6, 0.8, 0.4, 0.7],
            COL_IS_TP: [True, False, True, True, False],
            COL_GT_IDX: [0, None, 0, 1, None],
            COL_IOU: [0.9, 0.0, 0.85, 0.7, 0.0],
            COL_DET_IDX: [0, 1, 0, 1, 0],
        },
        schema={
            COL_IMAGE_ID: pl.String,
            COL_CLASS_ID: pl.String,
            COL_SCORE: pl.Float64,
            COL_IS_TP: pl.Boolean,
            COL_GT_IDX: pl.UInt32,
            COL_IOU: pl.Float64,
            COL_DET_IDX: pl.UInt32,
        },
    )
    meta_df = pl.DataFrame(
        {
            COL_IMAGE_ID: ["a", "b", "c"],
            COL_CLASS_ID: [DEFAULT_CLASS] * 3,
            COL_N_GTS: [1, 2, 0],
            COL_WEIGHT: [1.0, 1.0, 1.0],
            COL_GT_LABEL: [True, True, False],
        }
    )
    return DetectionTable.from_matched(det_df, meta_df)


class TestSequentialBootstrap:
    """Tests for bootstrap_metric_sequential."""

    def test_returns_bootstrap_result(self) -> None:
        """Sequential bootstrap returns BootstrapResult."""
        result = bootstrap_metric_sequential(
            image_ids=["a", "b", "c"],
            metric_fn=lambda ids: len(ids) * 0.1,
            point_estimate=0.3,
            n_bootstrap=50,
            seed=42,
        )
        assert isinstance(result, BootstrapResult)
        assert len(result.distribution) == 50

    def test_ci_bounds_contain_point(self) -> None:
        """CI should be reasonable relative to the point estimate."""
        result = bootstrap_metric_sequential(
            image_ids=["a", "b", "c"],
            metric_fn=lambda ids: 0.5,
            point_estimate=0.5,
            n_bootstrap=100,
            seed=42,
        )
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_invalid_params(self) -> None:
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="n_bootstrap"):
            bootstrap_metric_sequential(
                image_ids=["a"],
                metric_fn=lambda ids: 0.0,
                point_estimate=0.0,
                n_bootstrap=0,
            )
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_metric_sequential(
                image_ids=["a"],
                metric_fn=lambda ids: 0.0,
                point_estimate=0.0,
                n_bootstrap=10,
                confidence=1.5,
            )


class TestVectorizedPrAucBootstrap:
    """Tests for bootstrap_pr_auc (vectorized path)."""

    def test_returns_bootstrap_result(self, bootstrap_table: DetectionTable) -> None:
        """Vectorized bootstrap returns BootstrapResult."""
        result = bootstrap_pr_auc(
            bootstrap_table,
            n_bootstrap=20,
            seed=42,
        )
        assert isinstance(result, BootstrapResult)
        assert len(result.distribution) == 20

    def test_point_estimate_matches_ap(self, bootstrap_table: DetectionTable) -> None:
        """Point estimate should match direct AP computation."""
        result = bootstrap_pr_auc(
            bootstrap_table,
            n_bootstrap=10,
            seed=42,
        )
        direct_ap = average_precision(bootstrap_table)
        assert abs(result.point_estimate - direct_ap) < 0.01

    def test_ci_bounds_valid(self, bootstrap_table: DetectionTable) -> None:
        """CI lower should be <= upper."""
        result = bootstrap_pr_auc(
            bootstrap_table,
            n_bootstrap=50,
            seed=42,
        )
        assert result.ci_lower <= result.ci_upper
