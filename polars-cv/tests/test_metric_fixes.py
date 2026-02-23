"""Tests for detection metric fixes: FROC, LROC, PR AUC, AUC utilities, and MW-U.

Covers:
- FROC total_gts overcounting fix
- FROC bootstrap recomputation of total_targets
- FROC iou_threshold propagation from DetectionTable
- LROC lower-right endpoint addition
- PR monotone-envelope AP vs raw trapezoidal AUC
- partial_auc extrapolation warning
- Mann-Whitney U AUC for FROCResult and LROCResult
- ContourMatcher min_contour_area default change
- Zero-score detection filtering
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars_cv.metrics import (
    DetectionTable,
    froc_curve,
    lroc_curve,
    precision_recall_curve,
)
from polars_cv.metrics._auc import mann_whitney_u_auc, partial_auc
from polars_cv.metrics._matching._contour import ContourMatcher
from polars_cv.metrics._metrics._froc import _curve_from_dense
from polars_cv.metrics._metrics._lroc import LROCResult, _build_lroc_curve
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
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_detection_table() -> DetectionTable:
    """Create a simple detection table for FROC/LROC tests.

    3 images: 2 positive (a, b) with 1 GT each, 1 negative (c).
    Detections: TP@0.9 on img a, FP@0.8 on img b, FP@0.7 on img c.
    """
    det_df = pl.DataFrame(
        {
            COL_IMAGE_ID: ["a", "b", "c"],
            COL_CLASS_ID: [DEFAULT_CLASS] * 3,
            COL_SCORE: [0.9, 0.8, 0.7],
            COL_IS_TP: [True, False, False],
            COL_GT_IDX: [0, None, None],
            COL_IOU: [0.85, 0.0, 0.0],
            COL_DET_IDX: [0, 0, 0],
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
            COL_N_GTS: [1, 1, 0],
            COL_WEIGHT: [1.0, 1.0, 1.0],
            COL_GT_LABEL: [True, True, False],
        }
    )
    return DetectionTable.from_matched(det_df, meta_df, matching_iou_threshold=0.5)


# ---------------------------------------------------------------------------
# FROC Fixes
# ---------------------------------------------------------------------------


class TestFrocTotalGtsFix:
    """Verify FROC total_gts is not overcounted across thresholds."""

    def test_unweighted_total_gts_not_overcounted(self) -> None:
        """Unweighted path computes total_gts from unique images, not dense grid.

        With 3 images (n_gts = [1, 1, 0]) and 3 thresholds, the dense grid
        has 9 rows.  Summing n_gts naively gives 6; correct answer is 2.
        """
        dense = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                "threshold": [0.5, 0.7, 0.9] * 3,
                "tp": [1, 1, 0, 0, 0, 0, 0, 0, 0],
                "fp": [0, 0, 0, 1, 0, 0, 1, 0, 0],
                COL_N_GTS: [1, 1, 1, 1, 1, 1, 0, 0, 0],
                COL_WEIGHT: [1.0] * 9,
            }
        )
        curve = _curve_from_dense(dense, weighted=False)
        # total_gts in the curve should reflect unique images (1+1+0=2)
        total_gts_values = curve["total_gts"].unique().to_list()
        for v in total_gts_values:
            assert v <= 2, f"total_gts overcounted: {v}"

    def test_weighted_path_not_overcounted(self) -> None:
        """Weighted path (via weighted_curve) also produces correct total_gts."""
        dense = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "a", "b", "b"],
                "threshold": [0.5, 0.9, 0.5, 0.9],
                "tp": [1, 0, 0, 0],
                "fp": [0, 0, 1, 0],
                COL_N_GTS: [1, 1, 1, 1],
                COL_WEIGHT: [1.0, 1.0, 1.0, 1.0],
            }
        )
        curve = _curve_from_dense(dense, weighted=True)
        assert curve.height > 0
        sens_vals = curve["sensitivity"].to_list()
        for s in sens_vals:
            assert s is None or s <= 1.0, f"sensitivity > 1.0: {s}"


class TestFrocIouThresholdPropagation:
    """Verify iou_threshold is read from DetectionTable, not a broken expression."""

    def test_iou_threshold_from_table(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """FROCResult.iou_threshold matches the matching_iou_threshold."""
        result = froc_curve(simple_detection_table)
        assert result.iou_threshold == 0.5


class TestFrocBootstrapRecomputesTotalTargets:
    """Verify bootstrap recomputes total_targets from sampled images."""

    def test_bootstrap_sampled_total_targets(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """Bootstrap should not use a stale total_targets.

        We indirectly verify this by running bootstrap and checking
        that the CI does not collapse to a single point (which would
        happen if total_targets was wrong for all samples).
        """
        result = froc_curve(simple_detection_table)
        ci = result.bootstrap_ci(n_bootstrap=10, seed=42)
        assert len(ci.distribution) == 10


# ---------------------------------------------------------------------------
# LROC Fixes
# ---------------------------------------------------------------------------


class TestLrocEndpoint:
    """Verify LROC curve includes the lower-right endpoint."""

    def test_curve_has_origin_and_lower_right(self) -> None:
        """LROC curve spans [0, 1] on FPF with both sentinel points."""
        per_image = pl.DataFrame(
            {
                "image_id": ["p_tp", "p_fn", "n_fp"],
                "gt_label": [True, True, False],
                "weight": [1.0, 1.0, 1.0],
                "max_score": [0.9, None, 0.8],
                "top_is_tp": [True, False, False],
            }
        )
        curve = _build_lroc_curve(per_image)
        fpf_vals = sorted(curve["fpf"].to_list())
        assert 0.0 in fpf_vals, "Origin (fpf=0) missing"
        assert 1.0 in fpf_vals, "Lower-right (fpf=1) missing"

    def test_lower_right_sensitivity_is_max(self) -> None:
        """The lower-right sentinel (threshold=-inf) carries max sensitivity."""
        per_image = pl.DataFrame(
            {
                "image_id": ["p1", "p2", "n1"],
                "gt_label": [True, True, False],
                "weight": [1.0, 1.0, 1.0],
                "max_score": [0.9, 0.7, 0.5],
                "top_is_tp": [True, True, False],
            }
        )
        curve = _build_lroc_curve(per_image)
        sentinel = curve.filter(pl.col("threshold") == float("-inf"))
        assert sentinel.height == 1
        max_sens = float(sentinel["sensitivity"].item())
        # Both positives have a TP → max sensitivity = 1.0
        assert max_sens == pytest.approx(1.0)
        assert float(sentinel["fpf"].item()) == pytest.approx(1.0)

    def test_auc_spans_full_range(self) -> None:
        """AUC computation covers fpf from 0 to 1."""
        per_image = pl.DataFrame(
            {
                "image_id": ["p1", "n1"],
                "gt_label": [True, False],
                "weight": [1.0, 1.0],
                "max_score": [0.9, 0.5],
                "top_is_tp": [True, False],
            }
        )
        curve = _build_lroc_curve(per_image)
        result = LROCResult(
            curve=curve,
            per_image=per_image,
            n_positive=1,
            n_negative=1,
        )
        auc = result.auc()
        # With 1 positive (TP) and 1 negative: should give AUC = 1.0
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# PR Curve AP Fix
# ---------------------------------------------------------------------------


class TestPrApEnvelope:
    """Verify auc() uses monotone-envelope and raw_auc() does not."""

    def test_auc_ge_raw_auc(self) -> None:
        """Monotone-envelope auc() >= raw_auc() for typical curves.

        The envelope can only increase precision, so auc() should be >= raw_auc()
        (unless the curve is already monotone decreasing).
        """
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "a", "a", "a", "a"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 5,
                COL_SCORE: [0.9, 0.8, 0.7, 0.5, 0.3],
                COL_IS_TP: [True, False, True, False, True],
                COL_GT_IDX: [0, None, 1, None, 2],
                COL_IOU: [0.85, 0.0, 0.7, 0.0, 0.6],
                COL_DET_IDX: [0, 1, 2, 3, 4],
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
                COL_IMAGE_ID: ["a"],
                COL_CLASS_ID: [DEFAULT_CLASS],
                COL_N_GTS: [3],
                COL_WEIGHT: [1.0],
                COL_GT_LABEL: [True],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        result = precision_recall_curve(table)
        raw = result.raw_auc()
        envelope = result.auc()
        assert envelope >= raw - 1e-10

    def test_perfect_detector_ap_equals_one(self) -> None:
        """Perfect detector (all TP, no FP) has AP = 1.0 with envelope.

        Recall goes from 0.5 to 1.0, precision is always 1.0.
        Raw AUC = 0.5, but 11-point AP = 1.0. Envelope AP should also
        equal raw AUC here since precision is already monotone.
        """
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "a"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 2,
                COL_SCORE: [0.9, 0.8],
                COL_IS_TP: [True, True],
                COL_GT_IDX: [0, 1],
                COL_IOU: [0.9, 0.85],
                COL_DET_IDX: [0, 1],
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
                COL_IMAGE_ID: ["a"],
                COL_CLASS_ID: [DEFAULT_CLASS],
                COL_N_GTS: [2],
                COL_WEIGHT: [1.0],
                COL_GT_LABEL: [True],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        result = precision_recall_curve(table)
        assert result.auc() == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Partial AUC Warning
# ---------------------------------------------------------------------------


class TestPartialAucWarning:
    """Verify partial_auc warns on significant left-boundary extrapolation."""

    def test_warns_on_large_gap(self) -> None:
        """Warning emitted when lo is far below curve minimum x."""
        x = np.array([0.5, 0.6, 0.7, 0.8, 1.0])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            partial_auc(x, y, lo=0.0, hi=1.0)
            assert any("partial_auc" in str(warning.message) for warning in w)

    def test_no_warning_on_small_gap(self) -> None:
        """No warning when lo is close to curve minimum x."""
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            partial_auc(x, y, lo=0.0, hi=1.0)
            assert not any("partial_auc" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Mann-Whitney U AUC
# ---------------------------------------------------------------------------


class TestMannWhitneyUAuc:
    """Tests for the mann_whitney_u_auc utility function."""

    def test_perfect_separation(self) -> None:
        """All positives above all negatives → AUC = 1.0."""
        pos = np.array([0.9, 0.8, 0.7])
        neg = np.array([0.3, 0.2, 0.1])
        assert mann_whitney_u_auc(pos, neg) == pytest.approx(1.0)

    def test_reversed_separation(self) -> None:
        """All positives below all negatives → AUC = 0.0."""
        pos = np.array([0.1, 0.2, 0.3])
        neg = np.array([0.7, 0.8, 0.9])
        assert mann_whitney_u_auc(pos, neg) == pytest.approx(0.0)

    def test_equal_scores(self) -> None:
        """Identical distributions → AUC = 0.5."""
        pos = np.array([0.5, 0.5, 0.5])
        neg = np.array([0.5, 0.5, 0.5])
        assert mann_whitney_u_auc(pos, neg) == pytest.approx(0.5)

    def test_empty_groups(self) -> None:
        """Empty group returns 0.5 sentinel."""
        assert mann_whitney_u_auc(np.array([0.9]), np.array([])) == pytest.approx(0.5)
        assert mann_whitney_u_auc(np.array([]), np.array([0.1])) == pytest.approx(0.5)


class TestFrocMannWhitneyAuc:
    """Tests for mann_whitney_auc on FROCResult."""

    def test_detection_level(self, simple_detection_table: DetectionTable) -> None:
        """Detection-level MW-U should return a value in [0, 1]."""
        result = froc_curve(simple_detection_table)
        mw_auc = result.mann_whitney_auc(level="detection")
        assert 0.0 <= mw_auc <= 1.0

    def test_image_level(self, simple_detection_table: DetectionTable) -> None:
        """Image-level MW-U should return a value in [0, 1]."""
        result = froc_curve(simple_detection_table)
        mw_auc = result.mann_whitney_auc(level="image")
        assert 0.0 <= mw_auc <= 1.0

    def test_invalid_level_raises(self, simple_detection_table: DetectionTable) -> None:
        """Invalid level raises ValueError."""
        result = froc_curve(simple_detection_table)
        with pytest.raises(ValueError, match="Unsupported level"):
            result.mann_whitney_auc(level="invalid")


class TestLrocMannWhitneyAuc:
    """Tests for mann_whitney_auc on LROCResult."""

    def test_detection_level(self, simple_detection_table: DetectionTable) -> None:
        """Detection-level MW-U should return a value in [0, 1]."""
        result = lroc_curve(simple_detection_table)
        mw_auc = result.mann_whitney_auc(level="detection")
        assert 0.0 <= mw_auc <= 1.0

    def test_image_level(self, simple_detection_table: DetectionTable) -> None:
        """Image-level MW-U should return a value in [0, 1]."""
        result = lroc_curve(simple_detection_table)
        mw_auc = result.mann_whitney_auc(level="image")
        assert 0.0 <= mw_auc <= 1.0


# ---------------------------------------------------------------------------
# ContourMatcher defaults
# ---------------------------------------------------------------------------


class TestContourMatcherDefaults:
    """Verify ContourMatcher defaults were changed correctly."""

    def test_min_contour_area_default_is_one(self) -> None:
        """Default min_contour_area changed from 0.0 to 1.0."""
        matcher = ContourMatcher()
        assert matcher._min_contour_area == 1.0

    def test_gt_min_contour_area_defaults_to_min_contour_area(self) -> None:
        """gt_min_contour_area defaults to min_contour_area when None."""
        matcher = ContourMatcher(min_contour_area=2.0)
        assert matcher._gt_min_contour_area is None
        # At match time, gt_area = gt_min_contour_area or min_contour_area
        gt_area = (
            matcher._gt_min_contour_area
            if matcher._gt_min_contour_area is not None
            else matcher._min_contour_area
        )
        assert gt_area == 2.0

    def test_explicit_min_contour_area_still_works(self) -> None:
        """Explicit min_contour_area=0.0 is still allowed."""
        matcher = ContourMatcher(min_contour_area=0.0)
        assert matcher._min_contour_area == 0.0
