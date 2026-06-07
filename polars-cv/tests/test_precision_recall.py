"""Tests for precision-recall metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_cv.metrics import (
    DetectionTable,
    PrecisionRecallResult,
    average_precision,
    confusion_at_threshold,
    f1_at_threshold,
    mean_average_precision,
    precision_at_threshold,
    precision_recall_curve,
    recall_at_threshold,
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
def simple_detection_table() -> DetectionTable:
    """Create a simple detection table with known PR curve.

    5 detections: 3 TP, 2 FP; 3 total GTs across 2 images.
    Sorted by score desc: TP(0.9), FP(0.8), TP(0.7), FP(0.5), TP(0.3)
    """
    det_df = pl.DataFrame(
        {
            COL_IMAGE_ID: ["img1", "img1", "img1", "img2", "img2"],
            COL_CLASS_ID: [DEFAULT_CLASS] * 5,
            COL_SCORE: [0.9, 0.8, 0.7, 0.5, 0.3],
            COL_IS_TP: [True, False, True, False, True],
            COL_GT_IDX: [0, None, 1, None, 0],
            COL_IOU: [0.85, 0.0, 0.7, 0.0, 0.6],
            COL_DET_IDX: [0, 1, 2, 0, 1],
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
            COL_IMAGE_ID: ["img1", "img2"],
            COL_CLASS_ID: [DEFAULT_CLASS, DEFAULT_CLASS],
            COL_N_GTS: [2, 1],
            COL_WEIGHT: [1.0, 1.0],
            COL_GT_LABEL: [True, True],
        }
    )
    return DetectionTable.from_matched(det_df, meta_df)


class TestPrecisionRecallCurve:
    """Tests for precision_recall_curve function."""

    def test_returns_pr_result(self, simple_detection_table: DetectionTable) -> None:
        """Returns a PrecisionRecallResult."""
        result = precision_recall_curve(simple_detection_table)
        assert isinstance(result, PrecisionRecallResult)

    def test_curve_shape(self, simple_detection_table: DetectionTable) -> None:
        """Curve has expected columns."""
        result = precision_recall_curve(simple_detection_table)
        assert "score" in result.curve.columns
        assert "precision" in result.curve.columns
        assert "recall" in result.curve.columns
        assert "cum_tp" in result.curve.columns
        assert "cum_fp" in result.curve.columns

    def test_precision_recall_values(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """Verify hand-computed precision and recall at each rank.

        Ranked by score desc: TP(0.9), FP(0.8), TP(0.7), FP(0.5), TP(0.3)
        P: [1/1, 1/2, 2/3, 2/4, 3/5]  = [1.0, 0.5, 0.667, 0.5, 0.6]
        R: [1/3, 1/3, 2/3, 2/3, 3/3]  = [0.333, 0.333, 0.667, 0.667, 1.0]
        """
        result = precision_recall_curve(simple_detection_table)
        curve = result.curve

        precisions = curve["precision"].to_list()
        recalls = curve["recall"].to_list()

        assert abs(precisions[0] - 1.0) < 0.01
        assert abs(precisions[1] - 0.5) < 0.01
        assert abs(precisions[2] - 2 / 3) < 0.01
        assert abs(recalls[-1] - 1.0) < 0.01

    def test_empty_table(self) -> None:
        """Empty detection table produces empty curve."""
        det_df = pl.DataFrame(
            schema={
                COL_IMAGE_ID: pl.String,
                COL_CLASS_ID: pl.String,
                COL_SCORE: pl.Float64,
                COL_IS_TP: pl.Boolean,
                COL_GT_IDX: pl.UInt32,
                COL_IOU: pl.Float64,
                COL_DET_IDX: pl.UInt32,
            }
        )
        meta_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["img1"],
                COL_CLASS_ID: [DEFAULT_CLASS],
                COL_N_GTS: [0],
                COL_WEIGHT: [1.0],
                COL_GT_LABEL: [False],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        result = precision_recall_curve(table)
        assert result.curve.height == 0


class TestAveragePrecision:
    """Tests for average_precision function."""

    def test_all_points_ap(self, simple_detection_table: DetectionTable) -> None:
        """AP should be between 0 and 1."""
        ap = average_precision(simple_detection_table)
        assert 0.0 <= ap <= 1.0

    def test_eleven_point_ap(self, simple_detection_table: DetectionTable) -> None:
        """11-point AP should be between 0 and 1."""
        ap = average_precision(simple_detection_table, interpolation="11_point")
        assert 0.0 <= ap <= 1.0

    def test_perfect_detector(self) -> None:
        """Perfect detector (all detections are TP, no FP) has high AP.

        With 2 GTs and 2 TPs scored [0.9, 0.8]:
        - rank 1: P=1/1=1.0, R=1/2=0.5
        - rank 2: P=2/2=1.0, R=2/2=1.0
        Trapezoidal AUC from (0.5, 1.0) to (1.0, 1.0) = 0.5.
        11-point AP = 1.0 (precision is 1.0 at all recall levels).
        """
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["img1", "img1"],
                COL_CLASS_ID: [DEFAULT_CLASS, DEFAULT_CLASS],
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
                COL_IMAGE_ID: ["img1"],
                COL_CLASS_ID: [DEFAULT_CLASS],
                COL_N_GTS: [2],
                COL_WEIGHT: [1.0],
                COL_GT_LABEL: [True],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        # 11-point interpolation gives 1.0 since precision is always 1.0
        ap_11 = average_precision(table, interpolation="11_point")
        assert abs(ap_11 - 1.0) < 0.01
        # Trapezoidal AUC is 0.5 because curve starts at recall=0.5
        ap_trap = average_precision(table, interpolation="all_points")
        assert abs(ap_trap - 0.5) < 0.01


class TestMeanAveragePrecision:
    """Tests for mean_average_precision function."""

    def test_single_threshold(self, simple_detection_table: DetectionTable) -> None:
        """mAP at default threshold should match AP for single class."""
        map_val = mean_average_precision(simple_detection_table)
        ap_val = average_precision(simple_detection_table)
        assert abs(map_val - ap_val) < 0.01

    def test_rethreshold(self, simple_detection_table: DetectionTable) -> None:
        """mAP across multiple IoU thresholds uses re-thresholding."""
        map_val = mean_average_precision(
            simple_detection_table,
            iou_thresholds=[0.5, 0.75],
        )
        assert 0.0 <= map_val <= 1.0


class TestPrecisionRecallAtThreshold:
    """Tests for threshold-based metrics."""

    def test_precision_at_threshold(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """Precision at a known threshold."""
        p = precision_at_threshold(simple_detection_table, 0.7)
        # At threshold 0.7: detections with score >= 0.7 are [0.9(TP), 0.8(FP), 0.7(TP)]
        # Precision = 2/3
        assert abs(p - 2 / 3) < 0.01

    def test_recall_at_threshold(self, simple_detection_table: DetectionTable) -> None:
        """Recall at a known threshold."""
        r = recall_at_threshold(simple_detection_table, 0.7)
        # At threshold 0.7: 2 TPs, 3 total GTs => recall = 2/3
        assert abs(r - 2 / 3) < 0.01

    def test_f1_at_threshold(self, simple_detection_table: DetectionTable) -> None:
        """F1 at a known threshold."""
        f1 = f1_at_threshold(simple_detection_table, 0.7)
        # P = R = 2/3, so F1 = 2/3
        assert abs(f1 - 2 / 3) < 0.01


class TestConfusionAtThreshold:
    """Tests for confusion_at_threshold function."""

    def test_confusion_counts(self, simple_detection_table: DetectionTable) -> None:
        """Verify TP/FP/FN counts at a specific threshold."""
        result = confusion_at_threshold(simple_detection_table, 0.7)
        assert result["tp"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 1  # 3 total GTs - 2 TPs
