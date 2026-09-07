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
from polars_cv.metrics._metrics._precision_recall import (
    _all_points_ap,
    all_points_ap_by_group,
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
        """Perfect detector (all detections are TP, no FP) has AP = 1.0.

        With 2 GTs and 2 TPs scored [0.9, 0.8]:
        - rank 1: P=1/1=1.0, R=1/2=0.5
        - rank 2: P=2/2=1.0, R=2/2=1.0
        With the recall=0 anchor, all-points AP integrates to 1.0
        (matching 11-point AP and sklearn average_precision_score).
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
        ap_11 = average_precision(table, interpolation="11_point")
        assert abs(ap_11 - 1.0) < 0.01
        ap_all = average_precision(table, interpolation="all_points")
        assert abs(ap_all - 1.0) < 0.01


@pytest.fixture()
def multiclass_detection_table() -> DetectionTable:
    """Two classes (``cat``/``dog``), distinct scores, one GT per (image, class).

    Distinct scores keep every all-points AP well-defined (no tie-order
    ambiguity), so the pinned mAP values below are exact ground truth.
    """
    det_df = pl.DataFrame(
        {
            COL_IMAGE_ID: ["i1", "i1", "i2", "i1", "i2", "i2"],
            COL_CLASS_ID: ["cat", "cat", "cat", "dog", "dog", "dog"],
            COL_SCORE: [0.9, 0.8, 0.6, 0.85, 0.7, 0.4],
            COL_IS_TP: [True, False, True, True, False, True],
            COL_GT_IDX: [0, None, 1, 0, None, 1],
            COL_IOU: [0.95, 0.0, 0.72, 0.9, 0.0, 0.55],
            COL_DET_IDX: [0, 1, 2, 0, 1, 2],
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
            COL_IMAGE_ID: ["i1", "i2", "i1", "i2"],
            COL_CLASS_ID: ["cat", "cat", "dog", "dog"],
            COL_N_GTS: [1, 1, 1, 1],
            COL_WEIGHT: [1.0, 1.0, 1.0, 1.0],
            COL_GT_LABEL: [True, True, True, True],
        }
    )
    return DetectionTable.from_matched(det_df, meta_df, matching_iou_threshold=0.5)


class TestMeanAveragePrecision:
    """Tests for mean_average_precision function.

    The exact-value pins below lock the numeric output of ``mAP`` so the
    vectorization onto the grouped all-points authority (CR-07) cannot change
    any result. All fixtures use distinct scores, where the all-points estimator
    is tie-order-independent and therefore exactly reproducible.
    """

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

    def test_single_class_exact_values(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """Exact pins on the single-class fixture (both interpolations)."""
        assert mean_average_precision(simple_detection_table) == pytest.approx(
            0.7555555555555555, abs=1e-9
        )
        assert mean_average_precision(
            simple_detection_table, interpolation="11_point"
        ) == pytest.approx(0.7636363636363636, abs=1e-9)

    def test_multi_threshold_exact_values(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """Exact pins across a COCO-style set of IoU thresholds."""
        assert mean_average_precision(
            simple_detection_table, iou_thresholds=[0.5, 0.75]
        ) == pytest.approx(0.5444444444444444, abs=1e-9)
        assert mean_average_precision(
            simple_detection_table, iou_thresholds=[0.5, 0.75], interpolation="11_point"
        ) == pytest.approx(0.5636363636363637, abs=1e-9)

    def test_multiclass_exact_values(
        self, multiclass_detection_table: DetectionTable
    ) -> None:
        """Exact pins for multiple classes, single and multiple thresholds."""
        t = multiclass_detection_table
        assert mean_average_precision(t) == pytest.approx(0.8333333333333333, abs=1e-9)
        assert mean_average_precision(t, interpolation="11_point") == pytest.approx(
            0.8484848484848484, abs=1e-9
        )
        assert mean_average_precision(t, iou_thresholds=[0.5, 0.75]) == pytest.approx(
            0.6666666666666666, abs=1e-9
        )
        assert mean_average_precision(
            t, iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]
        ) == pytest.approx(0.6333333333333333, abs=1e-9)
        assert mean_average_precision(
            t, iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9], interpolation="11_point"
        ) == pytest.approx(0.6666666666666664, abs=1e-9)

    def test_class_without_detections_contributes_zero(self) -> None:
        """A class with GTs but no detections averages in as AP = 0.

        ``cat`` is a perfect detector (AP = 1.0); ``dog`` has metadata (2 GTs)
        but no detections, so its AP is 0.0. mAP = (1.0 + 0.0) / 2 = 0.5. This
        pins the grid-denominator behaviour the vectorized path must preserve:
        every ``(threshold, class)`` cell counts, present in the detections or
        not.
        """
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["i1", "i1"],
                COL_CLASS_ID: ["cat", "cat"],
                COL_SCORE: [0.9, 0.5],
                COL_IS_TP: [True, True],
                COL_GT_IDX: [0, 1],
                COL_IOU: [0.9, 0.8],
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
                COL_IMAGE_ID: ["i1", "i1"],
                COL_CLASS_ID: ["cat", "dog"],
                COL_N_GTS: [2, 2],
                COL_WEIGHT: [1.0, 1.0],
                COL_GT_LABEL: [True, True],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        assert mean_average_precision(table) == pytest.approx(0.5, abs=1e-9)
        assert mean_average_precision(table, interpolation="11_point") == pytest.approx(
            0.5, abs=1e-9
        )

    def test_class_with_zero_gts_contributes_zero(self) -> None:
        """A class whose GT count is zero scores AP = 0 (no division blow-up).

        ``cat`` has 1 GT and a perfect TP (AP = 1.0); ``dog`` has detections but
        ``n_gts = 0``, so recall is undefined and its AP is defined as 0.0. mAP =
        (1.0 + 0.0) / 2 = 0.5.
        """
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["i1", "i2"],
                COL_CLASS_ID: ["cat", "dog"],
                COL_SCORE: [0.9, 0.7],
                COL_IS_TP: [True, False],
                COL_GT_IDX: [0, None],
                COL_IOU: [0.9, 0.0],
                COL_DET_IDX: [0, 0],
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
                COL_IMAGE_ID: ["i1", "i2"],
                COL_CLASS_ID: ["cat", "dog"],
                COL_N_GTS: [1, 0],
                COL_WEIGHT: [1.0, 1.0],
                COL_GT_LABEL: [True, False],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        assert mean_average_precision(table) == pytest.approx(0.5, abs=1e-9)


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


class TestAllPointsAPAuthority:
    """The scalar ``_all_points_ap`` and grouped ``all_points_ap_by_group``.

    Both implement the same monotone-envelope + anchored-trapezoid estimator.
    They agree *exactly* on any curve with distinct scores. They *can* diverge
    on exact score ties, because the all-points AP is tie-order-sensitive and
    each path presents a differently ordered frame to an unstable Polars sort
    — the tie-break, and thus the AP, is arbitrary in both. Whether a given
    tied fixture actually diverges depends on the sort behaviour of whichever
    Polars engine/platform runs it, so no specific fixture is guaranteed to
    diverge everywhere. That possible divergence is the documented reason
    CR-06 keeps them as two functions rather than folding
    ``PrecisionRecallResult.auc`` onto the grouped path (which would change its
    already-arbitrary tie output); see ``metrics/AGENTS.md`` and CR-30.
    """

    @staticmethod
    def _grouped_ap(scores: list[float], is_tp: list[bool], total_gts: float) -> float:
        expanded = pl.LazyFrame(
            {
                COL_SCORE: scores,
                COL_IS_TP: is_tp,
                "total_gts": [float(total_gts)] * len(scores),
                "_g": [0] * len(scores),
            }
        )
        out = all_points_ap_by_group(expanded, group_col="_g").collect()
        return float(out["ap"][0]) if out.height else 0.0

    @staticmethod
    def _scalar_ap(scores: list[float], is_tp: list[bool], total_gts: int) -> float:
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: [f"i{i}" for i in range(len(scores))],
                COL_CLASS_ID: [DEFAULT_CLASS] * len(scores),
                COL_SCORE: scores,
                COL_IS_TP: is_tp,
                COL_GT_IDX: [i if t else None for i, t in enumerate(is_tp)],
                COL_IOU: [0.9 if t else 0.0 for t in is_tp],
                COL_DET_IDX: list(range(len(scores))),
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
                COL_IMAGE_ID: ["m"],
                COL_CLASS_ID: [DEFAULT_CLASS],
                COL_N_GTS: [total_gts],
                COL_WEIGHT: [1.0],
                COL_GT_LABEL: [True],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        return _all_points_ap(precision_recall_curve(table).curve)

    @pytest.mark.parametrize(
        ("scores", "is_tp", "gts"),
        [
            ([0.9, 0.8, 0.7, 0.5, 0.3], [True, False, True, False, True], 3),
            ([0.9], [True], 1),
            ([0.9], [False], 1),
            ([0.95, 0.85, 0.75], [True, True, True], 3),
            ([0.95, 0.85, 0.75], [False, False, False], 3),
            ([0.9, 0.8, 0.7, 0.6], [True, True, False, True], 5),
            ([0.42, 0.31, 0.20, 0.11], [False, True, True, False], 3),
        ],
    )
    def test_scalar_matches_grouped_on_distinct_scores(
        self, scores: list[float], is_tp: list[bool], gts: int
    ) -> None:
        """On distinct scores the two authorities are bit-identical."""
        scalar = self._scalar_ap(scores, is_tp, gts)
        grouped = self._grouped_ap(scores, is_tp, float(gts))
        assert scalar == pytest.approx(grouped, abs=1e-12)

    def test_tie_curve_is_valid_on_both_paths(self) -> None:
        """On exact score ties the two paths may legitimately disagree.

        Which way the tie-break falls depends on the sort implementation of
        whichever Polars engine/platform runs it (see the class docstring),
        so the two values are *not* pinned to differ: they coincided on
        macOS-arm64 CI while diverging on Linux for this exact fixture. What
        is guaranteed, and what this pins, is that both remain valid
        all-points APs for the tied curve — see CR-30 for the eventual merge
        of the two authorities.
        """
        scores = [0.5, 0.5, 0.5, 0.5]
        is_tp = [True, False, True, False]
        scalar = self._scalar_ap(scores, is_tp, 4)
        grouped = self._grouped_ap(scores, is_tp, 4.0)
        assert 0.0 <= scalar <= 1.0
        assert 0.0 <= grouped <= 1.0


class TestConfusionAtThreshold:
    """Tests for confusion_at_threshold function."""

    def test_confusion_counts(self, simple_detection_table: DetectionTable) -> None:
        """Verify TP/FP/FN counts at a specific threshold."""
        result = confusion_at_threshold(simple_detection_table, 0.7)
        assert result.tp == 2
        assert result.fp == 1
        assert result.fn == 1  # 3 total GTs - 2 TPs
        # Legacy mapping access remains available via to_dict().
        assert result.to_dict() == {"tp": 2, "fp": 1, "fn": 1}
        # Derived metrics: precision = 2/3, recall = 2/3.
        assert result.precision == 2 / 3
        assert result.recall == 2 / 3
