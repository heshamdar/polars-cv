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
    ap_by_group,
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

    Both implement the same monotone-envelope + anchored-trapezoid estimator and
    agree *exactly* — on distinct scores and on ties alike. CR-30 gave them a
    shared canonical tie convention (collapse every detection sharing a score
    into one PR point, at the cumulative counts after the whole tied block), so
    the all-points AP no longer depends on the input row order among equal scores
    and the two paths no longer diverge; see ``metrics/AGENTS.md`` and CR-30.
    """

    @staticmethod
    def _grouped_ap(scores: list[float], is_tp: list[bool], total_gts: float) -> float:
        expanded = pl.LazyFrame(
            {
                COL_SCORE: scores,
                COL_IS_TP: is_tp,
                COL_WEIGHT: [1.0] * len(scores),
                "gt_mass": [float(total_gts)] * len(scores),
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

    def test_tie_handling_is_order_independent_and_the_paths_agree(self) -> None:
        """On exact score ties both paths collapse the tie into one PR point.

        The all-points AP used to be tie-order-sensitive and the two paths could
        disagree (the documented CR-06 gap, which earlier only pinned that both
        stayed valid all-points APs because the divergence was platform-
        dependent). CR-30's canonical tie convention -- collapse detections
        sharing a score to the cumulative counts after the whole tied block --
        makes the AP independent of the row order among equal scores and makes
        the scalar and grouped authorities agree exactly.
        """
        scores = [0.5, 0.5, 0.5, 0.5]
        is_tp = [True, False, True, False]
        # Same score multiset, tied rows permuted -- must not change the AP.
        permuted_is_tp = [False, True, False, True]

        scalar = self._scalar_ap(scores, is_tp, 4)
        grouped = self._grouped_ap(scores, is_tp, 4.0)
        # Both remain valid all-points APs...
        assert 0.0 <= scalar <= 1.0
        assert 0.0 <= grouped <= 1.0
        # ...and now agree exactly on the tied curve.
        assert scalar == pytest.approx(grouped, abs=1e-12)
        # And the value is invariant to the input order among the tied scores.
        assert self._scalar_ap(scores, permuted_is_tp, 4) == pytest.approx(
            scalar, abs=1e-12
        )
        assert self._grouped_ap(scores, permuted_is_tp, 4.0) == pytest.approx(
            grouped, abs=1e-12
        )


def _ref_step_ap(scores: list[float], is_tp: list[bool], total_gts: float) -> float:
    """Independent reference: all-points AP by the step rule.

    One PR point per distinct score (after its whole tied block), precision
    replaced by its right-to-left envelope, then ``Σ (Rₖ − Rₖ₋₁) · P̂ₖ`` with
    ``R₀ = 0`` — the Pascal VOC 2010+ / COCO all-points definition. A
    trapezoid ``(P̂ₖ + P̂ₖ₋₁)/2`` instead overstates AP whenever a tied block
    mixes TPs and FPs.
    """
    points: list[tuple[float, float]] = []
    tp = fp = 0
    for s in sorted(set(scores), reverse=True):
        block = [t for sc, t in zip(scores, is_tp, strict=True) if sc == s]
        tp += sum(block)
        fp += len(block) - sum(block)
        points.append((tp / total_gts, tp / (tp + fp)))
    ap, prev_r = 0.0, 0.0
    for k, (r, _) in enumerate(points):
        envelope = max(p for _, p in points[k:])
        ap += (r - prev_r) * envelope
        prev_r = r
    return ap


class TestAllPointsAPStepRule:
    """All-points AP integrates the envelope as a step function (F5).

    Score ties are where the rules differ: a tied block holding a TP and FPs
    drops precision while raising recall, and a trapezoid credits that recall
    step with the *higher* precision of the point before it.
    """

    CASES = [
        # TP@0.9, then {TP, FP, FP}@0.5 against 2 GTs: 0.5·1 + 0.5·0.5.
        ([0.9, 0.5, 0.5, 0.5], [True, True, False, False], 2, 0.75),
        ([0.9, 0.7, 0.7, 0.3, 0.3], [True, True, False, True, False], 4, None),
        ([0.8, 0.8, 0.6, 0.6, 0.6], [True, False, True, False, False], 3, None),
        # Untied scores: unchanged.
        ([0.9, 0.8, 0.7, 0.5, 0.3], [True, False, True, False, True], 3, None),
    ]

    @pytest.mark.parametrize(("scores", "is_tp", "gts", "expected"), CASES)
    def test_grouped_and_scalar_authorities_follow_the_step_rule(
        self,
        scores: list[float],
        is_tp: list[bool],
        gts: int,
        expected: float | None,
    ) -> None:
        ref = _ref_step_ap(scores, is_tp, float(gts))
        if expected is not None:
            assert ref == pytest.approx(expected, abs=1e-12)
        authority = TestAllPointsAPAuthority
        assert authority._grouped_ap(scores, is_tp, float(gts)) == pytest.approx(
            ref, abs=1e-12
        )
        assert authority._scalar_ap(scores, is_tp, gts) == pytest.approx(ref, abs=1e-12)

    def test_public_average_precision_on_a_tied_block(self) -> None:
        det = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a"] * 4,
                COL_CLASS_ID: [DEFAULT_CLASS] * 4,
                COL_SCORE: [0.9, 0.5, 0.5, 0.5],
                COL_IS_TP: [True, True, False, False],
                COL_GT_IDX: [0, 1, None, None],
                COL_IOU: [0.9, 0.9, None, None],
                COL_DET_IDX: [0, 1, 2, 3],
            },
            schema_overrides={COL_GT_IDX: pl.UInt32, COL_DET_IDX: pl.UInt32},
        )
        meta = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a"],
                COL_CLASS_ID: [DEFAULT_CLASS],
                COL_N_GTS: [2],
                COL_WEIGHT: [1.0],
                COL_GT_LABEL: [True],
            }
        )
        table = DetectionTable.from_matched(det, meta)
        assert average_precision(table) == pytest.approx(0.75, abs=1e-12)
        assert mean_average_precision(table) == pytest.approx(0.75, abs=1e-12)


def _ref_coco_101(scores: list[float], is_tp: list[bool], total_gts: float) -> float:
    """pycocotools' ``COCOeval.accumulate`` for one (class, IoU, area) cell.

    The PR points are taken per score bucket (this library's tie convention;
    COCO takes them per detection in mergesort order, which only differs
    inside a tied block), the envelope is built right to left, and precision is
    read at ``np.searchsorted(recall, np.linspace(0, 1, 101), side="left")``.
    """
    import numpy as np

    rc: list[float] = []
    pr: list[float] = []
    tp = fp = 0
    for s in sorted(set(scores), reverse=True):
        block = [t for sc, t in zip(scores, is_tp, strict=True) if sc == s]
        tp += sum(block)
        fp += len(block) - sum(block)
        rc.append(tp / total_gts)
        pr.append(tp / (tp + fp))
    for i in range(len(pr) - 1, 0, -1):
        pr[i - 1] = max(pr[i - 1], pr[i])
    rec_thrs = np.linspace(0.0, 1.00, 101)
    inds = np.searchsorted(np.array(rc), rec_thrs, side="left")
    q = [pr[i] if i < len(pr) else 0.0 for i in inds]
    return float(np.mean(q))


class TestNPointAP:
    """11-point (VOC 2007) and 101-point (COCO) AP share one grouped authority."""

    CASES = [
        ([0.9, 0.8, 0.7, 0.5, 0.3], [True, False, True, False, True], 3),
        ([0.9, 0.5, 0.5, 0.5], [True, True, False, False], 2),
        ([0.95, 0.9, 0.6, 0.4, 0.2], [True, True, True, False, True], 10),
        ([0.9, 0.8, 0.7], [False, False, False], 3),
        # Recall lands exactly on grid values (k/100).
        ([0.9 - i / 100 for i in range(7)], [True] * 7, 100),
    ]

    @pytest.mark.parametrize(("scores", "is_tp", "gts"), CASES)
    def test_101_point_matches_cocoeval(
        self, scores: list[float], is_tp: list[bool], gts: int
    ) -> None:
        expanded = pl.LazyFrame(
            {
                COL_SCORE: scores,
                COL_IS_TP: is_tp,
                COL_WEIGHT: [1.0] * len(scores),
                "gt_mass": [float(gts)] * len(scores),
                "_g": [0] * len(scores),
            }
        )
        out = ap_by_group(expanded, group_col="_g", interpolation="101_point").collect()
        assert float(out["ap"][0]) == pytest.approx(
            _ref_coco_101(scores, is_tp, float(gts)), abs=1e-12
        )
        public = average_precision(
            _single_class_table(scores, is_tp, gts), interpolation="101_point"
        )
        assert public == pytest.approx(float(out["ap"][0]), abs=1e-12)

    def test_an_unknown_interpolation_is_refused(self) -> None:
        table = _single_class_table([0.9], [True], 1)
        with pytest.raises(ValueError, match="interpolation"):
            mean_average_precision(table, interpolation="12_point")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="interpolation"):
            average_precision(table, interpolation="12_point")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def _single_class_table(scores: list[float], is_tp: list[bool], gts: int):
    det = pl.DataFrame(
        {
            COL_IMAGE_ID: [f"i{i}" for i in range(len(scores))],
            COL_CLASS_ID: [DEFAULT_CLASS] * len(scores),
            COL_SCORE: scores,
            COL_IS_TP: is_tp,
            COL_GT_IDX: [i if t else None for i, t in enumerate(is_tp)],
            COL_IOU: [0.9 if t else None for t in is_tp],
            COL_DET_IDX: list(range(len(scores))),
        },
        schema_overrides={
            COL_GT_IDX: pl.UInt32,
            COL_DET_IDX: pl.UInt32,
            COL_IOU: pl.Float64,
        },
    )
    meta = pl.DataFrame(
        {
            COL_IMAGE_ID: ["m"],
            COL_CLASS_ID: [DEFAULT_CLASS],
            COL_N_GTS: [gts],
            COL_WEIGHT: [1.0],
            COL_GT_LABEL: [True],
        }
    )
    return DetectionTable.from_matched(det, meta)


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


class TestPrecisionRecallResultAtThreshold:
    """The result object's `precision_at` / `recall_at`, including empty filters."""

    def test_precision_and_recall_at_a_reachable_threshold(
        self, simple_detection_table: DetectionTable
    ) -> None:
        result = precision_recall_curve(simple_detection_table)
        assert 0.0 <= result.precision_at(0.5) <= 1.0
        assert 0.0 <= result.recall_at(0.5) <= 1.0

    def test_threshold_above_every_score_returns_the_empty_defaults(
        self, simple_detection_table: DetectionTable
    ) -> None:
        # No detection scores >= 2.0, so the filtered curve is empty: precision
        # defaults to 1.0 (nothing predicted, nothing wrong) and recall to 0.0.
        result = precision_recall_curve(simple_detection_table)
        assert result.precision_at(2.0) == 1.0
        assert result.recall_at(2.0) == 0.0


# --- weighting -------------------------------------------------------------------

# (image, class, score, is_tp, iou) detections and (image, class, n_gts) metadata.
_W_DETS = [
    ("i0", "car", 0.95, True, 0.90),
    ("i0", "car", 0.60, False, 0.10),
    ("i0", "ped", 0.55, True, 0.60),
    ("i1", "car", 0.85, False, 0.30),
    ("i1", "ped", 0.80, True, 0.80),
    ("i1", "ped", 0.35, True, 0.55),
    ("i2", "car", 0.75, True, 0.70),
    ("i2", "car", 0.40, True, 0.52),
    ("i2", "ped", 0.65, False, 0.20),
    ("i3", "car", 0.50, False, 0.00),
    ("i3", "ped", 0.45, True, 0.95),
]
_W_GTS = {
    ("i0", "car"): 2,
    ("i0", "ped"): 1,
    ("i1", "car"): 1,
    ("i1", "ped"): 3,
    ("i2", "car"): 2,
    ("i2", "ped"): 1,
    ("i3", "car"): 1,
    ("i3", "ped"): 1,
}
# Integer weights, so a weight of k is the image drawn k times.
_W_WEIGHTS = {"i0": 2.0, "i1": 1.0, "i2": 3.0, "i3": 1.0}


def _weighted_pr_table(
    weights: dict[str, float], *, replicate: bool = False
) -> DetectionTable:
    """``_W_DETS`` with per-image ``weights``, or each image copied ``k`` times."""
    copies = {img: (int(w) if replicate else 1) for img, w in weights.items()}

    def ids(img: str) -> list[str]:  # an image absent from `weights` is absent
        return [f"{img}#{r}" for r in range(copies.get(img, 0))]

    det_rows = [
        (uid, cls, score, tp, iou, k)
        for k, (img, cls, score, tp, iou) in enumerate(_W_DETS)
        for uid in ids(img)
    ]
    det = pl.DataFrame(
        {
            COL_IMAGE_ID: [r[0] for r in det_rows],
            COL_CLASS_ID: [r[1] for r in det_rows],
            COL_SCORE: [r[2] for r in det_rows],
            COL_IS_TP: [r[3] for r in det_rows],
            COL_GT_IDX: [0 if r[3] else None for r in det_rows],
            COL_IOU: [r[4] for r in det_rows],
            COL_DET_IDX: [r[5] for r in det_rows],
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
    meta_rows = [
        (uid, cls, n, 1.0 if replicate else weights[img])
        for (img, cls), n in _W_GTS.items()
        for uid in ids(img)
    ]
    meta = pl.DataFrame(
        {
            COL_IMAGE_ID: [r[0] for r in meta_rows],
            COL_CLASS_ID: [r[1] for r in meta_rows],
            COL_N_GTS: [r[2] for r in meta_rows],
            COL_WEIGHT: [r[3] for r in meta_rows],
            COL_GT_LABEL: [r[2] > 0 for r in meta_rows],
        }
    )
    return DetectionTable.from_matched(det, meta, matching_iou_threshold=0.5)


class TestWeightedPrecisionRecall:
    """The PR family is weighted by ``image_metadata.weight``.

    Precision is ``Σw·tp / Σw·(tp+fp)`` and recall ``Σw·tp / Σw·n_gts``, each
    detection carrying its image's weight (scikit-learn's ``sample_weight``).
    The oracle: an integer weight ``k`` equals the image drawn ``k`` times.
    """

    @pytest.fixture()
    def pair(self) -> tuple[DetectionTable, DetectionTable]:
        return (
            _weighted_pr_table(_W_WEIGHTS),
            _weighted_pr_table(_W_WEIGHTS, replicate=True),
        )

    @pytest.mark.parametrize("class_id", ["car", "ped", None])
    def test_curve_matches_replication(self, pair, class_id: str | None) -> None:
        weighted, replicated = pair
        w = precision_recall_curve(weighted, class_id=class_id).curve
        r = precision_recall_curve(replicated, class_id=class_id).curve
        assert w["score"].to_list() == r["score"].to_list()
        assert w["precision"].to_list() == pytest.approx(r["precision"].to_list())
        assert w["recall"].to_list() == pytest.approx(r["recall"].to_list())

    @pytest.mark.parametrize("interpolation", ["all_points", "11_point"])
    @pytest.mark.parametrize("class_id", ["car", "ped"])
    def test_average_precision_matches_replication(
        self, pair, class_id: str, interpolation: str
    ) -> None:
        weighted, replicated = pair
        got = average_precision(
            weighted, class_id=class_id, interpolation=interpolation
        )
        want = average_precision(
            replicated, class_id=class_id, interpolation=interpolation
        )
        assert got == pytest.approx(want, abs=1e-12)
        # ...and the weighting is not inert on this fixture.
        unit = _weighted_pr_table(dict.fromkeys(_W_WEIGHTS, 1.0))
        assert got != pytest.approx(
            average_precision(unit, class_id=class_id, interpolation=interpolation)
        )

    @pytest.mark.parametrize("interpolation", ["all_points", "11_point"])
    def test_mean_average_precision_matches_replication(
        self, pair, interpolation: str
    ) -> None:
        weighted, replicated = pair
        kw = {"iou_thresholds": [0.5, 0.75], "interpolation": interpolation}
        assert mean_average_precision(weighted, **kw) == pytest.approx(
            mean_average_precision(replicated, **kw), abs=1e-12
        )

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.9])
    def test_threshold_metrics_match_replication(self, pair, threshold: float) -> None:
        weighted, replicated = pair
        for fn in (precision_at_threshold, recall_at_threshold, f1_at_threshold):
            assert fn(weighted, threshold) == pytest.approx(
                fn(replicated, threshold), abs=1e-12
            )

    @pytest.mark.parametrize("threshold", [0.3, 0.7])
    def test_confusion_weighted_counts_match_replication(
        self, pair, threshold: float
    ) -> None:
        weighted, replicated = pair
        w = confusion_at_threshold(weighted, threshold)
        r = confusion_at_threshold(replicated, threshold)
        # Raw counts stay counts of the table's own detections...
        assert (w.tp, w.fp) != (r.tp, r.fp)
        # ...while the weighted counts and every derived rate are the oracle's.
        assert (w.weighted_tp, w.weighted_fp, w.weighted_fn) == pytest.approx(
            (r.tp, r.fp, r.fn)
        )
        assert (w.precision, w.recall, w.f1) == pytest.approx(
            (r.precision, r.recall, r.f1)
        )

    def test_unit_weights_leave_counts_and_weighted_counts_equal(self) -> None:
        unit = _weighted_pr_table(dict.fromkeys(_W_WEIGHTS, 1.0))
        c = confusion_at_threshold(unit, 0.5)
        assert (c.weighted_tp, c.weighted_fp, c.weighted_fn) == (c.tp, c.fp, c.fn)

    def test_scale_invariant(self) -> None:
        scaled = _weighted_pr_table({k: 3.7 * v for k, v in _W_WEIGHTS.items()})
        base = _weighted_pr_table(_W_WEIGHTS)
        for cls in ("car", "ped"):
            assert average_precision(scaled, class_id=cls) == pytest.approx(
                average_precision(base, class_id=cls), abs=1e-12
            )
        assert f1_at_threshold(scaled, 0.5) == pytest.approx(
            f1_at_threshold(base, 0.5), abs=1e-12
        )

    def test_zero_weight_image_is_dropped(self) -> None:
        # A weight of 0 removes the image's detections *and* its ground truths.
        zeroed = _weighted_pr_table({**_W_WEIGHTS, "i2": 0.0})
        dropped = _weighted_pr_table(
            {k: v for k, v in _W_WEIGHTS.items() if k != "i2"}, replicate=True
        )
        for cls in ("car", "ped"):
            assert average_precision(zeroed, class_id=cls) == pytest.approx(
                average_precision(dropped, class_id=cls), abs=1e-12
            )
        assert recall_at_threshold(zeroed, 0.3) == pytest.approx(
            recall_at_threshold(dropped, 0.3), abs=1e-12
        )

    def test_grouped_authority_matches_scalar_under_weights(self, pair) -> None:
        # The bootstrap's grouped AP (the CI point column) is the scalar AP.
        from polars_cv.metrics import average_precision_ci_lazy

        weighted, _ = pair
        for cls in ("car", "ped"):
            point = (
                average_precision_ci_lazy(weighted, class_id=cls, n_bootstrap=5, seed=1)
                .collect()["ap"]
                .item()
            )
            assert point == pytest.approx(
                average_precision(weighted, class_id=cls), abs=1e-12
            )
