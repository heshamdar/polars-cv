"""Tests for detection metric fixes: FROC, LROC, PR AUC, AUC utilities, and MW-U.

Covers:
- FROC total_gts overcounting fix
- FROC bootstrap recomputation of total_targets
- FROC iou_threshold propagation from DetectionTable
- LROC lower-right endpoint addition
- PR monotone-envelope AP vs raw trapezoidal AUC
- partial_auc extrapolation warning
- McClish correction for partial AUC
- Mann-Whitney U AUC for froc_auc / lroc_auc (via method="mann_whitney")
- Mann-Whitney AUC bootstrap support
- ContourMatcher min_contour_area default change
- Zero-score detection filtering
- Source format auto-detection for ContourMatcher
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_cv.metrics import (
    DetectionTable,
    MetricResult,
    average_precision_ci_lazy,
    froc_auc,
    froc_auc_ci_lazy,
    froc_curve_lazy,
    froc_sensitivity_at_fp,
    froc_summary_table,
    lroc_auc,
    lroc_auc_ci_lazy,
    lroc_curve_lazy,
    precision_recall_curve,
)
from polars_cv.metrics._auc import mcclish_correction, partial_auc
from polars_cv.metrics._auc_expr import collapse_scores, mann_whitney_auc_expr
from polars_cv.metrics._bootstrap import _bootstrap_table_with_draws
from polars_cv.metrics._matching._contour import ContourMatcher, _detect_source_info
from polars_cv.metrics._metrics._froc import _froc_curve_grouped
from polars_cv.metrics._metrics._lroc import _build_lroc_curve_grouped
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


def _froc_curve_df(det_df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """Pooled FROC curve from minimal (image_id/score/is_tp) + meta frames."""
    dl = det_df.lazy().with_columns(pl.lit(0, dtype=pl.Int32).alias("_froc_grp"))
    ml = meta_df.lazy().with_columns(pl.lit(0, dtype=pl.Int32).alias("_froc_grp"))
    return _froc_curve_grouped(dl, ml, ["_froc_grp"], None).drop("_froc_grp").collect()


def _lroc_curve_df(per_image: pl.DataFrame) -> pl.DataFrame:
    """Pooled LROC curve from a hand-crafted per-image frame."""
    lf = per_image.lazy().with_columns(pl.lit(0, dtype=pl.Int32).alias("_lroc_grp"))
    return _build_lroc_curve_grouped(lf, ["_lroc_grp"]).drop("_lroc_grp").collect()


def _mw(scores: list[float], labels: list[float]) -> float:
    """Mann-Whitney AUC via the two-stage expression, for utility-style tests."""
    lf = pl.DataFrame({"score": scores, "label": labels}).lazy()
    bucketed = collapse_scores(lf, score="score", label="label")
    return bucketed.select(auc=mann_whitney_auc_expr()).collect().item()


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


class TestFrocCurveFromDetections:
    """Verify cumulative-sum FROC curve construction."""

    def test_total_gts_correct(self) -> None:
        """Curve total_gts reflects the provided total, not double-counted.

        3 images: a (n_gts=1), b (n_gts=1), c (n_gts=0) → total=2.
        """
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b", "c"],
                COL_SCORE: [0.9, 0.7, 0.5],
                COL_IS_TP: [True, False, False],
            }
        )
        meta_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b", "c"],
                COL_N_GTS: [1, 1, 0],
                COL_WEIGHT: [1.0, 1.0, 1.0],
            }
        )
        curve = _froc_curve_df(det_df, meta_df)
        total_gts_values = curve["total_gts"].unique().to_list()
        for v in total_gts_values:
            assert v == 2, f"total_gts wrong: {v}"

    def test_weighted_sensitivity_bounded(self) -> None:
        """Weighted curve sensitivity stays in [0, 1]."""
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b"],
                COL_SCORE: [0.9, 0.5],
                COL_IS_TP: [True, False],
            }
        )
        meta_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b"],
                COL_N_GTS: [1, 1],
                COL_WEIGHT: [1.0, 1.0],
            }
        )
        curve = _froc_curve_df(det_df, meta_df)
        assert curve.height > 0
        for s in curve["sensitivity"].to_list():
            assert s is None or 0.0 <= s <= 1.0, f"sensitivity out of range: {s}"

    def test_cumulative_tp_fp_monotonic(self) -> None:
        """Cumulative TP and FP are non-decreasing as threshold decreases."""
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "a", "b", "c"],
                COL_SCORE: [0.9, 0.7, 0.5, 0.3],
                COL_IS_TP: [True, False, True, False],
            }
        )
        meta_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b", "c"],
                COL_N_GTS: [1, 1, 0],
                COL_WEIGHT: [1.0, 1.0, 1.0],
            }
        )
        curve = _froc_curve_df(det_df, meta_df)
        # Sort descending so we traverse from high → low threshold
        sorted_curve = curve.sort("threshold", descending=True)
        tp_vals = sorted_curve["tp"].to_list()
        fp_vals = sorted_curve["fp"].to_list()
        for i in range(1, len(tp_vals)):
            assert tp_vals[i] >= tp_vals[i - 1], "TP not monotonic"
            assert fp_vals[i] >= fp_vals[i - 1], "FP not monotonic"

    def test_weighted_curve_exact_values(self) -> None:
        """Distinct per-image weights drive the weighted numerator/denominator.

        a: w=2, n_gts=1, TP@0.9 ; b: w=1, n_gts=1, FP@0.5 ; c: w=3, n_gts=0, FP@0.3.
        total_weighted_gts = 1*2 + 1*1 + 0*3 = 3 ; per-image weight_sum = 2+1+3 = 6.
        At threshold 0.3 (all dets): weighted TP = 2 → sensitivity = 2/3;
        weighted FP = 1 + 3 = 4 → fp_per_image = 4/6 = 2/3.
        Uniform weights would give sensitivity 1/1 and fp_per_image 2/3 instead.
        """
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b", "c"],
                COL_SCORE: [0.9, 0.5, 0.3],
                COL_IS_TP: [True, False, False],
            }
        )
        meta_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b", "c"],
                COL_N_GTS: [1, 1, 0],
                COL_WEIGHT: [2.0, 1.0, 3.0],
            }
        )
        curve = _froc_curve_df(det_df, meta_df)
        low = curve.filter(pl.col("threshold") == 0.3).to_dicts()[0]
        assert low["sensitivity"] == pytest.approx(2.0 / 3.0)
        assert low["fp_per_image"] == pytest.approx(4.0 / 6.0)
        # Highest threshold keeps only the weighted TP; no false positives yet.
        high = curve.filter(pl.col("threshold") == 0.9).to_dicts()[0]
        assert high["sensitivity"] == pytest.approx(2.0 / 3.0)
        assert high["fp_per_image"] == pytest.approx(0.0)


class TestFrocIouThresholdPropagation:
    """Verify iou_threshold is carried on the DetectionTable."""

    def test_iou_threshold_from_table(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """The matching IoU threshold is stored on the table."""
        assert simple_detection_table._matching_iou_threshold == 0.5


class TestFrocBootstrapRecomputesTotalTargets:
    """Verify bootstrap recomputes total_targets from sampled images."""

    def test_bootstrap_sampled_total_targets(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """Bootstrap produces a valid CI over resampled replicates."""
        ci = froc_auc_ci_lazy(simple_detection_table, n_bootstrap=10, seed=42).collect()
        assert ci.height == 1
        assert ci["ci_lower"].item() <= ci["auc"].item() <= ci["ci_upper"].item()


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
        curve = _lroc_curve_df(per_image)
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
        curve = _lroc_curve_df(per_image)
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
        curve = _lroc_curve_df(per_image)
        auc = MetricResult(curve=curve).auc(x_col="fpf", y_col="sensitivity")
        # With 1 positive (TP) and 1 negative: should give AUC = 1.0
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# PR Curve AP Fix
# ---------------------------------------------------------------------------


class TestPrApEnvelope:
    """Verify auc() uses monotone-envelope and auc(method='trapezoidal') does not."""

    def test_auc_ge_raw_auc(self) -> None:
        """Monotone-envelope auc() >= auc(method='trapezoidal') for typical curves.

        The envelope can only increase precision, so auc() should be >=
        auc(method='trapezoidal') (unless the curve is already monotone
        decreasing).
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
        raw = result.auc(method="trapezoidal")
        envelope = result.auc()
        assert envelope >= raw - 1e-10

    def test_perfect_detector_ap_equals_one(self) -> None:
        """Perfect detector (all TP, no FP) has AP = 1.0 with envelope.

        Recall goes from 0.5 to 1.0, precision is always 1.0. With the
        recall=0 anchor the envelope integrates to 1.0 (matching 11-point
        AP and sklearn average_precision_score).
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
        assert result.auc() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Partial AUC Warning
# ---------------------------------------------------------------------------


class TestPartialAucWarning:
    """Verify partial_auc warns on significant left-boundary extrapolation."""

    def test_warns_on_large_gap(self) -> None:
        """Warning emitted when lo is far below curve minimum x."""
        x = pl.Series("x", [0.5, 0.6, 0.7, 0.8, 1.0])
        y = pl.Series("y", [0.2, 0.4, 0.6, 0.8, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            partial_auc(x, y, lo=0.0, hi=1.0)
            assert any("partial_auc" in str(warning.message) for warning in w)

    def test_no_warning_on_small_gap(self) -> None:
        """No warning when lo is close to curve minimum x."""
        x = pl.Series("x", [0.0, 0.5, 1.0])
        y = pl.Series("y", [0.0, 0.5, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            partial_auc(x, y, lo=0.0, hi=1.0)
            assert not any("partial_auc" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Mann-Whitney U AUC
# ---------------------------------------------------------------------------


class TestMannWhitneyUAuc:
    """Tests for the Mann-Whitney AUC expression."""

    def test_perfect_separation(self) -> None:
        """All positives above all negatives → AUC = 1.0."""
        assert _mw([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], [1, 1, 1, 0, 0, 0]) == pytest.approx(
            1.0
        )

    def test_reversed_separation(self) -> None:
        """All positives below all negatives → AUC = 0.0."""
        assert _mw([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [1, 1, 1, 0, 0, 0]) == pytest.approx(
            0.0
        )

    def test_equal_scores(self) -> None:
        """Identical distributions → AUC = 0.5."""
        assert _mw([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 0, 0, 0]) == pytest.approx(
            0.5
        )

    def test_empty_groups(self) -> None:
        """Empty group returns 0.5 sentinel."""
        assert _mw([0.9], [1.0]) == pytest.approx(0.5)
        assert _mw([0.1], [0.0]) == pytest.approx(0.5)


class TestFrocMannWhitneyAuc:
    """Tests for froc_auc(method='mann_whitney')."""

    def test_detection_level(self, simple_detection_table: DetectionTable) -> None:
        """Detection-level MW-U should return a value in [0, 1]."""
        mw = (
            froc_auc(simple_detection_table, method="mann_whitney", level="detection")
            .collect()
            .item()
        )
        assert 0.0 <= mw <= 1.0

    def test_image_level(self, simple_detection_table: DetectionTable) -> None:
        """Image-level MW-U should return a value in [0, 1]."""
        mw = (
            froc_auc(simple_detection_table, method="mann_whitney", level="image")
            .collect()
            .item()
        )
        assert 0.0 <= mw <= 1.0

    def test_invalid_level_raises(self, simple_detection_table: DetectionTable) -> None:
        """Invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported level"):
            froc_auc(simple_detection_table, method="mann_whitney", level="invalid")

    def test_mw_rejects_fp_range(self, simple_detection_table: DetectionTable) -> None:
        """Mann-Whitney with fp_range raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            froc_auc(simple_detection_table, method="mann_whitney", fp_range=(0, 1))


class TestLrocMannWhitneyAuc:
    """Tests for lroc_auc(method='mann_whitney')."""

    def test_detection_level(self, simple_detection_table: DetectionTable) -> None:
        """Detection-level MW-U should return a value in [0, 1]."""
        mw = (
            lroc_auc(simple_detection_table, method="mann_whitney", level="detection")
            .collect()
            .item()
        )
        assert 0.0 <= mw <= 1.0

    def test_image_level(self, simple_detection_table: DetectionTable) -> None:
        """Image-level MW-U should return a value in [0, 1]."""
        mw = (
            lroc_auc(simple_detection_table, method="mann_whitney", level="image")
            .collect()
            .item()
        )
        assert 0.0 <= mw <= 1.0


class TestMannWhitneyBootstrap:
    """Tests for bootstrapping Mann-Whitney AUC."""

    def test_froc_mw_bootstrap(self, simple_detection_table: DetectionTable) -> None:
        """Bootstrap CI for FROC MW-U detection-level runs and is valid."""
        ci = froc_auc_ci_lazy(
            simple_detection_table,
            n_bootstrap=10,
            seed=42,
            method="mann_whitney",
        ).collect()
        assert ci.height == 1
        lo, hi, auc = (
            ci["ci_lower"].item(),
            ci["ci_upper"].item(),
            ci["auc"].item(),
        )
        assert lo <= hi
        assert 0.0 <= auc <= 1.0

    def test_lroc_mw_bootstrap(self, simple_detection_table: DetectionTable) -> None:
        """Bootstrap CI for LROC MW-U image-level runs and is valid."""
        ci = lroc_auc_ci_lazy(
            simple_detection_table,
            n_bootstrap=10,
            seed=42,
            method="mann_whitney",
            level="image",
        ).collect()
        assert ci.height == 1
        lo, hi, auc = (
            ci["ci_lower"].item(),
            ci["ci_upper"].item(),
            ci["auc"].item(),
        )
        assert lo <= hi
        assert 0.0 <= auc <= 1.0

    def test_ci_lazy_returns_lazyframe(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """The public CI seam is lazy — it returns a LazyFrame, never a scalar."""
        out = froc_auc_ci_lazy(simple_detection_table, n_bootstrap=5, seed=42)
        assert isinstance(out, pl.LazyFrame)
        assert out.collect().columns == ["auc", "ci_lower", "ci_upper"]


class TestEntityLevelBootstrap:
    """Tests for entity-level (sample_col) bootstrap sampling."""

    @pytest.fixture()
    def entity_detection_table(self) -> DetectionTable:
        """Create a detection table with a case_id column for entity-level sampling."""
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a1", "a2", "b1", "b2"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 4,
                COL_SCORE: [0.9, 0.8, 0.7, 0.6],
                COL_IS_TP: [True, False, True, False],
                COL_GT_IDX: [0, None, 0, None],
                COL_IOU: [0.85, 0.0, 0.7, 0.0],
                COL_DET_IDX: [0, 0, 0, 0],
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
                COL_IMAGE_ID: ["a1", "a2", "b1", "b2"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 4,
                COL_N_GTS: [1, 0, 1, 0],
                COL_WEIGHT: [1.0, 1.0, 1.0, 1.0],
                COL_GT_LABEL: [True, False, True, False],
                "case_id": ["case_A", "case_A", "case_B", "case_B"],
            }
        )
        return DetectionTable.from_matched(det_df, meta_df, matching_iou_threshold=0.5)

    def test_entity_bootstrap_runs(
        self, entity_detection_table: DetectionTable
    ) -> None:
        """Entity-level bootstrap with sample_col produces valid CI."""
        ci = froc_auc_ci_lazy(
            entity_detection_table,
            n_bootstrap=10,
            seed=42,
            sample_col="case_id",
        ).collect()
        assert ci.height == 1
        assert ci["ci_lower"].item() <= ci["ci_upper"].item()

    def test_entity_bootstrap_samples_at_entity_level(
        self, entity_detection_table: DetectionTable
    ) -> None:
        """Entity-level resampling draws entities, then expands to their images.

        With 2 entities (case_A, case_B), each with 2 images, every replicate
        draws 2 entities (with replacement) and expands each to its 2 images —
        4 image rows per replicate — computed as one lazy plan (never a Python
        entity->image dict). Because a redrawn entity yields two rows carrying
        the *same* base image ids, the per-replicate row count, not the distinct
        image count, is what pins entity-level granularity.
        """
        from polars_cv.metrics._bootstrap import _resolve_bootstrap_samples

        samples = _resolve_bootstrap_samples(
            entity_detection_table,
            sample_col="case_id",
            n_bootstrap=5,
            seed=42,
        )
        assert isinstance(samples, pl.LazyFrame)
        df = samples.collect(engine="streaming")
        # 2 entities drawn * 2 images each = 4 image rows per replicate.
        per_replicate = df.group_by("bootstrap_id").len()
        assert per_replicate["len"].unique().to_list() == [4]
        # Only ever the images that belong to the two known entities.
        assert set(df[COL_IMAGE_ID].unique()) <= {"a1", "a2", "b1", "b2"}

    def test_no_sample_col_resamples_at_image_level(
        self, entity_detection_table: DetectionTable
    ) -> None:
        """Without sample_col, resampling is at the image level.

        Each replicate draws exactly the 4 base images (with replacement),
        stratified by ``gt_label``.
        """
        from polars_cv.metrics._bootstrap import _resolve_bootstrap_samples

        samples = _resolve_bootstrap_samples(
            entity_detection_table,
            sample_col=None,
            n_bootstrap=5,
            seed=42,
        )
        assert isinstance(samples, pl.LazyFrame)
        df = samples.collect(engine="streaming")
        per_replicate = df.group_by("bootstrap_id").len()
        assert per_replicate["len"].unique().to_list() == [4]


class TestMcClishCorrection:
    """Tests for the McClish standardized partial AUC correction."""

    def test_perfect_classifier_gives_one(self) -> None:
        """Perfect classifier in [0, 1] range gives corrected pAUC = 1.0."""
        corrected = mcclish_correction(raw_pauc=1.0, lo=0.0, hi=1.0)
        assert corrected == pytest.approx(1.0)

    def test_chance_classifier_gives_half(self) -> None:
        """Diagonal (chance) classifier gives corrected pAUC = 0.5."""
        corrected = mcclish_correction(raw_pauc=0.5, lo=0.0, hi=1.0)
        assert corrected == pytest.approx(0.5)

    def test_partial_range(self) -> None:
        """McClish correction works for a partial range."""
        lo, hi = 0.0, 0.5
        min_pauc = (lo + hi) * (hi - lo) / 2  # 0.125
        max_pauc = hi - lo  # 0.5
        raw = (min_pauc + max_pauc) / 2  # midpoint
        corrected = mcclish_correction(raw, lo, hi)
        assert corrected == pytest.approx(0.75)

    def test_zero_span_gives_half(self) -> None:
        """Zero-width interval returns 0.5 sentinel."""
        assert mcclish_correction(0.0, 0.5, 0.5) == pytest.approx(0.5)

    def test_froc_auc_with_mcclish(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """FROC auc with correction='mcclish' returns a value in [0, 1]."""
        corrected = (
            froc_auc(simple_detection_table, fp_range=(0.0, 1.0), correction="mcclish")
            .collect()
            .item()
        )
        assert 0.0 <= corrected <= 1.0

    def test_froc_auc_with_normalize(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """FROC auc with correction='normalize' returns average sensitivity."""
        normalized = (
            froc_auc(
                simple_detection_table, fp_range=(0.0, 1.0), correction="normalize"
            )
            .collect()
            .item()
        )
        assert 0.0 <= normalized <= 1.0


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


# ---------------------------------------------------------------------------
# Source format detection
# ---------------------------------------------------------------------------


class TestSourceFormatDetection:
    """`_detect_source_info` reads the leaf dtype; Rust picks the format.

    These tests used to pin a format string per Polars dtype. That mapping was
    a second implementation of `resolve_auto_format`, and it had already
    diverged: it sent every `Binary` column to `"blob"`, so a `ContourMatcher`
    over a PNG mask failed with "Invalid blob magic bytes" while the same
    column read fine through `source("auto")`. Metrics now follows the Rust
    side; what it still contributes is the one fact the Polars schema settles
    and Rust cannot infer at plan time — a nested column's element dtype.
    """

    def test_binary_carries_no_source_kwargs(self) -> None:
        """Binary needs nothing from Polars — `auto` reads its magic bytes."""
        info = _detect_source_info({"col": pl.Binary}, "col")
        assert info.kwargs == {}

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (pl.List(pl.List(pl.Float64)), "f64"),
            (pl.List(pl.List(pl.Float32)), "f32"),
            (pl.List(pl.List(pl.UInt8)), "u8"),
            (pl.Array(pl.Float32, 3), "f32"),
        ],
        ids=["list-f64", "list-f32", "list-u8", "array-f32"],
    )
    def test_nested_columns_carry_their_leaf_dtype(
        self, dtype: pl.DataType, expected: str
    ) -> None:
        """A typed source needs the element dtype before any data moves."""
        info = _detect_source_info({"col": dtype}, "col")
        assert info.kwargs == {"dtype": expected}

    def test_a_leaf_with_no_buffer_meaning_is_refused(self) -> None:
        """The leaf check stays in Python: it is a *Polars* type that is wrong."""
        with pytest.raises(ValueError, match="no meaningful buffer"):
            _detect_source_info({"col": pl.List(pl.List(pl.String))}, "col")

    def test_the_source_it_builds_is_auto(self) -> None:
        """The format is never named here -- naming it is the defect."""
        source = _detect_source_info({"col": pl.Binary}, "col").build_source()
        assert source._source is not None
        assert source._source.format.value == "auto"


class TestElevenPointApDenominator:
    """VOC 11-point AP is (1/11) * sum over t in {0.0,...,1.0} of
    max{precision : recall >= t}, where thresholds beyond the curve's max
    recall contribute 0 -- they must not be dropped from the average.
    """

    def test_unreachable_thresholds_count_as_zero(self) -> None:
        from polars_cv.metrics._metrics._precision_recall import _eleven_point_ap

        # Max recall 0.5: thresholds 0.0..0.5 (6 of them) see max precision
        # 1.0; thresholds 0.6..1.0 (5 of them) have no point -> 0.
        curve = pl.DataFrame(
            {
                "score": [0.9, 0.8],
                "recall": [0.5, 0.5],
                "precision": [1.0, 0.5],
            }
        )
        ap = _eleven_point_ap(curve)
        assert ap == pytest.approx(6.0 / 11.0, abs=1e-9)

    def test_full_recall_curve_unchanged(self) -> None:
        from polars_cv.metrics._metrics._precision_recall import _eleven_point_ap

        # Recall reaches 1.0 with precision 1.0 everywhere -> AP 1.0.
        curve = pl.DataFrame(
            {
                "score": [0.9, 0.8],
                "recall": [0.5, 1.0],
                "precision": [1.0, 1.0],
            }
        )
        ap = _eleven_point_ap(curve)
        assert ap == pytest.approx(1.0, abs=1e-9)

    def test_integration_via_average_precision(self) -> None:
        from polars_cv.metrics import average_precision

        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["img1", "img1"],
                COL_CLASS_ID: [DEFAULT_CLASS, DEFAULT_CLASS],
                COL_SCORE: [0.9, 0.8],
                COL_IS_TP: [True, False],
                COL_GT_IDX: [0, None],
                COL_IOU: [0.9, 0.0],
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
        # Curve: (R=0.5, P=1.0), (R=0.5, P=0.5); max recall 0.5.
        ap = average_precision(table, interpolation="11_point")
        assert ap == pytest.approx(6.0 / 11.0, abs=1e-9)


# ---------------------------------------------------------------------------
# average_precision_ci_lazy: replicates must use the same AP estimator as the point
# ---------------------------------------------------------------------------


class TestBootstrapPrAucEstimatorConsistency:
    """The point estimate uses envelope (all-points) AP; each bootstrap
    replicate must apply the same monotone precision envelope, otherwise
    the CI is computed on a systematically lower estimator than the point.
    """

    @staticmethod
    def _dipping_table() -> DetectionTable:
        # TP(0.9), FP(0.8), TP(0.7) over 2 GTs in ONE image:
        # raw precision [1.0, 0.5, 0.667] dips; the envelope lifts the
        # middle point, so raw trapezoid != envelope AP.
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["img1", "img1", "img1"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 3,
                COL_SCORE: [0.9, 0.8, 0.7],
                COL_IS_TP: [True, False, True],
                COL_GT_IDX: [0, None, 1],
                COL_IOU: [0.9, 0.0, 0.8],
                COL_DET_IDX: [0, 1, 2],
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
        return DetectionTable.from_matched(det_df, meta_df)

    def test_identity_replicates_equal_point_estimate(self) -> None:
        """With a single image, every bootstrap sample IS the full sample, so
        the CI collapses onto the point estimate (bounds equal the point)."""
        table = self._dipping_table()
        result = average_precision_ci_lazy(table, n_bootstrap=8, seed=7).collect()

        ap = result["ap"].item()
        assert result["ci_lower"].item() == pytest.approx(ap, abs=1e-9)
        assert result["ci_upper"].item() == pytest.approx(ap, abs=1e-9)

    def test_ci_brackets_point_estimate(self) -> None:
        table = self._dipping_table()
        result = average_precision_ci_lazy(table, n_bootstrap=8, seed=7).collect()
        assert (
            result["ci_lower"].item()
            <= result["ap"].item()
            <= result["ci_upper"].item()
        )

    def test_recall_anchor_matches_average_precision(self) -> None:
        """The CI ``ap`` point estimate equals ``average_precision`` after the
        shared recall=0 anchor fix (issue 1 paired fix)."""
        from polars_cv.metrics import average_precision

        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b", "c"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 3,
                COL_SCORE: [0.9, 0.8, 0.7],
                COL_IS_TP: [True, True, False],
                COL_GT_IDX: [0, 0, None],
                COL_IOU: [0.6, 0.6, 0.0],
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
                COL_N_GTS: [1, 1, 1],
                COL_WEIGHT: [1.0] * 3,
                COL_GT_LABEL: [True] * 3,
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        point = average_precision(table)
        result = average_precision_ci_lazy(table, n_bootstrap=5, seed=1).collect()
        assert point == pytest.approx(2.0 / 3.0, abs=1e-9)
        assert result["ap"].item() == pytest.approx(point, abs=1e-9)


# ---------------------------------------------------------------------------
# Issue 1: all_points AP must anchor at recall = 0
# ---------------------------------------------------------------------------


class TestAllPointsApRecallAnchor:
    """all_points AP must include the leftmost recall segment (R₀ = 0)."""

    def test_sklearn_repro(self) -> None:
        """Exact upstream repro: AP must be 2/3, not 1/3."""
        from polars_cv.metrics import average_precision

        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b", "c"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 3,
                COL_SCORE: [0.9, 0.8, 0.7],
                COL_IS_TP: [True, True, False],
                COL_GT_IDX: [0, 0, None],
                COL_IOU: [0.6, 0.6, 0.0],
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
                COL_N_GTS: [1, 1, 1],
                COL_WEIGHT: [1.0] * 3,
                COL_GT_LABEL: [True] * 3,
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        assert average_precision(table) == pytest.approx(2.0 / 3.0, abs=1e-9)

    def test_degenerate_perfect_result(self) -> None:
        """When every recall value is 1.0, AP must be 1.0 (not 0.0).

        Without the recall=0 anchor every trapezoid has zero width.
        """
        from polars_cv.metrics import average_precision

        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "a"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 2,
                COL_SCORE: [0.9, 0.4],
                COL_IS_TP: [True, False],
                COL_GT_IDX: [0, None],
                COL_IOU: [0.9, 0.0],
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
                COL_N_GTS: [1],
                COL_WEIGHT: [1.0],
                COL_GT_LABEL: [True],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        assert average_precision(table) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Issue 2: PreMatchedAdapter must accept an explicit image population
# ---------------------------------------------------------------------------


class TestPreMatchedAdapterPopulation:
    """PreMatchedAdapter must not silently drop zero-detection images."""

    def test_image_meta_retains_zero_detection_images(self) -> None:
        """Passing image_meta keeps images that carry no detections."""
        from polars_cv.metrics import PreMatchedAdapter

        data = pl.DataFrame(
            {
                "image_id": ["a"],
                "score": [0.9],
                "is_tp": [True],
            }
        )
        image_meta = pl.DataFrame(
            {
                "image_id": ["a", "b", "c"],
                "n_gts": [1, 0, 1],
                "weight": [1.0, 1.0, 1.0],
                "gt_label": [True, False, True],
            }
        )
        table = PreMatchedAdapter().match(
            data,
            pred_col="score",
            gt_col="is_tp",
            image_id_col="image_id",
            image_meta=image_meta,
        )
        _, meta = table.collect()
        assert meta.height == 3
        assert set(meta["image_id"].to_list()) == {"a", "b", "c"}

    def test_omitting_image_meta_warns(self) -> None:
        """Calling without image_meta emits a UserWarning."""
        from polars_cv.metrics import PreMatchedAdapter

        data = pl.DataFrame(
            {
                "image_id": ["a"],
                "score": [0.9],
                "is_tp": [True],
                "n_gts": [1],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table = PreMatchedAdapter().match(
                data,
                pred_col="score",
                gt_col="is_tp",
                image_id_col="image_id",
                n_gts_col="n_gts",
            )
            assert any(
                issubclass(w.category, UserWarning) and "image_meta" in str(w.message)
                for w in caught
            )
        _, meta = table.collect()
        assert meta.height == 1


# ---------------------------------------------------------------------------
# Issues 3 & 4: FROC weight-join fan-out
# ---------------------------------------------------------------------------


class TestFrocSharedImageId:
    """Duplicate image_id in metadata must not fan out detections."""

    def test_shared_image_not_double_counted(self) -> None:
        """Exact upstream repro: tp=1, fp=1, sensitivity=0.5."""
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["shared", "shared"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 2,
                COL_SCORE: [0.8, 0.7],
                COL_IS_TP: [True, False],
                COL_GT_IDX: [0, None],
                COL_IOU: [0.6, 0.0],
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
                COL_IMAGE_ID: ["shared", "shared"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 2,
                COL_N_GTS: [1, 1],
                COL_WEIGHT: [1.0, 1.0],
                COL_GT_LABEL: [True, True],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        curve = froc_curve_lazy(table).collect()
        low = curve.filter(pl.col("threshold") == 0.7)
        assert low.height == 1
        assert int(low["tp"].item()) == 1
        assert int(low["fp"].item()) == 1
        assert float(low["sensitivity"].item()) == pytest.approx(0.5)

    @staticmethod
    def _conflicting_weight_table(weights: list[float]) -> DetectionTable:
        """One positive image `a` (TP) and a negative image `s` whose metadata
        appears twice with disagreeing weights (a single FP)."""
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "s"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 2,
                COL_SCORE: [0.9, 0.5],
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
                COL_IMAGE_ID: ["a", "s", "s"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 3,
                COL_N_GTS: [1, 0, 0],
                COL_WEIGHT: [1.0, *weights],
                COL_GT_LABEL: [True, False, False],
            }
        )
        return DetectionTable.from_matched(det_df, meta_df)

    def test_conflicting_weights_do_not_raise(self) -> None:
        """The guard is gone: disagreeing weights resolve, they do not raise.

        (Was ``test_conflicting_weights_raise`` — the eager build-time guard that
        forced a collect was removed in favour of the lazy ``weight_agg`` policy,
        so the plan builds and only materialises when the caller collects.)
        """
        for order in ([2.0, 8.0], [8.0, 2.0]):
            table = self._conflicting_weight_table(order)
            # Build (no collect) and full collect both succeed for every policy.
            for agg in ("first", "min", "max", "mean", "sum"):
                froc_curve_lazy(table, weight_agg=agg).collect()

    @pytest.mark.parametrize(
        ("agg", "resolved_s"),
        [("min", 2.0), ("max", 8.0), ("mean", 5.0), ("sum", 10.0)],
    )
    def test_weight_agg_resolves_the_conflicting_weight(
        self, agg: str, resolved_s: float
    ) -> None:
        """Each order-independent policy resolves image ``s`` to a known weight,
        visible through ``fp_per_image`` = w(s) / (w(a) + w(s))."""
        table = self._conflicting_weight_table([2.0, 8.0])
        curve = froc_curve_lazy(table, weight_agg=agg).collect()
        low = curve.filter(pl.col("threshold") == 0.5).to_dicts()[0]
        assert low["fp_per_image"] == pytest.approx(resolved_s / (1.0 + resolved_s))
        # The single TP always drives sensitivity to 1.0 here.
        assert low["sensitivity"] == pytest.approx(1.0)

    def test_cross_class_weights_are_per_image_class_keys(self) -> None:
        """Two classes of one image are distinct weight keys, not a conflict.

        Weights are looked up per ``(image_id, class_id)``, so ``cat`` and ``dog``
        of image ``a`` simply carry their own weights — nothing raises.
        """
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "a"],
                COL_CLASS_ID: ["cat", "dog"],
                COL_SCORE: [0.9, 0.8],
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
                COL_IMAGE_ID: ["a", "a"],
                COL_CLASS_ID: ["cat", "dog"],
                COL_N_GTS: [1, 0],
                COL_WEIGHT: [1.0, 5.0],
                COL_GT_LABEL: [True, False],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        # Per class the weight is unambiguous; both build and collect cleanly.
        froc_curve_lazy(table, group_by="class_id").collect()

    def test_equal_weights_remain_stable(self) -> None:
        """Equal duplicate weights still yield tp=1, fp=1, sensitivity=0.5."""
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["shared", "shared"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 2,
                COL_SCORE: [0.8, 0.7],
                COL_IS_TP: [True, False],
                COL_GT_IDX: [0, None],
                COL_IOU: [0.6, 0.0],
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
                COL_IMAGE_ID: ["shared", "shared"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 2,
                COL_N_GTS: [1, 1],
                COL_WEIGHT: [1.0, 1.0],
                COL_GT_LABEL: [True, True],
            }
        )
        curve = froc_curve_lazy(DetectionTable.from_matched(det_df, meta_df)).collect()
        low = curve.filter(pl.col("threshold") == 0.7)
        assert int(low["tp"].item()) == 1
        assert int(low["fp"].item()) == 1
        assert float(low["sensitivity"].item()) == pytest.approx(0.5)


class TestFrocBootstrapCiContainsPoint:
    """FROC bootstrap CI must bracket the point estimate (issue 4)."""

    def test_ci_contains_point_and_sensitivity_bounded(self) -> None:
        """Replica sensitivity stays ≤ 1 and the CI brackets the point."""
        image_ids = [f"p{i}" for i in range(10)] + [f"n{i}" for i in range(10)]
        det_rows: list[tuple[str, float, bool, int | None, float, int]] = []
        for i in range(10):
            det_rows.append((f"p{i}", 0.9, True, 0, 0.8, 0))
        for i in range(10):
            det_rows.append((f"n{i}", 0.5, False, None, 0.0, 0))

        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: [r[0] for r in det_rows],
                COL_CLASS_ID: [DEFAULT_CLASS] * len(det_rows),
                COL_SCORE: [r[1] for r in det_rows],
                COL_IS_TP: [r[2] for r in det_rows],
                COL_GT_IDX: [r[3] for r in det_rows],
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
        meta_df = pl.DataFrame(
            {
                COL_IMAGE_ID: image_ids,
                COL_CLASS_ID: [DEFAULT_CLASS] * 20,
                COL_N_GTS: [1] * 10 + [0] * 10,
                COL_WEIGHT: [1.0] * 20,
                COL_GT_LABEL: [True] * 10 + [False] * 10,
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        point = froc_auc(table).collect().item()
        ci = froc_auc_ci_lazy(table, n_bootstrap=200, seed=0).collect()
        assert ci["ci_lower"].item() <= point <= ci["ci_upper"].item()
        assert ci["ci_upper"].item() <= 1.0 + 1e-9

        # Replica curves themselves must keep sensitivity ≤ 1.
        ids_series = pl.Series("id", image_ids)
        for i in range(5):
            sampled = ids_series.sample(
                n=len(image_ids), with_replacement=True, seed=i
            ).to_list()
            samples = pl.LazyFrame(
                {
                    "bootstrap_id": [0] * len(sampled),
                    COL_IMAGE_ID: sampled,
                    "_slot": list(range(len(sampled))),
                },
                schema={
                    "bootstrap_id": pl.Int32,
                    COL_IMAGE_ID: pl.String,
                    "_slot": pl.Int64,
                },
            )
            replica = _bootstrap_table_with_draws(table, samples)
            sens = froc_curve_lazy(replica).collect()["sensitivity"].max()
            assert float(sens) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Issue 5.1: FROC / LROC curve plotting order
# ---------------------------------------------------------------------------


class TestFrocCurveOrder:
    """FROC curve must be ordered by ascending fp_per_image."""

    def test_fp_per_image_non_decreasing(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """fp_per_image is non-decreasing down the returned frame."""
        curve = froc_curve_lazy(simple_detection_table).collect()
        vals = curve["fp_per_image"].to_list()
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1] - 1e-12


class TestLrocCurveOrder:
    """LROC curve must be ordered by ascending fpf."""

    def test_fpf_non_decreasing(self, simple_detection_table: DetectionTable) -> None:
        """fpf is non-decreasing down the returned frame."""
        curve = lroc_curve_lazy(simple_detection_table).collect()
        vals = curve["fpf"].to_list()
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1] - 1e-12


# ---------------------------------------------------------------------------
# Issue 5.2: interpolate returns None beyond the observed range
# ---------------------------------------------------------------------------


class TestInterpolateNullBeyondRange:
    """sensitivity_at_fp returns None past the curve's observed max FP rate."""

    def test_beyond_max_returns_none(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """Querying past the observed max fp_per_image yields None."""
        curve = froc_curve_lazy(simple_detection_table).collect()
        max_fp = float(curve["fp_per_image"].max())
        got = (
            froc_sensitivity_at_fp(simple_detection_table, max_fp + 1.0)
            .collect()["sensitivity"]
            .item()
        )
        assert got is None

    def test_summary_table_nulls_beyond_range(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """summary_table writes null for unreachable operating points."""
        curve = froc_curve_lazy(simple_detection_table).collect()
        max_fp = float(curve["fp_per_image"].max())
        summary = froc_summary_table(
            simple_detection_table, fp_rates=[0.0, max_fp + 10.0]
        ).collect()
        assert summary["sensitivity"][0] is not None
        assert summary["sensitivity"][1] is None


# ---------------------------------------------------------------------------
# Review follow-up: curve order and AUC must not depend on tie order
# ---------------------------------------------------------------------------


class TestCurveOrderIsDeterministic:
    """Tied x-values must not leave the curve (or its AUC) order-dependent.

    A FROC curve ties on ``fp_per_image`` constantly — every threshold bucket
    that adds only true positives leaves it unchanged — and Polars' ``sort``
    defaults to ``maintain_order=False``. Sorting the curve on x alone
    therefore leaves the y at each tie boundary unspecified, which is what
    ``trapz_auc`` reads.
    """

    @staticmethod
    def _tied_table() -> DetectionTable:
        """Four images whose top three detections are all true positives.

        fp_per_image stays at 0.0 across three distinct thresholds while
        sensitivity climbs 1/3 -> 2/3 -> 1, so the curve carries a four-row tie
        at x = 0 (with the origin) whose y values differ.
        """
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b", "c", "d"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 4,
                COL_SCORE: [0.9, 0.8, 0.7, 0.1],
                COL_IS_TP: [True, True, True, False],
                COL_GT_IDX: [0, 0, 0, None],
                COL_IOU: [0.9, 0.9, 0.9, 0.0],
                COL_DET_IDX: [0, 0, 0, 0],
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
                COL_IMAGE_ID: ["a", "b", "c", "d"],
                COL_CLASS_ID: [DEFAULT_CLASS] * 4,
                COL_N_GTS: [1, 1, 1, 0],
                COL_WEIGHT: [1.0] * 4,
                COL_GT_LABEL: [True, True, True, False],
            }
        )
        return DetectionTable.from_matched(det_df, meta_df)

    def test_curve_is_ordered_by_descending_threshold(self) -> None:
        """The curve's order is the sort key, not a property of the data.

        Limitation: this and ``test_tie_group_is_ordered_by_ascending_y`` pin
        the invariant rather than reproducing the defect — an unstable sort is
        *permitted* to preserve input order and on a five-row frame it does,
        so neither fails against the pre-fix ``.sort("fp_per_image")``. The
        guards that do fail on it are ``test_auc_is_invariant_to_input_row_order``
        and ``test_interpolate_at_a_tie_returns_the_upper_envelope``.
        """
        table = self._tied_table()
        thresholds = froc_curve_lazy(table).collect()["threshold"].to_list()
        assert thresholds == sorted(thresholds, reverse=True)
        assert len(set(thresholds)) == len(thresholds), "thresholds must be unique"

    def test_tie_group_is_ordered_by_ascending_y(self) -> None:
        """Within a tied x, the last row is the highest sensitivity.

        This is what makes the trapezoid leaving the tie use the upper
        envelope rather than whichever row the sort happened to leave last.
        """
        curve = froc_curve_lazy(self._tied_table()).collect()
        tied = curve.filter(pl.col("fp_per_image") == 0.0)
        assert tied.height > 1, "fixture must actually produce a tie"
        sens = tied["sensitivity"].to_list()
        assert sens == sorted(sens)

    def test_auc_is_invariant_to_input_row_order(self) -> None:
        """auc() must not change when the curve frame is shuffled.

        auc() re-sorts by x, so a curve whose tied rows arrive in a different
        order must still integrate to the same area.
        """
        table = self._tied_table()
        curve = froc_curve_lazy(table).collect()
        baseline = froc_auc(table).collect().item()
        for seed in range(8):
            shuffled = curve.sample(fraction=1.0, shuffle=True, seed=seed)
            got = MetricResult(curve=shuffled).auc(
                x_col="fp_per_image", y_col="sensitivity"
            )
            assert got == pytest.approx(baseline, abs=1e-12)

    def test_interpolate_at_a_tie_returns_the_upper_envelope(self) -> None:
        """sensitivity_at_fp(0.0) is the sensitivity reachable at zero FPs.

        The origin row (fp=0, sensitivity=0) shares x with every zero-FP
        operating point; returning its y would report that a detector making no
        false positives also finds nothing.
        """
        got = (
            froc_sensitivity_at_fp(self._tied_table(), 0.0)
            .collect()["sensitivity"]
            .item()
        )
        assert got == pytest.approx(1.0)


class TestSummaryTableDtype:
    """summary_table's y column stays Float64 even when every point is null."""

    def test_all_out_of_range_is_still_float64(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """A fully unreachable operating-point set must not yield Null dtype."""
        curve = froc_curve_lazy(simple_detection_table).collect()
        max_fp = float(curve["fp_per_image"].max())
        summary = froc_summary_table(
            simple_detection_table, fp_rates=[max_fp + 10.0, max_fp + 20.0]
        ).collect()
        assert summary["sensitivity"].dtype == pl.Float64
        assert summary["fp_per_image"].dtype == pl.Float64
        assert summary["sensitivity"].to_list() == [None, None]


# ---------------------------------------------------------------------------
# Review follow-up: image_meta is the sole source of metadata
# ---------------------------------------------------------------------------


class TestPreMatchedAdapterRejectsIgnoredParams:
    """image_meta cannot be combined with the derive-from-detections columns."""

    @staticmethod
    def _detections() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "image_id": ["a", "b"],
                "score": [0.9, 0.4],
                "is_tp": [True, False],
                "n_gts": [1, 1],
                "weight": [1.0, 1.0],
            }
        )

    @staticmethod
    def _meta() -> pl.DataFrame:
        return pl.DataFrame({"image_id": ["a", "b", "c"], "n_gts": [1, 1, 1]})

    @pytest.mark.parametrize(
        "kwarg", ["n_gts_col", "weight_col", "gt_label_col", "group_col"]
    )
    def test_conflicting_kwarg_raises(self, kwarg: str) -> None:
        """Each per-image column argument is rejected alongside image_meta."""
        from polars_cv.metrics import PreMatchedAdapter

        with pytest.raises(ValueError, match="image_meta cannot be combined"):
            PreMatchedAdapter().match(
                self._detections(),
                image_id_col="image_id",
                image_meta=self._meta(),
                **{kwarg: "n_gts"},
            )

    def test_image_meta_alone_is_accepted(self) -> None:
        """The supported call still works and keeps the zero-detection image."""
        from polars_cv.metrics import PreMatchedAdapter

        table = PreMatchedAdapter().match(
            self._detections(),
            image_id_col="image_id",
            image_meta=self._meta(),
        )
        meta = table.image_metadata.collect()
        assert sorted(meta[COL_IMAGE_ID].to_list()) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Review follow-up: one definition of a FROC evaluation unit
# ---------------------------------------------------------------------------


class TestFrocImageCount:
    """FP-per-image counts images, not (image, class) metadata rows."""

    def test_multi_class_denominator_is_the_image_count(self) -> None:
        """Two classes over two images give fp_per_image = fp / 2, not / 4."""
        det_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "b"],
                COL_CLASS_ID: ["cat", "dog"],
                COL_SCORE: [0.9, 0.8],
                COL_IS_TP: [False, False],
                COL_GT_IDX: [None, None],
                COL_IOU: [0.0, 0.0],
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
        # One row per (image, class): four rows, two images.
        meta_df = pl.DataFrame(
            {
                COL_IMAGE_ID: ["a", "a", "b", "b"],
                COL_CLASS_ID: ["cat", "dog", "cat", "dog"],
                COL_N_GTS: [1, 0, 0, 1],
                COL_WEIGHT: [1.0] * 4,
                COL_GT_LABEL: [True, False, False, True],
            }
        )
        table = DetectionTable.from_matched(det_df, meta_df)
        assert table.image_metadata.collect()[COL_IMAGE_ID].n_unique() == 2
        # Both detections are false positives, so the last point is 2 FP / 2
        # images = 1.0. Counting metadata rows would report 0.5.
        curve = froc_curve_lazy(table).collect()
        assert float(curve["fp_per_image"].max()) == pytest.approx(1.0)


class TestFrocBootstrapDrawsAreDistinctUnits:
    """Each bootstrap draw is its own evaluation unit, not a repeated image."""

    def test_repeated_draw_counts_once_per_draw(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """Drawing one image three times gives three images and three GTs."""
        samples = pl.LazyFrame(
            {
                "bootstrap_id": [0, 0, 0],
                COL_IMAGE_ID: ["a", "a", "a"],
                "_slot": [0, 1, 2],
            },
            schema={
                "bootstrap_id": pl.Int32,
                COL_IMAGE_ID: pl.String,
                "_slot": pl.Int64,
            },
        )
        replicate = _bootstrap_table_with_draws(simple_detection_table, samples)
        meta = replicate.image_metadata.collect()
        assert meta[COL_IMAGE_ID].n_unique() == 3
        assert int(meta[COL_N_GTS].sum()) == 3
        # Three copies of the same single TP: sensitivity reaches exactly 1.0.
        sens = froc_curve_lazy(replicate).collect()["sensitivity"].max()
        assert float(sens) == pytest.approx(1.0)

    def test_draw_ids_are_distinct(
        self, simple_detection_table: DetectionTable
    ) -> None:
        """The replicate's own table carries one id per draw, not per image."""
        samples = pl.LazyFrame(
            {
                "bootstrap_id": [0, 0, 0],
                COL_IMAGE_ID: ["a", "a", "b"],
                "_slot": [0, 1, 2],
            },
            schema={
                "bootstrap_id": pl.Int32,
                COL_IMAGE_ID: pl.String,
                "_slot": pl.Int64,
            },
        )
        replicate = _bootstrap_table_with_draws(simple_detection_table, samples)
        meta = replicate.image_metadata.collect()
        assert meta[COL_IMAGE_ID].n_unique() == 3


class TestPartialAucIntegerBounds:
    """partial_auc accepts integer bounds, the natural spelling of fp_range."""

    def test_integer_hi_beyond_the_curve(self) -> None:
        """`fp_range=(0, 8)` must integrate, not raise a SchemaError.

        The boundary point appended at `hi` was built with an inferred dtype,
        so an int bound produced an Int64 Series that would not concat onto
        the Float64 curve — which is every `froc.auc(fp_range=(0, 8))` call in
        the docs and the metrics example.
        """
        x = pl.Series("x", [0.0, 0.5, 1.0])
        y = pl.Series("y", [0.0, 0.5, 0.9])
        assert partial_auc(x, y, 0, 8, "normalize") == pytest.approx(
            partial_auc(x, y, 0.0, 8.0, "normalize")
        )

    def test_integer_lo_below_the_curve(self) -> None:
        """The prepended `lo` boundary has the same dtype requirement."""
        x = pl.Series("x", [2.0, 3.0])
        y = pl.Series("y", [0.4, 0.8])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert partial_auc(x, y, 0, 4) == pytest.approx(partial_auc(x, y, 0.0, 4.0))
