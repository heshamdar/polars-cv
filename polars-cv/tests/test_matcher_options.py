"""``ContourMatcher`` options: score reduction, per-row parameters, contour
predictions.

- ``score_reduction`` / ``score_region_mode`` pass through to ``label_reduce``:
  a saturating heatmap scores every detection at its peak, so detections tie
  and the FROC curve collapses; the mean tells them apart.
- Every parameter that does not change the table's shape may be a Polars
  expression, read per row — a tolerance in millimetres is a different pixel
  count per image when the pixel spacing varies.
- Predictions may be contours with their own scores, for models that emit
  masks or polygons rather than heatmaps.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import CONTOUR_SET_SCHEMA
from polars_cv.metrics import ContourMatcher
from polars_cv.metrics._types import COL_IMAGE_ID, COL_IS_TP, COL_SCORE
from tests.conftest import plugin_required

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_blobs() -> pl.DataFrame:
    """Two 4x4 blobs that both peak at 255: one strong (250), one weak (60)."""
    heatmap = [[0] * 12 for _ in range(12)]
    for y in range(1, 5):
        for x in range(1, 5):
            heatmap[y][x] = 250
    for y in range(7, 11):
        for x in range(7, 11):
            heatmap[y][x] = 60
    heatmap[2][2] = heatmap[8][8] = 255
    gt = [
        [255 if 1 <= y < 5 and 1 <= x < 5 else 0 for x in range(12)] for y in range(12)
    ]
    return pl.DataFrame(
        {"heatmap": [heatmap], "gt": [gt]},
        schema={
            "heatmap": pl.List(pl.List(pl.UInt8)),
            "gt": pl.List(pl.List(pl.UInt8)),
        },
    )


_SIZE = 40


def _band_heatmap(r0: int, r1: int, score: float) -> list[list[float]]:
    grid = [[0.0] * _SIZE for _ in range(_SIZE)]
    for y in range(r0, r1):
        for x in range(5, 35):
            grid[y][x] = score
    return grid


def _line(y: float) -> dict[str, object]:
    return {
        "exterior": [{"x": 6.0, "y": y}, {"x": 34.0, "y": y}],
        "holes": [],
        "is_closed": False,
    }


def _square(x0: float, y0: float, size: float) -> dict[str, object]:
    return {
        "exterior": [
            {"x": x0, "y": y0},
            {"x": x0 + size, "y": y0},
            {"x": x0 + size, "y": y0 + size},
            {"x": x0, "y": y0 + size},
        ],
        "holes": [],
        "is_closed": True,
    }


def _det(table) -> pl.DataFrame:  # noqa: ANN001
    det, _ = table.collect()
    return det.sort(COL_IMAGE_ID, COL_SCORE, descending=[False, True])


# ---------------------------------------------------------------------------
# 02: score reduction
# ---------------------------------------------------------------------------


def test_an_unknown_score_reduction_is_refused() -> None:
    with pytest.raises(ValueError, match="score_reduction"):
        ContourMatcher(score_reduction="median")


def test_an_unknown_score_region_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="score_region_mode"):
        ContourMatcher(score_region_mode="outline")


@plugin_required
class TestScoreReduction:
    def test_default_scores_by_the_peak(self) -> None:
        table = ContourMatcher(extraction_threshold=25.0, auto_resize=False).match(
            _two_blobs(), pred_col="heatmap", gt_col="gt"
        )
        assert _det(table)[COL_SCORE].to_list() == [255.0, 255.0]

    def test_mean_separates_a_saturated_pair(self) -> None:
        table = ContourMatcher(
            extraction_threshold=25.0, auto_resize=False, score_reduction="mean"
        ).match(_two_blobs(), pred_col="heatmap", gt_col="gt")
        det = _det(table)
        assert [round(v, 1) for v in det[COL_SCORE]] == [250.3, 72.2]
        # The strong blob is the one on the GT.
        assert det[COL_IS_TP].to_list() == [True, False]


# ---------------------------------------------------------------------------
# 03: per-row parameters
# ---------------------------------------------------------------------------


def test_an_expression_parameter_is_accepted_without_a_literal_check() -> None:
    # Validated per row when the query runs; there is nothing to check here.
    ContourMatcher(
        match_by="coverage",
        coverage_tolerance=5.0 / pl.col("spacing_mm"),
        iou_threshold=pl.col("t"),
        extraction_threshold=pl.col("e"),
        min_contour_area=pl.col("a"),
        auto_resize=False,
    )


def test_a_literal_parameter_is_still_checked() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        ContourMatcher(iou_threshold=1.5)
    with pytest.raises(ValueError, match="coverage_tolerance"):
        ContourMatcher(match_by="coverage", coverage_tolerance=-1.0)


@plugin_required
class TestPerRowParameters:
    @staticmethod
    def _lines(tolerance: list[float]) -> pl.DataFrame:
        """Two images, each a band at rows 9..12 and a GT line 2 px below it."""
        return pl.DataFrame(
            {
                "image_id": ["a", "b"],
                "heatmap": [_band_heatmap(9, 12, 0.9)] * 2,
                "gt": [[_line(14.0)]] * 2,
                "tol": tolerance,
            },
            schema={
                "image_id": pl.String,
                "heatmap": pl.List(pl.List(pl.Float64)),
                "gt": CONTOUR_SET_SCHEMA,
                "tol": pl.Float64,
            },
        )

    def test_a_per_row_coverage_tolerance(self) -> None:
        table = ContourMatcher(
            match_by="coverage",
            coverage_tolerance=pl.col("tol"),
            auto_resize=False,
        ).match(
            self._lines([0.5, 3.0]),
            pred_col="heatmap",
            gt_col="gt",
            image_id_col="image_id",
        )
        assert _det(table)[COL_IS_TP].to_list() == [False, True]

    def test_a_per_row_threshold_matches_the_literal_per_image(self) -> None:
        # A per-row threshold gives each image what a literal one would.
        df = self._lines([3.0, 3.0]).with_columns(t=pl.Series([1.0, 0.5]))
        literal = [
            _det(
                ContourMatcher(
                    match_by="coverage",
                    coverage_tolerance=3.0,
                    iou_threshold=t,
                    auto_resize=False,
                ).match(
                    df.filter(pl.col("image_id") == img),
                    pred_col="heatmap",
                    gt_col="gt",
                    image_id_col="image_id",
                )
            )[COL_IS_TP].to_list()
            for img, t in [("a", 1.0), ("b", 0.5)]
        ]
        per_row = ContourMatcher(
            match_by="coverage",
            coverage_tolerance=3.0,
            iou_threshold=pl.col("t"),
            auto_resize=False,
        ).match(df, pred_col="heatmap", gt_col="gt", image_id_col="image_id")
        assert _det(per_row)[COL_IS_TP].to_list() == literal[0] + literal[1]

    def test_a_per_row_extraction_threshold_and_min_area(self) -> None:
        # Image a extracts the band (threshold 0.5 < 0.9); image b's
        # threshold is above it, so b has no detection at all. A per-row
        # min area drops a's band in the third image.
        df = pl.DataFrame(
            {
                "image_id": ["a", "b", "c"],
                "heatmap": [_band_heatmap(9, 12, 0.9)] * 3,
                "gt": [[_line(10.5)]] * 3,
                "e": [0.5, 0.95, 0.5],
                "a": [0.0, 0.0, 1000.0],
            },
            schema={
                "image_id": pl.String,
                "heatmap": pl.List(pl.List(pl.Float64)),
                "gt": CONTOUR_SET_SCHEMA,
                "e": pl.Float64,
                "a": pl.Float64,
            },
        )
        table = ContourMatcher(
            match_by="coverage",
            coverage_tolerance=1.0,
            extraction_threshold=pl.col("e"),
            min_contour_area=pl.col("a"),
            auto_resize=False,
        ).match(df, pred_col="heatmap", gt_col="gt", image_id_col="image_id")
        assert _det(table)[COL_IMAGE_ID].to_list() == ["a"]

    def test_min_contour_area_fraction(self) -> None:
        # The band is 3 x 30 = 90 px of a 1600 px image (5.6%).
        df = self._lines([1.0, 1.0])
        kept = ContourMatcher(
            match_by="coverage",
            coverage_tolerance=1.0,
            min_contour_area_fraction=0.05,
            auto_resize=False,
        ).match(df, pred_col="heatmap", gt_col="gt", image_id_col="image_id")
        dropped = ContourMatcher(
            match_by="coverage",
            coverage_tolerance=1.0,
            min_contour_area_fraction=0.06,
            auto_resize=False,
        ).match(df, pred_col="heatmap", gt_col="gt", image_id_col="image_id")
        assert _det(kept).height == 2
        assert _det(dropped).height == 0

    def test_a_per_row_iou_threshold_stores_no_matching_threshold(self) -> None:
        df = self._lines([3.0, 3.0]).with_columns(t=pl.Series([0.5, 0.5]))
        table = ContourMatcher(
            match_by="coverage",
            coverage_tolerance=3.0,
            iou_threshold=pl.col("t"),
            auto_resize=False,
        ).match(df, pred_col="heatmap", gt_col="gt", image_id_col="image_id")
        # No single matching threshold to compare a re-threshold against.
        assert table._matching_iou_threshold is None


# ---------------------------------------------------------------------------
# 04: contour predictions
# ---------------------------------------------------------------------------


def _contour_preds() -> pl.DataFrame:
    """Image a: a hit (0.9) and a miss (0.4) on one GT square; b: no preds."""
    return pl.DataFrame(
        {
            "image_id": ["a", "b"],
            "pred": [[_square(0, 0, 10), _square(30, 30, 5)], []],
            "scores": [[0.9, 0.4], []],
            "gt": [[_square(0, 0, 10)], [_square(5, 5, 5)]],
        },
        schema={
            "image_id": pl.String,
            "pred": CONTOUR_SET_SCHEMA,
            "scores": pl.List(pl.Float64),
            "gt": CONTOUR_SET_SCHEMA,
        },
    )


def _match_contours(df: pl.DataFrame, **kwargs):  # noqa: ANN202
    return ContourMatcher(auto_resize=False).match(
        df, pred_col="pred", gt_col="gt", image_id_col="image_id", **kwargs
    )


@plugin_required
class TestContourPredictions:
    def test_contour_set_predictions_with_scores(self) -> None:
        table = _match_contours(_contour_preds(), score_col="scores")
        det, meta = table.collect()
        det = det.sort(COL_SCORE, descending=True)
        assert det[COL_SCORE].to_list() == [0.9, 0.4]
        assert det[COL_IS_TP].to_list() == [True, False]
        assert meta.sort(COL_IMAGE_ID)["n_gts"].to_list() == [1, 1]

    def test_a_zero_score_is_kept(self) -> None:
        # A caller's 0.0 is a confidence, not a heatmap with no evidence.
        df = _contour_preds().with_columns(scores=pl.Series([[0.9, 0.0], []]))
        assert _det(_match_contours(df, score_col="scores")).height == 2

    def test_single_contour_predictions_with_scalar_scores(self) -> None:
        df = pl.DataFrame(
            {
                "image_id": ["a"],
                "pred": [_square(0, 0, 10)],
                "scores": [0.7],
                "gt": [[_square(0, 0, 10)]],
            },
            schema={
                "image_id": pl.String,
                "pred": CONTOUR_SET_SCHEMA.inner,
                "scores": pl.Float64,
                "gt": CONTOUR_SET_SCHEMA,
            },
        )
        det = _det(_match_contours(df, score_col="scores"))
        assert det[COL_SCORE].to_list() == [0.7]
        assert det[COL_IS_TP].to_list() == [True]

    def test_contour_predictions_need_scores(self) -> None:
        with pytest.raises(ValueError, match="score_col"):
            _match_contours(_contour_preds())

    def test_contour_predictions_refuse_auto_resize(self) -> None:
        with pytest.raises(ValueError, match="auto_resize"):
            ContourMatcher().match(
                _contour_preds(), pred_col="pred", gt_col="gt", score_col="scores"
            )

    def test_a_score_count_mismatch_is_refused(self) -> None:
        df = _contour_preds().with_columns(scores=pl.Series([[0.9], []]))
        with pytest.raises(pl.exceptions.PolarsError):
            _match_contours(df, score_col="scores").collect()

    def test_a_null_score_is_refused(self) -> None:
        # A null-scored contour used to take part in matching -- claiming the
        # GT here, as an exact copy of it -- and then be dropped from the
        # table, so the 0.9 hit beside it was scored a false positive.
        df = _contour_preds().with_columns(
            pred=pl.Series(
                [[_square(0, 0, 10), _square(0, 0, 10)], []], dtype=CONTOUR_SET_SCHEMA
            ),
            scores=pl.Series([[None, 0.9], []], dtype=pl.List(pl.Float64)),
        )
        with pytest.raises(pl.exceptions.PolarsError):
            _match_contours(df, score_col="scores").collect()

    def test_a_null_score_list_beside_contours_is_refused(self) -> None:
        # A null list read as "no scores", so the row's contours vanished.
        df = _contour_preds().with_columns(
            scores=pl.Series([None, []], dtype=pl.List(pl.Float64))
        )
        with pytest.raises(pl.exceptions.PolarsError):
            _match_contours(df, score_col="scores").collect()

    def test_a_null_scalar_score_is_refused(self) -> None:
        df = pl.DataFrame(
            {"image_id": ["a"], "pred": [_square(0, 0, 10)], "scores": [None]},
            schema={
                "image_id": pl.String,
                "pred": CONTOUR_SET_SCHEMA.inner,
                "scores": pl.Float64,
            },
        ).with_columns(gt=pl.Series([[_square(0, 0, 10)]], dtype=CONTOUR_SET_SCHEMA))
        with pytest.raises(pl.exceptions.PolarsError):
            _match_contours(df, score_col="scores").collect()

    def test_a_null_prediction_row_has_no_detections(self) -> None:
        # A null contour set (with or without scores) is an image without
        # predictions, as for a heatmap: its GT still counts.
        df = _contour_preds().with_columns(
            pred=pl.Series([None, []], dtype=CONTOUR_SET_SCHEMA),
            scores=pl.Series([None, []], dtype=pl.List(pl.Float64)),
        )
        det, meta = _match_contours(df, score_col="scores").collect()
        assert det.height == 0
        assert meta["n_gts"].sum() == 2

    def test_scores_must_be_a_list_for_a_contour_set(self) -> None:
        df = _contour_preds().with_columns(scores=pl.Series([0.9, 0.1]))
        with pytest.raises(ValueError, match="score_col"):
            _match_contours(df, score_col="scores")

    def test_a_heatmap_prediction_refuses_a_score_column(self) -> None:
        with pytest.raises(ValueError, match="score_col"):
            ContourMatcher(auto_resize=False).match(
                _two_blobs().with_columns(s=pl.lit(1.0)),
                pred_col="heatmap",
                gt_col="gt",
                score_col="s",
            )
