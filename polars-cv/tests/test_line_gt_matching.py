"""Detection matching against line-shaped ground truth.

Some landmarks are annotated as lines (a skinfold, a muscle edge) but
predicted as thin regions. IoU between a line and a region is zero, so
``ContourMatcher(match_by="coverage")`` pairs them by how much of each GT line
lies within a tolerance of a prediction; GT contours may be given as a
contour column instead of a mask; and ``duplicates="ignore"`` drops repeated
hits on one GT (the LUNA16/CAMELYON convention) instead of counting them as
false positives.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import CONTOUR_SET_SCHEMA
from polars_cv.metrics import ContourMatcher, froc_curve_lazy, lroc_auc
from polars_cv.metrics._types import COL_IS_TP
from tests.conftest import plugin_required

_SIZE = 40


def _heatmap(bands: list[tuple[int, int, float]]) -> list[list[float]]:
    """Horizontal bands ``(row0, row1, score)`` across columns 5..35."""
    grid = [[0.0] * _SIZE for _ in range(_SIZE)]
    for r0, r1, score in bands:
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


def _frame(bands: list[tuple[int, int, float]], lines: list[float]) -> pl.DataFrame:
    """One image with GT lines at the given ``y`` values, plus an empty one."""
    return pl.DataFrame(
        {
            "image_id": ["a", "b"],
            "heatmap": [_heatmap(bands), _heatmap([])],
            "gt": [[_line(y) for y in lines], []],
        },
        schema={
            "image_id": pl.String,
            "heatmap": pl.List(pl.List(pl.Float64)),
            "gt": CONTOUR_SET_SCHEMA,
        },
    )


def _match(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    matcher = ContourMatcher(
        iou_threshold=0.5, auto_resize=False, extraction_threshold=0.1, **kwargs
    )
    table = matcher.match(df, pred_col="heatmap", gt_col="gt", image_id_col="image_id")
    det, _ = table.collect()
    return det.sort("score", descending=True)


@plugin_required
class TestCoverageMatching:
    def test_a_line_under_a_thin_prediction_is_a_true_positive(self) -> None:
        df = _frame([(9, 12, 0.9)], [10.5])
        det = _match(df, match_by="coverage", coverage_tolerance=1.0)
        assert det[COL_IS_TP].to_list() == [True]

    def test_a_line_away_from_every_prediction_is_missed(self) -> None:
        df = _frame([(9, 12, 0.9)], [25.0])
        det = _match(df, match_by="coverage", coverage_tolerance=1.0)
        assert det[COL_IS_TP].to_list() == [False]

    def test_iou_cannot_score_a_line(self) -> None:
        df = _frame([(9, 12, 0.9)], [10.5])
        with pytest.raises(pl.exceptions.ComputeError, match="open contour"):
            _match(df)

    def test_the_table_feeds_froc_and_lroc(self) -> None:
        df = _frame([(9, 12, 0.9)], [10.5])
        matcher = ContourMatcher(
            iou_threshold=0.5,
            auto_resize=False,
            match_by="coverage",
            coverage_tolerance=1.0,
        )
        table = matcher.match(
            df, pred_col="heatmap", gt_col="gt", image_id_col="image_id"
        )
        # One TP and no FP: the curve reaches full sensitivity at 0 FP/image.
        curve = froc_curve_lazy(table).collect()
        assert curve["sensitivity"].max() == pytest.approx(1.0)
        assert curve["fp_per_image"].max() == 0.0
        assert lroc_auc(table).collect().item() == pytest.approx(1.0)


@plugin_required
class TestDuplicates:
    """Two predictions on one GT line: the second is a duplicate hit."""

    _BANDS = [(9, 12, 0.9), (13, 15, 0.6)]

    def test_a_duplicate_is_a_false_positive_by_default(self) -> None:
        det = _match(
            _frame(self._BANDS, [10.5]), match_by="coverage", coverage_tolerance=3.0
        )
        assert det[COL_IS_TP].to_list() == [True, False]

    def test_ignore_drops_the_duplicate(self) -> None:
        det = _match(
            _frame(self._BANDS, [10.5]),
            match_by="coverage",
            coverage_tolerance=3.0,
            duplicates="ignore",
        )
        assert det[COL_IS_TP].to_list() == [True]

    def test_ignore_keeps_a_plain_miss(self) -> None:
        det = _match(
            _frame([(9, 12, 0.9), (30, 32, 0.6)], [10.5]),
            match_by="coverage",
            coverage_tolerance=3.0,
            duplicates="ignore",
        )
        assert det[COL_IS_TP].to_list() == [True, False]


class TestConfiguration:
    def test_coverage_needs_a_tolerance(self) -> None:
        with pytest.raises(ValueError, match="coverage_tolerance"):
            ContourMatcher(match_by="coverage")

    def test_a_tolerance_needs_coverage(self) -> None:
        with pytest.raises(ValueError, match="coverage_tolerance"):
            ContourMatcher(coverage_tolerance=1.0)

    @pytest.mark.parametrize(
        ("kwarg", "value"), [("match_by", "dice"), ("duplicates", "drop")]
    )
    def test_an_unknown_policy_is_refused(self, kwarg: str, value: str) -> None:
        with pytest.raises(ValueError, match=kwarg):
            ContourMatcher(**{kwarg: value})

    @plugin_required
    def test_contour_ground_truth_needs_auto_resize_off(self) -> None:
        df = _frame([(9, 12, 0.9)], [10.5])
        with pytest.raises(ValueError, match="auto_resize"):
            ContourMatcher(match_by="coverage", coverage_tolerance=1.0).match(
                df, pred_col="heatmap", gt_col="gt", image_id_col="image_id"
            )
