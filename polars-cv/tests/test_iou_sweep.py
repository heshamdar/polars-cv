"""Matching at several IoU thresholds, and multi-class matcher input.

COCO's mAP@[.5:.95] matches *again* at every threshold: a detection that
claimed a ground truth at 0.5 may lose it at 0.75 to a lower-scoring but
better-overlapping one. Re-thresholding a single 0.5 match cannot see that, so
the matchers take a sequence of thresholds and return one table carrying an
``iou_threshold`` column. Such a table refuses any evaluation that would pool
its thresholds (a pooled AP would count every ground truth once per
threshold); grouping by ``iou_threshold``, averaging over it (``MeanOver``) or
selecting one (``at_iou_threshold``) are the ways in.
"""

from __future__ import annotations

import polars as pl
import pytest

import polars_cv.metrics as M
from polars_cv.geometry import BBOX_SCHEMA
from polars_cv.metrics import (
    AP,
    FROCAUC,
    BBoxMatcher,
    DetectionTable,
    bootstrap_ci,
    mean_ap,
)

from .conftest import plugin_required


def _box(x: float, y: float, w: float, h: float) -> dict:
    return {"x": float(x), "y": float(y), "width": float(w), "height": float(h)}


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).cast(
        {"pred": pl.List(BBOX_SCHEMA), "gt": pl.List(BBOX_SCHEMA)}
    )


def _rematch_case() -> pl.DataFrame:
    """At 0.5 the 0.9 box (IoU 0.6) claims the GT and the 0.8 box (IoU 0.8)
    is a duplicate; at 0.75 the 0.9 box no longer qualifies and the 0.8 box
    claims it. Plus a second image that matches cleanly at both."""
    return _frame(
        [
            {
                "img": "a",
                "pred": [_box(0, 0, 10, 6), _box(0, 0, 10, 8)],
                "gt": [_box(0, 0, 10, 10)],
                "s": [0.9, 0.8],
            },
            {
                "img": "b",
                "pred": [_box(50, 50, 10, 10)],
                "gt": [_box(50, 50, 10, 10)],
                "s": [0.7],
            },
        ]
    )


def _match(thresholds, df=None, **kw) -> DetectionTable:
    return BBoxMatcher(iou_threshold=thresholds).match(
        _rematch_case() if df is None else df,
        pred_col="pred",
        gt_col="gt",
        score_col="s",
        image_id_col="img",
        **kw,
    )


def _same(a: DetectionTable, b: DetectionTable) -> bool:
    (da, ma), (db, mb) = a.collect(), b.collect()
    return da.sort(pl.all()).equals(db.sort(pl.all())) and ma.sort(pl.all()).equals(
        mb.sort(pl.all())
    )


@plugin_required
class TestMultiClassInput:
    """One row per (image, class): each detection keeps its own row's class."""

    def _two_classes(self) -> pl.DataFrame:
        return _frame(
            [
                {
                    "img": "a",
                    "cls": "cat",
                    "pred": [_box(0, 0, 10, 10)],
                    "gt": [_box(0, 0, 10, 10)],
                    "s": [0.9],
                },
                {
                    "img": "a",
                    "cls": "dog",
                    "pred": [_box(50, 0, 10, 10), _box(80, 0, 5, 5)],
                    "gt": [_box(50, 0, 10, 10)],
                    "s": [0.8, 0.3],
                },
            ]
        )

    def test_detections_are_not_duplicated_across_an_images_classes(self) -> None:
        table = BBoxMatcher().match(
            self._two_classes(),
            pred_col="pred",
            gt_col="gt",
            score_col="s",
            class_col="cls",
            image_id_col="img",
        )
        det = table.detections.collect().sort("score", descending=True)
        assert det.height == 3
        assert det["class_id"].to_list() == ["cat", "dog", "dog"]
        assert det["is_tp"].to_list() == [True, True, False]
        assert M.average_precision(table, class_id="dog") == pytest.approx(1.0)

    def test_contour_matcher_keeps_each_rows_class(self) -> None:
        from polars_cv.geometry import contour_set_from_coords

        sq = lambda x: [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]]  # noqa: E731
        df = pl.DataFrame(
            {
                "img": ["a", "a"],
                "cls": ["cat", "dog"],
                "pred": [[sq(0)], [sq(50), sq(80)]],
                "gt": [[sq(0)], [sq(50)]],
                "s": [[0.9], [0.8, 0.3]],
            }
        ).with_columns(
            pred=contour_set_from_coords(pl.col("pred")),
            gt=contour_set_from_coords(pl.col("gt")),
        )
        table = M.ContourMatcher(auto_resize=False).match(
            df,
            pred_col="pred",
            gt_col="gt",
            score_col="s",
            class_col="cls",
            image_id_col="img",
        )
        det = table.detections.collect().sort("score", descending=True)
        assert det["class_id"].to_list() == ["cat", "dog", "dog"]
        assert det["is_tp"].to_list() == [True, True, False]


@plugin_required
class TestSweepTable:
    def test_a_sequence_matches_once_per_threshold(self) -> None:
        sweep = _match([0.5, 0.75])
        assert sweep.iou_thresholds == (0.5, 0.75)
        for t in (0.5, 0.75):
            assert _same(sweep.at_iou_threshold(t), _match(t))

    def test_rematching_differs_from_rethresholding(self) -> None:
        exact = _match([0.5, 0.75]).at_iou_threshold(0.75)
        det = exact.collect()[0].sort("score", descending=True)
        assert det["is_tp"].to_list() == [False, True, True]
        approx = _match(0.5).at_iou_threshold(0.75)
        det = approx.collect()[0].sort("score", descending=True)
        assert det["is_tp"].to_list() == [False, False, True]

    def test_a_threshold_between_matched_ones_rethresholds_the_one_below(
        self,
    ) -> None:
        sweep = _match([0.5, 0.75])
        between = sweep.at_iou_threshold(0.8)
        assert _same(between, _match(0.75).at_iou_threshold(0.8))
        with pytest.warns(UserWarning, match="Lowering"):
            sweep.at_iou_threshold(0.3)

    def test_pooling_the_thresholds_is_refused(self) -> None:
        sweep = _match([0.5, 0.75])
        for read in (lambda t: t.detections, lambda t: t.image_metadata):
            with pytest.raises(ValueError, match="iou_threshold"):
                read(sweep)
        with pytest.raises(ValueError, match="iou_threshold"):
            M.average_precision(sweep)
        with pytest.raises(ValueError, match="iou_threshold"):
            M.froc_auc(sweep, fp_range=(0.0, 1.0))
        with pytest.raises(ValueError, match="iou_threshold"):
            M.lroc_auc(sweep)
        with pytest.raises(ValueError, match="iou_threshold"):
            AP().by_group(sweep)
        with pytest.raises(ValueError, match="iou_threshold"):
            bootstrap_ci(sweep, AP(), n_bootstrap=5)

    def test_grouping_by_the_threshold_reads_each_slice(self) -> None:
        sweep = _match([0.5, 0.75])
        per = AP().by_group(sweep, "iou_threshold").collect().sort("iou_threshold")
        for t, ap in zip(per["iou_threshold"], per["ap"], strict=True):
            assert ap == pytest.approx(AP().value(sweep.at_iou_threshold(t)))
        froc = (
            M.froc_auc(sweep, fp_range=(0.0, 1.0), group_by="iou_threshold")
            .collect()
            .sort("iou_threshold")
        )
        for t, auc in zip(froc["iou_threshold"], froc["auc"], strict=True):
            one = M.froc_auc(sweep.at_iou_threshold(t), fp_range=(0.0, 1.0))
            assert auc == pytest.approx(one.collect().item())
        lroc = M.lroc_auc(sweep, group_by="iou_threshold").collect()
        assert lroc.height == 2

    def test_map_averages_the_rematched_thresholds(self) -> None:
        sweep = _match([0.5, 0.75])
        each = [AP().value(sweep.at_iou_threshold(t)) for t in (0.5, 0.75)]
        assert mean_ap().value(sweep) == pytest.approx(sum(each) / 2)
        assert M.mean_average_precision(sweep) == pytest.approx(sum(each) / 2)
        # A single matching re-thresholded keeps its historical meaning.
        single = _match(0.5)
        rethresholded = [AP().value(single.at_iou_threshold(t)) for t in (0.5, 0.75)]
        assert M.mean_average_precision(
            single, iou_thresholds=[0.5, 0.75]
        ) == pytest.approx(sum(rethresholded) / 2)

    def test_a_map_interval_over_a_sweep(self) -> None:
        sweep = _match([0.5, 0.75])
        out = bootstrap_ci(sweep, mean_ap(), n_bootstrap=20, seed=1).collect()
        assert out["map"].item() == pytest.approx(mean_ap().value(sweep))
        grouped = bootstrap_ci(
            sweep, FROCAUC(fp_range=(0.0, 1.0)), group_by="iou_threshold", n_bootstrap=5
        ).collect()
        assert grouped.height == 2

    def test_views_keep_the_sweep(self) -> None:
        sweep = _match([0.5, 0.75])
        for view in (
            sweep.filter_class("__all__"),
            sweep.filter_images(["a"]),
            sweep.with_group("class_id"),
        ):
            assert view.iou_thresholds == (0.5, 0.75)
            with pytest.raises(ValueError, match="iou_threshold"):
                _ = view.detections

    def test_thresholds_are_validated(self) -> None:
        for bad in ([], [0.5, 0.5], [0.5, 1.5]):
            with pytest.raises(ValueError, match="iou_threshold"):
                BBoxMatcher(iou_threshold=bad)
