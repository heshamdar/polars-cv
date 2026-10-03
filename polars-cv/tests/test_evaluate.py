"""One-call evaluation: evaluate_detections / evaluate_heatmaps / evaluate_segmentation.

The reports compose the lower layers, so most tests here pin that the
composition is right — COCO numbers worked out by hand, agreement with the
manual route, summary/table consistency, intervals through the shared engine —
rather than re-testing the estimators.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import polars_cv.metrics as M
from polars_cv.metrics import (
    COCO_IOU_THRESHOLDS,
    evaluate_detections,
    evaluate_heatmaps,
    evaluate_segmentation,
    segmentation_measures,
)

from .conftest import plugin_required

# One GT box. The 0.9 box overlaps it at IoU 0.6, the 0.8 box at IoU 0.8.
# COCO re-matches at each threshold:
#   0.50-0.60: the 0.9 box is the TP, the 0.8 box a duplicate -> AP 1
#   0.65-0.80: the 0.9 box no longer qualifies, the 0.8 box is the TP -> AP 0.5
#   0.85-0.95: nothing qualifies -> AP 0
# so mAP = (3 * 1 + 4 * 0.5) / 10 = 0.5, AP50 = 1, AP75 = 0.5, and the
# recall at each threshold is 1 (x7) then 0 (x3): mAR = 0.7. Re-thresholding a
# 0.5 match instead would score 0.65-0.80 as 0 (mAP 0.3).
COCO_PREDS = pl.DataFrame(
    {
        "image_id": ["a", "a"],
        "class_id": ["cat", "cat"],
        "bbox": [[0, 0, 10, 6], [0, 0, 10, 8]],
        "score": [0.9, 0.8],
    }
)
COCO_GTS = pl.DataFrame(
    {"image_id": ["a"], "class_id": ["cat"], "bbox": [[0, 0, 10, 10]]}
)


@plugin_required
class TestEvaluateDetections:
    def report(self):
        return evaluate_detections(COCO_PREDS, COCO_GTS, box_format="xywh")

    def test_coco_numbers_by_hand(self) -> None:
        r = self.report()
        assert r.iou_thresholds == COCO_IOU_THRESHOLDS
        assert r.interpolation == "101_point"
        assert r.value("map") == pytest.approx(0.5)
        assert r.value("map_50") == pytest.approx(1.0)
        assert r.value("map_75") == pytest.approx(0.5)
        assert r.value("mar") == pytest.approx(0.7)

    def test_the_tables_agree_with_the_summary(self) -> None:
        r = evaluate_detections(
            pl.concat(
                [
                    COCO_PREDS,
                    pl.DataFrame(
                        {
                            "image_id": ["b", "c"],
                            "class_id": ["dog", "dog"],
                            "bbox": [[5, 5, 10, 10], [0, 0, 4, 4]],
                            "score": [0.7, 0.6],
                        }
                    ),
                ]
            ),
            pl.concat(
                [
                    COCO_GTS,
                    pl.DataFrame(
                        {
                            "image_id": ["b"],
                            "class_id": ["dog"],
                            "bbox": [[5, 5, 10, 9]],
                        }
                    ),
                ]
            ),
            box_format="xywh",
        )
        pc = r.per_class
        assert pc["class_id"].to_list() == ["cat", "dog"]
        assert r.value("map") == pytest.approx(pc["ap"].mean())
        at50 = r.per_threshold.filter(pl.col("iou_threshold") == 0.5)
        assert r.value("map_50") == pytest.approx(at50["ap"].mean())
        assert pc.filter(pl.col("class_id") == "dog")["n_preds"].item() == 2
        assert r.per_threshold.height == 10 * 2

    def test_a_class_without_truth_is_left_out_of_the_mean(self) -> None:
        preds = pl.concat(
            [
                COCO_PREDS,
                pl.DataFrame(
                    {
                        "image_id": ["a"],
                        "class_id": ["bird"],
                        "bbox": [[50, 50, 5, 5]],
                        "score": [0.3],
                    }
                ),
            ]
        )
        r = evaluate_detections(preds, COCO_GTS, box_format="xywh")
        assert r.value("map") == pytest.approx(0.5)
        bird = r.per_class.filter(pl.col("class_id") == "bird")
        assert bird["ap"].item() is None and bird["n_preds"].item() == 1

    def test_pascal_voc_is_one_threshold(self) -> None:
        r = evaluate_detections(
            COCO_PREDS, COCO_GTS, box_format="xywh", iou_thresholds=0.5
        )
        assert r.interpolation == "all_points"
        assert list(r.summary["metric"]) == ["map", "mar"]
        assert r.value("map") == pytest.approx(1.0)

    def test_intervals_through_the_shared_engine(self) -> None:
        r = self.report()
        ci = r.ci("map", n_bootstrap=20, seed=3)
        assert ci.columns == ["metric", "value", "ci_lower", "ci_upper"]
        assert ci["value"].item() == pytest.approx(0.5)
        direct = (
            M.bootstrap_ci(r.table, r.metrics["map"].statistic, n_bootstrap=20, seed=3)
            .collect()
            .rename({"map": "value"})
        )
        assert ci.drop("metric").equals(direct)
        with pytest.raises(ValueError, match="unknown metric"):
            r.ci("nope")

    def test_matches_are_the_prediction_rows(self) -> None:
        r = self.report()
        at50 = r.matches(0.5).sort("score", descending=True)
        assert at50.columns == [*COCO_PREDS.columns, "is_tp", "iou", "matched_gt_row"]
        assert at50["is_tp"].to_list() == [True, False]
        assert at50["matched_gt_row"].to_list() == [0, None]
        at75 = r.matches(0.75).sort("score", descending=True)
        assert at75["is_tp"].to_list() == [False, True]

    def test_the_lower_layers_are_one_step_away(self) -> None:
        r = self.report()
        assert r.pr_curve(iou_threshold=0.75).auc() == pytest.approx(0.5)
        assert r.confusion(0.85, iou_threshold=0.5).tp == 1
        assert r.froc(0.5).height > 0
        assert "map" in repr(r)

    @pytest.mark.parametrize("bad", ["voc", "COCO"])
    def test_an_unknown_preset_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError, match="'coco'"):
            evaluate_detections(
                COCO_PREDS, COCO_GTS, box_format="xywh", iou_thresholds=bad
            )  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def _mask(h: int, w: int, rects: list[tuple[int, int, int, int]], v: float = 1.0):
    """``rects`` as (x0, y0, x1, y1), half-open."""
    grid = [[0.0] * w for _ in range(h)]
    for x0, y0, x1, y1 in rects:
        for y in range(y0, y1):
            for x in range(x0, x1):
                grid[y][x] = v
    return grid


@plugin_required
class TestEvaluateHeatmaps:
    def data(self) -> pl.DataFrame:
        heat = [
            _mask(32, 32, [(4, 4, 12, 12)], 0.9),
            _mask(32, 32, [(20, 20, 28, 28)], 0.6),
            _mask(32, 32, [(2, 2, 6, 6)], 0.8),
        ]
        # Add a lower-confidence false positive to the first image.
        for y in range(20, 24):
            for x in range(20, 24):
                heat[0][y][x] = 0.4
        gts = [
            _mask(32, 32, [(4, 4, 12, 12)]),
            _mask(32, 32, []),
            _mask(32, 32, [(20, 2, 26, 8)]),
        ]
        return pl.DataFrame({"heat": heat, "gt": gts})

    def test_it_is_the_matcher_and_froc_route(self) -> None:
        r = evaluate_heatmaps(self.data(), heatmap="heat", gt="gt", extrapolate="flat")
        table = M.ContourMatcher().match(self.data(), pred_col="heat", gt_col="gt")
        manual = M.froc_summary_table(
            table, list(M.FROC_RATES), extrapolate="flat"
        ).collect()
        for rate, sens in manual.iter_rows():
            assert r.value(f"sensitivity@{rate:g}") == pytest.approx(sens)
        assert r.value("cpm") == pytest.approx(manual["sensitivity"].mean())
        assert r.value("map") == pytest.approx(M.average_precision(table))

    def test_matcher_options_pass_through_and_are_checked(self) -> None:
        r = evaluate_heatmaps(
            self.data(), heatmap="heat", gt="gt", extraction_threshold=0.5
        )
        # The 0.4 false positive is below the extraction threshold now.
        assert r.table.detections.collect().height == 3
        with pytest.raises(TypeError):
            evaluate_heatmaps(self.data(), heatmap="heat", gt="gt", bogus=1)

    def test_a_cpm_interval(self) -> None:
        r = evaluate_heatmaps(self.data(), heatmap="heat", gt="gt", extrapolate="flat")
        ci = r.ci("cpm", n_bootstrap=20, seed=1)
        assert ci["value"].item() == pytest.approx(r.value("cpm"))


@plugin_required
class TestSegmentation:
    def frame(self, pairs) -> pl.DataFrame:
        return pl.DataFrame(
            {"pred": [p for p, _ in pairs], "gt": [g for _, g in pairs]}
        )

    def test_perfect_shifted_and_empty(self) -> None:
        sq = _mask(32, 32, [(8, 8, 20, 20)])
        shifted = _mask(32, 32, [(10, 8, 22, 20)])
        empty = _mask(32, 32, [])
        df = self.frame([(sq, sq), (shifted, sq), (empty, empty), (sq, empty)])
        out = df.select(segmentation_measures("pred", "gt")).unnest("segmentation")
        perfect, moved, both_empty, one_empty = out.iter_rows(named=True)
        assert perfect["dice"] == 1.0 and perfect["hd"] == 0.0
        assert moved["hd"] == pytest.approx(2.0)
        assert moved["iou"] == pytest.approx((10 * 12) / (14 * 12))
        assert both_empty["dice"] == 1.0 and both_empty["hd"] is None
        assert one_empty["dice"] == 0.0 and one_empty["hd"] is None

    def test_holes_are_boundary(self) -> None:
        ring = _mask(32, 32, [(4, 4, 28, 28)])
        for y in range(12, 20):
            for x in range(12, 20):
                ring[y][x] = 0.0
        solid = _mask(32, 32, [(4, 4, 28, 28)])
        out = (
            self.frame([(ring, solid)])
            .select(segmentation_measures("pred", "gt"))
            .unnest("segmentation")
            .row(0, named=True)
        )
        # The hole's border is 8 px from the solid square's outline.
        assert out["hd"] == pytest.approx(8.0)

    def test_spacing_gives_physical_units(self) -> None:
        sq = _mask(32, 32, [(8, 8, 20, 20)])
        shifted = _mask(32, 32, [(10, 8, 22, 20)])
        out = (
            self.frame([(shifted, sq)])
            .select(segmentation_measures("pred", "gt", spacing=(1.0, 0.5)))
            .unnest("segmentation")
            .row(0, named=True)
        )
        assert out["hd"] == pytest.approx(1.0)  # 2 columns x 0.5

    def test_the_frame_leaves_the_image_edge_unmeasured(self) -> None:
        full = _mask(16, 16, [(0, 0, 16, 16)])
        df = self.frame([(full, full)])
        framed = df.select(segmentation_measures("pred", "gt")).unnest("segmentation")
        assert framed["hd"].item() is None  # all boundary is the image edge
        unframed = df.select(segmentation_measures("pred", "gt", frame=None)).unnest(
            "segmentation"
        )
        assert unframed["hd"].item() == 0.0

    def test_the_report(self) -> None:
        sq = _mask(32, 32, [(8, 8, 20, 20)])
        shifted = _mask(32, 32, [(10, 8, 22, 20)])
        empty = _mask(32, 32, [])
        df = self.frame([(sq, sq), (shifted, sq), (empty, sq)])
        r = evaluate_segmentation(df, pred="pred", target="gt")
        assert r.per_image.height == 3
        hd = r.summary.filter(pl.col("metric") == "hd").row(0, named=True)
        assert hd["value"] == pytest.approx(1.0)  # mean of 0 and 2
        assert hd["n_undefined"] == 1
        ci = r.ci(["dice", "hd"], n_bootstrap=50, seed=2)
        again = r.ci(["dice", "hd"], n_bootstrap=50, seed=2)
        assert ci.equals(again)
        for row in ci.iter_rows(named=True):
            assert row["ci_lower"] <= row["value"] <= row["ci_upper"]
        assert "dice" in repr(r)
        assert not math.isnan(r.value("dice"))
