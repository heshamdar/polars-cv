"""``carry=``: caller columns ride the matcher's pass into ``image_metadata``.

A detection evaluation often wants per-image measures (pixel Dice / IoU of a
heatmap) from the same images the matcher reads. Computing them in a second
plan over the image rows means two readers of those rows, and a streaming
cache then holds every image until the slower one catches up. With ``carry``
the measure is a column of the frame the matcher reads, evaluated in its own
pass, and lands in ``image_metadata``, one value per (image, class) row.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_cv.metrics as M
from polars_cv.geometry.schemas import BBOX_SCHEMA
from tests._streaming_guard import in_memory_nodes, unexplained

from .conftest import plugin_required


def _box(x: float, y: float, w: float, h: float) -> dict[str, float]:
    return {"x": x, "y": y, "width": w, "height": h}


_BOXES = pl.DataFrame(
    {
        "image_id": ["a", "b", "c"],
        "pred": [[_box(0, 0, 4, 4)], [_box(1, 1, 2, 2)], []],
        "pred_score": [[0.9], [0.4], []],
        "gt": [[_box(0, 0, 4, 4)], [], [_box(3, 3, 2, 2)]],
        "site": ["x", "y", None],
        "exposure": [1.5, 2.5, 3.5],
    },
    schema={
        "image_id": pl.String,
        "pred": pl.List(BBOX_SCHEMA),
        "pred_score": pl.List(pl.Float64),
        "gt": pl.List(BBOX_SCHEMA),
        "site": pl.String,
        "exposure": pl.Float64,
    },
)


def _bbox(**carry: object) -> M.DetectionTable:
    init = carry.pop("init", {})
    assert isinstance(init, dict)
    return M.BBoxMatcher(**init).match(
        _BOXES,
        pred_col="pred",
        gt_col="gt",
        score_col="pred_score",
        image_id_col="image_id",
        **carry,  # ty: ignore[invalid-argument-type]
    )


def _meta(table: M.DetectionTable) -> pl.DataFrame:
    return table.image_metadata.collect().sort("image_id")


@plugin_required
class TestBBoxCarry:
    def test_columns_and_expressions_reach_the_metadata(self) -> None:
        meta = _meta(_bbox(carry=["site", (pl.col("exposure") * 2).alias("double")]))
        assert meta["site"].to_list() == ["x", "y", None]
        assert meta["double"].to_list() == [3.0, 5.0, 7.0]

    def test_detections_are_unchanged(self) -> None:
        plain = _bbox().detections.collect()
        carried = _bbox(carry=["site"]).detections.collect()
        assert carried.equals(plain)

    def test_a_sweep_carries_each_threshold_row(self) -> None:
        table = _bbox(init={"iou_threshold": [0.5, 0.75]}, carry=["exposure"])
        meta = (
            table.frames(["iou_threshold"])[1]
            .collect()
            .sort("iou_threshold", "image_id")
        )
        assert meta["exposure"].to_list() == [1.5, 2.5, 3.5] * 2

    def test_a_carried_column_groups(self) -> None:
        table = _bbox(carry=["site"])
        by_site = M.FROCSensitivity(1.0).by_group(table, "site").collect()
        as_group = M.FROCSensitivity(1.0).by_group(table.with_group("site"), "group_id")
        assert by_site.equals(as_group.collect().rename({"group_id": "site"}))

    def test_the_plan_stays_streaming(self) -> None:
        table = _bbox(carry=["site", pl.col("exposure").alias("e")])
        for lf in (table.detections, table.image_metadata):
            assert unexplained(in_memory_nodes(lf)) == []

    @pytest.mark.parametrize(
        ("carry", "match"),
        [
            (["nope"], "nope"),
            (["weight"], "reserved"),
            ([pl.col("site").alias("group_id")], "reserved"),
            (["site", pl.col("exposure").alias("site")], "more than once"),
            ([pl.lit(1)], "literal"),
        ],
        ids=["missing", "reserved", "reserved-expr", "duplicate", "unnamed"],
    )
    def test_refused(self, carry: list, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _bbox(carry=carry)


def _heatmaps() -> pl.DataFrame:
    a = np.zeros((16, 16))
    a[3:9, 3:9] = 0.9
    g = np.zeros((16, 16))
    g[4:10, 3:9] = 1.0
    return pl.DataFrame(
        {
            "image_id": ["a", "b"],
            "pred": [a.tolist(), a.tolist()],
            "gt": [g.tolist(), a.tolist()],
        }
    )


@plugin_required
class TestContourCarry:
    def test_segmentation_measures_ride_the_match(self) -> None:
        data = _heatmaps()
        seg = M.segmentation_measures("pred", "gt")
        table = M.ContourMatcher().match(
            data,
            pred_col="pred",
            gt_col="gt",
            image_id_col="image_id",
            carry=[seg.struct.field("dice"), seg.struct.field("iou")],
        )
        meta = _meta(table)
        want = (
            data.select("image_id", seg)
            .unnest("segmentation")
            .select("image_id", "dice", "iou")
            .sort("image_id")
        )
        assert meta.select("image_id", "dice", "iou").equals(want)
        for lf in (table.detections, table.image_metadata):
            assert unexplained(in_memory_nodes(lf)) == []


class TestPreMatchedCarry:
    _DET = pl.DataFrame(
        {
            "image_id": ["a", "a", "b"],
            "score": [0.9, 0.4, 0.7],
            "is_tp": [True, False, True],
            "scanner": ["s1", "s1", "s2"],
        }
    )

    def test_from_image_meta(self) -> None:
        meta = pl.DataFrame(
            {"image_id": ["a", "b", "c"], "n_gts": [1, 1, 0], "age": [30, 40, 50]}
        )
        table = M.PreMatchedAdapter().match(
            self._DET, image_id_col="image_id", image_meta=meta, carry=["age"]
        )
        assert _meta(table)["age"].to_list() == [30, 40, 50]

    def test_from_the_detection_rows(self) -> None:
        with pytest.warns(UserWarning, match="without image_meta"):
            table = M.PreMatchedAdapter().match(
                self._DET, image_id_col="image_id", carry=["scanner"]
            )
        assert _meta(table)["scanner"].to_list() == ["s1", "s2"]


@plugin_required
def test_match_detections_carries_images_columns() -> None:
    preds = pl.DataFrame(
        {
            "image_id": ["a", "b"],
            "bbox": [[0, 0, 4, 4], [1, 1, 2, 2]],
            "score": [0.9, 0.5],
        }
    )
    gts = pl.DataFrame({"image_id": ["a", "c"], "bbox": [[0, 0, 4, 4], [2, 2, 2, 2]]})
    images = pl.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "w": [1.0, 2.0, 3.0],
            "scanner": ["s1", "s2", None],
        }
    )
    table = M.match_detections(
        preds,
        gts,
        box_format="xywh",
        class_id=None,
        images=images,
        weight="w",
        carry=["scanner", (pl.col("w") * 10).alias("w10")],
    )
    meta = _meta(table)
    assert meta["scanner"].to_list() == ["s1", "s2", None]
    assert meta["w10"].to_list() == [10.0, 20.0, 30.0]
    assert unexplained(in_memory_nodes(table.image_metadata)) == []
