"""Object-level matching: long tables of boxes, polygons or instance masks."""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.geometry import BBOX_SCHEMA, contour_from_coords
from polars_cv.metrics import AP, group_objects, match_detections

from .conftest import plugin_required

# Two images, two classes, xyxy boxes. img3 has truth only; img4 has
# predictions only (false positives on an empty image).
PREDS = pl.DataFrame(
    {
        "image_id": ["img1", "img1", "img1", "img2", "img4"],
        "class_id": ["cat", "cat", "dog", "cat", "dog"],
        "box": [
            [0, 0, 10, 10],
            [50, 50, 60, 60],
            [20, 0, 30, 10],
            [0, 0, 8, 8],
            [0, 0, 5, 5],
        ],
        "score": [0.9, 0.4, 0.8, 0.7, 0.6],
    }
)
GTS = pl.DataFrame(
    {
        "image_id": ["img1", "img1", "img2", "img3"],
        "class_id": ["cat", "dog", "cat", "dog"],
        "box": [[0, 0, 10, 10], [20, 0, 30, 10], [0, 0, 10, 10], [5, 5, 9, 9]],
    }
)


def _xywh(df: pl.DataFrame) -> pl.DataFrame:
    b = pl.col("box")
    return df.with_columns(
        box=pl.concat_list(
            b.list.get(0),
            b.list.get(1),
            b.list.get(2) - b.list.get(0),
            b.list.get(3) - b.list.get(1),
        )
    )


def _frames(table) -> tuple[pl.DataFrame, pl.DataFrame]:
    det, meta = table.collect()
    return det.sort(pl.all()), meta.sort(pl.all())


class TestGroupObjects:
    def test_the_population_is_every_image_crossed_with_every_class(self) -> None:
        out = group_objects(
            PREDS, GTS, geometry="box", images=["img1", "img2", "img3", "img4", "img5"]
        ).collect()
        assert out.height == 5 * 2
        img5 = out.filter(pl.col("image_id") == "img5")
        assert img5["pred"].list.len().to_list() == [0, 0]
        assert img5["gt"].list.len().to_list() == [0, 0]
        img4 = out.filter(pl.col("image_id") == "img4").sort("class_id")
        assert img4["pred"].list.len().to_list() == [0, 1]

    def test_predictions_are_ranked_and_capped(self) -> None:
        out = group_objects(PREDS, GTS, geometry="box", max_detections=1).collect()
        cats = out.filter(
            (pl.col("image_id") == "img1") & (pl.col("class_id") == "cat")
        )
        assert cats["pred_score"].to_list() == [[0.9]]
        assert cats["pred_row"].to_list() == [[0]]

    def test_missing_columns_are_named(self) -> None:
        with pytest.raises(ValueError, match="'score'"):
            group_objects(PREDS.drop("score"), GTS, geometry="box")
        with pytest.raises(TypeError, match="list of image ids"):
            group_objects(PREDS, GTS, geometry="box", images="img1")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


@plugin_required
class TestBoxes:
    def test_every_layout_gives_the_same_table(self) -> None:
        xyxy = match_detections(PREDS, GTS, geometry="box", box_format="xyxy")
        xywh = match_detections(
            _xywh(PREDS), _xywh(GTS), geometry="box", box_format="xywh"
        )
        cols = ["x1", "y1", "x2", "y2"]
        split = lambda df: df.with_columns(  # noqa: E731
            pl.col("box").list.to_struct(fields=cols)
        ).unnest("box")
        four = match_detections(
            split(PREDS), split(GTS), geometry=tuple(cols), box_format="xyxy"
        )
        from polars_cv.geometry import bbox_from_coords

        boxed = lambda df: df.with_columns(bbox_from_coords("box", format="xyxy"))  # noqa: E731
        struct = match_detections(boxed(PREDS), boxed(GTS), geometry="box")
        want = _frames(xyxy)
        for other in (xywh, four, struct):
            got = _frames(other)
            assert got[0].equals(want[0]) and got[1].equals(want[1])

    def test_the_table_is_population_complete(self) -> None:
        table = match_detections(PREDS, GTS, geometry="box", box_format="xyxy")
        det, meta = _frames(table)
        assert det.height == PREDS.height
        assert meta.height == 4 * 2  # four images seen, two classes
        assert meta["n_gts"].sum() == GTS.height
        # The dog predicted on img4 is a false positive; img3's dog is missed.
        dog = AP().value(table.filter_class("dog"))
        assert dog == pytest.approx(0.5)

    def test_the_format_is_explicit(self) -> None:
        with pytest.raises(ValueError, match="box_format"):
            match_detections(PREDS, GTS, geometry="box")
        with pytest.raises(ValueError, match="expected a BBOX_SCHEMA"):
            match_detections(
                PREDS.with_columns(box=pl.lit("x")),
                GTS.with_columns(box=pl.lit("x")),
                geometry="box",
            )

    def test_a_sweep(self) -> None:
        table = match_detections(
            PREDS, GTS, geometry="box", box_format="xyxy", iou_thresholds=[0.5, 0.75]
        )
        assert table.iou_thresholds == (0.5, 0.75)

    def test_images_frame_weights_and_groups(self) -> None:
        images = pl.DataFrame(
            {
                "image_id": ["img1", "img2", "img3", "img4", "img5"],
                "w": [2.0, 1.0, 1.0, 1.0, 1.0],
                "site": ["a", "a", "b", "b", "b"],
            }
        )
        table = match_detections(
            PREDS,
            GTS,
            geometry="box",
            box_format="xyxy",
            images=images,
            weight="w",
            group="site",
        )
        meta = table.image_metadata.collect()
        assert meta.height == 5 * 2
        assert meta.filter(pl.col("image_id") == "img1")["weight"].to_list() == [
            2.0,
            2.0,
        ]
        assert set(meta["group_id"]) == {"a", "b"}

    def test_det_idx_maps_back_to_the_input_row(self) -> None:
        table = match_detections(PREDS, GTS, geometry="box", box_format="xyxy")
        grouped = group_objects(PREDS, GTS, geometry="box").collect()
        det = table.detections.collect()
        rows = (
            det.join(grouped, on=["image_id", "class_id"])
            .select(pl.col("pred_row").list.get(pl.col("det_idx")), "score")
            .sort("pred_row")
        )
        assert rows["score"].to_list() == PREDS["score"].to_list()


def _square(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


@plugin_required
class TestPolygonsAndMasks:
    def _polygons(self, df: pl.DataFrame) -> pl.DataFrame:
        rings = [_square(*b) for b in df["box"].to_list()]
        return df.with_columns(
            poly=pl.Series(rings, dtype=pl.List(pl.List(pl.Float64)))
        ).with_columns(poly=contour_from_coords(pl.col("poly")))

    def test_rectangles_as_polygons_match_as_boxes(self) -> None:
        boxes = match_detections(PREDS, GTS, geometry="box", box_format="xyxy")
        polys = match_detections(
            self._polygons(PREDS), self._polygons(GTS), geometry="poly"
        )
        a, b = _frames(boxes)[0], _frames(polys)[0]
        assert a["is_tp"].to_list() == b["is_tp"].to_list()
        assert a["iou"].to_list() == pytest.approx(b["iou"].to_list())

    @staticmethod
    def _mask(
        boxes: list[tuple[int, int, int, int]], size: int = 64
    ) -> list[list[int]]:
        grid = [[0] * size for _ in range(size)]
        for x0, y0, x1, y1 in boxes:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    grid[y][x] = 1
        return grid

    def test_instance_masks_match_as_their_outlines(self) -> None:
        preds = PREDS.with_columns(
            mask=pl.Series([self._mask([tuple(b)]) for b in PREDS["box"].to_list()])
        )
        gts = GTS.with_columns(
            mask=pl.Series([self._mask([tuple(b)]) for b in GTS["box"].to_list()])
        )
        masks = match_detections(preds, gts, geometry="mask")
        boxes = match_detections(PREDS, GTS, geometry="box", box_format="xyxy")
        a, b = _frames(boxes)[0], _frames(masks)[0]
        assert a["is_tp"].to_list() == b["is_tp"].to_list()

    def test_a_mask_of_two_regions_is_refused(self) -> None:
        preds = PREDS.head(1).with_columns(
            mask=pl.Series([self._mask([(0, 0, 4, 4), (10, 10, 14, 14)])])
        )
        gts = GTS.head(1).with_columns(mask=pl.Series([self._mask([(0, 0, 4, 4)])]))
        table = match_detections(preds, gts, geometry="mask")
        with pytest.raises(pl.exceptions.ComputeError, match="'img1' holds 2 contours"):
            table.collect()

    def test_box_geometry_is_not_mixed_with_polygons(self) -> None:
        with pytest.raises(ValueError, match="in predictions but"):
            match_detections(
                self._polygons(PREDS),
                self._polygons(GTS).with_columns(poly=pl.col("box")),
                geometry="poly",
            )


def test_bbox_schema_constant_is_the_geometry_contract() -> None:
    assert BBOX_SCHEMA.to_schema() == {
        "x": pl.Float64,
        "y": pl.Float64,
        "width": pl.Float64,
        "height": pl.Float64,
    }
