"""The ``.bbox`` per-box accessors and ``nms``.

Each accessor is checked against a NumPy definition of the same quantity, on
a single-box column and on a box-set column (the arity is the column's, as for
``.contour``), with the declared schema equal to the data's. ``nms`` is held to
OpenCV in ``tests/reference/test_bbox_ref.py``; its refusals are here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import BBOX_SCHEMA, CONTOUR_SCHEMA
from polars_cv.geometry import BoxFormat, bbox_from_coords

from .conftest import plugin_required

pytestmark = plugin_required

_RNG = np.random.default_rng(7)
#: Integer-valued boxes: every accessor below is exact on them.
_XYWH = np.column_stack(
    [_RNG.integers(-20, 100, (12, 2)), _RNG.integers(0, 40, (12, 2))]
).astype(float)


def _box(row: np.ndarray) -> dict[str, float]:
    return dict(zip(("x", "y", "width", "height"), map(float, row), strict=True))


def _frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """The same boxes one per row, and as two sets (plus a null row each)."""
    single = pl.DataFrame(
        {"b": [_box(r) for r in _XYWH] + [None]}, schema={"b": BBOX_SCHEMA}
    )
    sets = pl.DataFrame(
        {"b": [[_box(r) for r in _XYWH[:5]], [_box(r) for r in _XYWH[5:]], None]},
        schema={"b": pl.List(BBOX_SCHEMA)},
    )
    return single, sets


def _xywh(value: dict) -> list[float]:
    return [value["x"], value["y"], value["width"], value["height"]]


#: accessor -> (call, per-box NumPy definition on [x, y, w, h], how a result
#: reads back as numbers).
_CASES: dict[str, tuple[Callable[[pl.Expr], pl.Expr], Callable, Callable]] = {
    "area": (lambda e: e.bbox.area(), lambda b: b[2] * b[3], float),
    "to_coords xyxy": (
        lambda e: e.bbox.to_coords("xyxy"),
        lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]],
        list,
    ),
    "to_coords xywh": (lambda e: e.bbox.to_coords("xywh"), list, list),
    "to_coords cxcywh": (
        lambda e: e.bbox.to_coords(BoxFormat.CXCYWH),
        lambda b: [b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]],
        list,
    ),
    "translate": (
        lambda e: e.bbox.translate(dx=3.0, dy=-2.0),
        lambda b: [b[0] + 3, b[1] - 2, b[2], b[3]],
        _xywh,
    ),
    "scale": (
        lambda e: e.bbox.scale(sx=2.0, sy=0.5),
        lambda b: [b[0] * 2, b[1] * 0.5, b[2] * 2, b[3] * 0.5],
        _xywh,
    ),
    "normalize": (
        lambda e: e.bbox.normalize(width=128.0, height=64.0),
        lambda b: [b[0] / 128, b[1] / 64, b[2] / 128, b[3] / 64],
        _xywh,
    ),
    "to_absolute": (
        lambda e: e.bbox.to_absolute(width=128.0, height=64.0),
        lambda b: [b[0] * 128, b[1] * 64, b[2] * 128, b[3] * 64],
        _xywh,
    ),
    "clip": (
        lambda e: e.bbox.clip(width=50.0, height=40.0),
        lambda b: [
            np.clip(b[0], 0, 50),
            np.clip(b[1], 0, 40),
            np.clip(b[0] + b[2], 0, 50) - np.clip(b[0], 0, 50),
            np.clip(b[1] + b[3], 0, 40) - np.clip(b[1], 0, 40),
        ],
        _xywh,
    ),
    "pad": (
        lambda e: e.bbox.pad(4.0),
        lambda b: [b[0] - 4, b[1] - 4, b[2] + 8, b[3] + 8],
        _xywh,
    ),
    "expand": (
        lambda e: e.bbox.expand(1.5),
        lambda b: [
            b[0] + b[2] / 2 - b[2] * 0.75,
            b[1] + b[3] / 2 - b[3] * 0.75,
            b[2] * 1.5,
            b[3] * 1.5,
        ],
        _xywh,
    ),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_each_accessor_is_its_definition_on_boxes_and_on_sets(name: str) -> None:
    call, definition, numbers = _CASES[name]
    single, sets = _frames()
    want = [np.asarray(definition(b), dtype=float) for b in _XYWH]

    out = single.select(call(pl.col("b")))
    assert (
        out.lazy().collect_schema()
        == single.lazy().select(call(pl.col("b"))).collect_schema()
    )
    *values, null = out["b"].to_list()
    assert null is None
    np.testing.assert_allclose([numbers(v) for v in values], want, rtol=1e-12)

    out = sets.select(call(pl.col("b")))
    assert out.schema == sets.lazy().select(call(pl.col("b"))).collect_schema()
    first, second, null = out["b"].to_list()
    assert null is None
    got = [numbers(v) for v in first + second]
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("fmt", [f.value for f in BoxFormat])
def test_to_coords_and_bbox_from_coords_round_trip_exactly(fmt: str) -> None:
    single, _ = _frames()
    back = single.select(
        bbox_from_coords(pl.col("b").bbox.to_coords(fmt), format=fmt).alias("b")
    )
    assert back.equals(single)


def test_to_contour_is_the_box_as_a_region() -> None:
    single, _ = _frames()
    out = single.select(
        pl.col("b").bbox.to_contour().alias("c"),
        pl.col("b").bbox.area().alias("a"),
    )
    assert out.schema["c"] == CONTOUR_SCHEMA
    region = out.select(
        pl.col("c").contour.area().alias("area"),
        pl.col("c").contour.bounding_box().alias("box"),
    )
    assert region["area"].to_list() == out["a"].to_list()
    assert region["box"].to_list()[:-1] == single["b"].to_list()[:-1]


def test_a_parameter_may_be_per_row() -> None:
    single, _ = _frames()
    dx = pl.Series("dx", np.arange(len(single), dtype=float))
    out = single.with_columns(dx).select(
        pl.col("b").bbox.translate(dx=pl.col("dx"), dy=0.0)
    )
    xs = [v["x"] for v in out["b"].to_list()[:-1]]
    assert xs == list(_XYWH[:, 0] + np.arange(len(_XYWH)))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda e: e.bbox.scale(sx=-1.0, sy=1.0), "at least 0"),
        (lambda e: e.bbox.expand(-0.5), "at least 0"),
        (lambda e: e.bbox.pad(-1000.0), "negative size"),
        (lambda e: e.bbox.normalize(width=0.0, height=10.0), "positive"),
        (lambda e: e.bbox.clip(width=-1.0, height=10.0), "at least 0"),
        (lambda e: e.bbox.to_absolute(width=-1.0, height=10.0), "at least 0"),
    ],
)
def test_a_result_no_box_can_have_is_refused(call: Callable, message: str) -> None:
    single, _ = _frames()
    with pytest.raises(pl.exceptions.ComputeError, match=message):
        single.select(call(pl.col("b")))


def test_an_unknown_format_is_refused_naming_the_formats() -> None:
    single, _ = _frames()
    with pytest.raises((ValueError, pl.exceptions.ComputeError), match="xyxy"):
        single.select(pl.col("b").bbox.to_coords("yxyx"))


class TestNms:
    def _frame(self, boxes, scores) -> pl.DataFrame:
        return pl.DataFrame(
            {"b": [boxes], "s": [scores]},
            schema={"b": pl.List(BBOX_SCHEMA), "s": pl.List(pl.Float64)},
        )

    def test_kept_positions_select_the_other_columns(self) -> None:
        boxes = [_box([0, 0, 10, 10]), _box([1, 1, 10, 10]), _box([50, 50, 5, 5])]
        df = self._frame(boxes, [0.8, 0.9, 0.1]).with_columns(
            labels=pl.lit(["a", "b", "c"])
        )
        out = df.with_columns(kept=pl.col("b").bbox.nms(pl.col("s"))).select(
            pl.col("kept"), pl.col("labels").list.gather(pl.col("kept"))
        )
        assert out.to_dicts() == [{"kept": [1, 2], "labels": ["b", "c"]}]

    def test_one_box_per_row_is_refused(self) -> None:
        df = pl.DataFrame(
            {"b": [_box([0, 0, 1, 1])], "s": [[1.0]]},
            schema={"b": BBOX_SCHEMA, "s": pl.List(pl.Float64)},
        )
        with pytest.raises(pl.exceptions.ComputeError, match="set of boxes"):
            df.select(pl.col("b").bbox.nms(pl.col("s")))

    @pytest.mark.parametrize(
        ("scores", "message"),
        [
            ([0.5], "one score per box"),
            ([0.5, None], "must not be null"),
            ([0.5, float("nan")], "finite"),
        ],
    )
    def test_scores_must_match_the_boxes(self, scores, message) -> None:
        df = self._frame([_box([0, 0, 1, 1]), _box([5, 5, 1, 1])], scores)
        with pytest.raises(pl.exceptions.ComputeError, match=message):
            df.select(pl.col("b").bbox.nms(pl.col("s")))

    def test_a_null_box_in_the_set_is_refused(self) -> None:
        df = self._frame([_box([0, 0, 1, 1]), None], [0.5, 0.4])
        with pytest.raises(pl.exceptions.ComputeError, match="every box must be"):
            df.select(pl.col("b").bbox.nms(pl.col("s")))

    @pytest.mark.parametrize("threshold", [-0.1, 1.5])
    def test_a_threshold_outside_the_unit_interval_is_refused(self, threshold) -> None:
        df = self._frame([_box([0, 0, 1, 1])], [0.5])
        with pytest.raises(pl.exceptions.ComputeError, match=r"\[0, 1\]"):
            df.select(pl.col("b").bbox.nms(pl.col("s"), iou_threshold=threshold))

    def test_a_null_row_is_null(self) -> None:
        df = pl.DataFrame(
            {"b": [None], "s": [None]},
            schema={"b": pl.List(BBOX_SCHEMA), "s": pl.List(pl.Float64)},
        )
        assert df.select(pl.col("b").bbox.nms(pl.col("s")))["b"].to_list() == [None]


class TestCropBox:
    """``Pipeline.crop_box``: the smallest whole-pixel window containing each
    row's box, through ``crop_clip``."""

    _IMG = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)

    def _crop(self, boxes: list) -> list:
        import io

        from PIL import Image

        from polars_cv import Pipeline, numpy_from_column

        buf = io.BytesIO()
        Image.fromarray(self._IMG).save(buf, "PNG")
        df = pl.DataFrame(
            {"img": [buf.getvalue()] * len(boxes), "bbox": boxes},
            schema={"img": pl.Binary, "bbox": BBOX_SCHEMA},
        )
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .crop_box(pl.col("bbox"))
            .on_null_param("null")
        )
        out = df.select(pl.col("img").cv.pipe(pipe).sink("tensor"))["img"]
        return numpy_from_column(out)

    def test_a_fractional_box_takes_every_pixel_it_touches(self) -> None:
        (got,) = self._crop([_box([2.5, 3.2, 4.0, 5.0])])
        np.testing.assert_array_equal(got, self._IMG[3:9, 2:7])

    def test_a_box_past_the_edge_keeps_the_part_inside_and_a_null_box_is_null(
        self,
    ) -> None:
        clipped, null = self._crop([_box([25.0, -4.0, 10.0, 10.0]), None])
        np.testing.assert_array_equal(clipped, self._IMG[0:6, 25:30])
        assert null is None

    def test_a_box_with_no_pixel_inside_the_image_is_refused(self) -> None:
        with pytest.raises(pl.exceptions.ComputeError):
            self._crop([_box([100.0, 100.0, 5.0, 5.0])])

    def test_a_negative_size_box_is_refused_as_every_reader_refuses_it(self) -> None:
        with pytest.raises(pl.exceptions.ComputeError, match="negative width"):
            self._crop([_box([10.0, 10.0, -5.0, 5.0])])
