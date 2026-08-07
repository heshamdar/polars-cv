"""
Tests for source('contour') pipeline source type.

These tests verify that contours can be used as pipeline sources,
with rasterization happening inside the pipeline execution.

A geometry column comes in two shapes and the source reads both: one contour
per row (`CONTOUR_SCHEMA`), and a whole set per row (`List(CONTOUR_SCHEMA)`) —
the shape `extract_contours().sink("native")` emits, which is what closes the
mask -> contours -> mask loop. `TestContourSetSource` covers the set;
`TestMaskContourRoundTrip` closes the loop through a real image column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from polars_cv.geometry import CONTOUR_SCHEMA
from tests.conftest import plugin_required

if TYPE_CHECKING:
    from collections.abc import Callable

CONTOUR_SET_SCHEMA = pl.List(CONTOUR_SCHEMA)


def create_square_contour(x: float, y: float, size: float) -> dict:
    """Create a square contour at given position."""
    return {
        "exterior": [
            {"x": x, "y": y},
            {"x": x + size, "y": y},
            {"x": x + size, "y": y + size},
            {"x": x, "y": y + size},
        ],
        "holes": [],
        "is_closed": True,
    }


def create_triangle_contour(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
) -> dict:
    """Create a triangle contour."""
    return {
        "exterior": [
            {"x": x1, "y": y1},
            {"x": x2, "y": y2},
            {"x": x3, "y": y3},
        ],
        "holes": [],
        "is_closed": True,
    }


def _ring(points: list[tuple[float, float]]) -> list[dict]:
    return [{"x": float(x), "y": float(y)} for x, y in points]


def _square_with_hole(x: float, y: float, size: float, hole: float) -> dict:
    """A square with a concentric square hole, both on integer coordinates."""
    inset = (size - hole) / 2
    return {
        "exterior": _ring([(x, y), (x + size, y), (x + size, y + size), (x, y + size)]),
        "holes": [
            _ring(
                [
                    (x + inset, y + inset),
                    (x + inset + hole, y + inset),
                    (x + inset + hole, y + inset + hole),
                    (x + inset, y + inset + hole),
                ]
            )
        ],
        "is_closed": True,
    }


def _rasterize(column: pl.Series, **source_kwargs) -> np.ndarray:
    """Rasterize a geometry column through `source("contour")`."""
    pipe = Pipeline().source("contour", **source_kwargs)
    frame = pl.DataFrame({"c": column})
    return numpy_from_struct(
        frame.select(m=pl.col("c").cv.pipe(pipe).sink("numpy"))["m"][0]
    )


class TestContourSourceExplicitDims:
    """Tests for contour source with explicit width/height dimensions."""

    def test_basic_rasterization(self) -> None:
        """Basic contour rasterization produces correct output shape."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = Pipeline().source("contour", width=100, height=100)
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))

        assert isinstance(result["mask"].dtype, pl.Struct)
        assert result["mask"].len() == 1

        # Parse the numpy output
        mask_bytes = result["mask"][0]
        arr = numpy_from_struct(mask_bytes)
        assert arr.shape == (100, 100, 1)
        assert arr.dtype == np.uint8

    def test_rasterization_with_fill_values(self) -> None:
        """Fill value and background parameters work correctly."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = Pipeline().source(
            "contour", width=100, height=100, fill_value=128, background=64
        )
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))

        arr = numpy_from_struct(result["mask"][0])

        # Check that we have the expected fill values
        # Inside contour should be 128, outside should be 64
        assert np.any(arr == 128), "Should have pixels with fill_value=128"
        assert np.any(arr == 64), "Should have pixels with background=64"

    def test_rasterization_multiple_contours(self) -> None:
        """Multiple contours can be rasterized in a single operation."""
        df = pl.DataFrame(
            {
                "contour": [
                    create_square_contour(10, 10, 20),
                    create_triangle_contour(50, 50, 80, 50, 65, 80),
                ],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = Pipeline().source("contour", width=100, height=100)
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))

        assert result["mask"].len() == 2

        # Both should produce valid arrays
        arr1 = numpy_from_struct(result["mask"][0])
        arr2 = numpy_from_struct(result["mask"][1])

        assert arr1.shape == (100, 100, 1)
        assert arr2.shape == (100, 100, 1)

    def test_rasterization_with_operations(self) -> None:
        """Contour rasterization followed by pipeline operations."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        # Rasterize and then blur
        pipe = Pipeline().source("contour", width=100, height=100).blur(2.0)
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))

        arr = numpy_from_struct(result["mask"][0])
        assert arr.shape == (100, 100, 1)

        # Blurred mask should have gradual transitions (not just 0 and 255)
        unique_values = np.unique(arr)
        assert len(unique_values) > 2, "Blurred mask should have smooth transitions"

    def test_rasterization_resize(self) -> None:
        """Contour rasterization followed by resize."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = (
            Pipeline()
            .source("contour", width=100, height=100)
            .resize(width=50, height=50)
        )
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))

        arr = numpy_from_struct(result["mask"][0])
        assert arr.shape == (50, 50, 1)


class TestContourSourceDynamicDims:
    """Tests for contour source with dynamic dimensions from columns."""

    def test_dynamic_width_height(self) -> None:
        """Width and height from column expressions."""
        df = pl.DataFrame(
            {
                "contour": [
                    create_square_contour(5, 5, 20),
                    create_square_contour(10, 10, 30),
                ],
                "w": [50, 100],
                "h": [50, 100],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = Pipeline().source("contour", width=pl.col("w"), height=pl.col("h"))
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))

        # First row: 50x50
        arr1 = numpy_from_struct(result["mask"][0])
        assert arr1.shape == (50, 50, 1)

        # Second row: 100x100
        arr2 = numpy_from_struct(result["mask"][1])
        assert arr2.shape == (100, 100, 1)


class TestContourSourceValidation:
    """Tests for contour source validation."""

    def test_missing_dimensions_error(self) -> None:
        """Error when neither dimensions nor shape provided."""
        with pytest.raises(ValueError, match="Contour source requires"):
            Pipeline().source("contour")

    def test_partial_dimensions_error(self) -> None:
        """Error when only width or only height provided."""
        with pytest.raises(ValueError, match="must be specified together"):
            Pipeline().source("contour", width=100)

        with pytest.raises(ValueError, match="must be specified together"):
            Pipeline().source("contour", height=100)

    def test_both_shape_and_dims_error(self) -> None:
        """Error when both shape and explicit dimensions provided."""
        # This should work (just width/height)
        Pipeline().source("contour", width=100, height=100)

        # Can't easily test the shape + dims conflict without a real LazyPipelineExpr


class TestContourSourceNullHandling:
    """Tests for null handling in contour source."""

    def test_null_contour_produces_null_output(self) -> None:
        """Null contours should produce null outputs."""
        df = pl.DataFrame(
            {
                "contour": [
                    create_square_contour(10, 10, 50),
                    None,
                    create_triangle_contour(20, 20, 60, 20, 40, 60),
                ],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = Pipeline().source("contour", width=100, height=100)
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))

        # First and third rows should have data
        assert result["mask"][0].get("data") is not None
        # Second row should have null fields
        assert result["mask"][1].get("data") is None
        # Third row should have data
        assert result["mask"][2].get("data") is not None


@plugin_required
class TestContourSetSource:
    """`source("contour")` on a `List(CONTOUR_SCHEMA)` column.

    A contour set rasterizes to the **union** of its members' regions — the same
    thing the in-pipeline `rasterize()` op does with the set `extract_contours()`
    produces, so a mask does not depend on which of the two routes made it.

    Before contour sets were accepted, feeding one in failed with
    ``Point struct missing 'x' field``: the list was read as a single contour's
    ring of points, and a contour struct is not a point struct.
    """

    def test_two_contours_paint_both_regions(self) -> None:
        """Disjoint squares on integer coordinates count exactly."""
        column = pl.Series(
            "c",
            [[create_square_contour(10, 10, 30), create_square_contour(60, 60, 40)]],
            dtype=CONTOUR_SET_SCHEMA,
        )
        arr = _rasterize(column, width=128, height=128, fill_value=1, background=0)

        assert arr.shape == (128, 128, 1)
        assert int(arr.sum()) == 30 * 30 + 40 * 40

    def test_a_single_element_set_matches_the_bare_struct(self) -> None:
        """The set of one and the lone contour are the same mask."""
        contour = create_square_contour(10, 10, 50)
        as_struct = _rasterize(
            pl.Series("c", [contour], dtype=CONTOUR_SCHEMA), width=100, height=100
        )
        as_set = _rasterize(
            pl.Series("c", [[contour]], dtype=CONTOUR_SET_SCHEMA), width=100, height=100
        )

        np.testing.assert_array_equal(as_struct, as_set)

    def test_holes_survive_and_stay_local_to_their_contour(self) -> None:
        """One member's hole cuts its own region, not its neighbour's.

        The hole here is covered by nothing else, so the count is exact; the
        neighbour is disjoint from it. A set painted contour-by-contour into one
        canvas would give the same answer only for some orderings, which is why
        `test_the_mask_does_not_depend_on_the_sets_order` exists alongside this.
        """
        column = pl.Series(
            "c",
            [[_square_with_hole(10, 10, 50, 20), create_square_contour(70, 70, 40)]],
            dtype=CONTOUR_SET_SCHEMA,
        )
        arr = _rasterize(column, width=128, height=128, fill_value=1, background=0)

        assert int(arr.sum()) == (50 * 50 - 20 * 20) + 40 * 40

    def test_the_mask_does_not_depend_on_the_sets_order(self) -> None:
        """An overlapping pair, both ways round.

        The second contour's hole overlaps the first's fill, so sequential
        painting would erase pixels the union keeps — and only in one of the two
        orderings.
        """
        holed = _square_with_hole(10, 10, 50, 30)
        overlapping = create_square_contour(30, 30, 40)

        forward = _rasterize(
            pl.Series("c", [[holed, overlapping]], dtype=CONTOUR_SET_SCHEMA),
            width=128,
            height=128,
        )
        reversed_ = _rasterize(
            pl.Series("c", [[overlapping, holed]], dtype=CONTOUR_SET_SCHEMA),
            width=128,
            height=128,
        )

        np.testing.assert_array_equal(forward, reversed_)
        # A pixel in the hole that the overlapping square covers stays filled.
        assert forward[35, 35, 0] == 255
        # A pixel in the hole that nothing else covers does not.
        assert forward[22, 22, 0] == 0

    def test_inverted_fill_and_background(self) -> None:
        """`fill_value < background` paints the same region, inverted.

        Folding per-contour masks together with `max` — the graph executor's old
        approach — returned an all-background canvas here as soon as the set held
        more than one contour.
        """
        contours = [
            create_square_contour(10, 10, 30),
            create_square_contour(60, 60, 40),
        ]
        column = pl.Series("c", [contours], dtype=CONTOUR_SET_SCHEMA)

        upright = _rasterize(
            column, width=128, height=128, fill_value=255, background=0
        )
        inverted = _rasterize(
            column, width=128, height=128, fill_value=0, background=255
        )

        np.testing.assert_array_equal(upright, 255 - inverted)
        assert int((inverted == 0).sum()) == 30 * 30 + 40 * 40

    def test_empty_set_is_all_background(self) -> None:
        """No contours is a valid set: nothing is painted."""
        column = pl.Series("c", [[]], dtype=CONTOUR_SET_SCHEMA)
        arr = _rasterize(column, width=32, height=32, fill_value=255, background=7)

        assert arr.shape == (32, 32, 1)
        assert np.all(arr == 7)

    def test_null_members_are_skipped(self) -> None:
        """A null contour inside the set drops out; the rest still paint."""
        column = pl.Series(
            "c",
            [[create_square_contour(10, 10, 30), None]],
            dtype=CONTOUR_SET_SCHEMA,
        )
        arr = _rasterize(column, width=128, height=128, fill_value=1, background=0)

        assert int(arr.sum()) == 30 * 30

    def test_a_null_row_is_still_null(self) -> None:
        """A null *row* nulls the output, as it does for a lone contour."""
        column = pl.Series(
            "c",
            [[create_square_contour(10, 10, 30)], None],
            dtype=CONTOUR_SET_SCHEMA,
        )
        pipe = Pipeline().source("contour", width=32, height=32)
        result = pl.DataFrame({"c": column}).with_columns(
            m=pl.col("c").cv.pipe(pipe).sink("numpy")
        )

        assert result["m"][0].get("data") is not None
        assert result["m"][1].get("data") is None

    def test_a_bare_ring_of_points_is_still_one_contour(self) -> None:
        """`List[{x, y}]` keeps meaning a single contour, not a set.

        The list forms are told apart by element dtype, so admitting the contour
        set did not turn a ring of points into a set of degenerate contours.
        """
        square = create_square_contour(10, 10, 30)
        ring = pl.Series(
            "c",
            [square["exterior"]],
            dtype=pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64})),
        )
        arr = _rasterize(ring, width=128, height=128, fill_value=1, background=0)

        assert int(arr.sum()) == 30 * 30


@plugin_required
class TestMaskContourRoundTrip:
    """mask -> `extract_contours()` -> `List[Contour]` column -> mask.

    The leg that closes the loop is the sink's output being re-readable by the
    source. It is lossy in one known direction — the tracer walks the *centres*
    of the boundary pixels, so each region comes back inset by half a pixel all
    round (see `test_contour_raster_crosscheck.TestRoundTripThroughExtraction`) —
    and the assertions below are written around that inset rather than a
    tolerance wide enough to hide a real shift.
    """

    CANVAS = 128
    BOXES = ((10, 40, 10, 40), (70, 110, 70, 110))  # (y0, y1, x0, x1)

    def _image(self, encode_png: "Callable") -> bytes:
        image = np.zeros((self.CANVAS, self.CANVAS, 3), dtype=np.uint8)
        for y0, y1, x0, x1 in self.BOXES:
            image[y0:y1, x0:x1] = 255
        return encode_png(image)

    def _contour_sets(self, encode_png: "Callable") -> pl.Series:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours(mode="external", method="simple")
        )
        frame = pl.DataFrame({"img": [self._image(encode_png)]})
        return frame.select(c=pl.col("img").cv.pipe(pipe).sink("native"))["c"]

    def test_the_sinks_output_feeds_the_source_unchanged(
        self, encode_png: "Callable"
    ) -> None:
        """No reshaping between the two ends: the column goes straight back in."""
        sets = self._contour_sets(encode_png)
        assert sets.dtype == CONTOUR_SET_SCHEMA
        assert len(sets[0]) == len(self.BOXES)

        arr = _rasterize(sets, width=self.CANVAS, height=self.CANVAS, fill_value=1)

        # Each region returns as its own (w-1) x (h-1) inset box.
        expected = sum((y1 - y0 - 1) * (x1 - x0 - 1) for y0, y1, x0, x1 in self.BOXES)
        assert int(arr.sum()) == expected

    def test_the_returned_mask_sits_inside_the_original(
        self, encode_png: "Callable"
    ) -> None:
        """Inset, never outset — every returned pixel was set in the original."""
        sets = self._contour_sets(encode_png)
        returned = (
            _rasterize(sets, width=self.CANVAS, height=self.CANVAS, fill_value=1)[
                :, :, 0
            ]
            > 0
        )

        original = np.zeros((self.CANVAS, self.CANVAS), dtype=bool)
        for y0, y1, x0, x1 in self.BOXES:
            original[y0:y1, x0:x1] = True

        assert not (returned & ~original).any()
        iou = (returned & original).sum() / (returned | original).sum()
        assert iou > 0.9

    def test_the_two_routes_to_a_mask_agree(self, encode_png: "Callable") -> None:
        """Rasterizing in the graph and rasterizing from a column give one mask.

        `extract_contours().rasterize()` keeps the set inside a single pipeline;
        sinking to `native` and reading it back through `source("contour")` sends
        it out through Polars and in again. Neither is allowed to be the odd one
        out.
        """
        frame = pl.DataFrame({"img": [self._image(encode_png)]})
        in_graph = numpy_from_struct(
            frame.select(
                m=pl.col("img")
                .cv.pipe(
                    Pipeline()
                    .source("image_bytes")
                    .grayscale()
                    .threshold(128)
                    .extract_contours(mode="external", method="simple")
                    .rasterize(width=self.CANVAS, height=self.CANVAS)
                )
                .sink("numpy")
            )["m"][0]
        )
        via_column = _rasterize(
            self._contour_sets(encode_png), width=self.CANVAS, height=self.CANVAS
        )

        np.testing.assert_array_equal(in_graph, via_column)


class TestContourSourceIntegration:
    """Integration tests combining contour source with other features."""

    def test_contour_to_threshold(self) -> None:
        """Rasterize contour and apply threshold."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = Pipeline().source("contour", width=100, height=100).threshold(128)
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))

        arr = numpy_from_struct(result["mask"][0])

        # Threshold should produce binary output
        unique_values = set(np.unique(arr))
        assert unique_values.issubset({0, 255}), f"Expected binary, got {unique_values}"
