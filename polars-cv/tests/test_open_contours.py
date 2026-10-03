"""Open contours (``is_closed = False``) are polylines, and are measured as such.

``is_closed`` used to be written ``True`` and never read: an open polyline was
measured as the ring its closing edge would make, so the point ``(0, 10)`` was
7.07 from the L ``(0,0) -> (10,0) -> (10,10)`` instead of 10. Now every
``.contour`` accessor (and every ``.point`` function of a contour) either
measures the boundary an open contour has, or refuses it, because the
function needs a region the polyline does not bound. The table below decides
which, and is completeness-asserted against the namespaces' case tables, so a
new accessor cannot join without deciding.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import CONTOUR_SCHEMA, CONTOUR_SET_SCHEMA, POINT_SCHEMA

from .conftest import plugin_required
from .test_schema_parity_namespaces import CONTOUR_CASES, POINT_CASES, _square

_L = [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 0.0}, {"x": 10.0, "y": 10.0}]


def _l_shape(*, closed: bool) -> dict:
    return {"exterior": _L, "holes": [], "is_closed": closed}


#: Every case that reads a contour: ``"outline"`` measures an open one's
#: boundary, ``"region"`` refuses it (no area, overlap, inside or centroid).
READS: dict[str, str] = {
    "area": "region",
    "centroid": "region",
    "is_convex": "region",
    "winding": "region",
    "ensure_winding": "region",
    "contains_point": "region",
    "iou": "region",
    "dice": "region",
    "pairwise_iou": "region",
    "largest": "region",
    "correspond": "region",
    "correspond_by_coverage": "outline",
    "perimeter": "outline",
    "bounding_box": "outline",
    "hausdorff_distance": "outline",
    "boundary_distances": "outline",
    "translate": "outline",
    "scale": "outline",
    "simplify": "outline",
    "convex_hull": "outline",
    "normalize": "outline",
    "to_absolute": "outline",
    "flip": "outline",
    "to_coords": "outline",
    "close_along_border": "outline",
    "point.distance_to_contour": "outline",
    "point.nearest_point_on_contour": "outline",
    "point.signed_distance_to_contour": "region",
}

_CASES = {
    **CONTOUR_CASES,
    **{
        f"point.{name}": case
        for name, case in POINT_CASES.items()
        if name.endswith("_contour")
    },
}


#: An outline case whose default spelling needs a region: ``scale`` defaults to
#: the centroid, which a polyline lacks (refused, see
#: ``test_scaling_a_polyline_about_its_centroid_is_refused``).
_OPEN_SPELLING = {
    "scale": lambda: pl.col("a").contour.scale(2.0, 2.0, origin="bbox_center"),
    # The case table reads its open-line column; here every column is open.
    "close_along_border": lambda: pl.col("a").contour.close_along_border(10, 10),
}


def test_every_contour_reader_is_classified() -> None:
    missing = set(_CASES) - set(READS)
    stale = set(READS) - set(_CASES)
    assert not missing, f"unclassified contour readers: {sorted(missing)}"
    assert not stale, f"classified readers that no longer exist: {sorted(stale)}"


def _open_df() -> pl.DataFrame:
    """The namespaces' case frame with every contour open (sets included)."""
    open_square = {**_square(0, 0, 10), "is_closed": False}
    return pl.DataFrame(
        {
            "a": [open_square],
            "b": [open_square],
            "aset": [[open_square]],
            "bset": [[open_square]],
            "pa": [{"x": 1.0, "y": 2.0}],
            "pb": [{"x": 3.0, "y": 4.0}],
        },
        schema={
            "a": CONTOUR_SCHEMA,
            "b": CONTOUR_SCHEMA,
            "aset": CONTOUR_SET_SCHEMA,
            "bset": CONTOUR_SET_SCHEMA,
            "pa": POINT_SCHEMA,
            "pb": POINT_SCHEMA,
        },
    )


@plugin_required
@pytest.mark.parametrize(
    "name", sorted(n for n, kind in READS.items() if kind == "region")
)
def test_a_region_function_refuses_an_open_contour(name: str) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="open contour"):
        _open_df().select(_CASES[name]())


@plugin_required
@pytest.mark.parametrize(
    "name", sorted(n for n, kind in READS.items() if kind == "outline")
)
def test_a_boundary_function_measures_an_open_contour(name: str) -> None:
    out = _open_df().select(_OPEN_SPELLING.get(name, _CASES[name])())
    assert out.height == 1
    assert out.to_series().null_count() == 0


@plugin_required
class TestOpenPolylineMeasures:
    """The reporter's L: an open polyline has no closing edge."""

    def _frame(self, *, closed: bool) -> pl.DataFrame:
        return pl.DataFrame(
            {"c": [_l_shape(closed=closed)], "p": [{"x": 0.0, "y": 10.0}]},
            schema={"c": CONTOUR_SCHEMA, "p": POINT_SCHEMA},
        )

    def test_distance_to_an_open_polyline(self) -> None:
        d = pl.col("p").point.distance_to_contour(pl.col("c"))
        assert self._frame(closed=False).select(d).item() == pytest.approx(10.0)
        assert self._frame(closed=True).select(d).item() == pytest.approx(50**0.5)

    def test_nearest_point_on_an_open_polyline(self) -> None:
        near = pl.col("p").point.nearest_point_on_contour(pl.col("c"))
        assert self._frame(closed=False).select(near).item() == {"x": 10.0, "y": 10.0}
        assert self._frame(closed=True).select(near).item() == {"x": 5.0, "y": 5.0}

    def test_perimeter_of_an_open_polyline(self) -> None:
        length = pl.col("c").contour.perimeter()
        assert self._frame(closed=False).select(length).item() == pytest.approx(20.0)
        assert self._frame(closed=True).select(length).item() == pytest.approx(
            20.0 + 200**0.5
        )

    @pytest.mark.parametrize(
        "transform",
        [
            lambda c: c.contour.translate(1.0, 1.0),
            lambda c: c.contour.scale(2.0, 2.0, origin="bbox_center"),
            lambda c: c.contour.simplify(0.5),
            lambda c: c.contour.flip(),
            lambda c: c.contour.normalize(10, 10),
            lambda c: c.contour.to_absolute(10, 10),
        ],
    )
    def test_a_transform_keeps_a_polyline_open(self, transform) -> None:
        out = self._frame(closed=False).select(transform(pl.col("c"))).item()
        assert out["is_closed"] is False
        assert len(out["exterior"]) == 3

    def test_the_hull_of_a_polyline_is_a_region(self) -> None:
        hull = self._frame(closed=False).select(pl.col("c").contour.convex_hull())
        assert hull.item()["is_closed"] is True

    def test_scaling_a_polyline_about_its_centroid_is_refused(self) -> None:
        with pytest.raises(pl.exceptions.ComputeError, match="centroid"):
            self._frame(closed=False).select(
                pl.col("c").contour.scale(2.0, 2.0, origin="centroid")
            )


@plugin_required
class TestOpenContourReading:
    def test_an_open_contour_with_holes_is_refused(self) -> None:
        df = pl.DataFrame(
            {"c": [{"exterior": _L, "holes": [_L], "is_closed": False}]},
            schema={"c": CONTOUR_SCHEMA},
        )
        with pytest.raises(pl.exceptions.ComputeError, match="has holes"):
            df.select(pl.col("c").contour.perimeter())

    def test_an_unspecified_is_closed_is_closed(self) -> None:
        """A dict without the key becomes a null under ``CONTOUR_SCHEMA``; like
        an unspecified ``holes`` (none), it means the default: closed."""
        df = pl.DataFrame(
            {"c": [{"exterior": _L, "holes": []}]}, schema={"c": CONTOUR_SCHEMA}
        )
        assert df["c"].struct.field("is_closed").null_count() == 1
        assert df.select(pl.col("c").contour.area()).item() == pytest.approx(50.0)

    def test_a_contour_without_the_field_is_closed(self) -> None:
        df = pl.DataFrame({"c": [{"exterior": _L}]})
        assert df.select(pl.col("c").contour.area()).item() == pytest.approx(50.0)

    def test_the_pipeline_contour_source_refuses_an_open_contour(self) -> None:
        """The pipeline's contour domain holds regions: rasterizing a polyline
        has no region to fill."""
        from polars_cv import Pipeline

        df = pl.DataFrame(
            {"c": [[_l_shape(closed=False)]]}, schema={"c": CONTOUR_SET_SCHEMA}
        )
        pipe = Pipeline().source("contour").area()
        with pytest.raises(pl.exceptions.ComputeError, match="open contour"):
            df.select(pl.col("c").cv.pipe(pipe).sink("native"))


@plugin_required
class TestCloseAlongBorder:
    """An open line ending on the image frame, closed along it into a region."""

    @staticmethod
    def _line(points: list[tuple[float, float]]) -> pl.DataFrame:
        ring = [{"x": x, "y": y} for x, y in points]
        return pl.DataFrame(
            {"c": [{"exterior": ring, "holes": [], "is_closed": False}]},
            schema={"c": CONTOUR_SCHEMA},
        )

    def test_a_pectoral_edge_closes_into_its_corner(self) -> None:
        df = self._line([(60.0, 0.0), (30.0, 20.0), (0.0, 40.0)])
        region = pl.col("c").contour.close_along_border(100, 100)
        out = df.select(region.contour.area()).item()
        assert out == pytest.approx(1200.0)
        other = pl.col("c").contour.close_along_border(100, 100, arc="counterclockwise")
        assert df.select(other.contour.area()).item() == pytest.approx(8800.0)
        assert df.select(region).item()["is_closed"] is True

    def test_the_frame_may_be_per_row(self) -> None:
        df = self._line([(60.0, 0.0), (0.0, 40.0)]).with_columns(
            w=pl.lit(100.0), h=pl.lit(100.0)
        )
        out = df.select(
            pl.col("c")
            .contour.close_along_border(pl.col("w"), pl.col("h"))
            .contour.area()
        )
        assert out.item() == pytest.approx(1200.0)

    def test_an_end_off_the_frame_is_refused(self) -> None:
        df = self._line([(60.0, 5.0), (0.0, 40.0)])
        with pytest.raises(pl.exceptions.ComputeError, match="max_snap"):
            df.select(pl.col("c").contour.close_along_border(100, 100))
        ok = df.select(pl.col("c").contour.close_along_border(100, 100, max_snap=6.0))
        assert ok.item()["exterior"][-1] == {"x": 60.0, "y": 0.0}

    def test_a_closed_contour_is_refused(self) -> None:
        df = pl.DataFrame({"c": [_l_shape(closed=True)]}, schema={"c": CONTOUR_SCHEMA})
        with pytest.raises(pl.exceptions.ComputeError, match="already closed"):
            df.select(pl.col("c").contour.close_along_border(10, 10))

    def test_a_nan_max_snap_is_refused(self) -> None:
        """A per-row NaN ``max_snap`` passed every distance test, so a line in
        the middle of the image was "snapped" to the frame and closed."""
        df = self._line([(50.0, 50.0), (60.0, 60.0)]).with_columns(
            s=pl.lit(float("nan"))
        )
        with pytest.raises(pl.exceptions.ComputeError, match="max_snap must be >= 0"):
            df.select(
                pl.col("c").contour.close_along_border(100, 100, max_snap=pl.col("s"))
            )
