"""One contour parser for the whole plugin.

Contours arrive from Polars as ``{exterior, holes, is_closed}`` structs (or a
bare point list). Historically three independent Rust parsers consumed them —
the contour namespace, the point namespace, and the contour *source* decoder —
with diverging semantics: the point parser silently dropped holes, and the
source decoder rejected bare lists. These tests pin the unified contract:
every consumer routes through ``contour.rs::parse_contour``, so holes are
respected everywhere, bare lists work everywhere, and a malformed contour
produces the same error text everywhere.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SCHEMA
from tests.conftest import plugin_required


def square_with_hole() -> dict:
    """A 10x10 square with a 2x2 hole centered at (5, 5)."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 10.0, "y": 0.0},
            {"x": 10.0, "y": 10.0},
            {"x": 0.0, "y": 10.0},
        ],
        "holes": [
            [
                {"x": 4.0, "y": 4.0},
                {"x": 6.0, "y": 4.0},
                {"x": 6.0, "y": 6.0},
                {"x": 4.0, "y": 6.0},
            ]
        ],
        "is_closed": True,
    }


@plugin_required
class TestPointOpsRespectHoles:
    """Point-namespace ops must honor contour holes.

    view-buffer's predicates are hole-aware; the old point-namespace parser
    silently dropped holes before they could matter.
    """

    def test_signed_distance_positive_inside_hole(self) -> None:
        """A point inside a hole is OUTSIDE the contour: positive distance."""
        df = pl.DataFrame(
            {"pt": [{"x": 5.0, "y": 5.0}], "contour": [square_with_hole()]}
        ).cast({"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            sdf=pl.col("pt").point.signed_distance_to_contour(pl.col("contour"))
        )
        assert result["sdf"][0] > 0.0, (
            "point at the hole's center must be outside the contour "
            f"(got signed distance {result['sdf'][0]})"
        )

    def test_distance_measures_hole_boundary(self) -> None:
        """Distance-to-boundary includes hole edges (1.0, not 5.0)."""
        df = pl.DataFrame(
            {"pt": [{"x": 5.0, "y": 5.0}], "contour": [square_with_hole()]}
        ).cast({"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            dist=pl.col("pt").point.distance_to_contour(pl.col("contour"))
        )
        assert abs(result["dist"][0] - 1.0) < 1e-9, (
            "nearest boundary is the hole edge at distance 1.0 "
            f"(got {result['dist'][0]})"
        )


@plugin_required
class TestContourSourceFormats:
    """The contour source decoder accepts every parse_contour input form."""

    def test_contour_source_accepts_bare_point_list(self) -> None:
        """A bare List[{x, y}] column decodes as a simple contour."""
        square = [
            {"x": 10.0, "y": 10.0},
            {"x": 50.0, "y": 10.0},
            {"x": 50.0, "y": 50.0},
            {"x": 10.0, "y": 50.0},
        ]
        df = pl.DataFrame({"contour": [square]})
        pipe = Pipeline().source("contour", width=64, height=64)
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))
        assert result["mask"].null_count() == 0

    def test_contour_source_respects_holes(self) -> None:
        """Rasterizing a contour with a hole leaves the hole unfilled."""
        import numpy as np

        from polars_cv import numpy_from_struct

        contour = square_with_hole()
        df = pl.DataFrame({"contour": [contour]}).cast({"contour": CONTOUR_SCHEMA})
        pipe = Pipeline().source("contour", width=12, height=12)
        result = df.with_columns(mask=pl.col("contour").cv.pipe(pipe).sink("numpy"))
        arr = np.asarray(numpy_from_struct(result["mask"][0]))
        assert arr[5, 5, 0] == 0, "hole center must stay background"
        assert arr[2, 2, 0] == 255, "solid region must be filled"


@plugin_required
class TestUniformParseErrors:
    """A malformed contour produces the same parser error text everywhere."""

    def test_error_text_shared_across_consumers(self) -> None:
        bogus = {"not_a_contour": 1.0}
        errors: list[str] = []

        df = pl.DataFrame({"pt": [{"x": 0.0, "y": 0.0}], "contour": [bogus]})
        with pytest.raises(pl.exceptions.ComputeError) as exc_info:
            df.with_columns(d=pl.col("pt").point.distance_to_contour(pl.col("contour")))
        errors.append(str(exc_info.value))

        df = pl.DataFrame({"contour": [bogus]})
        pipe = Pipeline().source("contour", width=8, height=8)
        with pytest.raises(pl.exceptions.ComputeError) as exc_info:
            df.with_columns(m=pl.col("contour").cv.pipe(pipe).sink("numpy"))
        errors.append(str(exc_info.value))

        for err in errors:
            assert "exterior/points" in err, (
                f"parser error must name the expected fields, got: {err}"
            )
