"""A non-finite float literal is refused where it is written, by name (CR-103).

JSON, the wire every op and geometry call crosses, has no NaN or infinity:
``json.dumps`` writes ``NaN``, which serde refused as "expected value at line 1
column N" — naming neither the op nor the parameter. The literal is now refused
as it is encoded (``_types._literal``, the one place values reach the wire),
naming the parameter. A value a row needs is passed per row, through an
expression, where the op checks it with its own rule. A ``bound`` field
(``histogram(bins=)`` edges) still takes an infinity, which it spells ``"inf"``.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from polars_cv import Pipeline

from .conftest import plugin_required

pytestmark = plugin_required

NON_FINITE = [math.nan, math.inf, -math.inf]


def _contour() -> Pipeline:
    return Pipeline().source("contour")


def _image() -> Pipeline:
    return Pipeline().source("image_bytes", dtype="u8")


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize(
    ("build", "where"),
    [
        (lambda v: _contour().simplify(v), r"simplify\(tolerance=\)"),
        (lambda v: _contour().offset(v), r"offset\(distance=\)"),
        (lambda v: _image().scale(v), r"scale\(factor=\)"),
        (
            lambda v: _contour().rasterize(width=4, height=4, line_width=v),
            r"rasterize\(line_width=\)",
        ),
        # An element of a list field names its position.
        (
            lambda v: _image().grayscale().apply_lut([[0.0], [v]], dtype="f32"),
            r"apply_lut\(table=\)\[1\]\[0\]",
        ),
    ],
    ids=["simplify", "offset", "scale", "rasterize", "apply_lut-entry"],
)
def test_an_op_literal_is_refused_by_name(build, where: str, value: float) -> None:
    with pytest.raises(ValueError, match=rf"{where}.*not a finite number"):
        build(value)


@pytest.mark.parametrize("value", NON_FINITE)
def test_a_geometry_literal_is_refused_by_name(value: float) -> None:
    with pytest.raises(
        ValueError, match=r"simplify\(tolerance=\).*not a finite number"
    ):
        pl.col("c").contour.simplify(value)


def test_a_bound_still_takes_an_infinity() -> None:
    """``histogram(bins=)`` edges are bounds: open outer edges are allowed."""
    assert _image().histogram(bins=[-math.inf, 128.0, math.inf]) is not None
    with pytest.raises(
        ValueError, match=r"histogram\(bins=\)\[1\].*not a finite number"
    ):
        _image().histogram(bins=[0.0, math.nan, 255.0])
    with pytest.raises(
        ValueError, match=r"histogram\(range=\)\[1\].*not a finite number"
    ):
        _image().histogram(bins=4, range=(0.0, math.inf))


def test_the_same_value_per_row_reaches_the_ops_own_check() -> None:
    """The expression path is unchanged: the op refuses the value for its row."""
    from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

    square = {
        "exterior": [{"x": 0.0, "y": 0.0}, {"x": 4.0, "y": 0.0}, {"x": 4.0, "y": 4.0}],
        "holes": [],
        "is_closed": True,
    }
    df = pl.DataFrame(
        {"c": [[square]], "d": [math.nan]},
        schema={"c": CONTOUR_SET_SCHEMA, "d": pl.Float64},
    )
    with pytest.raises(pl.exceptions.ComputeError, match="distance must be finite"):
        df.select(pl.col("c").cv.pipe(_contour().offset(pl.col("d"))).sink("native"))
