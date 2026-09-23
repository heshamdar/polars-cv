"""An image with no contours yields an empty contour set, not a null.

A null means "no value": the input was null, or the row failed under
``on_error("null")``. An image that was processed and simply contains no
shapes has a value — the empty set — and publishing it as null made
``list.len()`` read null instead of 0, dropped the row from counts, and
disagreed with the ``.contour`` transforms, which already keep ``[]``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline

from .conftest import plugin_required


@pytest.fixture
def frame(encode_png) -> pl.DataFrame:  # noqa: ANN001
    blank = np.zeros((16, 16, 3), dtype=np.uint8)
    square = blank.copy()
    square[4:12, 4:12] = 255
    return pl.DataFrame(
        {"img": [encode_png(square), encode_png(blank), None, b"not an image"]},
        schema={"img": pl.Binary},
    )


def _contours() -> Pipeline:
    return (
        Pipeline()
        .source("image_bytes", dtype="u8")
        .on_error("null")
        .grayscale()
        .threshold(128)
        .extract_contours()
    )


@plugin_required
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_no_contours_is_an_empty_set_and_failures_are_null(
    frame: pl.DataFrame, engine: str
) -> None:
    out = (
        frame.lazy()
        .select(c=pl.col("img").cv.pipe(_contours()).sink("native"))
        .with_columns(n=pl.col("c").list.len())
        .collect(engine=engine)  # ty: ignore[invalid-argument-type]
    )
    assert out["c"].is_null().to_list() == [False, False, True, True]
    assert out["n"].to_list() == [1, 0, None, None]


@plugin_required
def test_the_sink_and_the_transforms_agree_on_an_empty_set(
    frame: pl.DataFrame,
) -> None:
    out = frame.select(c=pl.col("img").cv.pipe(_contours()).sink("native"))
    moved = out.select(m=pl.col("c").contour.translate(dx=1.0, dy=0.0))
    assert moved["m"][1] is not None
    assert moved["m"][1].len() == 0
