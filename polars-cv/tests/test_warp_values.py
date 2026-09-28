"""Values the affine warp used to get wrong, through the public pipeline.

- ``rotate(0)`` ran the full bilinear warp. For finite pixels that returned the
  input, but a NaN or infinity spread into its left and upper neighbours
  (``NaN * 0.0`` is NaN). It now returns the input.
- A u64/i64 value at the top of its range was stored as 0: the warp clamped
  to ``u64::MAX as f64`` (2^64, one past the range), which the conversion
  after it refused. It now saturates, as every float→integer store does.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

pytestmark = plugin_required


@pytest.mark.parametrize("angle", [0.0, 360.0, -360.0])
def test_rotate_zero_leaves_every_value_where_it_was(angle: float) -> None:
    row = [[1.0, 2.0, 3.0], [4.0, float("nan"), 6.0], [7.0, 8.0, float("inf")]]
    df = pl.DataFrame({"a": [row]}, schema={"a": pl.List(pl.List(pl.Float32))})
    pipe = Pipeline().source("list", dtype="f32").rotate(angle)
    out = df.select(pl.col("a").cv.pipe(pipe).sink("list"))["a"][0].to_list()
    flat_in = [v for r in row for v in r]
    flat_out = [v for r in out for v in r]
    assert [math.isnan(v) for v in flat_out] == [math.isnan(v) for v in flat_in]
    assert [v for v in flat_out if not math.isnan(v)] == [
        v for v in flat_in if not math.isnan(v)
    ]


@pytest.mark.parametrize(
    ("dtype", "name", "top"),
    [(pl.UInt64, "u64", 2**64 - 1), (pl.Int64, "i64", 2**63 - 1)],
)
def test_a_warped_64_bit_maximum_stays_the_maximum(
    dtype: pl.DataType, name: str, top: int
) -> None:
    df = pl.DataFrame({"a": [[[top] * 4] * 4]}, schema={"a": pl.List(pl.List(dtype))})
    # A mirror: every source coordinate is an exact pixel centre.
    pipe = (
        Pipeline()
        .source("list", dtype=name)
        .warp_affine(matrix=[-1.0, 0.0, 3.0, 0.0, 1.0, 0.0], output_size=[4, 4])
    )
    out = df.select(pl.col("a").cv.pipe(pipe).sink("list"))["a"][0].to_list()
    assert out == [[top] * 4] * 4


@pytest.mark.parametrize(
    ("dtype", "name", "border", "fill"),
    [
        (pl.UInt8, "u8", 7.5, 8),
        (pl.UInt8, "u8", 7.4, 7),
        (pl.UInt8, "u8", 300.0, 255),
        (pl.UInt8, "u8", -1.0, 0),
        (pl.Int8, "i8", -200.0, -128),
        (pl.Float32, "f32", 7.5, 7.5),
    ],
)
def test_the_border_fill_is_stored_like_every_other_pixel(
    dtype: pl.DataType, name: str, border: float, fill: float
) -> None:
    # A translation far off the image: every output pixel is border. It is
    # stored by the conversion rule, rounded and saturated for an integer
    # dtype, as blended pixels are; it was truncated (7.5 -> 7) and an
    # out-of-range border became 0.
    df = pl.DataFrame({"a": [[[1] * 2] * 2]}, schema={"a": pl.List(pl.List(dtype))})
    pipe = (
        Pipeline()
        .source("list", dtype=name)
        .warp_affine(
            matrix=[1.0, 0.0, -400.0, 0.0, 1.0, -400.0],
            output_size=[3, 2],
            border_value=border,
        )
    )
    out = df.select(pl.col("a").cv.pipe(pipe).sink("list"))["a"][0].to_list()
    assert out == [[fill] * 2] * 3
