"""``numpy_from_column``: a numpy-sink column's rows as arrays, without copies.

Every row's array is a view of the column's own Arrow memory: nothing is
copied into Python ``bytes`` on the way. Each assertion compares against
``numpy_from_struct(copy=True)``, the row-at-a-time reader, so layout handling
(strides, offsets, flips) is held to the same answer.
"""

from __future__ import annotations

import gc

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_column, numpy_from_struct

from .conftest import make_image_png, plugin_required

pytestmark = plugin_required


def _images() -> pl.DataFrame:
    """RGB and RGBA rows with a null between them."""
    return pl.DataFrame(
        {
            "img": [
                make_image_png(6, 9, 3, seed=1),
                None,
                make_image_png(4, 5, 4, seed=2),
            ]
        },
        schema={"img": pl.Binary},
    )


PIPELINES = {
    "decoded": Pipeline().source("image_bytes"),
    "transposed": Pipeline().source("image_bytes").transpose(axes=[1, 0, 2]),
    "flipped": Pipeline().source("image_bytes").flip(axes=[0, 1]),
    "grayscale": Pipeline().source("image_bytes").grayscale(),
    "float": Pipeline().source("image_bytes").cast("f32"),
}


def _column(pipe: Pipeline, sink: str = "numpy") -> pl.Series:
    return _images().select(out=pl.col("img").cv.pipe(pipe).sink(sink)).to_series()


@pytest.mark.parametrize("name", PIPELINES)
def test_rows_match_numpy_from_struct(name: str) -> None:
    column = _column(PIPELINES[name])
    arrays = numpy_from_column(column)
    assert len(arrays) == len(column)
    for i, (array, row) in enumerate(zip(arrays, column.to_list(), strict=True)):
        if row is None or row["data"] is None:
            assert array is None, f"row {i}: a null row is None"
            continue
        assert array is not None
        expected = numpy_from_struct(row, copy=True)
        assert array.dtype == expected.dtype, f"row {i}"
        np.testing.assert_array_equal(array, expected, err_msg=f"row {i}")


def test_arrays_view_the_column_memory() -> None:
    column = _column(PIPELINES["decoded"])
    first = numpy_from_column(column)
    second = numpy_from_column(column)
    for a, b in zip(first, second, strict=True):
        if a is None:
            continue
        # Two reads of one row share memory only if neither copied it.
        assert np.shares_memory(a, b)
        # Polars memory is immutable: the views are read-only.
        assert not a.flags.writeable


def test_arrays_outlive_the_column() -> None:
    column = _column(PIPELINES["transposed"])
    arrays = numpy_from_column(column)
    expected = [None if a is None else a.copy() for a in arrays]
    del column
    gc.collect()
    for a, e in zip(arrays, expected, strict=True):
        if e is None:
            continue
        np.testing.assert_array_equal(a, e)


def test_copy_returns_owned_arrays() -> None:
    column = _column(PIPELINES["flipped"])
    views = numpy_from_column(column)
    copies = numpy_from_column(column, copy=True)
    for v, c in zip(views, copies, strict=True):
        if v is None:
            assert c is None
            continue
        assert c.flags.writeable and c.flags.c_contiguous
        assert not np.shares_memory(v, c)
        np.testing.assert_array_equal(v, c)


def test_an_ndarray_column_reads_the_same() -> None:
    pipe = PIPELINES["transposed"]
    plain = numpy_from_column(_column(pipe, "numpy"))
    tagged = numpy_from_column(_column(pipe, "ndarray"))
    for p, t in zip(plain, tagged, strict=True):
        if p is None:
            assert t is None
            continue
        np.testing.assert_array_equal(p, t)


def _struct_column(
    data: bytes, shape: list[int], strides: list[int], offset: int
) -> pl.Series:
    return (
        pl.DataFrame(
            {
                "data": [data],
                "dtype": ["uint8"],
                "shape": [shape],
                "strides": [strides],
                "offset": [offset],
            },
            schema={
                "data": pl.Binary,
                "dtype": pl.String,
                "shape": pl.List(pl.UInt64),
                "strides": pl.List(pl.Int64),
                "offset": pl.UInt64,
            },
        )
        .select(pl.struct(pl.all()).alias("out"))
        .to_series()
    )


@pytest.mark.parametrize(
    ("shape", "strides", "offset"),
    [
        ([64, 64], [64, 1], 0),  # far more elements than the 32 bytes hold
        ([4, 4], [4, 1], 20),  # starts inside, ends past the end
        ([4, 4], [-4, 1], 0),  # walks backwards off the start
    ],
)
def test_a_view_outside_the_row_bytes_is_refused(
    shape: list[int], strides: list[int], offset: int
) -> None:
    column = _struct_column(bytes(range(32)), shape, strides, offset)
    with pytest.raises(ValueError, match="outside"):
        numpy_from_column(column)


def test_a_view_inside_the_row_bytes_reads() -> None:
    column = _struct_column(bytes(range(32)), [4, 4], [-4, 1], 12)
    (array,) = numpy_from_column(column)
    np.testing.assert_array_equal(
        array, np.arange(16, dtype=np.uint8).reshape(4, 4)[::-1]
    )


def test_a_column_that_is_not_the_numpy_struct_is_refused() -> None:
    with pytest.raises(TypeError, match="numpy"):
        numpy_from_column(pl.Series("x", [b"abc"]))
