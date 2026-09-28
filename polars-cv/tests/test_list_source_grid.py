"""A ``List`` row is decoded only when it is a grid of values.

The ``list`` source (and ``.contour.label_reduce``'s image operand, which
decodes the same way) took each level's size from its first element and read
the row's values as that shape: ``[[1, 2], [3]]`` read past the end of its
values (garbage in the last slot), ``[[1, 2], [3], [4, 5, 6]]`` came back
silently re-rowed as ``[[1, 2], [3, 4], [5, 6]]``, and a null read as 0.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

pytestmark = plugin_required

LIST_SOURCE = Pipeline().source("list", dtype="f64")


def _decode(rows: list) -> list:
    df = pl.DataFrame({"a": rows}, schema={"a": pl.List(pl.List(pl.Float64))})
    return df.select(pl.col("a").cv.pipe(LIST_SOURCE).sink("list"))["a"].to_list()


def test_a_grid_decodes() -> None:
    assert _decode([[[1.0, 2.0], [3.0, 4.0]]]) == [[[1.0, 2.0], [3.0, 4.0]]]


@pytest.mark.parametrize(
    "row",
    [
        [[1.0, 2.0], [3.0]],
        [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
    ],
)
def test_a_jagged_row_is_refused(row: list) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="jagged"):
        _decode([row])


@pytest.mark.parametrize("row", [[[1.0, None], [3.0, 4.0]], [[1.0, 2.0], None]])
def test_a_null_in_a_row_is_refused(row: list) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="null"):
        _decode([row])


def test_label_reduce_refuses_a_heatmap_with_a_null_pixel() -> None:
    square = {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 2.0, "y": 0.0},
            {"x": 2.0, "y": 2.0},
            {"x": 0.0, "y": 2.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    df = pl.DataFrame(
        {
            "c": [[square]],
            # A dropped null used to shift the rest of the row left.
            "h": [[[1.0, None, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
        }
    )
    with pytest.raises(pl.exceptions.ComputeError, match="null"):
        df.select(pl.col("c").contour.label_reduce(pl.col("h")))


def test_require_contiguous_reads_a_rectangular_list() -> None:
    """A grid ``List`` row is read in place, so ``require_contiguous`` holds.

    It used to refuse every ``List`` row, grid or not: only ``Array`` rows
    were read in place.
    """
    pipe = Pipeline().source("list", dtype="f64", require_contiguous=True)
    df = pl.DataFrame(
        {"a": [[[1.0, 2.0], [3.0, 4.0]]]}, schema={"a": pl.List(pl.List(pl.Float64))}
    )
    out = df.select(pl.col("a").cv.pipe(pipe).sink("list"))["a"].to_list()
    assert out == [[[1.0, 2.0], [3.0, 4.0]]]
    jagged = pl.DataFrame(
        {"a": [[[1.0, 2.0], [3.0]]]}, schema={"a": pl.List(pl.List(pl.Float64))}
    )
    with pytest.raises(pl.exceptions.ComputeError, match="jagged"):
        jagged.select(pl.col("a").cv.pipe(pipe).sink("list"))


def _as_u8(values: list, dtype: pl.DataType, **source: object) -> list:
    df = pl.DataFrame({"a": [values]}, schema={"a": pl.List(dtype)})
    pipe = Pipeline().source("list", dtype="u8", **source)
    return df.select(pl.col("a").cv.pipe(pipe).sink("list"))["a"].to_list()


def test_other_dtypes_convert_to_the_declared_one() -> None:
    assert _as_u8([0, 7, 255], pl.Int64) == [[0, 7, 255]]
    assert _as_u8([3.7, 0.0], pl.Float64) == [[3, 0]]


@pytest.mark.parametrize(
    ("values", "dtype", "shown"),
    [
        ([7, 300], pl.Int64, "300"),
        ([7, -1], pl.Int64, "-1"),
        ([1.0, float("nan")], pl.Float64, "NaN"),
    ],
)
def test_a_value_the_declared_dtype_cannot_hold_is_refused(
    values: list, dtype: pl.DataType, shown: str
) -> None:
    """It used to be stored as 0 (CR-61)."""
    with pytest.raises(pl.exceptions.ComputeError, match=shown):
        _as_u8(values, dtype)
    assert _as_u8(values, dtype, on_error="null") == [None]
