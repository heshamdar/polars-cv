"""``sink("numpy"|"ndarray"|"torch", compact=True)``: rows that hold only their own bytes.

By default a tensor sink hands each row's storage over as it lies, so a view
(a crop, a transpose, a flip, a blob's payload) arrives with the whole buffer
behind it, plus the strides and offset that read the view out of it. That is
zero-copy, but whatever persists the column (Parquet, IPC) writes the whole
buffer. ``compact=True`` makes each row's ``data`` exactly its elements,
row-major, at offset zero, and copies only a row that is not already that.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_column
from polars_cv._lib import binary_rows

from .conftest import plugin_required

pytestmark = plugin_required

H, W, C = 64, 48, 3
ROWS = 3


def _raw_frame() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 255, (H, W, C), dtype=np.uint8) for _ in range(ROWS)]
    return pl.DataFrame(
        {"raw": [a.tobytes() for a in images[:-1]] + [None, images[-1].tobytes()]},
        schema={"raw": pl.Binary},
    )


RAW = Pipeline().source("raw", dtype="u8").reshape([H, W, C])

#: Every way a row reaches the sink: read in place, as a view of its input
#: (strided, flipped, an offset into it, a contiguous run of it), as a new
#: allocation, and as a blob's payload behind its header.
LAYOUTS = {
    "passthrough": RAW,
    "transpose": RAW.transpose([1, 0, 2]),
    "flip_h": RAW.flip_h(),
    "crop window": RAW.crop(top=8, left=4, height=40, width=30),
    "crop full rows": RAW.crop(top=8, left=0, height=40, width=W),
    "resize": RAW.resize(height=32, width=24),
    "float": RAW.cast("f32"),
}

TENSOR_SINKS = ["numpy", "ndarray", "torch"]


def _fields(column: pl.Series) -> pl.DataFrame:
    if isinstance(column.dtype, pl.datatypes.BaseExtension):
        column = column.ext.storage()
    return column.struct.unnest()


def _sink(
    frame: pl.DataFrame, pipe: Pipeline, sink: str, **kwargs: object
) -> pl.Series:
    return frame.select(
        out=pl.col("raw").cv.pipe(pipe).sink(sink, **kwargs)
    ).to_series()


def _assert_compact(column: pl.Series, label: str) -> None:
    for i, row in enumerate(_fields(column).iter_rows(named=True)):
        if row["data"] is None:
            continue
        itemsize = np.dtype(row["dtype"]).itemsize
        shape = row["shape"]
        c_strides = [
            int(np.prod(shape[k + 1 :], dtype=np.int64)) * itemsize
            for k in range(len(shape))
        ]
        assert len(row["data"]) == int(np.prod(shape)) * itemsize, (
            f"{label} row {i}: extra bytes"
        )
        assert row["offset"] == 0, f"{label} row {i}"
        assert row["strides"] == c_strides, f"{label} row {i}"


@pytest.mark.parametrize("sink", TENSOR_SINKS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_compact_rows_hold_exactly_their_elements(layout: str, sink: str) -> None:
    frame = _raw_frame()
    pipe = LAYOUTS[layout]
    compact = _sink(frame, pipe, sink, compact=True)
    _assert_compact(compact, f"{layout}/{sink}")
    expected = numpy_from_column(_sink(frame, pipe, sink))
    got = numpy_from_column(compact)
    for i, (e, g) in enumerate(zip(expected, got, strict=True)):
        if e is None:
            assert g is None, f"row {i}: a null row stays null"
            continue
        assert g is not None
        assert g.dtype == e.dtype
        np.testing.assert_array_equal(g, e, err_msg=f"{layout}/{sink} row {i}")


def test_compact_blob_rows_drop_the_header() -> None:
    blobs = _raw_frame().select(raw=pl.col("raw").cv.pipe(RAW).sink("blob"))
    pipe = Pipeline().source("blob")
    _assert_compact(_sink(blobs, pipe, "numpy", compact=True), "blob")


def test_compact_half_precision_rows_are_compact() -> None:
    column = _sink(
        _raw_frame(),
        LAYOUTS["float"].transpose([1, 0, 2]),
        "numpy",
        compact=True,
        dtype="f16",
    )
    _assert_compact(column, "f16")
    assert _fields(column)["dtype"].drop_nulls().unique().to_list() == ["float16"]


@pytest.mark.parametrize("layout", ["passthrough", "crop full rows"])
def test_compact_rows_that_already_are_stay_in_place(layout: str) -> None:
    """A row whose elements are one packed run of its input is sliced, not copied."""
    frame = _raw_frame()
    inputs = [
        (addr, addr + n) for _, addr, n in filter(None, binary_rows(frame["raw"]))
    ]
    out = _fields(_sink(frame, LAYOUTS[layout], "numpy", compact=True))["data"]
    for i, row in enumerate(binary_rows(out)):
        if row is None:
            continue
        _, addr, _ = row
        assert any(lo <= addr < hi for lo, hi in inputs), f"{layout} row {i} was copied"


def test_compact_persists_only_the_elements() -> None:
    """A crop above the default sink's 50% rule keeps each row's whole input,
    and Parquet writes it; a compact one writes only the crop."""
    frame = _raw_frame()
    height, width = 56, 40  # 73% of the image
    pipe = RAW.crop(top=4, left=2, height=height, width=width)
    rows = frame["raw"].drop_nulls().len()
    logical = rows * height * width * C

    def written(**kwargs: object) -> int:
        buf = io.BytesIO()
        pl.DataFrame({"o": _sink(frame, pipe, "ndarray", **kwargs)}).write_parquet(
            buf, compression="uncompressed"
        )
        return buf.tell()

    full, compact = written(), written(compact=True)
    assert full > rows * H * W * C, "the default sink writes each row's whole input"
    assert compact < logical + 4096, (
        f"{compact} bytes written for {logical} of elements"
    )
