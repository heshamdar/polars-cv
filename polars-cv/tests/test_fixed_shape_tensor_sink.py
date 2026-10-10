"""``sink("fixed_shape_tensor")``: Arrow's canonical ``arrow.fixed_shape_tensor``.

A Polars ``Array`` sink nests one ``FixedSizeList`` per axis. The canonical
tensor type is a single flat ``FixedSizeList`` of each row's elements in
row-major order, with the shape in the type's metadata (``{"shape": [...]}``).
That is the layout PyArrow, Ray, Lance and others read as a tensor column. The
column is Polars' generic ``pl.Extension`` over ``Array(dtype, n)``: polars-cv
registers no class for an ``arrow.`` name, which is not its own.
"""

from __future__ import annotations

import io
import json

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polars_cv import Pipeline, numpy_from_column

from .conftest import plugin_required

pytestmark = plugin_required

H, W, C = 12, 10, 3
NAME = "arrow.fixed_shape_tensor"


def _raw_frame() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    rows = [rng.integers(0, 255, (H, W, C), dtype=np.uint8).tobytes() for _ in range(3)]
    return pl.DataFrame(
        {"raw": [rows[0], None, rows[1], rows[2]]}, schema={"raw": pl.Binary}
    )


RAW = Pipeline().source("raw", dtype="u8").reshape([H, W, C])

#: Rows reaching the sink contiguous, strided, flipped, offset, cast, and at
#: another rank.
LAYOUTS = {
    "passthrough": RAW,
    "transpose": RAW.transpose([1, 0, 2]),
    "flip_h": RAW.flip_h(),
    "crop": RAW.crop(top=2, left=1, height=7, width=6),
    "float": RAW.cast("f32"),
    "rank 2": RAW.reshape([H, W * C]),
}


def _sink(pipe: Pipeline, **kwargs: object) -> pl.LazyFrame:
    return (
        _raw_frame()
        .lazy()
        .select(t=pl.col("raw").cv.pipe(pipe).sink("fixed_shape_tensor", **kwargs))
    )


def _expected(pipe: Pipeline) -> list[np.ndarray | None]:
    column = (
        _raw_frame().select(n=pl.col("raw").cv.pipe(pipe).sink("numpy")).to_series()
    )
    return numpy_from_column(column, copy=True)


def _assert_tensor_dtype(
    dtype: pl.DataType, shape: list[int], inner: pl.DataType
) -> None:
    assert isinstance(dtype, pl.Extension), dtype
    assert dtype.ext_name() == NAME
    assert dtype.ext_storage() == pl.Array(inner, int(np.prod(shape)))
    assert json.loads(dtype.ext_metadata()) == {"shape": shape}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_planned_type_is_the_executed_type(layout: str) -> None:
    pipe = LAYOUTS[layout]
    first = next(e for e in _expected(pipe) if e is not None)
    inner = pl.Float32 if first.dtype == np.float32 else pl.UInt8
    lazy = _sink(pipe)
    planned = lazy.collect_schema()["t"]
    _assert_tensor_dtype(planned, list(first.shape), inner)
    assert lazy.collect().schema["t"] == planned


@pytest.mark.parametrize("layout", LAYOUTS)
def test_pyarrow_reads_the_rows_as_tensors(layout: str) -> None:
    pipe = LAYOUTS[layout]
    table = _sink(pipe).collect().to_arrow()
    column = table.column("t").combine_chunks()
    assert isinstance(column.type, pa.FixedShapeTensorType), column.type
    expected = _expected(pipe)
    assert column.null_count == sum(e is None for e in expected)
    for i, e in enumerate(expected):
        if e is None:
            assert not column[i].is_valid, f"row {i}: a null row is null"
            continue
        np.testing.assert_array_equal(
            column[i].to_numpy(), e, err_msg=f"{layout} row {i}"
        )


def test_an_explicit_shape_is_checked_against_the_rows() -> None:
    _assert_tensor_dtype(
        _sink(RAW, shape=[H, W, C]).collect_schema()["t"], [H, W, C], pl.UInt8
    )
    with pytest.raises(Exception, match=r"shape \[10, 12, 3\] does not match"):
        _sink(RAW, shape=[W, H, C]).collect()


def test_a_shape_unknown_at_planning_is_refused_before_any_row_runs() -> None:
    pipe = RAW.resize(height=pl.col("raw").bin.size().cast(pl.UInt32) // 60, width=5)
    with pytest.raises(
        Exception, match=r"'fixed_shape_tensor' sink needs the full output shape"
    ):
        _sink(pipe).collect_schema()


def test_a_non_buffer_output_is_refused() -> None:
    with pytest.raises(
        Exception, match=r"domain 'scalar' with sink format 'fixed_shape_tensor'"
    ):
        _sink(RAW.reduce_sum()).collect()


def test_the_type_survives_parquet() -> None:
    """Polars round-trips the column, nulls included; PyArrow reads it back as
    a tensor column.

    PyArrow is shown a column without null rows: it cannot read a Parquet
    fixed-size list with a null row that Polars wrote, nor write one of its
    own ("Lists with non-zero length null components are not supported", as
    of PyArrow 22). That holds for any fixed-size list, the ``array`` sink's
    included, and is not this sink's to work around.
    """
    frame = _sink(LAYOUTS["transpose"]).collect()
    buf = io.BytesIO()
    frame.write_parquet(buf)
    read = pl.read_parquet(io.BytesIO(buf.getvalue()))
    assert read.schema == frame.schema
    assert read.equals(frame)

    dense = frame.drop_nulls()
    buf = io.BytesIO()
    dense.write_parquet(buf)
    table = pq.read_table(io.BytesIO(buf.getvalue()))
    assert isinstance(table.schema.field("t").type, pa.FixedShapeTensorType)
    expected = [e for e in _expected(LAYOUTS["transpose"]) if e is not None]
    np.testing.assert_array_equal(
        table.column("t").combine_chunks().to_numpy_ndarray(), np.stack(expected)
    )
