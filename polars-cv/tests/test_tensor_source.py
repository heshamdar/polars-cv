"""``source("tensor")``: the numpy/ndarray/torch sink struct, read back (CR-138).

The three tensor sinks share one struct (``NUMPY_OUTPUT_SCHEMA``), and no
source took it: their output could not feed another pipeline, even after a
Parquet round trip. ``source("tensor")`` reads a row where it lies, at its own
strides and offset, and ``source()`` (auto) routes that struct to it.

A row is column data, so its ``shape``, ``strides`` and ``offset`` are
untrusted: a row whose view does not lie inside its ``data`` is a row error,
never a read outside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from polars_cv.extension_types import NUMPY_OUTPUT_SCHEMA

from ._schema_parity import assert_plan_equals_exec
from .conftest import make_image_png, plugin_required

pytestmark = plugin_required

SINKS = ["tensor"]
SOURCES = [(), ("tensor",)]  # auto, and the explicit format


def _images() -> pl.DataFrame:
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


def _rows(series: pl.Series) -> list[np.ndarray | None]:
    return [None if row is None else numpy_from_struct(row) for row in series]


def _same(got: pl.Series, want: pl.Series) -> None:
    for g, w in zip(_rows(got), _rows(want), strict=True):
        if w is None:
            assert g is None
        else:
            assert g is not None
            assert g.dtype == w.dtype
            assert g.shape == w.shape
            assert np.array_equal(g, w, equal_nan=True)


def _tensors(pipe: Pipeline, sink: str, **sink_kwargs: Any) -> pl.DataFrame:
    return _images().select(t=pl.col("img").cv.pipe(pipe).sink(sink, **sink_kwargs))


def _read(df: pl.DataFrame, source: tuple[str, ...], **kwargs: Any) -> pl.Series:
    pipe = Pipeline().source(*source, **kwargs)
    return df.select(o=pl.col("t").cv.pipe(pipe).sink("tensor"))["o"]


@pytest.mark.parametrize("sink", SINKS)
@pytest.mark.parametrize("source", SOURCES, ids=["auto", "tensor"])
def test_a_tensor_sink_reads_back(sink: str, source: tuple[str, ...]) -> None:
    df = _tensors(Pipeline().source("image_bytes"), sink)
    _same(_read(df, source), df["t"])


@pytest.mark.parametrize("sink", SINKS)
@pytest.mark.parametrize("source", SOURCES, ids=["auto", "tensor"])
def test_a_strided_row_reads_back(sink: str, source: tuple[str, ...]) -> None:
    """Flips and a transpose: negative strides, an offset, no packed order."""
    pipe = Pipeline().source("image_bytes").flip_h().flip_v().transpose([1, 0, 2])
    df = _tensors(pipe, sink)
    storage = df["t"].ext.storage()
    strides = storage.struct.field("strides").drop_nulls()
    assert any(s < 0 for row in strides for s in row), (
        "the case must hold a negative stride to mean anything"
    )
    _same(_read(df, source), df["t"])


@pytest.mark.parametrize("source", SOURCES, ids=["auto", "tensor"])
def test_a_compact_row_reads_back(source: tuple[str, ...]) -> None:
    pipe = (
        Pipeline().source("image_bytes").flip_h().crop(top=1, left=1, height=2, width=3)
    )
    df = _tensors(pipe, "tensor", compact=True)
    _same(_read(df, source), df["t"])


@pytest.mark.parametrize("source", SOURCES, ids=["auto", "tensor"])
def test_a_float16_row_reads_back_as_f32(source: tuple[str, ...]) -> None:
    """The engine has no f16: a half-precision row widens to f32, exactly."""
    pipe = Pipeline().source("image_bytes").scale(1 / 255)
    df = _tensors(pipe, "tensor", dtype="f16")
    halves = _rows(df["t"])
    assert halves[0] is not None and halves[0].dtype == np.float16
    got = _rows(_read(df, source))
    for g, h in zip(got, halves, strict=True):
        if h is None:
            assert g is None
        else:
            assert g is not None and g.dtype == np.float32
            assert np.array_equal(g, h.astype(np.float32))


@pytest.mark.parametrize("sink", ["tensor"])
def test_a_parquet_round_trip_reads_back(sink: str, tmp_path: Any) -> None:
    df = _tensors(Pipeline().source("image_bytes").flip_h(), sink)
    path = tmp_path / "t.parquet"
    df.write_parquet(path)
    _same(_read(pl.read_parquet(path), ()), df["t"])


def test_ops_run_on_a_tensor_source() -> None:
    df = _tensors(Pipeline().source("image_bytes").flip_v(), "tensor")
    pipe = Pipeline().source("tensor").grayscale().resize(height=3, width=3)
    out = df.select(o=pl.col("t").cv.pipe(pipe).sink("tensor"))["o"]
    direct = _images().select(
        o=pl.col("img")
        .cv.pipe(
            Pipeline()
            .source("image_bytes")
            .flip_v()
            .grayscale()
            .resize(height=3, width=3)
        )
        .sink("tensor")
    )["o"]
    _same(out, direct)


@pytest.mark.parametrize("sink", ["tensor", "array", "list", "blob", "png"])
def test_planned_schema_is_the_executed_one(sink: str) -> None:
    """A row carries its own rank, so a nested sink is told it (as for a blob)."""
    df = _tensors(Pipeline().source("image_bytes"), "tensor")
    pipe = Pipeline().source("tensor", dtype="u8")
    if sink in ("array", "list"):
        pipe = pipe.resize(height=4, width=4).grayscale().assert_shape(dims=[4, 4, 1])
    assert_plan_equals_exec(df, pl.col("t").cv.pipe(pipe).sink(sink))


def test_the_rank_is_unknown_at_planning() -> None:
    pipe = Pipeline().source("tensor", dtype="u8")
    with pytest.raises(ValueError, match="needs the output rank at planning time"):
        pl.col("t").cv.pipe(pipe).sink("list")


def test_a_declared_dtype_is_checked_against_the_row() -> None:
    df = _tensors(Pipeline().source("image_bytes"), "tensor")
    with pytest.raises(pl.exceptions.ComputeError, match="declares dtype"):
        _read(df, ("tensor",), dtype="f32")
    out = _read(df, ("tensor",), dtype="f32", on_error="null")
    assert out.null_count() == df.height


# Malformed rows. The good row is 2x3 u8 over six bytes.


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "data": bytes(range(6)),
        "dtype": "uint8",
        "shape": [2, 3],
        "strides": [3, 1],
        "offset": 0,
    }
    row.update(overrides)
    return row


def _frame(*rows: dict[str, Any] | None) -> pl.DataFrame:
    return pl.DataFrame({"t": list(rows)}, schema={"t": NUMPY_OUTPUT_SCHEMA})


def test_a_hand_built_row_reads() -> None:
    (got,) = _rows(_read(_frame(_row()), ("tensor",)))
    assert got is not None
    assert np.array_equal(got, np.arange(6, dtype=np.uint8).reshape(2, 3))
    # A reversed view of the same bytes: offset at the last element.
    (got,) = _rows(_read(_frame(_row(strides=[-3, -1], offset=5)), ("tensor",)))
    assert got is not None
    assert np.array_equal(got, np.arange(6, dtype=np.uint8)[::-1].reshape(2, 3))


_MALFORMED = {
    "offset past the end": _row(offset=7),
    "offset at the end": _row(offset=6),
    "stride walks past the end": _row(strides=[4, 1]),
    "stride walks below the start": _row(strides=[-3, 1], offset=0),
    "negative reach from a small offset": _row(strides=[3, -1], offset=1),
    "shape larger than data": _row(shape=[3, 3]),
    "huge shape": _row(shape=[2**40, 2**40]),
    "shape product overflows": _row(shape=[2**63, 2**63], strides=[0, 0]),
    "huge stride": _row(strides=[2**62, 1]),
    "most negative stride": _row(strides=[-(2**63), 1]),
    "rank mismatch": _row(strides=[1]),
    "unknown dtype": _row(dtype="complex64"),
    "stride not a whole element": _row(dtype="uint16", shape=[3], strides=[1]),
    "data not a whole element": _row(
        dtype="uint16", shape=[3], strides=[2], data=b"12345"
    ),
    "null data": _row(data=None),
    "null dtype": _row(dtype=None),
    "null shape": _row(shape=None),
    "null strides": _row(strides=None),
    "null offset": _row(offset=None),
    "null dimension": _row(shape=[2, None]),
}


@pytest.mark.parametrize("bad", _MALFORMED.values(), ids=list(_MALFORMED))
def test_a_malformed_row_is_a_row_error(bad: dict[str, Any]) -> None:
    df = _frame(_row(), bad)
    with pytest.raises(pl.exceptions.ComputeError):
        _read(df, ("tensor",))
    out = _read(df, ("tensor",), on_error="null")
    assert out[0] is not None
    assert out[1] is None


def test_a_null_row_is_a_null_row() -> None:
    out = _read(_frame(_row(), None), ("tensor",))
    assert out[0] is not None
    assert out[1] is None


def test_random_layouts_never_read_outside_data() -> None:
    """A sweep: every row either reads exactly what NumPy reads, or is null."""
    rng = np.random.default_rng(138)
    rows, expected = [], []
    for _ in range(400):
        size = int(rng.integers(0, 24))
        data = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        rank = int(rng.integers(0, 4))
        shape = [int(rng.integers(0, 5)) for _ in range(rank)]
        strides = [int(rng.integers(-9, 10)) for _ in range(rank)]
        offset = int(rng.integers(0, 30))
        rows.append(_row(data=data, shape=shape, strides=strides, offset=offset))
        n = int(np.prod(shape)) if shape else 1
        lo = offset + sum((d - 1) * s for d, s in zip(shape, strides) if s < 0 and d)
        hi = offset + sum((d - 1) * s for d, s in zip(shape, strides) if s > 0 and d)
        if n == 0:
            expected.append(np.zeros(shape, dtype=np.uint8) if offset <= size else None)
        elif lo < 0 or hi >= size:
            expected.append(None)
        else:
            flat = np.frombuffer(data, dtype=np.uint8)
            expected.append(
                np.lib.stride_tricks.as_strided(flat[offset:], shape, strides).copy()
            )
    assert sum(e is not None for e in expected) > 40, "the sweep must read some rows"
    assert sum(e is None for e in expected) > 40, "and refuse some"
    got = _rows(_read(_frame(*rows), ("tensor",), on_error="null"))
    for i, (g, e) in enumerate(zip(got, expected, strict=True)):
        if e is None:
            assert g is None, f"row {i} ({rows[i]}) lies outside its data"
        else:
            assert g is not None, f"row {i} ({rows[i]}) lies inside its data"
            assert g.shape == e.shape and np.array_equal(g, e), f"row {i}"


def test_another_struct_is_not_a_tensor() -> None:
    """Auto matches the whole schema, dtypes included, not five field names."""
    other = pl.Struct(
        {
            "data": pl.Binary,
            "dtype": pl.String,
            "shape": pl.List(pl.Int64),  # the tensor struct's is UInt64
            "strides": pl.List(pl.Int64),
            "offset": pl.UInt64,
        }
    )
    df = pl.DataFrame({"t": [_row()]}, schema={"t": other})
    with pytest.raises(pl.exceptions.ComputeError, match="cannot infer a decode path"):
        _read(df, ())
