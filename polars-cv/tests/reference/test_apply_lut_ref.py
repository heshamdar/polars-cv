"""Reference tests for ``apply_lut`` (CR-95).

``apply_lut(table, dtype=)`` reads each u8/u16 value as an index into
``table``. A table of single-entry rows maps every element in place, on any
channel count (``cv2.LUT`` with one table); rows of ``K > 1`` entries take a
single-channel image to ``K`` channels (a palette). Oracles: NumPy fancy
indexing (``table[x]``) everywhere, ``cv2.LUT`` for u8 -> u8.

An index past the table is a row error (owner decision 3, ``ISSUES.md``):
clamping is explicit, ``clamp(0, n - 1, out_dtype=...)`` first (``clamp_max``
alone promotes to f32, which ``apply_lut`` refuses).
"""

from __future__ import annotations

import cv2
import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_column
from tests.conftest import plugin_required

pytestmark = plugin_required

_POLARS = {np.dtype(np.uint8): pl.UInt8, np.dtype(np.uint16): pl.UInt16}
_OUT = {
    "u8": np.uint8,
    "i8": np.int8,
    "u16": np.uint16,
    "i16": np.int16,
    "u32": np.uint32,
    "i32": np.int32,
    "u64": np.uint64,
    "i64": np.int64,
    "f32": np.float32,
    "f64": np.float64,
}


def _frame(images: list[np.ndarray]) -> pl.DataFrame:
    image = images[0]
    return pl.DataFrame(
        {"m": images}, schema={"m": pl.Array(_POLARS[image.dtype], image.shape)}
    )


def _run(image: np.ndarray, table, **kw) -> np.ndarray:
    out = _frame([image]).select(
        pl.col("m")
        .cv.pipe(Pipeline().source("array").apply_lut(table, **kw))
        .sink("numpy")
    )
    (result,) = numpy_from_column(out.to_series(), copy=True)
    assert result is not None
    return result


def _table(n: int, k: int, dtype: str, seed: int = 95) -> np.ndarray:
    rng = np.random.default_rng(seed)
    np_dtype = np.dtype(_OUT[dtype])
    if np_dtype.kind == "f":
        return (rng.random((n, k)) * 2000 - 1000).astype(np_dtype)
    info = np.iinfo(np_dtype)
    lo, hi = max(info.min, -(2**40)), min(info.max, 2**40)
    return rng.integers(lo, hi, size=(n, k), endpoint=True).astype(np_dtype)


def test_u8_to_u8_matches_cv2_lut() -> None:
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (31, 17, 3), dtype=np.uint8)
    table = _table(256, 1, "u8")
    got = _run(image, table.tolist())
    np.testing.assert_array_equal(got, cv2.LUT(image, table[:, 0]))
    assert got.shape == image.shape


@pytest.mark.parametrize("dtype", list(_OUT))
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_single_entry_rows_map_every_channel_in_place(
    dtype: str, channels: int
) -> None:
    rng = np.random.default_rng(channels)
    image = rng.integers(0, 40, (9, 13, channels), dtype=np.uint8)
    table = _table(40, 1, dtype)
    got = _run(image, table.tolist(), dtype=dtype)
    assert got.dtype == np.dtype(_OUT[dtype])
    np.testing.assert_array_equal(got, table[image, 0])


@pytest.mark.parametrize("dtype", ["u8", "i16", "f32"])
@pytest.mark.parametrize("k", [2, 3, 4, 5])
@pytest.mark.parametrize("shape", [(11, 7), (11, 7, 1)])
def test_a_palette_takes_one_channel_to_k(dtype: str, k: int, shape) -> None:
    image = np.random.default_rng(k).integers(0, 6, shape, dtype=np.uint8)
    table = _table(6, k, dtype)
    got = _run(image, table.tolist(), dtype=dtype)
    assert got.shape == (11, 7, k)
    np.testing.assert_array_equal(got, table[image.reshape(11, 7)])


def test_u16_input_indexes_by_value() -> None:
    rng = np.random.default_rng(3)
    image = rng.integers(0, 1000, (8, 9, 1), dtype=np.uint16)
    table = _table(1000, 3, "u16")
    np.testing.assert_array_equal(
        _run(image, table.tolist(), dtype="u16"), table[image[:, :, 0]]
    )


def test_the_full_u16_domain_with_a_strided_input() -> None:
    values = np.arange(65536, dtype=np.uint16).reshape(256, 256)[:, ::-1]
    image = np.ascontiguousarray(values)
    table = (np.arange(65536, dtype=np.int64)[::-1] % 997).reshape(-1, 1)
    np.testing.assert_array_equal(
        _run(image, table.tolist(), dtype="i32"), table[image, 0].astype(np.int32)
    )


def test_an_index_past_the_table_is_a_row_error() -> None:
    image = np.array([[0, 1, 2, 3]], np.uint8)
    df = _frame([image])
    pipe = Pipeline().source("array").apply_lut([[10], [20], [30]])
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r'value 3 is past.*3 entries.*clamp\(0, 2, out_dtype="u8"\)',
    ):
        df.select(pl.col("m").cv.pipe(pipe).sink("numpy"))
    # Clamping is explicit, and then every value has an entry.
    clamped = (
        Pipeline()
        .source("array")
        .clamp(0, 2, out_dtype="u8")
        .apply_lut([[10], [20], [30]])
    )
    out = df.select(pl.col("m").cv.pipe(clamped).sink("list"))
    assert out.item().to_list() == [[10, 20, 30, 30]]


def test_on_error_null_nulls_only_the_offending_row() -> None:
    df = _frame([np.array([[0, 1]], np.uint8), np.array([[0, 9]], np.uint8)])
    pipe = Pipeline().source("array").apply_lut([[5], [6]]).on_error("null")
    out = df.select(pl.col("m").cv.pipe(pipe).sink("list")).to_series().to_list()
    assert out == [[[5, 6]], None]


@pytest.mark.parametrize(
    ("table", "kw", "match"),
    [
        ([], {}, "at least one|empty"),
        ([[1, 2], [3]], {}, "same length|ragged"),
        ([[300]], {}, "300.*u8|fit"),
        ([[-1]], {}, "-1.*u8|fit"),
        ([[1.5]], {}, "1.5.*u8|fit|integer"),
        ([[1e300]], {"dtype": "f32"}, "f32|fit"),
    ],
)
def test_a_table_no_row_can_use_is_refused_at_build(table, kw, match) -> None:
    with pytest.raises(Exception, match=match):
        Pipeline().source("image_bytes", dtype="u8").apply_lut(table, **kw)


def test_float_input_is_refused() -> None:
    with pytest.raises(Exception, match="(?i)u8.*u16"):
        Pipeline().source("image_bytes", dtype="u8").cast("f32").apply_lut([[0]])


def test_a_palette_needs_one_channel() -> None:
    with pytest.raises(Exception, match="single-channel|one channel"):
        (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .assert_shape(channels=3)
            .apply_lut([[0, 0, 0]])
        )


def test_the_plan_publishes_the_table_dtype_and_k_channels() -> None:
    pipe = Pipeline().source("image_bytes", dtype="u8").grayscale()
    palette = pipe.apply_lut([[0, 0, 0], [1, 2, 3]], dtype="f32")
    assert palette.output_dtype() == "f32"
    df = _frame([np.zeros((3, 4, 1), np.uint8)])
    out = df.select(
        pl.col("m")
        .cv.pipe(Pipeline().source("array").apply_lut([[0, 0, 0]], dtype="i16"))
        .sink("array", shape=[3, 4, 3])
    )
    assert out.schema["m"] == pl.Array(pl.Int16, (3, 4, 3))
