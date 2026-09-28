"""Tests for the ``dtype="f16"`` half-precision numpy/torch sink downcast.

The engine has no native f16 dtype, so f16 is produced purely as an encode-time
downcast at the sink boundary (halving the output-tensor bytes / H2D transfer).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required


class TestF16SinkValidation:
    """Builder-time validation of the sink ``dtype`` kwarg (no plugin needed)."""

    def test_f16_rejected_on_non_tensor_sink(self) -> None:
        expr = pl.col("img").cv.pipe(Pipeline().source("image_bytes"))
        with pytest.raises(ValueError, match="numpy.*torch"):
            expr.sink("png", dtype="f16")

    def test_non_f16_dtype_rejected(self) -> None:
        expr = pl.col("img").cv.pipe(Pipeline().source("image_bytes"))
        with pytest.raises(ValueError, match="'dtype'.*SinkDType.*f16"):
            expr.sink("numpy", dtype="u8")

    def test_f16_accepted_on_numpy_and_torch(self) -> None:
        # Should build a valid expression without raising.
        for fmt in ("numpy", "torch"):
            expr = pl.col("img").cv.pipe(Pipeline().source("image_bytes"))
            assert expr.sink(fmt, dtype="f16") is not None


@plugin_required
class TestF16SinkExecution:
    """End-to-end: the numpy sink emits float16 with the right shape/values."""

    def _buffer_df(self) -> pl.DataFrame:
        # [2, 2, 1] f32 buffer.
        img = [[[0.0], [1.0]], [[2.0], [3.0]]]
        return pl.DataFrame(
            {"x": [img]},
            schema={"x": pl.List(pl.List(pl.List(pl.Float64)))},
        )

    def test_numpy_f16_downcast(self) -> None:
        from polars_cv import numpy_from_struct

        df = self._buffer_df()
        pipe = Pipeline().source("list", dtype="f32")
        out = (
            df.lazy()
            .with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy", dtype="f16"))
            .collect()
        )
        arr = numpy_from_struct(out["out"][0])
        assert arr.dtype == np.float16
        assert arr.shape == (2, 2, 1)
        np.testing.assert_array_equal(
            arr.astype(np.float32).ravel(), [0.0, 1.0, 2.0, 3.0]
        )

    def test_f16_halves_byte_cost_vs_f32(self) -> None:
        from polars_cv import numpy_from_struct

        df = self._buffer_df()
        pipe = Pipeline().source("list", dtype="f32")
        f32 = (
            df.lazy()
            .with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy"))
            .collect()
        )
        f16 = (
            df.lazy()
            .with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy", dtype="f16"))
            .collect()
        )
        a32 = numpy_from_struct(f32["out"][0])
        a16 = numpy_from_struct(f16["out"][0])
        assert a16.nbytes * 2 == a32.nbytes

    def test_f16_from_strided_buffer(self) -> None:
        # A transpose yields a non-contiguous (permuted-stride) buffer; the f16
        # downcast must materialize its logical (transposed) layout correctly.
        from polars_cv import numpy_from_struct

        img = [[[0.0], [1.0]], [[2.0], [3.0]]]  # [2, 2, 1]
        df = pl.DataFrame(
            {"x": [img]}, schema={"x": pl.List(pl.List(pl.List(pl.Float64)))}
        )
        pipe = Pipeline().source("list", dtype="f32").transpose([1, 0, 2])
        f32 = (
            df.lazy()
            .with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy"))
            .collect()
        )
        f16 = (
            df.lazy()
            .with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy", dtype="f16"))
            .collect()
        )
        a32 = numpy_from_struct(f32["out"][0])
        a16 = numpy_from_struct(f16["out"][0])
        assert a16.dtype == np.float16
        assert a16.shape == a32.shape
        np.testing.assert_array_equal(a16.astype(np.float32), a32)

    def test_f16_matches_numpy_rounding_on_edge_values(self) -> None:
        # Every edge of binary16, against NumPy's own float32 -> float16
        # conversion (round to nearest-even): values that round to f16's
        # largest finite value or overflow to infinity, subnormals and the
        # underflow edge, ties, ±0, ±inf and NaN. Many rows, so they are
        # converted on the plugin's row threads, and a transposed view.
        from polars_cv import numpy_from_struct

        edges = np.array(
            [
                0.0,
                -0.0,
                np.inf,
                -np.inf,
                np.nan,
                65504.0,
                65519.99,
                65520.0,
                -65520.0,
                1e10,
                6.1035156e-5,
                5.9604645e-8,
                2.9802322e-8,
                2.980233e-8,
                1e-30,
                1.0 + 1.0 / 2048.0,
                1.0 + 3.0 / 2048.0,
                0.1,
                -1234.567,
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(7)
        noise = rng.standard_normal((64, 62 * 4 - edges.size)).astype(np.float32)
        rows = [
            np.concatenate([edges * np.float32(10.0 ** (i % 7 - 3)), noise[i]]).reshape(
                62, 4
            )
            for i in range(64)
        ]
        df = pl.DataFrame(
            {"x": [r.tolist() for r in rows]},
            schema={"x": pl.List(pl.List(pl.Float32))},
        )
        for transpose in (False, True):
            pipe = Pipeline().source("list", dtype="f32")
            if transpose:
                pipe = pipe.transpose([1, 0])
            out = df.with_columns(
                out=pl.col("x").cv.pipe(pipe).sink("numpy", dtype="f16")
            )
            for row, value in zip(rows, out["out"], strict=True):
                # 65520 and 1e10 overflow to infinity, as intended.
                with np.errstate(over="ignore"):
                    want = (row.T if transpose else row).astype(np.float16)
                got = numpy_from_struct(value)
                assert got.dtype == np.float16
                assert got.shape == want.shape
                nan = np.isnan(want)
                np.testing.assert_array_equal(np.isnan(got), nan)
                np.testing.assert_array_equal(
                    got[~nan].view(np.uint16), want[~nan].view(np.uint16)
                )

    def test_f16_of_an_integer_image_reads_it_as_f32(self) -> None:
        # u16 values above 2048 are not all representable in f16: each
        # rounds to nearest-even, as NumPy's float32 -> float16 does.
        from polars_cv import numpy_from_struct

        img = np.arange(0, 65536, 7, dtype=np.uint16)[: 90 * 100].reshape(90, 100)
        df = pl.DataFrame(
            {"x": [img.tolist()]}, schema={"x": pl.List(pl.List(pl.UInt16))}
        )
        pipe = Pipeline().source("list", dtype="u16")
        out = df.with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy", dtype="f16"))
        got = numpy_from_struct(out["out"][0])
        want = img.astype(np.float32).astype(np.float16)
        np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))
