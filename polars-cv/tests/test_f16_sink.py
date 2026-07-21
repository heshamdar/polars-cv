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
        with pytest.raises(ValueError, match="only supports 'f16'"):
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
