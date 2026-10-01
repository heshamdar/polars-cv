"""
Tests filling gaps in binary operation coverage.

Covers: blend via LazyPipelineExpr execution, bitwise_xor execution,
maximum/minimum with NumPy reference comparison, and apply_mask with
invert=True.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required


@pytest.fixture
def sample_pair(encode_png: Callable) -> tuple[np.ndarray, np.ndarray, bytes, bytes]:
    """Two 50×50 RGB images with known seed, plus their PNG encodings."""
    rng = np.random.default_rng(123)
    img1 = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    img2 = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    return img1, img2, encode_png(img1), encode_png(img2)


# ---------------------------------------------------------------------------
# blend execution + reference
# ---------------------------------------------------------------------------


@plugin_required
class TestBlendExecution:
    """Test blend operation end-to-end against NumPy reference."""

    def test_blend_matches_reference(
        self,
        sample_pair: tuple[np.ndarray, np.ndarray, bytes, bytes],
    ) -> None:
        """Blend should match (a*b+127)//255 semantics."""
        img1, img2, png1, png2 = sample_pair

        # NumPy reference: rounding blend
        expected = (
            (img1.astype(np.uint32) * img2.astype(np.uint32) + 127) // 255
        ).astype(np.uint8)

        df = pl.DataFrame({"img1": [png1], "img2": [png2]})
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")
        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(out=expr1.blend(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)


# ---------------------------------------------------------------------------
# bitwise_xor execution
# ---------------------------------------------------------------------------


@plugin_required
class TestBitwiseXorExecution:
    """Test bitwise_xor end-to-end."""

    def test_xor_matches_numpy(
        self,
        sample_pair: tuple[np.ndarray, np.ndarray, bytes, bytes],
    ) -> None:
        """XOR should match np.bitwise_xor."""
        img1, img2, png1, png2 = sample_pair
        expected = np.bitwise_xor(img1, img2)

        df = pl.DataFrame({"img1": [png1], "img2": [png2]})
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")
        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(out=expr1.bitwise_xor(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# maximum / minimum reference
# ---------------------------------------------------------------------------


@plugin_required
class TestMaximumMinimumReference:
    """Test maximum/minimum against NumPy reference."""

    def test_maximum_matches_numpy(
        self,
        sample_pair: tuple[np.ndarray, np.ndarray, bytes, bytes],
    ) -> None:
        """Element-wise maximum should match np.maximum."""
        img1, img2, png1, png2 = sample_pair
        expected = np.maximum(img1, img2)

        df = pl.DataFrame({"img1": [png1], "img2": [png2]})
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")
        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(out=expr1.maximum(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_array_equal(actual, expected)

    def test_minimum_matches_numpy(
        self,
        sample_pair: tuple[np.ndarray, np.ndarray, bytes, bytes],
    ) -> None:
        """Element-wise minimum should match np.minimum."""
        img1, img2, png1, png2 = sample_pair
        expected = np.minimum(img1, img2)

        df = pl.DataFrame({"img1": [png1], "img2": [png2]})
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")
        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(out=expr1.minimum(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# apply_mask with invert=True
# ---------------------------------------------------------------------------


@plugin_required
class TestApplyMaskInvert:
    """Test apply_mask with invert parameter."""

    def test_apply_mask_inverted(self, encode_png: Callable) -> None:
        """Inverted mask should zero out inside the mask, keep outside."""
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        # Mask: center 20×20 is white (255)
        mask = np.zeros((50, 50, 3), dtype=np.uint8)
        mask[15:35, 15:35] = 255

        df = pl.DataFrame(
            {
                "image": [encode_png(img)],
                "mask": [encode_png(mask)],
            }
        )

        img_pipe = Pipeline().source("image_bytes")
        mask_pipe = Pipeline().source("image_bytes").grayscale()

        img_expr = pl.col("image").cv.pipe(img_pipe)
        mask_expr = pl.col("mask").cv.pipe(mask_pipe)

        result = df.select(
            out=img_expr.apply_mask(mask_expr, invert=True).sink("numpy")
        )
        actual = numpy_from_struct(result.row(0)[0])

        # Inverted: center should be zeroed, edges should be preserved
        assert actual.shape == (50, 50, 3)
        # Center pixel (inside mask) should be zero or near-zero
        assert actual[25, 25, 0] < 10
        # Corner pixel (outside mask) should be near-original
        assert actual[0, 0, 0] > 190

    def test_apply_mask_normal_vs_inverted_are_different(
        self, encode_png: Callable
    ) -> None:
        """Normal and inverted mask should produce different results."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
        mask = np.zeros((30, 30, 3), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        df = pl.DataFrame(
            {
                "image": [encode_png(img)],
                "mask": [encode_png(mask)],
            }
        )

        img_pipe = Pipeline().source("image_bytes")
        mask_pipe = Pipeline().source("image_bytes").grayscale()

        img_expr = pl.col("image").cv.pipe(img_pipe)
        mask_expr = pl.col("mask").cv.pipe(mask_pipe)

        r_normal = df.select(out=img_expr.apply_mask(mask_expr).sink("numpy"))
        r_invert = df.select(
            out=img_expr.apply_mask(mask_expr, invert=True).sink("numpy")
        )

        arr_normal = numpy_from_struct(r_normal.row(0)[0])
        arr_invert = numpy_from_struct(r_invert.row(0)[0])

        assert not np.array_equal(arr_normal, arr_invert)


@plugin_required
class TestMixedDtypePromotion:
    """Operands of different dtypes combine in NumPy's promoted dtype, at plan
    time and at execution alike: u8 with i8 is i16, so 200 + (-100) is 100.
    The larger-integer rule promoted to i8 and lost the 200."""

    @staticmethod
    def _frame() -> pl.DataFrame:
        a = np.array([[[200], [0]]], dtype=np.uint8)
        b = np.array([[[-100], [-128]]], dtype=np.int8)
        return pl.DataFrame(
            {
                "a": pl.Series([a], dtype=pl.Array(pl.UInt8, (1, 2, 1))),
                "b": pl.Series([b], dtype=pl.Array(pl.Int8, (1, 2, 1))),
            }
        )

    def test_u8_with_i8_is_i16(self) -> None:
        expr = (
            pl.col("a")
            .cv.pipe(Pipeline().source("array"))
            .add(pl.col("b").cv.pipe(Pipeline().source("array")))
            .sink("array")
        )
        lf = self._frame().lazy().select(out=expr)
        assert lf.collect_schema()["out"] == pl.Array(pl.Int16, (1, 2, 1))
        out = np.asarray(lf.collect()["out"][0].to_list())
        np.testing.assert_array_equal(out.ravel(), [100, -128])
        ref = np.array([200, 0], np.uint8) + np.array([-100, -128], np.int8)
        assert ref.dtype == np.int16

    def test_bitwise_without_a_common_integer_is_refused(self) -> None:
        df = pl.DataFrame(
            {
                "a": pl.Series(
                    [np.ones((1, 1, 1), np.uint64)],
                    dtype=pl.Array(pl.UInt64, (1, 1, 1)),
                ),
                "b": pl.Series(
                    [np.ones((1, 1, 1), np.int64)], dtype=pl.Array(pl.Int64, (1, 1, 1))
                ),
            }
        )
        expr = (
            pl.col("a")
            .cv.pipe(Pipeline().source("array"))
            .bitwise_and(pl.col("b").cv.pipe(Pipeline().source("array")))
            .sink("numpy")
        )
        with pytest.raises(
            (ValueError, pl.exceptions.ComputeError), match="no common integer"
        ):
            df.select(expr)
