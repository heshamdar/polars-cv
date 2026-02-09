"""
Tests filling gaps in binary operation coverage.

Covers: blend/ratio via LazyPipelineExpr execution, bitwise_xor execution,
maximum/minimum with NumPy reference comparison, and apply_mask with
invert=True.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct


def _plugin_available() -> bool:
    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


@pytest.fixture
def encode_png() -> Callable[[np.ndarray], bytes]:
    def _encode(arr: np.ndarray) -> bytes:
        from PIL import Image

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return _encode


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
# ratio execution + reference
# ---------------------------------------------------------------------------


@plugin_required
class TestRatioExecution:
    """Test ratio operation end-to-end against NumPy reference."""

    def test_ratio_matches_reference(
        self,
        sample_pair: tuple[np.ndarray, np.ndarray, bytes, bytes],
    ) -> None:
        """Ratio should match (a/b)*255, clamped semantics."""
        img1, img2, png1, png2 = sample_pair

        # NumPy reference
        expected = np.zeros_like(img1, dtype=np.uint8)
        zero_mask = img2 == 0
        nonzero_mask = ~zero_mask
        expected[nonzero_mask] = np.clip(
            (img1[nonzero_mask].astype(np.uint32) * 255)
            // img2[nonzero_mask].astype(np.uint32),
            0,
            255,
        ).astype(np.uint8)
        expected[zero_mask & (img1 == 0)] = 0
        expected[zero_mask & (img1 != 0)] = 255

        df = pl.DataFrame({"img1": [png1], "img2": [png2]})
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")
        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(out=expr1.ratio(expr2).sink("numpy"))
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
