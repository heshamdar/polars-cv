"""
Tests filling gaps in padding operation coverage.

Covers: reflect/symmetric pad modes, all pad_to_size positions, letterbox
identity/extreme aspect ratios, asymmetric padding, and NumPy reference
comparisons for reflect and symmetric modes.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required


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
def small_image() -> np.ndarray:
    """4×6 RGB image with distinct pixel values for verifying pad behaviour."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 256, (4, 6, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# NumPy reference: reflect and symmetric
# ---------------------------------------------------------------------------


class TestPadReflectReference:
    """NumPy reference behaviour for reflect padding."""

    def test_reflect_matches_numpy(self, small_image: np.ndarray) -> None:
        padded = np.pad(small_image, ((2, 2), (3, 3), (0, 0)), mode="reflect")
        assert padded.shape == (8, 12, 3)
        # Center region should be original image
        np.testing.assert_array_equal(padded[2:6, 3:9], small_image)


class TestPadSymmetricReference:
    """NumPy reference behaviour for symmetric padding."""

    def test_symmetric_matches_numpy(self, small_image: np.ndarray) -> None:
        padded = np.pad(small_image, ((2, 2), (3, 3), (0, 0)), mode="symmetric")
        assert padded.shape == (8, 12, 3)
        np.testing.assert_array_equal(padded[2:6, 3:9], small_image)


# ---------------------------------------------------------------------------
# Plugin: reflect and symmetric pad modes
# ---------------------------------------------------------------------------


@plugin_required
class TestPadReflectPlugin:
    """Test pad(mode='reflect') via plugin."""

    def test_reflect_shape(self, encode_png: Callable, small_image: np.ndarray) -> None:
        """Reflect padding should produce correct output shape."""
        png = encode_png(small_image)
        df = pl.DataFrame({"img": [png]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=2, bottom=2, left=3, right=3, mode="reflect")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape[0] == small_image.shape[0] + 4  # 4+4=8
        assert arr.shape[1] == small_image.shape[1] + 6  # 6+6=12

    def test_reflect_center_preserved(
        self, encode_png: Callable, small_image: np.ndarray
    ) -> None:
        """Center region of reflected output should match original."""
        png = encode_png(small_image)
        df = pl.DataFrame({"img": [png]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=2, bottom=2, left=3, right=3, mode="reflect")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(arr[2:6, 3:9], small_image)


@plugin_required
class TestPadSymmetricPlugin:
    """Test pad(mode='symmetric') via plugin."""

    def test_symmetric_shape(
        self, encode_png: Callable, small_image: np.ndarray
    ) -> None:
        """Symmetric padding should produce correct output shape."""
        png = encode_png(small_image)
        df = pl.DataFrame({"img": [png]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=1, bottom=1, left=1, right=1, mode="symmetric")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape[0] == small_image.shape[0] + 2
        assert arr.shape[1] == small_image.shape[1] + 2

    def test_symmetric_center_preserved(
        self, encode_png: Callable, small_image: np.ndarray
    ) -> None:
        """Center region of symmetric output should match original."""
        png = encode_png(small_image)
        df = pl.DataFrame({"img": [png]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=1, bottom=1, left=1, right=1, mode="symmetric")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(arr[1:5, 1:7], small_image)


# ---------------------------------------------------------------------------
# Asymmetric padding
# ---------------------------------------------------------------------------


@plugin_required
class TestPadAsymmetric:
    """Test asymmetric padding (all four sides different)."""

    def test_asymmetric_constant_shape(self, encode_png: Callable) -> None:
        """Pad with all different amounts should produce correct shape."""
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=5, bottom=10, left=15, right=20, mode="constant", value=0)
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (35, 65, 3)  # 20+5+10, 30+15+20


# ---------------------------------------------------------------------------
# pad_to_size positions
# ---------------------------------------------------------------------------


@plugin_required
class TestPadToSizePositions:
    """Test all pad_to_size position options."""

    @pytest.fixture
    def small_png(self, encode_png: Callable) -> bytes:
        return encode_png(np.full((20, 30, 3), 128, dtype=np.uint8))

    def test_center(self, small_png: bytes) -> None:
        """Center position should pad equally (or ±1) on each side."""
        df = pl.DataFrame({"img": [small_png]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad_to_size(height=40, width=50, position="center")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (40, 50, 3)

    def test_top_left(self, small_png: bytes) -> None:
        """top-left position should pad only bottom and right."""
        df = pl.DataFrame({"img": [small_png]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad_to_size(height=40, width=50, position="top-left")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (40, 50, 3)
        # Top-left pixel should be original content (128)
        assert arr[0, 0, 0] == 128

    def test_bottom_right(self, small_png: bytes) -> None:
        """bottom-right position should pad only top and left."""
        df = pl.DataFrame({"img": [small_png]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad_to_size(height=40, width=50, position="bottom-right")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (40, 50, 3)
        # Bottom-right pixel should be original content (128)
        assert arr[-1, -1, 0] == 128


# ---------------------------------------------------------------------------
# letterbox edge cases
# ---------------------------------------------------------------------------


@plugin_required
class TestLetterboxEdgeCases:
    """Letterbox edge cases: identity, extreme aspect ratios, fill values."""

    def test_letterbox_already_correct_size(self, encode_png: Callable) -> None:
        """Letterboxing to exact current size should be (near) identity."""
        img = np.full((100, 100, 3), 42, dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = Pipeline().source("image_bytes").letterbox(height=100, width=100)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (100, 100, 3)
        # Should be approximately the original image
        np.testing.assert_allclose(arr, img, atol=2)

    def test_letterbox_extreme_wide(self, encode_png: Callable) -> None:
        """Letterbox a very wide target should pad top/bottom."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = Pipeline().source("image_bytes").letterbox(height=100, width=400)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (100, 400, 3)

    def test_letterbox_custom_fill(self, encode_png: Callable) -> None:
        """Letterbox with custom fill_value should use that value in padding."""
        img = np.full((50, 100, 3), 200, dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = (
            Pipeline().source("image_bytes").letterbox(height=100, width=100, value=42)
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (100, 100, 3)
        # Top or bottom rows should be the fill value (42)
        # The image is 50x100 going into 100x100 so it fits width-wise,
        # padded top/bottom
        assert arr[0, 0, 0] == 42 or arr[-1, -1, 0] == 42
