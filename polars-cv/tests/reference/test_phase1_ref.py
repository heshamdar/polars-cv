"""Reference tests for Phase 1 operations: channel ops, padding fixes, intensity adjustments.

Compares polars-cv plugin output against NumPy/PIL ground truth to verify correctness.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


def _encode_png(arr: np.ndarray) -> bytes:
    """Encode a numpy array as PNG bytes."""
    from PIL import Image

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def rgb_image() -> np.ndarray:
    """256x256x3 RGB test image with known pixel values."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def rgb_png(rgb_image: np.ndarray) -> bytes:
    """PNG-encoded version of the RGB test image."""
    return _encode_png(rgb_image)


# ===========================================================================
# 1A. Channel Operations
# ===========================================================================


@plugin_required
class TestChannelSelect:
    """Reference tests for channel_select vs NumPy indexing."""

    def test_select_red_channel(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify channel_select(0) matches NumPy img[:, :, 0]."""
        expected = rgb_image[:, :, 0]

        pipe = Pipeline().source("image_bytes").channel_select(index=0)
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, expected)

    def test_select_green_channel(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify channel_select(1) matches NumPy img[:, :, 1]."""
        expected = rgb_image[:, :, 1]

        pipe = Pipeline().source("image_bytes").channel_select(index=1)
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, expected)

    def test_select_blue_channel(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify channel_select(2) matches NumPy img[:, :, 2]."""
        expected = rgb_image[:, :, 2]

        pipe = Pipeline().source("image_bytes").channel_select(index=2)
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, expected)


@plugin_required
class TestChannelSwap:
    """Reference tests for channel_swap vs NumPy indexing."""

    def test_rgb_to_bgr(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify channel_swap([2,1,0]) matches img[:, :, ::-1]."""
        expected = rgb_image[:, :, ::-1]

        pipe = Pipeline().source("image_bytes").channel_swap(order=[2, 1, 0])
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, expected)

    def test_identity_swap(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify channel_swap([0,1,2]) is a no-op."""
        pipe = Pipeline().source("image_bytes").channel_swap(order=[0, 1, 2])
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, rgb_image)


# ===========================================================================
# 1B. Reflect / Symmetric Padding
# ===========================================================================


@plugin_required
class TestReflectPadding:
    """Reference tests for reflect and symmetric padding vs NumPy."""

    def test_reflect_pad_small(self) -> None:
        """Verify reflect padding matches np.pad(mode='reflect')."""
        arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
        # Grayscale PNG decodes as [H, W, 3] (all channels equal), then
        # .grayscale() produces [H, W, 1]. Expected must match this layout.
        expected_2d = np.pad(arr, ((1, 1), (2, 2)), mode="reflect")
        expected = expected_2d[:, :, np.newaxis]

        png = _encode_png(arr)
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .pad(top=1, bottom=1, left=2, right=2, mode="reflect")
        )
        df = pl.DataFrame({"img": [png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, expected)

    def test_symmetric_pad_small(self) -> None:
        """Verify symmetric padding matches np.pad(mode='symmetric')."""
        arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
        expected_2d = np.pad(arr, ((1, 1), (2, 2)), mode="symmetric")
        expected = expected_2d[:, :, np.newaxis]

        png = _encode_png(arr)
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .pad(top=1, bottom=1, left=2, right=2, mode="symmetric")
        )
        df = pl.DataFrame({"img": [png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, expected)

    def test_reflect_pad_rgb(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify reflect padding works for multi-channel images."""
        pad_width = ((2, 3), (4, 5), (0, 0))
        expected = np.pad(rgb_image, pad_width, mode="reflect")

        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=2, bottom=3, left=4, right=5, mode="reflect")
        )
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, expected)


# ===========================================================================
# 1C. Intensity Adjustments
# ===========================================================================


@plugin_required
class TestAdjustContrast:
    """Reference tests for adjust_contrast."""

    def test_contrast_factor_1_is_noop(
        self, rgb_image: np.ndarray, rgb_png: bytes
    ) -> None:
        """Factor=1.0 should produce the same image (within float precision)."""
        pipe = Pipeline().source("image_bytes").adjust_contrast(factor=1.0)
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_allclose(actual, rgb_image.astype(np.float32), atol=1e-4)

    def test_contrast_vs_numpy(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify contrast adjustment matches (pixel - mean) * factor + mean."""
        factor = 1.5
        img_f32 = rgb_image.astype(np.float32)
        mean = img_f32.mean()
        expected = (img_f32 - mean) * factor + mean

        pipe = Pipeline().source("image_bytes").adjust_contrast(factor=factor)
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_allclose(actual, expected, atol=1e-2)


@plugin_required
class TestAdjustGamma:
    """Reference tests for adjust_gamma (power-law)."""

    def test_gamma_1_is_noop(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Gamma=1.0 should preserve pixel values."""
        pipe = Pipeline().source("image_bytes").adjust_gamma(gamma=1.0)
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_allclose(actual, rgb_image.astype(np.float32), atol=1.0)

    def test_gamma_vs_numpy(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify gamma correction matches (pixel/255)^gamma * 255."""
        gamma = 0.5
        img_f32 = rgb_image.astype(np.float32)
        expected = np.power(img_f32 / 255.0, gamma) * 255.0

        pipe = Pipeline().source("image_bytes").adjust_gamma(gamma=gamma)
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_allclose(actual, expected, atol=1.0)


@plugin_required
class TestInvert:
    """Reference tests for invert vs NumPy."""

    def test_invert_u8(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify invert matches 255 - pixel for u8 images."""
        expected = 255 - rgb_image

        pipe = Pipeline().source("image_bytes").invert()
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, expected)

    def test_invert_roundtrip(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Double invert should recover the original image."""
        pipe = Pipeline().source("image_bytes").invert().invert()
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_array_equal(actual, rgb_image)


@plugin_required
class TestAdjustBrightness:
    """Reference tests for adjust_brightness (convenience method)."""

    def test_brightness_factor_1_preserves(
        self, rgb_image: np.ndarray, rgb_png: bytes
    ) -> None:
        """Factor=1.0 should produce equivalent values (after f32 promotion + clamp)."""
        pipe = Pipeline().source("image_bytes").adjust_brightness(factor=1.0)
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_allclose(actual, rgb_image.astype(np.float32), atol=1e-4)

    def test_brightness_saturates(self) -> None:
        """High brightness factor should saturate at 255."""
        arr = np.full((8, 8, 3), 200, dtype=np.uint8)
        png = _encode_png(arr)

        pipe = Pipeline().source("image_bytes").adjust_brightness(factor=2.0)
        df = pl.DataFrame({"img": [png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        np.testing.assert_allclose(actual, np.full((8, 8, 3), 255.0), atol=1e-4)


# ===========================================================================
# Pipeline builder contract tests (no plugin required)
# ===========================================================================


class TestPhase1Contracts:
    """Test that pipeline builder correctly tracks dtype/ndim for Phase 1 ops."""

    def test_channel_select_reduces_ndim(self) -> None:
        """channel_select should reduce ndim from 3 to 2."""
        pipe = Pipeline().source("image_bytes").channel_select(index=0)
        assert pipe.current_domain() == "buffer"

    def test_channel_swap_preserves_ndim(self) -> None:
        """channel_swap should preserve ndim."""
        pipe = Pipeline().source("image_bytes").channel_swap(order=[2, 1, 0])
        assert pipe.current_domain() == "buffer"

    def test_invert_preserves_dtype(self) -> None:
        """invert should preserve the input dtype (not promote to float)."""
        pipe = Pipeline().source("image_bytes").invert()
        assert pipe._output_dtype == "auto"

    def test_adjust_contrast_promotes_dtype(self) -> None:
        """adjust_contrast should promote integer dtypes to float."""
        pipe = Pipeline().source("image_bytes", dtype="u8").adjust_contrast(factor=1.5)
        assert pipe._output_dtype == "f32"

    def test_adjust_gamma_promotes_dtype(self) -> None:
        """adjust_gamma should promote integer dtypes to float."""
        pipe = Pipeline().source("image_bytes", dtype="u8").adjust_gamma(gamma=0.5)
        assert pipe._output_dtype == "f32"

    def test_brightness_chains_scale_clamp(self) -> None:
        """adjust_brightness should produce scale + clamp ops."""
        pipe = Pipeline().source("image_bytes").adjust_brightness(factor=1.5)
        op_names = [op.op for op in pipe._ops]
        assert "scale" in op_names
        assert "clamp" in op_names
