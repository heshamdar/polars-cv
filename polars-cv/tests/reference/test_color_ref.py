"""Reference tests for Phase 2: Color space conversions.

Compares polars-cv plugin output against OpenCV / NumPy / PIL ground truth
to verify correctness of cvt_color and convenience methods.
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
    """64x64x3 RGB test image with diverse pixel values."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def rgb_png(rgb_image: np.ndarray) -> bytes:
    """PNG-encoded version of the RGB test image."""
    return _encode_png(rgb_image)


# ===========================================================================
# Helper: run pipeline and get numpy result
# ===========================================================================


def _run_pipe(pipe: Pipeline, png_bytes: bytes) -> np.ndarray:
    """Execute a pipeline on a single PNG image and return numpy result."""
    df = pl.DataFrame({"img": [png_bytes]})
    result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
    return numpy_from_struct(result.row(0)[0])


# ===========================================================================
# RGB ↔ BGR
# ===========================================================================


@plugin_required
class TestRgbBgr:
    """Reference tests for RGB ↔ BGR conversion."""

    def test_rgb_to_bgr(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify RGB to BGR matches NumPy channel flip."""
        expected = rgb_image[:, :, ::-1].copy()
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "bgr")
        actual = _run_pipe(pipe, rgb_png)
        np.testing.assert_array_equal(actual, expected)

    def test_bgr_to_rgb_roundtrip(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify RGB → BGR → RGB round-trip recovers original."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .cvt_color("rgb", "bgr")
            .cvt_color("bgr", "rgb")
        )
        actual = _run_pipe(pipe, rgb_png)
        np.testing.assert_array_equal(actual, rgb_image)

    def test_to_bgr_convenience(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify to_bgr() convenience method matches cvt_color."""
        pipe = Pipeline().source("image_bytes").to_bgr()
        actual = _run_pipe(pipe, rgb_png)
        expected = rgb_image[:, :, ::-1].copy()
        np.testing.assert_array_equal(actual, expected)


# ===========================================================================
# RGB ↔ HSV
# ===========================================================================


@plugin_required
class TestRgbHsv:
    """Reference tests for RGB ↔ HSV conversion against OpenCV."""

    def test_rgb_to_hsv_vs_opencv(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify RGB to HSV matches OpenCV cv2.cvtColor.

        Near-gray pixels have undefined hue, so we compare S/V channels with
        tight tolerance and H channel with wider tolerance for edge cases.
        """
        cv2 = pytest.importorskip("cv2")
        # OpenCV expects BGR input for COLOR_BGR2HSV
        bgr = rgb_image[:, :, ::-1].copy()
        expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
        actual = _run_pipe(pipe, rgb_png)
        # S and V channels: tight tolerance (±2)
        np.testing.assert_allclose(actual[:, :, 1], expected[:, :, 1], atol=2)
        np.testing.assert_allclose(actual[:, :, 2], expected[:, :, 2], atol=2)
        # H channel: wider tolerance — near-gray pixels have unstable hue
        # and hue wraps around at 0/180 boundary
        h_diff = np.abs(actual[:, :, 0].astype(int) - expected[:, :, 0].astype(int))
        h_diff_wrapped = np.minimum(h_diff, 180 - h_diff)
        assert np.percentile(h_diff_wrapped, 99.9) <= 2

    def test_hsv_roundtrip(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify RGB → HSV → RGB round-trip within tolerance.

        HSV uses OpenCV's [0,180] hue range for u8, which causes significant
        quantization on hue → u8 → hue. Combined with the u8 clamping on S/V,
        round-trip error of ±8 is expected for some pixels.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .cvt_color("rgb", "hsv")
            .cvt_color("hsv", "rgb")
        )
        actual = _run_pipe(pipe, rgb_png)
        np.testing.assert_allclose(actual, rgb_image, atol=8)

    def test_to_hsv_convenience(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify to_hsv() matches cvt_color("rgb", "hsv")."""
        pipe_a = Pipeline().source("image_bytes").to_hsv()
        pipe_b = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
        actual_a = _run_pipe(pipe_a, rgb_png)
        actual_b = _run_pipe(pipe_b, rgb_png)
        np.testing.assert_array_equal(actual_a, actual_b)

    def test_pure_red_hsv(self) -> None:
        """Verify pure red pixel converts to H=0, S=255, V=255."""
        img = np.array([[[255, 0, 0]]], dtype=np.uint8)
        png = _encode_png(img)
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
        actual = _run_pipe(pipe, png)
        # H=0, S=255, V=255 (OpenCV convention)
        assert actual[0, 0, 0] == 0  # Hue
        assert actual[0, 0, 1] == 255  # Saturation
        assert actual[0, 0, 2] == 255  # Value


# ===========================================================================
# RGB ↔ YCbCr
# ===========================================================================


@plugin_required
class TestRgbYcbcr:
    """Reference tests for RGB ↔ YCbCr conversion."""

    def test_ycbcr_roundtrip(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify RGB → YCbCr → RGB round-trip within ±3.

        Two u8 quantization passes (RGB→YCbCr→u8, then YCbCr→RGB→u8)
        accumulate ±3 rounding error.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .cvt_color("rgb", "ycbcr")
            .cvt_color("ycbcr", "rgb")
        )
        actual = _run_pipe(pipe, rgb_png)
        np.testing.assert_allclose(actual, rgb_image, atol=3)

    def test_ycbcr_luma_formula(self) -> None:
        """Verify Y channel follows BT.601 luma formula."""
        img = np.array([[[100, 150, 200]]], dtype=np.uint8)
        png = _encode_png(img)
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "ycbcr")
        actual = _run_pipe(pipe, png)
        # Y = 0.299*100 + 0.587*150 + 0.114*200 = 29.9 + 88.05 + 22.8 = 140.75
        y_expected = 0.299 * 100 + 0.587 * 150 + 0.114 * 200
        assert abs(actual[0, 0, 0] - y_expected) <= 1

    def test_to_ycbcr_convenience(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify to_ycbcr() matches cvt_color("rgb", "ycbcr")."""
        pipe_a = Pipeline().source("image_bytes").to_ycbcr()
        pipe_b = Pipeline().source("image_bytes").cvt_color("rgb", "ycbcr")
        actual_a = _run_pipe(pipe_a, rgb_png)
        actual_b = _run_pipe(pipe_b, rgb_png)
        np.testing.assert_array_equal(actual_a, actual_b)


# ===========================================================================
# RGB ↔ LAB
# ===========================================================================


@plugin_required
class TestRgbLab:
    """Reference tests for RGB ↔ CIE LAB conversion."""

    def test_lab_roundtrip(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify RGB → LAB → RGB round-trip within ±2."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .cvt_color("rgb", "lab")
            .cvt_color("lab", "rgb")
        )
        actual = _run_pipe(pipe, png_bytes=rgb_png)
        # LAB round-trip stays in f32, compare in float space
        np.testing.assert_allclose(actual, rgb_image.astype(np.float32), atol=2.0)

    def test_lab_output_is_float(self, rgb_png: bytes) -> None:
        """Verify LAB conversion produces f32 output."""
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "lab")
        df = pl.DataFrame({"img": [rgb_png]})
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out_struct = result.row(0)[0]
        actual = numpy_from_struct(out_struct)
        assert actual.dtype == np.float32

    def test_lab_range(self, rgb_png: bytes) -> None:
        """Verify L is in [0, 100] and a/b are in reasonable range."""
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "lab")
        actual = _run_pipe(pipe, rgb_png)
        assert actual[:, :, 0].min() >= -1.0  # L (allow slight numerical noise)
        assert actual[:, :, 0].max() <= 101.0
        assert actual[:, :, 1].min() >= -130.0  # a
        assert actual[:, :, 1].max() <= 130.0
        assert actual[:, :, 2].min() >= -130.0  # b
        assert actual[:, :, 2].max() <= 130.0

    def test_pure_white_lab(self) -> None:
        """Verify pure white = L=100, a=0, b=0."""
        img = np.array([[[255, 255, 255]]], dtype=np.uint8)
        png = _encode_png(img)
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "lab")
        actual = _run_pipe(pipe, png)
        assert abs(actual[0, 0, 0] - 100.0) < 1.0
        assert abs(actual[0, 0, 1]) < 1.0
        assert abs(actual[0, 0, 2]) < 1.0

    def test_pure_black_lab(self) -> None:
        """Verify pure black = L=0, a=0, b=0."""
        img = np.array([[[0, 0, 0]]], dtype=np.uint8)
        png = _encode_png(img)
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "lab")
        actual = _run_pipe(pipe, png)
        assert abs(actual[0, 0, 0]) < 1.0
        assert abs(actual[0, 0, 1]) < 1.0
        assert abs(actual[0, 0, 2]) < 1.0

    def test_to_lab_convenience(self, rgb_png: bytes) -> None:
        """Verify to_lab() matches cvt_color("rgb", "lab")."""
        pipe_a = Pipeline().source("image_bytes").to_lab()
        pipe_b = Pipeline().source("image_bytes").cvt_color("rgb", "lab")
        actual_a = _run_pipe(pipe_a, rgb_png)
        actual_b = _run_pipe(pipe_b, rgb_png)
        np.testing.assert_array_equal(actual_a, actual_b)


# ===========================================================================
# RGB ↔ Grayscale
# ===========================================================================


@plugin_required
class TestRgbGray:
    """Reference tests for RGB ↔ Grayscale via cvt_color."""

    def test_rgb_to_gray(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify RGB to grayscale matches BT.601 formula."""
        # BT.601: Y = 0.299*R + 0.587*G + 0.114*B
        expected = np.round(
            0.299 * rgb_image[:, :, 0].astype(np.float64)
            + 0.587 * rgb_image[:, :, 1].astype(np.float64)
            + 0.114 * rgb_image[:, :, 2].astype(np.float64)
        ).astype(np.uint8)

        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "gray")
        actual = _run_pipe(pipe, rgb_png)
        # Allow ±1 for fixed-point rounding
        np.testing.assert_allclose(actual.squeeze(), expected, atol=1)


# ===========================================================================
# Edge Cases
# ===========================================================================


@plugin_required
class TestEdgeCases:
    """Edge cases for color conversion operations."""

    def test_noop_conversion(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify RGB → RGB is identity."""
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "rgb")
        actual = _run_pipe(pipe, rgb_png)
        np.testing.assert_array_equal(actual, rgb_image)

    def test_saturated_primaries_hsv(self) -> None:
        """Verify saturated primary colors convert correctly to HSV."""
        primaries = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        png = _encode_png(primaries)
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
        actual = _run_pipe(pipe, png)

        # Red: H=0 (or 180 for wrap), S=255, V=255
        assert actual[0, 0, 0] == 0
        assert actual[0, 0, 1] == 255
        assert actual[0, 0, 2] == 255

        # Green: H=60 (in [0,180] scale), S=255, V=255
        assert actual[0, 1, 0] == 60
        assert actual[0, 1, 1] == 255
        assert actual[0, 1, 2] == 255

        # Blue: H=120 (in [0,180] scale), S=255, V=255
        assert actual[0, 2, 0] == 120
        assert actual[0, 2, 1] == 255
        assert actual[0, 2, 2] == 255

    def test_pure_gray_hsv(self) -> None:
        """Verify pure gray (128,128,128) has S=0 in HSV."""
        img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        png = _encode_png(img)
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
        actual = _run_pipe(pipe, png)
        assert actual[0, 0, 1] == 0  # Saturation = 0 for gray

    def test_invalid_color_space(self) -> None:
        """Verify invalid color space name raises ValueError."""
        with pytest.raises(ValueError, match="not a valid ColorSpace"):
            Pipeline().source("image_bytes").cvt_color("rgb", "xyz")

    def test_pipeline_dtype_tracking_lab(self) -> None:
        """Verify pipeline tracks f32 dtype after LAB conversion."""
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "lab")
        assert pipe._output_dtype == "f32"

    def test_pipeline_dtype_tracking_hsv(self) -> None:
        """Verify pipeline preserves u8 dtype for HSV conversion."""
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
        # auto from image_bytes, preserved by cvt_color (non-LAB)
        assert pipe._output_dtype == "auto"


# ===========================================================================
# Cross-conversion chains
# ===========================================================================


@plugin_required
class TestCrossConversions:
    """Test conversions that route through intermediate color spaces."""

    def test_bgr_to_hsv(self, rgb_image: np.ndarray, rgb_png: bytes) -> None:
        """Verify BGR → HSV via intermediate RGB."""
        pipe_direct = (
            Pipeline()
            .source("image_bytes")
            .cvt_color("rgb", "bgr")
            .cvt_color("bgr", "hsv")
        )
        pipe_via_rgb = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
        actual_direct = _run_pipe(pipe_direct, rgb_png)
        actual_via_rgb = _run_pipe(pipe_via_rgb, rgb_png)
        np.testing.assert_allclose(actual_direct, actual_via_rgb, atol=2)

    def test_hsv_to_ycbcr(self, rgb_png: bytes) -> None:
        """Verify HSV → YCbCr round-trip through RGB.

        Multiple conversions with u8 quantization at each stage
        accumulate up to ±10 rounding error.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .cvt_color("rgb", "hsv")
            .cvt_color("hsv", "ycbcr")
            .cvt_color("ycbcr", "rgb")
        )
        pipe_ref = Pipeline().source("image_bytes")
        actual = _run_pipe(pipe, rgb_png)
        ref = _run_pipe(pipe_ref, rgb_png)
        np.testing.assert_allclose(actual, ref, atol=10)
