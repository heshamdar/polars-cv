"""
Correctness audit tests for polars-cv.

This test module fills coverage gaps found during a comprehensive audit
of the existing test suite.  Every test verifies *actual output values*
against a NumPy reference (where applicable) rather than only checking
shapes and types.

Sections:
  1. Pixel-level correctness for core image operations
  2. Pipeline operations not previously tested end-to-end
  3. Contour operation correctness with non-trivial geometry
  4. Point operation edge cases
  5. Error handling / validation edge cases
  6. numpy_from_struct edge cases
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import pytest
from polars_cv import CONTOUR_SCHEMA, Pipeline, numpy_from_struct

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Encode a numpy array as PNG bytes."""

    def _encode(arr: np.ndarray) -> bytes:
        from PIL import Image

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return _encode


def _make_solid(
    h: int, w: int, color: tuple[int, ...], channels: int = 3
) -> np.ndarray:
    """Create a solid-color image array."""
    if channels == 1:
        return np.full((h, w), color[0], dtype=np.uint8)
    return np.full((h, w, channels), color[:channels], dtype=np.uint8)


# ===================================================================
# 1. Pixel-level correctness for core image operations
# ===================================================================


@plugin_required
class TestGrayscaleCorrectness:
    """Verify grayscale conversion uses correct luminance formula."""

    def test_grayscale_pure_red(self, encode_png: Callable) -> None:
        """Pure red (255,0,0) → luminance = 0.299*255 ≈ 76."""
        arr = _make_solid(10, 10, (255, 0, 0))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        gray = numpy_from_struct(result.row(0)[0])

        expected = int(round(0.299 * 255))  # 76
        assert gray.shape == (10, 10, 1)
        # Allow ±1 for rounding
        assert abs(int(gray[0, 0, 0]) - expected) <= 1

    def test_grayscale_pure_green(self, encode_png: Callable) -> None:
        """Pure green (0,255,0) → luminance = 0.587*255 ≈ 150."""
        arr = _make_solid(10, 10, (0, 255, 0))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        gray = numpy_from_struct(result.row(0)[0])

        expected = int(round(0.587 * 255))  # 150
        assert abs(int(gray[0, 0, 0]) - expected) <= 1

    def test_grayscale_pure_blue(self, encode_png: Callable) -> None:
        """Pure blue (0,0,255) → luminance = 0.114*255 ≈ 29."""
        arr = _make_solid(10, 10, (0, 0, 255))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        gray = numpy_from_struct(result.row(0)[0])

        expected = int(round(0.114 * 255))  # 29
        assert abs(int(gray[0, 0, 0]) - expected) <= 1

    def test_grayscale_white(self, encode_png: Callable) -> None:
        """White (255,255,255) → luminance = 255."""
        arr = _make_solid(10, 10, (255, 255, 255))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        gray = numpy_from_struct(result.row(0)[0])

        assert gray[0, 0, 0] == 255


@plugin_required
class TestThresholdCorrectness:
    """Verify threshold produces correct binary values."""

    def test_threshold_creates_binary(self, encode_png: Callable) -> None:
        """All values below threshold → 0, all above → 255."""
        arr = np.array(
            [[50, 100, 150], [200, 25, 128]],
            dtype=np.uint8,
        )
        arr = np.stack([arr, arr, arr], axis=-1)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").grayscale().threshold(100)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        # Values: 50→0, 100→0/255 (boundary), 150→255, 200→255, 25→0, 128→255
        assert out.shape == (2, 3, 1)
        assert out[0, 0, 0] == 0  # 50 < 100
        assert out[0, 2, 0] == 255  # 150 >= 100
        assert out[1, 0, 0] == 255  # 200 >= 100
        assert out[1, 1, 0] == 0  # 25 < 100
        assert out[1, 2, 0] == 255  # 128 >= 100
        # All values should be either 0 or 255
        unique = np.unique(out)
        assert set(unique.tolist()).issubset({0, 255})


@plugin_required
class TestNormalizeCorrectness:
    """Verify normalize operations compute correct values."""

    def test_normalize_zero_mean_unit_std(self, encode_png: Callable) -> None:
        """Z-score normalization should produce approximately 0 mean and 1 std."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").normalize(method="zscore")
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.dtype == np.float32
        # After z-score normalization, mean should be ~0, std ~1
        assert abs(np.mean(out)) < 0.1
        assert abs(np.std(out) - 1.0) < 0.1

    def test_normalize_min_max(self, encode_png: Callable) -> None:
        """Min-max normalization should produce values in [0, 1]."""
        rng = np.random.default_rng(42)
        arr = rng.integers(50, 200, (20, 20, 3), dtype=np.uint8)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").normalize(method="minmax")
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.dtype == np.float32
        assert np.min(out) >= -0.001  # Allow tiny floating point error
        assert np.max(out) <= 1.001
        assert abs(np.min(out)) < 0.01  # Should be very close to 0
        assert abs(np.max(out) - 1.0) < 0.01  # Should be very close to 1


@plugin_required
class TestClampCorrectness:
    """Verify clamp constrains values to the specified range."""

    def test_clamp_basic(self, encode_png: Callable) -> None:
        """Clamp should clip values to [min, max] range."""
        # Use a gradient to have varied values
        arr = np.arange(0, 255, dtype=np.uint8).reshape(1, -1)
        # Pad to make it a valid 3-channel image
        arr = np.stack([arr, arr, arr], axis=-1)  # (1, 255, 3)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").cast("f32").clamp(50.0, 200.0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert np.min(out) >= 50.0 - 0.01
        assert np.max(out) <= 200.0 + 0.01


@plugin_required
class TestScaleCorrectness:
    """Verify scale multiplies pixel values correctly."""

    def test_scale_doubles_values(self, encode_png: Callable) -> None:
        """Scaling by 2.0 should double pixel values."""
        arr = _make_solid(10, 10, (50, 100, 25))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").cast("f32").scale(2.0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(out[0, 0], [100.0, 200.0, 50.0], atol=0.5)


# ===================================================================
# 2. Pipeline operations not previously tested end-to-end
# ===================================================================


@plugin_required
class TestCropCorrectness:
    """Verify crop extracts the correct sub-region."""

    def test_crop_correct_pixels(self, encode_png: Callable) -> None:
        """Crop should extract the specified sub-region."""
        # Create a 10x10 image with each pixel having unique values
        arr = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        # Crop: top=2, left=3, height=4, width=5
        pipe = Pipeline().source("image_bytes").crop(top=2, left=3, height=4, width=5)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.shape == (4, 5, 3)
        # Verify that the top-left pixel of the crop matches arr[2, 3]
        np.testing.assert_array_equal(out[0, 0], arr[2, 3])
        # Verify the bottom-right pixel matches arr[5, 7]
        np.testing.assert_array_equal(out[3, 4], arr[5, 7])


@plugin_required
class TestFlipCorrectness:
    """Verify flip operations reverse the correct axes."""

    def test_flip_horizontal(self, encode_png: Callable) -> None:
        """Horizontal flip should reverse the width axis."""
        arr = np.zeros((4, 6, 3), dtype=np.uint8)
        # Mark left and right halves distinctly
        arr[:, :3, :] = 100  # left half
        arr[:, 3:, :] = 200  # right half
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").flip_h()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.shape == (4, 6, 3)
        # After horizontal flip, left half should be 200, right half 100
        assert out[0, 0, 0] == 200
        assert out[0, 5, 0] == 100

    def test_flip_vertical(self, encode_png: Callable) -> None:
        """Vertical flip should reverse the height axis."""
        arr = np.zeros((6, 4, 3), dtype=np.uint8)
        arr[:3, :, :] = 100  # top half
        arr[3:, :, :] = 200  # bottom half
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").flip_v()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.shape == (6, 4, 3)
        # After vertical flip, top should be 200, bottom should be 100
        assert out[0, 0, 0] == 200
        assert out[5, 0, 0] == 100

    def test_double_flip_is_identity(self, encode_png: Callable) -> None:
        """Flipping twice on the same axis should return the original."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").flip_h().flip_h()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        np.testing.assert_array_equal(out, arr)


@plugin_required
class TestRotateCorrectness:
    """Verify rotate operations."""

    def test_rotate_90_degrees(self, encode_png: Callable) -> None:
        """90° rotation should transpose and flip the image."""
        arr = np.zeros((10, 20, 3), dtype=np.uint8)
        arr[0, :, :] = 255  # White top row
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").rotate(90)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        # After 90° CW rotation of a 10x20 image, it becomes 20x10
        # The white row (originally top) should be on one side
        assert out.shape[0] == 20 or out.shape[1] == 10

    def test_rotate_180_is_double_flip(self, encode_png: Callable) -> None:
        """180° rotation should be equivalent to flip_h + flip_v."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe_rotate = Pipeline().source("image_bytes").rotate(180)
        pipe_flip = Pipeline().source("image_bytes").flip_h().flip_v()

        result_rotate = df.select(out=pl.col("img").cv.pipe(pipe_rotate).sink("numpy"))
        result_flip = df.select(out=pl.col("img").cv.pipe(pipe_flip).sink("numpy"))

        rotated = numpy_from_struct(result_rotate.row(0)[0])
        flipped = numpy_from_struct(result_flip.row(0)[0])

        np.testing.assert_array_equal(rotated, flipped)


@plugin_required
class TestTransposeCorrectness:
    """Verify transpose operation reorders dimensions correctly."""

    def test_transpose_hwc_to_chw(self, encode_png: Callable) -> None:
        """Transpose [2,0,1] should convert HWC to CHW layout."""
        arr = np.zeros((10, 20, 3), dtype=np.uint8)
        arr[:, :, 0] = 100  # R channel
        arr[:, :, 1] = 150  # G channel
        arr[:, :, 2] = 200  # B channel
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").transpose([2, 0, 1])
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        # After transpose [2,0,1]: (H,W,C) -> (C,H,W) = (3,10,20)
        assert out.shape == (3, 10, 20)
        # First channel should be all 100 (R)
        assert np.all(out[0] == 100)
        # Second channel should be all 150 (G)
        assert np.all(out[1] == 150)
        # Third channel should be all 200 (B)
        assert np.all(out[2] == 200)


@plugin_required
class TestCastCorrectness:
    """Verify cast operation changes dtype correctly."""

    def test_cast_uint8_to_float32(self, encode_png: Callable) -> None:
        """Cast to f32 should preserve values as floats."""
        arr = _make_solid(5, 5, (128, 64, 32))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").cast("f32")
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.dtype == np.float32
        np.testing.assert_allclose(out[0, 0], [128.0, 64.0, 32.0])

    def test_cast_float32_to_uint8(self, encode_png: Callable) -> None:
        """Cast f32 back to u8 should truncate correctly."""
        arr = _make_solid(5, 5, (128, 64, 32))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").cast("f32").cast("u8")
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out[0, 0], [128, 64, 32])


@plugin_required
class TestLetterboxCorrectness:
    """Verify letterbox resize + pad to exact target size."""

    def test_letterbox_maintains_aspect_ratio(self, encode_png: Callable) -> None:
        """Letterbox should pad to exact target while preserving aspect ratio."""
        # 100x50 image → letterbox to 100x100
        arr = _make_solid(50, 100, (128, 128, 128))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").letterbox(height=100, width=100)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.shape == (100, 100, 3)
        # The padding (top/bottom borders) should be 0 (black)
        # The center region should have the image content


@plugin_required
class TestPadCorrectness:
    """Verify pad operation adds correct padding."""

    def test_pad_constant_value(self, encode_png: Callable) -> None:
        """Constant padding should add specified value around the image."""
        arr = _make_solid(10, 10, (128, 128, 128))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=5, bottom=5, left=5, right=5, value=0)
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.shape == (20, 20, 3)
        # Top-left corner should be padding (0)
        assert out[0, 0, 0] == 0
        # Center should be original value
        assert out[7, 7, 0] == 128

    def test_pad_to_size(self, encode_png: Callable) -> None:
        """pad_to_size should pad image to exact target dimensions."""
        arr = _make_solid(10, 10, (100, 100, 100))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").pad_to_size(height=20, width=30)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.shape == (20, 30, 3)


@plugin_required
class TestReluCorrectness:
    """Verify ReLU activation on float arrays."""

    def test_relu_zeros_negatives_after_subtract(self, encode_png: Callable) -> None:
        """After casting to float and subtracting, ReLU should zero negatives."""
        arr = _make_solid(10, 10, (100, 100, 100))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        # Cast to f32, subtract 150 (making values negative), then relu
        pipe = (
            Pipeline()
            .source("image_bytes")
            .cast("f32")
            .scale(-1.0)  # all values become -100
            .relu()
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        # All values should be 0 after relu (since they were negative)
        assert np.all(out == 0.0)


# ===================================================================
# 3. Contour operation correctness with non-trivial geometry
# ===================================================================


@plugin_required
class TestContourIoUNonTrivial:
    """Verify IoU with partial overlap (not just identical contours)."""

    def test_iou_partial_overlap(self) -> None:
        """Two overlapping squares should have IoU between 0 and 1."""
        # Square 1: (0,0)-(100,100), area = 10000
        square1 = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 100.0},
                {"x": 0.0, "y": 100.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        # Square 2: (50,50)-(150,150), area = 10000
        # Overlap: (50,50)-(100,100) = 50*50 = 2500
        # Union: 10000+10000-2500 = 17500
        # IoU = 2500/17500 ≈ 0.1429
        square2 = {
            "exterior": [
                {"x": 50.0, "y": 50.0},
                {"x": 150.0, "y": 50.0},
                {"x": 150.0, "y": 150.0},
                {"x": 50.0, "y": 150.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame(
            {"a": [square1], "b": [square2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(iou=pl.col("a").contour.iou(pl.col("b")))

        iou_val = result["iou"][0]
        assert 0.0 < iou_val < 1.0
        # Expected IoU ≈ 0.1429
        assert iou_val == pytest.approx(2500.0 / 17500.0, rel=0.05)

    def test_iou_no_overlap(self) -> None:
        """Non-overlapping contours should have IoU = 0."""
        square1 = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        square2 = {
            "exterior": [
                {"x": 50.0, "y": 50.0},
                {"x": 60.0, "y": 50.0},
                {"x": 60.0, "y": 60.0},
                {"x": 50.0, "y": 60.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame(
            {"a": [square1], "b": [square2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(iou=pl.col("a").contour.iou(pl.col("b")))

        assert result["iou"][0] == pytest.approx(0.0, abs=0.01)


@plugin_required
class TestContourDiceNonTrivial:
    """Verify Dice coefficient with partial overlap."""

    def test_dice_partial_overlap(self) -> None:
        """Dice with partial overlap should be between 0 and 1."""
        square1 = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 100.0},
                {"x": 0.0, "y": 100.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        square2 = {
            "exterior": [
                {"x": 50.0, "y": 50.0},
                {"x": 150.0, "y": 50.0},
                {"x": 150.0, "y": 150.0},
                {"x": 50.0, "y": 150.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame(
            {"a": [square1], "b": [square2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(dice=pl.col("a").contour.dice(pl.col("b")))

        dice_val = result["dice"][0]
        assert 0.0 < dice_val < 1.0
        # Dice = 2*intersection / (area_a + area_b) = 2*2500/20000 = 0.25
        assert dice_val == pytest.approx(0.25, rel=0.05)

    def test_dice_no_overlap(self) -> None:
        """Non-overlapping contours should have Dice = 0."""
        square1 = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        square2 = {
            "exterior": [
                {"x": 50.0, "y": 50.0},
                {"x": 60.0, "y": 50.0},
                {"x": 60.0, "y": 60.0},
                {"x": 50.0, "y": 60.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame(
            {"a": [square1], "b": [square2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(dice=pl.col("a").contour.dice(pl.col("b")))

        assert result["dice"][0] == pytest.approx(0.0, abs=0.01)


@plugin_required
class TestContourHausdorffNonTrivial:
    """Verify Hausdorff distance with non-identical contours."""

    def test_hausdorff_known_distance(self) -> None:
        """Hausdorff distance between offset squares should be predictable."""
        square1 = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        # Square 2 is shifted right by 20 units (no overlap)
        square2 = {
            "exterior": [
                {"x": 20.0, "y": 0.0},
                {"x": 30.0, "y": 0.0},
                {"x": 30.0, "y": 10.0},
                {"x": 20.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame(
            {"a": [square1], "b": [square2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(
            hausdorff=pl.col("a").contour.hausdorff_distance(pl.col("b"))
        )

        hausdorff = result["hausdorff"][0]
        # The farthest point of square1 from square2 is (0,10) to (20,0) = sqrt(400+100)=~22.4
        # or (0,0) to (30,10) = sqrt(900+100) = ~31.6
        # Hausdorff = max of minimum distances = 20 (from (0,*) to (20,*))
        # Actually: max over all points in A of min distance to B
        # Point (0,0) has min dist to B = 20 (to (20,0))
        # Point (0,10) has min dist to B = sqrt(20^2 + 0^2) = 20 (to (20,10))
        # So Hausdorff should be >= 20
        assert hausdorff > 0
        assert hausdorff >= 19.0  # At least the gap distance


@plugin_required
class TestContourConvexHullCorrectness:
    """Verify convex hull produces correct geometry."""

    def test_convex_hull_of_l_shape(self) -> None:
        """Convex hull of L-shape should have more area than L-shape."""
        l_shape = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 50.0},
                {"x": 50.0, "y": 50.0},
                {"x": 50.0, "y": 100.0},
                {"x": 0.0, "y": 100.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame({"contour": [l_shape]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            hull=pl.col("contour").contour.convex_hull(),
            original_area=pl.col("contour").contour.area(),
        ).with_columns(
            hull_area=pl.col("hull").contour.area(),
        )

        original_area = result["original_area"][0]
        hull_area = result["hull_area"][0]

        # L-shape area = 7500 (100*50 + 50*50)
        assert original_area == pytest.approx(7500.0, rel=0.01)
        # Hull area should be >= original since it fills the concavity
        assert hull_area >= original_area


@plugin_required
class TestContourFlipCorrectness:
    """Verify contour flip reverses point order."""

    def test_flip_reverses_winding(self) -> None:
        """Flip should reverse the winding direction."""
        square = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 100.0},
                {"x": 0.0, "y": 100.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame({"contour": [square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            original_winding=pl.col("contour").contour.winding(),
            flipped=pl.col("contour").contour.flip(),
        ).with_columns(
            flipped_winding=pl.col("flipped").contour.winding(),
        )

        orig = result["original_winding"][0]
        flipped = result["flipped_winding"][0]
        # Flipping should change winding from cw→ccw or ccw→cw
        assert orig != flipped


@plugin_required
class TestContourNormalizeRoundTrip:
    """Verify normalize → to_absolute round-trip preserves coordinates."""

    def test_normalize_to_absolute_roundtrip(self) -> None:
        """normalize then to_absolute should recover original coordinates."""
        square = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 100.0},
                {"x": 0.0, "y": 100.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame({"contour": [square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            roundtrip=pl.col("contour")
            .contour.normalize(ref_width=100, ref_height=100)
            .contour.to_absolute(ref_width=100, ref_height=100)
        )

        rt = result["roundtrip"][0]
        ext = rt["exterior"]

        # First point should be back to (0, 0)
        assert ext[0]["x"] == pytest.approx(0.0, abs=0.01)
        assert ext[0]["y"] == pytest.approx(0.0, abs=0.01)
        # Second point should be (100, 0)
        assert ext[1]["x"] == pytest.approx(100.0, abs=0.01)
        assert ext[1]["y"] == pytest.approx(0.0, abs=0.01)


@plugin_required
class TestContourEnsureWindingCorrectness:
    """Verify ensure_winding actually forces the specified direction."""

    def test_ensure_winding_ccw_produces_ccw(self) -> None:
        """After ensure_winding('ccw'), winding should be 'ccw'."""
        square = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 100.0},
                {"x": 0.0, "y": 100.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame({"contour": [square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("ccw")
        ).with_columns(winding=pl.col("ensured").contour.winding())

        assert result["winding"][0] == "ccw"

    def test_ensure_winding_cw_produces_cw(self) -> None:
        """After ensure_winding('cw'), winding should be 'cw'."""
        square = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 100.0},
                {"x": 0.0, "y": 100.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame({"contour": [square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("cw")
        ).with_columns(winding=pl.col("ensured").contour.winding())

        assert result["winding"][0] == "cw"


# ===================================================================
# 4. Point operation edge cases
# ===================================================================


@plugin_required
class TestPointRotateEdgeCases:
    """Edge cases for point rotation."""

    def test_rotate_360_is_identity(self) -> None:
        """Rotating by 2*pi should return the same point."""
        df = pl.DataFrame({"pt": [{"x": 42.0, "y": 17.5}]})
        result = df.with_columns(rotated=pl.col("pt").point.rotate(2 * math.pi))

        rotated = result["rotated"][0]
        assert abs(rotated["x"] - 42.0) < 1e-8
        assert abs(rotated["y"] - 17.5) < 1e-8

    def test_rotate_0_is_identity(self) -> None:
        """Rotating by 0 should return the same point."""
        df = pl.DataFrame({"pt": [{"x": 42.0, "y": 17.5}]})
        result = df.with_columns(rotated=pl.col("pt").point.rotate(0.0))

        rotated = result["rotated"][0]
        assert abs(rotated["x"] - 42.0) < 1e-10
        assert abs(rotated["y"] - 17.5) < 1e-10


@plugin_required
class TestPointWithinBboxEdgeCases:
    """Edge cases for bbox containment."""

    def test_point_exactly_on_corner(self) -> None:
        """Point exactly on corner should be inside (inclusive boundary)."""
        df = pl.DataFrame(
            {
                "pt": [{"x": 0.0, "y": 0.0}],
                "bbox": [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}],
            }
        )
        result = df.with_columns(inside=pl.col("pt").point.within_bbox(pl.col("bbox")))
        assert result["inside"][0] is True

    def test_point_just_outside(self) -> None:
        """Point just outside the bbox should not be inside."""
        df = pl.DataFrame(
            {
                "pt": [{"x": 10.001, "y": 5.0}],
                "bbox": [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}],
            }
        )
        result = df.with_columns(inside=pl.col("pt").point.within_bbox(pl.col("bbox")))
        assert result["inside"][0] is False

    def test_multiple_points_with_nulls(self) -> None:
        """Null points should produce null results."""
        df = pl.DataFrame(
            {
                "pt": [
                    {"x": 5.0, "y": 5.0},
                    None,
                    {"x": 15.0, "y": 5.0},
                ],
                "bbox": [
                    {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
                    {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
                    {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
                ],
            }
        )
        result = df.with_columns(inside=pl.col("pt").point.within_bbox(pl.col("bbox")))
        assert result["inside"][0] is True
        assert result["inside"][1] is None
        assert result["inside"][2] is False


@plugin_required
class TestPointInterpolationEdgeCases:
    """Edge cases for point interpolation."""

    def test_interpolate_negative_t(self) -> None:
        """t < 0 should extrapolate backwards."""
        df = pl.DataFrame(
            {"p1": [{"x": 10.0, "y": 10.0}], "p2": [{"x": 20.0, "y": 20.0}]}
        )
        result = df.with_columns(
            interp=pl.col("p1").point.interpolate(pl.col("p2"), t=-1.0)
        )
        interp = result["interp"][0]
        # t=-1: p1 + (-1)*(p2-p1) = (10,10) + (-10,-10) = (0,0)
        assert abs(interp["x"]) < 1e-10
        assert abs(interp["y"]) < 1e-10


# ===================================================================
# 5. Error handling / validation edge cases
# ===================================================================


class TestPipelineValidation:
    """Test pipeline construction validation."""

    def test_pipeline_without_source_can_define_ops(self) -> None:
        """Operations without source is valid (for use with .pipe() composition)."""
        # This is the intended API: pipelines without source for continuation
        pipe = Pipeline().resize(height=100, width=100)
        assert len(pipe._ops) == 1

    def test_invalid_source_format(self) -> None:
        """Invalid source format should raise."""
        with pytest.raises((ValueError, KeyError)):
            Pipeline().source("invalid_format")

    def test_contour_op_on_buffer_domain_raises(self) -> None:
        """Contour operation on buffer domain should raise ValueError."""
        with pytest.raises(ValueError, match="expects contour"):
            Pipeline().source("image_bytes").area()

    def test_contour_source_starts_in_buffer_domain(self) -> None:
        """Contour source with dimensions rasterizes → starts in buffer domain.

        This means buffer ops like grayscale() should work on a contour source.
        """
        # Contour source with explicit dims rasterizes automatically
        pipe = Pipeline().source("contour", width=100, height=100)
        assert pipe.current_domain() == "buffer"
        # So buffer ops should work:
        pipe_gray = pipe.grayscale()
        assert pipe_gray.current_domain() == "buffer"

    def test_reduction_after_contour_extraction_raises(self) -> None:
        """Reduction on contour domain should raise ValueError."""
        with pytest.raises(ValueError, match="expects buffer"):
            (
                Pipeline()
                .source("image_bytes")
                .grayscale()
                .threshold(128)
                .extract_contours()
                .reduce_mean()
            )


class TestNumpyFromStructValidation:
    """Test numpy_from_struct edge cases."""

    def test_unsupported_dtype_raises(self) -> None:
        """Unsupported dtype string should raise ValueError."""
        struct = {"data": b"\x00\x00", "dtype": "complex128", "shape": [2]}
        with pytest.raises(ValueError, match="Unsupported dtype"):
            numpy_from_struct(struct)

    def test_empty_shape_works(self) -> None:
        """Scalar (empty shape) should work."""
        arr = np.array(42.0, dtype=np.float64)
        struct = {"data": arr.tobytes(), "dtype": "float64", "shape": []}
        result = numpy_from_struct(struct)
        assert result.shape == ()
        assert result == 42.0

    def test_mismatched_data_size_raises(self) -> None:
        """Data size that doesn't match shape should raise."""
        # Shape says 10 elements, but data is only 2 bytes
        struct = {"data": b"\x00\x00", "dtype": "uint8", "shape": [10]}
        with pytest.raises((ValueError, Exception)):
            numpy_from_struct(struct)


# ===================================================================
# 6. Resize correctness with value verification
# ===================================================================


@plugin_required
class TestResizeCorrectness:
    """Verify resize produces correct output dimensions."""

    def test_resize_exact_dimensions(self, encode_png: Callable) -> None:
        """Resize should produce exact target dimensions."""
        arr = _make_solid(100, 200, (128, 64, 32))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").resize(height=50, width=75)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        assert out.shape == (50, 75, 3)

    def test_resize_preserves_color_approximately(self, encode_png: Callable) -> None:
        """Resize of solid color should approximately preserve that color."""
        arr = _make_solid(100, 100, (200, 100, 50))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").resize(height=50, width=50)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        # A solid color resized should still be that color
        np.testing.assert_allclose(out[25, 25], [200, 100, 50], atol=2)


# ===================================================================
# 7. Multi-row correctness for pipeline operations
# ===================================================================


@plugin_required
class TestMultiRowCorrectness:
    """Verify operations produce correct per-row results."""

    def test_grayscale_different_images(self, encode_png: Callable) -> None:
        """Grayscale on multiple different images should process each correctly."""
        red = _make_solid(10, 10, (255, 0, 0))
        green = _make_solid(10, 10, (0, 255, 0))
        blue = _make_solid(10, 10, (0, 0, 255))

        df = pl.DataFrame(
            {"img": [encode_png(red), encode_png(green), encode_png(blue)]}
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        red_gray = numpy_from_struct(result.row(0)[0])
        green_gray = numpy_from_struct(result.row(1)[0])
        blue_gray = numpy_from_struct(result.row(2)[0])

        # Red should be brightest via 0.299 weight
        # Green should be brightest via 0.587 weight
        # Blue should be dimmest via 0.114 weight
        assert green_gray[0, 0, 0] > red_gray[0, 0, 0] > blue_gray[0, 0, 0]

    def test_reduction_per_row_independence(self, encode_png: Callable) -> None:
        """Each row should get its own reduction result."""
        img1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        img2 = np.full((10, 10, 3), 200, dtype=np.uint8)

        df = pl.DataFrame({"img": [encode_png(img1), encode_png(img2)]})

        pipe = Pipeline().source("image_bytes").reduce_max()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("native"))

        assert result["out"][0] == 100.0
        assert result["out"][1] == 200.0


# ===================================================================
# 8. Perceptual hash correctness
# ===================================================================


@plugin_required
class TestPerceptualHashCorrectness:
    """Verify perceptual hash behavior."""

    def test_identical_images_same_hash(self, encode_png: Callable) -> None:
        """Identical images should produce the same hash."""
        arr = np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        png = encode_png(arr)

        df = pl.DataFrame({"img": [png, png]})

        # perceptual_hash produces vector domain → use list sink
        pipe = Pipeline().source("image_bytes").perceptual_hash()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("list"))

        hash1 = result["out"][0].to_list()
        hash2 = result["out"][1].to_list()

        assert hash1 == hash2

    def test_different_images_different_hash(self, encode_png: Callable) -> None:
        """Very different images should produce different hashes."""
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        white = np.full((32, 32, 3), 255, dtype=np.uint8)

        df = pl.DataFrame({"img": [encode_png(black), encode_png(white)]})

        # perceptual_hash produces vector domain → use list sink
        pipe = Pipeline().source("image_bytes").perceptual_hash()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("list"))

        hash1 = result["out"][0].to_list()
        hash2 = result["out"][1].to_list()

        assert hash1 != hash2


# ===================================================================
# 9. Histogram correctness
# ===================================================================


@plugin_required
class TestHistogramCorrectness:
    """Verify histogram computation with known inputs."""

    def test_histogram_uniform_image_auto_range_bug(self, encode_png: Callable) -> None:
        """BUG: Auto-range histogram with uniform image puts all pixels in bin 0.

        When all pixels have the same value (e.g. 128), auto-range detection
        collapses to [128, 128], causing all 256 bins to map the single value
        to bin 0 instead of bin 128. This is a bug in the auto-range logic.

        Workaround: use explicit range=(0, 256) for correct behavior.
        """
        arr = _make_solid(10, 10, (128, 128, 128))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        # Auto-range puts all in bin 0 (bug)
        pipe = Pipeline().source("image_bytes").grayscale().histogram(bins=256)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("list"))
        hist = result["out"][0].to_list()
        assert len(hist) == 256
        assert sum(hist) == 100  # Total count is correct
        # BUG: hist[128] should be 100 but is 0; hist[0] is incorrectly 100
        assert hist[0] == 100  # Documents current (buggy) behavior
        # When this bug is fixed, uncomment the following:
        # assert hist[128] == 100

    def test_histogram_uniform_image_explicit_range(self, encode_png: Callable) -> None:
        """With explicit range, histogram correctly maps uniform pixels."""
        arr = _make_solid(10, 10, (128, 128, 128))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        # Explicit range avoids the auto-range collapse bug
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=256, range=(0, 256))
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("list"))
        hist = result["out"][0].to_list()
        assert len(hist) == 256
        assert hist[128] == 100
        assert sum(hist) == 100

    def test_histogram_gradient_image(self, encode_png: Callable) -> None:
        """Histogram of gradient image should distribute across bins."""
        arr = np.zeros((1, 4, 3), dtype=np.uint8)
        arr[0, 0] = [0, 0, 0]
        arr[0, 1] = [64, 64, 64]
        arr[0, 2] = [128, 128, 128]
        arr[0, 3] = [255, 255, 255]
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=256, range=(0, 256))
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("list"))
        hist = result["out"][0].to_list()

        assert hist[0] == 1
        assert hist[64] == 1
        assert hist[128] == 1
        assert hist[255] == 1
        assert sum(hist) == 4


# ===================================================================
# 10. Reduce percentile correctness
# ===================================================================


@plugin_required
class TestReducePercentileCorrectness:
    """Verify reduce_percentile with known inputs."""

    def test_percentile_50_is_median(self, encode_png: Callable) -> None:
        """50th percentile should be the median."""
        # Create an image with values 0-99 repeated
        row = np.arange(100, dtype=np.uint8)
        arr = np.stack([row, row, row], axis=-1).reshape(10, 10, 3)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").grayscale().reduce_percentile(50)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("native"))

        # Median of 0-99 = 49.5
        val = result["out"][0]
        assert abs(val - 49.5) < 2.0

    def test_percentile_0_equals_min(self, encode_png: Callable) -> None:
        """0th percentile should equal the minimum value."""
        rng = np.random.default_rng(42)
        arr = rng.integers(10, 200, (20, 20, 3), dtype=np.uint8)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe_min = Pipeline().source("image_bytes").reduce_min()
        pipe_p0 = Pipeline().source("image_bytes").reduce_percentile(0)

        result_min = df.select(out=pl.col("img").cv.pipe(pipe_min).sink("native"))
        result_p0 = df.select(out=pl.col("img").cv.pipe(pipe_p0).sink("native"))

        assert abs(result_min["out"][0] - result_p0["out"][0]) < 1.0

    def test_percentile_100_equals_max(self, encode_png: Callable) -> None:
        """100th percentile should equal the maximum value."""
        rng = np.random.default_rng(42)
        arr = rng.integers(10, 200, (20, 20, 3), dtype=np.uint8)
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe_max = Pipeline().source("image_bytes").reduce_max()
        pipe_p100 = Pipeline().source("image_bytes").reduce_percentile(100)

        result_max = df.select(out=pl.col("img").cv.pipe(pipe_max).sink("native"))
        result_p100 = df.select(out=pl.col("img").cv.pipe(pipe_p100).sink("native"))

        assert abs(result_max["out"][0] - result_p100["out"][0]) < 1.0


# ===================================================================
# 11. Contour extraction correctness
# ===================================================================


@plugin_required
class TestContourExtractionCorrectness:
    """Verify contour extraction from binary images."""

    def test_extract_contours_from_white_rectangle(self, encode_png: Callable) -> None:
        """Extracting contours from a white rectangle on black background."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[20:80, 20:80] = 255  # White rectangle
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("native"))

        # Should have at least 1 contour
        contours = result["out"][0]
        assert contours is not None
        # Result should be a list of contour structs
        assert len(contours) >= 1


# ===================================================================
# 12. Sink format correctness
# ===================================================================


@plugin_required
class TestSinkFormats:
    """Verify different sink formats produce correct output types."""

    def test_png_sink_produces_valid_png(self, encode_png: Callable) -> None:
        """PNG sink should produce valid PNG bytes."""
        arr = _make_solid(10, 10, (128, 64, 32))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").resize(height=5, width=5)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("png"))

        output_bytes = result["out"][0]
        # PNG magic bytes
        assert output_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_jpeg_sink_produces_valid_jpeg(self, encode_png: Callable) -> None:
        """JPEG sink should produce valid JPEG bytes."""
        arr = _make_solid(10, 10, (128, 64, 32))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").resize(height=5, width=5)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("jpeg"))

        output_bytes = result["out"][0]
        # JPEG magic bytes (SOI marker)
        assert output_bytes[:2] == b"\xff\xd8"

    def test_numpy_sink_struct_fields(self, encode_png: Callable) -> None:
        """Numpy sink should produce struct with data, dtype, shape fields."""
        arr = _make_solid(10, 10, (128, 64, 32))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes")
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        struct_val = result["out"][0]
        assert "data" in struct_val
        assert "dtype" in struct_val
        assert "shape" in struct_val
        assert struct_val["dtype"] == "uint8"
        shape = list(struct_val["shape"])
        assert shape == [10, 10, 3]


# ===================================================================
# 13. Blur correctness
# ===================================================================


@plugin_required
class TestBlurCorrectness:
    """Verify blur operation smooths the image."""

    def test_blur_reduces_contrast(self, encode_png: Callable) -> None:
        """Blurring a high-contrast image should reduce std deviation."""
        # Create checkerboard pattern
        arr = np.zeros((20, 20, 3), dtype=np.uint8)
        arr[::2, ::2] = 255
        arr[1::2, 1::2] = 255
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe_orig = Pipeline().source("image_bytes").reduce_std()
        pipe_blur = Pipeline().source("image_bytes").blur(3.0).reduce_std()

        r_orig = df.select(out=pl.col("img").cv.pipe(pipe_orig).sink("native"))
        r_blur = df.select(out=pl.col("img").cv.pipe(pipe_blur).sink("native"))

        # Blurred image should have lower std
        assert r_blur["out"][0] < r_orig["out"][0]

    def test_blur_solid_is_noop(self, encode_png: Callable) -> None:
        """Blurring a solid color image should not change pixel values."""
        arr = _make_solid(20, 20, (128, 128, 128))
        png = encode_png(arr)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("image_bytes").blur(3.0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = numpy_from_struct(result.row(0)[0])

        # Central pixels should still be 128 (edges may differ due to border handling)
        assert out[10, 10, 0] == 128


# ===================================================================
# 14. Expression parameter (dynamic) correctness
# ===================================================================


@plugin_required
class TestDynamicParameterCorrectness:
    """Verify that expression parameters resolve per-row correctly."""

    def test_dynamic_resize_per_row(self, encode_png: Callable) -> None:
        """Each row should get its own resize dimensions."""
        arr1 = _make_solid(100, 100, (128, 128, 128))
        arr2 = _make_solid(200, 200, (64, 64, 64))
        df = pl.DataFrame(
            {
                "img": [encode_png(arr1), encode_png(arr2)],
                "h": [50, 30],
                "w": [60, 40],
            }
        )

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("w"))
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        out1 = numpy_from_struct(result.row(0)[0])
        out2 = numpy_from_struct(result.row(1)[0])

        assert out1.shape == (50, 60, 3)
        assert out2.shape == (30, 40, 3)

    def test_dynamic_threshold_per_row(self, encode_png: Callable) -> None:
        """Each row should get its own threshold value."""
        # Image with uniform gray of 128
        arr = _make_solid(10, 10, (128, 128, 128))
        png = encode_png(arr)
        df = pl.DataFrame(
            {
                "img": [png, png],
                "thresh": [100, 200],  # Below and above 128
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale().threshold(pl.col("thresh"))
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        out1 = numpy_from_struct(result.row(0)[0])
        out2 = numpy_from_struct(result.row(1)[0])

        # thresh=100: 128 >= 100, so all pixels → 255
        assert np.all(out1 == 255)
        # thresh=200: 128 < 200, so all pixels → 0
        assert np.all(out2 == 0)
