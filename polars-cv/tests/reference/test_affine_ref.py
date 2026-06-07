"""
Reference tests for affine transform operations.

Compares polars-cv warp_affine output against OpenCV cv2.warpAffine
ground truth to ensure correctness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


def _run_affine_pipeline(
    image_arr: np.ndarray,
    pipe: Pipeline,
    encode_fn: Callable[[np.ndarray], bytes],
) -> np.ndarray:
    """
    Run a polars-cv affine pipeline and return the result as ndarray.

    Args:
        image_arr: Input image (H, W, 3) or (H, W) uint8.
        pipe: Pipeline to execute.
        encode_fn: Function to encode ndarray to PNG bytes.

    Returns:
        Output image as ndarray.
    """
    from polars_cv import numpy_from_struct

    png_bytes = encode_fn(image_arr)
    df = pl.DataFrame({"image": [png_bytes]})
    result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
    return numpy_from_struct(result["output"][0])


@plugin_required
class TestAffineIdentity:
    """Tests that identity affine transform preserves the image."""

    def test_identity_rgb(
        self,
        test_image_rgb: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Identity matrix should produce the same image."""
        h, w = test_image_rgb.shape[:2]
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(h, w),
            )
        )
        actual = _run_affine_pipeline(test_image_rgb, pipe, encode_png)
        np.testing.assert_array_equal(actual, test_image_rgb)


@plugin_required
class TestAffineTranslation:
    """Tests for pure translation via affine transform."""

    def test_translation_vs_opencv(
        self,
        test_image_rgb: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Translation matches OpenCV warpAffine."""
        cv2 = pytest.importorskip("cv2")

        h, w = test_image_rgb.shape[:2]
        tx, ty = 50.0, 30.0

        # OpenCV reference: M is [a, b, tx; c, d, ty]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        expected = cv2.warpAffine(
            test_image_rgb,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )

        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, tx, 0.0, 1.0, ty],
                output_size=(h, w),
            )
        )
        actual = _run_affine_pipeline(test_image_rgb, pipe, encode_png)

        # Allow small numerical differences from interpolation
        np.testing.assert_allclose(actual, expected, atol=2)


@plugin_required
class TestAffineRotation:
    """Tests for rotation via affine transform."""

    def test_90_degree_rotation_vs_opencv(
        self,
        test_image_rgb: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """90-degree rotation via affine matches OpenCV."""
        cv2 = pytest.importorskip("cv2")

        h, w = test_image_rgb.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, -90, 1.0)
        expected = cv2.warpAffine(
            test_image_rgb,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )

        # Build the same matrix for polars-cv
        # OpenCV convention: M[0] = [cos, sin, tx], M[1] = [-sin, cos, ty]
        # polars-cv convention: [a, b, tx, c, d, ty]
        matrix = [M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2]]
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=matrix,
                output_size=(h, w),
            )
        )
        actual = _run_affine_pipeline(test_image_rgb, pipe, encode_png)

        np.testing.assert_allclose(actual, expected, atol=2)


@plugin_required
class TestAffineScale:
    """Tests for scaling via affine transform."""

    def test_scale_2x_vs_opencv(
        self,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """2x scale via affine matches OpenCV."""
        cv2 = pytest.importorskip("cv2")

        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        M = np.float32([[2, 0, 0], [0, 2, 0]])
        expected = cv2.warpAffine(
            img,
            M,
            (w * 2, h * 2),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )

        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                output_size=(h * 2, w * 2),
            )
        )
        actual = _run_affine_pipeline(img, pipe, encode_png)

        np.testing.assert_allclose(actual, expected, atol=2)


@plugin_required
class TestAffineShear:
    """Tests for shear via affine transform."""

    def test_shear_vs_opencv(
        self,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Shear matches OpenCV warpAffine with same matrix."""
        cv2 = pytest.importorskip("cv2")

        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        sx = 0.3
        M = np.float32([[1, sx, 0], [0, 1, 0]])
        expected = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )

        pipe = Pipeline().source("image_bytes").shear(sx=sx, sy=0.0, output_size=(h, w))
        actual = _run_affine_pipeline(img, pipe, encode_png)

        # Minor interpolation differences are expected between implementations
        np.testing.assert_allclose(actual, expected, atol=5)


@plugin_required
class TestAffineInterpolation:
    """Tests for interpolation modes."""

    def test_nearest_vs_bilinear(
        self,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Nearest and bilinear interpolation produce different results for rotation."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        import math

        angle = 30.0
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cx, cy = w / 2.0, h / 2.0
        tx = (1 - cos_a) * cx + sin_a * cy
        ty = -sin_a * cx + (1 - cos_a) * cy
        matrix = [cos_a, sin_a, tx, -sin_a, cos_a, ty]

        pipe_bilinear = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(matrix=matrix, output_size=(h, w), interpolation="bilinear")
        )
        pipe_nearest = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(matrix=matrix, output_size=(h, w), interpolation="nearest")
        )

        result_bilinear = _run_affine_pipeline(img, pipe_bilinear, encode_png)
        result_nearest = _run_affine_pipeline(img, pipe_nearest, encode_png)

        # They should differ (bilinear produces smoother results)
        assert not np.array_equal(result_bilinear, result_nearest)


@plugin_required
class TestAffineBorderValue:
    """Tests for border value handling."""

    def test_custom_border_value(
        self,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Out-of-bounds pixels use the specified border_value."""
        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (50, 50, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        # Large translation to push most of the image out of frame
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 200.0, 0.0, 1.0, 200.0],
                output_size=(h, w),
                border_value=128.0,
            )
        )
        actual = _run_affine_pipeline(img, pipe, encode_png)

        # Most pixels should be the border value since the image is shifted far
        border_pixels = actual[0, 0]
        assert all(v == 128 for v in border_pixels.flat)


@plugin_required
class TestAffineMultiChannel:
    """Tests for multi-channel handling."""

    def test_rgb_channels_independent(
        self,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Each channel is interpolated independently during affine transform."""
        # Create image where channels differ significantly
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Red channel full
        img[:, :, 1] = 0  # Green channel zero
        img[:, :, 2] = 128  # Blue channel half

        h, w = img.shape[:2]
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(h, w),
            )
        )
        actual = _run_affine_pipeline(img, pipe, encode_png)

        # Identity transform should preserve channel values
        np.testing.assert_array_equal(actual, img)


@plugin_required
class TestRotateAndScaleConvenience:
    """Tests for the rotate_and_scale convenience method."""

    def test_rotate_and_scale_vs_opencv(
        self,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """rotate_and_scale matches OpenCV getRotationMatrix2D + warpAffine."""
        cv2 = pytest.importorskip("cv2")

        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        angle_deg = 30.0
        scale = 1.5
        center = (w / 2.0, h / 2.0)

        # OpenCV uses negative angle for CW rotation
        M = cv2.getRotationMatrix2D(center, -angle_deg, scale)
        expected = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )

        pipe = (
            Pipeline()
            .source("image_bytes")
            .rotate_and_scale(
                angle=angle_deg,
                scale=scale,
                center=center,
                output_size=(h, w),
            )
        )
        actual = _run_affine_pipeline(img, pipe, encode_png)

        # Minor interpolation differences between implementations
        np.testing.assert_allclose(actual, expected, atol=7)
