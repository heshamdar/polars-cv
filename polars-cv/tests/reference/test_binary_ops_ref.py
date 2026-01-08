"""
Reference tests for binary array operations using NumPy.

These tests establish the expected behavior for element-wise operations
between arrays, and verify that polars-cv implementations match
the NumPy reference behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct

if TYPE_CHECKING:
    pass


# Check if plugin is available
def _plugin_available() -> bool:
    """Check if the compiled plugin is available."""
    from pathlib import Path

    lib_path = Path(__file__).parent.parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


# Mark tests with plugin_required marker
plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


class TestBinaryOpsReference:
    """Establish expected behavior for binary operations using NumPy as reference."""

    def test_add_reference(self, sample_images: tuple[np.ndarray, np.ndarray]) -> None:
        """
        Element-wise addition with saturation for uint8.

        Expected behavior: Add values with clamping to [0, 255].
        """
        img1, img2 = sample_images

        # NumPy reference behavior - saturating addition
        result = np.clip(img1.astype(np.int16) + img2.astype(np.int16), 0, 255).astype(
            np.uint8
        )

        assert result.shape == img1.shape
        assert result.dtype == np.uint8

        # Verify no overflow occurred
        assert result.max() <= 255
        assert result.min() >= 0

    def test_subtract_reference(
        self, sample_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Element-wise subtraction with saturation for uint8.

        Expected behavior: Subtract values with clamping to [0, 255].
        """
        img1, img2 = sample_images

        # NumPy reference behavior - saturating subtraction
        result = np.clip(img1.astype(np.int16) - img2.astype(np.int16), 0, 255).astype(
            np.uint8
        )

        assert result.shape == img1.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0

    def test_multiply_reference(
        self, sample_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Element-wise multiplication (scaled for images).

        For images, normalize to [0,1] range, multiply, scale back.
        """
        img1, img2 = sample_images

        # For images, typically normalize then multiply
        result = (
            (img1.astype(np.float32) / 255) * (img2.astype(np.float32) / 255) * 255
        ).astype(np.uint8)

        assert result.shape == img1.shape
        assert result.dtype == np.uint8

    def test_divide_reference(
        self, sample_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Element-wise division with zero handling.

        Expected behavior: Divide with inf/nan handling.
        """
        img1, img2 = sample_images

        # Use float for division to handle zeros properly
        img2_safe = img2.astype(np.float32)
        img2_safe[img2_safe == 0] = 1  # Avoid division by zero

        result = np.clip(img1.astype(np.float32) / img2_safe * 255, 0, 255).astype(
            np.uint8
        )

        assert result.shape == img1.shape
        assert result.dtype == np.uint8

    def test_apply_mask_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        binary_mask: np.ndarray,
    ) -> None:
        """
        Apply binary mask to image.

        Expected behavior: Multiply image by 0/1 mask, broadcasting channels.
        """
        img, _ = sample_images

        # Broadcast mask to image channels
        mask_3d = np.expand_dims(binary_mask, axis=-1)
        result = img * mask_3d

        # Verify masked regions
        assert np.all(result[0, 0] == 0)  # Outside mask (corner)
        assert np.all(result[50, 50] == img[50, 50])  # Inside mask (center)

        # Verify shape preserved
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_apply_mask_inverted_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        binary_mask: np.ndarray,
    ) -> None:
        """
        Apply inverted binary mask to image.

        Expected behavior: Keep exterior, zero interior.
        """
        img, _ = sample_images

        # Invert mask
        inverted_mask = 1 - binary_mask
        mask_3d = np.expand_dims(inverted_mask, axis=-1)
        result = img * mask_3d

        # Verify inverted masked regions
        assert np.all(result[50, 50] == 0)  # Inside original mask = zeroed
        assert np.all(result[0, 0] == img[0, 0])  # Outside mask = preserved

    def test_maximum_reference(
        self, sample_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Element-wise maximum.

        Expected behavior: Max of corresponding elements.
        """
        img1, img2 = sample_images

        result = np.maximum(img1, img2)

        assert result.shape == img1.shape
        assert result.dtype == img1.dtype
        # Result should be >= both inputs at every position
        assert np.all(result >= img1)
        assert np.all(result >= img2)

    def test_minimum_reference(
        self, sample_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Element-wise minimum.

        Expected behavior: Min of corresponding elements.
        """
        img1, img2 = sample_images

        result = np.minimum(img1, img2)

        assert result.shape == img1.shape
        assert result.dtype == img1.dtype
        # Result should be <= both inputs at every position
        assert np.all(result <= img1)
        assert np.all(result <= img2)

    def test_bitwise_and_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """
        Bitwise AND operation.

        Useful for combining binary masks.
        """
        img1, img2 = sample_images

        result = np.bitwise_and(img1, img2)

        assert result.shape == img1.shape
        assert result.dtype == img1.dtype

    def test_bitwise_or_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """
        Bitwise OR operation.

        Useful for combining binary masks.
        """
        img1, img2 = sample_images

        result = np.bitwise_or(img1, img2)

        assert result.shape == img1.shape
        assert result.dtype == img1.dtype

    def test_broadcasting_scalar_reference(self) -> None:
        """
        Verify scalar broadcasting matches NumPy behavior.
        """
        img = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        scalar = np.array([1.5])

        result = np.clip(img.astype(np.float32) * scalar, 0, 255).astype(np.uint8)

        assert result.shape == img.shape

    def test_broadcasting_per_channel_reference(self) -> None:
        """
        Verify per-channel broadcasting matches NumPy behavior.

        Common use case: RGB to grayscale weights.
        """
        img = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        channel_weights = np.array([0.299, 0.587, 0.114])  # Grayscale weights

        result = (img.astype(np.float32) * channel_weights).astype(np.float32)

        assert result.shape == img.shape

    def test_shape_mismatch_raises(self) -> None:
        """
        Verify incompatible shapes raise an error.
        """
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((50, 50, 3), dtype=np.uint8)

        # NumPy raises ValueError for shape mismatch that can't broadcast
        with pytest.raises(ValueError):
            _ = img1 + img2


@plugin_required
class TestBinaryOpsPolarsCV:
    """
    Tests that compare polars-cv binary operations against NumPy reference.

    These tests will FAIL until binary operations are implemented in the Rust backend.
    This is the expected behavior - we're testing that polars-cv matches NumPy.
    """

    @pytest.fixture
    def encode_png(self) -> Callable[[np.ndarray], bytes]:
        """Encode a numpy array as PNG bytes."""

        def _encode(arr: np.ndarray) -> bytes:
            from io import BytesIO

            from PIL import Image

            img = Image.fromarray(arr)
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        return _encode

    def test_add_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv add should match NumPy reference (saturating addition)."""
        img1, img2 = sample_images

        # NumPy reference behavior - saturating addition
        expected = np.clip(
            img1.astype(np.int16) + img2.astype(np.int16), 0, 255
        ).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        # This will FAIL until add is implemented in execute.rs
        result = df.select(output=expr1.add(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    def test_subtract_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv subtract should match NumPy reference (saturating subtraction)."""
        img1, img2 = sample_images

        # NumPy reference behavior - saturating subtraction
        expected = np.clip(
            img1.astype(np.int16) - img2.astype(np.int16), 0, 255
        ).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        # This will FAIL until subtract is implemented in execute.rs
        result = df.select(output=expr1.subtract(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    def test_multiply_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv multiply should match saturating multiplication semantics."""
        img1, img2 = sample_images

        # NumPy reference: saturating multiplication (clamp to 255)
        expected = np.clip(
            img1.astype(np.uint16) * img2.astype(np.uint16), 0, 255
        ).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(output=expr1.multiply(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    def test_blend_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv blend should match normalized multiplication semantics."""
        img1, img2 = sample_images

        # NumPy reference: normalize to [0,1], multiply, scale back
        # Using rounding division to match Rust: (a * b + 127) / 255
        expected = (
            (img1.astype(np.uint32) * img2.astype(np.uint32) + 127) // 255
        ).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(output=expr1.blend(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    def test_divide_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv divide should match integer division semantics."""
        img1, img2 = sample_images

        # NumPy reference: integer division with zero protection (returns 0)
        expected = np.zeros_like(img1)
        nonzero_mask = img2 != 0
        expected[nonzero_mask] = img1[nonzero_mask] // img2[nonzero_mask]

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(output=expr1.divide(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    def test_ratio_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv ratio should match scaled division semantics."""
        img1, img2 = sample_images

        # NumPy reference: (a/b) * 255, clamped to [0, 255]
        # With zero protection: returns 0 if a==0 and b==0, else 255 if b==0
        expected = np.zeros_like(img1, dtype=np.uint8)
        zero_mask = img2 == 0
        nonzero_mask = ~zero_mask

        # Where denominator is non-zero: compute scaled ratio
        expected[nonzero_mask] = np.clip(
            (img1[nonzero_mask].astype(np.uint32) * 255)
            // img2[nonzero_mask].astype(np.uint32),
            0,
            255,
        ).astype(np.uint8)

        # Where denominator is zero: 0 if numerator is 0, else 255
        expected[zero_mask & (img1 == 0)] = 0
        expected[zero_mask & (img1 != 0)] = 255

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(output=expr1.ratio(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    def test_apply_mask_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        binary_mask: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv apply_mask should match NumPy reference."""
        img, _ = sample_images

        # For polars-cv, the mask needs to be a grayscale image
        # Convert binary mask (0/1) to 0/255 grayscale
        mask_image = (binary_mask * 255).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "image": [encode_png(img)],
                "mask": [encode_png(np.stack([mask_image] * 3, axis=-1))],
            }
        )

        img_pipe = Pipeline().source("image_bytes")
        mask_pipe = Pipeline().source("image_bytes").grayscale()

        img_expr = pl.col("image").cv.pipe(img_pipe)
        mask_expr = pl.col("mask").cv.pipe(mask_pipe)

        # This will FAIL until apply_mask is implemented in execute.rs
        result = df.select(output=img_expr.apply_mask(mask_expr).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        # Scale expected to match normalized mask behavior (0/1 -> 0/255)
        expected_scaled = (img.astype(np.float32) * (binary_mask[:, :, None])).astype(
            np.uint8
        )

        np.testing.assert_allclose(actual, expected_scaled, atol=1)
