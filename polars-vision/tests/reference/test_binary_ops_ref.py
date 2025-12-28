"""
Reference tests for binary array operations using NumPy.

These tests establish the expected behavior for element-wise operations
between arrays, serving as ground truth for polars-vision implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


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
