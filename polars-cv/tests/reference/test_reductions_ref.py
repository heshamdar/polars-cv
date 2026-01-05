"""
Reference tests for statistical reduction operations using NumPy.

These tests establish the expected behavior for array statistics,
serving as ground truth for polars-cv implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


class TestReductionsReference:
    """Establish expected behavior for statistical reductions using NumPy."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Sample image with known statistical properties."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_max_global_reference(self, sample_image: np.ndarray) -> None:
        """
        Global maximum across entire array.

        Expected: Single scalar value.
        """
        result = np.max(sample_image)

        assert isinstance(result, (np.integer, int))
        assert result == 255  # With enough random pixels, should hit max

    def test_max_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """
        Maximum per channel (reduce over H, W axes).

        Expected: Array of shape (3,) for RGB.
        """
        result = np.max(sample_image, axis=(0, 1))

        assert result.shape == (3,)
        assert np.all(result == 255)  # Each channel should have max 255

    def test_max_per_row_reference(self, sample_image: np.ndarray) -> None:
        """
        Maximum per row (reduce over W axis).
        """
        result = np.max(sample_image, axis=1)

        assert result.shape == (100, 3)

    def test_min_global_reference(self, sample_image: np.ndarray) -> None:
        """
        Global minimum across entire array.
        """
        result = np.min(sample_image)

        assert isinstance(result, (np.integer, int))
        assert result == 0  # With enough random pixels, should hit min

    def test_min_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """
        Minimum per channel.
        """
        result = np.min(sample_image, axis=(0, 1))

        assert result.shape == (3,)
        assert np.all(result == 0)

    def test_mean_global_reference(self, sample_image: np.ndarray) -> None:
        """
        Global mean across entire array.

        For uniform random [0,256), expected mean ≈ 127.5.
        """
        result = np.mean(sample_image)

        assert isinstance(result, (np.floating, float))
        # Uniform [0,256) should be around 127.5
        assert 120 < result < 135

    def test_mean_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """
        Mean per channel.
        """
        result = np.mean(sample_image, axis=(0, 1))

        assert result.shape == (3,)
        # Each channel should be around 127.5
        assert np.all((result > 120) & (result < 135))

    def test_std_population_reference(self, sample_image: np.ndarray) -> None:
        """
        Population standard deviation (ddof=0).

        For uniform [0,256), std ≈ 74.
        """
        result = np.std(sample_image, ddof=0)

        assert isinstance(result, (np.floating, float))
        # Uniform [0,256) has std ≈ 256/sqrt(12) ≈ 74
        assert 70 < result < 78

    def test_std_sample_reference(self, sample_image: np.ndarray) -> None:
        """
        Sample standard deviation (ddof=1).

        Should be slightly higher than population std.
        """
        result_pop = np.std(sample_image, ddof=0)
        result_sample = np.std(sample_image, ddof=1)

        # Sample std is always slightly larger
        assert result_sample > result_pop

    def test_std_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """
        Standard deviation per channel.
        """
        result = np.std(sample_image, axis=(0, 1))

        assert result.shape == (3,)

    def test_sum_global_reference(self, sample_image: np.ndarray) -> None:
        """
        Global sum across entire array.
        """
        result = np.sum(sample_image, dtype=np.int64)

        assert isinstance(result, (np.integer, int))
        # 100 * 100 * 3 pixels * ~127.5 mean ≈ 3,825,000
        expected_approx = 100 * 100 * 3 * 127.5
        assert abs(result - expected_approx) < expected_approx * 0.1

    def test_sum_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """
        Sum per channel.
        """
        result = np.sum(sample_image, axis=(0, 1), dtype=np.int64)

        assert result.shape == (3,)

    def test_argmax_global_reference(self, sample_image: np.ndarray) -> None:
        """
        Index of global maximum (flattened).
        """
        result = np.argmax(sample_image)

        assert isinstance(result, (np.integer, int))
        # Verify the index points to a max value
        assert sample_image.flat[result] == np.max(sample_image)

    def test_argmax_axis_reference(self, sample_image: np.ndarray) -> None:
        """
        Index of maximum along axis.
        """
        result = np.argmax(sample_image, axis=0)

        assert result.shape == (100, 3)

    def test_argmin_global_reference(self, sample_image: np.ndarray) -> None:
        """
        Index of global minimum (flattened).
        """
        result = np.argmin(sample_image)

        assert isinstance(result, (np.integer, int))
        # Verify the index points to a min value
        assert sample_image.flat[result] == np.min(sample_image)

    def test_reduction_preserves_type_reference(self) -> None:
        """
        Verify dtype handling for different input types.

        Note: NumPy's behavior depends on version and architecture.
        The key property is that mean returns a floating point type.
        """
        # Float32 input
        float_arr = np.random.default_rng(42).random((10, 10)).astype(np.float32)
        result = np.mean(float_arr)
        # NumPy may keep float32 or promote to float64 depending on version
        assert result.dtype in (np.float32, np.float64)

        # Float64 input
        float64_arr = float_arr.astype(np.float64)
        result = np.mean(float64_arr)
        assert result.dtype == np.float64

    def test_empty_axis_keeps_shape_reference(self) -> None:
        """
        Reducing with keepdims=True preserves dimensions.
        """
        arr = np.random.default_rng(42).integers(0, 256, (10, 20, 3), dtype=np.uint8)

        result = np.mean(arr, axis=1, keepdims=True)

        assert result.shape == (10, 1, 3)
