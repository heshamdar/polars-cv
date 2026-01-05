"""
Reference tests for histogram operations using NumPy.

These tests establish the expected behavior for histogram computation
and image quantization, serving as ground truth for polars-cv.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


class TestHistogramReference:
    """Establish expected behavior for histogram operations using NumPy."""

    @pytest.fixture
    def grayscale_image(self) -> np.ndarray:
        """Grayscale image for histogram tests."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (100, 100), dtype=np.uint8)

    @pytest.fixture
    def uniform_image(self) -> np.ndarray:
        """Image with known uniform distribution."""
        # Create image with exactly uniform distribution
        values = np.arange(256, dtype=np.uint8)
        # Repeat to fill 256x256 = 65536 pixels, 256 of each value
        return np.tile(values, 256).reshape(256, 256)

    def test_histogram_counts_reference(self, grayscale_image: np.ndarray) -> None:
        """
        Compute histogram counts.

        Expected: Array of bin counts with shape (bins,).
        """
        bins = 256
        counts, edges = np.histogram(
            grayscale_image.flatten(), bins=bins, range=(0, 256)
        )

        assert counts.shape == (bins,)
        assert counts.sum() == grayscale_image.size
        assert len(edges) == bins + 1

    def test_histogram_normalized_reference(self, grayscale_image: np.ndarray) -> None:
        """
        Compute normalized histogram (sums to 1.0).
        """
        bins = 256
        counts, _ = np.histogram(grayscale_image.flatten(), bins=bins, range=(0, 256))
        normalized = counts.astype(np.float64) / counts.sum()

        assert normalized.shape == (bins,)
        assert np.isclose(normalized.sum(), 1.0)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_histogram_uniform_distribution_reference(
        self, uniform_image: np.ndarray
    ) -> None:
        """
        Histogram of uniform distribution should have equal bin counts.
        """
        bins = 256
        counts, _ = np.histogram(uniform_image.flatten(), bins=bins, range=(0, 256))

        # Each bin should have exactly 256 pixels (65536 / 256)
        assert np.all(counts == 256)

    def test_histogram_custom_range_reference(
        self, grayscale_image: np.ndarray
    ) -> None:
        """
        Histogram with custom value range.
        """
        bins = 10
        value_range = (50, 200)
        counts, edges = np.histogram(
            grayscale_image.flatten(), bins=bins, range=value_range
        )

        assert counts.shape == (bins,)
        assert edges[0] == value_range[0]
        assert edges[-1] == value_range[1]

    def test_histogram_fewer_bins_reference(self, grayscale_image: np.ndarray) -> None:
        """
        Histogram with fewer bins aggregates values.
        """
        # 256 values into 8 bins = 32 values per bin
        bins = 8
        counts, edges = np.histogram(
            grayscale_image.flatten(), bins=bins, range=(0, 256)
        )

        assert counts.shape == (bins,)
        assert counts.sum() == grayscale_image.size

        # Verify bin width
        bin_width = (256 - 0) / bins
        assert np.isclose(bin_width, 32.0)

    def test_histogram_edges_reference(self, grayscale_image: np.ndarray) -> None:
        """
        Get histogram bin edges.
        """
        bins = 16
        _, edges = np.histogram(grayscale_image.flatten(), bins=bins, range=(0, 256))

        assert len(edges) == bins + 1
        assert edges[0] == 0
        assert edges[-1] == 256
        # Edges should be evenly spaced
        expected_width = 256 / bins
        actual_widths = np.diff(edges)
        assert np.allclose(actual_widths, expected_width)

    def test_quantize_image_reference(self, grayscale_image: np.ndarray) -> None:
        """
        Quantize image by replacing pixels with bin indices.
        """
        bins = 8

        # Quantize: map each pixel to its bin index
        bin_indices = (
            np.digitize(grayscale_image, np.linspace(0, 256, bins + 1)[:-1]) - 1
        )
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        assert bin_indices.shape == grayscale_image.shape
        assert bin_indices.min() >= 0
        assert bin_indices.max() < bins

    def test_quantize_preserves_shape_reference(
        self, grayscale_image: np.ndarray
    ) -> None:
        """
        Quantization preserves image shape.
        """
        bins = 4
        edges = np.linspace(0, 256, bins + 1)
        quantized = np.digitize(grayscale_image, edges[:-1]) - 1
        quantized = np.clip(quantized, 0, bins - 1)

        assert quantized.shape == grayscale_image.shape

    def test_histogram_3d_image_per_channel_reference(self) -> None:
        """
        Compute per-channel histograms for RGB image.
        """
        rng = np.random.default_rng(42)
        rgb_image = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)

        bins = 256
        histograms = []
        for c in range(3):
            counts, _ = np.histogram(
                rgb_image[:, :, c].flatten(), bins=bins, range=(0, 256)
            )
            histograms.append(counts)

        histograms = np.array(histograms)

        assert histograms.shape == (3, bins)
        assert np.all(histograms.sum(axis=1) == 100 * 100)

    def test_histogram_auto_range_reference(self) -> None:
        """
        Histogram with automatic range detection from data.
        """
        # Image with limited value range
        rng = np.random.default_rng(42)
        limited_image = rng.integers(50, 150, (100, 100), dtype=np.uint8)

        # Auto range
        counts, edges = np.histogram(limited_image.flatten(), bins=10)

        assert edges[0] == limited_image.min()
        assert edges[-1] == limited_image.max()

    def test_histogram_density_reference(self) -> None:
        """
        Histogram with density=True (integrates to 1 over the range).
        """
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (100, 100), dtype=np.uint8)

        counts, edges = np.histogram(
            image.flatten(), bins=256, range=(0, 256), density=True
        )

        # For density histogram, integral over range should be 1
        bin_widths = np.diff(edges)
        integral = np.sum(counts * bin_widths)
        assert np.isclose(integral, 1.0)
