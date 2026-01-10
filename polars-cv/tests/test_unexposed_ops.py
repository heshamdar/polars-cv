"""
Tests for previously unexposed operations: maximum, minimum, histogram.

These operations exist in view-buffer but lacked Python bindings.
This test file validates the Python API and behavior against NumPy reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    pass


# Check if plugin is available
def _plugin_available() -> bool:
    """Check if the compiled plugin is available."""
    from pathlib import Path

    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


# Mark tests with plugin_required marker
plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


# --- Fixtures ---


@pytest.fixture
def sample_images() -> tuple[np.ndarray, np.ndarray]:
    """
    Two sample images for binary operations testing.

    Returns:
        tuple: Two RGB images with shape (100, 100, 3) and dtype uint8.
    """
    rng = np.random.default_rng(123)
    img1 = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    img2 = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    return img1, img2


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """
    Grayscale test image for histogram operations.

    Returns:
        np.ndarray: Grayscale image with shape (100, 100) and dtype uint8.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (100, 100), dtype=np.uint8)


@pytest.fixture
def encode_png() -> Callable[[np.ndarray], bytes]:
    """
    Encode a numpy array as PNG bytes.

    Returns:
        A callable that encodes a numpy array as PNG bytes.
    """

    def _encode(arr: np.ndarray) -> bytes:
        """
        Encode numpy array as PNG bytes.

        Args:
            arr: NumPy array with shape (H, W, 3) or (H, W) and dtype uint8.

        Returns:
            PNG bytes.
        """
        from io import BytesIO

        from PIL import Image

        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return _encode


# --- Reference Tests (NumPy baseline) ---


class TestMaximumMinimumReference:
    """Establish expected behavior for maximum/minimum using NumPy as reference."""

    def test_maximum_reference(
        self, sample_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        NumPy reference: element-wise maximum.

        Expected behavior: Max of corresponding elements at every position.
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
        NumPy reference: element-wise minimum.

        Expected behavior: Min of corresponding elements at every position.
        """
        img1, img2 = sample_images

        result = np.minimum(img1, img2)

        assert result.shape == img1.shape
        assert result.dtype == img1.dtype
        # Result should be <= both inputs at every position
        assert np.all(result <= img1)
        assert np.all(result <= img2)

    def test_maximum_minimum_inverse(
        self, sample_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Maximum and minimum should partition the data.

        For any position: max(a, b) >= min(a, b)
        """
        img1, img2 = sample_images

        max_result = np.maximum(img1, img2)
        min_result = np.minimum(img1, img2)

        assert np.all(max_result >= min_result)

    def test_maximum_with_constant(self) -> None:
        """
        Maximum with constant array (like relu).
        """
        img = np.array([[10, 50, 100], [0, 255, 128]], dtype=np.uint8)
        zeros = np.zeros_like(img)

        result = np.maximum(img, zeros)

        # For uint8, this is identity since already >= 0
        np.testing.assert_array_equal(result, img)


class TestHistogramReference:
    """Establish expected behavior for histogram operations using NumPy."""

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

    def test_histogram_fewer_bins_reference(self, grayscale_image: np.ndarray) -> None:
        """
        Histogram with fewer bins aggregates values.
        """
        bins = 8
        counts, edges = np.histogram(
            grayscale_image.flatten(), bins=bins, range=(0, 256)
        )

        assert counts.shape == (bins,)
        assert counts.sum() == grayscale_image.size

        # Verify bin width
        bin_width = (256 - 0) / bins
        assert np.isclose(bin_width, 32.0)

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


# --- polars-cv Integration Tests ---


@plugin_required
class TestMaximumMinimumPolarsCV:
    """
    Tests that compare polars-cv maximum/minimum operations against NumPy reference.

    These tests verify that the Python API is correctly wired to the Rust backend.
    """

    def test_maximum_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv maximum should match NumPy np.maximum."""
        from polars_cv import Pipeline, numpy_from_struct

        img1, img2 = sample_images

        # NumPy reference
        expected = np.maximum(img1, img2)

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

        result = df.select(output=expr1.maximum(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_array_equal(actual, expected)

    def test_minimum_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv minimum should match NumPy np.minimum."""
        from polars_cv import Pipeline, numpy_from_struct

        img1, img2 = sample_images

        # NumPy reference
        expected = np.minimum(img1, img2)

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

        result = df.select(output=expr1.minimum(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_array_equal(actual, expected)

    def test_maximum_minimum_chain(
        self,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Test chaining maximum and minimum operations."""
        from polars_cv import Pipeline, numpy_from_struct

        # Create test images with known values
        img1 = np.array([[10, 200], [50, 100]], dtype=np.uint8)
        img2 = np.array([[100, 100], [100, 100]], dtype=np.uint8)
        img3 = np.array([[50, 150], [75, 75]], dtype=np.uint8)

        # Extend to 3-channel for PNG encoding
        img1_rgb = np.stack([img1] * 3, axis=-1)
        img2_rgb = np.stack([img2] * 3, axis=-1)
        img3_rgb = np.stack([img3] * 3, axis=-1)

        # NumPy reference: clamp img1 between img2 (min) and img3 (max-ish)
        # max(min(img1, img3), img2) = clamp img1 to [img2, img3]
        expected = np.maximum(np.minimum(img1_rgb, img3_rgb), img2_rgb)

        df = pl.DataFrame(
            {
                "img1": [encode_png(img1_rgb)],
                "img2": [encode_png(img2_rgb)],
                "img3": [encode_png(img3_rgb)],
            }
        )

        pipe = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe)
        expr2 = pl.col("img2").cv.pipe(pipe)
        expr3 = pl.col("img3").cv.pipe(pipe)

        # Chain: minimum(img1, img3) then maximum with img2
        result = df.select(output=expr1.minimum(expr3).maximum(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_array_equal(actual, expected)

    def test_maximum_with_self(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Maximum with itself should return the same image."""
        from polars_cv import Pipeline, numpy_from_struct

        img1, _ = sample_images

        df = pl.DataFrame({"img": [encode_png(img1)]})

        pipe = Pipeline().source("image_bytes")

        expr = pl.col("img").cv.pipe(pipe)

        result = df.select(output=expr.maximum(expr).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_array_equal(actual, img1)


@plugin_required
class TestHistogramPolarsCV:
    """
    Tests for polars-cv histogram operations.

    These tests validate that histogram() is properly exposed and matches NumPy.
    """

    def test_histogram_counts_matches_reference(
        self,
        grayscale_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv histogram counts should match NumPy histogram."""
        from polars_cv import Pipeline

        # NumPy reference
        expected_counts, _ = np.histogram(
            grayscale_image.flatten(), bins=256, range=(0, 256)
        )

        # polars-cv implementation
        # Encode as grayscale PNG
        gray_rgb = np.stack([grayscale_image] * 3, axis=-1)
        df = pl.DataFrame({"img": [encode_png(gray_rgb)]})

        # Histogram should transition to vector domain and return counts
        pipe = Pipeline().source("image_bytes").grayscale().histogram(bins=256)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("list"))

        # Extract the list result
        actual_counts = np.array(result.row(0)[0])

        np.testing.assert_array_equal(actual_counts, expected_counts)

    def test_histogram_normalized_matches_reference(
        self,
        grayscale_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv normalized histogram should sum to 1.0."""
        from polars_cv import Pipeline

        # NumPy reference
        expected_counts, _ = np.histogram(
            grayscale_image.flatten(), bins=256, range=(0, 256)
        )
        expected_normalized = expected_counts.astype(np.float64) / expected_counts.sum()

        # polars-cv implementation
        gray_rgb = np.stack([grayscale_image] * 3, axis=-1)
        df = pl.DataFrame({"img": [encode_png(gray_rgb)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=256, output="normalized")
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("list"))
        actual_normalized = np.array(result.row(0)[0])

        np.testing.assert_allclose(actual_normalized, expected_normalized, rtol=1e-10)
        assert np.isclose(actual_normalized.sum(), 1.0)

    def test_histogram_fewer_bins(
        self,
        grayscale_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Histogram with fewer bins should aggregate values correctly."""
        from polars_cv import Pipeline

        bins = 8

        # NumPy reference
        expected_counts, _ = np.histogram(
            grayscale_image.flatten(), bins=bins, range=(0, 256)
        )

        # polars-cv implementation
        gray_rgb = np.stack([grayscale_image] * 3, axis=-1)
        df = pl.DataFrame({"img": [encode_png(gray_rgb)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=bins, range=(0, 256))
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("list"))
        actual_counts = np.array(result.row(0)[0])

        assert len(actual_counts) == bins
        np.testing.assert_array_equal(actual_counts, expected_counts)

    def test_histogram_custom_range(
        self,
        grayscale_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """
        Histogram with custom range.

        Note: polars-cv clips values outside the range to the first/last bins,
        while NumPy excludes them. This test verifies the polars-cv behavior
        is internally consistent.
        """
        from polars_cv import Pipeline

        value_range = (50, 200)
        bins = 10

        # polars-cv implementation
        gray_rgb = np.stack([grayscale_image] * 3, axis=-1)
        df = pl.DataFrame({"img": [encode_png(gray_rgb)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=bins, range=value_range)
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("list"))
        actual_counts = np.array(result.row(0)[0])

        assert len(actual_counts) == bins
        # Total counts should equal total pixels (polars-cv clips, doesn't exclude)
        assert actual_counts.sum() == grayscale_image.size

    def test_histogram_edges_output(
        self,
        grayscale_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Histogram with edges output should return bin edges."""
        from polars_cv import Pipeline

        bins = 16

        # NumPy reference
        _, expected_edges = np.histogram(
            grayscale_image.flatten(), bins=bins, range=(0, 256)
        )

        # polars-cv implementation
        gray_rgb = np.stack([grayscale_image] * 3, axis=-1)
        df = pl.DataFrame({"img": [encode_png(gray_rgb)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=bins, range=(0, 256), output="edges")
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("list"))
        actual_edges = np.array(result.row(0)[0])

        assert len(actual_edges) == bins + 1
        np.testing.assert_allclose(actual_edges, expected_edges)

    def test_histogram_quantized_output(
        self,
        grayscale_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Histogram with quantized output should replace pixels with bin indices."""
        from polars_cv import Pipeline, numpy_from_struct

        bins = 8

        # NumPy reference: quantize pixels to bin indices
        bin_indices = (grayscale_image.astype(np.float32) / (256 / bins)).astype(
            np.uint32
        )
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        # polars-cv implementation
        gray_rgb = np.stack([grayscale_image] * 3, axis=-1)
        df = pl.DataFrame({"img": [encode_png(gray_rgb)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=bins, range=(0, 256), output="quantized")
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        # Grayscale output has shape (H, W, 1), squeeze to (H, W)
        actual = actual.squeeze()

        assert actual.shape == grayscale_image.shape
        assert actual.min() >= 0
        assert actual.max() < bins
        # Values should match reference quantization
        np.testing.assert_array_equal(actual, bin_indices)
