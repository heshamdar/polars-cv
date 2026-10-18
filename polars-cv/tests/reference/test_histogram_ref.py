"""
Reference tests for histogram operations against NumPy.

Each test computes the NumPy histogram / quantization and the same thing with
polars-cv's ``histogram`` on the same pixels (PNG-encoded u8 image), then
asserts the two agree.
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required


def _png(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _hist(
    arr: np.ndarray,
    output: str,
    *,
    bins: int | Sequence[float] = 256,
    range: tuple[float, float] | None = None,
    channel: int | None = None,
) -> np.ndarray:
    """Run ``histogram(output=...)`` over a PNG-encoded image; return a 1-D array."""
    pipe = Pipeline().source("image_bytes")
    if channel is not None:
        pipe = pipe.channel_select(channel)
    pipe = pipe.histogram(bins=bins, range=range, output=output)
    df = pl.DataFrame({"img": [_png(arr)]})
    out = df.select(o=pl.col("img").cv.pipe(pipe).sink("native"))
    return np.asarray(out["o"][0])


def _quantized(arr: np.ndarray, bins: int) -> np.ndarray:
    pipe = (
        Pipeline()
        .source("image_bytes")
        .histogram(bins=bins, range=(0, 256), output="quantized")
    )
    df = pl.DataFrame({"img": [_png(arr)]})
    out = df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy"))
    return numpy_from_struct(out.row(0)[0])


@plugin_required
@pytest.mark.parity(
    ops=["histogram"],
    against="NumPy `np.histogram` (counts, edges, density) and `np.digitize` (quantized)",
    tolerance="exact for counts and quantized indices; rtol 1e-12 for edges and normalized",
)
class TestHistogramReference:
    """polars-cv ``histogram`` must equal NumPy's histogram on the same pixels."""

    @pytest.fixture
    def grayscale_image(self) -> np.ndarray:
        """Grayscale image for histogram tests."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (100, 100), dtype=np.uint8)

    @pytest.fixture
    def uniform_image(self) -> np.ndarray:
        """Image with known uniform distribution."""
        # 256x256 = 65536 pixels, 256 of each value
        values = np.arange(256, dtype=np.uint8)
        return np.tile(values, 256).reshape(256, 256)

    def test_histogram_counts_reference(self, grayscale_image: np.ndarray) -> None:
        """Histogram counts: array of bin counts with shape (bins,)."""
        bins = 256
        counts, edges = np.histogram(
            grayscale_image.flatten(), bins=bins, range=(0, 256)
        )
        assert counts.shape == (bins,)
        assert counts.sum() == grayscale_image.size
        assert len(edges) == bins + 1

        actual = _hist(grayscale_image, "counts", bins=bins, range=(0, 256))
        np.testing.assert_array_equal(actual, counts)

    def test_histogram_normalized_reference(self, grayscale_image: np.ndarray) -> None:
        """Normalized histogram (sums to 1.0)."""
        bins = 256
        counts, _ = np.histogram(grayscale_image.flatten(), bins=bins, range=(0, 256))
        normalized = counts.astype(np.float64) / counts.sum()
        assert np.isclose(normalized.sum(), 1.0)

        actual = _hist(grayscale_image, "normalized", bins=bins, range=(0, 256))
        np.testing.assert_allclose(actual, normalized, rtol=1e-12)

    def test_histogram_uniform_distribution_reference(
        self, uniform_image: np.ndarray
    ) -> None:
        """Histogram of uniform distribution has equal bin counts."""
        bins = 256
        counts, _ = np.histogram(uniform_image.flatten(), bins=bins, range=(0, 256))
        assert np.all(counts == 256)  # 65536 / 256

        actual = _hist(uniform_image, "counts", bins=bins, range=(0, 256))
        np.testing.assert_array_equal(actual, counts)

    def test_histogram_custom_range_reference(
        self, grayscale_image: np.ndarray
    ) -> None:
        """Histogram with custom value range (out-of-range pixels are not counted)."""
        bins = 10
        value_range = (50, 200)
        counts, edges = np.histogram(
            grayscale_image.flatten(), bins=bins, range=value_range
        )
        assert counts.shape == (bins,)
        assert edges[0] == value_range[0]
        assert edges[-1] == value_range[1]

        actual = _hist(grayscale_image, "counts", bins=bins, range=value_range)
        np.testing.assert_array_equal(actual, counts)
        actual_edges = _hist(grayscale_image, "edges", bins=bins, range=value_range)
        np.testing.assert_allclose(actual_edges, edges, rtol=1e-12)

    def test_histogram_fewer_bins_reference(self, grayscale_image: np.ndarray) -> None:
        """Histogram with fewer bins aggregates values (256 values / 8 bins = 32)."""
        bins = 8
        counts, _ = np.histogram(grayscale_image.flatten(), bins=bins, range=(0, 256))
        assert counts.shape == (bins,)
        assert counts.sum() == grayscale_image.size
        assert np.isclose((256 - 0) / bins, 32.0)

        actual = _hist(grayscale_image, "counts", bins=bins, range=(0, 256))
        np.testing.assert_array_equal(actual, counts)

    def test_histogram_edges_reference(self, grayscale_image: np.ndarray) -> None:
        """Bin edges: evenly spaced, bins + 1 of them."""
        bins = 16
        _, edges = np.histogram(grayscale_image.flatten(), bins=bins, range=(0, 256))
        assert len(edges) == bins + 1
        assert np.allclose(np.diff(edges), 256 / bins)

        actual = _hist(grayscale_image, "edges", bins=bins, range=(0, 256))
        np.testing.assert_allclose(actual, edges, rtol=1e-12)

    def test_quantize_image_reference(self, grayscale_image: np.ndarray) -> None:
        """Quantize image by replacing pixels with bin indices."""
        bins = 8
        bin_indices = (
            np.digitize(grayscale_image, np.linspace(0, 256, bins + 1)[:-1]) - 1
        )
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        assert bin_indices.min() >= 0
        assert bin_indices.max() < bins

        actual = _quantized(grayscale_image, bins)
        np.testing.assert_array_equal(actual.squeeze(), bin_indices)

    def test_quantize_preserves_shape_reference(
        self, grayscale_image: np.ndarray
    ) -> None:
        """Quantization preserves image shape."""
        bins = 4
        edges = np.linspace(0, 256, bins + 1)
        quantized = np.digitize(grayscale_image, edges[:-1]) - 1
        quantized = np.clip(quantized, 0, bins - 1)
        assert quantized.shape == grayscale_image.shape

        actual = _quantized(grayscale_image, bins)
        # polars-cv keeps the image's trailing channel axis: [H, W, 1].
        assert actual.shape[:2] == grayscale_image.shape
        np.testing.assert_array_equal(actual.squeeze(), quantized)

    def test_histogram_3d_image_per_channel_reference(self) -> None:
        """Per-channel histograms for an RGB image (``channel_select`` then histogram)."""
        rng = np.random.default_rng(42)
        rgb_image = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)

        bins = 256
        histograms = np.array(
            [
                np.histogram(rgb_image[:, :, c].flatten(), bins=bins, range=(0, 256))[0]
                for c in range(3)
            ]
        )
        assert histograms.shape == (3, bins)
        assert np.all(histograms.sum(axis=1) == 100 * 100)

        actual = np.array(
            [
                _hist(rgb_image, "counts", bins=bins, range=(0, 256), channel=c)
                for c in range(3)
            ]
        )
        np.testing.assert_array_equal(actual, histograms)

    def test_histogram_auto_range_reference(self) -> None:
        """Histogram with automatic range detection from data (min..max)."""
        rng = np.random.default_rng(42)
        limited_image = rng.integers(50, 150, (100, 100), dtype=np.uint8)

        counts, edges = np.histogram(limited_image.flatten(), bins=10)
        assert edges[0] == limited_image.min()
        assert edges[-1] == limited_image.max()

        np.testing.assert_array_equal(_hist(limited_image, "counts", bins=10), counts)
        np.testing.assert_allclose(
            _hist(limited_image, "edges", bins=10), edges, rtol=1e-12
        )

    def test_histogram_density_reference(self) -> None:
        """
        Histogram with density=True (integrates to 1 over the range).

        polars-cv has no density output; ``normalized`` (count / total) divided by
        the bin width is NumPy's density.
        """
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (100, 100), dtype=np.uint8)

        density, edges = np.histogram(
            image.flatten(), bins=256, range=(0, 256), density=True
        )
        assert np.isclose(np.sum(density * np.diff(edges)), 1.0)

        normalized = _hist(image, "normalized", bins=256, range=(0, 256))
        np.testing.assert_allclose(normalized / np.diff(edges), density, rtol=1e-12)
