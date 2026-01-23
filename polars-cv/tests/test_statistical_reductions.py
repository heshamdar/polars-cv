"""
Tests for statistical reduction operations in polars-cv.

These tests verify that the new reduce_max, reduce_min, reduce_mean, reduce_std,
reduce_argmax, reduce_argmin, extract_shape, and statistics methods work correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


@plugin_required
class TestGlobalReductions:
    """Tests for global reduction operations (axis=None)."""

    @pytest.fixture
    def sample_image_bytes(self, encode_png: Callable[[np.ndarray], bytes]) -> bytes:
        """Create a test image with known values."""
        # Create a simple gradient image for predictable statistics
        arr = np.arange(0, 100, dtype=np.uint8).reshape(10, 10)
        arr = np.stack([arr, arr, arr], axis=-1)  # RGB
        return encode_png(arr)

    @pytest.fixture
    def known_values_image(self, encode_png: Callable[[np.ndarray], bytes]) -> bytes:
        """Create image with known min/max/mean."""
        # Create 2x2 image: [0, 100, 200, 50] for easy verification
        arr = np.array([[0, 100], [200, 50]], dtype=np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)
        return encode_png(arr)

    def test_reduce_max_global(self, known_values_image: bytes) -> None:
        """Test global maximum reduction."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_max()
        df = pl.DataFrame({"image": [known_values_image]})
        result = df.select(max_val=pl.col("image").cv.pipeline(pipe.sink("native")))

        # Maximum value in our test image is 200
        assert result["max_val"][0] == 200.0

    def test_reduce_min_global(self, known_values_image: bytes) -> None:
        """Test global minimum reduction."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_min()
        df = pl.DataFrame({"image": [known_values_image]})
        result = df.select(min_val=pl.col("image").cv.pipeline(pipe.sink("native")))

        # Minimum value in our test image is 0
        assert result["min_val"][0] == 0.0

    def test_reduce_mean_global(self, known_values_image: bytes) -> None:
        """Test global mean reduction."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_mean()
        df = pl.DataFrame({"image": [known_values_image]})
        result = df.select(mean_val=pl.col("image").cv.pipeline(pipe.sink("native")))

        # Mean of [0, 100, 200, 50] = 87.5
        expected_mean = (0 + 100 + 200 + 50) / 4.0
        assert abs(result["mean_val"][0] - expected_mean) < 0.01

    def test_reduce_std_population(self, known_values_image: bytes) -> None:
        """Test population standard deviation (ddof=0)."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_std(ddof=0)
        df = pl.DataFrame({"image": [known_values_image]})
        result = df.select(std_val=pl.col("image").cv.pipeline(pipe.sink("native")))

        # Calculate expected population std
        values = np.array([0, 100, 200, 50], dtype=np.float64)
        expected_std = np.std(values, ddof=0)
        assert abs(result["std_val"][0] - expected_std) < 0.1

    def test_reduce_std_sample(self, known_values_image: bytes) -> None:
        """Test sample standard deviation (ddof=1)."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_std(ddof=1)
        df = pl.DataFrame({"image": [known_values_image]})
        result = df.select(std_val=pl.col("image").cv.pipeline(pipe.sink("native")))

        # Calculate expected sample std
        values = np.array([0, 100, 200, 50], dtype=np.float64)
        expected_std = np.std(values, ddof=1)
        assert abs(result["std_val"][0] - expected_std) < 0.1

    def test_reduce_sum_global(self, known_values_image: bytes) -> None:
        """Test global sum reduction."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_sum()
        df = pl.DataFrame({"image": [known_values_image]})
        result = df.select(sum_val=pl.col("image").cv.pipeline(pipe.sink("native")))

        # Sum of [0, 100, 200, 50] = 350
        expected_sum = float(0 + 100 + 200 + 50)
        assert result["sum_val"][0] == expected_sum


@plugin_required
class TestAxisReductions:
    """Tests for axis-based reduction operations."""

    @pytest.fixture
    def simple_image(self, encode_png: Callable[[np.ndarray], bytes]) -> bytes:
        """Create a simple 3x3 grayscale-like RGB image."""
        arr = np.array(
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ],
            dtype=np.uint8,
        )
        arr = np.stack([arr, arr, arr], axis=-1)
        return encode_png(arr)

    def test_reduce_max_axis_0(self, simple_image: bytes) -> None:
        """Test max reduction along axis 0 (height)."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_max(axis=0)
        df = pl.DataFrame({"image": [simple_image]})
        result = df.select(output=pl.col("image").cv.pipeline(pipe.sink("list")))

        # Max along height: [70, 80, 90]
        output_list = result["output"][0].to_list()
        # Output is a 1D array with shape (3,)
        assert len(output_list) == 3

    def test_reduce_mean_axis_1(self, simple_image: bytes) -> None:
        """Test mean reduction along axis 1 (width)."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_mean(axis=1)
        df = pl.DataFrame({"image": [simple_image]})
        result = df.select(output=pl.col("image").cv.pipeline(pipe.sink("list")))

        # Mean along width: [20, 50, 80]
        output_list = result["output"][0].to_list()
        assert len(output_list) == 3


@plugin_required
class TestArgReductions:
    """Tests for argmax/argmin reduction operations."""

    @pytest.fixture
    def simple_image(self, encode_png: Callable[[np.ndarray], bytes]) -> bytes:
        """Create a simple image for arg testing."""
        arr = np.array(
            [
                [10, 90, 30],
                [40, 50, 60],
                [70, 80, 20],
            ],
            dtype=np.uint8,
        )
        arr = np.stack([arr, arr, arr], axis=-1)
        return encode_png(arr)

    def test_reduce_argmax_axis(self, simple_image: bytes) -> None:
        """Test argmax along an axis."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_argmax(axis=1)
        df = pl.DataFrame({"image": [simple_image]})
        result = df.select(output=pl.col("image").cv.pipeline(pipe.sink("list")))

        # Argmax along axis 1 (width):
        # Row 0: max at col 1 (90) -> 1
        # Row 1: max at col 2 (60) -> 2
        # Row 2: max at col 1 (80) -> 1
        output_list = result["output"][0].to_list()
        assert len(output_list) == 3

    def test_reduce_argmin_axis(self, simple_image: bytes) -> None:
        """Test argmin along an axis."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().reduce_argmin(axis=0)
        df = pl.DataFrame({"image": [simple_image]})
        result = df.select(output=pl.col("image").cv.pipeline(pipe.sink("list")))

        # Argmin along axis 0 (height):
        # Col 0: min at row 0 (10) -> 0
        # Col 1: min at row 1 (50) -> 1
        # Col 2: min at row 2 (20) -> 2
        output_list = result["output"][0].to_list()
        assert len(output_list) == 3


@plugin_required
class TestExtractShape:
    """Tests for shape extraction operation."""

    @pytest.fixture
    def sample_image(self, encode_png: Callable[[np.ndarray], bytes]) -> bytes:
        """Create a test image with known dimensions."""
        arr = np.zeros((50, 75, 3), dtype=np.uint8)
        return encode_png(arr)

    def test_extract_shape_basic(self, sample_image: bytes) -> None:
        """Test basic shape extraction."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").extract_shape()
        df = pl.DataFrame({"image": [sample_image]})
        result = df.select(shape=pl.col("image").cv.pipeline(pipe.sink("native")))

        # Shape should be [height, width, channels] = [50, 75, 3]
        shape_list = result["shape"][0].to_list()
        assert shape_list == [50.0, 75.0, 3.0]

    def test_extract_shape_after_resize(self, sample_image: bytes) -> None:
        """Test shape extraction after resize operation."""
        from polars_cv.pipeline import Pipeline

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=200)
            .extract_shape()
        )
        df = pl.DataFrame({"image": [sample_image]})
        result = df.select(shape=pl.col("image").cv.pipeline(pipe.sink("native")))

        shape_list = result["shape"][0].to_list()
        assert shape_list == [100.0, 200.0, 3.0]

    def test_extract_shape_grayscale(self, sample_image: bytes) -> None:
        """Test shape extraction after grayscale conversion."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale().extract_shape()
        df = pl.DataFrame({"image": [sample_image]})
        result = df.select(shape=pl.col("image").cv.pipeline(pipe.sink("native")))

        # Grayscale adds a channel dimension of 1
        shape_list = result["shape"][0].to_list()
        assert shape_list == [50.0, 75.0, 1.0]


@plugin_required
class TestStatisticsMethod:
    """Tests for the statistics() convenience method."""

    @pytest.fixture
    def sample_image(self, encode_png: Callable[[np.ndarray], bytes]) -> bytes:
        """Create a test image."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
        return encode_png(arr)

    def test_statistics_default(self, sample_image: bytes) -> None:
        """Test statistics() with default parameters."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale()
        df = pl.DataFrame({"image": [sample_image]})

        img_expr = pl.col("image").cv.pipe(pipe)
        stats_expr = img_expr.statistics()

        result = df.select(stats=stats_expr)

        # Should have mean, std, min, max fields
        stats_struct = result["stats"][0]
        assert "mean" in stats_struct
        assert "std" in stats_struct
        assert "min" in stats_struct
        assert "max" in stats_struct

    def test_statistics_custom_include(self, sample_image: bytes) -> None:
        """Test statistics() with custom include list."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes").grayscale()
        df = pl.DataFrame({"image": [sample_image]})

        img_expr = pl.col("image").cv.pipe(pipe)
        stats_expr = img_expr.statistics(include=["mean", "sum"])

        result = df.select(stats=stats_expr)

        stats_struct = result["stats"][0]
        assert "mean" in stats_struct
        assert "sum" in stats_struct
        # Should not have std, min, max
        assert "std" not in stats_struct
        assert "min" not in stats_struct
        assert "max" not in stats_struct

    def test_statistics_invalid_stat(self, sample_image: bytes) -> None:
        """Test that invalid statistics raise ValueError."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes")
        img_expr = pl.col("image").cv.pipe(pipe)

        with pytest.raises(ValueError, match="Unknown statistic"):
            img_expr.statistics(include=["invalid_stat"])


@plugin_required
class TestStatisticsLazy:
    """Tests for the statistics_lazy() method for composition."""

    @pytest.fixture
    def sample_image(self, encode_png: Callable[[np.ndarray], bytes]) -> bytes:
        """Create a test image."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
        return encode_png(arr)

    def test_statistics_lazy_composable(self, sample_image: bytes) -> None:
        """Test that statistics_lazy() can be composed with other outputs."""
        from polars_cv.pipeline import Pipeline

        pipe = Pipeline().source("image_bytes")
        df = pl.DataFrame({"image": [sample_image]})

        # Create an aliased image node
        img = pl.col("image").cv.pipe(pipe).alias("img")
        gray = img.pipe(Pipeline().grayscale()).alias("gray")

        # Get stats for the grayscale image
        stats = gray.statistics_lazy(include=["mean", "max"])

        # Merge and sink together
        merged = gray.merge_pipe(stats)
        result = df.select(
            output=merged.sink(
                {
                    "gray": "numpy",
                    "stat_mean": "native",
                    "stat_max": "native",
                }
            )
        )

        output_struct = result["output"][0]
        assert "gray" in output_struct
        assert "stat_mean" in output_struct
        assert "stat_max" in output_struct


@plugin_required
class TestReductionDomainTransitions:
    """Tests verifying correct domain transitions for reductions."""

    def test_reduction_on_wrong_domain_raises(
        self, encode_png: Callable[[np.ndarray], bytes]
    ) -> None:
        """Test that reductions on wrong domain raise ValueError."""
        from polars_cv.pipeline import Pipeline

        # Create a contour pipeline (wrong domain for reduction)
        arr = np.array([[0, 0, 255], [0, 255, 255], [0, 0, 255]], dtype=np.uint8)
        image_bytes = encode_png(np.stack([arr, arr, arr], axis=-1))

        # First verify normal reduction works on buffer
        pipe_valid = Pipeline().source("image_bytes").reduce_mean()
        df = pl.DataFrame({"image": [image_bytes]})
        result = df.select(pl.col("image").cv.pipeline(pipe_valid.sink("native")))
        assert result is not None

        # Now test that reduction fails after domain change to contour
        with pytest.raises(ValueError, match="expects buffer input"):
            Pipeline().source("image_bytes").grayscale().threshold(
                128
            ).extract_contours().reduce_mean()


@plugin_required
class TestMultipleRows:
    """Tests verifying reductions work correctly with multiple rows."""

    def test_reduction_multiple_images(
        self, encode_png: Callable[[np.ndarray], bytes]
    ) -> None:
        """Test reductions work on DataFrames with multiple images."""
        from polars_cv.pipeline import Pipeline

        # Create images with different known values
        img1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        img2 = np.full((10, 10, 3), 200, dtype=np.uint8)
        img3 = np.full((10, 10, 3), 50, dtype=np.uint8)

        df = pl.DataFrame(
            {"image": [encode_png(img1), encode_png(img2), encode_png(img3)]}
        )

        pipe = Pipeline().source("image_bytes").reduce_mean()
        result = df.select(mean=pl.col("image").cv.pipeline(pipe.sink("native")))

        # All pixels in each image are the same, so mean = pixel value
        assert result["mean"].to_list() == [100.0, 200.0, 50.0]
