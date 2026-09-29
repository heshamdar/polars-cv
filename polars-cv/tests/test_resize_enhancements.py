"""
Tests for resize enhancements: scale-factor resize and aspect-ratio preserving resize.

These tests validate:
- resize_scale(): Resize by scale factor (e.g., 0.5x, 2x)
- resize_to_height(): Resize to specific height, preserving aspect ratio
- resize_to_width(): Resize to specific width, preserving aspect ratio
- resize_max(): Resize so max dimension equals target, preserving aspect ratio
- resize_min(): Resize so min dimension equals target, preserving aspect ratio
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


# Check if plugin is available
# Mark tests with plugin_required marker
# --- Fixtures ---


@pytest.fixture
def rectangular_image() -> np.ndarray:
    """
    Rectangular test image (200x100 = 2:1 aspect ratio).

    Returns:
        np.ndarray: RGB image with shape (100, 200, 3) and dtype uint8.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def square_image() -> np.ndarray:
    """
    Square test image (100x100 = 1:1 aspect ratio).

    Returns:
        np.ndarray: RGB image with shape (100, 100, 3) and dtype uint8.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)


# Pixel-value parity for every resize in this file against Pillow (height
# pass first, alpha premultiplied, nearest ties excused exactly) lives in the
# parity suite: tests/parity/oracle/test_parity_single_ops.py, one property
# per method, over drawn sizes, dtypes, channel counts and filters. The
# "reference" classes that stood here resized with Pillow and asserted
# Pillow's own output size, never running polars-cv.


# --- polars-cv Integration Tests ---


@plugin_required
class TestResizeScalePolarsCV:
    """Tests for polars-cv resize_scale() method."""

    def test_resize_scale_uniform(
        self,
        rectangular_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Uniform scaling with scale parameter."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(rectangular_image)]})

        # Scale 0.5 on 200x100 -> 100x50
        pipe = Pipeline().source("image_bytes").resize_scale(scale=0.5)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        # Original: (100, 200, 3), scaled: (50, 100, 3)
        assert actual.shape == (50, 100, 3)

    def test_resize_scale_non_uniform(
        self,
        rectangular_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Non-uniform scaling with scale_x and scale_y."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(rectangular_image)]})

        # scale_x=0.5, scale_y=2.0 on 200x100 -> 100x200
        pipe = Pipeline().source("image_bytes").resize_scale(scale_x=0.5, scale_y=2.0)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        # Original: (100, 200, 3), scaled: (200, 100, 3)
        # Note: shape is (height, width, channels)
        assert actual.shape == (200, 100, 3)

    def test_resize_scale_upscale(
        self,
        square_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Upscaling with scale > 1."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(square_image)]})

        pipe = Pipeline().source("image_bytes").resize_scale(scale=2.0)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        # Original: (100, 100, 3), scaled: (200, 200, 3)
        assert actual.shape == (200, 200, 3)

    def test_resize_scale_with_expression(
        self,
        square_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Scale factor from column expression."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame(
            {
                "img": [encode_png(square_image)],
                "scale_factor": [0.5],
            }
        )

        pipe = (
            Pipeline().source("image_bytes").resize_scale(scale=pl.col("scale_factor"))
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (50, 50, 3)


@plugin_required
class TestAspectRatioResizePolarsCV:
    """Tests for polars-cv aspect-ratio preserving resize methods."""

    def test_resize_to_height(
        self,
        rectangular_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Resize to specific height, preserving aspect ratio."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(rectangular_image)]})

        # 200x100 (2:1) to height=50 -> 100x50
        pipe = Pipeline().source("image_bytes").resize_to_height(50)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape[0] == 50  # Height
        # Width should be proportional: 50 * 2 = 100
        assert actual.shape[1] == 100

    def test_resize_to_width(
        self,
        rectangular_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Resize to specific width, preserving aspect ratio."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(rectangular_image)]})

        # 200x100 (2:1) to width=100 -> 100x50
        pipe = Pipeline().source("image_bytes").resize_to_width(100)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape[1] == 100  # Width
        # Height should be proportional: 100 / 2 = 50
        assert actual.shape[0] == 50

    def test_resize_max(
        self,
        rectangular_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Resize so max dimension equals target."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(rectangular_image)]})

        # 200x100, max_size=50 -> 50x25 (max dim = 50)
        pipe = Pipeline().source("image_bytes").resize_max(50)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert max(actual.shape[0], actual.shape[1]) == 50
        assert actual.shape == (25, 50, 3)

    def test_resize_min(
        self,
        rectangular_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Resize so min dimension equals target."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(rectangular_image)]})

        # 200x100, min_size=50 -> 100x50 (min dim = 50)
        pipe = Pipeline().source("image_bytes").resize_min(50)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert min(actual.shape[0], actual.shape[1]) == 50
        assert actual.shape == (50, 100, 3)

    def test_resize_to_height_with_expression(
        self,
        rectangular_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Target height from column expression."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame(
            {
                "img": [encode_png(rectangular_image)],
                "target_h": [50],
            }
        )

        pipe = Pipeline().source("image_bytes").resize_to_height(pl.col("target_h"))

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape[0] == 50
        assert actual.shape[1] == 100

    def test_resize_max_square_image(
        self,
        square_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """resize_max on square image."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(square_image)]})

        # 100x100 with max_size=50 -> 50x50
        pipe = Pipeline().source("image_bytes").resize_max(50)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (50, 50, 3)
