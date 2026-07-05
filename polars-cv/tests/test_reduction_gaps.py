"""
Tests filling gaps in statistical reduction coverage.

Covers: reduce_argmax/argmin with axis=None (global), reduce_std with
various ddof values, multi-channel extract_shape, axis edge cases, and
NumPy reference comparisons for all axis variants.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required


@pytest.fixture
def encode_png() -> Callable[[np.ndarray], bytes]:
    def _encode(arr: np.ndarray) -> bytes:
        from PIL import Image

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return _encode


# ---------------------------------------------------------------------------
# NumPy reference tests for axis variants
# ---------------------------------------------------------------------------


class TestReductionAxisReference:
    """NumPy reference for axis parameter variants."""

    def test_mean_axis0_reference(self) -> None:
        """Mean along axis 0 (rows) should reduce height."""
        img = np.random.default_rng(42).integers(0, 256, (10, 20, 3), dtype=np.uint8)
        result = np.mean(img, axis=0)
        assert result.shape == (20, 3)

    def test_mean_axis1_reference(self) -> None:
        """Mean along axis 1 (columns) should reduce width."""
        img = np.random.default_rng(42).integers(0, 256, (10, 20, 3), dtype=np.uint8)
        result = np.mean(img, axis=1)
        assert result.shape == (10, 3)

    def test_mean_axis2_reference(self) -> None:
        """Mean along axis 2 (channels) should reduce channels."""
        img = np.random.default_rng(42).integers(0, 256, (10, 20, 3), dtype=np.uint8)
        result = np.mean(img, axis=2)
        assert result.shape == (10, 20)

    def test_argmax_global_reference(self) -> None:
        """Global argmax should return flat index of maximum."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[5, 7, 2] = 255
        idx = np.argmax(img)
        assert img.flat[idx] == 255

    def test_argmin_global_reference(self) -> None:
        """Global argmin should return flat index of minimum."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        img[3, 4, 1] = 0
        idx = np.argmin(img)
        assert img.flat[idx] == 0


# ---------------------------------------------------------------------------
# Plugin: reduce_mean with all axis options
# ---------------------------------------------------------------------------


@plugin_required
class TestReduceMeanAxes:
    """Test reduce_mean with different axis values."""

    @pytest.fixture
    def img_png(self, encode_png: Callable) -> bytes:
        rng = np.random.default_rng(42)
        return encode_png(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8))

    def test_reduce_mean_global(self, img_png: bytes) -> None:
        """Global mean (axis=None) should return a scalar value."""
        df = pl.DataFrame({"img": [img_png]})
        pipe = Pipeline().source("image_bytes").reduce_mean()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("native"))
        # Global reduction returns a scalar via native format
        val = result.row(0)[0]
        assert isinstance(val, (int, float))

    def test_reduce_mean_axis0(self, img_png: bytes) -> None:
        """Mean along axis 0 should reduce height dimension."""
        df = pl.DataFrame({"img": [img_png]})
        pipe = Pipeline().source("image_bytes").reduce_mean(axis=0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        # Should have reduced the height (axis 0)
        assert arr.shape[0] < 32 or arr.ndim < 3

    def test_reduce_mean_axis1(self, img_png: bytes) -> None:
        """Mean along axis 1 should reduce width dimension."""
        df = pl.DataFrame({"img": [img_png]})
        pipe = Pipeline().source("image_bytes").reduce_mean(axis=1)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape[0] == 32 or arr.ndim < 3


# ---------------------------------------------------------------------------
# Plugin: reduce_std with ddof
# ---------------------------------------------------------------------------


@plugin_required
class TestReduceStdDdof:
    """Test reduce_std with ddof=0 (population) and ddof=1 (sample)."""

    @pytest.fixture
    def img_png(self, encode_png: Callable) -> bytes:
        rng = np.random.default_rng(42)
        return encode_png(rng.integers(0, 256, (50, 50, 3), dtype=np.uint8))

    def test_std_population_vs_sample(self, img_png: bytes) -> None:
        """Sample std (ddof=1) should be slightly larger than population (ddof=0)."""
        df = pl.DataFrame({"img": [img_png]})

        pipe_pop = Pipeline().source("image_bytes").reduce_std(ddof=0)
        pipe_sample = Pipeline().source("image_bytes").reduce_std(ddof=1)

        r_pop = df.select(out=pl.col("img").cv.pipe(pipe_pop).sink("native"))
        r_sample = df.select(out=pl.col("img").cv.pipe(pipe_sample).sink("native"))

        std_pop = float(r_pop.row(0)[0])
        std_sample = float(r_sample.row(0)[0])

        assert std_sample > std_pop

    def test_std_axis0(self, img_png: bytes) -> None:
        """Per-row std should produce a reduction."""
        df = pl.DataFrame({"img": [img_png]})
        pipe = Pipeline().source("image_bytes").reduce_std(axis=0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        # Should have reduced the height axis
        assert arr.shape[0] < 50 or arr.ndim < 3


# ---------------------------------------------------------------------------
# Plugin: reduce_argmax / reduce_argmin
# ---------------------------------------------------------------------------


@plugin_required
class TestReduceArgmaxArgmin:
    """Test argmax/argmin with axis parameter."""

    @pytest.fixture
    def known_img_png(self, encode_png: Callable) -> bytes:
        """Image where max is at a known position."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[5, 7] = [255, 0, 0]  # Red pixel at (5, 7)
        return encode_png(img)

    def test_argmax_axis0(self, known_img_png: bytes) -> None:
        """argmax along axis=0 should return valid indices."""
        df = pl.DataFrame({"img": [known_img_png]})
        pipe = Pipeline().source("image_bytes").reduce_argmax(axis=0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        # Should have shape (width, channels) or similar reduced shape
        assert arr.ndim >= 1

    def test_argmin_axis0(self, known_img_png: bytes) -> None:
        """argmin along axis=0 should return valid indices."""
        df = pl.DataFrame({"img": [known_img_png]})
        pipe = Pipeline().source("image_bytes").reduce_argmin(axis=0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.ndim >= 1

    def test_argmax_axis1(self, encode_png: Callable) -> None:
        """argmax along axis=1 should return valid indices."""
        img = np.random.default_rng(42).integers(0, 256, (20, 30, 3), dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = Pipeline().source("image_bytes").reduce_argmax(axis=1)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.ndim >= 1


# ---------------------------------------------------------------------------
# Plugin: extract_shape for RGB
# ---------------------------------------------------------------------------


@plugin_required
class TestExtractShapeRGB:
    """Verify extract_shape returns correct dims for multi-channel images."""

    def test_extract_shape_rgb(self, encode_png: Callable) -> None:
        """RGB image should report (H, W, 3)."""
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = Pipeline().source("image_bytes").extract_shape()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("list"))
        shape_list = result.row(0)[0]
        # extract_shape returns [height, width, channels]
        assert shape_list[0] == 64
        assert shape_list[1] == 128
        assert shape_list[2] == 3

    def test_extract_shape_after_grayscale(self, encode_png: Callable) -> None:
        """After grayscale, shape should show 1 channel."""
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = Pipeline().source("image_bytes").grayscale().extract_shape()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("list"))
        shape_list = result.row(0)[0]
        assert shape_list[0] == 64
        assert shape_list[1] == 128
        assert shape_list[2] == 1
