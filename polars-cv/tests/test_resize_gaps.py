"""
Tests filling gaps in resize operation coverage.

Covers: all filter types (nearest, bilinear, lanczos3), resize_scale with
various scale factors, and edge cases.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct


def _plugin_available() -> bool:
    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


@pytest.fixture
def encode_png() -> Callable[[np.ndarray], bytes]:
    def _encode(arr: np.ndarray) -> bytes:
        from PIL import Image

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return _encode


@pytest.fixture
def test_image_png(encode_png: Callable) -> bytes:
    """A 64×64 RGB test image."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return encode_png(img)


# ---------------------------------------------------------------------------
# Filter types
# ---------------------------------------------------------------------------


@plugin_required
class TestResizeFilterTypes:
    """Test resize with each supported filter type."""

    @pytest.mark.parametrize("filter_name", ["nearest", "bilinear", "lanczos3"])
    def test_resize_with_filter(self, test_image_png: bytes, filter_name: str) -> None:
        """Each filter type should produce correct output dimensions."""
        df = pl.DataFrame({"img": [test_image_png]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=32, width=48, filter=filter_name)
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (32, 48, 3)

    def test_invalid_filter_raises(self) -> None:
        """Invalid filter name should raise ValueError at pipeline build time."""
        with pytest.raises(ValueError, match="Invalid filter"):
            Pipeline().source("image_bytes").resize(
                height=32, width=32, filter="cubic_magic"
            )


# ---------------------------------------------------------------------------
# resize_scale
# ---------------------------------------------------------------------------


@plugin_required
class TestResizeScale:
    """Test resize_scale with various scale factors."""

    def test_uniform_upscale(self, test_image_png: bytes) -> None:
        """scale=2.0 should double both dimensions."""
        df = pl.DataFrame({"img": [test_image_png]})
        pipe = Pipeline().source("image_bytes").resize_scale(scale=2.0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (128, 128, 3)

    def test_uniform_downscale(self, test_image_png: bytes) -> None:
        """scale=0.5 should halve both dimensions."""
        df = pl.DataFrame({"img": [test_image_png]})
        pipe = Pipeline().source("image_bytes").resize_scale(scale=0.5)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (32, 32, 3)

    def test_asymmetric_scale(self, test_image_png: bytes) -> None:
        """scale_x and scale_y can differ."""
        df = pl.DataFrame({"img": [test_image_png]})
        pipe = Pipeline().source("image_bytes").resize_scale(scale_x=0.5, scale_y=2.0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (128, 32, 3)

    def test_scale_1_is_identity(self, test_image_png: bytes) -> None:
        """scale=1.0 should produce identical dimensions."""
        df = pl.DataFrame({"img": [test_image_png]})
        pipe = Pipeline().source("image_bytes").resize_scale(scale=1.0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (64, 64, 3)

    def test_very_small_scale(self, test_image_png: bytes) -> None:
        """Very small scale factor should produce a small image (≥ 1px)."""
        df = pl.DataFrame({"img": [test_image_png]})
        pipe = Pipeline().source("image_bytes").resize_scale(scale=0.05)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape[0] >= 1
        assert arr.shape[1] >= 1


# ---------------------------------------------------------------------------
# resize_to_height / resize_to_width / resize_max / resize_min
# ---------------------------------------------------------------------------


@plugin_required
class TestResizeAspectPreserving:
    """Test aspect-ratio-preserving resize variants."""

    def test_resize_to_height(self, encode_png: Callable) -> None:
        """resize_to_height should set height, scale width proportionally."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = Pipeline().source("image_bytes").resize_to_height(height=50)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape[0] == 50
        assert arr.shape[1] == 100  # 200 * (50/100)

    def test_resize_to_width(self, encode_png: Callable) -> None:
        """resize_to_width should set width, scale height proportionally."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = Pipeline().source("image_bytes").resize_to_width(width=100)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape[1] == 100
        assert arr.shape[0] == 50  # 100 * (100/200)

    def test_resize_max(self, encode_png: Callable) -> None:
        """resize_max should constrain largest dimension."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = Pipeline().source("image_bytes").resize_max(100)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert max(arr.shape[0], arr.shape[1]) == 100
        assert arr.shape == (50, 100, 3)

    def test_resize_min(self, encode_png: Callable) -> None:
        """resize_min should constrain smallest dimension."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})
        pipe = Pipeline().source("image_bytes").resize_min(200)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert min(arr.shape[0], arr.shape[1]) == 200
        assert arr.shape == (200, 400, 3)


# ---------------------------------------------------------------------------
# Reference: nearest vs bilinear quality
# ---------------------------------------------------------------------------


@plugin_required
class TestResizeFilterQuality:
    """Verify that nearest and bilinear produce visibly different results."""

    def test_nearest_vs_bilinear_differ(self, encode_png: Callable) -> None:
        """Nearest and bilinear should produce different pixel values."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        png = encode_png(img)
        df = pl.DataFrame({"img": [png]})

        pipe_nearest = (
            Pipeline()
            .source("image_bytes")
            .resize(height=128, width=128, filter="nearest")
        )
        pipe_bilinear = (
            Pipeline()
            .source("image_bytes")
            .resize(height=128, width=128, filter="bilinear")
        )

        r1 = df.select(out=pl.col("img").cv.pipe(pipe_nearest).sink("numpy"))
        r2 = df.select(out=pl.col("img").cv.pipe(pipe_bilinear).sink("numpy"))

        arr1 = numpy_from_struct(r1.row(0)[0])
        arr2 = numpy_from_struct(r2.row(0)[0])

        # Both should be 128×128×3
        assert arr1.shape == arr2.shape == (128, 128, 3)
        # They should differ (bilinear interpolates, nearest doesn't)
        assert not np.array_equal(arr1, arr2)
