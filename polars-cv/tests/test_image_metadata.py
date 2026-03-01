"""Tests for expression-level image metadata extraction (header-only)."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import polars_cv  # noqa: F401 — registers .cv namespace

from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


def _encode_jpeg(arr: np.ndarray, quality: int = 85) -> bytes:
    """Encode a numpy array as JPEG bytes."""
    from PIL import Image

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _encode_webp(arr: np.ndarray) -> bytes:
    """Encode a numpy array as WebP bytes."""
    from PIL import Image

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


def _make_view_blob(shape: list[int], dtype_code: int, data: bytes) -> bytes:
    """Build a minimal VIEW protocol blob for testing."""
    import struct

    magic = b"VIEW"
    version = 1
    rank = len(shape)
    # data_offset: header (64) + shape dims (rank * 8) + strides (rank * 8)
    data_offset = 64 + rank * 8 + rank * 8

    header = bytearray(64)
    header[0:4] = magic
    struct.pack_into("<H", header, 4, version)
    header[6] = dtype_code
    header[7] = rank
    struct.pack_into("<Q", header, 8, data_offset)

    shape_bytes = b""
    for dim in shape:
        shape_bytes += struct.pack("<Q", dim)

    element_size = {1: 1, 3: 2, 5: 4, 7: 4, 8: 8}[dtype_code]
    strides_bytes = b""
    stride = element_size
    strides_list = []
    for dim in reversed(shape):
        strides_list.insert(0, stride)
        stride *= dim
    for s in strides_list:
        strides_bytes += struct.pack("<q", s)

    return bytes(header) + shape_bytes + strides_bytes + data


@plugin_required
class TestImageWidth:
    """Tests for pl.col('img').cv.width()."""

    def test_png_width(self, create_test_png: Callable) -> None:
        """PNG images report correct width."""
        png = create_test_png(width=320, height=240)
        df = pl.DataFrame({"img": [png]})
        result = df.select(pl.col("img").cv.width())
        assert result["img"][0] == 320

    def test_jpeg_width(self) -> None:
        """JPEG images report correct width."""
        arr = np.zeros((50, 80, 3), dtype=np.uint8)
        jpeg = _encode_jpeg(arr)
        df = pl.DataFrame({"img": [jpeg]})
        result = df.select(pl.col("img").cv.width())
        assert result["img"][0] == 80

    def test_webp_width(self) -> None:
        """WebP images report correct width."""
        arr = np.zeros((30, 60, 3), dtype=np.uint8)
        webp = _encode_webp(arr)
        df = pl.DataFrame({"img": [webp]})
        result = df.select(pl.col("img").cv.width())
        assert result["img"][0] == 60

    def test_null_returns_null(self) -> None:
        """Null input produces null output."""
        df = pl.DataFrame({"img": [None]}, schema={"img": pl.Binary})
        result = df.select(pl.col("img").cv.width())
        assert result["img"][0] is None

    def test_corrupt_returns_null(self) -> None:
        """Unrecognised bytes produce null."""
        df = pl.DataFrame({"img": [b"not an image"]})
        result = df.select(pl.col("img").cv.width())
        assert result["img"][0] is None


@plugin_required
class TestImageHeight:
    """Tests for pl.col('img').cv.height()."""

    def test_png_height(self, create_test_png: Callable) -> None:
        """PNG images report correct height."""
        png = create_test_png(width=320, height=240)
        df = pl.DataFrame({"img": [png]})
        result = df.select(pl.col("img").cv.height())
        assert result["img"][0] == 240

    def test_jpeg_height(self) -> None:
        """JPEG images report correct height."""
        arr = np.zeros((50, 80, 3), dtype=np.uint8)
        jpeg = _encode_jpeg(arr)
        df = pl.DataFrame({"img": [jpeg]})
        result = df.select(pl.col("img").cv.height())
        assert result["img"][0] == 50


@plugin_required
class TestImageChannels:
    """Tests for pl.col('img').cv.channels()."""

    def test_rgb_channels(self, create_test_png: Callable) -> None:
        """RGB PNG has 3 channels."""
        png = create_test_png(width=10, height=10)
        df = pl.DataFrame({"img": [png]})
        result = df.select(pl.col("img").cv.channels())
        assert result["img"][0] == 3

    def test_grayscale_channels(self) -> None:
        """Grayscale PNG has 1 channel."""
        from PIL import Image

        img = Image.new("L", (10, 10), 128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()

        df = pl.DataFrame({"img": [png]})
        result = df.select(pl.col("img").cv.channels())
        assert result["img"][0] == 1

    def test_rgba_channels(self) -> None:
        """RGBA PNG has 4 channels."""
        from PIL import Image

        img = Image.new("RGBA", (10, 10), (128, 128, 128, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()

        df = pl.DataFrame({"img": [png]})
        result = df.select(pl.col("img").cv.channels())
        assert result["img"][0] == 4


@plugin_required
class TestImageDtype:
    """Tests for pl.col('img').cv.image_dtype()."""

    def test_uint8_png(self, create_test_png: Callable) -> None:
        """Standard 8-bit PNG reports uint8."""
        png = create_test_png(width=10, height=10)
        df = pl.DataFrame({"img": [png]})
        result = df.select(pl.col("img").cv.image_dtype())
        assert result["img"][0] == "uint8"

    def test_uint16_png(self) -> None:
        """16-bit PNG reports uint16."""
        from PIL import Image

        arr = np.zeros((10, 10), dtype=np.uint16)
        img = Image.fromarray(arr, mode="I;16")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()

        df = pl.DataFrame({"img": [png]})
        result = df.select(pl.col("img").cv.image_dtype())
        assert result["img"][0] == "uint16"


@plugin_required
class TestViewProtocol:
    """Tests for VIEW protocol blob metadata extraction."""

    def test_view_blob_3d(self) -> None:
        """3D VIEW blob reports correct dimensions and dtype."""
        data = bytes(224 * 224 * 3)
        blob = _make_view_blob([224, 224, 3], dtype_code=1, data=data)
        df = pl.DataFrame({"img": [blob]})

        w = df.select(pl.col("img").cv.width())["img"][0]
        h = df.select(pl.col("img").cv.height())["img"][0]
        c = df.select(pl.col("img").cv.channels())["img"][0]
        dt = df.select(pl.col("img").cv.image_dtype())["img"][0]

        assert w == 224
        assert h == 224
        assert c == 3
        assert dt == "uint8"

    def test_view_blob_2d_float(self) -> None:
        """2D float32 VIEW blob reports correct metadata."""
        data = bytes(100 * 200 * 4)
        blob = _make_view_blob([100, 200], dtype_code=7, data=data)
        df = pl.DataFrame({"img": [blob]})

        w = df.select(pl.col("img").cv.width())["img"][0]
        h = df.select(pl.col("img").cv.height())["img"][0]
        c = df.select(pl.col("img").cv.channels())["img"][0]
        dt = df.select(pl.col("img").cv.image_dtype())["img"][0]

        assert w == 200
        assert h == 100
        assert c == 1
        assert dt == "float32"


@plugin_required
class TestMixedFormats:
    """Tests for mixed image formats in the same column."""

    def test_mixed_png_jpeg(self, create_test_png: Callable) -> None:
        """Mixed PNG and JPEG images both return correct widths."""
        png = create_test_png(width=100, height=50)
        arr = np.zeros((30, 60, 3), dtype=np.uint8)
        jpeg = _encode_jpeg(arr)

        df = pl.DataFrame({"img": [png, jpeg]})
        result = df.select(pl.col("img").cv.width())
        assert result["img"].to_list() == [100, 60]

    def test_mixed_with_null(self, create_test_png: Callable) -> None:
        """Null rows produce null results alongside valid images."""
        png = create_test_png(width=100, height=50)
        df = pl.DataFrame({"img": [png, None, png]}, schema={"img": pl.Binary})
        result = df.select(pl.col("img").cv.height())
        vals = result["img"].to_list()
        assert vals[0] == 50
        assert vals[1] is None
        assert vals[2] == 50

    def test_filter_by_resolution(self, create_test_png: Callable) -> None:
        """Metadata expressions can be used in filter/group operations."""
        small = create_test_png(width=50, height=50)
        large = create_test_png(width=200, height=200)

        df = pl.DataFrame({"img": [small, large, small]})
        big = df.filter(pl.col("img").cv.width() > 100)
        assert big.height == 1
