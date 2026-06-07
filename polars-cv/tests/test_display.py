"""Tests for the show_images display utility."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

from polars_cv.display import (
    _detect_mime,
    _is_notebook,
    _ndarray_to_png,
    _to_png_bytes,
    show_images,
)

if TYPE_CHECKING:
    pass


class TestDetectMime:
    """Tests for format detection from magic bytes."""

    def test_png(self) -> None:
        """Detect PNG from magic bytes."""
        data = b"\x89PNG\x0d\x0a\x1a\x0a" + b"\x00" * 50
        assert _detect_mime(data) == "image/png"

    def test_jpeg(self) -> None:
        """Detect JPEG from magic bytes."""
        data = b"\xff\xd8\xff" + b"\x00" * 50
        assert _detect_mime(data) == "image/jpeg"

    def test_webp(self) -> None:
        """Detect WebP from magic bytes."""
        data = b"RIFF" + b"\x00" * 50
        assert _detect_mime(data) == "image/webp"

    def test_gif(self) -> None:
        """Detect GIF from magic bytes."""
        data = b"GIF89a" + b"\x00" * 50
        assert _detect_mime(data) == "image/gif"

    def test_view_protocol(self) -> None:
        """Detect VIEW protocol from magic bytes."""
        data = b"VIEW" + b"\x00" * 60
        assert _detect_mime(data) == "view"

    def test_unknown(self) -> None:
        """Unknown format returns None."""
        assert _detect_mime(b"random junk data") is None


class TestNdarrayToPng:
    """Tests for NumPy array to PNG conversion."""

    def test_uint8_rgb(self) -> None:
        """uint8 RGB array encodes to valid PNG."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :, 0] = 255
        png = _ndarray_to_png(arr)
        assert png[:4] == b"\x89PNG"

    def test_float32_normalised(self) -> None:
        """float32 array is normalised to [0, 255] uint8."""
        arr = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        png = _ndarray_to_png(arr)
        assert png[:4] == b"\x89PNG"

    def test_single_channel_squeeze(self) -> None:
        """3D array with C=1 is squeezed to 2D before encoding."""
        arr = np.zeros((10, 10, 1), dtype=np.uint8)
        png = _ndarray_to_png(arr)
        assert png[:4] == b"\x89PNG"


class TestToPngBytes:
    """Tests for the _to_png_bytes converter."""

    def test_png_passthrough(self) -> None:
        """PNG bytes pass through unchanged."""
        from PIL import Image

        img = Image.new("RGB", (5, 5), (100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()

        result = _to_png_bytes(png, "auto")
        assert result == png

    def test_jpeg_passthrough(self) -> None:
        """JPEG bytes pass through unchanged (shown inline by browsers)."""
        from PIL import Image

        img = Image.new("RGB", (5, 5), (100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg = buf.getvalue()

        result = _to_png_bytes(jpeg, "auto")
        assert result == jpeg

    def test_unknown_returns_none(self) -> None:
        """Unknown bytes return None."""
        assert _to_png_bytes(b"not an image", "auto") is None

    def test_numpy_struct(self) -> None:
        """numpy-sink struct is converted to PNG."""
        arr = np.zeros((5, 5, 3), dtype=np.uint8)
        struct_val = {
            "data": arr.tobytes(),
            "dtype": "uint8",
            "shape": [5, 5, 3],
            "strides": None,
            "offset": 0,
        }
        result = _to_png_bytes(struct_val, "numpy")
        assert result is not None
        assert result[:4] == b"\x89PNG"

    def test_numpy_struct_null_data(self) -> None:
        """numpy-sink struct with null data returns None."""
        struct_val = {
            "data": None,
            "dtype": None,
            "shape": None,
        }
        assert _to_png_bytes(struct_val, "numpy") is None


class TestIsNotebook:
    """Tests for notebook detection."""

    def test_not_in_notebook(self) -> None:
        """Running in pytest is not a notebook."""
        assert _is_notebook() is False


class TestShowImages:
    """Tests for the main show_images function."""

    def test_missing_column_raises(self) -> None:
        """Missing column name raises KeyError."""
        df = pl.DataFrame({"img": [b"data"]})
        with pytest.raises(KeyError, match="not_here"):
            show_images(df, "not_here")

    def test_text_mode_png(self, create_test_png: Callable) -> None:
        """Text mode prints format summary for PNG images."""
        png = create_test_png(width=10, height=10)
        df = pl.DataFrame({"img": [png, None]}, schema={"img": pl.Binary})

        import io as sio
        import sys

        captured = sio.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            show_images(df, "img")
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "image/png" in output
        assert "null" in output

    def test_text_mode_unknown(self) -> None:
        """Text mode prints 'unknown format' for unrecognised data."""
        df = pl.DataFrame({"img": [b"random bytes"]})

        import io as sio
        import sys

        captured = sio.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            show_images(df, "img")
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "unknown format" in output

    def test_max_rows_respected(self, create_test_png: Callable) -> None:
        """Only max_rows images are displayed."""
        png = create_test_png(width=5, height=5)
        df = pl.DataFrame({"img": [png] * 20})

        import io as sio
        import sys

        captured = sio.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            show_images(df, "img", max_rows=3)
        finally:
            sys.stdout = old_stdout

        lines = [line for line in captured.getvalue().strip().split("\n") if line]
        assert len(lines) == 3
