"""
Tests for source types including list and array sources.

These tests verify that:
1. Existing source types (image_bytes, blob, raw, file_path, contour) work correctly
2. New list/array source types can accept Polars nested List/Array columns
3. The sink -> source round-trip preserves data correctly
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct

if TYPE_CHECKING:
    pass


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_image_bytes() -> bytes:
    """Create a simple 4x4 grayscale test image."""
    img = np.full((4, 4), 128, dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def rgb_image_bytes() -> bytes:
    """Create a 4x4 RGB test image."""
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# Test Class: Raw Source Type
# ============================================================


class TestRawSource:
    """Tests for the 'raw' source format."""

    def test_raw_source_1d_float32(self) -> None:
        """Raw source with f32 dtype should produce Float32 output."""
        data = np.array([1.0, 2.5, 100.0], dtype=np.float32)
        raw_bytes = data.tobytes()

        df = pl.DataFrame({"raw": [raw_bytes]})
        pipe = Pipeline().source("raw", dtype="f32").sink("list")
        result = df.with_columns(out=pl.col("raw").cv.pipeline(pipe))

        assert result["out"].dtype == pl.List(pl.Float32)
        assert result["out"][0].to_list() == [1.0, 2.5, 100.0]

    def test_raw_source_reshaped_2d(self) -> None:
        """Raw source reshaped to 2D should produce nested lists."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        raw_bytes = data.tobytes()

        df = pl.DataFrame({"raw": [raw_bytes]})
        pipe = Pipeline().source("raw", dtype="f32").reshape([2, 2]).sink("list")
        result = df.with_columns(out=pl.col("raw").cv.pipeline(pipe))

        assert result["out"].dtype == pl.List(pl.List(pl.Float32))
        assert result["out"][0].to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_raw_source_requires_dtype(self) -> None:
        """Raw source without dtype should raise error."""
        with pytest.raises(ValueError, match="dtype is required"):
            Pipeline().source("raw")


# ============================================================
# Test Class: List Sink Shape Preservation
# ============================================================


class TestListSinkShapePreservation:
    """Tests verifying list sink preserves any dimensionality shape."""

    def test_1d_shape_flat_list(self, rgb_image_bytes: bytes) -> None:
        """1D buffer (e.g., perceptual hash) should produce flat List."""
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(out=pl.col("image").cv.pipeline(pipe))

        # 1D shape [8] -> List(UInt8)
        assert result["out"].dtype == pl.List(pl.UInt8)
        values = result["out"][0].to_list()
        assert len(values) == 8

    def test_2d_shape_nested_list(self, simple_image_bytes: bytes) -> None:
        """2D buffer should produce 2-level nested List."""
        df = pl.DataFrame({"image": [simple_image_bytes]})

        # Grayscale produces [4, 4, 1], reshape to [4, 4]
        pipe = Pipeline().source("image_bytes").grayscale().reshape([4, 4]).sink("list")
        result = df.with_columns(out=pl.col("image").cv.pipeline(pipe))

        # 2D shape [4, 4] -> List(List(UInt8))
        assert result["out"].dtype == pl.List(pl.List(pl.UInt8))
        values = result["out"][0].to_list()
        assert len(values) == 4  # 4 rows
        assert len(values[0]) == 4  # 4 columns

    def test_3d_shape_triple_nested_list(self, rgb_image_bytes: bytes) -> None:
        """3D buffer should produce 3-level nested List."""
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").sink("list")
        result = df.with_columns(out=pl.col("image").cv.pipeline(pipe))

        # 3D shape [4, 4, 3] -> List(List(List(UInt8)))
        assert result["out"].dtype == pl.List(pl.List(pl.List(pl.UInt8)))
        values = result["out"][0].to_list()
        assert len(values) == 4  # 4 rows
        assert len(values[0]) == 4  # 4 columns
        assert len(values[0][0]) == 3  # 3 channels


# ============================================================
# Test Class: List Source Type (TO BE IMPLEMENTED)
# ============================================================


class TestListSource:
    """Tests for the 'list' source format (accepting Polars nested Lists)."""

    def test_list_source_2d_basic(self) -> None:
        """List source should accept 2D nested list input."""
        # Create a 3x3 grayscale image as nested list
        pixels = [[100, 150, 200], [50, 100, 150], [0, 50, 100]]
        df = pl.DataFrame({"pixels": [pixels]})

        # Process with list source
        pipe = Pipeline().source("list", dtype="u8").sink("list")
        result = df.with_columns(out=pl.col("pixels").cv.pipeline(pipe))

        # Should preserve the structure
        assert result["out"].dtype == pl.List(pl.List(pl.UInt8))
        assert result["out"][0].to_list() == pixels

    def test_list_source_3d_rgb(self) -> None:
        """List source should accept 3D nested list input (RGB)."""
        # Create a 2x2 RGB image as nested list
        pixels = [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ]
        df = pl.DataFrame({"pixels": [pixels]})

        # Process with list source - just sink back to list to verify shape preserved
        pipe = Pipeline().source("list", dtype="u8").sink("list")
        result = df.with_columns(out=pl.col("pixels").cv.pipeline(pipe))

        # Should preserve 3D nested structure
        assert result["out"].dtype == pl.List(pl.List(pl.List(pl.UInt8)))
        assert result["out"][0].to_list() == pixels

    def test_list_source_round_trip(self, rgb_image_bytes: bytes) -> None:
        """Data should round-trip: image_bytes -> list sink -> list source."""
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        # First: convert image to list
        pipe1 = Pipeline().source("image_bytes").sink("list")
        df2 = df.with_columns(pixels=pl.col("image").cv.pipeline(pipe1))

        # Second: use list source for processing
        pipe2 = Pipeline().source("list", dtype="u8").grayscale().sink("numpy")
        result = df2.with_columns(gray=pl.col("pixels").cv.pipeline(pipe2))

        # Verify grayscale output
        gray_arr = numpy_from_struct(result["gray"][0])
        assert gray_arr.shape == (4, 4, 1)


# ============================================================
# Test Class: Array Source Type (TO BE IMPLEMENTED)
# ============================================================


class TestArraySource:
    """Tests for the 'array' source format (accepting Polars fixed-size Arrays)."""

    def test_array_source_1d(self) -> None:
        """Array source should accept 1D fixed-size array input."""
        # Create fixed-size array column
        df = pl.DataFrame({"data": [[1, 2, 3, 4]]}).cast(
            {"data": pl.Array(pl.UInt8, 4)}
        )

        pipe = Pipeline().source("array", dtype="u8").sink("list")
        result = df.with_columns(out=pl.col("data").cv.pipeline(pipe))

        assert result["out"].dtype == pl.List(pl.UInt8)
        assert result["out"][0].to_list() == [1, 2, 3, 4]

    def test_array_source_2d(self) -> None:
        """Array source should accept 2D fixed-size array input."""
        # Create 2D fixed-size array: Array[Array[UInt8, 3], 2]
        data = [[1, 2, 3], [4, 5, 6]]
        df = pl.DataFrame({"data": [data]}).cast(
            {"data": pl.Array(pl.Array(pl.UInt8, 3), 2)}
        )

        pipe = Pipeline().source("array", dtype="u8").sink("list")
        result = df.with_columns(out=pl.col("data").cv.pipeline(pipe))

        assert result["out"].dtype == pl.List(pl.List(pl.UInt8))
        assert result["out"][0].to_list() == data


# ============================================================
# Test Class: Source Format Validation
# ============================================================


class TestSourceFormatValidation:
    """Tests for source format validation."""

    def test_invalid_source_format_error(self) -> None:
        """Invalid source format should raise ValueError with valid options."""
        with pytest.raises(ValueError) as exc_info:
            Pipeline().source("invalid_format")

        assert "Invalid source format" in str(exc_info.value)
        # Should list valid formats
        assert "image_bytes" in str(exc_info.value)

    def test_image_bytes_source(self, simple_image_bytes: bytes) -> None:
        """image_bytes source should decode PNG/JPEG."""
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(out=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["out"][0])
        assert arr.shape == (4, 4, 3)  # Decoded to RGB

    def test_blob_source(self, simple_image_bytes: bytes) -> None:
        """blob source should decode VIEW protocol binary."""
        df = pl.DataFrame({"image": [simple_image_bytes]})

        # First encode as blob
        pipe1 = Pipeline().source("image_bytes").sink("blob")
        df2 = df.with_columns(blob=pl.col("image").cv.pipeline(pipe1))

        # Then decode from blob
        pipe2 = Pipeline().source("blob").sink("numpy")
        result = df2.with_columns(out=pl.col("blob").cv.pipeline(pipe2))

        arr = numpy_from_struct(result["out"][0])
        assert arr.shape == (4, 4, 3)
