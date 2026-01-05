"""
Tests for numpy helper functions.

Tests the numpy_from_bytes, numpy_shape, numpy_dtype, and numpy_header_size
functions that parse the polars-cv numpy sink output format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from polars_cv import numpy_dtype, numpy_from_bytes, numpy_header_size, numpy_shape

if TYPE_CHECKING:
    pass


def create_test_bytes(dtype_code: int, shape: list[int], data: bytes) -> bytes:
    """
    Create test bytes in the polars-cv numpy format.

    Args:
        dtype_code: The dtype code (0-9).
        shape: The shape as a list of dimensions.
        data: The raw array data.

    Returns:
        The encoded bytes.
    """
    result = bytearray()
    result.append(dtype_code)
    result.append(len(shape))

    for dim in shape:
        result.extend(dim.to_bytes(8, "little"))

    result.extend(data)
    return bytes(result)


class TestNumpyFromBytes:
    """Test numpy_from_bytes function."""

    def test_simple_uint8_array(self) -> None:
        """Test parsing a simple uint8 array."""
        data = bytes([1, 2, 3, 4, 5, 6])
        encoded = create_test_bytes(0, [2, 3], data)

        result = numpy_from_bytes(encoded)

        assert result.dtype == np.uint8
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result.flatten(), [1, 2, 3, 4, 5, 6])

    def test_float32_array(self) -> None:
        """Test parsing a float32 array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        encoded = create_test_bytes(8, [2, 2], arr.tobytes())

        result = numpy_from_bytes(encoded)

        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result.flatten(), [1.0, 2.0, 3.0, 4.0])

    def test_float64_array(self) -> None:
        """Test parsing a float64 array."""
        arr = np.array([1.5, 2.5], dtype=np.float64)
        encoded = create_test_bytes(9, [2], arr.tobytes())

        result = numpy_from_bytes(encoded)

        assert result.dtype == np.float64
        assert result.shape == (2,)
        np.testing.assert_array_almost_equal(result, [1.5, 2.5])

    def test_3d_array(self) -> None:
        """Test parsing a 3D array (e.g., image)."""
        # 2x3x4 uint8 array
        data = bytes(range(24))
        encoded = create_test_bytes(0, [2, 3, 4], data)

        result = numpy_from_bytes(encoded)

        assert result.dtype == np.uint8
        assert result.shape == (2, 3, 4)

    def test_all_dtypes(self) -> None:
        """Test all supported dtypes."""
        dtype_map = {
            0: np.uint8,
            1: np.int8,
            2: np.uint16,
            3: np.int16,
            4: np.uint32,
            5: np.int32,
            6: np.uint64,
            7: np.int64,
            8: np.float32,
            9: np.float64,
        }

        for code, dtype in dtype_map.items():
            arr = np.array([1, 2], dtype=dtype)
            encoded = create_test_bytes(code, [2], arr.tobytes())
            result = numpy_from_bytes(encoded)
            assert result.dtype == dtype, f"Failed for dtype code {code}"

    def test_empty_data_error(self) -> None:
        """Test that empty data raises an error."""
        with pytest.raises(ValueError, match="too short"):
            numpy_from_bytes(b"")

    def test_short_data_error(self) -> None:
        """Test that short data raises an error."""
        with pytest.raises(ValueError, match="too short"):
            numpy_from_bytes(b"\x00")

    def test_invalid_dtype_code(self) -> None:
        """Test that invalid dtype code raises an error."""
        # dtype code 99 is invalid
        encoded = create_test_bytes(99, [2], b"\x00\x00")
        with pytest.raises(ValueError, match="Unknown dtype code"):
            numpy_from_bytes(encoded)

    def test_size_mismatch_error(self) -> None:
        """Test that size mismatch raises an error."""
        # Shape says 4 elements but only 2 bytes of data
        encoded = create_test_bytes(0, [4], b"\x00\x00")
        with pytest.raises(ValueError, match="size mismatch"):
            numpy_from_bytes(encoded)


class TestNumpyShape:
    """Test numpy_shape function."""

    def test_1d_shape(self) -> None:
        """Test extracting 1D shape."""
        encoded = create_test_bytes(0, [10], b"\x00" * 10)
        assert numpy_shape(encoded) == (10,)

    def test_2d_shape(self) -> None:
        """Test extracting 2D shape."""
        encoded = create_test_bytes(0, [5, 3], b"\x00" * 15)
        assert numpy_shape(encoded) == (5, 3)

    def test_3d_shape(self) -> None:
        """Test extracting 3D shape."""
        encoded = create_test_bytes(0, [224, 224, 3], b"\x00" * (224 * 224 * 3))
        assert numpy_shape(encoded) == (224, 224, 3)

    def test_empty_data(self) -> None:
        """Test with empty data."""
        assert numpy_shape(b"") == ()


class TestNumpyDtype:
    """Test numpy_dtype function."""

    def test_all_dtypes(self) -> None:
        """Test extracting all dtype codes."""
        dtype_names = {
            0: "uint8",
            1: "int8",
            2: "uint16",
            3: "int16",
            4: "uint32",
            5: "int32",
            6: "uint64",
            7: "int64",
            8: "float32",
            9: "float64",
        }

        for code, name in dtype_names.items():
            encoded = create_test_bytes(code, [1], b"\x00" * 8)
            assert numpy_dtype(encoded) == name

    def test_empty_data_error(self) -> None:
        """Test with empty data."""
        with pytest.raises(ValueError, match="too short"):
            numpy_dtype(b"")


class TestNumpyHeaderSize:
    """Test numpy_header_size function."""

    def test_1d_header(self) -> None:
        """Test header size for 1D array."""
        encoded = create_test_bytes(0, [10], b"\x00" * 10)
        assert numpy_header_size(encoded) == 2 + 1 * 8  # 10 bytes

    def test_2d_header(self) -> None:
        """Test header size for 2D array."""
        encoded = create_test_bytes(0, [5, 3], b"\x00" * 15)
        assert numpy_header_size(encoded) == 2 + 2 * 8  # 18 bytes

    def test_3d_header(self) -> None:
        """Test header size for 3D array."""
        encoded = create_test_bytes(0, [2, 3, 4], b"\x00" * 24)
        assert numpy_header_size(encoded) == 2 + 3 * 8  # 26 bytes

    def test_empty_data(self) -> None:
        """Test with empty data."""
        assert numpy_header_size(b"") == 0


class TestRoundTrip:
    """Test round-trip conversion matches expected shapes and dtypes."""

    def test_image_like_array(self) -> None:
        """Test with an image-like array (H, W, C)."""
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        encoded = create_test_bytes(0, list(arr.shape), arr.tobytes())

        result = numpy_from_bytes(encoded)

        np.testing.assert_array_equal(result, arr)

    def test_grayscale_image(self) -> None:
        """Test with a grayscale image (H, W)."""
        arr = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        encoded = create_test_bytes(0, list(arr.shape), arr.tobytes())

        result = numpy_from_bytes(encoded)

        np.testing.assert_array_equal(result, arr)
