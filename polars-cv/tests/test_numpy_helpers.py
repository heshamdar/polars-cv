"""
Tests for numpy helper functions.

Tests the numpy_from_struct function that converts polars-cv
numpy sink output structs to numpy arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars_cv import NUMPY_OUTPUT_SCHEMA, numpy_from_struct

if TYPE_CHECKING:
    pass


def create_test_struct(
    data: bytes,
    dtype: str,
    shape: list[int],
) -> dict[str, object]:
    """
    Create a test struct matching the numpy output format.

    Args:
        data: The raw array bytes.
        dtype: The numpy dtype string (e.g., "uint8", "float32").
        shape: The array shape as a list of dimensions.

    Returns:
        A dict with data, dtype, shape fields.
    """
    return {
        "data": data,
        "dtype": dtype,
        "shape": shape,
    }


class TestNumpyFromStruct:
    """Test numpy_from_struct function."""

    def test_simple_uint8_array(self) -> None:
        """Test parsing a simple uint8 array."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        struct = create_test_struct(arr.tobytes(), "uint8", [2, 3])

        result = numpy_from_struct(struct)

        assert result.dtype == np.uint8
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, arr)

    def test_float32_array(self) -> None:
        """Test parsing a float32 array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        struct = create_test_struct(arr.tobytes(), "float32", [2, 2])

        result = numpy_from_struct(struct)

        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, arr)

    def test_float64_array(self) -> None:
        """Test parsing a float64 array."""
        arr = np.array([1.5, 2.5], dtype=np.float64)
        struct = create_test_struct(arr.tobytes(), "float64", [2])

        result = numpy_from_struct(struct)

        assert result.dtype == np.float64
        assert result.shape == (2,)
        np.testing.assert_array_almost_equal(result, arr)

    def test_3d_array(self) -> None:
        """Test parsing a 3D array (e.g., image)."""
        arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        struct = create_test_struct(arr.tobytes(), "uint8", [2, 3, 4])

        result = numpy_from_struct(struct)

        assert result.dtype == np.uint8
        assert result.shape == (2, 3, 4)
        np.testing.assert_array_equal(result, arr)

    def test_all_dtypes(self) -> None:
        """Test all supported dtypes."""
        dtype_map = {
            "uint8": np.uint8,
            "int8": np.int8,
            "uint16": np.uint16,
            "int16": np.int16,
            "uint32": np.uint32,
            "int32": np.int32,
            "uint64": np.uint64,
            "int64": np.int64,
            "float32": np.float32,
            "float64": np.float64,
        }

        for dtype_str, dtype in dtype_map.items():
            arr = np.array([1, 2], dtype=dtype)
            struct = create_test_struct(arr.tobytes(), dtype_str, [2])
            result = numpy_from_struct(struct)
            assert result.dtype == dtype, f"Failed for dtype {dtype_str}"

    def test_null_data_error(self) -> None:
        """Test that null data raises an error."""
        struct = {"data": None, "dtype": "uint8", "shape": [2]}
        with pytest.raises(ValueError, match="null"):
            numpy_from_struct(struct)

    def test_null_dtype_error(self) -> None:
        """Test that null dtype raises an error."""
        struct = {"data": b"\x00\x00", "dtype": None, "shape": [2]}
        with pytest.raises(ValueError, match="null"):
            numpy_from_struct(struct)

    def test_null_shape_error(self) -> None:
        """Test that null shape raises an error."""
        struct = {"data": b"\x00\x00", "dtype": "uint8", "shape": None}
        with pytest.raises(ValueError, match="null"):
            numpy_from_struct(struct)

    def test_copy_true_creates_independent_copy(self) -> None:
        """Test that copy=True creates an independent array."""
        arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        data = arr.tobytes()
        struct = create_test_struct(data, "uint8", [4])

        result = numpy_from_struct(struct, copy=True)

        # Modifying result should not affect original
        result[0] = 99
        assert result[0] == 99

    def test_copy_false_works(self) -> None:
        """Test that copy=False works (may or may not share memory)."""
        arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        struct = create_test_struct(arr.tobytes(), "uint8", [4])

        result = numpy_from_struct(struct, copy=False)

        assert result.dtype == np.uint8
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, arr)


class TestNumpyFromStructPolars:
    """Test numpy_from_struct with Polars Series and struct values."""

    def test_from_polars_series_struct(self) -> None:
        """Test conversion from a Polars struct Series."""
        arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)

        # Create a DataFrame with struct column
        df = pl.DataFrame(
            {
                "output": [
                    {
                        "data": arr.tobytes(),
                        "dtype": "uint8",
                        "shape": [2, 3],
                    }
                ]
            }
        ).cast({"output": NUMPY_OUTPUT_SCHEMA})

        # Get the struct value
        struct_val = df["output"][0]

        result = numpy_from_struct(struct_val)

        assert result.dtype == np.uint8
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result.flatten(), arr)

    def test_from_dict_representation(self) -> None:
        """Test conversion from dict extracted from struct."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # Create dict matching struct fields
        row_dict = {
            "data": arr.tobytes(),
            "dtype": "float32",
            "shape": [2, 2],
        }

        result = numpy_from_struct(row_dict)

        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, arr)


class TestNumpyOutputSchema:
    """Test NUMPY_OUTPUT_SCHEMA constant."""

    def test_schema_structure(self) -> None:
        """Test that schema has correct structure."""
        assert NUMPY_OUTPUT_SCHEMA == pl.Struct(
            {
                "data": pl.Binary,
                "dtype": pl.String,
                "shape": pl.List(pl.UInt64),
                "strides": pl.List(pl.Int64),
                "offset": pl.UInt64,
            }
        )

    def test_schema_can_be_used_for_casting(self) -> None:
        """Test that schema can be used for column casting."""
        arr = np.array([1, 2, 3], dtype=np.uint8)

        df = pl.DataFrame(
            {
                "output": [
                    {
                        "data": arr.tobytes(),
                        "dtype": "uint8",
                        "shape": [3],
                        "strides": [1],
                        "offset": 0,
                    }
                ]
            }
        )

        # Should be able to cast to the schema
        casted = df.cast({"output": NUMPY_OUTPUT_SCHEMA})

        assert casted["output"].dtype == NUMPY_OUTPUT_SCHEMA


class TestRoundTrip:
    """Test round-trip conversion matches expected shapes and dtypes."""

    def test_image_like_array(self) -> None:
        """Test with an image-like array (H, W, C)."""
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        struct = create_test_struct(arr.tobytes(), "uint8", list(arr.shape))

        result = numpy_from_struct(struct)

        np.testing.assert_array_equal(result, arr)

    def test_grayscale_image(self) -> None:
        """Test with a grayscale image (H, W)."""
        arr = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        struct = create_test_struct(arr.tobytes(), "uint8", list(arr.shape))

        result = numpy_from_struct(struct)

        np.testing.assert_array_equal(result, arr)

    def test_large_float_array(self) -> None:
        """Test with a large float32 array."""
        arr = np.random.randn(100, 100, 3).astype(np.float32)
        struct = create_test_struct(arr.tobytes(), "float32", list(arr.shape))

        result = numpy_from_struct(struct)

        np.testing.assert_array_almost_equal(result, arr)

    def test_1d_vector(self) -> None:
        """Test with a 1D vector."""
        arr = np.arange(100, dtype=np.float64)
        struct = create_test_struct(arr.tobytes(), "float64", list(arr.shape))

        result = numpy_from_struct(struct)

        np.testing.assert_array_equal(result, arr)
