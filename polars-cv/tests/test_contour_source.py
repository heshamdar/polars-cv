"""
Tests for source('contour') pipeline source type.

These tests verify that contours can be used as pipeline sources,
with rasterization happening inside the pipeline execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from polars_cv.geometry import CONTOUR_SCHEMA

if TYPE_CHECKING:
    pass


def create_square_contour(x: float, y: float, size: float) -> dict:
    """Create a square contour at given position."""
    return {
        "exterior": [
            {"x": x, "y": y},
            {"x": x + size, "y": y},
            {"x": x + size, "y": y + size},
            {"x": x, "y": y + size},
        ],
        "holes": [],
        "is_closed": True,
    }


def create_triangle_contour(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
) -> dict:
    """Create a triangle contour."""
    return {
        "exterior": [
            {"x": x1, "y": y1},
            {"x": x2, "y": y2},
            {"x": x3, "y": y3},
        ],
        "holes": [],
        "is_closed": True,
    }


class TestContourSourceExplicitDims:
    """Tests for contour source with explicit width/height dimensions."""

    def test_basic_rasterization(self) -> None:
        """Basic contour rasterization produces correct output shape."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = Pipeline().source("contour", width=100, height=100).sink("numpy")
        result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))

        assert isinstance(result["mask"].dtype, pl.Struct)
        assert result["mask"].len() == 1

        # Parse the numpy output
        mask_bytes = result["mask"][0]
        arr = numpy_from_struct(mask_bytes)
        assert arr.shape == (100, 100, 1)
        assert arr.dtype == np.uint8

    def test_rasterization_with_fill_values(self) -> None:
        """Fill value and background parameters work correctly."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = (
            Pipeline()
            .source("contour", width=100, height=100, fill_value=128, background=64)
            .sink("numpy")
        )
        result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))

        arr = numpy_from_struct(result["mask"][0])

        # Check that we have the expected fill values
        # Inside contour should be 128, outside should be 64
        assert np.any(arr == 128), "Should have pixels with fill_value=128"
        assert np.any(arr == 64), "Should have pixels with background=64"

    def test_rasterization_multiple_contours(self) -> None:
        """Multiple contours can be rasterized in a single operation."""
        df = pl.DataFrame(
            {
                "contour": [
                    create_square_contour(10, 10, 20),
                    create_triangle_contour(50, 50, 80, 50, 65, 80),
                ],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = Pipeline().source("contour", width=100, height=100).sink("numpy")
        result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))

        assert result["mask"].len() == 2

        # Both should produce valid arrays
        arr1 = numpy_from_struct(result["mask"][0])
        arr2 = numpy_from_struct(result["mask"][1])

        assert arr1.shape == (100, 100, 1)
        assert arr2.shape == (100, 100, 1)

    def test_rasterization_with_operations(self) -> None:
        """Contour rasterization followed by pipeline operations."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        # Rasterize and then blur
        pipe = (
            Pipeline().source("contour", width=100, height=100).blur(2.0).sink("numpy")
        )
        result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))

        arr = numpy_from_struct(result["mask"][0])
        assert arr.shape == (100, 100, 1)

        # Blurred mask should have gradual transitions (not just 0 and 255)
        unique_values = np.unique(arr)
        assert len(unique_values) > 2, "Blurred mask should have smooth transitions"

    def test_rasterization_resize(self) -> None:
        """Contour rasterization followed by resize."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = (
            Pipeline()
            .source("contour", width=100, height=100)
            .resize(width=50, height=50)
            .sink("numpy")
        )
        result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))

        arr = numpy_from_struct(result["mask"][0])
        assert arr.shape == (50, 50, 1)


class TestContourSourceDynamicDims:
    """Tests for contour source with dynamic dimensions from columns."""

    def test_dynamic_width_height(self) -> None:
        """Width and height from column expressions."""
        df = pl.DataFrame(
            {
                "contour": [
                    create_square_contour(5, 5, 20),
                    create_square_contour(10, 10, 30),
                ],
                "w": [50, 100],
                "h": [50, 100],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = (
            Pipeline()
            .source("contour", width=pl.col("w"), height=pl.col("h"))
            .sink("numpy")
        )
        result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))

        # First row: 50x50
        arr1 = numpy_from_struct(result["mask"][0])
        assert arr1.shape == (50, 50, 1)

        # Second row: 100x100
        arr2 = numpy_from_struct(result["mask"][1])
        assert arr2.shape == (100, 100, 1)


class TestContourSourceValidation:
    """Tests for contour source validation."""

    def test_missing_dimensions_error(self) -> None:
        """Error when neither dimensions nor shape provided."""
        with pytest.raises(ValueError, match="Contour source requires"):
            Pipeline().source("contour")

    def test_partial_dimensions_error(self) -> None:
        """Error when only width or only height provided."""
        with pytest.raises(ValueError, match="must be specified together"):
            Pipeline().source("contour", width=100)

        with pytest.raises(ValueError, match="must be specified together"):
            Pipeline().source("contour", height=100)

    def test_both_shape_and_dims_error(self) -> None:
        """Error when both shape and explicit dimensions provided."""
        # This should work (just width/height)
        Pipeline().source("contour", width=100, height=100)

        # Can't easily test the shape + dims conflict without a real LazyPipelineExpr


class TestContourSourceNullHandling:
    """Tests for null handling in contour source."""

    def test_null_contour_produces_null_output(self) -> None:
        """Null contours should produce null outputs."""
        df = pl.DataFrame(
            {
                "contour": [
                    create_square_contour(10, 10, 50),
                    None,
                    create_triangle_contour(20, 20, 60, 20, 40, 60),
                ],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = Pipeline().source("contour", width=100, height=100).sink("numpy")
        result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))

        # First and third rows should have data
        assert result["mask"][0].get("data") is not None
        # Second row should have null fields
        assert result["mask"][1].get("data") is None
        # Third row should have data
        assert result["mask"][2].get("data") is not None


class TestContourSourceIntegration:
    """Integration tests combining contour source with other features."""

    def test_contour_to_threshold(self) -> None:
        """Rasterize contour and apply threshold."""
        df = pl.DataFrame(
            {
                "contour": [create_square_contour(10, 10, 50)],
            }
        ).cast({"contour": CONTOUR_SCHEMA})

        pipe = (
            Pipeline()
            .source("contour", width=100, height=100)
            .threshold(128)
            .sink("numpy")
        )
        result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))

        arr = numpy_from_struct(result["mask"][0])

        # Threshold should produce binary output
        unique_values = set(np.unique(arr))
        assert unique_values.issubset({0, 255}), f"Expected binary, got {unique_values}"
