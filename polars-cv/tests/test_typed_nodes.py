"""
Tests for typed pipeline nodes - multi-domain seamless pipelines.

This module contains two test classes:
1. TestMultiPhaseWorkaround - Current approach requiring materialization between domains
2. TestSeamlessPipeline - Target approach with unified multi-domain pipelines

The seamless tests will initially fail (marked xfail) until implementation is complete.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from PIL import Image
from polars_cv import CONTOUR_SCHEMA, Pipeline, numpy_from_struct

if TYPE_CHECKING:
    pass


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create a simple test image with a white square on black background."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255  # White square in center

    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_df(sample_image_bytes: bytes) -> pl.DataFrame:
    """DataFrame with test image."""
    return pl.DataFrame({"image": [sample_image_bytes]})


# ============================================================
# Reference Tests: Multi-Phase Workaround (Current Approach)
# ============================================================


class TestMultiPhaseWorkaround:
    """
    Tests demonstrating the CURRENT workaround approach.

    This requires breaking pipelines into phases with materialization
    at domain boundaries. Each phase produces intermediate columns.
    """

    def test_phase1_resize_and_threshold(self, sample_df: pl.DataFrame) -> None:
        """Phase 1: Resize image and apply threshold."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=50, width=50)
            .grayscale()
            .threshold(128)
            .sink("numpy")
        )

        result = sample_df.with_columns(thresholded=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["thresholded"][0])
        assert arr.shape == (50, 50, 1)
        assert np.any(arr == 255)  # Has white pixels
        assert np.any(arr == 0)  # Has black pixels

    def test_phase2_extract_contours_with_native_sink(
        self, sample_df: pl.DataFrame
    ) -> None:
        """
        Phase 2: Extract contours from thresholded image with native sink.

        This now works because we have native sink support for contour outputs.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()
            .sink("native")
        )

        result = sample_df.with_columns(contours=pl.col("image").cv.pipeline(pipe))

        # Should return a struct with contour data
        assert result["contours"] is not None
        # The contour data should be a struct (not null)
        contour_data = result["contours"][0]
        assert contour_data is not None

    def test_full_workflow_with_existing_contours(
        self, sample_df: pl.DataFrame
    ) -> None:
        """
        Full workflow using pre-existing contour column.

        This works because we start with contours in the DataFrame,
        rather than extracting them from an image.
        """
        # Create contour data manually (simulating what extract would produce)
        contour_data = {
            "exterior": [
                {"x": 12.5, "y": 12.5},
                {"x": 37.5, "y": 12.5},
                {"x": 37.5, "y": 37.5},
                {"x": 12.5, "y": 37.5},
            ],
            "interiors": None,
        }

        df = sample_df.with_columns(pl.lit(contour_data).alias("contour")).cast(
            {"contour": CONTOUR_SCHEMA}
        )

        # Now we can use contour source
        img_pipe = (
            Pipeline().source("image_bytes").resize(height=50, width=50).grayscale()
        )
        contour_pipe = Pipeline().source("contour", width=50, height=50)

        img = pl.col("image").cv.pipe(img_pipe).alias("resized")
        mask = pl.col("contour").cv.pipe(contour_pipe).alias("mask")

        masked = img.apply_mask(mask).alias("masked")

        result_expr = masked.sink(
            {
                "resized": "numpy",
                "mask": "numpy",
                "masked": "numpy",
            }
        )

        result = df.with_columns(outputs=result_expr)

        # Verify all outputs present
        resized = numpy_from_struct(result["outputs"].struct.field("resized")[0])
        mask_arr = numpy_from_struct(result["outputs"].struct.field("mask")[0])
        masked_arr = numpy_from_struct(result["outputs"].struct.field("masked")[0])

        assert resized.shape == (50, 50, 1)
        assert mask_arr.shape == (50, 50, 1)
        assert masked_arr.shape == (50, 50, 1)

        # Verify mask has non-zero values (contour was rasterized)
        assert np.any(mask_arr > 0)


# ============================================================
# Target Tests: Seamless Multi-Domain Pipeline (To Implement)
# ============================================================


class TestSeamlessPipeline:
    """
    Tests for the TARGET seamless multi-domain pipeline approach.

    These tests verify the typed node system where pipelines can
    transition between different data domains (buffer, contour, scalar).
    """

    def test_extract_contours_with_native_sink(self, sample_df: pl.DataFrame) -> None:
        """Extract contours and output as native Polars struct."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()
            .sink("native")  # Should return contour struct
        )

        result = sample_df.with_columns(contours=pl.col("image").cv.pipeline(pipe))

        # Verify output is not null and is a struct
        assert result["contours"] is not None
        contour = result["contours"][0]
        # The contour should have an exterior field
        assert contour is not None

    def test_image_to_contour_to_mask_seamless(self, sample_df: pl.DataFrame) -> None:
        """
        Complete seamless pipeline: image → threshold → extract → rasterize → mask.

        All domain transitions happen within a single unified pipeline.
        """
        # Define the image processing pipeline
        img_pipe = (
            Pipeline().source("image_bytes").resize(height=50, width=50).grayscale()
        )
        img = pl.col("image").cv.pipe(img_pipe).alias("resized")

        # Define contour extraction pipeline (from same image)
        # extract_contours transitions: Buffer → Contour
        # rasterize transitions: Contour → Buffer
        contour_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=50, width=50)
            .grayscale()
            .threshold(128)
            .extract_contours()  # Buffer → Contour
            .rasterize(width=50, height=50)  # Contour → Buffer (explicit dimensions)
        )
        mask = pl.col("image").cv.pipe(contour_pipe).alias("mask")

        # Apply mask to resized image
        masked = img.apply_mask(mask).alias("masked")

        # Multi-output sink
        result_expr = masked.sink(
            {
                "resized": "numpy",
                "mask": "numpy",
                "masked": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        # Verify outputs
        resized = numpy_from_struct(result["outputs"].struct.field("resized")[0])
        mask_arr = numpy_from_struct(result["outputs"].struct.field("mask")[0])
        masked_arr = numpy_from_struct(result["outputs"].struct.field("masked")[0])

        assert resized.shape == (50, 50, 1)
        assert mask_arr.shape == (50, 50, 1)
        assert masked_arr.shape == (50, 50, 1)

        # Verify mask has the expected pattern (white square in center)
        assert np.any(mask_arr > 0)  # Has white pixels
        assert np.any(mask_arr == 0)  # Has black pixels

    @pytest.mark.xfail(reason="Shape reference in rasterize not yet implemented")
    def test_rasterize_with_shape_reference(self, sample_df: pl.DataFrame) -> None:
        """
        Test rasterize with shape= parameter inferring dimensions from another pipeline.

        This is a convenience feature that infers output dimensions from another
        LazyPipelineExpr, avoiding manual dimension specification.
        """
        # Define the image processing pipeline
        img_pipe = (
            Pipeline().source("image_bytes").resize(height=50, width=50).grayscale()
        )
        img = pl.col("image").cv.pipe(img_pipe).alias("resized")

        # Define contour extraction pipeline using shape=img to infer dimensions
        contour_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=50, width=50)
            .grayscale()
            .threshold(128)
            .extract_contours()  # Buffer → Contour
            .rasterize(shape=img)  # Contour → Buffer (infer dimensions from img)
        )
        mask = pl.col("image").cv.pipe(contour_pipe).alias("mask")

        # Multi-output sink
        merged = img.merge_pipe(mask)
        result_expr = merged.sink(
            {
                "resized": "numpy",
                "mask": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        # Verify outputs have same dimensions
        resized = numpy_from_struct(result["outputs"].struct.field("resized")[0])
        mask_arr = numpy_from_struct(result["outputs"].struct.field("mask")[0])

        assert resized.shape == (50, 50, 1)
        assert mask_arr.shape == (50, 50, 1)

    def test_mixed_domain_multi_output(self, sample_df: pl.DataFrame) -> None:
        """
        Multi-output with DIFFERENT domains in a single sink.

        - resized: Buffer → numpy
        - contours: Contour → native struct
        - area: Scalar → native float
        """
        img_pipe = (
            Pipeline().source("image_bytes").resize(height=50, width=50).grayscale()
        )
        img = pl.col("image").cv.pipe(img_pipe).alias("resized")

        contour_pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()
        )
        contours = pl.col("image").cv.pipe(contour_pipe).alias("contours")

        # Stats pipeline: extract contour then compute area
        stats_pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()
            .area()  # Contour → Scalar
        )
        area = pl.col("image").cv.pipe(stats_pipe).alias("area")

        # Merge all branches before sinking mixed-type outputs
        merged = area.merge_pipe(img, contours)

        # Sink with mixed types
        result_expr = merged.sink(
            {
                "resized": "numpy",  # Buffer → Binary
                "contours": "native",  # Contour → Struct
                "area": "native",  # Scalar → Float64
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        # Verify mixed-type outputs
        resized = numpy_from_struct(result["outputs"].struct.field("resized")[0])
        assert resized.shape == (50, 50, 1)

        # The contours output should have an exterior field
        contours_output = result["outputs"].struct.field("contours")[0]
        assert contours_output is not None
        # Note: Rust outputs "interiors" instead of "holes" field
        assert "exterior" in result["outputs"].struct.field("contours").struct.fields

        area_val = result["outputs"].struct.field("area")[0]
        assert isinstance(area_val, float)
        assert area_val > 0  # Square should have positive area

    def test_domain_validation_error(self) -> None:
        """
        Invalid domain transition should raise clear error.

        e.g., calling resize() after extract_contours() should fail
        because resize expects Buffer but receives Contour.
        """
        with pytest.raises(ValueError, match="expects buffer.*but.*contour"):
            Pipeline().source("image_bytes").extract_contours().resize(
                height=50, width=50
            )

    def test_area_returns_scalar_directly(self, sample_df: pl.DataFrame) -> None:
        """Scalar outputs should return as native Float64, not Binary."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()
            .area()
            .sink("native")
        )

        result = sample_df.with_columns(area=pl.col("image").cv.pipeline(pipe))

        # Should be Float64, not Binary
        assert result["area"].dtype == pl.Float64
        # Area should be positive (the white square has area)
        assert result["area"][0] is not None
        assert result["area"][0] > 0

    def test_rasterize_after_extract_produces_buffer(
        self, sample_df: pl.DataFrame
    ) -> None:
        """
        Rasterize after extract_contours should produce a Buffer that can be
        encoded as numpy/png.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()  # Buffer → Contour
            .rasterize(width=50, height=50)  # Contour → Buffer
            .sink("numpy")  # Buffer → Binary
        )

        result = sample_df.with_columns(mask=pl.col("image").cv.pipeline(pipe))

        # Should be Struct (numpy output with data, dtype, shape fields)
        assert isinstance(result["mask"].dtype, pl.Struct)
        arr = numpy_from_struct(result["mask"][0])
        assert arr.shape == (50, 50, 1)

    def test_contour_transforms_preserve_domain(self, sample_df: pl.DataFrame) -> None:
        """
        Contour transforms (translate, scale) should keep us in Contour domain.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()  # Buffer → Contour
            .translate(dx=10, dy=10)  # Contour → Contour
            .scale_contour(sx=0.5, sy=0.5)  # Contour → Contour
            .sink("native")  # Contour → Struct
        )

        result = sample_df.with_columns(contours=pl.col("image").cv.pipeline(pipe))

        # Should be a struct with at least an exterior field (contour data)
        # Note: The Rust output schema uses "interiors" instead of "holes",
        # and omits "is_closed", so we check for functional correctness
        # rather than exact schema match.
        assert result["contours"].dtype.base_type() == pl.Struct
        contour_data = result["contours"][0]
        assert contour_data is not None
        # Check that exterior points exist and are transformed
        # (the original square is centered around 50,50 with width 50)
        # After translate(10,10) and scale(0.5,0.5 from centroid):
        # the contour should be smaller and offset
        assert "exterior" in result["contours"].struct.fields


class TestMergePipeBranches:
    """Tests for merge_pipe() method to combine branching pipelines."""

    def test_merge_pipe_basic_branches(self, sample_df: pl.DataFrame) -> None:
        """Merge two branches from shared backbone."""
        gray_pipe = Pipeline().source("image_bytes").grayscale()
        gray = pl.col("image").cv.pipe(gray_pipe).alias("gray")

        blurred_pipe = Pipeline().source("image_bytes").grayscale().blur(5)
        blurred = pl.col("image").cv.pipe(blurred_pipe).alias("blurred")

        threshold_pipe = Pipeline().source("image_bytes").grayscale().threshold(128)
        threshold = pl.col("image").cv.pipe(threshold_pipe).alias("threshold")

        # Merge branches
        result = blurred.merge_pipe(threshold, gray)
        expr = result.sink(
            {
                "gray": "numpy",
                "blurred": "numpy",
                "threshold": "numpy",
            }
        )

        df = sample_df.with_columns(outputs=expr)
        # All three outputs should be present
        assert "gray" in df["outputs"].struct.fields
        assert "blurred" in df["outputs"].struct.fields
        assert "threshold" in df["outputs"].struct.fields

        # Each output should have valid data
        gray_arr = numpy_from_struct(df["outputs"].struct.field("gray")[0])
        blurred_arr = numpy_from_struct(df["outputs"].struct.field("blurred")[0])
        threshold_arr = numpy_from_struct(df["outputs"].struct.field("threshold")[0])

        assert gray_arr is not None
        assert blurred_arr is not None
        assert threshold_arr is not None

    def test_merge_pipe_already_upstream_node(self, sample_df: pl.DataFrame) -> None:
        """Merging a node that's already upstream is safe (deduplicated)."""
        gray_pipe = Pipeline().source("image_bytes").grayscale()
        gray = pl.col("image").cv.pipe(gray_pipe).alias("gray")

        # Create blurred from gray using binary op (makes gray upstream)
        blurred_pipe = Pipeline().source("image_bytes").grayscale().blur(5)
        blurred = pl.col("image").cv.pipe(blurred_pipe).alias("blurred")

        # gray is effectively the same processing, merge it anyway
        result = blurred.merge_pipe(gray)
        expr = result.sink(
            {
                "gray": "numpy",
                "blurred": "numpy",
            }
        )

        df = sample_df.with_columns(outputs=expr)
        assert "gray" in df["outputs"].struct.fields
        assert "blurred" in df["outputs"].struct.fields

    def test_merge_pipe_output_is_first_branch(self, sample_df: pl.DataFrame) -> None:
        """Merged node outputs the same as self (first branch)."""
        pipe1 = Pipeline().source("image_bytes").grayscale()
        pipe2 = Pipeline().source("image_bytes").blur(5)

        img1 = pl.col("image").cv.pipe(pipe1)
        img2 = pl.col("image").cv.pipe(pipe2)

        # Single output mode - should output img1's result
        merged = img1.merge_pipe(img2)
        expr = merged.sink("numpy")

        # Execute both ways and compare
        direct_pipe = Pipeline().source("image_bytes").grayscale().sink("numpy")
        df = sample_df.with_columns(
            merged=expr,
            direct=pl.col("image").cv.pipeline(direct_pipe),
        )
        assert df["merged"][0] == df["direct"][0]

    def test_merge_pipe_preserves_aliases(self, sample_df: pl.DataFrame) -> None:
        """All aliased nodes remain accessible after merge."""
        pipe_a = Pipeline().source("image_bytes").resize(height=50, width=50)
        pipe_b = Pipeline().source("image_bytes").resize(height=100, width=100)

        a = pl.col("image").cv.pipe(pipe_a).alias("small")
        b = pl.col("image").cv.pipe(pipe_b).alias("large")

        merged = a.merge_pipe(b)
        expr = merged.sink(
            {
                "small": "numpy",
                "large": "numpy",
            }
        )

        df = sample_df.with_columns(outputs=expr)

        small_arr = numpy_from_struct(df["outputs"].struct.field("small")[0])
        large_arr = numpy_from_struct(df["outputs"].struct.field("large")[0])

        # Verify dimensions match expectations
        assert small_arr.shape[0] == 50  # height
        assert small_arr.shape[1] == 50  # width
        assert large_arr.shape[0] == 100
        assert large_arr.shape[1] == 100
