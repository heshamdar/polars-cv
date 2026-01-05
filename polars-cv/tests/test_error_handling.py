"""
Tests for batch-level error handling in pipeline execution.

These tests verify that:
1. Batches with all valid images succeed
2. Corrupted images produce clear errors (not panics)
3. Error messages are informative

These tests establish the expected error handling behavior before and after
the batch-level panic optimization is implemented.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_cv import Pipeline

if TYPE_CHECKING:
    pass


def create_test_png(
    width: int = 10, height: int = 10, color: tuple[int, int, int] = (255, 0, 0)
) -> bytes:
    """
    Create a test PNG image.

    Args:
        width: Image width.
        height: Image height.
        color: RGB color tuple.

    Returns:
        PNG bytes.
    """
    try:
        from PIL import Image

        img = Image.new("RGB", (width, height), color)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""


def _plugin_available() -> bool:
    """Check if the compiled plugin is available."""
    from pathlib import Path

    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


@plugin_required
class TestBatchWithValidImages:
    """Test that batches with all valid images process correctly."""

    def test_single_valid_image_succeeds(self) -> None:
        """Single valid image should process without error."""
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
        assert "processed" in result.columns
        assert result["processed"].dtype == pl.Binary
        assert result["processed"][0] is not None

    def test_multiple_valid_images_succeed(self) -> None:
        """Multiple valid images should all process correctly."""
        pipe = Pipeline().source("image_bytes").resize(height=32, width=32).sink("blob")

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes] * 10})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
        assert result["processed"].len() == 10
        for i in range(10):
            assert result["processed"][i] is not None

    def test_large_batch_succeeds(self) -> None:
        """Large batch of valid images should process correctly."""
        pipe = Pipeline().source("image_bytes").grayscale().sink("blob")

        png_bytes = create_test_png(50, 50)
        df = pl.DataFrame({"images": [png_bytes] * 100})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
        assert result["processed"].len() == 100
        # Verify all results are non-null
        assert result["processed"].null_count() == 0


@plugin_required
class TestCorruptedImageHandling:
    """Test handling of corrupted or invalid image data."""

    def test_corrupted_bytes_fails_gracefully(self) -> None:
        """Corrupted image bytes should produce an error, not panic."""
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        # Completely random garbage bytes
        corrupted_bytes = b"this is not a valid image at all"
        df = pl.DataFrame({"images": [corrupted_bytes]})

        with pytest.raises(Exception) as exc_info:
            df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Verify we got a meaningful error message
        error_message = str(exc_info.value).lower()
        # Error should mention something about decoding/image/format
        assert any(
            word in error_message for word in ["decode", "image", "format", "failed"]
        )

    def test_truncated_png_fails_gracefully(self) -> None:
        """Truncated PNG should produce clear error."""
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        # Create a valid PNG then truncate it
        png_bytes = create_test_png(100, 100)
        truncated = png_bytes[: len(png_bytes) // 2]  # Cut it in half

        df = pl.DataFrame({"images": [truncated]})

        with pytest.raises(Exception) as exc_info:
            df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Should get an error about decoding
        error_message = str(exc_info.value).lower()
        assert any(
            word in error_message for word in ["decode", "image", "failed", "error"]
        )

    def test_empty_bytes_fails_gracefully(self) -> None:
        """Empty bytes should produce clear error."""
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        df = pl.DataFrame({"images": [b""]})

        with pytest.raises(Exception) as exc_info:
            df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Should get an error about decoding
        error_message = str(exc_info.value).lower()
        assert any(
            word in error_message
            for word in ["decode", "image", "failed", "error", "empty"]
        )


@plugin_required
class TestNullHandling:
    """Test handling of null values in image columns."""

    def test_null_value_produces_null_output(self) -> None:
        """Null input values should produce null outputs."""
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes, None, png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # First and last should be non-null
        assert result["processed"][0] is not None
        assert result["processed"][2] is not None
        # Middle should be null
        assert result["processed"][1] is None

    def test_all_null_values(self) -> None:
        """All null inputs should produce all null outputs."""
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        df = pl.DataFrame({"images": [None, None, None]}).cast({"images": pl.Binary})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        assert result["processed"].null_count() == 3

    def test_null_first_then_blobs(self) -> None:
        """Test [None, blob, blob] ordering - null comes first.

        This tests whether null handling at the start affects subsequent rows.
        Regression test for potential memory/state issues after encountering nulls.
        """
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [None, png_bytes, png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # First should be null
        assert result["processed"][0] is None
        # Second and third should be non-null and valid
        assert result["processed"][1] is not None
        assert result["processed"][2] is not None

    def test_multiple_nulls_between_blobs(self) -> None:
        """Test [blob, None, None, blob] ordering - consecutive nulls.

        This tests whether multiple consecutive nulls affect subsequent processing.
        """
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes, None, None, png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # First and last should be non-null
        assert result["processed"][0] is not None
        assert result["processed"][3] is not None
        # Middle two should be null
        assert result["processed"][1] is None
        assert result["processed"][2] is None

    def test_null_last_after_blobs(self) -> None:
        """Test [blob, blob, None] ordering - null comes last.

        This tests whether null at the end is handled correctly.
        """
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes, png_bytes, None]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # First two should be non-null
        assert result["processed"][0] is not None
        assert result["processed"][1] is not None
        # Last should be null
        assert result["processed"][2] is None

    def test_alternating_null_and_blobs(self) -> None:
        """Test [blob, None, blob, None, blob] alternating pattern.

        This tests state management with frequent null/non-null transitions.
        """
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes, None, png_bytes, None, png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Odd indices (0, 2, 4) should be non-null
        assert result["processed"][0] is not None
        assert result["processed"][2] is not None
        assert result["processed"][4] is not None
        # Even indices (1, 3) should be null
        assert result["processed"][1] is None
        assert result["processed"][3] is None

    def test_separate_blob_objects_per_row(self) -> None:
        """Test using separate blob objects for each row.

        Creates distinct PNG bytes for each row to test if object identity
        or memory aliasing affects null handling behavior.
        Regression test for potential issues with shared byte references.
        """
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        # Create SEPARATE blob objects (different byte instances)
        png1 = create_test_png(100, 100, (255, 0, 0))  # Red
        png2 = create_test_png(100, 100, (0, 255, 0))  # Green
        png3 = create_test_png(100, 100, (0, 0, 255))  # Blue

        df = pl.DataFrame({"images": [png1, None, png2, None, png3]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Non-null rows should be processed
        assert result["processed"][0] is not None
        assert result["processed"][2] is not None
        assert result["processed"][4] is not None
        # Null rows should remain null
        assert result["processed"][1] is None
        assert result["processed"][3] is None

    def test_same_blob_object_reused(self) -> None:
        """Test using the same blob object reference multiple times.

        Uses the same bytes object for all non-null rows to test if
        Polars/Arrow memory handling causes issues with aliased data.
        Regression test for potential pointer/slice aliasing bugs.
        """
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        # Use the SAME bytes object for all rows
        shared_png = create_test_png(100, 100)
        df = pl.DataFrame({"images": [shared_png, None, shared_png]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # First and last should be non-null
        assert result["processed"][0] is not None
        assert result["processed"][2] is not None
        # Middle should be null
        assert result["processed"][1] is None
        # Both non-null results should be valid (same size output)
        assert len(result["processed"][0]) > 0
        assert len(result["processed"][2]) > 0

    def test_null_handling_with_larger_batch(self) -> None:
        """Test null handling with a larger batch of mixed values.

        Stress test for null handling in realistic batch sizes.
        """
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        png_bytes = create_test_png(100, 100)
        # Create pattern: blob, null, blob, null, ... (50 total)
        images = [png_bytes if i % 2 == 0 else None for i in range(50)]
        df = pl.DataFrame({"images": images})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Check all results
        for i in range(50):
            if i % 2 == 0:
                assert result["processed"][i] is not None, f"Row {i} should be non-null"
            else:
                assert result["processed"][i] is None, f"Row {i} should be null"

    def test_null_handling_with_complex_pipeline(self) -> None:
        """Test null handling with a more complex pipeline.

        Ensures null propagation works correctly through multiple operations.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .grayscale()
            .threshold(128)
            .sink("blob")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes, None, png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # First and last should be non-null
        assert result["processed"][0] is not None
        assert result["processed"][2] is not None
        # Middle should be null
        assert result["processed"][1] is None


@plugin_required
class TestErrorMessageQuality:
    """Test that error messages are informative."""

    def test_error_indicates_row_or_nature_of_failure(self) -> None:
        """Error message should be informative about the failure."""
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        # Invalid image data
        invalid_bytes = b"PNG\x00\x00\x00\x00invalid"
        df = pl.DataFrame({"images": [invalid_bytes]})

        with pytest.raises(Exception) as exc_info:
            df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        error_message = str(exc_info.value)
        # Error should not be completely empty or generic
        assert len(error_message) > 10, "Error message should be descriptive"

    def test_error_with_complex_pipeline(self) -> None:
        """Errors in complex pipelines should be informative."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=100)
            .grayscale()
            .threshold(128)
            .sink("blob")
        )

        # Invalid data should fail at decode stage
        invalid_bytes = bytes([0xFF] * 50)  # Not a valid image
        df = pl.DataFrame({"images": [invalid_bytes]})

        with pytest.raises(Exception) as exc_info:
            df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        error_message = str(exc_info.value).lower()
        # Should mention decode or image failure
        assert any(
            word in error_message for word in ["decode", "image", "failed", "format"]
        )


@plugin_required
class TestStreamingErrorHandling:
    """Test error handling in streaming execution mode."""

    def test_streaming_corrupted_image_error(self) -> None:
        """Corrupted images in streaming mode should fail gracefully."""
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        corrupted_bytes = b"not an image"
        df = pl.DataFrame({"images": [corrupted_bytes]})

        with pytest.raises(Exception):
            (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )

    def test_streaming_null_handling(self) -> None:
        """Null values in streaming mode should produce null outputs."""
        pipe = Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes, None, png_bytes]})

        result = (
            df.lazy()
            .with_columns(processed=pl.col("images").cv.pipeline(pipe))
            .collect(engine="streaming")
        )

        assert result["processed"][0] is not None
        assert result["processed"][1] is None
        assert result["processed"][2] is not None
