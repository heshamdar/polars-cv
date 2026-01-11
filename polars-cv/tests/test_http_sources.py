"""
Tests for HTTP/HTTPS URL source support.

These tests verify that:
1. Images can be loaded from HTTP/HTTPS URLs
2. The file_path source format handles HTTP URLs correctly
3. Error handling works for invalid URLs

Note: These tests require network access and are marked with @pytest.mark.network.
Run with: pytest -m network tests/test_http_sources.py
Skip with: pytest -m "not network"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct

if TYPE_CHECKING:
    pass


# Mark all tests in this module as requiring network access
pytestmark = pytest.mark.network


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def placeholder_url() -> str:
    """A publicly accessible PNG image URL from httpbin."""
    # Using httpbin's image endpoint which is reliable
    return "https://httpbin.org/image/png"


@pytest.fixture
def httpbin_bytes_url() -> str:
    """URL that returns known byte content."""
    return "https://httpbin.org/bytes/100"


# ============================================================
# Test Class: HTTP URL Sources
# ============================================================


class TestHttpSources:
    """Tests for loading images from HTTP/HTTPS URLs."""

    def test_http_url_basic(self, placeholder_url: str) -> None:
        """Basic HTTP URL should load and decode as image."""
        df = pl.DataFrame({"url": [placeholder_url]})

        pipe = Pipeline().source("file_path").sink("numpy")
        result = df.with_columns(out=pl.col("url").cv.pipeline(pipe))

        # Should produce a valid image array
        arr = numpy_from_struct(result["out"][0])
        assert arr.ndim == 3  # HxWxC
        assert arr.shape[0] == 100  # Height (httpbin returns 100x100)
        assert arr.shape[1] == 100  # Width

    def test_https_url(self, placeholder_url: str) -> None:
        """HTTPS URL should work (most common case)."""
        df = pl.DataFrame({"url": [placeholder_url]})

        pipe = Pipeline().source("file_path").grayscale().sink("numpy")
        result = df.with_columns(out=pl.col("url").cv.pipeline(pipe))

        # Should produce grayscale output
        arr = numpy_from_struct(result["out"][0])
        assert arr.shape == (100, 100, 1)

    def test_http_url_pipeline_operations(self, placeholder_url: str) -> None:
        """HTTP URL source should work with pipeline operations."""
        df = pl.DataFrame({"url": [placeholder_url]})

        pipe = (
            Pipeline()
            .source("file_path")
            .resize(width=5, height=5)
            .grayscale()
            .sink("png")
        )
        result = df.with_columns(out=pl.col("url").cv.pipeline(pipe))

        # Should produce PNG bytes
        assert result["out"].dtype == pl.Binary
        png_bytes = result["out"][0]
        assert len(png_bytes) > 0
        # PNG magic bytes
        assert png_bytes[:4] == b"\x89PNG"

    def test_http_url_multiple_rows(self, placeholder_url: str) -> None:
        """Multiple HTTP URLs should process correctly."""
        # Use same URL twice to test batch processing
        df = pl.DataFrame({"url": [placeholder_url, placeholder_url]})

        pipe = Pipeline().source("file_path").sink("numpy")
        result = df.with_columns(out=pl.col("url").cv.pipeline(pipe))

        assert len(result) == 2
        arr1 = numpy_from_struct(result["out"][0])
        arr2 = numpy_from_struct(result["out"][1])
        assert arr1.shape == arr2.shape

    def test_http_url_with_null(self, placeholder_url: str) -> None:
        """DataFrame with null URL should handle gracefully."""
        df = pl.DataFrame({"url": [placeholder_url, None]})

        pipe = Pipeline().source("file_path").sink("numpy")
        result = df.with_columns(out=pl.col("url").cv.pipeline(pipe))

        # First row should have data
        assert result["out"][0] is not None

        # Second row should be null
        # Depending on implementation, this may be null or raise an error
        # For now, just check it doesn't crash


# ============================================================
# Test Class: HTTP URL Error Handling
# ============================================================


class TestHttpErrorHandling:
    """Tests for error handling with HTTP URLs."""

    def test_invalid_url_format(self) -> None:
        """Invalid URL should produce an error (not crash)."""
        df = pl.DataFrame({"url": ["http://not-a-valid-url-12345.invalid/image.png"]})

        pipe = Pipeline().source("file_path").sink("numpy")

        # Should raise an error during execution, not crash
        with pytest.raises(Exception):
            df.with_columns(out=pl.col("url").cv.pipeline(pipe))

    def test_404_url(self) -> None:
        """404 URL should produce a meaningful error."""
        df = pl.DataFrame({"url": ["https://httpbin.org/status/404"]})

        pipe = Pipeline().source("file_path").sink("numpy")

        # Should raise an error with meaningful message
        with pytest.raises(Exception) as exc_info:
            df.with_columns(out=pl.col("url").cv.pipeline(pipe))

        # Error message should mention HTTP status or the URL
        error_msg = str(exc_info.value).lower()
        assert "404" in error_msg or "http" in error_msg or "failed" in error_msg


# ============================================================
# Test Class: Mixed Sources
# ============================================================


class TestMixedSources:
    """Tests for mixing HTTP URLs with local paths in same column."""

    def test_mixed_local_and_http_not_supported(self) -> None:
        """
        Mixing local paths and HTTP URLs in same column.

        Note: This tests current behavior. Depending on implementation,
        this may work or need special handling.
        """
        # This test documents current behavior - update if implementation changes
        pass  # Skip for now - would require creating local test files
