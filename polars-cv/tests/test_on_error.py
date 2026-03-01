"""Tests for on_error source decoding parameter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import polars as pl
import polars_cv  # noqa: F401 — registers .cv namespace
import pytest
from polars_cv import Pipeline

from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


# ============================================================================
# Python-layer validation (no plugin needed)
# ============================================================================


class TestOnErrorValidation:
    """Tests for on_error parameter validation at pipeline construction time."""

    def test_default_is_raise(self) -> None:
        """Default on_error is 'raise'."""
        pipe = Pipeline().source("image_bytes")
        assert pipe._source is not None
        assert pipe._source.on_error == "raise"

    def test_on_error_null(self) -> None:
        """on_error='null' is accepted."""
        pipe = Pipeline().source("image_bytes", on_error="null")
        assert pipe._source is not None
        assert pipe._source.on_error == "null"

    def test_invalid_on_error(self) -> None:
        """Invalid on_error value raises ValueError."""
        with pytest.raises(ValueError, match="on_error must be"):
            Pipeline().source("image_bytes", on_error="skip")

    def test_on_error_serialized(self) -> None:
        """on_error='null' is included in serialized dict."""
        pipe = Pipeline().source("image_bytes", on_error="null")
        assert pipe._source is not None
        d = pipe._source.to_dict()
        assert d["on_error"] == "null"

    def test_on_error_raise_not_serialized(self) -> None:
        """on_error='raise' (default) is omitted from serialized dict."""
        pipe = Pipeline().source("image_bytes")
        assert pipe._source is not None
        d = pipe._source.to_dict()
        assert "on_error" not in d


# ============================================================================
# Plugin integration tests
# ============================================================================


@plugin_required
class TestOnErrorRaise:
    """Tests for default on_error='raise' behaviour (existing behaviour)."""

    def test_corrupt_bytes_raises(self) -> None:
        """Corrupt bytes with default on_error='raise' raises an error."""
        df = pl.DataFrame({"img": [b"not a valid image"]})
        pipe = Pipeline().source("image_bytes").grayscale()
        expr = pl.col("img").cv.pipe(pipe).sink("png")
        with pytest.raises(Exception):
            df.with_columns(out=expr).collect()


@plugin_required
class TestOnErrorNull:
    """Tests for on_error='null' behaviour."""

    def test_corrupt_bytes_null(self) -> None:
        """Corrupt bytes with on_error='null' produces null output."""
        df = pl.DataFrame({"img": [b"not a valid image"]})
        pipe = Pipeline().source("image_bytes", on_error="null").grayscale()
        expr = pl.col("img").cv.pipe(pipe).sink("png")
        result = df.with_columns(out=expr)
        assert result["out"][0] is None

    def test_mixed_valid_corrupt(self, create_test_png: Callable) -> None:
        """Mix of valid and corrupt rows: valid rows processed, corrupt rows null."""
        good = create_test_png(width=10, height=10)
        bad = b"garbage data"
        df = pl.DataFrame({"img": [good, bad, good]})
        pipe = Pipeline().source("image_bytes", on_error="null").grayscale()
        expr = pl.col("img").cv.pipe(pipe).sink("png")
        result = df.with_columns(out=expr)

        assert result["out"][0] is not None
        assert result["out"][1] is None
        assert result["out"][2] is not None

    def test_all_corrupt_null(self) -> None:
        """All corrupt rows produces all null outputs."""
        df = pl.DataFrame({"img": [b"bad1", b"bad2"]})
        pipe = Pipeline().source("image_bytes", on_error="null")
        expr = pl.col("img").cv.pipe(pipe).sink("png")
        result = df.with_columns(out=expr)
        assert result["out"].null_count() == 2

    def test_null_input_still_null(self) -> None:
        """Explicit null input remains null (unchanged behaviour)."""
        df = pl.DataFrame({"img": [None]}, schema={"img": pl.Binary})
        pipe = Pipeline().source("image_bytes", on_error="null")
        expr = pl.col("img").cv.pipe(pipe).sink("png")
        result = df.with_columns(out=expr)
        assert result["out"][0] is None

    def test_file_path_bad_path(self) -> None:
        """Bad file path with on_error='null' produces null."""
        df = pl.DataFrame({"path": ["/no/such/file.png"]})
        pipe = Pipeline().source("file_path", on_error="null")
        expr = pl.col("path").cv.pipe(pipe).sink("png")
        result = df.with_columns(out=expr)
        assert result["out"][0] is None

    def test_numpy_sink_with_on_error(self, create_test_png: Callable) -> None:
        """on_error='null' works correctly with numpy sink (struct output)."""
        good = create_test_png(width=10, height=10)
        bad = b"corrupt"
        df = pl.DataFrame({"img": [good, bad]})
        pipe = Pipeline().source("image_bytes", on_error="null")
        expr = pl.col("img").cv.pipe(pipe).sink("numpy")
        result = df.with_columns(out=expr)

        row0 = result["out"][0]
        assert row0 is not None
        assert row0["data"] is not None

        # Corrupt row produces struct with all-null fields
        row1 = result["out"][1]
        assert row1["data"] is None

    def test_backwards_compatible(self, create_test_png: Callable) -> None:
        """Omitting on_error keeps existing behaviour (raise on error)."""
        good = create_test_png(width=10, height=10)
        df = pl.DataFrame({"img": [good]})
        pipe = Pipeline().source("image_bytes")
        expr = pl.col("img").cv.pipe(pipe).sink("png")
        result = df.with_columns(out=expr)
        assert result["out"][0] is not None
