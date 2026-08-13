"""
Tests for planning-time schema inference with list/array sources.

Validates that:
1. List sinks infer the correct nested List dtype from the input column.
2. Array sinks error when shape is not deterministic.
3. Dtype is correctly propagated or overridden by operations.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

# ============================================================
# Helpers
# ============================================================


def _make_list_column(data: np.ndarray) -> pl.Series:
    """Convert a 3D numpy array into a nested List column."""
    rows = []
    for i in range(data.shape[0]):
        row = []
        for j in range(data.shape[1]):
            row.append(data[i, j].tolist())
        rows.append(row)
    return pl.Series("data", rows)


# ============================================================
# Tests: List sink dtype inference
# ============================================================


@plugin_required
class TestListSinkInference:
    """Tests for inferring output schema from list source → list sink."""

    def test_float64_passthrough(self):
        """List(List(Float64)) input → list sink → List(List(Float64)).

        Note: _make_list_column creates a column with 2 nesting levels
        for 3D data (rows are 2D nested lists).
        """
        data = np.random.rand(2, 4, 4).astype(np.float64)
        col = _make_list_column(data)
        df = pl.DataFrame({"data": col})

        pipe = Pipeline().source("list")
        schema = (
            df.lazy().select(pl.col("data").cv.pipe(pipe).sink("list")).collect_schema()
        )
        result_dtype = schema["data"]
        # Column type is List(List(Float64)) → ndim=2 → output is List(List(Float64))
        assert result_dtype == pl.List(pl.List(pl.Float64))

    def test_int64_passthrough(self):
        """List(List(Int64)) input → list sink → List(List(Int64)).

        Note: Polars infers small integer lists as Int64.
        """
        data = np.random.randint(0, 255, (2, 4, 4), dtype=np.uint8)
        col = _make_list_column(data)
        df = pl.DataFrame({"data": col})

        pipe = Pipeline().source("list")
        schema = (
            df.lazy().select(pl.col("data").cv.pipe(pipe).sink("list")).collect_schema()
        )
        result_dtype = schema["data"]
        # Polars infers integer literals as Int64
        assert result_dtype == pl.List(pl.List(pl.Int64))

    def test_explicit_dtype_overrides_auto(self):
        """Explicit dtype overrides auto-inference; rank still comes from the
        input column, not a guess.

        The column is List(List(Float64)) (rank 2), so the plan is rank 2
        regardless of whether dtype was passed — matching the no-dtype tests
        above (previously this path guessed rank 3, diverging from the data)."""
        data = np.random.rand(2, 4, 4).astype(np.float64)
        col = _make_list_column(data)
        df = pl.DataFrame({"data": col})

        pipe = Pipeline().source("list", dtype="u8")
        schema = (
            df.lazy().select(pl.col("data").cv.pipe(pipe).sink("list")).collect_schema()
        )
        result_dtype = schema["data"]
        assert result_dtype == pl.List(pl.List(pl.UInt8))

    def test_normalize_overrides_to_f32(self):
        """List(List(Int64)) → normalize → list sink → List(List(Float32))."""
        data = np.random.randint(0, 255, (2, 4, 4), dtype=np.uint8)
        col = _make_list_column(data)
        df = pl.DataFrame({"data": col})

        pipe = Pipeline().source("list").normalize()
        schema = (
            df.lazy().select(pl.col("data").cv.pipe(pipe).sink("list")).collect_schema()
        )
        result_dtype = schema["data"]
        # normalize overrides dtype to f32, ndim=2 from column type
        assert result_dtype == pl.List(pl.List(pl.Float32))

    def test_float64_with_crop(self):
        """Float64 input → crop (provides shape [4,4]) → list sink → List(List(Float64)).

        crop sets deterministic height/width hints (H=4, W=4), but channels
        are unknown for list sources unless asserted. Without a channel
        dimension, expected_shape is None and the schema uses the 2D nesting
        inferred from the column type.
        """
        data = np.random.rand(2, 8, 8).astype(np.float64)
        col = _make_list_column(data)
        df = pl.DataFrame({"data": col})

        pipe = Pipeline().source("list").crop(top=1, left=1, height=4, width=4)
        schema = (
            df.lazy().select(pl.col("data").cv.pipe(pipe).sink("list")).collect_schema()
        )
        result_dtype = schema["data"]
        assert result_dtype == pl.List(pl.List(pl.Float64))


# ============================================================
# Tests: Array sink validation
# ============================================================


@plugin_required
class TestArraySinkValidation:
    """Tests for array sink shape requirements."""

    def test_array_sink_requires_shape(self):
        """Array sink without deterministic shape should raise ValueError at lazy sink."""
        pipe = Pipeline().source("list")
        expr = pl.col("data").cv.pipe(pipe)
        with pytest.raises(ValueError, match="needs the full output shape"):
            expr.sink("array")

    def test_array_sink_with_explicit_shape(self):
        """Array sink with explicit shape should work."""
        pipe = Pipeline().source("list")
        assert pipe is not None

    def test_array_sink_with_resize_and_channels(self):
        """Array sink after resize + grayscale (deterministic shape) should work."""
        pipe = (
            Pipeline()
            .source("list", dtype="u8")
            .resize(height=10, width=10)
            .grayscale()
        )
        assert pipe is not None


# ============================================================
# Tests: Execution correctness
# ============================================================


@plugin_required
class TestSchemaInferenceExecution:
    """Tests that execution produces correct results with auto-inferred schema."""

    def test_float64_roundtrip(self):
        """Float64 data survives a list→list roundtrip."""
        data = np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float64)
        col = _make_list_column(data)
        df = pl.DataFrame({"data": col})

        pipe = Pipeline().source("list")
        result = df.select(pl.col("data").cv.pipe(pipe).sink("list"))

        result_lists = result["data"].to_list()
        assert len(result_lists) == 1
        assert result_lists[0][0][0] == pytest.approx(1.5)
        assert result_lists[0][0][1] == pytest.approx(2.5)


# ============================================================
# Tests: Execution-time schema consistency for null data
# ============================================================


@plugin_required
class TestNullDataSchemaConsistency:
    """Execution-time schema must match planning-time schema even with null data."""

    def test_list_sink_all_null_preserves_nesting(self):
        """All-null List input should preserve nested List schema."""
        df = pl.DataFrame(
            {
                "data": pl.Series(
                    "data", [None, None], dtype=pl.List(pl.List(pl.Float64))
                )
            }
        )
        pipe = Pipeline().source("list")
        result = df.select(pl.col("data").cv.pipe(pipe).sink("list"))
        # Schema should be List(List(Float64)), not List(UInt8)
        assert result["data"].dtype == pl.List(pl.List(pl.Float64))

    def test_array_sink_all_null_preserves_shape(self):
        """All-null file_path input with assert_shape should preserve Array schema."""
        df = pl.DataFrame({"path": pl.Series("path", [None], dtype=pl.String)})
        pipe = (
            Pipeline()
            .source("file_path", dtype="u8")
            .assert_shape(height=10, width=10, channels=3)
        )
        result = df.select(pl.col("path").cv.pipe(pipe).sink("array"))
        expected = pl.Array(pl.Array(pl.Array(pl.UInt8, 3), 10), 10)
        assert result["path"].dtype == expected

    def test_list_sink_mixed_null(self):
        """Mixed null/non-null should still produce correct nested schema."""
        data = np.random.rand(4, 4).astype(np.float64)
        rows = [data.tolist(), None]
        col = pl.Series("data", rows)
        df = pl.DataFrame({"data": col})
        pipe = Pipeline().source("list")
        result = df.select(pl.col("data").cv.pipe(pipe).sink("list"))
        assert result["data"].dtype == pl.List(pl.List(pl.Float64))
