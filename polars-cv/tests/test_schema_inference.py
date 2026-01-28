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
        """When dtype is explicitly given, it overrides auto-inference."""
        data = np.random.rand(2, 4, 4).astype(np.float64)
        col = _make_list_column(data)
        df = pl.DataFrame({"data": col})

        pipe = Pipeline().source("list", dtype="u8")
        schema = (
            df.lazy().select(pl.col("data").cv.pipe(pipe).sink("list")).collect_schema()
        )
        result_dtype = schema["data"]
        assert result_dtype == pl.List(pl.List(pl.List(pl.UInt8)))

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
        """Float64 input → crop (provides shape [4,4,3]) → list sink → List(List(List(Float64))).

        crop sets deterministic shape hints (H=4, W=4, C=3 default),
        which gives expected_shape=[4,4,3] → 3 nesting levels.
        """
        data = np.random.rand(2, 8, 8).astype(np.float64)
        col = _make_list_column(data)
        df = pl.DataFrame({"data": col})

        pipe = Pipeline().source("list").crop(top=1, left=1, height=4, width=4)
        schema = (
            df.lazy().select(pl.col("data").cv.pipe(pipe).sink("list")).collect_schema()
        )
        result_dtype = schema["data"]
        assert result_dtype == pl.List(pl.List(pl.List(pl.Float64)))


# ============================================================
# Tests: Array sink validation
# ============================================================


class TestArraySinkValidation:
    """Tests for array sink shape requirements."""

    def test_array_sink_requires_shape(self):
        """Array sink without deterministic shape should raise ValueError."""
        pipe = Pipeline().source("list")
        with pytest.raises(ValueError, match="shape is required"):
            pipe.sink("array")

    def test_array_sink_with_explicit_shape(self):
        """Array sink with explicit shape should work."""
        pipe = Pipeline().source("list").sink("array", shape=[4, 4, 3])
        assert pipe is not None

    def test_array_sink_with_resize_and_channels(self):
        """Array sink after resize + grayscale (deterministic shape) should work."""
        pipe = (
            Pipeline()
            .source("list", dtype="u8")
            .resize(height=10, width=10)
            .grayscale()
            .sink("array")
        )
        assert pipe is not None


# ============================================================
# Tests: Execution correctness
# ============================================================


class TestSchemaInferenceExecution:
    """Tests that execution produces correct results with auto-inferred schema."""

    def test_float64_roundtrip(self):
        """Float64 data survives a list→list roundtrip."""
        data = np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float64)
        col = _make_list_column(data)
        df = pl.DataFrame({"data": col})

        pipe = Pipeline().source("list").sink("list")
        result = df.select(pl.col("data").cv.pipeline(pipe))

        result_lists = result["data"].to_list()
        assert len(result_lists) == 1
        assert result_lists[0][0][0] == pytest.approx(1.5)
        assert result_lists[0][0][1] == pytest.approx(2.5)
