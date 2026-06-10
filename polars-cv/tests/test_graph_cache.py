"""
Cache-safety tests for the compiled-graph cache.

The Rust plugin caches the *compiled* form of a graph (parsed spec, topo
order, slot-bound params, pre-resolved static ops) keyed by the plugin
kwargs. The cache must contain nothing data-derived, so one cached graph has
to behave identically across calls with different row counts, image sizes,
dtypes, null patterns, and per-row dynamic parameters. These tests exercise
exactly those axes by running the *same* pipeline object repeatedly over
heterogeneous data.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required


def _png(width: int, height: int, value: int = 128) -> bytes:
    """An RGB u8 PNG of the given size."""
    from PIL import Image

    arr = np.full((height, width, 3), value, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_u16(width: int, height: int, value: int = 1000) -> bytes:
    """A single-channel 16-bit PNG of the given size."""
    from PIL import Image

    arr = np.full((height, width), value, dtype=np.uint16)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _shape_of(struct_val) -> list[int]:
    return list(numpy_from_struct(struct_val).shape)


@plugin_required
class TestCacheHeterogeneousData:
    """One pipeline (one cached graph) over data of varying shape/dtype."""

    def test_same_pipeline_different_dataframes(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale()
        df_small = pl.DataFrame({"img": [_png(8, 8)]})
        df_large = pl.DataFrame({"img": [_png(32, 16)]})

        out_small = df_small.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out_large = df_large.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        assert _shape_of(out_small["out"][0]) == [8, 8, 1]
        assert _shape_of(out_large["out"][0]) == [16, 32, 1]

    def test_varying_sizes_within_one_column(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale()
        df = pl.DataFrame({"img": [_png(4, 4), _png(10, 6), _png(7, 13)]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        assert _shape_of(out["out"][0]) == [4, 4, 1]
        assert _shape_of(out["out"][1]) == [6, 10, 1]
        assert _shape_of(out["out"][2]) == [13, 7, 1]

    def test_same_pipeline_u8_and_u16_images(self) -> None:
        # The cached graph must not bake in a decoded dtype: u8 and u16
        # sources through the same pipeline keep their respective dtypes.
        pipe = Pipeline().source("image_bytes")
        df = pl.DataFrame({"img": [_png(4, 4)]})
        out8 = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert numpy_from_struct(out8["out"][0]).dtype == np.uint8

        df16 = pl.DataFrame({"img": [_png_u16(4, 4)]})
        out16 = df16.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert numpy_from_struct(out16["out"][0]).dtype == np.uint16

    def test_repeated_calls_are_consistent(self) -> None:
        # Many calls through the cache must give byte-identical results.
        pipe = Pipeline().source("image_bytes").grayscale().resize(height=5, width=5)
        df = pl.DataFrame({"img": [_png(16, 16, value=200)]})
        first = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        ref = numpy_from_struct(first["out"][0])
        for _ in range(5):
            out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
            np.testing.assert_array_equal(numpy_from_struct(out["out"][0]), ref)


@plugin_required
class TestCacheNullSafety:
    """Null rows must stay per-row decisions, never cached."""

    def test_null_rows_produce_null_outputs(self) -> None:
        # Null rows through a numpy (struct) sink encode as structs whose
        # fields are all null.
        pipe = Pipeline().source("image_bytes").grayscale()
        df = pl.DataFrame(
            {"img": [_png(4, 4), None, _png(6, 6)]},
            schema={"img": pl.Binary},
        )
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None
        assert out["out"][2]["data"] is not None

    def test_all_null_then_all_valid_batches(self) -> None:
        # An all-null batch first must not poison the cached graph for a
        # following all-valid batch (and vice versa).
        pipe = Pipeline().source("image_bytes").grayscale()
        df_null = pl.DataFrame({"img": [None, None]}, schema={"img": pl.Binary})
        out_null = df_null.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert out_null["out"].struct.field("data").null_count() == 2

        df_ok = pl.DataFrame({"img": [_png(4, 4)]})
        out_ok = df_ok.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert out_ok["out"].struct.field("data").null_count() == 0


@plugin_required
class TestCacheDynamicParams:
    """Slot-bound expression params resolve per row through the cache."""

    def test_per_row_resize_dimensions(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("w"))
        )
        df = pl.DataFrame(
            {
                "img": [_png(16, 16), _png(16, 16), _png(16, 16)],
                "h": [4, 8, 2],
                "w": [6, 3, 9],
            }
        )
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert _shape_of(out["out"][0]) == [4, 6, 3]
        assert _shape_of(out["out"][1]) == [8, 3, 3]
        assert _shape_of(out["out"][2]) == [2, 9, 3]

    def test_aggregation_param_broadcasts(self) -> None:
        # An aggregation expression yields a one-element series that must
        # broadcast to every row.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h").max(), width=pl.col("h").max())
        )
        df = pl.DataFrame(
            {
                "img": [_png(16, 16), _png(8, 8)],
                "h": [3, 7],
            }
        )
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert _shape_of(out["out"][0]) == [7, 7, 3]
        assert _shape_of(out["out"][1]) == [7, 7, 3]

    def test_param_dtype_varies_between_calls(self) -> None:
        # The same cached graph must accept the param column arriving as
        # different numeric dtypes on different calls.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
        )
        img = _png(16, 16)
        for dtype in (pl.Int64, pl.Int32, pl.UInt16, pl.Float64):
            df = pl.DataFrame({"img": [img], "h": pl.Series([5], dtype=dtype)})
            out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
            assert _shape_of(out["out"][0]) == [5, 5, 3]


@plugin_required
class TestCacheIdentity:
    """Structurally similar but distinct graphs must never collide."""

    def test_same_structure_different_literals(self) -> None:
        pipe_a = Pipeline().source("image_bytes").resize(height=4, width=4)
        pipe_b = Pipeline().source("image_bytes").resize(height=9, width=9)
        df = pl.DataFrame({"img": [_png(16, 16)]})

        out_a = df.with_columns(out=pl.col("img").cv.pipe(pipe_a).sink("numpy"))
        out_b = df.with_columns(out=pl.col("img").cv.pipe(pipe_b).sink("numpy"))
        # Interleave again to exercise cache hits on both entries.
        out_a2 = df.with_columns(out=pl.col("img").cv.pipe(pipe_a).sink("numpy"))

        assert _shape_of(out_a["out"][0]) == [4, 4, 3]
        assert _shape_of(out_b["out"][0]) == [9, 9, 3]
        assert _shape_of(out_a2["out"][0]) == [4, 4, 3]

    def test_many_distinct_graphs_exceeding_cache_capacity(self) -> None:
        # More distinct graphs than the cache holds (32): eviction must be
        # invisible to correctness.
        df = pl.DataFrame({"img": [_png(64, 64)]})
        for size in range(2, 40):
            pipe = Pipeline().source("image_bytes").resize(height=size, width=size)
            out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
            assert _shape_of(out["out"][0]) == [size, size, 3]


@plugin_required
class TestCacheStreaming:
    """The streaming engine invokes the plugin per morsel, in parallel."""

    def test_streaming_engine_many_rows(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
            .grayscale()
        )
        n = 200
        rng = np.random.default_rng(0)
        heights = rng.integers(2, 12, size=n).tolist()
        df = pl.DataFrame(
            {
                "img": [_png(16, 16)] * n,
                "h": heights,
            }
        )
        out = (
            df.lazy()
            .with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
            .collect(engine="streaming")
        )
        assert out.height == n
        for i in (0, n // 2, n - 1):
            assert _shape_of(out["out"][i]) == [heights[i], heights[i], 1]

    def test_streaming_with_nulls(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale()
        imgs = [_png(4, 4) if i % 3 else None for i in range(90)]
        df = pl.DataFrame({"img": imgs}, schema={"img": pl.Binary})
        out = (
            df.lazy()
            .with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
            .collect(engine="streaming")
        )
        for i in range(90):
            if i % 3 == 0:
                assert out["out"][i]["data"] is None
            else:
                assert out["out"][i]["data"] is not None


@plugin_required
class TestGraphVersionField:
    """The graph JSON carries a wire-format version the plugin validates."""

    def test_graph_json_carries_version(self) -> None:
        import json

        from polars_cv._graph import PipelineGraph

        graph = PipelineGraph()
        pipe = Pipeline().source("image_bytes").grayscale()
        graph.add_node("n0", pipe, column=pl.col("img"))
        graph.set_output("n0", "numpy")
        spec = json.loads(graph._to_json())
        assert spec["version"] == 1

    def test_future_version_is_rejected(self) -> None:
        # A graph claiming a newer format version must error clearly rather
        # than be misparsed.
        import json

        from polars_cv._graph import PipelineGraph

        graph = PipelineGraph()
        pipe = Pipeline().source("image_bytes").grayscale()
        graph.add_node("n0", pipe, column=pl.col("img"))
        graph.set_output("n0", "numpy")
        spec = json.loads(graph._to_json())
        spec["version"] = 999

        from polars.plugins import register_plugin_function

        from polars_cv._graph import LIB_PATH

        expr = register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="vb_graph",
            args=[pl.col("img")],
            kwargs={"graph_json": json.dumps(spec), "expr_column_names": []},
            is_elementwise=True,
        )
        df = pl.DataFrame({"img": [_png(4, 4)]})
        with pytest.raises(pl.exceptions.ComputeError, match="version"):
            df.with_columns(out=expr)
