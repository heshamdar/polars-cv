"""
Tests for Common Subexpression Elimination (CSE) optimization.

CSE automatically detects and deduplicates common operation prefixes
across pipelines, reducing redundant computation.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct
from tests._plan_view import ops_of
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a DataFrame with a test image."""
    # Create a simple test image with distinct regions
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    # Add a white square in the center for contour detection
    for x in range(25, 75):
        for y in range(25, 75):
            img.putpixel((x, y), (255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    return pl.DataFrame({"image": [img_bytes]})


# ============================================================
# Test Classes
# ============================================================


def _shared_prefix(a: Pipeline, b: Pipeline) -> int:
    """How many leading ops CSE shares between *a* and *b* on one column: the
    ops of the nodes both outputs' input chains pass through."""
    from polars_cv import OptFlags
    from polars_cv._graph import PipelineGraph

    graph = PipelineGraph()
    graph.add_node("a", a, pl.col("image"), alias="a")
    graph.add_node("b", b, pl.col("image"), alias="b")
    graph.set_multi_output({"a": "numpy", "b": "numpy"})
    graph.optimize(OptFlags.parse("common_subexpression_elimination"))

    def chain(alias: str) -> list[str]:
        node_id = graph._alias_to_node[alias]
        ids = [node_id]
        while (before := graph._nodes[ids[-1]].input) is not None:
            ids.append(before)
        return ids[::-1]

    shared = 0
    for x, y in zip(chain("a"), chain("b"), strict=False):
        if x != y:
            break
        shared += len(ops_of(graph._nodes[x].pipeline))
    return shared


@plugin_required
class TestCSEPrefixKey:
    """CSE shares an op iff it is the same op over the same graph inputs."""

    BASE = Pipeline().source("image_bytes")

    def test_identical_ops_are_shared(self) -> None:
        a = self.BASE.resize(width=100, height=100).grayscale()
        b = self.BASE.resize(width=100, height=100).threshold(128)
        assert _shared_prefix(a, b) == 1

    def test_different_params_are_not_shared(self) -> None:
        a = self.BASE.resize(width=100, height=100)
        b = self.BASE.resize(width=200, height=100)
        assert _shared_prefix(a, b) == 0

    def test_different_ops_are_not_shared(self) -> None:
        assert _shared_prefix(self.BASE.grayscale(), self.BASE.threshold(128)) == 0

    def test_the_same_expression_is_shared(self) -> None:
        a = self.BASE.resize(width=pl.col("w"), height=100).grayscale()
        b = self.BASE.resize(width=pl.col("w"), height=100).threshold(1)
        assert _shared_prefix(a, b) == 1

    def test_the_same_expression_at_different_local_slots_is_shared(self) -> None:
        # Each pipeline numbers its own expressions; CSE compares graph inputs.
        a = self.BASE.scale(pl.col("s")).resize(width=pl.col("w"), height=1)
        b = self.BASE.scale(pl.col("s")).resize(width=pl.col("w"), height=1)
        b2 = self.BASE.resize(width=pl.col("w"), height=1)
        assert _shared_prefix(a, b) == 2
        assert _shared_prefix(b2, a) == 0

    def test_different_expressions_are_not_shared(self) -> None:
        a = self.BASE.resize(width=pl.col("w1"), height=100)
        b = self.BASE.resize(width=pl.col("w2"), height=100)
        assert _shared_prefix(a, b) == 0


class TestCSEBasic:
    """Basic CSE optimization tests."""

    def test_shared_prefix_extracted(self, sample_df: pl.DataFrame) -> None:
        """Two pipelines with same prefix should share computation."""
        # Define two pipelines with common prefix: resize + grayscale
        gray_pipe = (
            Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        )

        threshold_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(128)
        )

        # Create lazy expressions
        gray = pl.col("image").cv.pipe(gray_pipe).alias("gray")
        thresh = pl.col("image").cv.pipe(threshold_pipe).alias("thresh")

        # Compose and sink
        result_expr = gray.merge_pipe(thresh).sink(
            {
                "gray": "numpy",
                "thresh": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        # Both outputs should be present and valid
        gray_arr = numpy_from_struct(result["outputs"].struct.field("gray")[0])
        thresh_arr = numpy_from_struct(result["outputs"].struct.field("thresh")[0])

        assert gray_arr.shape == (50, 50, 1)
        assert thresh_arr.shape == (50, 50, 1)

    def test_no_sharing_when_prefix_differs(self, sample_df: pl.DataFrame) -> None:
        """Pipelines with different prefixes should not share."""
        # Different resize dimensions = no common prefix
        pipe1 = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        pipe2 = (
            Pipeline().source("image_bytes").resize(width=100, height=100).grayscale()
        )

        expr1 = pl.col("image").cv.pipe(pipe1).alias("small")
        expr2 = pl.col("image").cv.pipe(pipe2).alias("large")

        result_expr = expr1.merge_pipe(expr2).sink(
            {
                "small": "numpy",
                "large": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        small = numpy_from_struct(result["outputs"].struct.field("small")[0])
        large = numpy_from_struct(result["outputs"].struct.field("large")[0])

        # Both should work, just with different sizes
        assert small.shape == (50, 50, 1)
        assert large.shape == (100, 100, 1)

    def test_partial_prefix_sharing(self, sample_df: pl.DataFrame) -> None:
        """Pipelines that diverge mid-way should share the common part."""
        # Common: resize + grayscale
        # Diverge: threshold vs blur
        thresh_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(128)
        )

        blur_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .blur(3)
        )

        thresh = pl.col("image").cv.pipe(thresh_pipe).alias("thresh")
        blur = pl.col("image").cv.pipe(blur_pipe).alias("blur")

        result_expr = thresh.merge_pipe(blur).sink(
            {
                "thresh": "numpy",
                "blur": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        thresh_arr = numpy_from_struct(result["outputs"].struct.field("thresh")[0])
        blur_arr = numpy_from_struct(result["outputs"].struct.field("blur")[0])

        assert thresh_arr.shape == (50, 50, 1)
        assert blur_arr.shape == (50, 50, 1)

        # Threshold and blur should produce different results
        # (threshold is binary, blur is smooth)
        assert not np.array_equal(thresh_arr, blur_arr)


class TestCSEMultiplePipelines:
    """Tests for CSE with more than two pipelines."""

    def test_three_pipelines_share_prefix(self, sample_df: pl.DataFrame) -> None:
        """Three pipelines with common prefix should all share."""
        # All three pipes share: source → resize → grayscale
        pipe_a = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(64)
        )
        pipe_b = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(128)
        )
        pipe_c = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(192)
        )

        a = pl.col("image").cv.pipe(pipe_a).alias("low")
        b = pl.col("image").cv.pipe(pipe_b).alias("mid")
        c = pl.col("image").cv.pipe(pipe_c).alias("high")

        result_expr = a.merge_pipe(b, c).sink(
            {
                "low": "numpy",
                "mid": "numpy",
                "high": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        low = numpy_from_struct(result["outputs"].struct.field("low")[0])
        mid = numpy_from_struct(result["outputs"].struct.field("mid")[0])
        high = numpy_from_struct(result["outputs"].struct.field("high")[0])

        # All should be valid 50x50 grayscale images
        assert low.shape == (50, 50, 1)
        assert mid.shape == (50, 50, 1)
        assert high.shape == (50, 50, 1)

        # Lower threshold = more white pixels
        assert np.sum(low) >= np.sum(mid) >= np.sum(high)


class TestCSEEdgeCases:
    """Edge cases for CSE optimization."""

    def test_single_pipeline_no_optimization(self, sample_df: pl.DataFrame) -> None:
        """Single pipeline should work without any optimization."""
        pipe = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()

        result = sample_df.with_columns(
            output=pl.col("image").cv.pipe(pipe).sink("numpy")
        )

        arr = numpy_from_struct(result["output"][0])
        assert arr.shape == (50, 50, 1)

    def test_identical_pipelines_fully_shared(self, sample_df: pl.DataFrame) -> None:
        """Two identical pipelines should fully share (prefix = entire pipeline)."""
        pipe1 = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        pipe2 = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()

        expr1 = pl.col("image").cv.pipe(pipe1).alias("a")
        expr2 = pl.col("image").cv.pipe(pipe2).alias("b")

        result_expr = expr1.merge_pipe(expr2).sink(
            {
                "a": "numpy",
                "b": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        a = numpy_from_struct(result["outputs"].struct.field("a")[0])
        b = numpy_from_struct(result["outputs"].struct.field("b")[0])

        # Both should be identical
        assert np.array_equal(a, b)

    def test_empty_prefix_no_sharing(self, sample_df: pl.DataFrame) -> None:
        """Pipelines with no common ops should not crash."""
        pipe1 = Pipeline().source("image_bytes").grayscale()
        pipe2 = Pipeline().source("image_bytes").resize(width=50, height=50)

        expr1 = pl.col("image").cv.pipe(pipe1).alias("gray")
        expr2 = pl.col("image").cv.pipe(pipe2).alias("resized")

        result_expr = expr1.merge_pipe(expr2).sink(
            {
                "gray": "numpy",
                "resized": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        gray = numpy_from_struct(result["outputs"].struct.field("gray")[0])
        resized = numpy_from_struct(result["outputs"].struct.field("resized")[0])

        # Gray should preserve original dimensions (100x100)
        assert gray.shape == (100, 100, 1)
        # Resized should be 50x50 RGB
        assert resized.shape == (50, 50, 3)

    def test_aliases_preserved_after_optimization(
        self, sample_df: pl.DataFrame
    ) -> None:
        """User aliases should be preserved through CSE optimization."""
        pipe1 = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        pipe2 = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(128)
        )

        # Both have meaningful aliases
        base = pl.col("image").cv.pipe(pipe1).alias("preprocessed")
        mask = pl.col("image").cv.pipe(pipe2).alias("binary_mask")

        result_expr = base.merge_pipe(mask).sink(
            {
                "preprocessed": "numpy",
                "binary_mask": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        # Aliases should be preserved in output
        assert "preprocessed" in result["outputs"].struct.fields
        assert "binary_mask" in result["outputs"].struct.fields


# ============================================================
# Full CSE: global value numbering over ops, then chain fusion
# ============================================================


def _physical(expr, fmt, opt_flags=True) -> dict:
    """The serialized graph ``expr.sink(fmt)`` runs."""
    return expr.sink(fmt, return_expr=False, opt_flags=opt_flags)._to_dict()


def _ops(graph: dict) -> list[str]:
    return [op["op"] for node in graph["nodes"].values() for op in node["ops"]]


def _roots(graph: dict) -> list[dict]:
    return [n for n in graph["nodes"].values() if "decode" in n["input"]]


def _decode(column: str = "image") -> object:
    return pl.col(column).cv.pipe(Pipeline().source("image_bytes", dtype="u8"))


class TestFullCSE:
    """Two nodes that compute the same thing from the same inputs are one
    node, whatever pipelines and chains they were written as; a linear run
    no one else reads is one node."""

    BASE = Pipeline().source("image_bytes", dtype="u8")

    def _three_branches(self):
        img = pl.col("image").cv
        a = img.pipe(self.BASE.grayscale()).alias("a")
        b = img.pipe(self.BASE.blur(1.0)).alias("b")
        c = img.pipe(self.BASE).alias("c")
        return a.merge_pipe(b, c), {"a": "numpy", "b": "numpy", "c": "numpy"}

    def test_pipelines_with_no_common_op_share_their_decode(self) -> None:
        expr, fmt = self._three_branches()
        assert len(_roots(_physical(expr, fmt))) == 1
        assert len(_roots(_physical(expr, fmt, opt_flags=False))) == 3

    def test_a_node_that_only_passes_its_input_on_vanishes(self) -> None:
        gray = _decode().pipe(Pipeline().grayscale())
        passthrough = gray.pipe(Pipeline()).alias("same")
        graph = _physical(passthrough, {"same": "numpy"})
        assert all(n["ops"] or "decode" in n["input"] for n in graph["nodes"].values())
        assert len(graph["nodes"]) == 1

    def test_a_prefix_is_shared_by_the_pipelines_that_have_it(self) -> None:
        img = pl.col("image").cv
        small = self.BASE.resize(height=8, width=8)
        a = img.pipe(small.grayscale()).alias("a")
        b = img.pipe(small.blur(1.0)).alias("b")
        c = img.pipe(self.BASE.flip_h()).alias("c")
        fmt = {"a": "numpy", "b": "numpy", "c": "numpy"}
        graph = _physical(a.merge_pipe(b, c), fmt)
        # `c` shares only the decode, which no longer stops `a` and `b`
        # sharing their resize.
        assert _ops(graph).count("resize") == 1
        assert len(_roots(graph)) == 1

    def test_siblings_below_a_node_share_their_prefix(self) -> None:
        gray = _decode().pipe(Pipeline().grayscale())
        x = gray.pipe(Pipeline().blur(1.0).threshold(10)).alias("x")
        y = gray.pipe(Pipeline().blur(1.0).canny()).alias("y")
        graph = _physical(x.merge_pipe(y), {"x": "numpy", "y": "numpy"})
        assert _ops(graph).count("blur") == 1

    def test_duplicates_merge_all_the_way_down(self) -> None:
        def branch(alias: str):
            return (
                _decode()
                .pipe(Pipeline().grayscale())
                .pipe(Pipeline().blur(1.0))
                .alias(alias)
            )

        a, b = branch("a"), branch("b")
        graph = _physical(a.merge_pipe(b), {"a": "numpy", "b": "png"})
        assert graph["outputs"]["a"]["node"] == graph["outputs"]["b"]["node"]
        assert sorted(_ops(graph)) == ["blur", "grayscale"]

    def test_an_op_reading_duplicate_operands_is_shared(self) -> None:
        def brightest(alias: str):
            img = _decode()
            gray = img.pipe(Pipeline().grayscale())
            blurred = img.pipe(Pipeline().blur(1.0).grayscale())
            return gray.maximum(blurred).alias(alias)

        a, b = brightest("a"), brightest("b")
        graph = _physical(a.merge_pipe(b), {"a": "numpy", "b": "numpy"})
        assert _ops(graph).count("maximum") == 1
        assert _ops(graph).count("grayscale") == 2

    def test_different_inputs_are_kept_apart(self) -> None:
        img = pl.col("image").cv
        a = img.pipe(self.BASE.resize(height=8, width=pl.col("w1"))).alias("a")
        b = img.pipe(self.BASE.resize(height=8, width=pl.col("w2"))).alias("b")
        c = pl.col("other").cv.pipe(self.BASE.resize(height=8, width=8)).alias("c")
        d = img.pipe(Pipeline().source("image_bytes").resize(height=8, width=8))
        fmt = {"a": "numpy", "b": "numpy", "c": "numpy", "d": "numpy"}
        graph = _physical(a.merge_pipe(b, c, d.alias("d")), fmt)
        assert _ops(graph).count("resize") == 4
        # The u8 and the auto-dtype decode of `image`, and `other`'s.
        assert len(_roots(graph)) == 3

    def test_a_run_nothing_else_reads_is_one_node(self) -> None:
        chained = _decode().grayscale().blur(1.0).threshold(10)
        graph = _physical(chained, "numpy")
        assert len(graph["nodes"]) == 1
        assert _ops(graph) == ["grayscale", "blur", "threshold"]
        assert len(_physical(chained, "numpy", opt_flags=False)["nodes"]) == 4

    def test_outputs_and_operands_stay_nodes(self) -> None:
        gray = _decode().grayscale().alias("gray")
        edges = gray.blur(1.0).canny()
        both = edges.maximum(gray.threshold(100)).alias("both")
        graph = _physical(both.merge_pipe(gray), {"both": "numpy", "gray": "numpy"})
        nodes = graph["nodes"]
        gray_node = graph["outputs"]["gray"]["node"]
        assert [op["op"] for op in nodes[gray_node]["ops"]] == ["grayscale"]
        (maximum,) = (n for n in nodes.values() if "maximum" in _ops({"nodes": {0: n}}))
        operand = next(op["other"] for op in maximum["ops"] if op["op"] == "maximum")
        assert [op["op"] for op in nodes[operand]["ops"]] == ["threshold"]
        # blur → canny → maximum: one node, reading the gray node.
        assert [op["op"] for op in maximum["ops"]] == ["blur", "canny", "maximum"]

    def test_with_cse_off_the_graph_is_as_written(self) -> None:
        expr, fmt = self._three_branches()
        graph = _physical(expr, fmt, opt_flags=False)
        assert len(graph["nodes"]) == 3


@plugin_required
class TestFullCSEIsByteIdentical:
    """Every output is the same, byte for byte, with CSE on and off."""

    @pytest.fixture
    def df(self, sample_df: pl.DataFrame) -> pl.DataFrame:
        return sample_df.with_columns(t=pl.lit(120, dtype=pl.UInt8))

    def test_a_branching_and_rejoining_graph(self, df: pl.DataFrame) -> None:
        base = Pipeline().source("image_bytes", dtype="u8").resize_max(64)
        img = pl.col("image").cv
        bright = img.pipe(base.grayscale().channel_select(0).threshold(pl.col("t")))
        dark = img.pipe(base.grayscale().channel_select(0).threshold(40))
        outline = bright.maximum(dark).pipe(
            Pipeline().extract_contours().largest().convex_hull()
        )
        photo = img.pipe(base)
        alpha = outline.rasterize(shape=photo, supersample=2).channel_select(0)
        r, g, b = (photo.channel_select(i) for i in range(3))
        cut = r.channel_merge(g, b, alpha).alias("cut")
        expr = cut.merge_pipe(outline.area().alias("area"), photo.alias("photo"))
        fmt = {"cut": "png", "area": "native", "photo": "numpy"}
        on = df.select(o=expr.sink(fmt, opt_flags=True)).unnest("o")
        off = df.select(o=expr.sink(fmt, opt_flags=False)).unnest("o")
        assert on["cut"].item() == off["cut"].item()
        assert on["area"].to_list() == off["area"].to_list()
        np.testing.assert_array_equal(
            numpy_from_struct(on["photo"].item()),
            numpy_from_struct(off["photo"].item()),
        )
