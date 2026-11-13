"""`explain()` on a graph: the text the planner renders for what will run (CR-148).

`PipelineGraph.explain()` is rendered in Rust from the same planning
`check()` and the schema read, so these tests hold the text to the graph
(`sink(return_expr=False)`) and to `collect_schema()`, never to a copy of
the planner in Python.
"""

from __future__ import annotations

import re
import sys

import polars as pl
import pytest

from polars_cv import Pipeline, Sink
from tests.conftest import plugin_required

pytestmark = plugin_required

NODE_LINE = re.compile(r"^(n\d+): ", re.MULTILINE)


def _cutout():
    """The README's example (``scripts/make_readme_cutout.py``), up to the sink."""
    base = Pipeline().source("file_path", allowed_roots=["*"]).resize_max(512)
    img = pl.col("path").cv
    bright = img.pipe(base.grayscale().threshold(190))
    saturated = img.pipe(
        base.to_hsv().channel_select(1).threshold(pl.col("min_saturation"))
    )
    outline = bright.maximum(saturated).pipe(
        Pipeline().morphology_open(ksize=5).extract_contours().largest().convex_hull()
    )
    photo = img.pipe(base)
    alpha = outline.rasterize(shape=photo, supersample=4)
    cutout = photo.channel_merge(alpha).alias("cutout")
    return cutout.merge_pipe(outline.area().alias("area"))


CUTOUT_SINKS = {"cutout": "png", "area": "native"}


def _decodes(text: str) -> int:
    return len(re.findall(r"^n\d+: decode ", text, re.MULTILINE))


class TestNodes:
    @pytest.mark.parametrize("optimized", [True, False])
    def test_every_node_of_the_graph_once_in_execution_order(self, optimized):
        expr = _cutout()
        flags = None if optimized else False
        graph = expr.sink(CUTOUT_SINKS, return_expr=False, opt_flags=flags)
        text = expr.explain(CUTOUT_SINKS, optimized=optimized)
        assert len(NODE_LINE.findall(text)) == len(graph._nodes)
        # Execution order: a node is listed after every node it reads.
        listed = NODE_LINE.findall(text)
        for line in text.splitlines():
            m = re.match(r"^(n\d+): reads (n\d+)", line)
            if m:
                assert listed.index(m.group(2)) < listed.index(m.group(1))

    def test_the_optimised_cutout_decodes_and_runs_less(self):
        expr = _cutout()
        optimised = expr.explain(CUTOUT_SINKS)
        written = expr.explain(CUTOUT_SINKS, optimized=False)
        assert _decodes(optimised) < _decodes(written)
        assert len(NODE_LINE.findall(optimised)) < len(NODE_LINE.findall(written))

    def test_a_node_shows_its_planned_state_after_each_op(self):
        pipe = Pipeline().source("image_bytes", dtype="u8").resize(height=8, width=6)
        text = pl.col("img").cv.pipe(pipe.grayscale()).explain("tensor")
        assert re.search(
            r"\| resize\(.*height=8, width=6\) +-> buffer u8 \[8, 6, \?\]", text
        )
        assert re.search(r"grayscale\(.*\).*-> buffer u8 \[8, 6, 1\]", text)

    def test_a_per_row_parameter_is_shown_as_its_expression(self):
        pipe = Pipeline().source("image_bytes").resize(height=pl.col("h"), width=4)
        text = pl.col("img").cv.pipe(pipe).explain("tensor")
        assert 'height=col("h")' in text
        assert re.search(r"-> buffer \w+ \[\?, 4, \?\]", text)

    def test_a_root_names_its_column_and_source(self):
        pipe = Pipeline().source("image_bytes", dtype="u8")
        text = pl.col("img").cv.pipe(pipe).explain("tensor")
        assert re.search(r'^n0: decode col\("img"\) as image_bytes\(', text, re.M)


class TestSegments:
    def test_fused_buffer_ops_are_marked_and_a_domain_change_is_not(self):
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .grayscale()
            .threshold(128)
            .extract_contours()
        )
        lines = pl.col("img").cv.pipe(pipe).explain("native").splitlines()
        (gray,) = [ln for ln in lines if "grayscale(" in ln]
        (thresh,) = [ln for ln in lines if "threshold(" in ln]
        (contours,) = [ln for ln in lines if "extract_contours(" in ln]
        assert gray.lstrip().startswith("|") and thresh.lstrip().startswith("|")
        assert not contours.lstrip().startswith("|")

    def test_the_engine_passes_are_named_as_per_segment(self):
        from polars_cv._optimize import OPTIMIZATION_PASSES

        engine = [s.name for s in OPTIMIZATION_PASSES if s.tier == "engine"]
        pipe = Pipeline().source("image_bytes", dtype="u8").flip_h().flip_h()
        text = pl.col("img").cv.pipe(pipe).explain("tensor")
        line = next(ln for ln in text.splitlines() if ln.startswith("engine passes"))
        assert all(name in line for name in engine)
        written = pl.col("img").cv.pipe(pipe).explain("tensor", optimized=False)
        line = next(ln for ln in written.splitlines() if ln.startswith("engine passes"))
        assert not any(name in line for name in engine)


class TestOutputs:
    def test_every_output_and_its_dtype_as_collect_schema_plans_it(self):
        frame = pl.LazyFrame(
            schema={"path": pl.String, "min_saturation": pl.Float64},
        )
        expr = _cutout()
        schema = frame.select(out=expr.sink(CUTOUT_SINKS)).collect_schema()
        text = expr.explain(CUTOUT_SINKS)
        for field in schema["out"].fields:
            line = next(
                ln
                for ln in text.splitlines()
                if ln.startswith(f"output '{field.name}'")
            )
            assert line.endswith(f":: {field.dtype._string_repr()}"), line

    def test_a_single_output_is_its_sink_and_dtype(self):
        pipe = Pipeline().source("image_bytes", dtype="u8").resize(height=4, width=4)
        expr = pl.col("img").cv.pipe(pipe.grayscale())
        frame = pl.LazyFrame(schema={"img": pl.Binary})
        dtype = frame.select(expr.sink("array")).collect_schema().dtypes()[0]
        text = expr.explain("array")
        (line,) = [ln for ln in text.splitlines() if ln.startswith("output")]
        assert line.startswith("output: n0 as array")
        assert line.endswith(f":: {dtype._string_repr()}")

    def test_a_sink_with_settings_is_given_as_a_sink_value(self):
        pipe = Pipeline().source("image_bytes", dtype="u8")
        text = pl.col("img").cv.pipe(pipe).explain(Sink.jpeg(quality=70))
        assert "as jpeg(quality=70)" in text

    def test_an_output_the_column_decides_says_so(self):
        # A list source takes its element type from the column, not the plan.
        pipe = Pipeline().source("list")
        text = pl.col("x").cv.pipe(pipe).explain("list")
        (line,) = [ln for ln in text.splitlines() if ln.startswith("output")]
        assert line.endswith(":: from the input column")


class TestLazyState:
    def test_output_dtype_and_domain_are_the_nodes_planned_state(self):
        pipe = Pipeline().source("image_bytes", dtype="u8").cast("f32")
        expr = pl.col("img").cv.pipe(pipe)
        assert (expr.current_domain(), expr.output_dtype()) == ("buffer", "f32")
        contours = expr.pipe(Pipeline().threshold(0.5).extract_contours())
        assert contours.current_domain() == "contour"


def test_explain_needs_no_graphviz(monkeypatch):
    monkeypatch.setitem(sys.modules, "pydot", None)
    monkeypatch.setitem(sys.modules, "networkx", None)
    pipe = Pipeline().source("image_bytes", dtype="u8")
    assert pl.col("img").cv.pipe(pipe).explain("tensor").startswith("n0: ")
