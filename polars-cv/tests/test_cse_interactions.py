"""CSE where it meets the rest of the system.

``test_cse_optimization.py`` pins what CSE shares; this module pins how it
behaves alongside everything else that shapes a graph: the node-scope passes
(which can make two pipelines equal only after they run), per-row expression
parameters and their nulls, the per-row error policy, several input columns,
every DAG shape (forks, rejoins, an op reading its own input, aliased
interiors), determinism of the wire, streaming execution, and a property over
random graphs.

Two kinds of claim run through it: **structure** (what the optimised graph
is: which values are one node) and **equivalence** (every output, byte for
byte, is what the graph as written computes: ``opt_flags=False``).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest
from hypothesis import example
from hypothesis import strategies as st

from polars_cv import LazyPipelineExpr, OptFlags, Pipeline, numpy_from_struct
from tests.conftest import plugin_required
from tests.parity.framework.budget import property_lanes

U8 = Pipeline().source("image_bytes", dtype="u8")


def _physical(expr: LazyPipelineExpr, fmt: Any, opt_flags: Any = True) -> dict:
    """The serialized graph ``expr.sink(fmt)`` runs."""
    return expr.sink(fmt, return_expr=False, opt_flags=opt_flags)._to_dict()


def _ops(graph: dict) -> list[str]:
    return [op["op"] for node in graph["nodes"].values() for op in node["ops"]]


def _assert_same_outputs(df: pl.DataFrame, expr: LazyPipelineExpr, fmt: dict) -> None:
    """Every output, with CSE on and off (and with every pass off), equal."""
    runs = {
        name: df.select(o=expr.sink(fmt, opt_flags=flags)).unnest("o")
        for name, flags in (
            ("all", True),
            ("none", False),
            ("no-cse", OptFlags.parse("all,-common_subexpression_elimination")),
        )
    }
    reference = runs.pop("none")
    for name, frame in runs.items():
        for alias in fmt:
            assert frame[alias].to_list() == reference[alias].to_list(), (name, alias)


def _images(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (6, 7, 3), dtype=np.uint8) for _ in range(n)]


@pytest.fixture
def df(encode_png: Callable) -> pl.DataFrame:
    a, b, c = _images(3)
    return pl.DataFrame(
        {
            "image": [encode_png(a), encode_png(b), encode_png(c)],
            "other": [encode_png(c), encode_png(a), encode_png(b)],
            "w": pl.Series([4, 5, 6], dtype=pl.UInt32),
            "t": pl.Series([60, None, 200], dtype=pl.UInt8),
        }
    )


# ---------------------------------------------------------------------------
# With the node-scope passes
# ---------------------------------------------------------------------------


class TestWithNodePasses:
    """A node pass can make two pipelines equal; CSE must then share them.
    The logical passes repeat until the graph stops changing."""

    def test_a_no_op_removed_by_identity_elimination(self) -> None:
        img = pl.col("image").cv
        a = img.pipe(U8.cast("u8").grayscale()).alias("a")
        b = img.pipe(U8.grayscale()).alias("b")
        graph = _physical(a.merge_pipe(b), {"a": "numpy", "b": "numpy"})
        assert graph["outputs"]["a"]["node"] == graph["outputs"]["b"]["node"]
        assert _ops(graph).count("grayscale") == 1

    def test_an_order_the_pushdown_normalises(self) -> None:
        img = pl.col("image").cv
        crop = {"top": 0, "left": 0, "height": 4, "width": 4}
        a = img.pipe(U8.invert().crop(**crop)).alias("a")
        b = img.pipe(U8.crop(**crop).invert()).alias("b")
        graph = _physical(a.merge_pipe(b), {"a": "numpy", "b": "numpy"})
        assert graph["outputs"]["a"]["node"] == graph["outputs"]["b"]["node"]

    def test_a_pass_off_keeps_its_duplicates_apart(self) -> None:
        img = pl.col("image").cv
        a = img.pipe(U8.cast("u8").grayscale()).alias("a")
        b = img.pipe(U8.grayscale()).alias("b")
        flags = OptFlags.parse("all,-identity_elimination")
        graph = _physical(a.merge_pipe(b), {"a": "numpy", "b": "numpy"}, flags)
        assert graph["outputs"]["a"]["node"] != graph["outputs"]["b"]["node"]

    @plugin_required
    def test_outputs_are_unchanged(self, df: pl.DataFrame) -> None:
        img = pl.col("image").cv
        crop = {"top": 1, "left": 1, "height": 4, "width": 4}
        a = img.pipe(U8.cast("u8").invert().crop(**crop)).alias("a")
        b = img.pipe(U8.crop(**crop).invert()).alias("b")
        _assert_same_outputs(df, a.merge_pipe(b), {"a": "numpy", "b": "png"})


# ---------------------------------------------------------------------------
# With per-row parameters
# ---------------------------------------------------------------------------


class TestWithPerRowParameters:
    def test_an_expression_built_twice_is_one_input(self) -> None:
        img = pl.col("image").cv
        a = img.pipe(U8.resize(height=4, width=pl.col("w")).invert()).alias("a")
        b = img.pipe(U8.resize(height=4, width=pl.col("w")).flip_h()).alias("b")
        graph = _physical(a.merge_pipe(b), {"a": "numpy", "b": "numpy"})
        assert _ops(graph).count("resize") == 1

    def test_a_literal_and_an_expression_are_not_one_op(self) -> None:
        img = pl.col("image").cv
        a = img.pipe(U8.resize(height=4, width=4)).alias("a")
        b = img.pipe(U8.resize(height=4, width=pl.col("w"))).alias("b")
        graph = _physical(a.merge_pipe(b), {"a": "numpy", "b": "numpy"})
        assert _ops(graph).count("resize") == 2

    @plugin_required
    def test_per_row_values_reach_every_branch(self, df: pl.DataFrame) -> None:
        img = pl.col("image").cv
        a = img.pipe(U8.resize(height=4, width=pl.col("w")).invert()).alias("a")
        b = img.pipe(U8.resize(height=4, width=pl.col("w")).flip_h()).alias("b")
        _assert_same_outputs(df, a.merge_pipe(b), {"a": "numpy", "b": "numpy"})

    @plugin_required
    def test_a_null_parameter_in_a_shared_prefix_nulls_every_branch(
        self, df: pl.DataFrame
    ) -> None:
        img = pl.col("image").cv
        base = U8.on_null_param("null").grayscale().threshold(pl.col("t"))
        a = img.pipe(base.invert()).alias("a")
        b = img.pipe(base.flip_h()).alias("b")
        fmt = {"a": "numpy", "b": "numpy"}
        out = df.select(o=a.merge_pipe(b).sink(fmt)).unnest("o")
        assert out["a"].is_null().to_list() == [False, True, False]
        assert out["b"].is_null().to_list() == [False, True, False]
        _assert_same_outputs(df, a.merge_pipe(b), fmt)

    @plugin_required
    def test_a_policy_set_on_one_branch_holds_after_merging(
        self, df: pl.DataFrame
    ) -> None:
        img = pl.col("image").cv
        a = img.pipe(U8.on_null_param("null").grayscale().threshold(pl.col("t")))
        b = img.pipe(U8.grayscale().invert())
        fmt = {"a": "numpy", "b": "numpy"}
        expr = a.alias("a").merge_pipe(b.alias("b"))
        out = df.select(o=expr.sink(fmt)).unnest("o")
        assert out["a"].is_null().to_list() == [False, True, False]
        assert out["b"].is_null().to_list() == [False, False, False]
        _assert_same_outputs(df, expr, fmt)


# ---------------------------------------------------------------------------
# With the per-row error policy
# ---------------------------------------------------------------------------


@plugin_required
class TestWithErrorPolicy:
    def test_a_failing_shared_decode_nulls_every_output(self, df: pl.DataFrame) -> None:
        bad = df.with_columns(
            image=pl.when(pl.int_range(pl.len()) == 1)
            .then(pl.lit(b"not an image"))
            .otherwise(pl.col("image"))
        )
        img = pl.col("image").cv
        base = U8.on_error("null")
        a = img.pipe(base.grayscale()).alias("a")
        b = img.pipe(base.flip_v()).alias("b")
        fmt = {"a": "numpy", "b": "numpy"}
        out = bad.select(o=a.merge_pipe(b).sink(fmt)).unnest("o")
        assert out["a"].is_null().to_list() == [False, True, False]
        assert out["b"].is_null().to_list() == [False, True, False]
        _assert_same_outputs(bad, a.merge_pipe(b), fmt)


# ---------------------------------------------------------------------------
# Across columns
# ---------------------------------------------------------------------------


class TestAcrossColumns:
    def test_the_same_pipeline_on_two_columns_is_two_values(self) -> None:
        a = pl.col("image").cv.pipe(U8.grayscale()).alias("a")
        b = pl.col("other").cv.pipe(U8.grayscale()).alias("b")
        graph = _physical(a.merge_pipe(b), {"a": "numpy", "b": "numpy"})
        assert len([n for n in graph["nodes"].values() if "decode" in n["input"]]) == 2
        assert _ops(graph).count("grayscale") == 2

    @plugin_required
    def test_an_operand_from_another_column(self, df: pl.DataFrame) -> None:
        def brightest(alias: str) -> LazyPipelineExpr:
            a = pl.col("image").cv.pipe(U8.grayscale())
            b = pl.col("other").cv.pipe(U8.grayscale())
            return a.maximum(b).alias(alias)

        expr = brightest("x").merge_pipe(brightest("y"))
        fmt = {"x": "numpy", "y": "numpy"}
        graph = _physical(expr, fmt)
        assert graph["outputs"]["x"]["node"] == graph["outputs"]["y"]["node"]
        assert _ops(graph).count("maximum") == 1
        _assert_same_outputs(df, expr, fmt)


# ---------------------------------------------------------------------------
# DAG shapes
# ---------------------------------------------------------------------------


@plugin_required
class TestDagShapes:
    def test_a_diamond(self, df: pl.DataFrame) -> None:
        gray = pl.col("image").cv.pipe(U8.grayscale())
        left = gray.pipe(Pipeline().blur(1.0))
        right = gray.pipe(Pipeline().invert())
        expr = left.maximum(right).alias("out")
        graph = _physical(expr, {"out": "numpy"})
        # The fork at `gray` keeps it a node; `left` runs into the maximum,
        # `right` (an operand) is a node of its own.
        assert len(graph["nodes"]) == 3
        _assert_same_outputs(df, expr, {"out": "numpy"})

    def test_an_op_reading_its_own_input(self, df: pl.DataFrame) -> None:
        gray = pl.col("image").cv.pipe(U8.grayscale())
        expr = gray.maximum(gray).alias("out")
        _assert_same_outputs(df, expr, {"out": "numpy"})

    def test_an_aliased_interior_stays_addressable(self, df: pl.DataFrame) -> None:
        gray = pl.col("image").cv.pipe(U8.grayscale()).alias("gray")
        edges = gray.pipe(Pipeline().blur(1.0).canny()).alias("edges")
        fmt = {"gray": "numpy", "edges": "png"}
        graph = _physical(edges.merge_pipe(gray), fmt)
        nodes = graph["nodes"]
        assert [op["op"] for op in nodes[graph["outputs"]["gray"]["node"]]["ops"]] == [
            "grayscale"
        ]
        _assert_same_outputs(df, edges.merge_pipe(gray), fmt)

    def test_one_value_in_two_formats(self, df: pl.DataFrame) -> None:
        gray = pl.col("image").cv.pipe(U8.grayscale())
        twin = pl.col("image").cv.pipe(U8.grayscale())
        expr = gray.alias("raw").merge_pipe(twin.alias("png"))
        fmt = {"raw": "numpy", "png": "png"}
        graph = _physical(expr, fmt)
        assert graph["outputs"]["raw"]["node"] == graph["outputs"]["png"]["node"]
        _assert_same_outputs(df, expr, fmt)

    def test_a_single_output_merged_with_its_twin(self, df: pl.DataFrame) -> None:
        a = pl.col("image").cv.pipe(U8.invert())
        b = pl.col("image").cv.pipe(U8.invert())
        expr = a.merge_pipe(b)
        assert len(_physical(expr, "numpy")["nodes"]) == 1
        on = df.select(o=expr.sink("numpy")).to_series().to_list()
        off = df.select(o=expr.sink("numpy", opt_flags=False)).to_series().to_list()
        assert [numpy_from_struct(x).tobytes() for x in on] == [
            numpy_from_struct(x).tobytes() for x in off
        ]

    def test_a_contour_branch_and_a_buffer_branch(self, df: pl.DataFrame) -> None:
        img = pl.col("image").cv
        mask = img.pipe(U8.grayscale().threshold(120))
        hull = mask.pipe(Pipeline().extract_contours().largest().convex_hull())
        photo = img.pipe(U8)
        cut = photo.apply_contour_mask(hull).alias("cut")
        rgba = photo.channel_merge(hull.rasterize(shape=photo)).alias("rgba")
        expr = cut.merge_pipe(rgba, hull.area().alias("area"))
        fmt = {"cut": "png", "rgba": "png", "area": "native"}
        assert (
            len(
                [
                    n
                    for n in _physical(expr, fmt)["nodes"].values()
                    if "decode" in n["input"]
                ]
            )
            == 1
        )
        _assert_same_outputs(df, expr, fmt)


# ---------------------------------------------------------------------------
# Determinism, idempotence and streaming
# ---------------------------------------------------------------------------


def _branching() -> tuple[LazyPipelineExpr, dict]:
    img = pl.col("image").cv
    small = U8.resize(height=4, width=pl.col("w"))
    a = img.pipe(small.grayscale()).alias("a")
    b = img.pipe(small.invert()).alias("b")
    c = img.pipe(U8.flip_h()).alias("c")
    return a.merge_pipe(b, c), {"a": "numpy", "b": "numpy", "c": "numpy"}


class TestDeterminism:
    def test_the_same_expression_built_twice_serializes_the_same(self) -> None:
        """Polars' own CSE and the compiled-graph cache key on the JSON."""
        (x, fmt), (y, _) = _branching(), _branching()
        jx = x.sink(fmt, return_expr=False)._to_json()
        jy = y.sink(fmt, return_expr=False)._to_json()
        assert jx == jy

    def test_optimizing_twice_changes_nothing(self) -> None:
        expr, fmt = _branching()
        graph = expr.sink(fmt, return_expr=False)
        once = graph._to_json()
        graph.optimize(OptFlags.all())
        assert graph._to_json() == once


@plugin_required
def test_streaming_matches_in_memory(df: pl.DataFrame) -> None:
    expr, fmt = _branching()
    big = pl.concat([df] * 40)
    eager = big.select(o=expr.sink(fmt)).unnest("o")
    streamed = (
        big.lazy().select(o=expr.sink(fmt)).unnest("o").collect(engine="streaming")
    )
    for alias in fmt:
        assert streamed[alias].to_list() == eager[alias].to_list(), alias


# ---------------------------------------------------------------------------
# Random graphs
# ---------------------------------------------------------------------------

#: Ops that keep a u8 image's dtype, by the shape class they keep or make.
_RGB_OPS = ("invert", "flip_h", "flip_v", "blur", "channel_swap")
_GRAY_OPS = ("invert", "flip_h", "blur", "threshold")


def _apply(pipe: Pipeline, op: str) -> Pipeline:
    return {
        "invert": lambda: pipe.invert(),
        "flip_h": lambda: pipe.flip_h(),
        "flip_v": lambda: pipe.flip_v(),
        "blur": lambda: pipe.blur(1.0),
        "channel_swap": lambda: pipe.channel_swap([2, 1, 0]),
        "threshold": lambda: pipe.threshold(100),
        "grayscale": lambda: pipe.grayscale(),
        "channel_select": lambda: pipe.channel_select(1),
    }[op]()


@st.composite
def _programs(draw: st.DrawFn) -> tuple[list[tuple], list[int]]:
    """A random program and the steps it outputs: each step a decode, ops on
    an earlier step (one ``pipe`` of a run, or one lazy call per op), a
    maximum of two earlier steps of one shape, or a fresh rebuild of an
    earlier step."""
    steps: list[tuple] = [("decode",)]
    shapes = ["rgb"]
    for _ in range(draw(st.integers(2, 9), label="steps")):
        kind = draw(st.sampled_from(["ops", "ops", "maximum", "rebuild", "decode"]))
        if kind == "decode":
            steps.append(("decode",))
            shapes.append("rgb")
        elif kind == "rebuild":
            i = draw(st.integers(0, len(steps) - 1))
            steps.append(("rebuild", i))
            shapes.append(shapes[i])
        elif kind == "maximum":
            i = draw(st.integers(0, len(steps) - 1))
            same = [j for j, s in enumerate(shapes) if s == shapes[i]]
            j = draw(st.sampled_from(same))
            steps.append(("maximum", i, j))
            shapes.append(shapes[i])
        else:
            i = draw(st.integers(0, len(steps) - 1))
            shape = shapes[i]
            ops = []
            for _ in range(draw(st.integers(1, 3))):
                if shape == "rgb" and draw(st.booleans()) and draw(st.booleans()):
                    ops.append(draw(st.sampled_from(["grayscale", "channel_select"])))
                    shape = "gray"
                else:
                    ops.append(
                        draw(st.sampled_from(_RGB_OPS if shape == "rgb" else _GRAY_OPS))
                    )
            steps.append(("ops", i, tuple(ops), draw(st.booleans())))
            shapes.append(shape)
    outputs = draw(
        st.lists(st.integers(0, len(steps) - 1), min_size=1, max_size=3, unique=True),
        label="outputs",
    )
    return steps, outputs


def _build(steps: list[tuple], at: int) -> LazyPipelineExpr:
    """Step ``at`` built from scratch: fresh objects, so a rebuilt step and
    its original are equal only by what they compute."""
    step = steps[at]
    if step[0] == "decode":
        return pl.col("image").cv.pipe(U8)
    if step[0] == "rebuild":
        return _build(steps, step[1])
    if step[0] == "maximum":
        return _build(steps, step[1]).maximum(_build(steps, step[2]))
    _, parent, ops, one_pipe = step
    expr = _build(steps, parent)
    if one_pipe:
        pipe = Pipeline()
        for op in ops:
            pipe = _apply(pipe, op)
        return expr.pipe(pipe)
    for op in ops:
        expr = expr.pipe(_apply(Pipeline(), op))
    return expr


def _operands(graph: dict, node: dict) -> list[str]:
    """The node ids a wire node's ops name (an operand: ``other``, ``mask``,
    ``size``, ``others``)."""

    def named(value: Any) -> list[str]:
        if isinstance(value, str) and value in graph["nodes"]:
            return [value]
        if isinstance(value, list):
            return [n for v in value for n in named(v)]
        return []

    return [
        n for op in node["ops"] for k, v in op.items() if k != "op" for n in named(v)
    ]


def _wire_dependencies(graph: dict, node: dict) -> list[str]:
    """What a wire node depends on: the node it reads, then its operands."""
    reads = [node["input"]["node"]] if "node" in node["input"] else []
    return reads + _operands(graph, node)


def _assert_fully_shared(graph: dict) -> None:
    """The optimised graph's invariants: no two nodes compute the same thing,
    no node only passes its input on, and every node read as the input of
    exactly one other node, and nothing else, has been fused into it."""
    specs = [
        json.dumps([n["input"], n["ops"]], sort_keys=True)
        for n in graph["nodes"].values()
    ]
    assert len(specs) == len(set(specs)), "two nodes compute one value"
    for node in graph["nodes"].values():
        assert node["ops"] or "decode" in node["input"], "a pass-through node survived"
    outputs = {o["node"] for o in graph["outputs"].values()}
    first_reads: dict[str, int] = {}
    operands: set[str] = set()
    for node in graph["nodes"].values():
        if "node" in node["input"]:
            read = node["input"]["node"]
            first_reads[read] = first_reads.get(read, 0) + 1
        operands.update(_operands(graph, node))
    for wire_id in graph["nodes"]:
        if first_reads.get(wire_id) == 1:
            assert wire_id in outputs or wire_id in operands, f"{wire_id} was not fused"


@plugin_required
@property_lanes(weight=3, program=_programs())
# Found by the deep lane: a node that is both an op's input and its operand.
@example(program=([("decode",), ("maximum", 0, 0), ("maximum", 0, 0)], [1]))
def test_random_graphs_share_fully_and_compute_the_same(
    program: tuple[list[tuple], list[int]],
) -> None:
    steps, picks = program
    expr = _build(steps, picks[0]).alias("o0")
    others = [_build(steps, i).alias(f"o{k}") for k, i in enumerate(picks[1:], 1)]
    if others:
        expr = expr.merge_pipe(*others)
    fmt = {f"o{k}": "numpy" for k in range(len(picks))}

    optimised = _physical(expr, fmt)
    _assert_fully_shared(optimised)
    assert len(optimised["nodes"]) <= len(
        _physical(expr, fmt, opt_flags=False)["nodes"]
    )

    images = _images(2, seed=len(steps))
    import io

    from PIL import Image

    def png(a: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG")
        return buf.getvalue()

    frame = pl.DataFrame({"image": [png(a) for a in images]})
    on = frame.select(o=expr.sink(fmt)).unnest("o")
    off = frame.select(o=expr.sink(fmt, opt_flags=False)).unnest("o")
    for alias in fmt:
        assert [numpy_from_struct(x).tobytes() for x in on[alias]] == [
            numpy_from_struct(x).tobytes() for x in off[alias]
        ], alias


class TestTheSharingChecker:
    """``_assert_fully_shared`` on hand-written graphs: what it must reject
    and what it must accept, so a checker that matches nothing cannot read
    as green."""

    SRC = {"format": "image_bytes"}

    def _graph(self, nodes: dict, outputs: list[str]) -> dict:
        return {
            "nodes": nodes,
            "outputs": {f"o{i}": {"node": n} for i, n in enumerate(outputs)},
        }

    def _node(self, reads: str | None, ops: list[dict]) -> dict:
        """A root decoding column 0 (``reads=None``), or a node reading ``reads``."""
        source = {"decode": {"column": 0, "source": self.SRC}}
        return {"input": source if reads is None else {"node": reads}, "ops": ops}

    def test_rejects_two_nodes_computing_one_value(self) -> None:
        inv = [{"op": "invert"}]
        graph = self._graph(
            {
                "n0": self._node(None, []),
                "n1": self._node("n0", inv),
                "n2": self._node("n0", inv),
            },
            ["n1", "n2"],
        )
        with pytest.raises(AssertionError, match="one value"):
            _assert_fully_shared(graph)

    def test_rejects_a_pass_through_node(self) -> None:
        graph = self._graph(
            {
                "n0": self._node(None, [{"op": "invert"}]),
                "n1": self._node("n0", []),
            },
            ["n1"],
        )
        with pytest.raises(AssertionError, match="pass-through"):
            _assert_fully_shared(graph)

    def test_rejects_an_unfused_single_reader(self) -> None:
        graph = self._graph(
            {
                "n0": self._node(None, [{"op": "invert"}]),
                "n1": self._node("n0", [{"op": "flip"}]),
            },
            ["n1"],
        )
        with pytest.raises(AssertionError, match="n0 was not fused"):
            _assert_fully_shared(graph)

    def test_accepts_an_operand_an_output_and_a_fork(self) -> None:
        graph = self._graph(
            {
                "n0": self._node(None, []),
                "n1": self._node("n0", [{"op": "invert"}]),  # operand of n2
                "n2": self._node("n0", [{"op": "maximum", "other": "n1"}]),
                "n3": self._node("n2", [{"op": "flip"}]),  # n2 is an output
            },
            ["n2", "n3"],
        )
        _assert_fully_shared(graph)

    def test_accepts_an_op_reading_its_own_input(self) -> None:
        graph = self._graph(
            {
                "n0": self._node(None, []),
                "n1": self._node("n0", [{"op": "maximum", "other": "n0"}]),
            },
            ["n1"],
        )
        _assert_fully_shared(graph)


# ---------------------------------------------------------------------------
# Expression arguments: only the same expression is the same input
# ---------------------------------------------------------------------------


def _resized(width: pl.Expr | int) -> LazyPipelineExpr:
    return pl.col("image").cv.pipe(U8.resize(height=4, width=width).invert())


class TestExpressionArguments:
    """Two ops are one only when every expression argument is the same
    expression (``meta.eq``): a different expression over the same column,
    however close, is a different input."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (pl.col("w"), pl.col("w") + 1),
            (pl.col("w"), pl.col("w").fill_null(4)),
            (pl.col("w") * 2, pl.col("w") + pl.col("w")),
            (pl.col("w"), pl.col("w").cast(pl.UInt16)),
            (pl.col("w") + 1, pl.col("w") + 2),
        ],
        ids=["plus-one", "fill-null", "same-value-other-form", "cast", "constants"],
    )
    def test_different_expressions_are_not_shared(self, left, right) -> None:
        expr = _resized(left).alias("a").merge_pipe(_resized(right).alias("b"))
        graph = _physical(expr, {"a": "numpy", "b": "numpy"})
        assert _ops(graph).count("resize") == 2
        assert graph["outputs"]["a"]["node"] != graph["outputs"]["b"]["node"]

    @pytest.mark.parametrize(
        "make",
        [
            lambda: pl.col("w"),
            lambda: pl.col("w") + 1,
            lambda: pl.col("w").fill_null(4),
        ],
        ids=["column", "plus-one", "fill-null"],
    )
    def test_the_same_expression_built_twice_is_shared(self, make) -> None:
        expr = _resized(make()).alias("a").merge_pipe(_resized(make()).alias("b"))
        graph = _physical(expr, {"a": "numpy", "b": "numpy"})
        assert graph["outputs"]["a"]["node"] == graph["outputs"]["b"]["node"]

    def test_a_shared_prefix_stops_at_the_first_different_expression(self) -> None:
        img = pl.col("image").cv
        base = U8.resize(height=4, width=pl.col("w")).invert()
        a = img.pipe(base.threshold(pl.col("t"))).alias("a")
        b = img.pipe(base.threshold(pl.col("t") + 1)).alias("b")
        graph = _physical(a.merge_pipe(b), {"a": "numpy", "b": "numpy"})
        assert _ops(graph).count("resize") == 1
        assert _ops(graph).count("invert") == 1
        assert _ops(graph).count("threshold") == 2

    @plugin_required
    def test_each_branch_reads_its_own_expression(self, df: pl.DataFrame) -> None:
        a = _resized(pl.col("w")).alias("a")
        b = _resized(pl.col("w") + 1).alias("b")
        fmt = {"a": "numpy", "b": "numpy"}
        out = df.select(o=a.merge_pipe(b).sink(fmt)).unnest("o")
        widths = [
            (numpy_from_struct(x).shape[1], numpy_from_struct(y).shape[1])
            for x, y in zip(out["a"], out["b"], strict=True)
        ]
        assert widths == [(4, 5), (5, 6), (6, 7)]
        _assert_same_outputs(df, a.merge_pipe(b), fmt)


# ---------------------------------------------------------------------------
# Branching where only some intermediates are collected
# ---------------------------------------------------------------------------


class TestSomeIntermediatesCollected:
    """A tree of branches, some of whose intermediate values are outputs and
    some not. A collected value is a node an output reads; an uncollected one
    is fused into the run it feeds, even when it carries an alias the sink
    does not select."""

    @staticmethod
    def _tree() -> dict[str, LazyPipelineExpr]:
        img = pl.col("image").cv
        gray = img.pipe(U8.grayscale()).alias("gray")
        blurred = gray.pipe(Pipeline().blur(1.0)).alias("blurred")
        edges = blurred.pipe(Pipeline().canny()).alias("edges")
        dilated = edges.pipe(Pipeline().dilate(ksize=3)).alias("dilated")
        inverted = gray.pipe(Pipeline().invert()).alias("inverted")
        flipped = inverted.pipe(Pipeline().flip_h()).alias("flipped")
        both = dilated.maximum(flipped).alias("both")
        rgb = img.pipe(U8.flip_v()).alias("rgb")
        return {
            "gray": gray,
            "blurred": blurred,
            "edges": edges,
            "dilated": dilated,
            "inverted": inverted,
            "flipped": flipped,
            "both": both,
            "rgb": rgb,
        }

    COLLECTIONS = [
        ("both",),
        ("both", "gray"),
        ("both", "edges"),
        ("both", "blurred", "inverted"),
        ("edges", "flipped"),
        ("dilated", "rgb"),
        ("gray", "blurred", "edges", "dilated", "inverted", "flipped", "both", "rgb"),
    ]

    @staticmethod
    def _sink(tree: dict, collected: tuple[str, ...]) -> tuple[LazyPipelineExpr, dict]:
        first, *rest = (tree[name] for name in collected)
        # Merge with every branch, so uncollected aliases are reachable too.
        expr = first.merge_pipe(*rest, tree["both"], tree["rgb"])
        return expr, dict.fromkeys(collected, "numpy")

    @pytest.mark.parametrize("collected", COLLECTIONS, ids=lambda c: "+".join(c))
    def test_collected_values_are_nodes_and_the_rest_are_fused(self, collected) -> None:
        tree = self._tree()
        expr, fmt = self._sink(tree, collected)
        graph = _physical(expr, fmt)
        _assert_fully_shared(graph)
        nodes = graph["nodes"]
        last_op = {
            "gray": "grayscale",
            "blurred": "blur",
            "edges": "canny",
            "dilated": "dilate",
            "inverted": "invert",
            "flipped": "flip",
            "both": "maximum",
            "rgb": "flip",
        }
        for name in collected:
            node = nodes[graph["outputs"][name]["node"]]
            assert node["ops"][-1]["op"] == last_op[name], name
        # Nothing an output does not need is serialized.
        needed = set()
        stack = [o["node"] for o in graph["outputs"].values()]
        while stack:
            n = stack.pop()
            if n not in needed:
                needed.add(n)
                stack.extend(_wire_dependencies(graph, nodes[n]))
        assert needed == set(nodes)

    @plugin_required
    @pytest.mark.parametrize("collected", COLLECTIONS, ids=lambda c: "+".join(c))
    def test_outputs_are_unchanged(self, df: pl.DataFrame, collected) -> None:
        expr, fmt = self._sink(self._tree(), collected)
        _assert_same_outputs(df, expr, fmt)
