"""How lazy expressions become a graph: aliases, ``merge_pipe``, and which
nodes reach the wire.

An alias names an output, so an alias names exactly one node (two nodes under
one alias are refused), and a node may carry several (each an output).
``merge_pipe`` only makes branches reachable: its result is its first
operand's output, and it is never a node of its own. Only nodes an output
needs are serialized.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import LazyPipelineExpr, Pipeline, numpy_from_struct
from tests.conftest import plugin_required


def _img(column: str = "image") -> LazyPipelineExpr:
    return pl.col(column).cv.pipe(Pipeline().source("image_bytes", dtype="u8"))


def _wire_nodes(expr: LazyPipelineExpr, fmt, **kwargs) -> dict:
    return expr.sink(fmt, return_expr=False, **kwargs)._to_dict()["nodes"]


class TestAliases:
    def test_one_alias_on_two_nodes_is_refused(self) -> None:
        img = _img()
        a = img.pipe(Pipeline().grayscale()).alias("x")
        b = img.pipe(Pipeline().blur(2.0)).alias("x")
        with pytest.raises(ValueError, match="'x'.*two different nodes"):
            a.merge_pipe(b).sink({"x": "numpy"})

    def test_a_node_may_carry_several_aliases(self) -> None:
        gray = _img().pipe(Pipeline().grayscale())
        graph = (
            gray.alias("p")
            .merge_pipe(gray.alias("q"))
            .sink({"p": "numpy", "q": "png"}, return_expr=False)
        )
        outputs = graph._to_dict()["outputs"]
        assert set(outputs) == {"p", "q"}
        assert outputs["p"]["node"] == outputs["q"]["node"]

    def test_the_same_alias_on_the_same_node_twice_is_one_output(self) -> None:
        gray = _img().pipe(Pipeline().grayscale()).alias("g")
        graph = gray.merge_pipe(gray).sink({"g": "numpy"}, return_expr=False)
        assert set(graph._to_dict()["outputs"]) == {"g"}


class TestMergePipe:
    def test_the_merge_node_is_not_serialized(self) -> None:
        img = _img()
        a = img.pipe(Pipeline().grayscale()).alias("a")
        b = img.pipe(Pipeline().blur(1.0)).alias("b")
        merged = a.merge_pipe(b)
        for opt in (True, False):
            nodes = _wire_nodes(merged, {"a": "numpy", "b": "numpy"}, opt_flags=opt)
            # The source, `a` and `b`: nothing reads the merge node.
            assert len(nodes) == 3, (opt, nodes)

    def test_a_nested_merge_reaches_every_branch(self) -> None:
        img = _img()
        a = img.pipe(Pipeline().grayscale()).alias("a")
        b = img.pipe(Pipeline().blur(1.0)).alias("b")
        c = img.pipe(Pipeline().flip_h()).alias("c")
        graph = (
            a.merge_pipe(b)
            .merge_pipe(c)
            .sink({"a": "numpy", "b": "numpy", "c": "numpy"}, return_expr=False)
        )
        assert set(graph._to_dict()["outputs"]) == {"a", "b", "c"}


@plugin_required
class TestMergePipeExecutes:
    @pytest.fixture
    def df(self, create_test_png: Callable[..., bytes]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "image": [create_test_png(8, 6, (40, 80, 120))],
                "other": [create_test_png(4, 4, (200, 10, 10))],
            }
        )

    @staticmethod
    def _run(df: pl.DataFrame, expr: pl.Expr) -> object:
        return df.select(o=expr).item()

    def test_a_single_sink_is_the_first_operands_output(self, df: pl.DataFrame) -> None:
        gray = _img().pipe(Pipeline().grayscale())
        brighter = gray.pipe(Pipeline().add_constant(50))
        merged = brighter.merge_pipe(gray)
        np.testing.assert_array_equal(
            numpy_from_struct(self._run(df, merged.sink("numpy"))),
            numpy_from_struct(self._run(df, brighter.sink("numpy"))),
        )

    def test_branches_from_different_columns(self, df: pl.DataFrame) -> None:
        a = _img().pipe(Pipeline().grayscale()).alias("a")
        b = _img("other").pipe(Pipeline().grayscale()).alias("b")
        out = self._run(df, a.merge_pipe(b).sink({"a": "numpy", "b": "numpy"}))
        assert numpy_from_struct(out["a"]).shape == (6, 8, 1)
        assert numpy_from_struct(out["b"]).shape == (4, 4, 1)

    def test_merging_an_upstream_node(self, df: pl.DataFrame) -> None:
        img = _img().alias("img")
        gray = img.pipe(Pipeline().grayscale()).alias("gray")
        out = self._run(
            df, gray.merge_pipe(img).sink({"img": "numpy", "gray": "numpy"})
        )
        assert numpy_from_struct(out["img"]).shape == (6, 8, 3)
        assert numpy_from_struct(out["gray"]).shape == (6, 8, 1)


class TestOnlyNeededNodesAreSerialized:
    def test_a_node_no_output_reads_is_dropped(self) -> None:
        img = _img()
        used = img.pipe(Pipeline().grayscale()).alias("used")
        unused = img.pipe(Pipeline().blur(3.0)).alias("unused")
        for opt in (True, False):
            nodes = _wire_nodes(
                used.merge_pipe(unused), {"used": "numpy"}, opt_flags=opt
            )
            ops = [op["op"] for n in nodes.values() for op in n["ops"]]
            assert "blur" not in ops, (opt, nodes)
