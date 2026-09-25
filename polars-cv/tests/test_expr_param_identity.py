"""Expression parameters are identified by what they compute, not how they print.

Polars' ``str(expr)`` is a display form, not an identity: every
``pl.lit(pl.Series("f", ...))`` prints ``Series[f]`` whatever its values, and
two different Python UDFs print the same ``python_udf`` text. polars-cv used
that text as the identity of an expression parameter in every place one was
compared — the plugin input slot it binds to, op equality (and so
CSE), and root-column deduplication — so two different expressions with equal
text silently became one, and the second op read the first op's values
(CR-31).

The identity authority is now :class:`polars_cv._types.SlotTable`: a graph
assigns each distinct expression (by ``Expr.meta.eq``) a plugin input
position, and a parameter is serialized as that position. No text is involved,
so text collisions cannot merge expressions.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import SlotTable

from ._plan_view import exprs_of, op_json, ops_of
from .conftest import make_test_png, plugin_required

N = 3


def _series(value: float) -> pl.Series:
    # Same name on purpose: both print as ``Series[f]``.
    return pl.Series("f", [value] * N)


class TestSlotTable:
    """Positions are injective over distinct expressions, shared by equal ones."""

    def test_distinct_literal_series_with_equal_text_get_distinct_slots(self) -> None:
        a, b = pl.lit(_series(1.0)), pl.lit(_series(3.0))
        assert str(a) == str(b)  # the premise: the display text collides
        table = SlotTable()
        assert table.add(a) != table.add(b)

    def test_distinct_udfs_with_equal_text_get_distinct_slots(self) -> None:
        f = pl.col("x").map_batches(lambda s: s * 2)
        g = pl.col("x").map_batches(lambda s: s * 3)
        assert str(f) == str(g)
        table = SlotTable()
        assert table.add(f) != table.add(g)

    def test_equal_expressions_built_separately_share_a_slot(self) -> None:
        # Deduplication of genuinely-equal expressions must survive.
        table = SlotTable()
        assert table.add(pl.col("h") * 2) == table.add(pl.col("h") * 2)
        assert len(table) == 1

    def test_an_unregistered_expression_is_an_error(self) -> None:
        table = SlotTable()
        table.add(pl.col("a"))
        with pytest.raises(KeyError, match="not a registered plugin input"):
            table.index(pl.col("b"))


@plugin_required
class TestPipelineSlots:
    """A pipeline numbers its expressions by ``meta.eq``, which drives CSE, so
    distinct expressions must not merge."""

    def _slots(self, pipe: Pipeline) -> list[object]:
        return [
            json.loads(op_json(pipe, i))["factor"] for i in range(len(ops_of(pipe)))
        ]

    def test_params_with_colliding_text_get_distinct_slots(self) -> None:
        pipe = Pipeline().source().scale(pl.lit(_series(1.0)))
        pipe = pipe.scale(pl.lit(_series(3.0)))
        assert self._slots(pipe) == [{"$slot": 0}, {"$slot": 1}]

    def test_equal_expressions_share_a_slot(self) -> None:
        pipe = Pipeline().source().scale(pl.col("h") + 1).scale(pl.col("h") + 1)
        assert self._slots(pipe) == [{"$slot": 0}, {"$slot": 0}]
        assert len(exprs_of(pipe)) == 1


@plugin_required
class TestExecution:
    """Verified at the user-facing entry point, not the helpers."""

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return pl.DataFrame({"img": [make_test_png(4, 4, (10, 10, 10))] * N})

    def _first_value(self, df: pl.DataFrame, pipe: Pipeline) -> float:
        out = df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy"))
        from polars_cv import numpy_from_struct

        return float(np.asarray(numpy_from_struct(out["o"][0])).ravel()[0])

    def test_two_literal_series_params_both_apply(self, df: pl.DataFrame) -> None:
        base = Pipeline().source("image_bytes", dtype="u8").cast("f32")
        pixel = self._first_value(df, base)
        pipe = base.scale(pl.lit(_series(2.0))).scale(pl.lit(_series(3.0)))
        assert self._first_value(df, pipe) == pytest.approx(pixel * 6.0)

    def test_two_udf_params_both_apply(self, df: pl.DataFrame) -> None:
        df = df.with_columns(k=pl.lit(1.0))
        base = Pipeline().source("image_bytes", dtype="u8").cast("f32")
        pixel = self._first_value(df, base)
        pipe = base.scale(pl.col("k").map_batches(lambda s: s * 2)).scale(
            pl.col("k").map_batches(lambda s: s * 5)
        )
        assert self._first_value(df, pipe) == pytest.approx(pixel * 10.0)

    def test_root_columns_with_colliding_text_stay_distinct(self) -> None:
        a = make_test_png(4, 4, (10, 10, 10))
        b = make_test_png(4, 4, (20, 20, 20))
        pipe = Pipeline().source("image_bytes", dtype="u8").cast("f32")
        left = pl.lit(pl.Series("img", [a] * N)).cv.pipe(pipe)
        right = pl.lit(pl.Series("img", [b] * N)).cv.pipe(pipe)
        assert str(pl.lit(pl.Series("img", [a] * N))) == str(
            pl.lit(pl.Series("img", [b] * N))
        )
        out = pl.DataFrame({"i": range(N)}).select(o=left.add(right).sink("numpy"))
        from polars_cv import numpy_from_struct

        value = float(np.asarray(numpy_from_struct(out["o"][0])).ravel()[0])
        assert value == pytest.approx(30.0)
