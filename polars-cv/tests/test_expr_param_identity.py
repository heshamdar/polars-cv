"""Expression parameters are identified by what they compute, not how they print.

Polars' ``str(expr)`` is a display form, not an identity: every
``pl.lit(pl.Series("f", ...))`` prints ``Series[f]`` whatever its values, and
two different Python UDFs print the same ``python_udf`` text. polars-cv used
that text as the identity of an expression parameter in every place one was
compared — the plugin input slot it binds to, ``ParamValue`` equality (and so
CSE), and root-column deduplication — so two different expressions with equal
text silently became one, and the second op read the first op's values
(CR-31).

The identity authority is now :func:`polars_cv._types.expr_key`, which keeps
the readable text as the key when it is unambiguous and disambiguates by
``Expr.meta.eq`` when it is not.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import OpSpec, ParamValue, expr_key

from .conftest import make_test_png, plugin_required

N = 3


def _series(value: float) -> pl.Series:
    # Same name on purpose: both print as ``Series[f]``.
    return pl.Series("f", [value] * N)


class TestExprKey:
    """The key is injective over distinct expressions and stable over equal ones."""

    def test_distinct_literal_series_with_equal_text_get_distinct_keys(self) -> None:
        a, b = pl.lit(_series(1.0)), pl.lit(_series(3.0))
        assert str(a) == str(b)  # the premise: the display text collides
        assert expr_key(a) != expr_key(b)

    def test_distinct_udfs_with_equal_text_get_distinct_keys(self) -> None:
        f = pl.col("x").map_batches(lambda s: s * 2)
        g = pl.col("x").map_batches(lambda s: s * 3)
        assert str(f) == str(g)
        assert expr_key(f) != expr_key(g)

    def test_equal_expressions_built_separately_share_a_key(self) -> None:
        # Deduplication of genuinely-equal expressions must survive.
        assert expr_key(pl.col("h") * 2) == expr_key(pl.col("h") * 2)

    def test_unambiguous_key_is_the_readable_text(self) -> None:
        assert expr_key(pl.col("my_column")) == 'col("my_column")'

    def test_key_is_stable_for_the_same_expression(self) -> None:
        a = pl.lit(_series(5.0))
        pl.lit(_series(6.0))  # a colliding neighbour registered in between
        assert expr_key(a) == expr_key(a)


class TestParamValueIdentity:
    """``ParamValue`` equality drives CSE, so it must not merge distinct exprs."""

    def test_params_with_colliding_text_are_not_equal(self) -> None:
        a = ParamValue.from_arg(pl.lit(_series(1.0)))
        b = ParamValue.from_arg(pl.lit(_series(3.0)))
        assert a != b

    def test_ops_with_colliding_param_text_are_not_equal(self) -> None:
        a = OpSpec("scale", {"factor": ParamValue.from_arg(pl.lit(_series(1.0)))})
        b = OpSpec("scale", {"factor": ParamValue.from_arg(pl.lit(_series(3.0)))})
        assert a != b

    def test_equal_expressions_are_equal_params(self) -> None:
        a = ParamValue.from_arg(pl.col("h") + 1)
        b = ParamValue.from_arg(pl.col("h") + 1)
        assert a == b
        assert hash(a) == hash(b)

    def test_serialized_keys_differ(self) -> None:
        # Both held, as a pipeline holds every expression it references until
        # the graph is serialized.
        a = ParamValue.from_arg(pl.lit(_series(1.0)))
        b = ParamValue.from_arg(pl.lit(_series(3.0)))
        assert a.to_dict()["col"] != b.to_dict()["col"]


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
