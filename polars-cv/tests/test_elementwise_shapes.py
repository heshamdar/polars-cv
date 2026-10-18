"""Element-wise binary ops take operands of one shape; ``squeeze`` and
``expand_dims`` change a rank by one axis.

The binary ops used to broadcast as NumPy does, aligned from the last axis.
That made ``[H, W, 1]`` against ``[H, W]`` a ``[H, W, W]`` cube on a square
canvas (4 GB of f32 at 1024 px) and an error on any other. An operand of a
different shape is now refused, at build where the sizes are known and per
row where they are not; a rank is changed explicitly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_column
from tests.conftest import make_image_png, plugin_required


def _array(*arrays: np.ndarray) -> pl.DataFrame:
    shape = arrays[0].shape
    return pl.DataFrame(
        {"x": pl.Series(list(arrays), dtype=pl.Array(pl.Float32, shape))}
    )


def _x():
    return pl.col("x").cv.pipe(Pipeline().source("array"))


def _sized(*dims: int) -> Pipeline:
    return Pipeline().source("array", dtype="f32").assert_shape(dims=list(dims))


def _run(df: pl.DataFrame, expr) -> list[np.ndarray]:
    return numpy_from_column(df.select(expr.sink("numpy")).to_series(), copy=True)


@plugin_required
class TestShapesMustMatch:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            pytest.param((4, 4, 1), (4, 4), id="square-rank"),
            pytest.param((4, 5, 1), (4, 5), id="rank"),
            pytest.param((4, 5, 1), (4, 5, 3), id="size-one-axis"),
            pytest.param((4, 5, 3), (3,), id="trailing-vector"),
        ],
    )
    def test_a_different_shape_is_refused_at_build(self, left, right) -> None:
        x = pl.col("x").cv.pipe(_sized(*left))
        y = pl.col("y").cv.pipe(_sized(*right))
        with pytest.raises(ValueError, match=r"same shape"):
            x.add(y)
        with pytest.raises(ValueError, match=r"same shape"):
            y.multiply(x)

    def test_equal_shapes_combine(self) -> None:
        a = np.arange(20, dtype=np.float32).reshape(4, 5, 1)
        (out,) = _run(_array(a), _x().add(_x().scale(2.0)))
        np.testing.assert_array_equal(out, a * 3)

    def test_a_per_row_mismatch_fails_only_its_row(self) -> None:
        df = pl.DataFrame(
            {
                "a": [make_image_png(4, 5, 3, seed=1), make_image_png(4, 5, 3, seed=2)],
                "b": [make_image_png(4, 5, 3, seed=3), make_image_png(6, 5, 3, seed=4)],
            }
        )
        a = pl.col("a").cv.pipe(Pipeline().source("image_bytes").on_error("null"))
        b = pl.col("b").cv.pipe(Pipeline().source("image_bytes"))
        out = df.select(a.add(b).sink("numpy")).to_series()
        assert out.is_null().to_list() == [False, True]
        with pytest.raises(Exception, match=r"same shape"):
            plain = pl.col("a").cv.pipe(Pipeline().source("image_bytes"))
            df.select(plain.add(b).sink("numpy"))


@plugin_required
class TestSqueezeAndExpandDims:
    def test_expand_dims_then_squeeze_round_trips(self) -> None:
        a = np.arange(20, dtype=np.float32).reshape(4, 5)
        df = pl.DataFrame({"x": pl.Series([a], dtype=pl.Array(pl.Float32, (4, 5)))})
        expanded = _x().expand_dims(-1)
        (e,) = _run(df, expanded)
        assert e.shape == (4, 5, 1)
        np.testing.assert_array_equal(e[..., 0], a)
        (front,) = _run(df, _x().expand_dims(0))
        assert front.shape == (1, 4, 5)
        (back,) = _run(df, expanded.squeeze(-1))
        np.testing.assert_array_equal(back, a)

    def test_the_planned_schema_carries_the_new_rank(self) -> None:
        df = pl.DataFrame(
            {
                "x": pl.Series(
                    [np.zeros((4, 5), np.float32)], dtype=pl.Array(pl.Float32, (4, 5))
                )
            }
        )
        schema = df.lazy().select(o=_x().expand_dims(1).sink("array")).collect_schema()
        assert schema["o"] == pl.Array(pl.Float32, (4, 1, 5))

    def test_a_rank_is_fixed_before_combining(self) -> None:
        """The trap the refusal replaces: say which axis to add."""
        a = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
        x = _x()
        (out,) = _run(
            _array(a), x.add(x.channel_select(0, keepdims=False).expand_dims(-1))
        )
        np.testing.assert_array_equal(out, a * 2)

    def test_squeeze_refuses_an_axis_of_known_size_other_than_one(self) -> None:
        with pytest.raises(ValueError, match=r"squeeze\(\).*size 1"):
            _sized(4, 5, 3).squeeze(-1)

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda p: p.squeeze(3), id="squeeze-past-rank"),
            pytest.param(lambda p: p.squeeze(-4), id="squeeze-before-first"),
            pytest.param(lambda p: p.expand_dims(4), id="expand-past-rank"),
            pytest.param(lambda p: p.expand_dims(-5), id="expand-before-first"),
        ],
    )
    def test_an_axis_outside_the_rank_is_refused(self, build) -> None:
        with pytest.raises(ValueError, match=r"axis"):
            build(_sized(4, 5, 1))
