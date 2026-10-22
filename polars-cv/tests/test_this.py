"""``this``: op parameters derived from the buffer the op reads, per row.

``crop(height=this.height // 2)`` halves every image whatever its size, and
after a ``resize`` ``this.height`` is the resized height: each op reads the
buffer it is given. An integer parameter takes the expression's value rounded
to nearest.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct, this

from ._plan_view import state_of
from .conftest import plugin_required

RNG = np.random.default_rng(23)


def _column(*arrays: np.ndarray) -> pl.DataFrame:
    rows = [a.tolist() for a in arrays]
    dtype: pl.DataType = pl.UInt8()
    for _ in range(arrays[0].ndim):
        dtype = pl.List(dtype)
    return pl.DataFrame({"a": pl.Series("a", rows, dtype=dtype)})


def _run(pipe: Pipeline, *arrays: np.ndarray) -> list[np.ndarray]:
    out = _column(*arrays).select(pl.col("a").cv.pipe(pipe).sink("numpy"))
    return [numpy_from_struct(v) for v in out["a"]]


def _img(*shape: int) -> np.ndarray:
    return RNG.integers(0, 256, shape, dtype=np.uint8)


class TestExpressions:
    """The Python side builds a tree; nothing is evaluated there."""

    def test_arithmetic_builds_the_wire_tree(self) -> None:
        expr = (this.height - 4) * 0.5 + this.width // 3
        assert expr._wire() == {
            "$this": ["+", ["*", ["-", "height", 4], 0.5], ["//", "width", 3]]
        }

    def test_reflected_and_negated(self) -> None:
        assert (10 - this.height)._wire() == {"$this": ["-", 10, "height"]}
        assert (2 / this.width)._wire() == {"$this": ["/", 2, "width"]}
        assert (-this.channels)._wire() == {"$this": ["-", 0, "channels"]}

    def test_reads_as_written(self) -> None:
        assert repr((this.height - 4) * 0.5) == "(this.height - 4) * 0.5"
        assert repr(this.height / (this.width + 1)) == "this.height / (this.width + 1)"

    @pytest.mark.parametrize("other", [pl.col("x"), "3", True, float("nan")])
    def test_refuses_what_is_not_a_finite_number(self, other: object) -> None:
        with pytest.raises(TypeError):
            this.height + other  # type: ignore[operator]

    def test_an_unknown_attribute_is_an_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            this.depth  # type: ignore[attr-defined]  # noqa: B018


@plugin_required
class TestPerRow:
    def test_each_row_reads_its_own_size(self) -> None:
        pipe = (
            Pipeline()
            .source("list")
            .crop(height=this.height // 2, width=this.width // 2)
        )
        a, b = _img(10, 8), _img(7, 12)
        got = _run(pipe, a, b)
        np.testing.assert_array_equal(got[0], a[:5, :4])
        np.testing.assert_array_equal(got[1], b[:3, :6])

    def test_an_op_reads_the_buffer_it_is_given(self) -> None:
        pipe = (
            Pipeline()
            .source("list")
            .resize(height=40, width=60)
            .crop(top=this.height / 4, height=this.height / 2, width=this.width - 10)
        )
        (got,) = _run(pipe, _img(13, 17))
        assert got.shape == (20, 50)

    def test_channels_and_ndim(self) -> None:
        rgb = _img(4, 5, 3)
        (got,) = _run(Pipeline().source("list").clamp_max(this.channels * 10), rgb)
        np.testing.assert_array_equal(got, np.minimum(rgb, 30))
        gray = _img(3, 3)
        (got,) = _run(Pipeline().source("list").clamp_max(this.channels), gray)
        np.testing.assert_array_equal(got, np.minimum(gray, 1))
        pipe = Pipeline().source("list").reshape([9]).clamp_max(this.ndim)
        (got,) = _run(pipe, gray)
        np.testing.assert_array_equal(got, np.minimum(gray.ravel(), 1))

    def test_an_integer_parameter_rounds_to_nearest(self) -> None:
        # 0.29 * 100 is 28.999999999999996 in floating point.
        pipe = Pipeline().source("list").crop(height=0.29 * this.height)
        (got,) = _run(pipe, _img(100, 3))
        assert got.shape[0] == 29

    def test_a_float_parameter_takes_the_value(self) -> None:
        a, b = _img(20, 20), _img(40, 40)
        got = _run(Pipeline().source("list").blur(sigma=this.height / 10), a, b)
        want = [
            _run(Pipeline().source("list").blur(sigma=2.0), a)[0],
            _run(Pipeline().source("list").blur(sigma=4.0), b)[0],
        ]
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w)

    def test_it_works_on_a_lazy_expression_and_beside_a_column(self) -> None:
        frame = _column(_img(10, 10)).with_columns(t=pl.lit(2))
        img = pl.col("a").cv.pipe(Pipeline().source("list"))
        out = frame.select(
            img.crop(top=pl.col("t"), height=this.height - 4).sink("numpy").alias("a")
        )
        assert numpy_from_struct(out["a"][0]).shape == (6, 10)

    def test_background_divide_scales_with_the_image(self) -> None:
        a = _img(20, 30).astype(np.float32)
        frame = pl.DataFrame(
            {"a": pl.Series("a", [a.tolist()], dtype=pl.List(pl.List(pl.Float32)))}
        )
        img = pl.col("a").cv.pipe(Pipeline().source("list"))
        got, want = (
            numpy_from_struct(
                frame.select(img.background_divide(sigma).sink("numpy"))["a"][0]
            )
            for sigma in (this.height / 4, 5.0)
        )
        np.testing.assert_array_equal(got, want)

    def test_the_planned_size_is_per_row(self) -> None:
        """A size from ``this`` is unknown when the query is planned, as a
        column's is; the axes it does not set stay known."""
        base = Pipeline().source("image_bytes").resize(height=8, width=8)
        assert state_of(base.crop(height=4)).dims == (4, 8, None)
        assert state_of(base.crop(height=this.height // 2)).dims == (None, 8, None)


@plugin_required
class TestRefusals:
    def test_a_structural_parameter_is_refused(self) -> None:
        with pytest.raises(TypeError, match="structural"):
            Pipeline().source("list").perceptual_hash(hash_size=this.height)  # type: ignore[arg-type]

    def test_a_function_with_no_buffer_is_refused(self) -> None:
        contours = pl.col("c")
        with pytest.raises(TypeError, match=r"this.height / 2 reads the buffer"):
            contours.contour.simplify(tolerance=this.height / 2)

    def test_a_non_numeric_parameter_is_refused(self) -> None:
        with pytest.raises(ValueError, match="`this` attributes give a number"):
            Pipeline().source("list").flip_h(when=this.height)  # type: ignore[arg-type]

    def test_an_attribute_the_buffer_does_not_have_is_refused(self) -> None:
        pipe = Pipeline().source("list").reshape([9]).clamp_max(this.height)
        with pytest.raises(
            pl.exceptions.ComputeError, match=r"this.height reads an \[H, W\]"
        ):
            _run(pipe, _img(3, 3))

    def test_division_by_zero_is_an_error_naming_the_parameter(self) -> None:
        pipe = (
            Pipeline()
            .source("list")
            .blur(sigma=this.height / (this.width - this.width))
        )
        with pytest.raises(pl.exceptions.ComputeError, match="sigma.*division by zero"):
            _run(pipe, _img(4, 4))

    def test_a_negative_size_is_refused(self) -> None:
        pipe = Pipeline().source("list").crop(height=this.height - 100)
        with pytest.raises(pl.exceptions.ComputeError, match="height.*negative"):
            _run(pipe, _img(4, 4))
