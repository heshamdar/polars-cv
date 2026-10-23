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

    def test_a_numpy_scalar_is_a_number(self) -> None:
        assert (this.height * np.int64(2))._wire() == {"$this": ["*", "height", 2]}
        assert (np.float32(0.5) * this.width)._wire() == {"$this": ["*", 0.5, "width"]}

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

    def test_an_op_after_rasterize_reads_the_raster(self) -> None:
        """The op before was given a contour set, not a buffer; the size of
        what it made is still known before the crop runs."""
        from polars_cv import CONTOUR_SET_SCHEMA

        square = {
            "exterior": [{"x": x, "y": y} for x, y in [(1, 1), (6, 1), (6, 5), (1, 5)]],
            "holes": [],
            "is_closed": True,
        }
        frame = pl.DataFrame(
            {"c": [[square]]}, schema_overrides={"c": CONTOUR_SET_SCHEMA}
        )
        pipe = (
            Pipeline()
            .source("contour")
            .rasterize(width=10, height=8)
            .crop(height=this.height // 2, width=this.width - 1)
        )
        out = frame.select(pl.col("c").cv.pipe(pipe).sink("numpy"))
        assert numpy_from_struct(out.item()).shape[:2] == (4, 9)

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

    def test_the_plan_uses_this_where_it_knows_the_size(self) -> None:
        """After a resize the plan knows the height, so ``this.height // 2`` is
        4 when the query is planned; over a size it does not know, the axis is
        unknown, as for a column."""
        base = Pipeline().source("image_bytes").resize(height=8, width=8)
        assert state_of(base.crop(height=this.height // 2)).dims == (4, 8, None)
        unsized = Pipeline().source("image_bytes")
        assert state_of(unsized.crop(height=this.height // 2)).dims == (
            None,
            None,
            None,
        )

    def test_a_planned_size_reaches_a_fixed_shape_sink(self) -> None:
        pipe = (
            Pipeline()
            .source("list", dtype="u8")
            .assert_shape(dims=[10, 8])
            .crop(height=this.height // 2, width=this.width - 4)
        )
        frame = _column(_img(10, 8))
        out = frame.select(pl.col("a").cv.pipe(pipe).sink("array"))
        assert out.schema["a"] == pl.Array(pl.UInt8, (5, 4))


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
        # The rank is known when the pipeline is built, so it is refused then.
        with pytest.raises(ValueError, match=r"this.height reads an \[H, W\]"):
            Pipeline().source("list").reshape([9]).clamp_max(this.height)

    def test_an_op_given_no_buffer_refuses_this_as_it_is_built(self) -> None:
        with pytest.raises(ValueError, match="`this` reads the buffer an op is given"):
            Pipeline().source("contour").rasterize(width=this.width, height=10)

    def test_a_value_the_plan_knows_is_wrong_is_refused_as_it_is_built(self) -> None:
        base = Pipeline().source("image_bytes").resize(height=8, width=8)
        with pytest.raises(ValueError, match="height.*negative"):
            base.crop(height=this.height - 100)

    @pytest.mark.parametrize(
        ("divides", "around"),
        [
            (
                (this.height - this.height) * 2 // (this.width - this.width),
                lambda e: e + 1,
            ),
            ((this.height - this.height) / (0 * this.ndim), lambda e: 10 - e),
            (this.height / (this.width - this.width), lambda e: e),
            # Parenthesised only by precedence: `//` binds tighter than `-`.
            (
                (this.height - this.height // 1) // (this.width - this.width),
                lambda e: e,
            ),
        ],
        ids=["floordiv", "nested", "div", "precedence"],
    )
    def test_the_engine_quotes_an_expression_as_python_writes_it(
        self, divides: object, around: object
    ) -> None:
        """The operators are Python's syntax and the engine's display: both
        sides write each one, so the error quotes the dividing part as
        ``repr`` writes it."""
        expr = around(divides)  # type: ignore[operator]
        pipe = Pipeline().source("list").blur(sigma=expr)
        with pytest.raises(pl.exceptions.ComputeError) as err:
            _run(pipe, _img(4, 4))
        assert f"{divides!r}: division by zero" in str(err.value)

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


def test_a_parameter_takes_this_only_where_its_op_reads_a_buffer() -> None:
    """The annotations say what is accepted: ``IntOrThisOrExpr`` on an op whose
    input may be a buffer, never on one that only reads contours or vectors
    (``rasterize``, the contour ops), where ``this`` is always refused."""
    import inspect

    from polars_cv._ops_generated import OP_DOMAINS, TYPED_OPS

    catalogue = {
        op["python"]: op["name"]
        for op in __import__("json").loads(
            __import__("polars_cv._lib", fromlist=["op_catalog"]).op_catalog()
        )
    }
    wrong = []
    for python, name in sorted(catalogue.items()):
        method = getattr(Pipeline, python, None)
        if method is None or name not in TYPED_OPS:
            continue
        reads_buffer = any(d["input"] == "buffer" for d in OP_DOMAINS[name])
        for param in inspect.signature(method).parameters.values():
            takes_this = "ThisOr" in str(param.annotation)
            if takes_this and not reads_buffer:
                wrong.append(
                    f"{python}({param.name}) takes this, input {OP_DOMAINS[name]}"
                )
    assert not wrong, wrong
    # The sweep saw both kinds.
    assert "ThisOr" in str(
        inspect.signature(Pipeline.crop).parameters["top"].annotation
    )


def _golden_render_cases() -> list:
    import json
    from pathlib import Path

    path = Path(__file__).parent / "golden" / "this_render.json"
    return json.loads(path.read_text())["cases"]


@pytest.mark.parametrize(("tree", "text"), _golden_render_cases())
def test_reads_as_the_shared_golden_says(tree, text) -> None:
    """The text Rust's ``Display`` gives the same tree (``mode.rs`` reads this
    file too), so an error quotes the expression as Python writes it."""
    from polars_cv._this import ThisExpr

    assert repr(ThisExpr(tree)) == text


def test_the_operators_are_the_engines() -> None:
    """Each operator the engine spells (``ThisOp``, generated) is reachable
    from Python, and Python builds no other."""
    from polars_cv._ops_generated import THIS_OPERATORS
    from polars_cv._this import _PRECEDENCE

    built = {
        (this.height + 1)._tree[0],
        (this.height - 1)._tree[0],
        (this.height * 1)._tree[0],
        (this.height / 1)._tree[0],
        (this.height // 1)._tree[0],
        (-this.height)._tree[0],
    }
    assert built == set(THIS_OPERATORS) == set(_PRECEDENCE)
