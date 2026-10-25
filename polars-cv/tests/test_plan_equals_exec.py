"""Plan == Exec: the dtype and shape planned at ``collect_schema`` equal what execution produces.

The headline invariant the architecture rests on.
"""

from __future__ import annotations

import polars as pl
import pytest

import polars_cv
from polars_cv import Pipeline
from tests._structural_helpers import _leaf_dtype, _planned_and_realized, _png
from tests.conftest import plugin_required

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


# (label, build-pipeline-callable) for image-source pipelines whose output dtype
# is deterministic and must agree between planning and execution.
# Typed list sinks require a known element dtype at plan time: either an
# explicit source dtype, or a dtype-fixing op (cast). Both forms are exercised.
_DETERMINISTIC_LIST_PIPELINES = [
    (
        "resize_u8",
        lambda: Pipeline().source("image_bytes", dtype="u8").resize(height=6, width=6),
    ),
    (
        "grayscale_u8",
        lambda: (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .resize(height=6, width=6)
            .grayscale()
        ),
    ),
    (
        "cast_f32",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=6).cast("f32"),
    ),
    (
        "cast_f64",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=6).cast("f64"),
    ),
    (
        "grayscale_then_cast_f64",
        lambda: (
            Pipeline()
            .source("image_bytes")
            .resize(height=6, width=6)
            .grayscale()
            .cast("f64")
        ),
    ),
]


@plugin_required
@pytest.mark.parametrize(
    "label,build",
    _DETERMINISTIC_LIST_PIPELINES,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_plan_equals_exec_list_sink(label, build):
    """Planned list-sink dtype must equal the realized dtype (real PNG data)."""
    if not isinstance(label, str):  # parametrize passes both tuple elements
        pytest.skip("paired param")
    df = pl.DataFrame({"out": [_png()]})
    expr = pl.col("out").cv.pipe(build()).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized, f"{label}: planned {planned} != realized {realized}"


@plugin_required
def test_plan_equals_exec_blur_preserves_float_dtype():
    """blur preserves dtype: `cast(f32).blur()` must plan AND produce f32.

    Regression for a Plan == Exec contract drift: the Python blur contract declared
    FIXED_U8 while view-buffer's blur preserves the input dtype, so this
    pipeline previously failed the execution-time dtype guard.
    """
    df = pl.DataFrame({"out": [_png()]})
    pipe = (
        Pipeline()
        .source("image_bytes")
        .resize(height=6, width=6)
        .cast("f32")
        .blur(sigma=1.0)
    )
    expr = pl.col("out").cv.pipe(pipe).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized
    assert realized == pl.List(pl.List(pl.List(pl.Float32)))


@plugin_required
def test_plan_equals_exec_array_sink():
    """Array sink with explicit shape: planned == realized."""
    df = pl.DataFrame({"out": [_png(width=6, height=6)]})
    pipe = Pipeline().source("image_bytes", dtype="u8").resize(height=6, width=6)
    expr = pl.col("out").cv.pipe(pipe).sink("array", shape=[6, 6, 3])
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized


@plugin_required
def test_plan_equals_exec_scalar_reduction():
    """Global reduction to a scalar: planned == realized."""
    df = pl.DataFrame({"out": [_png()]})
    pipe = Pipeline().source("image_bytes").grayscale().reduce_sum()
    expr = pl.col("out").cv.pipe(pipe).sink("native")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized


@plugin_required
def test_auto_16bit_image_list_sink_requires_explicit_dtype():
    """Plan == Exec: an image source with no known dtype cannot feed a typed list sink.

    The decoded dtype is only known at runtime, so the planner must refuse to
    guess (it would silently fall back to u8 and diverge from the realized u16).
    The user is required to supply the dtype instead.
    """
    pipe = Pipeline().source("image_bytes")
    with pytest.raises(ValueError, match="(?i)explicit"):
        pl.col("out").cv.pipe(pipe).sink("list")


@plugin_required
def test_auto_16bit_with_explicit_dtype_plan_equals_exec():
    """Plan == Exec: with an explicit source dtype, the 16-bit image plans and realizes u16."""
    df = pl.DataFrame({"out": [_png(mode="I;16")]})
    pipe = Pipeline().source("image_bytes", dtype="u16")
    expr = pl.col("out").cv.pipe(pipe).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized
    assert _leaf_dtype(realized) == pl.UInt16


@plugin_required
def test_plan_equals_exec_binary_promote():
    """Plan == Exec: a promoting binary op (true division) declares the promoted dtype.

    ``divide`` of two u8 images promotes to f32 (true division), and the planned
    dtype must match what execution produces — not the left operand's u8. The
    operands carry an explicit dtype so the typed sink is plannable.
    """
    df = pl.DataFrame({"out": [_png()]})
    left = pl.col("out").cv.pipe(
        Pipeline().source("image_bytes", dtype="u8").grayscale()
    )
    right = pl.col("out").cv.pipe(
        Pipeline().source("image_bytes", dtype="u8").grayscale()
    )
    expr = left.divide(right).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized
    assert _leaf_dtype(realized) == pl.Float32


def _planned_shape(pipe):
    """The pipeline's plan-time [H, W, C], using None for unknown/expr dims."""

    return list(pipe._state.dims)


# (label, build-pipeline, png-mode) exercising the rank/channel rules end-to-end.
_SHAPE_PIPELINES = [
    (
        "resize_rgb",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=5),
        "RGB",
    ),
    (
        "resize_grayscale",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=6).grayscale(),
        "RGB",
    ),
    (
        "resize_rgba_preserves_4ch",
        lambda: (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=4)
            .resize(height=6, width=6)
        ),
        "RGBA",
    ),
    (
        "cvt_gray_on_rgba_is_graya",
        lambda: (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=4)
            .resize(height=6, width=6)
            .convert_color(from_space="rgb", to_space="gray")
        ),
        "RGBA",
    ),
    (
        "cvt_hsv_on_rgba_preserves_4ch",
        lambda: (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=4)
            .resize(height=6, width=6)
            .convert_color(from_space="rgb", to_space="hsv")
        ),
        "RGBA",
    ),
]


@plugin_required
@pytest.mark.parametrize(
    "label,build,mode", _SHAPE_PIPELINES, ids=[c[0] for c in _SHAPE_PIPELINES]
)
def test_plan_equals_exec_shape(label, build, mode):
    """Each shape dimension the planner claims to know must match the realized
    array shape (plan == exec for shape, the WS-1 rank/channel invariant)."""
    df = pl.DataFrame({"out": [_png(width=8, height=8, mode=mode)]})
    pipe = build()
    realized = list(
        polars_cv.numpy_from_struct(
            df.lazy()
            .select(out=pl.col("out").cv.pipe(pipe).sink("numpy"))
            .collect()["out"][0]
        ).shape
    )
    planned = _planned_shape(pipe)
    for i, p in enumerate(planned):
        if p is not None:
            assert p == realized[i], (
                f"{label}: planned dim {i} = {p} but realized = {realized[i]} "
                f"(planned {planned} vs realized {realized})"
            )
