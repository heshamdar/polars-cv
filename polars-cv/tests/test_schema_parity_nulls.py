"""Plan == exec on the null and error paths, in every row layout.

The schema is fixed before a single row is read, so nothing about *which* rows
are null may change it. That is easy to state and easy to get wrong, because
the runtime series builders derive the leaf dtype and the nesting depth from
the first non-null row (``build_typed_list_series_from_rows_with_dtype``) and
resolve each output's ``"auto"`` dtype from ``inputs.first()``
(``resolved_output_specs``). Both are "look at the data to decide the schema",
and both behave differently depending on where the nulls fall.

So the axis here is the layout itself: nulls first, nulls last, an all-null
column, a null run longer than a morsel, alternating rows across a streaming
batch boundary, and rows whose *shapes* differ from one another. Every layout
must produce the schema that was planned, and the same one as every other
layout.

Everything runs under the streaming engine first (see ``ENGINES``), because
that is where a column is split into morsels and "the first row" stops being a
single well-defined thing.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import SinkFormat
from tests._op_cases import IMAGE_ENCODER_SINKS
from tests._schema_parity import (
    HETEROGENEOUS_PATTERNS,
    HOMOGENEOUS_PATTERNS,
    ROW_PATTERNS,
    assert_not_vacuous,
    assert_plan_equals_exec,
    encodable_by_image_codec,
    frame,
    plan_or_reject,
    rows_for,
)
from tests.conftest import make_image_png, plugin_required

H, W, C = 8, 10, 3
SINKS: tuple[str, ...] = tuple(sorted(m.value for m in SinkFormat))


def _images() -> list[bytes]:
    return [make_image_png(H, W, C, seed=s) for s in (1, 2, 3)]


def _base() -> Pipeline:
    return (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=H, width=W, channels=C)
        .cast("u8")
    )


def _corrupt() -> bytes:
    """Bytes that start like a PNG and are not one."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


# ---------------------------------------------------------------------------
# Every layout, every sink
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("pattern", sorted(ROW_PATTERNS))
def test_every_row_layout_plans_the_same_schema(pattern: str) -> None:
    """The planned schema must not depend on where the nulls are."""
    # Heterogeneous layouts carry differently-shaped rows, so the pipeline
    # normalises with a resize; the point is that execution must not re-derive
    # the shape from whichever row it reads first.
    pipe = (
        _base()
        if pattern in HOMOGENEOUS_PATTERNS
        else (Pipeline().source("image_bytes").resize(height=6, width=6).cast("u8"))
    )
    images = (
        _images()
        if pattern in HOMOGENEOUS_PATTERNS
        else [
            make_image_png(4, 5, C, seed=1),
            make_image_png(16, 9, C, seed=2),
            make_image_png(7, 7, C, seed=3),
        ]
    )
    df = frame(rows_for(pattern, images), pl.Binary)

    sinks = (
        SINKS
        if encodable_by_image_codec(pipe)
        else tuple(s for s in SINKS if s not in IMAGE_ENCODER_SINKS)
    )
    results = {
        sink: plan_or_reject(df, lambda s=sink: pl.col("img").cv.pipe(pipe).sink(s))
        for sink in sinks
    }
    assert_not_vacuous(results, f"row layout {pattern}")


@plugin_required
def test_all_layouts_agree_with_one_another() -> None:
    """Not just plan==exec per layout: the layouts must agree *across* runs.

    A per-layout check passes if every layout is self-consistently wrong. This
    compares them, which is what catches a schema that tracks the data.
    """
    pipe = _base()
    images = _images()

    for sink in ("list", "numpy", "blob", "array"):
        planned: dict[str, pl.DataType] = {}
        for pattern in HOMOGENEOUS_PATTERNS:
            df = frame(rows_for(pattern, images), pl.Binary)
            result = plan_or_reject(
                df, lambda s=sink: pl.col("img").cv.pipe(pipe).sink(s)
            )
            if result.ok:
                planned[pattern] = result.planned

        assert planned, f"no layout planned the {sink!r} sink"
        distinct = {str(d) for d in planned.values()}
        assert len(distinct) == 1, (
            f"{sink!r} sink planned different schemas for different null "
            f"layouts: { {k: str(v) for k, v in planned.items()} }"
        )


@plugin_required
@pytest.mark.parametrize("pattern", HETEROGENEOUS_PATTERNS)
def test_differently_shaped_rows_under_a_fixed_size_sink(pattern: str) -> None:
    """The strictest combination: heterogeneous rows into ``sink("array")``.

    The pipeline resizes, so the shape is genuinely fixed and the array sink is
    legitimate. If execution took the shape from row 0 the fixed-size column
    could not be built at all.
    """
    images = [
        make_image_png(4, 5, C, seed=1),
        make_image_png(16, 9, C, seed=2),
        make_image_png(7, 7, C, seed=3),
    ]
    df = frame(rows_for(pattern, images), pl.Binary)
    pipe = (
        Pipeline()
        .source("image_bytes")
        .resize(height=6, width=6)
        .assert_shape(channels=C)
        .cast("u8")
    )

    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("array"))
    assert series.dtype == pl.Array(pl.UInt8, (6, 6, C))


@plugin_required
def test_an_all_null_column_still_collects_to_the_planned_dtype() -> None:
    """No data at all is the case where the plan is the only source of truth.

    It is also the least-travelled path: it is the only one where the runtime
    builders fall back to the declared spec rather than reading a row, and
    ``null_row_result_for_spec`` has a ``_ => Binary(None)`` fallback whose
    arms do not mirror ``dtype_for_output``'s one for one.
    """
    pipe = _base()
    df = pl.DataFrame({"img": [None, None, None]}, schema={"img": pl.Binary})

    for sink in ("list", "numpy", "blob", "array", "png"):
        series = assert_plan_equals_exec(
            df, pl.col("img").cv.pipe(pipe).sink(sink), name=f"out_{sink}"
        )
        # A struct output nulls its *fields*, not the outer row, so
        # `null_count()` is 0 there by ordinary Polars semantics.
        if isinstance(series.dtype, pl.Struct):
            for field in series.dtype.fields:
                assert series.struct.field(field.name).null_count() == 3, (
                    f"{sink}: field {field.name} should be all-null"
                )
        else:
            assert series.null_count() == 3, (
                f"{sink}: an all-null input should stay all-null, got {series}"
            )


# ---------------------------------------------------------------------------
# Null parameters
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_null_parameter_rows_do_not_change_the_schema(position: str) -> None:
    """``on_null_param("null")`` nulls the row, not the column's type."""
    heights = {
        "first": [None, 5, 6],
        "middle": [5, None, 6],
        "last": [5, 6, None],
    }[position]
    images = _images()
    df = pl.DataFrame(
        {"img": images, "h": heights},
        schema={"img": pl.Binary, "h": pl.Int64},
    )

    pipe = (
        Pipeline()
        .source("image_bytes")
        .on_null_param("null")
        .resize(height=pl.col("h"), width=4)
        .cast("u8")
    )
    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("list"))
    assert series.null_count() == 1, (
        f"exactly the null-parameter row should be null, got {series.null_count()}"
    )


@plugin_required
def test_a_fully_null_parameter_column_keeps_the_schema() -> None:
    """Every row nulled by its parameter: still the planned dtype."""
    df = pl.DataFrame(
        {"img": _images(), "h": [None, None, None]},
        schema={"img": pl.Binary, "h": pl.Int64},
    )
    pipe = (
        Pipeline()
        .source("image_bytes")
        .on_null_param("null")
        .resize(height=pl.col("h"), width=4)
        .cast("u8")
    )
    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("list"))
    assert series.null_count() == 3


# ---------------------------------------------------------------------------
# Row error policies
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_on_error_null_keeps_the_planned_schema(position: str) -> None:
    """Corrupt rows go null; the column type is unchanged.

    The corrupt row is placed first as well as elsewhere: a failing *first*
    row is the case where a schema derived from "the first row" has nothing
    valid to derive from.
    """
    images = _images()
    rows = {
        "first": [_corrupt(), images[0], images[1]],
        "middle": [images[0], _corrupt(), images[1]],
        "last": [images[0], images[1], _corrupt()],
    }[position]
    df = frame(rows, pl.Binary)

    pipe = (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=H, width=W, channels=C)
        .cast("u8")
        .on_error("null")
    )
    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("list"))
    assert series.null_count() == 1


@plugin_required
def test_every_row_failing_keeps_the_planned_schema() -> None:
    """All rows corrupt: no successful row exists to shape the output."""
    df = frame([_corrupt(), _corrupt(), _corrupt()], pl.Binary)
    pipe = (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=H, width=W, channels=C)
        .cast("u8")
        .on_error("null")
    )
    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("list"))
    assert series.null_count() == 3


@plugin_required
@pytest.mark.parametrize("position", ["first", "last"])
def test_null_with_message_struct_is_planned_field_for_field(position: str) -> None:
    """``null_with_message`` turns a single output into a two-field struct.

    The struct is synthesised in the planner (``lib.rs``) and again at
    execution (``compiled.rs``), so the field list is one more thing computed
    twice.
    """
    images = _images()
    rows = (
        [_corrupt(), images[0], images[1]]
        if position == "first"
        else [images[0], images[1], _corrupt()]
    )
    df = frame(rows, pl.Binary)

    pipe = (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=H, width=W, channels=C)
        .cast("u8")
        .on_error("null_with_message")
    )
    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("list"))

    assert isinstance(series.dtype, pl.Struct)
    names = [f.name for f in series.dtype.fields]
    assert "_error" in names, f"expected an _error field, got {names}"
    assert dict((f.name, f.dtype) for f in series.dtype.fields)["_error"] == pl.String

    errors = series.struct.field("_error")
    assert errors.null_count() == 2, "only the corrupt row should carry a message"
