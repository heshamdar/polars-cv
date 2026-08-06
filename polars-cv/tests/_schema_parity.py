"""Harness for the plan-time == execution-time schema invariant.

Polars publishes an expression's output dtype at planning time
(``LazyFrame.collect_schema()``) and again, implicitly, when the data arrives
(``collect().schema``). Those two must be the same value. Everything the lazy
API rests on — a user reading a schema before any data moves, a downstream
expression type-checking against it, a fixed-size ``pl.Array`` column — is a
promise the planner makes and execution has to keep.

In this plugin the two halves are computed by *different code*:

* plan:    ``unified_output_dtype`` (src/lib.rs) -> ``dtype_for_output``
           (src/graph/decode.rs), from the hint bundle Python folded into the
           graph JSON (``expected_domain``/``_dtype``/``_shape``/``_ndim``).
* runtime: ``build_series_from_spec`` (src/graph/decode.rs) -> the typed
           builders in src/graph/encode.rs.

and the runtime half reads the *data* to decide. ``build_typed_list_series_
from_rows_with_dtype`` takes both the leaf dtype and the nesting depth from
the first non-null row; ``resolved_output_specs`` resolves every output's
``"auto"`` dtype from ``inputs.first()``. That is sound when row 0 happens to
be a representative value and undefined otherwise — which is why
``ROW_PATTERNS`` below exists and why null-first and all-null are not exotic
cases but the point.

Three test files used to carry their own copy of the plan-vs-exec assertion
(``_assert_plan_matches_data``, ``_planned_and_realized``,
``_assert_plan_equals_exec``). They are one mechanism and they live here now.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import polars as pl

# Streaming leads deliberately: it is the primary execution mode for this
# library, and it is the engine that splits a column into morsels — so it is
# the one where a schema derived from "the first row" can differ per batch.
STREAMING = "streaming"
IN_MEMORY = "in-memory"
ENGINES: tuple[str, ...] = (STREAMING, IN_MEMORY)


class Outcome(Enum):
    """What happened to one cell of a sweep.

    Rejection is an acceptable outcome for any cell: that is the planner
    refusing a combination before any data moves, which is what makes
    ``.sink("png")`` on a scalar a build error rather than a runtime surprise.
    What must never happen is planning succeeding and execution then failing
    or producing something else.
    """

    REJECTED_AT_BUILD = auto()  # .sink() raised while building the expression
    REJECTED_AT_PLAN = auto()  # collect_schema() raised
    OK = auto()  # planned, executed, and the two agreed


@dataclass(frozen=True)
class ParityResult:
    """One sweep cell's outcome, plus what it produced when it got that far."""

    outcome: Outcome
    planned: pl.DataType | None = None
    series: pl.Series | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


def assert_plan_equals_exec(
    df: pl.DataFrame,
    expr: pl.Expr,
    *,
    name: str = "out",
    engines: Sequence[str] = ENGINES,
) -> pl.Series:
    """Assert the planned dtype for *name* is what every engine produces.

    Returns the produced Series (from the first engine) so callers can make
    value-level assertions on top.
    """
    lf = df.lazy().with_columns(**{name: expr})
    planned = lf.collect_schema()[name]

    series: pl.Series | None = None
    produced: dict[str, pl.DataType] = {}
    for engine in engines:
        frame = lf.collect(engine=engine)
        produced[engine] = frame.schema[name]
        if series is None:
            series = frame[name]
        assert produced[engine] == planned, (
            f"engine={engine}: planner promised {planned!r} but execution "
            f"produced {produced[engine]!r}"
        )
        assert frame.height == df.height, (
            f"engine={engine}: {df.height} input rows produced {frame.height} "
            f"output rows — a schema that matches on a truncated result is not "
            f"the same promise"
        )

    distinct = {str(dtype) for dtype in produced.values()}
    assert len(distinct) == 1, f"engines disagree on the output dtype: {produced}"

    assert series is not None  # engines is never empty
    return series


def plan_or_reject(
    df: pl.DataFrame,
    build_expr: Callable[[], pl.Expr],
    *,
    name: str = "out",
    engines: Sequence[str] = ENGINES,
) -> ParityResult:
    """Tri-state form of :func:`assert_plan_equals_exec`, for sweeps.

    A cell may be rejected while building the expression or at plan time and
    that is fine — the sweep records it and moves on. Once the planner has
    published a dtype, though, execution raising is as much a contract
    violation as execution producing the wrong dtype, so both are assertion
    failures rather than a third kind of "rejected".
    """
    try:
        expr = build_expr()
    except (ValueError, TypeError) as exc:
        return ParityResult(Outcome.REJECTED_AT_BUILD, reason=str(exc))

    lf = df.lazy().with_columns(**{name: expr})
    try:
        planned = lf.collect_schema()[name]
    except Exception as exc:  # noqa: BLE001 - any plan-time refusal is fine
        return ParityResult(Outcome.REJECTED_AT_PLAN, reason=str(exc))

    series: pl.Series | None = None
    produced: dict[str, pl.DataType] = {}
    for engine in engines:
        try:
            frame = lf.collect(engine=engine)
        except Exception as exc:  # noqa: BLE001 - re-raised as a contract failure
            raise AssertionError(
                f"engine={engine}: the planner published {planned!r}, so execution "
                f"is obliged to produce it — instead it raised: {exc}"
            ) from exc
        produced[engine] = frame.schema[name]
        if series is None:
            series = frame[name]
        assert produced[engine] == planned, (
            f"engine={engine}: planner promised {planned!r} but execution "
            f"produced {produced[engine]!r}"
        )
        assert frame.height == df.height, (
            f"engine={engine}: {df.height} input rows produced {frame.height} rows"
        )

    distinct = {str(dtype) for dtype in produced.values()}
    assert len(distinct) == 1, f"engines disagree on the output dtype: {produced}"

    return ParityResult(Outcome.OK, planned=planned, series=series)


def assert_not_vacuous(results: dict[Any, ParityResult], label: str) -> None:
    """A sweep in which nothing executed proves nothing.

    Every sweep here treats rejection as acceptable, which means a change that
    makes *everything* reject — a renamed ``.sink()`` keyword, a builder that
    starts raising — would turn the sweep green while testing nothing. This
    repo has shipped that failure mode more than once, so each sweep asserts
    it still has at least one cell that made it all the way through.
    """
    ok = [key for key, result in results.items() if result.ok]
    assert ok, (
        f"{label}: no cell reached execution — every one was rejected at build "
        f"or plan time. The sweep is passing vacuously. Outcomes: "
        f"{ {k: r.outcome.name for k, r in results.items()} }"
    )


# ---------------------------------------------------------------------------
# Row composition
# ---------------------------------------------------------------------------

#: How a column's rows are laid out, given a sequence of distinct sample values.
#:
#: The plan is fixed before any row is seen, so it cannot depend on which rows
#: are null or on any individual row's shape. Each pattern below is a way for
#: that to go wrong:
#:
#: * ``all_null``   — no non-null row at all, so the runtime builders fall back
#:                    to the declared spec. The only path where the plan is
#:                    actually honoured, and the least exercised one.
#: * ``null_first`` / ``null_leading_run`` — the column's first row is null
#:                    while the first *non-null* row is further in.
#: * ``many_alternating`` — long enough to cross streaming morsel boundaries,
#:                    so different batches see different leading rows.
#: * ``heterogeneous*`` — rows of genuinely different shapes, so "the shape of
#:                    row 0" is not the shape of the column.
ROW_PATTERNS: dict[str, Callable[[Sequence[Any]], list[Any]]] = {
    "single": lambda v: [v[0]],
    "all_null": lambda v: [None, None],
    "null_first": lambda v: [None, v[0]],
    "null_last": lambda v: [v[0], None],
    "null_sandwich": lambda v: [None, v[0], None],
    "null_leading_run": lambda v: [None] * 8 + [v[0]],
    "many_alternating": lambda v: [None if i % 2 else v[i % len(v)] for i in range(64)],
    "heterogeneous": lambda v: list(v),
    "heterogeneous_null_first": lambda v: [None, *v],
}

#: Patterns whose non-null rows are all the same value. Sweeps that pin an
#: exact output shape (the ``array`` sink without a normalising ``resize``)
#: use these; the heterogeneous ones need a shape-normalising pipeline.
HOMOGENEOUS_PATTERNS: tuple[str, ...] = (
    "single",
    "all_null",
    "null_first",
    "null_last",
    "null_sandwich",
    "null_leading_run",
    "many_alternating",
)

#: The two patterns whose rows differ in shape from one another.
HETEROGENEOUS_PATTERNS: tuple[str, ...] = (
    "heterogeneous",
    "heterogeneous_null_first",
)


def rows_for(pattern: str, values: Sequence[Any]) -> list[Any]:
    """Materialise *pattern* over *values* (which must hold >= 3 samples)."""
    assert len(values) >= 3, "row patterns index up to three distinct samples"
    return ROW_PATTERNS[pattern](values)


def frame(
    rows: Sequence[Any],
    dtype: pl.DataType,
    *,
    name: str = "img",
    chunked: bool = True,
) -> pl.DataFrame:
    """Build a one-column frame of *rows* with an explicit *dtype*.

    The dtype is required, not inferred: an all-null list infers as
    ``pl.Null``, which is a different column type than the Binary/List/Array
    the pipeline is being asked about, and would quietly test something else.

    When *chunked* and there is more than one row, the frame is assembled from
    two chunks split at an offset that does not line up with the null pattern,
    so the streaming engine sees a morsel boundary mid-column.
    """
    schema = {name: dtype}
    if not chunked or len(rows) < 2:
        return pl.DataFrame({name: list(rows)}, schema=schema)

    split = 1 if len(rows) == 2 else len(rows) // 3 or 1
    return pl.concat(
        [
            pl.DataFrame({name: list(rows[:split])}, schema=schema),
            pl.DataFrame({name: list(rows[split:])}, schema=schema),
        ],
        rechunk=False,
    )


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------


#: The sinks that re-encode a buffer through an image codec.
IMAGE_CODEC_SINKS: frozenset[str] = frozenset({"png", "jpeg", "webp", "tiff"})


def encodable_by_image_codec(pipe: Any) -> bool:
    """Whether *pipe*'s output can go through png/jpeg/webp/tiff at all.

    Those four sinks enforce a dtype and a rank precondition inside the
    encoder rather than in the planner, so a pipeline that violates either
    plans as ``Binary`` and then raises at ``collect()``. That is one finding
    with one cause and it is pinned once, in
    ``test_schema_parity_encoder_sinks.py``; the broad sweeps skip the cells it
    owns instead of each rediscovering it. u8 at rank 2 or 3 clears all four.
    """
    return pipe.output_dtype() == "u8" and pipe._expected_ndim in (2, 3)


def leaf_dtype(dtype: pl.DataType) -> pl.DataType:
    """Peel nested List/Array wrappers down to the innermost element dtype."""
    while isinstance(dtype, (pl.List, pl.Array)):
        dtype = dtype.inner
    return dtype


def nesting_depth(dtype: pl.DataType) -> int:
    """How many List/Array wrappers *dtype* carries."""
    depth = 0
    while isinstance(dtype, (pl.List, pl.Array)):
        depth += 1
        dtype = dtype.inner
    return depth


def array_dims(dtype: pl.DataType) -> list[int]:
    """The fixed dimensions of a (possibly nested) ``pl.Array``, outermost first."""
    dims: list[int] = []
    while isinstance(dtype, pl.Array):
        dims.append(dtype.size)
        dtype = dtype.inner
    return dims
