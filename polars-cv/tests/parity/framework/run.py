"""Execute a case along any combination of axes.

A *case* is some input images (one per row, ``None`` for a null row) and a
sequence of :class:`Step` s. How it is executed is an :class:`Axes` value:
which source carries the images, which sink reads the result back, eager or
lazy, which streaming engine, whether parameters are literals or expressions,
how the pipeline is composed, and whether the optimizer runs.

The invariance suite asserts the result does not depend on the axes. The
reference suite executes along the default axes and compares against NumPy,
OpenCV and PIL. Both go through :func:`execute`, so an axis cannot be
exercised by one and silently skipped by the other.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from polars_cv import OptFlags, Pipeline
from polars_cv._optimize import PASS_NAMES
from tests._expr_param_cases import expression_eligible_parameters
from tests._plan_view import planned
from tests.parity.framework.io import (
    SINKS,
    SOURCES,
    OutputInfo,
    SourceSpec,
    source_pipeline,
)

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerRow:
    """A parameter value that differs per row (one entry per input row)."""

    values: tuple[Any, ...]

    def __repr__(self) -> str:
        return f"PerRow({list(self.values)!r})"


@dataclass(frozen=True)
class Step:
    """One ``Pipeline`` method call: its name and its literal arguments.

    A value may be a :class:`PerRow`, in which case it can only be passed as
    an expression (the ``literal`` parameter style cannot express it).
    """

    method: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def value_for_row(self, name: str, row: int) -> Any:
        """The literal value parameter *name* takes on *row*."""
        value = self.params[name]
        return value.values[row] if isinstance(value, PerRow) else value

    def literal_for_row(self, row: int) -> Step:
        """This step with every :class:`PerRow` resolved to *row*'s value."""
        return Step(self.method, {k: self.value_for_row(k, row) for k in self.params})

    @property
    def varies(self) -> bool:
        """Whether any parameter differs per row."""
        return any(isinstance(v, PerRow) for v in self.params.values())

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f".{self.method}({args})"


def render_chain(steps: Sequence[Step]) -> str:
    """The chain as the Python a user would write, for failure reports."""
    return "".join(repr(s) for s in steps)


_ELIGIBLE: dict[str, str] | None = None


def expression_eligible(method: str, name: str) -> str | None:
    """The annotation of ``method.name`` if it admits an expression, else ``None``.

    Read from the live ``Pipeline`` signatures through the same helper the
    expression-parameter ratchet uses, so this harness and that one cannot
    disagree about which parameters are per-row.
    """
    global _ELIGIBLE
    if _ELIGIBLE is None:
        _ELIGIBLE = expression_eligible_parameters()
    return _ELIGIBLE.get(f"{method}.{name}")


def admits_expression(annotation: str | None, value: Any) -> bool:
    """Whether *value* (a :class:`PerRow`'s values alike) can be an expression
    of a parameter annotated *annotation*: a scalar where the annotation
    admits an expression, a sequence where its elements do
    (``Sequence[FloatOrExpr]``). A parameter that is a scalar or a list
    (``histogram(bins=)``: a per-row count, or literal edges) admits one only
    in its scalar form."""
    if annotation is None:
        return False
    values = value.values if isinstance(value, PerRow) else (value,)
    if all(isinstance(v, (list, tuple)) for v in values):
        return "OrExpr]" in annotation
    if any(isinstance(v, (list, tuple)) for v in values):
        return False
    scalars = [p.strip() for p in annotation.split("|") if "[" not in p]
    return any(p.endswith("OrExpr") or p == "pl.Expr" for p in scalars)


def _is_sequence_param(annotation: str) -> bool:
    return any(t in annotation for t in ("Sequence[", "tuple[", "list["))


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

#: ``eager`` is ``DataFrame.select`` pinned to the in-memory engine (see
#: :func:`_collect`); the others are ``LazyFrame.collect(engine=...)``.
ENGINES = ("eager", "in-memory", "streaming")

#: How a parameter reaches the op.
#:
#: * ``literal`` — the Python value.
#: * ``column`` — ``pl.col`` of a column holding the value.
#: * ``lit`` — ``pl.lit(value)``: an expression with no column behind it.
#: * ``derived`` — ``pl.col(...).fill_null(value)``: a computed expression.
PARAM_STYLES = ("literal", "column", "lit", "derived")

#: How the steps are put together.
#:
#: * ``chain`` — one ``Pipeline``.
#: * ``continuation`` — split in two, the second half a source-less
#:   ``Pipeline`` continued with ``.pipe()``.
#: * ``aliased`` — every prefix aliased and sunk from one multi-output graph.
#: * ``materialized`` — every step in its own graph, the buffer materialized
#:   between them through a ``blob`` round trip (nothing can fuse).
COMPOSITIONS = ("chain", "continuation", "aliased", "materialized")


def _optimization_values() -> tuple[str, ...]:
    """Every optimizer setting the axis sweeps, from the pass registry.

    ``all`` and ``none``, then each pass alone (``only:<pass>``) and each pass
    removed from the full set (``without:<pass>``) — the same shape as
    ``test_optimize_equivalence``'s flag subsets. Read from
    ``polars_cv._optimize.PASS_NAMES`` (generated from the Rust pass
    catalogue), so a new pass joins the sweep without an edit here.
    """
    names = tuple(PASS_NAMES)
    return (
        "all",
        "none",
        *(f"only:{n}" for n in names),
        *(f"without:{n}" for n in names),
    )


OPTIMIZATION = _optimization_values()


def opt_flags_for(value: str) -> OptFlags:
    """The :class:`OptFlags` an optimize-axis value names."""
    if value == "all":
        return OptFlags.all()
    if value == "none":
        return OptFlags.none()
    kind, _, name = value.partition(":")
    if name not in PASS_NAMES or kind not in ("only", "without"):
        msg = f"unknown optimize setting {value!r}; expected one of {OPTIMIZATION}"
        raise ValueError(msg)
    on = kind == "without"
    flags = {n: on for n in PASS_NAMES}
    flags[name] = not on
    return OptFlags(**flags)


@dataclass(frozen=True)
class Axes:
    """How to execute a case. The defaults are the reference execution."""

    source: str = "array"
    sink: str = "numpy"
    engine: str = "eager"
    params: str = "literal"
    composition: str = "chain"
    optimize: str = "all"
    chunked: bool = False

    def __post_init__(self) -> None:
        for value, allowed in (
            (self.source, SOURCES),
            (self.sink, SINKS),
            (self.engine, ENGINES),
            (self.params, PARAM_STYLES),
            (self.composition, COMPOSITIONS),
            (self.optimize, OPTIMIZATION),
        ):
            if value not in allowed:
                msg = f"unknown axis value {value!r}; expected one of {list(allowed)}"
                raise ValueError(msg)

    def but(self, **changes: Any) -> Axes:
        """A copy with some axes changed."""
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


class _Params:
    """Turns a step's parameters into literals or expressions, per style."""

    def __init__(self, style: str, n_rows: int) -> None:
        self.style = style
        self.n_rows = n_rows
        self.columns: dict[str, list[Any]] = {}

    def value(self, step_index: int, step: Step, name: str) -> Any:
        value = step.params[name]
        annotation = expression_eligible(step.method, name)
        as_expr = self.style != "literal" and admits_expression(annotation, value)
        if isinstance(value, PerRow) and not as_expr:
            msg = (
                f"{step!r}: {name} varies per row, which the {self.style!r} "
                "parameter style cannot express"
            )
            raise ValueError(msg)
        if not as_expr:
            return list(value) if isinstance(value, tuple) else value
        key = f"p{step_index}_{name}"
        if (
            isinstance(value, PerRow)
            and isinstance(value.values[0], (list, tuple))
            and _is_sequence_param(annotation)
        ):
            # A per-row list: one expression per element, each varying by row.
            first = value.values[0]
            exprs = [
                self._expr(f"{key}_{j}", PerRow(tuple(v[j] for v in value.values)))
                for j in range(len(first))
            ]
            return tuple(exprs) if isinstance(first, tuple) else exprs
        if isinstance(value, (list, tuple)) and _is_sequence_param(annotation):
            exprs = [
                self._expr(f"{key}_{j}", element) for j, element in enumerate(value)
            ]
            return tuple(exprs) if isinstance(value, tuple) else exprs
        return self._expr(key, value)

    def _expr(self, key: str, value: Any) -> pl.Expr:
        if self.style == "lit":
            if isinstance(value, PerRow):
                msg = "a per-row value cannot be a pl.lit"
                raise ValueError(msg)
            return pl.lit(value)
        per_row = value.values if isinstance(value, PerRow) else (value,) * self.n_rows
        self.columns[key] = list(per_row)
        if self.style == "column":
            return pl.col(key)
        first = per_row[0]
        return pl.col(key).fill_null(first)


def apply_steps(
    pipe: Pipeline, steps: Sequence[Step], params: _Params, *, offset: int = 0
) -> Pipeline:
    """Append *steps* to *pipe*, parameters rendered by *params*."""
    for i, step in enumerate(steps, start=offset):
        method = getattr(pipe, step.method)
        kwargs = {name: params.value(i, step, name) for name in step.params}
        positional = _positional_only(method)
        args = [kwargs.pop(name) for name in positional if name in kwargs]
        pipe = method(*args, **kwargs)
    return pipe


def _positional_only(method: Any) -> list[str]:
    return [
        p.name
        for p in inspect.signature(method).parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_ONLY
    ]


# ---------------------------------------------------------------------------
# Executing
# ---------------------------------------------------------------------------


class PlanRefused(ValueError):
    """The planner refused the graph before any data moved.

    Either at ``sink()`` (``graph.check()``) or in Polars' own planning
    (``collect_schema()``) — a ``source("auto")`` learns its column's type
    only there. A refusal is the planner doing its job
    (``test_sink_contract.py``): an invariance variant it refuses is skipped,
    not failed. A *baseline* or reference execution it refuses is still an
    error — the case was drawn from inputs the op's contract admits.
    """


class NotApplicable(ValueError):
    """This harness cannot express the execution (a source that cannot carry
    the rows, a sink that cannot carry the output)."""


@dataclass
class Result:
    """What one execution produced.

    Attributes:
        rows: The decoded output per row (``None`` for a null row).
        info: The plan's view of the output (domain, dtype, shapes).
        planned: The Polars dtype the plan published for the output column.
        realized: The Polars dtype the column arrived with.
    """

    rows: list[Any]
    info: OutputInfo
    planned: pl.DataType
    realized: pl.DataType


def build_pipeline(
    source: SourceSpec,
    sample: np.ndarray,
    steps: Sequence[Step],
    params: _Params,
) -> Pipeline:
    """The single ``Pipeline`` for *steps* over *source*."""
    return apply_steps(source_pipeline(source, sample), steps, params)


def plan_info(pipe: Pipeline, shapes: Sequence[tuple[int, ...]]) -> OutputInfo:
    """The planner's domain and dtype for *pipe*, with the known row shapes."""
    return OutputInfo(
        domain=pipe.current_domain(),
        dtype=pipe.output_dtype(),
        shapes=tuple(shapes),
    )


def input_frame(
    source: SourceSpec,
    images: Sequence[np.ndarray | None],
    params: _Params,
    *,
    chunked: bool,
) -> pl.DataFrame:
    """The input frame: the image column plus any parameter columns."""
    values, column_type = source.frame_column(images)
    data: dict[str, pl.Series] = {"img": pl.Series("img", values, dtype=column_type)}
    for key, column in params.columns.items():
        data[key] = pl.Series(key, column)
    frame = pl.DataFrame(data)
    if not chunked or frame.height < 2:
        return frame
    # Two chunks, split off the null pattern, so the streaming engine sees a
    # morsel boundary mid-column (the same device as _schema_parity.frame).
    split = max(1, frame.height // 3)
    return pl.concat([frame[:split], frame[split:]], rechunk=False)


def execute(
    images: Sequence[np.ndarray | None],
    steps: Sequence[Step],
    axes: Axes = Axes(),
    *,
    shapes: Sequence[tuple[int, ...]] | None = None,
) -> Result:
    """Run *steps* over *images* along *axes* and decode the output.

    Args:
        images: One ``[H, W, C]`` array (or ``None``) per row. All non-null
            images share a dtype.
        steps: The operations.
        axes: How to execute.
        shapes: The non-null rows' output shapes, when the caller already
            knows them (a previous execution). Needed by the ``array`` sink's
            ``shape=`` and by the codec sinks' decoders; when omitted, a
            ``numpy`` execution along the same axes supplies them.
    """
    present = [im for im in images if im is not None]
    if not present:
        msg = "a case needs at least one non-null image"
        raise ValueError(msg)
    source = SOURCES[axes.source]
    if not source.can_carry(present):
        msg = (
            f"source {axes.source!r} cannot carry {present[0].dtype}{present[0].shape}"
        )
        raise NotApplicable(msg)
    sink = SINKS[axes.sink]

    params = _Params(axes.params, len(images))
    pipe = build_pipeline(source, present[0], steps, params)
    if shapes is None:
        if axes.sink == "numpy" or pipe.current_domain() != "buffer":
            shapes = []
        else:
            probe = execute(images, steps, axes.but(sink="numpy"))
            shapes = [r.shape for r in probe.rows if r is not None]
    info = plan_info(pipe, shapes)
    if not sink.applies(info):
        msg = f"sink {axes.sink!r} does not apply to {info}"
        raise NotApplicable(msg)
    sink_kwargs = sink.kwargs(info)
    opt_flags = opt_flags_for(axes.optimize)

    try:
        expr = _compose(
            source, present[0], steps, params, axes, sink.name, sink_kwargs, opt_flags
        )
    except ValueError as exc:
        raise PlanRefused(str(exc)) from exc
    frame = input_frame(source, images, params, chunked=axes.chunked)
    query = frame.lazy().select(out=expr)
    try:
        planned = query.collect_schema()["out"]
    except pl.exceptions.ComputeError as exc:
        raise PlanRefused(str(exc)) from exc
    out = _collect(frame, query, expr, axes.engine)
    series = out["out"]
    if axes.composition == "aliased":
        series = series.struct.field(_alias(len(steps)))
    rows = sink.decode(series, info)
    return Result(rows=rows, info=info, planned=planned, realized=out.schema["out"])


def _collect(
    frame: pl.DataFrame, query: pl.LazyFrame, expr: pl.Expr, engine: str
) -> pl.DataFrame:
    """Run *expr* on *engine*.

    ``eager`` is the DataFrame API — whose engine is otherwise whatever
    ``POLARS_ENGINE_AFFINITY`` says (the suite's conftest defaults it to
    streaming), which would make "eager" silently mean streaming. It is
    pinned to in-memory so the three axis values are three engines.
    """
    if engine != "eager":
        return query.collect(engine=engine)
    with pl.Config():
        pl.Config.set_engine_affinity("in-memory")
        return frame.select(out=expr)


def _alias(k: int) -> str:
    return f"s{k}"


def _blob_reader(upstream: Pipeline) -> Pipeline:
    """A ``blob`` source restating what the plan knew at *upstream*'s output.

    A materialized buffer arrives with no plan-time facts; restating the
    dtype, rank and sizes the single-pipeline plan held there means the next
    stage is planned from the same knowledge, so the only difference between
    the two compositions is the materialization itself.
    """
    view = planned(upstream)
    kwargs = {} if view.dtype == "auto" else {"dtype": view.dtype}
    reader = Pipeline().source("blob", **kwargs)
    if view.ndim is not None:
        dims = [d if isinstance(d, int) else None for d in view.dims]
        dims += [None] * (view.ndim - len(dims))
        reader = reader.assert_shape(dims=dims)
    return reader


def _compose(
    source: SourceSpec,
    sample: np.ndarray,
    steps: Sequence[Step],
    params: _Params,
    axes: Axes,
    sink: str,
    sink_kwargs: dict[str, Any],
    opt_flags: OptFlags,
) -> pl.Expr:
    col = pl.col("img")
    base = source_pipeline(source, sample)
    kind = axes.composition

    if kind == "chain" or (kind == "continuation" and len(steps) < 2):
        pipe = apply_steps(base, steps, params)
        return col.cv.pipe(pipe).sink(sink, opt_flags=opt_flags, **sink_kwargs)

    if kind == "continuation":
        k = len(steps) // 2
        head = apply_steps(base, steps[:k], params)
        tail = apply_steps(Pipeline(), steps[k:], params, offset=k)
        return (
            col.cv.pipe(head).pipe(tail).sink(sink, opt_flags=opt_flags, **sink_kwargs)
        )

    if kind == "aliased":
        node = col.cv.pipe(base).alias(_alias(0))
        outputs = {_alias(0): "blob"} if steps else {_alias(0): sink}
        for i, step in enumerate(steps, start=1):
            node = node.pipe(apply_steps(Pipeline(), [step], params, offset=i - 1))
            node = node.alias(_alias(i))
            # Intermediate buffers ride along as blobs: every prefix is an
            # output of the graph, which is what forces the shared prefix to
            # be computed once and read by several consumers.
            outputs[_alias(i)] = "blob"
        outputs[_alias(len(steps))] = sink
        if sink_kwargs:
            msg = "the aliased composition cannot pass per-output sink kwargs"
            raise ValueError(msg)
        return node.sink(outputs, opt_flags=opt_flags)

    if kind == "materialized":
        if not steps:
            return col.cv.pipe(base).sink(sink, opt_flags=opt_flags, **sink_kwargs)
        node = col.cv.pipe(base).sink("blob", opt_flags=opt_flags)
        fused = base  # the single-pipeline plan, read for what each barrier knew
        for i, step in enumerate(steps):
            reader = _blob_reader(fused)
            fused = apply_steps(fused, [step], params, offset=i)
            stage = apply_steps(reader, [step], params, offset=i)
            last = i == len(steps) - 1
            node = node.cv.pipe(stage).sink(
                sink if last else "blob",
                opt_flags=opt_flags,
                **(sink_kwargs if last else {}),
            )
        return node

    msg = f"unknown composition {kind!r}"  # pragma: no cover - Axes validates
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Graphs with two branches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryCase:
    """Two branches joined by a lazy-only binary op, with an optional tail.

    With *shared*, both branches read the ``left`` column — one source
    feeding two chains, which is where common-subexpression elimination
    shares the prefix between them. Otherwise the right branch reads a second
    column holding ``right``.
    """

    left: tuple[np.ndarray | None, ...]
    right: tuple[np.ndarray | None, ...]
    left_steps: tuple[Step, ...]
    right_steps: tuple[Step, ...]
    method: str
    tail: tuple[Step, ...] = ()
    shared: bool = False

    def describe(self) -> str:
        right_col = "img" if self.shared else "img2"
        return (
            f"pl.col('img').cv.pipe(src{render_chain(self.left_steps)})"
            f".{self.method}(pl.col('{right_col}').cv.pipe(src"
            f"{render_chain(self.right_steps)})){render_chain(self.tail)}"
        )


def execute_binary(
    case: BinaryCase, *, engine: str = "eager", optimize: str = "all"
) -> Result:
    """Execute a :class:`BinaryCase` (``array`` source, ``numpy`` sink)."""
    source = SOURCES["array"]
    present = [im for im in case.left if im is not None]
    params = _Params("literal", len(case.left))
    left_pipe = apply_steps(
        source_pipeline(source, present[0]), case.left_steps, params
    )
    right_sample = (
        present[0] if case.shared else next(im for im in case.right if im is not None)
    )
    right_pipe = apply_steps(
        source_pipeline(source, right_sample), case.right_steps, params
    )
    left = pl.col("img").cv.pipe(left_pipe)
    right = pl.col("img" if case.shared else "img2").cv.pipe(right_pipe)
    node = getattr(left, case.method)(right)
    if case.tail:
        node = node.pipe(apply_steps(Pipeline(), case.tail, params))
    opt_flags = opt_flags_for(optimize)
    try:
        expr = node.sink("numpy", opt_flags=opt_flags)
    except ValueError as exc:
        raise PlanRefused(str(exc)) from exc

    values, column_type = source.frame_column(case.left)
    data = {"img": pl.Series("img", values, dtype=column_type)}
    if not case.shared:
        values2, type2 = source.frame_column(case.right)
        data["img2"] = pl.Series("img2", values2, dtype=type2)
    frame = pl.DataFrame(data)
    query = frame.lazy().select(out=expr)
    try:
        planned_dtype = query.collect_schema()["out"]
    except pl.exceptions.ComputeError as exc:
        raise PlanRefused(str(exc)) from exc
    out = _collect(frame, query, expr, engine)
    info = OutputInfo(domain="buffer", dtype="auto", shapes=())
    rows = SINKS["numpy"].decode(out["out"], info)
    return Result(
        rows=rows, info=info, planned=planned_dtype, realized=out.schema["out"]
    )
