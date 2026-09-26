"""The one place tests read a pipeline's *planned* state.

The planner's state is an implementation detail: a pipeline's Rust ``Plan``
(its source, typed ops and the state at every op boundary). Tests that assert
on planned domain, dtype, rank or shape hints, or on the ops and source, read
them through :func:`planned`, :func:`ops_of` and :func:`source_of`, so a change
to how the plan is held changes this module and nothing else in the suite.

Values are plain data: a hint is its literal size, the string ``"expr"`` when
it is a per-row expression, or ``None`` when unknown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polars_cv import LazyPipelineExpr, Pipeline

#: Marker for a hint that is known to come from a per-row expression.
EXPR = "expr"


@dataclass(frozen=True)
class PlanView:
    """What the planner believes a pipeline produces."""

    domain: str
    dtype: str
    ndim: int | None
    height: int | str | None
    width: int | str | None
    channels: int | str | None

    @property
    def hw(self) -> tuple[int | str | None, int | str | None]:
        return (self.height, self.width)


def _pipeline(p: "Pipeline | LazyPipelineExpr") -> "Pipeline":
    inner = getattr(p, "_pipeline", None)
    return inner if inner is not None else p  # type: ignore[return-value]


def planned(p: "Pipeline | LazyPipelineExpr") -> PlanView:
    """The planned output state of *p* (a ``Pipeline`` or lazy node)."""
    state = _pipeline(p)._state
    height, width, channels = state.dims
    return PlanView(
        domain=state.domain,
        dtype=state.dtype,
        ndim=state.ndim,
        height=height,
        width=width,
        channels=channels,
    )


@dataclass(frozen=True)
class OpView:
    """One appended op: its name and its parameters as plain values.

    A literal parameter is its value (lists element-wise); a per-row
    expression is :data:`EXPR`. Independent of the wire encoding, which the
    typed-op migration changes.
    """

    op: str
    params: dict[str, Any]


def _plain(value: Any) -> Any:
    if isinstance(value, dict) and value.keys() == {"$slot"}:
        return EXPR
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def ops_of(p: "Pipeline | LazyPipelineExpr") -> list[OpView]:
    """Each op appended to *p*, in order (an absent optional field omitted)."""
    views = []
    for text in _pipeline(p)._plan.ops_json():
        op = json.loads(text)
        name = op.pop("op")
        views.append(
            OpView(name, {k: _plain(v) for k, v in op.items() if v is not None})
        )
    return views


def op_names(p: "Pipeline | LazyPipelineExpr") -> list[str]:
    """The name of each op appended to *p*, in order."""
    return [op.op for op in ops_of(p)]


class SourceView:
    """A source's settings by name, as the caller passed them (a per-row one
    as its expression).

    An absent setting reads as its default (``on_error`` is ``"raise"``,
    everything else ``None``), and ``cloud_options`` as a ``CloudOptions``.
    """

    def __init__(self, pipeline: "Pipeline") -> None:
        self._pipeline = pipeline
        self._settings = pipeline._unwire(json.loads(pipeline._plan.source_json()))

    @property
    def format(self) -> Any:
        from polars_cv._types import SourceFormat

        return SourceFormat(self._settings["format"])

    def to_dict(self, slot_of: Any) -> dict[str, Any]:
        return self._pipeline._to_spec_dict(slot_of)["source"]

    def __getattr__(self, name: str) -> Any:
        from polars_cv._types import normalize_cloud_options

        settings = self._settings
        if name == "cloud_options":
            return normalize_cloud_options(settings.get("cloud_options"))
        value = settings.get(name)
        return ("raise" if name == "on_error" else None) if value is None else value


def source_of(p: "Pipeline | LazyPipelineExpr") -> "SourceView | None":
    """*p*'s source settings (``None`` for a continuation pipeline)."""
    pipeline = _pipeline(p)
    return SourceView(pipeline) if pipeline._plan.has_source else None


def op_json(p: "Pipeline | LazyPipelineExpr", index: int) -> str:
    """The wire JSON of one appended op, slots numbered over *p*'s own
    expression table."""
    return _pipeline(p)._plan.ops_json()[index]


def exprs_of(p: "Pipeline | LazyPipelineExpr") -> list[Any]:
    """The expressions *p*'s slots name, by slot number."""
    return list(_pipeline(p)._exprs)
