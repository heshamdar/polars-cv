"""The one place tests read a pipeline's *planned* state.

The planner's state is an implementation detail that the typed-op migration
moves from Python fields into a Rust ``Plan`` (``TYPED_OPS_PLAN.md``, P7). Tests
that assert on planned domain, dtype, rank or shape hints read them through
:func:`planned` and :func:`ops_of`, so that move changes this module and nothing
else in the suite.

Values are plain data: a hint is its literal size, the string ``"expr"`` when
it is a per-row expression, or ``None`` when unknown.
"""

from __future__ import annotations

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


def _hint(value: Any) -> int | str | None:
    if value is None:
        return None
    if value.is_expr:
        return EXPR
    return value.value


def planned(p: "Pipeline | LazyPipelineExpr") -> PlanView:
    """The planned output state of *p* (a ``Pipeline`` or lazy node)."""
    pipe = _pipeline(p)
    hints = pipe._shape_hints
    return PlanView(
        domain=pipe._current_domain,
        dtype=pipe._output_dtype,
        ndim=pipe._expected_ndim,
        height=_hint(hints.height),
        width=_hint(hints.width),
        channels=_hint(hints.channels),
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
    if hasattr(value, "is_expr") and hasattr(value, "value"):  # a ParamValue
        return EXPR if value.is_expr else _plain(value.value)
    if isinstance(value, dict) and value.get("type") in ("literal", "expr"):
        return EXPR if value["type"] == "expr" else _plain(value.get("value"))
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def ops_of(p: "Pipeline | LazyPipelineExpr") -> list[OpView]:
    """Each op appended to *p*, in order."""
    return [
        OpView(op.op, {name: _plain(v) for name, v in op.params.items()})
        for op in _pipeline(p)._ops
    ]


def op_names(p: "Pipeline | LazyPipelineExpr") -> list[str]:
    """The name of each op appended to *p*, in order."""
    return [op.op for op in ops_of(p)]


def source_of(p: "Pipeline | LazyPipelineExpr") -> Any:
    """*p*'s source specification (``None`` for a continuation pipeline).

    Returned as-is until the typed-source phase (P4) gives it a stable view.
    """
    return _pipeline(p)._source


def op_json(p: "Pipeline | LazyPipelineExpr", index: int) -> str:
    """The wire JSON of one appended op, for tests of the op-level FFI.

    Only the ``op_*`` FFI tests need the wire form; they are rewritten when the
    planner moves into Rust (``TYPED_OPS_PLAN.md``, P7) and this goes with them.
    """
    import json

    return json.dumps(_pipeline(p)._ops[index].to_dict())
