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


def ops_of(p: "Pipeline | LazyPipelineExpr") -> list[dict]:
    """The wire form of each op appended to *p*, in order."""
    return [op.to_dict() for op in _pipeline(p)._ops]
