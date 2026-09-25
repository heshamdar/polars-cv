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
    #: A declaration (``assert_shape``, a canvas from another node) reached
    #: this lineage, so the sizes may rest on a claim.
    declared: bool = False

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
        declared=state.declared,
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


class SourceView:
    """A source's settings by name, independent of how the spec stores them.

    An absent setting reads as its default (``on_error`` is ``"raise"``,
    everything else ``None``); a contour canvas taken from another node reads
    as ``shape_node``, and ``cloud_options`` as a ``CloudOptions``.
    """

    def __init__(self, spec: Any) -> None:
        self._spec = spec

    @property
    def format(self) -> Any:
        return self._spec.format

    def to_dict(self, slot_of: Any) -> dict[str, Any]:
        return self._spec.to_dict(slot_of)

    def __getattr__(self, name: str) -> Any:
        from polars_cv._types import normalize_cloud_options

        settings = {key: param.value for key, param in self._spec.params.items()}
        if name == "shape_node":
            size = settings.get("size")
            return size if isinstance(size, str) else None
        if name == "cloud_options":
            return normalize_cloud_options(settings.get("cloud_options"))
        return settings.get(name, "raise" if name == "on_error" else None)


def source_of(p: "Pipeline | LazyPipelineExpr") -> "SourceView | None":
    """*p*'s source settings (``None`` for a continuation pipeline)."""
    spec = _pipeline(p)._source
    return None if spec is None else SourceView(spec)


def op_json(p: "Pipeline | LazyPipelineExpr", index: int) -> str:
    """The wire JSON of one appended op, for tests of the op-level FFI.

    Only the ``op_*`` FFI tests need the wire form; they are rewritten when the
    planner moves into Rust (``TYPED_OPS_PLAN.md``, P7) and this goes with them.
    """
    import json

    from polars_cv._types import planning_slots

    return json.dumps(_pipeline(p)._ops[index].to_dict(planning_slots))
