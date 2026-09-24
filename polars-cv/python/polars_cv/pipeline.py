"""
Pipeline builder for polars-cv.

This module provides the Pipeline class for building lazy image/array
processing pipelines that can be applied to Polars DataFrame columns.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import polars as pl

from polars_cv._ops_generated import OP_FIELDS, _OpsMixin
from polars_cv._types import (
    HINT_DIMS,
    CloudOptions,
    Domain,
    DType,
    FloatOrExpr,
    HashAlgorithm,
    IntOrExpr,
    NullParamPolicy,
    OpSpec,
    ParamValue,
    RowErrorPolicy,
    ScaleOrigin,
    ShapeAssertion,
    ShapeHints,
    SlotTable,
    SourceFormat,
    SourceSpec,
    _reject_expr,
    _validate_enum,
    is_supplied,
    normalize_cloud_options,
    planning_slots,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polars_cv._graph import PipelineGraph
    from polars_cv._optimize import OptFlags
    from polars_cv._types import SlotOf
    from polars_cv.lazy import LazyPipelineExpr


def _rotation_matrix(
    angle_deg: FloatOrExpr,
    center: tuple[FloatOrExpr, FloatOrExpr],
    scale: FloatOrExpr,
) -> list[FloatOrExpr]:
    """Build a 2x3 forward-mapping rotation+scale matrix around *center*.

    Matches OpenCV's ``getRotationMatrix2D(center, -angle_deg, scale)``
    convention where positive *angle_deg* = clockwise in image coordinates.

    Any argument may be a Polars expression. Only the trigonometry needs to
    know the difference — the remaining arithmetic is written with plain
    operators, which compose identically for floats and for ``pl.Expr``.

    Args:
        angle_deg: Rotation angle in degrees (positive = clockwise).
        center: ``(cx, cy)`` center of rotation.
        scale: Scale factor.

    Returns:
        Six-element list ``[a, b, tx, c, d, ty]`` (forward mapping).
    """
    cx, cy = center
    if not any(isinstance(v, pl.Expr) for v in (angle_deg, cx, cy, scale)):
        # All-literal: read the matrix from the Rust authority
        # (`AffineParams::rotation_matrix_2d`) rather than transliterating the
        # trig, so the formula lives in exactly one place. The `pl.Expr` path
        # below cannot -- the engine evaluates those operands per row at
        # execution, not at plan time -- and is the one guard-sanctioned copy.
        from polars_cv._lib import rotation_matrix_2d

        return list(
            rotation_matrix_2d(float(angle_deg), float(cx), float(cy), float(scale))  # ty: ignore[invalid-argument-type]
        )
    if isinstance(angle_deg, pl.Expr):
        rad = angle_deg.radians()
        cos_a = rad.cos() * scale
        sin_a = rad.sin() * scale
    else:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad) * scale
        sin_a = math.sin(rad) * scale
    tx = (1 - cos_a) * cx + sin_a * cy
    ty = -sin_a * cx + (1 - cos_a) * cy
    return [cos_a, -sin_a, tx, sin_a, cos_a, ty]


def _matrix_param_from_floats(values: "list[float]") -> "ParamValue":
    """Build a ``warp_affine`` ``matrix`` param from six literal floats.

    Matrix elements are serialized as individual ``ParamValue`` dicts so any of
    them may be a per-row expression; a fully-literal matrix (fusion output,
    converted-rotate matrix) still goes through the same per-element shape.
    """
    return ParamValue(
        is_expr=False,
        value=[ParamValue(is_expr=False, value=float(v)) for v in values],
    )


#: view-buffer's identity domain (`Domain::Any`): a step declaring it accepts
#: whatever it is handed, mirroring `Domain::accepts` on the Rust side. It is
#: deliberately *not* a member of the user-facing `Domain` enum — no pipeline is
#: ever *in* this domain, so `test_enum_parity_domain` excludes it from the
#: surfaced variant set. No step currently declares it — binary ops and
#: reductions list `["buffer", "vector"]` explicitly rather than opting out of
#: the check entirely — but the contract may return it, so the reader honours it.
_DOMAIN_ANY = "any"


def _source_param_defaults() -> "dict[str, Any]":
    """Each ``Pipeline.source`` keyword's default, read from its signature.

    The signature is the authority for what a default *is*, so "the caller
    passed this" cannot drift from what the function actually does with it.
    Cached: the signature never changes at runtime, and `source()` is on the
    builder's hot path.
    """
    global _SOURCE_DEFAULTS
    if _SOURCE_DEFAULTS is None:
        import inspect

        _SOURCE_DEFAULTS = {
            name: param.default
            for name, param in inspect.signature(Pipeline.source).parameters.items()
            if param.default is not inspect.Parameter.empty
        }
    return _SOURCE_DEFAULTS


_SOURCE_DEFAULTS: "dict[str, Any] | None" = None


def _op_contract_for(spec: "OpSpec") -> dict:
    """Read one operation's Rust contract (spatial rule, window flag, ...).

    The spatial-window pushdown's reader; appends go through ``plan_step``.
    """
    from polars_cv._lib import op_contract

    return op_contract(json.dumps(spec.to_dict(planning_slots)))


def _op_reads_sibling_nodes(op: "OpSpec") -> bool:
    """Whether ``op`` consumes another graph node's buffer.

    A binary op (``apply_mask``, ``add``) or ``channel_merge`` combines this
    node's buffer with a sibling node's at matching ``(y, x)``. Such an op is
    spatially ``Pointwise``, but hoisting a crop earlier past it would shrink only
    *this* operand and leave the sibling full-size, so a spatial window may not
    cross it whatever its spatial rule. The catalogue says which fields name
    another node (a ``node`` type, alone or in a list), so this reads the op's
    declared fields rather than a list of parameter names.
    """
    return any(_names_nodes(ty) for ty in OP_FIELDS.get(op.op, {}).values())


def _names_nodes(ty: "dict[str, Any]") -> bool:
    """Whether a catalogue field type holds a graph-node reference."""
    if ty["kind"] == "node":
        return True
    nested = [ty["inner"]] if "inner" in ty else ty.get("options", [])
    return any(_names_nodes(t) for t in nested)


def _output_shape_equals_input(
    out_dims: "Sequence[int | None]", entering_dims: "Sequence[int | None]"
) -> bool:
    """Whether an op's inferred output shape equals the shape entering it.

    Used by identity elimination to decide a ``WhenShapePreserved`` op. Two
    ``op_infer_shape`` conventions are folded in:

    * a **negative** output dim is ``op_infer_shape``'s "this is the unknown
      input axis, carried through unchanged" (e.g. a crop leaving the channel
      axis to the input), so it counts as preserved and matches any entering
      size;
    * a concrete output dim must equal the entering size exactly; an entering
      size that is unknown (``None``) therefore cannot match a concrete output,
      and an unknown output (``None``) is never treated as a match.

    So a full-frame crop or a same-shape reshape returns ``True`` while a partial
    crop or a real reshape returns ``False`` — and any unproven dimension keeps
    the op (the pass removes only what it can prove is a no-op).
    """
    if len(out_dims) != len(entering_dims):
        return False
    for out, enter in zip(out_dims, entering_dims):
        if out is not None and out < 0:
            continue  # preserved sentinel — same as the input dim
        if out is None or out != enter:
            return False
    return True


#: The spatial-window pushdown transfer function returns this when a window may
#: not cross an op — a hard stop, distinct from "crosses unchanged" (the window
#: itself). See :meth:`Pipeline._spatial_transfer`.
_SPATIAL_BARRIER = object()


def _asserted_rank(dims: "Sequence[int | None]") -> int:
    """Validate an ``assert_shape(dims=...)`` list and return the rank it pins.

    Entries are literal ``int``\\ s or ``None`` (dimension left unknown). An
    expression is refused rather than tracked: ``dims=`` publishes the output
    schema, and a per-row size is not a plan-time fact — ``height=`` remains
    available for a per-row dimension, where it correctly publishes nothing.

    Rank is capped at ``len(HINT_DIMS)`` because that is how many dimensions
    :class:`ShapeHints` tracks. Accepting a longer list would silently file
    dimension 0 under ``height`` and drop everything past dimension 2, which is
    a mis-assignment rather than a missing feature — so it is refused here.
    """
    if not isinstance(dims, (list, tuple)):
        msg = f"assert_shape(dims=...) must be a list of ints, got {dims!r}"
        raise ValueError(msg)
    dims = list(dims)
    if not dims:
        msg = "assert_shape(dims=[]) declares a rank-0 output, which no sink produces."
        raise ValueError(msg)
    if len(dims) > len(HINT_DIMS):
        msg = (
            f"assert_shape(dims=...) supports up to {len(HINT_DIMS)} dimensions "
            f"({', '.join(HINT_DIMS)}), got {len(dims)}. Higher-rank shapes are "
            f"not tracked by the planner; pass the shape to the sink instead "
            f"(.sink('array', shape=[...]))."
        )
        raise ValueError(msg)
    for axis, size in enumerate(dims):
        if size is None:
            continue
        _reject_expr(size, f"'dims[{axis}]'")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            msg = (
                f"assert_shape(dims=...) entry {axis} must be a positive int "
                f"or None, got {size!r}"
            )
            raise ValueError(msg)
    return len(dims)


def _same(value: "Any") -> "Any":
    """Carry a field across a copy by reference (immutable or deliberately shared)."""
    return value


#: Every field of a :class:`Pipeline`'s state, and how a copy of it is made.
#:
#: This is the single authority for "what *is* a Pipeline's state", and
#: :meth:`Pipeline._copy_state_from` is the only reader. Every constructor of a
#: derived pipeline — ``_clone``, ``_create_sub_pipeline``, and CSE's
#: ``_create_shared_node`` in ``_graph.py`` — copies the whole state through it
#: and *then* overrides the few fields it means to change, so a new field is
#: carried by default instead of by remembering three call sites.
#:
#: It replaced three hand-written field-by-field copies that had already
#: drifted: ``_create_sub_pipeline`` copied 11 of the 14 fields, so the public
#: ``Pipeline.on_error(...).to_graph(...)`` silently executed under ``"raise"``
#: — ``PipelineGraph._to_dict`` reads the policy off the *node* pipeline, and
#: the sub-pipeline it built had the default. Prefer a mechanism callers cannot
#: step around to a convention each caller must re-enact.
#:
#: A field added to ``__init__`` and not here fails
#: ``test_pipeline_state_copy_is_complete``.
_STATE_COPIERS: "dict[str, Callable[[Any], Any]]" = {
    # Specs and tracked scalars: immutable, shared by reference.
    "_source": _same,
    "_current_domain": _same,
    "_output_dtype": _same,
    "_expected_ndim": _same,
    "_on_error": _same,
    "_on_null_param": _same,
    "_shape_declared": _same,
    # Containers: copied so the clone cannot mutate its origin.
    "_ops": list,
    "_expr_refs": list,
    "_asserted_dims": set,
    # Per-op entering states: immutable records, so a shallow copy suffices.
    "_entering": list,
    # `pl.Expr` / `LazyPipelineExpr` elements are shared deliberately — they are
    # graph identities, and deep-copying one would break node reference.
    "_shape_refs": list,
    # Mutable value objects the planner writes through: deep-copied.
    "_shape_hints": copy.deepcopy,
    "_assertions": copy.deepcopy,
}


@dataclass(frozen=True)
class PlanState:
    """The planner's tracked state at one op boundary.

    What :meth:`Pipeline._push_op` advances and what a replay starts from:
    domain, dtype, rank, the shape hints, which of them the user asserted, and
    whether a shape declaration reached this lineage. ``hints`` is a private
    copy, never mutated.
    """

    domain: str
    dtype: str
    ndim: "int | None"
    hints: ShapeHints
    asserted_dims: "frozenset[str]"
    shape_declared: bool


class _Position(NamedTuple):
    """What the planner keeps for one op: the state entering it, and a binary
    op's other operand dtype (its two-input dtype rule reads it on replay)."""

    state: PlanState
    other_dtype: "str | None"


def _assertion_window(
    assertions: "dict[int, ShapeAssertion]", start: int, end: int
) -> "dict[int, ShapeAssertion]":
    """The assertions of op boundaries ``start..=end``, re-keyed from 0.

    Assertions are keyed by op *boundary* (an assertion at ``k`` applies after
    op ``k - 1``), so a slice ``[start, end)`` keeps both of its end boundaries.
    """
    return {
        i - start: copy.deepcopy(a) for i, a in assertions.items() if start <= i <= end
    }


def _encode_field(
    p: "Pipeline", value: Any, ty: "dict[str, Any]", where: str
) -> "ParamValue | None":
    """Encode one typed-op argument per its catalogue type (``OP_FIELDS``).

    ``None`` for an absent optional field. A sequence field is encoded element
    by element, so each element may be an expression; anything else in its
    place is passed through for the Rust definition to reject. An expression
    for a structural (literal-only) field is refused by ``ParamValue``.
    """
    kind = ty["kind"]
    if kind == "optional":
        return None if value is None else _encode_field(p, value, ty["inner"], where)
    if kind == "column":
        # An input column the step reads as data (`label_reduce(contours=)`):
        # only an expression has a column to give.
        if not isinstance(value, pl.Expr):
            msg = f"{where} must be a Polars expression, got {type(value).__name__}"
            raise TypeError(msg)
        return p._track_expr(value)
    if kind == "node":
        # An operand expression crosses as its node id; the graph wiring
        # (upstream edges) is the lazy layer's job, not the op's.
        from polars_cv.lazy import LazyPipelineExpr

        if isinstance(value, LazyPipelineExpr):
            value = value._node_id
        return ParamValue(is_expr=False, value=value)
    if kind == "one_of":
        # The options differ in shape: a sequence picks the sequence option.
        wants_seq = _is_sequence(value)
        for option in ty["options"]:
            if (option["kind"] in ("array", "list")) == wants_seq:
                return _encode_field(p, value, option, where)
        msg = f"{where}: no catalogue option takes {type(value).__name__}"
        raise TypeError(msg)
    if kind in ("array", "list") and _is_sequence(value):
        return ParamValue(
            is_expr=False,
            value=[
                _encode_field(p, v, ty["inner"], f"{where}[{i}]")
                for i, v in enumerate(value)
            ],
        )
    if kind == "scalar" and ty["per_row"]:
        return p._track_expr(value)
    return ParamValue(is_expr=False, value=value)


def _is_sequence(value: Any) -> bool:
    """A list-like argument (list, tuple, numpy array), not a string or expr."""
    return not isinstance(value, (str, bytes, pl.Expr)) and hasattr(value, "__iter__")


class Pipeline(_OpsMixin):
    """
    Modular pipeline builder for image and array operations.

    A pipeline defines a sequence of operations that can be applied to a Polars
    expression using the `.cv.pipe()` accessor. The pipeline is executed when
    `.sink()` is called on the resulting expression.

    Parameters that carry a *value* — sizes, offsets, factors, thresholds, fill
    values, kernel coefficients, and non-structural enums such as ``filter`` or
    ``interpolation`` — accept either a literal or a Polars expression, resolved
    per row at execution time.

    Parameters that fix the output **shape, rank, or dtype** at planning time
    are literal-only, so the lazy schema cannot desync from the produced data:
    reduction ``axis``, ``perceptual_hash(hash_size)``, ``reshape``/``transpose``
    /``flip`` axis lists, ``rotate(expand)``, ``cast(dtype)``,
    ``normalize(method``/``out_dtype)`` and ``histogram(closed``/``output)``.
    Passing an expression to one of those raises ``TypeError`` at build time.

    Example:
        ```python
        >>> from polars_cv import Pipeline
        >>> import polars as pl
        >>>
        >>> # Define a reusable pipeline (without a sink)
        >>> preprocess = (
        ...     Pipeline()
        ...     .source("image_bytes")
        ...     .resize(height=224, width=224)
        ...     .grayscale()
        ... )
        >>>
        >>> # Apply to a DataFrame and choose the output format at the sink
        >>> df = pl.DataFrame({"image": [img_bytes]})
        >>> result = df.with_columns(
        ...     processed=pl.col("image").cv.pipe(preprocess).sink("numpy")
        ... )
        ```

    Pipelines support typed domain tracking for transitions between images,
    geometry, and numeric results:
    - buffer: Image/array data (default)
    - contour: Polygon geometry
    - scalar: Single numeric values
    - vector: Multiple numeric values (e.g., bounding boxes)
    """

    def __init__(self) -> None:
        """Initialize an empty pipeline."""
        self._source: SourceSpec | None = None
        self._shape_hints: ShapeHints = ShapeHints()
        self._ops: list[OpSpec] = []
        self._expr_refs: list[pl.Expr] = []
        # Domain tracking for typed pipelines
        self._current_domain: str = Domain.BUFFER.value
        # Output dtype tracking — "auto" means unknown until runtime or
        # until an operation with a deterministic output dtype resolves it.
        self._output_dtype: str = "auto"
        # Number of dimensions tracking
        self._expected_ndim: int | None = None
        # The state entering each op, in step with `_ops`. A slice, a
        # reorder or a deletion of the ops replays them from one of these
        # (`_replay`), and identity elimination judges an op against its own.
        self._entering: list[_Position] = []
        # Shape dimensions the user asserted via assert_shape(), keyed by the
        # op position the assertion was written at. Distinguishes a user
        # assertion (authoritative, must survive a continuation replay) from a
        # hint an operation computed (recomputed by the replay).
        self._assertions: dict[int, ShapeAssertion] = {}
        # Which hints currently hold a value the *user* asserted rather than
        # one the ops' contracts inferred. Recomputed with the hints: cleared
        # by the schema fold, re-filled by `_apply_assertions_at`. Published as
        # `shape_asserted` so a plan/exec divergence is attributed to whoever
        # actually made the claim.
        self._asserted_dims: set[str] = set()
        # Sticky: has any shape declaration (an assert_shape, or a shape_ref
        # canvas) been applied anywhere in this pipeline's lineage? Unlike
        # `_asserted_dims` it is never cleared by the schema fold, because a
        # declared H/W stays a *claim* after flowing through a shape-preserving
        # op. Identity elimination reads it to refuse proving a shape-preserving
        # no-op from hints that may rest on a claim rather than a fact. Carried
        # into lazy continuations, whose hints are seeded from the upstream node.
        self._shape_declared: bool = False
        # Per-row error policy for the executed graph ("raise" by default).
        self._on_error: str = "raise"
        # What a null in a per-row expression parameter means ("raise" by
        # default). Independent of _on_error — see on_null_param().
        self._on_null_param: str = "raise"
        # LazyPipelineExpr nodes referenced by ops (e.g. rasterize(shape=...));
        # consumers wiring this pipeline into a graph add them as upstream
        # dependencies so the referenced node executes first.
        self._shape_refs: "list[LazyPipelineExpr]" = []

    def _track_expr(self, value: IntOrExpr | FloatOrExpr) -> ParamValue:
        """
        Create a ParamValue and track the expression if needed.

        Args:
            value: Literal or expression value.

        Returns:
            ParamValue instance.
        """
        param = ParamValue.from_arg(value)
        if param.is_expr and isinstance(value, pl.Expr):
            # Track each distinct expression once (by meta.eq, never by text).
            if not any(e is value or e.meta.eq(value) for e in self._expr_refs):
                self._expr_refs.append(value)
        return param

    def _copy_state_from(self, other: "Pipeline") -> None:
        """Copy *other*'s entire state onto this pipeline.

        The one way a derived pipeline inherits state, driven by
        :data:`_STATE_COPIERS`. Callers that mean to change a field override it
        *after* this returns, so anything they do not mention survives — the
        opposite of building a pipeline up field by field, which is how
        ``_create_sub_pipeline`` came to drop ``on_error`` / ``on_null_param``.
        """
        for name, copier in _STATE_COPIERS.items():
            setattr(self, name, copier(getattr(other, name)))

    def _clone(self) -> "Pipeline":
        """Create a shallow clone of this pipeline for chaining."""
        new = Pipeline()
        new._copy_state_from(self)
        return new

    def on_error(self, policy: str) -> "Pipeline":
        """
        Set the per-row error policy for the executed pipeline graph.

        Controls what happens when producing a single row fails (source
        decode, operation execution, or output encoding):

        - ``"raise"`` (default): the first failing row fails the whole
          expression with its error.
        - ``"null"``: failing rows yield null for **all** of the graph's
          outputs; other rows are unaffected.
        - ``"null_with_message"``: as ``"null"``, plus the output becomes a
          struct with a reserved ``_error`` string field carrying the failure
          message for bad rows (null for good rows). Single-output pipelines
          become a two-field struct (``_output`` + ``_error``).

        This is a graph-level setting: when pipelines are composed
        (``merge_pipe``, binary ops), all composed pipelines must agree on
        the policy.

        Note: the per-source ``source(..., on_error="null")`` setting remains
        independent — it nulls only the outputs that depend on a failing
        source decode, while this policy covers any error producing the row.

        Args:
            policy: One of ``"raise"``, ``"null"``, ``"null_with_message"``.

        Returns:
            New Pipeline with the error policy set.

        Example:
            >>> pipe = Pipeline().source("image_bytes").resize(height=224, width=224).on_error("null")
        """
        valid = tuple(p.value for p in RowErrorPolicy)
        if policy not in valid:
            msg = f"on_error must be one of {valid}, got '{policy}'"
            raise ValueError(msg)
        new = self._clone()
        new._on_error = policy
        return new

    def on_null_param(self, policy: str) -> "Pipeline":
        """
        Set what a null in a per-row expression parameter means.

        Parameters that take a Polars expression are read from ordinary
        columns, which may contain nulls:

        - ``"raise"`` (default): a null parameter fails the whole expression.
        - ``"null"``: rows whose parameter is null yield null, exactly as a
          null *input image* already does. Other rows are unaffected.

        Under ``"null"`` only the outputs that actually depend on the affected
        operation go null — unlike ``on_error("null")``, which nulls every
        output of a failing row. The two settings are independent: this one
        does not weaken error reporting for decode, encode or genuine
        operation failures.

        To substitute a **fallback value** instead of nulling, fill the null in
        the expression itself — ``pl.col("scale").fill_null(1.0)`` — which is
        the idiomatic Polars way and needs nothing from this API.

        This is a graph-level setting. Only a non-default policy is collected
        from the composed pipelines, so an explicit ``"raise"`` is
        indistinguishable from leaving it unset: composing a ``"null"``
        pipeline with a ``"raise"`` one gives the whole graph ``"null"``,
        rather than being rejected as a conflict. (With only two values there
        is no combination that can conflict; ``on_error``, which has two
        non-default values, does reject genuine disagreement.)

        Args:
            policy: One of ``"raise"``, ``"null"``.

        Returns:
            New Pipeline with the null-parameter policy set.

        Example:
            >>> pipe = (
            ...     Pipeline()
            ...     .source("image_bytes")
            ...     .resize(height=pl.col("h"), width=pl.col("w"))
            ...     .on_null_param("null")
            ... )
        """
        valid = tuple(p.value for p in NullParamPolicy)
        if policy not in valid:
            msg = f"on_null_param must be one of {valid}, got '{policy}'"
            raise ValueError(msg)
        new = self._clone()
        new._on_null_param = policy
        return new

    def current_domain(self) -> str:
        """
        Get the current data domain of the pipeline.

        Returns:
            Current domain: "buffer", "contour", "scalar", or "vector".
        """
        return self._current_domain

    def output_dtype(self) -> str:
        """
        Get the expected output dtype of the pipeline.

        This is the dtype of the buffer after all operations have been applied.
        Used for static type inference in list/array sinks.  May be ``"auto"``
        if the dtype has not yet been determined (e.g. an image source with
        no dtype-fixing operation applied).

        Returns:
            Output dtype string: ``"u8"``, ``"f32"``, ``"f64"``, ``"auto"``, etc.
        """
        return self._output_dtype

    def output_encoding(self) -> str | None:
        """Get the sink encoding selector for this pipeline's output, if any.

        Most outputs are encoded by their (domain, sink-format) pair. A few share
        a domain but need a distinct Polars schema; this names that encoding so it
        can be carried alongside the domain rather than overloading it.

        Currently the only such case is histogram ``buckets``: a ``vector``-domain
        output encoded as ``List(Struct[lower_edge, upper_edge, count,
        normalized])``. Returns ``"histogram_buckets"`` for it, else ``None``.
        """
        if self._ops:
            last = self._ops[-1]
            if last.op == "histogram":
                mode = last.params.get("output")
                if mode is not None and not mode.is_expr and mode.value == "buckets":
                    return "histogram_buckets"
        return None

    def _append_op(
        self,
        op_name: str,
        build_params: "Callable[[Pipeline], dict[str, ParamValue]]",
    ) -> "Pipeline":
        """Append one operation and apply its full plan-time effect.

        Every builder method routes through this. An operation therefore
        cannot be appended without also validating its input domain and
        updating **both** the tracked domain/dtype/ndim and the shape hints.
        Skipping the latter is what let ``transpose``/``channel_select``
        desync the planned schema from execution; making the sequence
        unskippable is the fix.

        Args:
            op_name: The operation's wire name (an op in the generated catalogue).
            build_params: Callable receiving the *cloned* pipeline and
                returning the op's parameters. It runs after the clone so it
                can register per-row expressions via that clone's
                ``_track_expr`` (and, for ``rasterize``, record a shape
                reference) without mutating the receiver.

        Returns:
            A new Pipeline with the operation appended and all state updated.
        """
        new = self._clone()
        spec = OpSpec(op=op_name, params=build_params(new))
        # One contract read serves both the input-domain check and the channel
        # rule, so an append still costs a constant number of FFI calls.
        new._push_op(spec)
        return new

    def _append_typed(self, op_name: str, values: "dict[str, Any]") -> "Pipeline":
        """Append a typed op (one in the generated catalogue).

        The generated builder methods (``_ops_generated._OpsMixin``) call this
        with their arguments as given; each field is encoded by the one rule
        its catalogue type names. Values are *not* validated here: the op's
        Rust definition rejects a wrong type, a value out of range, an unknown
        enum name or a wrong length when :meth:`_push_op` plans the op, so
        there is no second copy of any of those rules.
        """
        fields = OP_FIELDS[op_name]

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            params: dict[str, ParamValue] = {}
            for name, value in values.items():
                encoded = _encode_field(p, value, fields[name], f"{op_name}({name}=)")
                if encoded is not None:
                    params[name] = encoded
            return params

        return self._append_op(op_name, _params)

    def _push_op(self, spec: "OpSpec", *, other_dtype: "str | None" = None) -> None:
        """Append ``spec`` **in place** and apply its full plan-time effect.

        **The single mutator of ``_ops`` in the package.** :meth:`_append_op`
        wraps it for the immutable builder path; the graph hook
        (:meth:`_add_node_op`) calls it directly because it mutates an
        already-cloned pipeline. The effect — input-domain check, schema fold,
        H/W, channels, rank clipping — is one Rust call (:meth:`_plan_step`),
        made before anything changes, so an op cannot be appended with only part
        of it applied.

        The guard is ``test_op_append_is_structurally_exclusive``, which walks
        this module's AST and fails if anything else mutates ``_ops``.

        Args:
            spec: The operation to append.
            other_dtype: A binary op's other operand's dtype, which its
                two-input dtype rule reads. Rust refuses it for any other op,
                and refuses a binary op without it.
        """
        planned = self._plan_step(spec, other_dtype=other_dtype)
        self._entering.append(_Position(self._state(), other_dtype))
        self._ops.append(spec)
        self._apply_step(planned)
        # An assertion recorded *after* this op outranks what the contract
        # inferred. rasterize(shape=<node>) is the case that needs it: its
        # canvas comes from another node's buffer, which no contract on this
        # op can describe.
        self._apply_assertions_at(len(self._ops))

    def _replay(
        self,
        positions: "Sequence[int]",
        *,
        start: PlanState,
        assertions: "dict[int, ShapeAssertion]",
    ) -> None:
        """Rebuild the op list from ``positions`` of the current one, in place.

        **The one wholesale rewrite of ``_ops``**: a slice (CSE's prefix and
        suffix, a sub-pipeline), a reorder (the spatial pushdown) and a
        deletion (identity elimination) all name the ops they keep, in order,
        and the state they start from. Each op is then appended again through
        :meth:`_push_op`, so every per-position fact is *recomputed* for the new
        order rather than re-keyed by the caller — the re-key arithmetic each
        rewrite used to carry is where the CSE path once forgot the
        assertions. ``assertions`` is required, and keyed for the new list.
        """
        steps = [(self._ops[i], self._entering[i].other_dtype) for i in positions]
        self._ops = []
        self._entering = []
        self._assertions = assertions
        self._restore(start)
        self._apply_assertions_at(0)
        for spec, other_dtype in steps:
            self._push_op(spec, other_dtype=other_dtype)

    def _state(self) -> PlanState:
        """The current tracked state, as an immutable record."""
        return PlanState(
            domain=self._current_domain,
            dtype=self._output_dtype,
            ndim=self._expected_ndim,
            hints=copy.deepcopy(self._shape_hints),
            asserted_dims=frozenset(self._asserted_dims),
            shape_declared=self._shape_declared,
        )

    def _restore(self, state: PlanState) -> None:
        """Adopt ``state`` as the current tracked state."""
        self._current_domain = state.domain
        self._output_dtype = state.dtype
        self._expected_ndim = state.ndim
        self._shape_hints = copy.deepcopy(state.hints)
        self._asserted_dims = set(state.asserted_dims)
        self._shape_declared = state.shape_declared

    def _state_at(self, position: int) -> PlanState:
        """The state at op boundary ``position``: entering op ``position``, or
        the current state at the end."""
        if position < len(self._ops):
            return self._entering[position].state
        return self._state()

    def _plan_step(
        self, spec: "OpSpec", *, other_dtype: "str | None" = None
    ) -> "dict[str, Any]":
        """``spec``'s plan-time effect on the current state, from Rust.

        ``plan_step`` is the one authority: the input-domain check, the schema
        fold (domain, dtype, rank), the H/W the op's ``infer_shape`` gives, its
        channel rule and the clipping of hints to the output rank. A per-row
        hint enters as unknown, since its size is not a plan-time integer.
        """
        from polars_cv._lib import plan_step

        dims = [
            None
            if (hint := self._shape_hints.get(dim)) is None or hint.is_expr
            else int(hint.value)
            for dim in HINT_DIMS
        ]
        return plan_step(
            json.dumps(spec.to_dict(planning_slots)),
            self._current_domain,
            self._output_dtype,
            self._expected_ndim,
            dims,
            other_dtype,
        )

    def _apply_step(self, planned: "dict[str, Any]") -> None:
        """Adopt a :meth:`_plan_step` result.

        Only the hints the op replaces are touched; the rest (say, a per-row
        H/W assertion under an op whose input rank is unknown) are kept. Every
        replaced hint is the ops' inference, not the user's claim, so the
        attribution is cleared; :meth:`_apply_assertions_at` re-marks what it
        re-declares.
        """
        self._current_domain = planned["domain"]
        self._output_dtype = planned["dtype"]
        self._expected_ndim = planned["ndim"]
        self._asserted_dims.clear()
        for axis, size in planned["dims"]:
            setattr(
                self._shape_hints,
                HINT_DIMS[axis],
                None if size is None else ParamValue(is_expr=False, value=size),
            )

    def _apply_assertions_at(self, position: int) -> None:
        """Check and overlay any shape declaration recorded at op ``position``.

        A user assertion outranks whatever the ops inferred, but only from the
        point it was written — which is why it is replayed positionally rather
        than applied once at the end.

        **The single place a declaration is validated as well as applied.**
        ``assert_shape`` records into ``_assertions`` and calls this rather than
        assigning the hints itself, so the eager spelling and the lazy
        continuation's replay run the same checks. A declaration used to be
        applied unconditionally, which is how ``resize(224, 224)
        .assert_shape(height=999)`` reached execution: the contradiction was
        accepted here, published as ``expected_shape``, and only surfaced from
        ``validate_output_schema`` at ``collect()`` — as a *plugin* contract
        bug, for what the user had written three lines earlier.
        """
        assertion = self._assertions.get(position)
        if assertion is None:
            return
        self._shape_declared = True
        if assertion.ndim is not None:
            self._require_ndim_is_consistent(assertion)
            self._expected_ndim = assertion.ndim
        for dim, param in assertion.dims.items():
            # A `None` entry declares the dimension *unknown* — the `shape_ref`
            # source's answer when the referenced node's own hint is per-row.
            # There is nothing to contradict, and nothing to attribute.
            if param is None:
                setattr(self._shape_hints, dim, None)
                self._asserted_dims.discard(dim)
                continue
            self._require_dim_is_assertable(dim, param)
            setattr(self._shape_hints, dim, param)
            if assertion.source == "assert_shape":
                self._asserted_dims.add(dim)

    def _require_ndim_is_consistent(self, assertion: "ShapeAssertion") -> None:
        """Reject a rank declaration that contradicts the tracked rank."""
        current = self._expected_ndim
        if current is None or current == assertion.ndim:
            return
        msg = (
            f"assert_shape(dims=...) declares a rank-{assertion.ndim} output, "
            f"but this pipeline is already known to produce rank "
            f"{current}. Drop the assertion, or correct its length."
        )
        raise ValueError(msg)

    def _require_dim_is_assertable(self, dim: str, param: "ParamValue") -> None:
        """Reject a declaration the pipeline's own state contradicts.

        Two ways a declaration is not merely redundant but wrong:

        - the dimension does not exist at the tracked rank — the same invariant
          :meth:`_plan_step` enforces against the ops (hints are clipped to
          the output rank), applied to
          the user;
        - the dimension is already known concretely and the declaration
          disagrees. One of the two is wrong and the planner cannot tell which,
          so it refuses rather than picking.

        Declaring a dimension the planner does *not* know is the supported case
        and passes silently — it is the whole point of ``assert_shape`` on a
        list/array source, whose shape is not knowable until execution.
        """
        ndim = self._expected_ndim
        axis = HINT_DIMS.index(dim)
        if ndim is not None and axis >= ndim:
            msg = (
                f"assert_shape({dim}=...) names dimension {axis}, which a "
                f"rank-{ndim} output does not have. The shape hints are "
                f"positional — {', '.join(HINT_DIMS)} are dimensions "
                f"0, 1 and 2 — so use assert_shape(dims=[...]) for anything "
                f"that is not an [H, W, C] image."
            )
            raise ValueError(msg)
        known = self._shape_hints.get(dim)
        if known is None or known.is_expr or param.is_expr:
            return
        if int(known.value) == int(param.value):
            return
        where = f"the {self._ops[-1].op}() before it" if self._ops else "the source"
        msg = (
            f"assert_shape({dim}={param.value}) contradicts the {dim} "
            f"{known.value} that {where} already establishes. An assertion "
            f"cannot change what the data is — remove it, or fix the value."
        )
        raise ValueError(msg)

    @staticmethod
    def _shape_ref_dims(
        shape: "LazyPipelineExpr",
    ) -> "dict[str, ParamValue | None]":
        """The canvas a ``shape=<node>`` reference supplies, per dimension.

        The referenced node's own published hints are the authority: no
        contract on the rasterize itself can describe a canvas that comes from
        another node's buffer. A per-row (expression) dimension there is not a
        plan-time fact, so it reads as unknown.

        Shared by ``rasterize(shape=)`` and ``source("contour", shape=)`` —
        the same mask from the same reference, so they cannot disagree.
        """
        hints = shape._pipeline._shape_hints
        dims: dict[str, ParamValue | None] = {}
        for dim in ("height", "width"):
            value = getattr(hints, dim)
            dims[dim] = value if value is not None and not value.is_expr else None
        return dims

    def _seed_from_contour_rasterize(self, *, shape: "LazyPipelineExpr | None") -> None:
        """Publish the ``contour`` source's plan-time buffer contract.

        The source decodes by rasterizing (Rust ``decode_contour_source``), so
        what it hands the first op is what the ``rasterize`` op hands its
        successor — an ``[H, W, 1]`` u8 mask. Rank, dtype, channels and canvas
        are therefore read from ``GeometryOp::Rasterize``'s contract, through
        the same :meth:`_plan_step` :meth:`_push_op` uses, and are not restated
        here. Hard-coding rank 3
        and leaving the dtype ``"auto"`` is what made ``sink("list")`` and
        ``sink("array")`` unplannable on a contour source (both need a concrete
        element dtype) and forced a no-op ``.cast("u8")``.

        The fold runs from the *contour* domain, because that is what the
        column holds — the same transition the op declares, so the two routes
        to a mask cannot publish different plan-time state.

        The spec built here is **not** appended to ``_ops``: the rasterize
        happens inside the source's own decode, and appending it would
        rasterize a second time. Only its plan-time effect is read.

        Args:
            shape: The node a ``shape=`` source takes its canvas from, or
                ``None`` for the explicit ``width``/``height`` form. Carried
                into the spec as its ``size`` so the op's own contract reports
                the canvas as unknown, and read for its published H/W below —
                the two halves ``rasterize(shape=)`` also uses.
        """
        source = self._source
        assert source is not None  # set by the caller, immediately above
        if shape is not None:
            size = ParamValue(is_expr=False, value=shape._node_id)
        else:
            # Both are present together; the builder rejected a lone one above.
            size = ParamValue(is_expr=False, value=[source.height, source.width])
        params: dict[str, ParamValue] = {
            "size": size,
            "fill_value": source.fill_value,
            "background": source.background,
        }  # ty: ignore[invalid-assignment]
        spec = OpSpec(op="rasterize", params=params)
        # The column holds contours, of no buffer rank yet.
        self._current_domain = Domain.CONTOUR.value
        self._expected_ndim = None
        self._apply_step(self._plan_step(spec))
        if shape is not None:
            for dim, concrete in self._shape_ref_dims(shape).items():
                setattr(self._shape_hints, dim, concrete)

    # --- Source (required, starts the chain) ---

    def source(
        self,
        format: str = "auto",
        *,
        dtype: str | None = None,
        # Contour source parameters
        width: IntOrExpr | None = None,
        height: IntOrExpr | None = None,
        shape: "LazyPipelineExpr | None" = None,
        fill_value: IntOrExpr = 255,
        background: IntOrExpr = 0,
        # Cloud storage options for file_path sources
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        # Contiguity option for list/array sources
        require_contiguous: bool = False,
        # Error handling for source decoding
        on_error: str = "raise",
        # Explicit decode-scale assertion for image sources
        decode_max_size: int | None = None,
        # Path sandboxing for file_path sources
        allowed_roots: "Sequence[str] | None" = None,
    ) -> "Pipeline":
        """
        Define the input source format.

        The default ``"auto"`` infers the decode path from the column's Polars
        dtype at runtime: a ``String`` column reads as ``"file_path"``, a
        ``List``/``Array`` column as ``"list"``/``"array"``, and a ``Binary``
        column as ``"blob"`` when it carries the VIEW protocol magic and
        ``"image_bytes"`` otherwise. Pass an explicit format to override the
        inference (or when the column dtype cannot be routed, such as a plain
        numeric column).

        Image sources (``"image_bytes"`` and ``"file_path"``) auto-detect the
        format and preserve native dtype.  PNG/JPEG decode to u8, 16-bit PNG
        to u16, and TIFF may produce u8, u16, f32, or f64.  All decoded
        images are always 3D ``[H, W, C]``.

        Each keyword below applies to some formats and not others, and one that
        does not apply to the format you chose is **rejected** rather than
        ignored: a ``width`` on an image source, or ``cloud_options`` on a
        source that never opens a path, has no effect and is a mistake worth
        hearing about.

        Because the dtype is not known until runtime, it starts as ``"auto"``
        in the contract system.  Operations with deterministic output dtypes
        (e.g. ``normalize`` -> f32, ``threshold`` -> u8, ``cast``) resolve it.
        If you sink to ``"list"`` or ``"array"``, the dtype must be known at
        planning time — either via an explicit ``dtype`` here, a ``cast()`` in
        the pipeline, or an operation that fixes the output dtype.

        Args:
            format: How to interpret input data.
                - "auto" (default): Infer the decode path from the column's
                  Polars dtype (String → file_path, List/Array → list/array,
                  Binary → blob if VIEW-tagged else image_bytes)
                - "image_bytes": Decode PNG/JPEG/TIFF (auto-detect format
                  and dtype; always 3D ``[H, W, C]``)
                - "blob": VIEW protocol binary (self-describing)
                - "raw": Raw bytes (requires dtype)
                - "list": Polars nested List column
                - "array": Polars fixed-size Array column
                - "file_path": Read from path (local, s3://, gs://, az://,
                  http://); decodes like ``"image_bytes"``
                - "contour": Rasterize geometry to a binary mask. The column
                  may hold one contour per row (``CONTOUR_SCHEMA``) or a whole
                  set (``List(CONTOUR_SCHEMA)``, what ``extract_contours()``
                  sinks); a set paints the union of its members. The mask is
                  ``[H, W, 1]`` u8 — the same contract the ``rasterize`` op
                  publishes — so a typed ``list``/``array`` sink needs neither
                  a dtype nor a shape.
            dtype: For ``"raw"``: required data type of the raw bytes.
                For ``"image_bytes"`` / ``"file_path"``: asserts the expected
                dtype — at runtime, images with a different dtype are cast to
                this type (no-op if already matching).  For ``"list"`` /
                ``"array"``: override for the inferred column element type.
                Rejected for ``"contour"``: rasterizing always produces u8, so
                there is nothing to assert — use ``.cast(...)`` to convert.
            width: Output mask width for "contour" format.
            height: Output mask height for "contour" format.
            shape: Infer dimensions from another pipeline for "contour" format.
            fill_value: Value for pixels inside contour (default 255). Accepts
                a Polars expression for per-row dynamic values, matching the
                identical parameter on :meth:`rasterize`.
            background: Value for pixels outside contour (default 0). Accepts
                a Polars expression for per-row dynamic values.
            cloud_options: Credentials for cloud storage (S3, GCS, Azure).
            require_contiguous: For "list"/"array", whether to require rectangular data.
            on_error: Error handling strategy for source decoding.
                - ``"raise"`` (default): propagate decode errors (fails the
                  entire batch).
                - ``"null"``: treat decode errors as null output for that row,
                  allowing the rest of the batch to succeed.
            decode_max_size: Explicit assertion that the pipeline only needs
                at least this many pixels on the decoded image's long side
                (for ``"image_bytes"`` / ``"file_path"`` sources). JPEG
                decoding then uses IDCT scaling (1/8, 1/4 or 1/2) to skip
                work — a large CPU and memory win for thumbnail pipelines.
                The decoded long side never drops below
                ``min(decode_max_size, original)``, so a downstream resize
                down to this size never upscales. Other formats (PNG, …)
                ignore the assertion and decode at full size. Note that a
                scaled decode followed by a resize is not bit-identical to a
                full decode followed by the same resize (different
                resampling path) — hence the explicit opt-in.
            allowed_roots: Restrict which locations the path column may read
                from, for ``"file_path"`` (and ``"auto"`` resolving to it).
                Default ``None`` reads whatever the column names, which is
                right when the paths are your own and wrong when they are not.

                One list covers local and remote: an entry that parses as a
                remote URI (``"s3://bucket/public/"``) is matched as a URI
                prefix, anything else (``"/srv/images"``) as a local directory.
                Local paths are canonicalized before the comparison, so
                ``"/srv/images/../../etc/passwd"`` and a symlink out of the
                tree are both refused rather than compared as text, and
                matching is component-wise, so ``"/srv/images"`` does not also
                admit ``"/srv/images-private"``.

                A path matching no entry is refused — the sandbox denies by
                default once you ask for one — and the refusal is subject to
                ``on_error``, so ``on_error="null"`` nulls those rows instead
                of failing the query::

                    >>> pipe = Pipeline().source(
                    ...     "file_path", allowed_roots=["/srv/images"]
                    ... )

        Example:
            ```python
            >>> # Decode PNG/JPEG bytes from a column
            >>> pipe = Pipeline().source("image_bytes").resize(height=224, width=224)
            >>>
            >>> # Read from file paths or URLs
            >>> df = pl.DataFrame({"url": ["https://example.com/image.png"]})
            >>> pipe = Pipeline().source("file_path").grayscale()
            >>> expr = pl.col("url").cv.pipe(pipe).sink("numpy")
            >>>
            >>> # Assert dtype for list sink (cast if needed at runtime)
            >>> pipe = Pipeline().source("image_bytes", dtype="f32").resize(height=224, width=224)
            >>> expr = pl.col("img").cv.pipe(pipe).sink("list")
            >>>
            >>> # Gracefully handle corrupt images as null
            >>> pipe = Pipeline().source("image_bytes", on_error="null").resize(height=224, width=224)
            >>> expr = pl.col("img").cv.pipe(pipe).sink("png")
            ```
        """
        # Taken before anything else binds a name: these *are* the parameters,
        # so what is sent below cannot be a stale or partial list of them
        # (`test_source_applicability_reads_every_parameter`).
        passed = dict(locals())

        from polars_cv._lib import io_check
        from polars_cv.lazy import LazyPipelineExpr

        new = self._clone()
        fmt = _validate_enum(format, SourceFormat, "source format")
        defaults = _source_param_defaults()
        supplied = {
            name
            for name, value in passed.items()
            if name not in ("self", "format") and is_supplied(value, defaults[name])
        }

        if decode_max_size is not None and (
            not isinstance(decode_max_size, int) or decode_max_size <= 0
        ):
            msg = f"decode_max_size must be a positive int, got {decode_max_size!r}"
            raise ValueError(msg)

        dtype_enum = None
        if dtype is not None:
            dtype_enum = _validate_enum(dtype, DType, "dtype")

        if fmt == SourceFormat.CONTOUR:
            has_explicit_dims = width is not None or height is not None
            has_shape = shape is not None

            if has_explicit_dims and has_shape:
                msg = (
                    "Cannot specify both 'shape' and explicit dimensions (width/height)"
                )
                raise ValueError(msg)

            if not has_explicit_dims and not has_shape:
                msg = (
                    "Contour source requires either:\n"
                    "  1. Both 'width' and 'height' parameters, or\n"
                    "  2. A 'shape' LazyPipelineExpr to infer dimensions from"
                )
                raise ValueError(msg)

            if has_explicit_dims and (width is None or height is None):
                msg = "Both 'width' and 'height' must be specified together"
                raise ValueError(msg)

        # The canvas node, by id: Rust takes that node's already-computed
        # buffer.
        shape_node = None
        if shape is not None:
            if not isinstance(shape, LazyPipelineExpr):
                msg = "'shape' must be a LazyPipelineExpr"
                raise TypeError(msg)
            shape_node = shape._node_id
            # Referencing a node by id is not enough to get it executed:
            # `_shape_refs` is what `cv.pipe` / `LazyPipelineExpr.pipe` turn
            # into upstream edges, and only an upstream edge puts a node into
            # the dependency graph. Mirrors `rasterize(shape=...)`.
            new._shape_refs.append(shape)

        def _given(name: str, value: Any) -> Any:
            """The value when the caller passed it, else ``None`` (absent)."""
            return value if name in supplied else None

        # Every setting the caller passed goes into the spec, whichever format
        # it is for: the format's Rust definition refuses one it does not
        # read, naming where it does apply (`io_check` below). The contour
        # colours always go — that decode reads them.
        is_contour = fmt == SourceFormat.CONTOUR
        new._source = SourceSpec(
            format=fmt,
            dtype=dtype_enum,
            width=new._track_expr(width) if width is not None else None,
            height=new._track_expr(height) if height is not None else None,
            fill_value=new._track_expr(fill_value)
            if is_contour or "fill_value" in supplied
            else None,
            background=new._track_expr(background)
            if is_contour or "background" in supplied
            else None,
            shape_node=shape_node,
            cloud_options=normalize_cloud_options(
                _given("cloud_options", cloud_options)
            ),
            require_contiguous=require_contiguous,
            on_error=on_error,
            decode_max_size=decode_max_size,
            allowed_roots=tuple(allowed_roots) if allowed_roots is not None else None,
        )
        io_check("source", json.dumps(new._source.to_dict(planning_slots)))

        if is_contour:
            new._seed_from_contour_rasterize(shape=shape)
        else:
            # Set dtype and ndim based on source format
            if fmt == SourceFormat.RAW:
                # Raw bytes always carry a dtype: the typed raw source requires
                # one, so `io_check` above refused a spec without it.
                # Raw decodes to a flat 1-D buffer (decode.rs), so rank 1 is a
                # true known value — never guess 3. reshape()/assert_shape()
                # lifts the rank when the caller needs a higher-rank sink.
                assert dtype_enum is not None
                new._expected_ndim = 1
                new._output_dtype = dtype_enum.value
            elif fmt in (SourceFormat.BLOB, SourceFormat.AUTO):
                # Blob and Auto are both non-self-declaring at plan time:
                # dtype/rank are unknown here, so an explicit dtype assertion
                # (e.g. for list/array sinks) is the only thing that can pin
                # them. Blob is self-describing at decode. For Auto the concrete
                # decode path is chosen from the column dtype at runtime; for
                # List/Array columns Rust does resolve the leaf dtype at
                # plan-time-with-input (resolved_output_specs), while a
                # Binary/String column stays "auto" (image dtype isn't known
                # until decode).
                new._expected_ndim = None
                if dtype_enum is not None:
                    new._output_dtype = dtype_enum.value
                else:
                    new._output_dtype = "auto"
            elif fmt in (SourceFormat.IMAGE_BYTES, SourceFormat.FILE_PATH):
                # Decoded images are always 3D [H, W, C]
                new._expected_ndim = 3
                if dtype_enum is not None:
                    # User asserted dtype — at runtime, decoded images with
                    # a different dtype will be cast to this type.
                    new._output_dtype = dtype_enum.value
                else:
                    # Dtype unknown until runtime (TIFF=f32, PNG=u8, etc.)
                    new._output_dtype = "auto"
            elif fmt in (SourceFormat.LIST, SourceFormat.ARRAY):
                # For list/array sources, infer dtype and ndim from the
                # Polars column at planning time when not explicitly given.
                if dtype_enum is not None:
                    # User provided explicit dtype — use it. Rank stays unknown
                    # here and is derived from the polars column's true nesting
                    # depth at plan-time-with-input (resolved_output_specs),
                    # never guessed as 3. (Consistent with the no-dtype branch.)
                    new._output_dtype = dtype_enum.value
                    new._expected_ndim = None
                else:
                    # Mark as "auto" so Rust resolves from input_fields
                    new._output_dtype = "auto"
                    new._expected_ndim = None

        return new

    def thumbnail(self, max_size: int) -> "Pipeline":
        """
        Decode only a downscaled *thumbnail* of an image source.

        Explicit, chainable form of ``source(..., decode_max_size=...)``: it
        asserts the pipeline needs at most ``max_size`` pixels on the decoded
        image's long side, so JPEG decoding uses IDCT scaling (1/8, 1/4 or 1/2)
        to skip work — a large CPU and memory win. The decoded long side never
        drops below ``min(max_size, original)``, so a downstream resize down to
        ``max_size`` never upscales. Non-JPEG formats (PNG, …) ignore the
        assertion and decode at full size.

        This is the cheap front of a *decode-aware curation* pass: decode a
        thumbnail, compute a cheap signal (perceptual hash, mean, blur/quality
        score), filter on it, and only full-decode the survivors in a second
        pass. A scaled decode followed by a resize is not bit-identical to a full
        decode then the same resize (different resampling path) — hence the
        explicit opt-in.

        Must be called after ``source(...)`` on an ``image_bytes``/``file_path``
        source.

        Domain: buffer -> buffer (source assertion; does not add an op).

        Args:
            max_size: Maximum pixels on the decoded long side. Positive int.

        Returns:
            A new Pipeline whose source carries the decode-scale assertion.

        Raises:
            ValueError: If there is no source, the source is not an image
                source, or ``max_size`` is not a positive int.

        Example:
            ```python
            >>> # Cheap perceptual-hash over thumbnails for dedup/curation
            >>> pipe = (
            ...     Pipeline().source("image_bytes").thumbnail(64).perceptual_hash()
            ... )
            ```
        """
        import dataclasses

        if self._source is None:
            msg = "thumbnail() requires a source; call .source(...) first"
            raise ValueError(msg)
        if not isinstance(max_size, int) or isinstance(max_size, bool) or max_size <= 0:
            msg = f"max_size must be a positive int, got {max_size!r}"
            raise ValueError(msg)

        from polars_cv._lib import io_check

        new = self._clone()
        assert new._source is not None  # guaranteed: checked on self above
        new._source = dataclasses.replace(new._source, decode_max_size=max_size)
        # `thumbnail()` writes `decode_max_size`, so it applies exactly where
        # that field does: the source format's own definition decides, as it
        # does for `source(decode_max_size=)`. The two used to disagree about
        # `auto` when each kept its own list.
        try:
            io_check("source", json.dumps(new._source.to_dict(planning_slots)))
        except ValueError as e:
            msg = f"thumbnail() only applies where decode_max_size does: {e}"
            raise ValueError(msg) from None
        return new

    # --- Shape Assertions (optional, helps planner) ---

    def assert_shape(
        self,
        *,
        dims: "Sequence[int | None] | None" = None,
        height: IntOrExpr | None = None,
        width: IntOrExpr | None = None,
        channels: IntOrExpr | None = None,
    ) -> "Pipeline":
        """
        Declare a shape the planner cannot work out for itself.

        Use this when the source does not reveal its shape at plan time — a
        ``list``/``array`` column's rank and sizes are only known once the data
        arrives, so a fixed-shape ``.sink("array")`` has nothing to publish
        without a declaration. For a source the planner *can* read (image bytes,
        a file path), the shape is already inferred and an assertion is at best
        redundant.

        **An assertion states a fact; it does not change one.** A declaration
        that contradicts what the pipeline already knows — or that names a
        dimension the output rank does not have — is rejected here, at the line
        that wrote it. Previously it was accepted, published as the output
        schema, and reported at ``collect()`` as a plugin contract bug.

        Two spellings:

        - ``dims=[8, 8, 3]`` — positional and complete. Entry *i* is the size
          of dimension *i*; ``None`` leaves one unknown. This also pins the
          output **rank** to ``len(dims)``, which is what lets a list/array
          source reach an ``array`` sink. Entries are literal ``int``\\ s: a
          rank and its per-dimension sizes are plan-time schema facts, so
          unlike most parameters they cannot be per-row expressions.
        - ``height=``/``width=``/``channels=`` — the ``[H, W, C]`` spelling of
          dimensions 0, 1 and 2, for the common image case. The hints are
          **positional**, so these names only describe an ``[H, W, C]`` buffer:
          after a ``transpose([2, 0, 1])`` dimension 0 is the channel axis, and
          calling it ``height`` would be a lie. They are therefore rejected once
          the rank is known to be anything but 3 — use ``dims=`` there.
          Expressions are accepted and resolved per row, but a per-row size is
          not a plan-time fact, so it publishes no shape.

        Args:
            dims: Full positional shape. Mutually exclusive with the
                ``height``/``width``/``channels`` keywords.
            height: Size of dimension 0 (literal or expression).
            width: Size of dimension 1 (literal or expression).
            channels: Size of dimension 2 (literal or expression).

        Returns:
            A new pipeline carrying the declaration.

        Raises:
            ValueError: If the declaration contradicts a known dimension or
                rank, names a dimension the rank does not have, or mixes
                ``dims=`` with the per-dimension keywords.

        Example:
            ```python
            # A list column's shape is not knowable at plan time; declare it.
            pipe = Pipeline().source("list", dtype="f32").assert_shape(dims=[8, 8, 3])
            df.with_columns(pl.col("arr").cv.pipe(pipe).sink("array"))
            ```
        """
        named = {"height": height, "width": width, "channels": channels}
        given = {dim: value for dim, value in named.items() if value is not None}
        if dims is not None and given:
            msg = (
                f"assert_shape() takes either dims=[...] or the per-dimension "
                f"keywords {sorted(given)}, not both — dims= already gives "
                f"every dimension a position."
            )
            raise ValueError(msg)
        if dims is None and not given:
            msg = "assert_shape() needs a declaration: dims=[...] or height=/width=/channels=."
            raise ValueError(msg)

        new = self._clone()
        # Recorded against the current op position so a continuation can replay
        # the assertion at the point the user wrote it. Without the position an
        # assertion could not be told apart from a hint an op computed, and
        # replaying it at the end would override later ops that legitimately
        # change the shape (assert channels=3, then grayscale → 1).
        assertion = new._assertions.setdefault(len(new._ops), ShapeAssertion())
        if dims is not None:
            assertion.ndim = _asserted_rank(dims)
            for axis, size in enumerate(dims):
                if size is None:
                    continue
                if axis < len(HINT_DIMS):
                    assertion.dims[HINT_DIMS[axis]] = ParamValue(
                        is_expr=False, value=size
                    )
        else:
            for dim, value in given.items():
                assertion.dims[dim] = new._track_expr(value)
        # Applied (and checked) through the one path the lazy replay also uses,
        # rather than assigning the hints here — see `_apply_assertions_at`.
        new._apply_assertions_at(len(new._ops))
        return new

    # --- View Operations (zero-copy where possible) ---

    def flip_h(self) -> "Pipeline":
        """
        Flip horizontally (along width axis).

        Returns:
            Self for chaining.
        """
        return self.flip(axes=[1])

    def flip_v(self) -> "Pipeline":
        """
        Flip vertically (along height axis).

        Returns:
            Self for chaining.
        """
        return self.flip(axes=[0])

    # --- Compute Operations ---

    def _out_dtype_target(
        self, op_name: str, out_dtype: str | None, preserve_dtype: bool
    ) -> str | None:
        """Resolve the dtype a float-promoting scalar op's result ends in.

        ``out_dtype`` and ``preserve_dtype=True`` are two spellings of one
        request — "leave this op in a dtype other than the promoted float" —
        so they resolve to a single target here and are lowered by a single
        mechanism (:meth:`_apply_out_dtype`). ``out_dtype`` names the target
        outright; ``preserve_dtype`` computes it from the pipeline's dtype
        *before* the op. Returns ``None`` when neither was asked for, leaving
        the op's own ``OutputDTypeRule`` to decide.

        Raises when the request cannot be honored: both spellings at once, an
        unrecognised dtype name, or ``preserve_dtype`` over a pipeline whose
        dtype is not concrete (image sources are "auto" until the source
        declares one).
        """
        if out_dtype is not None and preserve_dtype:
            msg = (
                f"{op_name}: preserve_dtype=True and out_dtype are mutually "
                "exclusive; pass one or the other."
            )
            raise ValueError(msg)
        if out_dtype is not None:
            return _validate_enum(out_dtype, DType, "out_dtype").value
        if not preserve_dtype:
            return None
        pre_dtype = self._output_dtype
        # `DType` is the dtype-name authority, so "concrete" is membership in
        # it rather than a hand-listed set of sentinels ("auto", …) that would
        # go stale the day another one is added.
        if pre_dtype not in {member.value for member in DType}:
            msg = (
                f"{op_name}: preserve_dtype=True requires a known input dtype, "
                f"but the pipeline's dtype is {pre_dtype!r}. Declare the source "
                "dtype (e.g. .source('image_bytes', dtype='u8')) or use an "
                "explicit .cast(...) instead."
            )
            raise ValueError(msg)
        return pre_dtype

    def _apply_out_dtype(self, new: "Pipeline", target: str | None) -> "Pipeline":
        """Append the cast that lands a scalar op on ``target``, when needed.

        This is how ``out_dtype`` and ``preserve_dtype`` are *honored*, and
        deliberately not by giving the op its own output dtype: a trailing
        ``cast`` lowers into the existing fused-kernel cast support
        (round-then-saturate for float→int), which ``try_fuse`` already pins to
        the dtype the unfused chain would have produced. An op-carried dtype
        would instead need `extract_ops` and that pinning taught about it, and
        would turn the op's ``PromoteToFloat`` rule — which preserves f64 input
        — into a fixed-dtype one that silently downgrades it to f32.

        A no-op cast (the op already produced ``target``) is skipped.
        """
        if target is None or new._output_dtype == target:
            return new
        return new.cast(target)

    def scale(
        self,
        factor: FloatOrExpr,
        out_dtype: str | None = None,
        preserve_dtype: bool = False,
    ) -> "Pipeline":
        """
        Multiply all values by a factor.

        Args:
            factor: Scale factor.
            out_dtype: Dtype to leave the result in. Any :class:`DType` name;
                the multiply itself still happens in f32 (f64 for f64 input)
                and the result is cast, round-then-saturate for integer
                targets. Defaults to None — the op's own promote-to-float rule,
                which turns integers into f32 and preserves float input.
                Mutually exclusive with ``preserve_dtype``.
            preserve_dtype: If True, cast the result back to the input dtype,
                e.g. u8 in → u8 out instead of the promoted f32. The same
                mechanism as ``out_dtype``, with the target read off the
                pipeline, so it requires that dtype to be known (not "auto").

        Raises:
            ValueError: If domain is not buffer, or the output dtype cannot be
                honored (unknown dtype name, unknown input dtype under
                ``preserve_dtype``, or both keywords at once).
        """
        target = self._out_dtype_target("scale", out_dtype, preserve_dtype)
        new = self._scale(factor)
        return self._apply_out_dtype(new, target)

    def clamp(
        self,
        min_val: FloatOrExpr,
        max_val: FloatOrExpr,
        out_dtype: str | None = None,
        preserve_dtype: bool = False,
    ) -> "Pipeline":
        """
        Clamp values to a range.

        This operation accepts any numeric input dtype and automatically handles
        type promotion. Integers are promoted to float32; floats are preserved.

        Args:
            min_val: Minimum value (literal or expression).
            max_val: Maximum value (literal or expression).
            out_dtype: Dtype to leave the result in. Any :class:`DType` name;
                the clamp itself still happens in f32 (f64 for f64 input) and
                the result is cast, round-then-saturate for integer targets.
                Defaults to None — the op's own promote-to-float rule, which
                turns integers into f32 and preserves float input. Mutually
                exclusive with ``preserve_dtype``.
            preserve_dtype: If True, cast the result back to the input dtype,
                e.g. u8 in → u8 out instead of the promoted f32. The same
                mechanism as ``out_dtype``, with the target read off the
                pipeline, so it requires that dtype to be known (not "auto").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If domain is not buffer, or the output dtype cannot be
                honored (unknown dtype name, unknown input dtype under
                ``preserve_dtype``, or both keywords at once).
        """
        target = self._out_dtype_target("clamp", out_dtype, preserve_dtype)

        new = self._clamp(min_val, max_val)
        return self._apply_out_dtype(new, target)

    # --- Core math primitives ---
    #
    # Pure elementwise scalar ops. Each promotes to float (integers → f32, f64
    # preserved) like ``scale``/``relu`` and fuses automatically with adjacent
    # scalar ops into a single kernel pass — the user never manages fusion.

    # --- Channel Operations ---

    # --- Intensity Adjustments ---

    def adjust_brightness(
        self, *, factor: FloatOrExpr, preserve_dtype: bool = False
    ) -> "Pipeline":
        """
        Adjust image brightness by scaling pixel values.

        Convenience method equivalent to ``.scale(factor).clamp(min_val=0, max_val=255)``.

        Domain: buffer → buffer

        Args:
            factor: Brightness factor. 1.0 = no change, >1 = brighter, <1 = darker.
            preserve_dtype: If True, cast the result back to the dtype the
                pipeline had *before* this op (round-then-saturate for integer
                targets), e.g. u8 in → u8 out instead of the promoted f32.
                Requires the pipeline's dtype to be known (not "auto").

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").adjust_brightness(factor=1.2)
            ```
        """
        target = self._out_dtype_target("adjust_brightness", None, preserve_dtype)
        new = self.scale(factor=factor).clamp(min_val=0.0, max_val=255.0)
        return self._apply_out_dtype(new, target)

    # --- Color Space Conversion ---

    def to_hsv(self) -> "Pipeline":
        """Convert from RGB to HSV color space.

        Returns:
            Self for chaining.
        """
        return self.convert_color("rgb", "hsv")

    def to_lab(self) -> "Pipeline":
        """Convert from RGB to CIE LAB color space.

        Output dtype is promoted to f32 (L=[0,100], a/b~[-128,127]).

        Returns:
            Self for chaining.
        """
        return self.convert_color("rgb", "lab")

    def to_bgr(self) -> "Pipeline":
        """Convert from RGB to BGR channel order.

        Returns:
            Self for chaining.
        """
        return self.convert_color("rgb", "bgr")

    def to_ycbcr(self) -> "Pipeline":
        """Convert from RGB to YCbCr color space.

        Returns:
            Self for chaining.
        """
        return self.convert_color("rgb", "ycbcr")

    # --- Convolution / Filtering ---

    def sobel(self, *, axis: str = "x", ksize: int = 3) -> "Pipeline":
        """
        Sobel gradient operator.

        Convenience method that delegates to :meth:`convolve2d` with standard
        Sobel kernels.

        Domain: buffer → buffer

        Args:
            axis: Gradient direction — ``"x"`` (horizontal) or ``"y"`` (vertical).
            ksize: Kernel size (currently only 3 is supported).

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> gx = Pipeline().source("image_bytes").grayscale().sobel(axis="x")
            ```
        """
        if ksize != 3:
            msg = f"Only ksize=3 is currently supported for Sobel, got {ksize}"
            raise ValueError(msg)

        sobel_x_3: list[FloatOrExpr] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]
        sobel_y_3: list[FloatOrExpr] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]
        kernel = sobel_x_3 if axis == "x" else sobel_y_3
        return self.convolve2d(kernel, ksize, normalize=False)

    def laplacian(self, *, ksize: int = 3) -> "Pipeline":
        """
        Laplacian second-derivative operator.

        Convenience method that delegates to :meth:`convolve2d` with a standard
        Laplacian kernel.

        Domain: buffer → buffer

        Args:
            ksize: Kernel size (currently only 3 is supported).

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> lap = Pipeline().source("image_bytes").grayscale().laplacian()
            ```
        """
        if ksize != 3:
            msg = f"Only ksize=3 is currently supported for Laplacian, got {ksize}"
            raise ValueError(msg)

        laplacian_3 = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]
        return self.convolve2d(laplacian_3, ksize, normalize=False)

    def sharpen(self, *, strength: FloatOrExpr = 1.0) -> "Pipeline":
        """
        Sharpen using an unsharp-mask-style kernel.

        The kernel sum is 1 (brightness-preserving) with ``strength`` controlling
        how aggressively edges are enhanced. ``strength=0`` produces the
        identity; higher values increase edge emphasis.

        Domain: buffer → buffer

        Args:
            strength: Sharpening strength (default 1.0). Accepts a Polars
                expression: the kernel coefficients are built from it
                element-wise, and ``convolve2d`` resolves each coefficient per
                row.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> sharp = Pipeline().source("image_bytes").sharpen(strength=1.5)
            >>> # Per-row strength from a column
            >>> sharp = Pipeline().source("image_bytes").sharpen(
            ...     strength=pl.col("sharpness")
            ... )
            ```
        """
        s = strength
        # Expression arithmetic mirrors the float arithmetic, so both paths
        # produce the same brightness-preserving kernel (sum == 1).
        center = 1.0 + 8.0 * s
        neg = -s
        k = [neg, neg, neg, neg, center, neg, neg, neg, neg]
        return self.convolve2d(k, 3, normalize=False)

    # --- Edge Detection ---

    # --- Morphological Operations ---

    def morphology_open(self, *, ksize: IntOrExpr = 3) -> "Pipeline":
        """
        Morphological opening (erode then dilate).

        Removes small bright spots while preserving larger structures.
        Equivalent to ``.erode(ksize=ksize).dilate(ksize=ksize)``.

        Domain: buffer → buffer

        Args:
            ksize: Size of the square structuring element. Must be odd and >= 1.
                Accepts a Polars expression for per-row dynamic values.

        Returns:
            New Pipeline with opening applied.

        Example:
            ```python
            >>> cleaned = Pipeline().source("image_bytes").grayscale().threshold(128).morphology_open(ksize=3)
            ```
        """
        return self.erode(ksize=ksize).dilate(ksize=ksize)

    def morphology_close(self, *, ksize: IntOrExpr = 3) -> "Pipeline":
        """
        Morphological closing (dilate then erode).

        Fills small dark holes while preserving larger structures.
        Equivalent to ``.dilate(ksize=ksize).erode(ksize=ksize)``.

        Domain: buffer → buffer

        Args:
            ksize: Size of the square structuring element. Must be odd and >= 1.
                Accepts a Polars expression for per-row dynamic values.

        Returns:
            New Pipeline with closing applied.

        Example:
            ```python
            >>> filled = Pipeline().source("image_bytes").grayscale().threshold(128).morphology_close(ksize=3)
            ```
        """
        return self.dilate(ksize=ksize).erode(ksize=ksize)

    # --- Histogram Equalization ---

    # --- Image Operations ---

    def resize_scale(
        self,
        *,
        scale: FloatOrExpr | None = None,
        scale_x: FloatOrExpr | None = None,
        scale_y: FloatOrExpr | None = None,
        filter: str | pl.Expr = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image by scale factor.

        Target dimensions are computed at runtime as:
        - new_width = input_width * scale_x
        - new_height = input_height * scale_y

        Domain: buffer → buffer

        Args:
            scale: Uniform scale factor (applies to both x and y).
            scale_x: X (width) scale factor. If None, uses scale.
            scale_y: Y (height) scale factor. If None, uses scale.
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If neither scale nor scale_x/scale_y specified.
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> # Uniform 50% downscale
            >>> pipe = Pipeline().source("image_bytes").resize_scale(scale=0.5)
            >>>
            >>> # Non-uniform: half width, double height
            >>> pipe = Pipeline().source("image_bytes").resize_scale(scale_x=0.5, scale_y=2.0)
            >>>
            >>> # Dynamic scale from column
            >>> pipe = Pipeline().source("image_bytes").resize_scale(scale=pl.col("zoom"))
            ```
        """

        # Resolve scale factors
        if scale is None and scale_x is None and scale_y is None:
            msg = "Must specify 'scale' or 'scale_x'/'scale_y'"
            raise ValueError(msg)

        actual_scale_x = scale_x if scale_x is not None else scale
        actual_scale_y = scale_y if scale_y is not None else scale

        if actual_scale_x is None or actual_scale_y is None:
            msg = "Must specify both scale factors or use 'scale' for uniform scaling"
            raise ValueError(msg)

        return self._resize_scale(
            scale_x=actual_scale_x, scale_y=actual_scale_y, filter=filter
        )

    # --- Padding Operations ---

    # --- Affine Transform Operations ---

    def shear(
        self,
        *,
        sx: FloatOrExpr = 0.0,
        sy: FloatOrExpr = 0.0,
        output_size: tuple[IntOrExpr, IntOrExpr],
    ) -> "Pipeline":
        """
        Apply a shear transformation.

        Convenience wrapper that builds a shear matrix and delegates to
        :meth:`warp_affine`.

        Domain: buffer → buffer

        Args:
            sx: Horizontal shear factor (literal or per-row Polars expression).
            sy: Vertical shear factor (literal or per-row Polars expression).
            output_size: ``(height, width)`` of the output. Required, because the
                output shape is part of the plan-time schema and a shear does not
                imply one — an image source's height/width are not known until
                execution. Each element accepts a per-row Polars expression.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").shear(sx=0.2, output_size=(100, 100))
            >>>
            >>> # Per-sample random shear from a column
            >>> pipe = Pipeline().source("image_bytes").shear(
            ...     sx=pl.col("shear_x"), output_size=(100, 100)
            ... )
            ```
        """
        # sx/sy may be per-row expressions; warp_affine tracks each matrix
        # element independently, so the shear matrix passes them through.
        matrix: list[FloatOrExpr] = [1.0, sx, 0.0, sy, 1.0, 0.0]
        return self.warp_affine(matrix, output_size)

    def rotate_and_scale(
        self,
        *,
        angle: FloatOrExpr,
        center: tuple[FloatOrExpr, FloatOrExpr],
        output_size: tuple[IntOrExpr, IntOrExpr],
        scale: FloatOrExpr = 1.0,
    ) -> "Pipeline":
        """
        Combined rotation and scaling around a center point.

        Convenience wrapper that builds a rotation+scale matrix and delegates
        to :meth:`warp_affine`.

        Domain: buffer → buffer

        Args:
            angle: Rotation angle in degrees (positive = clockwise). Accepts a
                Polars expression for a per-row angle.
            center: ``(cx, cy)`` center of rotation. Required — an image source's
                height/width are not known until execution, so there is no
                plan-time centre to default to. Each element accepts an
                expression.
            output_size: ``(height, width)`` of the output. Required, because the
                output shape is part of the plan-time schema (same reason as
                *center*). Each element accepts an expression.
            scale: Scale factor (default 1.0). Accepts an expression.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").rotate_and_scale(
            ...     angle=45.0, scale=1.2, center=(112, 112), output_size=(224, 224)
            ... )
            >>> # Per-row angle from a column
            >>> pipe = Pipeline().source("image_bytes").rotate_and_scale(
            ...     angle=pl.col("theta"), center=(112, 112), output_size=(224, 224)
            ... )
            ```
        """
        matrix = _rotation_matrix(angle, center, scale)
        return self.warp_affine(matrix, output_size)

    def perceptual_hash(
        self,
        algorithm: HashAlgorithm | str = HashAlgorithm.PERCEPTUAL,
        hash_size: int = 64,
    ) -> "Pipeline":
        """
        Compute a perceptual hash fingerprint.

        Args:
            algorithm: "perceptual" (pHash), "average" (aHash), "difference" (dHash).
            hash_size: Number of bits in the hash (must be power of 2).

        Example:
            >>> Pipeline().source("image_bytes").perceptual_hash()
        """

        # The Rust definition validates both (an unknown algorithm, a
        # non-positive or per-row `hash_size`); this method only keeps the
        # signature, whose default is the Python enum member.
        return self._perceptual_hash(algorithm=algorithm, hash_size=hash_size)

    # --- Contour/Geometry Operations ---

    def rasterize(
        self,
        *,
        width: IntOrExpr | None = None,
        height: IntOrExpr | None = None,
        shape: "LazyPipelineExpr | None" = None,
        fill_value: IntOrExpr = 255,
        background: IntOrExpr = 0,
    ) -> "Pipeline":
        """
        Rasterize contours to a binary mask.

        A pixel is filled when its centre — ``(x + 0.5, y + 0.5)`` — lies inside
        the contour, boundary included; holes are cut out by the same rule. This
        is the convention ``contains_point`` and the area measures follow, so for
        a shape whose vertices are integers on axis-aligned edges the mask holds
        exactly ``area()`` pixels.

        The contour domain carries a *set* — ``extract_contours()`` generally
        yields more than one — and the mask is their union: each member's
        exterior minus its own holes. One member's hole never erases another's
        fill, and the result does not depend on the set's order. ``fill_value``
        and ``background`` may be inverted; the same region is painted either way.

        Args:
            width: Mask width.
            height: Mask height.
            shape: Match dimensions from another pipeline.
            fill_value: Inside value (default 255). Accepts a Polars expression
                for per-row dynamic values.
            background: Outside value (default 0). Accepts a Polars expression
                for per-row dynamic values.

        Domain transition: contour → buffer
        """
        has_explicit = width is not None or height is not None
        has_shape = shape is not None

        if not has_explicit and not has_shape:
            msg = "Must specify width/height or shape, not neither"
            raise ValueError(msg)
        if has_explicit and has_shape:
            msg = "Specify width/height or shape, not both"
            raise ValueError(msg)

        if shape is None:
            if width is None or height is None:
                msg = "Both width and height must be specified"
                raise ValueError(msg)
            # H/W come from `GeometryOp::Rasterize::infer_shape` and the
            # single-channel output from the op's `fixed:1` channel rule; none
            # of it is re-derived here.
            return self._rasterize(
                size=[height, width], fill_value=fill_value, background=background
            )

        from polars_cv.lazy import LazyPipelineExpr

        if not isinstance(shape, LazyPipelineExpr):
            msg = "'shape' must be a LazyPipelineExpr"
            raise TypeError(msg)
        new = self._rasterize(size=shape, fill_value=fill_value, background=background)
        # The referenced node must execute before this one; graph wiring
        # (cv.pipe / LazyPipelineExpr.pipe) adds it as an upstream dep.
        new._shape_refs.append(shape)
        # The canvas comes from another node's buffer, so no contract on this
        # op can supply it: it is recorded as an assertion at this op's
        # position, which the lazy continuation replays like a user
        # `assert_shape`, and applied the way `_push_op` applies one — last,
        # over the op's own (unknown) inferred size.
        #
        # Tagged `shape_ref`, not `assert_shape`: the canvas comes from another
        # node's *inferred* hints, so if execution disagrees that is a contract
        # bug and keeps the contract-bug wording.
        position = len(new._ops)
        asserted = new._assertions.setdefault(
            position, ShapeAssertion(source="shape_ref")
        )
        for dim, concrete in Pipeline._shape_ref_dims(shape).items():
            asserted.dims[dim] = concrete
        new._apply_assertions_at(position)
        return new

    def scale_contour(
        self,
        *,
        sx: FloatOrExpr,
        sy: FloatOrExpr,
        origin: "ScaleOrigin | str | pl.Expr" = ScaleOrigin.CENTROID,
    ) -> "Pipeline":
        """
        Scale the contour about *origin*.

        Domain: contour → contour

        Args:
            sx: X scale factor.
            sy: Y scale factor.
            origin: Point to scale about — ``"centroid"`` (the default),
                ``"bbox_center"`` or ``"origin"``. Accepts an expression for a
                per-row choice: which point the scale is measured from changes
                no output shape, rank or dtype, so it meets the eligibility
                rule for a per-row parameter.

        Note:
            The default is ``"centroid"``, which is what this method has always
            done — it previously hardcoded it with no way to choose. The
            ``.contour.scale`` accessor defaults to ``"origin"`` instead; pass
            *origin* explicitly if you need the two to agree.
        """
        # The Rust definition validates every argument; this method only keeps
        # the signature, whose default is the Python enum member.
        return self._scale_contour(sx=sx, sy=sy, origin=origin)

    # --- Validation ---

    def validate(self) -> None:
        """
        Validate that the pipeline is well-formed.

        Raises:
            ValueError: If pipeline is invalid.
        """
        if self._source is None:
            msg = "Pipeline must have a source. Call .source() first."
            raise ValueError(msg)

    def has_source(self) -> bool:
        """
        Check if the pipeline has a source defined.

        Returns:
            True if the pipeline has a source defined.
        """
        return self._source is not None

    # --- Graph Conversion ---

    def to_graph(self, column: pl.Expr | None = None) -> "PipelineGraph":
        """
        Convert this linear pipeline to a graph representation.

        This is the unified execution path - all pipelines are converted to
        graphs before execution. A Pipeline becomes a single node in the graph.

        For multi-output with intermediate checkpoints, use LazyPipelineExpr
        composition with .pipe() and .alias() instead.

        Args:
            column: The input column expression. If None, must be set later
                via graph.set_root_column().

        Returns:
            PipelineGraph representation of this pipeline.

        Example:
            ```python
            >>> from polars_cv import OptFlags
            >>> pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
            >>> graph = pipe.to_graph(pl.col("image"))
            >>> expr = graph.optimize(OptFlags.all()).to_expr()
            ```

            ``to_expr()`` requires the optimization phase to have run — call
            ``graph.optimize(flags)`` first (or use the higher-level
            ``LazyPipelineExpr.sink``, which does it for you).
        """
        from polars_cv._graph import PipelineGraph

        graph = PipelineGraph()

        # Create single node with all operations
        node_id = "_node_0"
        # Create a sub-pipeline with source and all ops (no sink - handled separately)
        sub_pipe = self._create_sub_pipeline(0, len(self._ops))
        graph.add_node(
            node_id=node_id,
            pipeline=sub_pipe,
            column=column,
            upstream=[],
            alias="_output",  # Implicit terminal alias
        )
        graph._alias_to_node["_output"] = node_id

        return graph

    def _create_sub_pipeline(
        self,
        start_op: int,
        end_op: int,
        source_format: str | None = None,
    ) -> "Pipeline":
        """
        Create a sub-pipeline with a subset of operations.

        Args:
            start_op: Starting operation index (inclusive).
            end_op: Ending operation index (exclusive).
            source_format: Override source format (e.g., "blob" for non-root nodes).

        Returns:
            New Pipeline with the specified operations.
        """
        # Inherit the whole state, then override only what this slice changes.
        # The per-row policies (`_on_error`, `_on_null_param`) ride along that
        # way: `PipelineGraph._to_dict` reads them off the node pipeline, and
        # `to_graph()` makes this sub-pipeline the graph's *only* node, so a
        # dropped policy here silently reverted the user's `on_error("null")`.
        sub = Pipeline()
        sub._copy_state_from(self)

        if source_format is not None:
            # Non-root node: source is blob (receives from upstream)
            sub._source = SourceSpec(format=SourceFormat(source_format))

        # The slice starts from the state entering its first op.
        sub._replay(
            range(start_op, end_op),
            start=self._state_at(start_op),
            assertions=_assertion_window(self._assertions, start_op, end_op),
        )
        return sub

    # --- Graph Composition Support ---

    def _add_node_op(
        self,
        op_name: str,
        values: "dict[str, Any]",
        *,
        other_dtype: "str | None" = None,
    ) -> None:
        """Append a ``lazy_only`` op — one reading other graph nodes — in place.

        Used by the ``LazyPipelineExpr`` methods that combine expressions
        (the binary ops, ``apply_mask``, ``channel_merge``) on a pipeline they
        have already cloned. Each field is encoded by its catalogue type, as
        :meth:`_append_typed` does: an operand expression becomes its node id,
        and any other value (e.g. ``apply_mask(invert=)``) may be per-row.

        Args:
            op_name: The op's wire name.
            values: Its arguments, by catalogue field name.
            other_dtype: A binary op's other operand's dtype (see
                :meth:`_push_op`).
        """
        fields = OP_FIELDS[op_name]
        params: dict[str, ParamValue] = {}
        for name, value in values.items():
            encoded = _encode_field(self, value, fields[name], f"{op_name}({name}=)")
            if encoded is not None:
                params[name] = encoded
        # Binary ops are elementwise, so H/W pass through unchanged — but the
        # append still routes through `_push_op`, which records the
        # entering-hints snapshot and applies the channel rule.
        self._push_op(OpSpec(op=op_name, params=params), other_dtype=other_dtype)

    # --- Spatial-window pushdown ---
    #
    # Structured as a pushdown, the way Polars' ``slice_pushdown`` carries a
    # slice toward the source: a spatial window (a crop / ROI) moves earlier
    # past each op it commutes with, the op's ``SpatialDependency`` (read from
    # ``op_contract``'s ``spatial_rule``) deciding whether — and how — it passes.
    # The three pieces are the transfer function (:meth:`_spatial_transfer`), the
    # driver (:meth:`_compute_spatial_pushdown`), and the commit
    # (:meth:`_replay`); later spatial optimizations widen the
    # transfer function's arms rather than adding a pass. Phase 1 moves a crop
    # past a run of ``Pointwise`` ops within one node.

    def _spatial_transfer(
        self, window: "OpSpec", op: "OpSpec", contract: dict
    ) -> "OpSpec | object":
        """How a spatial ``window`` crosses one preceding ``op``.

        Returns the window rewritten for crossing ``op`` — unchanged for a
        ``Pointwise`` op, whose output at ``(y, x)`` depends only on its input at
        ``(y, x)``, so a crop commutes exactly — or :data:`_SPATIAL_BARRIER` if
        it may not cross.

        The arms are exactly the ``SpatialDependency`` vocabulary
        (``op_contract``'s ``spatial_rule``), so this is the honest consumer of
        that single authority. Widening it — not adding a pass — is how later
        spatial optimizations land: ``neighborhood:<r>`` would return the window
        dilated by ``r`` (a halo, not bit-exact); ``geometric`` would return the
        window mapped through the op's inverse transform (needs the
        coordinate-remap descriptor ``GeometricEffect`` does not carry yet).
        ``global`` is always a barrier.

        A multi-input op (one reading a sibling node's buffer) is a hard barrier
        regardless of its spatial rule: hoisting the window past it would crop
        only this operand and leave the sibling full-size. See
        :func:`_op_reads_sibling_nodes`.
        """
        if _op_reads_sibling_nodes(op):
            return _SPATIAL_BARRIER
        rule = contract["spatial_rule"]
        if rule == "pointwise":
            return window
        return _SPATIAL_BARRIER

    @staticmethod
    def _is_spatial_window(op: "OpSpec") -> bool:
        """Whether ``op`` is a spatial window this pass hoists.

        Reads the Rust ``is_spatial_window`` authority (``op_contract``) rather
        than matching an op name: "is a hoistable H/W crop/ROI" is an op-identity
        fact the engine owns, the counterpart to the ``spatial_rule`` the transfer
        function reads. Only an H/W-only crop qualifies today — the engine leaves
        the channel axis at full extent — so a window commutes with a
        channel-changing pointwise op (e.g. ``grayscale``). The assertion pins the
        builder's guarantee that a recognised window carries no channel parameter.
        """
        if not _op_contract_for(op)["is_spatial_window"]:
            return False
        assert "channel" not in op.params and "channels" not in op.params, (
            "an is_spatial_window op unexpectedly carries a channel parameter; "
            "the H/W-only commutation assumption no longer holds"
        )
        return True

    def _compute_spatial_pushdown(self, ops: "list[OpSpec]") -> "list[int]":
        """Hoist each crop to the front of the ``Pointwise`` run before it.

        Returns the new order, as the original index of each op.
        A crop is moved to the start of the maximal contiguous run of ops
        immediately preceding it that the transfer function lets it cross; the
        run stops at the first barrier. An ``assert_shape`` op-boundary in the
        run is also a barrier — a crop is never moved across a shape the user
        pinned — and, to stay simple, a crop whose run contains such a boundary
        is left in place.

        Two crops never contend: a crop is itself a barrier (``Geometric``), so
        one crop's pointwise run cannot reach across another. Processing crops
        left to right therefore keeps the invariant that, when a crop at
        original index ``i`` is reached, the entries already placed for original
        indices ``j..i-1`` (its pointwise run) are the last ``i-j`` of
        ``result`` — so slicing ``result[j:]`` picks out exactly that run.
        """
        assertion_boundaries = set(self._assertions.keys())
        result: list[int] = []  # original indices, in new order
        for i, op in enumerate(ops):
            if not self._is_spatial_window(op):
                result.append(i)
                continue
            # Extend the run leftward over ops the window crosses unchanged.
            j = i
            while j - 1 >= 0:
                prev = ops[j - 1]
                if (
                    self._spatial_transfer(op, prev, _op_contract_for(prev))
                    is _SPATIAL_BARRIER
                ):
                    break
                j -= 1
            # Moving the crop to boundary j changes the shape at every boundary
            # in (j, i]; a user assertion on any of them would be violated, so
            # leave the crop where it is when one is in the way.
            if any(b in assertion_boundaries for b in range(j + 1, i + 1)):
                result.append(i)
                continue
            run = result[j:]
            result[j:] = [i, *run]
        return result

    def _hoist_spatial_windows_inplace(self) -> None:
        """Apply the spatial-window pushdown to this pipeline's ops, in place.

        The Tier-1 entry point (called by ``PipelineGraph.optimize`` per node).
        A no-op when nothing moves, so it is safe to call unconditionally on an
        already-optimized or window-free pipeline.
        """
        order = self._compute_spatial_pushdown(self._ops)
        if order == list(range(len(self._ops))):
            return
        # A crop only moves within a stretch holding no assertion boundary
        # (`_compute_spatial_pushdown`), so no assertion key moves; the replay
        # recomputes every entering shape for the new order.
        self._replay(order, start=self._state_at(0), assertions=self._assertions)

    # ---- Identity elimination (Tier-1) --------------------------------------
    #
    # Delete ops that are value-, dtype-, shape- and channel-preserving no-ops.
    # Which ops *can* be a no-op is the Rust ``IdentityRule`` authority
    # (``op_identity_rule``); the condition is evaluated here against the state
    # entering each op and the state it leaves, both recorded by ``_push_op``
    # (``_entering``). No shape/dtype math is re-implemented, and — unlike the
    # crop-specific ``_is_spatial_window`` recogniser in the pushdown — no op
    # name is matched: the classification lives entirely in the Rust contract.

    def _eliminate_identities_inplace(self) -> None:
        """Drop no-op ops from this pipeline's ops, in place.

        The Tier-1 entry point (called by ``PipelineGraph.optimize`` per node).
        A removed op is a no-op, so it changes no output byte — elimination is
        output-preserving even inside a dict-sink observed or multi-consumer
        node — and every verdict is read off the states recorded before any op
        drops out.

        Conservative around user shape assertions: a node carrying any
        ``assert_shape`` is left untouched, so no positional assertion key has to
        be re-derived across a deletion. Assertions are rare; this keeps the pass
        simple and never silently moves a pinned shape.
        """
        if not self._ops or self._assertions:
            return
        survivors = [
            i for i, op in enumerate(self._ops) if not self._op_is_identity_at(i, op)
        ]
        if len(survivors) == len(self._ops):
            return
        self._replay(survivors, start=self._state_at(0), assertions=self._assertions)

    def _op_is_identity_at(self, index: int, spec: "OpSpec") -> bool:
        """Whether ``spec`` at ``index`` is a removable no-op.

        Reads the op's ``IdentityRule`` and evaluates it against the state
        entering the op and the one it leaves (:meth:`_state_at`). Any unknown —
        an ``auto`` dtype, an unknown dimension, or an expression where a
        literal value is required — resolves to *not* an identity: the pass
        removes an op only when it can prove it does nothing.
        """
        from polars_cv._lib import op_identity_rule, op_infer_shape

        op_json = json.dumps(spec.to_dict(planning_slots))
        rule = op_identity_rule(op_json)
        if rule == "never":
            return False
        if rule == "always":
            # ``Always`` names its identity-deciding params, and the FFI forces
            # ``never`` when any of them is per-row — so an op whose deciding
            # param is an expression (a ``pad`` amount) never reaches here. A
            # remaining expression on an irrelevant param (a ``pad`` fill
            # ``value`` behind zero amounts) leaves the op a genuine no-op, so
            # nothing more to check.
            return True

        entering = self._state_at(index)
        if rule == "when_dtype_preserved":
            if entering.dtype == "auto":
                return False
            return self._state_at(index + 1).dtype == entering.dtype
        if rule == "when_shape_preserved":
            entering_dims = self._entering_dims(entering)
            if entering_dims is None:
                return False
            out_dims = op_infer_shape(op_json, entering_dims)
            if out_dims is None:
                return False
            return _output_shape_equals_input(out_dims, entering_dims)
        return False

    def _entering_dims(self, entering: PlanState) -> "list[int | None] | None":
        """The dimensions of ``entering``, or ``None`` when rank is unknown.

        Length ``ndim``; H/W come from its hints, and every other axis is
        reported ``None`` (unknown). That is enough for the WhenShapePreserved
        ops, whose H/W is the only axis they resize.

        ``None`` too when a shape declaration reached this pipeline
        (``_shape_declared``): the hints may then carry a *claimed* H/W — via a
        CSE suffix that kept the hints but not the assertion, or a lazy
        continuation seeded from an asserting upstream — and deleting an op on
        the strength of a claim changes the output whenever the claim is wrong.
        """
        ndim = entering.ndim
        if ndim is None or self._shape_declared:
            return None
        dims: "list[int | None]" = [None] * ndim
        for axis, dim in enumerate(HINT_DIMS[:2][:ndim]):
            hint = entering.hints.get(dim)
            if hint is not None and not hint.is_expr:
                dims[axis] = int(hint.value)
        return dims

    def _to_spec_dict(self, slot_of: "SlotOf") -> dict:
        """
        Convert pipeline to specification dictionary (without sink).

        Used for graph serialization where sink is handled separately.

        Serialization only serializes: it emits ``self._ops`` verbatim and runs
        no optimization — every pass is applied by ``PipelineGraph.optimize``
        before serialization (see ``polars_cv._optimize``).

        Shape hints are deliberately *not* emitted: no Rust code ever read the
        key, and because ``graph_json`` is the compiled-graph cache key, two
        pipelines that execute identically but carry different hints occupied
        separate cache entries. Plan-time shape still crosses the boundary as
        ``expected_shape`` on the output spec, which Rust does read.

        Args:
            slot_of: The graph's slot resolver (``SlotTable.index``), mapping
                each expression parameter to its plugin input position.

        Returns:
            Dictionary with source and ops.
        """
        return {
            "source": self._source.to_dict(slot_of) if self._source else None,
            "ops": [op.to_dict(slot_of) for op in self._ops],
        }

    # --- Serialization ---

    def _to_json(self) -> str:
        """
        Serialize a linear pipeline spec to JSON for compatibility tests.

        Returns:
            JSON string representation of the pipeline.

        Raises:
            ValueError: If pipeline is incomplete.
        """
        self.validate()

        # A lone pipeline's inputs: its column at 0, then its expressions.
        table = SlotTable()
        table.add(pl.col("__input__"))
        for expr in self._expr_refs:
            table.add(expr)
        return json.dumps(self._to_spec_dict(table.index))

    def _get_expr_columns(self) -> list[pl.Expr]:
        """
        Get all expression columns referenced by this pipeline.

        Returns:
            List of Polars expressions that need to be passed to the plugin.
        """
        return self._expr_refs.copy()

    # --- Repr ---

    def __repr__(self) -> str:
        """Return string representation of pipeline."""
        parts = []
        if self._source:
            parts.append(f"source({self._source.format.value!r})")
        if self._shape_hints.has_any():
            hints = []
            if self._shape_hints.height:
                hints.append(f"height={self._shape_hints.height.value}")
            if self._shape_hints.width:
                hints.append(f"width={self._shape_hints.width.value}")
            if self._shape_hints.channels:
                hints.append(f"channels={self._shape_hints.channels.value}")
            parts.append(f"assert_shape({', '.join(hints)})")
        for op in self._ops:
            params_str = ", ".join(f"{k}={v.value}" for k, v in op.params.items())
            parts.append(f"{op.op}({params_str})")

        return f"Pipeline().{'.'.join(parts)}" if parts else "Pipeline()"

    def explain(
        self,
        *,
        optimized: bool = True,
        opt_flags: "OptFlags | bool | None" = None,
    ) -> str:
        """Render the pipeline's op chain, logical or physical.

        Mirrors Polars' ``.explain(optimized=...)``: the same pipeline can be
        inspected before and after the plan-time optimization phase, so a user
        sees the physical graph differ while the output stays identical.

        Args:
            optimized: When ``False``, render the op chain exactly as written
                (the logical plan). When ``True`` (default), render it after the
                optimization passes selected by ``opt_flags``.
            opt_flags: Which passes to apply when ``optimized`` is ``True`` —
                same coercion as :meth:`LazyPipelineExpr.sink` (``None`` reads
                the env default). Every node-scope logical pass runs, in
                :data:`polars_cv._optimize.OPTIMIZATION_PASSES` order — identity
                elimination, then spatial-window pushdown. Graph-scope CSE is
                inert for a single ``Pipeline`` (it only shares a prefix once
                sibling pipelines meet in a graph), and engine-tier passes are
                Rust lowering with no effect on the logical op chain; both are
                skipped.

        Returns:
            A one-line ``Pipeline().…`` rendering of the chain.
        """
        from polars_cv._graph import PipelineGraph
        from polars_cv._optimize import OPTIMIZATION_PASSES, resolve_opt_flags

        if not optimized:
            return repr(self)
        flags = resolve_opt_flags(opt_flags)
        physical = self._clone()
        # Drive the node passes from the one registry `optimize()` uses, so a new
        # pass is applied here automatically and this never drifts from `.sink()`
        # (which is exactly how identity elimination went missing when it was
        # hand-listed). Only logical, node-scope passes change the op chain this
        # renders: CSE is graph-scope and inert for a lone pipeline, and
        # engine-tier passes are Rust lowering with no effect on the logical ops.
        handlers = PipelineGraph._pass_handlers()
        for spec in OPTIMIZATION_PASSES:
            if spec.tier != "logical":
                continue
            scope, run = handlers[spec.name]
            if scope == "node" and flags.enabled(spec.name):
                run(physical)
        return repr(physical)
