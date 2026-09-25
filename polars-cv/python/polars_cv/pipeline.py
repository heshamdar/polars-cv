"""
Pipeline builder for polars-cv.

This module provides the Pipeline class for building lazy image/array
processing pipelines that can be applied to Polars DataFrame columns.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import polars as pl

from polars_cv._ops_generated import OP_FIELDS, LogicalPass, _OpsMixin
from polars_cv._types import (
    HINT_DIMS,
    CloudOptions,
    DType,
    FloatOrExpr,
    IntOrExpr,
    NullParamPolicy,
    OpSpec,
    ParamValue,
    RowErrorPolicy,
    ScaleOrigin,
    SlotTable,
    SourceFormat,
    SourceSpec,
    _reject_expr,
    _validate_enum,
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


def _asserted_rank(dims: "Sequence[int | None]") -> int:
    """Validate an ``assert_shape(dims=...)`` list and return the rank it pins.

    Entries are literal ``int``\\ s or ``None`` (dimension left unknown). An
    expression is refused rather than tracked: ``dims=`` publishes the output
    schema, and a per-row size is not a plan-time fact — ``height=`` remains
    available for a per-row dimension, where it correctly publishes nothing.

    Rank is capped at ``len(HINT_DIMS)`` because that is how many dimensions
    :class:`PlanState` tracks. Accepting a longer list would silently file
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
    # The tracked state: an immutable record Rust hands back each step.
    "_state": _same,
    "_on_error": _same,
    "_on_null_param": _same,
    # Containers: copied so the clone cannot mutate its origin.
    "_ops": list,
    "_expr_refs": list,
    # Per-op entering states: immutable records, so a shallow copy suffices.
    "_entering": list,
    # `pl.Expr` / `LazyPipelineExpr` elements are shared deliberately — they are
    # graph identities, and deep-copying one would break node reference.
    "_shape_refs": list,
    # Assertion dicts the builders fill in place: deep-copied.
    "_assertions": copy.deepcopy,
}


@dataclass(frozen=True)
class PlanState:
    """The planner's state at one op boundary.

    Computed in Rust (``src/plan.rs``'s ``State``, whose field names these
    are): :meth:`Pipeline._push_op` hands the current one to ``plan_step`` and
    keeps what comes back; a source's comes from ``plan_source`` and an
    assertion's from ``plan_assert``. Python never edits one.

    Attributes:
        domain: ``buffer`` / ``contour`` / ``scalar`` / ``vector``.
        dtype: The element dtype, or ``"auto"`` when not known until decode.
        ndim: The rank, or ``None`` when not known at plan time.
        dims: Known sizes of dimensions 0, 1, 2 (``[H, W, C]`` for an image,
            named by :data:`HINT_DIMS`); ``None`` is unknown or per-row.
        asserted: Which of ``dims`` the user asserted rather than an op
            inferred — a divergence at execution is then theirs.
        declared: A shape declaration reached this lineage, so ``dims`` may
            rest on a claim rather than a fact.
    """

    domain: str = "buffer"
    dtype: str = "auto"
    ndim: "int | None" = None
    dims: "tuple[int | None, int | None, int | None]" = (None, None, None)
    asserted: "tuple[bool, bool, bool]" = (False, False, False)
    declared: bool = False

    @classmethod
    def of(cls, planned: "dict[str, Any]") -> "PlanState":
        """The record for a state Rust returned (a dict of the fields)."""
        return cls(
            domain=planned["domain"],
            dtype=planned["dtype"],
            ndim=planned["ndim"],
            dims=tuple(planned["dims"]),
            asserted=tuple(planned["asserted"]),
            declared=planned["declared"],
        )

    def dim(self, name: str) -> "int | None":
        """The known size of the dimension *name* (one of :data:`HINT_DIMS`)."""
        return self.dims[HINT_DIMS.index(name)]

    def has_all_dims(self) -> bool:
        """Are H, W and C all known?"""
        return all(size is not None for size in self.dims)


class _Position(NamedTuple):
    """What the planner keeps for one op: the state entering it, and a binary
    op's other operand dtype (its two-input dtype rule reads it on replay)."""

    state: PlanState
    other_dtype: "str | None"


#: A shape declaration at one op boundary, in the wire form ``plan_assert``
#: reads (``src/plan.rs``'s ``Assertion``): ``{"ndim": int | None, "dims":
#: [d0, d1, d2], "by_user": bool}``, each ``d`` ``None`` (not declared),
#: ``{"size": n}``, ``"per_row"`` or ``"unknown"``.
Assertion = dict


def _new_assertion(*, by_user: bool) -> Assertion:
    """An assertion declaring nothing yet."""
    return {"ndim": None, "dims": [None, None, None], "by_user": by_user}


def _assertion_window(
    assertions: "dict[int, Assertion]", start: int, end: int
) -> "dict[int, Assertion]":
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
        self._ops: list[OpSpec] = []
        self._expr_refs: list[pl.Expr] = []
        # The planned state after the last op (see `PlanState`): domain,
        # dtype, rank, known sizes, which of them the user asserted, and
        # whether a declaration reached this lineage.
        self._state: PlanState = PlanState()
        # The state entering each op, in step with `_ops`. A slice, a
        # reorder or a deletion of the ops replays them from one of these
        # (`_replay`), and identity elimination judges an op against its own.
        self._entering: list[_Position] = []
        # Shape declarations (`assert_shape`, a canvas taken from another
        # node), keyed by the op boundary they were written at, so a replay or
        # a lazy continuation applies each where it was written.
        self._assertions: dict[int, Assertion] = {}
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
        return self._state.domain

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
        return self._state.dtype

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
        H/W, channels, rank clipping — is one Rust call (``plan_step``),
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
        from polars_cv._lib import plan_step

        planned = PlanState.of(
            plan_step(
                json.dumps(spec.to_dict(planning_slots)), self._state, other_dtype
            )
        )
        self._entering.append(_Position(self._state, other_dtype))
        self._ops.append(spec)
        self._state = planned
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
        assertions: "dict[int, Assertion]",
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
        self._state = start
        self._apply_assertions_at(0)
        for spec, other_dtype in steps:
            self._push_op(spec, other_dtype=other_dtype)

    def _state_at(self, position: int) -> PlanState:
        """The state at op boundary ``position``: entering op ``position``, or
        the current state at the end."""
        if position < len(self._ops):
            return self._entering[position].state
        return self._state

    def _apply_assertions_at(self, position: int) -> None:
        """Check and apply any shape declaration recorded at op ``position``.

        A user assertion outranks whatever the ops inferred, but only from the
        point it was written — which is why it is replayed positionally rather
        than applied once at the end.

        **The single place a declaration is validated as well as applied**, and
        the checks are Rust's (``plan_assert``): a rank already known
        differently, a dimension the rank does not have, or a size that
        disagrees with a known one is refused at the line that wrote it.
        ``assert_shape`` records into ``_assertions`` and calls this, so the
        eager spelling and the lazy continuation's replay run the same checks.
        """
        assertion = self._assertions.get(position)
        if assertion is None:
            return
        from polars_cv._lib import plan_assert

        after_op = self._ops[-1].op if self._ops else None
        self._state = PlanState.of(
            plan_assert(self._state, json.dumps(assertion), after_op)
        )

    @staticmethod
    def _canvas_of(shape: "LazyPipelineExpr") -> "list[Any]":
        """The canvas a ``shape=<node>`` reference declares, per dimension.

        The referenced node's planned H/W are the authority: no contract on the
        rasterize itself can describe a canvas that comes from another node's
        buffer. An unknown size there is declared unknown. Shared by
        ``rasterize(shape=)`` and ``source("contour", shape=)`` — the same mask
        from the same reference, so they cannot disagree.
        """
        height, width, _ = shape._pipeline._state.dims
        return [
            "unknown" if size is None else {"size": size} for size in (height, width)
        ] + [None]

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
        fill_value: IntOrExpr | None = None,
        background: IntOrExpr | None = None,
        # Cloud storage options for file_path sources
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        # Contiguity option for list/array sources
        require_contiguous: bool | None = None,
        # Error handling for source decoding
        on_error: str | None = None,
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

        Each keyword below applies to some formats and not others. Every one
        defaults to ``None`` (the format's own default), and one you pass that
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
            require_contiguous: For "list"/"array", whether to require
                rectangular data (default ``False``).
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
        # (`test_source_applicability_reads_every_parameter`). A keyword was
        # passed iff it is not None.
        passed = {k: v for k, v in locals().items() if k != "self" and v is not None}

        from polars_cv._lib import plan_source
        from polars_cv.lazy import LazyPipelineExpr

        new = self._clone()
        fmt = _validate_enum(passed.pop("format"), SourceFormat, "source format")

        if decode_max_size is not None and (
            not isinstance(decode_max_size, int) or decode_max_size <= 0
        ):
            msg = f"decode_max_size must be a positive int, got {decode_max_size!r}"
            raise ValueError(msg)
        if fmt == SourceFormat.CONTOUR:
            if shape is not None and (width is not None or height is not None):
                msg = (
                    "Cannot specify both 'shape' and explicit dimensions (width/height)"
                )
                raise ValueError(msg)
            if shape is None and (width is None) != (height is None):
                msg = "Both 'width' and 'height' must be specified together"
                raise ValueError(msg)
            if shape is None and width is None:
                msg = (
                    "Contour source requires either:\n"
                    "  1. Both 'width' and 'height' parameters, or\n"
                    "  2. A 'shape' LazyPipelineExpr to infer dimensions from"
                )
                raise ValueError(msg)
        if shape is not None and not isinstance(shape, LazyPipelineExpr):
            msg = "'shape' must be a LazyPipelineExpr"
            raise TypeError(msg)

        # Every keyword the caller passed goes into the spec, whichever format
        # it is for: the format's Rust definition refuses one it does not
        # read, naming where it does apply (`plan_source` below).
        def literal(value: Any) -> ParamValue:
            return ParamValue(is_expr=False, value=value)

        params: dict[str, ParamValue] = {}
        for name, value in passed.items():
            if name == "dtype":
                params[name] = literal(_validate_enum(value, DType, "dtype").value)
            elif name == "shape":
                # The canvas node, by id: Rust takes that node's already-computed
                # buffer. `_shape_refs` is what turns the reference into an
                # upstream edge, so the node is executed (as for `rasterize`).
                params["size"] = literal(value._node_id)
                new._shape_refs.append(value)
            elif name in ("height", "width"):
                params["size"] = literal(
                    [
                        literal(None) if d is None else new._track_expr(d)
                        for d in (height, width)
                    ]
                )
            elif name in ("fill_value", "background"):
                params[name] = new._track_expr(value)
            elif name == "cloud_options":
                options = normalize_cloud_options(value)
                params[name] = literal(None if options is None else options.to_dict())
            elif name == "allowed_roots":
                params[name] = literal(list(value))
            else:
                params[name] = literal(value)
        new._source = SourceSpec(format=fmt, params=params)
        # The format's Rust definition validates the spec (refusing a setting
        # it does not read, naming where it applies) and says what state the
        # decode starts the pipeline in.
        new._state = PlanState.of(
            plan_source(json.dumps(new._source.to_dict(planning_slots)))
        )
        if shape is not None:
            # A contour canvas taken from another node: that node's planned
            # H/W, which no definition of this source can know.
            height, width, _ = shape._pipeline._state.dims
            new._state = dataclasses.replace(
                new._state, dims=(height, width, new._state.dims[2])
            )

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

        if self._source is None:
            msg = "thumbnail() requires a source; call .source(...) first"
            raise ValueError(msg)
        if not isinstance(max_size, int) or isinstance(max_size, bool) or max_size <= 0:
            msg = f"max_size must be a positive int, got {max_size!r}"
            raise ValueError(msg)

        from polars_cv._lib import plan_source

        new = self._clone()
        assert new._source is not None  # guaranteed: checked on self above
        new._source = SourceSpec(
            format=new._source.format,
            params={
                **new._source.params,
                "decode_max_size": ParamValue(is_expr=False, value=max_size),
            },
        )
        # `thumbnail()` writes `decode_max_size`, so it applies exactly where
        # that field does: the source format's own definition decides, as it
        # does for `source(decode_max_size=)`. The two used to disagree about
        # `auto` when each kept its own list.
        try:
            plan_source(json.dumps(new._source.to_dict(planning_slots)))
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
        assertion = new._assertions.setdefault(
            len(new._ops), _new_assertion(by_user=True)
        )
        if dims is not None:
            assertion["ndim"] = _asserted_rank(dims)
            for axis, size in enumerate(dims):
                if size is not None:
                    assertion["dims"][axis] = {"size": size}
        else:
            for dim, value in given.items():
                # A per-row size is declared but is no plan-time fact.
                assertion["dims"][HINT_DIMS.index(dim)] = (
                    "per_row" if isinstance(value, pl.Expr) else {"size": value}
                )
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
        pre_dtype = self._state.dtype
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
        if target is None or new._state.dtype == target:
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

        new = self._clamp(min=min_val, max=max_val)
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
        return self.convert_color(from_space="rgb", to_space="hsv")

    def to_lab(self) -> "Pipeline":
        """Convert from RGB to CIE LAB color space.

        Output dtype is promoted to f32 (L=[0,100], a/b~[-128,127]).

        Returns:
            Self for chaining.
        """
        return self.convert_color(from_space="rgb", to_space="lab")

    def to_bgr(self) -> "Pipeline":
        """Convert from RGB to BGR channel order.

        Returns:
            Self for chaining.
        """
        return self.convert_color(from_space="rgb", to_space="bgr")

    def to_ycbcr(self) -> "Pipeline":
        """Convert from RGB to YCbCr color space.

        Returns:
            Self for chaining.
        """
        return self.convert_color(from_space="rgb", to_space="ycbcr")

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
        return self.convolve2d(kernel=kernel, ksize=ksize, normalize=False)

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
        return self.convolve2d(kernel=laplacian_3, ksize=ksize, normalize=False)

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
        return self.convolve2d(kernel=k, ksize=3, normalize=False)

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
        return self.warp_affine(matrix=matrix, output_size=output_size)

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
        return self.warp_affine(matrix=matrix, output_size=output_size)

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
            # H/W come from `GeometryOp::Rasterize`'s `shape` and the
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
        asserted = new._assertions.setdefault(position, _new_assertion(by_user=False))
        asserted["dims"] = Pipeline._canvas_of(shape)
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

    # --- Node-scope optimisation passes ---

    def _run_node_pass(self, name: str) -> None:
        """Apply the node-scope logical pass ``name`` to this pipeline, in place.

        The pass itself is Rust (``node_pass``, ``src/passes.rs``): it reads
        the ops and the state at every op boundary and answers with the new op
        order — a subset for identity elimination, a permutation for the
        spatial-window pushdown — or ``None`` when nothing changes. The new
        order is committed by :meth:`_replay`, so every per-position fact is
        recomputed for it. Assertion boundaries do not move: identity
        elimination leaves an asserting node alone, and the pushdown never
        moves a crop across one.
        """
        from polars_cv._lib import node_pass

        if not self._ops:
            return
        order = node_pass(
            name,
            [json.dumps(op.to_dict(planning_slots)) for op in self._ops],
            [self._state_at(p) for p in range(len(self._ops) + 1)],
            sorted(self._assertions),
        )
        if order is not None:
            self._replay(order, start=self._state_at(0), assertions=self._assertions)

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
        separate cache entries. Plan-time shape still crosses the boundary in
        each output's ``planned`` state, which Rust does read.

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
        known = [
            f"{dim}={size}"
            for dim, size in zip(HINT_DIMS, self._state.dims)
            if size is not None
        ]
        if known:
            parts.append(f"assert_shape({', '.join(known)})")
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
        for spec in OPTIMIZATION_PASSES:
            if (
                spec.tier == "logical"
                and spec.name != LogicalPass.COMMON_SUBEXPRESSION_ELIMINATION
                and flags.enabled(spec.name)
            ):
                physical._run_node_pass(spec.name)
        return repr(physical)
