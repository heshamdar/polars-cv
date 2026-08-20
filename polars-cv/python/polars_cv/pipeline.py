"""
Pipeline builder for polars-cv.

This module provides the Pipeline class for building lazy image/array
processing pipelines that can be applied to Polars DataFrame columns.
"""

from __future__ import annotations

import copy
import json
import math
from typing import TYPE_CHECKING, Any

import polars as pl

from polars_cv._types import (
    HINT_DIMS,
    SOURCE_PARAM_APPLIES,
    ApproxMethod,
    BoolOrExpr,
    BorderMode,
    CloudOptions,
    ColorSpace,
    Domain,
    DType,
    ExtractMode,
    FetchErrorPolicy,
    FilterType,
    FloatOrExpr,
    HashAlgorithm,
    HistogramClosed,
    HistogramOutput,
    InterpolationType,
    IntOrExpr,
    LabelReduction,
    LabelRegionMode,
    NormalizeMethod,
    NullParamPolicy,
    OpSpec,
    PadMode,
    PadPosition,
    ParamValue,
    RowErrorPolicy,
    ScaleOrigin,
    ShapeAssertion,
    ShapeHints,
    SourceFormat,
    SourceSpec,
    StrOrExpr,
    _reject_expr,
    _validate_enum,
    is_supplied,
    normalize_cloud_options,
    reject_inapplicable_params,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polars_cv._graph import PipelineGraph
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
    operators, which compose identically for floats and for ``pl.Expr``. With
    all-literal inputs every element comes back a plain float, which is what
    keeps plan-time affine fusion (``_literal_matrix_values``) working.

    Args:
        angle_deg: Rotation angle in degrees (positive = clockwise).
        center: ``(cx, cy)`` center of rotation.
        scale: Scale factor.

    Returns:
        Six-element list ``[a, b, tx, c, d, ty]`` (forward mapping).
    """
    if isinstance(angle_deg, pl.Expr):
        rad = angle_deg.radians()
        cos_a = rad.cos() * scale
        sin_a = rad.sin() * scale
    else:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad) * scale
        sin_a = math.sin(rad) * scale
    cx, cy = center
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
        value=[{"type": "literal", "value": float(v)} for v in values],
    )


def _param_list(
    values: "Sequence[Any]",
    track: "Callable[[Any], ParamValue]",
) -> "ParamValue":
    """Serialize a fixed-length parameter list element by element.

    Each element becomes its own ``ParamValue`` dict, so any of them may be a
    per-row expression while the list *length* stays structural — it fixes the
    kernel size, channel count, or target rank at planning time. This is the
    encoding ``reshape`` and ``warp_affine`` already use; Rust reads it back
    with ``resolve_f32_list`` / ``resolve_usize_list``.

    ``track`` is the owning pipeline's ``_track_expr``, so expression elements
    are registered as plugin inputs.
    """
    return ParamValue(
        is_expr=False,
        value=[track(v).to_dict() for v in values],
    )


def _literal_matrix_values(matrix_param: "ParamValue") -> "list[float] | None":
    """Return the six literal floats of a ``warp_affine`` matrix param.

    Returns ``None`` when any element is a per-row expression — such a matrix is
    only resolvable at execution, so it cannot participate in plan-time affine
    fusion (matrix composition needs concrete numbers).
    """
    elements = matrix_param.value
    if not isinstance(elements, list):
        return None
    out: list[float] = []
    for elem in elements:
        if not isinstance(elem, dict) or elem.get("type") != "literal":
            return None
        out.append(float(elem["value"]))
    return out


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
    """Read one operation's Rust contract (domains + rank/channel rules).

    Single entry point so an append reads the contract exactly once and shares
    it between the input-domain check and channel inference.
    """
    from polars_cv._lib import op_contract

    return op_contract(json.dumps(spec.to_dict()))


def _literal_axes(axes: "Sequence[int]", label: str) -> "ParamValue":
    """Build an axis-list parameter, rejecting expressions element-wise.

    Axis lists reorder or select dimensions, so they fix the output rank at
    planning time — unlike value-carrying lists (``convolve2d``'s kernel, a
    ``normalize`` mean/std pair), whose elements may be per-row.
    """
    for axis in axes:
        _reject_expr(axis, f"'{label}'")
    return ParamValue(is_expr=False, value=list(axes))


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


def _enum_param(
    value: "str | pl.Expr",
    enum_cls: type,
    label: str,
    track: "Callable[[Any], ParamValue]",
) -> "ParamValue":
    """Build an enum-valued parameter that may vary per row.

    A literal is validated eagerly against *enum_cls*, exactly as
    :func:`_validate_enum` does. An expression cannot be checked at build time,
    so validation moves to execution, where Rust rejects an unknown value with
    the same "expected one of [...]" error.

    Only use this for enums with **no effect on output shape, rank or dtype** —
    the invariant that lets plan-time shape probing substitute the default (see
    ``ParamCtx::probe`` in ``params.rs``). Structural enums (``cast(dtype)``,
    ``normalize(method)``, ``histogram(output)``) must stay on
    :func:`_validate_enum` plus a literal ``ParamValue``.
    """
    if isinstance(value, pl.Expr):
        return track(value)
    return ParamValue(is_expr=False, value=_validate_enum(value, enum_cls, label).value)


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
    "_initial_output_dtype": _same,
    "_initial_expected_ndim": _same,
    "_on_error": _same,
    "_on_null_param": _same,
    # Containers: copied so the clone cannot mutate its origin.
    "_ops": list,
    "_expr_refs": list,
    "_asserted_dims": set,
    "_hint_snapshots": dict,
    # `pl.Expr` / `LazyPipelineExpr` elements are shared deliberately — they are
    # graph identities, and deep-copying one would break node reference.
    "_shape_refs": list,
    # Mutable value objects the planner writes through: deep-copied.
    "_shape_hints": copy.deepcopy,
    "_assertions": copy.deepcopy,
}


class Pipeline:
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

    # Registry of every operation name a pipeline can emit (via builder methods
    # here and the binary-op helpers in lazy.py). It must be *equal* to the Rust
    # executor's registry (``_lib.known_ops()`` / ``KNOWN_OPS``), not merely a
    # subset: an op here that Rust cannot resolve fails at execution, and an op
    # Rust knows that is missing here cannot be built at all. Both directions
    # are enforced by ``test_registry_parity_*``, and by
    # ``test_op_names_matches_rust_known_ops_without_the_plugin``, which reads
    # KNOWN_OPS from the Rust source so the check still runs when the extension
    # is stale or unbuilt (when the other two quietly skip).
    #
    # It is a hand-written mirror on purpose — deriving it from ``known_ops()``
    # would make importing the builder require the compiled plugin, which the
    # plan-time test lane deliberately does without.
    OP_NAMES: frozenset[str] = frozenset(
        {
            "add",
            "adjust_contrast",
            "adjust_gamma",
            "apply_mask",
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "blend",
            "blur",
            "canny",
            "cast",
            "channel_merge",
            "channel_select",
            "channel_swap",
            "clamp",
            "contour_area",
            "contour_bounding_box",
            "contour_centroid",
            "contour_convex_hull",
            "contour_perimeter",
            "contour_scale",
            "contour_simplify",
            "contour_translate",
            "convolve2d",
            "crop",
            "cvt_color",
            "dilate",
            "divide",
            "equalize_histogram",
            "erode",
            "extract_contours",
            "extract_shape",
            "flip",
            "grayscale",
            "histogram",
            "invert",
            "label_reduce",
            "letterbox",
            "maximum",
            "minimum",
            "morphology_gradient",
            "multiply",
            "normalize",
            "pad",
            "pad_to_size",
            "perceptual_hash",
            "rasterize",
            "ratio",
            "reduce_argmax",
            "reduce_argmin",
            "reduce_max",
            "reduce_mean",
            "reduce_min",
            "reduce_percentile",
            "reduce_popcount",
            "reduce_std",
            "reduce_sum",
            "relu",
            "reshape",
            "resize",
            "resize_max",
            "resize_min",
            "resize_scale",
            "resize_to_height",
            "resize_to_width",
            "rotate",
            "scale",
            "subtract",
            "threshold",
            "transpose",
            "warp_affine",
        }
    )

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
        # Post-source state (before any op), captured by source(). Batch
        # re-folds over the op list (to_graph, CSE prefixes) must seed from
        # here — seeding from the final state double-applies every op.
        self._initial_output_dtype: str = "auto"
        self._initial_expected_ndim: int | None = None
        # Height/width hints as they were ENTERING each op, keyed by op
        # index. Affine fusion reads these so a rotate converts with the
        # shape at its own position, not the pipeline's final shape.
        self._hint_snapshots: dict[
            int, tuple[ParamValue | None, ParamValue | None]
        ] = {}
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
        # Per-row error policy for the executed graph ("raise" by default).
        self._on_error: str = "raise"
        # What a null in a per-row expression parameter means ("raise" by
        # default). Independent of _on_error — see on_null_param().
        self._on_null_param: str = "raise"
        # LazyPipelineExpr nodes referenced by ops (e.g. rasterize(shape=...));
        # consumers wiring this pipeline into a graph add them as upstream
        # dependencies so the referenced node executes first.
        self._shape_refs: "list[LazyPipelineExpr]" = []

    @staticmethod
    def _compute_output_domain_dtype_ndim(
        ops: list["OpSpec"],
        initial_domain: str = "buffer",
        initial_dtype: str = "u8",
        initial_ndim: int | None = None,
    ) -> tuple[str, str, int | None]:
        """
        Fold every operation's schema effect over an initial state.

        Each op's (domain, dtype, ndim) effect comes from the single Rust
        authority ``op_schema`` — including the param-dependent cases (cast
        target, histogram output mode, reduction axis presence) that used to
        be re-implemented here as Python special cases.

        Used by lazy continuations, which seed the fold with the upstream
        node's state; incremental per-append tracking uses the same authority
        via ``_update_output_dtype``, so the two cannot diverge (guarded by
        ``test_pipeline_state_matches_batch_fold``).
        """
        from polars_cv._lib import op_schema

        domain, dtype, ndim = initial_domain, initial_dtype, initial_ndim
        for op_spec in ops:
            domain, dtype, ndim = op_schema(
                json.dumps(op_spec.to_dict()), domain, dtype, ndim
            )
        return domain, dtype, ndim

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
            # Check if we already track this expression
            expr_str = str(value)
            if not any(str(e) == expr_str for e in self._expr_refs):
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
            op_name: The operation name (must be in :attr:`OP_NAMES`).
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

    def _push_op(
        self,
        spec: "OpSpec",
        contract: dict | None = None,
        *,
        update_dtype: bool = True,
    ) -> None:
        """Append ``spec`` **in place** and run its full plan-time update.

        **The single mutator of ``_ops`` in the package.** :meth:`_append_op`
        wraps it for the immutable builder path; the graph hooks
        (:meth:`_add_binary_op`, :meth:`_add_channel_merge`) call it directly
        because they mutate an already-cloned pipeline. Both the schema fold
        and the shape-hint update are unconditional, so no caller can append
        an op while tracking only half its effect.

        The guard is ``test_op_append_is_structurally_exclusive``, which walks
        this module's AST and fails if anything else mutates ``_ops``.

        Args:
            spec: The operation to append.
            contract: A pre-read contract, reused to avoid a second FFI call.
            update_dtype: Only :meth:`_add_binary_op` passes False. A
                two-input dtype rule is not expressible through ``op_schema``;
                the lazy layer resolves it via ``binary_output_dtype``
                instead. The shape-hint update still runs.
        """
        if contract is None:
            contract = _op_contract_for(spec)
        self._require_input_domain(spec, contract)
        # The rank the op *consumes*, captured before the schema fold below
        # advances it — `op_infer_shape` describes a transform of the input.
        input_ndim = self._expected_ndim
        self._ops.append(spec)
        if update_dtype:
            self._update_output_dtype(spec)
        self._update_shape_hints(contract=contract, input_ndim=input_ndim)
        # An assertion recorded *after* this op outranks what the contract
        # inferred. rasterize(shape=<node>) is the case that needs it: its
        # canvas comes from another node's buffer, which no contract on this
        # op can describe.
        self._apply_assertions_at(len(self._ops))

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
          :meth:`_drop_hints_below_rank` enforces against the ops, applied to
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

    def _set_ops_slice(self, ops: "list[OpSpec]", *, shift: int) -> None:
        """Replace the whole op list, re-keying everything keyed by op index.

        The sanctioned wholesale replacement, for CSE (``_graph.py``), which
        splits one pipeline's ops across a shared prefix node and a suffix
        node. Distinct from :meth:`_push_op`, which appends a single op and
        advances the tracked state; here the state is supplied by the caller
        and only the index-keyed side tables move.

        It exists so no code outside this module has to assign ``_ops``
        directly — every such assignment has to remember which side tables are
        keyed by op position, and the CSE path had already forgotten
        ``_assertions``.

        Args:
            ops: The new op list.
            shift: How far each surviving op moved left (``prefix_len`` for a
                suffix node, ``0`` when keeping a prefix).
        """
        self._ops = list(ops)
        self._hint_snapshots = {
            i - shift: v
            for i, v in self._hint_snapshots.items()
            if shift <= i < shift + len(ops)
        }
        self._assertions = {
            i - shift: copy.deepcopy(a)
            for i, a in self._assertions.items()
            if shift <= i <= shift + len(ops)
        }

    def _require_axes_within_rank(self, axes: "Sequence[int]", label: str) -> None:
        """Reject an axis list that does not address the tracked rank.

        The rank is only known some of the time (an ``auto`` source leaves it
        ``None``), so this is a check that fires when it can rather than a
        guarantee. It has to exist because ``infer_shape`` indexes the input
        shape directly: a ``transpose`` carrying three axes over rank-2 data
        would otherwise reach the engine and abort, and the planner calls
        ``infer_shape`` from an ordinary builder where a clean ValueError is
        the contract.
        """
        ndim = self._expected_ndim
        if ndim is None:
            return
        # Only range-check plain integers. A Polars expression here is a
        # *structural* violation with its own error, raised by `_literal_axes`
        # further down; pre-empting it with a range message would bury the
        # real problem.
        bad = [a for a in axes if isinstance(a, int) and not -ndim <= a < ndim]
        if bad:
            msg = (
                f"{label} {list(axes)} is out of range for a {ndim}-dimensional "
                f"input (valid axes: 0..{ndim - 1})."
            )
            raise ValueError(msg)

    def _require_input_domain(self, spec: "OpSpec", contract: dict) -> None:
        """Reject an operation whose input domain is not the current domain.

        The accepted domains are read from the op's Rust contract
        (``op_contract(...)["input_domains"]``) rather than restated in Python.
        It is the same authority the executor dispatches on, so the builder
        cannot disagree with what will actually run — the input-domain mirror
        of ``op_schema`` supplying the output domain.

        It is a *set*: binary ops and reductions accept a buffer or a vector,
        because a perceptual hash is a 1-D buffer encoded as a vector.
        ``Domain::Any`` means the step accepts whatever it is handed.
        """
        accepted = contract["input_domains"]
        if _DOMAIN_ANY in accepted or self._current_domain in accepted:
            return
        expected = " or ".join(accepted)
        raise ValueError(
            f"{spec.op}() expects {expected} input but pipeline is currently "
            f"in {self._current_domain} domain. Add a domain-converting "
            f"operation (e.g., rasterize() for contour→buffer, "
            f"extract_contours() for buffer→contour)."
        )

    def _update_output_dtype(self, spec: "OpSpec") -> None:
        """
        Apply an operation's schema effect (domain, dtype, ndim) to the
        pipeline's tracked state.

        Incremental: exactly one ``op_schema`` FFI call per appended op (the
        old implementation replayed every prior op from the already-evolved
        state — O(n²) FFI calls, and a latent non-idempotency for axis
        reductions' ndim). Domain now comes from the same single authority
        as dtype and ndim; builder methods no longer assign
        ``_current_domain`` by hand.

        ``spec`` is passed rather than read off ``_ops[-1]`` so the contour
        source can fold the same rasterize contract without appending an op it
        does not execute (:meth:`_seed_from_contour_rasterize`).
        """
        from polars_cv._lib import op_schema

        domain, dtype, ndim = op_schema(
            json.dumps(spec.to_dict()),
            self._current_domain,
            self._output_dtype,
            self._expected_ndim,
        )
        self._current_domain = domain
        self._output_dtype = dtype
        self._expected_ndim = ndim

    def _update_shape_hints(self, contract: dict, input_ndim: "int | None") -> None:
        """
        Update shape hints based on the operation being added.

        Height/width come from the op's view-buffer ``infer_shape`` (via
        ``op_infer_shape``) — the single geometry authority — and channels from
        its channel rule via :meth:`_update_channels_from_rule`. No shape math
        is re-implemented in Python.

        Always describes the op just appended (``_ops[-1]``); there is one
        caller, :meth:`_push_op`, and both arguments are required so the
        method cannot be invoked with a silently wrong default.

        Args:
            contract: The op contract :meth:`_push_op` already read, so an
                append still costs a constant number of FFI calls.
            input_ndim: The rank the op consumes, captured before the schema
                fold advances ``_expected_ndim`` — ``infer_shape`` describes a
                transform *of the input*, so the post-op rank would misstate
                every rank-changing op. ``None`` means the rank is genuinely
                unknown, not "look it up".
        """
        # Record the hints ENTERING this op (before the update below) so
        # affine fusion can convert a rotate with the shape at its own
        # position. Any assert_shape() between ops is naturally captured:
        # it mutated _shape_hints before this append.
        # `_push_op` appends before calling, so there is always an op here.
        idx = len(self._ops) - 1
        self._hint_snapshots[idx] = (
            copy.deepcopy(self._shape_hints.height),
            copy.deepcopy(self._shape_hints.width),
        )
        self._apply_shape_contract(self._ops[idx], contract, input_ndim)

    def _apply_shape_contract(
        self, spec: "OpSpec", contract: dict, input_ndim: "int | None"
    ) -> None:
        """Fold one op's shape contract into the hints: H/W, channels, rank.

        Height/width come from the op's view-buffer ``infer_shape`` (via
        ``op_infer_shape``) — the single geometry authority — channels from its
        channel rule, and both are then clipped to the output rank. No shape
        math is re-implemented in Python.

        Shared with the contour source, whose decode *is* a rasterize
        (:meth:`_seed_from_contour_rasterize`), so the source and the
        ``rasterize`` op cannot publish different shapes for the same mask.
        """
        # Every hint below is about to be recomputed from the op's contracts,
        # so nothing survives as "the user asserted this". `_apply_assertions_at`
        # runs immediately after and re-marks whatever it re-declares.
        self._asserted_dims.clear()
        self._update_hw_from_infer_shape(
            spec, self._input_dims_for(contract, input_ndim)
        )
        self._update_channels_from_rule(spec)
        self._drop_hints_below_rank()

    def _drop_hints_below_rank(self) -> None:
        """Discard hints for dimensions the output rank does not have.

        Rank is the authority (``op_schema``); a hint is only meaningful when
        the dimension exists. This is its own invariant, not a patch over the
        channel rule: an op can drop rank while the channel rule still has
        something to say, and a dimension that does not exist cannot have a
        size whatever any rule reports.

        ``channel_select`` is the case that made it load-bearing — it drops
        rank 3 → 2, and a stale channel count surviving onto a rank-2 output is
        how ``expected_shape`` came to publish a three-dimensional shape for
        two-dimensional data.
        """
        ndim = self._expected_ndim
        if ndim is None:
            return
        if ndim < 3:
            self._shape_hints.channels = None
        if ndim < 2:
            self._shape_hints.width = None
        if ndim < 1:
            self._shape_hints.height = None

    def _update_channels_from_rule(self, spec: "OpSpec") -> None:
        """Set the channel hint from the op's view-buffer channel rule.

        Defers to ``op_output_channels``, which runs view-buffer's
        ``OutputChannelRule::apply`` — the same authority that declares the
        rule. Python holds no copy of the arithmetic: alpha handling
        (``StripProcessRestore``), fixed counts, and every "not determinable"
        case are answered once, in Rust.

        This used to re-derive the answer by parsing the stringified rule, and
        the two readings disagreed on ``NotApplicable``: ``apply`` returns
        "no channel count", Python left the hint untouched. See
        ``op_output_channels`` for why that stayed invisible.

        An expression-valued incoming hint enters as ``None`` and so leaves as
        ``None``: a per-row channel count is not a plan-time integer, which is
        exactly how ``expected_shape`` and ``_current_input_dims`` already read
        it. The assertion that produced it is replayed from ``_assertions``, not
        from this hint, so nothing is lost.
        """
        from polars_cv._lib import op_output_channels

        current = self._shape_hints.channels
        input_channels = (
            None if current is None or current.is_expr else int(current.value)
        )
        out = op_output_channels(json.dumps(spec.to_dict()), input_channels)
        self._shape_hints.channels = (
            None if out is None else ParamValue(is_expr=False, value=out)
        )

    def _current_input_dims(self, ndim: int) -> list[int | None]:
        """The current per-dimension sizes as ``op_infer_shape`` input.

        Length ``ndim``; each entry is the known size or ``None`` (unknown /
        expression). The tracked hints hold H (dim 0), W (dim 1), C (dim 2);
        higher dims are unknown.
        """
        dims: list[int | None] = [None] * ndim
        h, w, c = (
            self._shape_hints.height,
            self._shape_hints.width,
            self._shape_hints.channels,
        )
        if ndim >= 1 and h is not None and not h.is_expr:
            dims[0] = int(h.value)
        if ndim >= 2 and w is not None and not w.is_expr:
            dims[1] = int(w.value)
        if ndim >= 3 and c is not None and not c.is_expr:
            dims[2] = int(c.value)
        return dims

    def _input_dims_for(
        self, contract: dict, input_ndim: "int | None"
    ) -> "list[int | None] | None":
        """The input shape to hand ``op_infer_shape``, or ``None`` to not ask.

        ``input_ndim`` is the rank the op *consumes*, and is required rather
        than defaulted: falling back to ``self._expected_ndim`` would read the
        *post*-op rank the schema fold just wrote, which is exactly the
        misstatement this argument exists to prevent.

        An unknown input rank normally means "do not ask" — ``infer_shape``
        indexes the input shape, so a fabricated one would publish a fabricated
        result. A step that *builds* a buffer out of a non-buffer domain is the
        exception, and not by special-casing an op name: it consumes no buffer
        (``input_domains`` excludes it) and produces one, so its output geometry
        comes from its own parameters and there is no input shape to be unknown
        about. ``rasterize`` is the case — its canvas is its ``width``/
        ``height`` — and it is why its explicit-dims form published no shape at
        all while its docstring said ``infer_shape`` supplied one.
        """
        if input_ndim is not None and input_ndim >= 1:
            return self._current_input_dims(input_ndim)
        buffer = Domain.BUFFER.value
        if (
            buffer not in contract["input_domains"]
            and contract["output_domain"] == buffer
        ):
            return []
        return None

    def _update_hw_from_infer_shape(
        self, spec: "OpSpec", dims: "list[int | None] | None"
    ) -> None:
        """Set H/W hints from the op's view-buffer ``infer_shape`` (single
        authority), replacing the old per-op geometry.

        Reads ``op_infer_shape`` — which propagates unknowns (an unknown input
        dim or a per-row expression param yields a ``None`` output dim) — and
        maps the leading two output dims onto the H/W hints. Channels stay with
        :meth:`_update_channels_from_rule`; rank stays with ``op_schema``.

        ``dims`` is the input shape :meth:`_input_dims_for` resolved, or
        ``None`` when the op must not be asked at all.
        """
        if dims is None:
            return
        from polars_cv._lib import op_infer_shape

        try:
            out = op_infer_shape(json.dumps(spec.to_dict()), dims)
        except ValueError:
            # No inferable shape for this step — an axis reduction, a
            # histogram, a channel merge, a binary op, or an op whose params
            # disagree with the input rank.
            #
            # Invalidate rather than keep the pre-op values. Several of these
            # steps *do* change H/W (an axis reduction drops a dimension), so
            # leaving the old hints in place is how a pipeline came to publish
            # `[100, 200, 2]` for data that executes as `[200, 3, 2]`. Unknown
            # is always safe: `expected_shape` reports None and the sink asks
            # for an explicit shape.
            self._shape_hints.height = None
            self._shape_hints.width = None
            return

        def _dim(i: int) -> "ParamValue | None":
            if i < len(out) and out[i] is not None:
                return ParamValue(is_expr=False, value=int(out[i]))
            return None

        self._shape_hints.height = _dim(0)
        self._shape_hints.width = _dim(1)

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
        the same ``op_contract`` / ``op_schema`` / ``op_infer_shape`` FFI
        :meth:`_push_op` uses, and are not restated here. Hard-coding rank 3
        and leaving the dtype ``"auto"`` is what made ``sink("list")`` and
        ``sink("array")`` unplannable on a contour source (both need a concrete
        element dtype) and forced a no-op ``.cast("u8")``.

        The fold runs from the *contour* domain, because that is what the
        column holds — the same transition the op declares, so the two routes
        to a mask cannot publish different plan-time state.

        The spec built here is **not** appended to ``_ops``: the rasterize
        happens inside the source's own decode, and appending it would
        rasterize a second time. Only its contract is read.

        Args:
            shape: The node a ``shape=`` source takes its canvas from, or
                ``None`` for the explicit ``width``/``height`` form. Carried
                into the spec as ``shape_ref`` so the op's own contract reports
                the canvas as unknown, and read for its published H/W below —
                the two halves ``rasterize(shape=)`` also uses.
        """
        source = self._source
        assert source is not None  # set by the caller, immediately above
        params: dict[str, ParamValue] = {
            "fill_value": source.fill_value,
            "background": source.background,
        }
        if shape is not None:
            params["shape_ref"] = ParamValue(is_expr=False, value=shape._node_id)
        else:
            # Both are present together; the builder rejected a lone one above.
            params["width"] = source.width
            params["height"] = source.height
        spec = OpSpec(op="rasterize", params=params)
        contract = _op_contract_for(spec)

        self._current_domain = Domain.CONTOUR.value
        self._update_output_dtype(spec)
        self._apply_shape_contract(spec, contract, input_ndim=None)
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
        # so the applicability check below cannot be given a stale or partial
        # list of them (`test_source_applicability_reads_every_parameter`).
        passed = dict(locals())

        from polars_cv.lazy import LazyPipelineExpr

        new = self._clone()
        fmt = _validate_enum(format, SourceFormat, "source format")
        reject_inapplicable_params(
            kind="source",
            fmt=fmt,
            supplied={
                name: value
                for name, value in passed.items()
                if name not in ("self", "format")
                and is_supplied(value, _source_param_defaults()[name])
            },
            applies=SOURCE_PARAM_APPLIES,
        )

        fetch_policies = tuple(p.value for p in FetchErrorPolicy)
        if on_error not in fetch_policies:
            msg = f"on_error must be one of {fetch_policies}, got '{on_error}'"
            raise ValueError(msg)

        if decode_max_size is not None and (
            not isinstance(decode_max_size, int) or decode_max_size <= 0
        ):
            msg = f"decode_max_size must be a positive int, got {decode_max_size!r}"
            raise ValueError(msg)

        dtype_enum = None
        if dtype is not None:
            dtype_enum = _validate_enum(dtype, DType, "dtype")

        # RAW format always requires dtype (no type metadata in raw bytes)
        # LIST and ARRAY can auto-infer dtype from Polars column type
        if fmt == SourceFormat.RAW and dtype_enum is None:
            msg = "dtype is required for 'raw' source format (raw bytes have no type metadata)"
            raise ValueError(msg)

        # Handle contour source format
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

            # Track expressions for width/height if they are expressions
            width_param = new._track_expr(width) if width is not None else None
            height_param = new._track_expr(height) if height is not None else None

            # Serialize shape pipeline if provided
            shape_pipeline_dict = None
            if shape is not None:
                if not isinstance(shape, LazyPipelineExpr):
                    msg = "'shape' must be a LazyPipelineExpr"
                    raise TypeError(msg)
                # Collect the graph from the shape expression
                shape_pipeline_dict = {
                    "node_id": shape._node_id,
                    "column": str(shape._column),
                    "pipeline": shape._pipeline._to_spec_dict(),
                    "upstream": [u._node_id for u in shape._upstream],
                }
                # Referencing a node by id is not enough to get it executed:
                # `_shape_refs` is what `cv.pipe` / `LazyPipelineExpr.pipe`
                # turn into upstream edges, and only an upstream edge puts a
                # node into the dependency graph. Without this the reference
                # dangles unless the node happens to be reachable some other
                # way (e.g. it is also the image being masked). Mirrors
                # `rasterize(shape=...)` below.
                new._shape_refs.append(shape)

            new._source = SourceSpec(
                format=fmt,
                dtype=dtype_enum,
                width=width_param,
                height=height_param,
                fill_value=new._track_expr(fill_value),
                background=new._track_expr(background),
                shape_pipeline=shape_pipeline_dict,
                on_error=on_error,
            )
            new._seed_from_contour_rasterize(shape=shape)
        else:
            # Reaching here at all means the format accepts them: the
            # applicability check rejected every other format above, so the
            # `fmt in (FILE_PATH, AUTO)` test that used to guard this — and the
            # warn-and-drop branch beside it — are gone rather than restated.
            cloud_opts = normalize_cloud_options(cloud_options)

            new._source = SourceSpec(
                format=fmt,
                dtype=dtype_enum,
                cloud_options=cloud_opts,
                require_contiguous=require_contiguous,
                on_error=on_error,
                decode_max_size=decode_max_size,
                allowed_roots=tuple(allowed_roots)
                if allowed_roots is not None
                else None,
            )
            # Set dtype and ndim based on source format
            if fmt == SourceFormat.RAW:
                # Raw bytes always require explicit dtype (validated above).
                # Raw decodes to a flat 1-D buffer (decode.rs), so rank 1 is a
                # true known value — never guess 3. reshape()/assert_shape()
                # lifts the rank when the caller needs a higher-rank sink.
                assert dtype_enum is not None
                new._expected_ndim = 1
                new._output_dtype = dtype_enum.value
            elif fmt == SourceFormat.BLOB:
                # Blob is self-describing; dtype/ndim unknown until runtime.
                # User may assert dtype for planning (e.g., list/array sinks).
                new._expected_ndim = None
                if dtype_enum is not None:
                    new._output_dtype = dtype_enum.value
                else:
                    new._output_dtype = "auto"
            elif fmt == SourceFormat.AUTO:
                # The concrete decode path is chosen from the column dtype at
                # runtime, so dtype/rank are not knowable here — treat like blob.
                # For List/Array columns Rust does resolve the leaf dtype at
                # plan-time-with-input (resolved_output_specs); a Binary/String
                # column stays "auto" (image dtype isn't known until decode).
                # An explicit dtype assertion still flows through for planning.
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

        # Snapshot the post-source state: batch re-folds over the op list
        # (to_graph, CSE prefixes) seed from these, never from the final
        # per-op-tracked values.
        new._initial_output_dtype = new._output_dtype
        new._initial_expected_ndim = new._expected_ndim

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
        # `thumbnail()` writes `decode_max_size`, so it applies exactly where
        # that parameter does — read from the table rather than restated, which
        # is how the two came to disagree about `auto` (`source()` accepted it,
        # `thumbnail()` refused it, for the same field on the same spec).
        applies = SOURCE_PARAM_APPLIES["decode_max_size"]
        if self._source.format not in applies:
            spelled = ", ".join(sorted(f.value for f in applies))
            msg = (
                f"thumbnail() only applies to {spelled} sources, "
                f"got '{self._source.format.value}'"
            )
            raise ValueError(msg)
        if not isinstance(max_size, int) or isinstance(max_size, bool) or max_size <= 0:
            msg = f"max_size must be a positive int, got {max_size!r}"
            raise ValueError(msg)

        new = self._clone()
        new._source = dataclasses.replace(new._source, decode_max_size=max_size)
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

    def transpose(self, axes: list[int]) -> "Pipeline":
        """
        Transpose dimensions.

        Args:
            axes: New order of axes.

        Returns:
            Self for chaining.
        """
        # Axes are always literals (list of ints)
        self._require_axes_within_rank(axes, "transpose axes")
        if (
            self._expected_ndim is not None
            and all(isinstance(a, int) for a in axes)
            and len(axes) != self._expected_ndim
        ):
            msg = (
                f"transpose axes {list(axes)} must name every one of the "
                f"{self._expected_ndim} input dimensions exactly once."
            )
            raise ValueError(msg)
        return self._append_op(
            "transpose", lambda p: {"axes": _literal_axes(axes, "axes")}
        )

    def reshape(self, shape: list[int | pl.Expr]) -> "Pipeline":
        """
        Reshape array to new dimensions.

        Args:
            shape: New shape (list of ints or expressions).

        Returns:
            Self for chaining.
        """
        # Mixed literal/expr shapes: each entry is tracked independently, so
        # the entry *count* stays structural while any element may be per-row.
        return self._append_op(
            "reshape",
            lambda p: {"shape": _param_list(shape, p._track_expr)},
        )

    def flip(self, axes: list[int]) -> "Pipeline":
        """
        Flip along specified axes.

        Args:
            axes: Axes to flip.

        Returns:
            Self for chaining.
        """
        self._require_axes_within_rank(axes, "flip axes")
        return self._append_op("flip", lambda p: {"axes": _literal_axes(axes, "axes")})

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

    def crop(
        self,
        *,
        top: IntOrExpr = 0,
        left: IntOrExpr = 0,
        height: IntOrExpr | None = None,
        width: IntOrExpr | None = None,
    ) -> "Pipeline":
        """
        Extract a rectangular region.

        Args:
            top: Top offset.
            left: Left offset.
            height: Crop height (None = to end).
            width: Crop width (None = to end).
        """

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            params: dict[str, ParamValue] = {
                "top": p._track_expr(top),
                "left": p._track_expr(left),
            }
            if height is not None:
                params["height"] = p._track_expr(height)
            if width is not None:
                params["width"] = p._track_expr(width)
            return params

        return self._append_op("crop", _params)

    # --- Compute Operations ---

    def cast(self, dtype: str) -> "Pipeline":
        """
        Cast to a different data type.

        Args:
            dtype: Target data type (e.g., "f32", "u8").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If dtype is invalid or domain is not buffer.
        """
        dtype_enum = _validate_enum(dtype, DType, "dtype")
        return self._append_op(
            "cast",
            lambda p: {"dtype": ParamValue(is_expr=False, value=dtype_enum.value)},
        )

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
        — into a ``Configurable`` one that silently downgrades it to f32.

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
        new = self._append_op("scale", lambda p: {"factor": p._track_expr(factor)})
        return self._apply_out_dtype(new, target)

    def normalize(
        self,
        method: str = "minmax",
        mean: list[FloatOrExpr] | None = None,
        std: list[FloatOrExpr] | None = None,
        out_dtype: str | None = None,
    ) -> "Pipeline":
        """
        Normalize values to a standard range.

        Args:
            method: Normalization method. One of:
                - ``"minmax"``: Scale values to [0, 1] range using per-element
                  min/max. Output dtype is f32 by default.
                - ``"zscore"``: Standardize to mean=0, std=1 using per-element
                  statistics. Output dtype is f32 by default.
                - ``"preset"``: Apply ImageNet-style channel-wise normalization
                  using provided ``mean`` and ``std`` values. Each channel is
                  normalized as ``(x - mean[c]) / std[c]``.
            mean: Per-channel mean values. Required when ``method="preset"``.
                Common preset: ``[0.485, 0.456, 0.406]`` (ImageNet). **Each
                element may be a literal float or a Polars expression**, so
                per-row statistics can be joined in as columns; the list
                *length* is the channel count and must be literal.
            std: Per-channel standard deviation values. Required when
                ``method="preset"``. Common preset: ``[0.229, 0.224, 0.225]``
                (ImageNet). Each element accepts an expression, as with
                ``mean``.
            out_dtype: Output dtype (default f32). Normalization always computes
                in f32; the result is then cast to this dtype at execution, so
                the produced dtype always matches the planned dtype. Accepts any
                :class:`DType` name. For half precision use the sink dtype
                instead — ``.sink("numpy", dtype="f16")`` — since the engine has
                no native f16 type.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If method is invalid or preset is missing mean/std.

        Example:
            >>> Pipeline().source().normalize(method="minmax")
            >>> Pipeline().source().normalize(
            ...     method="preset",
            ...     mean=[0.485, 0.456, 0.406],
            ...     std=[0.229, 0.224, 0.225],
            ... )
        """
        method_enum = _validate_enum(method, NormalizeMethod, "normalize method")

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            params: dict[str, ParamValue] = {
                "method": ParamValue(is_expr=False, value=method_enum.value),
            }

            # Handle preset method with mean/std
            if method_enum == NormalizeMethod.PRESET:
                if mean is None or std is None:
                    msg = "method='preset' requires both 'mean' and 'std' parameters"
                    raise ValueError(msg)
                if len(mean) != len(std):
                    msg = (
                        f"mean length ({len(mean)}) must match std length ({len(std)})"
                    )
                    raise ValueError(msg)
                params["mean"] = _param_list(mean, p._track_expr)
                params["std"] = _param_list(std, p._track_expr)
            elif mean is not None or std is not None:
                msg = "mean/std parameters are only valid for method='preset'"
                raise ValueError(msg)

            # Add out_dtype if specified. Normalization computes in f32 and
            # casts the result to this dtype at execution (so plan ==
            # production). Unlike `scale`/`clamp`, this one rides on the op:
            # `Normalize`'s dtype rule is `Configurable(F32)`, the one rule
            # `output_dtype_for` honours an override for, and the runner's
            # `apply_normalize` performs the cast.
            if out_dtype is not None:
                out_dtype_enum = _validate_enum(out_dtype, DType, "out_dtype")
                params["out_dtype"] = ParamValue(
                    is_expr=False, value=out_dtype_enum.value
                )
            return params

        return self._append_op("normalize", _params)

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

        new = self._append_op(
            "clamp",
            lambda p: {
                "min": p._track_expr(min_val),
                "max": p._track_expr(max_val),
            },
        )
        return self._apply_out_dtype(new, target)

    def relu(self) -> "Pipeline":
        """
        Apply ReLU activation (max(0, x)).

        All negative values are set to zero, positive values are unchanged.
        Works on any numeric dtype.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If domain is not buffer.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").relu()
            ```
        """
        return self._append_op("relu", lambda p: {})

    # --- Channel Operations ---

    def channel_select(self, *, index: IntOrExpr) -> "Pipeline":
        """
        Extract a single channel from a multi-channel image.

        Produces a 2D [H, W] buffer from a [H, W, C] input.

        Domain: buffer → buffer

        Args:
            index: Channel index to extract (0-based). Accepts a Polars
                expression for per-row dynamic selection.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").channel_select(index=0)  # Red channel
            ```
        """
        return self._append_op(
            "channel_select", lambda p: {"index": p._track_expr(index)}
        )

    def channel_swap(self, *, order: list[IntOrExpr]) -> "Pipeline":
        """
        Reorder channels in a multi-channel image.

        Domain: buffer → buffer

        Args:
            order: New channel ordering, e.g. [2, 1, 0] for RGB-to-BGR.
                **Each index may be a literal or a Polars expression**, so the
                permutation can vary per row. The list *length* is the channel
                count and must be literal.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").channel_swap(order=[2, 1, 0])
            ```
        """
        return self._append_op(
            "channel_swap", lambda p: {"order": _param_list(order, p._track_expr)}
        )

    # --- Intensity Adjustments ---

    def adjust_contrast(self, *, factor: FloatOrExpr) -> "Pipeline":
        """
        Adjust image contrast.

        Scales pixel deviation from the mean: ``(pixel - mean) * factor + mean``.

        Domain: buffer → buffer

        Args:
            factor: Contrast factor. 1.0 = no change, >1 = more contrast, <1 = less.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").adjust_contrast(factor=1.5)
            ```
        """
        return self._append_op(
            "adjust_contrast", lambda p: {"factor": p._track_expr(factor)}
        )

    def adjust_gamma(self, *, gamma: FloatOrExpr) -> "Pipeline":
        """
        Apply gamma (power-law) correction.

        Normalizes to [0,1], applies ``pixel^gamma``, then denormalizes.

        Domain: buffer → buffer

        Args:
            gamma: Gamma value. <1 = brighter, >1 = darker, 1.0 = no change.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").adjust_gamma(gamma=0.5)
            ```
        """
        return self._append_op(
            "adjust_gamma", lambda p: {"gamma": p._track_expr(gamma)}
        )

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

    def invert(self) -> "Pipeline":
        """
        Invert pixel values.

        For u8: ``255 - pixel``. For float [0,1]: ``1.0 - pixel``.

        Domain: buffer → buffer

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").invert()
            ```
        """
        return self._append_op("invert", lambda p: {})

    # --- Color Space Conversion ---

    def convert_color(self, from_space: str, to_space: str) -> "Pipeline":
        """
        Convert between color spaces.

        Domain: buffer → buffer

        Args:
            from_space: Source color space (rgb, bgr, hsv, lab, ycbcr, gray).
            to_space: Target color space (rgb, bgr, hsv, lab, ycbcr, gray).

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").convert_color("rgb", "hsv")
            ```
        """
        # Validate enum values
        ColorSpace(from_space)
        ColorSpace(to_space)
        return self._append_op(
            "cvt_color",
            lambda p: {
                "from_space": ParamValue(is_expr=False, value=from_space),
                "to_space": ParamValue(is_expr=False, value=to_space),
            },
        )

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

    def convolve2d(
        self,
        kernel: list[FloatOrExpr],
        ksize: IntOrExpr,
        *,
        normalize: BoolOrExpr = False,
        border: StrOrExpr = "replicate",
    ) -> "Pipeline":
        """
        Apply generic 2D convolution with an arbitrary kernel.

        Domain: buffer → buffer

        Args:
            kernel: Flattened kernel values (row-major, ``ksize × ksize``).
                **Each coefficient may be a literal float or a Polars
                expression**, so a batch can convolve with a different kernel
                per row. The kernel *length* is structural and must be a
                literal odd square.
            ksize: Kernel dimension (must be odd; kernel is ``ksize × ksize``).
                Accepts a Polars expression for per-row dynamic values.
            normalize: If True, divide output by the sum of absolute kernel values.
            border: Border handling mode (``"replicate"``, ``"zero"``, ``"reflect"``).

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> edge = Pipeline().source("image_bytes").convolve2d(
            ...     kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
            ...     ksize=3,
            ... )
            ```
        """
        if isinstance(ksize, pl.Expr):
            # `ksize` is only known per row, but the kernel *length* is
            # structural and still checkable: it must be an odd perfect square.
            # Previously an expression `ksize` skipped every check, letting a
            # mismatched kernel reach Rust unvalidated.
            side = math.isqrt(len(kernel))
            if side * side != len(kernel) or side % 2 == 0:
                msg = (
                    f"convolve2d kernel length {len(kernel)} must be the square "
                    "of an odd number (9 for 3x3, 25 for 5x5, ...)"
                )
                raise ValueError(msg)
        else:
            if ksize % 2 == 0:
                msg = f"convolve2d ksize must be odd, got {ksize}"
                raise ValueError(msg)
            if len(kernel) != ksize * ksize:
                msg = (
                    f"kernel length {len(kernel)} doesn't match ksize²={ksize * ksize}"
                )
                raise ValueError(msg)
        return self._append_op(
            "convolve2d",
            lambda p: {
                "kernel": _param_list(kernel, p._track_expr),
                "ksize": p._track_expr(ksize),
                "normalize": p._track_expr(normalize),
                "border": _enum_param(border, BorderMode, "border mode", p._track_expr),
            },
        )

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

        sobel_x_3 = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]
        sobel_y_3 = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]
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

    def canny(
        self,
        *,
        low_threshold: FloatOrExpr = 50.0,
        high_threshold: FloatOrExpr = 150.0,
    ) -> "Pipeline":
        """
        Canny edge detection.

        Applies Gaussian blur, computes Sobel gradients, performs non-maximum
        suppression, and applies double-threshold hysteresis. Output is a U8
        binary edge map (0 or 255).

        Domain: buffer → buffer

        Args:
            low_threshold: Lower hysteresis threshold.
            high_threshold: Upper hysteresis threshold.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> edges = Pipeline().source("image_bytes").canny(low_threshold=50, high_threshold=150)
            ```
        """
        return self._append_op(
            "canny",
            lambda p: {
                "low_threshold": p._track_expr(low_threshold),
                "high_threshold": p._track_expr(high_threshold),
            },
        )

    # --- Morphological Operations ---

    def erode(self, *, ksize: IntOrExpr = 3, iterations: IntOrExpr = 1) -> "Pipeline":
        """
        Morphological erosion (local minimum filter).

        Shrinks bright regions / grows dark regions by computing the minimum
        value in a ``ksize × ksize`` rectangular neighborhood.  Requires
        single-channel input (e.g., after ``.grayscale()`` or ``.threshold()``).

        Domain: buffer → buffer

        Args:
            ksize: Size of the square structuring element. Must be odd and >= 1.
                Accepts a Polars expression for per-row dynamic values.
            iterations: Number of times the erosion is applied.
                Accepts a Polars expression for per-row dynamic values.

        Returns:
            New Pipeline with erosion applied.

        Example:
            ```python
            >>> mask = Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3)
            ```
        """
        return self._append_op(
            "erode",
            lambda p: {
                "ksize": p._track_expr(ksize),
                "iterations": p._track_expr(iterations),
            },
        )

    def dilate(self, *, ksize: IntOrExpr = 3, iterations: IntOrExpr = 1) -> "Pipeline":
        """
        Morphological dilation (local maximum filter).

        Grows bright regions / shrinks dark regions by computing the maximum
        value in a ``ksize × ksize`` rectangular neighborhood.  Requires
        single-channel input (e.g., after ``.grayscale()`` or ``.threshold()``).

        Domain: buffer → buffer

        Args:
            ksize: Size of the square structuring element. Must be odd and >= 1.
                Accepts a Polars expression for per-row dynamic values.
            iterations: Number of times the dilation is applied.
                Accepts a Polars expression for per-row dynamic values.

        Returns:
            New Pipeline with dilation applied.

        Example:
            ```python
            >>> mask = Pipeline().source("image_bytes").grayscale().threshold(128).dilate(ksize=3)
            ```
        """
        return self._append_op(
            "dilate",
            lambda p: {
                "ksize": p._track_expr(ksize),
                "iterations": p._track_expr(iterations),
            },
        )

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

    def morphology_gradient(self, *, ksize: IntOrExpr = 3) -> "Pipeline":
        """
        Morphological gradient (dilate - erode).

        Produces an edge outline by computing the difference between dilation
        and erosion on the same input.  Requires single-channel input.

        Domain: buffer → buffer

        Args:
            ksize: Size of the square structuring element. Must be odd and >= 1.
                Accepts a Polars expression for per-row dynamic values.

        Returns:
            New Pipeline with morphological gradient applied.

        Example:
            ```python
            >>> edges = Pipeline().source("image_bytes").grayscale().threshold(128).morphology_gradient(ksize=3)
            ```
        """
        return self._append_op(
            "morphology_gradient",
            lambda p: {
                "ksize": p._track_expr(ksize),
            },
        )

    # --- Histogram Equalization ---

    def equalize_histogram(self) -> "Pipeline":
        """
        Apply histogram equalization for contrast enhancement.

        Computes the cumulative histogram and maps each pixel through the
        normalized CDF. Operates per-channel on multi-channel images.
        Output is U8.

        Domain: buffer → buffer

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> eq = Pipeline().source("image_bytes").grayscale().equalize_histogram()
            ```
        """
        return self._append_op("equalize_histogram", lambda p: {})

    # --- Image Operations ---

    def resize(
        self,
        *,
        height: IntOrExpr,
        width: IntOrExpr,
        filter: str | pl.Expr = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image to specified dimensions.

        Args:
            height: Target height.
            width: Target width.
            filter: Interpolation: "nearest", "bilinear", "lanczos3" (default).

        Example:
            >>> Pipeline().source("image_bytes").resize(height=224, width=224)
        """

        return self._append_op(
            "resize",
            lambda p: {
                "height": p._track_expr(height),
                "width": p._track_expr(width),
                "filter": _enum_param(filter, FilterType, "filter", p._track_expr),
            },
        )

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

        return self._append_op(
            "resize_scale",
            lambda p: {
                "scale_x": p._track_expr(actual_scale_x),
                "scale_y": p._track_expr(actual_scale_y),
                "filter": _enum_param(filter, FilterType, "filter", p._track_expr),
            },
        )

    def resize_to_height(
        self,
        height: IntOrExpr,
        *,
        filter: str | pl.Expr = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image to target height, preserving aspect ratio.

        Width is computed at runtime as: new_width = height * (input_width / input_height)

        Domain: buffer → buffer

        Args:
            height: Target height (literal or expression).
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").resize_to_height(224)
            ```
        """

        return self._append_op(
            "resize_to_height",
            lambda p: {
                "height": p._track_expr(height),
                "filter": _enum_param(filter, FilterType, "filter", p._track_expr),
            },
        )

    def resize_to_width(
        self,
        width: IntOrExpr,
        *,
        filter: str | pl.Expr = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image to target width, preserving aspect ratio.

        Height is computed at runtime as: new_height = width * (input_height / input_width)

        Domain: buffer → buffer

        Args:
            width: Target width (literal or expression).
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").resize_to_width(224)
            ```
        """

        return self._append_op(
            "resize_to_width",
            lambda p: {
                "width": p._track_expr(width),
                "filter": _enum_param(filter, FilterType, "filter", p._track_expr),
            },
        )

    def resize_max(
        self,
        max_size: IntOrExpr,
        *,
        filter: str | pl.Expr = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image so the maximum dimension equals target, preserving aspect ratio.

        If input is 200x100 and max_size=50, output is 50x25 (width was max, now 50).

        Domain: buffer → buffer

        Args:
            max_size: Target for the maximum dimension (literal or expression).
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> # Ensure no dimension exceeds 224
            >>> pipe = Pipeline().source("image_bytes").resize_max(224)
            ```
        """

        return self._append_op(
            "resize_max",
            lambda p: {
                "max_size": p._track_expr(max_size),
                "filter": _enum_param(filter, FilterType, "filter", p._track_expr),
            },
        )

    def resize_min(
        self,
        min_size: IntOrExpr,
        *,
        filter: str | pl.Expr = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image so the minimum dimension equals target, preserving aspect ratio.

        If input is 200x100 and min_size=50, output is 100x50 (height was min, now 50).

        Domain: buffer → buffer

        Args:
            min_size: Target for the minimum dimension (literal or expression).
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> # Ensure min dimension is at least 224
            >>> pipe = Pipeline().source("image_bytes").resize_min(224)
            ```
        """

        return self._append_op(
            "resize_min",
            lambda p: {
                "min_size": p._track_expr(min_size),
                "filter": _enum_param(filter, FilterType, "filter", p._track_expr),
            },
        )

    # --- Padding Operations ---

    def pad(
        self,
        *,
        top: IntOrExpr = 0,
        bottom: IntOrExpr = 0,
        left: IntOrExpr = 0,
        right: IntOrExpr = 0,
        value: FloatOrExpr = 0.0,
        mode: str | pl.Expr = "constant",
    ) -> "Pipeline":
        """
        Add padding to the image.

        Domain: buffer → buffer

        Args:
            top: Padding on top edge.
            bottom: Padding on bottom edge.
            left: Padding on left edge.
            right: Padding on right edge.
            value: Fill value for "constant" mode (default 0). Accepts a
                Polars expression for per-row dynamic values.
            mode: Padding mode - "constant", "edge", "reflect", "symmetric".

        Returns:
            Self for chaining.

        Raises:
            ValueError: If mode is invalid or current domain is not buffer.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").pad(top=10, bottom=10)
            >>> pipe = Pipeline().source("image_bytes").pad(left=20, right=20, value=128)
            ```
        """

        return self._append_op(
            "pad",
            lambda p: {
                "top": p._track_expr(top),
                "bottom": p._track_expr(bottom),
                "left": p._track_expr(left),
                "right": p._track_expr(right),
                "value": p._track_expr(value),
                "mode": _enum_param(mode, PadMode, "pad mode", p._track_expr),
            },
        )

    def pad_to_size(
        self,
        *,
        height: IntOrExpr,
        width: IntOrExpr,
        position: str | pl.Expr = "center",
        value: FloatOrExpr = 0.0,
    ) -> "Pipeline":
        """
        Pad image to exact target size.

        Dimensions are computed at runtime. If image is larger than target,
        it will NOT be cropped - use resize first if needed.

        Domain: buffer → buffer

        Args:
            height: Target height.
            width: Target width.
            position: Where to place original content:
                - "center": Center content in padded area (default)
                - "top-left": Place at top-left corner
                - "bottom-right": Place at bottom-right corner
            value: Fill value for padding (default 0). Accepts a Polars
                expression for per-row dynamic values.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If position is invalid or current domain is not buffer.

        Example:
            ```python
            >>> # Pad 50x100 image to 100x200, centered
            >>> pipe = Pipeline().source("image_bytes").pad_to_size(height=100, width=200)
            ```
        """

        return self._append_op(
            "pad_to_size",
            lambda p: {
                "height": p._track_expr(height),
                "width": p._track_expr(width),
                "position": _enum_param(
                    position, PadPosition, "position", p._track_expr
                ),
                "value": p._track_expr(value),
            },
        )

    def letterbox(
        self,
        *,
        height: IntOrExpr,
        width: IntOrExpr,
        value: FloatOrExpr = 0.0,
        filter: str | pl.Expr = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image maintaining aspect ratio and pad to exact target size.

        This is a composed operation that:
        1. Resizes the image so it fits within the target dimensions
        2. Pads to reach exact target size with centered positioning

        Domain: buffer → buffer

        Args:
            height: Target height (literal or expression).
            width: Target width (literal or expression).
            value: Fill value for padding (default 0, typically black). Accepts a
                Polars expression for per-row dynamic values.
            filter: Resampling filter for the resize step. Defaults to
                ``"lanczos3"``, which is what letterbox has always used.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> # Letterbox any image to 224x224 for VLM input
            >>> pipe = Pipeline().source("image_bytes").letterbox(height=224, width=224)
            ```
        """

        return self._append_op(
            "letterbox",
            lambda p: {
                "height": p._track_expr(height),
                "width": p._track_expr(width),
                "value": p._track_expr(value),
                "filter": _enum_param(filter, FilterType, "filter", p._track_expr),
            },
        )

    def grayscale(self) -> "Pipeline":
        """
        Convert to grayscale.

        Uses standard luminance formula: 0.299R + 0.587G + 0.114B.
        """
        return self._append_op("grayscale", lambda p: {})

    def threshold(self, value: "IntOrExpr | FloatOrExpr") -> "Pipeline":
        """
        Apply binary threshold.

        Each element is compared against the threshold; the output is a
        U8 binary mask (255 if element > value, 0 otherwise).

        The threshold value range depends on the input dtype:
        - For u8 input: typically 0-255.
        - For float input (e.g., normalized [0, 1]): use a float value like 0.5.

        Args:
            value: Threshold value (int or float, or Polars expression).
        """
        return self._append_op("threshold", lambda p: {"value": p._track_expr(value)})

    def blur(self, sigma: FloatOrExpr) -> "Pipeline":
        """
        Apply Gaussian blur.

        Args:
            sigma: Standard deviation for Gaussian kernel.
        """
        return self._append_op("blur", lambda p: {"sigma": p._track_expr(sigma)})

    def rotate(
        self,
        angle: FloatOrExpr,
        *,
        expand: bool = False,
        interpolation: str | pl.Expr = "bilinear",
        border_value: FloatOrExpr = 0.0,
    ) -> "Pipeline":
        """
        Rotate image by specified angle.

        For angles of 90, 180, or 270 degrees, this uses zero-copy view
        operations (``interpolation`` and ``border_value`` are ignored).
        For arbitrary angles, the rotation is performed via an affine
        transformation using the specified interpolation and border value.

        This is a convenience wrapper around the affine transform family.
        For more control (e.g., combined rotation + scale, or explicit
        output sizing), use :meth:`rotate_and_scale` or :meth:`warp_affine`.

        Domain: buffer -> buffer

        Args:
            angle: Rotation angle in degrees (positive = clockwise).
                Can be a literal float or Polars expression.
            expand: If True, expand output dimensions to fit rotated image.
                If False (default), keep original dimensions (corners may
                be cropped).
            interpolation: Interpolation method for arbitrary angles --
                ``"bilinear"`` (default) or ``"nearest"``. Ignored for
                90/180/270 degree rotations.
            border_value: Fill value for out-of-bounds pixels (default 0).
                Ignored for 90/180/270 degree rotations.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Zero-copy 90-degree rotation
            >>> pipe = Pipeline().source("image_bytes").rotate(90)
            >>>
            >>> # Arbitrary angle with expansion
            >>> pipe = Pipeline().source("image_bytes").rotate(45, expand=True)
            >>>
            >>> # Dynamic angle from column
            >>> pipe = Pipeline().source("image_bytes").rotate(pl.col("angle"))
            >>>
            >>> # Nearest-neighbor interpolation for pixel-art
            >>> pipe = Pipeline().source("image_bytes").rotate(30, interpolation="nearest")
            ```
        """
        return self._append_op(
            "rotate",
            lambda p: {
                "angle": p._track_expr(angle),
                "expand": ParamValue(is_expr=False, value=expand),
                "interpolation": _enum_param(
                    interpolation, InterpolationType, "interpolation", p._track_expr
                ),
                "border_value": p._track_expr(border_value),
            },
        )

    # --- Affine Transform Operations ---

    def warp_affine(
        self,
        matrix: list[FloatOrExpr],
        output_size: tuple[IntOrExpr, IntOrExpr],
        *,
        interpolation: str | pl.Expr = "bilinear",
        border_value: FloatOrExpr = 0.0,
    ) -> "Pipeline":
        """
        Apply a 2x3 affine transformation matrix.

        The matrix ``[a, b, tx, c, d, ty]`` is a **forward** mapping from
        source to destination (same convention as OpenCV ``warpAffine``)::

            x_dst = a * x_src + b * y_src + tx
            y_dst = c * x_src + d * y_src + ty

        The kernel inverts this matrix internally for interpolation.

        Domain: buffer → buffer

        Args:
            matrix: Six-element sequence representing the 2x3 affine matrix
                ``[a, b, tx, c, d, ty]`` (forward mapping). **Each element may be
                a literal float or a Polars expression**, so a batch can apply a
                different (e.g. random) affine per row in one call — the matrix is
                resolved per row at execution.
            output_size: ``(height, width)`` of the output image. Each element
                accepts a Polars expression for per-row dynamic values.
            interpolation: Interpolation method -- ``"bilinear"`` (default)
                or ``"nearest"``.
            border_value: Pixel value for out-of-bounds regions (default 0).

        Returns:
            Self for chaining.

        Raises:
            ValueError: If *matrix* does not have 6 elements or domain is wrong.

        Example:
            ```python
            >>> # Translate image by (50, 30)
            >>> pipe = Pipeline().source("image_bytes").warp_affine(
            ...     matrix=[1.0, 0.0, 50.0, 0.0, 1.0, 30.0],
            ...     output_size=(224, 224),
            ... )
            >>>
            >>> # Per-sample random affine: each row uses its own matrix columns
            >>> pipe = Pipeline().source("image_bytes").warp_affine(
            ...     matrix=[pl.col("a"), pl.col("b"), pl.col("tx"),
            ...             pl.col("c"), pl.col("d"), pl.col("ty")],
            ...     output_size=(224, 224),
            ... )
            ```
        """
        matrix = list(matrix)
        if len(matrix) != 6:
            msg = f"Affine matrix must have 6 elements, got {len(matrix)}"
            raise ValueError(msg)
        h, w = output_size
        # Each matrix element is tracked independently so any of them may be a
        # per-row expression (resolved element-by-element in Rust via
        # as_param_list); the element *count* stays structural.
        return self._append_op(
            "warp_affine",
            lambda p: {
                "matrix": _param_list(matrix, p._track_expr),
                "output_height": p._track_expr(h),
                "output_width": p._track_expr(w),
                "interpolation": _enum_param(
                    interpolation, InterpolationType, "interpolation", p._track_expr
                ),
                "border_value": p._track_expr(border_value),
            },
        )

    def shear(
        self,
        *,
        sx: FloatOrExpr = 0.0,
        sy: FloatOrExpr = 0.0,
        output_size: tuple[IntOrExpr, IntOrExpr] | None = None,
    ) -> "Pipeline":
        """
        Apply a shear transformation.

        Convenience wrapper that builds a shear matrix and delegates to
        :meth:`warp_affine`.

        Domain: buffer → buffer

        Args:
            sx: Horizontal shear factor (literal or per-row Polars expression).
            sy: Vertical shear factor (literal or per-row Polars expression).
            output_size: ``(height, width)`` of the output. Required
                (auto-sizing not yet implemented).

        Returns:
            Self for chaining.

        Raises:
            ValueError: If *output_size* is not provided.

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
        # TODO: auto-compute output_size from input shape + shear if not provided
        if output_size is None:
            msg = "output_size is required for shear (auto-size not yet implemented)"
            raise ValueError(msg)
        # sx/sy may be per-row expressions; warp_affine tracks each matrix
        # element independently, so the shear matrix passes them through.
        matrix: list[FloatOrExpr] = [1.0, sx, 0.0, sy, 1.0, 0.0]
        return self.warp_affine(matrix, output_size)

    def rotate_and_scale(
        self,
        *,
        angle: FloatOrExpr,
        scale: FloatOrExpr = 1.0,
        center: tuple[FloatOrExpr, FloatOrExpr] | None = None,
        output_size: tuple[IntOrExpr, IntOrExpr] | None = None,
    ) -> "Pipeline":
        """
        Combined rotation and scaling around a center point.

        Convenience wrapper that builds a rotation+scale matrix and delegates
        to :meth:`warp_affine`.

        Domain: buffer → buffer

        Args:
            angle: Rotation angle in degrees (positive = clockwise). Accepts a
                Polars expression for a per-row angle.
            scale: Scale factor (default 1.0). Accepts an expression.
            center: ``(cx, cy)`` center of rotation. Required
                (auto-compute not yet implemented). Each element accepts an
                expression.
            output_size: ``(height, width)`` of the output. Required
                (auto-sizing not yet implemented). Each element accepts an
                expression.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If *center* or *output_size* is not provided.

        Note:
            A matrix built from expressions cannot participate in plan-time
            affine fusion, which needs concrete numbers to compose matrices;
            such a call executes as its own warp instead of being folded into
            a neighbouring one.

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
        # TODO: auto-compute center from input shape if not provided
        if center is None:
            msg = "center is required for rotate_and_scale (auto-compute not yet implemented)"
            raise ValueError(msg)
        if output_size is None:
            msg = "output_size is required for rotate_and_scale (auto-size not yet implemented)"
            raise ValueError(msg)
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

        # `algorithm` is paired with the structural `hash_size` and stays
        # literal; reject an expression here rather than letting it fall past
        # the isinstance check and explode on `.value`.
        _reject_expr(algorithm, "perceptual_hash 'algorithm'")
        if isinstance(algorithm, str):
            algorithm = _validate_enum(algorithm, HashAlgorithm, "algorithm")

        if isinstance(hash_size, pl.Expr):
            msg = (
                "hash_size is structural (it fixes the output vector length at "
                "planning time) and must be a literal, not a Polars expression."
            )
            raise TypeError(msg)
        if hash_size <= 0:
            msg = "hash_size must be a positive integer"
            raise ValueError(msg)

        # Transitions to the vector domain (fixed-length 1-D u8 fingerprint).
        # The domain comes from the op's Rust contract (GraphStep::PerceptualHash
        # → Domain::Vector), read via op_schema — not assigned here.
        return self._append_op(
            "perceptual_hash",
            lambda p: {
                "algorithm": ParamValue(is_expr=False, value=algorithm.value),
                "hash_size": ParamValue(is_expr=False, value=hash_size),
            },
        )

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

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            params: dict[str, ParamValue] = {
                "fill_value": p._track_expr(fill_value),
                "background": p._track_expr(background),
            }

            if has_explicit:
                if width is None or height is None:
                    msg = "Both width and height must be specified"
                    raise ValueError(msg)
                params["width"] = p._track_expr(width)
                params["height"] = p._track_expr(height)
                # No hint assignment here: the canvas size is fixed by these
                # params, so `GeometryOp::Rasterize::infer_shape` is the
                # authority and `_push_op` reads it via `op_infer_shape`.
                # Setting the hints here instead made them a side effect of
                # building the params, which the lazy continuation replay
                # (which re-pushes an already-built spec) silently skipped.
            else:
                # 'shape' parameter - store as reference for graph composition.
                # This will be resolved during graph execution.
                from polars_cv.lazy import LazyPipelineExpr

                if not isinstance(shape, LazyPipelineExpr):
                    msg = "'shape' must be a LazyPipelineExpr"
                    raise TypeError(msg)
                params["shape_ref"] = ParamValue(is_expr=False, value=shape._node_id)
                # The referenced node must execute before this one; graph wiring
                # (cv.pipe / LazyPipelineExpr.pipe) adds it as an upstream dep.
                p._shape_refs.append(shape)
                # Recorded as an assertion at this op's position: the canvas
                # comes from another node's buffer, so no contract on *this*
                # op can supply it. Assertions are replayed positionally, so
                # this survives a continuation like a user `assert_shape`.
                # Recorded one position *past* this op, so it is applied after
                # the op's own (unknown) inferred shape rather than before.
                #
                # Tagged `shape_ref`, not `assert_shape`: the canvas comes from
                # another node's *inferred* hints, so if execution disagrees
                # that is a contract bug and keeps the contract-bug wording.
                asserted = p._assertions.setdefault(
                    len(p._ops) + 1, ShapeAssertion(source="shape_ref")
                )
                for dim, concrete in Pipeline._shape_ref_dims(shape).items():
                    setattr(p._shape_hints, dim, concrete)
                    asserted.dims[dim] = concrete
            return params

        # H/W come from `GeometryOp::Rasterize::infer_shape` for the explicit
        # width/height form, and from the referenced node for the `shape=`
        # form; the single-channel output comes from the op's `fixed:1`
        # channel rule. None of it is re-derived here.
        return self._append_op("rasterize", _params)

    def extract_contours(
        self,
        *,
        mode: str | pl.Expr = "external",
        method: str | pl.Expr = "simple",
        min_area: FloatOrExpr | None = None,
    ) -> "Pipeline":
        """
        Extract contours from binary mask.

        Args:
            mode: "external" (outer only), "tree" (full hierarchy), "all".
            method: "simple" (remove redundant), "none" (all points), "approx".
            min_area: Filter small contours. Accepts a Polars expression for
                per-row dynamic thresholds.

        The traced outline passes through the **centres** of the boundary pixels,
        so it sits half a pixel inside the region it describes: a blob filling
        ``w x h`` pixels comes back bounding ``(w-1) x (h-1)``. Rasterizing the
        result therefore erodes it by a pixel per round trip.

        Borders come back as a flat list with no hierarchy. ``mode="all"`` yields
        the exterior plus one border for each enclosed background region — holes
        that touch or nest enclose one region between them — and reassembling a
        holed contour from those is the caller's job. ``mode="external"`` keeps
        only the outermost, discarding hole borders.

        Domain transition: buffer → contour
        """

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            params: dict[str, ParamValue] = {
                "mode": _enum_param(mode, ExtractMode, "mode", p._track_expr),
                "method": _enum_param(method, ApproxMethod, "method", p._track_expr),
            }
            if min_area is not None:
                params["min_area"] = p._track_expr(min_area)
            return params

        return self._append_op("extract_contours", _params)

    # --- Buffer Reduction Operations (buffer → scalar) ---

    def reduce_sum(self) -> "Pipeline":
        """
        Sum all elements in the buffer.

        Domain transition: buffer → scalar
        """
        return self._append_op("reduce_sum", lambda p: {})

    def reduce_percentile(self, q: FloatOrExpr) -> "Pipeline":
        """
        Compute the q-th percentile of all values.

        Uses linear interpolation matching numpy.percentile default behavior.

        Args:
            q: Percentile to compute, in [0, 100]. Accepts a Polars expression
                for per-row dynamic values.

        Domain transition: buffer -> scalar
        """
        return self._append_op("reduce_percentile", lambda p: {"q": p._track_expr(q)})

    def reduce_popcount(self) -> "Pipeline":
        """
        Count set bits (1s) in the buffer.

        Domain transition: buffer → scalar
        """
        return self._append_op("reduce_popcount", lambda p: {})

    def reduce_max(self, axis: int | None = None) -> "Pipeline":
        """
        Reduce buffer by computing the maximum value.

        When axis is None, computes the global maximum across all elements,
        returning a single scalar. When axis is specified, reduces along that
        axis, returning a buffer with one fewer dimension.

        Domain transition:
            - axis=None: buffer → scalar
            - axis=N: buffer → buffer (reduced shape)

        Args:
            axis: Axis to reduce along. None for global reduction.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Global maximum
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_max()
            >>> df.with_columns(max_val=pl.col("image").cv.pipe(pipe).sink("native"))
            >>>
            >>> # Maximum along height axis (returns 1D array per column)
            >>> pipe = Pipeline().source("image_bytes").reduce_max(axis=0)
            ```
        """

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            params: dict[str, ParamValue] = {}
            if axis is not None:
                params["axis"] = ParamValue(is_expr=False, value=axis)
            return params

        return self._append_op("reduce_max", _params)

    def reduce_min(self, axis: int | None = None) -> "Pipeline":
        """
        Reduce buffer by computing the minimum value.

        When axis is None, computes the global minimum across all elements,
        returning a single scalar. When axis is specified, reduces along that
        axis, returning a buffer with one fewer dimension.

        Domain transition:
            - axis=None: buffer → scalar
            - axis=N: buffer → buffer (reduced shape)

        Args:
            axis: Axis to reduce along. None for global reduction.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Global minimum
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_min()
            >>> df.with_columns(min_val=pl.col("image").cv.pipe(pipe).sink("native"))
            >>>
            >>> # Minimum along width axis
            >>> pipe = Pipeline().source("image_bytes").reduce_min(axis=1)
            ```
        """

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            params: dict[str, ParamValue] = {}
            if axis is not None:
                params["axis"] = ParamValue(is_expr=False, value=axis)
            return params

        return self._append_op("reduce_min", _params)

    def reduce_mean(self, axis: int | None = None) -> "Pipeline":
        """
        Compute arithmetic mean.

        Args:
            axis: Axis to reduce along. If None, computes global mean.

        Domain transition:
            - axis=None: buffer → scalar
            - axis=N: buffer → buffer (reduced shape)
        """

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            params: dict[str, ParamValue] = {}
            if axis is not None:
                params["axis"] = ParamValue(is_expr=False, value=axis)
            return params

        return self._append_op("reduce_mean", _params)

    def reduce_std(self, axis: int | None = None, ddof: IntOrExpr = 0) -> "Pipeline":
        """
        Reduce buffer by computing the standard deviation.

        When axis is None, computes the global standard deviation across all
        elements, returning a single scalar. When axis is specified, reduces
        along that axis, returning a buffer with one fewer dimension.

        Domain transition:
            - axis=None: buffer -> scalar
            - axis=N: buffer -> buffer (reduced shape)

        Args:
            axis: Axis to reduce along. None for global reduction.
            ddof: Delta degrees of freedom. 0 for population std (default),
                1 for sample std. Accepts a Polars expression for per-row
                dynamic values.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Global standard deviation
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_std()
            >>> df.with_columns(std=pl.col("image").cv.pipe(pipe).sink("native"))
            >>>
            >>> # Sample std (ddof=1)
            >>> pipe = Pipeline().source("image_bytes").reduce_std(ddof=1)
            ```
        """

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            params: dict[str, ParamValue] = {"ddof": p._track_expr(ddof)}
            if axis is not None:
                params["axis"] = ParamValue(is_expr=False, value=axis)
            return params

        return self._append_op("reduce_std", _params)

    def reduce_argmax(self, axis: int) -> "Pipeline":
        """
        Reduce buffer by finding the index of the maximum value along an axis.

        Unlike other reductions, argmax always requires an axis since the global
        argmax would be ambiguous for multi-dimensional arrays.

        Domain transition: buffer → buffer (reduced shape, i64 dtype)

        Args:
            axis: Axis along which to find the maximum index.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Find column with max value per row
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_argmax(axis=1)
            >>> df.with_columns(max_col=pl.col("image").cv.pipe(pipe).sink("list"))
            ```
        """
        # argmax always returns a buffer with reduced shape (indices)
        return self._append_op(
            "reduce_argmax",
            lambda p: {"axis": ParamValue(is_expr=False, value=axis)},
        )

    def reduce_argmin(self, axis: int) -> "Pipeline":
        """
        Reduce buffer by finding the index of the minimum value along an axis.

        Unlike other reductions, argmin always requires an axis since the global
        argmin would be ambiguous for multi-dimensional arrays.

        Domain transition: buffer → buffer (reduced shape, i64 dtype)

        Args:
            axis: Axis along which to find the minimum index.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Find column with min value per row
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_argmin(axis=1)
            >>> df.with_columns(min_col=pl.col("image").cv.pipe(pipe).sink("list"))
            ```
        """
        # argmin always returns a buffer with reduced shape (indices)
        return self._append_op(
            "reduce_argmin",
            lambda p: {"axis": ParamValue(is_expr=False, value=axis)},
        )

    def extract_shape(self) -> "Pipeline":
        """
        Extract buffer shape as a struct {height, width, channels}.

        Domain transition: buffer → vector
        """
        return self._append_op("extract_shape", lambda p: {})

    def label_reduce(
        self,
        *,
        contours: pl.Expr,
        reduction: str | pl.Expr = "max",
        region_mode: str | pl.Expr = "interior",
    ) -> "Pipeline":
        """
        Score contour regions against the current buffer values.

        This is the buffer-space variant of label reduction. It accepts contours
        via a Polars expression and returns one score per contour.

        Domain transition: buffer -> vector

        Args:
            contours: Contour-set expression (`List[Contour]`) to score.
            reduction: Reduction over contour region values (`"max"`, `"mean"`, `"sum"`).
            region_mode: Region selection mode.
                ``"interior"`` — only pixels strictly inside the contour polygon.
                ``"boundary"`` — interior pixels *plus* pixels on the contour boundary
                (avoids zero-score artifacts for sub-pixel contours).
                ``"bbox"`` — all pixels within the bounding box.

        Returns:
            New pipeline with label reduction appended.

        Raises:
            ValueError: If current domain is not buffer or args are invalid.
            TypeError: If `contours` is not a Polars expression.
        """
        if not isinstance(contours, pl.Expr):
            msg = "`contours` must be a Polars expression"
            raise TypeError(msg)
        return self._append_op(
            "label_reduce",
            lambda p: {
                "contours": p._track_expr(contours),
                "reduction": _enum_param(
                    reduction, LabelReduction, "reduction", p._track_expr
                ),
                "region_mode": _enum_param(
                    region_mode, LabelRegionMode, "region_mode", p._track_expr
                ),
            },
        )

    def histogram(
        self,
        bins: IntOrExpr | list[float] = 256,
        range: tuple[FloatOrExpr, FloatOrExpr] | None = None,
        closed: str = "left",
        output: str = "buckets",
    ) -> "Pipeline":
        """
        Compute pixel value histogram.

        Args:
            bins: Number of bins (default 256), a Polars expression for
                per-row dynamic bin count, or an explicit list of bin edges.
            range: (min, max) tuple. Auto-detected if None.
            closed: "left" or "right" interval inclusiveness (default "left").
            output: "buckets" (list of structs), "counts" (bin counts),
                    "normalized" (sum to 1.0), "quantized" (pixel indices),
                    "edges" (bin edges).

        Example:
            >>> Pipeline().source("image_bytes").grayscale().histogram(bins=8)
        """

        # Validate output mode
        output_mode = _validate_enum(output, HistogramOutput, "histogram output mode")
        closed_mode = _validate_enum(closed, HistogramClosed, "closed mode")

        def _params(p: "Pipeline") -> dict[str, ParamValue]:
            bins_param: ParamValue
            if isinstance(bins, list):
                bins_param = ParamValue(is_expr=False, value=bins)
            else:
                bins_param = p._track_expr(bins)

            params: dict[str, ParamValue] = {
                "bins": bins_param,
                "closed": ParamValue(is_expr=False, value=closed_mode.value),
                "output": ParamValue(is_expr=False, value=output_mode.value),
            }
            if range is not None:
                params["range_min"] = p._track_expr(range[0])
                params["range_max"] = p._track_expr(range[1])
            return params

        return self._append_op("histogram", _params)

    # --- Contour Measure Operations (contour → scalar/vector) ---

    def area(self, *, signed: BoolOrExpr = False) -> "Pipeline":
        """
        Compute the area of the contour using the Shoelace formula.

        Domain transition: contour → scalar

        Args:
            signed: If True, return signed area (negative for CW winding).

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        return self._append_op(
            "contour_area", lambda p: {"signed": p._track_expr(signed)}
        )

    def perimeter(self) -> "Pipeline":
        """
        Compute the perimeter (arc length) of the contour.

        Domain transition: contour → scalar

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        return self._append_op("contour_perimeter", lambda p: {})

    def centroid(self) -> "Pipeline":
        """
        Compute the centroid (center of mass) of the contour.

        Domain transition: contour → vector (returns [x, y])

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        return self._append_op("contour_centroid", lambda p: {})

    def bounding_box(self) -> "Pipeline":
        """
        Compute the axis-aligned bounding box of the contour.

        Domain transition: contour → vector (returns [x, y, width, height])

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        return self._append_op("contour_bounding_box", lambda p: {})

    # --- Contour Transform Operations (contour → contour) ---

    def translate(self, *, dx: FloatOrExpr, dy: FloatOrExpr) -> "Pipeline":
        """
        Translate the contour by an offset.

        Domain: contour → contour

        Args:
            dx: X offset (horizontal translation).
            dy: Y offset (vertical translation).

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        return self._append_op(
            "contour_translate",
            lambda p: {
                "dx": p._track_expr(dx),
                "dy": p._track_expr(dy),
            },
        )

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

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.

        Note:
            The default is ``"centroid"``, which is what this method has always
            done — it previously hardcoded it with no way to choose. The
            ``.contour.scale`` accessor defaults to ``"origin"`` instead; pass
            *origin* explicitly if you need the two to agree.
        """
        return self._append_op(
            "contour_scale",
            lambda p: {
                "sx": p._track_expr(sx),
                "sy": p._track_expr(sy),
                "origin": _enum_param(
                    origin, ScaleOrigin, "scale_contour origin", p._track_expr
                ),
            },
        )

    def simplify(self, *, tolerance: FloatOrExpr) -> "Pipeline":
        """
        Simplify the contour using Douglas-Peucker algorithm.

        Domain: contour → contour

        Args:
            tolerance: Maximum distance from original contour.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        return self._append_op(
            "contour_simplify", lambda p: {"tolerance": p._track_expr(tolerance)}
        )

    def convex_hull(self) -> "Pipeline":
        """
        Compute the convex hull of the contour.

        Domain: contour → contour

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        return self._append_op("contour_convex_hull", lambda p: {})

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
            >>> pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
            >>> graph = pipe.to_graph(pl.col("image"))
            >>> expr = graph.to_expr()
            ```
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

        # The op slice carries its position-keyed side tables with it, so
        # affine fusion in the sub-pipeline still sees per-position shapes.
        sub._set_ops_slice(self._ops[start_op:end_op], shift=start_op)

        # Compute the correct domain and dtype for this subset of operations.
        # The fold covers ops[0:end_op], so it must be seeded with the
        # post-source (pre-op) state — seeding with the pipeline's final
        # state would apply every op a second time.
        ops_to_compute = self._ops[0:end_op]
        domain, dtype, ndim = Pipeline._compute_output_domain_dtype_ndim(
            ops_to_compute,
            initial_dtype=self._initial_output_dtype,
            initial_ndim=self._initial_expected_ndim,
        )
        sub._current_domain = domain
        sub._output_dtype = dtype
        sub._expected_ndim = ndim

        return sub

    # --- Graph Composition Support ---

    def _add_binary_op(
        self,
        op: str,
        other_node_id: str,
        **kwargs,
    ) -> None:
        """
        Add a binary operation referencing another node.

        This is used internally by LazyPipelineExpr composition.

        Args:
            op: Operation name (e.g., "add", "multiply", "apply_mask").
            other_node_id: The node ID of the other operand.
            **kwargs: Additional operation parameters.
        """
        params: dict[str, ParamValue] = {
            "other_node": ParamValue(is_expr=False, value=other_node_id),
        }
        # `other_node` above is graph topology and stays literal; the
        # remaining kwargs are ordinary op params (e.g. `apply_mask(invert)`),
        # so an expression among them resolves per row like anywhere else.
        for key, value in kwargs.items():
            params[key] = self._track_expr(value)

        # Binary ops are elementwise, so H/W pass through unchanged — but the
        # append still routes through `_push_op`, which records the
        # entering-hints snapshot and applies the channel rule. `op_schema`
        # cannot express a two-input dtype rule, so the dtype is left to the
        # lazy layer's `binary_output_dtype`.
        self._push_op(OpSpec(op=op, params=params), update_dtype=False)

    def _add_channel_merge(self, other_node_ids: list[str]) -> None:
        """
        Add a ``channel_merge`` op referencing other buffer nodes.

        Stacks this pipeline's single-channel ``[H, W]`` buffer with the
        single-channel buffers produced by ``other_node_ids`` along a new
        channel axis, yielding ``[H, W, C]`` (``C = len(other_node_ids) + 1``).
        Used internally by :meth:`LazyPipelineExpr.channel_merge`.

        Args:
            other_node_ids: Node IDs of the other single-channel operands.
        """
        # Rank ([H, W] → [H, W, C]) and channel count change; both are sourced
        # from the Rust contract (op_schema for domain/dtype/ndim, the channel
        # rule for the channel hint) rather than re-declared here.
        self._push_op(
            OpSpec(
                op="channel_merge",
                params={
                    "other_nodes": ParamValue(is_expr=False, value=other_node_ids),
                },
            )
        )

    def _fuse_affine_ops(self, ops: list[OpSpec]) -> list[OpSpec]:
        """Compose consecutive affine-compatible ops into a single ``warp_affine``.

        Both ``warp_affine`` and ``rotate`` (with static, non-90/180/270
        angles) participate in fusion.  Matrix composition uses 3x3
        homogeneous multiplication so that ``rotate → translate → shear``
        becomes one affine warp at execution time, avoiding redundant
        interpolation passes.

        ``rotate`` ops with expression-based angles or zero-copy angles
        (90/180/270) are left as-is and break a fusion run.

        Args:
            ops: The list of pipeline operations.

        Returns:
            Optimized list where runs of affine ops are collapsed.
        """
        if len(ops) < 2:
            return ops

        result: list[OpSpec] = []
        i = 0
        while i < len(ops):
            converted = self._try_convert_rotate_to_affine(ops[i], op_index=i)
            if converted is None:
                result.append(ops[i])
                i += 1
                continue

            acc = converted
            j = i + 1
            while j < len(ops):
                next_converted = self._try_convert_rotate_to_affine(ops[j], op_index=j)
                if next_converted is None:
                    break
                acc = self._compose_affine_ops(acc, next_converted)
                j += 1
            if j == i + 1:
                # Run of one: nothing to fuse with. Keep the original op —
                # a lone runtime rotate computes its matrix from the actual
                # buffer dimensions, which beats baking in plan-time hints.
                result.append(ops[i])
            else:
                result.append(acc)
            i = j

        return result

    def _try_convert_rotate_to_affine(self, op: OpSpec, op_index: int) -> OpSpec | None:
        """Convert an op to a ``warp_affine`` ``OpSpec`` if it is fusible.

        Returns the op unchanged if it is already ``warp_affine``, converts
        ``rotate`` with a static arbitrary angle to ``warp_affine``, or
        returns ``None`` if the op is not affine-compatible.

        The conversion bakes the rotation center and output size into the
        matrix, so it uses the H/W hints ENTERING the op at ``op_index``
        (recorded when the op was appended) — never the pipeline's final
        hints, which reflect the shape after ALL ops.
        """
        if op.op == "warp_affine":
            # A per-row (expression) matrix can't be composed at plan time, so it
            # is not fusable — leave it to resolve per row at execution.
            if _literal_matrix_values(op.params["matrix"]) is None:
                return None
            return op

        if op.op != "rotate":
            return None

        angle_pv = op.params.get("angle")
        if angle_pv is None or angle_pv.is_expr:
            return None

        angle = float(angle_pv.value)
        norm = angle % 360
        if norm < 0:
            norm += 360

        eps = 0.001
        if (
            abs(norm - 90) < eps
            or abs(norm - 180) < eps
            or abs(norm - 270) < eps
            or abs(norm) < eps
            or abs(norm - 360) < eps
        ):
            return None

        expand_pv = op.params.get("expand")
        expand = bool(expand_pv and not expand_pv.is_expr and expand_pv.value)

        h_pv, w_pv = self._hint_snapshots.get(op_index, (None, None))
        if h_pv is None or w_pv is None or h_pv.is_expr or w_pv.is_expr:
            return None

        ih, iw = int(h_pv.value), int(w_pv.value)
        # Read the matrix from the engine rather than recomputing it. This is
        # the same `AffineParams::from_rotation` an *unfused* rotate executes
        # through, so a rotate produces identical geometry whether or not a
        # neighbouring op happened to make it fusible. Python used to
        # transliterate that function line for line, and the two had already
        # drifted: the angle was normalised here (`angle % 360`) and not there,
        # and the expand bounding box was rounded half-to-even here against
        # half-away-from-zero there.
        #
        # `angle`, not `norm`: the engine takes the raw angle, and passing it
        # what it would have received unfused is the whole point.
        from polars_cv._lib import rotate_affine_params

        matrix, new_h, new_w = rotate_affine_params(angle, ih, iw, expand)

        interpolation = op.params.get("interpolation")
        border_value = op.params.get("border_value")

        return OpSpec(
            op="warp_affine",
            params={
                "matrix": _matrix_param_from_floats(matrix),
                "output_height": ParamValue(is_expr=False, value=new_h),
                "output_width": ParamValue(is_expr=False, value=new_w),
                "interpolation": interpolation
                or ParamValue(is_expr=False, value="bilinear"),
                "border_value": border_value or ParamValue(is_expr=False, value=0.0),
            },
        )

    @staticmethod
    def _compose_affine_ops(first: OpSpec, second: OpSpec) -> OpSpec:
        """Compose two ``warp_affine`` ``OpSpec`` by matrix multiplication.

        The composed matrix is ``second.matrix @ first.matrix`` (in
        homogeneous 3x3 form).  Output dimensions, interpolation, and
        border value are taken from *second*.

        Args:
            first: The earlier warp_affine op.
            second: The later warp_affine op.

        Returns:
            A single fused ``OpSpec``.
        """
        # Both matrices are literal here: `_try_convert_rotate_to_affine` only
        # admits an op for fusion when its matrix has no per-row expression.
        m1 = _literal_matrix_values(first.params["matrix"])
        m2 = _literal_matrix_values(second.params["matrix"])
        assert m1 is not None and m2 is not None, "fusion requires literal matrices"
        a1, b1, tx1, c1, d1, ty1 = m1
        a2, b2, tx2, c2, d2, ty2 = m2

        fused_matrix = [
            a2 * a1 + b2 * c1,
            a2 * b1 + b2 * d1,
            a2 * tx1 + b2 * ty1 + tx2,
            c2 * a1 + d2 * c1,
            c2 * b1 + d2 * d1,
            c2 * tx1 + d2 * ty1 + ty2,
        ]
        return OpSpec(
            op="warp_affine",
            params={
                "matrix": _matrix_param_from_floats(fused_matrix),
                "output_height": second.params["output_height"],
                "output_width": second.params["output_width"],
                "interpolation": second.params["interpolation"],
                "border_value": second.params["border_value"],
            },
        )

    def _to_spec_dict(self) -> dict:
        """
        Convert pipeline to specification dictionary (without sink).

        Used for graph serialization where sink is handled separately.
        Applies affine fusion optimization before serialization.

        The node-level ``domain``/``output_dtype`` are Python-side
        visualization metadata (consumed by ``_graph_viz.parse_logical_graph``
        for intermediate nodes, which the terminal-only ``OutputSpec`` cannot
        supply). Rust's ``GraphNode`` declares but ignores them, computing its
        own schema from the ops; both are derived from the same ``op_schema``
        authority, so they cannot drift.

        Shape hints are deliberately *not* emitted: no Rust code ever read the
        key, and because ``graph_json`` is the compiled-graph cache key, two
        pipelines that execute identically but carry different hints occupied
        separate cache entries. Plan-time shape still crosses the boundary as
        ``expected_shape`` on the output spec, which Rust does read.

        Returns:
            Dictionary with source, ops, domain, and output_dtype.
        """
        optimized_ops = self._fuse_affine_ops(self._ops)
        spec: dict = {
            "source": self._source.to_dict() if self._source else None,
            "ops": [op.to_dict() for op in optimized_ops],
            "domain": self._current_domain,
            "output_dtype": self._output_dtype,
        }

        return spec

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

        spec: dict = {
            "source": self._source.to_dict() if self._source else None,
            "ops": [op.to_dict() for op in self._ops],
        }
        return json.dumps(spec)

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
