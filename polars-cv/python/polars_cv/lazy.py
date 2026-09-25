"""
Lazy pipeline expressions for composable vision operations.

This module provides the LazyPipelineExpr class which enables composable,
lazy pipeline operations that are fused into a single plugin call when
.sink() is called.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from polars_cv._graph import PipelineGraph
    from polars_cv._optimize import OptFlags
    from polars_cv.pipeline import Pipeline


def _generate_node_id() -> str:
    """Generate a unique node ID for the pipeline graph."""
    return f"node_{uuid.uuid4().hex[:8]}"


#: Statistic name -> the ``Pipeline`` reduction that computes it.
#:
#: One authority for both halves of the job: the accepted names are this
#: mapping's keys, and the dispatch reads its values. ``statistics()`` and
#: ``statistics_lazy()`` used to carry a ``valid_stats`` set *and* a five-arm
#: ``if/elif`` chain each — four copies of one list — and both chains ended in
#: ``else: continue``, so a name the validator accepted but the chain did not
#: know would have been silently dropped from the output rather than raising.
#:
#: ``test_stat_reducers_are_all_pipeline_methods`` rejects a value that is not
#: a real reduction, which is what stops this becoming a list of names nothing
#: resolves.
_STAT_REDUCERS: "dict[str, str]" = {
    "mean": "reduce_mean",
    "std": "reduce_std",
    "min": "reduce_min",
    "max": "reduce_max",
    "sum": "reduce_sum",
}

#: What ``include=None`` means. ``sum`` is deliberately outside it: it scales
#: with element count rather than describing a distribution, so it is opt-in.
_DEFAULT_STATS: "tuple[str, ...]" = ("mean", "std", "min", "max")


def _check_sink(
    fmt: str, kwargs: "dict[str, Any]", pipeline: Any, alias: str | None = None
) -> None:
    """Check one output's sink against its Rust definition and the plan.

    ``plan_sink`` validates the format and its keywords (a keyword the format
    does not read, a misspelled one, a ``dtype`` other than half precision) and
    then refuses a sink the output's planned state cannot give a Polars
    schema: a typed ``list``/``array`` element with no known dtype, an
    ``array`` with no shape, a ``list`` with no rank — unless the source
    resolves them from the input column. All of it raises here, while the
    pipeline is built, rather than at ``collect()``.
    """
    from polars_cv._lib import plan_sink
    from polars_cv._types import planning_slots

    source = pipeline._source
    plan_sink(
        json.dumps({"format": fmt, **kwargs}, default=list),
        pipeline._state,
        None if source is None else json.dumps(source.to_dict(planning_slots)),
        alias,
    )


class LazyPipelineExpr:
    """
    Lazy pipeline expression for composed operations.

    This class represents a deferred computation that can be composed with
    other expressions. The entire graph is fused and executed when `.sink()`
    is called.

    Example:
        >>> preprocess = Pipeline().source("image_bytes").resize(height=100, width=200)
        >>> img = pl.col("image").cv.pipe(preprocess)
        >>> expr = img.sink("numpy")
    """

    # The private instance state, declared once here so it is a single authority:
    # `gen_lazy_stub.py` reads these annotations to emit the `.pyi` (other modules
    # read a node's `_node_id` / `_pipeline` / ... across the package boundary).
    _column: pl.Expr
    _pipeline: Pipeline
    _node_id: str
    _upstream: list[LazyPipelineExpr]
    _alias: str | None

    def __init__(
        self,
        column: pl.Expr,
        pipeline: "Pipeline",
        node_id: str | None = None,
        upstream: list["LazyPipelineExpr"] | None = None,
        alias: str | None = None,
    ) -> None:
        """
        Initialize a LazyPipelineExpr.

        Args:
            column: The Polars column expression this pipeline reads from.
            pipeline: The Pipeline instance defining operations.
            node_id: Unique identifier for this node in the graph.
            upstream: List of upstream LazyPipelineExpr dependencies.
            alias: Optional user-defined name for this node (for multi-output).
        """
        self._column = column
        self._pipeline = pipeline
        self._node_id = node_id or _generate_node_id()
        self._upstream = upstream or []
        self._alias = alias

    @property
    def node_id(self) -> str:
        """Get the unique node ID for this expression."""
        return self._node_id

    @property
    def column(self) -> pl.Expr:
        """Get the underlying column expression."""
        return self._column

    @property
    def pipeline(self) -> "Pipeline":
        """Get the pipeline specification."""
        return self._pipeline

    @property
    def alias_name(self) -> str | None:
        """Get the user-defined alias for this node, if any."""
        return self._alias

    # --- Alias (named checkpoint) ---

    def alias(self, name: str) -> "LazyPipelineExpr":
        """
        Name this checkpoint for multi-output extraction.

        Example:
            >>> base = pl.col("img").cv.pipe(pipe).alias("base")
            >>> gray = base.pipe(Pipeline().grayscale()).alias("gray")
            >>> expr = gray.sink({"base": "numpy", "gray": "png"})
        """
        # Create a new LazyPipelineExpr with the alias set
        # This effectively creates a "checkpoint" that can be referenced
        return LazyPipelineExpr(
            column=self._column,
            pipeline=self._pipeline,
            node_id=self._node_id,  # Keep the same node_id
            upstream=self._upstream,
            alias=name,
        )

    # --- Pipeline Chaining ---

    def pipe(self, pipeline: "Pipeline") -> "LazyPipelineExpr":
        """
        Chain a Pipeline onto this expression.

        Args:
            pipeline: Operations to apply. If no source(), it continues from here.
        """
        if pipeline._source is None:
            # Continuation: new node receives input from self, only has NEW ops
            import copy as _copy

            from polars_cv._types import SourceFormat, SourceSpec
            from polars_cv.pipeline import Pipeline as PipelineClass

            new_pipeline = PipelineClass()
            # BLOB source means "receive from upstream node"
            new_pipeline._source = SourceSpec(format=SourceFormat.BLOB)
            new_pipeline._expr_refs = pipeline._expr_refs.copy()
            # Carry the graph-level policies through continuations.
            new_pipeline._on_error = pipeline._on_error
            new_pipeline._on_null_param = pipeline._on_null_param

            # A continuation's pre-op state IS the upstream node's output
            # state, hints included. Seed from it, then re-apply each new op
            # through the same mandatory append path the eager builders use,
            # so domain/dtype/ndim and the shape hints advance together, one
            # op at a time.
            #
            # Folding per-op is the point: the previous code replayed only the
            # hints and assigned the rank afterwards, so every replayed
            # op saw `ndim = None` and the H/W update was skipped at its
            # opening guard — the H/W half of the replay never ran. It
            # cannot be fixed by hoisting that assignment, either: a
            # rank-changing op must infer against its own input rank, not the
            # chain's final one.
            #
            # Its sizes are the upstream's, but none is this node's user's
            # assertion; whether a declaration reached the lineage carries
            # over (see `PlanState.declared`).
            new_pipeline._state = dataclasses.replace(
                self._pipeline._state, asserted=(False, False, False)
            )
            new_pipeline._assertions = _copy.deepcopy(pipeline._assertions)
            # An assert_shape() written before the first op has no preceding
            # append to apply it; every later position is applied by the
            # `_push_op` that lands on it, so the replay is just the append
            # path run once per op.
            new_pipeline._apply_assertions_at(0)
            for op_spec in pipeline._ops:
                new_pipeline._push_op(op_spec)

            # Ops referencing other nodes (rasterize(shape=...)) make those
            # nodes upstream dependencies so they execute first.
            new_pipeline._shape_refs = pipeline._shape_refs.copy()
            return LazyPipelineExpr(
                column=None,  # No column - receives from upstream, not from DataFrame
                pipeline=new_pipeline,
                node_id=_generate_node_id(),
                upstream=[self, *new_pipeline._shape_refs],
            )
        else:
            # Has source: create new root node
            # This is like calling pl.col(...).cv.pipe(...) again
            return LazyPipelineExpr(
                column=self._column,
                pipeline=pipeline,
                node_id=_generate_node_id(),
                upstream=list(pipeline._shape_refs),
            )

    # --- Sink (materializes to pl.Expr) ---

    def sink(
        self,
        format: str | dict[str, str] = "native",
        return_expr: bool = True,
        opt_flags: "OptFlags | bool | None" = None,
        **kwargs: Any,
    ) -> "pl.Expr | PipelineGraph":
        """
        Finalize the pipeline graph and return a Polars expression.

        Args:
            format: Output format string (e.g., "numpy", "png") or a dict
                    mapping aliases to formats for multi-output. ``"ndarray"``
                    is the ``"numpy"`` struct tagged with the
                    ``polars_cv.ndarray`` extension type
                    (:class:`polars_cv.NdArrayType`).
            return_expr: If True (default), return a pl.Expr. If False, return the PipelineGraph.
            opt_flags: Which plan-time optimizations to apply. ``None`` (default)
                    reads the ``POLARS_CV_OPTIMIZATIONS`` env var, falling back
                    to all-on; ``True``/``False`` are all/none shorthands; an
                    :class:`polars_cv.OptFlags` selects passes individually.
                    Every pass is output-preserving, so this only changes the
                    physical graph — see :mod:`polars_cv._optimize`.
            kwargs: Parameters for the sink. ``quality`` for the jpeg sink
                    (the other encoders take none); ``shape`` for the array
                    sink; ``dtype="f16"`` for the numpy/torch/ndarray sink to downcast
                    the output tensor to half precision at the encode boundary
                    (halving the tensor bytes and H2D transfer). ``dtype`` only
                    accepts half precision, as ``"f16"`` or ``"float16"`` —
                    cast inside the pipeline for other
                    dtypes. A keyword that does not apply to the chosen format
                    is rejected rather than ignored, as is one that is not a
                    sink parameter at all.

        Returns:
            A Polars expression (or PipelineGraph if return_expr=False).
        """
        from polars_cv._graph import PipelineGraph

        # Validate no cycles
        self._validate_no_cycles()

        # Collect all nodes in topological order
        all_nodes = self._collect_dependency_graph()

        # Build the fused pipeline graph with alias information
        graph = PipelineGraph()
        for node in all_nodes:
            graph.add_node(
                node_id=node._node_id,
                pipeline=node._pipeline,
                column=node._column,
                upstream=[u._node_id for u in node._upstream],
                alias=node._alias,
            )

        if isinstance(format, dict):
            # Multi-output: the shared kwargs apply to every alias's sink.
            for alias, fmt_str in format.items():
                node = self._find_node_by_alias(alias, all_nodes)
                _check_sink(
                    fmt_str,
                    kwargs,
                    (node or self)._pipeline,
                    alias,
                )
            graph.set_multi_output(format, **kwargs)
        else:
            _check_sink(format, kwargs, self._pipeline)
            graph.set_output(self._node_id, format, **kwargs)

        # The explicit optimization phase: rewrite the logical graph into its
        # physical form before serialization. Both return paths see the same
        # optimized graph.
        from polars_cv._optimize import resolve_opt_flags

        graph.optimize(resolve_opt_flags(opt_flags))

        if return_expr:
            # Register and return the fused expression
            return graph.to_expr()

        # Return the graph
        return graph

    # --- Composition Methods ---

    def apply_mask(
        self,
        mask: "LazyPipelineExpr",
        *,
        invert: bool | pl.Expr = False,
    ) -> "LazyPipelineExpr":
        """
        Apply a binary mask to this image.

        The mask can be from another image pipeline or a contour pipeline
        (which will be auto-rasterized to match dimensions).

        Args:
            mask: LazyPipelineExpr producing the mask.
            invert: If True, invert the mask (keep exterior, zero interior).

        Returns:
            New LazyPipelineExpr with the mask operation composed.
        """

        # Create a new pipeline that references the mask
        new_pipeline = self._pipeline._clone()
        new_pipeline._add_node_op("apply_mask", {"mask": mask, "invert": invert})

        return LazyPipelineExpr(
            column=self._column,
            pipeline=new_pipeline,
            node_id=_generate_node_id(),
            upstream=[self, mask],
        )

    def channel_merge(self, *others: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Merge single-channel buffers into one multi-channel image.

        Stacks this expression's ``[H, W]`` buffer with each ``others`` buffer
        along a new channel axis, producing ``[H, W, C]`` where
        ``C = len(others) + 1``. Each operand must be a single-channel ``[H, W]``
        buffer (e.g. produced by :meth:`Pipeline.channel_select`) with matching
        height and width. This is the inverse of ``channel_select``.

        Args:
            others: One or more single-channel LazyPipelineExpr operands, in
                channel order after this one.

        Returns:
            New LazyPipelineExpr producing the merged multi-channel image.

        Example:
            ```python
            >>> sel = lambda i: pl.col("img").cv.pipe(
            ...     Pipeline().source("image_bytes").channel_select(index=i)
            ... )
            >>> merged = sel(2).channel_merge(sel(1), sel(0))  # RGB -> BGR
            ```
        """
        # The op's Rust definition rejects an empty `others`.
        new_pipeline = self._pipeline._clone()
        new_pipeline._add_node_op("channel_merge", {"others": list(others)})

        return LazyPipelineExpr(
            column=self._column,
            pipeline=new_pipeline,
            node_id=_generate_node_id(),
            upstream=[self, *others],
        )

    # NOTE: `label_reduce` is intentionally NOT hand-defined here. Its
    # `Pipeline` counterpart takes a plain `pl.Expr` (not a `LazyPipelineExpr`
    # operand), so it does not qualify for the bespoke-lazy exception and is
    # generated by `_install_pipeline_forwarders` like every other ordinary op.
    # A hand-written copy drifted its docstring in the past — the generation
    # scheme exists precisely to prevent that.

    def add(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Element-wise addition with another array.

        For u8/u16: Saturating addition (clamps to max value, e.g., 255 for u8).
        For f32/f64: Standard addition.

        Args:
            other: LazyPipelineExpr to add.

        Returns:
            New LazyPipelineExpr with the add operation composed.

        Example:
            ```python
            >>> img1 = pl.col("image1").cv.pipe(pipe1)
            >>> img2 = pl.col("image2").cv.pipe(pipe2)
            >>> result = img1.add(img2).sink("numpy")  # 200 + 100 = 255 (saturated)
            ```
        """
        return self._binary_op("add", other)

    def subtract(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Element-wise subtraction.

        For u8/u16: Saturating subtraction (clamps to 0).
        For f32/f64: Standard subtraction.

        Args:
            other: LazyPipelineExpr to subtract.

        Returns:
            New LazyPipelineExpr with the subtract operation composed.

        Example:
            ```python
            >>> img1 = pl.col("image1").cv.pipe(pipe1)
            >>> img2 = pl.col("image2").cv.pipe(pipe2)
            >>> result = img1.subtract(img2).sink("numpy")  # 50 - 100 = 0 (saturated)
            ```
        """
        return self._binary_op("subtract", other)

    def multiply(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Element-wise multiplication.

        For u8/u16: Saturating multiplication (clamps to max value).
        For f32/f64: Standard multiplication.

        For normalized image blending (treating values as [0,1] range),
        use blend() instead.

        Args:
            other: LazyPipelineExpr to multiply by.

        Returns:
            New LazyPipelineExpr with the multiply operation composed.

        Example:
            ```python
            >>> img1 = pl.col("image1").cv.pipe(pipe1)
            >>> img2 = pl.col("image2").cv.pipe(pipe2)
            >>> result = img1.multiply(img2).sink("numpy")  # 16 * 16 = 255 (saturated)
            ```

        See Also:
            blend: For normalized multiplication ((a/255) * (b/255) * 255)
        """
        return self._binary_op("multiply", other)

    def divide(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Element-wise division.

        For u8/u16: Integer division with zero protection (returns 0 for divide by 0).
        For f32/f64: Standard division.

        Args:
            other: LazyPipelineExpr to divide by.

        Returns:
            New LazyPipelineExpr with the divide operation composed.
        """
        return self._binary_op("divide", other)

    def blend(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Normalized blend (element-wise).

        Performs normalized multiplication useful for image blending/compositing.

        For u8: (a/255) * (b/255) * 255
        For u16: (a/65535) * (b/65535) * 65535
        For f32/f64: Standard multiplication.

        Args:
            other: LazyPipelineExpr to blend with.

        Returns:
            New LazyPipelineExpr with the blend operation composed.

        Example:
            Blend two images together with proper normalization:

            >>> img1 = pl.col("image1").cv.pipe(pipe1)
            >>> img2 = pl.col("image2").cv.pipe(pipe2)
            >>> blended = img1.blend(img2).sink("numpy")
        """
        return self._binary_op("blend", other)

    def ratio(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Scaled ratio division.

        Computes a/b scaled to the full range of the data type.

        For u8: (a/b) * 255, clamped to [0, 255]
        For u16: (a/b) * 65535, clamped to [0, 65535]
        For f32/f64: Standard division.

        Args:
            other: LazyPipelineExpr to divide by.

        Returns:
            New LazyPipelineExpr with the ratio operation composed.

        Example:
            Compute normalized ratio between two images:

            >>> img1 = pl.col("image1").cv.pipe(pipe1)
            >>> img2 = pl.col("image2").cv.pipe(pipe2)
            >>> result = img1.ratio(img2).sink("numpy")
        """
        return self._binary_op("ratio", other)

    def bitwise_and(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Element-wise bitwise AND.

        For binary masks (0/255 values), this computes the intersection.

        Args:
            other: LazyPipelineExpr to AND with.

        Returns:
            New LazyPipelineExpr with the bitwise AND operation composed.

        Example:
            Compute intersection of two binary masks:

            >>> mask1 = pl.col("pred_mask").cv.pipe(mask_pipe)
            >>> mask2 = pl.col("gt_mask").cv.pipe(mask_pipe)
            >>> intersection = mask1.bitwise_and(mask2).sink("list")
        """
        return self._binary_op("bitwise_and", other)

    def bitwise_or(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Element-wise bitwise OR.

        For binary masks (0/255 values), this computes the union.

        Args:
            other: LazyPipelineExpr to OR with.

        Returns:
            New LazyPipelineExpr with the bitwise OR operation composed.

        Example:
            Compute union of two binary masks:

            >>> mask1 = pl.col("pred_mask").cv.pipe(mask_pipe)
            >>> mask2 = pl.col("gt_mask").cv.pipe(mask_pipe)
            >>> union = mask1.bitwise_or(mask2).sink("list")
        """
        return self._binary_op("bitwise_or", other)

    def bitwise_xor(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Element-wise bitwise XOR.

        For binary masks (0/255 values), this computes the symmetric difference.

        Args:
            other: LazyPipelineExpr to XOR with.

        Returns:
            New LazyPipelineExpr with the bitwise XOR operation composed.

        Example:
            Compute symmetric difference of two binary masks:

            >>> mask1 = pl.col("pred_mask").cv.pipe(mask_pipe)
            >>> mask2 = pl.col("gt_mask").cv.pipe(mask_pipe)
            >>> diff = mask1.bitwise_xor(mask2).sink("list")
        """
        return self._binary_op("bitwise_xor", other)

    def maximum(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Element-wise maximum of two arrays.

        Returns the maximum value at each position between this and another array.
        Useful for operations like image compositing, clamping, and non-linear
        image processing.

        Args:
            other: LazyPipelineExpr to compare with.

        Returns:
            New LazyPipelineExpr with the maximum operation composed.

        Example:
            Compute element-wise maximum of two images:

            >>> img1 = pl.col("image1").cv.pipe(pipe1)
            >>> img2 = pl.col("image2").cv.pipe(pipe2)
            >>> result = img1.maximum(img2).sink("numpy")
        """
        return self._binary_op("maximum", other)

    def minimum(self, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Element-wise minimum of two arrays.

        Returns the minimum value at each position between this and another array.
        Useful for operations like image compositing, clamping, and non-linear
        image processing.

        Args:
            other: LazyPipelineExpr to compare with.

        Returns:
            New LazyPipelineExpr with the minimum operation composed.

        Example:
            Compute element-wise minimum of two images:

            >>> img1 = pl.col("image1").cv.pipe(pipe1)
            >>> img2 = pl.col("image2").cv.pipe(pipe2)
            >>> result = img1.minimum(img2).sink("numpy")
        """
        return self._binary_op("minimum", other)

    def apply_contour_mask(
        self,
        contour: "LazyPipelineExpr",
        *,
        invert: bool | pl.Expr = False,
    ) -> "LazyPipelineExpr":
        """
        Apply a contour as a mask to this image.

        The contour will be auto-rasterized to match the current image dimensions.
        This is a convenience for:
            mask_pipe = Pipeline().source("contour", shape=img_expr)
            mask = pl.col("contour").cv.pipe(mask_pipe)
            img.apply_mask(mask)

        Args:
            contour: LazyPipelineExpr from a contour source (dimensions will be
                inferred from this image's output shape).
            invert: If True, mask exterior instead of interior.

        Returns:
            New LazyPipelineExpr with the contour mask applied.
        """
        from polars_cv.pipeline import Pipeline

        # Carry the original contour source's fill/background across to the new
        # shape-referencing source. Both are ``ParamValue | None`` (they accept
        # per-row expressions), so unwrap back to the value the caller passed —
        # re-wrapping a ``ParamValue`` would nest it and fail JSON encoding.
        orig_source = contour._pipeline._source

        def _unwrap(name: str, default: int) -> int | pl.Expr:
            param = orig_source.params.get(name) if orig_source else None
            return default if param is None else param.value

        fill_value = _unwrap("fill_value", 255)
        background = _unwrap("background", 0)

        # Create new contour source with shape= referencing this image for dimensions
        raster_pipeline = Pipeline().source(
            "contour",
            shape=self,  # Infer dimensions from this image's output
            fill_value=fill_value,
            background=background,
        )

        rasterized = LazyPipelineExpr(
            column=contour._column,
            pipeline=raster_pipeline,
            node_id=_generate_node_id(),
            upstream=[self],  # Depends on image for dimensions (shape inference)
        )

        return self.apply_mask(rasterized, invert=invert)

    def merge_pipe(self, *others: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """
        Merge multiple pipeline branches into a single terminal node.

        This creates a node that depends on self and all others, making them
        all reachable for multi-output sinking. The merged node outputs the
        same as self (the first/primary branch).

        Use this when you have branching pipelines that share a backbone and
        want to sink multiple branches in a single expression.

        Args:
            others: Other LazyPipelineExpr nodes to include in the graph.

        Returns:
            New LazyPipelineExpr with all branches as upstream dependencies.

        Example:
            ```python
            >>> gray = pl.col("image").cv.pipe(gray_pipe).alias("gray")
            >>> contours = gray.extract_contours().alias("contours")
            >>> blurred = gray.blur(5).alias("blurred")
            >>>
            >>> # Merge branches for multi-output
            >>> result = contours.merge_pipe(blurred)
            >>> expr = result.sink({
            ...     "gray": "png",
            ...     "contours": "native",
            ...     "blurred": "numpy"
            ... })
            ```
        Note:
            - Safe to merge nodes that are already upstream (deduplication handled)
            - Can merge pipelines from different source columns (multi-source graph)
            - The merged node's output is the same as self (first argument)
        """

        # Clone the pipeline - the merge node acts as a passthrough
        new_pipeline = self._pipeline._clone()

        return LazyPipelineExpr(
            column=self._column,
            pipeline=new_pipeline,
            node_id=_generate_node_id(),
            upstream=[self, *others],
        )

    def statistics(
        self,
        include: list[str] | None = None,
    ) -> pl.Expr:
        """
        Compute multiple statistics from the buffer in a single pass.

        Returns a Struct column with named fields for each statistic.
        By default includes: mean, std, min, max.

        This is a convenience method that creates branching reduction pipelines
        and merges them into a single multi-output expression.

        Args:
            include: List of statistics to compute. Valid options:
                - "mean": Arithmetic mean
                - "std": Standard deviation
                - "min": Minimum value
                - "max": Maximum value
                - "sum": Sum of all values
                If None, defaults to ["mean", "std", "min", "max"].

        Returns:
            A Polars expression that returns a Struct column with the
            requested statistics as fields.

        Example:
            ```python
            >>> # Get statistics for processed images
            >>> pipe = Pipeline().source("image_bytes").grayscale()
            >>> img = pl.col("image").cv.pipe(pipe)
            >>> stats_expr = img.statistics()
            >>> df.with_columns(stats=stats_expr)
            >>> # Access: df["stats"].struct.field("mean")
            >>>
            >>> # Only compute specific stats
            >>> stats_expr = img.statistics(include=["min", "max"])
            ```
        """
        names, stat_nodes = self._stat_nodes(include)
        merged = self._merge_stat_nodes(stat_nodes)
        return merged.sink({name: "native" for name in names})

    def statistics_lazy(
        self,
        include: list[str] | None = None,
    ) -> "LazyPipelineExpr":
        """
        Create a lazy pipeline for computing multiple statistics.

        Unlike `statistics()` which returns a finalized pl.Expr, this method
        returns a LazyPipelineExpr that can be merged with other pipelines
        for multi-output composition.

        When sunk, each statistic becomes a separate field in the output struct.
        The stat nodes are aliased with prefixed names: "stat_mean", "stat_std", etc.

        Args:
            include: List of statistics to compute. Valid options:
                - "mean": Arithmetic mean
                - "std": Standard deviation
                - "min": Minimum value
                - "max": Maximum value
                - "sum": Sum of all values
                If None, defaults to ["mean", "std", "min", "max"].

        Returns:
            A LazyPipelineExpr representing the merged statistics pipelines.
            Can be composed with other pipelines using merge_pipe().

        Example:
            ```python
            >>> # Compose stats with other outputs
            >>> pipe = Pipeline().source("image_bytes")
            >>> img = pl.col("image").cv.pipe(pipe).alias("img")
            >>> gray = img.pipe(Pipeline().grayscale()).alias("gray")
            >>> stats = gray.statistics_lazy()  # Creates stat_mean, stat_std, etc.
            >>>
            >>> # Merge and sink together
            >>> result = img.merge_pipe(gray, stats)
            >>> expr = result.sink({
            ...     "img": "numpy",
            ...     "gray": "numpy",
            ...     "stat_mean": "native",
            ...     "stat_std": "native",
            ...     "stat_min": "native",
            ...     "stat_max": "native",
            ... })
            ```
        """
        _, stat_nodes = self._stat_nodes(include, prefix="stat_")
        return self._merge_stat_nodes(stat_nodes)

    # --- Internal Helpers ---

    def _stat_nodes(
        self, include: "list[str] | None", prefix: str = ""
    ) -> "tuple[list[str], list[LazyPipelineExpr]]":
        """Build one aliased reduction node per requested statistic.

        Shared by :meth:`statistics` and :meth:`statistics_lazy`, which differ
        only in the alias prefix and in whether they sink the result. They used
        to carry a copy each of the accepted-name set and of the five-arm
        dispatch, both ending ``else: continue`` — so a name that passed
        validation but missed the chain vanished from the output silently.
        Reading :data:`_STAT_REDUCERS` for both halves makes that unrepresentable.

        Args:
            include: Statistic names, or ``None`` for :data:`_DEFAULT_STATS`.
            prefix: Prepended to each output alias.

        Returns:
            The output aliases and their nodes, in ``include`` order.

        Raises:
            ValueError: If a name is not a known statistic, or none were asked
                for.
        """
        from polars_cv.pipeline import Pipeline

        # Materialised before anything reads it. The validation loop below and
        # the node comprehension further down each iterate `include`, so a
        # one-shot iterable (a generator, `iter([...])`) was exhausted by the
        # first pass and produced zero nodes — surfacing as `IndexError` from
        # `_merge_stat_nodes` rather than the "at least one statistic" message.
        include = list(_DEFAULT_STATS) if include is None else list(include)

        for stat in include:
            if stat not in _STAT_REDUCERS:
                msg = (
                    f"Unknown statistic '{stat}'. "
                    f"Valid options: {sorted(_STAT_REDUCERS)}"
                )
                raise ValueError(msg)

        if not include:
            raise ValueError("At least one statistic must be included")

        names = [f"{prefix}{stat}" for stat in include]
        nodes = [
            self.pipe(getattr(Pipeline(), _STAT_REDUCERS[stat])()).alias(name)
            for stat, name in zip(include, names)
        ]
        return names, nodes

    @staticmethod
    def _merge_stat_nodes(nodes: "list[LazyPipelineExpr]") -> "LazyPipelineExpr":
        """Fold the stat nodes into one multi-output expression."""
        return nodes[0] if len(nodes) == 1 else nodes[0].merge_pipe(*nodes[1:])

    def _continuation(self) -> "Pipeline":
        """A sourceless Pipeline seeded with this expression's planner state.

        Domain-sensitive builders (contour measures, reductions) validate
        their input domain at construction time; a bare ``Pipeline()`` starts
        in the buffer domain and would reject e.g. ``.area()`` after
        ``.extract_contours()``. ``pipe()`` recomputes the continuation
        node's state from upstream regardless — this seed only exists so the
        builder-time validation sees the truth.
        """
        from polars_cv.pipeline import Pipeline, PlanState

        inner = Pipeline()
        upstream = self._pipeline._state
        inner._state = PlanState(
            domain=upstream.domain, dtype=upstream.dtype, ndim=upstream.ndim
        )
        return inner

    def _binary_op(self, op: str, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """Create a binary operation between this and another LazyPipelineExpr."""
        from polars_cv._types import SourceFormat, SourceSpec
        from polars_cv.pipeline import Pipeline as PipelineClass
        from polars_cv.pipeline import PlanState

        # Create a new pipeline that receives from upstream (BLOB source)
        # and only applies the binary op - don't clone self's ops as they're
        # already applied by the upstream node
        new_pipeline = PipelineClass()
        new_pipeline._source = SourceSpec(format=SourceFormat.BLOB)
        # The op applies to the left operand's output, so it starts from that
        # state; its dtype rule reads both operands (true division of two u8
        # is f32), so the other operand's dtype goes with it.
        left = self._pipeline._state
        new_pipeline._state = PlanState(
            domain=left.domain, dtype=left.dtype, ndim=left.ndim
        )
        new_pipeline._add_node_op(
            op, {"other": other}, other_dtype=other._pipeline._state.dtype
        )

        return LazyPipelineExpr(
            column=None,  # No direct column - receives from upstream
            pipeline=new_pipeline,
            node_id=_generate_node_id(),
            upstream=[self, other],
        )

    def _find_node_by_alias(
        self, alias: str, nodes: list["LazyPipelineExpr"]
    ) -> "LazyPipelineExpr | None":
        """Find a node in the graph by its alias."""
        for node in nodes:
            if node._alias == alias:
                return node
        return None

    def _collect_dependency_graph(self) -> list["LazyPipelineExpr"]:
        """
        Collect all nodes in the dependency graph in topological order.

        Returns:
            List of LazyPipelineExpr in execution order (dependencies first).
        """
        visited: set[str] = set()
        order: list[LazyPipelineExpr] = []

        def dfs(node: LazyPipelineExpr) -> None:
            if node._node_id in visited:
                return
            visited.add(node._node_id)

            for upstream in node._upstream:
                dfs(upstream)

            order.append(node)

        dfs(self)
        return order

    def _validate_no_cycles(self) -> None:
        """
        Detect circular dependencies in the pipeline graph.

        Raises:
            ValueError: If a cycle is detected.
        """
        visited: set[str] = set()
        path: set[str] = set()

        def dfs(node: LazyPipelineExpr) -> None:
            if node._node_id in path:
                raise ValueError(
                    f"Circular dependency detected: node '{node._node_id}' "
                    f"depends on itself. Check your pipeline composition."
                )
            if node._node_id in visited:
                return

            path.add(node._node_id)
            for upstream in node._upstream:
                dfs(upstream)
            path.remove(node._node_id)
            visited.add(node._node_id)

        dfs(self)

    # --- Prevent accidental use as pl.Expr ---

    def __repr__(self) -> str:
        """Return string representation with guidance."""
        upstream_ids = [u._node_id for u in self._upstream]
        alias_str = f", alias={self._alias!r}" if self._alias else ""
        return (
            f"LazyPipelineExpr(node={self._node_id!r}{alias_str}, "
            f"upstream={upstream_ids}) - call .sink(format) to execute"
        )

    def __str__(self) -> str:
        """Return string representation."""
        return self.__repr__()

    def _ipython_display_(self) -> None:
        """Display in Jupyter with guidance."""
        print(self.__repr__())
        if self._alias:
            print(f"\nThis node is aliased as '{self._alias}'")
        print("\nTo use in DataFrame operations, call .sink(format) first:")
        print(f"    expr = {self._node_id}.sink('numpy')")
        print("    df.with_columns(result=expr)")
        print("\nFor multi-output, use a dict:")
        print("    expr = node.sink({'alias1': 'numpy', 'alias2': 'png'})")


# ---------------------------------------------------------------------------
# Auto-generated Pipeline forwarders
# ---------------------------------------------------------------------------
#
# Every chainable ``Pipeline`` operation is exposed on ``LazyPipelineExpr`` as a
# thin forwarder that applies the op to a continuation pipeline (so build-time
# domain validation runs against the upstream state) and pipes the result back.
# Rather than hand-mirror ~70 method signatures — which drifted badly in the
# past and needed a dedicated parity test to police — the forwarders are
# generated once from ``Pipeline`` at import time, so each operation is defined a
# single place. The committed ``lazy.pyi`` stub mirrors the generated surface for
# static type checkers and IDEs; regenerate it with
# ``python scripts/gen_lazy_stub.py`` whenever ``Pipeline`` changes.

#: ``Pipeline`` methods that are intentionally NOT chainable lazy operations:
#: ``source`` starts a chain rather than continuing one, ``thumbnail`` is a
#: source-modifier that requires an existing image source (so it only applies
#: directly after ``source``, never on a sourceless lazy continuation), and the
#: rest are builder/planner introspection helpers.
PIPELINE_ONLY_METHODS = frozenset(
    {
        "source",
        "thumbnail",
        "validate",
        "has_source",
        "to_graph",
        "current_domain",
        "output_dtype",
        "output_encoding",
        # Introspection, not a chainable op: returns a str rendering, and
        # optimization is a graph-level phase a single lazy expr cannot stand in
        # for, so it is not forwarded onto LazyPipelineExpr.
        "explain",
    }
)


def _chainable_pipeline_ops() -> list[str]:
    """Names of ``Pipeline`` methods that forward onto ``LazyPipelineExpr``."""
    from polars_cv.pipeline import Pipeline

    return sorted(
        name
        for name in dir(Pipeline)
        if not name.startswith("_")
        and name not in PIPELINE_ONLY_METHODS
        and callable(getattr(Pipeline, name))
    )


def _make_forwarder(name: str) -> Any:
    """Build a lazy forwarder for the chainable ``Pipeline`` method ``name``.

    The returned function carries ``Pipeline``'s signature (so ``inspect``,
    ``help`` and the parity test see the true parameters) but delegates the
    keyword/positional handling to the wrapped ``Pipeline`` method at runtime.
    """
    import inspect

    from polars_cv.pipeline import Pipeline

    def forwarder(
        self: "LazyPipelineExpr", *args: Any, **kwargs: Any
    ) -> "LazyPipelineExpr":
        return self.pipe(getattr(self._continuation(), name)(*args, **kwargs))

    forwarder.__name__ = name
    forwarder.__qualname__ = f"LazyPipelineExpr.{name}"
    forwarder.__doc__ = f"Lazy counterpart of :meth:`polars_cv.Pipeline.{name}`."
    # Marker so tests can tell a generated forwarder from a hand-written lazy
    # method (both land in ``vars(LazyPipelineExpr)``).
    forwarder.__polars_cv_generated__ = True
    forwarder.__signature__ = inspect.signature(getattr(Pipeline, name)).replace(
        return_annotation="LazyPipelineExpr"
    )
    return forwarder


def _install_pipeline_forwarders() -> None:
    """Attach a forwarder for every chainable ``Pipeline`` op not already defined.

    Methods explicitly defined on ``LazyPipelineExpr`` (the binary operators,
    ``apply_mask``, ``channel_merge`` and friends — all of which take a
    ``LazyPipelineExpr`` operand) take precedence and are left untouched.
    """
    for name in _chainable_pipeline_ops():
        if name in vars(LazyPipelineExpr):
            continue
        setattr(LazyPipelineExpr, name, _make_forwarder(name))


_install_pipeline_forwarders()
