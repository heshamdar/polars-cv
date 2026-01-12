"""
Lazy pipeline expressions for composable vision operations.

This module provides the LazyPipelineExpr class which enables composable,
lazy pipeline operations that are fused into a single plugin call when
.sink() is called.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from polars_cv.pipeline import Pipeline


def _generate_node_id() -> str:
    """Generate a unique node ID for the pipeline graph."""
    return f"node_{uuid.uuid4().hex[:8]}"


class LazyPipelineExpr:
    """
    Lazy pipeline expression that can be composed before execution.

    This class represents a deferred pipeline computation that is NOT a pl.Expr.
    Multiple LazyPipelineExpr instances can be composed together, and the entire
    graph is fused into a single plugin call when .sink() is invoked.

    Using this directly in DataFrame operations will raise a TypeError with
    helpful guidance to call .sink() first.

    Example (basic):
        >>> img_pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
        >>> img = pl.col("image_data").cv.pipe(img_pipe)  # Returns LazyPipelineExpr
        >>> expr = img.sink("numpy")  # Returns pl.Expr
        >>> df.with_columns(output=expr)  # Now works!

    Example (chaining with .pipe() for composition):
        >>> # Define base processing once
        >>> base = pl.col("image").cv.pipe(
        ...     Pipeline().source("image_bytes").resize(50, 50)
        ... ).alias("resized")
        >>>
        >>> # Branch into different outputs using .pipe() (no source = continuation)
        >>> gray = base.pipe(Pipeline().grayscale()).alias("gray")
        >>> thresh = gray.pipe(Pipeline().threshold(128)).alias("thresh")
        >>> blur = gray.pipe(Pipeline().blur(5)).alias("blur")
        >>>
        >>> # CSE automatically shares the common prefix
        >>> result = thresh.merge_pipe(blur).sink({
        ...     "resized": "numpy",
        ...     "gray": "numpy",
        ...     "thresh": "numpy",
        ...     "blur": "numpy",
        ... })
    """

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
        self._upstream: list[LazyPipelineExpr] = upstream or []
        self._alias: str | None = alias

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
        Mark this node with a name for multi-output.

        Aliases create named checkpoints that can be referenced when calling
        `.sink()` with multiple outputs. The alias allows you to extract
        intermediate results from the pipeline graph.

        Args:
            name: Unique name for this node.

        Returns:
            New LazyPipelineExpr with the alias set.

        Example:
            ```python
            >>> base = pl.col("image").cv.pipe(
            ...     Pipeline().source("image_bytes").resize(100, 200)
            ... ).alias("resized")
            >>>
            >>> gray = base.pipe(Pipeline().grayscale()).alias("gray")
            >>> thresh = gray.pipe(Pipeline().threshold(128)).alias("thresh")
            >>>
            >>> # Return multiple outputs as Struct column
            >>> expr = thresh.sink({
            ...     "resized": "numpy",
            ...     "gray": "png",
            ...     "thresh": "numpy"
            ... })
            ```
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
        Chain additional operations from a Pipeline onto this expression.

        If the pipeline has no source, it continues from this expression's output.
        If it has a source, it creates a new root node (for multi-source graphs).

        This enables compositional reuse - define a base expression once and
        branch from it:

        Args:
            pipeline: Operations to apply. If no source(), inherits from this node.

        Returns:
            New LazyPipelineExpr with operations chained.

        Example:
            ```python
            >>> # Define base processing
            >>> base = pl.col("image").cv.pipe(
            ...     Pipeline().source("image_bytes").resize(50, 50)
            ... ).alias("resized")
            >>>
            >>> # Branch into different outputs (no source = continuation)
            >>> gray = base.pipe(Pipeline().grayscale()).alias("gray")
            >>> thresh = gray.pipe(Pipeline().threshold(128)).alias("thresh")
            >>> blur = gray.pipe(Pipeline().blur(5)).alias("blur")
            >>>
            >>> # CSE automatically shares the common prefix
            >>> result = thresh.merge_pipe(blur).sink({
            ...     "resized": "numpy",
            ...     "gray": "numpy",
            ...     "thresh": "numpy",
            ...     "blur": "numpy",
            ... })
            ```
        """
        if pipeline._source is None:
            # Continuation: new node receives input from self, only has NEW ops
            from polars_cv._types import SourceFormat, SourceSpec
            from polars_cv.pipeline import Pipeline as PipelineClass

            new_pipeline = PipelineClass()
            # BLOB source means "receive from upstream node"
            new_pipeline._source = SourceSpec(format=SourceFormat.BLOB)
            # Only the NEW operations (not self's ops)
            new_pipeline._ops = pipeline._ops.copy()
            new_pipeline._expr_refs = pipeline._expr_refs.copy()
            # Copy both domain and dtype for proper static type inference
            new_pipeline._current_domain = pipeline._current_domain
            new_pipeline._output_dtype = pipeline._output_dtype

            return LazyPipelineExpr(
                column=None,  # No column - receives from upstream, not from DataFrame
                pipeline=new_pipeline,
                node_id=_generate_node_id(),
                upstream=[self],
            )
        else:
            # Has source: create new root node
            # This is like calling pl.col(...).cv.pipe(...) again
            return LazyPipelineExpr(
                column=self._column,
                pipeline=pipeline,
                node_id=_generate_node_id(),
            )

    # --- Sink (materializes to pl.Expr) ---

    def sink(self, format: str | dict[str, str] = "native", **kwargs: Any) -> pl.Expr:
        """
        Finalize the pipeline graph and create executable pl.Expr.

        This is the ONLY way to convert LazyPipelineExpr to pl.Expr.
        It fuses all upstream dependencies into a single plugin call.

        Can be called with either:
        - A single format string: returns Binary column with terminal node output
        - A dict mapping alias names to formats: returns Struct column with
          multiple named outputs

        Args:
            format: Output format specification. For single output use a string:
                "numpy", "torch", "png", "jpeg", "blob". For multi-output use a dict:
                {"alias1": "numpy", "alias2": "png"}.
            kwargs: Additional sink parameters (e.g., quality for jpeg).

        Returns:
            A Polars expression that can be used in DataFrame operations.
            - For single output: Binary column
            - For multi-output: Struct column with named Binary fields

        Raises:
            ValueError: If circular dependencies are detected.
            ValueError: If multi-output aliases are not found in the graph.

        Example:
            ```python
            >>> # Single output
            >>> expr = result.sink("numpy")
            >>>
            >>> # Multi-output
            >>> expr = result.sink({
            ...     "original": "png",
            ...     "processed": "numpy",
            ...     "mask": "blob"
            ... })
            ```
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
            # Multi-output mode
            graph.set_multi_output(format, **kwargs)
        else:
            # Single output mode
            graph.set_output(self._node_id, format, **kwargs)

        # Register and return the fused expression
        return graph.to_expr()

    # --- Composition Methods ---

    def apply_mask(
        self,
        mask: "LazyPipelineExpr",
        *,
        invert: bool = False,
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
        new_pipeline._add_binary_op("apply_mask", mask._node_id, invert=invert)

        return LazyPipelineExpr(
            column=self._column,
            pipeline=new_pipeline,
            node_id=_generate_node_id(),
            upstream=[self, mask],
        )

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
        invert: bool = False,
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

        # Get contour source parameters (fill_value, background) from original pipeline
        orig_source = contour._pipeline._source
        fill_value = getattr(orig_source, "fill_value", 255)
        background = getattr(orig_source, "background", 0)

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

        if not isinstance(self, LazyPipelineExpr):
            raise TypeError("merge_pipe() must be called on a LazyPipelineExpr")

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
        from polars_cv.pipeline import Pipeline

        # Default statistics to compute
        if include is None:
            include = ["mean", "std", "min", "max"]

        # Validate include list
        valid_stats = {"mean", "std", "min", "max", "sum"}
        for stat in include:
            if stat not in valid_stats:
                msg = (
                    f"Unknown statistic '{stat}'. Valid options: {sorted(valid_stats)}"
                )
                raise ValueError(msg)

        # Create reduction pipelines for each requested statistic
        stat_nodes: list[LazyPipelineExpr] = []
        sink_spec: dict[str, str] = {}

        for stat in include:
            if stat == "mean":
                node = self.pipe(Pipeline().reduce_mean()).alias("mean")
            elif stat == "std":
                node = self.pipe(Pipeline().reduce_std()).alias("std")
            elif stat == "min":
                node = self.pipe(Pipeline().reduce_min()).alias("min")
            elif stat == "max":
                node = self.pipe(Pipeline().reduce_max()).alias("max")
            elif stat == "sum":
                node = self.pipe(Pipeline().reduce_sum()).alias("sum")
            else:
                continue

            stat_nodes.append(node)
            sink_spec[stat] = "native"

        if not stat_nodes:
            raise ValueError("At least one statistic must be included")

        # Merge all stat nodes and sink as multi-output struct
        if len(stat_nodes) == 1:
            merged = stat_nodes[0]
        else:
            merged = stat_nodes[0].merge_pipe(*stat_nodes[1:])

        return merged.sink(sink_spec)

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
        from polars_cv.pipeline import Pipeline

        # Default statistics to compute
        if include is None:
            include = ["mean", "std", "min", "max"]

        # Validate include list
        valid_stats = {"mean", "std", "min", "max", "sum"}
        for stat in include:
            if stat not in valid_stats:
                msg = (
                    f"Unknown statistic '{stat}'. Valid options: {sorted(valid_stats)}"
                )
                raise ValueError(msg)

        # Create reduction pipelines for each requested statistic
        stat_nodes: list[LazyPipelineExpr] = []

        for stat in include:
            alias_name = f"stat_{stat}"
            if stat == "mean":
                node = self.pipe(Pipeline().reduce_mean()).alias(alias_name)
            elif stat == "std":
                node = self.pipe(Pipeline().reduce_std()).alias(alias_name)
            elif stat == "min":
                node = self.pipe(Pipeline().reduce_min()).alias(alias_name)
            elif stat == "max":
                node = self.pipe(Pipeline().reduce_max()).alias(alias_name)
            elif stat == "sum":
                node = self.pipe(Pipeline().reduce_sum()).alias(alias_name)
            else:
                continue

            stat_nodes.append(node)

        if not stat_nodes:
            raise ValueError("At least one statistic must be included")

        # Merge all stat nodes
        if len(stat_nodes) == 1:
            return stat_nodes[0]
        else:
            return stat_nodes[0].merge_pipe(*stat_nodes[1:])

    # --- Internal Helpers ---

    def _binary_op(self, op: str, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """Create a binary operation between this and another LazyPipelineExpr."""
        from polars_cv._types import SourceFormat, SourceSpec
        from polars_cv.pipeline import Pipeline as PipelineClass

        # Create a new pipeline that receives from upstream (BLOB source)
        # and only applies the binary op - don't clone self's ops as they're
        # already applied by the upstream node
        new_pipeline = PipelineClass()
        new_pipeline._source = SourceSpec(format=SourceFormat.BLOB)
        # Copy both domain and dtype for proper static type inference
        new_pipeline._current_domain = self._pipeline._current_domain
        new_pipeline._output_dtype = self._pipeline._output_dtype
        new_pipeline._add_binary_op(op, other._node_id)

        return LazyPipelineExpr(
            column=None,  # No direct column - receives from upstream
            pipeline=new_pipeline,
            node_id=_generate_node_id(),
            upstream=[self, other],
        )

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
