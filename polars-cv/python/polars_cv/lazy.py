"""
Lazy pipeline expressions for composable vision operations.

This module provides the LazyPipelineExpr class which enables composable,
lazy pipeline operations that are fused into a single plugin call when
.sink() is called.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

from polars_cv._lazy_forwarders import _LazyForwardersMixin
from polars_cv._ops_generated import _LazyOpsMixin

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
#: The source of a node that receives its input from an upstream node.
_BLOB_SOURCE = json.dumps({"format": "blob"})

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


class LazyPipelineExpr(_LazyOpsMixin, _LazyForwardersMixin):
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

    # The private instance state, declared once here (other modules read a
    # node's `_node_id` / `_pipeline` / ... across the package boundary).
    #: The input column; ``None`` for a continuation node, which reads its
    #: upstream node instead.
    _column: pl.Expr | None
    _pipeline: Pipeline
    _node_id: str
    _upstream: list[LazyPipelineExpr]
    _alias: str | None

    def __init__(
        self,
        column: pl.Expr | None,
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
    def column(self) -> pl.Expr | None:
        """The input column expression; ``None`` for a continuation node."""
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
        if not pipeline._plan.has_source:
            # Continuation: a new node that receives this node's output (a
            # "blob" source) and applies only the new ops, each planned again
            # by Rust from this node's output state. The clone carries the
            # expressions the ops' slots name, the graph-level policies and
            # the nodes the ops read.
            new_pipeline = pipeline._clone()
            new_pipeline._plan = pipeline._plan.rebased(
                _BLOB_SOURCE, self._pipeline._state
            )
            return LazyPipelineExpr(
                column=None,  # No column - receives from upstream, not from DataFrame
                pipeline=new_pipeline,
                node_id=_generate_node_id(),
                upstream=[self, *new_pipeline._node_refs],
            )
        else:
            # Has source: create new root node
            # This is like calling pl.col(...).cv.pipe(...) again
            return LazyPipelineExpr(
                column=self._column,
                pipeline=pipeline,
                node_id=_generate_node_id(),
                upstream=list(pipeline._node_refs),
            )

    # --- Sink (materializes to pl.Expr) ---

    @overload
    def sink(
        self,
        format: str | dict[str, str] = ...,
        return_expr: Literal[True] = ...,
        opt_flags: OptFlags | bool | None = ...,
        **kwargs: Any,
    ) -> pl.Expr: ...

    @overload
    def sink(
        self,
        format: str | dict[str, str] = ...,
        *,
        return_expr: Literal[False],
        opt_flags: OptFlags | bool | None = ...,
        **kwargs: Any,
    ) -> PipelineGraph: ...

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
            graph.set_multi_output(format, **kwargs)
        else:
            graph.set_output(self._node_id, format, **kwargs)

        # The explicit optimization phase: rewrite the logical graph into its
        # physical form before serialization. Both return paths see the same
        # optimized graph.
        from polars_cv._optimize import resolve_opt_flags

        graph.optimize(resolve_opt_flags(opt_flags))
        # Refused here, where it is written, by the code the plugin runs: the
        # graph is compiled and planned, and every output's sink checked.
        graph.check()

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
    # a generated forwarder (`_lazy_forwarders.py`) like every other ordinary op.
    # A hand-written copy drifted its docstring in the past — the generation
    # scheme exists precisely to prevent that.

    def apply_contour_mask(
        self,
        contour: "LazyPipelineExpr",
        *,
        invert: bool | pl.Expr = False,
    ) -> "LazyPipelineExpr":
        """
        Apply a contour as a mask to this image.

        The contour is rasterized onto this image's canvas. This is a
        convenience for:
            mask = contour.rasterize(shape=img)
            img.apply_mask(mask)

        Args:
            contour: LazyPipelineExpr whose pipeline ends in contours (its ops
                all run), or in ``rasterize()``, whose canvas is replaced by
                this image's while its ``fill_value``/``background`` are kept.
            invert: If True, mask exterior instead of interior.

        Raises:
            ValueError: If ``contour`` ends in neither contours nor
                ``rasterize()``.

        Returns:
            New LazyPipelineExpr with the contour mask applied.
        """
        # The contour pipeline runs whole; only its canvas becomes this
        # image's. One that already rasterizes keeps its paint (a per-row
        # value as its expression) and loses only that rasterize's size.
        base = contour._pipeline
        ops = base._plan.ops_json()
        last = json.loads(ops[-1]) if ops else {}
        paint: dict[str, Any] = {}
        if last.get("op") == "rasterize":
            paint = {
                k: v for k, v in base._unwire(last).items() if k not in ("op", "size")
            }
            base = base._clone()
            base._plan = base._plan.select(list(range(len(ops) - 1)), start=0)
        elif base.current_domain() != "contour":
            msg = (
                "apply_contour_mask() takes a pipeline that ends in contours or "
                f"in rasterize(); this one ends in the {base.current_domain()!r} "
                "domain. Use apply_mask() for a mask that is already a buffer."
            )
            raise ValueError(msg)
        raster_pipeline = base.rasterize(shape=self, **paint)

        rasterized = LazyPipelineExpr(
            column=contour._column,
            pipeline=raster_pipeline,
            node_id=_generate_node_id(),
            # The contour node's own inputs, then this image for the canvas.
            upstream=[*contour._upstream, self],
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
        """A sourceless Pipeline that starts from this expression's planned state.

        Builders validate against the state they append to (domain, rank, a
        fully known shape), so a continuation starts from the upstream node's
        state as it is: ``.area()`` after ``.extract_contours()`` is accepted
        and ``.channel_select(2)`` after ``.grayscale()`` is refused.
        ``pipe()`` then plans the continuation node from upstream.
        """
        from polars_cv._lib import Plan
        from polars_cv.pipeline import Pipeline

        inner = Pipeline()
        inner._plan = Plan.continuing(self._pipeline._state)
        return inner

    def _binary_op(self, op: str, other: "LazyPipelineExpr") -> "LazyPipelineExpr":
        """Create a binary operation between this and another LazyPipelineExpr."""
        from polars_cv._lib import Plan
        from polars_cv.pipeline import Pipeline as PipelineClass

        # A node that receives the left operand's output (a "blob" source) and
        # applies only the binary op, planned from that state; its rules read
        # both operands (true division of two u8 is f32, the shapes
        # broadcast), so the other operand's state goes with it.
        new_pipeline = PipelineClass()
        new_pipeline._plan = Plan().rebased(_BLOB_SOURCE, self._pipeline._state)
        new_pipeline._add_node_op(op, {"other": other})

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
# Which Pipeline methods are chainable lazy operations
# ---------------------------------------------------------------------------
#
# Every chainable ``Pipeline`` operation is a ``LazyPipelineExpr`` method that
# applies the op to a continuation pipeline (so build-time domain validation
# runs against the upstream state) and pipes the result back. ``gen_ops.py``
# writes those methods into ``_lazy_forwarders.py`` from the built ``Pipeline``,
# reading the policy below; a method ``LazyPipelineExpr`` writes itself (it
# takes another node as an operand) is not generated.

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
