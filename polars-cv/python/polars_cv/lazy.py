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
    from polars_cv._types import FloatOrThisOrExpr
    from polars_cv.pipeline import Pipeline


def _generate_node_id() -> str:
    """Generate a unique node ID for the pipeline graph."""
    return f"node_{uuid.uuid4().hex[:8]}"


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
    #: The input column; ``None`` for a continuation node, which reads
    #: ``_input`` instead.
    _column: pl.Expr | None
    _pipeline: Pipeline
    _node_id: str
    #: The node a continuation reads; ``None`` for a root, which reads
    #: ``_column``.
    _input: LazyPipelineExpr | None
    #: ``merge_pipe``'s other branches: kept reachable (so their aliases can
    #: be sunk) though no op reads them.
    _merged: tuple[LazyPipelineExpr, ...]
    _alias: str | None

    def __init__(
        self,
        column: pl.Expr | None,
        pipeline: "Pipeline",
        node_id: str | None = None,
        input: "LazyPipelineExpr | None" = None,  # noqa: A002
        merged: "tuple[LazyPipelineExpr, ...]" = (),
        alias: str | None = None,
    ) -> None:
        """
        Initialize a LazyPipelineExpr.

        A node reads exactly one of ``column`` (a root) and ``input`` (a
        continuation). The other nodes it depends on are not passed: they are
        the nodes its pipeline's ops read (``Pipeline._node_refs``), so an
        operand cannot be declared twice or forgotten.

        Args:
            column: The Polars column expression a root reads.
            pipeline: The Pipeline instance defining operations.
            node_id: Unique identifier for this node in the graph.
            input: The node a continuation reads.
            merged: ``merge_pipe``'s other branches.
            alias: Optional user-defined name for this node (for multi-output).

        Raises:
            ValueError: If the node reads both a column and a node, or neither.
        """
        if (column is None) == (input is None):
            msg = (
                "a pipeline node reads a column or another node: exactly one of the two"
            )
            raise ValueError(msg)
        self._column = column
        self._pipeline = pipeline
        self._node_id = node_id or _generate_node_id()
        self._input = input
        self._merged = tuple(merged)
        self._alias = alias

    def _dependencies(self) -> "list[LazyPipelineExpr]":
        """The nodes this one depends on, each once: its input, the nodes its
        ops read (``Plan.operands``, Rust's ``operands()``), then
        ``merge_pipe``'s branches."""
        refs = {ref._node_id: ref for ref in self._pipeline._node_refs}
        read = [refs[node_id] for node_id in self._pipeline._plan.operands()]
        found: list[LazyPipelineExpr] = []
        for node in (
            *([self._input] if self._input is not None else []),
            *read,
            *self._merged,
        ):
            if all(node is not seen for seen in found):
                found.append(node)
        return found

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
            input=self._input,
            merged=self._merged,
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
            # Continuation: a new node that reads this node's output and
            # applies only the new ops, each planned again by Rust from this
            # node's output state. The clone carries the expressions the ops'
            # slots name, the graph-level policies and the nodes the ops read.
            new_pipeline = pipeline._clone()
            new_pipeline._plan = pipeline._plan.rebased(self._pipeline._state)
            return LazyPipelineExpr(
                column=None,
                pipeline=new_pipeline,
                node_id=_generate_node_id(),
                input=self,
            )
        else:
            # Has source: create new root node
            # This is like calling pl.col(...).cv.pipe(...) again
            if self._column is None:
                msg = (
                    "pipe() with a pipeline that has a source decodes a column, "
                    "and this expression continues another node, so it has no "
                    "column of its own. Pipe a pipeline without source() to "
                    "continue it, or start again from pl.col(...).cv.pipe(...)."
                )
                raise ValueError(msg)
            return LazyPipelineExpr(
                column=self._column,
                pipeline=pipeline,
                node_id=_generate_node_id(),
            )

    # --- Sink (materializes to pl.Expr) ---

    @overload
    def sink(
        self,
        format: str | dict[str, str | tuple[str, dict[str, Any]]] = ...,
        return_expr: Literal[True] = ...,
        opt_flags: OptFlags | bool | None = ...,
        **kwargs: Any,
    ) -> pl.Expr: ...

    @overload
    def sink(
        self,
        format: str | dict[str, str | tuple[str, dict[str, Any]]] = ...,
        *,
        return_expr: Literal[False],
        opt_flags: OptFlags | bool | None = ...,
        **kwargs: Any,
    ) -> PipelineGraph: ...

    def sink(
        self,
        format: str | dict[str, str | tuple[str, dict[str, Any]]] = "native",
        return_expr: bool = True,
        opt_flags: "OptFlags | bool | None" = None,
        **kwargs: Any,
    ) -> "pl.Expr | PipelineGraph":
        """
        Finalize the pipeline graph and return a Polars expression.

        Args:
            format: Output format string (e.g., "numpy", "png") or a dict
                    mapping aliases to formats for multi-output. A dict value
                    may be ``(format, {params})`` to give that output its own
                    sink parameters, e.g. ``{"x": ("array", {"shape": [224,
                    224, 3]}), "thumb": ("jpeg", {"quality": 85})}``. ``"ndarray"``
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
                    dtypes. ``compact=True`` for the numpy/torch/ndarray sink
                    makes each row's ``data`` exactly its elements, row-major,
                    at offset 0, instead of the storage a view reads from
                    (zero-copy, but written whole by Parquet/IPC); it copies
                    only a row that is not already compact. ``names=[...]``
                    (one per channel) and ``where="<name>"`` for the long
                    sink, which gives ``List[Struct{row, col, <names>...}]``,
                    one struct per pixel kept (``where``'s channel > 0, and
                    left out). A keyword that does not apply to the chosen format
                    is rejected rather than ignored, as is one that is not a
                    sink parameter at all. With a dict ``format`` these
                    apply to every output; naming one both here and in an
                    output's own params is refused.

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
                input=None if node._input is None else node._input._node_id,
                alias=node._alias,
            )
        for alias, node_id in self._aliases():
            graph.add_alias(alias, node_id)

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

    def background_divide(
        self, sigma: "FloatOrThisOrExpr", *, epsilon: float = 1e-6
    ) -> "LazyPipelineExpr":
        """
        Flat-field: divide this image by its own background,
        ``image / (blur(image, sigma) + epsilon)``.

        A wide blur is the slowly varying background (illumination, a
        detector's gain); dividing it out leaves the local structure. The
        result is a float image, as ``divide`` gives.

        Args:
            sigma: The background blur's sigma (wide: a fraction of the
                image's size). Accepts a Polars expression.
            epsilon: Added to the background so a black region does not
                divide by zero.

        Returns:
            New LazyPipelineExpr: the flat-fielded image.
        """
        from polars_cv.pipeline import Pipeline

        # Epsilon first: `add_constant` promotes to the image's float
        # (f32, or f64 for a 32/64-bit image), so the blur runs in it rather
        # than rounding an integer image's background to whole numbers. A
        # Gaussian's weights sum to one, so blur(x + e) is blur(x) + e.
        background = self.pipe(Pipeline().add_constant(epsilon).blur(sigma))
        return self.divide(background)

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

        # The contour node's own input; this image is the canvas, an operand
        # of the rasterize.
        rasterized = LazyPipelineExpr(
            column=contour._column,
            pipeline=raster_pipeline,
            node_id=_generate_node_id(),
            input=contour._input,
            merged=contour._merged,
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

        # The merge node is a passthrough of self's output.
        return LazyPipelineExpr(
            column=None,
            pipeline=self._receiving(),
            node_id=_generate_node_id(),
            input=self,
            merged=others,
        )

    def statistics(
        self,
        include: list[str] | None = None,
        *,
        mask: "LazyPipelineExpr | None" = None,
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
                - "median": The median
                - "pNN": The NN-th percentile, e.g. "p10", "p97.5"
                - "count": How many values (every channel; under ``mask``, the masked ones)
                If None, defaults to ["mean", "std", "min", "max"].
            mask: Reduce only the pixels this mask selects (u8 or float, set
                where > 0), as ``reduce_masked`` does. Over no pixel a
                statistic is NaN (0 for "sum" and "count").

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
        names, stat_nodes = self._stat_nodes(include, mask=mask)
        merged = self._merge_stat_nodes(stat_nodes)
        return merged.sink({name: "native" for name in names})

    def statistics_lazy(
        self,
        include: list[str] | None = None,
        *,
        mask: "LazyPipelineExpr | None" = None,
    ) -> "LazyPipelineExpr":
        """
        Create a lazy pipeline for computing multiple statistics.

        Unlike `statistics()` which returns a finalized pl.Expr, this method
        returns a LazyPipelineExpr that can be merged with other pipelines
        for multi-output composition.

        When sunk, each statistic becomes a separate field in the output struct.
        The stat nodes are aliased with prefixed names: "stat_mean", "stat_std", etc.

        Args:
            include: Statistics to compute, as ``statistics()`` names them
                ("mean", "std", "min", "max", "sum", "median", "count", or a
                percentile "pNN"). If None, defaults to
                ["mean", "std", "min", "max"].
            mask: Reduce only the pixels this mask selects, as
                ``statistics()``.

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
        _, stat_nodes = self._stat_nodes(include, prefix="stat_", mask=mask)
        return self._merge_stat_nodes(stat_nodes)

    # --- Internal Helpers ---

    def _stat_nodes(
        self,
        include: "list[str] | None",
        prefix: str = "",
        mask: "LazyPipelineExpr | None" = None,
    ) -> "tuple[list[str], list[LazyPipelineExpr]]":
        """Build one aliased reduction node per requested statistic.

        Shared by :meth:`statistics` and :meth:`statistics_lazy`, which differ
        only in the alias prefix and in whether they sink the result. They used
        to carry a copy each of the accepted-name set and of the five-arm
        dispatch, both ending ``else: continue`` — so a name that passed
        validation but missed the chain vanished from the output silently.
        Each statistic is now a ``Reduction`` the op parses (``reduce``), so
        there is no name list here. Under a mask every statistic reduces one
        ``masked_select`` node: the masked values are gathered once, not once
        per statistic.

        Args:
            include: Statistic names, or ``None`` for :data:`_DEFAULT_STATS`.
            prefix: Prepended to each output alias.
            mask: Reduce only the pixels this mask selects (``masked_select``).

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

        if not include:
            raise ValueError("At least one statistic must be included")

        names = [f"{prefix}{stat}" for stat in include]
        values = self if mask is None else self.masked_select(mask)
        nodes = [
            values.pipe(Pipeline().reduce(stat)).alias(name)
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

    def _receiving(self) -> "Pipeline":
        """The pipeline of a node that reads this expression's output, planned
        from its state.

        Every node that composes this expression with others starts here, so
        this expression's ops run once, in its own node. Cloning its pipeline
        instead replayed them on their own output (``add_constant(50)`` added
        100) or, from a root, decoded the column a second time.
        """
        from polars_cv._lib import Plan
        from polars_cv.pipeline import Pipeline

        new_pipeline = Pipeline()
        new_pipeline._plan = Plan.continuing(self._pipeline._state)
        return new_pipeline

    def _node_op(self, op: str, values: "dict[str, Any]") -> "LazyPipelineExpr":
        """Append the ``lazy_only`` op *op*, reading this and the other nodes
        among *values* (its arguments, by catalogue field name): every
        generated ``_LazyOpsMixin`` method."""
        # The op's rules read every operand (true division of two u8 is f32,
        # the shapes broadcast), so the operands' states go with it.
        new_pipeline = self._receiving()
        new_pipeline._add_node_op(op, values)

        return LazyPipelineExpr(
            column=None,
            pipeline=new_pipeline,
            node_id=_generate_node_id(),
            input=self,
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

            for dependency in node._dependencies():
                dfs(dependency)

            order.append(node)

        dfs(self)
        return order

    def _aliases(self) -> list[tuple[str, str]]:
        """Every ``(alias, node id)`` this expression reaches.

        ``alias()`` returns a copy under the same node id, so one node may be
        reached under several aliases: walked per object, not per node id,
        or all but one alias of a node would be lost.
        """
        seen: set[int] = set()
        found: list[tuple[str, str]] = []
        stack: list[LazyPipelineExpr] = [self]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if node._alias is not None:
                found.append((node._alias, node._node_id))
            stack.extend(node._dependencies())
        return found

    # --- Prevent accidental use as pl.Expr ---

    def __repr__(self) -> str:
        """Return string representation with guidance."""
        dependency_ids = [d._node_id for d in self._dependencies()]
        alias_str = f", alias={self._alias!r}" if self._alias else ""
        return (
            f"LazyPipelineExpr(node={self._node_id!r}{alias_str}, "
            f"depends_on={dependency_ids}) - call .sink(format) to execute"
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
