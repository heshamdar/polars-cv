"""
Pipeline graph representation and serialization.

This module provides the PipelineGraph class which represents a DAG of
pipeline operations and handles serialization for the Rust backend.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from polars_cv._graph_viz import get_graphviz_out
from polars_cv._types import SlotTable

if TYPE_CHECKING:
    import pydot

    from polars_cv._optimize import OptFlags
    from polars_cv._types import OpSpec
    from polars_cv.pipeline import Pipeline


@dataclass
class GraphNode:
    """
    A node in the pipeline graph.

    Attributes:
        node_id: Unique identifier for this node.
        pipeline: The Pipeline specification for this node.
        column: The Polars column expression this node reads from (None for non-root nodes).
        upstream: List of upstream node IDs this node depends on.
        alias: Optional user-defined alias for multi-output support.
    """

    node_id: str
    pipeline: "Pipeline"
    column: pl.Expr | None  # None for non-root nodes that receive from upstream
    upstream: list[str] = field(default_factory=list)
    alias: str | None = None

    @property
    def domain(self) -> str:
        """Get the output domain of this node's pipeline."""
        return self.pipeline.current_domain()

    @property
    def output_dtype(self) -> str:
        """Get the expected output dtype of this node's pipeline."""
        return self.pipeline.output_dtype()

    @property
    def output_encoding(self) -> str | None:
        """Get the sink encoding selector of this node's pipeline, if any."""
        return self.pipeline.output_encoding()

    @property
    def expected_ndim(self) -> int | None:
        """Get the expected number of dimensions of this node's pipeline."""
        return self.pipeline._expected_ndim

    @property
    def expected_shape(self) -> list[int] | None:
        """Get the expected output shape of this node's pipeline if deterministic.

        Only reported for a rank-3 ``[H, W, C]`` output. The hints track H/W/C
        specifically, so at any other rank they cannot describe the shape —
        publishing ``[H, W, C]`` for a rank-2 output is exactly how
        ``channel_select`` used to declare a schema execution could not produce.
        """
        if self.pipeline._expected_ndim != 3:
            return None
        hints = self.pipeline._shape_hints
        if (
            hints.height
            and not hints.height.is_expr
            and hints.width
            and not hints.width.is_expr
        ):
            if not hints.channels or hints.channels.is_expr:
                return None
            return [hints.height.value, hints.width.value, hints.channels.value]
        return None

    @property
    def shape_asserted(self) -> bool:
        """Did any dimension of :attr:`expected_shape` come from ``assert_shape``?

        Decides *who* a plan/exec divergence is reported against. A shape the
        ops' contracts inferred and execution then contradicted is a contract
        bug, and ``validate_output_schema`` says so. A shape the user asserted
        is a claim about their data, and blaming "the Rust implementation" for
        it — which is what happened — sends them to the wrong file.
        """
        return bool(self.pipeline._asserted_dims)


@dataclass
class GraphOutput:
    """
    Output specification for the pipeline graph (single output mode).

    Attributes:
        node_id: The node whose output to return.
        format: Output format (e.g., "numpy", "torch", "png").
        params: Additional output parameters.
    """

    node_id: str
    format: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiGraphOutput:
    """
    Output specification for multi-output mode.

    Attributes:
        outputs: Mapping from alias names to (node_id, format, params).
    """

    outputs: dict[str, tuple[str, str, dict[str, Any]]] = field(default_factory=dict)


class PipelineGraph:
    """
    Represents a DAG of pipeline operations for fused execution.

    This class collects multiple LazyPipelineExpr nodes and their dependencies,
    serializes them to a JSON graph specification, and registers them as a
    single Polars plugin function call.

    The graph structure enables:
    - Zero intermediate materialization between composed operations
    - Single plugin call for the entire graph
    - Automatic topological ordering of execution
    - Multi-output support via aliases

    Example:
        >>> graph = PipelineGraph()
        >>> graph.add_node("img", img_pipeline, pl.col("image"), alias="original")
        >>> graph.add_node("mask", mask_pipeline, pl.col("contour"), upstream=["img"])
        >>> graph.set_output("mask", "numpy")  # Single output
        >>> from polars_cv import OptFlags
        >>> expr = graph.optimize(OptFlags.all()).to_expr()  # Returns fused pl.Expr
        >>>
        >>> # Or for multi-output:
        >>> graph.set_multi_output({"original": "png", "mask": "numpy"})
        >>> expr = graph.optimize(OptFlags.all()).to_expr()  # Struct expression
    """

    def __init__(self) -> None:
        """Initialize an empty pipeline graph."""
        self._nodes: dict[str, GraphNode] = {}
        self._output: GraphOutput | None = None
        self._multi_output: MultiGraphOutput | None = None
        # Mapping from alias names to node IDs
        self._alias_to_node: dict[str, str] = {}
        # Set by ``optimize()``. Serialization (``to_expr``) refuses to run on a
        # graph that has not passed through the optimization phase, so no exit
        # can silently emit an unoptimized graph — optimization lives in exactly
        # one place (``optimize()``, invoked by ``LazyPipelineExpr.sink``).
        self._optimized: bool = False
        # Engine-tier optimization toggles, set by ``optimize()`` from the
        # ``OptFlags`` and serialized in the graph's ``opt`` object for Rust.
        # Empty until optimized; an empty/absent object means all-on in Rust.
        self._opt_config: dict[str, bool] = {}

    def add_node(
        self,
        node_id: str,
        pipeline: "Pipeline",
        column: pl.Expr | None = None,
        upstream: list[str] | None = None,
        alias: str | None = None,
    ) -> None:
        """
        Add a node to the pipeline graph.

        Args:
            node_id: Unique identifier for this node.
            pipeline: The Pipeline specification.
            column: The Polars column expression this node reads from.
                    None for non-root nodes that receive from upstream.
            upstream: List of upstream node IDs this depends on.
            alias: Optional user-defined alias for multi-output.
        """
        self._nodes[node_id] = GraphNode(
            node_id=node_id,
            pipeline=pipeline,
            column=column,
            upstream=upstream or [],
            alias=alias,
        )
        # Track alias -> node_id mapping
        if alias is not None:
            self._alias_to_node[alias] = node_id

    def set_root_column(self, column: pl.Expr) -> None:
        """
        Set the input column for all root nodes (nodes with no upstream).

        This is used when the column is not known at graph construction time,
        such as when converting a Pipeline to a graph.

        Args:
            column: The Polars column expression for root nodes.
        """
        for node in self._nodes.values():
            if not node.upstream:
                node.column = column

    def set_output(self, node_id_or_alias: str, format: str, **kwargs: Any) -> None:
        """
        Set the output node and format for single-output mode.

        Args:
            node_id_or_alias: The node ID or alias whose output to return.
            format: Output format (e.g., "numpy", "torch", "png").
            **kwargs: Additional output parameters (e.g., quality for jpeg).
        """
        # Resolve alias to node_id if needed
        if node_id_or_alias in self._alias_to_node:
            node_id = self._alias_to_node[node_id_or_alias]
        elif node_id_or_alias in self._nodes:
            node_id = node_id_or_alias
        else:
            raise ValueError(f"Node or alias '{node_id_or_alias}' not found in graph")

        self._output = GraphOutput(node_id=node_id, format=format, params=kwargs)
        self._multi_output = None

    def set_multi_output(
        self,
        outputs: dict[str, str],
        **kwargs: Any,
    ) -> None:
        """
        Set multiple outputs for multi-output mode.

        Args:
            outputs: Mapping from alias names to output formats.
            **kwargs: Additional output parameters (e.g., quality for jpeg).

        Raises:
            ValueError: If any alias is not found in the graph.
        """
        multi = MultiGraphOutput()

        for alias, fmt in outputs.items():
            # Find the node ID for this alias
            if alias not in self._alias_to_node:
                # List available aliases for helpful error message
                available = list(self._alias_to_node.keys())
                msg = (
                    f"Alias '{alias}' not found in graph. "
                    f"Available aliases: {available}. "
                    f"Use .alias('{alias}') to define it."
                )
                raise ValueError(msg)

            node_id = self._alias_to_node[alias]
            multi.outputs[alias] = (node_id, fmt, kwargs.copy())

        self._multi_output = multi
        self._output = None

    def is_multi_output(self) -> bool:
        """Check if the graph uses multi-output mode."""
        return self._multi_output is not None

    # --- Optimization phase ---

    def optimize(self, flags: "OptFlags") -> "PipelineGraph":
        """Rewrite the logical graph into an equivalent physical graph.

        The single optimization phase (see :mod:`polars_cv._optimize`).
        Construction and serialization never optimize; this applies each
        registered pass, gated by ``flags``, in :data:`OPTIMIZATION_PASSES`
        order, mutating the graph in place and returning ``self`` for chaining.

        The two tiers are handled differently, by each pass's ``tier``:

        - **logical** passes (CSE, identity elimination, spatial-window
          pushdown) are applied here, rewriting node pipelines.
        - **engine** passes are per-row lowering that runs later in Rust; their
          flags are captured into :attr:`_opt_config` and serialized in the
          graph's ``opt`` object (see :meth:`_to_dict`), gating the Rust
          ``OptConfig``. Nothing is rewritten here for them.

        The run order is *data* (the tuple's order), not hand-wired here, so
        adding a pass is a registry edit; a fixed order keeps the physical graph
        deterministic. The logical passes do not fight: identity elimination only
        deletes no-ops (it can expose more that follows), and pushdown moves a
        crop only past ``Pointwise`` ops.

        Every pass is output-preserving and byte-identical when toggled (see
        ``polars_cv._optimize.PassSpec.bit_exact``).

        The passes rewrite node pipelines in place, so the graph first takes its
        own clone of each one: ``cv.pipe(p)`` holds the caller's ``Pipeline`` by
        reference, and ``Pipeline`` is immutable from the caller's view — the
        physical graph must own independent copies to mutate.
        """
        from polars_cv._optimize import OPTIMIZATION_PASSES

        for node in self._nodes.values():
            node.pipeline = node.pipeline._clone()
        handlers = self._pass_handlers()
        for spec in OPTIMIZATION_PASSES:
            if spec.tier != "logical":
                continue
            if not flags.enabled(spec.name):
                continue
            scope, run = handlers[spec.name]
            if scope == "graph":
                run(self)
            else:  # "node": rewrite each node's ops in place
                for node in self._nodes.values():
                    run(node.pipeline)
        # Engine-tier flags do not rewrite the Python graph; they ride to Rust in
        # the serialized `opt` object, keyed by the Rust `OptConfig` field names.
        self._opt_config = flags.engine_opt()
        self._optimized = True
        return self

    @staticmethod
    def _pass_handlers() -> "dict[str, tuple[str, Any]]":
        """Each **logical**-tier pass's applicator, keyed by name.

        A ``"graph"`` handler takes the whole :class:`PipelineGraph` and may
        rewrite topology (CSE splits siblings onto a shared prefix node); a
        ``"node"`` handler takes one node :class:`Pipeline` and rewrites its ops
        in place. This map's keys must equal :data:`LOGICAL_PASS_NAMES` (engine
        passes have no Python handler) — a logical pass without a handler, or a
        handler without a logical pass, fails
        ``test_pass_handlers_cover_every_logical_pass``.
        """
        from polars_cv.pipeline import Pipeline

        return {
            "common_subexpression_elimination": (
                "graph",
                PipelineGraph._optimize_common_subexpressions,
            ),
            "identity_elimination": ("node", Pipeline._eliminate_identities_inplace),
            "spatial_window_pushdown": (
                "node",
                Pipeline._hoist_spatial_windows_inplace,
            ),
        }

    # --- CSE Optimization ---

    def _optimize_common_subexpressions(self) -> None:
        """
        Extract common operation prefixes into shared nodes.

        This optimization detects when multiple pipelines share the same
        sequence of operations (starting from the source) and creates a
        single shared node for that prefix. The original nodes are then
        updated to use the shared node as their upstream.

        Example:
            Before:
                gray_pipe: source → resize → grayscale
                mask_pipe: source → resize → grayscale → threshold → extract

            After:
                _shared:   source → resize → grayscale
                gray_pipe: (empty) ← upstream: _shared
                mask_pipe: threshold → extract ← upstream: _shared
        """
        # Group root nodes by (source column, source spec)
        groups = self._group_nodes_for_cse()

        for group_key, nodes in groups.items():
            if len(nodes) < 2:
                continue

            # Find common prefix among all nodes in this group
            ops_lists = [node.pipeline._ops for node in nodes]
            common_ops = self._find_common_prefix(ops_lists)

            if len(common_ops) == 0:
                continue

            # Create shared node for the common prefix
            shared_id = self._create_shared_node(nodes[0], common_ops)

            # Update original nodes to use shared node as upstream
            for node in nodes:
                self._update_node_to_use_shared(node, shared_id, len(common_ops))

    def _group_nodes_for_cse(self) -> dict[str, list[GraphNode]]:
        """
        Group nodes that could potentially share a common prefix.

        Nodes are grouped by:
        1. Same source column (or both have no column)
        2. Same source spec (format, dtype, etc.)

        Returns:
            Dict mapping group keys to lists of nodes in that group.
        """
        groups: dict[str, list[GraphNode]] = {}
        table = self._slot_table()

        for node in self._nodes.values():
            # Only consider root nodes (those with column bindings)
            if node.column is None:
                continue

            # Create a group key from column + source spec. Key on the source's
            # canonical serialization, not ``hash(source)``: a hash collision
            # would bucket two *different* sources together and fuse a shared
            # prefix node with the wrong source. String equality cannot collide.
            col_key = table.index(node.column)
            source = node.pipeline._source
            source_key = (
                json.dumps(source.to_dict(table.index), sort_keys=True)
                if source
                else "none"
            )
            group_key = f"{col_key}:{source_key}"

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(node)

        return groups

    def _find_common_prefix(self, ops_lists: list[list["OpSpec"]]) -> list["OpSpec"]:
        """
        Find the longest common prefix across all operation lists.

        Args:
            ops_lists: List of operation lists to compare.

        Returns:
            The common prefix (may be empty if no common ops).
        """
        if not ops_lists:
            return []

        # Find minimum length
        min_len = min(len(ops) for ops in ops_lists)
        if min_len == 0:
            return []

        prefix: list["OpSpec"] = []
        for i in range(min_len):
            first = ops_lists[0][i]
            # Check if all lists have the same op at position i
            if all(ops[i] == first for ops in ops_lists[1:]):
                prefix.append(first)
            else:
                break

        return prefix

    def _create_shared_node(
        self, template_node: GraphNode, prefix_ops: list["OpSpec"]
    ) -> str:
        """
        Create a shared node containing the common prefix operations.

        Args:
            template_node: A node to use as template for source/column.
            prefix_ops: The operations to include in the shared node.

        Returns:
            The node_id of the newly created shared node.
        """
        from polars_cv.pipeline import Pipeline

        shared_id = f"_cse_{uuid.uuid4().hex[:8]}"

        # Create a new pipeline with just the prefix operations. It inherits
        # the template's whole state through the one copy mechanism, then
        # overrides the ops; see `_STATE_COPIERS` in `pipeline.py`.
        #
        # The per-row policies come along, which is a no-op for the graph
        # spec today: `_to_dict` hoists the *set* of non-default policies
        # across all nodes, and the template node keeps its own. Copying them
        # is what keeps that true if the hoist ever reads one node.
        shared_pipeline = Pipeline()
        shared_pipeline._copy_state_from(template_node.pipeline)
        # The prefix ops keep their original indices, so everything keyed by
        # op position carries over unshifted (identity elimination reads the
        # entering-hints snapshots).
        shared_pipeline._set_ops_slice(prefix_ops, shift=0)

        # Compute the correct domain and dtype for the prefix operations.
        # The fold starts at op 0, so it is seeded with the template's
        # post-source (pre-op) state, not its final tracked state.
        domain, dtype, ndim = Pipeline._compute_output_domain_dtype_ndim(
            prefix_ops,
            initial_dtype=template_node.pipeline._initial_output_dtype,
            initial_ndim=template_node.pipeline._initial_expected_ndim,
        )
        shared_pipeline._current_domain = domain
        shared_pipeline._output_dtype = dtype
        shared_pipeline._expected_ndim = ndim

        # Create the shared node
        shared_node = GraphNode(
            node_id=shared_id,
            pipeline=shared_pipeline,
            column=template_node.column,
            upstream=[],
            alias=None,  # Shared nodes don't have user aliases
        )

        self._nodes[shared_id] = shared_node

        return shared_id

    def _update_node_to_use_shared(
        self, node: GraphNode, shared_id: str, prefix_len: int
    ) -> None:
        """
        Update a node to use a shared node as its upstream.

        Args:
            node: The node to update.
            shared_id: The ID of the shared node to use as upstream.
            prefix_len: Number of operations that are now in the shared node.
        """
        # Remove the prefix operations from this node's pipeline. Everything
        # keyed by op index (entering-hints snapshots, assert_shape positions)
        # shifts with them.
        node.pipeline._set_ops_slice(node.pipeline._ops[prefix_len:], shift=prefix_len)
        # The node's pre-op state is now the shared node's output state.
        shared_pipeline = self._nodes[shared_id].pipeline
        node.pipeline._initial_output_dtype = shared_pipeline._output_dtype
        node.pipeline._initial_expected_ndim = shared_pipeline._expected_ndim

        # Set the shared node as upstream
        if not node.upstream:
            node.upstream = [shared_id]
        else:
            # Prepend shared node to existing upstream
            node.upstream = [shared_id] + node.upstream

        # Clear column binding - now receives input from upstream
        # Keep the column reference for column_bindings but mark it as non-root
        # Actually, we need to keep track that this node no longer reads directly
        # The shared node will have the column binding instead
        node.column = None

    def to_expr(self) -> pl.Expr:
        """
        Convert the graph to a Polars expression.

        This serializes the entire graph to JSON and registers it as a
        single plugin function call.

        Returns:
            A Polars expression that executes the fused graph.
            - For single output ("_output" only): Binary column
            - For multi-output: Struct column with named Binary fields

        Raises:
            ValueError: If no output is set.
            RuntimeError: If the graph has not been optimized. Call
                ``optimize(flags)`` first (``.sink()`` does this).
        """
        from polars_cv import _plugin

        if self._output is None and self._multi_output is None:
            raise ValueError(
                "No output set. Call set_output() or set_multi_output() first."
            )

        # Serialization only serializes. Optimization (every logical pass) is
        # the explicit `optimize(flags)` phase — the single site, invoked by
        # `.sink()`. `to_expr()` refuses an un-optimized graph rather than
        # silently emitting one: that is how the sink-free `to_graph().to_expr()`
        # route (which does not run `sink`) is kept from diverging from `sink()`.
        if not self._optimized:
            raise RuntimeError(
                "PipelineGraph.to_expr() requires the optimization phase to run "
                "first. Use `.sink(...)` (which optimizes with opt_flags), or "
                "`graph.optimize(flags).to_expr()` for the low-level path."
            )

        # Validate that root nodes have columns
        for node in self._nodes.values():
            if not node.upstream and node.column is None:
                msg = (
                    f"Root node '{node.node_id}' has no column set. "
                    "Call set_root_column() or pass column when adding the node."
                )
                raise ValueError(msg)

        # One positional table for the plugin's inputs: root columns first,
        # then every expression parameter. Serialization reads the same table,
        # so positions and arguments cannot disagree.
        table = self._slot_table()

        # Unified graph execution handles both single and multi-output
        return _plugin.call(
            "vb_graph",
            args=table.columns,
            kwargs={"graph_json": self._to_json()},
            is_elementwise=True,
        )

    def _slot_table(self) -> SlotTable:
        """The plugin's inputs, in order: each distinct root column, then each
        distinct expression parameter (identity by ``Expr.meta.eq``).

        Derived from the nodes on demand, so it always describes the current
        graph (CSE rewrites nodes) and every reader gets the same positions.
        """
        table = SlotTable()
        for node in self._nodes.values():
            if node.column is not None:
                table.add(node.column)
        for node in self._nodes.values():
            for expr in node.pipeline._get_expr_columns():
                table.add(expr)
        return table

    def _to_dict(self) -> dict[str, Any]:
        if self._output is None and self._multi_output is None:
            raise ValueError("No output set")

        # Build nodes dict
        table = self._slot_table()
        nodes_dict: dict[str, Any] = {}
        for node_id, node in self._nodes.items():
            # Get the pipeline's JSON representation without sink
            # We'll add sink info to the output specification
            node_spec = node.pipeline._to_spec_dict(table.index)
            node_spec["upstream"] = node.upstream
            nodes_dict[node_id] = node_spec

        # Build unified outputs dict (always use "outputs" format)
        outputs_spec: dict[str, Any] = {}

        if self._multi_output is not None:
            # Multi-output mode
            for alias, (node_id, fmt, params) in self._multi_output.outputs.items():
                node = self._nodes.get(node_id)
                outputs_spec[alias] = {
                    "node": node_id,
                    "sink": {
                        "format": fmt,
                        **params,
                    },
                    # Add domain and dtype for static type inference
                    "expected_domain": node.domain if node else "buffer",
                    "expected_dtype": node.output_dtype if node else "u8",
                    "expected_shape": node.expected_shape if node else None,
                    "shape_asserted": node.shape_asserted if node else False,
                    "expected_ndim": node.expected_ndim if node else None,
                    "expected_encoding": node.output_encoding if node else None,
                }
        else:
            # Single output mode - use "_output" as the key
            assert self._output is not None
            node = self._nodes.get(self._output.node_id)
            outputs_spec["_output"] = {
                "node": self._output.node_id,
                "sink": {
                    "format": self._output.format,
                    **self._output.params,
                },
                # Add domain and dtype for static type inference
                "expected_domain": node.domain if node else "buffer",
                "expected_dtype": node.output_dtype if node else "u8",
                "expected_shape": node.expected_shape if node else None,
                "shape_asserted": node.shape_asserted if node else False,
                "expected_ndim": node.expected_ndim if node else None,
                "expected_encoding": node.output_encoding if node else None,
            }

        graph_spec = {
            # Wire-format version; the Rust side rejects versions newer than
            # it understands instead of misparsing them.
            "version": 1,
            "nodes": nodes_dict,
            "outputs": outputs_spec,
            "column_bindings": {
                node_id: table.index(node.column)
                for node_id, node in self._nodes.items()
                if node.column is not None
            },
            # Engine-tier optimization toggles → Rust `OptConfig`. Keys are the
            # `OptConfig` field names; missing keys default on. Distinct opt
            # settings key distinct compiled-graph cache entries, so a per-query
            # toggle actually re-executes.
            "opt": self._opt_config,
        }

        # Both per-row policies are graph-level settings collected from the
        # composed pipelines; distinct non-default policies are ambiguous.
        # Emitted only when non-default, so unaffected graphs serialize
        # byte-identically to before.
        for attr, key in (
            ("_on_error", "on_error"),
            ("_on_null_param", "on_null_param"),
        ):
            policies = {
                getattr(node.pipeline, attr)
                for node in self._nodes.values()
                if getattr(node.pipeline, attr) != "raise"
            }
            if len(policies) > 1:
                msg = (
                    f"Conflicting {key} policies in composed pipelines: "
                    f"{sorted(policies)}. All pipelines in a graph must agree."
                )
                raise ValueError(msg)
            if policies:
                graph_spec[key] = policies.pop()

        return graph_spec

    def _to_json(self) -> str:
        """
        Serialize the graph to JSON for the Rust backend.

        Always uses unified "outputs" format. Single output uses "_output" key.
        The Rust backend determines whether to return Binary or Struct based on
        the number of outputs.

        Returns:
            JSON string representation of the graph.
        """
        graph_spec = self._to_dict()
        return json.dumps(graph_spec)

    def topological_order(self) -> list[str]:
        """
        Get nodes in topological order (dependencies first).

        For multi-output graphs, includes all nodes reachable from any output.

        Returns:
            List of node IDs in execution order.
        """
        visited: set[str] = set()
        order: list[str] = []

        def dfs(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)

            node = self._nodes.get(node_id)
            if node:
                for upstream_id in node.upstream:
                    dfs(upstream_id)
                order.append(node_id)

        # Get all output nodes
        output_nodes: set[str] = set()
        if self._output:
            output_nodes.add(self._output.node_id)
        if self._multi_output:
            for node_id, _, _ in self._multi_output.outputs.values():
                output_nodes.add(node_id)

        # DFS from all output nodes
        for node_id in output_nodes:
            dfs(node_id)

        return order

    def get_output_nodes(self) -> set[str]:
        """
        Get the set of node IDs that are output targets.

        This is useful for optimization - these nodes should not be
        optimized away or fused past.

        Returns:
            Set of node IDs that are designated as outputs.
        """
        output_nodes: set[str] = set()
        if self._output:
            output_nodes.add(self._output.node_id)
        if self._multi_output:
            for node_id, _, _ in self._multi_output.outputs.values():
                output_nodes.add(node_id)
        return output_nodes

    def show_graph(self) -> pydot.Dot:
        """Build dot representation of graph."""
        return get_graphviz_out(self)  # ty: ignore[invalid-return-type]
