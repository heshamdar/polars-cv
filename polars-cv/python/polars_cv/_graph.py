"""
Pipeline graph representation and serialization.

This module provides the PipelineGraph class which represents a DAG of
pipeline operations and handles serialization for the Rust backend.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from polars_cv._graph_viz import get_graphviz_out

if TYPE_CHECKING:
    import pydot

    from polars_cv._types import OpSpec
    from polars_cv.pipeline import Pipeline

# Path to the compiled Rust library
LIB_PATH = Path(__file__).parent


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
    def expected_ndim(self) -> int | None:
        """Get the expected number of dimensions of this node's pipeline."""
        return self.pipeline._expected_ndim

    @property
    def expected_shape(self) -> list[int] | None:
        """Get the expected output shape of this node's pipeline if deterministic."""
        hints = self.pipeline._shape_hints
        if (
            hints.height
            and not hints.height.is_expr
            and hints.width
            and not hints.width.is_expr
        ):
            # Default to 3 channels if not specified for image sources
            channels = 3
            if hints.channels and not hints.channels.is_expr:
                channels = hints.channels.value
            return [hints.height.value, hints.width.value, channels]
        return None


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
        >>> expr = graph.to_expr()  # Returns fused pl.Expr
        >>>
        >>> # Or for multi-output:
        >>> graph.set_multi_output({"original": "png", "mask": "numpy"})
        >>> expr = graph.to_expr()  # Returns Struct expression
    """

    def __init__(self) -> None:
        """Initialize an empty pipeline graph."""
        self._nodes: dict[str, GraphNode] = {}
        self._output: GraphOutput | None = None
        self._multi_output: MultiGraphOutput | None = None
        self._column_bindings: dict[str, int] = {}
        # Mapping from alias names to node IDs
        self._alias_to_node: dict[str, str] = {}

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

        for node in self._nodes.values():
            # Only consider root nodes (those with column bindings)
            if node.column is None:
                continue

            # Create a group key from column + source spec
            col_str = str(node.column)
            source = node.pipeline._source
            source_key = hash(source) if source else "none"
            group_key = f"{col_str}:{source_key}"

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

        # Create a new pipeline with just the prefix operations
        shared_pipeline = Pipeline()
        shared_pipeline._source = template_node.pipeline._source
        shared_pipeline._shape_hints = template_node.pipeline._shape_hints
        shared_pipeline._ops = list(prefix_ops)  # Copy the prefix ops
        shared_pipeline._expr_refs = template_node.pipeline._expr_refs.copy()

        # Compute the correct domain and dtype for the prefix operations
        # This ensures static type inference matches runtime behavior
        domain, dtype, ndim = Pipeline._compute_output_domain_dtype_ndim(
            prefix_ops,
            initial_ndim=template_node.pipeline._expected_ndim,
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
        # Remove the prefix operations from this node's pipeline
        node.pipeline._ops = node.pipeline._ops[prefix_len:]

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
        """
        from polars.plugins import register_plugin_function

        if self._output is None and self._multi_output is None:
            raise ValueError(
                "No output set. Call set_output() or set_multi_output() first."
            )

        # Optimize: extract common operation prefixes into shared nodes
        self._optimize_common_subexpressions()

        # Validate that root nodes have columns
        for node in self._nodes.values():
            if not node.upstream and node.column is None:
                msg = (
                    f"Root node '{node.node_id}' has no column set. "
                    "Call set_root_column() or pass column when adding the node."
                )
                raise ValueError(msg)

        # Build column bindings (assign index to each unique column)
        self._build_column_bindings()

        # Collect all column expressions in order (source columns first)
        columns = self._get_ordered_columns()

        # Collect expression columns from all nodes' pipelines
        expr_columns, expr_column_names = self._get_expr_columns()

        # Add expression columns to args (after source columns)
        all_args = columns + expr_columns

        # Serialize graph to JSON
        graph_json = self._to_json()

        # Unified graph execution handles both single and multi-output
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="vb_graph",
            args=all_args,
            kwargs={
                "graph_json": graph_json,
                "expr_column_names": expr_column_names,
            },
            is_elementwise=True,
        )

    def _build_column_bindings(self) -> None:
        """Build mapping from node IDs to column indices.

        Only root nodes (those with columns) get bindings.
        Non-root nodes receive data from upstream nodes.
        """
        seen_columns: dict[str, int] = {}
        idx = 0

        for node_id, node in self._nodes.items():
            if node.column is not None:
                # Get a string representation of the column for deduplication
                col_str = str(node.column)
                if col_str not in seen_columns:
                    seen_columns[col_str] = idx
                    idx += 1
                self._column_bindings[node_id] = seen_columns[col_str]
            # Non-root nodes don't have column bindings - they receive from upstream

    def _get_ordered_columns(self) -> list[pl.Expr]:
        """Get unique column expressions in order.

        Only includes columns from root nodes (nodes with column expressions).
        """
        seen: set[str] = set()
        columns: list[pl.Expr] = []

        for node in self._nodes.values():
            if node.column is not None:
                col_str = str(node.column)
                if col_str not in seen:
                    seen.add(col_str)
                    columns.append(node.column)

        return columns

    def _get_expr_columns(self) -> tuple[list[pl.Expr], list[str]]:
        """Get expression columns from all node pipelines.

        Collects expression parameters (like pl.col("height")) from all
        pipeline operations in the graph, deduplicating by string representation.

        Returns:
            Tuple of (expression_list, column_names_list).
            The expressions and names are in the same order.

        Note:
            Uses the expression's string representation as the identifier name.
            This matches the key used in ParamValue.to_dict() to ensure expression
            values can be correctly looked up on the Rust side. This avoids
            collisions when multiple expressions share the same root column
            (e.g., col("x").list.get(0).max() and col("x").list.get(1).max()).
        """
        seen: set[str] = set()
        expr_columns: list[pl.Expr] = []
        expr_names: list[str] = []

        for node in self._nodes.values():
            # Get expression columns from this node's pipeline
            for expr in node.pipeline._get_expr_columns():
                expr_str = str(expr)
                if expr_str not in seen:
                    seen.add(expr_str)
                    expr_columns.append(expr)
                    # Use the expression's string representation as the identifier.
                    # This matches the key used in ParamValue.to_dict() for lookups.
                    expr_names.append(expr_str)

        return expr_columns, expr_names

    def _to_dict(self) -> dict[str, Any]:
        if self._output is None and self._multi_output is None:
            raise ValueError("No output set")

        # Build nodes dict
        nodes_dict: dict[str, Any] = {}
        for node_id, node in self._nodes.items():
            # Get the pipeline's JSON representation without sink
            # We'll add sink info to the output specification
            node_spec = node.pipeline._to_spec_dict()
            node_spec["upstream"] = node.upstream
            if node.alias is not None:
                node_spec["alias"] = node.alias
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
                    "expected_ndim": node.expected_ndim if node else None,
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
                "expected_ndim": node.expected_ndim if node else None,
            }

        graph_spec = {
            "nodes": nodes_dict,
            "outputs": outputs_spec,
            "column_bindings": self._column_bindings,
        }

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
        return get_graphviz_out(self)
