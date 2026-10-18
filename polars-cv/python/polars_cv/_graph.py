"""
Pipeline graph representation and serialization.

This module provides the PipelineGraph class which represents a DAG of
pipeline operations and handles serialization for the Rust backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from polars_cv._graph_viz import get_graphviz_out
from polars_cv._ops_generated import LogicalPass
from polars_cv._types import SlotTable

if TYPE_CHECKING:
    import pydot

    from polars_cv._optimize import OptFlags
    from polars_cv.pipeline import Pipeline


#: The graph-level per-row policies, by ``Pipeline`` attribute: wire key.
_POLICIES = {"_on_error": "on_error", "_on_null_param": "on_null_param"}


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
        # Build-time ids of nodes CSE replaced, by the node now computing
        # their value: plans name the nodes they read by build-time id.
        self._renamed: dict[str, str] = {}

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
        if alias is not None:
            self.add_alias(alias, node_id)

    def add_alias(self, alias: str, node_id: str) -> None:
        """Name ``node_id``'s output ``alias``. A node may carry several
        aliases; an alias names one node.

        Raises:
            ValueError: If ``alias`` already names a different node.
        """
        known = self._alias_to_node.get(alias)
        if known is not None and known != node_id:
            msg = (
                f"Alias {alias!r} names two different nodes; an alias names "
                "one output. Give each branch its own alias."
            )
            raise ValueError(msg)
        self._alias_to_node[alias] = node_id

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
        outputs: "dict[str, str | tuple[str, dict[str, Any]]]",
        **kwargs: Any,
    ) -> None:
        """
        Set multiple outputs for multi-output mode.

        Args:
            outputs: Mapping from alias names to an output format, or to
                ``(format, params)`` for parameters of that output alone
                (``("jpeg", {"quality": 85})``).
            **kwargs: Output parameters applied to every output.

        Raises:
            ValueError: If any alias is not found in the graph, or a parameter
                is given both per output and shared.
            TypeError: If an output is neither a format nor ``(format, params)``.
        """
        multi = MultiGraphOutput()

        for alias, spec in outputs.items():
            if isinstance(spec, str):
                fmt, own = spec, {}
            elif (
                isinstance(spec, tuple)
                and len(spec) == 2
                and isinstance(spec[0], str)
                and isinstance(spec[1], dict)
            ):
                fmt, own = spec
            else:
                msg = (
                    f"output '{alias}': expected a format string or "
                    f"(format, {{params}}), got {spec!r}"
                )
                raise TypeError(msg)
            both = sorted(own.keys() & kwargs.keys())
            if both:
                msg = (
                    f"output '{alias}': {', '.join(repr(k) for k in both)} given "
                    "both for this output and for every output; give it once"
                )
                raise ValueError(msg)

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
            multi.outputs[alias] = (node_id, fmt, {**kwargs, **own})

        self._multi_output = multi
        self._output = None

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
        ``tests/test_optimize_equivalence.py``).

        The passes rewrite node pipelines in place, so the graph first takes its
        own clone of each one: ``cv.pipe(p)`` holds the caller's ``Pipeline`` by
        reference, and ``Pipeline`` is immutable from the caller's view — the
        physical graph must own independent copies to mutate.
        """
        from polars_cv._optimize import OPTIMIZATION_PASSES

        self._check_roots()
        self._drop_unread_nodes()
        for node in self._nodes.values():
            node.pipeline = node.pipeline._clone()
        for spec in OPTIMIZATION_PASSES:
            if spec.tier != "logical" or not flags.enabled(spec.name):
                continue
            if spec.name == LogicalPass.COMMON_SUBEXPRESSION_ELIMINATION:
                # Graph scope: it needs the expressions' identities, which
                # only Python holds.
                self._optimize_common_subexpressions()
            else:
                # Node scope: Rust (`Plan.run_pass`) decides, and refuses a name
                # that is not a `LogicalPass`.
                for node in self._nodes.values():
                    node.pipeline._run_node_pass(spec.name)
        # Engine-tier flags do not rewrite the Python graph; they ride to Rust in
        # the serialized `opt` object, keyed by the Rust `OptConfig` field names.
        self._opt_config = flags.engine_opt()
        self._optimized = True
        return self

    def _check_roots(self) -> None:
        """Refuse a node that reads neither a column nor another node."""
        for node in self._nodes.values():
            if not node.upstream and node.column is None:
                msg = (
                    f"Root node '{node.node_id}' has no column set. "
                    "Pass the column when adding the node."
                )
                raise ValueError(msg)

    def _graph_policy(self, attr: str) -> str:
        """A graph-level per-row policy (``_on_error``/``_on_null_param``):
        the one non-default value the composed pipelines set, or ``"raise"``.

        Raises:
            ValueError: If they set different non-default values.
        """
        policies = {
            getattr(node.pipeline, attr)
            for node in self._nodes.values()
            if getattr(node.pipeline, attr) != "raise"
        }
        if len(policies) > 1:
            msg = (
                f"Conflicting {_POLICIES[attr]} policies in composed pipelines: "
                f"{sorted(policies)}. All pipelines in a graph must agree."
            )
            raise ValueError(msg)
        return policies.pop() if policies else "raise"

    def _output_node_ids(self) -> list[str]:
        """The nodes the outputs read, in output order. Before an output is
        set, nothing says which nodes matter, so every node counts as one."""
        if self._multi_output is not None:
            return [node_id for node_id, _, _ in self._multi_output.outputs.values()]
        if self._output is not None:
            return [self._output.node_id]
        return list(self._nodes)

    def _drop_unread_nodes(self) -> None:
        """Keep only the nodes some output reads, directly or through others.

        Not an optimization: Rust executes only what the outputs reach, so an
        unread node (``merge_pipe``'s own node, a branch no alias selects)
        changes no output, only the wire, the cache key and ``show_graph``.
        """
        reached: set[str] = set()
        stack = self._output_node_ids()
        while stack:
            node_id = stack.pop()
            if node_id not in reached:
                reached.add(node_id)
                stack.extend(self._nodes[node_id].upstream)
        self._nodes = {k: v for k, v in self._nodes.items() if k in reached}

    # --- CSE Optimization ---

    def _topological_ids(self) -> list[str]:
        """Every node, each after the nodes it reads."""
        order: list[str] = []
        seen: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in seen:
                return
            seen.add(node_id)
            for upstream_id in self._nodes[node_id].upstream:
                visit(upstream_id)
            order.append(node_id)

        for node_id in self._nodes:
            visit(node_id)
        return order

    def _optimize_common_subexpressions(self) -> None:
        """Compute every distinct value once, and give each linear run one node.

        **Value numbering.** Every op of every node is an *atom*, named by
        what it computes: the atom before it (the node's source, or the atom
        its upstream node ends on) and the op as the wire spells it, with its
        expression parameters as graph inputs and the nodes it reads as their
        atoms. A source atom is its column and source spec. Two atoms with one
        name are one atom, so equal work is shared wherever it was written:
        two decodes of a column, a prefix two of five pipelines have, ops
        below a shared node, an op whose operands were built twice, and a node
        with no ops (its upstream's atom).

        **Chain fusion.** The atoms the outputs need form a DAG. An atom ends a
        node where it must be addressable — an output, an operand some op
        reads by id — or where it is not read by exactly one atom; each run
        between such ends becomes one node, its ops replanned in order from
        the state entering the run. So a linear chain written as several
        nodes runs as one, and only a fork keeps a value apart.

        Build-time ids of replaced nodes keep resolving (``_renamed``): a
        plan's operands name nodes by build-time id, and the wire names each
        by the node that now computes it.
        """
        from polars_cv._lib import Plan
        from polars_cv.lazy import _BLOB_SOURCE

        table = self._slot_table()
        mark = "\x00atom:"  # an operand's atom on the wire; no field holds NUL

        index: dict[str, int] = {}
        prev: list[int | None] = []  # the atom each atom follows
        origin: list[tuple[str, int]] = []  # (node, op position); -1 for a source
        reads: list[list[int]] = []  # the atoms each op reads by id
        final: dict[str, int] = {}  # each node's last atom

        def intern(
            key: Any, after: int | None, at: tuple[str, int], read: list[int]
        ) -> int:
            name = json.dumps(key, sort_keys=True)
            if name not in index:
                index[name] = len(prev)
                prev.append(after)
                origin.append(at)
                reads.append(read)
            return index[name]

        def operands(value: Any) -> list[int]:
            if isinstance(value, str) and value.startswith(mark):
                return [int(value[len(mark) :])]
            if isinstance(value, dict):
                return [a for v in value.values() for a in operands(v)]
            if isinstance(value, list):
                return [a for v in value for a in operands(v)]
            return []

        old_order = list(self._nodes)
        source_atom: dict[str, int] = {}  # each root node's source atom
        for node_id in self._topological_ids():
            node = self._nodes[node_id]
            named = {n: f"{mark}{a}" for n, a in final.items()}
            spec = node.pipeline._to_spec_dict(table.index, named)
            if node.column is not None:
                key = ["source", table.index(node.column), spec["source"]]
                atom = intern(key, None, (node_id, -1), [])
                source_atom[node_id] = atom
            else:
                atom = final[node.upstream[0]]
            for position, op in enumerate(spec["ops"]):
                atom = intern([atom, op], atom, (node_id, position), operands(op))
            final[node_id] = atom

        # The atoms the outputs need, and which of them must end a node.
        outputs = {final[n] for n in self._output_node_ids()}
        needed: set[int] = set()
        stack = list(outputs)
        while stack:
            atom = stack.pop()
            if atom not in needed:
                needed.add(atom)
                stack.extend(reads[atom])
                before = prev[atom]
                if before is not None:
                    stack.append(before)
        readers: dict[int, list[int]] = {}
        for atom in sorted(needed):
            before = prev[atom]
            if before is not None:
                readers.setdefault(before, []).append(atom)
        operand_atoms = {r for atom in needed for r in reads[atom]}

        def ends_a_node(atom: int) -> bool:
            return (
                atom in outputs
                or atom in operand_atoms
                or len(readers.get(atom, [])) != 1
            )

        # Each run, after the runs it reads (its input, its ops' operands).
        run_of: dict[int, list[int]] = {}  # last atom -> run
        for atom in sorted(needed):
            before = prev[atom]
            if before is None or ends_a_node(before):
                run = [atom]
                while not ends_a_node(run[-1]):
                    run.append(readers[run[-1]][0])
                run_of[run[-1]] = run
        runs: list[list[int]] = []
        placed: set[int] = set()

        def place(last: int) -> None:
            if last in placed or last not in run_of:
                return
            placed.add(last)
            run = run_of[last]
            before = prev[run[0]]
            for dep in ([] if before is None else [before]) + [
                r for atom in run for r in reads[atom]
            ]:
                place(dep)
            runs.append(run)

        for last in run_of:
            place(last)

        old = self._nodes
        policies = {attr: self._graph_policy(attr) for attr in _POLICIES}
        node_of: dict[int, str] = {}  # a run's last atom -> its node
        nodes: dict[str, GraphNode] = {}
        for i, run in enumerate(runs):
            first_node, _ = origin[run[0]]
            # Derived through `_clone` (the one copy, so every setting carries
            # over), then given its own plan, expressions and node reads.
            pipeline = old[first_node].pipeline._clone()
            pipeline._exprs, pipeline._node_refs = [], []
            for attr, policy in policies.items():
                setattr(pipeline, attr, policy)
            template = old[first_node].pipeline
            before = prev[run[0]]
            source_json = template._plan.source_json()
            if before is None and source_json is not None:
                source = _rewired(json.loads(source_json), template, pipeline)
                pipeline._plan = Plan().with_source(json.dumps(source))
                ops, column = run[1:], old[first_node].column
                upstream = []
            elif before is not None:
                entering = self._state_after(old, origin, before)
                pipeline._plan = Plan().rebased(_BLOB_SOURCE, entering)
                ops, column = run, None
                upstream = [node_of[before]]
            else:
                msg = f"root node {first_node!r} has no source"
                raise ValueError(msg)
            for atom in ops:
                node_id, position = origin[atom]
                src = old[node_id].pipeline
                op = _rewired(json.loads(src._plan.ops_json()[position]), src, pipeline)
                pipeline._plan = pipeline._plan.push(json.dumps(op), src._refs())
                for ref in src._node_refs:
                    pipeline._read_node(ref)
                for r in reads[atom]:
                    if node_of[r] not in upstream:
                        upstream.append(node_of[r])
            # A node computing what one written node did keeps its id.
            same = [n for n in old_order if final[n] == run[-1]]
            node_id = same[0] if same else f"_cse_{i}"
            aliases = [old[n].alias for n in same if old[n].alias]
            nodes[node_id] = GraphNode(
                node_id=node_id,
                pipeline=pipeline,
                column=column,
                upstream=upstream,
                alias=aliases[0] if aliases else None,
            )
            node_of[run[-1]] = node_id

        # Root nodes first, in their columns' original order: the first column
        # is the plugin's first input, which names the output. (The wire is
        # not read in order; Rust orders nodes by `upstream`.)
        root_rank = {a: i for i, a in enumerate(dict.fromkeys(source_atom.values()))}
        first_atom = {node_of[run[-1]]: run[0] for run in runs}
        nodes = dict(
            sorted(
                nodes.items(),
                key=lambda item: root_rank.get(first_atom[item[0]], len(root_rank)),
            )
        )
        renamed = {n: node_of[a] for n, a in final.items() if a in node_of}
        self._renamed = {
            k: renamed.get(v, v) for k, v in self._renamed.items()
        } | renamed
        self._nodes = nodes
        self._alias_to_node = {
            alias: renamed[n]
            for alias, n in self._alias_to_node.items()
            if n in renamed
        }
        if self._output is not None:
            self._output.node_id = renamed[self._output.node_id]
        if self._multi_output is not None:
            self._multi_output.outputs = {
                alias: (renamed[n], fmt, params)
                for alias, (n, fmt, params) in self._multi_output.outputs.items()
            }

    @staticmethod
    def _state_after(
        nodes: dict[str, GraphNode], origin: list[tuple[str, int]], atom: int
    ) -> Any:
        """The planned state an atom produces."""
        node_id, position = origin[atom]
        return nodes[node_id].pipeline._plan.state_at(position + 1)

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

        self._check_roots()

        # One positional table for the plugin's inputs: root columns first,
        # then every expression parameter. Serialization reads the same table,
        # so positions and arguments cannot disagree.
        table = self._slot_table()

        # Unified graph execution handles both single and multi-output
        return _plugin.call(
            "vb_graph",
            args=table.columns,
            kwargs={"graph_json": self._to_json()},
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

    def _output_spec(
        self, node_id: str, fmt: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """One output on the wire: its node and its sink. Everything else about
        the output is planned by Rust from the graph itself."""
        return {"node": node_id, "sink": {"format": fmt, **params}}

    def check(self) -> None:
        """Refuse a graph the plugin would refuse, now: compile and plan it,
        and check every output's sink, through the plugin's own code
        (``_lib.check_graph``). Raises ``ValueError``."""
        from polars_cv._lib import check_graph

        check_graph(self._to_json())

    def _wire_ids(self) -> dict[str, str]:
        """Each node's id on the wire, keyed by its build-time id.

        Node ids are generated per build (``uuid4``), so the wire names each
        node by its position instead (ops that read a node included, through
        ``Plan.to_spec``): the same pipeline built twice then serializes to the
        same JSON, which is what lets polars' CSE merge the two plugin calls
        (it compares their kwargs) and the compiled-graph cache reuse one
        entry.
        """
        wire = {node_id: f"n{i}" for i, node_id in enumerate(self._nodes)}
        for old, now in self._renamed.items():
            wire.setdefault(old, wire[now])
        return wire

    def _to_dict(self) -> dict[str, Any]:
        if self._output is None and self._multi_output is None:
            raise ValueError("No output set")

        wire = self._wire_ids()

        # Build nodes dict
        table = self._slot_table()
        nodes_dict: dict[str, Any] = {}
        for node_id, node in self._nodes.items():
            # Get the pipeline's JSON representation without sink
            # We'll add sink info to the output specification
            node_spec = node.pipeline._to_spec_dict(table.index, wire)
            node_spec["upstream"] = [wire[u] for u in node.upstream]
            nodes_dict[wire[node_id]] = node_spec

        # Build unified outputs dict (always use "outputs" format)
        outputs_spec: dict[str, Any] = {}

        if self._multi_output is not None:
            # Multi-output mode
            for alias, (node_id, fmt, params) in self._multi_output.outputs.items():
                outputs_spec[alias] = self._output_spec(wire[node_id], fmt, params)
        else:
            # Single output mode - use "_output" as the key
            assert self._output is not None
            outputs_spec["_output"] = self._output_spec(
                wire[self._output.node_id], self._output.format, self._output.params
            )

        graph_spec = {
            # Wire-format version; the Rust side rejects versions newer than
            # it understands instead of misparsing them.
            "version": 1,
            "nodes": nodes_dict,
            "outputs": outputs_spec,
            "column_bindings": {
                wire[node_id]: table.index(node.column)
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
        for attr, key in _POLICIES.items():
            policy = self._graph_policy(attr)
            if policy != "raise":
                graph_spec[key] = policy

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

    def show_graph(self) -> pydot.Dot:
        """Build dot representation of graph."""
        return get_graphviz_out(self)  # ty: ignore[invalid-return-type]


def _rewired(value: Any, src: "Pipeline", dst: "Pipeline") -> Any:
    """A wire value of ``src``'s with each ``{"$slot": i}`` renumbered into
    ``dst``'s expression table (adding the expression there)."""
    if isinstance(value, dict):
        if value.keys() == {"$slot"}:
            return dst._slot(src._exprs[value["$slot"]])
        return {k: _rewired(v, src, dst) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewired(v, src, dst) for v in value]
    return value
