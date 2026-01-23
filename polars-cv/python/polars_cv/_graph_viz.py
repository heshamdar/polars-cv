from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import networkx as nx
from graphviz import Source
from networkx.drawing.nx_pydot import to_pydot

if TYPE_CHECKING:
    from polars_cv._graph import PipelineGraph


class NodeKind(str, Enum):
    SOURCE = "source"
    COMPUTE = "compute"
    OUTPUT = "output"


@dataclass(frozen=True)
class BaseNode:
    node_id: str
    kind: NodeKind


@dataclass(frozen=True)
class ComputeNode(BaseNode):
    alias: Optional[str]
    ops: List[Dict[str, Any]]
    domain: str
    dtype: str
    upstream: List[str]
    source_format: Optional[str] = None  # only set for sources


@dataclass(frozen=True)
class OutputNode(BaseNode):
    outputs: Dict[str, Dict[str, Any]]  # alias -> sink spec


@dataclass
class LogicalGraph:
    nodes: Dict[str, ComputeNode] = field(default_factory=dict)
    output: Optional[OutputNode] = None


OUTPUT_NODE_ID = "__output__"


def parse_logical_graph(spec: Dict[str, Any]) -> LogicalGraph:
    graph = LogicalGraph()

    # Parse compute + source nodes
    for node_id, payload in spec["nodes"].items():
        upstream = payload.get("upstream", [])
        is_source = len(upstream) == 0

        graph.nodes[node_id] = ComputeNode(
            node_id=node_id,
            kind=NodeKind.SOURCE if is_source else NodeKind.COMPUTE,
            alias=payload.get("alias"),
            ops=payload.get("ops", []),
            domain=payload["domain"],
            dtype=payload["output_dtype"],
            upstream=upstream,
            source_format=payload["source"]["format"] if is_source else None,
        )

    # Parse outputs as a single container
    graph.output = OutputNode(
        node_id=OUTPUT_NODE_ID,
        kind=NodeKind.OUTPUT,
        outputs=spec["outputs"],
    )

    return graph


def build_dag(graph: LogicalGraph) -> nx.DiGraph:
    g = nx.DiGraph()

    # Add compute/source nodes
    for node in graph.nodes.values():
        g.add_node(node.node_id, payload=node)

        for upstream in node.upstream:
            g.add_edge(upstream, node.node_id)

    # Add output container
    output_node = graph.output
    assert output_node is not None

    g.add_node(output_node.node_id, payload=output_node)

    for alias, out in output_node.outputs.items():
        g.add_edge(out["node"], output_node.node_id, alias=alias)

    # remove terminal node that isn't output
    # this occurs when we use a merge_pipe
    spurious_sink_nodes = [
        node
        for node, out_degree in g.out_degree()
        if (out_degree == 0) and (node != OUTPUT_NODE_ID)
    ]
    g.remove_nodes_from(spurious_sink_nodes)

    return g


## Visualization


def style_node(payload: BaseNode) -> Dict[str, Any]:
    if payload.kind == NodeKind.SOURCE:
        assert isinstance(payload, ComputeNode)
        return {
            "shape": "box",
            "style": "bold",
            "label": (
                "SOURCE\n"
                f"{payload.alias}\n"
                f"source: {payload.source_format}\n"
                f"output_type: {payload.dtype}"
            ),
        }

    if payload.kind == NodeKind.COMPUTE:
        assert isinstance(payload, ComputeNode)

        ops = ""
        for op in payload.ops:
            op_dict = {
                k: v.get("value", "Expr") for k, v in op.items() if k != "op"
            }  # if not a value must be an expression
            ops_string = op["op"] + str(op_dict)
            ops += ops_string + "\n"

        return {
            "shape": "box",
            "label": (f"{payload.alias}\n{ops}output_type: {payload.dtype}"),
        }

    if payload.kind == NodeKind.OUTPUT:
        assert isinstance(payload, OutputNode)
        lines = ["SINK"]
        for alias, spec in payload.outputs.items():
            lines.append(f"• {alias} ({spec['sink']['format']})")
        return {
            "shape": "box",
            "style": "bold",
            "label": "\n".join(lines),
        }

    raise ValueError(payload)


def visualize_dag(g: nx.DiGraph):
    for node_id, data in g.nodes(data=True):
        payload = data["payload"]
        data.update(style_node(payload))

    return to_pydot(g)


def get_graphviz_out(graph: PipelineGraph) -> Source:
    """
    Returns a Graphviz object for the given pipeline graph.
    """
    spec = graph._to_dict()

    logical = parse_logical_graph(spec)
    dag = build_dag(logical)
    dot = visualize_dag(dag)

    graphviz_object = Source(dot.to_string())

    return graphviz_object
