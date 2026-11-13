"""Draw a graph from the plugin's description of it.

The description (``_lib.describe_graph``) is the one ``PipelineGraph.explain``
prints: the nodes in execution order, each op with its planned state and
whether it runs fused, the outputs with their planned dtypes. This module only
lays it out, so the drawing and the text cannot disagree; nothing here reads
the Python graph or the wire.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import graphviz

    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

if TYPE_CHECKING:
    import graphviz

#: The one node every output edge ends at.
SINK = "SINK"


def _label(lines: list[str]) -> str:
    """Left-justified lines, each escaped so its text reads literally."""
    return "".join(graphviz.escape(line) + r"\l" for line in lines)


def draw(description: dict[str, Any]) -> graphviz.Digraph:
    """The graph *description* describes, as a ``graphviz.Digraph``.

    One box per node, its lines those ``explain`` prints: led by its wire id
    (the name an operand reads it by) and what it reads, then each op with its planned state (``|`` marks
    the fused ones, as in ``explain``); a root is bold. A solid edge is a
    node's input, a dashed one an op reading another node. Every output is an
    edge into the ``SINK`` box, which lists each with its sink and dtype.
    """
    if not _VIZ_AVAILABLE:
        msg = (
            "Graph drawing requires the 'graphviz' package (and Graphviz's "
            "`dot` program to render it). Install it with: "
            "pip install 'polars-cv[viz]'"
        )
        raise ImportError(msg)

    dot = graphviz.Digraph(
        "polars_cv", node_attr={"shape": "box", "fontname": "monospace"}
    )
    for node in description["nodes"]:
        source = node["input"].get("decode")
        reads = (
            f"decode {source['column']} as {source['source']}"
            if source is not None
            else f"reads {node['input']['node']}"
        )
        lines = [f"{node['id']}: {reads}  -> {node['state']}"]
        lines += [
            f"{'|' if op['fused'] else ' '} {op['label']}  -> {op['state']}"
            for op in node["ops"]
        ]
        dot.node(node["id"], _label(lines), style="bold" if source else "solid")
        if source is None:
            dot.edge(node["input"]["node"], node["id"])
        for op in node["ops"]:
            for other in op["reads"]:
                name = op["label"].split("(", 1)[0]
                dot.edge(other, node["id"], label=graphviz.escape(name), style="dashed")

    lines = ["SINK"]
    for out in description["outputs"]:
        name = out["name"] or "output"
        dtype = out["dtype"] or "set by the input column"
        where = (
            f"{out['node']} as {out['sink']}"
            if out["node"]
            else "each failing row's message"
        )
        lines.append(f"{name}: {where} :: {dtype}")
    dot.node(SINK, _label(lines), style="bold")
    for out in description["outputs"]:
        if out["node"]:
            dot.edge(out["node"], SINK, label=graphviz.escape(out["name"] or ""))
    return dot
