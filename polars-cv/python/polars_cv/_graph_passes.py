"""The graph-scope optimisation passes, which Python declares and runs.

A graph-scope pass rewrites the whole :class:`~polars_cv._graph.PipelineGraph`
— it compares and merges nodes — and reads the expressions' identities
(``meta.eq``), which only Python holds. So unlike the node-scope passes
(Rust's ``LogicalPass``, run through ``Plan.run_pass``) it is declared here, and
``scripts/gen_ops.py`` puts it at the head of the generated ``PASS_CATALOG``
with tier ``"graph"``, giving it its :class:`~polars_cv._optimize.OptFlags`
field. ``PipelineGraph`` maps each name to its implementation and refuses to
import with one missing.

Dependency-free: the generator loads this file by path.
"""

from __future__ import annotations

#: Every graph-scope pass, in the order they apply: (name, tier, summary).
GRAPH_PASSES: tuple[tuple[str, str, str], ...] = (
    (
        "common_subexpression_elimination",
        "graph",
        "Compute each distinct value once — the same op on the same inputs, "
        "however the pipelines were written — and fuse each linear run of ops "
        "no other node reads into one node.",
    ),
)
