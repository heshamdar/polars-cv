"""Guards for surfaces deliberately removed (Phase 2).

Deletion needs a guard as much as construction does: nothing stops a later
change from reintroducing a parameter that does nothing, or a wire field
nothing reads. Each test here pins one removal, and says why it happened so
the next author does not "restore" it.

The Rust-side removals in the same phase (view-buffer's unreachable
pipeline-composition layer, the cost-reporting subsystem) are guarded by the
compiler instead — they cannot be referenced because they no longer exist.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import polars as pl
import pytest

import polars_cv
from polars_cv import Pipeline

from .conftest import plugin_required

# ---------------------------------------------------------------------------
# anti_alias: a parameter that was accepted, plumbed six layers deep, discarded
# ---------------------------------------------------------------------------


def test_rasterize_has_no_anti_alias_parameter() -> None:
    """``rasterize`` must not accept ``anti_alias``.

    It was threaded from the builder through the op spec, the JSON graph,
    ``resolve_rasterize_style``, ``GeometryOp::Rasterize`` and into
    ``geometry::rasterize``, whose signature named it ``_anti_alias`` and
    ignored it. Beyond being a documented no-op it was not free: it entered the
    op's identity, so two pipelines that behave identically hashed differently
    for CSE and compiled to separate graph-cache entries.

    A caller that passes it now gets a TypeError rather than silence.
    """
    params = inspect.signature(Pipeline.rasterize).parameters
    assert "anti_alias" not in params

    contour_pipe = (
        Pipeline().source("image_bytes").grayscale().threshold(128).extract_contours()
    )
    with pytest.raises(TypeError):
        contour_pipe.rasterize(width=8, height=8, anti_alias=True)  # type: ignore[call-arg]


def test_anti_alias_is_gone_from_the_type_stub() -> None:
    """The generated stub must not advertise the removed parameter."""
    stub = (Path(polars_cv.__file__).parent / "lazy.pyi").read_text()
    assert "anti_alias" not in stub


# ---------------------------------------------------------------------------
# shape_hints: emitted onto every node, read by nothing
# ---------------------------------------------------------------------------


@plugin_required
def test_graph_json_carries_no_shape_hints() -> None:
    """Node-level ``shape_hints`` must not be serialized.

    Nothing in ``polars-cv/src`` or ``view-buffer/src`` ever read the key. It
    was not merely wasted bytes: ``graph_json`` is the compiled-graph cache
    key, so two pipelines that execute identically but carry different hints
    occupied separate cache entries.

    Plan-time shape still crosses the boundary — as ``expected_shape`` on the
    *output* spec, which Rust does read.
    """
    pipe = (
        Pipeline()
        .source("image_bytes", dtype="u8")
        .assert_shape(height=100, width=200, channels=3)
        .resize(height=8, width=8)
    )
    graph = pl.col("img").cv.pipe(pipe).sink("png", return_expr=False)
    spec = json.loads(graph._to_json())

    for node_id, node in spec["nodes"].items():
        assert "shape_hints" not in node, (
            f"node {node_id} still serializes shape_hints, which nothing reads"
        )


# ---------------------------------------------------------------------------
# The graph wire format is closed in both directions
# ---------------------------------------------------------------------------


@plugin_required
def test_graph_node_rejects_unknown_fields() -> None:
    """An unrecognised key on a graph node must fail loudly.

    ``GraphNode`` was permissive, so a stale or misspelled field was silently
    dropped — which is how node-level ``shape_hints`` went on being emitted
    long after the last reader was removed. Everything Python sends is now
    declared on the Rust struct, including the fields only Python consumes.

    Note this closes the *node*, not the op. ``OpSpec`` carries its parameters
    through ``#[serde(flatten)]``, which serde documents as incompatible with
    ``deny_unknown_fields``; an unknown key there is indistinguishable from an
    op parameter by construction. Op names and their parameters are guarded
    instead by the registry-parity tests and ``resolve_op``'s catch-all.
    """
    import polars as pl

    df = pl.DataFrame({"img": [b""]})
    graph = (
        pl.col("img")
        .cv.pipe(Pipeline().source("image_bytes", dtype="u8").grayscale())
        .sink("png", return_expr=False)
    )
    spec = json.loads(graph._to_json())
    for node in spec["nodes"].values():
        node["definitely_not_a_field"] = 1
    tampered = json.dumps(spec)

    expr = pl.col("img").cv._plugin(  # type: ignore[attr-defined]
        "vb_graph",
        kwargs={"graph_json": tampered, "expr_column_names": []},
    )
    with pytest.raises(Exception) as excinfo:
        df.lazy().select(out=expr).collect()
    assert "definitely_not_a_field" in str(excinfo.value) or "unknown field" in str(
        excinfo.value
    )
