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

#: Every test here is a structural guard: it checks the *shape* of the codebase
#: -- registries, authorities, removed surfaces, documented vocabularies --
#: rather than the numerical behaviour of a pipeline. `-m structural` is the
#: lane pre-commit runs; see `tests/AGENTS.md`. Note that the lane as a whole
#: does need the compiled extension: many structural facts are only observable
#: through the FFI, and those tests fail rather than skip without it.
pytestmark = pytest.mark.structural

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
    """The generated stub must not advertise the removed parameter.

    ``"anti_alias" not in stub`` is also true of an empty stub, a stub that
    lost ``rasterize`` altogether, and a stub whose path this test no longer
    finds — three ways to pass while checking nothing. Confirm the file is the
    populated stub it claims to be first.
    """
    stub = (Path(polars_cv.__file__).parent / "lazy.pyi").read_text()
    assert "def rasterize" in stub, (
        "lazy.pyi does not declare rasterize, so the assertion below holds "
        "vacuously. Regenerate with scripts/gen_lazy_stub.py."
    )
    assert "anti_alias" not in stub


# ---------------------------------------------------------------------------
# shape_hints: emitted onto every node, read by nothing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# assert_shape(batch=...): recorded, never read, never sent
# ---------------------------------------------------------------------------


def test_assert_shape_has_no_batch_parameter() -> None:
    """``assert_shape(batch=...)`` must raise, not be silently recorded.

    It reached ``ShapeHints.batch`` and stopped there. Nothing read it: not
    ``has_all_dims``, not ``expected_shape``, not ``_current_input_dims``, and
    not Rust — the node-level ``shape_hints`` wire field it was serialized into
    had already lost its last reader, and then the field itself. So a caller who
    declared a batch dimension got exactly the same plan as one who did not,
    while ``ShapeHints.to_dict`` went on emitting it.

    The hints are positional and track three dimensions; a fourth had no
    position to occupy. ``assert_shape(dims=[...])`` is the spelling for a shape
    the H/W/C names do not describe, and it rejects a rank the planner cannot
    track rather than accepting it and dropping the extra dimensions.
    """
    with pytest.raises(TypeError, match="batch"):
        Pipeline().source("image_bytes").assert_shape(batch=4)

    from polars_cv._types import ShapeHints

    assert not hasattr(ShapeHints(), "batch"), (
        "ShapeHints.batch is back; it was removed because nothing read it"
    )
    assert not hasattr(ShapeHints, "to_dict"), (
        "ShapeHints.to_dict is back; it serialized the node-level `shape_hints` "
        "wire field, which no longer exists"
    )


# ---------------------------------------------------------------------------
# source("contour", dtype=...): an assertion the decode never read
# ---------------------------------------------------------------------------


def test_contour_source_rejects_a_dtype_assertion() -> None:
    """``source("contour", dtype=...)`` must raise, not be quietly dropped.

    The parameter reached ``SourceSpec.dtype`` and stopped there: the contour
    decode rasterizes, and ``rasterize`` fixes its output at u8
    (``OutputDTypeRule::Fixed(U8)``). So an asserted ``"f32"`` bought a u8
    column — and for the other sources a dtype assertion is exactly what makes
    a typed ``list``/``array`` sink plannable, which is the reading a caller
    would bring to it.

    The dtype is now published from the rasterize contract instead, so the
    parameter has nothing left to say. ``.cast(...)`` after the source is the
    supported way to change it, and it runs through the real cast op.
    """
    with pytest.raises(ValueError, match="dtype does not apply"):
        Pipeline().source("contour", width=8, height=8, dtype="f32")
    with pytest.raises(ValueError, match="use .cast"):
        Pipeline().source("contour", width=8, height=8, dtype="u8")


# ---------------------------------------------------------------------------
# Python-side sink spec classes: unreachable, and wrong where they disagreed
# ---------------------------------------------------------------------------


def test_the_python_sink_spec_dataclasses_are_gone() -> None:
    """``_types`` must not carry ``SinkSpec``/``OutputSpec``/``MultiSinkSpec``.

    Nothing in the package or the tests referenced them: a sink is built from
    the raw ``.sink()`` kwargs in ``_graph.py`` and serialized straight into the
    graph JSON, and the live specs are the Rust ones in ``pipeline.rs``.

    They were not inert. Each held a copy of which sink parameters apply to
    which format (``if format == JPEG or WEBP: result["quality"]``), and that
    copy was wrong in the same way the docstrings were — the WebP encoder takes
    no quality. `SINK_PARAM_APPLIES` is the one place that fact now lives.
    """
    import polars_cv._types as types_module

    for name in ("SinkSpec", "OutputSpec", "MultiSinkSpec"):
        assert not hasattr(types_module, name), (
            f"{name} was deleted as unreachable; the sink's wire format is "
            f"Rust's SinkSpec and its parameter table is SINK_PARAM_APPLIES"
        )


# ---------------------------------------------------------------------------
# OutputDType: a partial second dtype table whose one distinct value was a
# synonym for the default
# ---------------------------------------------------------------------------


def test_the_output_dtype_strategy_enum_is_gone() -> None:
    """``_types`` must not carry ``OutputDType``.

    It listed ``f32``/``f64``/``u8`` — a partial second copy of the dtype
    spellings ``dtype_table!`` already owns — plus one value that was not a
    dtype: ``"preserve"``.

    ``"preserve"`` documented itself, in the enum and in ``clamp``'s docstring,
    as "keep input dtype (floats preserved, integers -> f32)". That is
    character for character what ``OutputDTypeRule::PromoteToFloat`` does, i.e.
    what passing nothing already did — so it was a synonym for the default, not
    an unimplemented feature. ``normalize`` had to reject it by hand for that
    reason, and ``scale``/``clamp`` accepted it into the op's identity and
    dropped it.

    The behaviour the word suggests (u8 in, u8 out) is ``preserve_dtype=True``,
    which is wired. ``out_dtype`` now validates against ``DType``, so every
    dtype is requestable and there is one table of dtype names.
    """
    import polars_cv._types as types_module

    assert not hasattr(types_module, "OutputDType"), (
        "OutputDType was deleted: out_dtype validates against DType, the single "
        "dtype-name authority, and 'preserve' was a synonym for the default "
        "(preserve_dtype=True is the input-dtype-preserving behaviour)"
    )


@pytest.mark.parametrize("op", ["scale", "clamp", "normalize"])
def test_out_dtype_rejects_the_preserve_strategy(op: str) -> None:
    """``out_dtype="preserve"`` must fail: it is not a dtype.

    Previously this was accepted and silently ignored by ``scale``/``clamp``
    (it never reached Rust, where ``parse_dtype`` has no ``"preserve"``), and
    rejected by a bespoke check in ``normalize``. One authority, one answer.
    """
    pipe = Pipeline().source("list", dtype="u8")
    build = {
        "scale": lambda: pipe.scale(2.0, out_dtype="preserve"),
        "clamp": lambda: pipe.clamp(0.0, 1.0, out_dtype="preserve"),
        "normalize": lambda: pipe.normalize(method="minmax", out_dtype="preserve"),
    }[op]
    with pytest.raises(ValueError, match="preserve"):
        build()


@pytest.mark.parametrize("op", ["scale", "clamp"])
def test_out_dtype_does_not_reach_the_op_params(op: str) -> None:
    """``scale``/``clamp`` must not carry ``out_dtype`` on the wire.

    They have no configurable output dtype — their rule is ``PromoteToFloat``,
    which ``output_dtype_for`` does not honour an override for, and neither
    ``resolve_op`` arm ever read the parameter. It rode in the op's identity
    (so two pipelines that behave identically hashed differently for CSE) and
    was discarded at execution.

    The request is now lowered to a trailing ``cast`` op, which is the
    mechanism that actually performs it.
    """
    pipe = Pipeline().source("list", dtype="u8")
    built = {
        "scale": lambda: pipe.scale(2.0, out_dtype="u8"),
        "clamp": lambda: pipe.clamp(0.0, 1.0, out_dtype="u8"),
    }[op]()
    assert [spec.op for spec in built._ops] == [op, "cast"], (
        f"{op}(out_dtype=...) must lower to the op plus a cast, got "
        f"{[spec.op for spec in built._ops]}"
    )
    assert "out_dtype" not in built._ops[0].params, (
        f"{op} must not serialize out_dtype: no resolve_op arm reads it"
    )
