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
import re
from pathlib import Path

import polars as pl
import pytest

import polars_cv
from polars_cv import Pipeline

from ._discovery import requires_checkout, rust_sources
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


def test_the_benchmark_does_not_restate_a_deleted_concurrency_constant() -> None:
    """The remote benchmark must not mirror a fetch-concurrency constant.

    ``fetch::DEFAULT_CONCURRENCY`` was a per-*plugin-call* cap of 16, and
    ``benchmarks/scenarios/remote_source.py`` carried a ``PLUGIN_CONCURRENCY``
    copy of it so its report could say "16 files per wave". Both are gone: the
    bound is polars' process-wide ``POLARS_CONCURRENCY_BUDGET`` semaphore, taken
    one permit per request, because a per-call cap could not bound anything under
    the streaming engine — the engine invokes the plugin once per morsel across
    threads, so the real in-flight count was (morsels x 16).

    A hand-copied mirror of a deleted Rust constant reads as documentation while
    being wrong, which is worse than silence. Re-adding one here would also
    re-suggest that the number is per call.
    """
    from benchmarks.scenarios import remote_source

    # Positive half first: without it, an import failure or a rename would
    # satisfy the absence check below while proving nothing.
    assert hasattr(remote_source, "run_remote_source"), (
        "remote_source.run_remote_source is gone; this guard reads that module "
        "as the thing it is guarding and is now checking nothing"
    )

    assert not hasattr(remote_source, "PLUGIN_CONCURRENCY"), (
        "PLUGIN_CONCURRENCY is back in remote_source.py. Fetch concurrency is "
        "polars' POLARS_CONCURRENCY_BUDGET, not a per-call constant, and it has "
        "no FFI accessor to mirror faithfully."
    )

    source = Path(remote_source.__file__).read_text()
    assert "files per wave" not in source, (
        "the 'N files per wave' claim is back; the fan-out is bounded by a "
        "global semaphore, not by fixed-size waves"
    )


# ---------------------------------------------------------------------------
# match_detections: an evaluation protocol that had colonised the CV layers
# ---------------------------------------------------------------------------


def test_match_detections_is_gone_from_both_namespaces() -> None:
    """Neither geometry namespace may offer ``match_detections`` again.

    It performed greedy one-to-one assignment walked in descending *confidence*
    order -- the COCO/PASCAL evaluation protocol -- from inside a crate whose
    job is strided arrays. Its replacement, ``correspond``, runs the same rule
    over the same overlap matrix but takes a visit *order* instead of scores,
    so deciding that confidence is what orders the walk stays with the caller
    who knows what the contours mean.

    This also subsumes the old ``strategy=`` guard: that parameter was typed
    ``Literal["greedy"]``, rode the wire unread, and selected nothing. There is
    no method left for it to sit on.
    """
    from polars_cv.geometry.bbox import BBoxNamespace
    from polars_cv.geometry.contours import ContourNamespace

    for namespace in (ContourNamespace, BBoxNamespace):
        # Probe first: an absence assertion is equally true of a namespace that
        # lost every set-level accessor, or was renamed out from under this.
        assert hasattr(namespace, "correspond"), (
            f"probe is broken: {namespace.__name__} has no `correspond`"
        )
        assert not hasattr(namespace, "match_detections"), (
            f"{namespace.__name__}.match_detections is back; detection matching "
            "is `correspond` plus an order the caller chooses."
        )


def test_the_correspondence_result_carries_no_counts() -> None:
    """``n_tp`` / ``n_fp`` / ``n_fn`` / ``pred_idx`` must not come back.

    All four were computed per row, published in the result schema, documented
    on two accessors -- and read by nothing in the repository. The accessor
    docstrings even warned against reading them: per-row counts undercount
    false negatives over any population containing ground-truth-bearing images
    with no detections, which is precisely the population a detection metric
    evaluates. ``pred_idx`` was worse than unread, it was ``0..n``: a second
    copy of the position its own list was already indexed by.

    Counting pairings is a question about a population, so it belongs to
    whoever defines the population.
    """
    import polars_cv

    schema = polars_cv.CORRESPONDENCE_SCHEMA
    names = {f.name for f in schema.fields}
    assert names == {"right_idx", "overlap"}, (
        f"CORRESPONDENCE_SCHEMA publishes {sorted(names)}; it is the pairing "
        "and its overlap, nothing about how many pairings a population holds."
    )

    contour_rs = (
        Path(__file__).resolve().parents[1] / "src" / "contour.rs"
    ).read_text()
    assert "fn correspondence_fields()" in contour_rs, (
        "probe is broken: correspondence_fields not found in src/contour.rs"
    )
    for dead in ("n_tp", "n_fp", "n_fn", "pred_idx"):
        assert dead not in contour_rs, (
            f"src/contour.rs names {dead!r} again -- nothing read it, and the "
            "docs argued against reading it."
        )


@requires_checkout
def test_the_engine_carries_no_detection_vocabulary() -> None:
    """view-buffer must not learn what a detection is again.

    The greedy matcher lived in the tensor engine, returning a struct whose
    field names were true-positive counts. What the engine owns now is
    ``greedy_assign``: an assignment over an overlap matrix that knows nothing
    about detections, confidence or populations.
    """
    saw_replacement = False
    for path in rust_sources():
        text = path.read_text()
        if "pub fn greedy_assign(" in text:
            saw_replacement = True
        for dead in ("DetectionMatchResult", "match_from_matrix", "fn score_order("):
            assert dead not in text, (
                f"{path.name} declares {dead} again; the visit order and the "
                f"tallying belong above the CV layer, not inside it."
            )
    assert saw_replacement, (
        "probe is broken: no source declares `greedy_assign`, so the sweep "
        "above would pass over an engine with no assignment rule at all"
    )


def test_the_contour_kwargs_wire_field_is_gone() -> None:
    """``strategy`` must not come back as a Rust kwargs field either.

    Removing it from the Python signature alone would leave the wire field
    accepting a value from any other caller, which is how an unread field goes
    on being emitted for releases (see the ``shape_hints`` guard above).
    """
    contour_rs = (
        Path(__file__).resolve().parents[1] / "src" / "contour.rs"
    ).read_text()
    assert "pub struct ContourKwargs" in contour_rs, (
        "probe is broken: ContourKwargs not found in src/contour.rs"
    )
    assert "pub strategy" not in contour_rs, (
        "ContourKwargs declares 'strategy' again -- nothing reads it."
    )


# ---------------------------------------------------------------------------
# The rotation matrix: transliterated into Python, compared against a copy of
# itself
# ---------------------------------------------------------------------------


def test_the_planner_does_not_recompute_the_rotation_matrix() -> None:
    """``pipeline.py`` must read the rotate matrix from Rust, never derive it.

    ``AffineParams::from_rotation`` is the authority, and it is what an
    *unfused* rotate executes through. The planner's affine fusion used to
    transliterate it line for line — same variable names, same matrix layout —
    so which implementation produced a user's rotation depended on whether a
    neighbouring op happened to be affine-fusible. The two had already drifted
    (Python normalised ``angle % 360``, Rust did not; Python's ``round`` is
    half-to-even, Rust's is half-away-from-zero), and nothing compared them:
    the test that looked like a cross-check compared Python against a *third*
    copy of the same arithmetic living in ``test_affine_builder.py``.

    A pixel-level test cannot replace this. It pins the values for the angles
    it happens to sweep, whereas the property is that there is only one
    implementation to disagree with.
    """
    source = (Path(polars_cv.__file__).resolve().parent / "pipeline.py").read_text()

    fusion_start = source.index("def _try_convert_rotate_to_affine")
    fusion_end = source.index("def _to_spec_dict")
    fusion = source[fusion_start:fusion_end]
    assert "rotate_affine_params" in fusion, (
        "affine fusion no longer calls rotate_affine_params -- if it derives "
        "the matrix itself again, a fused rotate and an unfused one can "
        "silently disagree."
    )

    # The trig that builds a rotation matrix. `_rotation_matrix` (used by
    # `rotate_and_scale`) legitimately keeps its own, because it must accept
    # `pl.Expr` operands the engine cannot evaluate at plan time -- so scope
    # this to the fusion helper rather than the whole module.
    #
    # Matched on the bare names as well as the `math.` attribute form: a
    # `from math import cos, sin` inside the helper reintroduces exactly the
    # second implementation this rejects, and an attribute-only scan reads
    # green through it.
    tokens = ("cos", "sin", "radians", "atan2", "hypot")
    offenders = sorted(
        token
        for token in tokens
        if re.search(rf"(?<![\w.]){token}\s*\(", fusion) or f"math.{token}" in fusion
    )
    assert not offenders, (
        f"affine fusion computes {offenders} again. The rotation matrix has "
        f"one authority (AffineParams::from_rotation, via the "
        f"rotate_affine_params FFI); a second one is what this guard exists "
        f"to reject."
    )


# ---------------------------------------------------------------------------
# DomainOp: a public trait nothing implemented, documented as if it were live
# ---------------------------------------------------------------------------


@requires_checkout
def test_domain_op_is_gone() -> None:
    """`DomainOp` must not come back.

    It was declared in `ops/traits.rs`, re-exported from `ops/mod.rs`, and
    described in `view-buffer/AGENTS.md` as a live part of the op contract —
    with a doc example implementing it for `ExtractContoursOp`, a type that
    does not exist. Nothing in either crate implemented it. Domain dispatch
    actually lives on `GraphStep` in the plugin, which is where
    `view-buffer/AGENTS.md` says graph-level concerns belong.

    It was also the only reason `ops/traits.rs` imported `NodeOutput` — a graph
    concept reaching into the engine's trait module — so deleting it closed
    that too. A trait nothing implements is not free: it reads as coverage, and
    every future `Domain` change has to be reconciled against it.
    """
    for path in rust_sources():
        text = path.read_text()
        assert "trait DomainOp" not in text, (
            f"{path.name} declares DomainOp again; domain dispatch belongs to "
            f"GraphStep in the plugin"
        )
        assert "DomainOp" not in text, f"{path.name} still references DomainOp"


# ---------------------------------------------------------------------------
# Eager FROC/LROC result API: replaced by expression-valued functions
# ---------------------------------------------------------------------------


def test_eager_froc_lroc_result_api_is_gone() -> None:
    """The eager FROC/LROC AUC surface must not come back.

    ``froc_curve``/``lroc_curve`` returned ``FROCResult``/``LROCResult`` whose
    ``.auc()`` reduced a curve to a Python float through the eager integrals in
    ``_auc.py``. That was a second implementation of the integral the
    expression path now owns (``froc_auc``/``lroc_auc`` +
    ``metrics._auc_expr``), so a lazy plan could not carry an AUC and grouping
    meant a Python loop. Callers use ``froc_auc(table).collect().item()`` (or
    ``froc_sensitivity_at_fp`` / ``froc_summary_table`` for the curve helpers,
    ``bootstrap_froc_auc`` / ``bootstrap_lroc_auc`` for CIs).

    The FROC/LROC-only Mann-Whitney helpers went with them:
    ``mann_whitney_u_auc`` / ``detection_level_mann_whitney`` are now
    ``metrics._auc_expr.mann_whitney_auc_expr``.
    """
    import polars_cv.metrics as m

    removed = [
        "froc_curve",
        "lroc_curve",
        "FROCResult",
        "LROCResult",
    ]
    for name in removed:
        assert not hasattr(m, name), f"polars_cv.metrics re-exports removed {name!r}"
        assert not hasattr(polars_cv, name), f"polars_cv re-exports removed {name!r}"

    from polars_cv.metrics import _auc

    for name in ("mann_whitney_u_auc", "detection_level_mann_whitney"):
        assert not hasattr(_auc, name), f"_auc still defines removed {name!r}"

    # The replacements are present.
    for name in ("froc_auc", "lroc_auc", "froc_sensitivity_at_fp"):
        assert hasattr(m, name), f"replacement {name!r} missing"


def test_conflicting_weight_guard_is_gone() -> None:
    """The eager conflicting-weight guard was removed for pure-lazy streaming.

    ``_froc._raise_on_conflicting_weights`` collected ``image_metadata`` at build
    time to fail loudly on disagreeing duplicate weights. That broke the lazy
    plan (a collect before the caller asked for one), so it was replaced by the
    order-independent ``weight_agg`` policy (``resolve_key_weights``): conflicting
    weights now resolve, they do not raise. Do not reintroduce a build-time guard.
    """
    from polars_cv.metrics import DetectionTable, froc_curve_lazy
    from polars_cv.metrics._metrics import _froc

    assert not hasattr(_froc, "_raise_on_conflicting_weights"), (
        "_raise_on_conflicting_weights is back — it forces an eager collect"
    )

    det = pl.DataFrame(
        {
            "image_id": ["s", "s"],
            "class_id": ["__all__", "__all__"],
            "score": [0.8, 0.7],
            "is_tp": [True, False],
            "gt_idx": pl.Series([0, None], dtype=pl.UInt32),
            "iou": [0.6, 0.0],
            "det_idx": pl.Series([0, 1], dtype=pl.UInt32),
        }
    )
    meta = pl.DataFrame(
        {
            "image_id": ["s", "s"],
            "class_id": ["__all__", "__all__"],
            "n_gts": [1, 1],
            "weight": [1.0, 5.0],  # conflicting
            "gt_label": [True, True],
        }
    )
    table = DetectionTable.from_matched(det, meta)
    # No raise, at build or at collect, under any policy.
    for agg in ("first", "min", "max", "mean", "sum"):
        froc_curve_lazy(table, weight_agg=agg).collect()


def test_bootstrap_reconstruct_hook_is_gone() -> None:
    """``MetricResult._reconstruct`` (and the PR override) must not come back.

    ``bootstrap_ci`` used to rebuild a whole result and collect once *per
    replicate per metric* through ``_reconstruct``. It is now vectorized: one
    lazy resample feeds ``_bootstrap_grouped``, which computes every replicate's
    metric grouped by ``bootstrap_id`` in a single streaming plan. The
    per-replicate reconstruct loop — the eager path — is deleted, so nothing
    should reintroduce ``_reconstruct``; the vectorized hook is the way in.
    """
    from polars_cv.metrics._metrics._precision_recall import PrecisionRecallResult
    from polars_cv.metrics._result import MetricResult

    assert not hasattr(MetricResult, "_reconstruct"), (
        "_reconstruct is back — bootstrap_ci must vectorize via _bootstrap_grouped, "
        "not a per-replicate reconstruct loop"
    )
    assert not hasattr(PrecisionRecallResult, "_reconstruct")
    # The vectorized hook is present in its place.
    assert hasattr(MetricResult, "_bootstrap_grouped")
    assert "_bootstrap_grouped" in vars(PrecisionRecallResult)
