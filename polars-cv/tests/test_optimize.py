"""Unit tests for the optimization control surface (``polars_cv._optimize``).

These need no compiled plugin — they pin the pass registry, ``OptFlags``, and
environment parsing. The execution-level differential-equivalence guard (same
output regardless of flags) lives in ``test_optimize_equivalence.py``.
"""

from __future__ import annotations

import dataclasses

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._graph import PipelineGraph
from polars_cv._optimize import (
    LOGICAL_PASSES,
    OPT_ENV_VAR,
    PASS_NAMES,
    OptFlags,
    PassSpec,
    resolve_opt_flags,
)
from tests.conftest import plugin_required


def _two_rotate_pipe() -> Pipeline:
    """A pipeline with a fusible run: two adjacent static rotates.

    The leading ``resize`` establishes the plan-time H/W the rotate→affine
    conversion needs; without known input dims a rotate cannot convert and the
    run does not fuse. Fusion therefore collapses the two rotates into one
    ``warp_affine`` while ``resize`` stays, giving ``[resize, warp_affine]``.
    """
    return (
        Pipeline()
        .source("image_bytes")
        .resize(height=64, width=64)
        .rotate(30.0)
        .rotate(15.0)
    )


def _graph_of(pipe: Pipeline) -> PipelineGraph:
    graph = PipelineGraph()
    graph.add_node(node_id="n", pipeline=pipe, column=pl.col("img"))
    return graph


def _node_ops(graph: PipelineGraph) -> list[str]:
    return [op.op for op in graph._nodes["n"].pipeline._ops]


class TestRegistry:
    def test_pass_names_are_unique(self) -> None:
        assert len(PASS_NAMES) == len(set(PASS_NAMES))

    def test_every_pass_declares_its_equivalence_class(self) -> None:
        # bit_exact is required (no default) so a new pass cannot omit it.
        for spec in LOGICAL_PASSES:
            assert isinstance(spec, PassSpec)
            assert isinstance(spec.bit_exact, bool)
            assert spec.summary

    def test_flags_match_registry_both_directions(self) -> None:
        """Every registered pass has an OptFlags field and vice versa.

        The canonical-path guard for this tier: a pass without a switch, or a
        switch without a pass, fails here rather than silently diverging.
        """
        flag_fields = {f.name for f in dataclasses.fields(OptFlags)}
        assert flag_fields == set(PASS_NAMES)

    def test_pass_handlers_cover_every_pass(self) -> None:
        """Every registered pass has an ``optimize()`` handler and vice versa.

        The applicator map in ``PipelineGraph.optimize`` is the third place a
        pass must appear (with ``LOGICAL_PASSES`` and ``OptFlags``); this pins
        it to ``PASS_NAMES`` so a pass cannot be registered without being
        runnable, nor a handler survive a removed pass.
        """
        assert set(PipelineGraph._pass_handlers()) == set(PASS_NAMES)

    def test_every_flag_field_is_boolean_defaulting_on(self) -> None:
        defaults = OptFlags()
        for name in PASS_NAMES:
            assert getattr(defaults, name) is True


class TestShorthands:
    def test_all_enables_everything(self) -> None:
        flags = OptFlags.all()
        assert all(flags.enabled(name) for name in PASS_NAMES)

    def test_none_disables_everything(self) -> None:
        flags = OptFlags.none()
        assert not any(flags.enabled(name) for name in PASS_NAMES)

    def test_default_is_all_on(self) -> None:
        assert OptFlags() == OptFlags.all()

    def test_enabled_rejects_unknown_pass(self) -> None:
        with pytest.raises(KeyError, match="Unknown optimization pass"):
            OptFlags().enabled("no_such_pass")

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            OptFlags().affine_fusion = False  # type: ignore[misc]


class TestParse:
    def test_all(self) -> None:
        assert OptFlags.parse("all") == OptFlags.all()

    def test_none(self) -> None:
        assert OptFlags.parse("none") == OptFlags.none()

    def test_bare_name_turns_only_that_on(self) -> None:
        flags = OptFlags.parse("affine_fusion")
        assert flags.enabled("affine_fusion")
        assert not flags.enabled("common_subexpression_elimination")

    def test_all_minus_one(self) -> None:
        flags = OptFlags.parse("all,-affine_fusion")
        assert not flags.enabled("affine_fusion")
        assert flags.enabled("common_subexpression_elimination")

    def test_whitespace_is_tolerated(self) -> None:
        assert OptFlags.parse("  all , -affine_fusion ") == OptFlags.parse(
            "all,-affine_fusion"
        )

    def test_unknown_pass_raises_not_ignored(self) -> None:
        with pytest.raises(ValueError, match="Unknown optimization pass"):
            OptFlags.parse("all,-bogus")


class TestFromEnv:
    def test_unset_is_all_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(OPT_ENV_VAR, raising=False)
        assert OptFlags.from_env() == OptFlags.all()

    def test_blank_is_all_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(OPT_ENV_VAR, "   ")
        assert OptFlags.from_env() == OptFlags.all()

    def test_none_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(OPT_ENV_VAR, "none")
        assert OptFlags.from_env() == OptFlags.none()

    def test_subset_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(OPT_ENV_VAR, "all,-affine_fusion")
        flags = OptFlags.from_env()
        assert flags.enabled("common_subexpression_elimination")
        assert not flags.enabled("affine_fusion")


class TestResolve:
    def test_none_defers_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(OPT_ENV_VAR, "none")
        assert resolve_opt_flags(None) == OptFlags.none()

    def test_true_is_all(self) -> None:
        assert resolve_opt_flags(True) == OptFlags.all()

    def test_false_is_none(self) -> None:
        assert resolve_opt_flags(False) == OptFlags.none()

    def test_optflags_passes_through(self) -> None:
        flags = OptFlags(affine_fusion=False)
        assert resolve_opt_flags(flags) is flags

    def test_bad_type_raises(self) -> None:
        with pytest.raises(TypeError, match="opt_flags must be"):
            resolve_opt_flags("all")  # type: ignore[arg-type]


class TestStaging:
    """The staging contract: optimization only happens in the optimize phase.

    Construction builds the logical graph and serialization emits it verbatim;
    neither fuses or CSEs. Toggling passes changes only the physical graph.
    """

    def test_serialization_is_verbatim(self) -> None:
        # _to_spec_dict used to fuse affine runs; now it serializes as written.
        spec = _two_rotate_pipe()._to_spec_dict()
        assert [op["op"] for op in spec["ops"]] == ["resize", "rotate", "rotate"]

    def test_optimize_none_changes_nothing(self) -> None:
        graph = _graph_of(_two_rotate_pipe())
        graph.optimize(OptFlags.none())
        assert _node_ops(graph) == ["resize", "rotate", "rotate"]

    def test_affine_fusion_flag_gates_the_pass(self) -> None:
        off = _graph_of(_two_rotate_pipe())
        off.optimize(OptFlags(affine_fusion=False))
        assert _node_ops(off) == ["resize", "rotate", "rotate"]

        on = _graph_of(_two_rotate_pipe())
        on.optimize(OptFlags(affine_fusion=True))
        assert _node_ops(on) == ["resize", "warp_affine"]

    def test_optimize_is_idempotent(self) -> None:
        graph = _graph_of(_two_rotate_pipe())
        graph.optimize(OptFlags.all())
        first = _node_ops(graph)
        graph.optimize(OptFlags.all())
        assert _node_ops(graph) == first == ["resize", "warp_affine"]


class TestImmutability:
    """Optimization must not mutate the caller's Pipeline (it is immutable)."""

    def test_optimize_does_not_touch_the_source_pipeline(self) -> None:
        pipe = _two_rotate_pipe()
        graph = _graph_of(pipe)
        graph.optimize(OptFlags.all())
        # The graph fused its own copy; the caller's pipeline is untouched.
        assert [op.op for op in pipe._ops] == ["resize", "rotate", "rotate"]
        assert _node_ops(graph) == ["resize", "warp_affine"]


class TestExplain:
    """Pipeline.explain surfaces the logical vs physical op chain."""

    def test_logical_shows_unfused(self) -> None:
        text = _two_rotate_pipe().explain(optimized=False)
        assert text.count("rotate") == 2
        assert "warp_affine" not in text

    def test_optimized_shows_fused(self) -> None:
        text = _two_rotate_pipe().explain(optimized=True)
        assert "warp_affine" in text
        assert "rotate(" not in text

    def test_optimized_respects_flags(self) -> None:
        text = _two_rotate_pipe().explain(opt_flags=OptFlags(affine_fusion=False))
        assert text.count("rotate") == 2

    def test_explain_does_not_mutate(self) -> None:
        pipe = _two_rotate_pipe()
        pipe.explain(optimized=True)
        assert [op.op for op in pipe._ops] == ["resize", "rotate", "rotate"]


def _shape_ref() -> "pl.Expr":
    """A shape sub-pipeline expression carrying a fusible affine run.

    ``resize`` gives the two static rotates their plan-time H/W (so the
    rotate→affine conversion fires), making ``[resize, rotate, rotate]`` a
    genuine fusion candidate — the same fixture shape as ``_two_rotate_pipe``,
    but delivered as a ``shape=`` reference for a contour source.
    """
    return pl.col("img").cv.pipe(
        Pipeline()
        .source("image_bytes")
        .resize(width=40, height=40)
        .rotate(30.0)
        .rotate(15.0)
    )


def _shape_node_ops(graph: PipelineGraph, node_id: str) -> list[str]:
    return [op.op for op in graph._nodes[node_id].pipeline._ops]


class TestToExprRequiresOptimization:
    """``to_expr`` refuses an un-optimized graph — optimization has one site.

    Before the fix, the public ``Pipeline.to_graph(col).to_expr()`` route (which
    never runs ``sink``) emitted an *unoptimized* graph: no CSE, no affine
    fusion, pixel-divergent from ``sink()``. ``to_expr`` now raises unless the
    optimize phase has run, so the low-level path cannot silently diverge.
    """

    def test_fresh_graph_is_not_marked_optimized(self) -> None:
        assert _graph_of(_two_rotate_pipe())._optimized is False

    def test_optimize_marks_the_graph(self) -> None:
        graph = _graph_of(_two_rotate_pipe())
        assert graph.optimize(OptFlags.all())._optimized is True

    def test_to_expr_rejects_unoptimized_graph(self) -> None:
        graph = _graph_of(_two_rotate_pipe())
        graph.set_output("n", "numpy")
        with pytest.raises(RuntimeError, match="optimization phase"):
            graph.to_expr()

    def test_to_expr_works_after_optimize(self) -> None:
        graph = _graph_of(_two_rotate_pipe())
        graph.set_output("n", "numpy")
        graph.optimize(OptFlags.all())
        # Does not raise; register_plugin_function builds the expr lazily and
        # needs no compiled .so at construction time.
        graph.to_expr()

    def test_sink_return_graph_is_optimized(self) -> None:
        graph = (
            pl.col("img").cv.pipe(_two_rotate_pipe()).sink("numpy", return_expr=False)
        )
        assert graph._optimized is True
        # sink() auto-generates the node id, so read the sole node generically.
        (only_node,) = graph._nodes.values()
        assert [op.op for op in only_node.pipeline._ops] == ["resize", "warp_affine"]


def _crop_after_pointwise_pipe() -> Pipeline:
    """A crop sitting after a run of pointwise ops — the pushdown candidate.

    ``cast``/``scale``/``grayscale`` are all ``Pointwise`` (spatial radius 0),
    so the crop commutes to the front of the run exactly (byte-identical).
    """
    return (
        Pipeline()
        .source("image_bytes")
        .cast("f32")
        .scale(0.5)
        .grayscale()
        .crop(top=1, left=1, height=8, width=8)
    )


class TestSpatialWindowPushdown:
    """The crop-hoisting pass: a crop moves earlier past a ``Pointwise`` run.

    The pass reads each op's ``SpatialDependency`` from the ``op_contract`` FFI
    (``spatial_rule``) — the single authority — and moves a crop to the front of
    the contiguous run of ``pointwise`` ops immediately preceding it, within one
    node's op list. ``neighborhood``/``geometric``/``global`` ops and any
    ``assert_shape`` boundary are barriers.
    """

    def test_pass_is_registered(self) -> None:
        assert "spatial_window_pushdown" in PASS_NAMES
        assert OptFlags().spatial_window_pushdown is True

    def test_optimize_none_leaves_order(self) -> None:
        graph = _graph_of(_crop_after_pointwise_pipe())
        graph.optimize(OptFlags.none())
        assert _node_ops(graph) == ["cast", "scale", "grayscale", "crop"]

    def test_flag_hoists_crop_to_front_of_pointwise_run(self) -> None:
        graph = _graph_of(_crop_after_pointwise_pipe())
        graph.optimize(
            OptFlags(
                spatial_window_pushdown=True,
                affine_fusion=False,
                common_subexpression_elimination=False,
            )
        )
        assert _node_ops(graph) == ["crop", "cast", "scale", "grayscale"]

    def test_flag_off_leaves_order(self) -> None:
        graph = _graph_of(_crop_after_pointwise_pipe())
        graph.optimize(OptFlags(spatial_window_pushdown=False))
        assert _node_ops(graph) == ["cast", "scale", "grayscale", "crop"]

    def test_idempotent(self) -> None:
        graph = _graph_of(_crop_after_pointwise_pipe())
        graph.optimize(OptFlags.all())
        first = _node_ops(graph)
        graph.optimize(OptFlags.all())
        assert _node_ops(graph) == first == ["crop", "cast", "scale", "grayscale"]

    def test_does_not_mutate_caller(self) -> None:
        pipe = _crop_after_pointwise_pipe()
        graph = _graph_of(pipe)
        graph.optimize(OptFlags.all())
        assert [op.op for op in pipe._ops] == [
            "cast",
            "scale",
            "grayscale",
            "crop",
        ]

    def test_neighborhood_op_is_a_barrier(self) -> None:
        # blur is Neighborhood: a crop may not cross it in this phase.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .blur(1.0)
            .crop(top=0, left=0, height=8, width=8)
        )
        graph = _graph_of(pipe)
        graph.optimize(OptFlags.all())
        assert _node_ops(graph) == ["grayscale", "blur", "crop"]

    def test_geometric_op_is_a_barrier(self) -> None:
        # resize is Geometric: a crop may not cross it in this phase.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=32, width=32)
            .crop(top=0, left=0, height=8, width=8)
        )
        graph = _graph_of(pipe)
        graph.optimize(OptFlags.all())
        assert _node_ops(graph) == ["resize", "crop"]

    def test_hoist_stops_at_barrier_mid_run(self) -> None:
        # [resize(geometric), grayscale(pointwise), crop] — the crop hoists past
        # grayscale but stops at the resize barrier.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=32, width=32)
            .grayscale()
            .crop(top=0, left=0, height=8, width=8)
        )
        graph = _graph_of(pipe)
        graph.optimize(OptFlags.all())
        assert _node_ops(graph) == ["resize", "crop", "grayscale"]

    def test_assert_shape_is_a_barrier(self) -> None:
        # A user shape assertion between the pointwise run and the crop pins the
        # pre-crop shape; the crop must not move across it.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .assert_shape(height=96, width=96)
            .crop(top=0, left=0, height=8, width=8)
        )
        graph = _graph_of(pipe)
        graph.optimize(OptFlags.all())
        assert _node_ops(graph) == ["grayscale", "crop"]

    def test_crop_does_not_cross_node_boundary(self) -> None:
        # .pipe() makes a new node; phase 1's barrier is the node boundary, so a
        # crop in the downstream node is not hoisted into the pointwise upstream.
        # This pins the boundary: enabling cross-node hoisting must update it.
        graph = (
            pl.col("img")
            .cv.pipe(Pipeline().source("image_bytes").grayscale())
            .pipe(Pipeline().crop(top=0, left=0, height=8, width=8))
            .sink("numpy", return_expr=False, opt_flags=OptFlags.all())
        )
        op_lists = [[op.op for op in n.pipeline._ops] for n in graph._nodes.values()]
        assert ["crop"] in op_lists
        assert ["grayscale"] in op_lists


class TestShapeSubpipelineStaging:
    """The contour-source shape sub-pipeline goes through the optimize phase.

    It used to be affine-fused at *construction* time, unconditionally — which
    both broke the "construction never optimizes" staging contract and ignored
    ``opt_flags``. The shape sub-pipeline is an ordinary graph node, so the
    single ``optimize()`` phase fuses it like any other, honoring the flags.
    """

    def test_construction_leaves_shape_subpipeline_logical(self) -> None:
        # No plugin: pure construction. The embedded shape spec must be the
        # verbatim logical op chain, NOT a construction-time fusion.
        pipe = Pipeline().source("contour", shape=_shape_ref())
        embedded = pipe._source.shape_pipeline["pipeline"]["ops"]
        assert [op["op"] for op in embedded] == ["resize", "rotate", "rotate"]

    def test_shape_subpipeline_fusion_respects_opt_flags(self) -> None:
        # No plugin: sink(return_expr=False) builds + optimizes the graph
        # without registering the expr. The shape node id is the shape ref's own
        # node id; inspect its ops under each flag.
        shape = _shape_ref()
        shape_id = shape._node_id
        contour = Pipeline().source("contour", shape=shape)

        on = (
            pl.col("c")
            .cv.pipe(contour)
            .sink("numpy", return_expr=False, opt_flags=OptFlags.all())
        )
        assert _shape_node_ops(on, shape_id) == ["resize", "warp_affine"]

        off = (
            pl.col("c")
            .cv.pipe(contour)
            .sink("numpy", return_expr=False, opt_flags=OptFlags(affine_fusion=False))
        )
        assert _shape_node_ops(off, shape_id) == ["resize", "rotate", "rotate"]

    @plugin_required
    def test_shape_subpipeline_output_identical_under_flags(self) -> None:
        # Plugin: the shape node's fusion changes pixels but not its H/W, and the
        # contour source reads only the shape buffer's dimensions — so the mask
        # is byte-identical whether or not the shape sub-pipeline was fused.
        import io

        import numpy as np
        from PIL import Image

        from polars_cv import numpy_from_struct
        from polars_cv.geometry import CONTOUR_SCHEMA

        buf = io.BytesIO()
        Image.new("RGB", (48, 48), color=(20, 120, 200)).save(buf, format="PNG")
        contour = {
            "exterior": [
                {"x": 5.0, "y": 5.0},
                {"x": 30.0, "y": 5.0},
                {"x": 30.0, "y": 30.0},
                {"x": 5.0, "y": 30.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        df = pl.DataFrame(
            {
                "img": [buf.getvalue()],
                "c": pl.Series([contour], dtype=CONTOUR_SCHEMA),
            }
        )
        pipe = Pipeline().source("contour", shape=_shape_ref())

        def run(flags: OptFlags) -> "np.ndarray":
            out = df.select(m=pl.col("c").cv.pipe(pipe).sink("numpy", opt_flags=flags))[
                "m"
            ][0]
            return numpy_from_struct(out)

        assert np.array_equal(run(OptFlags.all()), run(OptFlags.none()))


class TestIdentityElimination:
    """Staging for the identity-elimination pass.

    Needs the compiled plugin: the pass reads ``op_identity_rule`` /
    ``op_schema`` / ``op_infer_shape`` to classify each op and evaluate its
    condition against the entering state.
    """

    @plugin_required
    def test_zero_pad_is_removed(self) -> None:
        g = _graph_of(
            Pipeline()
            .source("image_bytes")
            .pad(top=0, bottom=0, left=0, right=0)
            .grayscale()
        )
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["grayscale"]

    @plugin_required
    def test_flag_gates_the_pass(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=0, bottom=0, left=0, right=0)
            .grayscale()
        )
        off = _graph_of(pipe)
        off.optimize(OptFlags(identity_elimination=False))
        assert _node_ops(off) == ["pad", "grayscale"]

        on = _graph_of(pipe)
        on.optimize(OptFlags(identity_elimination=True))
        assert _node_ops(on) == ["grayscale"]

    @plugin_required
    def test_promoting_scale_is_not_an_identity(self) -> None:
        # scale(1.0) promotes u8 -> f32, so deleting it would change the output
        # dtype: the indicator must report Never, and the op must survive.
        g = _graph_of(Pipeline().source("image_bytes").scale(1.0))
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["scale"]

    @plugin_required
    def test_expression_amount_is_not_a_zero_pad(self) -> None:
        # A per-row pad amount resolves to the neutralization placeholder (not
        # zero), so the FFI reports "never" and the pad survives — no op is
        # deleted on the strength of a spoofed placeholder value.
        g = _graph_of(Pipeline().source("image_bytes").pad(top=pl.col("t")))
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["pad"]

    @plugin_required
    def test_zero_pad_with_expression_fill_is_still_removed(self) -> None:
        # Amounts are literally zero, so no pixels are added and the fill value
        # is never used: an expression on the irrelevant `value` param does not
        # stop the elimination.
        g = _graph_of(
            Pipeline()
            .source("image_bytes")
            .pad(top=0, bottom=0, left=0, right=0, value=pl.col("v"))
            .grayscale()
        )
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["grayscale"]

    @plugin_required
    def test_redundant_cast_is_removed(self) -> None:
        # The second cast(u8) enters u8, so it copies rather than converts.
        g = _graph_of(Pipeline().source("image_bytes").cast("u8").cast("u8"))
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["cast"]

    @plugin_required
    def test_cast_to_a_new_dtype_is_kept(self) -> None:
        g = _graph_of(Pipeline().source("image_bytes").cast("u8").cast("f32"))
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["cast", "cast"]

    @plugin_required
    def test_full_frame_crop_is_removed(self) -> None:
        # resize fixes the entering H/W at 64x64; a 64x64 crop is a no-op view.
        g = _graph_of(
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .crop(top=0, left=0, height=64, width=64)
        )
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["resize"]

    @plugin_required
    def test_partial_crop_is_kept(self) -> None:
        g = _graph_of(
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .crop(top=0, left=0, height=32, width=32)
        )
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["resize", "crop"]

    @plugin_required
    def test_crop_with_unknown_entering_shape_is_kept(self) -> None:
        # No prior resize: entering H/W is unknown, so a full-frame crop cannot
        # be proven and the op is conservatively kept.
        g = _graph_of(
            Pipeline().source("image_bytes").crop(top=0, left=0, height=64, width=64)
        )
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["crop"]

    @plugin_required
    def test_assertion_bearing_node_is_left_untouched(self) -> None:
        # A node carrying an assert_shape is skipped wholesale so no positional
        # assertion key is re-derived across a deletion — the no-op crop stays.
        g = _graph_of(
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .crop(top=0, left=0, height=64, width=64)
            .assert_shape(height=64, width=64)
        )
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["resize", "crop"]
