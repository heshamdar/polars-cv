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
    ENGINE_PASS_NAMES,
    LOGICAL_PASS_NAMES,
    OPT_ENV_VAR,
    OPTIMIZATION_PASSES,
    PASS_NAMES,
    OptFlags,
    PassSpec,
    resolve_opt_flags,
)
from polars_cv._types import SlotTable
from tests._plan_view import op_names, ops_of
from tests.conftest import plugin_required


def _removable_op_pipe() -> Pipeline:
    """A pipeline with a removable no-op: a full-frame crop after a resize.

    The leading ``resize`` fixes the plan-time H/W at 64x64, so the 64x64 crop
    is provably a no-op and identity elimination deletes it, leaving ``[resize]``
    while ``resize`` stays. A logical (unoptimized) view still shows the crop.
    (Substitutes for the old affine-fusion fixture, removed with that pass — it
    likewise exercises a logical node pass that rewrites the op chain.)
    """
    return (
        Pipeline()
        .source("image_bytes")
        .resize(height=64, width=64)
        .crop(top=0, left=0, height=64, width=64)
    )


def _graph_of(pipe: Pipeline) -> PipelineGraph:
    graph = PipelineGraph()
    graph.add_node(node_id="n", pipeline=pipe, column=pl.col("img"))
    return graph


def _node_ops(graph: PipelineGraph) -> list[str]:
    return op_names(graph._nodes["n"].pipeline)


class TestRegistry:
    def test_pass_names_are_unique(self) -> None:
        assert len(PASS_NAMES) == len(set(PASS_NAMES))

    def test_every_pass_declares_a_summary_and_a_known_tier(self) -> None:
        for spec in OPTIMIZATION_PASSES:
            assert isinstance(spec, PassSpec)
            assert spec.summary
            assert spec.tier in ("logical", "engine")

    def test_tier_partitions_the_registry(self) -> None:
        assert set(LOGICAL_PASS_NAMES) | set(ENGINE_PASS_NAMES) == set(PASS_NAMES)
        assert set(LOGICAL_PASS_NAMES).isdisjoint(ENGINE_PASS_NAMES)

    def test_every_flag_field_is_boolean_defaulting_on(self) -> None:
        defaults = OptFlags()
        for name in PASS_NAMES:
            assert getattr(defaults, name) is True

    def test_engine_flags_round_trip_into_graph_opt(self) -> None:
        """Each engine flag surfaces in the serialized ``opt`` object.

        This is the toggle's only path to Rust, so a missing/renamed key would
        silently disable the switch. Logical passes are left off so no compiled
        plugin is needed.
        """
        import json

        # Engine on, logical off.
        flags_on = OptFlags(**{n: (n in ENGINE_PASS_NAMES) for n in PASS_NAMES})
        g = _graph_of(Pipeline().source("image_bytes").grayscale())
        g.set_output("n", "numpy")
        g.optimize(flags_on)
        opt = json.loads(g._to_json())["opt"]
        assert set(opt) == set(ENGINE_PASS_NAMES)
        assert all(opt[n] is True for n in ENGINE_PASS_NAMES)

        # Everything off → every engine flag serializes False.
        g2 = _graph_of(Pipeline().source("image_bytes").grayscale())
        g2.set_output("n", "numpy")
        g2.optimize(OptFlags.none())
        opt2 = json.loads(g2._to_json())["opt"]
        assert all(opt2[n] is False for n in ENGINE_PASS_NAMES)


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
            OptFlags().scalar_fusion = False  # type: ignore[misc]


class TestParse:
    def test_all(self) -> None:
        assert OptFlags.parse("all") == OptFlags.all()

    def test_none(self) -> None:
        assert OptFlags.parse("none") == OptFlags.none()

    def test_bare_name_turns_only_that_on(self) -> None:
        flags = OptFlags.parse("scalar_fusion")
        assert flags.enabled("scalar_fusion")
        assert not flags.enabled("common_subexpression_elimination")

    def test_all_minus_one(self) -> None:
        flags = OptFlags.parse("all,-scalar_fusion")
        assert not flags.enabled("scalar_fusion")
        assert flags.enabled("common_subexpression_elimination")

    def test_whitespace_is_tolerated(self) -> None:
        assert OptFlags.parse("  all , -scalar_fusion ") == OptFlags.parse(
            "all,-scalar_fusion"
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
        monkeypatch.setenv(OPT_ENV_VAR, "all,-scalar_fusion")
        flags = OptFlags.from_env()
        assert flags.enabled("common_subexpression_elimination")
        assert not flags.enabled("scalar_fusion")


class TestResolve:
    def test_none_defers_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(OPT_ENV_VAR, "none")
        assert resolve_opt_flags(None) == OptFlags.none()

    def test_true_is_all(self) -> None:
        assert resolve_opt_flags(True) == OptFlags.all()

    def test_false_is_none(self) -> None:
        assert resolve_opt_flags(False) == OptFlags.none()

    def test_optflags_passes_through(self) -> None:
        flags = OptFlags(scalar_fusion=False)
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
        # _to_spec_dict never optimizes; it serializes the ops as written.
        spec = _removable_op_pipe()._to_spec_dict(SlotTable().index)
        assert [op["op"] for op in spec["ops"]] == ["resize", "crop"]

    def test_optimize_none_changes_nothing(self) -> None:
        graph = _graph_of(_removable_op_pipe())
        graph.optimize(OptFlags.none())
        assert _node_ops(graph) == ["resize", "crop"]

    @plugin_required
    def test_identity_elimination_flag_gates_the_pass(self) -> None:
        off = _graph_of(_removable_op_pipe())
        off.optimize(OptFlags(identity_elimination=False))
        assert _node_ops(off) == ["resize", "crop"]

        on = _graph_of(_removable_op_pipe())
        on.optimize(OptFlags(identity_elimination=True))
        assert _node_ops(on) == ["resize"]

    @plugin_required
    def test_optimize_is_idempotent(self) -> None:
        graph = _graph_of(_removable_op_pipe())
        graph.optimize(OptFlags.all())
        first = _node_ops(graph)
        graph.optimize(OptFlags.all())
        assert _node_ops(graph) == first == ["resize"]


@plugin_required
class TestImmutability:
    """Optimization must not mutate the caller's Pipeline (it is immutable)."""

    def test_optimize_does_not_touch_the_source_pipeline(self) -> None:
        pipe = _removable_op_pipe()
        graph = _graph_of(pipe)
        graph.optimize(OptFlags.all())
        # The graph optimized its own copy; the caller's pipeline is untouched.
        assert op_names(pipe) == ["resize", "crop"]
        assert _node_ops(graph) == ["resize"]


class TestExplain:
    """Pipeline.explain surfaces the logical vs physical op chain."""

    @plugin_required
    def test_logical_shows_unoptimized(self) -> None:
        text = _removable_op_pipe().explain(optimized=False)
        assert "crop" in text

    @plugin_required
    def test_optimized_shows_eliminated(self) -> None:
        text = _removable_op_pipe().explain(optimized=True)
        assert "crop" not in text

    @plugin_required
    def test_optimized_respects_flags(self) -> None:
        text = _removable_op_pipe().explain(
            opt_flags=OptFlags(identity_elimination=False)
        )
        assert "crop" in text

    def test_explain_does_not_mutate(self) -> None:
        pipe = _removable_op_pipe()
        pipe.explain(optimized=False)
        assert op_names(pipe) == ["resize", "crop"]

    @plugin_required
    def test_optimized_reflects_identity_elimination(self) -> None:
        # `.sink()` eliminates the full-frame crop and the redundant same-dtype
        # cast; `explain(optimized=True)` must show that same physical chain, not
        # a logical one still carrying them. (This is the divergence that arose
        # when `explain` hand-listed its passes and missed identity elimination.)
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .crop(top=0, left=0, height=64, width=64)
            .cast("u8")
            .cast("u8")
        )
        text = pipe.explain(optimized=True)
        assert "crop" not in text
        assert text.count("cast(") == 1

        # Gated off, the logical ops survive.
        off = pipe.explain(opt_flags=OptFlags(identity_elimination=False))
        assert "crop" in off
        assert off.count("cast(") == 2


def _shape_ref() -> "pl.Expr":
    """A shape sub-pipeline expression carrying a removable no-op.

    ``resize`` fixes the plan-time H/W at 40x40, so the 40x40 crop is a provable
    no-op that identity elimination deletes — the same fixture shape as
    ``_removable_op_pipe``, but delivered as a ``shape=`` reference for a contour
    source, so it exercises optimization of an embedded shape node.
    """
    return pl.col("img").cv.pipe(
        Pipeline()
        .source("image_bytes")
        .resize(width=40, height=40)
        .crop(top=0, left=0, height=40, width=40)
    )


def _shape_node_ops(graph: PipelineGraph, node_id: str) -> list[str]:
    return op_names(graph._nodes[node_id].pipeline)


class TestToExprRequiresOptimization:
    """``to_expr`` refuses an un-optimized graph — optimization has one site.

    Before the fix, the public ``Pipeline.to_graph(col).to_expr()`` route (which
    never runs ``sink``) emitted an *unoptimized* graph, divergent from
    ``sink()``. ``to_expr`` now raises unless the optimize phase has run, so the
    low-level path cannot silently diverge.
    """

    def test_fresh_graph_is_not_marked_optimized(self) -> None:
        assert _graph_of(_removable_op_pipe())._optimized is False

    @plugin_required
    def test_optimize_marks_the_graph(self) -> None:
        graph = _graph_of(_removable_op_pipe())
        assert graph.optimize(OptFlags.all())._optimized is True

    def test_to_expr_rejects_unoptimized_graph(self) -> None:
        graph = _graph_of(_removable_op_pipe())
        graph.set_output("n", "numpy")
        with pytest.raises(RuntimeError, match="optimization phase"):
            graph.to_expr()

    @plugin_required
    def test_to_expr_works_after_optimize(self) -> None:
        graph = _graph_of(_removable_op_pipe())
        graph.set_output("n", "numpy")
        graph.optimize(OptFlags.all())
        # Does not raise; register_plugin_function builds the expr lazily and
        # needs no compiled .so at construction time.
        graph.to_expr()

    @plugin_required
    def test_sink_return_graph_is_optimized(self) -> None:
        graph = (
            pl.col("img").cv.pipe(_removable_op_pipe()).sink("numpy", return_expr=False)
        )
        assert graph._optimized is True
        # sink() auto-generates the node id, so read the sole node generically.
        (only_node,) = graph._nodes.values()
        assert op_names(only_node.pipeline) == ["resize"]


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

    The pass (``passes.rs``) reads each op's ``SpatialDependency`` — the single
    authority — and moves a crop to the front of
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
                identity_elimination=False,
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
        assert op_names(pipe) == [
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
        # A user shape assertion between the pointwise run and the crop checks
        # the pre-crop shape; the crop must not move across it.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .assert_shape(height=96, width=96)
            .crop(top=0, left=0, height=8, width=8)
        )
        graph = _graph_of(pipe)
        graph.optimize(OptFlags.all())
        assert _node_ops(graph) == ["grayscale", "assert_shape", "crop"]

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
        op_lists = [op_names(n.pipeline) for n in graph._nodes.values()]
        assert ["crop"] in op_lists
        assert ["grayscale"] in op_lists


class TestShapeSubpipelineStaging:
    """The contour-source shape sub-pipeline goes through the optimize phase.

    It used to be optimized at *construction* time, unconditionally — which both
    broke the "construction never optimizes" staging contract and ignored
    ``opt_flags``. The shape sub-pipeline is an ordinary graph node, so the
    single ``optimize()`` phase rewrites it like any other, honoring the flags.
    """

    def test_construction_leaves_shape_subpipeline_logical(self) -> None:
        # No plugin: pure construction. The rasterize step references the
        # shape node by id only, and that node's ops stay the verbatim logical
        # chain, NOT a construction-time rewrite.
        shape = _shape_ref()
        pipe = Pipeline().source("contour").rasterize(shape=shape)
        [rasterize] = ops_of(pipe)
        assert rasterize.params["size"] == shape._node_id
        assert op_names(shape) == ["resize", "crop"]

    @plugin_required
    def test_shape_subpipeline_optimization_respects_opt_flags(self) -> None:
        # sink(return_expr=False) builds + optimizes the graph without
        # registering the expr. The shape node id is the shape ref's own node id;
        # inspect its ops under each flag.
        shape = _shape_ref()
        shape_id = shape._node_id
        contour = Pipeline().source("contour").rasterize(shape=shape)

        on = (
            pl.col("c")
            .cv.pipe(contour)
            .sink("numpy", return_expr=False, opt_flags=OptFlags.all())
        )
        assert _shape_node_ops(on, shape_id) == ["resize"]

        off = (
            pl.col("c")
            .cv.pipe(contour)
            .sink(
                "numpy",
                return_expr=False,
                opt_flags=OptFlags(identity_elimination=False),
            )
        )
        assert _shape_node_ops(off, shape_id) == ["resize", "crop"]

    @plugin_required
    def test_shape_subpipeline_output_identical_under_flags(self) -> None:
        # Plugin: optimizing the shape node preserves its H/W (and output), and
        # the contour source reads only the shape buffer's dimensions — so the
        # mask is byte-identical whether or not the shape sub-pipeline was
        # optimized.
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
        pipe = Pipeline().source("contour").rasterize(shape=_shape_ref())

        def run(flags: OptFlags) -> "np.ndarray":
            out = df.select(m=pl.col("c").cv.pipe(pipe).sink("numpy", opt_flags=flags))[
                "m"
            ][0]
            return numpy_from_struct(out)

        assert np.array_equal(run(OptFlags.all()), run(OptFlags.none()))


class TestIdentityElimination:
    """Staging for the identity-elimination pass.

    Needs the compiled plugin: the pass (``passes.rs``) reads each op's identity
    rule and evaluates it against the recorded entering state. Its gating on
    per-row deciding params is unit-tested there.
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
    def test_an_assertion_is_kept_and_does_not_shield_its_node(self) -> None:
        # An assert_shape checks every row, so it is never an identity, even
        # where the plan already knows it holds. It is an op like any other:
        # the node's no-op crop still goes.
        g = _graph_of(
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .crop(top=0, left=0, height=64, width=64)
            .assert_shape(height=64, width=64)
        )
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["resize", "assert_shape"]

    @plugin_required
    def test_offset_crop_with_full_extent_is_kept(self) -> None:
        g = _graph_of(
            Pipeline()
            .source("image_bytes")
            .resize(height=20, width=20)
            .crop(top=5, left=5, height=20, width=20)
        )
        g.optimize(OptFlags.all())
        assert _node_ops(g) == ["resize", "crop"]

    @plugin_required
    def test_a_declared_shape_in_the_lineage_is_a_checked_fact(self) -> None:
        # The upstream assertion is checked where it was written, so every row
        # reaching the continuation is 10x10: the full-frame crop is provably
        # a no-op and goes, with the zero pad. (A row the assertion does not
        # describe fails at the assertion, optimized or not.)
        upstream = pl.col("image").cv.pipe(
            Pipeline().source("image_bytes").assert_shape(height=10, width=10)
        )
        graph = upstream.pipe(
            Pipeline()
            .pad(top=0, bottom=0, left=0, right=0)
            .crop(top=0, left=0, height=10, width=10)
        ).sink("numpy", return_expr=False, opt_flags=OptFlags.all())
        ops = [op_names(n.pipeline) for n in graph._nodes.values()]
        assert ["assert_shape"] in ops and ["crop"] not in ops

    @plugin_required
    def test_undeclared_continuation_still_eliminates_a_full_frame_crop(
        self,
    ) -> None:
        # Control for the test above: a continuation whose H/W an upstream op
        # *computed* (resize) is a proven shape, so the no-op crop is removed —
        # the guard keys on declarations, not on being a continuation.
        upstream = pl.col("image").cv.pipe(
            Pipeline().source("image_bytes").resize(height=10, width=10)
        )
        graph = upstream.pipe(Pipeline().crop(top=0, left=0, height=10, width=10)).sink(
            "numpy", return_expr=False, opt_flags=OptFlags.all()
        )
        ops = [op_names(n.pipeline) for n in graph._nodes.values()]
        assert ["resize"] in ops and ["crop"] not in ops


class TestSpatialPushdownGuard:
    """A crop is never hoisted past an op that reads a sibling node's buffer.

    A binary/merge op is spatially ``Pointwise``, but hoisting a crop earlier
    would shrink only this operand and leave the sibling full-size. The pass must
    treat such an op as a barrier regardless of its spatial rule.
    """

    @plugin_required
    def test_crop_does_not_cross_apply_mask(self) -> None:
        # Build a node whose ops are [grayscale, apply_mask, crop] directly —
        # node-splitting keeps this shape off the public API, so this is the
        # only way to exercise the barrier.
        pipe = Pipeline().source("image_bytes").grayscale()
        pipe._add_node_op("apply_mask", {"mask": "mask_node", "invert": False})
        pipe = pipe.crop(top=0, left=0, height=8, width=8)
        assert op_names(pipe) == ["grayscale", "apply_mask", "crop"]

        pipe._run_node_pass("spatial_window_pushdown")
        # The crop stays put: apply_mask reads a sibling node, so it is a barrier.
        assert op_names(pipe) == ["grayscale", "apply_mask", "crop"]

    @plugin_required
    def test_control_crop_crosses_a_pointwise_run(self) -> None:
        # Same shape without the sibling-reading op: the crop DOES move, proving
        # the barrier above is what stopped it (not an inert pass).
        pipe = Pipeline().source("image_bytes").grayscale().invert()
        pipe = pipe.crop(top=1, left=1, height=8, width=8)
        pipe._run_node_pass("spatial_window_pushdown")
        assert op_names(pipe) == ["crop", "grayscale", "invert"]
