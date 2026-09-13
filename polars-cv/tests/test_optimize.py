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
