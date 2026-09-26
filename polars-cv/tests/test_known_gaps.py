"""Executable specifications for defects that are verified but not yet fixed.

Every test in this module is ``xfail(strict=True)``. Each one:

* describes a defect that has been **confirmed against running code or source**,
  not a suspicion;
* asserts the behaviour the codebase *should* have, so it fails today; and
* names, in its docstring, what "fixed" looks like.

``strict=True`` is the point. When someone lands the fix, the test XPASSes and
the suite goes **red** — which is the signal to delete the marker and let the
test join the suite proper. A backlog that lives in prose is a backlog that
rots; this one cannot silently become stale in either direction. It is the same
reasoning as `AGENTS.md`'s "The Single-Authority Refactor: What Was Done, What
Is Left" section, made executable.

These are the items recorded as deferred in that review: the ones where a fix is
a design change rather than a correction, and so wants its own commit. Adding a
test here is not a way to avoid fixing something — it is a way to stop the
knowledge evaporating between sessions.

Do **not** put a flaky or environment-dependent test here. `xfail` marks "known
broken", never "sometimes fails".
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import LazyPipelineExpr, Pipeline
from tests._plan_view import planned
from tests._schema_parity import Outcome, plan_or_reject
from tests.conftest import make_image_png, plugin_required

# Each gap carries its own lane: a source scan is `structural` (pre-commit runs
# it with no compiled extension), a runtime one `plugin_required`.


def _gap(reason: str) -> pytest.MarkDecorator:
    """Mark a verified, unfixed defect. Strict, so a fix fails the suite.

    Only an ``AssertionError`` is the expected failure: every gap states the
    defect as an ``assert``, so a gap that breaks for any other reason (a
    renamed helper, a fixture that no longer builds) fails the suite rather
    than reading as the defect.
    """
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


def _build_error(build: Callable[[], object]) -> str | None:
    """The message a builder call refused with, or ``None`` if it built."""
    try:
        build()
    except ValueError as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Planned sizes: PLANNER_SIZES_PLAN.md (S1 rank-N shapes, S2 symbolic validate)
# ---------------------------------------------------------------------------

#: Image ops every row of a rank-1 buffer refuses (each confirmed to fail per
#: row with a shape or parameter error), with the sink each one reaches.
_IMAGE_OPS_ON_RANK_1: dict[str, tuple[Callable[[Pipeline], Pipeline], str]] = {
    "blur": (lambda p: p.blur(sigma=1.0), "numpy"),
    "resize": (lambda p: p.resize(height=4, width=4), "numpy"),
    "grayscale": (lambda p: p.grayscale(), "numpy"),
    "threshold": (lambda p: p.threshold(128), "numpy"),
    "channel_select": (lambda p: p.channel_select(0), "numpy"),
    "perceptual_hash": (lambda p: p.perceptual_hash(), "list"),
}


def _sized_image(height: int, width: int) -> Pipeline:
    """An image pipeline whose whole ``[H, W, 3]`` u8 shape is planned."""
    return (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=4, width=5, channels=3)
        .cast("u8")
        .resize(height=height, width=width)
    )


@plugin_required
class TestPlannedSizes:
    """What the planner knows about sizes and does not use, or gets wrong."""

    @_gap(
        "S2: check_rank keeps only errors whose *variant* is rank-level; "
        "require_hw_or_hwc and phash report a known-rank failure as "
        "ShapeRequirement/InvalidParameter, so it is dropped until each row "
        "fails. Fixed when the builder call raises."
    )
    @pytest.mark.parametrize("op", sorted(_IMAGE_OPS_ON_RANK_1))
    def test_a_known_rank_refuses_an_image_op_at_build(self, op: str) -> None:
        apply, sink = _IMAGE_OPS_ON_RANK_1[op]
        raw = pl.DataFrame({"b": [bytes(range(16))]})
        pipe = apply(Pipeline().source("raw", dtype="u8"))
        # The precondition this gap rests on: every row really does refuse.
        with pytest.raises(pl.exceptions.ComputeError):
            raw.select(pl.col("b").cv.pipe(pipe).sink(sink))
        err = _build_error(lambda: apply(Pipeline().source("raw", dtype="u8")))
        assert err is not None, f"{op}() on a rank-1 buffer built; every row fails"

    @_gap(
        "S2: OpShape::Broadcast falls back to the left operand's shape "
        "(`.unwrap_or(a)`) for two known, incompatible shapes, and the planner "
        "never validates a graph op, so .sink() publishes Array(u8, (4, 5, 3)) "
        "for a query every row of which fails. Fixed when .sink() refuses."
    )
    def test_incompatible_known_operands_refuse_at_build(self) -> None:
        df = pl.DataFrame({"i": [make_image_png(4, 5, 3, seed=1)]})
        left = pl.col("i").cv.pipe(_sized_image(4, 5))
        right = pl.col("i").cv.pipe(_sized_image(2, 5))
        err = _build_error(lambda: left.add(right).sink("array"))
        if err is None:
            planned_dtype = (
                df.lazy().select(o=left.add(right).sink("array")).collect_schema()["o"]
            )
            raise AssertionError(
                f".sink() accepted incompatible operands and planned {planned_dtype}"
            )
        assert "broadcast" in err.lower(), err

    @_gap(
        "S2: the same fallback over two Array columns whose shapes are known "
        "only once Polars plans with them: the schema is published and every "
        "row then fails. Fixed when planning refuses."
    )
    def test_incompatible_array_columns_refuse_at_plan(self) -> None:
        df = pl.DataFrame(
            {
                "x": pl.Series(
                    [np.zeros((4, 5, 3), np.uint8)], dtype=pl.Array(pl.UInt8, (4, 5, 3))
                ),
                "y": pl.Series(
                    [np.zeros((2, 5, 3), np.uint8)], dtype=pl.Array(pl.UInt8, (2, 5, 3))
                ),
            }
        )

        def build() -> pl.Expr:
            x = pl.col("x").cv.pipe(Pipeline().source("array"))
            y = pl.col("y").cv.pipe(Pipeline().source("array"))
            return x.add(y).sink("array")

        # plan_or_reject turns "published, then execution raised" into an
        # AssertionError, which is today's behaviour.
        result = plan_or_reject(df, build)
        assert result.outcome is Outcome.REJECTED_AT_PLAN, result.outcome
        assert "broadcast" in (result.reason or "").lower(), result.reason

    @_gap(
        "S2: OpShape::Broadcast knows a size only when every size of both "
        "operands is known, so one unknown channel count drops the H and W "
        "both operands agree on. Fixed when each axis broadcasts on its own."
    )
    def test_broadcast_keeps_the_sizes_both_operands_know(self) -> None:
        def resized() -> LazyPipelineExpr:
            return pl.col("i").cv.pipe(
                Pipeline().source("image_bytes", dtype="u8").resize(height=8, width=8)
            )

        left, right = resized(), resized()
        assert planned(left).hw == (8, 8)  # the operands' sizes are planned
        out = planned(left.add(right))
        assert out.hw == (8, 8), f"a.add(b) planned {out}"
