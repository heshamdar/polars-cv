"""A plan fact that rests on a *claim* must never let the optimizer change output.

A declared dtype or size is either checked where the data arrives or carried as
a declaration, so a pass that trusts it (identity elimination) cannot turn a
working query into a failing one. Each case runs the same pipeline with every
optimization off and on, at the user-facing entry point (``.sink``), and
requires the same result. Consolidation plan C0.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import LazyPipelineExpr, OptFlags, Pipeline
from tests._plan_view import planned
from tests._schema_parity import Outcome, plan_or_reject
from tests.conftest import make_image_png, plugin_required


def _png(height: int, width: int) -> bytes:
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(
        buf, format="PNG"
    )
    return buf.getvalue()


def _f32_blob_frame() -> pl.DataFrame:
    arr = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    df = pl.DataFrame(
        {"a": [arr.tolist()]}, schema={"a": pl.List(pl.List(pl.List(pl.Float32)))}
    )
    return df.select(b=pl.col("a").cv.pipe(Pipeline().source("list")).sink("blob"))


@plugin_required
class TestADtypeClaimIsChecked:
    def test_a_blob_whose_dtype_contradicts_the_claim_is_refused_at_decode(
        self,
    ) -> None:
        df = _f32_blob_frame()
        pipe = Pipeline().source("blob", dtype="u8")
        with pytest.raises(pl.exceptions.ComputeError, match=r"(?i)f32.*u8|u8.*f32"):
            df.select(pl.col("b").cv.pipe(pipe).sink("numpy"))

    @pytest.mark.parametrize("flags", [OptFlags.none(), OptFlags.all()])
    def test_the_claim_does_not_blame_the_planner(self, flags: OptFlags) -> None:
        df = _f32_blob_frame()
        pipe = Pipeline().source("blob", dtype="u8").cast("u8")
        with pytest.raises(pl.exceptions.ComputeError) as err:
            df.select(pl.col("b").cv.pipe(pipe).sink("numpy", opt_flags=flags))
        assert "planner" not in str(err.value)

    def test_a_matching_claim_decodes(self) -> None:
        df = _f32_blob_frame()
        out = df.select(
            pl.col("b").cv.pipe(Pipeline().source("blob", dtype="f32")).sink("numpy")
        ).to_series()[0]
        assert out["dtype"] == "float32"


@plugin_required
class TestACanvasFromAnotherNodeIsChecked:
    """``source("contour").rasterize(shape=)`` takes the canvas node's planned size, and
    that size is a checked fact: an ``assert_shape`` it rests on is checked
    where it was written, so a wrong one fails the row naming the assertion,
    with or without optimizations (consolidation plan C0.2, C2)."""

    CONTOUR = {
        "exterior": [
            {"x": 1.0, "y": 1.0},
            {"x": 4.0, "y": 1.0},
            {"x": 4.0, "y": 4.0},
            {"x": 1.0, "y": 4.0},
        ],
        "holes": [],
        "is_closed": True,
    }

    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame({"img": [_png(6, 8)], "c": [self.CONTOUR]})

    @pytest.mark.parametrize("flags", [OptFlags.none(), OptFlags.all()])
    def test_a_wrong_upstream_assertion_is_reported_as_the_users(
        self, flags: OptFlags
    ) -> None:
        shape = pl.col("img").cv.pipe(
            Pipeline().source("image_bytes").assert_shape(height=10, width=10)
        )
        pipe = (
            Pipeline()
            .source("contour")
            .rasterize(shape=shape)
            .pad_to_size(height=10, width=10)
        )
        with pytest.raises(pl.exceptions.ComputeError) as err:
            self._frame().select(
                pl.col("c").cv.pipe(pipe).sink("numpy", opt_flags=flags)
            )
        assert "assert_shape(height=10, width=10) does not hold" in str(err.value)
        assert "contract" not in str(err.value)

    @pytest.mark.parametrize("flags", [OptFlags.none(), OptFlags.all()])
    def test_the_canvas_is_the_nodes_size(self, flags: OptFlags) -> None:
        shape = pl.col("img").cv.pipe(Pipeline().source("image_bytes"))
        pipe = (
            Pipeline()
            .source("contour")
            .rasterize(shape=shape)
            .pad_to_size(height=6, width=8)
        )
        out = self._frame().select(
            pl.col("c").cv.pipe(pipe).sink("numpy", opt_flags=flags)
        )
        assert out.to_series()[0]["shape"] == [6, 8, 1]

    def test_the_source_state_takes_the_nodes_planned_size(self) -> None:
        shape = pl.col("img").cv.pipe(
            Pipeline().source("image_bytes").assert_shape(height=10, width=10)
        )
        assert planned(Pipeline().source("contour").rasterize(shape=shape)).hw == (
            10,
            10,
        )


@plugin_required
@pytest.mark.parametrize("size", [-5, 0])
def test_assert_shape_refuses_a_non_positive_size(size: int) -> None:
    """Refused by Rust (``Declared::Size``), whichever spelling carries it."""
    refused = "positive int|cannot be negative"
    with pytest.raises(ValueError, match=refused):
        Pipeline().source("image_bytes").assert_shape(height=size)
    with pytest.raises(ValueError, match=refused):
        Pipeline().source("list").assert_shape(dims=[8, size, 3])


@plugin_required
def test_a_fully_known_input_is_validated_at_build() -> None:
    pipe = Pipeline().source("image_bytes").assert_shape(height=6, width=8, channels=3)
    with pytest.raises(ValueError, match="(?i)channel"):
        pipe.grayscale().channel_select(2)


@plugin_required
def test_repr_shows_only_written_assertions() -> None:
    pipe = Pipeline().source("image_bytes").resize(height=4, width=4)
    assert "assert_shape" not in repr(pipe)
    asserted = Pipeline().source("image_bytes").assert_shape(height=6).grayscale()
    assert repr(asserted).index("assert_shape") < repr(asserted).index("grayscale")


#: Builder calls whose refusal would rest on a fact the plan does not have:
#: an image's decoded channel count, a list column's rank, an operand's size.
#: A plan-time verdict is only ever one execution would also reach, so each of
#: these must build (PLANNER_SIZES_PLAN.md S2: validation over partially known
#: sizes says nothing about a size it does not know).
#:
#: Watched failing: raising every verdict over placeholder sizes (the
#: planner's former rank check) turns `channel_select`, `channel_swap` and
#: `crop` red;
#: `require_single_channel` reading an unknown channel count as 3 turns
#: `threshold` red; `broadcast_dims` refusing a known size against an unknown
#: one turns `add` red. `blur` over an unknown rank guards a path the planner
#: does not take (an input of unknown rank gives `validate` nothing to decide)
#: and is to be watched failing if it ever does.
_UNKNOWN_FACT_BUILDS = {
    "threshold, channels unknown": lambda: (
        Pipeline().source("image_bytes").threshold(128)
    ),
    "channel_swap, channels unknown": lambda: (
        Pipeline().source("image_bytes").channel_swap([2, 1, 0])
    ),
    "channel_select, channels unknown": lambda: (
        Pipeline().source("image_bytes").resize(height=4, width=4).channel_select(2)
    ),
    "blur, rank unknown": lambda: Pipeline().source("list", dtype="u8").blur(sigma=1),
    "crop, height unknown": lambda: (
        Pipeline()
        .source("image_bytes")
        .assert_shape(width=8, channels=3)
        .crop(top=4, left=0, height=4, width=8)
    ),
    "add, one operand's sizes unknown": lambda: (
        pl.col("i")
        .cv.pipe(Pipeline().source("image_bytes").resize(height=8, width=8))
        .add(pl.col("i").cv.pipe(Pipeline().source("image_bytes")))
        .sink("numpy")
    ),
}


@plugin_required
@pytest.mark.parametrize("case", sorted(_UNKNOWN_FACT_BUILDS))
def test_an_unknown_fact_is_never_refused_at_build(case: str) -> None:
    _UNKNOWN_FACT_BUILDS[case]()


def _build_error(build: Callable[[], object]) -> str | None:
    """The message a builder call refused with, or ``None`` if it built."""
    try:
        build()
    except ValueError as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Plan-time validation over what the plan knows (PLANNER_SIZES_PLAN.md S2)
# ---------------------------------------------------------------------------

#: Image ops every row of a rank-1 buffer refuses.
_IMAGE_OPS_ON_RANK_1: dict[str, Callable[[Pipeline], Pipeline]] = {
    "blur": lambda p: p.blur(sigma=1.0),
    "resize": lambda p: p.resize(height=4, width=4),
    "grayscale": lambda p: p.grayscale(),
    "threshold": lambda p: p.threshold(128),
    "channel_select": lambda p: p.channel_select(0),
    "perceptual_hash": lambda p: p.perceptual_hash(),
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
    """Every fact the plan knows is used: a known rank or size that no row can
    run on is refused while the pipeline is built, and a binary op's operands
    broadcast axis by axis. (These were strict gaps in test_known_gaps.py.)"""

    @pytest.mark.parametrize("op", sorted(_IMAGE_OPS_ON_RANK_1))
    def test_a_known_rank_refuses_an_image_op_at_build(self, op: str) -> None:
        # Each op refuses every row of a rank-1 buffer (confirmed when this was
        # a gap), so the known rank is refused where the op is written.
        apply = _IMAGE_OPS_ON_RANK_1[op]
        err = _build_error(lambda: apply(Pipeline().source("raw", dtype="u8")))
        assert err is not None, f"{op}() on a rank-1 buffer built; every row fails"
        assert f"{op}()" in err, err

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

    def test_broadcast_keeps_the_sizes_both_operands_know(self) -> None:
        def resized() -> LazyPipelineExpr:
            return pl.col("i").cv.pipe(
                Pipeline().source("image_bytes", dtype="u8").resize(height=8, width=8)
            )

        left, right = resized(), resized()
        assert planned(left).hw == (8, 8)  # the operands' sizes are planned
        out = planned(left.add(right))
        assert out.hw == (8, 8), f"a.add(b) planned {out}"

    def test_a_merge_or_mask_is_checked_against_the_nodes_it_reads(self) -> None:
        """PLANNER_SIZES_PLAN.md S3: the nodes an op reads are its inputs to
        the planner, so their planned sizes are checked and planned from."""

        def select(height: int | None, width: int | None) -> LazyPipelineExpr:
            pipe = Pipeline().source("image_bytes", dtype="u8")
            if height is not None and width is not None:
                pipe = pipe.resize(height=height, width=width)
            return pl.col("i").cv.pipe(pipe.channel_select(0))

        # Channels of different known sizes cannot be stacked.
        err = _build_error(lambda: select(8, 6).channel_merge(select(8, 5)))
        assert err is not None and "channel_merge()" in err, err
        # An input of unknown size takes H and W from the operands that know.
        assert planned(select(None, None).channel_merge(select(8, 6))).dims == (8, 6, 2)
        # A mask that cannot broadcast against the buffer it masks.
        image = pl.col("i").cv.pipe(_sized_image(8, 8))
        err = _build_error(lambda: image.apply_mask(select(4, 4)))
        assert err is not None and "apply_mask()" in err, err
        assert planned(image.apply_mask(select(8, 8))).dims == (8, 8, 3)

    @pytest.mark.parametrize(
        "read",
        [
            lambda image, contour: image.apply_mask(contour),
            lambda image, contour: image.add(contour),
            lambda image, contour: image.channel_select(0).channel_merge(contour),
        ],
        ids=["apply_mask", "add", "channel_merge"],
    )
    def test_a_node_read_in_the_wrong_domain_is_refused_at_build(self, read) -> None:
        """An operand's domain is checked where the op's own input is: at
        build. A contour operand used to plan, sink, and then fail every row
        with "accepts Buffer input but received Contour"."""
        image = pl.col("i").cv.pipe(_sized_image(8, 8))
        contour = pl.col("c").cv.pipe(Pipeline().source("contour"))
        err = _build_error(lambda: read(image, contour))
        assert err is not None and "contour domain" in err, err
