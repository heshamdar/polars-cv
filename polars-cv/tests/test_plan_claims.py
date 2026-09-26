"""A plan fact that rests on a *claim* must never let the optimizer change output.

A declared dtype or size is either checked where the data arrives or carried as
a declaration, so a pass that trusts it (identity elimination) cannot turn a
working query into a failing one. Each case runs the same pipeline with every
optimization off and on, at the user-facing entry point (``.sink``), and
requires the same result. Consolidation plan C0.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import OptFlags, Pipeline
from tests._plan_view import planned
from tests.conftest import plugin_required


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
#: Watched failing: raising every `check_rank` verdict over its placeholder
#: sizes turns the `channel_select`, `channel_swap` and `crop` cases red. The
#: other three guard paths that do not exist yet — `threshold`'s channel check
#: happens to accept the placeholder 1, an unknown rank is never validated, and
#: a binary op is never validated — and are to be watched failing when S1/S2
#: add them.
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
