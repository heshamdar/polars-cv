"""The image-encoder sinks check their dtype precondition too late.

``png``, ``jpeg``, ``webp`` and ``tiff`` each accept only some buffer dtypes:
jpeg and webp are 8-bit formats, png takes u8 or u16, tiff rejects certain
dtype/channel combinations, and all four need a buffer shaped like an image
rather than a flat 1-D run. Every one of those preconditions is decidable at
planning time — ``OutputSpec`` carries ``expected_dtype`` and
``expected_ndim``, which is exactly what the encoder later complains about —
and none of them is checked there. So::

    Pipeline().source("image_bytes").scale(factor=2.0)   # promotes to f32
    ...sink("jpeg")

plans as ``Binary`` and then dies inside ``collect()`` with "JPEG is an 8-bit
format but the image is F32".

That is the failure mode ``test_sink_contract.py`` names as the one thing that
must never happen: *"planning succeeding and execution then failing"*. It was
invisible to that file because its single base pipeline ends in ``cast("u8")``,
and u8 satisfies all four encoders.

The contract test below is marked ``xfail(strict=True)``. Strict matters: the
day the planner starts rejecting these combinations the test passes
unexpectedly, pytest fails the run, and this file has to be deleted. The
marker cannot outlive the defect.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import DType
from tests._op_cases import IMAGE_ENCODER_SINKS
from tests._schema_parity import Outcome, plan_or_reject
from tests.conftest import make_image_png, plugin_required

HEIGHT, WIDTH, CHANNELS = 32, 24, 3

#: Dtypes reachable by casting a decoded u8 image. Every one of them is known
#: to the planner before a byte of data moves.
CAST_DTYPES: tuple[str, ...] = tuple(d.value for d in DType)


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {"img": [make_image_png(HEIGHT, WIDTH, CHANNELS, seed=1)]},
        schema={"img": pl.Binary},
    )


def _cell(dtype: str, sink: str, *, flatten: bool = False):
    pipe = (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=HEIGHT, width=WIDTH, channels=CHANNELS)
        .cast(dtype)
    )
    if flatten:
        # Rank 1: a legal buffer, and not a thing any image codec can encode.
        pipe = pipe.reshape([HEIGHT * WIDTH * CHANNELS])
    return plan_or_reject(_df(), lambda: pl.col("img").cv.pipe(pipe).sink(sink))


@plugin_required
@pytest.mark.parametrize("sink", IMAGE_ENCODER_SINKS)
def test_u8_is_encodable_by_every_image_sink(sink: str) -> None:
    """The baseline the rest of the suite leans on: u8 always encodes.

    Without this, the ``planned_dtype == "u8"`` carve-out in
    ``test_schema_parity_ops.py`` could be hiding a real failure rather than a
    known one.
    """
    result = _cell("u8", sink)
    assert result.outcome is Outcome.OK, (
        f"u8 -> {sink} should encode cleanly, got {result.outcome.name}: "
        f"{result.reason}"
    )
    assert result.planned == pl.Binary


@plugin_required
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known gap: the image-encoder sinks enforce their dtype precondition "
        "in the encoder instead of the planner, so a non-u8 buffer plans as "
        "Binary and raises at collect(). Delete this marker when the planner "
        "rejects the combination at plan time."
    ),
)
@pytest.mark.parametrize("sink", IMAGE_ENCODER_SINKS)
def test_image_encoder_sinks_decide_at_plan_time(sink: str) -> None:
    """For every dtype: reject while planning, or execute — never both.

    This asserts the *relationship* rather than a blessed list of dtypes, so it
    stays correct as encoder support changes: whether png grows f32 support or
    loses u16 support, the requirement is only that the decision is made before
    the schema is published.
    """
    plan_then_raise: list[str] = []
    for dtype in CAST_DTYPES:
        for flatten in (False, True):
            try:
                _cell(dtype, sink, flatten=flatten)
            except AssertionError:
                # plan_or_reject raises exactly when the planner published a
                # dtype and execution then failed to produce it.
                plan_then_raise.append(f"{dtype}{'@rank1' if flatten else ''}")

    assert not plan_then_raise, (
        f"'{sink}' sink published a schema and then failed at collect() for "
        f"{plan_then_raise}. Both the buffer dtype and its rank are on the "
        f"OutputSpec at plan time, so this belongs in dtype_for_output "
        f"(graph/decode.rs), alongside the ('buffer', 'native') arm that "
        f"already refuses there."
    )


@plugin_required
def test_the_gap_is_a_late_check_not_a_wrong_dtype() -> None:
    """Pin *which* half of the contract is broken.

    The planner is not lying about the dtype — every encoder sink is Binary and
    would be Binary if it succeeded. What it fails to do is refuse. Recording
    that distinction keeps the finding from being "read" as a dtype bug and
    fixed in the wrong place.
    """
    result = None
    raised = False
    try:
        result = _cell("f32", "jpeg")
    except AssertionError as exc:
        raised = True
        assert "planner published Binary" in str(exc), (
            f"expected a plan-then-raise on Binary, got: {exc}"
        )

    assert raised, (
        "f32 -> jpeg no longer plans-then-raises; if the planner now rejects "
        "it, delete this file along with the xfail above"
    )
    assert result is None
