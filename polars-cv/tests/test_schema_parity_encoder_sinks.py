"""The image-encoder sinks decide their preconditions at planning time.

``png``, ``jpeg``, ``webp`` and ``tiff`` each accept only some buffer dtypes:
jpeg and webp are 8-bit formats, png carries 8- or 16-bit samples, tiff has a
colour type per (dtype, channels) pair and stores floats greyscale-only. All
four also need a buffer shaped like an image rather than a flat 1-D run.

None of that requires looking at the pixels — it follows from the buffer's
*description*, which is exactly what `OutputSpec` carries. It used to be
checked in the encoder anyway, so::

    Pipeline().source("image_bytes", dtype="u8").cast("f32")...sink("jpeg")

planned as ``Binary`` and then died inside ``collect()``. That is the failure
``test_sink_contract.py`` names as the one thing that must never happen —
"planning succeeding and execution then failing" — and it was invisible there
because its single base pipeline ends in ``cast("u8")``, which every codec
accepts.

``ImageCodec::check_support`` (view-buffer, ``interop/image.rs``) is now the
one table, read by the planner (``dtype_for_output``), by ``encode_sink``, and
by the encoders themselves. These tests pin the contract from the outside.
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

#: Every dtype a `cast()` can reach. All of them are known to the planner
#: before a byte of data moves.
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

    Without this, the ``encodable_by_image_codec`` carve-out the broad sweeps
    apply could be hiding a real failure rather than a known-irrelevant cell —
    and a check that rejected everything would look like a fix.
    """
    result = _cell("u8", sink)
    assert result.outcome is Outcome.OK, (
        f"u8 -> {sink} should encode cleanly, got {result.outcome.name}: "
        f"{result.reason}"
    )
    assert result.planned == pl.Binary


@plugin_required
@pytest.mark.parametrize("sink", IMAGE_ENCODER_SINKS)
def test_image_encoder_sinks_decide_at_plan_time(sink: str) -> None:
    """For every dtype and rank: reject while planning, or execute — never both.

    This asserts the *relationship* rather than a blessed list of dtypes, so it
    stays correct as codec support changes: whether png grows f32 support or
    tiff loses a colour type, the requirement is only that the decision is made
    before the schema is published.
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
        f"OutputSpec at plan time — route the check through "
        f"ImageCodec::check_support in dtype_for_output."
    )


@plugin_required
@pytest.mark.parametrize(
    ("dtype", "sink"),
    [
        ("f32", "jpeg"),
        ("f32", "png"),
        ("f32", "webp"),
        ("f32", "tiff"),  # float TIFF is greyscale-only; this base is 3-channel
        ("u16", "jpeg"),
        ("u16", "webp"),
        ("i32", "png"),
    ],
)
def test_unencodable_combinations_are_refused_before_any_data_moves(
    dtype: str, sink: str
) -> None:
    """The refusal must happen at plan time, with a message naming the fix."""
    result = _cell(dtype, sink)
    assert result.outcome is Outcome.REJECTED_AT_PLAN, (
        f"{dtype} -> {sink} should be refused while planning, got "
        f"{result.outcome.name} (planned {result.planned!r})"
    )
    assert dtype.upper() in (result.reason or "").upper(), (
        f"the refusal should name the offending dtype: {result.reason}"
    )


@plugin_required
def test_u16_png_is_not_over_rejected() -> None:
    """The check must refuse only what cannot encode.

    PNG carries 16-bit samples, so a u16 buffer has to keep working — a check
    that rejected it would make the whole suite greener while breaking users.
    """
    result = _cell("u16", "png")
    assert result.outcome is Outcome.OK, f"u16 -> png was refused: {result.reason}"


@plugin_required
def test_float_tiff_is_allowed_when_single_channel() -> None:
    """TIFF stores f32/f64 greyscale, so channel count decides, not dtype alone."""
    single = (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=HEIGHT, width=WIDTH, channels=CHANNELS)
        .grayscale()
        .cast("f32")
    )
    result = plan_or_reject(_df(), lambda: pl.col("img").cv.pipe(single).sink("tiff"))
    assert result.outcome is Outcome.OK, (
        f"single-channel f32 -> tiff should encode, got {result.outcome.name}: "
        f"{result.reason}"
    )


@plugin_required
def test_a_promoted_but_unresolved_dtype_is_still_refused() -> None:
    """The planner does not need the exact dtype, only enough of it.

    ``source("image_bytes")`` has no dtype at plan time — a PNG decodes u8 or
    u16, a TIFF f32 or f64 — and the float-promoting ops keep it unresolved
    because the answer is f32 *or* f64 depending on the input. That used to
    read as "unknown", which `check_planned` treats as permission, so
    `scale(...)` into a JPEG sink planned `Binary` and died in the encoder.

    `PlannedDType::SomeFloat` carries the part that *is* known. No float is an
    8-bit sample whichever float it is, so the sink is refused while planning.
    """
    pipe = Pipeline().source("image_bytes").scale(factor=2.0)
    assert pipe.output_dtype() == "auto", (
        "the Python planner still reports 'auto'; the refinement happens in "
        "Rust's resolved_output_specs, which has the input column"
    )

    result = plan_or_reject(_df(), lambda: pl.col("img").cv.pipe(pipe).sink("jpeg"))
    assert result.outcome is Outcome.REJECTED_AT_PLAN, (
        f"expected a plan-time refusal, got {result.outcome.name} "
        f"(planned {result.planned!r})"
    )
    assert "floating point" in (result.reason or ""), result.reason


@plugin_required
def test_an_unresolved_dtype_that_could_encode_is_not_refused() -> None:
    """Permissiveness is preserved where the constraint does not bite.

    The same unresolved source *without* a promoting op could still decode to
    u8, so PNG must plan. A check that refused here would break every correct
    `auto`-dtype pipeline — the failure mode that made the naive fix unsafe.
    """
    for pipe, sink in [
        (Pipeline().source("image_bytes"), "png"),
        (Pipeline().source("image_bytes"), "jpeg"),
        (Pipeline().source("image_bytes").grayscale(), "png"),
        # A float that TIFF *can* carry: greyscale.
        (Pipeline().source("image_bytes").grayscale().scale(factor=2.0), "tiff"),
    ]:
        result = plan_or_reject(
            _df(), lambda p=pipe, s=sink: pl.col("img").cv.pipe(p).sink(s)
        )
        assert result.outcome is Outcome.OK, (
            f"{sink} was refused for an unresolved dtype that could encode: "
            f"{result.reason}"
        )
