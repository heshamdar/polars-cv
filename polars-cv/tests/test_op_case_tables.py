"""Every table in ``tests/_op_cases.py`` is pinned to an authority.

``_op_cases.py`` holds the shared tables the schema-parity and contract sweeps
are parametrized from. ``OP_CASES`` itself has been completeness-asserted
against ``_chainable_pipeline_ops()`` since it was introduced
(``test_append_contract.py::test_op_case_table_is_complete``), and deleting
``test_sanitation.py``'s private ``_OP_BUILDERS`` in favour of it took the
contract sweep from 22 ops to all 71.

The five tables sitting directly below ``OP_CASES`` had no such assertion. Each
is an axis some sweep iterates, so a table that under-lists does not fail — the
sweep just gets smaller, silently, which is the failure mode this repo has
shipped nine times over. This module gives each one the strongest tie its fact
supports:

============================ ==========================================
table                        authority
============================ ==========================================
``HISTOGRAM_OUTPUTS``        ``_types.HistogramOutput``
``COLOR_SPACES``             ``_types.ColorSpace``
``IMAGE_ENCODER_SINKS``      ``_types.SinkFormat``, partitioned
``SINGLE_CHANNEL_OPS``       the engine's own runtime precondition
``EXTRA_CASES``              ``OP_CASES`` (structure, not membership)
============================ ==========================================

Four of the five run without the compiled plugin. ``SINGLE_CHANNEL_OPS`` cannot:
its fact is not declared anywhere at plan time — the failure is a runtime
"requires single-channel input" from the kernel — so the only honest way to
check the list is to ask the kernels.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv._types import ColorSpace, HistogramOutput, SinkFormat

from ._op_cases import (
    BUFFER,
    COLOR_SPACES,
    EXTRA_CASES,
    HISTOGRAM_OUTPUTS,
    IMAGE_ENCODER_SINKS,
    OP_CASES,
    SINGLE_CHANNEL_OPS,
    base_pipeline,
    buffer_ops,
    build_case,
)
from .conftest import make_image_png, plugin_required


def test_histogram_outputs_are_the_whole_enum() -> None:
    """Every ``histogram(output=)`` mode is swept.

    Each mode lands in a different (rank, dtype, domain) corner, so a mode
    missing from the table is a corner with no plan-vs-exec test rather than a
    smaller sweep that still means something.
    """
    assert set(HISTOGRAM_OUTPUTS) == {member.value for member in HistogramOutput}


def test_color_spaces_are_the_whole_enum() -> None:
    """Every ``convert_color`` target is swept.

    The table exists for the ``StripProcessRestore`` channel rule, whose whole
    point is that different targets have different channel counts — so the one
    left out is exactly the one worth having.
    """
    assert set(COLOR_SPACES) == {member.value for member in ColorSpace}


#: Sink formats that do **not** re-encode a buffer through an image codec, and
#: why. Together with ``IMAGE_ENCODER_SINKS`` this must exhaust ``SinkFormat``:
#: the point is that adding a format forces a decision about which side it is
#: on, rather than defaulting to "not a codec" by omission.
NON_CODEC_SINKS: dict[str, str] = {
    "numpy": "zero-copy struct, no re-encode",
    "torch": "zero-copy struct, no re-encode",
    "blob": "the self-describing VIEW protocol, no codec preconditions",
    "array": "Polars Array, fixed shape",
    "list": "Polars nested List",
    "native": "the domain's Polars-native type; an error for buffers",
}


def test_image_encoder_sinks_partition_the_sink_formats() -> None:
    """Every ``SinkFormat`` is either an image codec or explicitly not one.

    ``IMAGE_ENCODER_SINKS`` drives ``test_schema_parity_encoder_sinks.py``,
    which is the only sweep that exercises the codec dtype preconditions
    (jpeg/webp are 8-bit only; png takes u8/u16; tiff refuses some
    dtype/channel pairs) — preconditions checked when the encoder runs, not
    when the query is planned. A new codec format left off the list would ship
    with none of that covered and nothing would say so.
    """
    codecs = set(IMAGE_ENCODER_SINKS)
    non_codecs = set(NON_CODEC_SINKS)
    overlap = codecs & non_codecs
    assert not overlap, f"formats claimed by both halves: {sorted(overlap)}"

    declared = {member.value for member in SinkFormat}
    assert codecs | non_codecs == declared, (
        f"unclassified sink formats: {sorted(declared - codecs - non_codecs)}; "
        f"classified formats that no longer exist: "
        f"{sorted((codecs | non_codecs) - declared)}"
    )


def test_non_codec_reasons_are_written_down() -> None:
    """An exemption without a reason is an exemption nobody can review."""
    blank = [name for name, why in NON_CODEC_SINKS.items() if not why.strip()]
    assert not blank, f"NON_CODEC_SINKS entries without a reason: {blank}"


def test_extra_cases_name_ops_that_have_a_base_case() -> None:
    """``EXTRA_CASES`` adds branches to ``OP_CASES``, so it cannot outlive it.

    Kept separate from ``OP_CASES`` so the one-case-per-op completeness check
    stays strict; that separation is also what lets an entry here go stale
    unnoticed after a rename.
    """
    stale = sorted({op for op, _, _ in EXTRA_CASES if OP_CASES.get(op) is None})
    assert not stale, (
        f"extra cases for ops with no callable base case: {stale}. Either the "
        f"op was renamed or its OP_CASES entry became an exemption."
    )


def test_extra_cases_reach_a_branch_the_base_case_does_not() -> None:
    """An extra case that repeats its base case exercises nothing extra.

    The table's stated purpose is "the interesting behaviour is in a branch the
    single case does not reach", so an entry whose kwargs and domain match the
    base case is coverage that reads as coverage without being any.
    """
    redundant = []
    seen: set[tuple[str, str, str]] = set()
    duplicates = []
    for op, domain, kwargs in EXTRA_CASES:
        base = OP_CASES.get(op)
        if base is not None and (domain, kwargs) == base:
            redundant.append(op)
        key = (op, domain, repr(sorted(kwargs.items(), key=str)))
        if key in seen:
            duplicates.append(op)
        seen.add(key)

    assert not redundant, f"extra cases identical to their OP_CASES entry: {redundant}"
    assert not duplicates, f"extra cases repeated verbatim: {duplicates}"


#: How the engine spells its single-channel precondition. Matched loosely: the
#: kernels word it per-op ("Erode requires single-channel input, but got 3
#: channels"), so the shared part is what is matched on.
_SINGLE_CHANNEL_REFUSAL = "single-channel"


@plugin_required
def test_single_channel_ops_are_exactly_the_ops_that_refuse_three_channels() -> None:
    """The table must equal what the kernels actually refuse.

    Nothing rejects a three-channel pipeline for these at build or plan time —
    the failure is a runtime refusal from the kernel — so there is no plan-time
    authority to derive the list from. Asking the kernels is the next strongest
    thing, and it fails in both directions: an op that starts refusing three
    channels without joining the table (so the sweeps that use the table start
    erroring), and an op that stops refusing while the table still puts a
    ``grayscale()`` in front of it (so the sweep quietly tests something else).
    """
    df = pl.DataFrame({"img": [make_image_png(height=16, width=24, channels=3)]})

    refusing: set[str] = set()
    for op in buffer_ops():
        pipe = build_case(op)
        try:
            df.lazy().with_columns(
                out=pl.col("img").cv.pipe(pipe).sink("blob")
            ).collect()
        except Exception as e:  # noqa: BLE001 - any refusal is read, then matched
            if _SINGLE_CHANNEL_REFUSAL in str(e):
                refusing.add(op)

    assert refusing == set(SINGLE_CHANNEL_OPS), (
        f"ops refusing three channels but not in SINGLE_CHANNEL_OPS: "
        f"{sorted(refusing - set(SINGLE_CHANNEL_OPS))}; "
        f"ops in SINGLE_CHANNEL_OPS that accept three channels: "
        f"{sorted(set(SINGLE_CHANNEL_OPS) - refusing)}"
    )


@plugin_required
def test_the_single_channel_probe_can_see_a_refusal() -> None:
    """The probe above is only as good as its ability to observe a refusal.

    A sweep that catches every exception and matches on a substring passes
    vacuously if the substring stops appearing — so one known-refusing op is
    driven through the same path and asserted to raise, naming the precondition.
    """
    df = pl.DataFrame({"img": [make_image_png(height=16, width=24, channels=3)]})
    pipe = base_pipeline(BUFFER).erode(ksize=3)
    with pytest.raises(Exception, match=_SINGLE_CHANNEL_REFUSAL):
        df.lazy().with_columns(out=pl.col("img").cv.pipe(pipe).sink("blob")).collect()
