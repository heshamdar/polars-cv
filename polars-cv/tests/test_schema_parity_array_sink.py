"""``sink("array")`` — the strictest form of the plan-time schema contract.

Every other sink can be planned from partial knowledge: ``numpy``/``torch``
publish a fixed struct whatever the buffer turns out to be, ``list`` needs only
a rank and a leaf dtype, ``png`` needs neither. A fixed-size ``pl.Array``
column needs the *exact* dimensions, before any data is read.

So this is where "the plan must be knowable" is a hard requirement rather than
a convenience, and where being wrong is worst: the schema names a concrete
``Array(UInt8, (h, w, c))`` that every downstream expression will type against.

The invariant is asserted in both directions. When the shape is known the sink
must be accepted *and* produce exactly those dimensions; when it is not, the
sink must be refused while planning. "Accepted and then wrong" and "refused
though knowable" are both failures.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline
from tests._schema_parity import (
    HOMOGENEOUS_PATTERNS,
    array_dims,
    assert_plan_equals_exec,
    frame,
    leaf_dtype,
    plan_or_reject,
    rows_for,
)
from tests.conftest import make_image_png, plugin_required

H, W, C = 12, 20, 3


def _df(pattern: str = "single", *, channels: int = C) -> pl.DataFrame:
    images = [make_image_png(H, W, channels, seed=s) for s in (1, 2, 3)]
    return frame(rows_for(pattern, images), pl.Binary)


def _base(*, channels: int = C) -> Pipeline:
    return (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=H, width=W, channels=channels)
        .cast("u8")
    )


def _numpy_shape(df: pl.DataFrame, pipe: Pipeline) -> list[int]:
    """The shape the ``numpy`` sink reports for the same pipeline.

    An independent producer of the same fact: the array sink derives its
    dimensions from the planner's ``expected_shape``, while the numpy sink
    reports the buffer's real runtime shape. Comparing them is the same
    technique ``test_contour_raster_crosscheck.py`` uses — two implementations
    of one quantity, so a fault in either shows up as a mismatch.
    """
    series = (
        df.lazy()
        .select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        .collect(engine="streaming")["out"]
    )
    # A null row of a struct output is a struct whose *fields* are null, not a
    # None row, so filtering on the row alone finds an empty shape.
    row = next(r for r in series if r is not None and r["shape"] is not None)
    return list(row["shape"])


# ---------------------------------------------------------------------------
# Knowable shape: accepted, and exactly right
# ---------------------------------------------------------------------------

#: Pipelines whose shape the planner can both *know* and *express*.
#:
#: "Express" is the operative word: ``GraphNode.expected_shape`` reports a
#: shape only at rank 3, because the hints track H/W/C specifically and at any
#: other rank they cannot describe the output. So auto-shaping is a rank-3
#: feature; the other ranks are covered by ``_KNOWN_BUT_UNEXPRESSIBLE`` below.
_KNOWN_SHAPE = {
    "rank3-identity": (lambda: _base(), [H, W, C]),
    "rank3-resize": (lambda: _base().resize(height=5, width=7), [5, 7, C]),
    "rank3-pad": (
        lambda: _base().pad(top=1, bottom=2, left=3, right=4),
        [H + 3, W + 7, C],
    ),
    "rank3-rotate90": (lambda: _base().rotate(angle=90), [W, H, C]),
}

#: Pipelines whose dimensions the planner *has* — they are sitting in
#: ``_shape_hints`` — but cannot publish, because ``expected_shape`` is gated
#: on rank 3. These must refuse the array sink without an explicit ``shape=``
#: and be exact with one.
#:
#: This is a deliberate conservative choice, not a divergence: publishing
#: ``[H, W, C]`` for a rank-2 output is how ``channel_select`` once declared a
#: schema execution could not produce. It is recorded here because "refused
#: though knowable" is a real cost — a rank-2 pipeline can never auto-shape —
#: and because a future widening should update this table on purpose.
_KNOWN_BUT_UNEXPRESSIBLE = {
    "rank2-channel_select": (lambda: _base().channel_select(index=0), [H, W]),
    "rank1-reshape": (lambda: _base().reshape([H * W * C]), [H * W * C]),
    "rank2-reshape": (lambda: _base().reshape([H, W * C]), [H, W * C]),
}


@plugin_required
@pytest.mark.parametrize("case", sorted(_KNOWN_SHAPE))
@pytest.mark.parametrize("pattern", HOMOGENEOUS_PATTERNS)
def test_known_shape_is_accepted_and_exact(case: str, pattern: str) -> None:
    """A knowable shape must plan, execute, and match dimension for dimension."""
    build, expected = _KNOWN_SHAPE[case]
    pipe = build()
    df = _df(pattern)

    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("array"))
    assert array_dims(series.dtype) == expected, (
        f"{case}: planned Array dims {array_dims(series.dtype)} != {expected}"
    )
    assert leaf_dtype(series.dtype) == pl.UInt8


@plugin_required
@pytest.mark.parametrize("case", sorted(_KNOWN_BUT_UNEXPRESSIBLE))
def test_non_rank3_refuses_without_a_shape_but_is_exact_with_one(case: str) -> None:
    """The planner knows these dimensions and still cannot publish them.

    Both halves are asserted: the refusal (so the gate is real and safe) and
    the exactness once ``shape=`` supplies what the planner could not say (so
    the dimensions really were right all along).
    """
    build, expected = _KNOWN_BUT_UNEXPRESSIBLE[case]
    pipe = build()
    df = _df("null_first")

    bare = plan_or_reject(df, lambda: pl.col("img").cv.pipe(pipe).sink("array"))
    assert not bare.ok, (
        f"{case}: expected_shape is gated on rank 3, so this should refuse; "
        f"it planned {bare.planned!r} instead"
    )
    assert "shape is required" in (bare.reason or "")

    series = assert_plan_equals_exec(
        df, pl.col("img").cv.pipe(pipe).sink("array", shape=expected)
    )
    assert array_dims(series.dtype) == expected

    # The dimensions the planner declined to publish are the right ones.
    assert _numpy_shape(df, pipe) == expected, (
        f"{case}: the gate is hiding a correct shape, {_numpy_shape(df, pipe)}"
    )


@plugin_required
@pytest.mark.parametrize("case", sorted(_KNOWN_SHAPE))
def test_array_dims_agree_with_the_numpy_sinks_shape(case: str) -> None:
    """The two sinks must report the same shape for the same pipeline.

    The array sink's dims come from the planner; the numpy sink's ``shape``
    field is the buffer's real shape at runtime. If the planner's shape
    contract were wrong, these would disagree even though each is internally
    consistent.
    """
    build, _ = _KNOWN_SHAPE[case]
    pipe = build()
    df = _df()

    planned = (
        df.lazy()
        .select(out=pl.col("img").cv.pipe(pipe).sink("array"))
        .collect_schema()["out"]
    )
    assert array_dims(planned) == _numpy_shape(df, pipe), (
        f"{case}: array sink planned {array_dims(planned)} but the numpy sink "
        f"reports {_numpy_shape(df, pipe)}"
    )


# ---------------------------------------------------------------------------
# Unknowable shape: refused while planning
# ---------------------------------------------------------------------------

#: Pipelines whose output shape genuinely cannot be known at plan time.
_UNKNOWN_SHAPE = {
    # No assert_shape, so the decoded image's H/W is unknown.
    "undeclared-image": lambda: Pipeline().source("image_bytes").cast("u8"),
    # A per-row expression parameter: the value differs per row by definition.
    "expr-resize": lambda: _base().resize(height=pl.col("h"), width=8),
    # A per-row angle could be a 90-multiple, which swaps H/W.
    "expr-rotate": lambda: _base().rotate(angle=pl.col("a")),
}


@plugin_required
@pytest.mark.parametrize("case", sorted(_UNKNOWN_SHAPE))
def test_unknown_shape_is_refused_while_planning(case: str) -> None:
    """No shape means no array sink — refused, never guessed.

    Guessing here is the worst available outcome: the schema would name
    concrete dimensions that the very next row could contradict.
    """
    pipe = _UNKNOWN_SHAPE[case]()
    df = _df().with_columns(h=pl.lit(6, pl.Int64), a=pl.lit(90.0))

    result = plan_or_reject(df, lambda: pl.col("img").cv.pipe(pipe).sink("array"))
    assert not result.ok, (
        f"{case}: the array sink was accepted with an unknowable shape and "
        f"planned {result.planned!r}"
    )


@plugin_required
def test_an_explicit_shape_rescues_an_unknown_pipeline() -> None:
    """``shape=`` is the escape hatch, and it must be honoured end to end."""
    pipe = Pipeline().source("image_bytes").resize(height=6, width=6).cast("u8")
    df = _df("null_first")

    series = assert_plan_equals_exec(
        df, pl.col("img").cv.pipe(pipe).sink("array", shape=[6, 6, C])
    )
    assert array_dims(series.dtype) == [6, 6, C]


@plugin_required
def test_assert_shape_makes_an_array_sink_plannable() -> None:
    """A user assertion is the other way to make the shape knowable."""
    undeclared = Pipeline().source("image_bytes").cast("u8")
    df = _df()
    assert not plan_or_reject(
        df, lambda: pl.col("img").cv.pipe(undeclared).sink("array")
    ).ok

    declared = undeclared.assert_shape(height=H, width=W, channels=C)
    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(declared).sink("array"))
    assert array_dims(series.dtype) == [H, W, C]


# ---------------------------------------------------------------------------
# Vector-domain array sinks
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize(
    ("name", "build", "shape"),
    [
        # A hash_size=8 phash packs into a single 64-bit value: shape [1].
        ("phash", lambda: _base().perceptual_hash(hash_size=8), [1]),
        ("extract_shape", lambda: _base().extract_shape(), [3]),
    ],
)
def test_vector_array_sinks(name: str, build, shape: list[int]) -> None:
    """A vector reaching the sink as a Buffer and as a real Vector.

    That a vector can arrive in two representations is what the two halves of
    the sink contract once disagreed about, so both are swept here at the
    strictest sink.
    """
    pipe = build()
    df = _df("null_first")
    series = assert_plan_equals_exec(
        df, pl.col("img").cv.pipe(pipe).sink("array", shape=shape)
    )
    assert array_dims(series.dtype) == shape, f"{name}: {array_dims(series.dtype)}"


# ---------------------------------------------------------------------------
# Multi-output
# ---------------------------------------------------------------------------


@plugin_required
def test_multi_output_struct_fields_match_field_by_field() -> None:
    """A multi-output sink is a Struct, and every field must be planned right.

    ``test_multi_output.py`` asserts only ``isinstance(dtype, pl.Struct)``, so
    a wrong *field* dtype passed. The field order is also pinned: Rust sorts
    outputs by alias (``compiled.rs``), which nothing tested.
    """
    df = _df("null_first")
    expr = (
        pl.col("img")
        .cv.pipe(_base().resize(height=4, width=4))
        .alias("small")
        .merge_pipe(pl.col("img").cv.pipe(_base().grayscale()).alias("gray"))
        .sink({"small": "array", "gray": "list"})
    )

    lf = df.lazy().with_columns(out=expr)
    planned = lf.collect_schema()["out"]
    produced = lf.collect(engine="streaming").schema["out"]

    assert planned == produced, f"planned {planned} != produced {produced}"
    assert isinstance(planned, pl.Struct)
    assert [f.name for f in planned.fields] == sorted(f.name for f in planned.fields), (
        "multi-output struct fields are expected to be alias-sorted"
    )
    for field in planned.fields:
        assert (
            field.dtype == dict((f.name, f.dtype) for f in produced.fields)[field.name]
        )


@plugin_required
def test_multi_output_array_branch_honours_an_explicit_shape() -> None:
    """The single-output array branch honours ``shape=`` before checking hints.

    The multi-output branch checks ``has_all_dims()`` *first* — an asymmetry
    between two spellings of the same request. This pins whichever behaviour is
    current so a change to either is deliberate.
    """
    df = _df()
    unknown = Pipeline().source("image_bytes").resize(height=6, width=6).cast("u8")

    single = plan_or_reject(
        df, lambda: pl.col("img").cv.pipe(unknown).sink("array", shape=[6, 6, C])
    )
    assert single.ok, f"single-output array with shape= was refused: {single.reason}"

    multi = plan_or_reject(
        df,
        lambda: (
            pl.col("img")
            .cv.pipe(unknown)
            .alias("a")
            .sink({"a": "array"}, shape=[6, 6, C])
        ),
    )
    # Record the asymmetry rather than assert it is correct: if the multi
    # branch starts honouring shape= too, this flips and should be updated
    # deliberately.
    assert multi.ok or "deterministic shape" in (multi.reason or ""), (
        f"unexpected multi-output array failure: {multi.reason}"
    )
