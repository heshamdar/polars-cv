"""Plan == exec across every source format, element dtype and channel count.

Three axes that the existing coverage barely touched:

* **Source format.** ``test_plan_matches_data.py`` covered five of the eight.
  ``auto`` — the *default* — had no plan-vs-exec test at all, which matters
  more than the others because its decode path is chosen at runtime from the
  column's Polars dtype, so it is the one source whose plan is made with the
  least information.
* **Element dtype.** Only ``u8``, ``u16``, ``f32`` and ``f64`` were ever
  compared plan-vs-exec, leaving ``i8 i16 u32 i32 u64 i64`` — and with them
  every dtype rule that can produce them — unchecked against data.
* **Channel count.** 1, 2, 3 and 4 all reach different arms of the alpha
  rules. Two channels (``GrayA``, what ``StripProcessRestore`` yields from
  RGBA) had never been fed through a sink.

Both vocabularies are completeness-asserted against the enums in
``polars_cv._types``, so adding a source format or a dtype without extending
this file fails here rather than silently narrowing the sweep.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import DType, SinkFormat, SourceFormat
from polars_cv.geometry.schemas import CONTOUR_SCHEMA
from tests._op_cases import IMAGE_ENCODER_SINKS
from tests._schema_parity import (
    HOMOGENEOUS_PATTERNS,
    ParityResult,
    assert_not_vacuous,
    assert_plan_equals_exec,
    encodable_by_image_codec,
    frame,
    leaf_dtype,
    plan_or_reject,
    rows_for,
)
from tests.conftest import make_image_png, plugin_required

SINKS: tuple[str, ...] = tuple(sorted(m.value for m in SinkFormat))
ALL_DTYPES: tuple[str, ...] = tuple(d.value for d in DType)

#: numpy's spelling of each ``DType``. This is a mirror, so it is pinned:
#: ``test_dtype_axis_covers_the_vocabulary`` fails if ``DType`` grows a member
#: this map does not have.
NUMPY_FOR_DTYPE: dict[str, type] = {
    "u8": np.uint8,
    "i8": np.int8,
    "u16": np.uint16,
    "i16": np.int16,
    "u32": np.uint32,
    "i32": np.int32,
    "u64": np.uint64,
    "i64": np.int64,
    "f32": np.float32,
    "f64": np.float64,
}

#: Polars leaf dtype per ``DType``, for building list/array source columns.
POLARS_FOR_DTYPE: dict[str, pl.DataType] = {
    "u8": pl.UInt8,
    "i8": pl.Int8,
    "u16": pl.UInt16,
    "i16": pl.Int16,
    "u32": pl.UInt32,
    "i32": pl.Int32,
    "u64": pl.UInt64,
    "i64": pl.Int64,
    "f32": pl.Float32,
    "f64": pl.Float64,
}

H, W, C = 4, 3, 2


def _sweep(
    df: pl.DataFrame,
    pipe: Pipeline,
    label: str,
    *,
    column: str = "img",
    sinks: tuple[str, ...] = SINKS,
) -> dict[str, ParityResult]:
    # See encodable_by_image_codec: the image codecs' dtype/rank preconditions
    # are a separate, separately-pinned finding, not this sweep's subject.
    if not encodable_by_image_codec(pipe):
        sinks = tuple(s for s in sinks if s not in IMAGE_ENCODER_SINKS)
    results = {
        sink: plan_or_reject(
            df, lambda s=sink: pl.col(column).cv.pipe(pipe).sink(s), name="out"
        )
        for sink in sinks
    }
    assert_not_vacuous(results, label)
    return results


# ---------------------------------------------------------------------------
# Completeness: both axes come from the real vocabularies
# ---------------------------------------------------------------------------


def test_source_axis_covers_the_vocabulary() -> None:
    """Every ``SourceFormat`` is exercised by a test that actually exists here.

    The first version of this guard compared a hand-written set against the
    enum, which says nothing about whether the tests are still present — and it
    passed happily while five of them were deleted by an editing mistake.
    Naming the covering test and resolving it in this module's namespace is
    what makes the check bite: delete or rename a test and this fails.
    """
    covered: dict[SourceFormat, str] = {
        SourceFormat.AUTO: "test_auto_source_on_a_binary_column",
        SourceFormat.IMAGE_BYTES: "test_image_bytes_source_every_channel_count",
        SourceFormat.BLOB: "test_blob_source_round_trips",
        SourceFormat.RAW: "test_raw_source_every_dtype",
        SourceFormat.FILE_PATH: "test_file_path_source",
        SourceFormat.LIST: "test_list_source_every_dtype",
        SourceFormat.ARRAY: "test_array_source_every_dtype",
        SourceFormat.CONTOUR: "test_contour_source",
    }
    assert set(covered) == set(SourceFormat), (
        f"source formats with no plan-vs-exec test: {set(SourceFormat) - set(covered)}"
    )
    missing = [name for name in covered.values() if name not in globals()]
    assert not missing, (
        f"named as covering a source format but not defined in this module: {missing}"
    )


def test_dtype_axis_covers_the_vocabulary() -> None:
    """The numpy/polars mirrors must list every ``DType``."""
    names = {d.value for d in DType}
    assert set(NUMPY_FOR_DTYPE) == names, (
        f"NUMPY_FOR_DTYPE out of step with DType: {names ^ set(NUMPY_FOR_DTYPE)}"
    )
    assert set(POLARS_FOR_DTYPE) == names, (
        f"POLARS_FOR_DTYPE out of step with DType: {names ^ set(POLARS_FOR_DTYPE)}"
    )


# ---------------------------------------------------------------------------
# Typed columnar sources, every dtype
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("pattern", HOMOGENEOUS_PATTERNS)
def test_list_source_every_dtype(dtype: str, pattern: str) -> None:
    """A 2-D List column of each element dtype, under each null layout."""
    inner = POLARS_FOR_DTYPE[dtype]
    col_dtype = pl.List(pl.List(inner))
    value = [[1, 2, 3], [4, 5, 6]]
    df = frame(rows_for(pattern, [value, value, value]), col_dtype)

    pipe = Pipeline().source("list", dtype=dtype)
    results = _sweep(df, pipe, f"list source {dtype} / {pattern}")

    listed = results["list"]
    if listed.ok:
        assert leaf_dtype(listed.planned) == inner, (
            f"list source declared {dtype} but planned leaf {listed.planned}"
        )


@plugin_required
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_array_source_every_dtype(dtype: str) -> None:
    """A fixed-size Array column of each element dtype."""
    inner = POLARS_FOR_DTYPE[dtype]
    col_dtype = pl.Array(inner, (2, 3))
    value = [[1, 2, 3], [4, 5, 6]]
    df = frame([value, None, value], col_dtype)

    pipe = Pipeline().source("array", dtype=dtype)
    _sweep(df, pipe, f"array source {dtype}")


@plugin_required
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_raw_source_every_dtype(dtype: str) -> None:
    """Raw bytes decode 1-D at the declared dtype, nulls included."""
    np_dtype = NUMPY_FOR_DTYPE[dtype]
    payload = np.arange(6, dtype=np_dtype).tobytes()
    df = frame([payload, None, payload], pl.Binary)

    pipe = Pipeline().source("raw", dtype=dtype)
    results = _sweep(df, pipe, f"raw source {dtype}")

    listed = results["list"]
    if listed.ok:
        assert listed.planned == pl.List(POLARS_FOR_DTYPE[dtype])


# ---------------------------------------------------------------------------
# Image sources
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("pattern", HOMOGENEOUS_PATTERNS)
def test_image_bytes_source_every_channel_count(channels: int, pattern: str) -> None:
    """1/2/3/4-channel PNGs, including the 2-channel GrayA nothing sank before."""
    images = [make_image_png(H, W, channels, seed=s) for s in (1, 2, 3)]
    df = frame(rows_for(pattern, images), pl.Binary)

    pipe = (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=H, width=W, channels=channels)
        .cast("u8")
    )
    _sweep(df, pipe, f"image_bytes {channels}ch / {pattern}")


@plugin_required
@pytest.mark.parametrize("pattern", HOMOGENEOUS_PATTERNS)
def test_auto_source_on_a_binary_column(pattern: str) -> None:
    """``auto`` is the default and picks its decode path at runtime.

    It is therefore the source whose plan is made with the least information,
    and it had no plan-vs-exec test of any kind. It used to be the one source
    that published a schema execution did not produce: on a Binary column the
    rank stayed unknown and the ``list`` sink silently planned a depth-1
    ``List(UInt8)`` for data that arrives three deep. ``dtype_for_output`` now
    refuses a list sink whose rank it cannot name, so that cell rejects at plan
    time and every other sink still holds.
    """
    images = [make_image_png(H, W, 3, seed=s) for s in (1, 2, 3)]
    df = frame(rows_for(pattern, images), pl.Binary)

    pipe = Pipeline().source("auto").cast("u8")
    _sweep(df, pipe, f"auto source on Binary / {pattern}")


@plugin_required
def test_auto_list_sink_resolves_rank_from_a_columnar_source() -> None:
    """The fix must refuse only what it cannot know, not everything.

    ``auto`` over a List/Array column *can* resolve its rank —
    ``resolved_output_specs`` folds the column's nesting depth through the op
    rank rules — and that path has to keep planning a correct schema. Over a
    Binary column the rank is settled by inspecting the bytes for the VIEW
    magic, so it is genuinely unknowable at plan time and the list sink now
    says so instead of guessing depth 1.
    """
    value = [[1, 2, 3], [4, 5, 6]]
    listed = pl.DataFrame({"img": [value]}, schema={"img": pl.List(pl.List(pl.UInt8))})
    lf = listed.lazy().with_columns(
        out=pl.col("img").cv.pipe(Pipeline().source("auto", dtype="u8")).sink("list")
    )
    assert lf.collect_schema()["out"] == lf.collect(engine="streaming").schema["out"]

    binary = pl.DataFrame(
        {"img": [make_image_png(H, W, 3, seed=1)]}, schema={"img": pl.Binary}
    )
    refused = plan_or_reject(
        binary,
        lambda: (
            pl.col("img").cv.pipe(Pipeline().source("auto").cast("u8")).sink("list")
        ),
    )
    assert not refused.ok, (
        f"auto over a Binary column planned {refused.planned!r} for a rank it "
        f"cannot know"
    )
    assert "rank at planning time" in (refused.reason or ""), refused.reason

    # Naming the source is the documented way out, and it must still work.
    named = Pipeline().source("image_bytes").cast("u8")
    assert_plan_equals_exec(binary, pl.col("img").cv.pipe(named).sink("list"))


@plugin_required
def test_auto_on_binary_still_plans_the_rank_free_sinks() -> None:
    """Refusing the list sink must not take the rest of ``auto`` down with it.

    ``numpy`` publishes a fixed struct whatever the rank turns out to be, so it
    is unaffected by the rank being unknown — the refusal has to be scoped to
    the sinks that actually encode the rank in their Polars dtype.
    """
    binary = pl.DataFrame(
        {"img": [None, make_image_png(H, W, 3, seed=1)]}, schema={"img": pl.Binary}
    )
    pipe = Pipeline().source("auto").cast("u8")
    for sink in ("numpy", "blob", "png"):
        assert_plan_equals_exec(
            binary, pl.col("img").cv.pipe(pipe).sink(sink), name=f"o_{sink}"
        )


@plugin_required
@pytest.mark.parametrize("pattern", HOMOGENEOUS_PATTERNS)
def test_auto_source_on_a_list_column(pattern: str) -> None:
    """The same ``auto`` source, routed to the list decode path instead."""
    value = [[1, 2, 3], [4, 5, 6]]
    df = frame(rows_for(pattern, [value] * 3), pl.List(pl.List(pl.UInt8)))

    pipe = Pipeline().source("auto", dtype="u8")
    _sweep(df, pipe, f"auto source on List / {pattern}")


@plugin_required
def test_16bit_png_keeps_u16_from_plan_to_data() -> None:
    """A 16-bit PNG decodes to u16; the plan must say so before it is read."""
    images = [make_image_png(H, W, 1, sixteen_bit=True, seed=s) for s in (1, 2, 3)]
    df = frame([None, images[0], images[1]], pl.Binary)

    pipe = (
        Pipeline()
        .source("image_bytes", dtype="u16")
        .assert_shape(height=H, width=W, channels=1)
    )
    results = _sweep(df, pipe, "16-bit png")
    if results["list"].ok:
        assert leaf_dtype(results["list"].planned) == pl.UInt16


@plugin_required
@pytest.mark.parametrize("pattern", HOMOGENEOUS_PATTERNS)
def test_file_path_source(tmp_path, pattern: str) -> None:
    """A String column of paths, with nulls interleaved."""
    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(make_image_png(H, W, 3, seed=i))
        paths.append(str(p))

    df = frame(rows_for(pattern, paths), pl.String)
    pipe = (
        Pipeline()
        .source("file_path")
        .assert_shape(height=H, width=W, channels=3)
        .cast("u8")
    )
    _sweep(df, pipe, f"file_path / {pattern}")


@plugin_required
@pytest.mark.parametrize("pattern", HOMOGENEOUS_PATTERNS)
def test_blob_source_round_trips(pattern: str) -> None:
    """VIEW-protocol blobs are self-describing; the plan still has to match.

    The blob column is produced by the library's own ``blob`` sink, so this is
    also the round-trip: what one half writes, the other half must plan for.
    """
    src = pl.DataFrame(
        {"img": [make_image_png(H, W, 3, seed=1)]}, schema={"img": pl.Binary}
    )
    blob = (
        src.lazy()
        .select(
            out=pl.col("img")
            .cv.pipe(Pipeline().source("image_bytes").cast("u8"))
            .sink("blob")
        )
        .collect()["out"][0]
    )
    df = frame(rows_for(pattern, [blob, blob, blob]), pl.Binary)

    pipe = Pipeline().source("blob", dtype="u8")
    _sweep(df, pipe, f"blob source / {pattern}")


@plugin_required
def test_contour_source() -> None:
    """The contour source rasterises geometry back into the buffer domain.

    Note the asymmetry this test had to work around: ``extract_contours()
    .sink("native")`` emits a contour *set* per row (``List(Struct)``), while
    ``source("contour")`` parses a *single* contour per row (``Struct``). The
    sink's output is therefore not directly re-readable by the source, and
    feeding it back reports ``Point struct missing 'x' field`` — the contour
    set's ``exterior``/``holes``/``is_closed`` struct being read as a point.
    """
    contour_set = (
        pl.DataFrame(
            {"img": [make_image_png(16, 16, 3, seed=1)]}, schema={"img": pl.Binary}
        )
        .lazy()
        .select(
            out=pl.col("img")
            .cv.pipe(
                Pipeline()
                .source("image_bytes")
                .grayscale()
                .threshold(128)
                .extract_contours()
            )
            .sink("native")
        )
        .collect()["out"]
    )
    one = contour_set[0][0]
    df = pl.DataFrame({"img": [one, None, one]}, schema={"img": CONTOUR_SCHEMA})

    pipe = Pipeline().source("contour", width=16, height=16).cast("u8")
    _sweep(df, pipe, "contour source", sinks=tuple(s for s in SINKS if s != "native"))


# ---------------------------------------------------------------------------
# The heterogeneous case: the plan is not row 0's shape
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("pattern", ["heterogeneous", "heterogeneous_null_first"])
def test_differently_shaped_rows_do_not_change_the_schema(pattern: str) -> None:
    """Rows of three different sizes, normalised by a resize.

    The planned shape comes from ``resize``, not from whichever row the
    executor happens to look at first — which is the thing
    ``build_typed_list_series_from_rows_with_dtype`` reads to decide both the
    leaf dtype and the nesting depth.
    """
    images = [
        make_image_png(8, 6, 3, seed=1),
        make_image_png(16, 32, 3, seed=2),
        make_image_png(4, 4, 3, seed=3),
    ]
    df = frame(rows_for(pattern, images), pl.Binary)

    pipe = Pipeline().source("image_bytes").resize(height=5, width=7).cast("u8")
    results = _sweep(
        df,
        pipe,
        f"heterogeneous rows / {pattern}",
        sinks=tuple(s for s in SINKS if s not in IMAGE_ENCODER_SINKS),
    )

    listed = results["list"]
    assert listed.ok, f"list sink rejected on heterogeneous rows: {listed.reason}"
    assert listed.planned == pl.List(pl.List(pl.List(pl.UInt8)))
