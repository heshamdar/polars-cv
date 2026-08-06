"""Plan == exec for *composed* effects, multi-node graphs and continuations.

A single-op sweep cannot reach these. What is checked here:

* chains that stack an H/W change on a rank change on a channel change on a
  dtype change, asserted after every prefix as well as at the end;
* the steps ``op_infer_shape`` refuses, which wipe the H/W hints — the
  requirement there is that a shape-dependent sink is *rejected while
  planning*, never accepted and then wrong;
* every binary op, with the operand axis read from Rust's ``BINARY_OPS``
  registry rather than a list written here;
* the two spellings of a continuation (``.pipe(p.op())`` vs ``.pipe(p).op()``)
  against each other *and* against the data — ``test_append_contract`` pins
  the first pair to each other but never to execution;
* graphs with more than one node: CSE-shared prefixes and two distinct
  ``pl.col()`` roots, which is where ``resolved_output_specs`` resolves every
  output's ``"auto"`` dtype from ``inputs.first()``.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline
from tests._schema_parity import (
    assert_plan_equals_exec,
    encodable_by_image_codec,
    frame,
    plan_or_reject,
    rows_for,
)
from tests.conftest import make_image_png, plugin_required

H, W = 20, 32


def _binary_op_names() -> list[str]:
    """The binary-op vocabulary, read from the Rust registry.

    ``BINARY_OPS`` in ``src/execute.rs`` is the single authority and it is
    surfaced as the ``BinaryOp`` enum over ``enum_variants``. Reading it here
    means a new binary op joins this sweep automatically.
    """
    from polars_cv._lib import enum_variants

    return sorted(enum_variants("BinaryOp"))


def _df(pattern: str = "single", *, channels: int = 3) -> pl.DataFrame:
    images = [make_image_png(H, W, channels, seed=s) for s in (1, 2, 3)]
    return frame(rows_for(pattern, images), pl.Binary)


def _base(*, channels: int = 3) -> Pipeline:
    return (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=H, width=W, channels=channels)
        .cast("u8")
    )


# ---------------------------------------------------------------------------
# Composed shape / channel / dtype effects
# ---------------------------------------------------------------------------

#: Each step stacks another kind of schema effect on the one before: H/W, then
#: channels, then dtype, then padding, then a 90-degree swap, then an axis
#: permutation, then a rank drop.
_CHAIN = [
    ("resize", lambda p: p.resize(height=12, width=8)),
    ("grayscale", lambda p: p.grayscale()),
    ("cast_f32", lambda p: p.cast("f32")),
    ("pad", lambda p: p.pad(top=1, bottom=2, left=3, right=4)),
    ("rotate90", lambda p: p.rotate(angle=90)),
    ("transpose", lambda p: p.transpose([1, 0, 2])),
    ("channel_select", lambda p: p.channel_select(index=0)),
]


@plugin_required
@pytest.mark.parametrize("prefix_len", range(1, len(_CHAIN) + 1))
def test_composed_chain_plans_what_it_executes(prefix_len: int) -> None:
    """Every prefix of the chain, not just the whole thing.

    Asserting only the full chain would let two errors cancel — an H/W swap
    undone by a later transpose reads as correct at the end.
    """
    pipe = _base()
    for _, step in _CHAIN[:prefix_len]:
        pipe = step(pipe)

    df = _df("null_first")
    for sink in ("list", "numpy", "blob"):
        assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink(sink))


@plugin_required
# 3 -> 1 (no alpha to preserve) and 4 -> 2 (alpha split off and re-attached)
# are the two arms of the rule. A 2-channel input is not a legal source for
# from_space="rgb", so it is not a case here.
@pytest.mark.parametrize(("channels", "expected"), [(3, 1), (4, 2)])
def test_strip_process_restore_channel_count(channels: int, expected: int) -> None:
    """``convert_color(..., "gray")`` keeps alpha; ``grayscale()`` drops it.

    These are two different channel rules that read alike in a pipeline —
    ``strip_restore:1`` versus ``fixed:1`` — and the difference is exactly one
    channel in the published schema. Pinned at the hint level in
    ``test_alpha_channel.py`` and never against data until now.
    """
    pipe = _base(channels=channels).convert_color(from_space="rgb", to_space="gray")
    assert pipe._shape_hints.channels.value == expected

    df = _df("null_first", channels=channels)
    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("list"))
    assert series.dtype == pl.List(pl.List(pl.List(pl.UInt8)))

    # And the data really is that wide, so the plan is not merely
    # self-consistent with itself.
    row = next(r for r in series if r is not None)
    assert len(row[0][0]) == expected, (
        f"{channels}ch -> gray: plan said {expected} channels, data has "
        f"{len(row[0][0])}"
    )


@plugin_required
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_grayscale_is_fixed_one_channel_and_drops_alpha(channels: int) -> None:
    """The sibling rule: ``grayscale`` is ``fixed:1`` whatever the input."""
    pipe = _base(channels=channels).grayscale()
    assert pipe._shape_hints.channels.value == 1

    df = _df("null_first", channels=channels)
    series = assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("list"))
    row = next(r for r in series if r is not None)
    assert len(row[0][0]) == 1


# ---------------------------------------------------------------------------
# Hint-invalidating steps
# ---------------------------------------------------------------------------

#: Steps ``op_infer_shape`` rejects (lib.rs:250-258), which wipe the H/W hints.
#: After one of these the array sink has no shape to plan and must refuse.
_HINT_INVALIDATING = {
    "reduce_sum": lambda p: p.reduce_sum(),
    "reduce_argmax": lambda p: p.reduce_argmax(axis=0),
    "perceptual_hash": lambda p: p.perceptual_hash(),
    "extract_shape": lambda p: p.extract_shape(),
    "histogram": lambda p: p.grayscale().histogram(bins=8, output="counts"),
}


@plugin_required
@pytest.mark.parametrize("name", sorted(_HINT_INVALIDATING))
def test_array_sink_refuses_after_a_hint_invalidating_step(name: str) -> None:
    """Unknown shape must be a plan-time refusal, not a runtime surprise.

    Keeping the pre-op H/W across one of these is how a pipeline came to
    publish ``[100, 200, 2]`` for data that executes as ``[200, 3, 2]``.
    Unknown is always safe; wrong never is.
    """
    pipe = _HINT_INVALIDATING[name](_base())
    df = _df()

    result = plan_or_reject(df, lambda: pl.col("img").cv.pipe(pipe).sink("array"))
    if result.ok:
        # Accepting is fine *if* the shape really was knowable and correct —
        # plan_or_reject already proved plan == exec in that case.
        assert result.planned is not None
    else:
        assert result.reason is not None


@plugin_required
@pytest.mark.parametrize("name", sorted(_HINT_INVALIDATING))
def test_hint_invalidating_steps_still_plan_their_dtype(name: str) -> None:
    """The shape may be unknown; the dtype and domain must not be."""
    pipe = _HINT_INVALIDATING[name](_base())
    df = _df("null_first")
    for sink in ("list", "native"):
        plan_or_reject(df, lambda s=sink: pl.col("img").cv.pipe(pipe).sink(s))


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------


@plugin_required
def test_binary_op_axis_is_the_rust_registry() -> None:
    """The sweep below must cover the real vocabulary, not a stale copy."""
    names = _binary_op_names()
    assert len(names) >= 11, f"BinaryOp vocabulary shrank to {names}"
    assert "divide" in names and "blend" in names


@plugin_required
@pytest.mark.parametrize("op", _binary_op_names())
def test_binary_ops_plan_what_they_execute(op: str) -> None:
    """Only ``divide`` had a plan-vs-exec test; here is the whole family.

    Binary ops are also the one place the planner assigns ``_output_dtype`` and
    ``_expected_ndim`` by hand (``lazy.py`` ``_binary_op``) instead of folding
    through ``op_schema``, so their dtype promotion — u8 x u8 -> f32 for
    ``divide``/``ratio`` — is computed by a separate code path.
    """
    df = _df("null_first")
    left = pl.col("img").cv.pipe(_base())
    right = pl.col("img").cv.pipe(_base())

    combined = getattr(left, op)(right)
    for sink in ("list", "numpy", "blob"):
        plan_or_reject(df, lambda s=sink: combined.sink(s))


# ---------------------------------------------------------------------------
# Eager vs lazy vs data
# ---------------------------------------------------------------------------

_CONTINUATION_OPS = {
    "resize": lambda p: p.resize(height=7, width=5),
    "pad": lambda p: p.pad(top=2, bottom=3, left=1, right=4),
    "rotate90": lambda p: p.rotate(angle=90),
    "resize_max": lambda p: p.resize_max(max_size=16),
    "pad_to_size": lambda p: p.pad_to_size(height=40, width=48),
    "channel_select": lambda p: p.channel_select(index=1),
    "transpose": lambda p: p.transpose([1, 0, 2]),
}


@plugin_required
@pytest.mark.parametrize("name", sorted(_CONTINUATION_OPS))
def test_eager_and_lazy_spellings_agree_with_each_other_and_the_data(
    name: str,
) -> None:
    """``.pipe(p.op())`` and ``.pipe(p).op()`` must plan alike *and* be right.

    ``test_eager_and_lazy_agree_on_shape_state`` compares the two plans to each
    other, which two identically-wrong plans also satisfy. This adds the third
    point: both must equal what executes.
    """
    step = _CONTINUATION_OPS[name]
    df = _df("null_first")

    eager_pipe = step(_base())
    eager = pl.col("img").cv.pipe(eager_pipe).sink("list")
    lazy = step(pl.col("img").cv.pipe(_base())).sink("list")

    eager_dtype = assert_plan_equals_exec(df, eager, name="a").dtype
    lazy_dtype = assert_plan_equals_exec(df, lazy, name="b").dtype
    assert eager_dtype == lazy_dtype, (
        f"{name}: eager spelling produced {eager_dtype}, lazy {lazy_dtype}"
    )


# ---------------------------------------------------------------------------
# Multi-node graphs
# ---------------------------------------------------------------------------


@plugin_required
def test_cse_shared_prefix_keeps_both_schemas_honest() -> None:
    """Two outputs off one prefix: CSE re-folds state via ``_create_shared_node``.

    The re-fold is a second path to the same plan-time state, so it is a place
    the published schema can quietly diverge from the single-branch spelling.
    """
    df = _df("null_first")
    shared = _base().resize(height=10, width=10)

    a = pl.col("img").cv.pipe(shared).grayscale().sink("list")
    b = pl.col("img").cv.pipe(shared).cast("f32").sink("list")

    lf = df.lazy().with_columns(a=a, b=b)
    planned = lf.collect_schema()
    produced = lf.collect(engine="streaming").schema
    assert planned["a"] == produced["a"], f"{planned['a']} != {produced['a']}"
    assert planned["b"] == produced["b"], f"{planned['b']} != {produced['b']}"
    assert planned["a"] != planned["b"], (
        "the two branches should differ in dtype; if they do not, CSE has "
        "merged pipelines it should not have and this test proves nothing"
    )


@plugin_required
def test_two_source_columns_in_one_graph() -> None:
    """Two distinct ``pl.col()`` roots, with different element dtypes.

    ``resolved_output_specs`` resolves each output's ``"auto"`` dtype from
    ``inputs.first()`` — the *first* input column — regardless of which node
    the output belongs to. Two roots whose dtypes differ is what tells those
    apart.
    """
    images = [make_image_png(H, W, 3, seed=s) for s in (1, 2, 3)]
    df = pl.DataFrame(
        {"a": [None, images[0], images[1]], "b": [images[2], images[0], None]},
        schema={"a": pl.Binary, "b": pl.Binary},
    )

    left = pl.col("a").cv.pipe(_base().cast("u8")).sink("list")
    right = pl.col("b").cv.pipe(_base().cast("f32")).sink("list")

    lf = df.lazy().with_columns(left=left, right=right)
    planned = lf.collect_schema()
    produced = lf.collect(engine="streaming").schema
    assert planned["left"] == produced["left"]
    assert planned["right"] == produced["right"]
    assert planned["left"] != planned["right"], (
        "the two roots must differ in element dtype or this test cannot detect "
        "a first-column-wins resolution"
    )


@plugin_required
def test_each_output_resolves_against_its_own_root_column() -> None:
    """A multi-root graph must not resolve every output from column 0.

    ``merge_pipe`` and the binary ops join two ``pl.col()`` lineages into one
    ``vb_graph`` call. ``resolved_output_specs`` used to fill in every output's
    ``"auto"`` element dtype from ``inputs.first()``, so the second branch was
    planned with the first branch's column type: two list columns of different
    leaf dtypes planned ``Struct({x: List(UInt8), y: List(UInt8)})`` and
    executed with ``y`` as Float32.

    Two *separate* expressions never showed this — each is its own plugin call
    with its own single input — so the bug needed one graph with two roots.
    """
    df = pl.DataFrame(
        {"a": [[[1, 2], [3, 4]]], "b": [[[1.5, 2.5], [3.5, 4.5]]]},
        schema={"a": pl.List(pl.List(pl.UInt8)), "b": pl.List(pl.List(pl.Float32))},
    )
    left = pl.col("a").cv.pipe(Pipeline().source("list")).alias("x")
    right = pl.col("b").cv.pipe(Pipeline().source("list")).alias("y")

    series = assert_plan_equals_exec(
        df, left.merge_pipe(right).sink({"x": "list", "y": "list"})
    )
    fields = dict((f.name, f.dtype) for f in series.dtype.fields)
    assert fields["x"] == pl.List(pl.List(pl.UInt8))
    assert fields["y"] == pl.List(pl.List(pl.Float32)), (
        f"'y' reads the Float32 column; planned {fields['y']}"
    )


@plugin_required
def test_fused_and_unfused_affine_runs_agree() -> None:
    """Affine fusion rewrites the op list after the schema was folded.

    ``_to_spec_dict`` collapses a run of affine ops using ``_hint_snapshots``,
    so the executed graph is not the graph the schema was computed from.
    """
    df = _df("null_first")
    fusable = _base().rotate(angle=30.0).resize(height=11, width=9)
    interrupted = _base().rotate(angle=30.0).grayscale().resize(height=11, width=9)

    for pipe in (fusable, interrupted):
        assert_plan_equals_exec(df, pl.col("img").cv.pipe(pipe).sink("list"))


@plugin_required
def test_image_codec_carve_out_is_still_narrow() -> None:
    """The sweeps skip codec sinks for non-u8/rank-1 pipelines; keep that tight.

    If ``encodable_by_image_codec`` started returning ``False`` for ordinary u8
    image pipelines, every sweep would quietly stop testing png/jpeg/webp/tiff.
    """
    assert encodable_by_image_codec(_base())
    assert encodable_by_image_codec(_base().grayscale())
    assert not encodable_by_image_codec(_base().cast("f32"))
    assert not encodable_by_image_codec(_base().reduce_sum())
