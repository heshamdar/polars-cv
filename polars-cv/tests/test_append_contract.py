"""Guards for the op-append contract.

Every plan-time effect of appending an operation — input-domain validation,
the domain/dtype/rank fold and the shape — is one Rust call, ``Plan.push``,
on an immutable plan Python cannot edit: a builder cannot append an op with
part of its effect skipped, because the plan it gets back was planned whole.

* :func:`test_eager_and_lazy_agree_on_shape_state` pins the two spellings of an
  operation (``.pipe(p.op())`` and ``.pipe(p).op()``) to the same state, and
  its op table is completeness-asserted against the real chainable-op list, so
  a new operation cannot join without a case.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline

from ._op_cases import (
    BUFFER,
    CONTOUR,
    EXTRA_CASES,
    OP_CASES,
    base_pipeline,
    case_base,
)
from ._schema_parity import assert_plan_equals_exec
from .conftest import plugin_required

#: Every test here is a structural guard: it checks the *shape* of the codebase
#: -- registries, authorities, removed surfaces, documented vocabularies --
#: rather than the numerical behaviour of a pipeline. `-m structural` is the
#: lane pre-commit runs; see `tests/AGENTS.md`. Note that the lane as a whole
#: does need the compiled extension: many structural facts are only observable
#: through the FFI, and those tests fail rather than skip without it.
pytestmark = pytest.mark.structural

# ---------------------------------------------------------------------------
# 1. The plan is Rust's, and immutable
# ---------------------------------------------------------------------------


@plugin_required
def test_the_plan_cannot_be_edited_from_python() -> None:
    """A pipeline's ops live in its Rust ``Plan``, which has no setter: the
    only way to change them is a method that plans every op it keeps
    (``push``, ``select``, ``with_source``, ``rebased``, ``run_pass``)."""
    plan = Pipeline().source("image_bytes").grayscale()._plan
    for name in ("state", "has_source", "source_format"):
        with pytest.raises(AttributeError):
            setattr(plan, name, None)
    with pytest.raises(AttributeError):
        plan.ops = []  # type: ignore[attr-defined]


@plugin_required
def test_a_derived_pipeline_keeps_every_setting_and_shares_no_list() -> None:
    """``_clone`` is the one copy: policies, expressions and node reads reach
    the copy, and appending to the copy leaves its origin as it was."""
    base = (
        Pipeline()
        .source("image_bytes")
        .on_error("null")
        .on_null_param("null")
        .resize(height=pl.col("h"), width=4)
    )
    derived = base.scale(pl.col("s"))
    assert (derived._on_error, derived._on_null_param) == ("null", "null")
    assert len(derived._exprs) == 2
    assert len(base._exprs) == 1
    assert len(base._plan) == 1
    graph_node = base.to_graph(pl.col("img"))._nodes["_node_0"].pipeline
    assert graph_node._on_error == "null"


# ---------------------------------------------------------------------------
# 2. Input domain comes from the Rust contract
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize(
    ("build", "op", "kwargs"),
    [
        (lambda: Pipeline().source("image_bytes"), "area", {}),
        (lambda: Pipeline().source("image_bytes"), "perimeter", {}),
        (lambda: Pipeline().source("image_bytes"), "convex_hull", {}),
        (lambda: Pipeline().source("image_bytes"), "simplify", {"tolerance": 1.0}),
    ],
)
def test_wrong_input_domain_is_rejected(build, op, kwargs) -> None:
    """A contour op on a buffer pipeline raises, naming the contract's domain."""
    pipe = build()
    with pytest.raises(ValueError, match="expects contour input"):
        getattr(pipe, op)(**kwargs)


@plugin_required
def test_input_domain_matches_the_rust_contract() -> None:
    """The rejection names the op and the domains its Rust contract accepts.

    Input domain used to be a hand-written argument at every builder call
    site; the check and its message are now ``Plan.push``'s. Binary ops and
    reductions accept two domains, and the message lists both.
    """
    contour_pipe = (
        Pipeline().source("image_bytes").grayscale().threshold(128).extract_contours()
    )
    with pytest.raises(ValueError, match=r"resize\(\) expects buffer input"):
        contour_pipe.resize(height=8, width=8)
    with pytest.raises(ValueError, match=r"reduce_sum\(\) expects buffer or vector"):
        contour_pipe.reduce_sum()


# ---------------------------------------------------------------------------
# 3. Eager and lazy must agree
# ---------------------------------------------------------------------------

#: Per-op arguments for the eager/lazy parity sweep.
#:
#: The table itself lives in ``tests/_op_cases.py`` because the schema-parity
#: matrix drives the same axis from it. It is completeness-asserted below, so a
#: new op cannot join the library without a case — and therefore without both
#: an eager/lazy cell here and a plan-vs-exec cell in the matrix.
_BUFFER, _CONTOUR = BUFFER, CONTOUR

_OP_CASES = OP_CASES


def _state(pipe: Pipeline) -> tuple:
    s = pipe._state
    return (s.dims, s.ndim, s.dtype, s.domain)


def test_op_case_table_is_complete() -> None:
    """Every chainable op needs a parity case (or an explicit exemption).

    Without this the parity sweep below would silently shrink as operations
    are added — the failure mode of every hand-maintained table in this repo.
    """
    from polars_cv.lazy import _chainable_pipeline_ops

    chainable = set(_chainable_pipeline_ops())
    missing = chainable - set(_OP_CASES)
    stale = set(_OP_CASES) - chainable
    assert not missing, f"chainable ops with no parity case: {sorted(missing)}"
    assert not stale, f"parity cases for ops that no longer exist: {sorted(stale)}"


#: Extra parameterisations for ops whose interesting behaviour is in a branch
#: the single case in ``_OP_CASES`` does not reach. Kept separate so the
#: completeness assertion above stays a strict one-case-per-op check.
_EXTRA_CASES = EXTRA_CASES


@plugin_required
@pytest.mark.parametrize(
    ("op", "domain", "kwargs"),
    [(o, d, k) for o, d, k in _EXTRA_CASES],
    ids=[f"{o}-{sorted(k)}" for o, _, k in _EXTRA_CASES],
)
def test_eager_and_lazy_agree_on_extra_branches(op, domain, kwargs) -> None:
    """Branch coverage for ops whose single parity case misses the interesting path."""
    base = base_pipeline(domain)
    eager = getattr(base, op)(**kwargs)
    lazy = getattr(pl.col("img").cv.pipe(base), op)(**kwargs)._pipeline
    assert _state(eager) == _state(lazy), (
        f"{op}({kwargs}): eager {_state(eager)} != lazy {_state(lazy)}"
    )


@plugin_required
@pytest.mark.parametrize(
    "op", sorted(name for name, case in _OP_CASES.items() if case is not None)
)
def test_eager_and_lazy_agree_on_shape_state(op) -> None:
    """``.pipe(p.op())`` and ``.pipe(p).op()`` must plan identically.

    The lazy continuation re-applies each op over the upstream state. It used
    to replay only the shape hints, and to assign the rank *after*
    that loop — so every replayed op saw ``ndim = None`` and the H/W half of
    the replay returned at its opening guard. Six of ten sampled ops disagreed
    with their eager spelling, ``pad`` and ``rotate`` among them.
    """
    domain, kwargs = _OP_CASES[op]
    base = case_base(op, domain)

    eager = getattr(base, op)(**kwargs)
    lazy = getattr(pl.col("img").cv.pipe(base), op)(**kwargs)._pipeline

    assert _state(eager) == _state(lazy), (
        f"{op}: eager {_state(eager)} != lazy {_state(lazy)} — the two "
        f"spellings of the same operation must plan identically"
    )


@plugin_required
def test_assert_shape_survives_a_continuation() -> None:
    """A user assertion outranks inference, and only from where it was written.

    ``assert_shape`` records against its op position so a continuation replays
    it in place. Applying it after the whole chain instead would let an
    assertion override ops that legitimately change the shape.
    """
    base = Pipeline().source("image_bytes", dtype="u8")
    lazy = (
        pl.col("img").cv.pipe(base).resize(height=6, width=5).assert_shape(channels=3)
    )
    assert lazy._pipeline._state.dims == (6, 5, 3)

    # Asserted before an op that changes the same dimension: the op wins.
    # Checked on both spellings — the positional replay is what makes the lazy
    # side work, and an end-of-chain overlay would pass the eager case alone.
    base_u8 = Pipeline().source("image_bytes", dtype="u8")
    eager = base_u8.assert_shape(channels=3).grayscale()
    assert eager._state.dims[2] == 1

    lazy = pl.col("img").cv.pipe(base_u8).assert_shape(channels=3).grayscale()._pipeline
    assert lazy._state.dims[2] == 1

    # And an assertion mid-chain in a single continuation pipeline.
    mid = (
        pl.col("img").cv.pipe(Pipeline().assert_shape(channels=3).grayscale())._pipeline
    )
    assert mid._state.dims[2] == 1


def test_a_contradicting_assertion_is_rejected_where_it_is_written() -> None:
    """``assert_shape`` states a fact; it may not overrule a known one.

    A contradiction used to be accepted here, published as ``expected_shape``,
    and reported at ``collect()`` by ``validate_output_schema`` as *"the
    planner's shape contract disagrees with the Rust implementation"* — the
    plugin taking the blame for a value the caller typed three lines earlier.
    Both spellings are checked: the lazy continuation replays the
    ``assert_shape`` op through the same ``Plan.push``, so a check that only
    ran in the eager builder would leave half the surface open.
    """
    base = Pipeline().source("image_bytes", dtype="u8")
    with pytest.raises(ValueError, match="contradicts the height 224"):
        base.resize(height=224, width=224).assert_shape(height=999)
    with pytest.raises(ValueError, match="contradicts the channels 1"):
        base.grayscale().assert_shape(channels=3)
    with pytest.raises(ValueError, match="contradicts the height 224"):
        pl.col("img").cv.pipe(base).resize(height=224, width=224).assert_shape(
            height=999
        )

    # An assertion the planner cannot check is the supported case: a list
    # source's shape is genuinely unknown until execution.
    Pipeline().source("list", dtype="f32").assert_shape(dims=[8, 8, 3])


def test_an_assertion_may_not_name_a_dimension_the_rank_lacks() -> None:
    """The hints are positional, so the H/W/C names only fit a rank-3 buffer."""
    flat = Pipeline().source("raw", dtype="u8")  # rank 1
    assert flat._state.ndim == 1
    with pytest.raises(ValueError, match="does not have"):
        flat.assert_shape(channels=3)
    # `dims=` is the spelling that does fit, and it pins the rank with it.
    lifted = flat.assert_shape(dims=[64])
    assert lifted._state.ndim == 1


def test_dims_pins_the_rank_a_list_source_could_not_supply() -> None:
    """``dims=`` is what makes ``.assert_shape()`` reach an ``array`` sink.

    A list/array source leaves the rank unknown, and an output's shape only
    publishes at rank 3 — so the H/W/C spelling set the hints and changed
    nothing, and the sink's advice to "use .assert_shape()" was circular.
    The output facts Rust plans from this state (the rank-3 gate) are pinned
    by ``output_facts_are_planned_from_the_ops``.
    """
    pipe = Pipeline().source("list", dtype="f32").assert_shape(dims=[8, 8, 3])
    assert (pipe._state.ndim, pipe._state.dims) == (3, (8, 8, 3))


def test_dims_rejects_what_it_cannot_track() -> None:
    with pytest.raises(ValueError, match="both"):
        Pipeline().source("list").assert_shape(dims=[8, 8, 3], height=8)
    with pytest.raises(ValueError, match="needs a declaration"):
        Pipeline().source("list").assert_shape()
    with pytest.raises(ValueError, match="1 to 3 dimensions"):
        Pipeline().source("list").assert_shape(dims=[2, 8, 8, 3])
    with pytest.raises(ValueError, match="positive int"):
        Pipeline().source("list").assert_shape(dims=[8, 0, 3])


# ---------------------------------------------------------------------------
# 4. End-to-end: plan must equal exec (the bugs this phase fixes)
# ---------------------------------------------------------------------------


@pytest.fixture
def non_square_png() -> bytes:
    """A 100x200 RGB PNG — non-square so an H/W swap cannot cancel out."""
    buf = io.BytesIO()
    Image.fromarray(np.zeros((100, 200, 3), np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


#: The plan-vs-exec assertion lives in ``tests/_schema_parity.py``; this file
#: used to carry its own copy, as did two others.
_assert_plan_equals_exec = assert_plan_equals_exec


@plugin_required
@pytest.mark.parametrize(
    ("label", "chain", "sink"),
    [
        ("eager transpose", lambda p: p.transpose([1, 0, 2]), "list"),
        ("eager channel_select", lambda p: p.channel_select(index=0), "list"),
        ("eager pad", lambda p: p.pad(top=10, bottom=10, left=0, right=0), "array"),
    ],
)
def test_eager_plan_equals_exec(non_square_png, label, chain, sink) -> None:
    """Shape-changing ops must not desync the planned schema (A-1)."""
    df = pl.DataFrame({"img": [non_square_png]})
    base = (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=100, width=200, channels=3)
        .cast("u8")
    )
    _assert_plan_equals_exec(df, pl.col("img").cv.pipe(chain(base)).sink(sink))


@plugin_required
@pytest.mark.parametrize(
    ("label", "chain", "sink"),
    [
        ("lazy pad", lambda e: e.pad(top=10, bottom=10, left=0, right=0), "array"),
        ("lazy rotate90", lambda e: e.rotate(angle=90), "array"),
        ("lazy pad_to_size", lambda e: e.pad_to_size(height=150, width=250), "array"),
        ("lazy resize_max", lambda e: e.resize_max(max_size=120), "array"),
        ("lazy channel_select", lambda e: e.channel_select(index=0), "list"),
    ],
)
def test_lazy_plan_equals_exec(non_square_png, label, chain, sink) -> None:
    """The lazy continuation must plan what it executes (A-2)."""
    df = pl.DataFrame({"img": [non_square_png]})
    base = (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=100, width=200, channels=3)
        .cast("u8")
    )
    _assert_plan_equals_exec(df, chain(pl.col("img").cv.pipe(base)).sink(sink))


def test_an_unknown_input_is_not_planned_as_square() -> None:
    """An aspect-preserving resize of an image of unknown size has unknown H/W.

    The retired shape prober stood the *same* placeholder in for every unknown
    input axis, so it planned every unknown image as square: with a per-row
    ``filter`` making the op per-row, ``resize_max(7)`` published ``[7, 7]``
    for a 100x50 image that executes as 7x4. Shapes are now symbolic, so only
    what the op fixes is known.
    """
    per_row = pl.col("filter")
    plan = Pipeline().source("image_bytes").resize_max(7, filter=per_row)._state
    assert plan.dims[:2] == (None, None)
    plan = Pipeline().source("image_bytes").resize_to_height(7, filter=per_row)._state
    assert plan.dims[:2] == (7, None)


@plugin_required
def test_a_per_row_parameter_is_not_validated_as_a_placeholder() -> None:
    """A per-row value is unknown at plan time, so no plan-time check reads it.

    The planner used to resolve each op with a stand-in value for every
    per-row parameter (``1`` for an integer) and validate that: a per-row
    ``channel_select`` on a known ``[4, 4]`` buffer was refused as "channel 1"
    though every row selects channel 0. A literal index is still checked
    while the pipeline is built.
    """
    base = Pipeline().source("array").assert_shape(dims=[4, 4])
    pipe = base.channel_select(index=pl.col("i"))
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    df = pl.DataFrame({"a": [arr.tolist()], "i": [0]})
    out = df.select(pl.col("a").cv.pipe(pipe).sink("list").alias("o"))["o"]
    assert out.to_list() == [arr.tolist()]
    with pytest.raises(ValueError, match="channel 1"):
        base.channel_select(index=1)
