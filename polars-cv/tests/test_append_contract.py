"""Guards for the mandatory op-append contract (Phase 1).

Every plan-time effect of appending an operation — input-domain validation,
the domain/dtype/ndim fold, and the shape hints — is applied by exactly one
function, ``Pipeline._push_op``. These tests exist to make that structural
rather than conventional:

* :func:`test_op_append_is_structurally_exclusive` forbids any other code from
  mutating ``_ops``, so a builder physically cannot append while tracking only
  part of the effect.
* :func:`test_eager_and_lazy_agree_on_shape_state` pins the two spellings of an
  operation (``.pipe(p.op())`` and ``.pipe(p).op()``) to the same state, and
  its op table is completeness-asserted against the real chainable-op list, so
  a new operation cannot join without a case.

The predecessor of the first test ratcheted only the dtype update while
naming this exact failure mode ("the eager/lazy drift class of bug"); an
enumerated guard that lists one of two required calls is how the transpose and
pad shape bugs shipped underneath it.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

import polars_cv
from polars_cv import Pipeline
from polars_cv._graph import GraphNode
from polars_cv._types import HINT_DIMS, Domain

from ._discovery import package_modules
from ._op_cases import BUFFER, CONTOUR, EXTRA_CASES, OP_CASES, base_pipeline
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
# 1. Only _push_op may mutate _ops
# ---------------------------------------------------------------------------

#: The only functions permitted to touch ``Pipeline._ops``.
#:
#: There are exactly two ways ``_ops`` (and ``_entering``, the state entering
#: each op, kept in step with it) is assigned:
#:
#: * ``_push_op`` appends one op at the end, records the state entering it and
#:   advances the tracked state.
#: * ``_replay`` is the single wholesale rewrite. A slice (CSE, a
#:   sub-pipeline), a reorder (the pushdown) and a deletion (identity
#:   elimination) name the ops they keep and the state to start from, and it
#:   appends them again through ``_push_op`` — so every per-position fact is
#:   recomputed, and no rewrite carries re-key arithmetic of its own (which is
#:   how the CSE path once forgot ``_assertions``).
#:
#: ``_clone`` is listed because it is the copy constructor: it duplicates every
#: field including all the side tables (via ``_copy_state_from`` /
#: ``_STATE_COPIERS``), so there is no position bookkeeping for it to get wrong.
_OPS_MUTATORS = frozenset(
    {
        "_push_op",
        "_replay",
        "_clone",
    }
)

#: The per-position fields only the mutators above may write.
_POSITIONAL = frozenset({"_ops", "_entering"})


def _pipeline_ast() -> ast.ClassDef:
    source = Path(polars_cv.pipeline.__file__).read_text()
    tree = ast.parse(source)
    return next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Pipeline"
    )


def _mutates_ops(node: ast.AST) -> bool:
    """True if *node* appends to, assigns into, replaces or aliases ``*._ops``
    (or ``*._entering``, which is kept in step with it).

    Aliasing counts (``ops = self._ops`` then ``ops.append(...)``) because it
    is the obvious way around a guard that only looks for ``._ops.append``.
    """
    for sub in ast.walk(node):
        # ops = x._ops  — an alias the mutation can then happen through
        if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Attribute):
            if sub.value.attr in _POSITIONAL:
                return True
        # x._ops.append(...) / .extend(...) / .insert(...) / .clear(...)
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"append", "extend", "insert", "clear", "pop"}
            and isinstance(sub.func.value, ast.Attribute)
            and sub.func.value.attr in _POSITIONAL
        ):
            return True
        # x._ops[i] = ... and x._ops += ...
        targets: list[ast.AST] = []
        if isinstance(sub, ast.Assign):
            targets = list(sub.targets)
        elif isinstance(sub, ast.AugAssign):
            targets = [sub.target]
        for t in targets:
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute):
                if t.value.attr in _POSITIONAL:
                    return True
            if isinstance(t, ast.Attribute) and t.attr in _POSITIONAL:
                return True
    return False


def test_op_append_is_structurally_exclusive() -> None:
    """``_push_op`` is the only function that may append to ``_ops``.

    This is the contract that makes the append sequence unskippable: a builder
    cannot add an operation without also running the domain check, the schema
    fold and the shape-hint update, because it never touches ``_ops`` at all.
    """
    offenders: list[str] = []
    # Discovery goes through `_discovery`, which refuses to return an empty
    # set: this guard passing over zero modules is the failure mode it exists
    # to prevent, not a pass.
    for module in package_modules():
        tree = ast.parse(module.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name in _OPS_MUTATORS:
                continue
            # Only the function's own statements, not those of nested defs
            # (which are reported under their own name).
            if _mutates_ops(fn):
                offenders.append(f"{module.name}:{fn.name}")
    assert not offenders, (
        f"only {sorted(_OPS_MUTATORS)} may touch Pipeline._ops, but these also "
        f"do: {sorted(set(offenders))}. Route appends through _append_op() / "
        f"_push_op() and wholesale rewrites through _replay() so the ops and "
        f"the state entering each cannot be updated by halves."
    )


def test_pipeline_state_copy_is_complete() -> None:
    """``_STATE_COPIERS`` must name every field ``Pipeline.__init__`` creates.

    A derived pipeline — ``_clone``, ``_create_sub_pipeline``, CSE's
    ``_create_shared_node`` — inherits its state through
    ``Pipeline._copy_state_from``, which reads only this table. A field the
    table omits is silently reset to its ``__init__`` default in every one of
    them, which is not a degradation the caller can see.

    That is not hypothetical: the three copies used to be written out by hand,
    ``_create_sub_pipeline`` carried 11 of the 14 fields, and because
    ``to_graph()`` makes its sub-pipeline the graph's only node, a public
    ``Pipeline().source(...).on_error("null").to_graph(col)`` executed under
    ``"raise"``. Guard the table rather than the three call sites: the call
    sites are what kept being forgotten.
    """
    from polars_cv.pipeline import _STATE_COPIERS

    declared = set(_STATE_COPIERS)
    actual = set(vars(Pipeline()))

    assert actual, "Pipeline() has no instance attributes -- the probe is broken"
    assert declared == actual, (
        f"_STATE_COPIERS is out of step with Pipeline.__init__.\n"
        f"  missing from the table (silently dropped by every copy): "
        f"{sorted(actual - declared)}\n"
        f"  named but no longer a field (stale entry): {sorted(declared - actual)}"
    )


def test_every_pipeline_field_survives_a_copy() -> None:
    """The table is honoured: a mutated field reaches the copy.

    ``test_pipeline_state_copy_is_complete`` checks the *names*; this checks
    that ``_copy_state_from`` actually transfers a value for each, so an entry
    whose copier silently drops data (or a field re-assigned after the copy)
    fails here rather than in a user's graph.
    """
    from polars_cv.pipeline import _STATE_COPIERS

    source = Pipeline()
    # A value distinguishable from every `__init__` default, per field type.
    sentinels = {
        "_source": object(),
        "_current_domain": "contour",
        "_output_dtype": "f64",
        "_expected_ndim": 7,
        "_on_error": "null",
        "_on_null_param": "null",
        "_shape_declared": True,
        "_ops": ["sentinel-op"],
        "_expr_refs": ["sentinel-expr"],
        "_asserted_dims": {"height"},
        "_entering": ["sentinel-position"],
        "_shape_refs": ["sentinel-ref"],
        "_shape_hints": None,
        "_assertions": {2: None},
    }
    assert set(sentinels) == set(_STATE_COPIERS), (
        "this test's sentinel table drifted from _STATE_COPIERS: "
        f"{sorted(set(sentinels) ^ set(_STATE_COPIERS))}"
    )
    for name, value in sentinels.items():
        setattr(source, name, value)

    copied = Pipeline()
    copied._copy_state_from(source)

    for name, value in sentinels.items():
        assert getattr(copied, name) == value, (
            f"_copy_state_from lost {name}: expected {value!r}, "
            f"got {getattr(copied, name)!r}"
        )

    # Equality alone cannot see the bug the table exists to prevent. A copier
    # that aliases instead of copying passes every check above and then lets a
    # clone mutate its origin -- which is what `_clone` returning a *new*
    # Pipeline is for. Containers must be distinct objects.
    aliased = sorted(
        name
        for name in _STATE_COPIERS
        if isinstance(getattr(source, name), (list, dict, set))
        and getattr(copied, name) is getattr(source, name)
    )
    assert not aliased, (
        f"these fields are shared with the origin rather than copied: "
        f"{aliased}. Mutating the clone would mutate the pipeline it came "
        f"from; `Pipeline` is immutable by contract."
    )


def test_replay_takes_its_assertions_explicitly() -> None:
    """``_replay`` cannot be called without saying where the assertions go.

    The one re-key a rewrite still owns is the assertions' (a slice shifts
    them); a keyword-only parameter with no default makes forgetting it a
    ``TypeError`` rather than an assertion silently kept at the wrong place.
    """
    import inspect

    params = inspect.signature(Pipeline._replay).parameters
    for name in ("start", "assertions"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert params[name].default is inspect.Parameter.empty, name


@plugin_required
def test_a_slice_replays_the_states_it_keeps() -> None:
    """A sub-pipeline over ``[start, end)`` has exactly the states the whole
    pipeline had there — entering each kept op, and at ``end``."""
    pipe = (
        Pipeline()
        .source("blob", dtype="u8")
        .assert_shape(dims=[10, 20, 3])
        .resize(height=4, width=6)
        .grayscale()
        .pad(top=1, bottom=1, left=1, right=1)
    )
    for start, end in [(0, 3), (1, 3), (1, 2), (0, 1)]:
        sub = pipe._create_sub_pipeline(start, end)
        assert sub._entering == pipe._entering[start:end], (start, end)
        assert sub._state() == pipe._state_at(end), (start, end)


def test_push_op_applies_the_whole_plan_step_unconditionally() -> None:
    """``_push_op`` must apply the op's whole plan-time effect, every time.

    Guards the body of the sole mutator itself: it is not enough that callers
    route through it if it were to become selective. The effect is one Rust
    call (``_plan_step``: domain check, schema, H/W, channels, rank clipping)
    and its adoption (``_apply_step``); neither may sit inside a compound
    statement, and the only parameter besides the op is the binary operand's
    dtype, which Rust itself requires for exactly the binary ops.
    """
    fn = next(
        m
        for m in _pipeline_ast().body
        if isinstance(m, ast.FunctionDef) and m.name == "_push_op"
    )
    called = {
        sub.func.attr
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
    }
    assert {"_plan_step", "_apply_step"} <= called, called

    args = [a.arg for a in fn.args.kwonlyargs] + [a.arg for a in fn.args.args]
    flags = [a for a in args if a not in {"self", "spec"}]
    assert flags == ["other_dtype"], (
        f"_push_op grew a new parameter: {flags}. Every additional flag is a "
        f"way to append an op while skipping part of its plan-time effect."
    )

    compound = (ast.If, ast.Try, ast.For, ast.While, ast.With)
    guarded = {
        sub.func.attr
        for branch in ast.walk(fn)
        if isinstance(branch, compound)
        for sub in ast.walk(branch)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
    }
    assert not guarded & {"_plan_step", "_apply_step"}, (
        "the plan step must run for every appended op, not conditionally"
    )


def test_python_holds_no_copy_of_the_channel_rule_arithmetic() -> None:
    """The package must not re-implement ``OutputChannelRule::apply``.

    A source scan, because the property is "this code does not exist". The
    rule is applied in Rust (``plan_step``); what must not come back is
    package code spelling its variants to compute a channel count. It scans the
    whole package, so the arithmetic cannot return under another name.
    Limits: a spelling built at runtime would pass unseen.
    """
    sources = {p: p.read_text() for p in package_modules()}
    for path, src in sources.items():
        for spelling in (
            "strip_restore",
            '"fixed:',
            '"preserve"',
            '"n/a"',
            '"channel_rule"',
        ):
            assert spelling not in src, (
                f"{spelling!r} is in {path.name}: the channel arithmetic "
                f"belongs to OutputChannelRule::apply, reached via plan_step"
            )


# ---------------------------------------------------------------------------
# 2. Input domain comes from the Rust contract
# ---------------------------------------------------------------------------


def test_domain_vocabulary_declared_once() -> None:
    """The domain vocabulary lives in ``_types.Domain``, nowhere else.

    ``Pipeline`` used to carry ``DOMAIN_BUFFER``/``DOMAIN_CONTOUR``/... string
    constants — a third copy behind Rust's ``Domain::NAMED`` and the Python
    ``Domain`` enum, and the only one nothing could pin.

    Every assertion below is an *absence*, which is equally true of a
    ``Pipeline`` that no longer checks domains at all. The positive half
    confirms the replacement is live: a wrong-domain op still raises, and the
    pipeline still tracks a domain drawn from the ``Domain`` vocabulary.
    """
    leaked = [n for n in dir(Pipeline) if n.startswith("DOMAIN_")]
    assert not leaked, f"Pipeline must not re-declare domain constants: {leaked}"
    assert not hasattr(Pipeline, "_validate_domain"), (
        "_validate_domain re-declared each op's input domain in Python; the "
        "check is plan_step's, from the op's Rust input_domains"
    )
    source = Path(polars_cv.pipeline.__file__).read_text()
    assert "_validate_domain" not in source
    assert "DOMAIN_BUFFER" not in source

    # The domain a pipeline reports must be a member of the one vocabulary...
    pipe = Pipeline().source("blob", dtype="u8")
    assert pipe._current_domain in {d.value for d in Domain}, (
        f"Pipeline reports domain {pipe._current_domain!r}, which is not in "
        f"_types.Domain — the vocabulary this test claims is the only one."
    )
    # ...and the check that reads it must still reject a mismatch. Without
    # this, deleting the domain check entirely passes every assertion above.
    with pytest.raises(ValueError, match="(?i)domain"):
        pipe.rasterize(width=8, height=8)


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
    site; the check and its message are now ``plan_step``'s. Binary ops and
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
    hints = pipe._shape_hints
    dims = tuple(
        None if (p := hints.get(dim)) is None or p.is_expr else p.value
        for dim in HINT_DIMS
    )
    return (dims, pipe._expected_ndim, pipe._output_dtype, pipe._current_domain)


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
    to replay only the shape hints, and to assign ``_expected_ndim`` *after*
    that loop — so every replayed op saw ``ndim = None`` and the H/W half of
    the replay returned at its opening guard. Six of ten sampled ops disagreed
    with their eager spelling, ``pad`` and ``rotate`` among them.
    """
    domain, kwargs = _OP_CASES[op]
    base = base_pipeline(domain)

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
    hints = lazy._pipeline._shape_hints
    assert (hints.height.value, hints.width.value, hints.channels.value) == (6, 5, 3)

    # Asserted before an op that changes the same dimension: the op wins.
    # Checked on both spellings — the positional replay is what makes the lazy
    # side work, and an end-of-chain overlay would pass the eager case alone.
    base_u8 = Pipeline().source("image_bytes", dtype="u8")
    eager = base_u8.assert_shape(channels=3).grayscale()
    assert eager._shape_hints.channels.value == 1

    lazy = pl.col("img").cv.pipe(base_u8).assert_shape(channels=3).grayscale()._pipeline
    assert lazy._shape_hints.channels.value == 1

    # And an assertion mid-chain in a single continuation pipeline.
    mid = (
        pl.col("img").cv.pipe(Pipeline().assert_shape(channels=3).grayscale())._pipeline
    )
    assert mid._shape_hints.channels.value == 1


def test_a_contradicting_assertion_is_rejected_where_it_is_written() -> None:
    """``assert_shape`` states a fact; it may not overrule a known one.

    A contradiction used to be accepted here, published as ``expected_shape``,
    and reported at ``collect()`` by ``validate_output_schema`` as *"the
    planner's shape contract disagrees with the Rust implementation"* — the
    plugin taking the blame for a value the caller typed three lines earlier.
    Both spellings are checked: the lazy continuation replays assertions through
    the same ``_apply_assertions_at``, so a check that only ran in the eager
    builder would leave half the surface open.
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
    assert flat._expected_ndim == 1
    with pytest.raises(ValueError, match="does not have"):
        flat.assert_shape(channels=3)
    # `dims=` is the spelling that does fit, and it pins the rank with it.
    lifted = flat.assert_shape(dims=[64])
    assert lifted._expected_ndim == 1


def test_dims_pins_the_rank_a_list_source_could_not_supply() -> None:
    """``dims=`` is what makes ``.assert_shape()`` reach an ``array`` sink.

    A list/array source leaves the rank unknown, and ``expected_shape`` only
    publishes at rank 3 — so the H/W/C spelling set the hints and changed
    nothing, and the sink's advice to "use .assert_shape()" was circular.
    """
    pipe = Pipeline().source("list", dtype="f32").assert_shape(dims=[8, 8, 3])
    assert pipe._expected_ndim == 3
    node = GraphNode(node_id="n", pipeline=pipe, column=None)
    assert node.expected_shape == [8, 8, 3]
    assert node.shape_asserted is True

    # An inferred shape is not attributed to the caller.
    inferred = Pipeline().source("image_bytes", dtype="u8").resize(height=8, width=8)
    assert (
        GraphNode(node_id="n", pipeline=inferred, column=None).shape_asserted is False
    )


def test_dims_rejects_what_it_cannot_track() -> None:
    with pytest.raises(ValueError, match="both"):
        Pipeline().source("list").assert_shape(dims=[8, 8, 3], height=8)
    with pytest.raises(ValueError, match="needs a declaration"):
        Pipeline().source("list").assert_shape()
    with pytest.raises(ValueError, match="up to 3 dimensions"):
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
