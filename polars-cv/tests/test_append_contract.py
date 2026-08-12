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

The predecessor of the first test ratcheted only ``_update_output_dtype`` while
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
from polars_cv._types import Domain

from ._discovery import package_modules
from ._op_cases import BUFFER, CONTOUR, EXTRA_CASES, OP_CASES, base_pipeline
from ._schema_parity import assert_plan_equals_exec
from .conftest import plugin_required

#: Every test here is a structural guard: it checks the *shape* of the
#: codebase rather than the behaviour of a pipeline, so it needs no compiled
#: extension and runs in milliseconds. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural

# ---------------------------------------------------------------------------
# 1. Only _push_op may mutate _ops
# ---------------------------------------------------------------------------

#: The only functions permitted to touch ``Pipeline._ops``.
#:
#: ``_push_op`` appends one op and advances the tracked state; ``_set_ops_slice``
#: replaces the list wholesale for CSE and re-keys everything keyed by op index.
#: Both live in ``pipeline.py`` next to the state they maintain. Anything else
#: assigning ``_ops`` has to remember which side tables are position-keyed —
#: which is how the CSE path came to re-key ``_hint_snapshots`` but not
#: ``_assertions``.
#:
#: ``_clone`` is listed because it is the copy constructor: it duplicates every
#: field including all the side tables, so there is no position bookkeeping for
#: it to get wrong. It is the one place where assigning ``_ops`` carries no
#: obligation.
_OPS_MUTATORS = frozenset({"_push_op", "_set_ops_slice", "_clone"})


def _pipeline_ast() -> ast.ClassDef:
    source = Path(polars_cv.pipeline.__file__).read_text()
    tree = ast.parse(source)
    return next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Pipeline"
    )


def _mutates_ops(node: ast.AST) -> bool:
    """True if *node* appends to, assigns into, replaces or aliases ``*._ops``.

    Aliasing counts (``ops = self._ops`` then ``ops.append(...)``) because it
    is the obvious way around a guard that only looks for ``._ops.append``.
    """
    for sub in ast.walk(node):
        # ops = x._ops  — an alias the mutation can then happen through
        if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Attribute):
            if sub.value.attr == "_ops":
                return True
        # x._ops.append(...) / .extend(...) / .insert(...) / .clear(...)
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"append", "extend", "insert", "clear", "pop"}
            and isinstance(sub.func.value, ast.Attribute)
            and sub.func.value.attr == "_ops"
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
                if t.value.attr == "_ops":
                    return True
            if isinstance(t, ast.Attribute) and t.attr == "_ops":
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
        f"do: {sorted(set(offenders))}. Route them through _append_op() / "
        f"_push_op() / _set_ops_slice() so the plan-time state and the "
        f"position-keyed side tables cannot be updated by halves."
    )


def test_push_op_updates_dtype_and_hints_unconditionally() -> None:
    """``_push_op`` must apply *both* halves of the plan-time effect.

    Guards the body of the sole mutator itself: it is no longer enough that
    callers route through it if it were to become selective about what it
    updates. ``update_dtype=False`` exists only for the two-input binary rule
    and is asserted to be the sole opt-out, with the hint update outside any
    conditional.
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
    assert "_update_output_dtype" in called
    assert "_update_shape_hints" in called
    assert "_require_input_domain" in called, (
        "_push_op must validate the input domain, so every append path gets "
        "the check — not just the builder path through _append_op"
    )

    # `update_dtype` is the only opt-out, and it opts out of exactly one thing.
    args = [a.arg for a in fn.args.kwonlyargs] + [a.arg for a in fn.args.args]
    flags = [a for a in args if a not in {"self", "spec", "contract"}]
    assert flags == ["update_dtype"], (
        f"_push_op grew a new opt-out: {flags}. Every additional flag is a way "
        f"to append an op while skipping part of its plan-time effect."
    )

    # The hint update must not sit inside *any* compound statement: it applies
    # to every op unconditionally. Checking only `ast.If` left try/for/while/with
    # as ways to make it conditional while still passing.
    compound = (ast.If, ast.Try, ast.For, ast.While, ast.With)
    guarded = {
        sub.func.attr
        for branch in ast.walk(fn)
        if isinstance(branch, compound)
        for sub in ast.walk(branch)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
    }
    assert "_update_shape_hints" not in guarded, (
        "_update_shape_hints must run for every appended op, not conditionally"
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
        "check now reads op_contract(...)['input_domains']"
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
    """The rejection message names the domain the Rust contract declares.

    The input-domain mirror of ``test_planner_domain_is_sourced_from_rust``:
    output domain was already sourced from Rust while input domain stayed a
    hand-written argument at every builder call site.
    """
    import json

    from polars_cv._lib import op_contract

    contour_pipe = (
        Pipeline().source("image_bytes").grayscale().threshold(128).extract_contours()
    )
    # A buffer op on a contour pipeline.
    with pytest.raises(ValueError) as excinfo:
        contour_pipe.resize(height=8, width=8)
    resize_spec = Pipeline().source("image_bytes").resize(height=8, width=8)._ops[-1]
    accepted = op_contract(json.dumps(resize_spec.to_dict()))["input_domains"]
    assert f"expects {' or '.join(accepted)} input" in str(excinfo.value)


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
        None if p is None or p.is_expr else p.value
        for p in (hints.height, hints.width, hints.channels, hints.batch)
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
