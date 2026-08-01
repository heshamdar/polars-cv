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

from .conftest import plugin_required

# ---------------------------------------------------------------------------
# 1. Only _push_op may mutate _ops
# ---------------------------------------------------------------------------

#: The one function permitted to mutate ``Pipeline._ops``.
_SOLE_MUTATOR = "_push_op"


def _pipeline_ast() -> ast.ClassDef:
    source = Path(polars_cv.pipeline.__file__).read_text()
    tree = ast.parse(source)
    return next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Pipeline"
    )


def _mutates_ops(node: ast.AST) -> bool:
    """True if *node* appends to, assigns into, or extends ``*._ops``."""
    for sub in ast.walk(node):
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
                # Whole-list rebinding is how a Pipeline is *constructed*
                # (_clone, _create_sub_pipeline), not how an op is appended.
                if isinstance(sub, ast.AugAssign):
                    return True
    return False


def test_op_append_is_structurally_exclusive() -> None:
    """``_push_op`` is the only function that may append to ``_ops``.

    This is the contract that makes the append sequence unskippable: a builder
    cannot add an operation without also running the domain check, the schema
    fold and the shape-hint update, because it never touches ``_ops`` at all.
    """
    offenders = [
        method.name
        for method in _pipeline_ast().body
        if isinstance(method, ast.FunctionDef)
        and method.name != _SOLE_MUTATOR
        and _mutates_ops(method)
    ]
    assert not offenders, (
        f"only Pipeline.{_SOLE_MUTATOR}() may mutate _ops, but these also do: "
        f"{offenders}. Route them through _append_op()/_push_op() so the "
        f"plan-time state cannot be updated by halves."
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
        if isinstance(m, ast.FunctionDef) and m.name == _SOLE_MUTATOR
    )
    called = {
        sub.func.attr
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
    }
    assert "_update_output_dtype" in called
    assert "_update_shape_hints" in called

    # The hint update must not sit inside an `if`: it applies to every op.
    guarded = {
        sub.func.attr
        for branch in ast.walk(fn)
        if isinstance(branch, ast.If)
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
    """
    leaked = [n for n in dir(Pipeline) if n.startswith("DOMAIN_")]
    assert not leaked, f"Pipeline must not re-declare domain constants: {leaked}"
    assert not hasattr(Pipeline, "_validate_domain"), (
        "_validate_domain re-declared each op's input domain in Python; the "
        "check now reads op_contract(...)['input_domain']"
    )
    source = Path(polars_cv.pipeline.__file__).read_text()
    assert "_validate_domain" not in source
    assert "DOMAIN_BUFFER" not in source


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
    expected = op_contract(json.dumps(resize_spec.to_dict()))["input_domain"]
    assert f"expects {expected} input" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. Eager and lazy must agree
# ---------------------------------------------------------------------------

#: Per-op arguments for the eager/lazy parity sweep.
#:
#: ``None`` marks an op that cannot be compared by this harness, with the
#: reason. Everything else is ``(domain, kwargs)`` where *domain* selects the
#: base pipeline. The table is completeness-asserted against the real
#: chainable-op list below, so a new op cannot be added without a decision here.
_BUFFER, _CONTOUR = "buffer", "contour"

_OP_CASES: dict[str, tuple[str, dict] | None] = {
    # --- buffer -> buffer -------------------------------------------------
    "adjust_brightness": (_BUFFER, {"factor": 1.2}),
    "adjust_contrast": (_BUFFER, {"factor": 1.2}),
    "adjust_gamma": (_BUFFER, {"gamma": 1.5}),
    "blur": (_BUFFER, {"sigma": 1.0}),
    "canny": (_BUFFER, {}),
    "cast": (_BUFFER, {"dtype": "f32"}),
    "channel_select": (_BUFFER, {"index": 0}),
    "channel_swap": (_BUFFER, {"order": [2, 1, 0]}),
    "clamp": (_BUFFER, {"min_val": 0.0, "max_val": 1.0}),
    "convert_color": (_BUFFER, {"from_space": "rgb", "to_space": "hsv"}),
    "convolve2d": (_BUFFER, {"kernel": [0.0] * 9, "ksize": 3}),
    "crop": (_BUFFER, {"top": 0, "left": 0, "height": 50, "width": 50}),
    "dilate": (_BUFFER, {"ksize": 3}),
    "equalize_histogram": (_BUFFER, {}),
    "erode": (_BUFFER, {"ksize": 3}),
    "flip": (_BUFFER, {"axes": [0]}),
    "flip_h": (_BUFFER, {}),
    "flip_v": (_BUFFER, {}),
    "grayscale": (_BUFFER, {}),
    "invert": (_BUFFER, {}),
    "laplacian": (_BUFFER, {}),
    "letterbox": (_BUFFER, {"height": 128, "width": 128}),
    "morphology_close": (_BUFFER, {"ksize": 3}),
    "morphology_gradient": (_BUFFER, {"ksize": 3}),
    "morphology_open": (_BUFFER, {"ksize": 3}),
    "normalize": (_BUFFER, {"method": "minmax"}),
    "pad": (_BUFFER, {"top": 10, "bottom": 10, "left": 0, "right": 0}),
    "pad_to_size": (_BUFFER, {"height": 150, "width": 250}),
    "relu": (_BUFFER, {}),
    "reshape": (_BUFFER, {"shape": [100, 200, 3]}),
    "resize": (_BUFFER, {"height": 64, "width": 32}),
    "resize_max": (_BUFFER, {"max_size": 120}),
    "resize_min": (_BUFFER, {"min_size": 40}),
    "resize_scale": (_BUFFER, {"scale_x": 0.5, "scale_y": 0.5}),
    "resize_to_height": (_BUFFER, {"height": 50}),
    "resize_to_width": (_BUFFER, {"width": 50}),
    "rotate": (_BUFFER, {"angle": 90}),
    "rotate_and_scale": (
        _BUFFER,
        {"angle": 45.0, "scale": 1.5, "center": (50.0, 100.0), "output_size": (64, 64)},
    ),
    "scale": (_BUFFER, {"factor": 2.0}),
    "sharpen": (_BUFFER, {}),
    "shear": (_BUFFER, {"sx": 0.2, "output_size": (100, 200)}),
    "sobel": (_BUFFER, {}),
    "threshold": (_BUFFER, {"value": 128}),
    "to_bgr": (_BUFFER, {}),
    "to_hsv": (_BUFFER, {}),
    "to_lab": (_BUFFER, {}),
    "to_ycbcr": (_BUFFER, {}),
    "transpose": (_BUFFER, {"axes": [1, 0, 2]}),
    "warp_affine": (
        _BUFFER,
        {"matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "output_size": (64, 64)},
    ),
    # --- buffer -> other domains -----------------------------------------
    "extract_contours": (_BUFFER, {}),
    "extract_shape": (_BUFFER, {}),
    "histogram": (_BUFFER, {"bins": 8}),
    "perceptual_hash": (_BUFFER, {}),
    "reduce_argmax": (_BUFFER, {"axis": 0}),
    "reduce_argmin": (_BUFFER, {"axis": 0}),
    "reduce_max": (_BUFFER, {}),
    "reduce_mean": (_BUFFER, {}),
    "reduce_min": (_BUFFER, {}),
    "reduce_percentile": (_BUFFER, {"q": 50.0}),
    "reduce_popcount": (_BUFFER, {}),
    "reduce_std": (_BUFFER, {}),
    "reduce_sum": (_BUFFER, {}),
    # --- contour domain ---------------------------------------------------
    "area": (_CONTOUR, {}),
    "bounding_box": (_CONTOUR, {}),
    "centroid": (_CONTOUR, {}),
    "convex_hull": (_CONTOUR, {}),
    "perimeter": (_CONTOUR, {}),
    "rasterize": (_CONTOUR, {"width": 32, "height": 32}),
    "scale_contour": (_CONTOUR, {"sx": 2.0, "sy": 2.0}),
    "simplify": (_CONTOUR, {"tolerance": 1.0}),
    "translate": (_CONTOUR, {"dx": 1.0, "dy": 2.0}),
    # --- not comparable by this harness -----------------------------------
    "assert_shape": None,  # sets hints directly; covered by its own test below
    "label_reduce": None,  # takes a contour *column*, not a plain value
    "on_error": None,  # graph-level policy, appends no op
    "on_null_param": None,  # graph-level policy, appends no op
}


def _base(domain: str) -> Pipeline:
    """A pipeline in *domain* with fully known, non-square shape hints."""
    pipe = (
        Pipeline().source("image_bytes").assert_shape(height=100, width=200, channels=3)
    )
    if domain == _CONTOUR:
        return pipe.grayscale().threshold(128).extract_contours()
    return pipe


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
    base = _base(domain)

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
    asserted_then_gray = (
        Pipeline()
        .source("image_bytes", dtype="u8")
        .assert_shape(channels=3)
        .grayscale()
    )
    assert asserted_then_gray._shape_hints.channels.value == 1


# ---------------------------------------------------------------------------
# 4. End-to-end: plan must equal exec (the bugs this phase fixes)
# ---------------------------------------------------------------------------


@pytest.fixture
def non_square_png() -> bytes:
    """A 100x200 RGB PNG — non-square so an H/W swap cannot cancel out."""
    buf = io.BytesIO()
    Image.fromarray(np.zeros((100, 200, 3), np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _assert_plan_equals_exec(df: pl.DataFrame, expr: pl.Expr) -> None:
    lf = df.lazy().select(out=expr)
    planned = lf.collect_schema()["out"]
    produced = lf.collect()["out"].dtype
    assert planned == produced, f"planned {planned} but produced {produced}"


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
