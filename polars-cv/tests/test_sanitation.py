"""
Sanitation suite — the permanent "implemented once / plan == exec" guards.

These tests are the enforcement mechanism for the foundational refactor described
in the project plan. They are intentionally strict: each one fails if a piece of
knowledge (output dtype, domain, an enum, an op registration) is implemented in
more than one place and the copies drift.

Categories
----------
1. ``test_plan_equals_exec_*`` — the headline invariant: the dtype Polars infers
   at planning time (``collect_schema``) MUST equal the dtype actually produced at
   execution time (``collect``). This is the contract the whole architecture rests
   on (assessment findings A1/A2/A3).
2. ``test_registry_parity_*`` — every operation known to the Python ``Pipeline``
   builder must be known to the Rust executor and have exactly one contract
   (findings B1/B2). Activated once the Rust introspection API lands; until then
   they skip with a clear reason so they switch on automatically.
3. ``test_enum_parity_*`` — Python user-facing enums must match the Rust enum
   variants exactly; no second hand-authored copy (finding A4).
4. ``test_no_duplicate_enum_*`` — the overlapping Python dtype enums collapse into
   one (finding A4 / "no repeated enums").

Tests that depend on not-yet-implemented introspection hooks are written now (so
the target is unambiguous) and skip until the hook exists, rather than being
deleted and re-added later.
"""

from __future__ import annotations

import io

import polars as pl
import pytest

import polars_cv
from polars_cv import Pipeline
from tests.conftest import plugin_required

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png(width: int = 8, height: int = 8, color=(128, 64, 32), mode: str = "RGB") -> bytes:
    """Encode a small test image to PNG bytes in the requested PIL mode."""
    PIL = pytest.importorskip("PIL.Image")
    if mode == "I;16":
        img = PIL.new("I;16", (width, height), 4096)
    elif mode == "L":
        img = PIL.new("L", (width, height), color[0])
    else:
        img = PIL.new(mode, (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _planned_and_realized(df: pl.DataFrame, expr: pl.Expr, col: str = "out"):
    """Return (planned_dtype, realized_dtype) for an expression over ``df``."""
    lf = df.lazy().select(**{col: expr})
    planned = lf.collect_schema()[col]
    realized = lf.collect()[col].dtype
    return planned, realized


# ---------------------------------------------------------------------------
# 1. Plan == Exec (A1/A2/A3) — the headline invariant
# ---------------------------------------------------------------------------

# (label, build-pipeline-callable) for image-source pipelines whose output dtype
# is deterministic and must agree between planning and execution.
_DETERMINISTIC_LIST_PIPELINES = [
    ("resize_u8", lambda: Pipeline().source("image_bytes").resize(height=6, width=6)),
    ("grayscale_u8", lambda: Pipeline().source("image_bytes").resize(height=6, width=6).grayscale()),
    ("cast_f32", lambda: Pipeline().source("image_bytes").resize(height=6, width=6).cast("f32")),
    ("cast_f64", lambda: Pipeline().source("image_bytes").resize(height=6, width=6).cast("f64")),
    (
        "grayscale_then_cast_f64",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=6).grayscale().cast("f64"),
    ),
]


@plugin_required
@pytest.mark.parametrize("label,build", _DETERMINISTIC_LIST_PIPELINES, ids=lambda v: v if isinstance(v, str) else "")
def test_plan_equals_exec_list_sink(label, build):
    """Planned list-sink dtype must equal the realized dtype (real PNG data)."""
    if not isinstance(label, str):  # parametrize passes both tuple elements
        pytest.skip("paired param")
    df = pl.DataFrame({"out": [_png()]})
    expr = pl.col("out").cv.pipe(build()).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized, f"{label}: planned {planned} != realized {realized}"


@plugin_required
def test_plan_equals_exec_blur_preserves_float_dtype():
    """blur preserves dtype: `cast(f32).blur()` must plan AND produce f32.

    Regression for an A1 contract drift: the Python blur contract declared
    FIXED_U8 while view-buffer's blur preserves the input dtype, so this
    pipeline previously failed the execution-time dtype guard.
    """
    df = pl.DataFrame({"out": [_png()]})
    pipe = Pipeline().source("image_bytes").resize(height=6, width=6).cast("f32").blur(sigma=1.0)
    expr = pl.col("out").cv.pipe(pipe).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized
    assert realized == pl.List(pl.List(pl.List(pl.Float32)))


@plugin_required
def test_plan_equals_exec_array_sink():
    """Array sink with explicit shape: planned == realized."""
    df = pl.DataFrame({"out": [_png(width=6, height=6)]})
    pipe = Pipeline().source("image_bytes").resize(height=6, width=6)
    expr = pl.col("out").cv.pipe(pipe).sink("array", shape=[6, 6, 3])
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized


@plugin_required
def test_plan_equals_exec_scalar_reduction():
    """Global reduction to a scalar: planned == realized."""
    df = pl.DataFrame({"out": [_png()]})
    pipe = Pipeline().source("image_bytes").grayscale().reduce_sum()
    expr = pl.col("out").cv.pipe(pipe).sink("native")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized


@plugin_required
@pytest.mark.xfail(
    reason="A2: image-source dtype is 'auto'; the plan-time guess can diverge from "
    "the actually-decoded dtype (e.g. 16-bit images). Fixed in Phase 1.",
    strict=False,
)
def test_plan_equals_exec_auto_16bit_image():
    """16-bit image with no dtype-fixing op: planned 'auto' must equal realized u16."""
    df = pl.DataFrame({"out": [_png(mode="I;16")]})
    pipe = Pipeline().source("image_bytes")
    expr = pl.col("out").cv.pipe(pipe).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized


@plugin_required
@pytest.mark.xfail(
    reason="A3: binary-op output dtype is copied from the left operand only, ignoring "
    "the operator's promotion semantics. Fixed in Phase 4.",
    strict=False,
)
def test_plan_equals_exec_binary_promote():
    """A promoting binary op (divide) must declare the promoted dtype, not the left's."""
    df = pl.DataFrame({"out": [_png()]})
    left = pl.col("out").cv.pipe(Pipeline().source("image_bytes").grayscale())
    right = pl.col("out").cv.pipe(Pipeline().source("image_bytes").grayscale())
    expr = left.divide(right).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized


def _planned_shape(pipe):
    """The pipeline's plan-time [H, W, C], using None for unknown/expr dims."""

    def known(p):
        return p.value if (p is not None and not p.is_expr) else None

    sh = pipe._shape_hints
    return [known(sh.height), known(sh.width), known(sh.channels)]


# (label, build-pipeline, png-mode) exercising the rank/channel rules end-to-end.
_SHAPE_PIPELINES = [
    ("resize_rgb", lambda: Pipeline().source("image_bytes").resize(height=6, width=5), "RGB"),
    (
        "resize_grayscale",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=6).grayscale(),
        "RGB",
    ),
    (
        "resize_rgba_preserves_4ch",
        lambda: Pipeline()
        .source("image_bytes")
        .assert_shape(channels=4)
        .resize(height=6, width=6),
        "RGBA",
    ),
    (
        "cvt_gray_on_rgba_is_graya",
        lambda: Pipeline()
        .source("image_bytes")
        .assert_shape(channels=4)
        .resize(height=6, width=6)
        .cvt_color("rgb", "gray"),
        "RGBA",
    ),
    (
        "cvt_hsv_on_rgba_preserves_4ch",
        lambda: Pipeline()
        .source("image_bytes")
        .assert_shape(channels=4)
        .resize(height=6, width=6)
        .cvt_color("rgb", "hsv"),
        "RGBA",
    ),
]


@plugin_required
@pytest.mark.parametrize(
    "label,build,mode", _SHAPE_PIPELINES, ids=[c[0] for c in _SHAPE_PIPELINES]
)
def test_plan_equals_exec_shape(label, build, mode):
    """Each shape dimension the planner claims to know must match the realized
    array shape (plan == exec for shape, the WS-1 rank/channel invariant)."""
    df = pl.DataFrame({"out": [_png(width=8, height=8, mode=mode)]})
    pipe = build()
    realized = list(
        polars_cv.numpy_from_struct(
            df.lazy()
            .select(out=pl.col("out").cv.pipe(pipe).sink("numpy"))
            .collect()["out"][0]
        ).shape
    )
    planned = _planned_shape(pipe)
    for i, p in enumerate(planned):
        if p is not None:
            assert p == realized[i], (
                f"{label}: planned dim {i} = {p} but realized = {realized[i]} "
                f"(planned {planned} vs realized {realized})"
            )


# ---------------------------------------------------------------------------
# 2. Registry parity (B1/B2) — one op, known everywhere, contracted once
# ---------------------------------------------------------------------------


def _known_ops_from_rust():
    """Op names the Rust executor accepts, or None if the hook isn't built yet."""
    lib = getattr(polars_cv, "_lib", None)
    fn = getattr(lib, "known_ops", None) if lib is not None else None
    return set(fn()) if callable(fn) else None


@plugin_required
def test_registry_parity_pipeline_ops_are_executable():
    """Every op a Pipeline can emit must be known to the Rust executor (B1)."""
    rust_ops = _known_ops_from_rust()
    if rust_ops is None:
        pytest.skip("_lib.known_ops() not implemented yet (Phase 3)")
    pipeline_ops = getattr(Pipeline, "OP_NAMES", None)
    if pipeline_ops is None:
        pytest.skip("Pipeline.OP_NAMES not implemented yet (Phase 3)")
    missing = set(pipeline_ops) - rust_ops
    assert not missing, f"Pipeline ops with no Rust executor arm: {sorted(missing)}"


@plugin_required
def test_registry_parity_no_dead_contracts():
    """Ops the Pipeline never emits are not executable (B2: sobel/laplacian/sharpen)."""
    import json

    lib = getattr(polars_cv, "_lib", None)
    contract_fn = getattr(lib, "op_contract", None) if lib is not None else None
    if not callable(contract_fn):
        pytest.skip("_lib.op_contract() not implemented yet (Phase 1/3)")
    # sobel/laplacian/sharpen lower to convolve2d; they are not real executable
    # ops, so resolving them must fail (their standalone contracts are dead, B2).
    for lowered in ("sobel", "laplacian", "sharpen"):
        with pytest.raises(Exception):
            contract_fn(json.dumps({"op": lowered}))


# ---------------------------------------------------------------------------
# 2b. Contract parity (A1) — Python dtype contracts match the Rust authority
# ---------------------------------------------------------------------------
#
# view-buffer's ViewDto::output_dtype_rule() is the single authority for an op's
# output dtype. The Python OPERATION_CONTRACTS table currently re-declares the
# same knowledge as DTypeEffect; this test fails if the two drift (the exact
# class of bug that made the blur contract say u8 while execution produced f32).

# Python DTypeEffect.value -> canonical rule string in _lib.op_contract["dtype_rule"].
_EFFECT_TO_RULE = {
    "preserve": "preserve",
    "promote": "promote",
    "u8": "fixed:u8",
    "f32": "fixed:f32",
    "f64": "fixed:f64",
    "i64": "fixed:i64",
    "u64": "fixed:u64",
    "u32": "fixed:u32",
    "config_f32": "config:f32",
}

# op name -> builder producing a Pipeline whose LAST op is the op under test,
# using only literal params (so resolve_op needs no expression columns).
_OP_BUILDERS = {
    "resize": lambda: Pipeline().source("image_bytes").resize(height=4, width=4),
    "grayscale": lambda: Pipeline().source("image_bytes").grayscale(),
    "threshold": lambda: Pipeline().source("image_bytes").grayscale().threshold(128),
    "blur": lambda: Pipeline().source("image_bytes").blur(sigma=1.0),
    "scale": lambda: Pipeline().source("image_bytes").scale(2.0),
    "clamp": lambda: Pipeline().source("image_bytes").clamp(0.0, 255.0),
    "relu": lambda: Pipeline().source("image_bytes").relu(),
    "invert": lambda: Pipeline().source("image_bytes").invert(),
    "adjust_contrast": lambda: Pipeline().source("image_bytes").adjust_contrast(factor=1.2),
    "adjust_gamma": lambda: Pipeline().source("image_bytes").adjust_gamma(gamma=1.2),
    "cvt_color": lambda: Pipeline().source("image_bytes").cvt_color("rgb", "hsv"),
    "convolve2d": lambda: Pipeline().source("image_bytes").convolve2d([0, 0, 0, 0, 1, 0, 0, 0, 0], 3),
    "erode": lambda: Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3),
    "dilate": lambda: Pipeline().source("image_bytes").grayscale().threshold(128).dilate(ksize=3),
    "morphology_gradient": lambda: Pipeline().source("image_bytes").grayscale().threshold(128).morphology_gradient(ksize=3),
    "canny": lambda: Pipeline().source("image_bytes").grayscale().canny(),
    "equalize_histogram": lambda: Pipeline().source("image_bytes").grayscale().equalize_histogram(),
    "channel_select": lambda: Pipeline().source("image_bytes").channel_select(index=0),
    "channel_swap": lambda: Pipeline().source("image_bytes").channel_swap(order=[2, 1, 0]),
    "flip": lambda: Pipeline().source("image_bytes").flip([0]),
}

# Ops deliberately NOT strict-compared here, each with a reason. Keeping this
# explicit (rather than "everything not in _OP_BUILDERS") means a newly-added
# contract must be classified — the completeness test below fails otherwise.
_OP_PARITY_EXCEPTIONS = {
    # Param-dependent output dtype (rule depends on the dtype/out_dtype param).
    "cast": "dtype set by param",
    "normalize": "configurable out_dtype",
    # Domain-changing ops: produce scalar/vector/contour, not a buffer dtype, so
    # the buffer-dtype rule is intentionally not the observable output dtype.
    "reduce_sum": "scalar domain", "reduce_mean": "scalar domain",
    "reduce_std": "scalar domain", "reduce_max": "scalar domain",
    "reduce_min": "scalar domain", "reduce_popcount": "scalar domain",
    "reduce_percentile": "scalar domain", "reduce_argmax": "scalar domain",
    "reduce_argmin": "scalar domain", "extract_shape": "vector domain",
    "label_reduce": "vector domain", "histogram": "vector domain",
    "perceptual_hash": "vector domain", "rasterize": "contour->buffer source op",
    "contour_area": "contour domain", "contour_perimeter": "contour domain",
    "contour_centroid": "contour domain", "contour_bounding_box": "contour domain",
    # Graph-level / multi-input or complex-param ops not yet covered by a builder.
    "channel_merge": "multi-input graph op", "warp_affine": "matrix params",
    "rotate": "param-dependent fast-path vs affine", "reshape": "param-dependent ndim",
    "transpose": "needs valid axes", "crop": "covered by view rule, builder TBD",
    "pad": "builder TBD", "pad_to_size": "builder TBD", "letterbox": "builder TBD",
    "resize_scale": "deferred resize, builder TBD",
    "resize_to_height": "deferred resize, builder TBD",
    "resize_to_width": "deferred resize, builder TBD",
    "resize_max": "deferred resize, builder TBD",
    "resize_min": "deferred resize, builder TBD",
}


def test_contract_parity_completeness():
    """Every contracted op must be either parity-checked or explicitly excepted."""
    from polars_cv._types import OPERATION_CONTRACTS

    classified = set(_OP_BUILDERS) | set(_OP_PARITY_EXCEPTIONS)
    uncovered = set(OPERATION_CONTRACTS) - classified
    assert not uncovered, (
        f"Contracts not classified for parity (add a builder or an exception): {sorted(uncovered)}"
    )


@plugin_required
@pytest.mark.parametrize("op_name", sorted(_OP_BUILDERS))
def test_contract_parity_dtype_rule(op_name):
    """Python DTypeEffect must equal the Rust ViewDto::output_dtype_rule (A1)."""
    import json

    from polars_cv._types import OPERATION_CONTRACTS

    contract_fn = getattr(getattr(polars_cv, "_lib", None), "op_contract", None)
    if not callable(contract_fn):
        pytest.skip("_lib.op_contract() not built")

    pipe = _OP_BUILDERS[op_name]()
    op_json = json.dumps(pipe._ops[-1].to_dict())
    rust_rule = contract_fn(op_json)["dtype_rule"]

    effect = OPERATION_CONTRACTS[op_name].dtype_effect.value
    expected = _EFFECT_TO_RULE[effect]
    assert rust_rule == expected, (
        f"{op_name}: Python contract says {expected!r} but Rust authority says {rust_rule!r}"
    )


@plugin_required
@pytest.mark.parametrize("op_name", sorted(_OP_BUILDERS))
def test_planner_domain_is_sourced_from_rust(op_name):
    """The planner derives each op's output domain from the view-buffer
    contract (ViewDto::output_domain) rather than a Python domain table (A10).

    The former Pipeline._OPERATION_OUTPUT_DOMAIN dict is gone; this guards
    against a special-case in _compute_output_domain_dtype_ndim diverging from
    the Rust authority for buffer-producing ops.
    """
    import json

    contract_fn = getattr(getattr(polars_cv, "_lib", None), "op_contract", None)
    if not callable(contract_fn):
        pytest.skip("_lib.op_contract() not built")

    pipe = _OP_BUILDERS[op_name]()
    rust_domain = contract_fn(json.dumps(pipe._ops[-1].to_dict()))["output_domain"]
    planned_domain, _, _ = Pipeline._compute_output_domain_dtype_ndim(
        pipe._ops, initial_domain="buffer", initial_dtype="u8"
    )
    assert planned_domain == rust_domain, (
        f"{op_name}: planner domain {planned_domain!r} != Rust authority "
        f"{rust_domain!r}"
    )


@plugin_required
@pytest.mark.parametrize("op_name", sorted(_OP_BUILDERS))
def test_contract_exposes_rank_and_channel_rules(op_name):
    """Every op's contract exposes a rank_rule and channel_rule in the known
    vocabulary — the single authority the Python planner reads instead of
    re-declaring an NdimEffect/AlphaMode of its own."""
    import json

    contract_fn = getattr(getattr(polars_cv, "_lib", None), "op_contract", None)
    if not callable(contract_fn):
        pytest.skip("_lib.op_contract() not built")

    contract = contract_fn(json.dumps(_OP_BUILDERS[op_name]()._ops[-1].to_dict()))
    rank, channel = contract["rank_rule"], contract["channel_rule"]

    assert rank in ("preserve", "reduce_one", "unknown") or (
        rank.startswith("fixed:") and rank.split(":", 1)[1].isdigit()
    ), f"{op_name}: unexpected rank_rule {rank!r}"
    assert channel in ("preserve", "n/a", "unknown") or (
        channel.startswith(("fixed:", "strip_restore:"))
        and channel.split(":", 1)[1].isdigit()
    ), f"{op_name}: unexpected channel_rule {channel!r}"


# ---------------------------------------------------------------------------
# 3. Enum parity (A4) — Python user enums match Rust variants
# ---------------------------------------------------------------------------


def _rust_enum_variants(enum_name):
    fn = getattr(getattr(polars_cv, "_lib", None), "enum_variants", None)
    return set(fn(enum_name)) if callable(fn) else None


@plugin_required
def test_enum_parity_dtype():
    """Python DType values must equal the Rust DType variant set (A4)."""
    rust = _rust_enum_variants("DType")
    if rust is None:
        pytest.skip("_lib.enum_variants() not built")
    import polars_cv._types as t

    py = {m.value for m in t.DType}
    assert py == rust, f"DType: python {py} != rust {rust}"


@plugin_required
@pytest.mark.xfail(
    reason="A4: Domain vocabularies diverge — Rust has `any`, Python has "
    "`histogram`. Reconciled in Phase 2 (one Domain enum across the boundary).",
    strict=True,
)
def test_enum_parity_domain():
    """Python Domain values must equal the Rust Domain variant set (A4)."""
    rust = _rust_enum_variants("Domain")
    if rust is None:
        pytest.skip("_lib.enum_variants() not built")
    import polars_cv._types as t

    py = {m.value for m in t.Domain}
    assert py == rust, f"Domain: python {py} != rust {rust}"


# SourceFormat/SinkFormat are intentionally NOT enum-parity-checked: view-buffer
# defines its own (CamelCase) format enums while the graph boundary uses plain
# strings and Python defines a third set — a three-way representation split to
# consolidate in Phase 2, not a simple drift to assert away here.


# ---------------------------------------------------------------------------
# 4. No duplicate enums (A4 / "no repeated enums")
# ---------------------------------------------------------------------------


def test_no_duplicate_expected_dtype_enum():
    """ExpectedDType was an exact duplicate of DType and must not exist (A4)."""
    import polars_cv._types as t

    assert not hasattr(t, "ExpectedDType"), "ExpectedDType should be folded into DType"


@pytest.mark.xfail(
    reason="A4: OutputDType (override options + PRESERVE) overlaps DType and is "
    "consolidated once the Rust OutputDTypeRule authority lands. Phase 2.",
    strict=True,
)
def test_no_duplicate_output_dtype_enum():
    """The configurable-output-dtype enum collapses into the single dtype authority."""
    import polars_cv._types as t

    assert not hasattr(t, "OutputDType"), "OutputDType should be folded into DType"
