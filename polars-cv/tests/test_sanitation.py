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


def _png(
    width: int = 8, height: int = 8, color=(128, 64, 32), mode: str = "RGB"
) -> bytes:
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


def _leaf_dtype(dtype: pl.DataType) -> pl.DataType:
    """Peel nested List/Array wrappers to the innermost element dtype."""
    while isinstance(dtype, (pl.List, pl.Array)):
        dtype = dtype.inner
    return dtype


# ---------------------------------------------------------------------------
# 1. Plan == Exec (A1/A2/A3) — the headline invariant
# ---------------------------------------------------------------------------

# (label, build-pipeline-callable) for image-source pipelines whose output dtype
# is deterministic and must agree between planning and execution.
# Typed list sinks require a known element dtype at plan time: either an
# explicit source dtype, or a dtype-fixing op (cast). Both forms are exercised.
_DETERMINISTIC_LIST_PIPELINES = [
    (
        "resize_u8",
        lambda: Pipeline().source("image_bytes", dtype="u8").resize(height=6, width=6),
    ),
    (
        "grayscale_u8",
        lambda: (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .resize(height=6, width=6)
            .grayscale()
        ),
    ),
    (
        "cast_f32",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=6).cast("f32"),
    ),
    (
        "cast_f64",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=6).cast("f64"),
    ),
    (
        "grayscale_then_cast_f64",
        lambda: (
            Pipeline()
            .source("image_bytes")
            .resize(height=6, width=6)
            .grayscale()
            .cast("f64")
        ),
    ),
]


@plugin_required
@pytest.mark.parametrize(
    "label,build",
    _DETERMINISTIC_LIST_PIPELINES,
    ids=lambda v: v if isinstance(v, str) else "",
)
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
    pipe = (
        Pipeline()
        .source("image_bytes")
        .resize(height=6, width=6)
        .cast("f32")
        .blur(sigma=1.0)
    )
    expr = pl.col("out").cv.pipe(pipe).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized
    assert realized == pl.List(pl.List(pl.List(pl.Float32)))


@plugin_required
def test_plan_equals_exec_array_sink():
    """Array sink with explicit shape: planned == realized."""
    df = pl.DataFrame({"out": [_png(width=6, height=6)]})
    pipe = Pipeline().source("image_bytes", dtype="u8").resize(height=6, width=6)
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
def test_auto_16bit_image_list_sink_requires_explicit_dtype():
    """A2: an image source with no known dtype cannot feed a typed list sink.

    The decoded dtype is only known at runtime, so the planner must refuse to
    guess (it would silently fall back to u8 and diverge from the realized u16).
    The user is required to supply the dtype instead.
    """
    pipe = Pipeline().source("image_bytes")
    with pytest.raises(ValueError, match="(?i)explicit dtype"):
        pl.col("out").cv.pipe(pipe).sink("list")


@plugin_required
def test_auto_16bit_with_explicit_dtype_plan_equals_exec():
    """A2: with an explicit source dtype, the 16-bit image plans and realizes u16."""
    df = pl.DataFrame({"out": [_png(mode="I;16")]})
    pipe = Pipeline().source("image_bytes", dtype="u16")
    expr = pl.col("out").cv.pipe(pipe).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized
    assert _leaf_dtype(realized) == pl.UInt16


@plugin_required
def test_plan_equals_exec_binary_promote():
    """A3: a promoting binary op (true division) declares the promoted dtype.

    ``divide`` of two u8 images promotes to f32 (true division), and the planned
    dtype must match what execution produces — not the left operand's u8. The
    operands carry an explicit dtype so the typed sink is plannable.
    """
    df = pl.DataFrame({"out": [_png()]})
    left = pl.col("out").cv.pipe(
        Pipeline().source("image_bytes", dtype="u8").grayscale()
    )
    right = pl.col("out").cv.pipe(
        Pipeline().source("image_bytes", dtype="u8").grayscale()
    )
    expr = left.divide(right).sink("list")
    planned, realized = _planned_and_realized(df, expr)
    assert planned == realized
    assert _leaf_dtype(realized) == pl.Float32


def _planned_shape(pipe):
    """The pipeline's plan-time [H, W, C], using None for unknown/expr dims."""

    def known(p):
        return p.value if (p is not None and not p.is_expr) else None

    sh = pipe._shape_hints
    return [known(sh.height), known(sh.width), known(sh.channels)]


# (label, build-pipeline, png-mode) exercising the rank/channel rules end-to-end.
_SHAPE_PIPELINES = [
    (
        "resize_rgb",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=5),
        "RGB",
    ),
    (
        "resize_grayscale",
        lambda: Pipeline().source("image_bytes").resize(height=6, width=6).grayscale(),
        "RGB",
    ),
    (
        "resize_rgba_preserves_4ch",
        lambda: (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=4)
            .resize(height=6, width=6)
        ),
        "RGBA",
    ),
    (
        "cvt_gray_on_rgba_is_graya",
        lambda: (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=4)
            .resize(height=6, width=6)
            .convert_color("rgb", "gray")
        ),
        "RGBA",
    ),
    (
        "cvt_hsv_on_rgba_preserves_4ch",
        lambda: (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=4)
            .resize(height=6, width=6)
            .convert_color("rgb", "hsv")
        ),
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


def _emitted_op_names_from_source():
    """Op names actually emitted by the Python builders, scanned from source.

    Pipeline builders emit ``op="<name>"`` literals (pipeline.py) and the binary
    helpers emit ``_binary_op("<name>")`` / ``_add_binary_op("<name>")``
    (lazy.py). Scanning the source keeps the comparison drift-proof without a
    second hand-maintained list.
    """
    import re
    from pathlib import Path

    pkg = Path(polars_cv.__file__).parent
    names: set[str] = set()
    text = (pkg / "pipeline.py").read_text()
    names |= set(re.findall(r'op="([a-z_0-9]+)"', text))
    lazy = (pkg / "lazy.py").read_text()
    names |= set(re.findall(r'_(?:add_)?binary_op\("([a-z_]+)"', lazy))
    return names


def test_op_names_covers_all_emitted_ops():
    """Pipeline.OP_NAMES must list exactly the ops the builders actually emit.

    Guards against OP_NAMES silently under-listing (a new builder op missing
    from the registry) or over-listing (a stale entry no builder emits).
    """
    emitted = _emitted_op_names_from_source()
    declared = set(Pipeline.OP_NAMES)
    assert emitted == declared, (
        f"OP_NAMES out of sync with builders: "
        f"missing={sorted(emitted - declared)} stale={sorted(declared - emitted)}"
    )


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


_REQUIRED_LIB_HOOKS = (
    "op_contract",
    "op_output_dtype",
    "binary_output_dtype",
    "known_ops",
    "enum_variants",
)


@plugin_required
def test_lib_introspection_api_is_present():
    """A built plugin MUST expose every introspection hook (no false-green CI).

    The other parity tests ``pytest.skip`` when a hook is missing so they switch
    on automatically as the API lands. Once the plugin is actually built, a
    missing hook is a real regression (e.g. a function dropped from the
    ``#[pymodule]``), not a not-yet-implemented feature — so this guard turns it
    into a hard failure instead of a silent skip.
    """
    lib = getattr(polars_cv, "_lib", None)
    assert lib is not None, "compiled _lib missing despite the plugin .so being present"
    missing = [
        name for name in _REQUIRED_LIB_HOOKS if not callable(getattr(lib, name, None))
    ]
    assert not missing, f"_lib is built but missing introspection hooks: {missing}"


def _binary_op_names_from_source() -> set[str]:
    """Binary op names the lazy API emits via ``self._binary_op("<name>")``."""
    import re
    from pathlib import Path

    lazy = (Path(polars_cv.__file__).parent / "lazy.py").read_text()
    return set(re.findall(r'self\._binary_op\("([a-z_]+)"', lazy))


@plugin_required
def test_binary_output_dtype_authority():
    """The two-input dtype FFI resolves every binary op and encodes true division.

    Guards both the new ``binary_output_dtype`` hook and its op-name mapping
    against the binary ops the Python API actually emits (drift-proof: the names
    are scanned from source).
    """
    from polars_cv._lib import binary_output_dtype

    emitted = _binary_op_names_from_source()
    assert emitted, "no binary ops scanned from lazy.py — scan regex out of date?"
    for op in emitted:
        # Every emitted binary op must resolve through the FFI without error.
        result = binary_output_dtype(op, "u8", "u8")
        assert result in {"u8", "f32"}, f"{op}: unexpected dtype {result}"

    # True division promotes integers to float; other ops use plain promotion.
    assert binary_output_dtype("divide", "u8", "u8") == "f32"
    assert binary_output_dtype("ratio", "u16", "u16") == "f32"
    assert binary_output_dtype("divide", "f64", "f64") == "f64"
    assert binary_output_dtype("add", "u8", "u8") == "u8"
    assert binary_output_dtype("add", "u8", "u16") == "u16"
    assert binary_output_dtype("add", "u8", "f32") == "f32"
    # An unknown operand dtype keeps the result unknown (handled by the sink).
    assert binary_output_dtype("divide", "auto", "u8") == "auto"
    assert binary_output_dtype("add", "u8", "auto") == "auto"


# ---------------------------------------------------------------------------
# 2b. Contract authority (A1/A10) — the planner reads view-buffer's per-op
# contract (dtype, domain, rank, channel) instead of re-declaring it in Python.
# ---------------------------------------------------------------------------
#
# view-buffer's ViewDto is the single authority; the Python planner no longer
# keeps a parallel dtype/ndim/alpha table. These tests pin the planner to that
# authority so a Python special-case can't silently drift from execution (the
# class of bug that made the old blur contract say u8 while execution produced
# f32).

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
    "adjust_contrast": lambda: (
        Pipeline().source("image_bytes").adjust_contrast(factor=1.2)
    ),
    "adjust_gamma": lambda: Pipeline().source("image_bytes").adjust_gamma(gamma=1.2),
    "cvt_color": lambda: Pipeline().source("image_bytes").convert_color("rgb", "hsv"),
    "convolve2d": lambda: (
        Pipeline().source("image_bytes").convolve2d([0, 0, 0, 0, 1, 0, 0, 0, 0], 3)
    ),
    "erode": lambda: (
        Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3)
    ),
    "dilate": lambda: (
        Pipeline().source("image_bytes").grayscale().threshold(128).dilate(ksize=3)
    ),
    "morphology_gradient": lambda: (
        Pipeline()
        .source("image_bytes")
        .grayscale()
        .threshold(128)
        .morphology_gradient(ksize=3)
    ),
    "canny": lambda: Pipeline().source("image_bytes").grayscale().canny(),
    "equalize_histogram": lambda: (
        Pipeline().source("image_bytes").grayscale().equalize_histogram()
    ),
    "channel_select": lambda: Pipeline().source("image_bytes").channel_select(index=0),
    "channel_swap": lambda: (
        Pipeline().source("image_bytes").channel_swap(order=[2, 1, 0])
    ),
    "flip": lambda: Pipeline().source("image_bytes").flip([0]),
}


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
    re-declaring its own ndim/alpha rules."""
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


# view-buffer's `any` Domain is an internal identity domain (materialize) that is
# never surfaced to a Python pipeline, so it is excluded from the comparison.
_RUST_INTERNAL_DOMAINS = {"any"}


@plugin_required
def test_enum_parity_domain():
    """Python Domain values must equal the surfaced Rust Domain variant set (A4).

    The former Python-only `histogram` "domain" is gone (histogram buckets are a
    `vector` output whose struct schema is selected by the sink encoding), so the
    only remaining difference is Rust's internal `any` domain, which is never
    surfaced to a pipeline.
    """
    rust = _rust_enum_variants("Domain")
    if rust is None:
        pytest.skip("_lib.enum_variants() not built")
    import polars_cv._types as t

    surfaced = rust - _RUST_INTERNAL_DOMAINS
    py = {m.value for m in t.Domain}
    assert py == surfaced, f"Domain: python {py} != surfaced rust {surfaced}"


@plugin_required
@pytest.mark.parametrize(
    "enum_name",
    [
        "NormalizeMethod",
        "ColorSpace",
        "HashAlgorithm",
        "HistogramOutput",
        "PadMode",
        "PadPosition",
    ],
)
def test_enum_parity_api_enums(enum_name):
    """Each user-facing API enum must equal its view-buffer authority set (A4)."""
    rust = _rust_enum_variants(enum_name)
    if rust is None:
        pytest.skip("_lib.enum_variants() not built")
    import polars_cv._types as t

    py = {m.value for m in getattr(t, enum_name)}
    assert py == rust, f"{enum_name}: python {py} != rust {rust}"


@plugin_required
def test_enum_parity_filter_type_is_subset():
    """Python FilterType is a documented subset of view-buffer's FilterType.

    Rust also offers catmullrom/gaussian (and surfaces `Triangle` as "bilinear");
    polars-cv intentionally exposes only nearest/bilinear/lanczos3. Guard the
    subset so a new Python value can't escape the Rust authority.
    """
    rust = _rust_enum_variants("FilterType")
    if rust is None:
        pytest.skip("_lib.enum_variants() not built")
    import polars_cv._types as t

    py = {m.value for m in t.FilterType}
    assert py <= rust, f"FilterType: python {py} is not a subset of rust {rust}"
    assert {"catmullrom", "gaussian"} <= rust, "expected Rust-only filter variants"


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


def test_output_dtype_is_strategy_not_dtype_duplicate():
    """OutputDType is an out-dtype *strategy*, not a second copy of DType (A4).

    It carries the `preserve` strategy (keep input dtype, promoting ints to f32)
    that DType cannot express, and exposes only the handful of dtypes worth
    requesting as an output override — so it is deliberately kept distinct rather
    than folded into DType. Guard that it stays a strategy (i.e. not equal to the
    full dtype set and still offering `preserve`).
    """
    import polars_cv._types as t

    out_values = {m.value for m in t.OutputDType}
    dtype_values = {m.value for m in t.DType}
    assert "preserve" in out_values, "OutputDType must offer the preserve strategy"
    assert "preserve" not in dtype_values, "DType must not carry a strategy value"
    assert out_values != dtype_values, (
        "OutputDType must remain a strategy enum distinct from DType, not a duplicate"
    )


# ---------------------------------------------------------------------------
# Pipeline <-> LazyPipelineExpr method parity
# ---------------------------------------------------------------------------


def test_lazy_pipeline_method_parity():
    """Every chainable Pipeline method must exist on LazyPipelineExpr with an
    identical parameter list.

    The lazy forwarders are generated from Pipeline at import time
    (``polars_cv.lazy._install_pipeline_forwarders``), so parity holds by
    construction. This test guards that the generator stays wired up — and that
    any explicitly hand-written lazy methods (binary ops, ``label_reduce``,
    ``apply_mask``) keep signatures aligned with their Pipeline counterparts."""
    import inspect

    from polars_cv.lazy import (
        PIPELINE_ONLY_METHODS,
        LazyPipelineExpr,
        _chainable_pipeline_ops,
    )
    from polars_cv.pipeline import Pipeline

    # The single source of truth for the eager/lazy asymmetry lives in lazy.py.
    pipeline_methods = {
        name
        for name in dir(Pipeline)
        if not name.startswith("_") and callable(getattr(Pipeline, name))
    }
    assert set(_chainable_pipeline_ops()) == pipeline_methods - PIPELINE_ONLY_METHODS

    missing = sorted(
        n for n in _chainable_pipeline_ops() if not hasattr(LazyPipelineExpr, n)
    )
    assert not missing, f"LazyPipelineExpr is missing Pipeline methods: {missing}"

    for name in _chainable_pipeline_ops():
        p_sig = inspect.signature(getattr(Pipeline, name))
        l_sig = inspect.signature(getattr(LazyPipelineExpr, name))
        # Compare parameter names, kinds and defaults (skip `self`); string
        # defaults compare equal to their str-enum counterparts.
        p_params = [
            (p.name, p.kind, p.default) for p in list(p_sig.parameters.values())[1:]
        ]
        l_params = [
            (p.name, p.kind, p.default) for p in list(l_sig.parameters.values())[1:]
        ]
        assert p_params == l_params, (
            f"Signature drift on '{name}':\n  Pipeline: {p_params}\n  Lazy:     {l_params}"
        )


def test_lazy_stub_is_current():
    """The committed ``lazy.pyi`` must match what ``gen_lazy_stub.py`` produces.

    The stub is generated from the runtime ``LazyPipelineExpr`` so IDEs and type
    checkers see the auto-generated forwarders. This guards against the stub
    drifting after a Pipeline change without a regeneration."""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "gen_lazy_stub.py"
    spec = importlib.util.spec_from_file_location("gen_lazy_stub", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    stub_path = Path(module._STUB_PATH)
    assert stub_path.exists(), "lazy.pyi is missing; run python scripts/gen_lazy_stub.py"
    assert stub_path.read_text() == module.generate_stub(), (
        "lazy.pyi is out of date. Run: python scripts/gen_lazy_stub.py"
    )
