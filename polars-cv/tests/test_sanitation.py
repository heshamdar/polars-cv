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


def _lib():
    """The compiled ``_lib`` submodule, importing it on demand (or ``None``).

    ``polars_cv._lib`` is a compiled submodule that ``polars_cv/__init__`` does
    not eagerly import, so ``_lib()`` is ``None`` until
    some other code triggers the import. Going through an explicit
    ``import polars_cv._lib`` makes these introspection tests behave identically
    whether run in isolation or after other tests — no order-dependent false
    skips (nor a false "compiled _lib missing" failure). Returns ``None`` only
    when the plugin genuinely isn't built.
    """
    try:
        import polars_cv._lib as lib

        return lib
    except ImportError:
        return None


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
    lib = _lib()
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
def test_registry_parity_all_rust_ops_are_reachable():
    """Every op the Rust executor knows must be reachable from the Python API.

    The forward test guards ``OP_NAMES ⊆ known_ops()``. This is the reverse
    direction: ``known_ops() ⊆ OP_NAMES``. Together they pin an exact equality,
    so a Rust ``resolve_op`` arm registered in ``KNOWN_OPS`` cannot sit
    unreachable from any ``Pipeline``/lazy builder (the gap that hid
    ``channel_merge`` and the graph-path contour ops before this suite existed).

    Graph geometry ops are exposed via the ``Pipeline`` builders; the separate
    ``.contour``/``.point``/``.bbox`` namespace plugins do NOT go through
    ``vb_graph``/``known_ops()`` and so are (correctly) not part of this set.
    """
    rust_ops = _known_ops_from_rust()
    if rust_ops is None:
        pytest.skip("_lib.known_ops() not implemented yet (Phase 3)")
    pipeline_ops = set(Pipeline.OP_NAMES)
    unreachable = rust_ops - pipeline_ops
    assert not unreachable, (
        "Rust ops in KNOWN_OPS with no Python builder that emits them "
        f"(dead or unconnected graph path): {sorted(unreachable)}"
    )


def _rust_src_dir():
    """The crate ``src/`` dir in a source checkout, or None (installed wheel)."""
    from pathlib import Path

    # python/polars_cv/__init__.py -> ../../src
    src = Path(polars_cv.__file__).resolve().parent.parent.parent / "src"
    return src if (src / "lib.rs").exists() else None


def test_namespace_plugin_symbols_match_registrations():
    """The namespace plugin surface is connected in BOTH directions.

    The ``.contour``/``.point``/``.bbox``/``.cv`` namespace accessors call
    individually-registered ``#[polars_expr]`` functions by name (bypassing the
    ``vb_graph``/``known_ops()`` graph path, so the registry-parity tests don't
    cover them). Both directions are pinned, mirroring the graph-path guarantee:

    - Forward: every ``_plugin("name")`` call resolves to a registered Rust
      symbol — a typo or rename (e.g. ``contour_bbox`` vs
      ``contour_bounding_box``) fails here instead of only at execution time.
    - Reverse: every registered namespace ``#[polars_expr]`` symbol is actually
      reached by a ``_plugin(...)`` call — a Rust namespace function left
      unconnected to the Python API (the same orphan class the graph-path
      reverse-parity test guards) fails here.
    """
    import re
    from pathlib import Path

    src = _rust_src_dir()
    if src is None:
        pytest.skip("Rust sources not available (installed wheel)")

    pkg = Path(polars_cv.__file__).parent
    called: set[str] = set()
    for py in pkg.rglob("*.py"):
        # `\s*` spans the newline for multi-line `_plugin(\n  "name", ...)` calls.
        called |= set(re.findall(r'_plugin\(\s*"([a-z_0-9]+)"', py.read_text()))
    assert called, "no _plugin(...) calls found — scan is broken"

    registered: set[str] = set()
    for rs in ("contour.rs", "point.rs", "image_metadata.rs", "read_bytes.rs"):
        text = (src / rs).read_text()
        # `#[polars_expr(...)]` immediately precedes `pub fn <name>` / `fn <name>`.
        registered |= set(
            re.findall(r"#\[polars_expr[^\]]*\]\s*(?:pub\s+)?fn\s+([a-z_0-9]+)", text)
        )
    assert registered, "no #[polars_expr] fns found — scan is broken"

    missing = sorted(called - registered)
    assert not missing, (
        "namespace _plugin() names with no matching #[polars_expr] Rust symbol "
        f"(typo or rename): {missing}"
    )
    unreached = sorted(registered - called)
    assert not unreached, (
        "registered #[polars_expr] namespace symbols with no _plugin() caller "
        f"(dead or unconnected namespace path): {unreached}"
    )


def test_lib_module_registration_matches_required_hooks():
    """The introspection FFI registered in ``#[pymodule]`` equals the hooks list.

    ``_REQUIRED_LIB_HOOKS`` is hand-maintained (the parity tests skip on any
    missing hook). This pins it to the actual ``wrap_pyfunction!`` registrations
    in ``lib.rs`` so the two cannot drift — e.g. dropping ``op_output_dtype``
    from the module must also drop it from the hooks list, and vice versa.
    """
    import re

    src = _rust_src_dir()
    if src is None:
        pytest.skip("Rust sources not available (installed wheel)")

    text = (src / "lib.rs").read_text()
    registered = set(re.findall(r"wrap_pyfunction!\(\s*([a-z_0-9]+)\s*,", text))
    assert registered, "no wrap_pyfunction! registrations found — scan is broken"
    assert registered == set(_REQUIRED_LIB_HOOKS), (
        "introspection FFI drift between lib.rs #[pymodule] and _REQUIRED_LIB_HOOKS: "
        f"in lib.rs only={sorted(registered - set(_REQUIRED_LIB_HOOKS))} "
        f"in hooks only={sorted(set(_REQUIRED_LIB_HOOKS) - registered)}"
    )


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

    lib = _lib()
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
    "op_schema",
    "op_infer_shape",
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
    lib = _lib()
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


def test_op_schema_rules_are_required_not_defaulted():
    """The three structural schema rules are REQUIRED trait methods (no default
    body). An op that omits one is a compile error, so a new op cannot silently
    inherit ``PreserveRank``/``PreserveChannels``/``PreserveInput`` and lie about
    its structure — contract by the type system, not convention. This ratchets
    against re-adding a default body to ``view-buffer``'s ``Op`` trait.
    """
    import re

    src = _rust_src_dir()
    if src is None:
        pytest.skip("Rust sources not available (installed wheel)")
    traits = src.parent.parent / "view-buffer" / "src" / "ops" / "traits.rs"
    if not traits.exists():
        pytest.skip("view-buffer sources not available")
    text = traits.read_text()
    for rule, ret in (
        ("output_rank_rule", "OutputRankRule"),
        ("output_channel_rule", "OutputChannelRule"),
        ("output_dtype_rule", "OutputDTypeRule"),
    ):
        required = re.search(rf"fn {rule}\(&self\) -> {ret};", text)
        defaulted = re.search(rf"fn {rule}\(&self\) -> {ret}\s*\{{", text)
        assert required is not None and defaulted is None, (
            f"Op::{rule} must be a required trait method with no default body "
            "so ops cannot inherit a silent, possibly-wrong structural default"
        )


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
    "perceptual_hash": lambda: Pipeline().source("image_bytes").perceptual_hash(),
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

    contract_fn = getattr(_lib(), "op_contract", None)
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

    contract_fn = getattr(_lib(), "op_contract", None)
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
    fn = getattr(_lib(), "enum_variants", None)
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
        "BorderMode",
        "HistogramClosed",
        "LabelReduction",
        "LabelRegionMode",
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
    any explicitly hand-written lazy methods (binary ops, ``apply_mask``,
    ``channel_merge``) keep signatures aligned with their Pipeline counterparts."""
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
    assert stub_path.exists(), (
        "lazy.pyi is missing; run python scripts/gen_lazy_stub.py"
    )
    assert stub_path.read_text() == module.generate_stub(), (
        "lazy.pyi is out of date. Run: python scripts/gen_lazy_stub.py"
    )


def test_source_modifiers_are_not_generated_lazy_forwarders():
    """A Pipeline method that mutates ``_source`` must be Pipeline-only.

    ``_install_pipeline_forwarders`` generates a lazy forwarder for every
    chainable Pipeline op, running it on a *sourceless* continuation
    (``_continuation()`` returns ``Pipeline()`` with ``_source is None``). A
    source-modifier (``source``, ``thumbnail``) would therefore unconditionally
    raise "requires a source" as a lazy method — a latent, always-failing
    forwarder. Such methods must live in ``PIPELINE_ONLY_METHODS`` so no
    forwarder is generated. This pins the fix for ``thumbnail`` and catches any
    future source-mutating builder that forgets the exclusion."""
    import inspect
    import re

    from polars_cv.lazy import PIPELINE_ONLY_METHODS
    from polars_cv.pipeline import Pipeline

    # `._source =` assignment, but not the `._source ==`/`is None` comparisons.
    assigns_source = re.compile(r"\._source\s*=(?!=)")
    offenders = []
    for name in dir(Pipeline):
        if name.startswith("_"):
            continue
        method = getattr(Pipeline, name)
        if not callable(method):
            continue
        try:
            src = inspect.getsource(method)
        except (OSError, TypeError):
            continue
        if assigns_source.search(src) and name not in PIPELINE_ONLY_METHODS:
            offenders.append(name)
    assert not offenders, (
        "source-mutating Pipeline methods must be in PIPELINE_ONLY_METHODS "
        f"(else they generate always-failing lazy forwarders): {offenders}"
    )


def test_explicit_lazy_methods_take_a_lazy_operand():
    """A hand-written lazy method is only justified when its Pipeline
    counterpart takes a ``LazyPipelineExpr`` operand (a binary op, mask/merge).

    Ordinary ops must be generated, not hand-mirrored: the ``label_reduce``
    regression (a redundant explicit copy whose docstring drifted) is exactly
    what this guards against. For every method explicitly defined on
    ``LazyPipelineExpr`` that ALSO exists as a chainable Pipeline op, the
    Pipeline signature must reference ``LazyPipelineExpr`` in a parameter
    annotation (the bespoke-lazy exception); otherwise it should be deleted and
    left to the generator."""
    import inspect

    from polars_cv.lazy import LazyPipelineExpr, _chainable_pipeline_ops
    from polars_cv.pipeline import Pipeline

    chainable = set(_chainable_pipeline_ops())
    offenders = []
    for name, member in vars(LazyPipelineExpr).items():
        if name.startswith("_") or not callable(member):
            continue
        if getattr(member, "__polars_cv_generated__", False):
            continue  # auto-generated forwarder — the desired path
        if name not in chainable:
            continue  # lazy-only method (merge_pipe, statistics_lazy, …) — fine
        p_sig = inspect.signature(getattr(Pipeline, name))
        takes_lazy_operand = any(
            "LazyPipelineExpr" in str(p.annotation) for p in p_sig.parameters.values()
        )
        if not takes_lazy_operand:
            offenders.append(name)
    assert not offenders, (
        "explicit LazyPipelineExpr methods without a LazyPipelineExpr operand "
        f"should be deleted and generated instead: {offenders}"
    )


# ---------------------------------------------------------------------------
# op_schema: the single per-op schema authority (domain, dtype, ndim)
# ---------------------------------------------------------------------------


def _op_json(op: str, **params: object) -> str:
    import json

    spec: dict = {"op": op}
    for k, v in params.items():
        spec[k] = {"type": "literal", "value": v}
    return json.dumps(spec)


@plugin_required
@pytest.mark.parametrize(
    ("op_json", "state_in", "expected"),
    [
        # cast: dtype comes from the op's own target parameter.
        (_op_json("cast", dtype="f32"), ("buffer", "u8", 3), ("buffer", "f32", 3)),
        # histogram modes: quantized keeps a buffer; buckets are struct-encoded
        # (dtype is an encoding concern -> "auto"); counts/normalized/edges are
        # typed vectors.
        (
            _op_json("histogram", bins=8, closed="left", output="quantized"),
            ("buffer", "u8", 3),
            ("buffer", "u32", 3),
        ),
        (
            _op_json("histogram", bins=8, closed="left", output="buckets"),
            ("buffer", "u8", 3),
            ("vector", "auto", 1),
        ),
        (
            _op_json("histogram", bins=8, closed="left", output="counts"),
            ("buffer", "u8", 3),
            ("vector", "u64", 1),
        ),
        (
            _op_json("histogram", bins=8, closed="left", output="normalized"),
            ("buffer", "u8", 3),
            ("vector", "f64", 1),
        ),
        (
            _op_json("histogram", bins=8, closed="left", output="edges"),
            ("buffer", "u8", 3),
            ("vector", "f64", 1),
        ),
        # reductions: axis presence decides scalar-vs-buffer.
        (_op_json("reduce_max", axis=0), ("buffer", "u8", 3), ("buffer", "u8", 2)),
        (_op_json("reduce_max"), ("buffer", "u8", 3), ("scalar", "u8", 0)),
        (_op_json("reduce_sum"), ("buffer", "u8", 3), ("scalar", "f64", 0)),
        # domain transitions.
        (_op_json("extract_shape"), ("buffer", "u8", 3), ("vector", "f64", 1)),
        # extract_contours: contour coordinates are f64 by the geometry contract.
        (_op_json("extract_contours"), ("buffer", "u8", 3), ("contour", "f64", None)),
        (
            _op_json("rasterize", width=8, height=8),
            ("contour", "u8", None),
            ("buffer", "u8", 3),
        ),
        # ordinary buffer op: rank/dtype preserved.
        (_op_json("grayscale"), ("buffer", "u8", 3), ("buffer", "u8", 3)),
    ],
)
def test_op_schema_authority(op_json, state_in, expected) -> None:
    """``op_schema`` resolves the param-dependent schema cases in Rust —
    including everything the Python planner used to special-case."""
    import polars_cv._lib as lib

    assert tuple(lib.op_schema(op_json, *state_in)) == expected


@plugin_required
def test_pipeline_state_matches_batch_fold() -> None:
    """Incrementally tracked builder state equals the fold over op_schema
    from the initial state — the two mechanisms share one authority."""
    corpus = [
        Pipeline().source("blob", dtype="u8").grayscale().threshold(128),
        Pipeline().source("blob", dtype="u8").cast("f32").scale(2.0),
        Pipeline()
        .source("blob", dtype="u8")
        .grayscale()
        .histogram(bins=8, output="counts"),
        Pipeline().source("blob", dtype="u8").reduce_max(axis=0),
        Pipeline().source("blob", dtype="u8").reduce_sum(),
        Pipeline()
        .source("blob", dtype="u8")
        .grayscale()
        .threshold(1)
        .extract_contours(),
        Pipeline().source("blob", dtype="u8").perceptual_hash(),
        # Ops whose builders skipped the incremental schema call entirely
        # (they append + update shape hints only) — regression coverage for
        # the tracked-state == fold invariant on that path.
        Pipeline().source("blob", dtype="u8").reshape([16]).cast("f32"),
        Pipeline().source("blob", dtype="u8").flip([0]).transpose([1, 0, 2]),
        Pipeline().source("blob", dtype="u8").crop(top=0, left=0, height=2, width=2),
        Pipeline().source("blob", dtype="u8").pad(top=1, bottom=1, left=1, right=1),
        Pipeline().source("blob", dtype="u8").pad_to_size(height=8, width=8),
        Pipeline().source("blob", dtype="u8").letterbox(height=8, width=8),
        Pipeline()
        .source("blob", dtype="u8")
        .grayscale()
        .threshold(1)
        .extract_contours()
        .simplify(tolerance=0.5)
        .convex_hull(),
    ]
    for pipe in corpus:
        folded = Pipeline._compute_output_domain_dtype_ndim(
            pipe._ops, initial_domain="buffer", initial_dtype="u8", initial_ndim=None
        )
        tracked = (pipe._current_domain, pipe._output_dtype, pipe._expected_ndim)
        assert tracked == folded, f"state drift for {[o.op for o in pipe._ops]}"


@plugin_required
def test_append_cost_is_linear(monkeypatch) -> None:
    """Appending N ops makes exactly N op_schema calls (no full replay)."""
    import polars_cv._lib as lib

    calls = {"n": 0}
    real = lib.op_schema

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(lib, "op_schema", counting)

    pipe = Pipeline().source("blob", dtype="u8")
    n_ops = 6
    for _ in range(n_ops // 2):
        pipe = pipe.scale(2.0).relu()
    assert len(pipe._ops) == n_ops
    assert calls["n"] == n_ops, (
        f"expected exactly {n_ops} op_schema calls, got {calls['n']} — "
        "per-append tracking must not replay prior ops"
    )


@plugin_required
def test_axis_reduction_ndim_decrements_exactly_once() -> None:
    """Regression: the old full-replay tracking re-subtracted axis
    reductions' ndim on every subsequent append."""
    pipe = Pipeline().source("blob", dtype="u8")
    pipe._expected_ndim = 3  # white-box: seed a known rank
    pipe = pipe.reduce_max(axis=0)
    assert pipe._expected_ndim == 2
    pipe = pipe.reduce_min(axis=0)
    assert pipe._expected_ndim == 1


@plugin_required
def test_reshape_rank_tracked_eagerly() -> None:
    """Regression: reshape's builder skipped the incremental schema call, so
    the eager pipeline kept the stale pre-reshape ndim while the lazy fold
    saw the op — eager and lazy tracking disagreed. Reshape's rank is
    structural (= len(shape)), so both paths now report it exactly."""
    pipe = Pipeline().source("blob", dtype="u8")
    pipe._expected_ndim = 3  # white-box: seed a known rank

    flat = pipe.reshape([16])
    assert flat._expected_ndim == 1

    grid = pipe.reshape([2, 2, 2, 2])
    assert grid._expected_ndim == 4

    # Per-row expression entries do not hide the rank: it is the entry count.
    dyn = pipe.reshape([pl.col("n"), 4])
    assert dyn._expected_ndim == 2


def test_every_op_append_updates_tracked_state() -> None:
    """Ratchet: every Pipeline builder method that appends an OpSpec must run
    the appended op through the op_schema authority (_update_output_dtype).
    A method that skips the call leaves the eager tracked state stale while
    the lazy fold sees the op — the eager/lazy drift class of bug.

    Exception: _add_binary_op is an internal hook whose schema effect is
    resolved by LazyPipelineExpr via binary_output_dtype (a two-input rule
    op_schema cannot express).
    """
    import ast
    from pathlib import Path

    import polars_cv.pipeline as pipeline_mod

    source = Path(pipeline_mod.__file__).read_text()
    tree = ast.parse(source)
    pipeline_cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Pipeline"
    )

    def calls_in(node: ast.AST, attr: str) -> bool:
        return any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == attr
            for sub in ast.walk(node)
        )

    offenders = [
        method.name
        for method in pipeline_cls.body
        if isinstance(method, ast.FunctionDef)
        and method.name != "_add_binary_op"
        and any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "append"
            and isinstance(sub.func.value, ast.Attribute)
            and sub.func.value.attr == "_ops"
            for sub in ast.walk(method)
        )
        and not calls_in(method, "_update_output_dtype")
    ]
    assert not offenders, (
        f"builder methods append an OpSpec without _update_output_dtype: {offenders}"
    )


def test_histogram_schema_declared_once() -> None:
    """Ratchet: histogram's mode->dtype mapping lives in Rust only. The
    Python builder must not re-declare the u32/u64 result dtypes."""
    from pathlib import Path

    import polars_cv.pipeline as pipeline_mod

    source = Path(pipeline_mod.__file__).read_text()
    assert '"u32"' not in source, "histogram quantized dtype re-declared in Python"
    assert '"u64"' not in source, "histogram counts dtype re-declared in Python"


def test_enum_validation_uniform() -> None:
    """Every enum-valued builder parameter fails with the uniform
    ``Invalid <label> '<value>'. Valid: [...]`` error from _validate_enum."""
    cases = [
        (
            lambda: (
                Pipeline()
                .source("blob", dtype="u8")
                .resize(height=8, width=8, filter="bogus")
            ),
            "filter",
        ),
        (
            lambda: Pipeline().source("blob", dtype="u8").normalize(method="bogus"),
            "normalize method",
        ),
        (
            lambda: (
                Pipeline().source("blob", dtype="u8").perceptual_hash(algorithm="bogus")
            ),
            "algorithm",
        ),
        (
            lambda: (
                Pipeline()
                .source("blob", dtype="u8")
                .grayscale()
                .histogram(output="bogus")
            ),
            "histogram output mode",
        ),
        (lambda: Pipeline().source("blob", dtype="u8").cast("bogus"), "dtype"),
    ]
    for build, label in cases:
        with pytest.raises(ValueError, match=rf"Invalid {label} 'bogus'"):
            build()


# ---------------------------------------------------------------------------
# Test-suite conformance: shared fixtures live in conftest.py only
# ---------------------------------------------------------------------------


def _test_files() -> list:
    from pathlib import Path

    tests_dir = Path(__file__).parent
    return [
        p
        for p in tests_dir.rglob("*.py")
        if p.name != "conftest.py" and p.name != Path(__file__).name
    ]


def test_no_local_plugin_available_definitions() -> None:
    """Plugin availability is checked in exactly one place (conftest.py).

    Import ``plugin_required`` from ``tests.conftest`` instead of redefining
    ``_plugin_available`` per file."""
    offenders = [
        str(p.name) for p in _test_files() if "def _plugin_available" in p.read_text()
    ]
    assert not offenders, f"local _plugin_available definitions in: {offenders}"


def test_no_local_png_factories() -> None:
    """PNG construction fixtures live in conftest.py only (create_test_png /
    encode_png); test files must not define their own."""
    offenders = [
        str(p.name) for p in _test_files() if "def create_test_png" in p.read_text()
    ]
    assert not offenders, f"local create_test_png definitions in: {offenders}"
