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

import ast
import io
import re
from pathlib import Path

import polars as pl
import pytest

import polars_cv
from polars_cv import Pipeline
from tests._discovery import (
    package_modules,
    requires_checkout,
    rust_sources,
    rust_src_dir,
    suite_files,
    suite_modules,
)
from tests._dtype_ratchet import dispatch_offenders
from tests._op_cases import build_case, comparable_ops
from tests._schema_parity import assert_plan_equals_exec, leaf_dtype
from tests.conftest import plugin_required

#: Every test here is a structural guard: it checks the *shape* of the
#: codebase rather than the behaviour of a pipeline, so it needs no compiled
#: extension and runs in milliseconds. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural

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
    """Return (planned_dtype, realized_dtype) for an expression over ``df``.

    Backed by the shared harness, so every call site here also gets the
    streaming engine and the engine-agreement check, not just in-memory.
    """
    planned = df.lazy().with_columns(**{col: expr}).collect_schema()[col]
    realized = assert_plan_equals_exec(df, expr, name=col).dtype
    return planned, realized


#: Peeling nested List/Array wrappers is one operation; it lives in the harness.
_leaf_dtype = leaf_dtype


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
    # These used to `pytest.skip` on the symbols being "not implemented yet
    # (Phase 3)". All three have existed for releases, so the skips were dead
    # guards: had the FFI regressed, this parity check would have gone quiet
    # instead of failing. Assert them instead.
    rust_ops = _known_ops_from_rust()
    assert rust_ops is not None, "_lib.known_ops() is missing from the compiled plugin"
    pipeline_ops = getattr(Pipeline, "OP_NAMES", None)
    assert pipeline_ops is not None, "Pipeline.OP_NAMES is missing"
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
    assert rust_ops is not None, "_lib.known_ops() is missing from the compiled plugin"
    pipeline_ops = set(Pipeline.OP_NAMES)
    unreachable = rust_ops - pipeline_ops
    assert not unreachable, (
        "Rust ops in KNOWN_OPS with no Python builder that emits them "
        f"(dead or unconnected graph path): {sorted(unreachable)}"
    )


@requires_checkout
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

    src = rust_src_dir()

    called: set[str] = set()
    # Two spellings reach the same plugin: the direct `self._plugin("name", ...)`
    # and `_ArgBinder.call(self, "name", ...)`, which routes through `_plugin`
    # after partitioning literal kwargs from per-row expression inputs.
    # `\s*` spans the newline for multi-line calls.
    patterns = (
        r'_plugin\(\s*"([a-z_0-9]+)"',
        r'\.call\(\s*self,\s*"([a-z_0-9]+)"',
    )
    for py in package_modules():
        text = py.read_text()
        for pattern in patterns:
            called |= set(re.findall(pattern, text))
    assert called, "no plugin calls found — scan is broken"

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


@requires_checkout
def test_lib_module_registration_matches_required_hooks():
    """The introspection FFI registered in ``#[pymodule]`` equals the hooks list.

    ``_REQUIRED_LIB_HOOKS`` is hand-maintained (the parity tests skip on any
    missing hook). This pins it to the actual ``wrap_pyfunction!`` registrations
    in ``lib.rs`` so the two cannot drift — e.g. dropping ``op_output_dtype``
    from the module must also drop it from the hooks list, and vice versa.
    """
    import re

    src = rust_src_dir()

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
    names |= set(re.findall(r'_append_op\(\s*"([a-z_0-9]+)"', text))
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


@requires_checkout
def test_op_names_matches_rust_known_ops_without_the_plugin() -> None:
    """``Pipeline.OP_NAMES`` must equal Rust's ``KNOWN_OPS``, checked from source.

    The two ``test_registry_parity_*`` tests already pin this equality in both
    directions, but both are ``@plugin_required`` and skip when the extension
    is not built. That is not a hypothetical lane: the editable install leaves
    the compiled ``.so`` at its last ``maturin develop`` while Python sources
    track the working tree, so a contributor adding a builder op and running
    the suite before rebuilding gets two skips where they expect two failures.

    Reading ``KNOWN_OPS`` out of the Rust source needs no plugin, so the drift
    is caught in that window too. Source-scanning is the weaker technique and
    is used here only because the stronger one is unavailable by construction;
    it asserts it parsed a plausible registry rather than matching nothing.
    """
    src = rust_src_dir()

    text = (src / "execute.rs").read_text()
    m = re.search(r"pub const KNOWN_OPS: &\[&str\] = &\[(.*?)\n\];", text, re.S)
    assert m, "could not find KNOWN_OPS in execute.rs — scan is out of date"
    # Strip comments first: this codebase explains absences inline (`// "sobel"
    # is deliberately absent`), and a quoted name in one would read as an op.
    body = re.sub(r"(?m)//.*$", "", m.group(1))
    rust_ops = set(re.findall(r'"([a-z0-9_]+)"', body))
    assert len(rust_ops) > 50, f"KNOWN_OPS scan found only {len(rust_ops)} ops"

    declared = set(Pipeline.OP_NAMES)
    assert declared == rust_ops, (
        "Pipeline.OP_NAMES has drifted from Rust KNOWN_OPS: "
        f"python-only={sorted(declared - rust_ops)}, "
        f"rust-only={sorted(rust_ops - declared)}"
    )


@plugin_required
def test_registry_parity_no_dead_contracts():
    """Ops the Pipeline never emits are not executable (B2: sobel/laplacian/sharpen)."""
    import json

    lib = _lib()
    contract_fn = getattr(lib, "op_contract", None) if lib is not None else None
    assert callable(contract_fn), "_lib.op_contract() is missing from the plugin"
    # sobel/laplacian/sharpen lower to convolve2d; they are not real executable
    # ops, so resolving them must fail (their standalone contracts are dead, B2).
    for lowered in ("sobel", "laplacian", "sharpen"):
        with pytest.raises(Exception):
            contract_fn(json.dumps({"op": lowered}))


_REQUIRED_LIB_HOOKS = (
    "op_contract",
    "op_schema",
    "op_infer_shape",
    "op_output_channels",
    "binary_output_dtype",
    "known_ops",
    "enum_variants",
    "enum_names",
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


@requires_checkout
def test_op_schema_rules_are_required_not_defaulted():
    """The three structural schema rules are REQUIRED trait methods (no default
    body). An op that omits one is a compile error, so a new op cannot silently
    inherit ``PreserveRank``/``PreserveChannels``/``PreserveInput`` and lie about
    its structure — contract by the type system, not convention. This ratchets
    against re-adding a default body to ``view-buffer``'s ``Op`` trait.
    """
    import re

    src = rust_src_dir()
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

# Every op with a callable case, driven from `tests/_op_cases.py` — the table
# `test_op_case_table_is_complete` pins to `_chainable_pipeline_ops()` in both
# directions. This replaced a local op -> builder map that named 22 of the ~90
# ops, so the other ~70 never had their domain or rank/channel rule checked
# against the Rust contract at all: the failure mode of every hand-maintained
# list in this repo, sitting inside the file that polices them.


@plugin_required
@pytest.mark.parametrize("op_name", comparable_ops())
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

    pipe = build_case(op_name)
    rust_domain = contract_fn(json.dumps(pipe._ops[-1].to_dict()))["output_domain"]
    planned_domain, _, _ = Pipeline._compute_output_domain_dtype_ndim(
        pipe._ops, initial_domain="buffer", initial_dtype="u8"
    )
    assert planned_domain == rust_domain, (
        f"{op_name}: planner domain {planned_domain!r} != Rust authority "
        f"{rust_domain!r}"
    )


@plugin_required
@pytest.mark.parametrize("op_name", comparable_ops())
def test_contract_exposes_rank_and_channel_rules(op_name):
    """Every op's contract exposes a rank_rule and channel_rule in the known
    vocabulary — the single authority the Python planner reads instead of
    re-declaring its own ndim/alpha rules."""
    import json

    contract_fn = getattr(_lib(), "op_contract", None)
    if not callable(contract_fn):
        pytest.skip("_lib.op_contract() not built")

    contract = contract_fn(json.dumps(build_case(op_name)._ops[-1].to_dict()))
    rank, channel = contract["rank_rule"], contract["channel_rule"]

    assert rank in ("preserve", "reduce_one", "unknown") or (
        rank.startswith("fixed:") and rank.split(":", 1)[1].isdigit()
    ), f"{op_name}: unexpected rank_rule {rank!r}"
    assert channel in ("preserve", "n/a", "unknown") or (
        channel.startswith(("fixed:", "strip_restore:"))
        and channel.split(":", 1)[1].isdigit()
    ), f"{op_name}: unexpected channel_rule {channel!r}"


#: Exactly the keys ``op_contract`` publishes. Pinned as a set, in both
#: directions, so the boundary cannot grow a second spelling of a fact it
#: already carries.
_CONTRACT_KEYS = frozenset(
    {
        "dtype_rule",
        "rank_rule",
        "channel_rule",
        "input_domains",
        "output_domain",
    }
)


@plugin_required
def test_contract_publishes_no_second_spelling():
    """``op_contract`` publishes each fact once.

    It used to carry both ``input_domain`` (a single ``Domain``) and
    ``input_domains`` (the accepted set). Only the set was read, and the two
    were free to disagree the moment a step accepted more than one domain —
    which binary ops and reductions do. An unread key on an FFI boundary is not
    inert: it is the next author's authority.
    """
    import json

    contract_fn = getattr(_lib(), "op_contract", None)
    if not callable(contract_fn):
        pytest.skip("_lib.op_contract() not built")

    spec = Pipeline().source("image_bytes").grayscale()._ops[-1]
    keys = set(contract_fn(json.dumps(spec.to_dict())))
    assert keys == _CONTRACT_KEYS, (
        f"op_contract's key set changed: added {sorted(keys - _CONTRACT_KEYS)}, "
        f"removed {sorted(_CONTRACT_KEYS - keys)}. Every key here is read by "
        f"the Python planner; add one only with the reader that needs it."
    )


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


# Enums whose Python mirror is a plain `_types` enum of the same name, checked
# uniformly below. `test_every_rust_enum_is_parity_checked` asserts this list
# plus the bespoke cases account for every enum Rust surfaces, so adding one in
# Rust fails here until it is either mirrored or explicitly excused.
_UNIFORM_PARITY_ENUMS = [
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
    "FilterType",
    "ExtractMode",
    "ApproxMethod",
    "InterpolationType",
    "ScaleOrigin",
    "Winding",
    # Owned by the plugin crate rather than the engine (PLUGIN_REGISTRY), which
    # `enum_variants` chains onto the engine's. Nothing about checking them
    # differs — that is the point of chaining rather than special-casing.
    "RowErrorPolicy",
    "NullParamPolicy",
    "FetchErrorPolicy",
]

# Checked, but not by the uniform test: their Python side needs special
# handling (a subtracted internal variant, an extra sub-assertion).
_BESPOKE_PARITY_ENUMS = {"DType", "Domain"}

# Surfaced by `enum_variants` with no Python enum to compare against.
_NO_PYTHON_MIRROR = {
    # Binary ops are Python *methods* (`.add()`, `.blend()`), not an enum, so
    # there is no member set to diff. `test_binary_ops_match_rust` pins the
    # names against the Rust table instead.
    #
    # This is now the *only* reason it is here. It used to be exempt for a
    # second reason as well — its name table lived in the plugin crate, so
    # `enum_variants` answered for it through a hand-written arm rather than a
    # registry. The table has moved beside the enum in view-buffer, so it is
    # registered and name-checked like everything else.
    "BinaryOp",
}


@plugin_required
@pytest.mark.parametrize("enum_name", _UNIFORM_PARITY_ENUMS)
def test_enum_parity_api_enums(enum_name):
    """Each user-facing API enum must equal its view-buffer authority set (A4)."""
    rust = _rust_enum_variants(enum_name)
    if rust is None:
        pytest.skip("_lib.enum_variants() not built")
    import polars_cv._types as t

    py = {m.value for m in getattr(t, enum_name)}
    assert py == rust, f"{enum_name}: python {py} != rust {rust}"


@plugin_required
def test_filter_type_exposes_every_rust_variant():
    """`FilterType` is full parity, not a subset.

    It was previously a deliberate subset (nearest/bilinear/lanczos3). Making
    `filter` a per-row parameter broke that: the literal path validated against
    the Python enum while a column value went straight to Rust's larger table,
    so an expression could reach a filter a literal could not. Rather than
    validate the same restriction twice, the subset was dropped — checked here
    alongside the other API enums via `test_enum_parity_api_enums`, with this
    test pinning the specific variants that used to be Rust-only.
    """
    rust = _rust_enum_variants("FilterType")
    if rust is None:
        pytest.skip("_lib.enum_variants() not built")
    import polars_cv._types as t

    py = {m.value for m in t.FilterType}
    assert {"catmullrom", "gaussian"} <= py, (
        "catmullrom/gaussian must stay reachable from Python; a subset here "
        "would be bypassable through a per-row `filter` expression"
    )
    assert py == rust, f"FilterType: python {py} != rust {rust}"


@plugin_required
def test_every_rust_enum_is_parity_checked():
    """Every enum ``enum_variants`` answers for must be checked by some test.

    The list of enums to check used to be hand-written, and had drifted:
    ``LabelReduction`` and ``LabelRegionMode`` both had authoritative Rust
    tables and neither appeared in any parity test, so a Python/Rust
    divergence in either would have shipped. Reading the enum names from Rust
    closes that: a newly registered enum lands in ``enum_names()`` and fails
    here until it is mirrored in ``_types`` or explicitly excused above.
    """
    fn = getattr(_lib(), "enum_names", None)
    if not callable(fn):
        pytest.skip("_lib.enum_names() not built")

    surfaced = set(fn())
    accounted = set(_UNIFORM_PARITY_ENUMS) | _BESPOKE_PARITY_ENUMS | _NO_PYTHON_MIRROR
    unchecked = surfaced - accounted
    assert not unchecked, (
        f"these Rust enums are surfaced to Python but no parity test covers "
        f"them: {sorted(unchecked)}. Add each to _UNIFORM_PARITY_ENUMS (with a "
        f"matching polars_cv._types enum), or to _NO_PYTHON_MIRROR with a "
        f"reason."
    )

    # The reverse direction: an excused or bespoke name that Rust no longer
    # surfaces is a stale entry that would quietly stop checking anything.
    stale = accounted - surfaced
    assert not stale, (
        f"these names are listed as parity-checked but Rust does not surface "
        f"them: {sorted(stale)}"
    )


@plugin_required
def test_binary_ops_match_rust():
    """``BinaryOp`` has no Python enum, so pin the method names instead."""
    rust = _rust_enum_variants("BinaryOp")
    if rust is None:
        pytest.skip("_lib.enum_variants() not built")
    # `hasattr(LazyPipelineExpr, name)` would be a weak proxy: the class
    # generates methods from Pipeline, so a Rust op whose name collided with an
    # unrelated generated method would pass. Compare against the names the lazy
    # API actually emits as binary ops.
    emitted = _binary_op_names_from_source()
    assert emitted, "no binary ops scanned from lazy.py — scan regex out of date?"
    assert emitted == rust, (
        f"BinaryOp drift between Rust and lazy.py: "
        f"rust-only={sorted(rust - emitted)}, python-only={sorted(emitted - rust)}"
    )


# SourceFormat/SinkFormat have no Rust *enum* to be checked against: the graph
# boundary carries them as plain strings, and view-buffer's shadowing copies
# were deleted along with its unreachable pipeline-composition layer. That is
# not the same as having nothing to pin them to, which an earlier note here
# claimed. Source formats do have a Rust vocabulary — `KNOWN_SOURCE_FORMATS` in
# graph/compiled.rs, which the graph validator rejects unknown formats against
# — so the two lists must be equal, and the test below pins them.
#
# Sink formats genuinely have no list: `SinkKind::resolve` (graph/sink_kind.rs)
# is the one place a (domain, format) pair is interpreted, and it errors on the
# fall-through, so an unhandled sink is rejected rather than enumerated. The
# four halves of the sink contract match on the resolved *kind*, so they cannot
# disagree about which pairs exist.
#
# This note used to say the pair was matched in two places that "error on the
# fall-through, so there is no second declaration to drift from". Both halves
# of that were false: there were four such matches, and two of them ended in
# `_ => Binary` rather than an error.


@requires_checkout
def test_source_formats_match_the_rust_vocabulary() -> None:
    """``SourceFormat`` must equal Rust's ``KNOWN_SOURCE_FORMATS``.

    Both are hand-written lists of the same vocabulary, one per side of the
    FFI. A Python-only format builds a graph the validator rejects at
    execution, with an error naming a format the user did pass; a Rust-only
    one is a decode path nothing can reach. Neither shows up until someone
    runs the query.

    Read from the Rust source rather than over the FFI so this runs in the
    plugin-free lane too — the drift is introduced by editing Python, which is
    exactly when the extension is stale.
    """
    src = rust_src_dir()

    text = (src / "graph" / "compiled.rs").read_text()
    m = re.search(r"const KNOWN_SOURCE_FORMATS: &\[&str\] = &\[(.*?)\n\];", text, re.S)
    assert m, (
        "could not find KNOWN_SOURCE_FORMATS in graph/compiled.rs — the scan is "
        "out of date, and a scan that matches nothing passes vacuously"
    )
    body = re.sub(r"(?m)//.*$", "", m.group(1))
    rust_formats = set(re.findall(r'"([a-z0-9_]+)"', body))
    assert len(rust_formats) > 3, (
        f"KNOWN_SOURCE_FORMATS scan found only {sorted(rust_formats)}"
    )

    from polars_cv._types import SourceFormat

    declared = {m.value for m in SourceFormat}
    assert declared == rust_formats, (
        "SourceFormat has drifted from Rust KNOWN_SOURCE_FORMATS: "
        f"python-only={sorted(declared - rust_formats)}, "
        f"rust-only={sorted(rust_formats - declared)}"
    )


# ---------------------------------------------------------------------------
# 4. No duplicate enums (A4 / "no repeated enums")
# ---------------------------------------------------------------------------


def test_no_duplicate_expected_dtype_enum():
    """ExpectedDType was an exact duplicate of DType and must not exist (A4)."""
    import polars_cv._types as t

    assert not hasattr(t, "ExpectedDType"), "ExpectedDType should be folded into DType"


def test_dtype_is_the_only_python_dtype_name_table():
    """No Python enum may re-list dtype spellings alongside DType (A4).

    `OutputDType` used to: `f32`/`f64`/`u8` plus a `preserve` value that was a
    synonym for the default rather than a dtype. It is gone (see
    `test_removed_surfaces.py`), and `out_dtype` validates against `DType` —
    which is itself checked against Rust's `dtype_table!` authority by the
    parity tests. Guard that a second partial table does not reappear.
    """
    import enum

    import polars_cv._types as t

    dtype_values = {m.value for m in t.DType}
    for name in dir(t):
        member = getattr(t, name)
        if name == "DType" or not isinstance(member, type):
            continue
        if not issubclass(member, enum.Enum):
            continue
        overlap = {m.value for m in member} & dtype_values
        assert not overlap, (
            f"{name} re-lists dtype names {sorted(overlap)}; DType is the single "
            f"Python dtype-name table (Rust's authority is dtype_table!)"
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


def test_op_append_ratchet_moved_to_the_append_contract_suite() -> None:
    """The per-call ratchet has been replaced by a structural guard.

    This test used to assert that every builder appending an OpSpec also
    called ``_update_output_dtype`` — one of the *two* updates an append
    requires. Its own docstring named the failure mode it was meant to stop
    ("the eager/lazy drift class of bug"), and the ops that skipped the other
    update (``_update_shape_hints``) shipped a plan/exec divergence underneath
    it: enumerating required calls only guards the calls you enumerated.

    ``tests/test_append_contract.py`` now forbids anything but
    ``Pipeline._push_op`` from mutating ``_ops`` at all, which makes the whole
    sequence unskippable rather than checked. Kept as a pointer so the weaker
    form is not reintroduced.
    """
    from tests import test_append_contract

    assert hasattr(test_append_contract, "test_op_append_is_structurally_exclusive")


def test_histogram_schema_declared_once() -> None:
    """Ratchet: histogram's mode->dtype mapping lives in Rust only. The
    Python builder must not re-declare the u32/u64 result dtypes."""
    from pathlib import Path

    import polars_cv.pipeline as pipeline_mod

    source = Path(pipeline_mod.__file__).read_text()
    assert '"u32"' not in source, "histogram quantized dtype re-declared in Python"
    assert '"u64"' not in source, "histogram counts dtype re-declared in Python"


#: ``(build, label, rust enum, a real variant)`` per enum-valued builder
#: parameter. ``build`` takes the value so the same call can be exercised with
#: a bogus one and a good one.
_ENUM_VALIDATION_CASES = [
    (
        lambda v: (
            Pipeline().source("blob", dtype="u8").resize(height=8, width=8, filter=v)
        ),
        "filter",
        "FilterType",
        "nearest",
    ),
    (
        lambda v: Pipeline().source("blob", dtype="u8").normalize(method=v),
        "normalize method",
        "NormalizeMethod",
        "minmax",
    ),
    (
        lambda v: Pipeline().source("blob", dtype="u8").perceptual_hash(algorithm=v),
        "algorithm",
        "HashAlgorithm",
        "average",
    ),
    (
        lambda v: Pipeline().source("blob", dtype="u8").grayscale().histogram(output=v),
        "histogram output mode",
        "HistogramOutput",
        "counts",
    ),
    (
        lambda v: Pipeline().source("blob", dtype="u8").cast(v),
        "dtype",
        "DType",
        "f32",
    ),
]


@pytest.mark.parametrize(
    ("build", "label", "enum_name", "good"),
    _ENUM_VALIDATION_CASES,
    ids=[c[1] for c in _ENUM_VALIDATION_CASES],
)
def test_enum_validation_uniform(build, label: str, enum_name: str, good: str) -> None:
    """Each enum parameter rejects a bad value *and* accepts a real one.

    The rejection half is the uniform ``Invalid <label> '<value>'. Valid: [...]``
    error from ``_validate_enum``.

    The acceptance half is what stops the rejection half being vacuous. On its
    own, "'bogus' raises" passes just as well when the builder rejects
    *everything* — an accidentally empty valid-set, a parameter renamed so the
    keyword lands in ``**kwargs`` and dies for an unrelated reason, or a domain
    check firing first. Each case therefore also builds with a real variant,
    and asserts that variant is one the Rust enum actually publishes rather
    than a name hard-coded here that both sides might have dropped.
    """
    with pytest.raises(ValueError, match=rf"Invalid {label} '__bogus__'"):
        build("__bogus__")

    build(good)  # must not raise

    rust = _rust_enum_variants(enum_name)
    if rust is None:
        pytest.skip("_lib.enum_variants() not built")
    assert good in rust, (
        f"'{good}' is used here as a known-good {enum_name}, but Rust publishes "
        f"{sorted(rust)}. The positive half of this test is checking a value "
        f"the engine no longer has."
    )


#: Geometry-accessor enum parameters that are deliberately literal-only, with
#: the reason. A parameter belongs here **only** if its value fixes the output
#: shape, rank or dtype — the eligibility rule in root ``CLAUDE.md``. Empty
#: today: none of the geometry enums is structural.
_LITERAL_ONLY_GEOM_ENUMS: dict[str, str] = {}


def test_non_structural_geometry_enums_accept_an_expression() -> None:
    """A geometry enum that changes no output schema must be per-row capable.

    The rule for per-row eligibility is not the parameter's *type*: it is
    whether the value affects output shape, rank or dtype. None of the contour
    namespace's enums does — a winding, a scale origin, a reduction and a
    region mode all leave `List(Struct(CONTOUR_SCHEMA))` (or the reduction's
    `List(Float64)`) exactly as it was.

    Two of the four were literal-only anyway, and the rejection they raised
    claimed the opposite: "'direction' is structural (it fixes the output
    shape/rank/dtype at planning time)". It fixes none of them. Reading the
    live signatures makes the rule enforced rather than remembered — a new
    non-structural enum has to be plumbed through ``_ArgBinder`` or listed
    above with a reason.
    """
    import inspect

    from polars_cv._types import (
        LabelReduction,
        LabelRegionMode,
        ScaleOrigin,
        Winding,
    )
    from polars_cv.geometry.bbox import BBoxNamespace
    from polars_cv.geometry.contours import ContourNamespace
    from polars_cv.geometry.points import PointNamespace

    enum_names = {
        cls.__name__ for cls in (Winding, ScaleOrigin, LabelReduction, LabelRegionMode)
    }
    found: dict[str, str] = {}
    for namespace in (ContourNamespace, PointNamespace, BBoxNamespace):
        for name, method in inspect.getmembers(namespace, inspect.isfunction):
            if name.startswith("_"):
                continue
            for param_name, param in inspect.signature(method).parameters.items():
                annotation = str(param.annotation)
                if any(enum in annotation for enum in enum_names):
                    found[f"{namespace.__name__}.{name}.{param_name}"] = annotation

    # Non-vacuity: an import rename or a signature-scan bug must fail here
    # rather than silently checking an empty set.
    assert len(found) >= 4, (
        f"found {len(found)} enum-annotated geometry parameters — the signature "
        f"scan is broken, not the annotations: {found}"
    )
    literal_only = {
        key: annotation
        for key, annotation in found.items()
        if "Expr" not in annotation and key not in _LITERAL_ONLY_GEOM_ENUMS
    }
    assert not literal_only, (
        f"these geometry enum parameters reject a Polars expression but change "
        f"no output shape, rank or dtype, so they are eligible to be per-row: "
        f"{literal_only}. Route them through `_ArgBinder`, or add them to "
        f"_LITERAL_ONLY_GEOM_ENUMS with the structural reason."
    )


def test_the_literal_only_geometry_exemptions_carry_a_reason() -> None:
    """An exemption is a decision about the eligibility rule, so it says why."""
    blank = [k for k, why in _LITERAL_ONLY_GEOM_ENUMS.items() if not why.strip()]
    assert not blank, f"exemptions without a reason: {blank}"


# ---------------------------------------------------------------------------
# Test-suite conformance: shared fixtures live in conftest.py only
# ---------------------------------------------------------------------------


def test_no_local_plugin_available_definitions() -> None:
    """Plugin availability is checked in exactly one place (conftest.py).

    Import ``plugin_required`` from ``tests.conftest`` instead of redefining
    ``_plugin_available`` per file."""
    offenders = [
        str(p.name) for p in suite_modules() if "def _plugin_available" in p.read_text()
    ]
    assert not offenders, f"local _plugin_available definitions in: {offenders}"


def test_no_local_png_factories() -> None:
    """PNG construction fixtures live in conftest.py only (create_test_png /
    encode_png); test files must not define their own."""
    offenders = [
        str(p.name) for p in suite_modules() if "def create_test_png" in p.read_text()
    ]
    assert not offenders, f"local create_test_png definitions in: {offenders}"


# ---------------------------------------------------------------------------
# dtype spellings: one authority (dtype_table! in view-buffer/src/core/dtype.rs)
# ---------------------------------------------------------------------------


def _load_script(name: str):
    """Import a ``scripts/`` module by path.

    The generators are not an importable package, and putting ``scripts/`` on
    ``sys.path`` at module scope would reorder every import in this file. Same
    approach ``test_lazy_stub_is_current`` already uses.
    """
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@requires_checkout
def test_dtype_names_module_is_current() -> None:
    """``_dtype_names.py`` must match what ``gen_dtype_names.py`` produces.

    Python cannot read `dtype_table!` at runtime — `numpy_name` does not cross
    the FFI, and `numpy_from_struct` has to work with no compiled extension —
    so the spellings are generated and checked in. Regenerate-and-diff is what
    keeps that copy honest; without it the file is just a fourth hand-written
    dtype table, which is what it replaced.
    """
    gen = _load_script("gen_dtype_names")
    generated = gen.generate()
    committed = Path(gen._OUT_PATH)
    assert committed.exists(), (
        "_dtype_names.py is missing; run python scripts/gen_dtype_names.py"
    )
    assert committed.read_text() == generated, (
        "_dtype_names.py is out of date. Run: python scripts/gen_dtype_names.py"
    )


def test_the_numpy_allowlist_holds_no_character_codes() -> None:
    """The names must be numpy's *spelled* dtypes, not its character codes.

    This is the specific defect the generated module replaced: the hand-written
    allowlist admitted ``"u8"``/``"i8"``/``"f2"``, where numpy reads ``"u8"`` as
    ``uint64`` — the opposite of this project's ``u8``. A caller hand-building a
    struct got a silent reinterpretation of the bytes. Pinned as its own
    assertion because a regenerated file would happily carry them back if the
    generator ever read the wrong column.
    """
    from polars_cv._dtype_names import NUMPY_NAMES
    from polars_cv._types import DType

    collisions = NUMPY_NAMES & {m.value for m in DType}
    assert not collisions, (
        f"these numpy names collide with engine dtype names: {sorted(collisions)}. "
        f"numpy reads 'u8' as uint64 and this project reads it as uint8."
    )
    assert all(not name[-1].isdigit() or len(name) > 3 for name in NUMPY_NAMES), (
        f"character-code-shaped entries in NUMPY_NAMES: {sorted(NUMPY_NAMES)}"
    )


def _dtype_table_rows() -> list[tuple[str, str, int, str]]:
    """``dtype_table!``'s rows, parsed by the generator that also reads them.

    The parse lives in ``scripts/gen_dtype_names.py`` because that script has to
    read the table anyway, and two parsers of one table is the shape these
    guards exist to catch. Imported rather than re-implemented.
    """
    return _load_script("gen_dtype_names").dtype_table_rows()


def test_engine_dtype_names_match_the_generated_table() -> None:
    """``_types.DType`` must spell exactly what ``dtype_table!`` spells.

    ``DType`` is in view-buffer's ``naming::REGISTRY``, so
    ``test_every_rust_enum_is_parity_checked`` already compares it to the Rust
    variants — but only through ``enum_variants``, which needs the compiled
    extension. Editing Python is exactly when the extension is stale, so the
    check that matters most runs in the plugin-free lane, against the generated
    module. This is also what gives ``SHORT_NAMES`` a reader: a generated
    constant nothing reads is a fourth dtype table with extra steps.
    """
    from polars_cv._dtype_names import SHORT_NAMES, WIRE_CODES
    from polars_cv._types import DType

    assert SHORT_NAMES == {member.value for member in DType}, (
        "polars_cv._types.DType has drifted from dtype_table!: "
        f"only in DType={sorted({m.value for m in DType} - SHORT_NAMES)}, "
        f"only in dtype_table!={sorted(SHORT_NAMES - {m.value for m in DType})}"
    )
    # And the wire codes cover the same names, which is what makes WIRE_CODES
    # usable as the authority `display.py` is pinned to below.
    assert set(WIRE_CODES) == SHORT_NAMES


def test_display_wire_codes_match_the_rust_dtype_table() -> None:
    """``display.py`` renders VIEW blobs without going through the plugin, so
    it keeps its own wire-code map. Pin it to the Rust table it copies.

    The numpy types are derived from the generated ``NUMPY_NAMES``. The
    hand-written ``{"uint8": np.uint8, ...}`` that used to sit here was a fifth
    copy of ``dtype_table!``'s numpy column, inside the file that polices
    duplicate spellings.
    """
    import numpy as np

    import polars_cv.display as display_mod
    from polars_cv._dtype_names import NUMPY_NAMES

    numpy_by_name = {name: getattr(np, name) for name in NUMPY_NAMES}
    expected = {
        code: numpy_by_name[numpy_name]
        for _variant, _short, code, numpy_name in _dtype_table_rows()
    }

    source = Path(display_mod.__file__).read_text()
    body = source.split("dtype_map = {", 1)[1].split("}", 1)[0]
    actual = {
        int(code): getattr(np, name)
        for code, name in re.findall(r"(\d+):\s*np\.(\w+)", body)
    }

    assert actual == expected, (
        "display.py's VIEW wire-code map has drifted from dtype_table! in "
        f"view-buffer/src/core/dtype.rs: {actual} != {expected}"
    )


def test_display_rejects_unknown_wire_codes() -> None:
    """An unrecognised dtype code must raise, not render as uint8. Guessing
    reinterprets the payload and produces a plausible but meaningless image."""
    import polars_cv.display as display_mod

    blob = bytearray(b"VIEW" + bytes(60))
    blob[6] = 200  # not a dtype code
    blob[7] = 2
    with pytest.raises(ValueError, match="unknown VIEW dtype code"):
        display_mod._view_to_png(bytes(blob))


# Functions that map to a *subset* of DType variants on purpose. The ratchet
# fails on any unlisted partial rather than skipping it: every blind spot it has
# had was something it chose to skip, so an unrecognised shape must be an
# explicit, reviewable entry here rather than silence.
ALLOWED_PARTIAL_VARIANT_MAPS = frozenset(
    {
        # Arrow FFI import supports a subset by design and rejects the rest
        # with "Unsupported Arrow type" rather than guessing, so the missing
        # arms are an honest refusal, not silent drift.
        "view-buffer/src/interop/arrow_ffi.rs::from_arrow_ffi",
    }
)


def test_no_second_dtype_spelling_table() -> None:
    """Ratchet: dtype names are declared in ``dtype_table!`` only.

    Some dispatch genuinely has to name every dtype: matching a string to a
    Polars ``DataType``, or to a macro arm that needs the variant as a literal
    token. Those are correspondences, not naming decisions, and cannot be
    derived away. What must not happen is one of them disagreeing with the
    authority — a missing arm or a stray ``"f16"`` is how a dtype ends up
    handled in nine places and forgotten in the tenth.

    So this does not forbid the tables; it requires any dtype dispatch to name
    *exactly* the ten that ``dtype_table!`` declares. New code that only needs
    the name should still use ``DType::short_name`` / ``numpy_name`` /
    ``wire_code`` rather than adding an eleventh site.

    Two earlier versions of this test were unsound, so the technique matters:

    - Checking per *file* let ``encode.rs``'s two dispatches cover for each
      other, so deleting an arm from one passed.
    - Checking per brace-matched ``match`` block fixed that but was
      non-monotonic: with a "six or more names must be all ten" threshold,
      dropping four arms failed while dropping five *passed*, because the
      block fell under the threshold. Damage concealed itself by growing. The
      brace matcher was also defeatable by an unbalanced ``{`` inside a string
      literal (``polars_bail!("expected one of {")``) and by a brace in the
      scrutinee (``match Key { id: 1 } {``), either of which merged sibling
      blocks and restored the per-file hole.

    This version does no brace matching and has no threshold. It groups arms
    by enclosing function and requires each function's dtype-arm set to be
    exactly the ten. Multiplicity is checked too: a function holding two
    dispatches names every dtype twice, so dropping one arm makes the counts
    uneven even though the *set* is still complete.

    Two limits, stated because a guard that overstates its reach is how this
    test went wrong twice already:

    - It checks that the arm *keys* are all present, never that a key maps to
      the right thing. ``"u32" | "u64" => build_typed_list_u32(..)`` folds two
      dtypes onto one builder and passes: the set is complete. Only a test that
      executes the dispatch can catch that.
    - It reads ``match`` arms. A dtype table written as an if/else chain or a
      ``HashMap`` literal is invisible. Arm keys and ``DType::`` variant keys
      can be found without parsing Rust, and the two parsing attempts above are
      what produced two unsound versions.
    """
    root = Path(__file__).resolve().parents[2]
    allowed = {
        # The authority itself.
        root / "view-buffer" / "src" / "core" / "dtype.rs",
        # Pins the frozen VIEW codes literally, on purpose.
        root / "view-buffer" / "tests" / "dtype_single_authority.rs",
    }
    expected_pairs = {
        (variant, short) for variant, short, _code, _numpy in _dtype_table_rows()
    }

    offenders = []
    for path in rust_sources():
        if path in allowed:
            continue
        offenders += dispatch_offenders(
            path.read_text(),
            str(path.relative_to(root)),
            expected_pairs,
            ALLOWED_PARTIAL_VARIANT_MAPS,
        )

    assert not offenders, (
        "these dtype dispatches do not agree with dtype_table! in "
        f"view-buffer/src/core/dtype.rs: {offenders}"
    )


def test_contour_dtype_map_matches_rust() -> None:
    """The metrics contour matcher builds pipeline sources without going
    through the plugin, so it keeps its own Polars-type -> dtype-name map.
    The correspondence is its own, but the names must be the Rust ones."""
    from polars_cv.metrics._matching._contour import _POLARS_TO_CV_DTYPE

    expected = {short for _variant, short, _code, _numpy in _dtype_table_rows()}
    actual = set(_POLARS_TO_CV_DTYPE.values())
    assert actual == expected, (
        "_POLARS_TO_CV_DTYPE has drifted from dtype_table!: "
        f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
    )


def test_contour_source_accepts_boolean_masks() -> None:
    """A ``List(Boolean)`` mask is the documented shape for ``gt_col``.

    Regression guard. It briefly raised, on the theory that Boolean is not a
    buffer element type and declaring one was a silent lie. The source decoder
    *casts* rather than reinterprets (``series_to_bytes`` in graph/decode.rs),
    so the declared dtype was always honest and boolean masks worked; the
    rejection broke them.
    """
    from polars_cv.metrics._matching._contour import _detect_source_info

    info = _detect_source_info({"mask": pl.List(pl.List(pl.Boolean))}, "mask")
    assert info.kwargs["dtype"] == "u8"


def test_contour_source_rejects_types_with_no_buffer_meaning() -> None:
    """Types whose cast would fail or lose meaning must still be refused up
    front, rather than becoming an f32 source that fails later and deeper."""
    from polars_cv.metrics._matching._contour import _detect_source_info

    for leaf in (pl.String, pl.Duration):
        with pytest.raises(ValueError, match="no meaningful buffer"):
            _detect_source_info({"mask": pl.List(pl.List(leaf))}, "mask")


# ---------------------------------------------------------------------------
# The verification entry point must not drift from CI
# ---------------------------------------------------------------------------


def test_verify_script_covers_every_ci_check() -> None:
    """``scripts/verify.sh`` must run every check CI runs.

    The script exists so a full verification is one command with per-check exit
    codes, rather than a set of commands whose output someone reads and
    summarises — reading a filtered view is what produced false "all green"
    reports here more than once. That only holds if the script stays complete,
    so the two are pinned together: a check added to CI and not to the script
    would leave the local run passing while CI fails.
    """
    root = Path(__file__).resolve().parents[2]
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    script = (root / "scripts" / "verify.sh").read_text()

    # (substring identifying the check in CI, substring identifying it in the
    # script). Matched loosely on purpose: flags legitimately differ (CI adds
    # -x/-v, the script adds -q), and pinning exact command lines would make
    # this fail on formatting rather than on a missing check.
    required = [
        ("cargo fmt --all -- --check", "cargo fmt --all -- --check"),
        ("cargo clippy", "cargo clippy"),
        ("cargo test -p view-buffer", "cargo test -p view-buffer"),
        ("maturin develop", "maturin develop"),
        ('-m "not network and not slow"', '-m "not network and not slow"'),
        ('-m "slow and not network"', '-m "slow and not network"'),
        ('-m "structural and not slow"', '-m "structural and not slow"'),
        ("ruff check", "ruff check"),
        ("ruff format --check", "ruff format --check"),
        ("mkdocs build --strict", "mkdocs build --strict"),
    ]
    missing = [
        ci_frag
        for ci_frag, script_frag in required
        if ci_frag in ci and script_frag not in script
    ]
    assert not missing, (
        f"scripts/verify.sh is missing checks that CI runs: {missing}. "
        f"A local 'PASS' would not mean CI passes."
    )

    stale = [ci_frag for ci_frag, _ in required if ci_frag not in ci]
    assert not stale, (
        f"this test expects CI to run checks it no longer does: {stale}. "
        f"Update the list rather than leaving it asserting nothing."
    )


#: Shell constructs in a `run:` block that invoke nothing checkable.
_CI_SHELL_NOISE = ("#", "if", "then", "else", "elif", "fi", "echo", "source", "cd")

#: How each command CI runs is classified: the value is a token that must
#: appear in `scripts/verify.sh`, or ``None`` when the line is setup rather
#: than a check.
#:
#: A key matches when its words appear as an ordered subsequence of the
#: command's tokens, starting at the first — so `uv run pytest` matches
#: `uv run --no-sync pytest tests/ -q`, and flags between the words do not
#: need enumerating. The longest match wins.
#:
#: A command line matching **no** entry fails the test. That is what makes this
#: more than a second hand-written list: adding a new checker to CI cannot be
#: silent, because the new command is unclassified until someone says which of
#: the two it is.
_CI_COMMAND_CLASSIFICATION: "dict[str, str | None]" = {
    "cargo fmt": "cargo fmt",
    "cargo clippy": "cargo clippy",
    "cargo test": "cargo test",
    "maturin develop": "maturin develop",
    "pytest": "pytest",
    "uvx ruff check": "ruff check",
    "uvx ruff format": "ruff format",
    "uv run mkdocs": "mkdocs build",
    "uv run pytest": "pytest",
    # Setup: installs and environment, nothing a code change can break.
    "uv sync": None,
    "uv venv": None,
    "uv pip": None,
    "uv run python": None,
}


def _matches_command(key_words: "list[str]", tokens: "list[str]") -> bool:
    """True when *key_words* appear in order among *tokens*, starting at the first.

    Anchoring on the first token keeps `cargo test` from matching a line that
    merely mentions cargo later; allowing gaps after that lets flags sit between
    the words (`uv run --no-sync pytest`) without every combination needing its
    own entry.
    """
    if not tokens or not key_words or tokens[0] != key_words[0]:
        return False
    remaining = iter(tokens[1:])
    return all(word in remaining for word in key_words[1:])


def _ci_run_commands(ci_yaml: str) -> list[str]:
    """Every command line inside a ``run: |`` block of *ci_yaml*.

    Read textually rather than by parsing the YAML: the `run:` bodies are plain
    shell, and what matters is which executables they invoke.
    """
    commands: list[str] = []
    in_run = False
    run_indent = 0
    for raw in ci_yaml.splitlines():
        if re.match(r"^\s*run:\s*\|", raw):
            in_run = True
            run_indent = len(raw) - len(raw.lstrip())
            continue
        if not in_run:
            continue
        if not raw.strip():
            continue
        if len(raw) - len(raw.lstrip()) <= run_indent:
            in_run = False
            continue
        line = raw.strip()
        # Strip leading `VAR=value` assignments: `PYTHONPATH=python uv run …`
        # is a `uv run` invocation, and classifying it by the variable name
        # would make every env-prefixed command look like a new tool.
        while re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+\S", line):
            line = line.split(None, 1)[1]
        if line.split()[0] in _CI_SHELL_NOISE:
            continue
        commands.append(line)
    return commands


def test_no_ci_check_is_missing_from_the_verify_script() -> None:
    """A check *added* to CI must be added to ``verify.sh`` too.

    ``test_verify_script_covers_every_ci_check`` above pins a hand-written list
    both ways — a check in that list must be in CI, and must be in the script.
    What neither direction catches is a check added to CI that nobody adds to
    the list: it is simply never mentioned, and the local run keeps reporting
    PASS while CI fails.

    This closes that direction by reading every command in ``ci.yml``'s
    ``run:`` blocks and requiring each to be *classified* — as a check that
    ``verify.sh`` must also run, or as setup. An unclassified command fails, so
    a checker added to CI cannot pass unnoticed the way an unlisted one could.
    """
    root = Path(__file__).resolve().parents[2]
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    script = (root / "scripts" / "verify.sh").read_text()

    commands = _ci_run_commands(ci)
    assert commands, (
        "no commands were found in ci.yml's run: blocks. Either the workflow "
        "was restructured or the extraction has rotted; this test is checking "
        "nothing."
    )

    unclassified: list[str] = []
    missing: list[str] = []
    for command in commands:
        tokens = command.split()
        matches = [
            key
            for key in _CI_COMMAND_CLASSIFICATION
            if _matches_command(key.split(), tokens)
        ]
        if not matches:
            unclassified.append(command)
            continue
        token = _CI_COMMAND_CLASSIFICATION[max(matches, key=lambda k: len(k.split()))]
        if token is not None and token not in script:
            missing.append(command)

    assert not unclassified, (
        f"ci.yml runs commands this test cannot classify: {sorted(set(unclassified))}. "
        f"Add each to _CI_COMMAND_CLASSIFICATION — with the token verify.sh must "
        f"contain if it is a check, or None if it is setup. Leaving it out is how "
        f"a check gets into CI and never into the local script."
    )
    assert not missing, (
        f"ci.yml runs {sorted(set(missing))}, which scripts/verify.sh does not. "
        f"A local 'PASS' would not mean CI passes. Add the check to the script."
    )


# ---------------------------------------------------------------------------
# Discovery: no guard may find its own files
# ---------------------------------------------------------------------------

#: Where a direct filesystem walk is legitimate, with the reason. Every entry
#: is a place where finding *nothing* is a meaningful answer rather than a
#: broken scan, which is exactly the property `_discovery` refuses to allow.
_DISCOVERY_EXEMPT: dict[str, str] = {
    "_discovery.py": "the discovery module itself",
    "conftest.py": (
        "plugin detection: an empty result means the extension is not built, "
        "which is the answer `plugin_required` needs rather than a failure"
    ),
    "test_streaming_ooc.py": (
        "asserts a spill directory stays empty, so an empty walk is the "
        "assertion rather than a broken scan"
    ),
}


@requires_checkout
def test_scans_go_through_discovery() -> None:
    """File discovery happens in ``tests/_discovery.py`` and nowhere else.

    A guard that finds its own files can find none of them, and then it passes
    while checking nothing — the failure mode `tests/AGENTS.md` calls worse
    than no guard, and one this suite had shipped twice: ``_test_files()`` and
    ``_PACKAGE_MODULES`` were both a bare ``rglob`` whose empty result was
    indistinguishable from a clean bill of health.

    Routing every scan through :mod:`tests._discovery` makes that impossible by
    construction rather than by remembering to assert non-emptiness at each
    site. This test is what stops the next scan from being written the old way,
    and it is deliberately a *mechanism* check (may you glob?) rather than a
    list of the scans that exist.
    """
    offenders: list[str] = []
    for module in suite_files():
        if module.name in _DISCOVERY_EXEMPT:
            continue
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"glob", "rglob"}:
                offenders.append(f"{module.name}:{node.lineno}")

    assert not offenders, (
        "these sites walk the filesystem directly instead of calling "
        f"tests._discovery: {offenders}. A scan that finds nothing passes "
        "vacuously; _discovery raises instead. If an empty result is genuinely "
        "the answer, add the file to _DISCOVERY_EXEMPT with the reason."
    )


@requires_checkout
def test_discovery_exemptions_are_real_files() -> None:
    """An exemption for a file that no longer exists is a stale exemption.

    Without this the allowlist could quietly grow to cover renamed files,
    re-opening the hole it documents closing.
    """
    names = {module.name for module in suite_files()}
    stale = sorted(name for name in _DISCOVERY_EXEMPT if name not in names)
    assert not stale, f"exemptions for files that do not exist: {stale}"


# ---------------------------------------------------------------------------
# The structural lane: pre-commit's selection must stay meaningful
# ---------------------------------------------------------------------------


def _module_marks(source: str) -> set[str]:
    """Mark names a module applies to itself via a top-level ``pytestmark``."""
    marks: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Attribute):
                marks.add(sub.attr)
    return marks


def test_every_source_scanning_module_declares_its_lane() -> None:
    """A module that reads repository files must say which lane it is in.

    ``-m structural`` is what pre-commit runs, so a guard outside it is a guard
    the hook does not enforce. The set is derived rather than listed: reading
    repository files means going through ``_discovery`` (that is
    ``test_scans_go_through_discovery``'s whole job), so importing it is the
    signal, and a new source-scanning guard cannot join without choosing.

    ``slow`` is the other valid answer — ``test_examples.py`` discovers files
    and then spends a minute running them, which is not a pre-commit check.

    Limit worth stating: this covers modules that scan *sets* of files. A guard
    reading one named file (``test_param_strictness.py`` opening ``execute.rs``)
    has no reason to import ``_discovery`` and so is not caught here; those are
    marked by hand.
    """
    offenders = []
    for module in suite_modules():
        source = module.read_text()
        if "_discovery import" not in source:
            continue
        marks = _module_marks(source)
        if not ({"structural", "slow"} & marks):
            offenders.append(module.name)

    assert not offenders, (
        f"these modules scan repository files but declare no lane: {offenders}. "
        f"Add `pytestmark = pytest.mark.structural` (a fast shape check that "
        f"pre-commit should run) or `pytest.mark.slow` (something heavier)."
    )


def test_the_structural_lane_is_not_empty() -> None:
    """``-m structural`` must select a substantial set of guards.

    The lane is the pre-commit hook's entire contents. If the marker were
    renamed, or the declaration dropped from ``pyproject.toml``, the hook would
    select nothing and pass in milliseconds — reading as a working guard
    forever. Count the modules that claim the marker rather than trusting that
    a selection happened.
    """
    marked = [
        m.name for m in suite_modules() if "structural" in _module_marks(m.read_text())
    ]
    assert len(marked) >= 8, (
        f"only {len(marked)} modules carry the structural marker ({marked}); "
        f"the pre-commit lane is meant to cover the guard suite."
    )

    markers = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "structural:" in markers, (
        "the `structural` marker is not declared in pyproject.toml, so "
        "`-m structural` warns and the hook's selection is unenforced."
    )
