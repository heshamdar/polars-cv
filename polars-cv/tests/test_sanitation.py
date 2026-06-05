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
    """No contract exists for an op the Pipeline never emits (B2: sobel/laplacian/sharpen)."""
    lib = getattr(polars_cv, "_lib", None)
    contract_fn = getattr(lib, "op_contract", None) if lib is not None else None
    if not callable(contract_fn):
        pytest.skip("_lib.op_contract() not implemented yet (Phase 1/3)")
    # sobel/laplacian/sharpen lower to convolve2d and must NOT have standalone contracts.
    for lowered in ("sobel", "laplacian", "sharpen"):
        with pytest.raises(Exception):
            contract_fn(lowered)


# ---------------------------------------------------------------------------
# 3. Enum parity (A4) — Python user enums match Rust variants
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("enum_name", ["Domain", "DType", "SourceFormat", "SinkFormat"])
def test_enum_parity_python_matches_rust(enum_name):
    """Python user-facing enum values must equal the Rust enum variant set (A4)."""
    lib = getattr(polars_cv, "_lib", None)
    variants_fn = getattr(lib, "enum_variants", None) if lib is not None else None
    if not callable(variants_fn):
        pytest.skip("_lib.enum_variants() not implemented yet (Phase 2)")
    import polars_cv._types as t

    py_enum = getattr(t, enum_name)
    py_values = {m.value for m in py_enum}
    rust_values = set(variants_fn(enum_name))
    assert py_values == rust_values, (
        f"{enum_name}: python {py_values} != rust {rust_values}"
    )


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
