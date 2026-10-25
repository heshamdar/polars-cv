"""Enum parity: Python's user-facing enums are the Rust variants, with no second hand-written copy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._structural_helpers import (
    _binary_op_names_from_source,
    _rust_enum_variants,
)
from tests.conftest import plugin_required

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


# Every other registered enum's Python class is generated from the enum
# catalogue (`scripts/gen_ops.py`, `tests/golden/enum_catalog.json`), so there is
# no second copy to compare; `test_the_committed_catalog_is_the_built_one` holds
# the catalogue to the built extension and the module to the catalogue.


def test_every_enum_exclusion_names_a_registered_enum() -> None:
    """``gen_ops.NOT_GENERATED`` may only excuse enums the catalogue holds.

    A registered enum gets a Python class by being registered; the exclusions
    are the stated exceptions. One naming an enum that no longer exists would
    read as coverage while excusing nothing.
    """
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "gen_ops", root / "scripts" / "gen_ops.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registered = {
        e["name"]
        for e in json.loads(
            (root / "tests" / "golden" / "enum_catalog.json").read_text()
        )
    }
    assert set(module.NOT_GENERATED) <= registered, (
        f"stale exclusions: {sorted(set(module.NOT_GENERATED) - registered)}"
    )


@plugin_required
def test_binary_ops_match_rust():
    """``BinaryOp`` has no Python enum, so pin the method names instead."""
    rust = _rust_enum_variants("BinaryOp")
    if rust is None:
        pytest.skip("_lib.enum_catalog() not built")
    # `hasattr(LazyPipelineExpr, name)` would be a weak proxy: the class
    # generates methods from Pipeline, so a Rust op whose name collided with an
    # unrelated generated method would pass. Compare against the names the lazy
    # API actually emits as binary ops.
    emitted = _binary_op_names_from_source()
    assert emitted, "no binary ops scanned from source — scan regex out of date?"
    assert emitted == rust, (
        f"BinaryOp drift between Rust and the lazy API: "
        f"rust-only={sorted(rust - emitted)}, python-only={sorted(emitted - rust)}"
    )


# SourceFormat/SinkFormat are generated from the typed Rust formats
# (`src/formats/`, via `io_catalog.json`), and the graph deserializes the same
# types, so there is no second list of either to pin.


def test_dtype_is_the_only_python_dtype_name_table():
    """No Python enum may re-list dtype spellings alongside DType (enum parity).

    `OutputDType` used to: `f32`/`f64`/`u8` plus a `preserve` value that was a
    synonym for the default rather than a dtype. It is gone, and `out_dtype`
    validates against `DType` —
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
