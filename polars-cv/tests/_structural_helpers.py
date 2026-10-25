"""Helpers shared by the structural guards (split out of ``test_sanitation.py``)."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import polars as pl
import pytest

import polars_cv
from polars_cv import Pipeline
from tests._schema_parity import assert_plan_equals_exec, leaf_dtype


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


def _plan_state(domain: str, dtype: str, ndim: "int | None") -> object:
    """A ``PlanState`` with nothing known about the sizes, built the only way
    one can be: by planning a real pipeline (``auto`` = no declared dtype)."""
    image = Pipeline().source("image_bytes", dtype=None if dtype == "auto" else dtype)
    if (domain, ndim) == ("buffer", 3):
        state = image._state
    elif (domain, ndim) == ("contour", None):
        state = image.grayscale().threshold(1).extract_contours()._state
    else:
        msg = f"no pipeline plans a ({domain}, {dtype}, {ndim}) state"
        raise ValueError(msg)
    assert (state.domain, state.ndim) == (domain, ndim), state
    return state


def _planned_after(op_json: str, state: object, refs: "dict | None" = None) -> object:
    """The state after planning *op_json* from *state*: one ``Plan.push``."""
    from polars_cv._lib import Plan

    return Plan.continuing(state).push(op_json, refs).state


def _binary_op_names_from_source() -> set[str]:
    """The element-wise binary ops among :func:`_node_op_calls_from_source`.

    ``_node_op`` builds every ``lazy_only`` op, so the contour set ops
    (``union``, ``intersection``, …, CR-99), the mask ops and ``channel_merge``
    go through it too; the element-wise ones are those that read any numeric container, a
    ``buffer`` and a ``vector`` alike, as their catalogued domains say.
    """
    from polars_cv._ops_generated import OP_DOMAINS

    numeric = {"buffer", "vector"}
    return {
        n
        for n in _node_op_calls_from_source()
        if {d["input"] for d in OP_DOMAINS[n]} == numeric
    }


def _rust_enum_variants(enum_name):
    """``enum_name``'s variant spellings, read from the built enum catalogue."""
    fn = getattr(_lib(), "enum_catalog", None)
    if not callable(fn):
        return None
    (desc,) = [e for e in json.loads(fn()) if e["name"] == enum_name]
    return set(desc["variants"])


def _node_op_calls_from_source() -> set[str]:
    """Every op name the lazy API emits via ``self._node_op("<name>")``.

    The methods are generated into ``_LazyOpsMixin``; ``lazy.py`` is scanned
    too, so a hand-written node-op method would be counted rather than missed.
    """

    pkg = Path(polars_cv.__file__).parent
    return {
        name
        for module in ("lazy.py", "_ops_generated.py")
        for name in re.findall(
            r'self\._node_op\(\s*"([a-z_]+)"', (pkg / module).read_text()
        )
    }
