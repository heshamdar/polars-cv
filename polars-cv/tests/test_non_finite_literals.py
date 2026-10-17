"""A non-finite float literal crosses the wire like any other value.

JSON, the wire every op and geometry call crosses, has no NaN or infinity, so
every float field reads and writes them as ``"nan"``, ``"inf"`` and ``"-inf"``
(``naming.rs``; ``_types._literal`` encodes them). A literal then meets the
op's own check, exactly as the same value passed per row through an
expression does. (CR-103 first refused such a literal in Python, which made
``apply_lut([[nan]], dtype="f32")`` an error while ``pl.lit(nan)`` worked.)

The sweep below is registry-driven: every float field of every pipeline op
(``OP_FIELDS``), at each non-finite value, must either be refused while the
pipeline is built, naming the op, or plan and run to a value or a row error —
never a panic, and never the JSON decoder's "expected value".
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_column
from polars_cv._ops_generated import OP_FIELDS
from tests._op_cases import OP_CASES, case_base

from .conftest import make_rect_png, plugin_required

pytestmark = plugin_required

NON_FINITE = [math.nan, math.inf, -math.inf]


def _float_paths(ty: dict, path: tuple[int, ...] = ()) -> Iterator[tuple[int, ...]]:
    """Where a field's float scalars sit: ``()`` for a scalar, ``(0,)`` for a
    list's first element, ``(0, 0)`` for a nested list's."""
    if ty.get("kind") == "scalar" and ty.get("py") in ("float", "bound"):
        yield path
    elif ty.get("kind") == "list":
        yield from _float_paths(ty["inner"], (*path, 0))


def _catalogue() -> dict[str, dict]:
    from polars_cv._lib import op_catalog

    return {op["python"]: op for op in json.loads(op_catalog())}


CATALOGUE = _catalogue()

#: An internal op is reached through its generated ``_<name>`` method, whose
#: fields its sugar may spell differently (``clamp(min_val=)`` for ``min``):
#: the values its required fields take here.
INTERNAL_FIELDS: dict[str, dict[str, Any]] = {"clamp": {"min": 0.0, "max": 1.0}}


def _float_fields() -> list[tuple[str, str, tuple[int, ...]]]:
    return sorted(
        (python, field["name"], path)
        for python, op in CATALOGUE.items()
        for field in op["fields"]
        for path in _float_paths(OP_FIELDS[op["name"]][field["name"]])
    )


FLOAT_FIELDS = _float_fields()


def test_the_sweep_covers_every_float_field() -> None:
    """Guard the guard: each op with a float field has a case to sweep it in."""
    assert len(FLOAT_FIELDS) > 25, f"only {len(FLOAT_FIELDS)} float fields found"
    uncovered = sorted({op for op, _, _ in FLOAT_FIELDS if OP_CASES.get(op) is None})
    assert not uncovered, f"ops with a float field and no case: {uncovered}"


def _with(value: Any, path: tuple[int, ...], new: float) -> Any:
    if not path:
        return new
    items = list(value)
    items[path[0]] = _with(items[path[0]], path[1:], new)
    return items


@pytest.mark.parametrize("value", NON_FINITE, ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize(
    ("op", "field", "path"),
    FLOAT_FIELDS,
    ids=[
        f"{op}.{field}{''.join(f'[{i}]' for i in p)}" for op, field, p in FLOAT_FIELDS
    ],
)
def test_every_float_literal_is_checked_by_its_op(
    op: str, field: str, path: tuple[int, ...], value: float
) -> None:
    domain, kwargs = OP_CASES[op]
    method = op
    if CATALOGUE[op]["visibility"] == "internal":
        method = f"_{op}"
        fields = {f["name"] for f in CATALOGUE[op]["fields"]}
        kwargs = {k: v for k, v in kwargs.items() if k in fields}
        kwargs = {**INTERNAL_FIELDS.get(op, {}), **kwargs}
    kwargs = dict(kwargs)
    if path and field not in kwargs:
        pytest.skip(f"{op}'s case does not set the list field {field}")
    kwargs[field] = _with(kwargs.get(field), path, value)
    try:
        pipe = getattr(case_base(op, domain), method)(**kwargs)
    except ValueError as e:
        assert "expected value" not in str(e), e
        assert op in str(e) or field in str(e), f"refused without naming it: {e}"
        return
    df = pl.DataFrame({"img": [make_rect_png(100, 200, 3)]})
    sink = "numpy" if pipe.current_domain() == "buffer" else "native"
    try:
        df.select(pl.col("img").cv.pipe(pipe).sink(sink))
    except pl.exceptions.ComputeError as e:
        assert "panicked" not in str(e), e


def test_a_nan_table_entry_is_a_nan() -> None:
    """The case CR-103 refused: a palette mapping a label to NaN."""
    table = [[0.5], [math.nan], [math.inf]]
    pipe = Pipeline().source("raw", dtype="u8").apply_lut(table, dtype="f32")
    data = np.array([[0, 1, 2]], np.uint8)
    df = pl.DataFrame({"b": [data.tobytes()]})
    out = df.select(pl.col("b").cv.pipe(pipe).sink("numpy"))
    (got,) = numpy_from_column(out.to_series(), copy=True)
    assert got.ravel()[0] == 0.5
    assert math.isnan(got.ravel()[1]) and got.ravel()[2] == math.inf


@pytest.mark.parametrize("value", NON_FINITE, ids=["nan", "inf", "-inf"])
def test_an_op_refuses_a_value_by_its_own_rule(value: float) -> None:
    """``offset`` needs a finite distance: its check names it, at build."""
    with pytest.raises(ValueError, match="distance must be finite"):
        Pipeline().source("contour").offset(value)


def test_a_geometry_literal_meets_the_functions_check_at_build() -> None:
    """The accessor refuses it as the pipeline op does: when written."""
    with pytest.raises(ValueError, match="distance must be finite"):
        pl.col("c").contour.offset(math.inf)


def test_a_bound_takes_an_infinity_and_refuses_a_nan() -> None:
    """``histogram(bins=)`` edges are bounds: open outer edges, never NaN."""
    image = Pipeline().source("image_bytes", dtype="u8")
    assert image.histogram(bins=[-math.inf, 128.0, math.inf]) is not None
    with pytest.raises(ValueError, match="a bound cannot be NaN"):
        image.histogram(bins=[0.0, math.nan, 255.0])


def test_the_same_value_per_row_reaches_the_ops_own_check() -> None:
    """The expression path is unchanged: the op refuses the value for its row."""
    from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

    square = {
        "exterior": [{"x": 0.0, "y": 0.0}, {"x": 4.0, "y": 0.0}, {"x": 4.0, "y": 4.0}],
        "holes": [],
        "is_closed": True,
    }
    df = pl.DataFrame(
        {"c": [[square]], "d": [math.nan]},
        schema={"c": CONTOUR_SET_SCHEMA, "d": pl.Float64},
    )
    pipe = Pipeline().source("contour").offset(pl.col("d"))
    with pytest.raises(pl.exceptions.ComputeError, match="distance must be finite"):
        df.select(pl.col("c").cv.pipe(pipe).sink("native"))


@pytest.mark.parametrize(
    ("lo", "hi", "match"),
    [(5.0, 1.0, "min 5 is greater than max 1"), (math.nan, 1.0, "min is NaN")],
)
def test_clamp_refuses_bounds_no_row_can_clamp_to(lo: float, hi: float, match: str):
    """Found by the sweep: ``f32::clamp`` panics on a NaN bound or crossed
    bounds, which a per-row pair already reached. Its check refuses them,
    at build for literals and with the row for expressions."""
    with pytest.raises(ValueError, match=match):
        Pipeline().source("raw", dtype="f32").clamp(lo, hi)
    df = pl.DataFrame(
        {"b": [np.zeros(4, np.float32).tobytes()], "lo": [lo], "hi": [hi]},
        schema={"b": pl.Binary, "lo": pl.Float32, "hi": pl.Float32},
    )
    pipe = Pipeline().source("raw", dtype="f32").clamp(pl.col("lo"), pl.col("hi"))
    with pytest.raises(pl.exceptions.ComputeError, match=match) as raised:
        df.select(pl.col("b").cv.pipe(pipe).sink("numpy"))
    assert "panicked" not in str(raised.value)
