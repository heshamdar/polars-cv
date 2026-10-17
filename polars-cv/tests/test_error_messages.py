"""A refused value names its op once, as the user spells it.

Every op's ``check()`` message names its op (``blur: sigma -1 is ...``); the
planner and the geometry accessors' build check used to prefix it again with
the wire name, so a message read ``rasterize: rasterize: ...`` or
``contour_densify: densify: ...``, while a check that did not name itself
(``k must be >= 1``) was named only by that wire prefix.
"""

from __future__ import annotations

import math
import re

import polars as pl
import pytest

from polars_cv import Pipeline

from .conftest import plugin_required

pytestmark = plugin_required


def _image() -> Pipeline:
    return Pipeline().source("image_bytes")


def _contour() -> Pipeline:
    return Pipeline().source("contour")


#: (the name the message must start with, a build the op's check refuses).
REFUSALS = {
    "densify": ("densify", lambda: _contour().densify(0.0)),
    "stroke": ("stroke", lambda: _contour().stroke(width=0.0)),
    "offset": ("offset", lambda: _contour().offset(math.inf)),
    "largest": ("largest", lambda: _contour().largest(k=0)),
    "rasterize": (
        "rasterize",
        lambda: _contour().rasterize(width=4, height=4, supersample=0),
    ),
    "apply_lut": ("apply_lut", lambda: _image().grayscale().apply_lut([])),
    "clamp": ("clamp", lambda: _image().clamp(5.0, 1.0)),
    "rotate": ("rotate", lambda: _image().rotate(math.nan)),
    "blur": ("blur", lambda: _image().blur(-1.0)),
    "normalize": ("normalize", lambda: _image().normalize(mean=[0.5], std=[0.2])),
    "perceptual_hash": (
        "perceptual_hash",
        lambda: _image().perceptual_hash(hash_size=0),
    ),
    "extract_contours": (
        "extract_contours",
        lambda: _image().grayscale().extract_contours(min_area_fraction=2.0),
    ),
    "convolve2d": ("convolve2d", lambda: _image().convolve2d([1.0] * 4)),
    "assert_shape": ("assert_shape", lambda: _image().assert_shape()),
    ".contour.densify": ("densify", lambda: pl.col("c").contour.densify(0.0)),
    ".contour.stroke": ("stroke", lambda: pl.col("c").contour.stroke(width=0.0)),
    ".contour.largest": ("largest", lambda: pl.col("c").contour.largest(k=0)),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_refusal_names_its_op_once(case: str) -> None:
    name, build = REFUSALS[case]
    with pytest.raises(ValueError) as refused:
        build()
    message = str(refused.value)
    assert message.startswith(f"{name}: "), message
    assert not re.search(rf"\b(?:contour_)?{name}\b.*\b{name}: ", message), message


def test_an_output_dtype_is_spelled_out_dtype() -> None:
    """``apply_lut`` sets its output dtype as ``clamp``, ``normalize`` and
    ``scale`` do: ``out_dtype``, never a bare ``dtype``, which elsewhere names
    an input's (``source(dtype=)``)."""
    import inspect

    from polars_cv._ops_generated import OP_FIELDS

    for method in ("apply_lut", "clamp", "normalize", "scale"):
        params = inspect.signature(getattr(Pipeline, method)).parameters
        assert "out_dtype" in params and "dtype" not in params, method
    # Only `cast` takes a bare `dtype`: the dtype it casts to is its argument.
    assert [op for op, fields in OP_FIELDS.items() if "dtype" in fields] == ["cast"]


def test_every_generated_enum_is_re_exported_by_types() -> None:
    """``_types`` re-exports the generated enums from the generator's own list
    (``_ops_generated.__all__``), not a hand-kept subset that drifts (it
    lacked ``LineCap``/``LineJoin`` while carrying ``Footprint``)."""
    import enum

    import polars_cv._ops_generated as generated
    import polars_cv._types as types

    enums = {
        name
        for name, value in vars(generated).items()
        if isinstance(value, type)
        and issubclass(value, enum.Enum)
        and value is not enum.Enum
    }
    assert set(getattr(generated, "__all__", ())) == enums
    missing = sorted(
        name
        for name in enums
        if getattr(types, name, None) is not getattr(generated, name)
    )
    assert not missing, f"generated enums _types does not re-export: {missing}"


def test_op_defaults_are_the_catalogues() -> None:
    """``OP_DEFAULTS`` is every declared default (``#[param(default)]``), which
    the hand-written sugar reads instead of restating a literal."""
    import json

    from polars_cv import _ops_generated
    from polars_cv._lib import op_catalog

    declared = {
        op["name"]: {f["name"]: f["default"] for f in op["fields"] if "default" in f}
        for op in json.loads(op_catalog())
    }
    assert getattr(_ops_generated, "OP_DEFAULTS", None) == {
        op: fields for op, fields in declared.items() if fields
    }


def _png(height: int = 4, width: int = 4) -> bytes:
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((height, width), np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("lo", "hi", "reason"),
    [(5.0, 1.0, "range min 5 must not exceed max 1"), (0.0, math.inf, "not finite")],
)
def test_a_histogram_range_no_row_can_use_is_refused_by_its_check(lo, hi, reason):
    """Refused at build for a literal and with its row for an expression,
    named once either way (it was refused only while running, as
    ``Histogram: histogram: ...``)."""
    with pytest.raises(ValueError, match=rf"^histogram: .*{reason}"):
        _image().grayscale().histogram(range=(lo, hi))
    df = pl.DataFrame({"i": [_png()], "lo": [lo], "hi": [hi]})
    pipe = _image().grayscale().histogram(range=(pl.col("lo"), pl.col("hi")))
    with pytest.raises(pl.exceptions.ComputeError, match=reason) as raised:
        df.select(pl.col("i").cv.pipe(pipe).sink("native"))
    assert not re.search(r"histogram: histogram:", str(raised.value), re.I)


@pytest.mark.parametrize("supersample", [0, 17])
def test_rasterize_onto_another_nodes_canvas_checks_its_row(supersample: int) -> None:
    """``rasterize(shape=...)`` resolves its per-row values on its own path;
    they meet the op's check there too (0 hit an engine assert, 17 ran past
    the 1..16 bound)."""
    from polars_cv.geometry.schemas import CONTOUR_SCHEMA

    square = {
        "exterior": [{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 0.0}, {"x": 2.0, "y": 2.0}],
        "holes": [],
        "is_closed": True,
    }
    df = pl.DataFrame(
        {"c": [square], "i": [_png()], "s": [supersample]},
        schema={"c": CONTOUR_SCHEMA, "i": pl.Binary, "s": pl.Int64},
    )
    canvas = pl.col("i").cv.pipe(_image())
    mask = pl.col("c").cv.pipe(
        _contour().rasterize(shape=canvas, supersample=pl.col("s"))
    )
    with pytest.raises(
        pl.exceptions.ComputeError, match="supersample must be between 1 and 16"
    ) as raised:
        df.select(mask.sink("numpy"))
    assert "panicked" not in str(raised.value)


def _contour_frame(contours: list[dict], **columns) -> pl.DataFrame:
    from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

    return pl.DataFrame(
        {"c": [contours], **{k: [v] for k, v in columns.items()}},
        schema={"c": CONTOUR_SET_SCHEMA, **{k: pl.Int64 for k in columns}},
    )


_SQUARE = {
    "exterior": [{"x": 0.0, "y": 0.0}, {"x": 4.0, "y": 0.0}, {"x": 4.0, "y": 4.0}],
    "holes": [],
    "is_closed": True,
}
_LINE = {**_SQUARE, "is_closed": False}


def test_a_per_row_value_is_refused_by_the_same_rule_as_a_literal() -> None:
    """``largest(k=0)`` per row is the rule its build check states, once (its
    accessor kept a second copy with other wording)."""
    with pytest.raises(ValueError) as built:
        pl.col("c").contour.largest(k=0)
    rule = str(built.value)
    with pytest.raises(pl.exceptions.ComputeError) as ran:
        _contour_frame([_SQUARE], k=0).select(
            pl.col("c").contour.largest(k=pl.col("k"))
        )
    assert f"{rule} (row 0)" in str(ran.value), (rule, str(ran.value))


@pytest.mark.parametrize(
    ("build", "engine_name"),
    [
        (lambda c: c.offset(1.0), "Offset"),
        (lambda c: c.union_all(), "UnionAll"),
        (lambda c: c.area(), "Area"),
    ],
)
def test_an_execution_error_names_its_step_as_every_step_does(build, engine_name):
    """A row refused while running is named by the engine's op name, the
    executor's one convention (``Offset:``, ``Histogram:``)."""
    pipe = build(_contour())
    with pytest.raises(
        pl.exceptions.ComputeError, match=rf"{engine_name}: an open contour"
    ):
        _contour_frame([_LINE]).select(pl.col("c").cv.pipe(pipe).sink("native"))


@pytest.mark.parametrize(
    ("op", "engine_name"),
    [("union", "Union"), ("symmetric_difference", "SymmetricDifference")],
)
def test_a_set_operation_names_its_step_as_every_step_does(op, engine_name):
    left = pl.col("c").cv.pipe(_contour())
    right = pl.col("c").cv.pipe(_contour())
    with pytest.raises(
        pl.exceptions.ComputeError, match=rf"{engine_name}: an open contour"
    ):
        _contour_frame([_LINE]).select(getattr(left, op)(right).sink("native"))
