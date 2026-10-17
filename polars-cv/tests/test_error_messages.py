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
