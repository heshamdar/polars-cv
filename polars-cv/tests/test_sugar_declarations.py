"""Hand-written sugar declares the ops it lowers to; every public ``Pipeline`` method has one kind."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


def test_every_public_pipeline_method_has_one_kind() -> None:
    """Every public ``Pipeline`` method is generated, declared sugar
    (``_sugar``), a policy setter, or Pipeline-only — so a hand-written
    method cannot join without a declaration its ``Domain:`` line and the
    truth test below are read from."""
    from polars_cv._ops_generated import _OpsMixin
    from polars_cv.lazy import PIPELINE_ONLY_METHODS
    from polars_cv.pipeline import POLICY_METHODS, SUGAR_OPS, Pipeline

    public = {n for n in dir(Pipeline) if not n.startswith("_")}
    public = {n for n in public if callable(getattr(Pipeline, n))}
    generated = {n for n in vars(_OpsMixin) if not n.startswith("_")}
    assert not generated & set(SUGAR_OPS), "a generated method declared as sugar"
    unclaimed = (
        public - generated - set(SUGAR_OPS) - POLICY_METHODS - PIPELINE_ONLY_METHODS
    )
    assert not unclaimed, (
        f"undeclared hand-written Pipeline methods: {sorted(unclaimed)}"
    )


def _image() -> Pipeline:
    return (
        Pipeline()
        .source("image_bytes", dtype="u8")
        .assert_shape(height=8, width=8, channels=3)
    )


#: One call per sugar method (and one per optional path), on a probe.
_SUGAR_PROBES: list[tuple[str, Callable[[], Pipeline], Callable[[], Pipeline]]] = [
    ("adjust_brightness", _image, lambda: _image().adjust_brightness(factor=1.2)),
    (
        "adjust_brightness",
        _image,
        lambda: _image().adjust_brightness(factor=1.2, preserve_dtype=True),
    ),
    (
        "assert_shape",
        lambda: Pipeline().source("list"),
        lambda: Pipeline().source("list").assert_shape(dims=[2, 2]),
    ),
    ("clamp", _image, lambda: _image().clamp(0.0, 1.0)),
    ("clamp", _image, lambda: _image().clamp(0.0, 1.0, out_dtype="u8")),
    ("flip_h", _image, lambda: _image().flip_h()),
    ("median_filter", _image, lambda: _image().median_filter()),
    ("flip_v", _image, lambda: _image().flip_v()),
    ("laplacian", _image, lambda: _image().laplacian()),
    ("sobel", _image, lambda: _image().sobel()),
    ("sharpen", _image, lambda: _image().sharpen()),
    (
        "morphology_close",
        lambda: _image().grayscale(),
        lambda: _image().grayscale().morphology_close(ksize=3),
    ),
    (
        "morphology_open",
        lambda: _image().grayscale(),
        lambda: _image().grayscale().morphology_open(ksize=3),
    ),
    (
        "rasterize",
        lambda: Pipeline().source("contour"),
        lambda: Pipeline().source("contour").rasterize(width=4, height=4),
    ),
    ("resize_scale", _image, lambda: _image().resize_scale(scale=0.5)),
    ("shear", _image, lambda: _image().shear(sx=0.1, output_size=(8, 8))),
    ("scale", _image, lambda: _image().scale(2.0)),
    ("scale", _image, lambda: _image().scale(2.0, out_dtype="u8")),
    ("to_bgr", _image, lambda: _image().to_bgr()),
    ("to_hsv", _image, lambda: _image().to_hsv()),
    ("to_lab", _image, lambda: _image().to_lab()),
    ("to_ycbcr", _image, lambda: _image().to_ycbcr()),
]


@plugin_required
def test_sugar_appends_exactly_the_ops_it_declares() -> None:
    """A sugar method's ``Domain:`` line is composed from the ops it declares,
    so the declaration must be what it does: its required ops in order, then
    only its declared optional ones."""
    from polars_cv.pipeline import SUGAR_OPS
    from tests._plan_view import op_names

    assert {name for name, _, _ in _SUGAR_PROBES} == set(SUGAR_OPS), (
        "every sugar method needs a probe here, and every probe a declaration"
    )
    for name, before, call in _SUGAR_PROBES:
        required, optional = SUGAR_OPS[name]
        appended = op_names(call())[len(op_names(before())) :]
        head, tail = appended[: len(required)], appended[len(required) :]
        assert tuple(head) == required and set(tail) <= set(optional), (
            f"{name} appends {appended}, declared {required} then any of {optional}"
        )


@plugin_required
def test_sugar_defaults_are_their_ops_defaults() -> None:
    """A sugar method's parameter named for a field of an op it declares
    defaults to that field's default, as the Rust definition states it.

    Sugar re-declares its signature by hand (``rasterize(supersample=1)``,
    ``morphology_open(footprint="square")``), so a default changed in the op
    would otherwise leave the sugar quietly passing the old one. Read from the
    catalogue (``op_catalog``), the one place defaults are declared.
    """
    import inspect

    from polars_cv._lib import op_catalog
    from polars_cv.pipeline import SUGAR_OPS

    defaults = {
        op["name"]: {f["name"]: f["default"] for f in op["fields"] if "default" in f}
        for op in json.loads(op_catalog())
    }
    checked, mismatched = 0, []
    for name, (required, optional) in SUGAR_OPS.items():
        params = inspect.signature(getattr(Pipeline, name)).parameters
        for op in (*required, *optional):
            for field, default in defaults[op].items():
                param = params.get(field)
                if param is None or param.default is inspect.Parameter.empty:
                    continue
                checked += 1
                if param.default != default:
                    mismatched.append(
                        f"{name}({field}={param.default!r}), but {op} defaults "
                        f"{field} to {default!r}"
                    )
    assert checked >= 5, f"only {checked} defaults compared; the scan matches nothing"
    assert not mismatched, "\n".join(mismatched)
