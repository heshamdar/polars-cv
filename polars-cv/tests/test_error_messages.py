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

from ._rows import ROW, assert_names_row
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
    assert rule in str(ran.value), (rule, str(ran.value))
    assert_names_row(str(ran.value), 0, 1)


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


def _row_error(run) -> str:
    """The refusal *run* raises, without either surface's wrapping: the
    pipeline's ``Pipeline execution failed:`` / ``Op resolution error:``, the
    row reference (``tests/_rows.py``), and polars' trailing expression trace."""
    with pytest.raises(pl.exceptions.ComputeError) as raised:
        run()
    message = str(raised.value).split("\n\nThis error occurred")[0]
    message = message.replace("the plugin failed with message: ", "")
    for wrapper in ("Pipeline execution failed: ", "Op resolution error: "):
        message = message.replace(wrapper, "")
    return ROW.sub("", message).rstrip()


#: (pipeline method, accessor method, arguments, the row, its step's name)
_REFUSED_ROWS = [
    ("area", "area", {}, _LINE, "Area"),
    ("centroid", "centroid", {}, _LINE, "Centroid"),
    ("offset", "offset", {"distance": 1.0}, _LINE, "Offset"),
    ("union_all", "union_all", {}, _LINE, "UnionAll"),
    ("largest", "largest", {"k": 1}, _LINE, "Largest"),
    ("boundary", "boundary", {}, _LINE, "Boundary"),
    ("close", "close", {}, _SQUARE, "Close"),
    ("scale_contour", "scale", {"sx": 2.0, "sy": 2.0}, _LINE, "Scale"),
    ("smooth", "smooth", {"iterations": pl.col("p").cast(pl.Int64)}, _SQUARE, "Smooth"),
    (
        "densify",
        "densify",
        {"max_segment_length": pl.col("p")},
        _SQUARE,
        "Densify",
    ),
]


@pytest.mark.parametrize(
    ("method", "accessor", "kwargs", "row", "step"),
    _REFUSED_ROWS,
    ids=[case[0] for case in _REFUSED_ROWS],
)
def test_a_refused_row_reads_the_same_on_both_surfaces(
    method, accessor, kwargs, row, step
) -> None:
    """A ``.contour`` accessor runs the pipeline step it is, so a row it
    refuses is refused in the step's words, named by the step once — as the
    executor names every step (``Offset:``, ``Histogram:``). The accessors
    named none of these, and the engine functions that named their own
    refusal did so in lower case on both surfaces."""
    from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

    p = {"smooth": 40.0, "densify": 1e-9}.get(method, 0.0)
    frame = pl.DataFrame(
        {"c": [[row]], "p": [p]}, schema={"c": CONTOUR_SET_SCHEMA, "p": pl.Float64}
    )
    pipe = getattr(_contour(), method)(**kwargs)
    by_pipeline = _row_error(
        lambda: frame.select(pl.col("c").cv.pipe(pipe).sink("native"))
    )
    by_accessor = _row_error(
        lambda: frame.select(getattr(pl.col("c").contour, accessor)(**kwargs))
    )
    assert by_pipeline == by_accessor
    assert by_pipeline.startswith(f"{step}: "), by_pipeline
    assert not re.match(rf"{step}: {step}:", by_pipeline, re.I), by_pipeline


def test_an_empty_contour_has_no_bounding_box_on_either_surface() -> None:
    """A contour with no points has no extent: the step gave a 0x0 box at the
    origin, the accessor a null. Both now read NaN (a measure vector has no
    null), never a box that is not there."""
    from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

    empty = {"exterior": [], "holes": [], "is_closed": True}
    frame = pl.DataFrame({"c": [[empty]]}, schema={"c": CONTOUR_SET_SCHEMA})
    by_pipeline = frame.select(
        pl.col("c").cv.pipe(_contour().bounding_box()).sink("native")
    ).item()
    by_accessor = frame.select(pl.col("c").contour.bounding_box()).item().to_list()
    assert len(by_pipeline) == 4 and all(math.isnan(v) for v in by_pipeline)
    (box,) = by_accessor
    assert box is not None and all(math.isnan(v) for v in box.values()), box


def _rgb_png(width: int, height: int) -> bytes:
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _first_line(err: pytest.ExceptionInfo) -> str:
    return str(err.value).split("\n\nThis error occurred")[0]


def test_a_failing_row_is_named_as_the_accessors_name_it() -> None:
    """A row an op refuses ends with its row reference, as a ``.contour``
    accessor's refusal does: the second row here is narrower than the crop."""
    frame = pl.DataFrame({"img": [_rgb_png(40, 12), _rgb_png(31, 12)]})
    pipe = _image().crop(left=8, width=24)
    with pytest.raises(pl.exceptions.ComputeError) as err:
        frame.select(pl.col("img").cv.pipe(pipe).sink("numpy"))
    first = _first_line(err)
    assert "crop window 8..32 on axis 1" in first, first
    assert ROW.search(first) and first.endswith(")"), first
    assert_names_row(first, 1, 2)


def test_a_bad_per_row_value_names_its_op_and_parameter() -> None:
    """A per-row value no op can take names the op, the parameter it was
    bound to and the column it came from, once each."""
    frame = pl.DataFrame({"img": [_rgb_png(31, 12)], "l": [-1]})
    pipe = _image().crop(left=0, width=pl.col("l"))
    with pytest.raises(pl.exceptions.ComputeError) as err:
        frame.select(pl.col("img").cv.pipe(pipe).sink("numpy"))
    first = _first_line(err)
    assert "crop: width: Parameter column 'l': -1 cannot be negative" in first, first
    assert first.count("crop:") == 1, first
    assert_names_row(first, 0, 1)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_a_row_is_named_within_its_batch_on_every_engine(engine: str) -> None:
    """The row a refusal names is the batch's, and the message says so.

    The 51st row is bad. In memory the plugin is called once with the whole
    frame, so the batch is the frame and the row is 50. Under streaming,
    with batches of at most 8 rows, the plugin cannot know where its batch
    starts: the row is within a batch of 8 or fewer, which the message states
    rather than presenting a batch index as the frame's.
    """
    rows = [_rgb_png(40, 12)] * 50 + [_rgb_png(31, 12)]
    lf = pl.DataFrame({"img": rows}).lazy()
    pipe = _image().crop(left=8, width=24)
    with pl.Config(streaming_chunk_size=8):
        with pytest.raises(pl.exceptions.ComputeError) as err:
            lf.select(pl.col("img").cv.pipe(pipe).sink("numpy")).collect(engine=engine)
    first = _first_line(err)
    assert_names_row(first, 50, 51)
    ((index, batch),) = [(int(i), int(n)) for i, n in ROW.findall(first)]
    if engine == "in-memory":
        assert (index, batch) == (50, 51), first
    else:
        assert batch <= 8, first


def test_a_window_bound_past_2_31_suggests_an_unsigned_wrap() -> None:
    """``UInt32`` arithmetic in polars wraps below zero to ~2^32 before
    polars-cv sees the value, so a window that far out says so."""
    frame = pl.DataFrame(
        {"img": [_rgb_png(31, 12)], "w": [31]}, schema_overrides={"w": pl.UInt32}
    )
    pipe = _image().crop(left=pl.col("w") - 32, width=4)
    with pytest.raises(pl.exceptions.ComputeError) as err:
        frame.select(pl.col("img").cv.pipe(pipe).sink("numpy"))
    first = _first_line(err)
    assert "4294967295" in first
    assert "unsigned subtraction" in first, first


def test_a_parameter_is_named_as_the_user_wrote_it_not_by_its_wrapper() -> None:
    """``resize(height=pl.col("h"))``'s null names ``resize: height:``. The
    engine's ``ImageOp { kind }`` wrapper is not a parameter, and its field
    name leaked into the message as ``resize: kind: height:``."""
    frame = pl.DataFrame(
        {"img": [_rgb_png(4, 4)], "h": [None]}, schema_overrides={"h": pl.Int64}
    )
    pipe = _image().resize(height=pl.col("h"), width=4)
    with pytest.raises(pl.exceptions.ComputeError) as err:
        frame.select(pl.col("img").cv.pipe(pipe).sink("numpy"))
    first = _first_line(err)
    assert "resize: height: Parameter column 'h' has a null value" in first, first
    assert "kind" not in first, first
