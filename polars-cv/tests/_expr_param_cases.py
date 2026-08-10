"""The one table of "every ``Pipeline`` parameter that accepts an expression".

A parameter is *expression-eligible* iff its value has no effect on the output
shape, rank or dtype — the rule stated in the root ``CLAUDE.md`` — and the way
a parameter opts in is ``Pipeline._track_expr`` (directly, or via
``_param_list``/``_enum_param``, which take it as a callback). The visible
consequence of opting in is the annotation: ``IntOrExpr``, ``FloatOrExpr``,
``BoolOrExpr``, ``StrOrExpr`` or a bare ``pl.Expr`` union.

``test_expression_op_params.py`` sweeps this table, and
``test_every_expression_parameter_has_a_case`` reads the *annotations* back off
``Pipeline`` and fails if one of them has no case here. That is the ratchet: a
new op whose parameter accepts an expression cannot join the library without
also getting swept. The annotation is the authority, so the ratchet cannot be
satisfied by editing this file alone.

Each case is a whole-pipeline factory rather than a kwargs dict. Several of the
eligible parameters are *elements* of a list or tuple argument — a
``warp_affine`` matrix coefficient, an ``output_size`` half, a ``normalize``
mean — and a kwargs table cannot express "this one element is an expression
while its siblings stay literal", which is precisely the case the
element-by-element ``_param_list`` lowering exists to serve.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import polars as pl

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

# --- Input columns the cases draw from -------------------------------------

#: An RGB image column. Content is noisy (``make_image_png``) so that a change
#: in any parameter shows up as a different result — a flat image absorbs most
#: of them and would make the "the value reached the kernel" assertion vacuous.
IMAGE = "image"

#: A binary-ish image with one filled rectangle, for contour extraction.
RECT = "rect"

#: An image with a rectangular hole, so ``extract_contours(mode=)`` has a
#: second border to find. ``mode="external"`` and ``mode="all"`` are identical
#: on a solid shape.
RING = "ring"

#: A hand-written ``List[Contour]`` column feeding the ``contour`` source and
#: ``label_reduce``'s operand.
CONTOURS = "contours"

#: A rotated square. Its bounding box is strictly larger than its interior,
#: which is what makes ``label_reduce(region_mode=)`` observable.
DIAMOND = "diamond"


# --- Base pipelines --------------------------------------------------------


def rgb() -> Pipeline:
    """Three-channel buffer straight off the image source.

    The source declares ``dtype="u8"`` because the sweep compares values
    through the ``list`` sink, which needs a concrete element dtype at
    planning time — an image source's decoded dtype is only known at runtime.
    All the fixtures here are 8-bit PNGs, so the declaration is true.
    """
    return Pipeline().source("image_bytes", dtype="u8")


def gray() -> Pipeline:
    """Single-channel buffer, for the kernels that require one."""
    return rgb().grayscale()


def mask() -> Pipeline:
    """Binary u8 mask, for the morphology and contour ops."""
    return gray().threshold(128)


def contour() -> Pipeline:
    """Contour domain, reached the ordinary way (buffer → contour)."""
    return mask().extract_contours()


# --- The case record -------------------------------------------------------


@dataclass(frozen=True)
class ExprCase:
    """One expression-eligible parameter, and how to exercise it.

    Attributes:
        op: The ``Pipeline`` method that owns the parameter.
        param: The parameter's name in that method's signature. ``op`` and
            ``param`` together form the key the coverage ratchet matches
            against the introspected annotations.
        build: Builds a complete pipeline from one parameter value. Called
            with a literal and with a ``pl.Expr`` — the same factory both
            times, so the two paths cannot drift apart in the test itself.
        values: One value per row, all distinct. Two is the minimum; three
            catches "row 0's value was broadcast to the rest" as well as a
            reversed pairing.
        column: Which input column the pipeline reads.
        dtype: Polars dtype for the parameter column. ``None`` lets Polars
            infer, which is right for most cases; naming it explicitly is how
            the integer-column-into-a-float-parameter path gets covered.
        varies: True when the values must produce distinct outputs. This is
            the assertion that proves the value is *used*: the
            expression-versus-literal comparison alone cannot catch a
            parameter the kernel ignores, because it would ignore it on both
            paths. Set False only with a ``note`` saying why the parameter
            cannot change the output.
        literal: False for a parameter that has no literal spelling
            (``label_reduce(contours=)`` is an operand column, not a value),
            which skips the expression-versus-literal leg.
        note: Required whenever ``varies`` is False.
    """

    op: str
    param: str
    build: Callable[[Any], Pipeline]
    values: tuple[Any, ...]
    column: str = IMAGE
    dtype: pl.DataType | None = None
    varies: bool = True
    literal: bool = True
    note: str = ""

    @property
    def key(self) -> str:
        """``method.parameter`` — the key the coverage ratchet matches on."""
        return f"{self.op}.{self.param}"

    def __post_init__(self) -> None:
        if not self.varies and not self.note:
            msg = (
                f"{self.key}: varies=False needs a note saying why the "
                "parameter cannot change the output"
            )
            raise ValueError(msg)
        if len(self.values) < 2:
            msg = f"{self.key}: needs at least two row values"
            raise ValueError(msg)
        if self.varies and len(set(map(repr, self.values))) != len(self.values):
            msg = f"{self.key}: row values must be distinct"
            raise ValueError(msg)


def square(x0: float, y0: float, x1: float, y1: float) -> dict:
    """An axis-aligned, counter-clockwise square contour."""
    return {
        "exterior": [
            {"x": x0, "y": y0},
            {"x": x1, "y": y0},
            {"x": x1, "y": y1},
            {"x": x0, "y": y1},
        ],
        "holes": [],
        "is_closed": True,
    }


#: The contour set every row of the ``contours`` column carries. It sits inside
#: the white block of the ``rect`` image, so ``label_reduce`` scores a region
#: that actually holds signal.
CONTOUR_SET = [square(4.0, 4.0, 12.0, 12.0)]

#: Two visibly different contour sets, for the one parameter that takes a
#: contour column rather than a value.
CONTOUR_SET_A = [square(4.0, 4.0, 12.0, 12.0)]
CONTOUR_SET_B = [square(0.0, 0.0, 4.0, 4.0)]

#: The rotated square behind the :data:`DIAMOND` column.
DIAMOND_SET = [
    {
        "exterior": [
            {"x": 8.0, "y": 2.0},
            {"x": 14.0, "y": 8.0},
            {"x": 8.0, "y": 14.0},
            {"x": 2.0, "y": 8.0},
        ],
        "holes": [],
        "is_closed": True,
    }
]


# --- The table -------------------------------------------------------------

CASES: list[ExprCase] = [
    # --- source (the contour source rasterizes onto a canvas) --------------
    ExprCase(
        "source",
        "width",
        lambda v: Pipeline().source("contour", width=v, height=12),
        (10, 14, 16),
        column=CONTOURS,
    ),
    ExprCase(
        "source",
        "height",
        lambda v: Pipeline().source("contour", width=12, height=v),
        (10, 14, 16),
        column=CONTOURS,
    ),
    ExprCase(
        "source",
        "fill_value",
        lambda v: Pipeline().source("contour", width=12, height=12, fill_value=v),
        (255, 128, 64),
        column=CONTOURS,
    ),
    ExprCase(
        "source",
        "background",
        lambda v: Pipeline().source("contour", width=12, height=12, background=v),
        (0, 32, 96),
        column=CONTOURS,
    ),
    # --- assert_shape ------------------------------------------------------
    # These four state a fact about the buffer rather than changing it, so a
    # correct assertion is invisible in the output by construction. What they
    # must do instead — reject the row whose buffer disagrees — is pinned by
    # `TestAssertShapeExpressions` in test_expression_op_params.py.
    ExprCase(
        "assert_shape",
        "height",
        lambda v: rgb().assert_shape(height=v),
        (16, 16),
        varies=False,
        note="an assertion that holds leaves the buffer untouched",
    ),
    ExprCase(
        "assert_shape",
        "width",
        lambda v: rgb().assert_shape(width=v),
        (16, 16),
        varies=False,
        note="an assertion that holds leaves the buffer untouched",
    ),
    ExprCase(
        "assert_shape",
        "channels",
        lambda v: rgb().assert_shape(channels=v),
        (3, 3),
        varies=False,
        note="an assertion that holds leaves the buffer untouched",
    ),
    ExprCase(
        "assert_shape",
        "batch",
        lambda v: rgb().assert_shape(batch=v),
        (1, 1),
        varies=False,
        note="an assertion that holds leaves the buffer untouched",
    ),
    # --- view / shape ------------------------------------------------------
    ExprCase(
        "reshape",
        "shape",
        # 16 x 16 x 3 = 768 elements. The entry *count* of `shape` is
        # structural, the entries themselves are not — and `768 // v` is
        # ordinary Polars arithmetic when `v` is an expression, so the same
        # factory serves both legs.
        lambda v: rgb().reshape([v, 768 // v]),
        (8, 16),
    ),
    ExprCase(
        "crop",
        "top",
        lambda v: rgb().crop(top=v, left=0, height=6, width=6),
        (0, 3, 6),
    ),
    ExprCase(
        "crop",
        "left",
        lambda v: rgb().crop(top=0, left=v, height=6, width=6),
        (0, 3, 6),
    ),
    ExprCase(
        "crop",
        "height",
        lambda v: rgb().crop(top=0, left=0, height=v, width=6),
        (4, 8, 12),
    ),
    ExprCase(
        "crop",
        "width",
        lambda v: rgb().crop(top=0, left=0, height=6, width=v),
        (4, 8, 12),
    ),
    # --- scalar arithmetic -------------------------------------------------
    ExprCase("scale", "factor", lambda v: rgb().scale(v), (0.5, 1.0, 2.0)),
    ExprCase(
        "normalize",
        "mean",
        lambda v: rgb().normalize(method="preset", mean=[v, 0.5, 0.5], std=[1.0] * 3),
        (0.0, 0.25, 0.5),
    ),
    ExprCase(
        "normalize",
        "std",
        lambda v: rgb().normalize(method="preset", mean=[0.5] * 3, std=[v, 1.0, 1.0]),
        (0.5, 1.0, 2.0),
    ),
    ExprCase(
        "clamp",
        "min_val",
        lambda v: rgb().clamp(min_val=v, max_val=255.0),
        (0.0, 64.0, 128.0),
    ),
    ExprCase(
        "clamp",
        "max_val",
        lambda v: rgb().clamp(min_val=0.0, max_val=v),
        (64.0, 128.0, 255.0),
    ),
    ExprCase(
        "channel_select",
        "index",
        lambda v: rgb().channel_select(index=v),
        (0, 1, 2),
    ),
    ExprCase(
        "channel_swap",
        "order",
        # Only the first entry moves; the list *length* stays structural.
        lambda v: rgb().channel_swap(order=[v, 1, 2]),
        (0, 1, 2),
    ),
    ExprCase(
        "adjust_contrast",
        "factor",
        lambda v: rgb().adjust_contrast(factor=v),
        (0.5, 1.0, 2.0),
    ),
    ExprCase(
        "adjust_gamma",
        "gamma",
        lambda v: rgb().adjust_gamma(gamma=v),
        (0.5, 1.0, 2.2),
    ),
    ExprCase(
        "adjust_brightness",
        "factor",
        lambda v: rgb().adjust_brightness(factor=v),
        (0.5, 1.0, 1.5),
    ),
    # --- filtering ---------------------------------------------------------
    ExprCase(
        "convolve2d",
        "kernel",
        # One coefficient varies; the other eight stay literal zeros.
        lambda v: gray().convolve2d([0.0] * 4 + [v] + [0.0] * 4, 3),
        (0.5, 1.0, 2.0),
    ),
    ExprCase(
        "convolve2d",
        "ksize",
        lambda v: gray().convolve2d([0.0] * 4 + [1.0] + [0.0] * 4, v),
        (3, 3),
        varies=False,
        note=(
            "ksize must equal the square root of the structural kernel length, "
            "so an expression can only restate it; a disagreeing value is "
            "rejected at execution (TestConvolveKsizeExpression)"
        ),
    ),
    ExprCase(
        "convolve2d",
        "normalize",
        lambda v: gray().convolve2d([1.0] * 9, 3, normalize=v),
        (True, False),
    ),
    ExprCase(
        "convolve2d",
        "border",
        # A 5x5 kernel pads two pixels. At a one-pixel pad "reflect" and
        # "replicate" both reach the edge pixel and coincide, which would make
        # the distinctness assertion unsatisfiable rather than informative.
        lambda v: gray().convolve2d([1.0] * 25, 5, normalize=True, border=v),
        ("replicate", "zero", "reflect"),
    ),
    ExprCase(
        "sharpen",
        "strength",
        lambda v: gray().sharpen(strength=v),
        (0.0, 0.5, 1.5),
    ),
    ExprCase(
        "canny",
        "low_threshold",
        lambda v: gray().canny(low_threshold=v, high_threshold=200.0),
        (10.0, 60.0, 120.0),
        column=RECT,
    ),
    ExprCase(
        "canny",
        "high_threshold",
        lambda v: gray().canny(low_threshold=10.0, high_threshold=v),
        # Above ~120 the noise image has no gradient left to keep, so a third
        # value would duplicate the second rather than add a case.
        (40.0, 120.0),
    ),
    # --- morphology (single-channel kernels) -------------------------------
    ExprCase("erode", "ksize", lambda v: mask().erode(ksize=v), (3, 5), column=RECT),
    ExprCase(
        "erode",
        "iterations",
        lambda v: mask().erode(ksize=3, iterations=v),
        (1, 2, 3),
        column=RECT,
    ),
    ExprCase("dilate", "ksize", lambda v: mask().dilate(ksize=v), (3, 5), column=RECT),
    ExprCase(
        "dilate",
        "iterations",
        lambda v: mask().dilate(ksize=3, iterations=v),
        (1, 2, 3),
        column=RECT,
    ),
    ExprCase(
        "morphology_open",
        "ksize",
        lambda v: mask().morphology_open(ksize=v),
        (3, 5),
        column=RING,
    ),
    ExprCase(
        "morphology_close",
        "ksize",
        lambda v: mask().morphology_close(ksize=v),
        (3, 5),
        column=RING,
    ),
    ExprCase(
        "morphology_gradient",
        "ksize",
        lambda v: mask().morphology_gradient(ksize=v),
        (3, 5),
        column=RING,
    ),
    # --- resizing ----------------------------------------------------------
    ExprCase("resize", "height", lambda v: rgb().resize(height=v, width=8), (4, 8, 12)),
    ExprCase("resize", "width", lambda v: rgb().resize(height=8, width=v), (4, 8, 12)),
    ExprCase(
        "resize",
        "filter",
        lambda v: rgb().resize(height=7, width=7, filter=v),
        ("nearest", "bilinear", "lanczos3"),
    ),
    ExprCase(
        "resize_scale",
        "scale",
        lambda v: rgb().resize_scale(scale=v),
        (0.25, 0.5, 0.75),
    ),
    ExprCase(
        "resize_scale",
        "scale_x",
        lambda v: rgb().resize_scale(scale_x=v, scale_y=0.5),
        (0.25, 0.5, 0.75),
    ),
    ExprCase(
        "resize_scale",
        "scale_y",
        lambda v: rgb().resize_scale(scale_x=0.5, scale_y=v),
        (0.25, 0.5, 0.75),
    ),
    ExprCase(
        "resize_scale",
        "filter",
        lambda v: rgb().resize_scale(scale=0.4, filter=v),
        ("nearest", "bilinear", "lanczos3"),
    ),
    ExprCase(
        "resize_to_height",
        "height",
        lambda v: rgb().resize_to_height(v),
        (4, 8, 12),
    ),
    ExprCase(
        "resize_to_height",
        "filter",
        lambda v: rgb().resize_to_height(7, filter=v),
        ("nearest", "bilinear", "lanczos3"),
    ),
    ExprCase(
        "resize_to_width", "width", lambda v: rgb().resize_to_width(v), (4, 8, 12)
    ),
    ExprCase(
        "resize_to_width",
        "filter",
        lambda v: rgb().resize_to_width(7, filter=v),
        ("nearest", "bilinear", "lanczos3"),
    ),
    ExprCase("resize_max", "max_size", lambda v: rgb().resize_max(v), (4, 8, 12)),
    ExprCase(
        "resize_max",
        "filter",
        lambda v: rgb().resize_max(7, filter=v),
        ("nearest", "bilinear", "lanczos3"),
    ),
    ExprCase("resize_min", "min_size", lambda v: rgb().resize_min(v), (4, 8, 12)),
    ExprCase(
        "resize_min",
        "filter",
        lambda v: rgb().resize_min(7, filter=v),
        ("nearest", "bilinear", "lanczos3"),
    ),
    # --- padding -----------------------------------------------------------
    ExprCase("pad", "top", lambda v: rgb().pad(top=v), (1, 2, 3)),
    ExprCase("pad", "bottom", lambda v: rgb().pad(bottom=v), (1, 2, 3)),
    ExprCase("pad", "left", lambda v: rgb().pad(left=v), (1, 2, 3)),
    ExprCase("pad", "right", lambda v: rgb().pad(right=v), (1, 2, 3)),
    ExprCase(
        "pad",
        "value",
        lambda v: rgb().pad(top=2, value=v),
        (0.0, 64.0, 255.0),
    ),
    ExprCase(
        "pad",
        "mode",
        lambda v: rgb().pad(top=2, left=2, mode=v),
        ("constant", "edge", "reflect", "symmetric"),
    ),
    ExprCase(
        "pad_to_size",
        "height",
        lambda v: rgb().pad_to_size(height=v, width=20),
        (18, 20, 24),
    ),
    ExprCase(
        "pad_to_size",
        "width",
        lambda v: rgb().pad_to_size(height=20, width=v),
        (18, 20, 24),
    ),
    ExprCase(
        "pad_to_size",
        "position",
        lambda v: rgb().pad_to_size(height=22, width=22, position=v),
        ("center", "top-left", "bottom-right"),
    ),
    ExprCase(
        "pad_to_size",
        "value",
        lambda v: rgb().pad_to_size(height=22, width=22, value=v),
        (0.0, 64.0, 255.0),
    ),
    ExprCase(
        "letterbox",
        "height",
        lambda v: rgb().letterbox(height=v, width=12),
        (10, 12, 14),
    ),
    ExprCase(
        "letterbox",
        "width",
        lambda v: rgb().letterbox(height=12, width=v),
        (10, 12, 14),
    ),
    ExprCase(
        "letterbox",
        "value",
        lambda v: rgb().letterbox(height=12, width=20, value=v),
        (0.0, 64.0, 255.0),
    ),
    ExprCase(
        "letterbox",
        "filter",
        lambda v: rgb().letterbox(height=7, width=7, filter=v),
        ("nearest", "bilinear", "lanczos3"),
    ),
    # --- pointwise / geometric --------------------------------------------
    ExprCase("threshold", "value", lambda v: gray().threshold(v), (50, 128, 200)),
    ExprCase("blur", "sigma", lambda v: gray().blur(v), (0.5, 1.5, 3.0)),
    ExprCase("rotate", "angle", lambda v: rgb().rotate(v), (0.0, 30.0, 90.0)),
    ExprCase(
        "rotate",
        "interpolation",
        lambda v: rgb().rotate(30.0, interpolation=v),
        ("nearest", "bilinear"),
    ),
    ExprCase(
        "rotate",
        "border_value",
        lambda v: rgb().rotate(30.0, border_value=v),
        (0.0, 128.0, 255.0),
    ),
    ExprCase(
        "warp_affine",
        "matrix",
        # Only the x-translation moves; the other five stay literal.
        lambda v: rgb().warp_affine([1.0, 0.0, v, 0.0, 1.0, 0.0], (12, 12)),
        (0.0, 2.0, 4.0),
    ),
    ExprCase(
        "warp_affine",
        "output_size",
        lambda v: rgb().warp_affine([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (v, 12)),
        (8, 12, 16),
    ),
    ExprCase(
        "warp_affine",
        "interpolation",
        lambda v: rgb().warp_affine(
            [1.0, 0.3, 0.0, 0.2, 1.0, 0.0], (12, 12), interpolation=v
        ),
        ("nearest", "bilinear"),
    ),
    ExprCase(
        "warp_affine",
        "border_value",
        lambda v: rgb().warp_affine(
            [1.0, 0.0, 4.0, 0.0, 1.0, 4.0], (16, 16), border_value=v
        ),
        (0.0, 128.0, 255.0),
    ),
    ExprCase(
        "shear",
        "sx",
        lambda v: rgb().shear(sx=v, output_size=(16, 16)),
        (0.0, 0.2, 0.4),
    ),
    ExprCase(
        "shear",
        "sy",
        lambda v: rgb().shear(sy=v, output_size=(16, 16)),
        (0.0, 0.2, 0.4),
    ),
    ExprCase(
        "shear",
        "output_size",
        lambda v: rgb().shear(sx=0.2, output_size=(v, 16)),
        (8, 12, 16),
    ),
    ExprCase(
        "rotate_and_scale",
        "angle",
        lambda v: rgb().rotate_and_scale(
            angle=v, center=(8.0, 8.0), output_size=(16, 16)
        ),
        (0.0, 30.0, 60.0),
    ),
    ExprCase(
        "rotate_and_scale",
        "scale",
        lambda v: rgb().rotate_and_scale(
            angle=15.0, scale=v, center=(8.0, 8.0), output_size=(16, 16)
        ),
        (0.5, 1.0, 1.5),
    ),
    ExprCase(
        "rotate_and_scale",
        "center",
        lambda v: rgb().rotate_and_scale(
            angle=30.0, center=(v, 8.0), output_size=(16, 16)
        ),
        (0.0, 4.0, 8.0),
    ),
    ExprCase(
        "rotate_and_scale",
        "output_size",
        lambda v: rgb().rotate_and_scale(
            angle=30.0, center=(8.0, 8.0), output_size=(v, 16)
        ),
        (8, 12, 16),
    ),
    # --- contour ops -------------------------------------------------------
    ExprCase(
        "rasterize",
        "width",
        lambda v: contour().rasterize(width=v, height=12),
        (10, 14, 16),
        column=RECT,
    ),
    ExprCase(
        "rasterize",
        "height",
        lambda v: contour().rasterize(width=12, height=v),
        (10, 14, 16),
        column=RECT,
    ),
    ExprCase(
        "rasterize",
        "fill_value",
        lambda v: contour().rasterize(width=12, height=12, fill_value=v),
        (255, 128, 64),
        column=RECT,
    ),
    ExprCase(
        "rasterize",
        "background",
        lambda v: contour().rasterize(width=12, height=12, background=v),
        (0, 32, 96),
        column=RECT,
    ),
    ExprCase(
        "extract_contours",
        "mode",
        lambda v: mask().extract_contours(mode=v),
        ("external", "all"),
        column=RING,
    ),
    ExprCase(
        "extract_contours",
        "method",
        lambda v: mask().extract_contours(method=v),
        ("simple", "none"),
        column=RECT,
    ),
    ExprCase(
        "extract_contours",
        "min_area",
        lambda v: mask().extract_contours(min_area=v),
        (0.0, 5000.0),
        column=RECT,
    ),
    ExprCase(
        "area",
        "signed",
        lambda v: contour().scale_contour(sx=1.0, sy=-1.0).area(signed=v),
        (True, False),
        column=RECT,
    ),
    ExprCase(
        "translate",
        "dx",
        lambda v: contour().translate(dx=v, dy=0.0),
        (0.0, 5.0, 10.0),
        column=RECT,
    ),
    ExprCase(
        "translate",
        "dy",
        lambda v: contour().translate(dx=0.0, dy=v),
        (0.0, 5.0, 10.0),
        column=RECT,
    ),
    ExprCase(
        "scale_contour",
        "sx",
        lambda v: contour().scale_contour(sx=v, sy=1.0),
        (0.5, 1.0, 2.0),
        column=RECT,
    ),
    ExprCase(
        "scale_contour",
        "sy",
        lambda v: contour().scale_contour(sx=1.0, sy=v),
        (0.5, 1.0, 2.0),
        column=RECT,
    ),
    ExprCase(
        "simplify",
        "tolerance",
        lambda v: contour().simplify(tolerance=v),
        (0.01, 5.0),
        column=RECT,
    ),
    # --- reductions and vectors -------------------------------------------
    ExprCase(
        "reduce_percentile",
        "q",
        lambda v: gray().reduce_percentile(v),
        (0.1, 0.5, 0.9),
    ),
    ExprCase(
        "reduce_std",
        "ddof",
        lambda v: gray().reduce_std(ddof=v),
        (0, 1, 2),
    ),
    ExprCase(
        "histogram",
        "bins",
        lambda v: gray().histogram(bins=v, output="counts"),
        (4, 8, 16),
    ),
    ExprCase(
        "histogram",
        "range",
        lambda v: gray().histogram(bins=8, range=(v, 255.0), output="counts"),
        (0.0, 32.0, 64.0),
    ),
    # --- label reduction ---------------------------------------------------
    ExprCase(
        "label_reduce",
        "contours",
        lambda v: gray().label_reduce(contours=v, reduction="mean"),
        (CONTOUR_SET_A, CONTOUR_SET_B),
        dtype=CONTOUR_SET_SCHEMA,
        # `contours` is an operand column: there is no literal spelling of a
        # contour set, so this case runs every leg of the sweep except the
        # comparison against a literal-built pipeline.
        literal=False,
    ),
    ExprCase(
        "label_reduce",
        "reduction",
        lambda v: gray().label_reduce(contours=pl.col(CONTOURS), reduction=v),
        ("max", "mean", "sum"),
    ),
    ExprCase(
        "label_reduce",
        "region_mode",
        # A diamond is the shape that separates the three modes: for an
        # axis-aligned rectangle the bounding box *is* the interior, and the
        # boundary is already included by the centre-inside rule, so all three
        # score identically and the case would prove nothing.
        lambda v: gray().label_reduce(
            contours=pl.col(DIAMOND), reduction="mean", region_mode=v
        ),
        ("interior", "boundary", "bbox"),
    ),
]


#: Expression-eligible parameters that :data:`CASES` deliberately does not
#: sweep, with the reason. The coverage ratchet accepts a key here *instead of*
#: a case, so an entry is a documented decision rather than a silent gap.
NOT_SWEPT: dict[str, str] = {
    # `to_graph` is graph plumbing, not an operation parameter: `column` names
    # the input the graph reads and is supplied by `.cv.pipe()` for every
    # pipeline the sweep builds.
    "to_graph.column": "the graph's root input column, not an op parameter",
}


def literal_cases() -> list[ExprCase]:
    """Cases whose parameter also has a literal spelling to compare against."""
    return [case for case in CASES if case.literal]


def varying_cases() -> list[ExprCase]:
    """Cases whose row values must produce distinct outputs."""
    return [case for case in CASES if case.varies]


def covered_keys() -> set[str]:
    """Every ``method.parameter`` the table speaks for."""
    return {case.key for case in CASES} | set(NOT_SWEPT)


# --- The authority the ratchet reads --------------------------------------


def expression_eligible_parameters() -> dict[str, str]:
    """Every ``Pipeline`` parameter whose annotation admits a ``pl.Expr``.

    Read off the live signatures rather than a second list, so the ratchet
    tracks the builder. ``LazyPipelineExpr`` annotations are excluded: those
    name another *node* in the graph (``rasterize(shape=)``), not a per-row
    value, and are wired by node id rather than by ``ParamValue``.

    Returns:
        Mapping of ``method.parameter`` to the annotation that qualified it.
    """
    found: dict[str, str] = {}
    for name, method in inspect.getmembers(Pipeline, inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        for param_name, param in signature.parameters.items():
            if param_name == "self" or param.annotation is inspect.Parameter.empty:
                continue
            annotation = str(param.annotation)
            if "Expr" not in annotation or "LazyPipelineExpr" in annotation:
                continue
            found[f"{name}.{param_name}"] = annotation
    return found
