"""The one table of "how do you call each chainable ``Pipeline`` op".

This started life inside ``test_append_contract.py``, where it drove the
eager-vs-lazy plan parity sweep. It lives here because the schema-parity
matrix needs the same axis, and a second copy of "every op and its arguments"
is exactly the hand-maintained list that goes stale the day an op is added.

The table is completeness-asserted against ``polars_cv.lazy.
_chainable_pipeline_ops()`` by ``test_op_case_table_is_complete`` in
``test_append_contract.py``, so a new operation cannot join the library
without also getting a case here — and therefore a plan-vs-exec cell.

``None`` marks an op this harness cannot call, with the reason. Everything
else is ``(domain, kwargs)``, where *domain* selects which base pipeline the
op is appended to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polars_cv import Pipeline

BUFFER = "buffer"
CONTOUR = "contour"

OP_CASES: dict[str, tuple[str, dict] | None] = {
    # --- buffer -> buffer -------------------------------------------------
    "adjust_brightness": (BUFFER, {"factor": 1.2}),
    "adjust_contrast": (BUFFER, {"factor": 1.2}),
    "adjust_gamma": (BUFFER, {"gamma": 1.5}),
    "blur": (BUFFER, {"sigma": 1.0}),
    "canny": (BUFFER, {}),
    "cast": (BUFFER, {"dtype": "f32"}),
    "channel_select": (BUFFER, {"index": 0}),
    "channel_swap": (BUFFER, {"order": [2, 1, 0]}),
    "clamp": (BUFFER, {"min_val": 0.0, "max_val": 1.0}),
    "convert_color": (BUFFER, {"from_space": "rgb", "to_space": "hsv"}),
    "convolve2d": (BUFFER, {"kernel": [0.0] * 9, "ksize": 3}),
    "crop": (BUFFER, {"top": 0, "left": 0, "height": 50, "width": 50}),
    "dilate": (BUFFER, {"ksize": 3}),
    "equalize_histogram": (BUFFER, {}),
    "erode": (BUFFER, {"ksize": 3}),
    "flip": (BUFFER, {"axes": [0]}),
    "flip_h": (BUFFER, {}),
    "flip_v": (BUFFER, {}),
    "grayscale": (BUFFER, {}),
    "invert": (BUFFER, {}),
    "laplacian": (BUFFER, {}),
    "letterbox": (BUFFER, {"height": 128, "width": 128}),
    "morphology_close": (BUFFER, {"ksize": 3}),
    "morphology_gradient": (BUFFER, {"ksize": 3}),
    "morphology_open": (BUFFER, {"ksize": 3}),
    "normalize": (BUFFER, {"method": "minmax"}),
    "pad": (BUFFER, {"top": 10, "bottom": 10, "left": 0, "right": 0}),
    "pad_to_size": (BUFFER, {"height": 150, "width": 250}),
    "relu": (BUFFER, {}),
    "reshape": (BUFFER, {"shape": [100, 200, 3]}),
    "resize": (BUFFER, {"height": 64, "width": 32}),
    "resize_max": (BUFFER, {"max_size": 120}),
    "resize_min": (BUFFER, {"min_size": 40}),
    "resize_scale": (BUFFER, {"scale_x": 0.5, "scale_y": 0.5}),
    "resize_to_height": (BUFFER, {"height": 50}),
    "resize_to_width": (BUFFER, {"width": 50}),
    "rotate": (BUFFER, {"angle": 90}),
    "rotate_and_scale": (
        BUFFER,
        {"angle": 45.0, "scale": 1.5, "center": (50.0, 100.0), "output_size": (64, 64)},
    ),
    "scale": (BUFFER, {"factor": 2.0}),
    "sharpen": (BUFFER, {}),
    "shear": (BUFFER, {"sx": 0.2, "output_size": (100, 200)}),
    "sobel": (BUFFER, {}),
    "threshold": (BUFFER, {"value": 128}),
    "to_bgr": (BUFFER, {}),
    "to_hsv": (BUFFER, {}),
    "to_lab": (BUFFER, {}),
    "to_ycbcr": (BUFFER, {}),
    "transpose": (BUFFER, {"axes": [1, 0, 2]}),
    "warp_affine": (
        BUFFER,
        {"matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "output_size": (64, 64)},
    ),
    # --- buffer -> other domains -----------------------------------------
    "extract_contours": (BUFFER, {}),
    "extract_shape": (BUFFER, {}),
    "histogram": (BUFFER, {"bins": 8}),
    "perceptual_hash": (BUFFER, {}),
    "reduce_argmax": (BUFFER, {"axis": 0}),
    "reduce_argmin": (BUFFER, {"axis": 0}),
    "reduce_max": (BUFFER, {}),
    "reduce_mean": (BUFFER, {}),
    "reduce_min": (BUFFER, {}),
    "reduce_percentile": (BUFFER, {"q": 50.0}),
    "reduce_popcount": (BUFFER, {}),
    "reduce_std": (BUFFER, {}),
    "reduce_sum": (BUFFER, {}),
    # --- contour domain ---------------------------------------------------
    "area": (CONTOUR, {}),
    "bounding_box": (CONTOUR, {}),
    "centroid": (CONTOUR, {}),
    "convex_hull": (CONTOUR, {}),
    "perimeter": (CONTOUR, {}),
    "rasterize": (CONTOUR, {"width": 32, "height": 32}),
    "scale_contour": (CONTOUR, {"sx": 2.0, "sy": 2.0}),
    "simplify": (CONTOUR, {"tolerance": 1.0}),
    "translate": (CONTOUR, {"dx": 1.0, "dy": 2.0}),
    # --- not comparable by this harness -----------------------------------
    "assert_shape": None,  # sets hints directly; covered by its own test
    "label_reduce": None,  # takes a contour *column*, not a plain value
    "on_error": None,  # graph-level policy, appends no op
    "on_null_param": None,  # graph-level policy, appends no op
}

#: Extra parameterisations for ops whose interesting behaviour is in a branch
#: the single case in ``OP_CASES`` does not reach. Kept separate so the
#: completeness assertion stays a strict one-case-per-op check.
EXTRA_CASES: list[tuple[str, str, dict]] = [
    # rotate's zero-copy 90/180/270 fast path vs the affine path vs expand.
    ("rotate", BUFFER, {"angle": 45.0}),
    ("rotate", BUFFER, {"angle": 45.0, "expand": True}),
    ("rotate", BUFFER, {"angle": 180}),
    # Axis reductions drop a dimension; the no-axis form does not.
    ("reduce_max", BUFFER, {"axis": 0}),
    ("reduce_min", BUFFER, {"axis": 1}),
    ("reduce_mean", BUFFER, {"axis": 2}),
    ("reduce_std", BUFFER, {"axis": 0}),
    # Non-square and rank-changing reshapes.
    ("reshape", BUFFER, {"shape": [200, 100, 3]}),
    ("reshape", BUFFER, {"shape": [60000]}),
    # crop with only an offset, and histogram's struct-encoded output.
    ("crop", BUFFER, {"top": 10, "left": 20}),
    ("histogram", BUFFER, {"bins": 8, "output": "buckets"}),
    # The float-promoting scalar ops asked to land somewhere other than the
    # promoted float. `out_dtype` was silently discarded — accepted, serialized
    # into the op's identity, and read by no `resolve_op` arm and no dtype rule
    # — so neither op had a plan-vs-exec cell for it, and planner and execution
    # agreed on the *wrong* answer.
    #
    # Only the `out_dtype` spelling appears here. `preserve_dtype=True` is the
    # same mechanism with the target read off the pipeline, so it requires a
    # concrete input dtype and cannot run against the `image_bytes` base these
    # tables share (`auto` until the source declares one). It is covered on a
    # typed source by `test_preserve_dtype.py` and by
    # `TestScalarOpOutDtypeIsHonored` in `test_dtype_contracts.py`, which
    # asserts the two spellings reach the same column.
    ("scale", BUFFER, {"factor": 2.0, "out_dtype": "u8"}),
    ("scale", BUFFER, {"factor": 2.0, "out_dtype": "f64"}),
    ("clamp", BUFFER, {"min_val": 0.0, "max_val": 1.0, "out_dtype": "u8"}),
    ("clamp", BUFFER, {"min_val": 0.0, "max_val": 1.0, "out_dtype": "i32"}),
]

#: The five ``histogram(output=...)`` modes. Each lands in a different
#: (rank, dtype, domain) corner — ``counts`` is ForceU64 rank 1, ``quantized``
#: stays a ForceU32 buffer, ``buckets`` is a rank-2 struct-encoded vector —
#: and none of them had a plan-vs-exec test before this matrix.
HISTOGRAM_OUTPUTS: tuple[str, ...] = (
    "counts",
    "normalized",
    "edges",
    "buckets",
    "quantized",
)

#: Every colour space ``convert_color`` can target, for the
#: ``StripProcessRestore`` channel rule.
COLOR_SPACES: tuple[str, ...] = ("rgb", "bgr", "hsv", "lab", "ycbcr", "gray")

#: Ops whose engine kernel requires a single-channel buffer.
#:
#: Nothing rejects a three-channel pipeline for these at build or plan time —
#: the failure is a runtime "Erode requires single-channel input, but got 3
#: channels". The table's argument sets were only ever exercised by the
#: eager/lazy *plan* parity sweep, which never executes, so this precondition
#: went unnoticed until the schema matrix started collecting. Sweeps that
#: execute must put a ``grayscale()`` in front of these.
SINGLE_CHANNEL_OPS: frozenset[str] = frozenset(
    {
        "threshold",
        "erode",
        "dilate",
        "morphology_open",
        "morphology_close",
        "morphology_gradient",
    }
)

#: The sinks that re-encode a buffer through an image codec. Each has a dtype
#: precondition (jpeg/webp are 8-bit only; png takes u8/u16; tiff rejects some
#: dtype/channel combinations) that is checked when the encoder runs, not when
#: the query is planned — see ``test_schema_parity_encoder_sinks.py``.
IMAGE_ENCODER_SINKS: tuple[str, ...] = ("png", "jpeg", "webp", "tiff")


def buffer_ops() -> list[str]:
    """Op names whose case runs on a buffer-domain pipeline."""
    return sorted(
        name
        for name, case in OP_CASES.items()
        if case is not None and case[0] == BUFFER
    )


def contour_ops() -> list[str]:
    """Op names whose case runs on a contour-domain pipeline."""
    return sorted(
        name
        for name, case in OP_CASES.items()
        if case is not None and case[0] == CONTOUR
    )


def comparable_ops() -> list[str]:
    """Every op with a callable case, in a stable order."""
    return sorted(name for name, case in OP_CASES.items() if case is not None)


def base_pipeline(domain: str) -> "Pipeline":
    """A pipeline in *domain* with fully known, non-square shape hints.

    ``assert_shape`` records an assertion and sets hints without appending to
    ``_ops``, so the op under test is still ``_ops[-1]`` and the domain fold
    still sees only real operations.
    """
    from polars_cv import Pipeline

    pipe = (
        Pipeline().source("image_bytes").assert_shape(height=100, width=200, channels=3)
    )
    if domain == CONTOUR:
        return pipe.grayscale().threshold(128).extract_contours()
    return pipe


def build_case(op: str) -> "Pipeline":
    """The pipeline for *op*'s case, with that op last.

    The one way to turn a row of this table into a pipeline. Guards that want
    "every op, called somehow" read it from here rather than keeping their own
    op → builder map: ``test_sanitation.py`` kept one covering 22 of the ~90
    ops, so the other ~70 never had their domain or rank/channel rule checked
    against the Rust contract at all.
    """
    case = OP_CASES[op]
    if case is None:
        msg = f"{op} has no callable case"
        raise ValueError(msg)
    domain, kwargs = case
    return getattr(base_pipeline(domain), op)(**kwargs)
