"""Algebraic laws: relations between ops that hold whatever the reference.

A reference comparison needs a second implementation; a law needs only the
engine, and states something no single-op check can: two chains that must
agree, an op that must undo another, an ordering that must hold pointwise.
Every law here is exact — they are chosen so that no rounding enters.

These absorb three of the four properties that used to live in
``tests/property/test_shape_and_param_invariants.py`` (double flip, double
transpose, grayscale's channel count) and extend them across dtypes, channel
counts and sources. The fourth, a literal and a column argument agreeing, is
now the ``params`` axis of ``invariance/test_parity_execution.py`` and the
per-row check in ``invariance/test_parity_rows.py``, for every op.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import assume, event, reject
from hypothesis import strategies as st
from scipy import ndimage

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required
from tests.parity.framework import known
from tests.parity.framework.budget import property_lanes
from tests.parity.framework.cases import lossless_sources
from tests.parity.framework.checks import ParityFailure, same_output
from tests.parity.framework.images import DTYPES, image_specs, sides
from tests.parity.framework.oracle import OPS
from tests.parity.framework.run import Axes, Step, execute

pytestmark = plugin_required


def _skip(divergence: known.Divergence) -> None:
    event(f"known divergence: {divergence.key}")
    reject()


def _run(image: np.ndarray, steps: list[Step], source: str = "array") -> np.ndarray:
    """Execute *steps*; raise :class:`KnownDivergence` if one would be hit.

    Laws compose ops freely, so they consult the registry at every step: a
    registered defect is pinned by its own strict xfail, and a law is not the
    place to rediscover it. Such an example is rejected (and counted).
    """
    axes = Axes(source=source)
    divergence = known.axes_divergence(axes, [image], [], steps)
    if divergence is not None:
        _skip(divergence)
    current = image
    for i, step in enumerate(steps):
        divergence = known.chain_divergence(steps[: i + 1])
        if divergence is None and isinstance(current, np.ndarray):
            divergence = known.step_divergence(step, current)
        if divergence is not None:
            _skip(divergence)
        domain = OPS[step.method].domain_out
        prefix_axes = axes if domain == "buffer" else axes.but(sink="native")
        current = execute([image], steps[: i + 1], prefix_axes).rows[0]
    return current if steps else execute([image], [], axes).rows[0]


def _agree(label: str, a: np.ndarray, b: np.ndarray) -> None:
    difference = same_output(a, b)
    if difference is not None:
        raise ParityFailure(f"{label}: {difference}")


_ANY_IMAGE = image_specs(max_side=24)


# ---------------------------------------------------------------------------
# Inverses
# ---------------------------------------------------------------------------


@property_lanes(spec=_ANY_IMAGE, data=st.data())
def test_involutions_are_the_identity(spec, data: st.DataObject) -> None:
    """flip twice, transpose then its inverse, four quarter turns: identity."""
    image = spec.render()
    source = data.draw(st.sampled_from(lossless_sources([image])), label="source")
    axes = data.draw(
        st.lists(st.integers(0, 2), min_size=1, max_size=3, unique=True),
        label="flip axes",
    )
    perm = data.draw(st.permutations([0, 1, 2]), label="perm")
    inverse = list(np.argsort(perm))
    for label, steps in (
        ("flip twice", [Step("flip", {"axes": axes})] * 2),
        (
            "transpose and back",
            [Step("transpose", {"axes": perm}), Step("transpose", {"axes": inverse})],
        ),
        ("four quarter turns", [Step("rotate", {"angle": 90.0})] * 4),
        ("two half turns", [Step("rotate", {"angle": 180.0})] * 2),
    ):
        _agree(label, _run(image, steps, source), image)


@property_lanes(
    spec=image_specs(
        dtypes=("u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64"), max_side=24
    )
)
def test_integer_invert_is_an_involution(spec) -> None:
    """``MAX + MIN - (MAX + MIN - x) == x`` exactly on every integer dtype."""
    image = spec.render()
    _agree("invert twice", _run(image, [Step("invert")] * 2), image)


#: Widening round trips: every value of the narrow dtype survives the wide one.
_WIDENINGS = {
    "u8": ("u16", "i16", "u32", "i32", "u64", "i64", "f32", "f64"),
    "i8": ("i16", "i32", "i64", "f32", "f64"),
    "u16": ("u32", "i32", "u64", "i64", "f32", "f64"),
    "i16": ("i32", "i64", "f32", "f64"),
    "u32": ("u64", "i64", "f64"),
    "i32": ("i64", "f64"),
    "f32": ("f64",),
}


@property_lanes(spec=image_specs(dtypes=tuple(_WIDENINGS), max_side=16), data=st.data())
def test_widening_cast_round_trips(spec, data: st.DataObject) -> None:
    """narrow -> wide -> narrow is the identity for every lossless widening."""
    image = spec.render()
    wide = data.draw(st.sampled_from(_WIDENINGS[spec.dtype]), label="wide")
    steps = [Step("cast", {"dtype": wide}), Step("cast", {"dtype": spec.dtype})]
    _agree(f"{spec.dtype} -> {wide} -> {spec.dtype}", _run(image, steps), image)


@property_lanes(spec=_ANY_IMAGE, data=st.data())
def test_pad_then_crop_is_the_identity(spec, data: st.DataObject) -> None:
    """Padding by any mode and cropping the padding off returns the input."""
    image = spec.render()
    h, w = image.shape[:2]
    mode = data.draw(
        st.sampled_from(["constant", "edge", "reflect", "symmetric"]), label="mode"
    )
    limit_h = {"reflect": h - 1, "symmetric": h}.get(mode, 5)
    limit_w = {"reflect": w - 1, "symmetric": w}.get(mode, 5)
    top, bottom = (data.draw(st.integers(0, min(5, limit_h))) for _ in range(2))
    left, right = (data.draw(st.integers(0, min(5, limit_w))) for _ in range(2))
    steps = [
        Step(
            "pad",
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "value": 0.0,
                "mode": mode,
            },
        ),
        Step("crop", {"top": top, "left": left, "height": h, "width": w}),
    ]
    _agree(f"pad({mode}) then crop", _run(image, steps), image)


def _mergeable_dtypes() -> tuple[str, ...]:
    """The dtypes no registered divergence rules out for channel_merge —
    drawn from the registry, so a fix widens this law automatically."""
    return tuple(
        name
        for name, dtype in DTYPES.items()
        if known.step_divergence(Step("channel_merge"), np.zeros((1, 1, 2), dtype))
        is None
    )


@property_lanes(
    spec=image_specs(dtypes=_mergeable_dtypes(), channels=(2, 3, 4), max_side=16)
)
def test_channel_select_then_merge_is_the_identity(spec) -> None:
    """Splitting an image into its channels and merging them back."""
    image = spec.render()
    frame = pl.DataFrame(
        {"img": [image]},
        schema={"img": pl.Array(_polars(spec.dtype), image.shape)},
    )
    source = Pipeline().source("array", dtype=spec.dtype)
    planes = [
        pl.col("img").cv.pipe(source.channel_select(c)) for c in range(image.shape[2])
    ]
    merged = planes[0].channel_merge(*planes[1:]).sink("numpy")
    out = numpy_from_struct(frame.select(out=merged)["out"][0])
    _agree("channel_select each, channel_merge back", out, image)


def _polars(dtype: str) -> pl.DataType:
    from tests.parity.framework.io import POLARS_DTYPES

    return POLARS_DTYPES[dtype]


# ---------------------------------------------------------------------------
# Commutation
# ---------------------------------------------------------------------------


@property_lanes(spec=_ANY_IMAGE, data=st.data())
def test_flip_commutes_with_a_mirrored_crop(spec, data: st.DataObject) -> None:
    """crop then flip == flip then the mirror-image crop."""
    image = spec.render()
    h, w = image.shape[:2]
    top = data.draw(st.integers(0, h - 1))
    left = data.draw(st.integers(0, w - 1))
    height = data.draw(st.integers(1, h - top))
    width = data.draw(st.integers(1, w - left))
    crop = Step("crop", {"top": top, "left": left, "height": height, "width": width})
    mirrored = Step(
        "crop",
        {
            "top": h - top - height,
            "left": w - left - width,
            "height": height,
            "width": width,
        },
    )
    flip = Step("flip", {"axes": [0, 1]})
    _agree(
        "crop.flip vs flip.mirrored crop",
        _run(image, [crop, flip]),
        _run(image, [flip, mirrored]),
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


@property_lanes(spec=_ANY_IMAGE, data=st.data())
def test_integer_nearest_upscale_repeats_pixels(spec, data: st.DataObject) -> None:
    """A nearest resize by whole factors is ``np.repeat`` (no ties arise)."""
    image = spec.render()
    fy = data.draw(st.integers(1, 3), label="fy")
    fx = data.draw(st.integers(1, 3), label="fx")
    h, w = image.shape[:2]
    step = Step("resize", {"height": h * fy, "width": w * fx, "filter": "nearest"})
    expected = np.repeat(np.repeat(image, fy, axis=0), fx, axis=1)
    _agree(f"nearest x({fy}, {fx})", _run(image, [step]), expected)


@property_lanes(spec=image_specs(dtypes=("u8",), max_side=24), data=st.data())
def test_resize_to_the_same_size_is_the_identity(spec, data: st.DataObject) -> None:
    """Every filter samples at the pixel centres when the grid is unchanged."""
    image = spec.render()
    h, w = image.shape[:2]
    flt = data.draw(
        st.sampled_from(["nearest", "bilinear", "catmullrom", "gaussian", "lanczos3"]),
        label="filter",
    )
    step = Step("resize", {"height": h, "width": w, "filter": flt})
    _agree(f"resize({flt}) to the same size", _run(image, [step]), image)


@property_lanes(
    # No alpha: an alpha image is resampled premultiplied in 8 bits, so a
    # flat colour under a nearly transparent alpha legitimately moves.
    height=sides(1, 16),
    width=sides(1, 16),
    channels=st.sampled_from([1, 3]),
    value=st.integers(0, 255),
    data=st.data(),
)
def test_smoothing_a_constant_image_leaves_it_constant(
    height, width, channels, value, data
) -> None:
    """Blur and resampling have weights summing to one: a flat image stays flat."""
    image = np.full((height, width, channels), value, dtype=np.uint8)
    step = data.draw(
        st.one_of(
            st.sampled_from([0.5, 1.0, 2.0, 3.0]).map(
                lambda s: Step("blur", {"sigma": s})
            ),
            st.tuples(
                st.integers(1, 24),
                st.integers(1, 24),
                st.sampled_from(
                    ["nearest", "bilinear", "catmullrom", "gaussian", "lanczos3"]
                ),
            ).map(
                lambda t: Step(
                    "resize", {"height": t[0], "width": t[1], "filter": t[2]}
                )
            ),
        ),
        label="step",
    )
    out = _run(image, [step])
    _agree(
        f"{step!r} of a constant image", out, np.full(out.shape, value, dtype=np.uint8)
    )


# ---------------------------------------------------------------------------
# Morphology and masks
# ---------------------------------------------------------------------------


@property_lanes(spec=image_specs(channels=(1,), max_side=24), data=st.data())
def test_morphology_orders_pointwise(spec, data: st.DataObject) -> None:
    """erode <= x <= dilate and open <= x <= close, at every pixel."""
    image = spec.render()
    k = data.draw(st.sampled_from([1, 3, 5]), label="ksize")
    eroded = _run(image, [Step("erode", {"ksize": k, "iterations": 1})])
    dilated = _run(image, [Step("dilate", {"ksize": k, "iterations": 1})])
    opened = _run(image, [Step("morphology_open", {"ksize": k})])
    closed = _run(image, [Step("morphology_close", {"ksize": k})])
    for label, low, high in (
        ("erode <= x", eroded, image),
        ("x <= dilate", image, dilated),
        ("open <= x", opened, image),
        ("x <= close", image, closed),
    ):
        if not np.all(low <= high):
            where = tuple(np.argwhere(~(low <= high))[0])
            raise ParityFailure(
                f"{label} fails at {where}: {low[where]} > {high[where]}"
            )


@property_lanes(
    height=sides(1, 20),
    width=sides(1, 20),
    density=st.floats(0.05, 0.95),
    seed=st.integers(0, 2**32 - 1),
    data=st.data(),
)
def test_extracted_contours_rasterize_back_to_the_mask(
    height, width, density, seed, data
) -> None:
    """mask -> extract_contours(external) -> rasterize == the mask, holes filled.

    Documented as lossless ("Rasterizing the result gives back the mask"); the
    external borders drop holes, which ``binary_fill_holes`` fills with the
    matching connectivity (8-connected foreground, 4-connected background).
    """
    rng = np.random.default_rng(seed)
    mask = (rng.random((height, width)) < density).astype(np.uint8) * 255
    assume(mask.any())
    method = data.draw(st.sampled_from(["none", "simple"]), label="method")
    steps = [
        Step("extract_contours", {"mode": "external", "method": method}),
        Step("rasterize", {"width": width, "height": height}),
    ]
    out = _run(mask[:, :, None], steps)
    expected = ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255
    _agree("contours rasterized", out.reshape(height, width), expected)


@property_lanes(
    spec=image_specs(dtypes=("u8",), channels=(1,), max_side=24), data=st.data()
)
def test_threshold_is_idempotent_on_a_mask(spec, data: st.DataObject) -> None:
    """A 0/255 mask thresholded anywhere in (0, 255) is itself."""
    mask = _run(spec.render(), [Step("threshold", {"value": 127.5})])
    again = _run(
        mask,
        [Step("threshold", {"value": float(data.draw(st.integers(0, 254))) + 0.5})],
    )
    _agree("threshold of a mask", again, mask)


# ---------------------------------------------------------------------------
# Shape facts (the former tests/property suite)
# ---------------------------------------------------------------------------


@property_lanes(spec=image_specs(channels=(1, 2, 3, 4), max_side=24))
def test_grayscale_yields_one_channel(spec) -> None:
    """grayscale is SingleChannel: ``[H, W, C] -> [H, W, 1]`` for every C."""
    image = spec.render()
    out = _run(image, [Step("grayscale")])
    if out.shape != (*image.shape[:2], 1):
        raise ParityFailure(f"grayscale of {image.shape} gave {out.shape}")
    if out.dtype != image.dtype:
        raise ParityFailure(f"grayscale of {image.dtype} gave {out.dtype}")
