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
    """erode <= x <= dilate and open <= x <= close, at every pixel, for
    either footprint (each holds its centre).

    A NaN in a window makes the extreme NaN (the one ordering rule), so a
    position where either side is NaN orders nothing and is skipped."""
    image = spec.render()
    k = data.draw(st.sampled_from([1, 3, 5, 7]), label="ksize")
    fp = data.draw(st.sampled_from(["square", "disk"]), label="footprint")
    eroded = _run(
        image, [Step("erode", {"ksize": k, "iterations": 1, "footprint": fp})]
    )
    dilated = _run(
        image, [Step("dilate", {"ksize": k, "iterations": 1, "footprint": fp})]
    )
    opened = _run(image, [Step("morphology_open", {"ksize": k, "footprint": fp})])
    closed = _run(image, [Step("morphology_close", {"ksize": k, "footprint": fp})])
    for label, low, high in (
        ("erode <= x", eroded, image),
        ("x <= dilate", image, dilated),
        ("open <= x", opened, image),
        ("x <= close", image, closed),
    ):
        bad = (
            ~(low <= high)
            & ~np.isnan(low.astype(np.float64))
            & ~np.isnan(high.astype(np.float64))
        )
        if bad.any():
            where = tuple(np.argwhere(bad)[0])
            raise ParityFailure(
                f"{label} fails at {where}: {low[where]} > {high[where]}"
            )


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------


@property_lanes(
    spec=image_specs(dtypes=("u8",), max_side=16),
    target=st.sampled_from(tuple(DTYPES)),
)
def test_the_identity_table_is_a_cast(spec, target: str) -> None:
    """``apply_lut([[0], [1], ...], out_dtype=d)`` is ``cast(d)`` wherever the
    value fits the target: every value is its own entry."""
    image = spec.render()
    info = np.iinfo(image.dtype)
    target_dtype = DTYPES[target]
    top = (
        int(info.max)
        if target_dtype.kind == "f"
        else min(int(info.max), int(np.iinfo(target_dtype).max))
    )
    if int(image.max()) > top:
        event("a value past the target's range")
        return
    table = [[i] for i in range(top + 1)]
    _agree(
        f"identity table to {target} vs cast",
        _run(image, [Step("apply_lut", {"table": table, "out_dtype": target})]),
        _run(image, [Step("cast", {"dtype": target})]),
    )


@property_lanes(
    spec=image_specs(dtypes=("u8",), channels=(1, 3), max_side=16),
    seed=st.integers(0, 2**32 - 1),
)
def test_two_tables_compose_into_one(spec, seed: int) -> None:
    """``lut(lut(x, A), B) == lut(x, B[A])``: a table read is a function of
    the value alone."""
    image = spec.render()
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, 256)
    b = rng.integers(-1000, 1000, 256)
    twice = _run(
        image,
        [
            Step("apply_lut", {"table": [[int(v)] for v in a], "out_dtype": "u8"}),
            Step("apply_lut", {"table": [[int(v)] for v in b], "out_dtype": "i16"}),
        ],
    )
    once = _run(
        image,
        [Step("apply_lut", {"table": [[int(v)] for v in b[a]], "out_dtype": "i16"})],
    )
    _agree("lut(lut(x, A), B) vs lut(x, B[A])", twice, once)


#: The grid's symmetries that keep a square pixel lattice: the two flips and
#: the transpose. A Euclidean distance commutes with each exactly.
_GRID_SYMMETRIES = (
    Step("flip", {"axes": [0]}),
    Step("flip", {"axes": [1]}),
    Step("transpose", {"axes": [1, 0, 2]}),
)

_MASKS = image_specs(channels=(1,), max_side=24, contents=("mask",))


@property_lanes(spec=_MASKS, data=st.data())
def test_distance_transform_commutes_with_the_grid_symmetries(
    spec, data: st.DataObject
) -> None:
    """dt(g(x)) == g(dt(x)) for every flip and the transpose, exactly."""
    image = spec.render()
    symmetry = data.draw(st.sampled_from(_GRID_SYMMETRIES), label="symmetry")
    dt = Step(
        "distance_transform", {"signed": data.draw(st.booleans(), label="signed")}
    )
    _agree(
        f"{symmetry.method}{symmetry.params} then dt vs dt then it",
        _run(image, [symmetry, dt]),
        _run(image, [dt, symmetry]),
    )


@property_lanes(
    spec=image_specs(dtypes=("u8",), channels=(1,), max_side=24, contents=("mask",)),
    data=st.data(),
)
def test_disk_morphology_is_the_distance_transform_thresholded(
    spec, data: st.DataObject
) -> None:
    """On a mask, a disk dilation sets exactly the pixels within ``r`` of a set
    pixel, and a disk erosion keeps exactly those more than ``r`` from an unset
    one: the morphology and the distance transform, two kernels, agree."""
    image = spec.render()
    k = data.draw(st.sampled_from([1, 3, 5, 7, 9]), label="ksize")
    r = k // 2
    mask = [Step("threshold", {"value": 0.5})]
    dilated = _run(image, [*mask, Step("dilate", {"ksize": k, "footprint": "disk"})])
    eroded = _run(image, [*mask, Step("erode", {"ksize": k, "footprint": "disk"})])
    # Distance to the nearest set pixel: the transform of the inverted mask.
    to_set = _run(image, [*mask, Step("invert", {}), Step("distance_transform", {})])
    to_unset = _run(image, [*mask, Step("distance_transform", {})])
    _agree("dilate(disk) vs dt(inverted) <= r", dilated != 0, to_set <= r)
    _agree("erode(disk) vs dt > r", eroded != 0, to_unset > r)


@property_lanes(spec=_MASKS, data=st.data())
def test_distance_transform_is_one_lipschitz(spec, data: st.DataObject) -> None:
    """Neighbouring pixels differ by at most their distance apart: 1 across
    an edge, sqrt(2) across a corner. Holds for the signed form too, across
    the boundary (-0.5 to 0.5). Infinite rows (one class only) skip."""
    image = spec.render()
    signed = data.draw(st.booleans(), label="signed")
    out = _run(image, [Step("distance_transform", {"signed": signed})])
    plane = out[:, :, 0].astype(np.float64)
    if not np.isfinite(plane).all():
        event("one class only: infinitely far")
        return
    for label, a, b, bound in (
        ("vertical", plane[1:, :], plane[:-1, :], 1.0),
        ("horizontal", plane[:, 1:], plane[:, :-1], 1.0),
        ("diagonal", plane[1:, 1:], plane[:-1, :-1], np.sqrt(2)),
        ("anti-diagonal", plane[1:, :-1], plane[:-1, 1:], np.sqrt(2)),
    ):
        jump = np.abs(a - b)
        if jump.size and jump.max() > bound + 1e-5:
            where = tuple(np.argwhere(jump > bound + 1e-5)[0])
            raise ParityFailure(f"{label} step {jump[where]} > {bound} at {where}")


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


# ---------------------------------------------------------------------------
# Contour shaping, stroking and set algebra, against the mask they trace
# ---------------------------------------------------------------------------
#
# ``extract_contours`` traces pixel edges, so a traced set has integer
# vertices on axis-aligned edges: it is exactly the union of its pixels'
# unit squares. On such a set each shaping op has a closed form over the
# pixel grid, computed here from the mask alone (never from the engine), and
# no pixel centre (a half-integer point) can tie: a centre's L∞ distance to
# the region is a half-integer, never an integer offset or half-width, and
# its squared Euclidean distance is a half-integer, never ``d²``. The round
# arcs are polygons within 1/64 px for these radii, and a centre is at least
# ``0.25 / d`` from a round boundary, so those are exact too up to ``d = 8``.


def _traced_masks(margin: int):
    """A random region with *margin* empty pixels around it, so a shape grown
    by up to *margin* stays on the canvas and the image border is background.

    Holes are filled (8-connected foreground, 4-connected background, as
    ``extract_contours`` traces): its ``"all"`` mode returns a hole as a flat
    member of its own, which paints, so only a hole-free mask is the union of
    its traced set."""

    @st.composite
    def draw(d: st.DrawFn) -> np.ndarray:
        h = d(sides(1, 14), label="height")
        w = d(sides(1, 14), label="width")
        density = d(st.floats(0.05, 0.95), label="density")
        rng = np.random.default_rng(d(st.integers(0, 2**32 - 1), label="seed"))
        core = rng.random((h, w)) < density
        mask = np.zeros((h + 2 * margin, w + 2 * margin), dtype=bool)
        mask[margin : margin + h, margin : margin + w] = core
        return ndimage.binary_fill_holes(mask)

    return draw()


def _gaps(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis gaps from each pixel centre to each set pixel's unit square:
    ``[H, W, N]`` arrays (``N`` set pixels), ``|Δ| - 1/2`` or 0 on the same row
    or column."""
    ys, xs = np.nonzero(mask)
    gy = np.abs(np.arange(mask.shape[0])[:, None, None] - ys[None, None, :]) - 0.5
    gx = np.abs(np.arange(mask.shape[1])[None, :, None] - xs[None, None, :]) - 0.5
    return np.clip(gy, 0, None), np.clip(gx, 0, None)


def _distance_to(mask: np.ndarray, join: str) -> np.ndarray:
    """Each pixel centre's distance to the union of *mask*'s unit squares:
    L∞ for a ``miter`` join (a square grown by its corners), Euclidean for
    ``round``. Infinite where *mask* is empty."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    gy, gx = _gaps(mask)
    d = np.maximum(gy, gx) if join == "miter" else np.hypot(gy, gx)
    return d.min(axis=2)


def _region_steps(mask: np.ndarray, data: st.DataObject) -> list[Step]:
    """Trace *mask* to its contour set, by either method."""
    method = data.draw(st.sampled_from(["none", "simple"]), label="method")
    return [Step("extract_contours", {"mode": "external", "method": method})]


def _raster(mask: np.ndarray, steps: list[Step], **kw) -> np.ndarray:
    """Run *steps* on *mask* and rasterize onto its own canvas, as booleans.

    The whole chain runs as one query (``_run`` would also run each prefix,
    and ``close`` has no reference entry to say where it ends)."""
    h, w = mask.shape
    image = (mask * np.uint8(255))[:, :, None]
    raster = Step("rasterize", {"width": w, "height": h, **kw})
    (out,) = execute([image], [*steps, raster], Axes(source="array")).rows
    return out.reshape(h, w) != 0


@property_lanes(mask=_traced_masks(0), data=st.data())
def test_a_traced_set_rasterizes_to_its_mask(mask, data) -> None:
    """A traced set rasterizes to the mask it was traced from; the
    shape-preserving ops
    (``union_all``, ``densify``, ``largest`` of every member, ``smooth(0)``)
    and ``supersample`` (full coverage or none, on these edges) leave it so."""
    assume(mask.any())
    traced = _region_steps(mask, data)
    members = len(_run((mask * np.uint8(255))[:, :, None], traced))
    for label, steps, kw in (
        ("traced", traced, {}),
        ("union_all", [*traced, Step("union_all")], {}),
        ("densify", [*traced, Step("densify", {"max_segment_length": 0.5})], {}),
        ("largest(all)", [*traced, Step("largest", {"k": members})], {}),
        ("smooth(0)", [*traced, Step("smooth", {"iterations": 0})], {}),
        ("supersample", traced, {"supersample": data.draw(st.integers(2, 4))}),
    ):
        _agree(label, _raster(mask, steps, **kw), mask)


@property_lanes(mask=_traced_masks(0), data=st.data())
def test_close_undoes_boundary(mask, data) -> None:
    """``boundary()`` draws each ring as a line returning to its start and
    ``close()`` makes each such line the region it bounds: on a hole-free set
    the round trip is the identity, as ``close()``'s docstring states."""
    assume(mask.any())
    steps = [*_region_steps(mask, data), Step("boundary"), Step("close")]
    _agree("close(boundary)", _raster(mask, steps), mask)


@property_lanes(mask=_traced_masks(0), data=st.data())
def test_largest_keeps_a_subset_of_the_members(mask, data) -> None:
    """``largest(k)`` paints within what the whole set paints, and as many
    pixels as its ``k`` largest members do between them."""
    assume(mask.any())
    traced = _region_steps(mask, data)
    k = data.draw(st.integers(1, 3), label="k")
    kept = _raster(mask, [*traced, Step("largest", {"k": k})])
    if (kept & ~mask).any():
        where = tuple(np.argwhere(kept & ~mask)[0])
        raise ParityFailure(f"largest({k}) paints {where}, outside the set")
    areas = sorted(
        (
            abs(a)
            for a in _run((mask * np.uint8(255))[:, :, None], [*traced, Step("area")])
        ),
        reverse=True,
    )
    # Traced members are disjoint, so the kept area is the k largest summed.
    _agree(f"largest({k}) area", np.float64(kept.sum()), np.float64(sum(areas[:k])))


_GROW = 4


@property_lanes(mask=_traced_masks(_GROW + 1), data=st.data())
def test_an_offset_is_the_set_grown_or_shrunk_by_its_distance(mask, data) -> None:
    """``offset(d)`` paints the pixels within ``d`` of the set (``d > 0``) or
    more than ``|d|`` inside it (``d < 0``): Minkowski growth by a square for
    a ``miter`` join on these right angles, by a disc for ``round``."""
    assume(mask.any())
    d = data.draw(st.integers(-_GROW, _GROW).filter(bool), label="distance")
    join = data.draw(st.sampled_from(["miter", "round"]), label="join")
    got = _raster(
        mask,
        [
            *_region_steps(mask, data),
            Step("offset", {"distance": float(d), "join": join}),
        ],
    )
    if d > 0:
        want = _distance_to(mask, join) <= d
    else:
        want = _distance_to(~mask, join) > -d
    _agree(f"offset({d}, join={join})", got, want)


@property_lanes(mask=_traced_masks(_GROW + 1), data=st.data())
def test_a_stroke_is_the_band_round_the_sets_rings(mask, data) -> None:
    """``stroke(width=2k)`` of a region set paints the pixels within ``k`` of
    its rings (each closed: no cap applies, whichever is asked for) — those
    within ``k`` of the set outside it, and within ``k`` of the background
    inside it."""
    assume(mask.any())
    k = data.draw(st.integers(1, _GROW), label="half width")
    join = data.draw(st.sampled_from(["miter", "round"]), label="join")
    cap = data.draw(st.sampled_from(["round", "square", "butt"]), label="cap")
    stroke = Step("stroke", {"width": 2.0 * k, "join": join, "cap": cap})
    got = _raster(mask, [*_region_steps(mask, data), stroke])
    to_ring = np.where(mask, _distance_to(~mask, join), _distance_to(mask, join))
    _agree(f"stroke(width={2 * k}, join={join}, cap={cap})", got, to_ring <= k)


_SET_ALGEBRA = {
    "union": np.logical_or,
    "intersection": np.logical_and,
    "difference": lambda a, b: a & ~b,
    "symmetric_difference": np.logical_xor,
}


@property_lanes(
    left=_traced_masks(0),
    seed=st.integers(0, 2**32 - 1),
    density=st.floats(0.05, 0.95),
    data=st.data(),
)
def test_a_set_operation_is_its_mask_operation(left, seed, density, data) -> None:
    """``a op b`` paints exactly ``mask(a) op mask(b)`` (``|``, ``&``, ``& ~``,
    ``^``), for two traced sets, by the pipeline op."""
    from tests.parity.framework.run import BinaryCase, execute_binary

    right = ndimage.binary_fill_holes(
        np.random.default_rng(seed).random(left.shape) < density
    )
    op = data.draw(st.sampled_from(sorted(_SET_ALGEBRA)), label="op")
    h, w = left.shape
    trace = tuple(_region_steps(left, data))
    case = BinaryCase(
        left=((left * np.uint8(255))[:, :, None],),
        right=((right * np.uint8(255))[:, :, None],),
        left_steps=trace,
        right_steps=trace,
        method=op,
        tail=(Step("rasterize", {"width": w, "height": h}),),
    )
    (got,) = execute_binary(case).rows
    _agree(f"{op} painted", got.reshape(h, w) != 0, _SET_ALGEBRA[op](left, right))


@st.composite
def _walks(draw: st.DrawFn, margin: int) -> list[tuple[int, int]]:
    """An open, axis-aligned polyline on integer points, at least *margin*
    from the canvas edge: steps of 1-6 px that never double straight back,
    ending away from its start (a line returning to its start is a loop,
    stroked as a closed path, which the ring law holds)."""
    size = 12
    x, y = draw(st.integers(0, size), label="x0"), draw(st.integers(0, size))
    points = [(x, y)]
    direction = None
    for _ in range(draw(st.integers(1, 6), label="segments")):
        choices = [d for d in ((1, 0), (-1, 0), (0, 1), (0, -1)) if d != direction]
        dx, dy = draw(st.sampled_from(choices))
        n = draw(st.integers(1, 6))
        x, y = min(max(x + dx * n, 0), size), min(max(y + dy * n, 0), size)
        if (x, y) != points[-1]:
            direction = (-dx, -dy)
            points.append((x, y))
    assume(len(points) > 1 and points[0] != points[-1])
    return [(px + margin, py + margin) for px, py in points]


def _to_polyline(shape: tuple[int, int], points, metric: str) -> np.ndarray:
    """Each pixel centre's distance to the polyline through *points*:
    Euclidean, or L∞ (exact here, every segment being axis-aligned)."""
    h, w = shape
    cy, cx = np.mgrid[0:h, 0:w] + 0.5
    best = np.full(shape, np.inf)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        # The nearest point of an axis-aligned segment: clamp per axis.
        nx = np.clip(cx, min(x0, x1), max(x0, x1))
        ny = np.clip(cy, min(y0, y1), max(y0, y1))
        dx, dy = np.abs(cx - nx), np.abs(cy - ny)
        d = np.hypot(dx, dy) if metric == "euclidean" else np.maximum(dx, dy)
        best = np.minimum(best, d)
    return best


@property_lanes(data=st.data())
def test_a_stroked_line_is_the_band_round_it(data) -> None:
    """``stroke(width=2k)`` of an open line paints the pixels within ``k`` of
    it: Euclidean with round caps and joins, L∞ with square caps and miter
    joins (the line grown by a square, on these right angles)."""
    from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

    k = data.draw(st.integers(1, _GROW), label="half width")
    points = data.draw(_walks(k + 1), label="line")
    cap, join, metric = data.draw(
        st.sampled_from([("round", "round", "euclidean"), ("square", "miter", "linf")]),
        label="style",
    )
    side = 12 + 2 * (k + 1) + 1
    line = {
        "exterior": [{"x": float(x), "y": float(y)} for x, y in points],
        "holes": [],
        "is_closed": False,
    }
    frame = pl.DataFrame({"c": [[line]]}, schema={"c": CONTOUR_SET_SCHEMA})
    pipe = (
        Pipeline()
        .source("contour")
        .stroke(width=2.0 * k, cap=cap, join=join)
        .rasterize(width=side, height=side)
    )
    out = frame.select(out=pl.col("c").cv.pipe(pipe).sink("numpy"))
    got = numpy_from_struct(out["out"][0]).reshape(side, side) != 0
    want = _to_polyline((side, side), points, metric) <= k
    _agree(f"stroke(width={2 * k}, cap={cap}, join={join}) of {points}", got, want)
