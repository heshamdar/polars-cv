"""Verified divergences this suite has found and not yet fixed.

The parity sweeps are generative, so a known bug cannot simply be left
failing: every run would rediscover it, and the per-push lane would be red
until it is fixed. Nor can it be absorbed into a tolerance or dropped from the
generator quietly — that is how a sweep ends up covering less than it claims.

So each divergence is recorded here once, with:

* a **predicate** saying which cases it covers, which the checks consult to
  *skip that one comparison* (reported through ``hypothesis.event``, so the
  statistics show how often it fires). The case is still generated and still
  executed along every other axis; only the comparison the bug breaks is
  withheld.
* a **repro**: the minimal case, asserting the *correct* behaviour. It runs as
  a strict ``xfail`` in ``oracle/test_parity_known_divergences.py``, so the day the
  bug is fixed it XPASSes, the suite goes red, and the entry (predicate and
  all) has to be deleted — which puts the fixed path back under the sweeps.

An entry must be a *confirmed* defect: reproduced against the running engine
with an independent decoder or reference. Suspicions do not go here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from tests.parity.framework.run import Axes, Step


@dataclass(frozen=True)
class Divergence:
    """One confirmed, unfixed defect.

    Attributes:
        key: Stable identifier (used as the xfail test id).
        summary: What is wrong, and what "fixed" looks like.
        affects_axes: ``(axes, images, output_shapes, steps) -> bool`` for a
            defect of an execution axis (a source, sink, frame layout,
            parameter style), possibly only for some ops.
        affects_step: ``(step, input) -> bool`` for a defect of one op on one
            kind of input.
        affects_chain: ``(steps) -> bool`` for a defect that depends on what
            came before: called with the chain up to and including the step
            about to be appended.
        repro: Runs the minimal case and asserts the correct behaviour.
        avoid: Whether the defect cannot be carried forward: it raises, or
            it leaves a state no later step is meant to handle (a zero-sized
            image). Such a case is never executed or appended to a chain; the
            checks skip it and count it with ``event``.
        raises: The exception the repro fails with today: ``AssertionError``
            for a wrong answer, the engine's error type for one it refuses.
        match: A fragment of that exception's message. Together with
            *raises* it pins the failure to the defect, so a repro broken for
            another reason (a renamed helper, a changed fixture) is reported
            rather than read as "still reproduces" (:func:`still_reproduces`).
    """

    key: str
    summary: str
    repro: Callable[[], None]
    affects_axes: (
        Callable[[Axes, Sequence[Any], Sequence[tuple[int, ...]], Sequence[Step]], bool]
        | None
    ) = None
    affects_step: Callable[[Step, np.ndarray], bool] | None = None
    affects_chain: Callable[[Sequence[Step]], bool] | None = None
    avoid: bool = False
    raises: type[BaseException] = AssertionError
    match: str = ""


def still_reproduces(divergence: Divergence) -> None:
    """Run *divergence*'s repro and require it to fail *for its defect*.

    Raises ``AssertionError`` when the repro passes (the defect is fixed:
    delete the entry, which puts the path back under the sweeps) or fails
    some other way (the repro is broken, not the engine).
    """
    try:
        divergence.repro()
    except divergence.raises as exc:
        if divergence.match not in str(exc):
            msg = (
                f"{divergence.key}: the repro raised {type(exc).__name__} but "
                f"not with {divergence.match!r}: {str(exc)[:300]}"
            )
            raise AssertionError(msg) from exc
        return
    except Exception as exc:
        msg = (
            f"{divergence.key}: the repro failed with {type(exc).__name__}, not "
            f"{divergence.raises.__name__}: {str(exc)[:300]}"
        )
        raise AssertionError(msg) from exc
    msg = (
        f"{divergence.key}: the repro passes. The defect is fixed: delete its "
        "entry (predicate included) so the sweeps cover the path again."
    )
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Repros
# ---------------------------------------------------------------------------


def _repro_tiff_gray_alpha() -> None:
    from tests.parity.framework.run import Axes, execute

    image = (np.arange(4 * 5 * 2).reshape(4, 5, 2) * 7 % 250).astype(np.uint8)
    out = execute([image], [], Axes(source="tiff"))
    assert np.array_equal(out.rows[0], image), (
        "a gray+alpha TIFF should decode as itself"
    )


def _repro_array_null_slice() -> None:
    from polars_cv import Pipeline

    image = np.zeros((1, 1, 1), dtype=np.uint8)
    rows = [None, None, None, image, image]
    frame = pl.DataFrame(
        {"x": pl.Series("x", rows, dtype=pl.Array(pl.UInt8, (1, 1, 1)))}
    )
    expr = pl.col("x").cv.pipe(Pipeline().source("array", dtype="u8")).sink("numpy")
    # One chunk; the streaming engine's morsels slice it.
    out = frame.lazy().select(expr).collect(engine="streaming")
    assert out["x"].null_count() == 3


def _repro_blob_numpy_rows() -> None:
    from tests.parity.framework.run import Axes, execute

    a = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    b = a + 100
    out = execute([a, b], [], Axes(source="blob", sink="numpy", engine="eager"))
    assert np.array_equal(out.rows[1], b), "row 1 came back as row 0's pixels"


def _repro_channel_swap_panic() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.arange(12, dtype=np.uint16).reshape(2, 2, 3)
    out = execute([image], [Step("channel_swap", {"order": [2, 1, 0]})], Axes())
    assert np.array_equal(out.rows[0], image[:, :, ::-1])


def _repro_channel_merge_dtype() -> None:
    from polars_cv import Pipeline, numpy_from_struct

    image = np.arange(12, dtype=np.uint16).reshape(2, 2, 3)
    frame = pl.DataFrame({"x": [image]}, schema={"x": pl.Array(pl.UInt16, image.shape)})
    source = Pipeline().source("array", dtype="u16")
    planes = [pl.col("x").cv.pipe(source.channel_select(c)) for c in range(3)]
    out = frame.select(o=planes[0].channel_merge(*planes[1:]).sink("numpy"))
    assert np.array_equal(numpy_from_struct(out["o"][0]), image)


def _repro_divide_contract() -> None:
    from tests.parity.framework.run import BinaryCase, execute_binary

    a = np.array([[[7]]], dtype=np.uint8)
    b = np.array([[[2]]], dtype=np.uint8)
    out = execute_binary(BinaryCase((a,), (b,), (), (), "divide")).rows[0]
    assert out.dtype == np.uint8 and out.ravel()[0] == 3, "documented: integer division"


def _repro_warp_per_row_matrix() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.random.default_rng(1).random((6, 7, 2))
    step = Step(
        "warp_affine",
        {
            # rotate_and_scale(angle=0.25, center=(0.0137, 0.0137), scale=0.5)
            "matrix": [
                0.49999524036036724,
                -0.0021816546423732855,
                0.006879953875663483,
                0.0021816546423732855,
                0.49999524036036724,
                0.006820176538462455,
            ],
            "output_size": (5, 6),
            "interpolation": "bilinear",
            "border_value": 0.0,
        },
    )
    literal = execute([image], [step], Axes(params="literal")).rows[0]
    per_row = execute([image], [step], Axes(params="column")).rows[0]
    assert np.array_equal(literal, per_row), "a per-row matrix changed the f64 result"


def _repro_threshold_wide_literal() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.array([[[18442240474082181119]]], dtype=np.uint64)
    step = Step("threshold", {"value": 1.8442240474082181e19})  # float(pixel)
    out = execute([image], [step], Axes(params="literal")).rows[0]
    assert out.ravel()[0] == 0, "the pixel does not exceed the threshold"


def _wide_int_threshold(step: Step, images: Sequence[Any]) -> bool:
    """A threshold beyond 2**50 over a 64-bit integer image."""
    if step.method != "threshold":
        return False
    value = step.params.get("value", 0.0)
    values = getattr(value, "values", (value,))  # a PerRow carries several
    return any(abs(float(v)) >= 2**50 for v in values) and any(
        im is not None and im.dtype in (np.uint64, np.int64) for im in images
    )


def _repro_nan_one_sided_clamp() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.array([[[np.nan]]], dtype=np.float32)
    for step in (
        Step("relu"),
        Step("clamp_min", {"value": 0.5}),
        Step("clamp_max", {"value": 0.5}),
    ):
        out = execute([image], [step], Axes()).rows[0]
        assert np.isnan(out.ravel()[0]), f"{step!r} turned NaN into a number"


def _repro_hsv_hue_180() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.array([[[255, 0, 1]]], dtype=np.uint8)  # hue 359.8 degrees
    out = execute([image], [Step("to_hsv")], Axes()).rows[0]
    assert out.ravel()[0] < 180, "8-bit hue is [0, 180): 359.8 degrees wraps to 0"


def _has_nan(x: np.ndarray) -> bool:
    return x.dtype.kind == "f" and bool(np.isnan(x).any())


def _hue_rounds_to_180(x: np.ndarray) -> bool:
    """Whether some RGB pixel's hue lies in [359, 360) degrees, where half of
    it rounds to 180."""
    import cv2

    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] not in (3, 4):
        return False
    rgb = np.ascontiguousarray(x[:, :, :3]).astype(np.float32) / 255.0
    hue = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 0]
    return bool((hue >= 359.0).any())


def _repro_fused_int_cast() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.array([[[1000], [60000], [300]]], dtype=np.uint16)
    steps = [Step("invert"), Step("cast", {"dtype": "u8"})]
    fused = execute([image], steps, Axes(optimize="all")).rows[0]
    unfused = execute([image], steps, Axes(optimize="none")).rows[0]
    # int -> int casts wrap (view-buffer/src/core/convert.rs): 64535 -> 23.
    assert np.array_equal(fused, unfused), f"fused {fused.ravel()} vs {unfused.ravel()}"


def _fused_int_cast(axes: Axes, images: Sequence[Any], steps: Sequence[Step]) -> bool:
    """scalar_fusion on, an integer image, and an integer cast after invert."""
    from tests.parity.framework.images import DTYPES
    from tests.parity.framework.run import opt_flags_for

    if not opt_flags_for(axes.optimize).scalar_fusion:
        return False
    if not any(im is not None and im.dtype.kind in "iu" for im in images):
        return False
    for i, step in enumerate(steps):
        target = step.params.get("dtype") if step.method == "cast" else None
        if target in DTYPES and DTYPES[target].kind in "iu":
            if any(prev.method == "invert" for prev in steps[:i]):
                return True
    return False


def _repro_derived_size_tie() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.zeros((14, 31, 1), dtype=np.uint8)  # 31 * 21 / 14 = 46.5
    out = execute(
        [image], [Step("resize_to_height", {"height": 21, "filter": "nearest"})], Axes()
    )
    # 7.5 -> 8, 10.5 -> 11, 1.5 -> 2 elsewhere: a tie rounds up.
    assert out.rows[0].shape[1] == 47, f"width {out.rows[0].shape[1]}"


def _derived_size_is_tie(step: Step, x: np.ndarray) -> bool:
    """Whether an aspect-preserving resize derives a size of exactly k + 1/2."""
    from fractions import Fraction

    if x.ndim < 2:
        return False
    h, w = x.shape[:2]
    p = step.params
    ratio = {
        "resize_to_height": lambda: (Fraction(p["height"], h),) * 2,
        "resize_to_width": lambda: (Fraction(p["width"], w),) * 2,
        "resize_max": lambda: (Fraction(p["max_size"], max(h, w)),) * 2,
        "resize_min": lambda: (Fraction(p["min_size"], min(h, w)),) * 2,
        "letterbox": lambda: (
            (min(Fraction(p["height"], h), Fraction(p["width"], w)),) * 2
        ),
        "resize_scale": lambda: (Fraction(p["scale_y"]), Fraction(p["scale_x"])),
    }.get(step.method)
    if ratio is None:
        return False
    fy, fx = ratio()
    return any((size * f) % 1 == Fraction(1, 2) for size, f in ((h, fy), (w, fx)))


def _repro_blend_int() -> None:
    from tests.parity.framework.run import BinaryCase, execute_binary

    a = np.array([[[81]]], dtype=np.int8)
    out = execute_binary(BinaryCase((a,), (a,), (), (), "blend")).rows[0]
    # (81 / 127) * (81 / 127) * 127 = 51.66
    assert out.ravel()[0] == 52, f"blend of i8 81 with itself is {out.ravel()[0]}"


def _repro_color_int_range() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    gray = np.full((1, 1, 3), 32768, dtype=np.uint16)
    out = execute([gray], [Step("to_ycbcr")], Axes()).rows[0].ravel()
    # A gray pixel has no chroma: Cb = Cr = the middle of the range.
    assert out.tolist() == [32768] * 3, f"u16 gray to YCbCr is {out.tolist()}"


_COLOR_SPACES = frozenset({"to_hsv", "to_ycbcr", "to_lab"})


def _color_int_range(step: Step, x: np.ndarray) -> bool:
    space = step.method in _COLOR_SPACES or (
        step.method == "convert_color"
        and step.params.get("to_space") not in (None, "rgb", "bgr", "gray")
    )
    return space and x.dtype.kind in "iu" and x.dtype != np.uint8


def _repro_convolve_f64() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.linspace(0, 1, 12).reshape(2, 2, 3)
    out = execute([image], [Step("sobel", {"axis": "x"})], Axes())
    assert out.rows[0].dtype == np.float64


def _repro_wide_int_through_f32() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.full((2, 2, 3), 16_777_217, dtype=np.uint32)
    out = execute([image], [Step("to_bgr")], Axes())
    assert np.array_equal(out.rows[0], image), (
        "a channel reorder must not round 16777217 to 16777216"
    )


def _repro_letterbox_zero_extent() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.array([[[10], [20], [30]]], dtype=np.uint8)  # 1 x 3
    step = Step(
        "letterbox", {"height": 1, "width": 1, "value": 0.0, "filter": "nearest"}
    )
    out = execute([image], [step], Axes())
    assert out.rows[0].ravel()[0] != 0, "the fitted content collapsed to zero rows"


def _repro_view_offset_lost() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.arange(6, dtype=np.uint8).reshape(3, 2, 1)
    crop = Step("crop", {"top": 1, "left": 0, "height": 2, "width": 2})
    out = execute([image], [crop, Step("channel_select", {"index": 0})], Axes())
    assert np.array_equal(out.rows[0], image[1:, :, 0]), "read from the uncropped start"


def _repro_reshape_after_view() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.arange(12, dtype=np.uint8).reshape(4, 3, 1)
    steps = [Step("flip", {"axes": [0]}), Step("reshape", {"shape": [12, 1]})]
    out = execute([image], steps, Axes())
    assert np.array_equal(out.rows[0], image[::-1].reshape(12, 1))


#: Ops that leave their output as a view of their input (no copy). A run of
#: them ending in a rank change is where the two view defects below live.
_VIEW_OPS = frozenset(
    {
        "crop",
        "flip",
        "flip_h",
        "flip_v",
        "transpose",
        "channel_select",
        "channel_swap",
        "reshape",
        "assert_shape",
    }
)


def _view_run(steps: Sequence[Step]) -> list[Step]:
    """The trailing run of view-producing steps before the last step."""
    run: list[Step] = []
    for step in reversed(steps[:-1]):
        quarter = (
            step.method == "rotate" and float(step.params.get("angle", 1)) % 90 == 0
        )
        if step.method not in _VIEW_OPS and not quarter:
            break
        run.append(step)
    return run


def _offset_view_then_rank_change(steps: Sequence[Step]) -> bool:
    if not steps or steps[-1].method not in ("channel_select", "reshape"):
        return False
    return any(
        s.method == "crop" and (s.params.get("top", 0) or s.params.get("left", 0))
        for s in _view_run(steps)
    )


def _reshape_of_a_view(steps: Sequence[Step]) -> bool:
    if not steps or steps[-1].method != "reshape":
        return False
    return any(s.method not in ("assert_shape",) for s in _view_run(steps))


#: Ops that convert or resample a 32/64-bit integer image through f32.
_THROUGH_F32 = frozenset(
    {
        "to_bgr",
        "convert_color",
        "blur",
        "resize",
        "resize_to_height",
        "resize_to_width",
        "resize_max",
        "resize_min",
        "resize_scale",
        "letterbox",
        "morphology_gradient",
        # binary ops (the DAG suite asks with Step(method) and each operand)
        "add",
        "subtract",
        "multiply",
        "blend",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "maximum",
        "minimum",
    }
)


def _not_representable_in_f32(x: np.ndarray) -> bool:
    """Whether some value of *x* changes on a round trip through f32."""
    if x.size == 0 or x.dtype.itemsize < 4 or x.dtype == np.float32:
        return False
    if x.dtype.kind in "iu":
        return float(np.max(np.abs(x.astype(np.float64)))) > 2**24
    return bool(np.any(x.astype(np.float32).astype(x.dtype) != x))


def _derived_extent_is_zero(step: Step, x: np.ndarray) -> bool:
    """Whether an aspect-preserving resize derives a size that rounds to 0."""
    if x.ndim < 2:
        return False
    h, w = x.shape[:2]
    p = step.params
    factors = {
        "letterbox": lambda: (min(p["height"] / h, p["width"] / w),) * 2,
        "resize_max": lambda: (p["max_size"] / max(h, w),) * 2,
        "resize_min": lambda: (p["min_size"] / min(h, w),) * 2,
        "resize_to_height": lambda: (p["height"] / h,) * 2,
        "resize_to_width": lambda: (p["width"] / w,) * 2,
        "resize_scale": lambda: (p["scale_y"], p["scale_x"]),
    }.get(step.method)
    if factors is None:
        return False
    fy, fx = factors()
    return min(h * fy, w * fx) < 0.5


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

DIVERGENCES: tuple[Divergence, ...] = (
    Divergence(
        key="tiff-gray-alpha",
        raises=pl.exceptions.ComputeError,
        match="Unsupported TIFF color type",
        summary=(
            "The image decoder cannot read a gray+alpha (2-sample) TIFF: one "
            "written by Pillow is refused ('Unsupported TIFF color type: "
            "Multiband { bit_depth: 8, num_samples: 2 }'), and one written by "
            "the engine's own tiff sink — which OpenCV decodes correctly — "
            "comes back as 4 channels of the wrong values. PNG round-trips "
            "the same image exactly. Fixed: a gray+alpha TIFF decodes as its "
            "two channels."
        ),
        repro=_repro_tiff_gray_alpha,
        affects_axes=lambda axes, images, shapes, steps: (
            axes.source == "tiff"
            and any(im is not None and im.shape[2] == 2 for im in images)
        ),
        avoid=True,
    ),
    Divergence(
        key="array-null-slice-panic",
        raises=pl.exceptions.ComputeError,
        match="panicked",
        summary=(
            "An Array column with null rows panics in polars-arrow ('the "
            "offset of the new Buffer cannot exceed the existing length') "
            "once it is sliced: across two chunks under any engine, or in one "
            "chunk under the streaming engine once its morsels split it (three "
            "leading nulls in five rows is enough). The plugin slices the "
            "FixedSizeList with an offset it has already applied. Fixed: the "
            "null rows come back null."
        ),
        repro=_repro_array_null_slice,
        affects_axes=lambda axes, images, shapes, steps: (
            axes.source == "array"
            and (axes.chunked or axes.engine == "streaming")
            and any(im is None for im in images)
        ),
        avoid=True,
    ),
    Divergence(
        key="binary-source-numpy-rows",
        match="row 1 came back as row 0's pixels",
        summary=(
            "A multi-row blob or raw column read by the eager or in-memory "
            "engine into a zero-copy tensor sink (numpy/ndarray/torch) returns "
            "the wrong pixels for every row but the first (blob: row 0's; "
            "raw: garbage). The blob, png and list sinks, a materializing op "
            "(cast) and the streaming engine are all correct, so the "
            "zero-copy output of a view over a Binary column mis-addresses "
            "rows. Fixed: every row comes back as itself."
        ),
        repro=_repro_blob_numpy_rows,
        affects_axes=lambda axes, images, shapes, steps: (
            (axes.source in ("blob", "raw") or axes.composition == "materialized")
            and axes.sink in ("numpy", "ndarray", "torch")
            and axes.engine != "streaming"
            and sum(im is not None for im in images) > 1
        ),
    ),
    Divergence(
        key="channel-swap-panic",
        raises=pl.exceptions.ComputeError,
        match="ImageOp contract violation: ChannelSwap",
        summary=(
            "channel_swap panics on every dtype but u8 and f32 ('ImageOp "
            "contract violation: ChannelSwap: expected output dtype U16 (rule "
            "PreserveInput ...), but got F32'): the kernel converts through "
            "f32 while its contract preserves the dtype. Fixed: the channels "
            "are reordered in the input dtype."
        ),
        repro=_repro_channel_swap_panic,
        affects_step=lambda step, x: (
            step.method == "channel_swap" and x.dtype not in (np.uint8, np.float32)
        ),
        avoid=True,
    ),
    Divergence(
        key="channel-merge-dtype",
        raises=pl.exceptions.ComputeError,
        match="planned dtype u16 but execution produced F32",
        summary=(
            "channel_merge of any dtype but u8 and f32 plans the operands' "
            "dtype and executes f32, which the plugin's output guard rejects "
            "('planned dtype u16 but execution produced F32'); the same "
            "through-f32 kernel as channel-swap-panic. Fixed: the planes are "
            "stacked in their own dtype."
        ),
        repro=_repro_channel_merge_dtype,
        affects_step=lambda step, x: (
            step.method == "channel_merge" and x.dtype not in (np.uint8, np.float32)
        ),
        avoid=True,
    ),
    Divergence(
        key="divide-ratio-contract",
        match="documented: integer division",
        summary=(
            "divide and ratio do not do what their docs say. Documented: "
            "divide is integer division for u8/u16 (x / 0 -> 0) and IEEE "
            "division for floats (x / 0 -> inf); ratio scales a / b by the "
            "dtype's maximum and clamps. Observed: both return f32 a / b on "
            "every dtype, with x / 0 -> 0.0 even for floats, so ratio is "
            "divide. Fixed: code or docs change, and the reference follows."
        ),
        repro=_repro_divide_contract,
        affects_step=lambda step, x: step.method in ("divide", "ratio"),
    ),
    Divergence(
        key="warp-per-row-matrix",
        match="a per-row matrix changed the f64 result",
        summary=(
            "warp_affine (and shear/rotate_and_scale, which lower to it) "
            "gives a different result when the matrix arrives as per-row "
            "expressions rather than literals. Both run the same f64 kernel; "
            "the literal matrix is parsed from the graph JSON inexactly "
            "(serde_json without float_roundtrip can land an ulp away), so "
            "the kernel gets a different matrix. It shows in the last bits of "
            "f64 (up to 8e-16) and of wide integers (u64 110184465182317 vs "
            "...318), and can flip a rounding on any integer dtype. Fixed: "
            "literal and per-row matrices give identical output."
        ),
        repro=_repro_warp_per_row_matrix,
        affects_axes=lambda axes, images, shapes, steps: (
            axes.params != "literal"
            and any(
                s.method in ("warp_affine", "shear", "rotate_and_scale") for s in steps
            )
        ),
    ),
    Divergence(
        key="threshold-wide-int",
        match="the pixel does not exceed the threshold",
        summary=(
            "threshold on a u64/i64 image near the edge of f64 precision "
            "(|value| >= 2**50) depends on whether the value arrives as a "
            "literal or an expression, on every source: u64 pixel "
            "18442240474082181119 against the literal 1.8442240474082181e19 "
            "gives 255 where the same value as an expression gives 0; i64 "
            "pixel 3797082577976981 against 3797082577976980.5 gives 0 as a "
            "literal and 255 as an expression. The literal is parsed from the "
            "graph JSON inexactly, and the kernel compares the pixel rounded "
            "to f64. Fixed: one exact answer on every path."
        ),
        repro=_repro_threshold_wide_literal,
        affects_step=lambda step, x: _wide_int_threshold(step, [x]),
        affects_axes=lambda axes, images, shapes, steps: any(
            _wide_int_threshold(s, images) for s in steps
        ),
    ),
    Divergence(
        key="nan-one-sided-clamp",
        match="turned NaN into a number",
        summary=(
            "relu, clamp_min and clamp_max turn NaN into their bound (relu(NaN) "
            "-> 0, clamp_min(0.5) of NaN -> 0.5), while clamp, abs, sign, "
            "round and scale propagate it — and NumPy and PyTorch propagate it "
            "through all of these. Fixed: NaN in, NaN out (or the rule is "
            "chosen and documented)."
        ),
        repro=_repro_nan_one_sided_clamp,
        affects_step=lambda step, x: (
            step.method in ("relu", "clamp_min", "clamp_max") and _has_nan(x)
        ),
    ),
    Divergence(
        key="hsv-hue-180",
        match="8-bit hue is [0, 180)",
        summary=(
            "to_hsv on u8 emits H = 180 for hues in [359, 360) degrees "
            "(RGB(255, 0, 1) -> H 180), outside 8-bit HSV's [0, 180); OpenCV "
            "wraps them to 0. A 180-entry hue table indexed with it reads out "
            "of bounds. Fixed: H is taken modulo 180."
        ),
        repro=_repro_hsv_hue_180,
        affects_step=lambda step, x: step.method == "to_hsv" and _hue_rounds_to_180(x),
    ),
    Divergence(
        key="scalar-fusion-int-cast",
        match="fused [255 255 255]",
        summary=(
            "The scalar_fusion pass changes results: it fuses an integer "
            "invert with a following int -> int cast into one kernel that "
            "converts by the float -> int rule (round, saturate) instead of "
            "the int -> int rule (wrap). u16 invert().cast('u8') of 1000 is 23 "
            "unfused (64535 wraps) and 255 fused; i16 targets likewise. Every "
            "pass is documented as byte-identical. Fixed: fused and unfused "
            "agree."
        ),
        repro=_repro_fused_int_cast,
        affects_axes=lambda axes, images, shapes, steps: _fused_int_cast(
            axes, images, steps
        ),
    ),
    Divergence(
        key="derived-size-tie",
        match="width 46",
        summary=(
            "An aspect-preserving resize whose derived size is exactly k + 1/2 "
            "rounds it inconsistently: resize_to_height(21) of a 14x31 image "
            "derives width 46 (46.5 down) while 7.5, 10.5 and 1.5 round up — "
            "the size is computed in floating point in an order that lands "
            "some ties just below .5. Fixed: ties round one way."
        ),
        repro=_repro_derived_size_tie,
        affects_step=_derived_size_is_tie,
    ),
    Divergence(
        key="blend-int",
        match="blend of i8 81 with itself is 127",
        summary=(
            "blend is normalized ((a/MAX)(b/MAX)MAX) only for u8 and u16; "
            "every other integer dtype takes the float path, where blend is "
            "a plain multiply saturated to the dtype (i8 81 blend 81 -> 127, "
            "not 52). The per-dtype semantics of the binary ops are written "
            "for u8/u16 and floats only. Fixed: blend normalizes by the "
            "dtype's maximum on every integer dtype."
        ),
        repro=_repro_blend_int,
        affects_step=lambda step, x: (
            step.method == "blend"
            and x.dtype.kind in "iu"
            and x.dtype not in (np.uint8, np.uint16)
        ),
    ),
    Divergence(
        key="color-int-range",
        match="u16 gray to YCbCr is [32768, 128, 128]",
        summary=(
            "The colour-space conversions (to_hsv, to_ycbcr, to_lab) use "
            "8-bit constants on every integer dtype: a u16 gray pixel 32768 "
            "converts to YCbCr (32768, 128, 128) (the chroma offset is 128, "
            "not half the range), HSV saturation is scaled to 255 while V "
            "keeps the input's range, and Lab reads the input as 0-255 (u16 "
            "mid-gray has L = 5393). The oracle models u8 only; widen its "
            "ref_accepts with the fix. Fixed: each integer dtype converts "
            "over its own range."
        ),
        repro=_repro_color_int_range,
        affects_step=_color_int_range,
    ),
    Divergence(
        key="convolve2d-f64",
        raises=pl.exceptions.ComputeError,
        match="planned dtype f64 but execution produced F32",
        summary=(
            "convolve2d (and sobel/laplacian/sharpen, which lower to it) on an "
            "f64 image plans f64 but executes f32, which the plugin's own "
            "output guard rejects ('planned dtype f64 but execution produced "
            "F32'). Fixed: f64 in, f64 out."
        ),
        repro=_repro_convolve_f64,
        affects_step=lambda step, x: (
            step.method in ("convolve2d", "sobel", "laplacian", "sharpen")
            and x.dtype == np.float64
        ),
        avoid=True,
    ),
    Divergence(
        key="through-f32",
        match="must not round 16777217",
        summary=(
            "32/64-bit integer and f64 images are converted or resampled "
            "through f32, so a value f32 cannot hold changes (u32 16777217 -> "
            "16777216; f64 0.63696169 -> 0.63696170) even where the op only "
            "moves data (to_bgr, convert_color rgb->bgr, a nearest resize), "
            "and in the binary ops: every operand pair but u8, u16, f32 and "
            "f64 is computed in f32 (u32 16777219 ^ 16777221 -> 0, not 6). "
            "Fixed: those ops are exact on every dtype."
        ),
        repro=_repro_wide_int_through_f32,
        affects_step=lambda step, x: (
            step.method in _THROUGH_F32 and _not_representable_in_f32(x)
        ),
    ),
    Divergence(
        key="derived-extent-zero",
        match="collapsed to zero rows",
        summary=(
            "The aspect-preserving resizes (resize_max/min/to_height/to_width/"
            "scale, letterbox) round a derived size without a floor of one "
            "pixel: resize_max(1) of a 3x1 image is 1x0, resize_scale(0.25) "
            "of a 1x1 image is 0x0, and letterbox of a 1x3 image into 1x1 "
            "fits the content to zero rows and returns pure padding. Fixed: "
            "every derived size is at least 1."
        ),
        repro=_repro_letterbox_zero_extent,
        affects_step=_derived_extent_is_zero,
        avoid=True,
    ),
    Divergence(
        key="view-offset-lost",
        match="read from the uncropped start",
        summary=(
            "A crop that starts below the first row (or right of the first "
            "column) leaves a view with an offset; a zero-copy rank change "
            "applied to it in the same pipeline — channel_select on a "
            "single-channel image, or reshape — reads from the *uncropped* "
            "start. crop(top=1).channel_select(0) of a 3x2x1 image returns "
            "rows 0-1, not 1-2. Materializing the crop first (a blob round "
            "trip) gives the right answer. Fixed: the rank change honours the "
            "view's offset."
        ),
        repro=_repro_view_offset_lost,
        affects_chain=_offset_view_then_rank_change,
        avoid=True,
    ),
    Divergence(
        key="reshape-after-view",
        raises=pl.exceptions.ComputeError,
        match="cannot reshape a non-contiguous view",
        summary=(
            "reshape after a flip, transpose, quarter rotation or crop fails at "
            "execution ('cannot reshape a non-contiguous view ... the elements "
            "are not in reshape order') where NumPy would copy. It is refused "
            "on purpose, but at run time rather than by the planner, and with "
            "no way to ask for the copy. Fixed (or decided): the view is "
            "materialized, or the planner refuses the chain."
        ),
        repro=_repro_reshape_after_view,
        affects_chain=_reshape_of_a_view,
        avoid=True,
    ),
)


def _first(matches: Sequence[Divergence]) -> Divergence | None:
    """The match a caller must act on: an ``avoid`` entry if any matches.

    Several entries can cover one case (a large i32 image resized to zero
    width is both ``through-f32`` and ``derived-extent-zero``). Returning the
    first in registry order would let a value-only entry mask one whose case
    must not be executed at all.
    """
    for divergence in matches:
        if divergence.avoid:
            return divergence
    return matches[0] if matches else None


def axes_divergence(
    axes: Axes,
    images: Sequence[Any],
    shapes: Sequence[tuple[int, ...]],
    steps: Sequence[Step] = (),
) -> Divergence | None:
    """The known divergence executing *steps* along *axes* would hit, if any."""
    return _first(
        [
            d
            for d in DIVERGENCES
            if d.affects_axes is not None
            and d.affects_axes(axes, images, shapes, steps)
        ]
    )


def step_divergence(step: Step, x: np.ndarray) -> Divergence | None:
    """The known divergence *step* on input *x* would hit, if any."""
    return _first(
        [
            d
            for d in DIVERGENCES
            if d.affects_step is not None and d.affects_step(step, x)
        ]
    )


def chain_divergence(steps: Sequence[Step]) -> Divergence | None:
    """The known divergence the last of *steps* would hit, given the rest."""
    return _first(
        [
            d
            for d in DIVERGENCES
            if d.affects_chain is not None and d.affects_chain(steps)
        ]
    )


def append_divergence(
    steps: Sequence[Step], step: Step, inputs: Sequence[np.ndarray]
) -> Divergence | None:
    """The divergence to avoid when appending *step* after *steps*, if any.

    *inputs* are the arrays the step would read (one per non-null row).
    Only ``avoid`` entries count: a value-only divergence is withheld at
    comparison time, not by refusing to build the case.
    """
    candidates = [chain_divergence([*steps, step])]
    candidates += [step_divergence(step, x) for x in inputs]
    for divergence in candidates:
        if divergence is not None and divergence.avoid:
            return divergence
    return None
