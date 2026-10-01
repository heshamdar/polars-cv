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


def _repro_hsv_hue_180() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.array([[[255, 0, 1]]], dtype=np.uint8)  # hue 359.8 degrees
    out = execute([image], [Step("to_hsv")], Axes()).rows[0]
    assert out.ravel()[0] < 180, "8-bit hue is [0, 180): 359.8 degrees wraps to 0"


def _hue_rounds_to_180(x: np.ndarray) -> bool:
    """Whether some RGB pixel's hue lies in [359, 360) degrees, where half of
    it rounds to 180."""
    import cv2

    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] not in (3, 4):
        return False
    rgb = np.ascontiguousarray(x[:, :, :3]).astype(np.float32) / 255.0
    hue = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 0]
    return bool((hue >= 359.0).any())


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


def _repro_letterbox_zero_extent() -> None:
    from tests.parity.framework.run import Axes, Step, execute

    image = np.array([[[10], [20], [30]]], dtype=np.uint8)  # 1 x 3
    step = Step(
        "letterbox", {"height": 1, "width": 1, "value": 0.0, "filter": "nearest"}
    )
    out = execute([image], [step], Axes())
    assert out.rows[0].ravel()[0] != 0, "the fitted content collapsed to zero rows"


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
)


def _first(matches: Sequence[Divergence]) -> Divergence | None:
    """The match a caller must act on: an ``avoid`` entry if any matches.

    Several entries can cover one case (a large i32 image resized to zero
    width was both a value-only precision entry and ``derived-extent-zero``).
    Returning the first in registry order would let a value-only entry mask
    one whose case must not be executed at all.
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
