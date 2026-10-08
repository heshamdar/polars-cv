"""The reference table: one entry per operation.

Each :class:`OpSpec` says, for one ``Pipeline`` method:

* **when it applies** (:attr:`OpSpec.accepts`) — the inputs the op's contract
  admits, so a generator only proposes calls that should work. If the planner
  refuses one of these, that is a finding, not a skipped example.
* **how to call it** (:attr:`OpSpec.params`) — a strategy for arguments that
  are valid on a given input.
* **what it should produce** (:attr:`OpSpec.ref`) — an independent
  implementation on NumPy / OpenCV / PIL / SciPy, which also states the
  output dtype on its own authority. ``None`` when there is no trustworthy
  reference, with the reason in :attr:`OpSpec.no_ref`.
* **how close is close enough** (:attr:`OpSpec.tol`, :attr:`OpSpec.gain`,
  :attr:`OpSpec.kind`) — see :mod:`tests.parity.framework.tolerance`.

Conventions the engine documents and a reference library does not share are
modelled in the reference, never absorbed into a tolerance, and each is
stated in the entry's ``note``: resize resamples height before width (PIL
does width first), rotation is clockwise about ``(w/2, h/2)`` (OpenCV's
``getRotationMatrix2D`` is counter-clockwise), and so on. A tolerance wide
enough to hide a convention would hide a bug of the same size.

Coverage is ratcheted (``meta/test_parity_ratchets.py``): every chainable
``Pipeline`` method and every lazy-only binary op must have an entry here or
an exemption in :data:`EXEMPT` / :data:`BINARY_EXEMPT` naming its reason.
"""

from __future__ import annotations

import bisect
import math
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import cv2
import numpy as np
from hypothesis import strategies as st
from PIL import Image
from scipy import ndimage

from tests.parity.framework.images import DTYPES, dtype_name
from tests.parity.framework.tolerance import EXACT, Tol, close, lsb, sparse, widen

Params = dict[str, Any]
ParamStrategy = Callable[[st.DrawFn, np.ndarray], Params]
Reference = Callable[[np.ndarray, Params], Any]


def _always(x: np.ndarray, p: Params | None = None) -> bool:
    return True


def _non_finite(x: Any) -> bool:
    arr = np.asarray(x)
    return arr.dtype.kind == "f" and arr.size > 0 and not bool(np.isfinite(arr).all())


@dataclass(frozen=True)
class OpSpec:
    """Everything the parity suites know about one ``Pipeline`` method."""

    method: str
    params: ParamStrategy
    accepts: Callable[[np.ndarray], bool] = _always
    ref: Reference | None = None
    ref_accepts: Callable[[np.ndarray, Params], bool] = _always
    tol: Tol | Callable[[np.ndarray, Params], Tol] = EXACT
    gain: float | Callable[[np.ndarray, Params], float] = 1.0
    kind: str = "pointwise"
    domain_in: str = "buffer"
    domain_out: str = "buffer"
    no_ref: str = ""
    note: str = ""
    #: A terminal op ends a buffer chain (its output is not an image).
    terminal: bool = False
    #: The dtypes the reference can model (``None``: all). The reference suites
    #: draw inputs from these so their examples are not spent on skips; the
    #: invariance suites ignore it and draw every dtype the op accepts.
    ref_dtypes: tuple[str, ...] | None = None
    #: Whether the arguments restate a row's whole shape (``assert_shape``,
    #: ``reshape``), so one literal set cannot serve rows of different sizes.
    uniform_rows: bool = False
    #: Whether the op defines a NaN/infinity rule the reference shares (numpy's
    #: for the ordering reductions and ``histogram``), so non-finite input is
    #: compared rather than withheld (see :meth:`has_reference`).
    defines_non_finite: bool = False
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.ref is None and not self.no_ref:
            msg = f"{self.method}: an entry without a reference must say why"
            raise ValueError(msg)

    def tolerance(self, x: np.ndarray, p: Params) -> Tol:
        """The step's own error bound on input *x* with arguments *p*."""
        return self.tol(x, p) if callable(self.tol) else self.tol

    def gain_for(self, x: np.ndarray, p: Params) -> float:
        """The step's Lipschitz bound on input *x* with arguments *p*."""
        return self.gain(x, p) if callable(self.gain) else self.gain

    def has_reference(self, x: np.ndarray, p: Params) -> bool:
        """Whether the reference models this call on this input.

        Non-finite input (a NaN or infinity an earlier step produced) is
        compared only through pointwise ops and data movement, where IEEE
        arithmetic defines the answer. A kernel mixing neighbours or whole-
        image statistics has no agreed NaN rule (OpenCV propagates a NaN
        through a zero interpolation weight, the engine skips the tap), so
        those comparisons are withheld; the invariance suites still cover it.
        """
        if self.ref is None or not self.ref_accepts(x, p):
            return False
        if (
            self.kind in ("spatial", "global")
            and not self.defines_non_finite
            and _non_finite(x)
        ):
            return False
        return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


#: The dtypes whose float is f64 (``DType::accumulator``): f32 cannot hold
#: the 32/64-bit integers, which NumPy promotes to f64.
_F64_ACCUMULATED = {
    np.dtype(t) for t in (np.float64, np.uint32, np.int32, np.uint64, np.int64)
}


def float_out(x: np.ndarray) -> np.dtype:
    """The dtype a float-promoting op produces (``PromoteToFloat``): f64 for
    f64 and the 32/64-bit integers, f32 otherwise."""
    return np.dtype(np.float64 if x.dtype in _F64_ACCUMULATED else np.float32)


def round_half_away(v: np.ndarray) -> np.ndarray:
    """Round half away from zero (the engine's float → integer rule, Rust's
    ``f64::round``), exactly.

    Not ``floor(|v| + 0.5)``: that addition rounds, so the largest double
    below 0.5 becomes 1 and an odd integer above 2**52 the even one after it.
    ``|v| - floor(|v|)`` is exact (the two are within 1 of each other), so the
    comparison with 0.5 decides the tie, and only the tie.
    """
    a = np.abs(v)
    f = np.floor(a)
    return np.sign(v) * (f + (a - f >= 0.5))


def to_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert float *values* to *dtype*: round half away and saturate."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return values.astype(dtype)
    info = np.iinfo(dtype)
    v = np.nan_to_num(round_half_away(values.astype(np.float64)), nan=0.0)
    lo, hi = float(info.min), float(info.max)
    clipped = np.clip(v, lo, hi)
    out = np.empty(clipped.shape, dtype=dtype)
    # float64 cannot represent the 64-bit maxima; place them exactly.
    top = clipped >= hi
    bottom = clipped <= lo
    mid = ~(top | bottom)
    out[mid] = clipped[mid].astype(dtype)
    out[top] = info.max
    out[bottom] = info.min
    return out


def is_rank3(x: np.ndarray) -> bool:
    return x.ndim == 3


def is_int(x: np.ndarray) -> bool:
    return x.dtype.kind in "iu"


def is_u8(x: np.ndarray) -> bool:
    return x.dtype == np.uint8


def dtype_max(x: np.ndarray) -> float:
    return float(np.iinfo(x.dtype).max) if is_int(x) else 1.0


def magnitude(x: np.ndarray) -> float:
    """Largest absolute value in *x* (floor 1), for scaling float tolerances."""
    if x.size == 0:
        return 1.0
    return max(1.0, float(np.max(np.abs(x.astype(np.float64)))))


def _accumulated(x: np.ndarray, own: Tol) -> Tol:
    """*own*, plus the accumulator's rounding on a 64-bit integer image.

    An interpolating op computes in ``DType::accumulator``: f64 for the
    64-bit integers, which holds 53 bits. Above 2**53 a computed value is good
    to a few f64 ulps of the magnitude, not to one unit, and the float64
    reference rounds the same way, so the two can differ by that much.
    """
    if x.dtype.kind in "iu" and x.dtype.itemsize == 8:
        return widen(own, atol=own.atol + 8 * 2.0**-52 * magnitude(x))
    return own


def per_channel(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Apply a 2-D function to each channel of ``[H, W, C]`` *x*."""
    return np.stack([fn(x[:, :, c]) for c in range(x.shape[2])], axis=2)


def color_channels(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """``ColorChannels``: run *fn* on RGB, carry alpha through unchanged."""
    if x.shape[2] == 4:
        return np.concatenate([fn(x[:, :, :3]), x[:, :, 3:]], axis=2)
    return fn(x)


def quiet(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Silence NumPy's divide/invalid warnings inside a reference."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)

    return wrapped


# Values exactly representable in f32, so a threshold or constant compares
# identically whether the engine widens the pixel or narrows the parameter.
_NICE_FLOATS = st.integers(-64, 64).map(lambda k: k / 8.0)


def _nice(lo: float, hi: float) -> st.SearchStrategy[float]:
    return st.integers(math.ceil(lo * 8), math.floor(hi * 8)).map(lambda k: k / 8.0)


def _no_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {}


# ---------------------------------------------------------------------------
# Data movement (exact on every dtype)
# ---------------------------------------------------------------------------


def _crop_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    h, w = x.shape[:2]
    top = draw(st.integers(0, h - 1), label="top")
    left = draw(st.integers(0, w - 1), label="left")
    return {
        "top": top,
        "left": left,
        "height": draw(st.integers(1, h - top), label="height"),
        "width": draw(st.integers(1, w - left), label="width"),
    }


def _crop_ref(x: np.ndarray, p: Params) -> np.ndarray:
    top, left = p["top"], p["left"]
    return x[top : top + p["height"], left : left + p["width"]].copy()


def _flip_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    axes = draw(
        st.lists(st.integers(0, x.ndim - 1), min_size=1, max_size=x.ndim, unique=True),
        label="axes",
    )
    return {"axes": sorted(axes)}


def _transpose_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"axes": draw(st.permutations(list(range(x.ndim))), label="axes")}


def _pad_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    h, w = x.shape[:2]
    mode = draw(st.sampled_from(["constant", "edge", "reflect", "symmetric"]))
    # reflect mirrors about the edge pixel, so it cannot pad past size - 1;
    # symmetric repeats it, so past size. Stay within one reflection.
    limit_h = {"reflect": h - 1, "symmetric": h}.get(mode, 6)
    limit_w = {"reflect": w - 1, "symmetric": w}.get(mode, 6)
    limit_h, limit_w = min(limit_h, 6), min(limit_w, 6)
    return {
        "top": draw(st.integers(0, limit_h), label="top"),
        "bottom": draw(st.integers(0, limit_h), label="bottom"),
        "left": draw(st.integers(0, limit_w), label="left"),
        "right": draw(st.integers(0, limit_w), label="right"),
        "value": draw(_fill_value(x), label="value"),
        "mode": mode,
    }


def _fill_value(x: np.ndarray) -> st.SearchStrategy[float]:
    """A pad/border fill the dtype can hold exactly."""
    if is_int(x):
        info = np.iinfo(x.dtype)
        return st.integers(max(info.min, -128), min(info.max, 255)).map(float)
    return _NICE_FLOATS


def _pad_ref(x: np.ndarray, p: Params) -> np.ndarray:
    widths = [(p["top"], p["bottom"]), (p["left"], p["right"])] + [(0, 0)] * (
        x.ndim - 2
    )
    if p["mode"] == "constant":
        return np.pad(
            x, widths, mode="constant", constant_values=x.dtype.type(p["value"])
        )
    return np.pad(x, widths, mode=p["mode"])


_POSITIONS = ("center", "top-left", "bottom-right")


def _pad_to_size_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    h, w = x.shape[:2]
    return {
        "height": draw(st.integers(h, h + 7), label="height"),
        "width": draw(st.integers(w, w + 7), label="width"),
        "position": draw(st.sampled_from(_POSITIONS), label="position"),
        "value": draw(_fill_value(x), label="value"),
    }


def _placement(extra: int, position: str) -> int:
    return {"center": extra // 2, "top-left": 0, "bottom-right": extra}[position]


def _pad_to_size_ref(x: np.ndarray, p: Params) -> np.ndarray:
    h, w = x.shape[:2]
    top = _placement(p["height"] - h, p["position"])
    left = _placement(p["width"] - w, p["position"])
    out = np.full((p["height"], p["width"], *x.shape[2:]), x.dtype.type(p["value"]))
    out[top : top + h, left : left + w] = x
    return out


def _channel_select_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"index": draw(st.integers(0, x.shape[2] - 1), label="index")}


def _channel_swap_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"order": draw(st.permutations(list(range(x.shape[2]))), label="order")}


def _reshape_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    h, w, c = x.shape
    options = [[h, w, c], [h * w, c], [h, w * c], [w, h, c], [h * w * c, 1, 1]]
    if c == 1:
        options.append([h, w])
    return {"shape": draw(st.sampled_from(options), label="shape")}


def _rot90_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"angle": draw(st.sampled_from([90.0, 180.0, 270.0, -90.0]), label="angle")}


def _rot90_ref(x: np.ndarray, p: Params) -> np.ndarray:
    # Positive angles are clockwise; np.rot90 turns counter-clockwise.
    return np.ascontiguousarray(np.rot90(x, k=-int(p["angle"]) // 90, axes=(0, 1)))


def _assert_shape_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"dims": list(x.shape)}


# ---------------------------------------------------------------------------
# Per-value maps
# ---------------------------------------------------------------------------


def _cast_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"dtype": draw(st.sampled_from(list(DTYPES)), label="dtype")}


@quiet
def _cast_ref(x: np.ndarray, p: Params) -> np.ndarray:
    target = DTYPES[p["dtype"]]
    if target.kind == "f":
        return x.astype(target)
    if x.dtype.kind == "f":
        return to_dtype(x.astype(np.float64), target)
    # Integer to integer is a plain `as`: narrowing wraps (two's complement),
    # which is what NumPy's unsafe astype does too.
    return x.astype(target)


def _cast_gain(x: np.ndarray, p: Params) -> float:
    """1, unless an integer-to-integer cast can wrap: then a one-level error
    on the input (127 vs 128 into ``i8``) is a 255-level one on the output,
    so the cast is discontinuous there. A float cast saturates instead."""
    target = DTYPES[p["dtype"]]
    if x.dtype.kind in "iu" and target.kind in "iu":
        return 1.0 if np.can_cast(x.dtype, target, casting="safe") else _INF
    return 1.0


def _cast_ref_accepts(x: np.ndarray, p: Params) -> bool:
    return not (x.dtype.kind == "f" and not np.all(np.isfinite(x)))


def _scalar(fn: Callable[[np.ndarray, Params], np.ndarray]) -> Reference:
    """A float-promoting per-value op, computed in the output float type."""

    @quiet
    def ref(x: np.ndarray, p: Params) -> np.ndarray:
        f = float_out(x)
        return np.asarray(fn(x.astype(f), p), dtype=f)

    return ref


def _value_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"value": draw(_NICE_FLOATS, label="value")}


def _scalar_tol(x: np.ndarray, p: Params) -> Tol:
    # One f32 rounding of the result may differ from computing the constant
    # in f64 first; allow one unit in the last place, relatively.
    eps = float(np.finfo(float_out(x)).eps)
    return Tol(atol=1e-30, rtol=2 * eps)


def _factor_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"factor": draw(_nice(-3, 3), label="factor")}


def _invert_ref(x: np.ndarray, p: Params) -> np.ndarray:
    if is_int(x):
        info = np.iinfo(x.dtype)
        return (info.max + info.min - x.astype(object)).astype(x.dtype)
    return (x.dtype.type(1) - x).astype(x.dtype)


def _threshold_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    if x.size and is_int(x):
        lo, hi = int(x.min()), int(x.max())
        value = draw(st.integers(lo - 1, hi).map(lambda v: v + 0.5), label="value")
        return {"value": float(value)}
    return {"value": draw(_NICE_FLOATS, label="value")}


def _threshold_ref(x: np.ndarray, p: Params) -> np.ndarray:
    """``255`` where a pixel exceeds the value, compared exactly: Python
    compares an int with a float exactly, where a 64-bit pixel rounded to
    f64 can land on the value."""
    t = p["value"]
    above = np.array([v > t for v in x.ravel().tolist()], dtype=bool)
    return np.where(above.reshape(x.shape), 255, 0).astype(np.uint8)


def _clamp_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    a = draw(_nice(-4, 300), label="min_val")
    b = draw(_nice(-4, 300), label="max_val")
    return {"min_val": min(a, b), "max_val": max(a, b)}


def _brightness_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"factor": draw(_nice(0, 3), label="factor")}


@quiet
def _contrast_ref(x: np.ndarray, p: Params) -> np.ndarray:
    f = float_out(x)
    xf = x.astype(np.float64)
    mean = xf.mean()
    return ((xf - mean) * p["factor"] + mean).astype(f)


def _stat_tol(x: np.ndarray, p: Params) -> Tol:
    """A float result through a whole-image statistic (mean, std, min/max).

    The engine may accumulate in f32 while the reference uses f64; the
    difference scales with the image's magnitude, not the element's.
    """
    eps = float(np.finfo(float_out(x)).eps)
    scale = magnitude(x) * max(1.0, abs(float(p.get("factor", 1.0))))
    # The statistic itself carries summation-order error growing with n.
    return Tol(
        atol=64 * eps * scale * max(1.0, math.log2(max(2, x.size))), rtol=8 * eps
    )


def _gamma_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"gamma": draw(st.sampled_from([0.25, 0.5, 1.0, 1.5, 2.0, 3.0]))}


@quiet
def _gamma_ref(x: np.ndarray, p: Params) -> np.ndarray:
    f = float_out(x)
    peak = dtype_max(x)
    return (np.power(x.astype(np.float64) / peak, p["gamma"]) * peak).astype(f)


def _gamma_ref_accepts(x: np.ndarray, p: Params) -> bool:
    # [0, MAX] for an integer image, [0, 1] for a float one; below 0 the
    # power is undefined and the op documents no behaviour.
    return x.dtype.kind in "u" or (
        x.dtype.kind == "f" and x.size and x.min() >= 0 and x.max() <= 1
    )


def _normalize_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    method = draw(st.sampled_from(["minmax", "zscore", "preset"]), label="method")
    if method != "preset":
        return {"method": method}
    c = x.shape[2]
    return {
        "method": method,
        "mean": draw(st.lists(_nice(0, 1), min_size=c, max_size=c), label="mean"),
        "std": draw(st.lists(_nice(0.125, 1), min_size=c, max_size=c), label="std"),
    }


@quiet
def _normalize_ref(x: np.ndarray, p: Params) -> np.ndarray:
    xf = x.astype(np.float64)
    if p["method"] == "minmax":
        out = (xf - xf.min()) / (xf.max() - xf.min())
    elif p["method"] == "zscore":
        out = (xf - xf.mean()) / xf.std()
    else:
        out = (xf - np.asarray(p["mean"])) / np.asarray(p["std"])
    return out.astype(np.float32)


def _normalize_ref_accepts(x: np.ndarray, p: Params) -> bool:
    # A constant image has no range or spread; the op documents no result.
    return p["method"] == "preset" or bool(x.size and x.max() != x.min())


def _normalize_tol(x: np.ndarray, p: Params) -> Tol:
    if p["method"] == "preset":
        return close(atol=1e-6, rtol=1e-5)
    # Dividing by a small range or spread amplifies f32 accumulation error.
    xf = x.astype(np.float64)
    spread = (xf.max() - xf.min()) if p["method"] == "minmax" else xf.std()
    return close(atol=1e-5 * magnitude(x) / max(spread, 1e-30), rtol=1e-5)


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

_LUMA = np.array([0.299, 0.587, 0.114])


def _grayscale_ref(x: np.ndarray, p: Params) -> np.ndarray:
    c = x.shape[2]
    if c in (1, 2):
        return x[:, :, :1].copy()
    luma = x[:, :, :3].astype(np.float64) @ _LUMA
    return to_dtype(luma, x.dtype)[:, :, None]


def _grayscale_tol(x: np.ndarray, p: Params) -> Tol:
    if x.shape[2] in (1, 2):
        return EXACT
    if not is_int(x):
        return close(atol=1e-6 * magnitude(x), rtol=1e-6)
    # The luma is computed in f64 (an integer's accumulator) on both sides,
    # in different orders: beyond 2**53 that is a few f64 spacings apart.
    return lsb(max(1.0, 4 * float(np.spacing(float(magnitude(x))))))


def _to_bgr_ref(x: np.ndarray, p: Params) -> np.ndarray:
    return color_channels(lambda rgb: rgb[:, :, ::-1].copy(), x)


def _ranged(x: np.ndarray) -> bool:
    """The dtypes with a colour range: unsigned integers (0..MAX) and floats
    ([0, 1]); the ranged conversions refuse signed integers."""
    return x.dtype.kind in "uf"


@quiet
def _hsv_definition(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HSV by its definition, in float64: hue in degrees [0, 360), S in
    [0, 1], V in the input's units."""
    r, g, b = (rgb[:, :, i].astype(np.float64) for i in range(3))
    v = np.maximum(np.maximum(r, g), b)
    diff = v - np.minimum(np.minimum(r, g), b)
    s = np.where(v == 0, 0.0, diff / np.where(v == 0, 1.0, v))
    d = np.where(diff == 0, 1.0, diff)
    hue = np.select(
        [diff == 0, v == r, v == g],
        [0.0, 60.0 * (g - b) / d, 60.0 * (b - r) / d + 120.0],
        60.0 * (r - g) / d + 240.0,
    )
    return np.where(hue < 0, hue + 360.0, hue), s, v


def _hsv_ref(x: np.ndarray, p: Params) -> np.ndarray:
    """OpenCV's 8-bit HSV (H in half-degrees); for u16 and floats HSV by its
    definition (OpenCV's float HSV divides by ``v + FLT_EPSILON``, which is
    visible on dark u16 pixels): float H in degrees with S, V in [0, 1], u16
    over 0..65535 with the hue covering one turn (``HSV_FULL``'s scheme),
    wrapped at 65536."""

    def hsv(rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype == np.uint8:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, val = _hsv_definition(rgb)
        if rgb.dtype.kind == "f":
            return np.stack([hue, sat, val], axis=2).astype(rgb.dtype)
        full = float(np.iinfo(rgb.dtype).max)
        hue = round_half_away(hue * (full + 1) / 360.0)
        hue = np.where(hue >= full + 1, hue - (full + 1), hue)
        return to_dtype(np.stack([hue, sat * full, val], axis=2), rgb.dtype)

    return color_channels(hsv, x)


@quiet
def _ycbcr_ref(x: np.ndarray, p: Params) -> np.ndarray:
    """Full-range BT.601 (JFIF) by its definition: ``Cb = (B - Y) / 1.772``
    and ``Cr = (R - Y) / 1.402`` about the dtype's chroma centre (128 for
    u8, 32768 for u16, 0.5 for floats). OpenCV rounds those factors to
    0.564 and 0.713, a 1e-4 difference of full scale."""
    off = 0.5 if x.dtype.kind == "f" else (float(np.iinfo(x.dtype).max) + 1) / 2

    def ycbcr(rgb: np.ndarray) -> np.ndarray:
        r, g, b = (rgb[:, :, i].astype(np.float64) for i in range(3))
        y = 0.299 * r + 0.587 * g + 0.114 * b
        # (B - Y) / 1.772 and (R - Y) / 1.402 expanded, so an infinite
        # channel gives the infinite limit, not inf - inf.
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        planes = np.stack([y, off + cb, off + cr], axis=2)
        return to_dtype(planes, rgb.dtype)

    return color_channels(ycbcr, x)


def _lab_ref(x: np.ndarray, p: Params) -> np.ndarray:
    """OpenCV's float Lab of the image scaled to [0, 1] (OpenCV has no
    16-bit Lab)."""
    scale = 1.0 if x.dtype.kind == "f" else float(np.iinfo(x.dtype).max)
    rgb = (x[:, :, :3].astype(np.float64) / scale).astype(np.float32)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)


def _color_ref_dtype(x: np.ndarray) -> bool:
    """What OpenCV's colour conversions take (8/16-bit and f32)."""
    return (
        x.ndim == 3
        and x.shape[2] in (3, 4)
        and x.dtype in (np.uint8, np.uint16, np.float32)
    )


def _unit(x: np.ndarray) -> bool:
    """A float image within [0, 1], a float colour's range (outside it,
    OpenCV's float HSV and Lab extrapolate differently from the engine)."""
    return x.dtype.kind != "f" or bool(np.all((x[:, :, :3] >= 0) & (x[:, :, :3] <= 1)))


def _hsv_tol(x: np.ndarray, p: Params) -> Tol:
    # Hue is undefined on gray pixels and wraps, hence the sparse allowance.
    if x.dtype == np.uint8:
        return sparse(atol=2, frac=0.005, frac_atol=180)
    if x.dtype == np.uint16:
        return sparse(atol=2, frac=0.005, frac_atol=65536)
    return sparse(atol=1e-3 * magnitude(x), frac=0.005, frac_atol=360)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

_PIL_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "catmullrom": Image.Resampling.BICUBIC,  # Pillow's bicubic is a = -0.5
    "lanczos3": Image.Resampling.LANCZOS,
}
_FILTERS = (*_PIL_FILTERS, "gaussian")


#: Each resampling kernel's L1 norm, ``integral |k(x)| dx``: how much a
#: per-tap rounding error can add up to through the filter (the negative
#: lobes count). Computed numerically from the kernels' definitions.
_FILTER_L1 = {"bilinear": 1.0, "catmullrom": 1.1667, "lanczos3": 1.3544}


def _pil_resize_plane(
    plane: np.ndarray, height: int, width: int, flt: str
) -> np.ndarray:
    """Resize one plane with Pillow, height pass first (the engine's order).

    Pillow resamples width first when both sizes change, and rounds an 8-bit
    image to 8 bits between the passes, as the engine does — so the pass order
    is observable (up to 43 levels on noise, measured) and is modelled here
    rather than absorbed into the tolerance.
    """
    mode = "L" if plane.dtype == np.uint8 else "F"
    image = Image.fromarray(
        plane.astype(np.float32) if mode == "F" else plane, mode=mode
    )
    h0, w0 = plane.shape
    resample = _PIL_FILTERS[flt]
    if height != h0:
        image = image.resize((w0, height), resample)
    if width != w0:
        image = image.resize((width, height), resample)
    return np.asarray(image)


def _nearest_indices(src: int, dst: int) -> np.ndarray:
    """Nearest-neighbour source index per output pixel, in exact arithmetic.

    Output pixel ``i`` samples the source pixel under its centre,
    ``floor((i + 0.5) * src / dst)``; a centre exactly on a source pixel
    boundary takes the pixel after it. The engine computes the same integers
    (``view-buffer``'s ``resample::nearest_indices``), so nearest is exact.
    """
    i = np.arange(dst)
    return ((2 * i + 1) * src) // (2 * dst)


def _pil_resize_premultiplied(
    x: np.ndarray, height: int, width: int, flt: str
) -> np.ndarray:
    """An 8-bit gray+alpha / RGBA resize with colour weighted by alpha.

    The engine resamples an image with alpha premultiplied — so a transparent
    pixel contributes no colour — as Pillow does for ``LA``/``RGBA``. Pillow's
    ``La``/``RGBa`` modes are the premultiplied forms; converting once and
    resizing in them keeps the two passes premultiplied throughout.

    A pixel whose resampled alpha is 0 has no colour: the engine
    un-premultiplies it to 0, where Pillow leaves the residue of its
    neighbours' ringing. The reference takes the engine's convention for
    exactly those pixels.
    """
    h0, w0 = x.shape[:2]
    if (height, width) == (h0, w0):
        return x.copy()  # nothing resampled, nothing premultiplied
    mode, pre = ("LA", "La") if x.shape[2] == 2 else ("RGBA", "RGBa")
    image = Image.fromarray(x, mode=mode).convert(pre)
    resample = _PIL_FILTERS[flt]
    if height != h0:
        image = image.resize((w0, height), resample)
    if width != w0:
        image = image.resize((width, height), resample)
    out = np.asarray(image.convert(mode)).copy()
    out[out[:, :, -1] == 0, :-1] = 0
    return out


def _resize_to(x: np.ndarray, height: int, width: int, flt: str) -> np.ndarray:
    if flt == "nearest":
        rows = _nearest_indices(x.shape[0], height)
        cols = _nearest_indices(x.shape[1], width)
        return x[rows][:, cols].copy()
    if x.dtype == np.uint8 and x.shape[2] in (2, 4):
        return _pil_resize_premultiplied(x, height, width, flt)
    out = per_channel(lambda pl_: _pil_resize_plane(pl_, height, width, flt), x)
    return out.astype(x.dtype)


def _resamplable(x: np.ndarray) -> bool:
    """The resampler's contract: ``[H, W, C]`` with at most four channels
    (view-buffer/src/ops/image.rs, "at most 4 channels for resampling")."""
    return is_rank3(x) and x.shape[2] <= 4


def _resize_ref_accepts(x: np.ndarray, p: Params) -> bool:
    if p.get("filter") not in _PIL_FILTERS:
        return False
    if p["filter"] == "nearest":
        return True  # pure data movement: every dtype, every channel count
    # Pillow's float mode has no alpha: premultiplied f32 is not modelled.
    return x.dtype == np.uint8 or (x.dtype == np.float32 and x.shape[2] in (1, 3))


def _resize_tol(x: np.ndarray, p: Params) -> Tol:
    flt = p.get("filter")
    if flt == "nearest":
        return EXACT
    if x.dtype == np.uint8 and x.shape[2] in (2, 4):
        # Both sides resample premultiplied, in 8 bits. Un-premultiplied, each
        # side's rounding is scaled by 255 / alpha (3.4x at alpha 74), so no
        # dense straight-colour bound holds on a translucent output. Compared
        # premultiplied, each side is off by at most: its premultiply's
        # rounding through the filter (1/2 * the kernel's L1 norm), the
        # resample's own rounding (1/2) and the straight output's (1/2, no
        # more once re-premultiplied): 2 * (L1 / 2 + 1) between them.
        return Tol(atol=2 * (_FILTER_L1[flt] / 2 + 1), space="premultiplied")
    if x.dtype == np.float32:
        return close(atol=1e-4 * magnitude(x), rtol=1e-4)
    # Each fixed-point resampler rounds within 1 of the exact resample, and two
    # correct ones can round to opposite sides of it (F11: bilinear, engine 34,
    # Pillow 36, exact 35.0) -- 2 apart for every filter.
    return lsb(2)


def _size_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {
        "height": draw(st.integers(1, 40), label="height"),
        "width": draw(st.integers(1, 40), label="width"),
        "filter": draw(st.sampled_from(_FILTERS), label="filter"),
    }


def _resize_ref(x: np.ndarray, p: Params) -> np.ndarray:
    return _resize_to(x, p["height"], p["width"], p["filter"])


def _scaled(size: int, factor: Fraction | float) -> int:
    """The engine's derived-size rule, exactly: round half up, at least one
    pixel. A ratio comes as a ``Fraction``; a scale factor as the f32 the op
    receives."""
    exact = (
        factor if isinstance(factor, Fraction) else Fraction(float(np.float32(factor)))
    )
    return max(1, math.floor(size * exact + Fraction(1, 2)))


def _resize_to_height_ref(x: np.ndarray, p: Params) -> np.ndarray:
    h, w = x.shape[:2]
    return _resize_to(x, p["height"], _scaled(w, Fraction(p["height"], h)), p["filter"])


def _resize_to_width_ref(x: np.ndarray, p: Params) -> np.ndarray:
    h, w = x.shape[:2]
    return _resize_to(x, _scaled(h, Fraction(p["width"], w)), p["width"], p["filter"])


def _resize_max_ref(x: np.ndarray, p: Params) -> np.ndarray:
    h, w = x.shape[:2]
    s = Fraction(p["max_size"], max(h, w))
    return _resize_to(x, _scaled(h, s), _scaled(w, s), p["filter"])


def _resize_min_ref(x: np.ndarray, p: Params) -> np.ndarray:
    h, w = x.shape[:2]
    s = Fraction(p["min_size"], min(h, w))
    return _resize_to(x, _scaled(h, s), _scaled(w, s), p["filter"])


def _resize_scale_ref(x: np.ndarray, p: Params) -> np.ndarray:
    h, w = x.shape[:2]
    return _resize_to(
        x, _scaled(h, p["scale_y"]), _scaled(w, p["scale_x"]), p["filter"]
    )


def _one_size_params(key: str, lo: int = 1, hi: int = 40) -> ParamStrategy:
    def draw_params(draw: st.DrawFn, x: np.ndarray) -> Params:
        return {
            key: draw(st.integers(lo, hi), label=key),
            "filter": draw(st.sampled_from(_FILTERS), label="filter"),
        }

    return draw_params


def _resize_scale_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    factors = st.sampled_from([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    return {
        "scale_x": draw(factors, label="scale_x"),
        "scale_y": draw(factors, label="scale_y"),
        "filter": draw(st.sampled_from(_FILTERS), label="filter"),
    }


def _letterbox_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {
        "height": draw(st.integers(1, 40), label="height"),
        "width": draw(st.integers(1, 40), label="width"),
        "value": draw(_fill_value(x), label="value"),
        "filter": draw(st.sampled_from(_FILTERS), label="filter"),
    }


def _letterbox_ref(x: np.ndarray, p: Params) -> np.ndarray:
    h, w = x.shape[:2]
    s = min(Fraction(p["height"], h), Fraction(p["width"], w))
    nh, nw = min(p["height"], _scaled(h, s)), min(p["width"], _scaled(w, s))
    inner = _resize_to(x, nh, nw, p["filter"])
    return _pad_to_size_ref(
        inner,
        {
            "height": p["height"],
            "width": p["width"],
            "position": "center",
            "value": p["value"],
        },
    )


# ---------------------------------------------------------------------------
# Geometry (affine)
# ---------------------------------------------------------------------------

_CV_INTERP = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR}
_CV_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def _warp(
    x: np.ndarray,
    matrix: list[float],
    size: tuple[int, int],
    interp: str,
    border: float,
) -> np.ndarray:
    h, w = size
    m = np.array(matrix, dtype=np.float64).reshape(2, 3)
    planes = [
        cv2.warpAffine(
            np.ascontiguousarray(x[:, :, c]),
            m,
            (w, h),
            flags=_CV_INTERP[interp],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
        for c in range(x.shape[2])
    ]
    return np.stack(planes, axis=2).astype(x.dtype)


def _warp_ref_accepts(x: np.ndarray, p: Params) -> bool:
    return x.dtype.type in _CV_DTYPES


def _rotate_ref_accepts(x: np.ndarray, p: Params) -> bool:
    """``rotate`` alone turns a quarter turn into data movement (every dtype);
    the other warps are warps at any angle, whatever their arguments."""
    return _is_quarter_turn(p) or _warp_ref_accepts(x, p)


def _rotate_tol(x: np.ndarray, p: Params) -> Tol:
    return EXACT if _is_quarter_turn(p) else _warp_tol(x, p)


def _warp_tol(x: np.ndarray, p: Params) -> Tol:
    # OpenCV computes interpolation weights in fixed point (5 bits); the
    # engine in float. A pixel whose source footprint straddles the image
    # edge blends with the border value, where those differences are
    # largest, so the sparse allowance is the output's border share plus a
    # little. Nearest is exact away from the edge (the parameters are drawn
    # off the half-pixel lattice); at the edge, in or out is a coin toss.
    h, w = p.get("output_size", x.shape[:2])
    border = min(1.0, 0.03 + 2 * (h + w) / (h * w))
    if p.get("interpolation", "bilinear") == "nearest":
        return sparse(atol=0, frac=border, frac_atol=math.inf)
    # 5-bit weights misplace a sample by up to 1/64 pixel, so the error is
    # bounded by the steepest step in the image over 32, plus rounding.
    xf = x.astype(np.float64)
    steep = 0.0
    if x.shape[0] > 1:
        steep = max(steep, float(np.abs(np.diff(xf, axis=0)).max()))
    if x.shape[1] > 1:
        steep = max(steep, float(np.abs(np.diff(xf, axis=1)).max()))
    # The image's edge is a step to the border value too.
    fill = float(p.get("border_value", 0.0))
    if x.size:
        steep = max(steep, float(np.abs(xf - fill).max()))
    rounding = 1.0 if is_int(x) else 1e-6 * magnitude(x)
    return sparse(atol=rounding + steep / 32, frac=border, frac_atol=math.inf)


#: A small non-dyadic offset that keeps drawn geometry off exact pixel ties.
_OFF_LATTICE = 0.0137

#: Shear factors with no exact binary form, so no sample lands on a tie.
_SHEARS = st.integers(-9, 9).map(lambda k: k / 10.0 + (0.013 if k else 0.0))


def _rotation(angle: float, cx: float, cy: float, scale: float = 1.0) -> list[float]:
    # Clockwise for positive angles: OpenCV's getRotationMatrix2D with -angle.
    m = cv2.getRotationMatrix2D((cx, cy), -angle, scale)
    return [float(v) for v in m.ravel()]


def _rotate_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    if draw(st.booleans(), label="quarter turn"):
        return _rot90_params(draw, x)
    return {
        "angle": draw(_nice(-180, 180).filter(lambda a: a % 90 != 0), label="angle"),
        "interpolation": draw(st.sampled_from(list(_CV_INTERP)), label="interpolation"),
        "border_value": draw(_fill_value(x), label="border_value"),
    }


def _is_quarter_turn(p: Params) -> bool:
    return float(p["angle"]) % 90 == 0


def _rotate_ref(x: np.ndarray, p: Params) -> np.ndarray:
    if _is_quarter_turn(p):
        return _rot90_ref(x, p)
    h, w = x.shape[:2]
    m = _rotation(p["angle"], w / 2.0, h / 2.0)
    return _warp(x, m, (h, w), p["interpolation"], p["border_value"])


def _warp_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    h, w = x.shape[:2]
    angle = draw(_nice(-180, 180), label="angle")
    scale = draw(st.sampled_from([0.5, 0.75, 1.0, 1.25, 2.0]), label="scale")
    m = _rotation(angle, w / 2.0, h / 2.0, scale)
    # Off the half-pixel lattice: a dyadic offset puts nearest-neighbour
    # samples exactly on ties, where the side taken is rounding noise.
    m[2] += draw(_nice(-4, 4), label="tx") + _OFF_LATTICE
    m[5] += draw(_nice(-4, 4), label="ty") + _OFF_LATTICE
    return {
        "matrix": m,
        "output_size": (draw(st.integers(1, 40)), draw(st.integers(1, 40))),
        "interpolation": draw(st.sampled_from(list(_CV_INTERP)), label="interpolation"),
        "border_value": draw(_fill_value(x), label="border_value"),
    }


def _warp_ref(x: np.ndarray, p: Params) -> np.ndarray:
    return _warp(
        x, p["matrix"], p["output_size"], p["interpolation"], p["border_value"]
    )


def _shear_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {
        "sx": draw(_SHEARS, label="sx"),
        "sy": draw(_SHEARS, label="sy"),
        "output_size": (draw(st.integers(1, 40)), draw(st.integers(1, 40))),
    }


def _shear_ref(x: np.ndarray, p: Params) -> np.ndarray:
    return _warp(
        x, [1.0, p["sx"], 0.0, p["sy"], 1.0, 0.0], p["output_size"], "bilinear", 0.0
    )


def _rotate_and_scale_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    h, w = x.shape[:2]
    return {
        "angle": draw(_nice(-180, 180), label="angle"),
        "center": (
            draw(_nice(0, w), label="cx") + _OFF_LATTICE,
            draw(_nice(0, h), label="cy") + _OFF_LATTICE,
        ),
        "output_size": (draw(st.integers(1, 40)), draw(st.integers(1, 40))),
        "scale": draw(st.sampled_from([0.5, 1.0, 1.5, 2.0]), label="scale"),
    }


def _rotate_and_scale_ref(x: np.ndarray, p: Params) -> np.ndarray:
    cx, cy = p["center"]
    m = _rotation(p["angle"], cx, cy, p["scale"])
    return _warp(x, m, p["output_size"], "bilinear", 0.0)


# ---------------------------------------------------------------------------
# Neighbourhood filters
# ---------------------------------------------------------------------------

_SIGMAS = (0.5, 0.8, 1.0, 1.5, 2.0, 3.0)


def _blur_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"sigma": draw(st.sampled_from(_SIGMAS), label="sigma")}


def _blur_ref(x: np.ndarray, p: Params) -> np.ndarray:
    sigma = float(np.float32(p["sigma"]))  # the op's sigma is an f32
    k = 2 * math.ceil(3 * sigma) + 1
    planes = per_channel(
        lambda plane: cv2.GaussianBlur(
            plane.astype(np.float64), (k, k), sigma, borderType=cv2.BORDER_REPLICATE
        ),
        x,
    )
    return to_dtype(planes, x.dtype)


def _blur_tol(x: np.ndarray, p: Params) -> Tol:
    own = lsb(1) if is_int(x) else close(atol=1e-5 * magnitude(x), rtol=1e-5)
    return _accumulated(x, own)


def _single_channel(x: np.ndarray) -> bool:
    return x.ndim == 3 and x.shape[2] == 1


def _morph_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {
        "ksize": draw(st.sampled_from([1, 3, 5]), label="ksize"),
        "iterations": draw(st.integers(1, 3), label="iterations"),
    }


def _ksize_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"ksize": draw(st.sampled_from([1, 3, 5]), label="ksize")}


def _window_extreme(
    plane: np.ndarray, k: int, reduce: Callable[..., np.ndarray]
) -> np.ndarray:
    """Min or max over each ``k x k`` window, edges replicated.

    Written on NumPy windows rather than ``scipy.ndimage.grey_erosion``,
    which goes through float64 and so is wrong for 64-bit integers.
    """
    r = k // 2
    padded = np.pad(plane, r, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    return reduce(windows, axis=(-2, -1)).astype(plane.dtype)


def _erode(x: np.ndarray, k: int, n: int = 1) -> np.ndarray:
    out = x[:, :, 0]
    for _ in range(n):
        out = _window_extreme(out, k, np.min)
    return out[:, :, None]


def _dilate(x: np.ndarray, k: int, n: int = 1) -> np.ndarray:
    out = x[:, :, 0]
    for _ in range(n):
        out = _window_extreme(out, k, np.max)
    return out[:, :, None]


#: convolve2d's border names in SciPy's spelling. Its "reflect" repeats the
#: edge pixel (d c b a | a b c d, SciPy "reflect") — unlike pad(mode=
#: "reflect"), which mirrors about it (NumPy "reflect", no repeat); pad calls
#: the repeating form "symmetric".
def _saturating_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    info = np.iinfo(a.dtype) if a.dtype.kind in "iu" else None
    diff = a.astype(object) - b.astype(object)
    if info is not None:
        diff = np.clip(diff, info.min, info.max)
    return diff.astype(a.dtype)


_BORDERS = {"replicate": "nearest", "zero": "constant", "reflect": "reflect"}


def _kernel_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    side = draw(st.sampled_from([1, 3, 5]), label="side")
    kernel = draw(
        st.lists(_nice(-2, 2), min_size=side * side, max_size=side * side),
        label="kernel",
    )
    return {
        "kernel": kernel,
        "normalize": draw(st.booleans(), label="normalize"),
        "border": draw(st.sampled_from(list(_BORDERS)), label="border"),
    }


@quiet
def _correlate(
    x: np.ndarray, kernel: list[float], *, normalize: bool, border: str
) -> np.ndarray:
    side = int(round(math.sqrt(len(kernel))))
    k = np.asarray(kernel, dtype=np.float64).reshape(side, side)
    if normalize:
        k = k / np.abs(k).sum()
    out = per_channel(
        lambda plane: ndimage.correlate(
            plane.astype(np.float64), k, mode=_BORDERS[border], cval=0.0
        ),
        x,
    )
    return out.astype(float_out(x))


def _convolve_ref(x: np.ndarray, p: Params) -> np.ndarray:
    return _correlate(x, p["kernel"], normalize=p["normalize"], border=p["border"])


def _convolve_ref_accepts(x: np.ndarray, p: Params) -> bool:
    return not (p.get("normalize") and not any(p["kernel"]))


def _convolve_tol(x: np.ndarray, p: Params) -> Tol:
    kernel = np.abs(np.asarray(p.get("kernel", [1.0])))
    weight = 1.0 if p.get("normalize") else max(1.0, float(kernel.sum()))
    eps = float(np.finfo(float_out(x)).eps)
    return close(atol=16 * eps * weight * magnitude(x), rtol=16 * eps)


def _kernel_gain(_x: np.ndarray, p: Params) -> float:
    kernel = np.abs(np.asarray(p["kernel"], dtype=np.float64))
    return 1.0 if p["normalize"] else float(kernel.sum())


_SOBEL = {
    "x": [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
    "y": [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
}
_LAPLACIAN = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]


def _sharpen_kernel(s: float) -> list[float]:
    return [-s] * 4 + [1.0 + 8.0 * s] + [-s] * 4


def _canny_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    lo = draw(st.integers(0, 300), label="low_threshold")
    hi = draw(st.integers(0, 300), label="high_threshold")
    return {"low_threshold": float(lo), "high_threshold": float(hi)}


def _canny_ref(x: np.ndarray, p: Params) -> np.ndarray:
    c = x.shape[2]
    planes = x[:, :, :1] if c in (1, 2) else x[:, :, :3]
    image = planes[:, :, 0] if planes.shape[2] == 1 else np.ascontiguousarray(planes)
    edges = cv2.Canny(image, p["low_threshold"], p["high_threshold"])
    return edges[:, :, None]


def _equalize_ref(x: np.ndarray, p: Params) -> np.ndarray:
    return per_channel(cv2.equalizeHist, x)


# ---------------------------------------------------------------------------
# Terminal ops (reductions, histograms, shape)
# ---------------------------------------------------------------------------


def _scalar_result(fn: Callable[[np.ndarray, Params], float]) -> Reference:
    @quiet
    def ref(x: np.ndarray, p: Params) -> np.float64:
        return np.float64(fn(x, p))

    return ref


def _mean_tol(x: np.ndarray, p: Params) -> Tol:
    """A mean inherits its sum's summation-order error (up to ``n * eps *
    max|x|`` for a naive sum); dividing by ``n`` does not shrink that bound
    relative to the result, so it is taken as is."""
    eps = float(np.finfo(float_out(x)).eps) if x.dtype.kind == "f" else 2.0**-52
    return close(atol=eps * max(1, x.size) * magnitude(x), rtol=1e-12)


def _std_tol(x: np.ndarray, p: Params) -> Tol:
    """A standard deviation is the square root of a variance whose rounding
    error scales with ``eps * max|x|**2``; near zero spread the root turns
    that into ``sqrt(eps * n) * max|x|`` (a constant image of 2700s comes
    back with std 4e-11, not 0)."""
    eps = float(np.finfo(float_out(x)).eps) if x.dtype.kind == "f" else 2.0**-52
    return close(atol=math.sqrt(eps * max(1, x.size) * 16) * magnitude(x), rtol=1e-9)


def _sum_tol(x: np.ndarray, p: Params) -> Tol:
    """Summation order: a sum of ``n`` terms may lose up to ``n * eps * sum|x|``
    (the classical bound), which for 64-bit integers near 2**52 is hundreds."""
    eps = float(np.finfo(float_out(x)).eps) if x.dtype.kind == "f" else 2.0**-52
    total = float(np.abs(x.astype(np.float64)).sum()) if x.size else 0.0
    return close(atol=eps * max(1, x.size) * max(total, 1.0), rtol=1e-12)


def _popcount_ref(x: np.ndarray, p: Params) -> np.float64:
    return np.float64(np.unpackbits(np.ascontiguousarray(x).view(np.uint8)).sum())


def _percentile_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"q": draw(_nice(0, 100), label="q")}


def _std_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"ddof": draw(st.integers(0, 1), label="ddof")}


def _std_ref_accepts(x: np.ndarray, p: Params) -> bool:
    return x.size > p.get("ddof", 0)


def _axis_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {"axis": draw(st.integers(0, x.ndim - 1), label="axis")}


def _axis_reduce(
    fn: Callable[..., np.ndarray], dtype: Callable[[np.ndarray], np.dtype]
) -> Reference:
    @quiet
    def ref(x: np.ndarray, p: Params) -> np.ndarray:
        return np.asarray(fn(x, axis=p["axis"])).astype(dtype(x))

    return ref


def _histogram_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    params: Params = {
        "output": draw(
            st.sampled_from(["counts", "normalized", "edges"]), label="output"
        ),
        "closed": draw(st.sampled_from(["left", "right"]), label="closed"),
    }
    if draw(st.booleans(), label="explicit_edges"):
        # Non-decreasing, ties allowed, the outer edges possibly open.
        inner = sorted(
            draw(st.lists(_nice(-8, 160), min_size=1, max_size=8), label="edges")
        )
        if draw(st.booleans(), label="tie") and inner:
            inner.insert(0, inner[0])
        lo = [-math.inf] if draw(st.booleans(), label="open_below") else []
        hi = [math.inf] if draw(st.booleans(), label="open_above") else []
        edges = lo + inner + hi
        if len(edges) < 2:
            edges.append(edges[-1] + 1.0)
        params["bins"] = edges
        return params
    params["bins"] = draw(st.integers(1, 16), label="bins")
    # Auto range over NaN or infinity is an error (numpy's rule, and the
    # engine's): there are no equal-width bins to make.
    if _non_finite(x) or draw(st.booleans(), label="explicit_range"):
        lo = draw(_nice(-4, 128), label="lo")
        params["range"] = (lo, lo + draw(_nice(1, 256), label="span"))
    return params


def _auto_range(values: list) -> tuple[float, float]:
    """The detected range, widened outward to hold every value: an integer
    beyond 2**53 can round inward to f64."""
    vmin, vmax = min(values), max(values)
    lo, hi = float(vmin), float(vmax)
    while lo > vmin:  # Python compares an int with a float exactly
        lo = math.nextafter(lo, -math.inf)
    while hi < vmax:
        hi = math.nextafter(hi, math.inf)
    if lo == hi:  # numpy widens a zero-width range by 0.5 each way
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def _histogram_ref(x: np.ndarray, p: Params) -> np.ndarray:
    """``numpy.histogram``'s rule, compared exactly: Python ints for integer
    pixels (numpy rounds a pixel beyond 2**53 to f64 first). A value outside
    the edges, or NaN, is in no bin."""
    values = x.ravel().tolist()  # Python ints / floats, exact
    if isinstance(p["bins"], list):
        edges = [float(e) for e in p["bins"]]
    else:
        lo, hi = p.get("range") or _auto_range(values)
        edges = np.linspace(lo, hi, p["bins"] + 1).tolist()
        edges[-1] = hi
    if p["output"] == "edges":
        return np.asarray(edges, np.float64)
    n = len(edges) - 1
    counts = np.zeros(n, np.uint64)
    for v in values:
        if v != v or not edges[0] <= v <= edges[-1]:
            continue  # NaN, or outside: in no bin
        if p.get("closed", "left") == "left":  # [e_i, e_i+1), the last closed
            b = min(bisect.bisect_right(edges, v) - 1, n - 1)
        else:  # (e_i, e_i+1], the first closed
            b = max(bisect.bisect_left(edges, v) - 1, 0)
        counts[b] += 1
    if p["output"] == "normalized":
        total = int(counts.sum())
        return counts / total if total else counts.astype(np.float64)
    return counts


def _histogram_ref_accepts(x: np.ndarray, p: Params) -> bool:
    return True


def _extract_shape_ref(x: np.ndarray, p: Params) -> np.ndarray:
    return np.asarray(x.shape, dtype=np.float64)


# ---------------------------------------------------------------------------
# Contour-domain tails (no reference here: invariance only)
# ---------------------------------------------------------------------------

_CONTOUR_NO_REF = (
    "contour geometry is checked by its own differential suite "
    "(test_contour_raster_crosscheck.py) and the contour reference tests; "
    "here it is exercised for execution invariance only"
)


def _extract_contours_params(draw: st.DrawFn, x: np.ndarray) -> Params:
    return {
        "mode": draw(st.sampled_from(["external", "all"]), label="mode"),
        "method": draw(st.sampled_from(["none", "simple", "approx"]), label="method"),
    }


def _mask_like(x: np.ndarray) -> bool:
    return is_u8(x) and _single_channel(x) and x.shape[0] * x.shape[1] <= 32 * 32


def _translate_params(draw: st.DrawFn, x: Any) -> Params:
    return {"dx": draw(_nice(-8, 8), label="dx"), "dy": draw(_nice(-8, 8), label="dy")}


def _scale_contour_params(draw: st.DrawFn, x: Any) -> Params:
    return {
        "sx": draw(_nice(0.25, 3), label="sx"),
        "sy": draw(_nice(0.25, 3), label="sy"),
        "origin": draw(
            st.sampled_from(["centroid", "bbox_center", "origin"]), label="origin"
        ),
    }


def _rasterize_params(draw: st.DrawFn, x: Any) -> Params:
    return {
        "width": draw(st.integers(1, 32), label="width"),
        "height": draw(st.integers(1, 32), label="height"),
    }


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

_movement = {"kind": "movement", "gain": 1.0}
_float_scalar = {"tol": _scalar_tol, "kind": "pointwise"}
_INF = math.inf


def _spec(method: str, params: ParamStrategy = _no_params, **kw: Any) -> OpSpec:
    return OpSpec(method=method, params=params, **kw)


OPS: dict[str, OpSpec] = {
    spec.method: spec
    for spec in (
        # --- data movement ---------------------------------------------------
        _spec("crop", _crop_params, accepts=is_rank3, ref=_crop_ref, **_movement),
        _spec(
            "flip",
            _flip_params,
            ref=lambda x, p: np.flip(x, p["axes"]).copy(),
            **_movement,
        ),
        _spec(
            "flip_h", ref=lambda x, p: x[:, ::-1].copy(), accepts=is_rank3, **_movement
        ),
        _spec("flip_v", ref=lambda x, p: x[::-1].copy(), accepts=is_rank3, **_movement),
        _spec(
            "transpose",
            _transpose_params,
            ref=lambda x, p: np.ascontiguousarray(np.transpose(x, p["axes"])),
            **_movement,
        ),
        _spec("pad", _pad_params, accepts=is_rank3, ref=_pad_ref, **_movement),
        _spec(
            "pad_to_size",
            _pad_to_size_params,
            uniform_rows=True,
            accepts=is_rank3,
            ref=_pad_to_size_ref,
            note="center places the odd pixel after the content (top = extra // 2)",
            **_movement,
        ),
        _spec(
            "channel_select",
            _channel_select_params,
            accepts=is_rank3,
            ref=lambda x, p: x[:, :, p["index"]].copy(),
            **_movement,
        ),
        _spec(
            "channel_swap",
            _channel_swap_params,
            accepts=is_rank3,
            ref=lambda x, p: x[:, :, p["order"]].copy(),
            **_movement,
        ),
        _spec(
            "reshape",
            _reshape_params,
            uniform_rows=True,
            accepts=is_rank3,
            ref=lambda x, p: x.reshape(p["shape"]).copy(),
            **_movement,
        ),
        _spec(
            "assert_shape",
            _assert_shape_params,
            uniform_rows=True,
            ref=lambda x, p: x.copy(),
            note="the exact shape the row has: the op must be the identity",
            **_movement,
        ),
        # --- per-value --------------------------------------------------------
        _spec(
            "cast",
            _cast_params,
            ref=_cast_ref,
            ref_accepts=_cast_ref_accepts,
            gain=_cast_gain,
            note="view-buffer/src/core/convert.rs: float -> int rounds half "
            "away from zero and saturates (NaN -> 0); int -> int is a plain "
            "`as`, so narrowing wraps",
        ),
        _spec("invert", ref=_invert_ref, gain=1.0),
        _spec(
            "threshold",
            _threshold_params,
            accepts=lambda x: x.ndim == 2 or (x.ndim == 3 and x.shape[2] == 1),
            ref=_threshold_ref,
            gain=_INF,
            note="thresholds are drawn off the data's lattice (k + 0.5 for "
            "integers, multiples of 1/8 for floats) so no pixel ties",
        ),
        _spec("abs", ref=_scalar(lambda x, p: np.abs(x)), **_float_scalar),
        _spec("neg", ref=_scalar(lambda x, p: -x), **_float_scalar),
        _spec(
            "add_constant",
            _value_params,
            ref=_scalar(lambda x, p: x + x.dtype.type(p["value"])),
            **_float_scalar,
        ),
        _spec(
            "subtract_constant",
            _value_params,
            ref=_scalar(lambda x, p: x - x.dtype.type(p["value"])),
            **_float_scalar,
        ),
        _spec("ceil", ref=_scalar(lambda x, p: np.ceil(x)), gain=_INF, **_float_scalar),
        _spec(
            "floor", ref=_scalar(lambda x, p: np.floor(x)), gain=_INF, **_float_scalar
        ),
        _spec(
            "round",
            ref=_scalar(lambda x, p: np.round(x)),
            gain=_INF,
            note="ties to even",
            **_float_scalar,
        ),
        _spec(
            "trunc", ref=_scalar(lambda x, p: np.trunc(x)), gain=_INF, **_float_scalar
        ),
        _spec("sign", ref=_scalar(lambda x, p: np.sign(x)), gain=_INF, **_float_scalar),
        _spec("sqrt", ref=_scalar(lambda x, p: np.sqrt(x)), gain=_INF, **_float_scalar),
        _spec("square", ref=_scalar(lambda x, p: x * x), gain=_INF, **_float_scalar),
        _spec(
            "reciprocal",
            ref=_scalar(lambda x, p: x.dtype.type(1) / x),
            gain=_INF,
            **_float_scalar,
        ),
        _spec(
            "relu",
            ref=_scalar(lambda x, p: np.maximum(x, x.dtype.type(0))),
            **_float_scalar,
        ),
        _spec(
            "clamp_min",
            _value_params,
            ref=_scalar(lambda x, p: np.maximum(x, x.dtype.type(p["value"]))),
            **_float_scalar,
        ),
        _spec(
            "clamp_max",
            _value_params,
            ref=_scalar(lambda x, p: np.minimum(x, x.dtype.type(p["value"]))),
            **_float_scalar,
        ),
        _spec(
            "clamp",
            _clamp_params,
            ref=_scalar(
                lambda x, p: np.clip(
                    x, x.dtype.type(p["min_val"]), x.dtype.type(p["max_val"])
                )
            ),
            **_float_scalar,
        ),
        _spec(
            "scale",
            _factor_params,
            ref=_scalar(lambda x, p: x * x.dtype.type(p["factor"])),
            gain=lambda _x, p: abs(p["factor"]),
            **_float_scalar,
        ),
        _spec(
            "adjust_brightness",
            _brightness_params,
            ref=_scalar(lambda x, p: np.clip(x * x.dtype.type(p["factor"]), 0, 255)),
            gain=lambda _x, p: abs(p["factor"]),
            note="scale then clamp to [0, 255] whatever the dtype",
            **_float_scalar,
        ),
        _spec(
            "adjust_contrast",
            _factor_params,
            ref=_contrast_ref,
            tol=_stat_tol,
            kind="global",
            gain=lambda _x, p: 2 * abs(p["factor"]) + 1,
            note="mean over every element (all channels together)",
        ),
        _spec(
            "adjust_gamma",
            _gamma_params,
            ref=_gamma_ref,
            ref_accepts=_gamma_ref_accepts,
            ref_dtypes=("u8", "u16", "u32", "f32", "f64"),
            tol=lambda x, p: close(atol=1e-5 * dtype_max(x), rtol=1e-5),
            gain=_INF,
            note="normalised by the dtype's maximum for integers, 1 for floats",
        ),
        _spec(
            "normalize",
            _normalize_params,
            accepts=is_rank3,
            ref=_normalize_ref,
            ref_accepts=_normalize_ref_accepts,
            tol=_normalize_tol,
            kind="global",
            gain=_INF,
            note="minmax/zscore over every element; preset per channel on raw values",
        ),
        # --- colour -------------------------------------------------------------
        _spec(
            "grayscale",
            accepts=is_rank3,
            ref=_grayscale_ref,
            tol=_grayscale_tol,
            gain=1.0,
            note="0.299R + 0.587G + 0.114B; alpha dropped; 1/2-channel input "
            "keeps its gray plane",
        ),
        _spec(
            "to_bgr",
            accepts=lambda x: is_rank3(x) and x.shape[2] in (3, 4),
            ref=_to_bgr_ref,
            **_movement,
        ),
        _spec(
            "to_hsv",
            accepts=lambda x: is_rank3(x) and x.shape[2] in (3, 4) and _ranged(x),
            ref=_hsv_ref,
            ref_accepts=lambda x, p: _color_ref_dtype(x) and _unit(x),
            ref_dtypes=("u8", "u16", "f32"),
            tol=_hsv_tol,
            gain=_INF,
            note="OpenCV's HSV: u8 H in [0, 180), u16 H over 0..65535 for one "
            "turn, float H in degrees with S, V in [0, 1]; hue is undefined on "
            "gray pixels and wraps, hence the sparse allowance",
        ),
        _spec(
            "to_ycbcr",
            accepts=lambda x: is_rank3(x) and x.shape[2] in (3, 4) and _ranged(x),
            ref=_ycbcr_ref,
            ref_accepts=lambda x, p: _color_ref_dtype(x),
            ref_dtypes=("u8", "u16", "f32"),
            tol=lambda x, p: (
                lsb(1) if is_int(x) else close(atol=1e-5 * magnitude(x), rtol=1e-5)
            ),
            gain=1.0,
            note="full-range BT.601 (JFIF) by its definition; chroma centred "
            "at 128 (u8), 32768 (u16) and 0.5 (float)",
        ),
        _spec(
            "to_lab",
            accepts=lambda x: is_rank3(x) and x.shape[2] in (3, 4) and _ranged(x),
            ref=_lab_ref,
            ref_accepts=lambda x, p: (
                _color_ref_dtype(x) and x.shape[2] == 3 and _unit(x)
            ),
            ref_dtypes=("u8", "u16", "f32"),
            tol=close(atol=0.5, rtol=1e-3),
            gain=_INF,
            note="OpenCV's float Lab of RGB scaled to [0, 1] (L in [0, 100]); a* and b* "
            "agree to ~0.2 (OpenCV's float path approximates the sRGB "
            "curve), well under one just-noticeable difference",
        ),
        _spec(
            "convert_color",
            lambda draw, x: {
                "from_space": "rgb",
                "to_space": draw(st.sampled_from(["bgr", "gray"])),
            },
            accepts=lambda x: is_rank3(x) and x.shape[2] in (3, 4),
            ref=lambda x, p: (
                _to_bgr_ref(x, p)
                if p["to_space"] == "bgr"
                else color_channels(lambda rgb: _grayscale_ref(rgb, p), x)
            ),
            tol=lambda x, p: EXACT if p["to_space"] == "bgr" else _grayscale_tol(x, p),
            gain=1.0,
            note="the named spaces have their own entries (to_hsv, ...); the "
            "generic call is checked on the two it shares with them. Unlike "
            "grayscale() (SingleChannel, alpha dropped), rgb->gray is "
            "ColorChannels: RGBA comes back gray+alpha",
        ),
        # --- resampling ----------------------------------------------------------
        _spec(
            "resize",
            _size_params,
            accepts=_resamplable,
            ref=_resize_ref,
            ref_accepts=_resize_ref_accepts,
            ref_dtypes=("u8", "f32"),
            tol=_resize_tol,
            kind="spatial",
            note="Pillow per plane, height pass first; 'gaussian' has no "
            "Pillow counterpart",
        ),
        _spec(
            "resize_to_height",
            _one_size_params("height"),
            accepts=_resamplable,
            ref=_resize_to_height_ref,
            ref_accepts=_resize_ref_accepts,
            ref_dtypes=("u8", "f32"),
            tol=_resize_tol,
            kind="spatial",
            note="derived width rounds half up",
        ),
        _spec(
            "resize_to_width",
            _one_size_params("width"),
            accepts=_resamplable,
            ref=_resize_to_width_ref,
            ref_accepts=_resize_ref_accepts,
            ref_dtypes=("u8", "f32"),
            tol=_resize_tol,
            kind="spatial",
        ),
        _spec(
            "resize_max",
            _one_size_params("max_size"),
            accepts=_resamplable,
            ref=_resize_max_ref,
            ref_accepts=_resize_ref_accepts,
            ref_dtypes=("u8", "f32"),
            tol=_resize_tol,
            kind="spatial",
        ),
        _spec(
            "resize_min",
            _one_size_params("min_size"),
            accepts=_resamplable,
            ref=_resize_min_ref,
            ref_accepts=_resize_ref_accepts,
            ref_dtypes=("u8", "f32"),
            tol=_resize_tol,
            kind="spatial",
        ),
        _spec(
            "resize_scale",
            _resize_scale_params,
            accepts=_resamplable,
            ref=_resize_scale_ref,
            ref_accepts=_resize_ref_accepts,
            ref_dtypes=("u8", "f32"),
            tol=_resize_tol,
            kind="spatial",
        ),
        _spec(
            "letterbox",
            _letterbox_params,
            accepts=_resamplable,
            ref=_letterbox_ref,
            ref_accepts=_resize_ref_accepts,
            ref_dtypes=("u8", "f32"),
            tol=_resize_tol,
            kind="spatial",
            note="fit (round half up), then pad centred",
        ),
        # --- affine ---------------------------------------------------------------
        _spec(
            "rotate",
            _rotate_params,
            accepts=is_rank3,
            ref=_rotate_ref,
            ref_accepts=_rotate_ref_accepts,
            ref_dtypes=("u8", "u16", "i16", "f32", "f64"),
            tol=_rotate_tol,
            kind="spatial",
            note="quarter turns are exact data movement (np.rot90, clockwise); "
            "other angles rotate clockwise about (w/2, h/2), OpenCV's matrix "
            "with -angle",
        ),
        _spec(
            "warp_affine",
            _warp_params,
            accepts=is_rank3,
            ref=_warp_ref,
            ref_accepts=_warp_ref_accepts,
            ref_dtypes=("u8", "u16", "i16", "f32", "f64"),
            tol=_warp_tol,
            kind="spatial",
            note="forward matrix, as cv2.warpAffine",
        ),
        _spec(
            "shear",
            _shear_params,
            accepts=is_rank3,
            ref=_shear_ref,
            ref_accepts=_warp_ref_accepts,
            ref_dtypes=("u8", "u16", "i16", "f32", "f64"),
            tol=_warp_tol,
            kind="spatial",
        ),
        _spec(
            "rotate_and_scale",
            _rotate_and_scale_params,
            accepts=is_rank3,
            ref=_rotate_and_scale_ref,
            ref_accepts=_warp_ref_accepts,
            ref_dtypes=("u8", "u16", "i16", "f32", "f64"),
            tol=_warp_tol,
            kind="spatial",
        ),
        # --- neighbourhood ----------------------------------------------------------
        _spec(
            "blur",
            _blur_params,
            accepts=is_rank3,
            ref=_blur_ref,
            tol=_blur_tol,
            kind="spatial",
            note="OpenCV GaussianBlur, ksize 2*ceil(3*sigma)+1, replicate border; "
            "sigma is an f32 on the wire",
        ),
        _spec(
            "erode",
            _morph_params,
            accepts=_single_channel,
            ref=lambda x, p: _erode(x, p["ksize"], p["iterations"]),
            kind="spatial",
        ),
        _spec(
            "dilate",
            _morph_params,
            accepts=_single_channel,
            ref=lambda x, p: _dilate(x, p["ksize"], p["iterations"]),
            kind="spatial",
        ),
        _spec(
            "morphology_open",
            _ksize_params,
            accepts=_single_channel,
            ref=lambda x, p: _dilate(_erode(x, p["ksize"]), p["ksize"]),
            kind="spatial",
        ),
        _spec(
            "morphology_close",
            _ksize_params,
            accepts=_single_channel,
            ref=lambda x, p: _erode(_dilate(x, p["ksize"]), p["ksize"]),
            kind="spatial",
        ),
        _spec(
            "morphology_gradient",
            _ksize_params,
            accepts=_single_channel,
            ref=lambda x, p: _saturating_difference(
                _dilate(x, p["ksize"]), _erode(x, p["ksize"])
            ),
            note="dilate - erode, saturating (i8 127 - -25 -> 127)",
            kind="spatial",
            gain=2.0,
        ),
        _spec(
            "convolve2d",
            _kernel_params,
            accepts=is_rank3,
            ref=_convolve_ref,
            ref_accepts=_convolve_ref_accepts,
            tol=_convolve_tol,
            gain=_kernel_gain,
            kind="spatial",
            note="correlation (cv2.filter2D), not flipped convolution; "
            "'reflect' repeats the edge pixel (SciPy reflect), which pad "
            "calls 'symmetric'",
        ),
        _spec(
            "sobel",
            lambda draw, x: {"axis": draw(st.sampled_from(["x", "y"]))},
            accepts=is_rank3,
            ref=lambda x, p: _correlate(
                x, _SOBEL[p["axis"]], normalize=False, border="replicate"
            ),
            tol=lambda x, p: _convolve_tol(x, {"kernel": _SOBEL["x"]}),
            gain=8.0,
            kind="spatial",
        ),
        _spec(
            "laplacian",
            accepts=is_rank3,
            ref=lambda x, p: _correlate(
                x, _LAPLACIAN, normalize=False, border="replicate"
            ),
            tol=lambda x, p: _convolve_tol(x, {"kernel": _LAPLACIAN}),
            gain=8.0,
            kind="spatial",
        ),
        _spec(
            "sharpen",
            lambda draw, x: {"strength": draw(_nice(0, 2), label="strength")},
            accepts=is_rank3,
            ref=lambda x, p: _correlate(
                x, _sharpen_kernel(p["strength"]), normalize=False, border="replicate"
            ),
            tol=lambda x, p: _convolve_tol(
                x, {"kernel": _sharpen_kernel(p["strength"])}
            ),
            gain=lambda _x, p: 1 + 16 * abs(p["strength"]),
            kind="spatial",
        ),
        _spec(
            "canny",
            _canny_params,
            accepts=is_rank3,
            ref=_canny_ref,
            ref_accepts=lambda x, p: is_u8(x),
            ref_dtypes=("u8",),
            gain=_INF,
            kind="spatial",
            note="cv2.Canny: strongest channel of a colour image, alpha ignored",
        ),
        _spec(
            "equalize_histogram",
            accepts=is_rank3,
            ref=_equalize_ref,
            tol=lsb(1),
            ref_accepts=lambda x, p: (
                is_u8(x)
                and all(len(np.unique(x[:, :, c])) > 1 for c in range(x.shape[2]))
            ),
            ref_dtypes=("u8",),
            gain=_INF,
            kind="global",
            note="cv2.equalizeHist per channel, to one level (the CDF's "
            "rounding differs); a single-valued channel has no "
            "CDF to spread (OpenCV keeps the value, the engine maps it to 0) "
            "and the op documents neither",
        ),
        # --- terminal: scalar ----------------------------------------------------------
        _spec(
            "reduce_sum",
            ref=_scalar_result(lambda x, p: x.astype(np.float64).sum()),
            tol=_sum_tol,
            kind="global",
            domain_out="scalar",
            terminal=True,
        ),
        _spec(
            "reduce_mean",
            ref=_scalar_result(lambda x, p: x.astype(np.float64).mean()),
            tol=_mean_tol,
            kind="global",
            domain_out="scalar",
            terminal=True,
        ),
        _spec(
            "reduce_min",
            ref=_scalar_result(lambda x, p: x.min()),
            kind="global",
            defines_non_finite=True,
            domain_out="scalar",
            terminal=True,
        ),
        _spec(
            "reduce_max",
            ref=_scalar_result(lambda x, p: x.max()),
            kind="global",
            defines_non_finite=True,
            domain_out="scalar",
            terminal=True,
        ),
        _spec(
            "reduce_std",
            _std_params,
            ref=_scalar_result(lambda x, p: x.astype(np.float64).std(ddof=p["ddof"])),
            ref_accepts=_std_ref_accepts,
            tol=_std_tol,
            kind="global",
            domain_out="scalar",
            terminal=True,
        ),
        _spec(
            "reduce_percentile",
            _percentile_params,
            ref=_scalar_result(
                lambda x, p: np.percentile(x.astype(np.float64), p["q"])
            ),
            # numpy interpolates an infinity into NaN (its p0 of [1, inf] is
            # NaN), no rule worth matching; NaN alone is defined (NaN out).
            ref_accepts=lambda x, p: not np.isinf(x).any(),
            tol=_stat_tol,
            kind="global",
            defines_non_finite=True,
            domain_out="scalar",
            terminal=True,
            note="linear interpolation, as numpy's default",
        ),
        _spec(
            "reduce_popcount",
            accepts=is_int,
            ref=_popcount_ref,
            ref_dtypes=("u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64"),
            kind="global",
            domain_out="scalar",
            terminal=True,
            note="set bits of each element's two's-complement bytes",
        ),
        # --- terminal: along an axis (buffer out) ----------------------------------
        _spec(
            "reduce_argmax",
            _axis_params,
            ref=_axis_reduce(np.argmax, lambda x: np.dtype(np.int64)),
            kind="global",
            defines_non_finite=True,
            terminal=True,
            note="first occurrence wins a tie, as numpy",
        ),
        _spec(
            "reduce_argmin",
            _axis_params,
            ref=_axis_reduce(np.argmin, lambda x: np.dtype(np.int64)),
            kind="global",
            defines_non_finite=True,
            terminal=True,
        ),
        # --- terminal: vector ----------------------------------------------------------
        _spec(
            "histogram",
            _histogram_params,
            ref=_histogram_ref,
            ref_accepts=_histogram_ref_accepts,
            tol=close(atol=1e-12, rtol=1e-9),
            kind="global",
            defines_non_finite=True,
            domain_out="vector",
            terminal=True,
            note="numpy.histogram (bins half-open [a, b), the last closed; "
            "closed='right' mirrors it), a value outside the edges in no bin, "
            "integers compared exactly",
        ),
        _spec(
            "extract_shape",
            ref=_extract_shape_ref,
            kind="global",
            domain_out="vector",
            terminal=True,
        ),
        _spec(
            "perceptual_hash",
            lambda draw, x: {
                "algorithm": draw(
                    st.sampled_from(
                        ["average", "difference", "perceptual", "blockhash"]
                    )
                ),
                "hash_size": draw(st.sampled_from([16, 64])),
            },
            accepts=lambda x: (
                is_rank3(x) and x.shape[0] >= 2 and x.shape[1] >= 2 and x.shape[2] <= 4
            ),
            no_ref=(
                "hashes match imagehash only up to a Hamming distance (resize "
                "differs); reference/test_perceptual_hash_ref.py checks that. "
                "Invariance only here"
            ),
            kind="global",
            domain_out="vector",
            terminal=True,
        ),
        _spec(
            "extract_contours",
            _extract_contours_params,
            accepts=_mask_like,
            no_ref=_CONTOUR_NO_REF,
            kind="global",
            domain_out="contour",
            terminal=True,
        ),
        # --- contour-domain tails ---------------------------------------------------
        *(
            _spec(
                name,
                params,
                domain_in="contour",
                domain_out=out,
                no_ref=_CONTOUR_NO_REF,
                kind="global",
                terminal=True,
            )
            for name, params, out in (
                ("area", lambda draw, x: {"signed": draw(st.booleans())}, "vector"),
                ("perimeter", _no_params, "vector"),
                ("centroid", _no_params, "vector"),
                ("bounding_box", _no_params, "vector"),
                ("convex_hull", _no_params, "contour"),
                (
                    "largest",
                    lambda draw, x: {"k": draw(st.integers(1, 3))},
                    "contour",
                ),
                ("translate", _translate_params, "contour"),
                ("scale_contour", _scale_contour_params, "contour"),
                (
                    "simplify",
                    lambda draw, x: {"tolerance": draw(_nice(0, 4))},
                    "contour",
                ),
                ("rasterize", _rasterize_params, "buffer"),
            )
        ),
    )
}

#: Chainable ``Pipeline`` methods with no entry, and why.
EXEMPT: dict[str, str] = {
    "label_reduce": (
        "scores a separate contour column against the buffer; it needs a "
        "second input the single-column harness does not build "
        "(test_expression_op_params.py covers it)"
    ),
    "on_error": "a graph policy, not an operation (test_on_error.py)",
    "on_null_param": "a graph policy, not an operation (test_null_params.py)",
}


# ---------------------------------------------------------------------------
# Binary (lazy-only) ops
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinarySpec:
    """A two-operand op on ``LazyPipelineExpr``.

    Both operands share a dtype and shape. ``ref(a, b)`` is the reference.
    As for :class:`OpSpec`, ``accepts`` is what the op's contract admits (the
    cases drawn, so the invariance suite covers all of it) and
    ``ref_accepts`` what the reference models (the cases compared).
    """

    method: str
    ref: Callable[[np.ndarray, np.ndarray], np.ndarray]
    accepts: Callable[[np.ndarray], bool] = _always
    ref_accepts: Callable[[np.ndarray], bool] = _always
    tol: Callable[[np.ndarray], Tol] = lambda x: EXACT
    note: str = ""


def _saturating(
    op: Callable[[Any, Any], Any],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    @quiet
    def ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.dtype.kind == "f":
            return op(a, b).astype(a.dtype)
        info = np.iinfo(a.dtype)
        values = op(a.astype(object), b.astype(object))
        return np.clip(values, info.min, info.max).astype(a.dtype)

    return ref


@quiet
def _divide_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy's true division into the operands' float: f64 for f64 and the
    32/64-bit integers, f32 otherwise. IEEE at zero."""
    out = float_out(a)
    return (a.astype(np.float64) / b.astype(np.float64)).astype(out)


def _divide_tol(x: np.ndarray) -> Tol:
    """The engine divides in ``DType::accumulator``; the reference rounds an
    f64 quotient to the output, so the two may differ by an ulp of it."""
    eps = np.finfo(float_out(x)).eps
    return close(atol=0, rtol=2 * float(eps))


@quiet
def _blend_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a/MAX)(b/MAX)MAX``: a plain product for floats; for integers
    ``round(a*b / MAX)`` in exact integers (MAX is odd, so no ties),
    saturated -- f64 cannot hold a 64-bit product."""
    if a.dtype.kind == "f":
        return (a.astype(np.float64) * b.astype(np.float64)).astype(a.dtype)
    info = np.iinfo(a.dtype)
    peak = int(info.max)

    def one(x: int, y: int) -> int:
        q, r = divmod(abs(x * y), peak)
        q += 2 * r > peak
        return min(max(q if x * y >= 0 else -q, int(info.min)), peak)

    flat = [one(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist())]
    return np.array(flat, dtype=a.dtype).reshape(np.broadcast_shapes(a.shape, b.shape))


def _float_or_lsb(x: np.ndarray) -> Tol:
    return lsb(1) if is_int(x) else close(atol=0, rtol=4 * float(np.finfo(x.dtype).eps))


BINARY: dict[str, BinarySpec] = {
    spec.method: spec
    for spec in (
        BinarySpec("add", _saturating(lambda a, b: a + b), note="saturating"),
        BinarySpec("subtract", _saturating(lambda a, b: a - b), note="saturating"),
        BinarySpec("multiply", _saturating(lambda a, b: a * b), note="saturating"),
        BinarySpec(
            "divide",
            _divide_ref,
            tol=_divide_tol,
            note="true division into a float; IEEE at zero (x / 0 -> inf)",
        ),
        BinarySpec(
            "blend",
            _blend_ref,
            tol=_float_or_lsb,
            note="normalized product: (a/MAX)(b/MAX)MAX, exact for integers",
        ),
        BinarySpec("maximum", lambda a, b: np.maximum(a, b)),
        BinarySpec("minimum", lambda a, b: np.minimum(a, b)),
        BinarySpec("bitwise_and", lambda a, b: a & b, accepts=is_int),
        BinarySpec("bitwise_or", lambda a, b: a | b, accepts=is_int),
        BinarySpec("bitwise_xor", lambda a, b: a ^ b, accepts=is_int),
    )
}

#: Lazy-only ops without a :class:`BinarySpec`, and why.
BINARY_EXEMPT: dict[str, str] = {
    "apply_mask": (
        "takes a mask operand of a different shape (single channel, or a "
        "contour to rasterize); covered by test_binary_ops_ref.py"
    ),
    "channel_merge": (
        "variadic over single-channel operands; its inverse law is checked in "
        "laws/test_parity_laws.py"
    ),
}


def spec_for(method: str) -> OpSpec:
    """The entry for *method* (a ``KeyError`` names what is missing)."""
    return OPS[method]


def buffer_specs() -> list[OpSpec]:
    """Every entry that reads a buffer."""
    return [s for s in OPS.values() if s.domain_in == "buffer"]


def dtype_of(x: Any) -> str:
    """The engine spelling of *x*'s dtype."""
    return dtype_name(np.asarray(x).dtype)
