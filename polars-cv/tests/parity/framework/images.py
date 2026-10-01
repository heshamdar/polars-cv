"""The image state space: dtypes, sizes, channel counts and pixel content.

Every image here is ``[H, W, C]`` because every image source decodes to that
rank (``list``/``array``/``raw`` can carry others, but the op vocabulary is an
image vocabulary). Pixel values come from a seeded NumPy generator rather than
from Hypothesis drawing each pixel: a pixel-level strategy is slow to generate
and shrinks towards all-zero images that exercise nothing, while a seed plus a
*content kind* shrinks towards a small image of the simplest kind that still
fails — which is the useful minimal example.

The content kinds are the inputs that break image kernels in practice:

* ``noise`` — every value in the kind's range; maximal high-frequency content.
* ``smooth`` — upsampled noise; what a real photo looks like to a filter.
* ``constant`` — one value; degenerate statistics (zero variance, flat
  histogram, no edges).
* ``extremes`` — only the dtype's min/max (and 0); saturation and overflow.
* ``edges`` — two-valued blocks; step edges for gradients, morphology, canny.
* ``gradient`` — a ramp; monotone content for histogram and resampling.
* ``halves`` (floats) — values on ``k/2``; ties for every rounding rule.
* ``nonfinite`` (floats) — smooth content with NaN, ``inf`` and ``-inf``
  sprinkled in; every op's NaN and infinity rule, and its panics.

Integer value ranges: ``byte`` (0..255, what most image data occupies),
``full`` (the dtype's range, within ±2**52 so a value survives f64) and, for
the 64-bit dtypes, ``wide``: the whole range with odd low bits, values f64
cannot hold (a kernel that computes through f64 rounds them).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st

#: Every dtype the engine's ``dtype_table!`` names, in NumPy spelling. The
#: ratchet in ``meta/test_parity_ratchets.py`` holds this to the table.
DTYPES: dict[str, np.dtype] = {
    "u8": np.dtype(np.uint8),
    "i8": np.dtype(np.int8),
    "u16": np.dtype(np.uint16),
    "i16": np.dtype(np.int16),
    "u32": np.dtype(np.uint32),
    "i32": np.dtype(np.int32),
    "u64": np.dtype(np.uint64),
    "i64": np.dtype(np.int64),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}

#: dtype spelling for a NumPy dtype (the inverse of :data:`DTYPES`).
NAME_OF: dict[np.dtype, str] = {v: k for k, v in DTYPES.items()}

INT_CONTENT = ("noise", "smooth", "constant", "extremes", "edges", "gradient")
FLOAT_CONTENT = (*INT_CONTENT, "halves", "nonfinite")

#: The dtypes whose integers reach beyond f64's 53-bit mantissa.
WIDE_DTYPES = ("u64", "i64")

#: Float value ranges. ``unit`` is what normalized images hold; ``signed``
#: exercises negative paths (abs, relu, sign); ``wide`` exercises magnitude.
FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "unit": (0.0, 1.0),
    "signed": (-2.0, 2.0),
    "wide": (-1.0e4, 1.0e4),
}


def dtype_name(dtype: np.dtype) -> str:
    """The engine spelling of a NumPy dtype."""
    return NAME_OF[np.dtype(dtype)]


@dataclass(frozen=True)
class ImageSpec:
    """A reproducible image: everything needed to regenerate its pixels."""

    height: int
    width: int
    channels: int
    dtype: str
    content: str
    value_range: str
    seed: int

    def render(self) -> np.ndarray:
        """The pixels, ``[H, W, C]`` of :attr:`dtype`."""
        return render(self)

    def __repr__(self) -> str:  # a short form for Hypothesis' falsifying example
        return (
            f"ImageSpec({self.height}x{self.width}x{self.channels} {self.dtype} "
            f"{self.content}/{self.value_range} seed={self.seed})"
        )


def _bounds(dtype: np.dtype, value_range: str) -> tuple[float, float]:
    if dtype.kind == "f":
        return FLOAT_RANGES[value_range]
    info = np.iinfo(dtype)
    if value_range == "byte":
        # The range most image data actually occupies, whatever its container.
        return (max(info.min, 0), min(info.max, 255))
    if value_range == "wide":
        # The largest floats inside the range: rendered through f64, then
        # given odd low bits (``_widen``).
        return (float(info.min), float(np.nextafter(float(info.max), 0.0)))
    # 64-bit extremes do not survive a trip through float64; stay in range.
    return (float(max(info.min, -(2**52))), float(min(info.max, 2**52)))


def render(spec: ImageSpec) -> np.ndarray:
    """Generate the pixels an :class:`ImageSpec` describes."""
    dtype = DTYPES[spec.dtype]
    rng = np.random.default_rng(spec.seed)
    lo, hi = _bounds(dtype, spec.value_range)
    shape = (spec.height, spec.width, spec.channels)

    if spec.content == "noise":
        unit = rng.random(shape)
    elif spec.content == "smooth":
        coarse = rng.random(
            (max(2, spec.height // 6 + 2), max(2, spec.width // 6 + 2), spec.channels)
        )
        rows = np.linspace(0, coarse.shape[0] - 1, spec.height)
        cols = np.linspace(0, coarse.shape[1] - 1, spec.width)
        unit = _bilinear(coarse, rows, cols)
    elif spec.content == "constant":
        unit = np.full(shape, rng.random())
    elif spec.content == "extremes":
        unit = rng.choice(np.array([0.0, 1.0, 0.5]), size=shape, p=[0.4, 0.4, 0.2])
    elif spec.content == "edges":
        block = max(1, min(spec.height, spec.width) // 3)
        yy, xx = np.indices((spec.height, spec.width))
        pattern = ((yy // block + xx // block) % 2).astype(float)
        levels = rng.random(2)
        unit = np.repeat(
            (levels[0] + pattern * (levels[1] - levels[0]))[:, :, None],
            spec.channels,
            axis=2,
        )
    elif spec.content == "gradient":
        ramp = np.linspace(0.0, 1.0, spec.height * spec.width).reshape(
            spec.height, spec.width
        )
        unit = np.repeat(ramp[:, :, None], spec.channels, axis=2)
    elif spec.content == "halves":
        values = rng.integers(-16, 17, size=shape) / 2.0
        return values.astype(dtype)
    elif spec.content == "nonfinite":
        values = (lo + rng.random(shape) * (hi - lo)).astype(dtype)
        special = rng.random(shape)
        values[special < 0.15] = np.nan
        values[(special >= 0.15) & (special < 0.2)] = np.inf
        values[(special >= 0.2) & (special < 0.25)] = -np.inf
        return values
    else:  # pragma: no cover - the strategy only draws known kinds
        msg = f"unknown content kind {spec.content!r}"
        raise ValueError(msg)

    values = lo + unit * (hi - lo)
    if dtype.kind == "f":
        return values.astype(dtype)
    ints = np.clip(np.rint(values), lo, hi).astype(dtype)
    return _widen(ints, rng) if spec.value_range == "wide" else ints


def _widen(ints: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Give the 64-bit values low bits f64 cannot hold, staying in range: a
    positive value steps down by 1-7, any other up (the values sit at least
    1024 inside the range, ``_bounds``). One step per image, so ``constant``
    stays constant."""
    step = ints.dtype.type(rng.integers(1, 8))
    return np.where(ints > 0, ints - step, ints + step)


def _bilinear(coarse: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    r0 = np.floor(rows).astype(int)
    c0 = np.floor(cols).astype(int)
    r1 = np.minimum(r0 + 1, coarse.shape[0] - 1)
    c1 = np.minimum(c0 + 1, coarse.shape[1] - 1)
    fr = (rows - r0)[:, None, None]
    fc = (cols - c0)[None, :, None]
    top = coarse[r0][:, c0] * (1 - fc) + coarse[r0][:, c1] * fc
    bottom = coarse[r1][:, c0] * (1 - fc) + coarse[r1][:, c1] * fc
    return top * (1 - fr) + bottom * fr


#: Sizes that break loops: 1 (no neighbours), 2, primes, and a spread of
#: ordinary sizes. Weighted towards the small end so shrinking has somewhere to
#: go and most examples stay cheap.
_AWKWARD_SIDES = (1, 2, 3, 5, 7, 8, 13, 16, 17, 31, 32)


def sides(min_side: int = 1, max_side: int = 48) -> st.SearchStrategy[int]:
    """An image side in ``[min_side, max_side]``, biased to awkward sizes."""
    awkward = [s for s in _AWKWARD_SIDES if min_side <= s <= max_side]
    plain = st.integers(min_value=min_side, max_value=max_side)
    return st.one_of(st.sampled_from(awkward), plain) if awkward else plain


def image_specs(
    *,
    dtypes: tuple[str, ...] = tuple(DTYPES),
    channels: tuple[int, ...] = (1, 2, 3, 4),
    min_side: int = 1,
    max_side: int = 48,
    contents: tuple[str, ...] | None = None,
) -> st.SearchStrategy[ImageSpec]:
    """Draw an :class:`ImageSpec` from the given slice of the state space."""

    @st.composite
    def build(draw: st.DrawFn) -> ImageSpec:
        dtype = draw(st.sampled_from(dtypes), label="dtype")
        is_float = DTYPES[dtype].kind == "f"
        kinds = contents or (FLOAT_CONTENT if is_float else INT_CONTENT)
        if is_float:
            value_range = draw(st.sampled_from(tuple(FLOAT_RANGES)), label="range")
        else:
            ranges = (
                ("byte", "full", "wide") if dtype in WIDE_DTYPES else ("byte", "full")
            )
            value_range = draw(st.sampled_from(ranges), label="range")
        side = sides(min_side, max_side)
        return ImageSpec(
            height=draw(side, label="height"),
            width=draw(side, label="width"),
            channels=draw(st.sampled_from(channels), label="channels"),
            dtype=dtype,
            content=draw(st.sampled_from(kinds), label="content"),
            value_range=value_range,
            seed=draw(st.integers(0, 2**32 - 1), label="seed"),
        )

    return build()
