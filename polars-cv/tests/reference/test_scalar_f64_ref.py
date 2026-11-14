"""The scalar family's constants are f64 where the arithmetic is.

Data whose accumulator is f64 (``Float64`` and the 32/64-bit integers)
computes in f64 and stores f64. Its constants were f32 on the wire and in
``ScalarOp``, so the arithmetic was f64 over a constant that had already lost
its low bits: ``scale(1/3)`` was wrong from the eighth digit and an integer
bound past 2^24 was off by one. Every case here is exact against NumPy.

``Float32`` and 8/16-bit data compute in f32 with the constant narrowed to
f32, as before: the last test holds that to NumPy's f32 result.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

pytestmark = [
    plugin_required,
    pytest.mark.parity(
        ops=[
            "scale",
            "clamp",
            "clamp_min",
            "clamp_max",
            "add_constant",
            "subtract_constant",
            "adjust_contrast",
            "adjust_gamma",
            "pad",
            "pad_to_size",
            "crop_pad",
            "convolve2d",
        ],
        against="NumPy float64 arithmetic (float32 on Float32 data)",
        tolerance="exact (adjust_gamma: 1 ulp)",
    ),
]

_POLARS = {
    np.dtype("float64"): pl.Float64,
    np.dtype("float32"): pl.Float32,
    np.dtype("int32"): pl.Int32,
    np.dtype("int64"): pl.Int64,
    np.dtype("uint32"): pl.UInt32,
}

Build = Callable[[Pipeline], Pipeline]


def _run(values: np.ndarray, build: Build, **columns: float) -> np.ndarray:
    """``build`` applied to one ``Array`` row holding ``values``."""
    df = pl.DataFrame(
        {"a": [values.tolist()], **{k: [v] for k, v in columns.items()}},
        schema_overrides={"a": pl.Array(_POLARS[values.dtype], len(values))},
    )
    pipe = build(Pipeline().source("array"))
    out = df.select(o=pl.col("a").cv.pipe(pipe).sink("tensor"))
    return numpy_from_struct(out.row(0)[0])


def _exact(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == want.dtype, f"dtype {got.dtype}, expected {want.dtype}"
    assert got.tolist() == want.tolist()


THIRD = 1.0 / 3.0
PAST_F32 = 16_777_217  # 2^24 + 1: the first integer f32 cannot hold


# The plan's table, one row each.


def test_scale_by_a_third_on_float64() -> None:
    values = np.array([1.0, 3.0])
    _exact(_run(values, lambda p: p.scale(THIRD)), values * THIRD)


def test_add_constant_a_tenth_on_float64() -> None:
    values = np.array([1.0])
    _exact(_run(values, lambda p: p.add_constant(0.1)), values + 0.1)


def test_clamp_min_leaves_a_float64_that_meets_the_bound() -> None:
    """``0.30000000000000004 >= 0.3``: the f32 bound (0.30000001192...) raised it."""
    values = np.array([0.1, 0.30000000000000004])
    _exact(_run(values, lambda p: p.clamp_min(0.3)), np.maximum(values, 0.3))


def test_clamp_bound_past_f32_on_int64() -> None:
    values = np.array([PAST_F32, PAST_F32 + 5], dtype=np.int64)
    got = _run(values, lambda p: p.clamp(0, PAST_F32))
    _exact(got, np.clip(values.astype(np.float64), 0, PAST_F32))


def test_add_constant_past_f32_on_int64() -> None:
    values = np.array([PAST_F32], dtype=np.int64)
    got = _run(values, lambda p: p.add_constant(PAST_F32))
    _exact(got, values.astype(np.float64) + PAST_F32)


def test_scale_by_a_float64_column() -> None:
    values = np.array([1.0, 3.0])
    got = _run(values, lambda p: p.scale(pl.col("k")), k=THIRD)
    _exact(got, values * THIRD)


# The family, on every dtype whose accumulator is f64.

_VALUES = {
    "float64": np.array([0.1, 0.30000000000000004, -2.5, 1.0, 3.0, 1e-9]),
    "int32": np.array([-7, 0, 3, PAST_F32, 2_000_000_011], dtype=np.int32),
    "int64": np.array([-7, 0, 3, PAST_F32, 9_007_199_254_740_991], dtype=np.int64),
    "uint32": np.array([0, 3, PAST_F32, 4_000_000_019], dtype=np.uint32),
}

_OPS: list[tuple[str, Build, Callable[[np.ndarray], np.ndarray]]] = [
    ("scale", lambda p: p.scale(THIRD), lambda a: a * THIRD),
    ("add_constant", lambda p: p.add_constant(0.1), lambda a: a + 0.1),
    ("subtract_constant", lambda p: p.subtract_constant(0.1), lambda a: a - 0.1),
    ("add_past_f32", lambda p: p.add_constant(PAST_F32), lambda a: a + PAST_F32),
    ("clamp_min", lambda p: p.clamp_min(0.3), lambda a: np.maximum(a, 0.3)),
    ("clamp_max", lambda p: p.clamp_max(0.3), lambda a: np.minimum(a, 0.3)),
    ("clamp", lambda p: p.clamp(0.1, PAST_F32), lambda a: np.clip(a, 0.1, PAST_F32)),
]


@pytest.mark.parametrize("dtype", sorted(_VALUES))
@pytest.mark.parametrize("name,build,reference", _OPS, ids=[c[0] for c in _OPS])
def test_scalar_family_matches_numpy_in_f64(
    dtype: str,
    name: str,
    build: Build,
    reference: Callable[[np.ndarray], np.ndarray],
) -> None:
    values = _VALUES[dtype]
    _exact(_run(values, build), reference(values.astype(np.float64)))


def test_adjust_contrast_factor_is_f64_on_float64() -> None:
    """``(x - mean) * factor + mean`` with the factor as given."""
    values = np.array([0.25, 0.5, 1.5])
    mean = (values[0] + values[1] + values[2]) / 3.0
    got = _run(values, lambda p: p.adjust_contrast(factor=THIRD))
    _exact(got, (values - mean) * THIRD + mean)


def test_adjust_gamma_exponent_is_f64_on_float64() -> None:
    values = np.array([0.25, 0.5, 0.9])
    got = _run(values, lambda p: p.adjust_gamma(gamma=THIRD))
    assert got.dtype == np.float64
    # ``powf`` is libm's on both sides, but not one build of it: allow an ulp.
    # The f32 exponent was wrong by 1e-8, a million ulps.
    np.testing.assert_array_max_ulp(got, values**THIRD, maxulp=1)


def test_a_literal_and_a_float64_column_agree() -> None:
    values = np.array([0.1, 3.0, -2.5])
    for build_lit, build_col in [
        (lambda p: p.scale(THIRD), lambda p: p.scale(pl.col("k"))),
        (lambda p: p.add_constant(THIRD), lambda p: p.add_constant(pl.col("k"))),
        (lambda p: p.clamp_min(THIRD), lambda p: p.clamp_min(pl.col("k"))),
        (lambda p: p.clamp(-1.0, THIRD), lambda p: p.clamp(-1.0, pl.col("k"))),
    ]:
        _exact(_run(values, build_col, k=THIRD), _run(values, build_lit))


@pytest.mark.parametrize("name,build,reference", _OPS, ids=[c[0] for c in _OPS])
def test_float32_data_still_computes_in_f32(
    name: str, build: Build, reference: Callable[[np.ndarray], np.ndarray]
) -> None:
    """The constant narrows to f32 at use, so f32 data is NumPy's f32 result."""
    values = np.array([0.1, 0.3, -2.5, 1.0, 3.0, 1e-9], dtype=np.float32)

    def as_f32(a: np.ndarray) -> np.ndarray:
        # NumPy 2 keeps f32 against a Python float: the constant narrows too.
        return reference(a).astype(np.float32)

    with np.errstate(over="ignore"):
        want = as_f32(values)
    assert want.dtype == np.float32
    _exact(_run(values, build), want)


# A fill value is written in the buffer's own dtype, so it is the same
# question: an f32 fill is the wrong value in an f64 or 32/64-bit buffer.

_FILLS = [
    ("float64", 0.1),
    ("float64", THIRD),
    ("int64", PAST_F32),
    ("int32", PAST_F32),
    ("uint32", 4_000_000_019),
]

_PADS: list[tuple[str, Callable[[Pipeline, float], Pipeline]]] = [
    ("pad", lambda p, v: p.pad(left=1, value=v)),
    ("pad_to_size", lambda p, v: p.pad_to_size(height=1, width=3, value=v)),
    ("crop_pad", lambda p, v: p.crop_pad(left=-1, height=1, width=3, value=v)),
]


@pytest.mark.parametrize("dtype,fill", _FILLS)
@pytest.mark.parametrize("name,build", _PADS, ids=[c[0] for c in _PADS])
def test_a_fill_value_is_exact_in_the_buffers_dtype(
    name: str, build: Callable[[Pipeline, float], Pipeline], dtype: str, fill: float
) -> None:
    values = np.array([[1, 2]], dtype=dtype)
    df = pl.DataFrame(
        {"a": [values.tolist()]},
        schema_overrides={"a": pl.Array(_POLARS[values.dtype], (1, 2))},
    )
    pipe = build(Pipeline().source("array"), fill)
    got = numpy_from_struct(
        df.select(o=pl.col("a").cv.pipe(pipe).sink("tensor")).row(0)[0]
    )
    assert got.dtype == values.dtype
    assert fill in got.ravel().tolist(), got


def test_convolve2d_coefficients_are_f64_on_float64() -> None:
    """A lone centre tap of 1/3 is ``x * (1/3)``: the taps accumulate in f64."""
    values = np.array([[1.0, 3.0, 0.1], [0.3, -2.5, 7.0], [1e-9, 2.0, 5.0]])
    df = pl.DataFrame(
        {"a": [values.tolist()]}, schema={"a": pl.Array(pl.Float64, (3, 3))}
    )
    kernel = [0.0, 0.0, 0.0, 0.0, THIRD, 0.0, 0.0, 0.0, 0.0]
    pipe = Pipeline().source("array").convolve2d(kernel=kernel, normalize=False)
    got = numpy_from_struct(
        df.select(o=pl.col("a").cv.pipe(pipe).sink("tensor")).row(0)[0]
    )
    _exact(got, values * THIRD)
