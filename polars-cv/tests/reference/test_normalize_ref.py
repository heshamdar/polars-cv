"""Reference tests for ``normalize`` (minmax, zscore) and ``normalize_preset``.

Compares polars-cv plugin output against a NumPy reference computed in the
documented precision. Inputs are fed through ``source("array")`` from
``pl.Array`` columns, which is the path that carries u8, u16 and f32 buffers
of any shape; views (``crop``, ``flip_h``, ``flip_v``) are applied before the
op and compared against the same NumPy view. Constant images exercise the
``max == min`` / ``std == 0`` all-zeros path.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

SIZES = [(1, 1), (1, 7), (7, 1), (13, 17), (33, 65), (64, 64), (100, 203)]
DTYPES = {"u8": np.uint8, "u16": np.uint16, "f32": np.float32}
POL_DTYPES = {"u8": pl.UInt8, "u16": pl.UInt16, "f32": pl.Float32}

# name -> (pipeline view op, numpy view); crop leaves a non-trivial window.
Views = tuple[Callable[[Pipeline], Pipeline], Callable[[np.ndarray], np.ndarray]]
VIEWS: dict[str, Views] = {
    "none": (lambda p: p, lambda a: a),
    "flip_h": (lambda p: p.flip_h(), lambda a: a[:, ::-1]),
    "flip_v": (lambda p: p.flip_v(), lambda a: a[::-1]),
}


def _make(kind: str, dtype: str, shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Random data spanning the full dtype range, or a constant image."""
    rng = np.random.default_rng([seed, *shape, len(dtype)])
    dt = DTYPES[dtype]
    if kind == "const":
        value = {"u8": 77, "u16": 40000, "f32": -3.25}[dtype]
        return np.full(shape, value, dtype=dt)
    if dtype == "f32":
        return rng.uniform(-1e3, 1e3, shape).astype(dt)
    info = np.iinfo(dt)
    arr = rng.integers(info.min, int(info.max) + 1, shape, dtype=dt)
    flat = arr.reshape(-1)
    flat[0] = info.min  # pin the extremes so min/max span the dtype range
    flat[-1] = info.max
    return arr


def _df(arrs: list[np.ndarray | None], dtype: str) -> pl.DataFrame:
    shape = next(a.shape for a in arrs if a is not None)
    ser = pl.Series(
        "a",
        [None if a is None else a.tolist() for a in arrs],
        dtype=pl.Array(POL_DTYPES[dtype], shape),
    )
    return pl.DataFrame({"a": ser})


def _run(df: pl.DataFrame, pipe: Pipeline) -> list[np.ndarray | None]:
    out = df.select(o=pl.col("a").cv.pipe(pipe).sink("tensor"))["o"]
    return [None if row is None else numpy_from_struct(row) for row in out.to_list()]


def _run1(arr: np.ndarray, dtype: str, pipe: Pipeline) -> np.ndarray:
    (got,) = _run(_df([arr], dtype), pipe)
    assert got is not None
    return got


def _ref_minmax(x: np.ndarray) -> np.ndarray:
    lo = np.float32(x.min())
    rng = np.float32(x.max()) - lo
    if rng == 0:
        return np.zeros(x.shape, dtype=np.float32)
    return (x.astype(np.float32) - lo) / rng


def _ref_zscore(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    mean = x64.mean()
    std = x64.std()
    if std == 0:
        return np.zeros(x.shape, dtype=np.float32)
    return (x.astype(np.float32) - np.float32(mean)) / np.float32(std)


def _check_minmax(got: np.ndarray, x: np.ndarray) -> None:
    expected = _ref_minmax(x)
    assert got.shape == expected.shape
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
    if np.float32(x.max()) - np.float32(x.min()) > 0:
        assert got.min() == 0
        assert got.max() == 1


def _check_zscore(got: np.ndarray, x: np.ndarray) -> None:
    expected = _ref_zscore(x)
    assert got.shape == expected.shape
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


KINDS = ["random", "const"]


@plugin_required
@pytest.mark.parity(
    ops=["normalize"],
    against="NumPy float32 (x - min) / (max - min)",
    tolerance="atol 1e-6, rtol 0; exact 0 and 1 extremes",
)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("hw", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_minmax_matches_numpy(
    hw: tuple[int, int], channels: int, dtype: str, kind: str
) -> None:
    x = _make(kind, dtype, (*hw, channels))
    got = _run1(x, dtype, Pipeline().source("array").normalize())
    _check_minmax(got, x)


@plugin_required
@pytest.mark.parity(
    ops=["normalize"],
    against="NumPy float64 mean/std cast to float32, (x - mean) / std",
    tolerance="rtol 1e-5, atol 1e-5",
)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("hw", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_zscore_matches_numpy(
    hw: tuple[int, int], channels: int, dtype: str, kind: str
) -> None:
    x = _make(kind, dtype, (*hw, channels))
    got = _run1(x, dtype, Pipeline().source("array").normalize(method="zscore"))
    _check_zscore(got, x)


PRESET_MEAN = [0.485, 0.456, 0.406]
PRESET_STD = [0.229, 0.224, 0.225]


def _preset_params(dtype: str, channels: int) -> tuple[list[float], list[float]]:
    scale = {"u8": 255.0, "u16": 65535.0, "f32": 1000.0}[dtype]
    mean = [m * scale for m in PRESET_MEAN[:channels]]
    std = [s * scale for s in PRESET_STD[:channels]]
    return mean, std


def _ref_preset(x: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    m = np.array(mean, dtype=np.float32)
    s = np.array(std, dtype=np.float32)
    return (x.astype(np.float32) - m) / s


@plugin_required
@pytest.mark.parity(
    ops=["normalize_preset"],
    against="NumPy float32 per-channel (x - mean[c]) / std[c]",
    tolerance="atol 1e-5, rtol 0",
)
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("hw", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_preset_matches_numpy(hw: tuple[int, int], channels: int, dtype: str) -> None:
    x = _make("random", dtype, (*hw, channels))
    mean, std = _preset_params(dtype, channels)
    pipe = Pipeline().source("array").normalize_preset(mean=mean, std=std)
    got = _run1(x, dtype, pipe)
    expected = _ref_preset(x, mean, std)
    assert got.shape == expected.shape
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)


VIEW_NAMES = [*VIEWS, "crop"]


def _apply_view(name: str, x: np.ndarray) -> tuple[Pipeline, np.ndarray]:
    pipe = Pipeline().source("array")
    if name == "crop":
        h, w = x.shape[:2]
        top, left = h // 4, w // 3
        height, width = max(1, h // 2), max(1, w // 2)
        pipe = pipe.crop(top=top, left=left, height=height, width=width)
        return pipe, x[top : top + height, left : left + width]
    op, view = VIEWS[name]
    return op(pipe), view(x)


@plugin_required
@pytest.mark.parity(
    ops=["normalize", "normalize_preset", "crop", "flip_h", "flip_v"],
    against="NumPy views of the same window, then the float32 references",
    tolerance="minmax atol 1e-6; zscore/preset atol 1e-5",
)
@pytest.mark.parametrize("view", VIEW_NAMES)
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("hw", [(13, 17), (33, 65), (100, 203)], ids=lambda s: str(s))
def test_normalize_after_view(
    hw: tuple[int, int], channels: int, dtype: str, view: str
) -> None:
    x = _make("random", dtype, (*hw, channels))
    pipe, xv = _apply_view(view, x)
    xv = np.ascontiguousarray(xv)

    _check_minmax(_run1(x, dtype, pipe.normalize()), xv)
    _check_zscore(_run1(x, dtype, pipe.normalize(method="zscore")), xv)

    mean, std = _preset_params(dtype, channels)
    got = _run1(x, dtype, pipe.normalize_preset(mean=mean, std=std))
    np.testing.assert_allclose(got, _ref_preset(xv, mean, std), rtol=0, atol=1e-5)


@plugin_required
@pytest.mark.parity(
    ops=["normalize", "normalize_preset", "crop", "flip_h", "flip_v"],
    against="NumPy views of the same window, then the float32 references",
    tolerance="minmax atol 1e-6; zscore/preset atol 1e-5",
)
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_normalize_after_flip_then_crop(dtype: str) -> None:
    x = _make("random", dtype, (33, 65, 3))
    pipe = (
        Pipeline()
        .source("array")
        .flip_h()
        .flip_v()
        .crop(top=3, left=5, height=20, width=31)
    )
    xv = np.ascontiguousarray(x[::-1, ::-1][3:23, 5:36])
    _check_minmax(_run1(x, dtype, pipe.normalize()), xv)
    _check_zscore(_run1(x, dtype, pipe.normalize(method="zscore")), xv)


@plugin_required
@pytest.mark.parity(
    ops=["normalize", "normalize_preset"],
    against="NumPy per-row references; null rows stay null",
    tolerance="as the non-null cases",
)
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("op", ["minmax", "zscore", "preset"])
def test_null_row_propagates(dtype: str, op: str) -> None:
    shape = (13, 17, 3)
    a, b, c = (_make("random", dtype, shape, seed=s) for s in (1, 2, 3))
    const = _make("const", dtype, shape)
    mean, std = _preset_params(dtype, 3)
    pipe = Pipeline().source("array")
    pipe = {
        "minmax": pipe.normalize(),
        "zscore": pipe.normalize(method="zscore"),
        "preset": pipe.normalize_preset(mean=mean, std=std),
    }[op]
    rows = [a, None, b, const, None, c]
    got = _run(_df(rows, dtype), pipe)
    assert len(got) == len(rows)
    for row, x in zip(got, rows):
        if x is None:
            assert row is None
            continue
        assert row is not None
        if op == "minmax":
            _check_minmax(row, x)
        elif op == "zscore":
            _check_zscore(row, x)
        else:
            np.testing.assert_allclose(
                row, _ref_preset(x, mean, std), rtol=0, atol=1e-5
            )
