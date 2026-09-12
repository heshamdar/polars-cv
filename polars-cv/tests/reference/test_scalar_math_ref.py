"""Reference tests for the core math primitive ops.

Compares polars-cv plugin output against NumPy ground truth for the pure
elementwise scalar ops (neg, abs, sqrt, square, reciprocal, sign, floor,
ceil, round, trunc, clamp_min, clamp_max, add_constant, subtract_constant).

The base image is mapped into a signed float range [-2, 2) so every op is
exercised on negatives, near-zero, and values large enough for the rounding
family to move — a [0, 1] base would make abs/sign/neg/floor vacuous.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required


def _encode_png(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def rgb_arr() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def rgb_png(rgb_arr: np.ndarray) -> bytes:
    return _encode_png(rgb_arr)


def _base(arr: np.ndarray) -> np.ndarray:
    """The signed-float reference base: ``arr * (4/255) - 2`` in float32."""
    return arr.astype(np.float32) * np.float32(4.0 / 255.0) - np.float32(2.0)


def _run(pipe: Pipeline, png: bytes) -> np.ndarray:
    df = pl.DataFrame({"img": [png]})
    return numpy_from_struct(
        df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy")).row(0)[0]
    )


#: (name, pipeline op applied after the signed base, numpy reference on the base).
_CASES: list[
    tuple[str, Callable[[Pipeline], Pipeline], Callable[[np.ndarray], np.ndarray]]
] = [
    ("neg", lambda p: p.neg(), lambda a: -a),
    ("abs", lambda p: p.abs(), np.abs),
    ("sqrt", lambda p: p.sqrt(), np.sqrt),  # NaN for a < 0 on both sides
    ("square", lambda p: p.square(), lambda a: a * a),
    ("reciprocal", lambda p: p.reciprocal(), lambda a: np.float32(1.0) / a),
    ("sign", lambda p: p.sign(), np.sign),
    ("floor", lambda p: p.floor(), np.floor),
    ("ceil", lambda p: p.ceil(), np.ceil),
    ("round", lambda p: p.round(), np.round),  # both ties-to-even
    ("trunc", lambda p: p.trunc(), np.trunc),
    ("clamp_min", lambda p: p.clamp_min(0.5), lambda a: np.maximum(a, np.float32(0.5))),
    ("clamp_max", lambda p: p.clamp_max(0.5), lambda a: np.minimum(a, np.float32(0.5))),
    ("add_constant", lambda p: p.add_constant(1.5), lambda a: a + np.float32(1.5)),
    (
        "subtract_constant",
        lambda p: p.subtract_constant(1.5),
        lambda a: a - np.float32(1.5),
    ),
]


@plugin_required
@pytest.mark.parametrize("name, op, ref", _CASES, ids=[c[0] for c in _CASES])
def test_scalar_op_matches_numpy(
    name: str,
    op: Callable[[Pipeline], Pipeline],
    ref: Callable[[np.ndarray], np.ndarray],
    rgb_arr: np.ndarray,
    rgb_png: bytes,
) -> None:
    base = _base(rgb_arr)
    base_pipe = (
        Pipeline()
        .source("image_bytes")
        .cast("f32")
        .scale(4.0 / 255.0)
        .subtract_constant(2.0)
    )
    got = _run(op(base_pipe), rgb_png)
    # sqrt of a negative and 1/0 legitimately produce NaN/inf on both sides;
    # silence NumPy's warning for the reference and compare NaN-aware.
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = ref(base).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5, equal_nan=True)


@plugin_required
def test_scalar_ops_promote_integer_input_to_f32(rgb_png: bytes) -> None:
    """A scalar op on a u8 image promotes to f32 (like ``scale``)."""
    out = _run(Pipeline().source("image_bytes").sqrt(), rgb_png)
    assert out.dtype == np.float32
