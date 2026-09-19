"""Property-based invariants over the pipeline's shape and dtype contracts.

Reference tests pin exact outputs on fixed inputs; these assert algebraic
properties across a range of randomly-sized images that a fixed grid would miss:
an involution applied twice is the identity, a fixed-channel op collapses the
channel count whatever the input, and — the property that guards the planner's
two parameter paths — a structural parameter passed as a literal produces the
same bytes as the same value passed as a column (the constant-fold path vs the
general per-row path).
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

pytestmark = plugin_required

# Bounded so each example builds and executes a real pipeline cheaply.
_dims = st.integers(min_value=2, max_value=16)
_settings = settings(max_examples=25, deadline=None)


def _png(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _rgb(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _run(arr: np.ndarray, pipe: Pipeline) -> np.ndarray:
    df = pl.DataFrame({"img": [_png(arr)]})
    out = df.with_columns(r=pl.col("img").cv.pipe(pipe).sink("numpy"))
    return numpy_from_struct(out["r"][0])


@given(height=_dims, width=_dims, seed=st.integers(0, 2**31))
@_settings
def test_double_row_flip_is_identity(height: int, width: int, seed: int) -> None:
    arr = _rgb(height, width, seed)
    pipe = Pipeline().source("image_bytes").flip(axes=[0]).flip(axes=[0])
    assert np.array_equal(_run(arr, pipe), arr)


@given(height=_dims, width=_dims, seed=st.integers(0, 2**31))
@_settings
def test_double_transpose_is_identity(height: int, width: int, seed: int) -> None:
    arr = _rgb(height, width, seed)
    # Swap the two spatial axes, then swap back; the channel axis stays put.
    pipe = (
        Pipeline()
        .source("image_bytes")
        .transpose(axes=[1, 0, 2])
        .transpose(axes=[1, 0, 2])
    )
    assert np.array_equal(_run(arr, pipe), arr)


@given(height=_dims, width=_dims, seed=st.integers(0, 2**31))
@_settings
def test_grayscale_collapses_to_one_channel(height: int, width: int, seed: int) -> None:
    arr = _rgb(height, width, seed)
    out = _run(arr, Pipeline().source("image_bytes").grayscale())
    # Fixed(1) channel rule: exactly one channel regardless of input, dropping
    # the two spatial axes' RGB triple to a single luma plane.
    assert out.shape[-1] == 1 or out.ndim == 2


@given(
    height=_dims,
    width=_dims,
    target_h=st.integers(2, 20),
    target_w=st.integers(2, 20),
    seed=st.integers(0, 2**31),
)
@_settings
def test_literal_and_column_resize_agree(
    height: int, width: int, target_h: int, target_w: int, seed: int
) -> None:
    # The planner constant-folds a literal structural parameter but resolves a
    # column one per row; both must produce identical bytes for the same value.
    arr = _rgb(height, width, seed)
    df = pl.DataFrame({"img": [_png(arr)], "h": [target_h], "w": [target_w]})

    literal = Pipeline().source("image_bytes").resize(height=target_h, width=target_w)
    column = (
        Pipeline().source("image_bytes").resize(height=pl.col("h"), width=pl.col("w"))
    )

    out = df.with_columns(
        lit=pl.col("img").cv.pipe(literal).sink("numpy"),
        col=pl.col("img").cv.pipe(column).sink("numpy"),
    )
    assert np.array_equal(
        numpy_from_struct(out["lit"][0]), numpy_from_struct(out["col"][0])
    )
