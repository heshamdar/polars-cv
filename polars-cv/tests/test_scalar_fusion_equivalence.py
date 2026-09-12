"""Fusion equivalence for the core math primitives.

A chain of scalar ops written in one pipeline collapses into a single fused
kernel (see the Rust guard ``scalar_chain_fuses_into_one_kernel``). This suite
proves that fusion does not change results: the fused chain must match the
*unfused* execution of the same ops.

The unfused reference blocks fusion by materialising the buffer between every
op through a ``blob`` sink → ``blob`` source round-trip (the VIEW protocol is
lossless for the f32 buffers here). Because each op then lives in its own graph
fed from a freshly decoded source, no two ops ever share a node and the
optimizer has nothing to fuse — the simplest available barrier.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

Op = Callable[[Pipeline], Pipeline]


def _png(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def png() -> bytes:
    rng = np.random.default_rng(11)
    return _png(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8))


def _base() -> Pipeline:
    """Signed float base in ~[-2, 2), so every op does something."""
    return (
        Pipeline()
        .source("image_bytes")
        .cast("f32")
        .scale(4.0 / 255.0)
        .subtract_constant(2.0)
    )


def _run_fused(png: bytes, ops: list[Op]) -> np.ndarray:
    """All ops in one pipeline → the optimizer fuses adjacent scalar ops."""
    pipe = _base()
    for op in ops:
        pipe = op(pipe)
    df = pl.DataFrame({"img": [png]})
    return numpy_from_struct(
        df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy")).row(0)[0]
    )


def _run_unfused(png: bytes, ops: list[Op]) -> np.ndarray:
    """Each op in its own graph, buffers materialised via a blob round-trip.

    Nothing is chained, so there is nothing to fuse — the reference for what
    the ops compute when run one materialised pass at a time.
    """
    df = pl.DataFrame({"img": [png]})
    # Materialise the signed base once.
    df = df.select(buf=pl.col("img").cv.pipe(_base()).sink("blob"))
    for op in ops:
        df = df.select(
            buf=pl.col("buf").cv.pipe(op(Pipeline().source("blob"))).sink("blob")
        )
    return numpy_from_struct(
        df.select(o=pl.col("buf").cv.pipe(Pipeline().source("blob")).sink("numpy")).row(
            0
        )[0]
    )


#: Chains that mix the new primitives with each other and with the existing
#: fusable ops (scale/clamp). Names are for test ids only.
_CHAINS: dict[str, list[Op]] = {
    "sqrt_square_roundtrip": [
        lambda p: p.abs(),
        lambda p: p.sqrt(),
        lambda p: p.square(),
    ],
    "shift_scale_clamp": [
        lambda p: p.add_constant(1.0),
        lambda p: p.scale(0.5),
        lambda p: p.clamp_min(0.0),
        lambda p: p.clamp_max(1.0),
    ],
    "rounding_family": [
        lambda p: p.scale(3.0),
        lambda p: p.floor(),
        lambda p: p.add_constant(0.5),
        lambda p: p.ceil(),
    ],
    "sign_abs_recip": [
        lambda p: p.abs(),
        lambda p: p.add_constant(0.25),
        lambda p: p.reciprocal(),
        lambda p: p.sign(),
    ],
    "long_mixed": [
        lambda p: p.neg(),
        lambda p: p.abs(),
        lambda p: p.sqrt(),
        lambda p: p.subtract_constant(0.3),
        lambda p: p.square(),
        lambda p: p.clamp_max(0.8),
        lambda p: p.round(),
    ],
}


@plugin_required
@pytest.mark.parametrize("chain", list(_CHAINS.values()), ids=list(_CHAINS))
def test_fused_matches_unfused(chain: list[Op], png: bytes) -> None:
    fused = _run_fused(png, chain)
    unfused = _run_unfused(png, chain)
    assert fused.shape == unfused.shape
    assert fused.dtype == unfused.dtype == np.float32
    np.testing.assert_allclose(fused, unfused, rtol=1e-5, atol=1e-5, equal_nan=True)


@plugin_required
def test_blob_roundtrip_is_lossless_for_f32(png: bytes) -> None:
    """The barrier itself must be lossless, or the equivalence test is vacuous."""
    df = pl.DataFrame({"img": [png]})
    direct = numpy_from_struct(
        df.select(o=pl.col("img").cv.pipe(_base()).sink("numpy")).row(0)[0]
    )
    viablob = df.select(buf=pl.col("img").cv.pipe(_base()).sink("blob"))
    roundtripped = numpy_from_struct(
        viablob.select(
            o=pl.col("buf").cv.pipe(Pipeline().source("blob")).sink("numpy")
        ).row(0)[0]
    )
    np.testing.assert_array_equal(direct, roundtripped)
