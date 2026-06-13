"""
Tests for the opt-in ``preserve_dtype`` parameter on scalar ops.

``preserve_dtype=True`` on ``scale`` / ``clamp`` / ``adjust_brightness``
lowers to a trailing ``cast(<pre-op dtype>)`` OpSpec: the computation still
runs in f32 per the PromoteToFloat contract, but the stored result is cast
back (round-then-saturate for integer targets), e.g. u8 in → u8 out instead
of the promoted f32 (4× smaller payloads).

The cast lowering rides the existing fused-kernel cast support; these tests
pin the dtype, the exact rounding/saturation convention, the equivalence law
vs an explicit f32 pipeline, fused-vs-unfused parity, plan-time schema, and
the error cases.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required


def _png_from(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr, "RGB" if arr.ndim == 3 else "L").save(buf, format="PNG")
    return buf.getvalue()


def _run(pipe: Pipeline, png: bytes) -> np.ndarray:
    df = pl.DataFrame({"img": [png]})
    out = df.with_columns(o=pl.col("img").cv.pipe(pipe).sink("numpy"))
    return numpy_from_struct(out["o"][0])


def _u8_src() -> Pipeline:
    return Pipeline().source("image_bytes", dtype="u8")


class TestPreserveDtypeValidation:
    def test_requires_known_dtype(self) -> None:
        with pytest.raises(ValueError, match="requires a known input dtype"):
            Pipeline().source("image_bytes").scale(2.0, preserve_dtype=True)

    def test_mutually_exclusive_with_out_dtype(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _u8_src().scale(2.0, out_dtype="f32", preserve_dtype=True)

    def test_planned_dtype_is_input_dtype(self) -> None:
        pipe = _u8_src().scale(2.0, preserve_dtype=True)
        assert pipe._output_dtype == "u8"
        pipe = _u8_src().clamp(0.0, 200.0, preserve_dtype=True)
        assert pipe._output_dtype == "u8"
        pipe = _u8_src().adjust_brightness(factor=1.3, preserve_dtype=True)
        assert pipe._output_dtype == "u8"

    def test_float_input_is_noop_cast(self) -> None:
        # f32 in → scale already produces f32: no cast op is appended.
        pipe = _u8_src().cast("f32").scale(2.0, preserve_dtype=True)
        assert pipe._output_dtype == "f32"
        assert [op.op for op in pipe._ops].count("cast") == 1  # only the explicit one


@plugin_required
class TestPreserveDtypeExecution:
    def test_scale_u8_saturates_and_rounds(self) -> None:
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        arr[0, 0] = 200  # 200*2 = 400 → saturates to 255
        arr[0, 1] = 100  # 100*2 = 200
        png = _png_from(arr)

        out = _run(_u8_src().scale(2.0, preserve_dtype=True), png)
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 255
        assert out[0, 1, 0] == 200

    def test_rounding_convention_half_away(self) -> None:
        # cast uses round-half-away (.round()), not banker's rounding:
        # 101 * 0.5 = 50.5 → 51, 103 * 0.5 = 51.5 → 52.
        arr = np.zeros((1, 2, 3), dtype=np.uint8)
        arr[0, 0] = 101
        arr[0, 1] = 103
        png = _png_from(arr)

        out = _run(_u8_src().scale(0.5, preserve_dtype=True), png)
        assert int(out[0, 0, 0]) == 51
        assert int(out[0, 1, 0]) == 52

    def test_equivalence_law_vs_promoted_pipeline(self) -> None:
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        png = _png_from(arr)

        preserved = _run(_u8_src().scale(1.7, preserve_dtype=True), png)
        promoted = _run(_u8_src().scale(1.7), png)

        # The cast rounds half-away-from-zero (Rust .round()), not numpy's
        # banker's rounding — floor(x + 0.5) for the non-negative range here.
        expected = np.clip(np.floor(promoted.astype(np.float64) + 0.5), 0, 255).astype(
            np.uint8
        )
        np.testing.assert_array_equal(preserved, expected)

    def test_adjust_brightness_preserves_u8(self) -> None:
        rng = np.random.default_rng(11)
        arr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        png = _png_from(arr)

        out = _run(_u8_src().adjust_brightness(factor=1.3, preserve_dtype=True), png)
        assert out.dtype == np.uint8

        promoted = _run(_u8_src().adjust_brightness(factor=1.3), png)
        expected = np.clip(np.floor(promoted.astype(np.float64) + 0.5), 0, 255).astype(
            np.uint8
        )
        np.testing.assert_array_equal(out, expected)

    def test_fused_vs_unfused_parity(self) -> None:
        # A single scale+cast may or may not fuse depending on chain length;
        # a longer chain (scale→clamp→cast) definitely exercises the fused
        # kernel with a trailing cast. Both must agree with the equivalence
        # law and with each other on the overlapping prefix semantics.
        rng = np.random.default_rng(13)
        arr = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
        png = _png_from(arr)

        short = _run(_u8_src().scale(1.5, preserve_dtype=True), png)
        long = _run(
            _u8_src().scale(1.5).clamp(0.0, 255.0, preserve_dtype=False).cast("u8"),
            png,
        )
        # scale(1.5) of u8 ≤ 382.5; clamp(0,255) saturates exactly like the
        # u8 cast's saturation, so the two pipelines agree everywhere.
        np.testing.assert_array_equal(short, long)

    def test_lazy_mirror_executes(self) -> None:
        import polars_cv.expressions  # noqa: F401

        rng = np.random.default_rng(17)
        arr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        png = _png_from(arr)
        df = pl.DataFrame({"img": [png]})

        expr = (
            pl.col("img")
            .cv.pipe(_u8_src())
            .adjust_brightness(factor=1.2, preserve_dtype=True)
            .sink("numpy")
        )
        out = numpy_from_struct(df.with_columns(o=expr)["o"][0])
        assert out.dtype == np.uint8

    def test_f64_input_unfused_path(self) -> None:
        # Fusion refuses f64; the unfused cast path must produce the same
        # round-then-saturate semantics.
        rng = np.random.default_rng(19)
        arr = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
        png = _png_from(arr)

        out = _run(
            _u8_src().cast("f64").scale(1.5, preserve_dtype=True),
            png,
        )
        assert out.dtype == np.float64  # pre-op dtype was f64
        promoted = _run(_u8_src().cast("f64").scale(1.5), png)
        np.testing.assert_allclose(out, promoted.astype(np.float64), rtol=1e-6)
