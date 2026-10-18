"""
Reference tests for binary array operations using NumPy.

These tests establish the expected behavior for element-wise operations
between arrays, and verify that polars-cv implementations match
the NumPy reference behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


def _img_df(encode_png: Callable[[np.ndarray], bytes], **images: np.ndarray):
    """One-row frame with a PNG-encoded image per keyword."""
    return pl.DataFrame({k: [encode_png(v)] for k, v in images.items()})


def _col(name: str) -> "pl.Expr":
    """An image column wrapped as a polars-cv lazy pipeline (``image_bytes``)."""
    return pl.col(name).cv.pipe(Pipeline().source("image_bytes"))


def _numpy(df: pl.DataFrame, lazy) -> np.ndarray:
    """Evaluate a lazy pipeline expression with the numpy sink."""
    return numpy_from_struct(df.select(output=lazy.sink("numpy")).row(0)[0])


@plugin_required
@pytest.mark.parity(
    ops=[
        "add",
        "subtract",
        "multiply",
        "divide",
        "apply_mask",
        "maximum",
        "minimum",
        "bitwise_and",
        "bitwise_or",
        "blend",
        "scale",
        "channel_select",
        "channel_merge",
    ],
    against="NumPy int16-clipped add/subtract, float32 normalized product, float32 division, "
    "0/1 mask product, np.maximum/minimum/bitwise_and/bitwise_or, scalar and per-channel scaling",
    tolerance="exact except the normalized product (atol 1: polars-cv rounds, the reference "
    "truncates) and per-channel weights (rtol 1e-6, f32 vs f64 weights)",
)
class TestBinaryOpsReference:
    """polars-cv binary operations must reproduce the NumPy reference semantics."""

    def test_add_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Element-wise addition with saturation for uint8, clamped to [0, 255]."""
        img1, img2 = sample_images

        result = np.clip(img1.astype(np.int16) + img2.astype(np.int16), 0, 255).astype(
            np.uint8
        )
        assert result.shape == img1.shape
        assert result.dtype == np.uint8

        df = _img_df(encode_png, a=img1, b=img2)
        actual = _numpy(df, _col("a").add(_col("b")))
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, result)

    def test_subtract_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Element-wise subtraction with saturation for uint8, clamped to [0, 255]."""
        img1, img2 = sample_images

        result = np.clip(img1.astype(np.int16) - img2.astype(np.int16), 0, 255).astype(
            np.uint8
        )
        assert result.shape == img1.shape
        assert result.dtype == np.uint8

        df = _img_df(encode_png, a=img1, b=img2)
        actual = _numpy(df, _col("a").subtract(_col("b")))
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, result)

    def test_multiply_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """
        Element-wise multiplication scaled for images: normalize to [0,1],
        multiply, scale back. That is polars-cv's ``blend``.
        """
        img1, img2 = sample_images

        result = (
            (img1.astype(np.float32) / 255) * (img2.astype(np.float32) / 255) * 255
        ).astype(np.uint8)
        assert result.shape == img1.shape
        assert result.dtype == np.uint8

        df = _img_df(encode_png, a=img1, b=img2)
        actual = _numpy(df, _col("a").blend(_col("b")))
        assert actual.dtype == np.uint8
        # atol 1: polars-cv rounds (a*b + 127) // 255, the reference truncates the f32 product.
        np.testing.assert_allclose(actual, result, atol=1)

    def test_divide_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """
        Element-wise division, then scaled to [0, 255] and clipped.

        The reference substitutes 1 for a zero divisor; polars-cv divides IEEE-style
        (inf / NaN at zero), so the comparison is over the non-zero divisors and the
        zero-divisor pixels are asserted non-finite.
        """
        img1, img2 = sample_images

        img2_safe = img2.astype(np.float32)
        img2_safe[img2_safe == 0] = 1  # Avoid division by zero
        result = np.clip(img1.astype(np.float32) / img2_safe * 255, 0, 255).astype(
            np.uint8
        )
        assert result.shape == img1.shape
        assert result.dtype == np.uint8

        df = _img_df(encode_png, a=img1, b=img2)
        quotient = _numpy(df, _col("a").divide(_col("b")))
        assert quotient.dtype == np.float32
        nonzero = img2 != 0
        assert not nonzero.all()  # the fixture does contain zero divisors
        actual = np.clip(quotient[nonzero] * np.float32(255), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(actual, result[nonzero])
        assert not np.isfinite(quotient[~nonzero]).any()

    def test_apply_mask_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        binary_mask: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Apply binary mask to image: multiply by 0/1 mask, broadcasting channels."""
        img, _ = sample_images

        mask_3d = np.expand_dims(binary_mask, axis=-1)
        result = img * mask_3d
        assert np.all(result[0, 0] == 0)  # Outside mask (corner)
        assert np.all(result[50, 50] == img[50, 50])  # Inside mask (center)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

        # polars-cv reads a u8 mask as m / 255, so the 0/1 mask becomes 0/255.
        df = _img_df(encode_png, image=img, mask=binary_mask * 255)
        mask = pl.col("mask").cv.pipe(Pipeline().source("image_bytes").grayscale())
        actual = _numpy(df, _col("image").apply_mask(mask))
        np.testing.assert_array_equal(actual, result)

    def test_apply_mask_inverted_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        binary_mask: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Apply inverted binary mask: keep exterior, zero interior."""
        img, _ = sample_images

        inverted_mask = 1 - binary_mask
        mask_3d = np.expand_dims(inverted_mask, axis=-1)
        result = img * mask_3d
        assert np.all(result[50, 50] == 0)  # Inside original mask = zeroed
        assert np.all(result[0, 0] == img[0, 0])  # Outside mask = preserved

        df = _img_df(encode_png, image=img, mask=binary_mask * 255)
        mask = pl.col("mask").cv.pipe(Pipeline().source("image_bytes").grayscale())
        actual = _numpy(df, _col("image").apply_mask(mask, invert=True))
        np.testing.assert_array_equal(actual, result)

    def test_maximum_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Element-wise maximum."""
        img1, img2 = sample_images

        result = np.maximum(img1, img2)
        assert result.shape == img1.shape
        assert result.dtype == img1.dtype
        assert np.all(result >= img1)
        assert np.all(result >= img2)

        df = _img_df(encode_png, a=img1, b=img2)
        actual = _numpy(df, _col("a").maximum(_col("b")))
        assert actual.dtype == result.dtype
        np.testing.assert_array_equal(actual, result)

    def test_minimum_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Element-wise minimum."""
        img1, img2 = sample_images

        result = np.minimum(img1, img2)
        assert result.shape == img1.shape
        assert result.dtype == img1.dtype
        assert np.all(result <= img1)
        assert np.all(result <= img2)

        df = _img_df(encode_png, a=img1, b=img2)
        actual = _numpy(df, _col("a").minimum(_col("b")))
        assert actual.dtype == result.dtype
        np.testing.assert_array_equal(actual, result)

    def test_bitwise_and_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Bitwise AND operation (useful for combining binary masks)."""
        img1, img2 = sample_images

        result = np.bitwise_and(img1, img2)
        assert result.shape == img1.shape
        assert result.dtype == img1.dtype

        df = _img_df(encode_png, a=img1, b=img2)
        actual = _numpy(df, _col("a").bitwise_and(_col("b")))
        assert actual.dtype == result.dtype
        np.testing.assert_array_equal(actual, result)

    def test_bitwise_or_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Bitwise OR operation (useful for combining binary masks)."""
        img1, img2 = sample_images

        result = np.bitwise_or(img1, img2)
        assert result.shape == img1.shape
        assert result.dtype == img1.dtype

        df = _img_df(encode_png, a=img1, b=img2)
        actual = _numpy(df, _col("a").bitwise_or(_col("b")))
        assert actual.dtype == result.dtype
        np.testing.assert_array_equal(actual, result)

    def test_broadcasting_scalar_reference(
        self, encode_png: Callable[[np.ndarray], bytes]
    ) -> None:
        """Scalar broadcasting: scale by 1.5, saturate at 255, truncate to uint8."""
        img = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        scalar = np.array([1.5])

        result = np.clip(img.astype(np.float32) * scalar, 0, 255).astype(np.uint8)
        assert result.shape == img.shape

        # `cast` rounds, so floor first to reproduce the reference's truncation.
        pipe = Pipeline().source("image_bytes").scale(1.5).clamp_max(255.0).floor()
        df = _img_df(encode_png, a=img)
        actual = _numpy(df, pl.col("a").cv.pipe(pipe.cast("u8")))
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, result)

    def test_broadcasting_per_channel_reference(
        self, encode_png: Callable[[np.ndarray], bytes]
    ) -> None:
        """
        Per-channel broadcasting (RGB to grayscale weights), as one scaled
        ``channel_select`` per channel merged back with ``channel_merge``.
        """
        img = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        channel_weights = np.array([0.299, 0.587, 0.114])  # Grayscale weights

        result = (img.astype(np.float32) * channel_weights).astype(np.float32)
        assert result.shape == img.shape

        df = _img_df(encode_png, a=img)
        scaled = [
            pl.col("a").cv.pipe(
                Pipeline()
                .source("image_bytes")
                .channel_select(c)
                .scale(float(channel_weights[c]))
            )
            for c in range(3)
        ]
        actual = _numpy(df, scaled[0].channel_merge(scaled[1], scaled[2]))
        assert actual.shape == result.shape
        assert actual.dtype == np.float32
        # rtol 1e-6: polars-cv multiplies in f32, the reference in f64 then rounds to f32.
        np.testing.assert_allclose(actual, result, rtol=1e-6)

    def test_shape_mismatch_raises(
        self, encode_png: Callable[[np.ndarray], bytes]
    ) -> None:
        """Incompatible shapes raise an error (nothing broadcasts in polars-cv)."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((50, 50, 3), dtype=np.uint8)

        # NumPy raises ValueError for shape mismatch that can't broadcast
        with pytest.raises(ValueError):
            _ = img1 + img2

        df = _img_df(encode_png, a=img1, b=img2)
        with pytest.raises(pl.exceptions.ComputeError, match="same shape"):
            _numpy(df, _col("a").add(_col("b")))


@plugin_required
class TestBinaryOpsPolarsCV:
    """
    Tests that compare polars-cv binary operations against NumPy reference.

    These tests will FAIL until binary operations are implemented in the Rust backend.
    This is the expected behavior - we're testing that polars-cv matches NumPy.
    """

    @pytest.mark.parity(
        ops=["add"],
        against="NumPy saturating uint8 add (`np.clip` of int16 sum)",
        tolerance="atol 1",
    )
    def test_add_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv add should match NumPy reference (saturating addition)."""
        img1, img2 = sample_images

        # NumPy reference behavior - saturating addition
        expected = np.clip(
            img1.astype(np.int16) + img2.astype(np.int16), 0, 255
        ).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        # This will FAIL until add is implemented in execute.rs
        result = df.select(output=expr1.add(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    @pytest.mark.parity(
        ops=["subtract"],
        against="NumPy saturating uint8 subtract (`np.clip` of int16 difference)",
        tolerance="atol 1",
    )
    def test_subtract_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv subtract should match NumPy reference (saturating subtraction)."""
        img1, img2 = sample_images

        # NumPy reference behavior - saturating subtraction
        expected = np.clip(
            img1.astype(np.int16) - img2.astype(np.int16), 0, 255
        ).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        # This will FAIL until subtract is implemented in execute.rs
        result = df.select(output=expr1.subtract(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    @pytest.mark.parity(
        ops=["multiply"],
        against="NumPy saturating uint8 multiply (`np.clip` of uint16 product)",
        tolerance="atol 1",
    )
    def test_multiply_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv multiply should match saturating multiplication semantics."""
        img1, img2 = sample_images

        # NumPy reference: saturating multiplication (clamp to 255)
        expected = np.clip(
            img1.astype(np.uint16) * img2.astype(np.uint16), 0, 255
        ).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(output=expr1.multiply(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    @pytest.mark.parity(
        ops=["blend"],
        against="NumPy rounded normalized product `(a*b + 127) // 255`, u8",
        tolerance="atol 1",
    )
    def test_blend_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv blend should match normalized multiplication semantics."""
        img1, img2 = sample_images

        # NumPy reference: normalize to [0,1], multiply, scale back
        # Using rounding division to match Rust: (a * b + 127) / 255
        expected = (
            (img1.astype(np.uint32) * img2.astype(np.uint32) + 127) // 255
        ).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(output=expr1.blend(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        np.testing.assert_allclose(actual, expected, atol=1)

    @pytest.mark.parity(
        ops=["divide"],
        against="NumPy true division into f32, IEEE inf/NaN at zero",
        tolerance="rtol 1e-6, NaN equal; dtype f32",
    )
    def test_divide_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv divide is NumPy's true division into f32, IEEE at zero."""
        img1, img2 = sample_images

        # NumPy reference: true division (x / 0 -> inf, 0 / 0 -> nan).
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = (img1.astype(np.float64) / img2.astype(np.float64)).astype(
                np.float32
            )

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "img1": [encode_png(img1)],
                "img2": [encode_png(img2)],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = df.select(output=expr1.divide(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-6, equal_nan=True)

    @pytest.mark.parity(
        ops=["apply_mask", "grayscale"],
        against="NumPy image times 0/1 mask, u8",
        tolerance="atol 1",
    )
    def test_apply_mask_matches_reference(
        self,
        sample_images: tuple[np.ndarray, np.ndarray],
        binary_mask: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """polars-cv apply_mask should match NumPy reference."""
        img, _ = sample_images

        # For polars-cv, the mask needs to be a grayscale image
        # Convert binary mask (0/1) to 0/255 grayscale
        mask_image = (binary_mask * 255).astype(np.uint8)

        # polars-cv implementation
        df = pl.DataFrame(
            {
                "image": [encode_png(img)],
                "mask": [encode_png(np.stack([mask_image] * 3, axis=-1))],
            }
        )

        img_pipe = Pipeline().source("image_bytes")
        mask_pipe = Pipeline().source("image_bytes").grayscale()

        img_expr = pl.col("image").cv.pipe(img_pipe)
        mask_expr = pl.col("mask").cv.pipe(mask_pipe)

        # This will FAIL until apply_mask is implemented in execute.rs
        result = df.select(output=img_expr.apply_mask(mask_expr).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        # Scale expected to match normalized mask behavior (0/1 -> 0/255)
        expected_scaled = (img.astype(np.float32) * (binary_mask[:, :, None])).astype(
            np.uint8
        )

        np.testing.assert_allclose(actual, expected_scaled, atol=1)
