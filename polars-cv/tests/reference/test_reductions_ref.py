"""
Reference tests for statistical reduction operations against NumPy.

Each test computes the NumPy reduction and the same reduction with polars-cv on
the same pixels (PNG-encoded RGB image, or a nested-list ``f32`` source), then
asserts the two agree.

polars-cv reduces over one axis at a time, so a per-channel reduction over
``(H, W)`` is expressed as ``reshape([H * W, C])`` followed by an ``axis=0``
reduction.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

H, W, C = 100, 100, 3


def _png(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _src() -> Pipeline:
    return Pipeline().source("image_bytes")


def _scalar(pipe: Pipeline, arr: np.ndarray) -> float:
    """Run a scalar-domain pipeline over a PNG-encoded image; return the value."""
    df = pl.DataFrame({"img": [_png(arr)]})
    return df.select(o=pl.col("img").cv.pipe(pipe).sink("native"))["o"][0]


def _array(pipe: Pipeline, arr: np.ndarray) -> np.ndarray:
    """Run a buffer-domain pipeline over a PNG-encoded image; return the array."""
    df = pl.DataFrame({"img": [_png(arr)]})
    out = df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy"))
    return numpy_from_struct(out.row(0)[0])


@plugin_required
@pytest.mark.parity(
    ops=[
        "reduce_max",
        "reduce_min",
        "reduce_mean",
        "reduce_std",
        "reduce_sum",
        "reduce_argmax",
        "reduce_argmin",
    ],
    against="NumPy `np.max/min/mean/std/sum/argmax/argmin` on a random u8 RGB image",
    tolerance="exact for max/min/argmax/argmin/sum; rtol 1e-9 for mean/std",
)
class TestReductionsReference:
    """polars-cv reductions must equal the NumPy reductions on the same pixels."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Sample image with known statistical properties."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (H, W, C), dtype=np.uint8)

    def _per_channel(self) -> Pipeline:
        """Flatten (H, W) so an axis=0 reduction is a per-channel one."""
        return _src().reshape([H * W, C])

    def test_max_global_reference(self, sample_image: np.ndarray) -> None:
        """Global maximum across entire array: a single scalar."""
        expected = np.max(sample_image)
        assert expected == 255
        assert _scalar(_src().reduce_max(), sample_image) == expected

    def test_max_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """Maximum per channel (reduce over H, W): shape (3,) for RGB."""
        expected = np.max(sample_image, axis=(0, 1))
        assert expected.shape == (3,)
        actual = _array(self._per_channel().reduce_max(axis=0), sample_image)
        np.testing.assert_array_equal(actual, expected)

    def test_max_per_row_reference(self, sample_image: np.ndarray) -> None:
        """Maximum per row (reduce over W axis)."""
        expected = np.max(sample_image, axis=1)
        assert expected.shape == (H, C)
        actual = _array(_src().reduce_max(axis=1), sample_image)
        np.testing.assert_array_equal(actual, expected)

    def test_min_global_reference(self, sample_image: np.ndarray) -> None:
        """Global minimum across entire array."""
        expected = np.min(sample_image)
        assert expected == 0
        assert _scalar(_src().reduce_min(), sample_image) == expected

    def test_min_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """Minimum per channel."""
        expected = np.min(sample_image, axis=(0, 1))
        assert expected.shape == (3,)
        actual = _array(self._per_channel().reduce_min(axis=0), sample_image)
        np.testing.assert_array_equal(actual, expected)

    def test_mean_global_reference(self, sample_image: np.ndarray) -> None:
        """Global mean across entire array (uniform [0,256) is about 127.5)."""
        expected = np.mean(sample_image)
        assert 120 < expected < 135
        actual = _scalar(_src().reduce_mean(), sample_image)
        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_mean_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """Mean per channel."""
        expected = np.mean(sample_image, axis=(0, 1))
        assert expected.shape == (3,)
        actual = _array(self._per_channel().reduce_mean(axis=0), sample_image)
        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_std_population_reference(self, sample_image: np.ndarray) -> None:
        """Population standard deviation (ddof=0); uniform [0,256) is about 74."""
        expected = np.std(sample_image, ddof=0)
        assert 70 < expected < 78
        actual = _scalar(_src().reduce_std(ddof=0), sample_image)
        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_std_sample_reference(self, sample_image: np.ndarray) -> None:
        """Sample standard deviation (ddof=1): slightly higher than population."""
        expected_pop = np.std(sample_image, ddof=0)
        expected_sample = np.std(sample_image, ddof=1)
        assert expected_sample > expected_pop

        actual_pop = _scalar(_src().reduce_std(ddof=0), sample_image)
        actual_sample = _scalar(_src().reduce_std(ddof=1), sample_image)
        np.testing.assert_allclose(actual_pop, expected_pop, rtol=1e-9)
        np.testing.assert_allclose(actual_sample, expected_sample, rtol=1e-9)
        assert actual_sample > actual_pop

    def test_std_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """Standard deviation per channel."""
        expected = np.std(sample_image, axis=(0, 1))
        assert expected.shape == (3,)
        actual = _array(self._per_channel().reduce_std(axis=0), sample_image)
        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_sum_global_reference(self, sample_image: np.ndarray) -> None:
        """Global sum across entire array (integer-exact in polars-cv's f64)."""
        expected = np.sum(sample_image, dtype=np.int64)
        assert abs(expected - H * W * C * 127.5) < H * W * C * 127.5 * 0.1
        actual = _scalar(_src().reduce_sum(), sample_image)
        assert actual == expected

    def test_sum_per_channel_reference(self, sample_image: np.ndarray) -> None:
        """
        Sum per channel.

        ``reduce_sum`` has no ``axis``, so the per-channel sum is the per-channel
        ``reduce_mean`` times the pixel count; the product is an exact integer in f64.
        """
        expected = np.sum(sample_image, axis=(0, 1), dtype=np.int64)
        assert expected.shape == (3,)
        mean = _array(self._per_channel().reduce_mean(axis=0), sample_image)
        np.testing.assert_array_equal(
            np.rint(mean * (H * W)).astype(np.int64), expected
        )

    def test_argmax_global_reference(self, sample_image: np.ndarray) -> None:
        """Index of global maximum (flattened); first occurrence on ties."""
        expected = np.argmax(sample_image)
        assert sample_image.flat[expected] == np.max(sample_image)
        # reduce_argmax needs an axis, so flatten to 1-D first.
        actual = _array(_src().reshape([H * W * C]).reduce_argmax(axis=0), sample_image)
        assert int(actual.squeeze()) == expected

    def test_argmax_axis_reference(self, sample_image: np.ndarray) -> None:
        """Index of maximum along axis."""
        expected = np.argmax(sample_image, axis=0)
        assert expected.shape == (H, C)
        actual = _array(_src().reduce_argmax(axis=0), sample_image)
        np.testing.assert_array_equal(actual, expected)

    def test_argmin_global_reference(self, sample_image: np.ndarray) -> None:
        """Index of global minimum (flattened); first occurrence on ties."""
        expected = np.argmin(sample_image)
        assert sample_image.flat[expected] == np.min(sample_image)
        actual = _array(_src().reshape([H * W * C]).reduce_argmin(axis=0), sample_image)
        assert int(actual.squeeze()) == expected

    def test_reduction_preserves_type_reference(self) -> None:
        """
        Mean of f32 input is a floating-point value matching NumPy.

        polars-cv's scalar mean is f64, so compare it against NumPy's mean of the
        same f32 values taken in f64.
        """
        float_arr = np.random.default_rng(42).random((10, 10)).astype(np.float32)
        assert np.mean(float_arr).dtype in (np.float32, np.float64)
        assert np.mean(float_arr.astype(np.float64)).dtype == np.float64

        df = pl.DataFrame(
            {"x": [float_arr.astype(np.float64).tolist()]},
            schema={"x": pl.List(pl.List(pl.Float64))},
        )
        pipe = Pipeline().source("list", dtype="f32").reduce_mean()
        out = df.select(o=pl.col("x").cv.pipe(pipe).sink("native"))
        assert out.schema["o"].is_float()
        np.testing.assert_allclose(
            out["o"][0], np.mean(float_arr.astype(np.float64)), rtol=1e-6
        )

    def test_empty_axis_keeps_shape_reference(self) -> None:
        """Reducing with keepdims=True preserves dimensions (``expand_dims`` after)."""
        arr = np.random.default_rng(42).integers(0, 256, (10, 20, 3), dtype=np.uint8)
        expected = np.mean(arr, axis=1, keepdims=True)
        assert expected.shape == (10, 1, 3)
        actual = _array(_src().reduce_mean(axis=1).expand_dims(axis=1), arr)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-9)
