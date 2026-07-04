"""
Reference tests for spectral / frequency-domain operations against NumPy.

The forward FFT is verified by feeding polars-cv's *own* grayscale output into
``numpy.fft.fft2`` and comparing — this isolates the transform under test from any
grayscale-formula differences. DCT is verified via round-trip (no scipy needed).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct


def _plugin_available() -> bool:
    lib_path = Path(__file__).parent.parent.parent / "python" / "polars_cv"
    return bool(list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd")))


plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


@pytest.fixture
def textured_png() -> bytes:
    """A small RGB image with varied spatial frequencies (good FFT content)."""
    from PIL import Image

    h, w = 16, 16
    yy, xx = np.mgrid[0:h, 0:w]
    r = (128 + 100 * np.cos(2 * np.pi * xx / 4)).astype(np.uint8)
    g = (128 + 80 * np.sin(2 * np.pi * yy / 8)).astype(np.uint8)
    b = ((xx * 16) % 256).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@plugin_required
class TestSpectralReference:
    def _gray(self, img_bytes: bytes) -> np.ndarray:
        """polars-cv grayscale output as a 2D float array."""
        df = pl.DataFrame({"img": [img_bytes]})
        pipe = Pipeline().source("image_bytes").grayscale()
        out = numpy_from_struct(
            df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy")).row(0)[0]
        )
        return out.reshape(out.shape[0], out.shape[1]).astype(np.float64)

    def test_fft2_matches_numpy(self, textured_png: bytes) -> None:
        gray = self._gray(textured_png)
        expected = np.fft.fft2(gray)

        df = pl.DataFrame({"img": [textured_png]})
        pipe = Pipeline().source("image_bytes").grayscale().fft2()
        spec = numpy_from_struct(
            df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy")).row(0)[0]
        )
        assert spec.shape == (16, 16, 2)
        actual = spec[..., 0] + 1j * spec[..., 1]
        # f32 transform: tolerance scaled to the DC magnitude.
        scale = float(np.abs(expected).max())
        np.testing.assert_allclose(actual, expected, atol=1e-3 * scale)

    def test_magnitude_and_power_spectra(self, textured_png: bytes) -> None:
        gray = self._gray(textured_png)
        fexp = np.fft.fft2(gray)
        df = pl.DataFrame({"img": [textured_png]})

        mag = numpy_from_struct(
            df.select(
                o=pl.col("img")
                .cv.pipe(Pipeline().source("image_bytes").grayscale().fft2().complex_magnitude())
                .sink("numpy")
            ).row(0)[0]
        )
        scale = float(np.abs(fexp).max())
        np.testing.assert_allclose(mag.reshape(16, 16), np.abs(fexp), atol=1e-3 * scale)

        power = numpy_from_struct(
            df.select(
                o=pl.col("img")
                .cv.pipe(Pipeline().source("image_bytes").grayscale().fft2().complex_power())
                .sink("numpy")
            ).row(0)[0]
        )
        np.testing.assert_allclose(
            power.reshape(16, 16), np.abs(fexp) ** 2, atol=1e-2 * scale**2
        )

    def test_fftshift_centers_spectrum(self, textured_png: bytes) -> None:
        gray = self._gray(textured_png)
        expected = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
        df = pl.DataFrame({"img": [textured_png]})
        pipe = (
            Pipeline().source("image_bytes").grayscale().fft2().fftshift().complex_magnitude()
        )
        mag = numpy_from_struct(
            df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy")).row(0)[0]
        ).reshape(16, 16)
        scale = float(expected.max())
        np.testing.assert_allclose(mag, expected, atol=1e-3 * scale)

    def test_fft_ifft_round_trip(self, textured_png: bytes) -> None:
        gray = self._gray(textured_png)
        df = pl.DataFrame({"img": [textured_png]})
        pipe = Pipeline().source("image_bytes").grayscale().fft2().ifft2()
        back = numpy_from_struct(
            df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy")).row(0)[0]
        ).reshape(16, 16)
        np.testing.assert_allclose(back, gray, atol=1e-2)

    def test_dct_idct_round_trip(self, textured_png: bytes) -> None:
        gray = self._gray(textured_png)
        df = pl.DataFrame({"img": [textured_png]})
        pipe = Pipeline().source("image_bytes").grayscale().dct2().idct2()
        back = numpy_from_struct(
            df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy")).row(0)[0]
        ).reshape(16, 16)
        np.testing.assert_allclose(back, gray, atol=1e-2)

    def test_complex_mul_convolution_theorem(self, textured_png: bytes) -> None:
        # ifft2(fft2(a) * fft2(b)) == circular convolution of a and b.
        gray = self._gray(textured_png)
        expected = np.fft.ifft2(np.fft.fft2(gray) * np.fft.fft2(gray)).real

        df = pl.DataFrame({"img": [textured_png]})
        a = pl.col("img").cv.pipe(Pipeline().source("image_bytes").grayscale().fft2())
        b = pl.col("img").cv.pipe(Pipeline().source("image_bytes").grayscale().fft2())
        product = a.complex_mul(b).ifft2()
        back = numpy_from_struct(df.select(o=product.sink("numpy")).row(0)[0]).reshape(
            16, 16
        )
        scale = float(np.abs(expected).max())
        np.testing.assert_allclose(back, expected, atol=1e-3 * scale)
