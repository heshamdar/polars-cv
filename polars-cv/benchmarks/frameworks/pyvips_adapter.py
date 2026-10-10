"""
pyvips (libvips) framework adapter for benchmarking.

libvips is demand-driven: an operation builds a node in a lazy pipeline and
nothing runs until pixels are asked for, when the pipeline runs in tiles over
libvips's own thread pool. Two things follow for a fair benchmark:

- every result is materialised (``copy_memory``), or a "pipeline" would time
  only building the graph;
- libvips's operation cache is disabled, or every timed iteration after the
  first would be a cache hit on identical arguments.

Ops that need a global statistic (``normalize``, ``adjust_contrast``) would
otherwise evaluate their input twice (once for the statistic, once for the
output), so they materialise their input first, as a libvips user would.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import (
    BaseFrameworkAdapter,
    OperationParams,
    gamma_lut,
    letterbox_geometry,
    rotation_matrix,
)

if TYPE_CHECKING:
    import numpy.typing as npt

#: polars-cv's grayscale weights (ITU-R BT.601).
_LUMA = [[0.299, 0.587, 0.114]]


class PyVipsAdapter(BaseFrameworkAdapter):
    """Adapter for pyvips, the Python binding of libvips.

    Images are ``pyvips.Image`` objects in RGB band order, u8 unless an op
    promotes to float (as polars-cv's does).
    """

    name: str = "pyvips"
    supports_gpu: bool = False

    def __init__(self) -> None:
        """Import pyvips (when installed) and disable its operation cache."""
        self._pyvips: Any = None
        if self.is_available():
            self._get_pyvips().cache_set_max(0)

    def is_available(self) -> bool:
        """
        Check if pyvips and its libvips are available.

        Returns:
            True if pyvips can be imported, False otherwise.
        """
        try:
            import pyvips  # noqa: F401

            return True
        except (ImportError, OSError):
            return False

    def _get_pyvips(self) -> Any:
        """Get the pyvips module."""
        if self._pyvips is None:
            import pyvips

            self._pyvips = pyvips
        return self._pyvips

    # --- loading ----------------------------------------------------------

    def load_from_file(self, path: Path) -> Any:
        """Open an image file (decoded on demand)."""
        return self._get_pyvips().Image.new_from_file(str(path))

    def load_from_bytes(self, data: bytes) -> Any:
        """Open encoded image bytes (decoded on demand)."""
        return self._get_pyvips().Image.new_from_buffer(data, "")

    def prepare_decoded_images(self, png_bytes_list: list[bytes]) -> list[Any]:
        """Decode into memory now, so timed runs start from pixels."""
        return [self.load_from_bytes(data).copy_memory() for data in png_bytes_list]

    def run_pipeline(
        self, images: list[Any], operations: list[OperationParams]
    ) -> list[Any]:
        """Apply ``operations`` to each image and materialise the result."""
        results = []
        for img in images:
            for op in operations:
                img = self.apply_operation(img, op)
            results.append(img.copy_memory())
        return results

    # --- helpers ------------------------------------------------------------

    def _gray(self, img: Any) -> Any:
        """BT.601 luma rounded to u8; a one-band image is returned as is."""
        if img.bands == 1:
            return img
        return (img.recomb(_LUMA) + 0.5).cast("uchar")

    def _matrix(self, values: list[list[float]]) -> Any:
        """A convolution mask (scale 1, offset 0) from rows of coefficients."""
        return self._get_pyvips().Image.new_from_list(values, scale=1, offset=0)

    def _correlate(self, img: Any, kernel: "npt.NDArray[np.float64]") -> Any:
        """Correlate every band with ``kernel`` in float, replicated border.

        libvips's ``conv`` correlates without flipping the mask (as polars-cv
        and ``cv2.filter2D`` do), but its output shrinks by the mask's rim, so
        the input is first extended by copying its edge pixels.
        """
        r = kernel.shape[0] // 2
        padded = img.embed(r, r, img.width + 2 * r, img.height + 2 * r, extend="copy")
        out = padded.conv(self._matrix(kernel.tolist()), precision="float")
        return out.crop(r, r, img.width, img.height) if out.width > img.width else out

    def _rank(self, img: Any, ksize: int, which: str) -> Any:
        """A ``ksize`` square min or max filter (grayscale erosion/dilation)."""
        index = 0 if which == "min" else ksize * ksize - 1
        return img.rank(ksize, ksize, index)

    def _affine(self, img: Any, mat: Any, out_h: int, out_w: int) -> Any:
        """Bilinear warp by the forward 2x3 ``mat`` (pixel-centre coordinates,
        which is also libvips's convention) onto an ``out_h`` x ``out_w``
        canvas with a zero border."""
        pyvips = self._get_pyvips()
        a, b, tx = (float(v) for v in mat[0])
        c, d, ty = (float(v) for v in mat[1])
        return img.affine(
            [a, b, c, d],
            interpolate=pyvips.Interpolate.new("bilinear"),
            oarea=[0, 0, out_w, out_h],
            odx=tx,
            ody=ty,
            background=[0],
            extend="background",
        )

    # --- operations --------------------------------------------------------

    def resize(self, img: Any, height: int, width: int) -> Any:
        """Bilinear, antialiased on a downscale, as polars-cv's and Pillow's.

        A shrink runs ``reduceh``/``reducev`` with the linear kernel (libvips's
        antialiased path); an enlargement runs ``affine`` with the half-pixel
        alignment polars-cv uses, ``src = (dst + 0.5) / scale - 0.5`` —
        ``resize`` itself aligns an enlargement differently. On a shrink
        the interior agrees with polars-cv to 2 levels; the outer two pixels
        differ more, since libvips extends the edge where polars-cv
        renormalises the clipped kernel.
        """
        pyvips = self._get_pyvips()
        sx, sy = width / img.width, height / img.height
        if sx < 1:
            img = img.reduceh(1 / sx, kernel="linear")
        if sy < 1:
            img = img.reducev(1 / sy, kernel="linear")
        ax, ay = max(sx, 1.0), max(sy, 1.0)
        if ax > 1 or ay > 1:
            img = img.affine(
                [ax, 0, 0, ay],
                interpolate=pyvips.Interpolate.new("bilinear"),
                oarea=[0, 0, width, height],
                idx=0.5,
                idy=0.5,
                odx=-0.5,
                ody=-0.5,
                extend="copy",
            )
        if (img.width, img.height) != (width, height):
            img = img.crop(0, 0, width, height)
        return img

    def grayscale(self, img: Any) -> Any:
        """BT.601 luma (``recomb``), rounded to u8."""
        return self._gray(img)

    def normalize(self, img: Any) -> Any:
        """Min-max to [0, 1] as f32 (one ``stats`` pass for both bounds)."""
        img = img.copy_memory()
        stats = img.stats()
        lo, hi = stats(0, 0)[0], stats(1, 0)[0]
        if hi - lo > 0:
            return img.linear(1 / (hi - lo), -lo / (hi - lo))
        return img.cast("float")

    def flip_horizontal(self, img: Any) -> Any:
        """``fliphor``."""
        return img.fliphor()

    def flip_vertical(self, img: Any) -> Any:
        """``flipver``."""
        return img.flipver()

    def crop(self, img: Any, top: int, left: int, height: int, width: int) -> Any:
        """``crop`` (libvips's ``extract_area``)."""
        return img.crop(left, top, width, height)

    def blur(self, img: Any, sigma: float) -> Any:
        """``gaussblur`` with polars-cv's radius, ``ceil(3 sigma)``.

        libvips sizes its mask by where the Gaussian falls below ``min_ampl``;
        the amplitude half a pixel past polars-cv's radius ends it there.
        """
        radius = math.ceil(3 * sigma)
        min_ampl = math.exp(-((radius + 0.5) ** 2) / (2 * sigma * sigma))
        return img.gaussblur(sigma, min_ampl=min_ampl, precision="integer")

    def threshold(self, img: Any, value: int) -> Any:
        """``> value`` on the grayscale image: libvips yields 255 or 0."""
        return self._gray(img) > value

    def rotate(self, img: Any, angle: float, *, expand: bool = False) -> Any:
        """Clockwise, as polars-cv's ``rotate``: lattice angles permute
        pixels, any other angle resamples about the centre."""
        quarter = angle % 360
        if quarter == 0:
            return img.copy()
        if quarter == 90:
            return img.rot90()
        if quarter == 180:
            return img.rot180()
        if quarter == 270:
            return img.rot270()
        mat, (out_h, out_w) = rotation_matrix(
            img.height, img.width, angle, expand=expand
        )
        return self._affine(img, mat, out_h, out_w)

    def erode(self, img: Any, ksize: int, iterations: int = 1) -> Any:
        """Grayscale erosion: a min ``rank`` filter."""
        img = self._gray(img)
        for _ in range(iterations):
            img = self._rank(img, ksize, "min")
        return img

    def dilate(self, img: Any, ksize: int, iterations: int = 1) -> Any:
        """Grayscale dilation: a max ``rank`` filter."""
        img = self._gray(img)
        for _ in range(iterations):
            img = self._rank(img, ksize, "max")
        return img

    def morphology(self, img: Any, op: str, ksize: int) -> Any:
        """Opening, closing or gradient from min/max ``rank`` filters."""
        img = self._gray(img)
        if op == "open":
            return self._rank(self._rank(img, ksize, "min"), ksize, "max")
        if op == "close":
            return self._rank(self._rank(img, ksize, "max"), ksize, "min")
        return (self._rank(img, ksize, "max") - self._rank(img, ksize, "min")).cast(
            "uchar"
        )

    def invert(self, img: Any) -> Any:
        """``invert`` (255 - pixel on u8)."""
        return img.invert()

    def adjust_contrast(self, img: Any, factor: float) -> Any:
        """``(pixel - mean) * factor + mean`` over all bands, as f32."""
        img = img.copy_memory()
        mean = img.avg()
        return img.linear(factor, mean * (1 - factor))

    def adjust_brightness(self, img: Any, factor: float) -> Any:
        """``pixel * factor`` clamped to [0, 255], as f32."""
        return img.linear(factor, 0).clamp(min=0, max=255)

    def adjust_gamma(self, img: Any, gamma: float) -> Any:
        """A u8 -> f32 lookup table (``maplut`` keeps the table's format)."""
        lut = self._get_pyvips().Image.new_from_array(gamma_lut(gamma)[None, :])
        return img.maplut(lut)

    def sharpen(self, img: Any, strength: float = 1.0) -> Any:
        """polars-cv's 3x3 kernel (sum 1), f32, replicated border."""
        kernel = np.full((3, 3), -strength)
        kernel[1, 1] = 1 + 8 * strength
        return self._correlate(img, kernel)

    def pad(
        self, img: Any, top: int, bottom: int, left: int, right: int, value: int = 0
    ) -> Any:
        """``embed`` on a constant background."""
        return img.embed(
            left,
            top,
            img.width + left + right,
            img.height + top + bottom,
            extend="background",
            background=[value],
        )

    def histogram_equalize(self, img: Any) -> Any:
        """``hist_equal`` on the grayscale image."""
        return self._gray(img).hist_equal()

    def sobel(self, img: Any, axis: str = "x") -> Any:
        """The 3x3 Sobel kernel on the grayscale image, f32. (libvips's own
        ``sobel`` returns the gradient magnitude, not one axis.)"""
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        return self._correlate(self._gray(img), kernel if axis == "x" else kernel.T)

    def laplacian(self, img: Any) -> Any:
        """The 4-neighbour 3x3 Laplacian on the grayscale image, f32."""
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        return self._correlate(self._gray(img), kernel)

    def convolve2d(self, img: Any, kernel: tuple[float, ...]) -> Any:
        """Correlate every band with the square ``kernel``, f32."""
        side = int(round(len(kernel) ** 0.5))
        return self._correlate(img, np.asarray(kernel).reshape(side, side))

    def letterbox(self, img: Any, height: int, width: int) -> Any:
        """Linear-kernel fit, ``embed``-ded centred on a black canvas."""
        new_h, new_w, top, left = letterbox_geometry(
            img.height, img.width, height, width
        )
        fitted = self.resize(img, new_h, new_w)
        return fitted.embed(
            left, top, width, height, extend="background", background=[0]
        )

    def warp_affine(
        self, img: Any, matrix: tuple[float, ...], height: int, width: int
    ) -> Any:
        """``affine`` with the forward matrix (:meth:`_affine`)."""
        return self._affine(img, np.asarray(matrix).reshape(2, 3), height, width)

    def to_numpy(self, img: Any) -> "npt.NDArray[Any]":
        """``(H, W)`` for one band, ``(H, W, C)`` otherwise."""
        arr = np.asarray(img.numpy())
        return arr[..., 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr
