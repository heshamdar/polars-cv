"""
Pixeltable framework adapter for benchmarking.

Pixeltable (https://pixeltable.com) is the third engine in this comparison that
puts images in a table, and it is architecturally the most different of the
three. Daft and polars-cv are in-process query engines: you hand them a batch,
they compute, they hand it back, and nothing persists. Pixeltable is a
**stateful store** — an embedded Postgres plus a media store on disk — where
you declare *computed columns* and it materializes and caches them, keeping
them up to date as rows arrive.

That difference drives two things the benchmark has to be honest about.

**There is no pre-decoded in-memory column.** Every other adapter implements
`prepare_decoded_images` by decoding once, outside the timed section, so the
scenarios measure operations alone. Pixeltable cannot: its images live in the
media store as files, and every query reads and decodes them again. So its
single-op and pipeline numbers **include a decode that the other frameworks'
numbers exclude**, and that is not an artifact this adapter can remove — it is
the data model. Decode is the larger share: a bare `select(img).collect()` runs
at ~646 img/s on 256^2 images against ~385 img/s for select-with-a-resize, so
roughly 60% of the work in any Pixeltable cell here is getting the pixels back
out of the store. The scenario where the comparison is genuinely like-for-like
is `e2e_workflow`, which starts from files for everyone.

**Recomputation is the wrong axis for it.** The scenarios re-run their pipeline
every iteration, which is precisely what Pixeltable is built never to do: a
computed column is calculated once and read thereafter. `udf_path_probe.py`'s
sibling, `incremental_probe.py`, measures that model on its own terms.

Two adapters, following the same rule as the Daft pair:

``pixeltable`` (:class:`PixeltableNativeAdapter`)
    Only Pixeltable's own image expressions, which are a passthrough to PIL:
    `resize`, `crop`, `rotate`, `convert`, `transpose`, `point`, `blend`,
    `histogram`, `entropy`, `thumbnail`, `quantize`, `getchannel`. That covers
    10 of the harness's 20 single ops — including flips, which Daft cannot do —
    with `point()` supplying threshold, invert and brightness as lookup tables.
    Anything else raises.

``pixeltable-udf`` (:class:`PixeltableUDFAdapter`)
    Fills the remaining gaps with a `@pxt.udf` over :class:`OpenCVAdapter`, so
    the kernels are byte-identical to the `opencv` adapter's and the difference
    is engine overhead alone.

Three constraints of Pixeltable's API shape the code below, all of them found
by validating output against OpenCV rather than by reading docs:

- **`resize` exposes no resampling filter.** Its signature is
  `resize(self, size)`, so it uses PIL's default, which is **bicubic** —
  verified against `PIL.Image.BICUBIC` exactly. Every other adapter here is
  pinned to bilinear, so Pixeltable's resize output legitimately differs.
- **`rotate` takes an integer angle and no `expand`.** Its signature is
  `rotate(self, angle: Int)`, so a rotation that grows the canvas is not
  expressible; `rotate_45` (expand=True) falls to the UDF path.
- **UDFs must live in a named module.** Pixeltable rejects a UDF defined in a
  script's global namespace or built on the fly inside a method, so the gap
  filler is the single module-level :func:`_apply_op` below, parameterized by a
  JSON operation spec rather than by closing over one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseFrameworkAdapter, OperationParams, OperationType
from .opencv_adapter import OpenCVAdapter

if TYPE_CHECKING:
    import numpy.typing as npt

#: PIL's `Transpose` codes, which `pxtf.image.transpose` takes directly.
_FLIP_LEFT_RIGHT = 0
_FLIP_TOP_BOTTOM = 1

#: Operations Pixeltable can express with its own (PIL-backed) expressions.
#: Widening this without a real expression behind it is the one way these
#: benchmarks can lie.
NATIVE_OPS: frozenset[OperationType] = frozenset(
    {
        OperationType.RESIZE,
        OperationType.GRAYSCALE,
        OperationType.CROP,
        OperationType.FLIP_H,
        OperationType.FLIP_V,
        OperationType.THRESHOLD,
        OperationType.INVERT,
        OperationType.ADJUST_BRIGHTNESS,
    }
)


def _lut(values: list[int], bands: int) -> list[int]:
    """
    Repeat a 256-entry lookup table once per band.

    PIL's `point()` wants `256 * bands` entries for a multi-band image; handing
    an RGB image a bare 256-entry table is a runtime error, not a broadcast.

    Args:
        values: The 256-entry table for one band.
        bands: Number of bands in the image the table will be applied to.

    Returns:
        A lookup table sized for `bands`.
    """
    return values * bands


def _threshold_lut(value: int) -> list[int]:
    """
    Build the one-band lookup table for a binary threshold.

    Args:
        value: Threshold; pixels above it become 255.

    Returns:
        A 256-entry table.
    """
    return [0 if i <= value else 255 for i in range(256)]


def _invert_lut() -> list[int]:
    """
    Build the one-band lookup table that inverts pixel values.

    Returns:
        A 256-entry table.
    """
    return [255 - i for i in range(256)]


def _brightness_lut(factor: float) -> list[int]:
    """
    Build the one-band lookup table for a brightness adjustment.

    Matches `OpenCVAdapter.adjust_brightness`: scale, then clip to [0, 255].

    Args:
        factor: Brightness multiplier.

    Returns:
        A 256-entry table.
    """
    return [min(255, max(0, int(i * factor))) for i in range(256)]


class PixeltableAdapter(BaseFrameworkAdapter):
    """
    Adapter for Pixeltable image processing.

    Attributes:
        name: Human-readable name of the adapter.
        allow_udf: Whether operations with no native Pixeltable expression may
            fall back to a `@pxt.udf` over OpenCV kernels.
    """

    supports_gpu: bool = False
    columnar: bool = True

    #: Pixeltable persists every table it is given, so a run that created one
    #: per benchmark cell would leave hundreds of copies of the image set in the
    #: media store — gigabytes. The scenarios hold at most two prepared sets
    #: alive at once (the benchmark set and the warmup set), so slots rotate
    #: between two fixed names and each `create_table` replaces the older one.
    _SLOTS = 2

    def __init__(self, allow_udf: bool = False) -> None:
        """
        Initialize the Pixeltable adapter.

        Args:
            allow_udf: If True, fill gaps in the native op surface with UDFs.
                If False, unsupported ops raise NotImplementedError.
        """
        self.allow_udf = allow_udf
        self.name = "pixeltable-udf" if allow_udf else "pixeltable"
        self._pxt: Any = None
        self._cv = OpenCVAdapter()
        self._slot = 0
        self._tmpdirs: list[Any] = [None] * self._SLOTS
        self._probe: "npt.NDArray[np.uint8] | None" = None

    def is_available(self) -> bool:
        """
        Check if Pixeltable is available.

        Returns:
            True if pixeltable can be imported, False otherwise.
        """
        try:
            import pixeltable  # noqa: F401
        except ImportError:
            return False
        return self._cv.is_available() if self.allow_udf else True

    def _get_pxt(self) -> Any:
        """Get the pixeltable module, initializing the store on first use."""
        if self._pxt is None:
            import pixeltable as pxt

            pxt.init()
            pxt.create_dir("bench", if_exists="ignore")
            self._pxt = pxt
        return self._pxt

    # ------------------------------------------------------------------
    # Expression construction
    # ------------------------------------------------------------------

    def _can_be_native(self, op: OperationParams) -> bool:
        """
        Decide whether Pixeltable can express this exact operation itself.

        Membership of :data:`NATIVE_OPS` is necessary but not sufficient:
        `rotate` is native only for a whole-degree angle with no canvas
        expansion, because its signature is `rotate(self, angle: Int)`.

        Args:
            op: Operation to check.

        Returns:
            True if a native expression covers it.
        """
        if op.operation is OperationType.ROTATE:
            return not op.expand and float(op.angle).is_integer()
        return op.operation in NATIVE_OPS

    def _native_expr(self, expr: Any, op: OperationParams, bands: int) -> Any:
        """
        Apply one natively-supported operation to a Pixeltable expression.

        Args:
            expr: Pixeltable expression over an image column.
            op: Operation to apply.
            bands: Channel count of the incoming image, needed to size the
                `point()` lookup tables.

        Returns:
            The extended expression.

        Raises:
            ValueError: If ``op`` has no native expression.
        """
        kind = op.operation
        if kind == OperationType.RESIZE:
            # PIL order is (width, height). No resampling filter is exposed, so
            # this is PIL's default (bicubic) — see the module docstring.
            return expr.resize((op.width, op.height))
        if kind == OperationType.GRAYSCALE:
            return expr.convert("L")
        if kind == OperationType.CROP:
            # PIL's box is (left, top, right, bottom), not (x, y, w, h).
            return expr.crop(
                [
                    op.crop_left,
                    op.crop_top,
                    op.crop_left + op.crop_width,
                    op.crop_top + op.crop_height,
                ]
            )
        if kind == OperationType.FLIP_H:
            return expr.transpose(_FLIP_LEFT_RIGHT)
        if kind == OperationType.FLIP_V:
            return expr.transpose(_FLIP_TOP_BOTTOM)
        if kind == OperationType.ROTATE:
            # PIL rotates counter-clockwise; `OpenCVAdapter.rotate` negates the
            # angle into getRotationMatrix2D, i.e. clockwise. Negate to agree.
            return expr.rotate(-int(op.angle))
        if kind == OperationType.THRESHOLD:
            # OpenCV's threshold grayscales first; match it. One band after.
            return expr.convert("L").point(_threshold_lut(op.threshold_value))
        if kind == OperationType.INVERT:
            return expr.point(_lut(_invert_lut(), bands))
        if kind == OperationType.ADJUST_BRIGHTNESS:
            return expr.point(_lut(_brightness_lut(op.brightness_factor), bands))
        msg = f"{kind} is not a native Pixeltable expression"
        raise ValueError(msg)

    def _build_expr(self, column: Any, operations: list[OperationParams]) -> Any:
        """
        Build the Pixeltable expression for a list of operations.

        A NumPy probe is threaded through the chain so two things are known at
        each step: the band count, which the `point()` lookup tables need, and
        whether the data has become float, which decides both that no image
        expression can apply any more and which UDF to reach for.

        Args:
            column: The table's image column reference.
            operations: Operations to apply, in order.

        Returns:
            A Pixeltable expression producing the processed result.

        Raises:
            NotImplementedError: If ``allow_udf`` is False and an operation has
                no native Pixeltable expression.
        """
        from . import _pixeltable_udfs as udfs

        self._get_pxt()
        if self._probe is None:
            msg = "call prepare_decoded_images() before building an expression"
            raise RuntimeError(msg)

        expr = column
        probe = self._probe
        is_float = False

        for op in operations:
            bands = 1 if probe.ndim == 2 else probe.shape[2]
            native = self._can_be_native(op) and not is_float

            if native:
                expr = self._native_expr(expr, op, bands)
                probe = self._cv.apply_operation(probe, op)
                continue

            if not self.allow_udf:
                if is_float:
                    msg = (
                        "Pixeltable cannot apply "
                        f"{op.operation.name.lower()} to a float column "
                        "(image expressions are 8-bit only)"
                    )
                else:
                    msg = (
                        "Pixeltable has no native expression for "
                        f"{op.operation.name.lower()}"
                        + (
                            " with expand=True or a fractional angle"
                            if op.operation is OperationType.ROTATE
                            else ""
                        )
                    )
                raise NotImplementedError(msg)

            probe = self._cv.apply_operation(probe, op)
            spec = udfs.spec_of(op)
            produces_float = probe.dtype != np.uint8

            if is_float:
                expr = udfs.apply_array_op(expr, spec)
            elif produces_float:
                expr = udfs.apply_op_float(expr, spec)
            else:
                expr = udfs.apply_op(expr, spec)
            is_float = produces_float

        return expr

    # ------------------------------------------------------------------
    # Base adapter contract
    # ------------------------------------------------------------------

    def load_from_file(self, path: Path) -> bytes:
        """
        Load image bytes from a file.

        Args:
            path: Path to the image file.

        Returns:
            Image bytes.
        """
        return path.read_bytes()

    def load_from_bytes(self, data: bytes) -> bytes:
        """
        Pass through image bytes.

        Args:
            data: Image bytes.

        Returns:
            The same bytes.
        """
        return data

    def to_numpy(self, img: Any) -> "npt.NDArray[Any]":
        """
        Convert a Pixeltable result cell to a NumPy array.

        Args:
            img: A PIL image from a collected result set.

        Returns:
            NumPy array representation.
        """
        return np.asarray(img)

    # ------------------------------------------------------------------
    # Columnar batch paths
    # ------------------------------------------------------------------

    def _write_files(self, png_bytes_list: list[bytes], slot: int) -> list[str]:
        """
        Materialize encoded images as files for Pixeltable to reference.

        Pixeltable ingests media by path, so bytes held in memory by the other
        adapters have to hit the filesystem first. The directory is kept alive
        in the slot for as long as the table that references it.

        Args:
            png_bytes_list: Encoded image bytes.
            slot: Rotation slot owning the directory.

        Returns:
            Paths to the written files.
        """
        self._tmpdirs[slot] = tempfile.TemporaryDirectory()
        root = Path(self._tmpdirs[slot].name)
        paths = []
        for index, data in enumerate(png_bytes_list):
            path = root / f"{index}.png"
            path.write_bytes(data)
            paths.append(str(path))
        return paths

    def prepare_decoded_images(self, png_bytes_list: list[bytes]) -> Any:
        """
        Build a Pixeltable table holding the images.

        Note that unlike every other adapter, this cannot pre-*decode*: the
        rows reference files in the media store and each query decodes them
        again. See the module docstring.

        Args:
            png_bytes_list: List of PNG image bytes.

        Returns:
            A Pixeltable table handle.
        """
        pxt = self._get_pxt()
        self._probe = self._cv.load_from_bytes(png_bytes_list[0])
        slot = self._slot
        self._slot = (self._slot + 1) % self._SLOTS
        paths = self._write_files(png_bytes_list, slot)
        table = pxt.create_table(
            f"bench.{self.name.replace('-', '_')}_{slot}",
            {"img": pxt.Image},
            if_exists="replace",
        )
        table.insert({"img": path} for path in paths)
        return table

    def run_pipeline_on_decoded(
        self,
        decoded_images: Any,
        operations: list[OperationParams],
    ) -> Any:
        """
        Run operations over the table and materialize the result.

        Args:
            decoded_images: Table from :meth:`prepare_decoded_images`.
            operations: Operations to apply.

        Returns:
            The collected result set.
        """
        expr = self._build_expr(decoded_images.img, operations)
        return decoded_images.select(out=expr).collect()

    def run_pipeline_batch(
        self,
        image_bytes_list: list[bytes],
        operations: list[OperationParams],
    ) -> list["npt.NDArray[Any]"]:
        """
        Run a full pipeline over a batch of encoded images.

        Args:
            image_bytes_list: List of encoded image bytes.
            operations: Operations to apply.

        Returns:
            List of processed images as NumPy arrays.
        """
        table = self.prepare_decoded_images(image_bytes_list)
        result = self.run_pipeline_on_decoded(table, operations)
        return [np.asarray(cell) for cell in result["out"]]

    # ------------------------------------------------------------------
    # Single-image methods (used by the validation harness)
    # ------------------------------------------------------------------

    def _single(self, img: bytes, op: OperationParams) -> "npt.NDArray[Any]":
        """
        Apply one operation to one image through Pixeltable.

        Args:
            img: Encoded image bytes.
            op: Operation to apply.

        Returns:
            Processed image as a NumPy array.
        """
        return self.run_pipeline_batch([img], [op])[0]

    def resize(self, img: bytes, height: int, width: int) -> "npt.NDArray[Any]":
        """Resize an image."""
        return self._single(
            img,
            OperationParams(operation=OperationType.RESIZE, height=height, width=width),
        )

    def grayscale(self, img: bytes) -> "npt.NDArray[Any]":
        """Convert an image to grayscale."""
        return self._single(img, OperationParams(operation=OperationType.GRAYSCALE))

    def normalize(self, img: bytes) -> "npt.NDArray[Any]":
        """Min-max normalize an image."""
        return self._single(img, OperationParams(operation=OperationType.NORMALIZE))

    def flip_horizontal(self, img: bytes) -> "npt.NDArray[Any]":
        """Flip an image horizontally."""
        return self._single(img, OperationParams(operation=OperationType.FLIP_H))

    def flip_vertical(self, img: bytes) -> "npt.NDArray[Any]":
        """Flip an image vertically."""
        return self._single(img, OperationParams(operation=OperationType.FLIP_V))

    def crop(
        self, img: bytes, top: int, left: int, height: int, width: int
    ) -> "npt.NDArray[Any]":
        """Crop a region from an image."""
        return self._single(
            img,
            OperationParams(
                operation=OperationType.CROP,
                crop_top=top,
                crop_left=left,
                crop_height=height,
                crop_width=width,
            ),
        )

    def blur(self, img: bytes, sigma: float) -> "npt.NDArray[Any]":
        """Apply Gaussian blur."""
        return self._single(
            img, OperationParams(operation=OperationType.BLUR, sigma=sigma)
        )

    def threshold(self, img: bytes, value: int) -> "npt.NDArray[Any]":
        """Apply a binary threshold."""
        return self._single(
            img,
            OperationParams(operation=OperationType.THRESHOLD, threshold_value=value),
        )


class PixeltableNativeAdapter(PixeltableAdapter):
    """Pixeltable using only its own (PIL-backed) image expressions."""

    def __init__(self) -> None:
        """Initialize the native-only adapter."""
        super().__init__(allow_udf=False)


class PixeltableUDFAdapter(PixeltableAdapter):
    """Pixeltable with `@pxt.udf` filling in the missing operations."""

    def __init__(self) -> None:
        """Initialize the UDF-backed adapter."""
        super().__init__(allow_udf=True)
