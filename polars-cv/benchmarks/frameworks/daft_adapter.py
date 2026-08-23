"""
Daft framework adapter for benchmarking.

Daft (https://daft.ai) is the closest thing to a direct competitor for
polars-cv's premise: a columnar dataframe engine with a Rust core that treats
images as a first-class column type rather than as opaque blobs.

The two projects overlap in *shape* but not in *depth*, and this adapter is
built so the benchmark numbers make that difference visible instead of hiding
it behind a `NotImplementedError`:

``daft`` (:class:`DaftNativeAdapter`)
    Only Daft's own image expressions. As of 0.7 the native vision surface is
    ``resize`` / ``crop`` / ``convert_image`` / ``decode_image`` /
    ``encode_image`` / ``image_to_tensor`` / ``image_attribute`` /
    ``image_hash`` — which covers three of the harness's twenty single-op
    benchmarks (resize, grayscale, crop). Every other op raises
    :class:`NotImplementedError` from :meth:`_build_expr`, so the results table
    shows a gap rather than a number Daft did not really earn. This is the
    engine-vs-engine measurement against polars-cv.

``daft-udf`` (:class:`DaftUDFAdapter`)
    What a Daft user actually writes for the other seventeen: native
    expressions where they exist, and a ``@daft.func.batch`` UDF where they do
    not. The UDF bodies call :class:`OpenCVAdapter` rather than reimplementing
    blur/canny/sobel/... a second time, so ``daft-udf`` vs ``opencv`` isolates
    Daft's per-batch UDF overhead, and ``daft-udf`` vs ``polars-cv-*`` compares
    the two dataframe routes over identical kernels.

Two Daft 0.7.24 limitations shape the code below and are load-bearing, not
incidental:

- **No arithmetic on Image or Tensor columns.** ``col("img") * 2`` raises
  ``Cannot multiply``, so normalize / invert / brightness / contrast are not
  expressible as native expressions at all — they are UDF-only.
- **Float image modes are declared but not readable.** ``DataType.image(...)``
  advertises ``RGB32F`` / ``RGBA32F``, but a UDF returning float32 arrays under
  that dtype panics the Rust worker on readback (``Attempting to downcast
  Float32 to DataArray<UInt8Type>``). :func:`_daft_dtype_for` therefore routes
  every float result to ``DataType.tensor(...)``, which round-trips correctly.

Because Daft cannot express a shape/dtype rule for a UDF the way polars-cv's
``Op`` contracts do, this adapter has to work out each UDF's return dtype
itself. :meth:`_build_expr` threads a small NumPy *probe array* through the
chain — applying each op to the probe to learn the shape and dtype it produces,
exactly the plan-time probing polars-cv does in Rust — and picks the Daft dtype
from the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseFrameworkAdapter, OperationParams, OperationType
from .opencv_adapter import OpenCVAdapter

if TYPE_CHECKING:
    import numpy.typing as npt

#: Operations Daft can express with its own image expressions. Everything else
#: in `OperationType` needs a UDF. Keep this list honest — widening it without
#: a real native expression behind it is the one way this benchmark can lie.
NATIVE_OPS: frozenset[OperationType] = frozenset(
    {
        OperationType.RESIZE,
        OperationType.GRAYSCALE,
        OperationType.CROP,
    }
)


def _daft_dtype_for(arr: "npt.NDArray[Any]", daft: Any) -> Any:
    """
    Pick the Daft return dtype that can hold ``arr``.

    uint8 results stay in the image type system (so a native expression can
    still be chained after a UDF); everything else becomes a tensor, because
    Daft's float image modes panic on readback (see the module docstring).

    Args:
        arr: A representative output array from the operation.
        daft: The imported ``daft`` module.

    Returns:
        A Daft ``DataType``.
    """
    if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        # Mode-less image(): Daft infers L/RGB/RGBA from the channel count.
        return daft.DataType.image()
    return daft.DataType.tensor(daft.DataType.from_numpy_dtype(arr.dtype))


def _as_image_array(arr: "npt.NDArray[Any]") -> "npt.NDArray[Any]":
    """
    Give a 2-D result the trailing channel axis Daft's image type requires.

    Daft rejects an ``(H, W)`` array for an image column ("Tensor array shapes
    are not compatible"); it wants ``(H, W, 1)``. OpenCV's grayscale-producing
    kernels return 2-D, so this is on the path of every gray op.

    Args:
        arr: Operation output.

    Returns:
        The same data with at least three dimensions.
    """
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    return arr


def _as_opencv_array(arr: "npt.NDArray[Any]") -> "npt.NDArray[Any]":
    """
    Drop a single trailing channel axis before handing an array to OpenCV.

    The inverse of :func:`_as_image_array`, and required for exactly the same
    reason it exists. Daft stores single-channel images as ``(H, W, 1)`` — both
    its own ``convert_image("L")`` and any UDF output do — but OpenCV decides
    whether an image is already grayscale by testing ``ndim == 2``. Hand it the
    ``(H, W, 1)`` form and a later ``cvtColor(RGB2GRAY)`` raises "Bad number of
    channels", which is how every gray-then-something chain (the heavy and
    medical pipelines) used to fail.

    Args:
        arr: Array as it comes out of a Daft image column.

    Returns:
        The array with a redundant trailing axis removed.
    """
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    return arr


class DaftAdapter(BaseFrameworkAdapter):
    """
    Adapter for Daft image processing.

    Attributes:
        name: Human-readable name of the adapter.
        allow_udf: Whether operations with no native Daft expression may fall
            back to a ``@daft.func.batch`` UDF over OpenCV kernels.
    """

    supports_gpu: bool = False
    columnar: bool = True

    def __init__(self, allow_udf: bool = False) -> None:
        """
        Initialize the Daft adapter.

        Args:
            allow_udf: If True, fill gaps in Daft's native op surface with
                batch UDFs. If False, unsupported ops raise NotImplementedError.
        """
        self.allow_udf = allow_udf
        self.name = "daft-udf" if allow_udf else "daft"
        self._daft: Any = None
        self._cv = OpenCVAdapter()
        self._probe: "npt.NDArray[np.uint8] | None" = None

    def is_available(self) -> bool:
        """
        Check if Daft is available.

        Returns:
            True if daft can be imported, False otherwise.
        """
        try:
            import daft  # noqa: F401
        except ImportError:
            return False
        # The UDF variant leans on OpenCV for the ops Daft has no expression for.
        return self._cv.is_available() if self.allow_udf else True

    def _get_daft(self) -> Any:
        """Get the daft module."""
        if self._daft is None:
            import daft

            self._daft = daft
        return self._daft

    # ------------------------------------------------------------------
    # Expression construction
    # ------------------------------------------------------------------

    def _native_expr(self, expr: Any, op: OperationParams) -> Any:
        """
        Apply one natively-supported operation to a Daft expression.

        Args:
            expr: Daft expression holding an image column.
            op: Operation to apply.

        Returns:
            The extended expression.

        Raises:
            ValueError: If ``op`` is not in :data:`NATIVE_OPS`.
        """
        if op.operation == OperationType.RESIZE:
            # Daft takes (width, height); no interpolation parameter is exposed,
            # and the fixed filter is bilinear (verified against PIL BILINEAR),
            # which is what the other adapters are pinned to.
            return expr.resize(op.width, op.height)
        if op.operation == OperationType.GRAYSCALE:
            return expr.convert_image("L")
        if op.operation == OperationType.CROP:
            # Daft's bbox is (x, y, width, height).
            return expr.crop((op.crop_left, op.crop_top, op.crop_width, op.crop_height))
        msg = f"{op.operation} is not a native Daft expression"
        raise ValueError(msg)

    def _udf_expr(self, expr: Any, op: OperationParams, probe: Any) -> Any:
        """
        Apply one operation as a batch UDF over OpenCV kernels.

        Args:
            expr: Daft expression holding an image or tensor column.
            op: Operation to apply.
            probe: Representative output array, used to pick the return dtype.

        Returns:
            The extended expression.
        """
        daft = self._get_daft()
        cv = self._cv
        return_dtype = _daft_dtype_for(probe, daft)

        @daft.func.batch(return_dtype=return_dtype)
        def _apply(series: Any) -> list["npt.NDArray[Any]"]:
            return [
                _as_image_array(
                    cv.apply_operation(_as_opencv_array(np.asarray(row)), op)
                )
                for row in series.to_pylist()
            ]

        return _apply(expr)

    def _build_expr(self, column: str, operations: list[OperationParams]) -> Any:
        """
        Build the Daft expression chain for a list of operations.

        A NumPy probe array is threaded through the chain so each UDF's return
        dtype is derived from what the operation actually produces rather than
        from a hand-maintained table. The probe also decides whether the column
        is still an image (so a native expression can be used) or has become a
        float tensor (so everything downstream must be a UDF).

        Args:
            column: Name of the input image column.
            operations: Operations to apply, in order.

        Returns:
            A Daft expression producing the processed column.

        Raises:
            NotImplementedError: If ``allow_udf`` is False and an operation has
                no native Daft expression.
        """
        daft = self._get_daft()
        expr = daft.col(column)
        probe = self._probe_array()

        for op in operations:
            # A native expression needs an image column: once the probe says the
            # data is float (i.e. it became a tensor), only UDFs can touch it.
            still_image = probe.dtype == np.uint8
            if op.operation in NATIVE_OPS and still_image:
                expr = self._native_expr(expr, op)
                probe = _as_image_array(
                    self._cv.apply_operation(_as_opencv_array(probe), op)
                )
                continue

            if not self.allow_udf:
                msg = (
                    f"Daft has no native expression for {op.operation.name.lower()}"
                    if op.operation not in NATIVE_OPS
                    else f"Daft cannot apply {op.operation.name.lower()} to a "
                    "non-uint8 column (no arithmetic or image ops on tensors)"
                )
                raise NotImplementedError(msg)

            probe = _as_image_array(
                self._cv.apply_operation(_as_opencv_array(probe), op)
            )
            expr = self._udf_expr(expr, op, probe)

        return expr

    def _probe_array(self) -> "npt.NDArray[np.uint8]":
        """
        Get the representative input array for plan-time shape probing.

        Returns:
            A small uint8 array matching the shape of the decoded input images.

        Raises:
            RuntimeError: If no images have been prepared yet.
        """
        if getattr(self, "_probe", None) is None:
            msg = (
                "No probe image available: call prepare_decoded_images() or "
                "run_pipeline_batch() before building an expression"
            )
            raise RuntimeError(msg)
        return self._probe

    def _set_probe(self, image_bytes: bytes) -> None:
        """
        Record the decoded shape of the input images for probing.

        Args:
            image_bytes: One encoded image from the benchmark set.
        """
        self._probe = self._cv.load_from_bytes(image_bytes)

    # ------------------------------------------------------------------
    # Base adapter contract
    # ------------------------------------------------------------------

    def load_from_file(self, path: Path) -> bytes:
        """
        Load image bytes from a file.

        Args:
            path: Path to the image file.

        Returns:
            Image bytes; Daft decodes them inside the dataframe.
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
        Convert a Daft result cell to a NumPy array for validation.

        Args:
            img: A cell from a collected Daft image or tensor column.

        Returns:
            NumPy array representation.
        """
        return np.asarray(img)

    # ------------------------------------------------------------------
    # Columnar batch paths (what the scenarios actually time)
    # ------------------------------------------------------------------

    def prepare_decoded_images(self, png_bytes_list: list[bytes]) -> Any:
        """
        Decode PNG bytes into a materialized Daft DataFrame.

        The decode and the DataFrame construction both happen here so that the
        timed section measures operations only, matching what the polars-cv and
        OpenCV adapters do.

        Args:
            png_bytes_list: List of PNG image bytes.

        Returns:
            A collected Daft DataFrame with an ``images`` Image column.
        """
        daft = self._get_daft()
        self._set_probe(png_bytes_list[0])
        return (
            daft.from_pydict({"encoded": png_bytes_list})
            .select(daft.col("encoded").decode_image().alias("images"))
            .collect()
        )

    def run_pipeline_on_decoded(
        self,
        decoded_images: Any,
        operations: list[OperationParams],
    ) -> Any:
        """
        Run operations on a pre-decoded Daft DataFrame.

        Args:
            decoded_images: DataFrame from :meth:`prepare_decoded_images`.
            operations: Operations to apply.

        Returns:
            A collected DataFrame with the ``processed`` column.
        """
        expr = self._build_expr("images", operations)
        return decoded_images.select(expr.alias("processed")).collect()

    def run_pipeline_batch(
        self,
        image_bytes_list: list[bytes],
        operations: list[OperationParams],
    ) -> list["npt.NDArray[Any]"]:
        """
        Run a full decode-and-process pipeline over a batch of images.

        Accepts either encoded bytes (the normal case, where Daft's
        ``decode_image`` does the decode inside the query) or already-decoded
        NumPy arrays, so that a caller chaining one operation at a time can feed
        this method its own output.

        Args:
            image_bytes_list: List of encoded image bytes, or decoded arrays.
            operations: Operations to apply.

        Returns:
            List of processed images as NumPy arrays.
        """
        daft = self._get_daft()
        first = image_bytes_list[0]

        if isinstance(first, (bytes, bytearray)):
            self._set_probe(bytes(first))
            frame = daft.from_pydict({"encoded": list(image_bytes_list)}).select(
                daft.col("encoded").decode_image().alias("images")
            )
        else:
            # Already decoded: hand Daft the arrays directly. Single-channel
            # results have to carry the trailing axis for the image dtype.
            arrays = [_as_image_array(np.asarray(row)) for row in image_bytes_list]
            self._probe = _as_opencv_array(arrays[0])
            frame = daft.from_pydict({"images": arrays})

        expr = self._build_expr("images", operations)
        result = frame.select(expr.alias("processed")).to_pydict()
        return [np.asarray(cell) for cell in result["processed"]]

    # ------------------------------------------------------------------
    # Single-image methods (used by the validation harness)
    # ------------------------------------------------------------------

    def _single(self, img: bytes, op: OperationParams) -> "npt.NDArray[Any]":
        """
        Apply one operation to one image through Daft.

        Args:
            img: Encoded image bytes.
            op: Operation to apply.

        Returns:
            Processed image as a NumPy array.
        """
        return self.run_pipeline_batch([img], [op])[0]

    def resize(self, img: bytes, height: int, width: int) -> "npt.NDArray[Any]":
        """Resize an image. See :meth:`BaseFrameworkAdapter.resize`."""
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


class DaftNativeAdapter(DaftAdapter):
    """Daft using only its own image expressions."""

    def __init__(self) -> None:
        """Initialize the native-only adapter."""
        super().__init__(allow_udf=False)


class DaftUDFAdapter(DaftAdapter):
    """Daft with `@daft.func.batch` UDFs filling in the missing operations."""

    def __init__(self) -> None:
        """Initialize the UDF-backed adapter."""
        super().__init__(allow_udf=True)
