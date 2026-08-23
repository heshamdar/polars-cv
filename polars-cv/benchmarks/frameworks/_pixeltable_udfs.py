"""
Pixeltable UDFs for the benchmark adapter.

This module exists because of a Pixeltable requirement rather than a design
preference: **a UDF must be defined in a named module.** Pixeltable rejects one
defined in a script's global namespace, and it rejects one constructed inside a
method, so the gap-filling functions cannot live next to the adapter logic that
uses them. It also registers UDFs by qualified name and raises
`AlreadyExistsError` on a second registration, so each is decorated exactly
once here, at import.

Three of them are needed, and the reason is Pixeltable's type system rather
than the operations themselves. A Pixeltable `Image` column is PIL-backed and
therefore 8-bit: it cannot hold the float output of `normalize` or `sobel`.
Those results have to become `Array[Float]` columns instead — and once a column
is an array, no image expression applies to it, so everything downstream of a
float-producing operation must also be a UDF.

    image -> image     `apply_op`         uint8 in, uint8 out
    image -> array     `apply_op_float`   uint8 in, float32 out
    array -> array     `apply_array_op`   float32 in, float32 out

Every body calls `OpenCVAdapter` rather than reimplementing a kernel, so
`pixeltable-udf` and the `opencv` adapter run byte-identical code and the
difference between their timings is engine overhead alone.

The operation to apply travels in as a JSON string, because a Pixeltable UDF
cannot close over a Python object.
"""

from __future__ import annotations

import json

import numpy as np
import PIL.Image
import pixeltable as pxt

from .base import OperationParams, OperationType
from .opencv_adapter import OpenCVAdapter

#: One shared adapter; it holds only a lazily-imported cv2 handle.
_CV = OpenCVAdapter()


def spec_of(op: OperationParams) -> str:
    """
    Serialize an operation so it can be passed into a UDF.

    Args:
        op: Operation to serialize.

    Returns:
        JSON accepted by :func:`_params_of`.
    """
    import dataclasses

    fields = dataclasses.asdict(op)
    fields["operation"] = op.operation.name
    return json.dumps(fields)


def _params_of(spec: str) -> OperationParams:
    """
    Rebuild an `OperationParams` from its JSON form.

    Args:
        spec: JSON produced by :func:`spec_of`.

    Returns:
        The reconstructed operation parameters.
    """
    fields = json.loads(spec)
    fields["operation"] = OperationType[fields["operation"]]
    return OperationParams(**fields)


def _for_opencv(arr: np.ndarray) -> np.ndarray:
    """
    Drop a redundant single-channel axis before handing an array to OpenCV.

    OpenCV decides an image is already grayscale by testing ``ndim == 2``, so a
    ``(H, W, 1)`` array makes a later `cvtColor(RGB2GRAY)` raise.

    Args:
        arr: Array from a Pixeltable cell.

    Returns:
        The array in the form OpenCV's grayscale checks expect.
    """
    return arr[:, :, 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr


@pxt.udf
def apply_op(img: PIL.Image.Image, spec: str) -> PIL.Image.Image:
    """
    Apply one uint8-producing operation to an image cell.

    Args:
        img: The input image.
        spec: JSON operation spec.

    Returns:
        The processed image.
    """
    out = _CV.apply_operation(_for_opencv(np.asarray(img)), _params_of(spec))
    return PIL.Image.fromarray(out)


@pxt.udf
def apply_op_float(img: PIL.Image.Image, spec: str) -> pxt.Array[pxt.Float]:
    """
    Apply one float-producing operation to an image cell.

    Args:
        img: The input image.
        spec: JSON operation spec.

    Returns:
        The processed data as a float32 array, since an Image column is 8-bit.
    """
    out = _CV.apply_operation(_for_opencv(np.asarray(img)), _params_of(spec))
    return np.ascontiguousarray(out, dtype=np.float32)


@pxt.udf
def apply_array_op(arr: pxt.Array[pxt.Float], spec: str) -> pxt.Array[pxt.Float]:
    """
    Apply one operation to a float array cell.

    Everything downstream of a float-producing operation lands here: once a
    column is an array, no Pixeltable image expression applies to it.

    Args:
        arr: The input array.
        spec: JSON operation spec.

    Returns:
        The processed float32 array.
    """
    out = _CV.apply_operation(_for_opencv(np.asarray(arr)), _params_of(spec))
    return np.ascontiguousarray(out, dtype=np.float32)
