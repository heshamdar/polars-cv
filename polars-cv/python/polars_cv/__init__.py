"""
polars-cv: A Polars plugin for vision/array operations.

This package provides lazy, zero-copy-where-possible image and array
operations on Polars DataFrame columns.

Two patterns are supported:

1. **Direct pipeline (eager)**: Define a complete pipeline with sink and apply directly.

    ```python
    >>> import polars as pl
    >>> from polars_cv import Pipeline
    >>>
    >>> pipe = (
    ...     Pipeline()
    ...     .source("image_bytes")
    ...     .resize(height=224, width=224)
    ...     .normalize(method="minmax")
    ...     .sink("numpy")
    ... )
    >>> result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
    ```
2. **Composable pipeline (lazy)**: Define pipelines without sinks, compose them,
   then call `.sink()` to finalize. Enables fused execution of multiple pipelines.

    ```python
    >>> from polars_cv import Pipeline
    >>>
    >>> img_pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
    >>> mask_pipe = Pipeline().source("contour")
    >>>
    >>> img = pl.col("image").cv.pipe(img_pipe)    # LazyPipelineExpr
    >>> mask = pl.col("contour").cv.pipe(mask_pipe)  # LazyPipelineExpr
    >>>
    >>> result = img.apply_contour_mask(mask).sink("numpy")  # Now a pl.Expr
    >>> df.with_columns(masked=result)
    ```
3. **NumPy Conversion**: Use `numpy_from_bytes()` to convert pipeline output to arrays.

    ```python
    >>> from polars_cv import Pipeline, numpy_from_bytes
    >>>
    >>> # Get the first row's output and convert to numpy
    >>> output_bytes = df.select("processed").row(0)[0]
    >>> array = numpy_from_bytes(output_bytes)
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import numpy as np

from polars_cv._types import (
    CloudOptions,
    HashAlgorithm,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from polars_cv.expressions import CvNamespace
from polars_cv.geometry import (
    BBOX_SCHEMA,
    CONTOUR_SCHEMA,
    CONTOUR_SET_SCHEMA,
    POINT_SCHEMA,
    POINT_SET_SCHEMA,
    RING_SCHEMA,
)
from polars_cv.geometry.contours import ContourNamespace
from polars_cv.geometry.points import PointNamespace
from polars_cv.lazy import LazyPipelineExpr
from polars_cv.pipeline import Pipeline

# Dtype code to numpy dtype mapping (matches Rust dtype_to_numpy_code)
_DTYPE_MAP = {
    0: "uint8",  # U8
    1: "int8",  # I8
    2: "uint16",  # U16
    3: "int16",  # I16
    4: "uint32",  # U32
    5: "int32",  # I32
    6: "uint64",  # U64
    7: "int64",  # I64
    8: "float32",  # F32
    9: "float64",  # F64
}


def numpy_from_bytes(data: bytes) -> "np.ndarray":
    """
    Convert polars-cv numpy/torch sink output to a numpy array.

    This function parses the header (dtype, ndim, shape) from the serialized
    bytes and returns a properly shaped numpy array.

    The format is:
    - 1 byte: dtype code (0=u8, 1=i8, 2=u16, 3=i16, 4=u32, 5=i32, 6=u64, 7=i64, 8=f32, 9=f64)
    - 1 byte: number of dimensions
    - 8 bytes per dimension: shape (uint64 little-endian)
    - Remaining bytes: array data

    Args:
        data: The bytes output from a pipeline with "numpy" or "torch" sink.

    Returns:
        A numpy array with the correct dtype and shape.

    Raises:
        ImportError: If numpy is not installed.
        ValueError: If the data cannot be parsed.

    Example:
        ```python
        >>> from polars_cv import Pipeline, numpy_from_bytes
        >>>
        >>> pipe = Pipeline().source("image_bytes").resize(height=100, width=100).sink("numpy")
        >>> result = df.select(processed=pl.col("images").cv.pipeline(pipe))
        >>>
        >>> # Convert first row to numpy array
        >>> array = numpy_from_bytes(result.row(0)[0])
        >>> print(array.shape)  # (100, 100, 3)
        ```
    """
    import numpy as np

    if len(data) < 2:
        msg = f"Data too short: expected at least 2 bytes for header, got {len(data)}"
        raise ValueError(msg)

    # Parse header
    dtype_code = data[0]
    ndim = data[1]

    if dtype_code not in _DTYPE_MAP:
        msg = f"Unknown dtype code: {dtype_code}. Valid codes: 0-9"
        raise ValueError(msg)

    dtype = np.dtype(_DTYPE_MAP[dtype_code])
    header_size = 2 + ndim * 8

    if len(data) < header_size:
        msg = (
            f"Data too short: expected {header_size} bytes for header, got {len(data)}"
        )
        raise ValueError(msg)

    # Parse shape
    shape = []
    offset = 2
    for _ in range(ndim):
        dim = int.from_bytes(data[offset : offset + 8], "little")
        shape.append(dim)
        offset += 8

    # Create array from data
    array_data = data[offset:]
    expected_size = int(np.prod(shape)) * dtype.itemsize

    if len(array_data) != expected_size:
        msg = (
            f"Data size mismatch: expected {expected_size} bytes, got {len(array_data)}"
        )
        raise ValueError(msg)

    return np.frombuffer(array_data, dtype=dtype).reshape(shape)


def numpy_header_size(data: bytes) -> int:
    """
    Get the header size for numpy sink output.

    This is useful for determining where the actual array data starts.

    Args:
        data: The bytes output from a pipeline with "numpy" or "torch" sink.

    Returns:
        The number of bytes in the header (2 + ndim * 8).

    Example:
        ```python
        >>> header_len = numpy_header_size(output_bytes)
        >>> raw_data = output_bytes[header_len:]
        ```
    """
    if len(data) < 2:
        return 0
    ndim = data[1]
    return 2 + ndim * 8


def numpy_shape(data: bytes) -> tuple[int, ...]:
    """
    Extract the shape from numpy sink output without loading the array.

    This is useful for inspecting the shape without allocating the full array.

    Args:
        data: The bytes output from a pipeline with "numpy" or "torch" sink.

    Returns:
        A tuple of dimensions.

    Example:
        ```python
        >>> shape = numpy_shape(output_bytes)
        >>> print(shape)  # (224, 224, 3)
        ```
    """
    if len(data) < 2:
        return ()

    ndim = data[1]
    shape = []
    offset = 2

    for _ in range(ndim):
        if offset + 8 > len(data):
            break
        dim = int.from_bytes(data[offset : offset + 8], "little")
        shape.append(dim)
        offset += 8

    return tuple(shape)


def numpy_dtype(data: bytes) -> str:
    """
    Extract the dtype from numpy sink output without loading the array.

    Args:
        data: The bytes output from a pipeline with "numpy" or "torch" sink.

    Returns:
        The numpy dtype string (e.g., "float32", "uint8").

    Example:
        ```python
        >>> dtype = numpy_dtype(output_bytes)
        >>> print(dtype)  # "uint8"
        ```
    """
    if len(data) < 1:
        msg = "Data too short to contain dtype"
        raise ValueError(msg)

    dtype_code = data[0]
    if dtype_code not in _DTYPE_MAP:
        msg = f"Unknown dtype code: {dtype_code}"
        raise ValueError(msg)

    return _DTYPE_MAP[dtype_code]


def mask_iou(
    pred: LazyPipelineExpr,
    target: LazyPipelineExpr,
    *,
    epsilon: float = 1e-7,
) -> pl.Expr:
    """
    Compute Intersection over Union (IoU) between two binary mask pipelines.

    IoU = intersection_sum / union_sum

    This function composes the bitwise operations and sum reductions, then
    uses Polars' native scalar operations for the final division.

    Args:
        pred: LazyPipelineExpr producing the prediction mask (binary 0/255).
        target: LazyPipelineExpr producing the ground truth mask (binary 0/255).
        epsilon: Small value to avoid division by zero (default 1e-7).

    Returns:
        A Polars expression that computes IoU as a Float64 value in [0, 1].

    Example:
        ```python
        >>> from polars_cv import Pipeline, mask_iou
        >>>
        >>> # Create mask pipelines
        >>> mask_pipe = Pipeline().source("image_bytes").grayscale().threshold(128)
        >>> pred = pl.col("pred_mask").cv.pipe(mask_pipe)
        >>> target = pl.col("gt_mask").cv.pipe(mask_pipe)
        >>>
        >>> # Compute IoU
        >>> iou_expr = mask_iou(pred, target)
        >>> result = df.select(iou=iou_expr)
        ```
    Note:
        Both masks should be binary (0 or non-zero values). The result is
        based on summing all pixel values, so for 0/255 masks, both intersection
        and union sums scale equally, yielding the correct IoU ratio.
    """
    # Compute intersection and union, then reduce to scalars
    intersection = (
        pred.bitwise_and(target)
        .pipe(Pipeline().reduce_sum())
        .alias("_iou_intersection")
    )
    union = pred.bitwise_or(target).pipe(Pipeline().reduce_sum()).alias("_iou_union")

    # Sink both as native scalars (Float64)
    result = intersection.merge_pipe(union).sink(
        {
            "_iou_intersection": "native",
            "_iou_union": "native",
        }
    )

    # Compute IoU using Polars scalar operations
    intersection_sum = result.struct.field("_iou_intersection")
    union_sum = result.struct.field("_iou_union")

    return intersection_sum / (union_sum + epsilon)


def hamming_distance(
    hash1: LazyPipelineExpr,
    hash2: LazyPipelineExpr,
) -> pl.Expr:
    """
    Compute Hamming distance between two hash buffers.

    Hamming distance is the number of positions at which the corresponding
    bits differ. This is useful for comparing perceptual hashes to determine
    image similarity.

    The computation is: XOR the two hash buffers, then count set bits (popcount).

    Args:
        hash1: LazyPipelineExpr producing a hash buffer (e.g., from perceptual_hash()).
        hash2: LazyPipelineExpr producing a hash buffer (same size as hash1).

    Returns:
        A Polars expression that computes the Hamming distance as a Float64 value.

    Example:
        ```python
        >>> from polars_cv import Pipeline, hamming_distance
        >>>
        >>> # Compute perceptual hashes
        >>> hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        >>> hash1 = pl.col("image1").cv.pipe(hash_pipe)
        >>> hash2 = pl.col("image2").cv.pipe(hash_pipe)
        >>>
        >>> # Compute Hamming distance (number of differing bits)
        >>> distance_expr = hamming_distance(hash1, hash2)
        >>> result = df.select(distance=distance_expr)
        ```
    Note:
        For 64-bit perceptual hashes, the maximum Hamming distance is 64 (completely
        different) and minimum is 0 (identical). Lower distance = more similar.
    """
    # XOR the hashes and count set bits
    xor_result = hash1.bitwise_xor(hash2).pipe(Pipeline().reduce_popcount())

    # Sink as native scalar (Float64)
    return xor_result.sink("native")


def hash_similarity(
    hash1: LazyPipelineExpr,
    hash2: LazyPipelineExpr,
    *,
    hash_bits: int = 64,
) -> pl.Expr:
    """
    Compute similarity percentage between two hash buffers.

    Similarity is computed as: (1 - hamming_distance / total_bits) * 100

    This provides an intuitive percentage where:
    - 100% = identical hashes
    - 0% = completely different hashes

    Args:
        hash1: LazyPipelineExpr producing a hash buffer (e.g., from perceptual_hash()).
        hash2: LazyPipelineExpr producing a hash buffer (same size as hash1).
        hash_bits: Total number of bits in the hash (default 64 for standard perceptual hash).

    Returns:
        A Polars expression that computes similarity as a Float64 percentage [0, 100].

    Example:
        ```python
        >>> from polars_cv import Pipeline, hash_similarity
        >>>
        >>> # Compute perceptual hashes
        >>> hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        >>> hash1 = pl.col("original_image").cv.pipe(hash_pipe)
        >>> hash2 = pl.col("modified_image").cv.pipe(hash_pipe)
        >>>
        >>> # Compute similarity percentage
        >>> similarity_expr = hash_similarity(hash1, hash2)
        >>> result = df.select(similarity=similarity_expr)
        >>>
        >>> # Filter for similar images (>90% similar)
        >>> similar = df.filter(hash_similarity(hash1, hash2) > 90)
        ```
    Note:
        The default hash_bits=64 is appropriate for standard perceptual hashes.
        For larger hashes (e.g., 256-bit), pass hash_bits=256.
    """
    # XOR the hashes and count set bits
    xor_popcount = hash1.bitwise_xor(hash2).pipe(Pipeline().reduce_popcount())

    # Sink as native scalar (Float64)
    distance = xor_popcount.sink("native")

    # Compute similarity: (1 - distance / total_bits) * 100
    return (1.0 - distance / hash_bits) * 100.0


def mask_dice(
    pred: LazyPipelineExpr,
    target: LazyPipelineExpr,
    *,
    epsilon: float = 1e-7,
) -> pl.Expr:
    """
    Compute Dice coefficient between two binary mask pipelines.

    Dice = 2 * intersection_sum / (pred_sum + target_sum)

    This function composes the bitwise operations and sum reductions, then
    uses Polars' native scalar operations for the final calculation.

    Args:
        pred: LazyPipelineExpr producing the prediction mask (binary 0/255).
        target: LazyPipelineExpr producing the ground truth mask (binary 0/255).
        epsilon: Small value to avoid division by zero (default 1e-7).

    Returns:
        A Polars expression that computes Dice coefficient as a Float64 value in [0, 1].

    Example:
        ```python
        >>> from polars_cv import Pipeline, mask_dice
        >>>
        >>> # Create mask pipelines
        >>> mask_pipe = Pipeline().source("image_bytes").grayscale().threshold(128)
        >>> pred = pl.col("pred_mask").cv.pipe(mask_pipe)
        >>> target = pl.col("gt_mask").cv.pipe(mask_pipe)
        >>>
        >>> # Compute Dice
        >>> dice_expr = mask_dice(pred, target)
        >>> result = df.select(dice=dice_expr)
        ```
    Note:
        Both masks should be binary (0 or non-zero values). The result is
        based on summing all pixel values, so for 0/255 masks, all sums
        scale equally, yielding the correct Dice coefficient.
    """
    # Compute intersection, pred sum, and target sum as scalars
    intersection = (
        pred.bitwise_and(target)
        .pipe(Pipeline().reduce_sum())
        .alias("_dice_intersection")
    )
    pred_sum = pred.pipe(Pipeline().reduce_sum()).alias("_dice_pred")
    target_sum = target.pipe(Pipeline().reduce_sum()).alias("_dice_target")

    # Sink all three as native scalars (Float64)
    result = intersection.merge_pipe(pred_sum, target_sum).sink(
        {
            "_dice_intersection": "native",
            "_dice_pred": "native",
            "_dice_target": "native",
        }
    )

    # Compute Dice using Polars scalar operations
    inter = result.struct.field("_dice_intersection")
    total = result.struct.field("_dice_pred") + result.struct.field("_dice_target")

    return (2.0 * inter) / (total + epsilon)


__all__ = [
    "Pipeline",
    "CvNamespace",
    "LazyPipelineExpr",
    # Types
    "CloudOptions",
    "HashAlgorithm",
    # ImageNet normalization constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # NumPy conversion utilities
    "numpy_from_bytes",
    "numpy_header_size",
    "numpy_shape",
    "numpy_dtype",
    # Mask comparison functions
    "mask_iou",
    "mask_dice",
    # Hash comparison functions
    "hamming_distance",
    "hash_similarity",
    # Geometry namespaces (registered automatically via decorators)
    "ContourNamespace",
    "PointNamespace",
    # Schemas
    "POINT_SCHEMA",
    "POINT_SET_SCHEMA",
    "RING_SCHEMA",
    "CONTOUR_SCHEMA",
    "CONTOUR_SET_SCHEMA",
    "BBOX_SCHEMA",
]
__version__ = "0.1.0"
