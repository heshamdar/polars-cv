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
3. **NumPy Conversion**: Use `numpy_from_struct()` to convert pipeline output to arrays.

    The numpy/torch sink returns a Struct with `data`, `dtype`, and `shape` fields,
    enabling zero-copy data transfer from Rust.

    ```python
    >>> from polars_cv import Pipeline, numpy_from_struct
    >>>
    >>> # Get the first row's output struct and convert to numpy
    >>> output_struct = df.select("processed")["processed"][0]
    >>> array = numpy_from_struct(output_struct)
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

# Schema for numpy/torch sink output struct
# Matches the Rust output module schema
NUMPY_OUTPUT_SCHEMA = pl.Struct({
    "data": pl.Binary,
    "dtype": pl.String,
    "shape": pl.List(pl.UInt64),
    "strides": pl.List(pl.Int64),
    "offset": pl.UInt64,
})


def numpy_from_struct(
    row: dict[str, object] | pl.Series,
    *,
    copy: bool = True,
) -> "np.ndarray":
    """
    Convert polars-cv numpy/torch sink output struct to a numpy array.

    The numpy/torch sink format is a Struct with five fields:
    - data: Binary (raw array bytes, may be larger for strided views)
    - dtype: String (numpy dtype name like "uint8", "float32")
    - shape: List[UInt64] (array dimensions)
    - strides: List[Int64] (byte strides per dimension)
    - offset: UInt64 (byte offset into data buffer)

    This format enables zero-copy data transfer from Rust to Python,
    including for non-contiguous strided buffers (e.g., after crop/flip).

    Args:
        row: A struct value from the output column. Can be:
            - A dict with 'data', 'dtype', 'shape', 'strides', 'offset' keys
            - A Series representing a single struct row
        copy: Whether to copy the data (default True). If False, the returned
            array may share memory with the underlying buffer, which can be
            more efficient but requires care with lifetime management.
            When False with strided data, creates a strided numpy view.

    Returns:
        A numpy array with the correct dtype and shape.

    Raises:
        ImportError: If numpy is not installed.
        ValueError: If the input is not a valid numpy output struct.
        KeyError: If required fields are missing.

    Example:
        ```python
        >>> from polars_cv import Pipeline, numpy_from_struct
        >>>
        >>> pipe = Pipeline().source("image_bytes").resize(height=100, width=100).sink("numpy")
        >>> result = df.select(processed=pl.col("images").cv.pipeline(pipe))
        >>>
        >>> # Convert first row to numpy array
        >>> row = result["processed"][0]
        >>> array = numpy_from_struct(row)
        >>> print(array.shape)  # (100, 100, 3)
        >>>
        >>> # Zero-copy strided view (for crop/flip pipelines)
        >>> array_view = numpy_from_struct(row, copy=False)
        ```
    """
    import numpy as np

    # Extract fields from struct
    if isinstance(row, dict):
        data = row.get("data")
        dtype_str = row.get("dtype")
        shape_list = row.get("shape")
        strides_list = row.get("strides")
        offset = row.get("offset", 0)
    elif isinstance(row, pl.Series):
        # Single-row Series from struct indexing
        if row.dtype == pl.Struct:
            struct_data = row.struct.unnest()
            data = struct_data["data"][0]
            dtype_str = struct_data["dtype"][0]
            shape_list = struct_data["shape"][0]
            strides_list = struct_data["strides"][0] if "strides" in struct_data.columns else None
            offset = struct_data["offset"][0] if "offset" in struct_data.columns else 0
        else:
            msg = f"Expected Struct Series, got {row.dtype}"
            raise ValueError(msg)
    else:
        # Assume it's a struct value that can be accessed like a dict
        try:
            data = row["data"]
            dtype_str = row["dtype"]
            shape_list = row["shape"]
            strides_list = row.get("strides") if hasattr(row, "get") else None
            offset = row.get("offset", 0) if hasattr(row, "get") else 0
        except (TypeError, KeyError) as e:
            msg = f"Cannot extract struct fields from {type(row)}: {e}"
            raise ValueError(msg) from e

    # Validate required fields
    if data is None:
        msg = "Struct field 'data' is null"
        raise ValueError(msg)
    if dtype_str is None:
        msg = "Struct field 'dtype' is null"
        raise ValueError(msg)
    if shape_list is None:
        msg = "Struct field 'shape' is null"
        raise ValueError(msg)

    # Convert shape to tuple
    if isinstance(shape_list, pl.Series):
        shape = tuple(int(x) for x in shape_list.to_list())
    else:
        shape = tuple(int(x) for x in shape_list)

    # Convert strides to tuple (if present)
    strides: tuple[int, ...] | None = None
    if strides_list is not None:
        if isinstance(strides_list, pl.Series):
            strides = tuple(int(x) for x in strides_list.to_list())
        else:
            strides = tuple(int(x) for x in strides_list)

    # Convert offset
    if offset is None:
        offset = 0
    else:
        offset = int(offset)

    # Create numpy dtype
    dtype = np.dtype(dtype_str)

    if copy:
        # Always copy: use frombuffer then reshape
        arr = np.frombuffer(bytes(data), dtype=dtype, offset=offset).copy()
        return arr.reshape(shape)
    else:
        # Zero-copy path: create strided view if strides are available
        if strides is not None:
            # Create strided numpy array view directly
            # This is the true zero-copy path for non-contiguous data
            arr = np.ndarray(
                shape=shape,
                dtype=dtype,
                buffer=bytes(data),
                offset=offset,
                strides=strides,
            )
            return arr
        else:
            # Legacy path: no strides, assume contiguous
            arr = np.frombuffer(bytes(data), dtype=dtype, offset=offset)
            return arr.reshape(shape)


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
    "numpy_from_struct",
    "NUMPY_OUTPUT_SCHEMA",
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
