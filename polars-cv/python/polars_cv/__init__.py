"""
polars-cv: High-performance vision and array processing for Polars.

This package provides modular image and array operations on Polars
DataFrame columns using modular pipelines.

Example:
    >>> from polars_cv import Pipeline
    >>> import polars as pl
    >>>
    >>> pipe = Pipeline().source("image_bytes").resize(224, 224)
    >>> df.with_columns(processed=pl.col("image").cv.pipe(pipe).sink("numpy"))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import numpy as np

from ._lib import configure_tiling as _configure_tiling
from ._lib import get_tiling_config as _get_tiling_config
from ._types import IMAGENET_MEAN, IMAGENET_STD, CloudOptions, ColorSpace, HashAlgorithm
from .display import show_images
from .expressions import CvNamespace
from .geometry import (
    BBOX_SCHEMA,
    CONTOUR_SCHEMA,
    CONTOUR_SET_SCHEMA,
    POINT_SCHEMA,
    POINT_SET_SCHEMA,
    RING_SCHEMA,
)
from .geometry.bbox import BBoxNamespace
from .geometry.contours import ContourNamespace
from .geometry.points import PointNamespace
from .lazy import LazyPipelineExpr
from .metrics import (
    BBoxMatcher,
    BootstrapResult,
    ContourMatcher,
    DetectionTable,
    FROCResult,
    LROCResult,
    MetricResult,
    PrecisionRecallResult,
    PreMatchedAdapter,
    average_precision,
    confusion_at_threshold,
    f1_at_threshold,
    froc_curve,
    lroc_curve,
    mean_average_precision,
    precision_at_threshold,
    precision_recall_curve,
    recall_at_threshold,
)
from .pipeline import Pipeline

# Schema for numpy/torch sink output struct
# Matches the Rust output module schema
NUMPY_OUTPUT_SCHEMA = pl.Struct(
    {
        "data": pl.Binary,
        "dtype": pl.String,
        "shape": pl.List(pl.UInt64),
        "strides": pl.List(pl.Int64),
        "offset": pl.UInt64,
    }
)


def numpy_from_struct(
    row: dict[str, object] | pl.Series,
    *,
    copy: bool = True,
) -> "np.ndarray":
    """
    Convert numpy sink output struct to a NumPy array.

    Args:
        row: Struct value from output column.
        copy: Whether to copy data (default True). If False, returns a view.
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
            strides_list = (
                struct_data["strides"][0] if "strides" in struct_data.columns else None
            )
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

    # Create numpy dtype — validate against allowlist to prevent arbitrary dtype strings
    _ALLOWED_DTYPES = frozenset(
        {
            "uint8",
            "u1",
            "int8",
            "i1",
            "uint16",
            "u2",
            "int16",
            "i2",
            "uint32",
            "u4",
            "int32",
            "i4",
            "uint64",
            "u8",
            "int64",
            "i8",
            "float16",
            "f2",
            "float32",
            "f4",
            "float64",
            "f8",
            "bool",
            "b1",
        }
    )
    if dtype_str not in _ALLOWED_DTYPES:
        msg = f"Unsupported dtype '{dtype_str}'. Allowed: {sorted(_ALLOWED_DTYPES)}"
        raise ValueError(msg)
    dtype = np.dtype(dtype_str)

    if copy:
        # Always copy: use frombuffer then reshape
        arr = np.frombuffer(bytes(data), dtype=dtype, offset=offset).copy()
        return arr.reshape(shape)
    else:
        # Zero-copy path: avoid bytes(data) which always copies.
        # Use the buffer protocol directly when available.
        buf = _as_buffer(data)

        if strides is not None:
            # Create strided numpy array view directly
            arr = np.ndarray(
                shape=shape,
                dtype=dtype,
                buffer=buf,
                offset=offset,
                strides=strides,
            )
            return arr
        else:
            # Contiguous path
            arr = np.frombuffer(buf, dtype=dtype, offset=offset)
            return arr.reshape(shape)


def _as_buffer(data: object) -> object:
    """Get a buffer-protocol object from data, avoiding unnecessary copies.

    Tries to use memoryview for zero-copy access. Falls back to bytes()
    if the object doesn't support the buffer protocol.

    Args:
        data: The data object (typically bytes from a Polars Binary column).

    Returns:
        A buffer-protocol compatible object.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return data
    # Try memoryview for objects that support the buffer protocol
    try:
        return memoryview(data)  # type: ignore[arg-type]
    except TypeError:
        # Fallback: copy into bytes
        return bytes(data)  # type: ignore[arg-type]


def mask_iou(
    pred: LazyPipelineExpr,
    target: LazyPipelineExpr,
    *,
    epsilon: float = 1e-7,
) -> pl.Expr:
    """
    Compute Intersection over Union (IoU) between two binary masks.

    Args:
        pred: Mask expression (binary 0/255).
        target: Target mask expression.
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


def configure_tiling(min_image_size: int | None = 512) -> None:
    """
    Configure tiled execution for large image processing.

    Tiled execution improves cache efficiency when processing large images
    by dividing them into smaller tiles (default 256x256) that fit in CPU cache.
    This can significantly improve performance for very large images.

    By default, tiling is **enabled** for images larger than 512 pixels
    in any dimension.

    Args:
        min_image_size: Minimum dimension (height or width) for tiling to activate.
            - ``None``: Disable tiling entirely (process all images as single buffers)
            - ``0``: Enable tiling for all images regardless of size
            - Positive integer: Only tile images larger than this threshold (default: 512)

    Example:
        ```python
        >>> import polars_cv
        >>>
        >>> # Check current configuration
        >>> print(polars_cv.get_tiling_config())
        {'enabled': True, 'tile_size': 256, 'min_image_size': 512}
        >>>
        >>> # Disable tiling (useful for debugging or small images only)
        >>> polars_cv.configure_tiling(None)
        >>>
        >>> # Only tile very large images (>2048 pixels)
        >>> polars_cv.configure_tiling(2048)
        >>>
        >>> # Reset to default (tile images > 512 pixels)
        >>> polars_cv.configure_tiling(512)
        ```

    Note:
        Tiling is **transparent** - results are identical whether tiling is
        on or off. The only difference is memory access patterns and cache
        efficiency. For very large images (e.g., 10000x10000), tiling can
        provide 2-5x speedups by keeping working data in CPU cache.
    """
    _configure_tiling(min_image_size)


def get_tiling_config() -> dict[str, int | bool] | None:
    """
    Get the current tiling configuration.

    Returns:
        A dict with tiling settings if enabled, or ``None`` if disabled.
        When enabled, the dict contains:
        - ``enabled``: Always ``True``
        - ``tile_size``: Size of each tile in pixels (default: 256)
        - ``min_image_size``: Minimum image dimension to trigger tiling

    Example:
        ```python
        >>> import polars_cv
        >>>
        >>> config = polars_cv.get_tiling_config()
        >>> if config:
        ...     print(f"Tiling: {config['tile_size']}x{config['tile_size']} tiles")
        ...     print(f"Activates for images > {config['min_image_size']}px")
        ... else:
        ...     print("Tiling disabled")
        Tiling: 256x256 tiles
        Activates for images > 512px
        ```
    """
    return _get_tiling_config()


def hamming_distance(
    hash1: LazyPipelineExpr,
    hash2: LazyPipelineExpr,
) -> pl.Expr:
    """
    Compute Hamming distance between two perceptual hashes.

    Args:
        hash1: First hash expression.
        hash2: Second hash expression.
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
    Compute similarity percentage [0, 100] between two hashes.

    Args:
        hash1: First hash expression.
        hash2: Second hash expression.
        hash_bits: Total bits in hash (default 64).
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
    Compute Dice coefficient between two binary masks.

    Args:
        pred: Mask expression (binary 0/255).
        target: Target mask expression.
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
    # Tiling configuration
    "configure_tiling",
    "get_tiling_config",
    # Types
    "CloudOptions",
    "ColorSpace",
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
    # Display utilities
    "show_images",
    # Detection metrics — core types
    "DetectionTable",
    "MetricResult",
    "BootstrapResult",
    # Detection metrics — matchers
    "ContourMatcher",
    "BBoxMatcher",
    "PreMatchedAdapter",
    # Detection metrics — functions
    "froc_curve",
    "lroc_curve",
    "precision_recall_curve",
    "average_precision",
    "mean_average_precision",
    "precision_at_threshold",
    "recall_at_threshold",
    "f1_at_threshold",
    "confusion_at_threshold",
    # Detection metrics — result types
    "FROCResult",
    "LROCResult",
    "PrecisionRecallResult",
    # Geometry namespaces (registered automatically via decorators)
    "BBoxNamespace",
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
__version__ = "0.8.0"
