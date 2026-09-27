"""
polars-cv: High-performance vision and array processing for Polars.

This package provides modular image and array operations on Polars
DataFrame columns using modular pipelines.

Example:
    >>> from polars_cv import Pipeline
    >>> import polars as pl
    >>>
    >>> pipe = Pipeline().source("image_bytes").resize(height=224, width=224)
    >>> df.with_columns(processed=pl.col("image").cv.pipe(pipe).sink("numpy"))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

from ._dtype_names import SINK_NUMPY_NAMES
from ._optimize import OptFlags
from ._types import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CloudOptions,
    ColorSpace,
    HashAlgorithm,
    dtype_name_for,
)
from .display import show_images
from .expressions import CvNamespace
from .extension_types import (
    NUMPY_OUTPUT_SCHEMA,
    BBoxType,
    ContourType,
    NdArrayType,
    PointType,
    register_extension_types,
)
from .geometry import (
    BBOX_SCHEMA,
    CONTOUR_SCHEMA,
    CONTOUR_SET_SCHEMA,
    CORRESPONDENCE_SCHEMA,
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
    ConfusionResult,
    ContourMatcher,
    DetectionTable,
    MetricResult,
    PrecisionRecallResult,
    PreMatchedAdapter,
    average_precision,
    average_precision_ci_lazy,
    confusion_at_threshold,
    f1_at_threshold,
    froc_auc,
    froc_auc_ci_lazy,
    froc_curve_lazy,
    froc_sensitivity_at_fp,
    froc_summary_table,
    lroc_auc,
    lroc_auc_ci_lazy,
    lroc_curve_lazy,
    lroc_sensitivity_at_fpf,
    mean_average_precision,
    precision_at_threshold,
    precision_recall_curve,
    recall_at_threshold,
)
from .pipeline import Pipeline

# Registered at import, not on first use: a Parquet/IPC read of a tagged column
# consults the registry, and a column read before registration decays to its
# storage. Pure Python, so importing polars-cv still loads no compiled code.
register_extension_types()

__version__ = "0.29.0"


def _source_hash_from_tree() -> str | None:
    """Recompute the extension's source hash from the working tree.

    Mirrors ``polars-cv/build.rs`` exactly — same inputs, same order, same
    FNV-1a — so the value can be compared against ``_lib.__source_hash__`` to
    tell whether the compiled extension was built from the sources now on disk.

    Returns ``None`` when the Rust sources are not present (an installed wheel
    rather than a checkout), where staleness is not a question that arises.
    """
    root = Path(__file__).resolve().parents[3]
    if not (root / "polars-cv" / "Cargo.toml").is_file():
        return None  # pragma: no cover  # installed wheel, not a source checkout

    contents: dict[str, bytes] = {}

    def _push(path: Path, key: str) -> None:
        try:
            contents[key] = path.read_bytes()
        except OSError:  # pragma: no cover  # unreadable source file; defensive
            pass

    for crate in ("polars-cv", "polars-cv-macros", "view-buffer"):
        crate_root = root / crate
        for rs in sorted((crate_root / "src").rglob("*.rs")):
            _push(rs, f"{crate}/{rs.relative_to(crate_root).as_posix()}")
        _push(crate_root / "Cargo.toml", f"{crate}/Cargo.toml")
    # Must match `build.rs`'s input set exactly, or the guard fires on a
    # correctly-built extension and gets deleted for crying wolf. The workspace
    # manifest and the toolchain pin change the artifact without touching a
    # `.rs` file; `build.rs` decides what the hash covers at all.
    for name in ("Cargo.lock", "Cargo.toml", "rust-toolchain.toml"):
        _push(root / name, name)
    _push(root / "polars-cv" / "build.rs", "polars-cv/build.rs")

    digest = 0xCBF29CE484222325  # FNV-1a offset basis
    for key in sorted(contents):
        for byte in key.encode() + contents[key]:
            digest ^= byte
            digest = (digest * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


def build_info() -> dict[str, str | None]:
    """
    Report the versions of the things that can disagree.

    The install is editable, so edits to the Python sources are live — but the
    compiled extension is not. After a ``git pull`` that touches Rust,
    ``_lib.abi3.so`` keeps its build-time version until ``maturin develop`` is
    re-run, and new Python then runs against old Rust. When these three values
    disagree, re-run ``maturin develop``.

    Returns:
        Dict with:
            - ``version``: ``polars_cv.__version__``, from the imported Python source.
            - ``plugin_version``: the compiled Rust extension's version, baked in
              from ``Cargo.toml`` at build time. ``None`` if the plugin is not built.
            - ``dist_version``: the installed distribution metadata version.
              ``None`` if the package is not installed (e.g. run from a checkout).
            - ``plugin_source_hash``: the hash of the Rust sources the extension
              was *built* from. ``None`` if the plugin is not built.
            - ``source_hash``: the same hash recomputed from the working tree.
              ``None`` outside a source checkout.

    The two hashes are what actually detect staleness. The versions cannot:
    they are the release version, identical until the next bump, so they agree
    throughout the entire window in which Rust is edited without rebuilding.

    Example:
        ```python
        >>> import polars_cv
        >>> info = polars_cv.build_info()
        >>> len(set(v for v in info.values() if v is not None)) == 1  # all agree
        True
        ```
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        from . import _lib

        plugin_version = getattr(_lib, "__version__", None)
        plugin_source_hash = getattr(_lib, "__source_hash__", None)
    except ImportError:  # pragma: no cover  # extension always built in CI/dev
        plugin_version = None
        plugin_source_hash = None

    try:
        dist_version = _dist_version("polars-cv")
    except PackageNotFoundError:  # pragma: no cover  # package always installed
        dist_version = None

    return {
        "version": __version__,
        "plugin_version": plugin_version,
        "dist_version": dist_version,
        "plugin_source_hash": plugin_source_hash,
        "source_hash": _source_hash_from_tree(),
    }


def numpy_from_struct(
    row: dict[str, object] | pl.Series,
    *,
    copy: bool = True,
) -> "np.ndarray":
    """
    Convert numpy sink output struct to a NumPy array.

    Reads both ``sink("numpy")`` and ``sink("ndarray")`` output: a row of either
    is the same dict, and a single-row Series may be the plain struct or the
    ``polars_cv.ndarray`` type over it.

    To read a whole column, use :func:`numpy_from_column`, which views each
    row in the column's own memory instead of copying it through a ``dict``.

    Args:
        row: Struct value from output column.
        copy: Whether to copy data (default True). If False, returns a view.
    """
    import numpy as np

    if row is None:
        # A null row of the numpy/torch/ndarray sink is a null value.
        msg = "row is null (a null input or a failed row); it has no array"
        raise ValueError(msg)

    # Extract fields from struct
    if isinstance(row, dict):
        data = row.get("data")
        dtype_str = row.get("dtype")
        shape_list = row.get("shape")
        strides_list = row.get("strides")
        offset = row.get("offset", 0)
    elif isinstance(row, pl.Series):
        # A `sink("ndarray")` column is the same struct under a type tag.
        if isinstance(row.dtype, NdArrayType):
            row = row.ext.storage()
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
        shape = tuple(int(x) for x in shape_list)  # ty: ignore[not-iterable]

    # Convert strides to tuple (if present)
    strides: tuple[int, ...] | None = None
    if strides_list is not None:
        if isinstance(strides_list, pl.Series):
            strides = tuple(int(x) for x in strides_list.to_list())
        else:
            strides = tuple(int(x) for x in strides_list)  # ty: ignore[not-iterable]

    # Convert offset
    if offset is None:
        offset = 0
    else:
        offset = int(offset)  # ty: ignore[invalid-argument-type]

    if strides is None:
        # No stride metadata (older/dict callers): C-contiguous.
        itemsize = _checked_dtype(str(dtype_str)).itemsize
        c_strides: list[int] = []
        step = itemsize
        for n in reversed(shape):
            c_strides.append(step)
            step *= n
        strides = tuple(reversed(c_strides))

    # The row's bytes as a uint8 array: it owns (references) `data` and
    # names its address, and `_view` reads the row through it, checked.
    backing = np.frombuffer(_as_buffer(data), dtype=np.uint8)  # ty: ignore[no-matching-overload]
    return _view(
        backing,
        backing.__array_interface__["data"][0],
        backing.nbytes,
        str(dtype_str),
        shape,
        strides,
        offset,
        copy=copy,
    )


def _checked_dtype(dtype_name: str) -> np.dtype:
    """The numpy dtype a sink struct names, refusing anything else.

    The sink writes `DType::numpy_name()` into the struct, so those names are
    exactly what can legitimately arrive. Generated from `dtype_table!` rather
    than listed here: the hand-written list this replaced admitted numpy's
    *character codes* too, which meant `"u8"` (numpy uint64) sat in the same
    set as this project's `"u8"` (uint8) — so a caller hand-building a struct
    with `dtype="u8"` got a uint64 reinterpretation of the bytes, silently.
    """
    import numpy as np

    if dtype_name not in SINK_NUMPY_NAMES:
        msg = f"Unsupported dtype '{dtype_name}'. Allowed: {sorted(SINK_NUMPY_NAMES)}"
        raise ValueError(msg)
    return np.dtype(dtype_name)


def _view(
    owner: object,
    address: int,
    nbytes: int,
    dtype_name: str,
    shape: Sequence[int],
    strides: Sequence[int],
    offset: int,
    *,
    copy: bool,
) -> np.ndarray:
    """A numpy array over `nbytes` of memory at `address`, as a sink struct
    describes it: `shape`, byte `strides` (negative for a flip) and a byte
    `offset` to element zero.

    **The one way a sink struct becomes an array**, for
    :func:`numpy_from_struct` and :func:`numpy_from_column` alike. The array is
    built through ``__array_interface__``, which numpy does not bounds-check,
    so the byte range the strides reach is checked against the row's `nbytes`
    here: a struct describing memory outside its row (only a hand-built one
    can) is refused rather than read. The view holds `owner`, which keeps the
    memory alive, and is read-only.
    """
    import numpy as np

    dtype = _checked_dtype(dtype_name)
    shape, strides, offset = tuple(shape), tuple(strides), int(offset)
    if offset % dtype.itemsize != 0:
        msg = (
            f"Byte offset {offset} is not a multiple of itemsize {dtype.itemsize}; "
            "cannot reconstruct a typed strided view from this struct."
        )
        raise ValueError(msg)
    if len(strides) != len(shape):
        msg = f"strides {strides} and shape {shape} have different rank"
        raise ValueError(msg)
    if all(n > 0 for n in shape):
        low = offset + sum(
            min(0, (n - 1) * s) for n, s in zip(shape, strides, strict=True)
        )
        high = offset + sum(
            max(0, (n - 1) * s) for n, s in zip(shape, strides, strict=True)
        )
        if low < 0 or high + dtype.itemsize > nbytes:
            msg = (
                f"shape {list(shape)} with strides {list(strides)} at offset {offset} "
                f"reaches bytes [{low}, {high + dtype.itemsize}), outside the row's "
                f"{nbytes} bytes"
            )
            raise ValueError(msg)
    view = np.asarray(
        _RowView(
            owner,
            {
                "version": 3,
                "shape": shape,
                "strides": strides,
                "typestr": dtype.str,
                "data": (address + offset, True),
            },
        )
    )
    # `copy()` is C-ordered; `ascontiguousarray` would turn a 0-d scalar 1-d.
    return view.copy() if copy else view


class _RowView:
    """One row's bytes, described to numpy through ``__array_interface__``.

    ``np.asarray`` over this object views the address it names and keeps the
    object as the array's ``.base``, so the Arrow buffer the object holds (an
    ``ArrowBytes`` from the plugin) lives exactly as long as the array.
    """

    __slots__ = ("__array_interface__", "_owner")

    def __init__(self, owner: object, interface: dict[str, object]) -> None:
        self._owner = owner
        self.__array_interface__ = interface


def numpy_from_column(
    column: pl.Series,
    *,
    copy: bool = False,
) -> list[np.ndarray | None]:
    """
    Read every row of a ``sink("numpy")`` / ``sink("ndarray")`` column as a
    NumPy array, without copying.

    Each array is a view of the column's own Arrow memory, with the row's
    shape, dtype and byte strides (a transposed or flipped output stays a
    strided view). Nothing passes through Python ``bytes``, which is what
    reading rows as dicts and calling :func:`numpy_from_struct` costs: one copy
    of every row into ``bytes``, and another with ``copy=True``. The views are
    read-only, since Polars memory is immutable, and each keeps the memory it
    reads alive after the column is gone.

    Args:
        column: A numpy- or ndarray-sink output column.
        copy: Return owned, writable, C-contiguous arrays instead of views.

    Returns:
        One array per row, and ``None`` for a null row.

    Raises:
        TypeError: The column is not a numpy/ndarray sink column.
        ValueError: A row names a dtype the sink does not emit, or a view
            reaching outside the row's bytes (a hand-built struct; the sink
            never produces one).
    """
    from ._lib import binary_rows

    if isinstance(column.dtype, NdArrayType):
        column = column.ext.storage()
    if column.dtype != NUMPY_OUTPUT_SCHEMA:
        msg = (
            f"numpy_from_column reads a numpy/ndarray sink column "
            f"({NUMPY_OUTPUT_SCHEMA}), got {column.dtype}"
        )
        raise TypeError(msg)

    fields = column.struct.unnest()
    arrays: list[np.ndarray | None] = []
    for row, dtype_name, shape, strides, offset in zip(
        binary_rows(fields["data"]),
        fields["dtype"].to_list(),
        fields["shape"].to_list(),
        fields["strides"].to_list(),
        fields["offset"].to_list(),
        strict=True,
    ):
        if row is None:
            arrays.append(None)
            continue
        owner, address, nbytes = row
        arrays.append(
            _view(owner, address, nbytes, dtype_name, shape, strides, offset, copy=copy)
        )
    return arrays


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
        return memoryview(data)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    except TypeError:  # pragma: no cover  # non-buffer input; defensive fallback
        # Fallback: copy into bytes
        return bytes(data)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


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
    "OptFlags",
    # Types
    "CloudOptions",
    "ColorSpace",
    "HashAlgorithm",
    "dtype_name_for",
    # ImageNet normalization constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # NumPy conversion utilities
    "numpy_from_struct",
    "numpy_from_column",
    "NUMPY_OUTPUT_SCHEMA",
    # Arrow extension types
    "NdArrayType",
    "PointType",
    "ContourType",
    "BBoxType",
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
    # Detection metrics — matchers
    "ContourMatcher",
    "BBoxMatcher",
    "PreMatchedAdapter",
    # Detection metrics — functions
    "froc_auc",
    "froc_curve_lazy",
    "froc_sensitivity_at_fp",
    "froc_summary_table",
    "lroc_auc",
    "lroc_curve_lazy",
    "lroc_sensitivity_at_fpf",
    "froc_auc_ci_lazy",
    "lroc_auc_ci_lazy",
    "average_precision_ci_lazy",
    "precision_recall_curve",
    "average_precision",
    "mean_average_precision",
    "precision_at_threshold",
    "recall_at_threshold",
    "f1_at_threshold",
    "confusion_at_threshold",
    # Detection metrics — result types
    "ConfusionResult",
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
    "CORRESPONDENCE_SCHEMA",
    "BBOX_SCHEMA",
    # Build/version introspection
    "__version__",
    "build_info",
]
