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

from ._build_info import source_hash_from_tree
from ._cloud import CloudCredentialError, check_cloud_access
from ._ops_generated import This, this
from ._optimize import OptFlags
from ._this import ThisExpr
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
from .hashing import hamming_distance, hash_similarity
from .lazy import LazyPipelineExpr
from .masks import mask_dice, mask_iou, region_stats
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
    bootstrap_ci,
    confusion_at_threshold,
    evaluate_detections,
    evaluate_heatmaps,
    evaluate_segmentation,
    f1_at_threshold,
    froc_auc,
    froc_auc_ci_lazy,
    froc_curve_lazy,
    froc_operating_range,
    froc_sensitivity_at_fp,
    froc_summary_table,
    lroc_auc,
    lroc_auc_ci_lazy,
    lroc_curve_lazy,
    lroc_sensitivity_at_fpf,
    match_detections,
    mean_average_precision,
    precision_at_threshold,
    precision_recall_curve,
    recall_at_threshold,
    segmentation_measures,
)
from .numpy_io import numpy_from_column, numpy_from_struct
from .pipeline import Pipeline

# Registered at import, not on first use: a Parquet/IPC read of a tagged column
# consults the registry, and a column read before registration decays to its
# storage. Pure Python, so importing polars-cv still loads no compiled code.
register_extension_types()

__version__ = "0.41.0"


def set_row_parallelism(enabled: bool) -> bool:
    """
    Turn polars-cv's own row parallelism on or off for this process.

    On (the default), a plugin call spreads its rows over polars-cv's thread
    pool when that pool has idle threads. Off, every call runs its rows on the
    thread polars called it on, so the only parallelism is polars' own — the
    streaming engine's morsels, or one thread per call on the in-memory engine.
    Turn it off when the streaming engine already keeps every core busy, or on
    hosts where waking the pool's threads costs more than the rows (many
    hyperthreaded cores and cheap rows). Outputs are identical either way.

    Args:
        enabled: Whether calls may split their rows. Must be a ``bool``.

    Returns:
        The previous setting, so a caller can restore it.

    Examples:
        >>> previous = polars_cv.set_row_parallelism(False)
        >>> lf.collect(engine="streaming")  # doctest: +SKIP
        >>> polars_cv.set_row_parallelism(previous)
    """
    from . import _lib

    return _lib.set_row_parallelism(enabled)


def row_parallelism() -> bool:
    """Whether polars-cv splits a call's rows over its own thread pool."""
    from . import _lib

    return _lib.row_parallelism()


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
        "source_hash": source_hash_from_tree(),
    }


__all__ = [
    "Pipeline",
    "this",
    "This",
    "ThisExpr",
    "CvNamespace",
    "LazyPipelineExpr",
    "region_stats",
    "OptFlags",
    # Types
    "CloudOptions",
    "CloudCredentialError",
    "check_cloud_access",
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
    # Evaluation — one call from predictions and ground truth to a report
    "evaluate_detections",
    "evaluate_heatmaps",
    "evaluate_segmentation",
    "segmentation_measures",
    "match_detections",
    "bootstrap_ci",
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
    "froc_operating_range",
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
    # Threading
    "set_row_parallelism",
    "row_parallelism",
]
