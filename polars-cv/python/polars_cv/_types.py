"""
Type definitions for polars-cv.

This module contains the core type definitions used throughout the package,
including ParamValue for handling literal vs expression parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Union

try:
    from typing import TypeAlias
except ImportError:
    # Python < 3.10 compatibility
    from typing_extensions import TypeAlias

import polars as pl

# Type alias for values that can be either literals or expressions
LiteralOrExpr: TypeAlias = Union[int, float, str, pl.Expr]
IntOrExpr: TypeAlias = Union[int, pl.Expr]
FloatOrExpr: TypeAlias = Union[float, pl.Expr]


class SourceFormat(str, Enum):
    """Supported input source formats."""

    IMAGE_BYTES = "image_bytes"  # Decode PNG/JPEG (auto-detect)
    BLOB = "blob"  # VIEW protocol binary
    RAW = "raw"  # Raw bytes (requires dtype and shape)
    FILE_PATH = "file_path"  # Read from file path (local, cloud, or HTTP URL)
    CONTOUR = "contour"  # Contour struct data
    LIST = "list"  # Polars nested List column (requires dtype)
    ARRAY = "array"  # Polars fixed-size Array column (requires dtype)


class SinkFormat(str, Enum):
    """Supported output sink formats."""

    NUMPY = "numpy"  # NumPy-compatible bytes
    TORCH = "torch"  # PyTorch-compatible bytes
    PNG = "png"  # Re-encode as PNG
    JPEG = "jpeg"  # Re-encode as JPEG
    WEBP = "webp"  # Re-encode as WebP
    TIFF = "tiff"  # Re-encode as TIFF with LZW compression (supports floating-point)
    BLOB = "blob"  # VIEW protocol (for chaining)
    ARRAY = "array"  # Polars Array type (fixed shape)
    LIST = "list"  # Polars nested List (variable shape)
    NATIVE = "native"  # Returns Polars-native type based on output domain
    #                   - Buffer → error (use explicit format)
    #                   - Contour → Struct matching CONTOUR_SCHEMA
    #                   - Scalar → Float64
    #                   - Vector → List[Float64]


class DType(str, Enum):
    """Supported data types."""

    U8 = "u8"
    I8 = "i8"
    U16 = "u16"
    I16 = "i16"
    U32 = "u32"
    I32 = "i32"
    U64 = "u64"
    I64 = "i64"
    F32 = "f32"
    F64 = "f64"


class NormalizeMethod(str, Enum):
    """Normalization methods."""

    MINMAX = "minmax"
    ZSCORE = "zscore"
    PRESET = "preset"  # Channel-wise with preset mean/std values


# ImageNet normalization constants
# These are the standard normalization values computed from the ImageNet dataset.
# Use with: normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


class OutputDType(str, Enum):
    """
    Output dtype specification for operations that support dtype configuration.

    This allows users to control the output dtype of operations like normalize,
    scale, etc. The default behavior promotes integers to float32.
    """

    # Explicit dtype options
    F32 = "f32"  # Always output float32 (default for most operations)
    F64 = "f64"  # Output float64 for higher precision
    U8 = "u8"  # Output uint8 (useful for image pipelines)

    # Special options
    PRESERVE = "preserve"  # Keep input dtype (floats preserved, integers -> f32)


class FilterType(str, Enum):
    """Image resize filter types."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    LANCZOS3 = "lanczos3"


class HashAlgorithm(str, Enum):
    """
    Perceptual hash algorithm selection.

    Different algorithms trade off speed vs robustness to transformations:
    - AVERAGE: Fastest, least robust. Good for exact/near-exact matches.
    - DIFFERENCE: Gradient-based, good balance of speed and robustness.
    - PERCEPTUAL: DCT-based, most robust to resize/compression. Recommended default.
    - BLOCKHASH: Block-based, good resistance to cropping.
    """

    AVERAGE = "average"
    DIFFERENCE = "difference"
    PERCEPTUAL = "perceptual"
    BLOCKHASH = "blockhash"


class HistogramOutput(str, Enum):
    """
    Histogram output mode selection.

    Controls what the histogram operation returns:
    - COUNTS: Bin counts as a 1D array (default)
    - NORMALIZED: Histogram normalized to sum to 1.0
    - QUANTIZED: Input array with pixels replaced by bin indices
    - EDGES: Bin edge values
    """

    COUNTS = "counts"
    NORMALIZED = "normalized"
    QUANTIZED = "quantized"
    EDGES = "edges"


class PadMode(str, Enum):
    """
    Padding mode selection.

    Controls how padding values are determined:
    - CONSTANT: Fill with a constant value (default)
    - EDGE: Replicate edge values
    - REFLECT: Reflect values at edge (not including edge)
    - SYMMETRIC: Reflect values at edge (including edge)
    """

    CONSTANT = "constant"
    EDGE = "edge"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"


class PadPosition(str, Enum):
    """
    Position for pad_to_size.

    Controls where the original content is placed:
    - CENTER: Center content in padded area (default)
    - TOP_LEFT: Place content at top-left corner
    - BOTTOM_RIGHT: Place content at bottom-right corner
    """

    CENTER = "center"
    TOP_LEFT = "top-left"
    BOTTOM_RIGHT = "bottom-right"


class Domain(str, Enum):
    """
    Data domain for typed pipeline nodes.

    Tracks what type of data is flowing through the pipeline for
    static type inference and sink validation.
    """

    BUFFER = "buffer"  # Image/array data
    CONTOUR = "contour"  # Extracted geometry
    SCALAR = "scalar"  # Single numeric value
    VECTOR = "vector"  # Fixed-length numeric array


class ExpectedDType(str, Enum):
    """
    Expected output dtype for list/array sinks.

    This is used for static type inference at Polars planning time.
    The values correspond to Polars dtypes.
    """

    UINT8 = "u8"
    INT8 = "i8"
    UINT16 = "u16"
    INT16 = "i16"
    UINT32 = "u32"
    INT32 = "i32"
    UINT64 = "u64"
    INT64 = "i64"
    FLOAT32 = "f32"
    FLOAT64 = "f64"


class DTypeEffect(str, Enum):
    """
    How an operation affects the buffer dtype.

    Mirrors the Rust ``OutputDTypeRule`` enum so that plan-time dtype inference
    on the Python side agrees with the Rust execution layer.  Each operation
    declares exactly one ``DTypeEffect`` in its ``OpContract``.
    """

    PRESERVE = "preserve"
    """Output dtype == input dtype (e.g. resize, pad, crop)."""

    FIXED_U8 = "u8"
    """Output is always UInt8 (e.g. grayscale, threshold)."""

    FIXED_F32 = "f32"
    """Output is always Float32."""

    FIXED_F64 = "f64"
    """Output is always Float64 (e.g. global reductions)."""

    FIXED_I64 = "i64"
    """Output is always Int64 (e.g. argmax/argmin)."""

    FIXED_U64 = "u64"
    """Output is always UInt64 (e.g. histogram counts)."""

    FIXED_U32 = "u32"
    """Output is always UInt32 (e.g. histogram quantized)."""

    PROMOTE_TO_FLOAT = "promote"
    """Integer inputs become Float32; float inputs are unchanged."""

    CONFIGURABLE_F32 = "config_f32"
    """Default Float32, but overridable via ``out_dtype`` parameter."""

    def resolve(self, input_dtype: str) -> str:
        """
        Resolve the concrete output dtype given the current input dtype.

        When ``input_dtype`` is ``"auto"`` (unknown at plan time, e.g. from
        image sources), only effects with a deterministic output can resolve
        it.  ``PRESERVE`` and ``PROMOTE_TO_FLOAT`` propagate ``"auto"``
        because their output depends on the actual input dtype.

        Args:
            input_dtype: The dtype of the data entering this operation,
                or ``"auto"`` when the dtype is not yet known.

        Returns:
            The expected output dtype string (e.g. ``"f32"``, ``"u8"``,
            or ``"auto"`` if still unknown).
        """
        if input_dtype == "auto":
            # Fixed-output effects resolve regardless of input
            if self in (
                DTypeEffect.FIXED_U8,
                DTypeEffect.FIXED_F32,
                DTypeEffect.FIXED_F64,
                DTypeEffect.FIXED_I64,
                DTypeEffect.FIXED_U64,
                DTypeEffect.FIXED_U32,
            ):
                return self.value
            if self is DTypeEffect.CONFIGURABLE_F32:
                # Default output is f32; caller handles out_dtype override.
                return "f32"
            # PRESERVE, PROMOTE_TO_FLOAT: output depends on input -> still unknown
            return "auto"

        if self is DTypeEffect.PRESERVE:
            return input_dtype
        if self is DTypeEffect.PROMOTE_TO_FLOAT:
            return input_dtype if input_dtype in ("f32", "f64") else "f32"
        if self is DTypeEffect.CONFIGURABLE_F32:
            # Caller should check params for out_dtype override first.
            # This default is used when no override is present.
            return "f32"
        # All FIXED_* variants: the enum value *is* the dtype string.
        return self.value


class NdimEffect(str, Enum):
    """
    How an operation affects the number of dimensions.

    Used at plan-time to track the expected dimensionality of the buffer
    through a pipeline.
    """

    PRESERVE = "preserve"
    """Ndim unchanged (e.g. resize, blur, pad)."""

    REDUCE_ONE = "reduce_one"
    """Ndim decreases by 1 (axis-based reduction)."""

    TO_ZERO = "to_zero"
    """Global reduction to scalar (ndim → 0)."""

    TO_ONE = "to_one"
    """Output is a 1-D vector (e.g. perceptual_hash, extract_shape)."""

    TO_THREE = "to_three"
    """Output is 3-D (e.g. rasterize → [H, W, C])."""


@dataclass(frozen=True)
class OpContract:
    """
    Plan-time declaration of an operation's effects on dtype and ndim.

    Every operation must have an ``OpContract`` entry in
    :data:`OPERATION_CONTRACTS`.  The contract is the **single source of truth**
    on the Python side for dtype and ndim inference; it mirrors the Rust
    ``OutputDTypeRule`` and is validated at execution time.
    """

    dtype_effect: DTypeEffect
    ndim_effect: NdimEffect

    def resolve_dtype(
        self,
        input_dtype: str,
        params: "dict[str, ParamValue] | None" = None,
    ) -> str:
        """
        Resolve the concrete output dtype for this contract.

        Handles the ``CONFIGURABLE_F32`` case by checking for an
        ``out_dtype`` parameter override; otherwise delegates to
        :meth:`DTypeEffect.resolve`.

        Args:
            input_dtype: Dtype string entering the operation.
            params: Operation parameters (may contain ``out_dtype``).

        Returns:
            The expected output dtype string.
        """
        if (
            self.dtype_effect is DTypeEffect.CONFIGURABLE_F32
            and params
            and (od := params.get("out_dtype"))
            and not od.is_expr
        ):
            return str(od.value)
        return self.dtype_effect.resolve(input_dtype)


# ---------------------------------------------------------------------------
# Operation contracts – one entry per operation name.
#
# This replaces the old ``OPERATION_OUTPUT_DTYPE`` flat dict.  Each entry
# declares the dtype and ndim effects so that ``_compute_output_domain_dtype_ndim``
# can infer the output schema at plan-time.
#
# Notes on ``ndim_effect``:
#   - Reductions with an optional ``axis`` parameter may be either
#     ``REDUCE_ONE`` (axis given) or ``TO_ZERO`` (global).  The dict stores
#     the *global* variant; the axis case is handled by a param-dependent
#     override inside ``_compute_output_domain_dtype_ndim``.
#   - ``cast`` and ``histogram`` are param-dependent for dtype; the dict
#     stores sensible defaults that are overridden by param inspection.
# ---------------------------------------------------------------------------
OPERATION_CONTRACTS: dict[str, OpContract] = {
    # --- Image (spatial) operations – preserve input dtype ---
    "resize": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "resize_scale": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "resize_to_height": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "resize_to_width": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "resize_max": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "resize_min": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    # --- Image operations ---
    # Grayscale is a channel reduction that preserves element dtype.
    "grayscale": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    # Threshold always produces a U8 binary mask (0 or 255) regardless of input dtype.
    "threshold": OpContract(DTypeEffect.FIXED_U8, NdimEffect.PRESERVE),
    # Blur uses the image crate internally — requires and produces U8.
    "blur": OpContract(DTypeEffect.FIXED_U8, NdimEffect.PRESERVE),
    # Rotate is a dtype-preserving spatial transformation.
    "rotate": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    # --- Perceptual hash ---
    "perceptual_hash": OpContract(DTypeEffect.FIXED_U8, NdimEffect.TO_ONE),
    # --- Compute operations ---
    "normalize": OpContract(DTypeEffect.CONFIGURABLE_F32, NdimEffect.PRESERVE),
    "scale": OpContract(DTypeEffect.PROMOTE_TO_FLOAT, NdimEffect.PRESERVE),
    "clamp": OpContract(DTypeEffect.PROMOTE_TO_FLOAT, NdimEffect.PRESERVE),
    "relu": OpContract(DTypeEffect.PROMOTE_TO_FLOAT, NdimEffect.PRESERVE),
    "cast": OpContract(
        DTypeEffect.FIXED_U8, NdimEffect.PRESERVE
    ),  # overridden by params
    # --- Reductions (global defaults – axis variants overridden in code) ---
    "reduce_sum": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ZERO),
    "reduce_mean": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ZERO),
    "reduce_std": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ZERO),
    "reduce_max": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ZERO),
    "reduce_min": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ZERO),
    "reduce_popcount": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ZERO),
    "reduce_percentile": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ZERO),
    "reduce_argmax": OpContract(DTypeEffect.FIXED_I64, NdimEffect.TO_ZERO),
    "reduce_argmin": OpContract(DTypeEffect.FIXED_I64, NdimEffect.TO_ZERO),
    # --- Shape / domain transitions ---
    "extract_shape": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ONE),
    "rasterize": OpContract(DTypeEffect.FIXED_U8, NdimEffect.TO_THREE),
    # --- Geometry scalars / vectors ---
    "contour_area": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ZERO),
    "contour_perimeter": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ZERO),
    "contour_centroid": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ONE),
    "contour_bounding_box": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ONE),
    "label_reduce": OpContract(DTypeEffect.FIXED_F64, NdimEffect.TO_ONE),
    # --- Histogram (default is counts mode – overridden by params) ---
    "histogram": OpContract(DTypeEffect.FIXED_U64, NdimEffect.TO_ONE),
    # --- Padding / spatial view operations – preserve dtype ---
    "pad": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "pad_to_size": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "letterbox": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "crop": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "reshape": OpContract(
        DTypeEffect.PRESERVE, NdimEffect.PRESERVE
    ),  # ndim param-dependent
    "flip": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
    "transpose": OpContract(DTypeEffect.PRESERVE, NdimEffect.PRESERVE),
}


@dataclass
class ParamValue:
    """
    A parameter value that can be either a literal or an expression reference.

    When serialized, expressions are stored as column references that are
    resolved at execution time per row.
    """

    is_expr: bool
    value: Any  # The literal value or expression

    def __eq__(self, other: object) -> bool:
        """Compare two ParamValues for equality."""
        if not isinstance(other, ParamValue):
            return NotImplemented
        if self.is_expr != other.is_expr:
            return False
        if self.is_expr:
            # Compare expressions by their string representation
            return str(self.value) == str(other.value)
        return self.value == other.value

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        if self.is_expr:
            # Hash expression by string representation
            return hash((True, str(self.value)))
        # For literals, hash the value directly (works for immutable types)
        try:
            return hash((False, self.value))
        except TypeError:
            # Fallback for unhashable types (e.g., lists)
            return hash((False, str(self.value)))

    @classmethod
    def from_arg(cls, arg: LiteralOrExpr) -> "ParamValue":
        """
        Create a ParamValue from a literal or expression.

        Args:
            arg: Either a literal value (int, float, str) or a Polars expression.

        Returns:
            ParamValue with appropriate type flag.
        """
        if isinstance(arg, pl.Expr):
            return cls(is_expr=True, value=arg)
        return cls(is_expr=False, value=arg)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary for JSON encoding.

        Returns:
            Dictionary with type and value/expr fields.

        Note:
            For expression parameters, we use the expression's string representation
            as the identifier. This ensures unique keys even when multiple expressions
            share the same root column (e.g., col("x").max() and col("x").min()).
            The same string representation is used in _get_expr_columns() to ensure
            the keys match when looking up expression values on the Rust side.
        """
        if self.is_expr:
            # Use the expression's string representation as a unique identifier.
            # This avoids collisions when multiple expressions share the same root
            # (e.g., height_expr.max() and width_expr.max() from the same source).
            expr = self.value
            expr_str = str(expr)
            return {"type": "expr", "col": expr_str}
        return {"type": "literal", "value": self.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParamValue":
        """
        Deserialize from dictionary.

        Args:
            d: Dictionary with type and value/expr fields.

        Returns:
            ParamValue instance.
        """
        if d["type"] == "literal":
            return cls(is_expr=False, value=d["value"])
        # For expressions, we store the serialized form
        # Actual expression is reconstructed on the Rust side
        return cls(is_expr=True, value=d)


@dataclass
class CloudOptions:
    """
    Cloud storage options for file_path sources.

    Used to configure credentials and access options for cloud storage providers.
    When not provided, the default credential chain is used:
    1. Anonymous access (for public buckets)
    2. Environment variables (AWS_ACCESS_KEY_ID, GOOGLE_APPLICATION_CREDENTIALS, etc.)
    3. Instance metadata / IAM roles

    Attributes:
        aws_region: AWS region (e.g., "us-east-1").
        aws_access_key_id: AWS access key ID.
        aws_secret_access_key: AWS secret access key.
        aws_session_token: AWS session token (for temporary credentials).
        gcs_service_account_key: Path to GCS service account key file.
        azure_storage_account: Azure storage account name.
        azure_storage_access_key: Azure storage access key.
        anonymous: Whether to use anonymous access (default: None, auto-detect).
    """

    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    gcs_service_account_key: str | None = None
    azure_storage_account: str | None = None
    azure_storage_access_key: str | None = None
    anonymous: bool | None = None

    # Fields that contain sensitive credential data and should be masked
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "aws_secret_access_key",
            "aws_access_key_id",
            "aws_session_token",
            "azure_storage_access_key",
        }
    )

    def __repr__(self) -> str:
        """Return string representation with sensitive fields masked."""
        parts: list[str] = []
        for field_name in [
            "aws_region",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "gcs_service_account_key",
            "azure_storage_account",
            "azure_storage_access_key",
            "anonymous",
        ]:
            value = getattr(self, field_name)
            if value is not None:
                if field_name in self._SENSITIVE_FIELDS:
                    parts.append(f"{field_name}='***'")
                else:
                    parts.append(f"{field_name}={value!r}")
        return f"CloudOptions({', '.join(parts)})"

    def to_dict(self) -> dict[str, str]:
        """
        Serialize to dictionary for JSON encoding.

        Returns:
            Dictionary with non-None credential fields.
        """
        result: dict[str, str] = {}
        if self.aws_region is not None:
            result["aws_region"] = self.aws_region
        if self.aws_access_key_id is not None:
            result["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key is not None:
            result["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token is not None:
            result["aws_session_token"] = self.aws_session_token
        if self.gcs_service_account_key is not None:
            result["gcs_service_account_key"] = self.gcs_service_account_key
        if self.azure_storage_account is not None:
            result["azure_storage_account"] = self.azure_storage_account
        if self.azure_storage_access_key is not None:
            result["azure_storage_access_key"] = self.azure_storage_access_key
        if self.anonymous is not None:
            result["anonymous"] = str(self.anonymous).lower()
        return result


@dataclass
class SourceSpec:
    """Specification for pipeline input source."""

    format: SourceFormat
    dtype: DType | None = None  # For "raw" format
    # Contour source parameters
    width: "ParamValue | None" = None
    height: "ParamValue | None" = None
    fill_value: int = 255
    background: int = 0
    shape_pipeline: dict | None = (
        None  # Serialized LazyPipelineExpr for shape inference
    )
    # Cloud options for file_path sources
    cloud_options: CloudOptions | None = None
    # Contiguity requirement for list/array sources
    # When True, requires data to be contiguous for zero-copy; errors on jagged data
    # When False (default), allows jagged data with copy-based flattening
    require_contiguous: bool = False

    def __eq__(self, other: object) -> bool:
        """Compare two SourceSpecs for equality."""
        if not isinstance(other, SourceSpec):
            return NotImplemented
        return (
            self.format == other.format
            and self.dtype == other.dtype
            and self.width == other.width
            and self.height == other.height
            and self.fill_value == other.fill_value
            and self.background == other.background
            and self.shape_pipeline == other.shape_pipeline
            and self.cloud_options == other.cloud_options
            and self.require_contiguous == other.require_contiguous
        )

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash(
            (
                self.format,
                self.dtype,
                self.width,
                self.height,
                self.fill_value,
                self.background,
                str(self.shape_pipeline) if self.shape_pipeline else None,
                str(self.cloud_options) if self.cloud_options else None,
                self.require_contiguous,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {"format": self.format.value}
        if self.dtype is not None:
            result["dtype"] = self.dtype.value
        # Include contour-specific parameters if source is contour
        if self.format == SourceFormat.CONTOUR:
            if self.width is not None:
                result["width"] = self.width.to_dict()
            if self.height is not None:
                result["height"] = self.height.to_dict()
            result["fill_value"] = self.fill_value
            result["background"] = self.background
            if self.shape_pipeline is not None:
                result["shape_pipeline"] = self.shape_pipeline
        # Include require_contiguous for list/array sources
        if self.format in (SourceFormat.LIST, SourceFormat.ARRAY):
            result["require_contiguous"] = self.require_contiguous
        return result


@dataclass
class SinkSpec:
    """Specification for pipeline output sink."""

    format: SinkFormat
    quality: int = 85  # For JPEG and WebP
    shape: list[int] | None = None  # For ARRAY format

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {"format": self.format.value}
        if self.format == SinkFormat.JPEG or self.format == SinkFormat.WEBP:
            result["quality"] = self.quality
        if self.format == SinkFormat.ARRAY and self.shape is not None:
            result["shape"] = self.shape
        return result


@dataclass
class ShapeHints:
    """Optional shape hints for pipeline planning."""

    height: ParamValue | None = None
    width: ParamValue | None = None
    channels: ParamValue | None = None
    batch: ParamValue | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, omitting None values."""
        result: dict[str, Any] = {}
        if self.height is not None:
            result["height"] = self.height.to_dict()
        if self.width is not None:
            result["width"] = self.width.to_dict()
        if self.channels is not None:
            result["channels"] = self.channels.to_dict()
        if self.batch is not None:
            result["batch"] = self.batch.to_dict()
        return result

    def has_any(self) -> bool:
        """Check if any hints are provided."""
        return any(
            x is not None for x in [self.height, self.width, self.channels, self.batch]
        )

    def has_all_dims(self) -> bool:
        """Check if all image dimensions (H, W, C) are provided."""
        return all(
            x is not None and not x.is_expr
            for x in [self.height, self.width, self.channels]
        )


@dataclass
class OpSpec:
    """Specification for a single operation in the pipeline."""

    op: str  # Operation name
    params: dict[str, ParamValue] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        """Compare two OpSpecs for equality (same op and params)."""
        if not isinstance(other, OpSpec):
            return NotImplemented
        if self.op != other.op:
            return False
        if set(self.params.keys()) != set(other.params.keys()):
            return False
        return all(self.params[k] == other.params[k] for k in self.params)

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        # Create a stable hash from op name and sorted params
        param_hashes = tuple((k, hash(v)) for k, v in sorted(self.params.items()))
        return hash((self.op, param_hashes))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {"op": self.op}
        for key, value in self.params.items():
            result[key] = value.to_dict()
        return result


@dataclass
class OutputSpec:
    """
    Specification for a single output in multi-output mode.

    Represents one output in a multi-output sink, mapping an alias name
    to a specific format and optional parameters.
    """

    alias: str  # The user-defined alias name
    format: SinkFormat  # Output format for this alias
    quality: int = 85  # For JPEG and WebP
    shape: list[int] | None = None  # For ARRAY format

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "alias": self.alias,
            "format": self.format.value,
        }
        if self.format == SinkFormat.JPEG or self.format == SinkFormat.WEBP:
            result["quality"] = self.quality
        if self.format == SinkFormat.ARRAY and self.shape is not None:
            result["shape"] = self.shape
        return result


@dataclass
class MultiSinkSpec:
    """
    Specification for multi-output sink mode.

    When `.sink()` is called with a dict of aliases to formats, this
    captures all the output specifications for the pipeline.
    """

    outputs: dict[str, OutputSpec]  # alias -> OutputSpec

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "outputs": {alias: spec.to_dict() for alias, spec in self.outputs.items()}
        }

    @classmethod
    def from_dict_spec(
        cls,
        spec: dict[str, str],
        quality: int = 85,
    ) -> "MultiSinkSpec":
        """
        Create from a simple dict mapping aliases to format strings.

        Args:
            spec: Dict mapping alias names to format strings (e.g., {"img": "numpy"})
            quality: JPEG quality for any jpeg outputs.

        Returns:
            MultiSinkSpec instance.

        Raises:
            ValueError: If any format is invalid.
        """
        outputs: dict[str, OutputSpec] = {}
        for alias, fmt_str in spec.items():
            try:
                fmt = SinkFormat(fmt_str)
            except ValueError as e:
                valid = [f.value for f in SinkFormat]
                msg = f"Invalid format '{fmt_str}' for alias '{alias}'. Valid: {valid}"
                raise ValueError(msg) from e
            outputs[alias] = OutputSpec(alias=alias, format=fmt, quality=quality)
        return cls(outputs=outputs)
