"""
Type definitions for polars-cv.

This module contains the core type definitions used throughout the package,
including ParamValue for handling literal vs expression parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

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


# Mapping of operations to their output dtype rules
# This mirrors the Rust OutputDTypeRule for each operation
OPERATION_OUTPUT_DTYPE: dict[str, str] = {
    # Image operations - Fixed(U8)
    "grayscale": "u8",
    "resize": "u8",
    "blur": "u8",
    "threshold": "u8",
    "rotate": "u8",
    # Perceptual hash - Fixed(U8)
    "perceptual_hash": "u8",
    # Compute operations - PromoteToFloat or Configurable(F32)
    "normalize": "f32",  # Configurable, default F32
    "scale": "f32",  # PromoteToFloat
    "clamp": "f32",  # PromoteToFloat
    "relu": "f32",  # PromoteToFloat
    # Reductions - ForceF64 for global, PreserveInput for axis-based
    "reduce_sum": "f64",
    "reduce_max": "f64",
    "reduce_min": "f64",
    "reduce_mean": "f64",
    "reduce_std": "f64",
    "reduce_popcount": "f64",
    # ArgMax/ArgMin always return i64 (indices)
    "reduce_argmax": "i64",
    "reduce_argmin": "i64",
    # Cast - Configurable
    "cast": "u8",  # Default, overridden by params
    # Shape extraction returns f64 values (vector domain uses f64)
    "extract_shape": "f64",
    # Rasterize - produces u8 buffer
    "rasterize": "u8",
    # Geometry -> scalar/vector
    "contour_area": "f64",
    "contour_perimeter": "f64",
    "contour_centroid": "f64",
    "contour_bounding_box": "f64",
    # Histogram - output dtype depends on mode
    # counts -> u64, normalized -> f64, quantized -> u32, edges -> f64
    "histogram": "u64",  # Default for counts mode
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
    quality: int = 85  # For JPEG format
    shape: list[int] | None = None  # For ARRAY format

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "alias": self.alias,
            "format": self.format.value,
        }
        if self.format == SinkFormat.JPEG:
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
