"""
Type definitions for polars-cv.

This module contains the core type definitions used throughout the package,
including ParamValue for handling literal vs expression parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Union

try:
    from typing import TypeAlias
except ImportError:
    # Python < 3.10 compatibility
    from typing_extensions import TypeAlias

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

# Type alias for values that can be either literals or expressions
LiteralOrExpr: TypeAlias = Union[int, float, str, pl.Expr]
IntOrExpr: TypeAlias = Union[int, pl.Expr]
FloatOrExpr: TypeAlias = Union[float, pl.Expr]
# For non-structural flags only. A flag that changes the output shape — such as
# ``rotate(expand)`` — stays a plain ``bool``.
BoolOrExpr: TypeAlias = Union[bool, pl.Expr]
StrOrExpr: TypeAlias = Union[str, pl.Expr]


class SourceFormat(str, Enum):
    """Supported input source formats."""

    AUTO = "auto"  # Infer decode path from the column dtype (the default)
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


class ColorSpace(str, Enum):
    """Supported color spaces for ``convert_color``."""

    RGB = "rgb"
    BGR = "bgr"
    HSV = "hsv"
    LAB = "lab"
    YCBCR = "ycbcr"
    GRAY = "gray"


class FilterType(str, Enum):
    """Image resize filter types.

    Full parity with view-buffer's ``FilterType`` authority. ``BILINEAR`` is the
    API name for view-buffer's ``Triangle`` variant; the parser also accepts
    ``"triangle"`` as a backwards-compatible alias.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CATMULLROM = "catmullrom"
    GAUSSIAN = "gaussian"
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
    - COUNTS: Bin counts as a 1D array
    - NORMALIZED: Histogram normalized to sum to 1.0
    - QUANTIZED: Input array with pixels replaced by bin indices
    - EDGES: Bin edge values
    - BUCKETS: List of bucket structs (lower_edge, upper_edge, count, normalized)
    """

    COUNTS = "counts"
    NORMALIZED = "normalized"
    QUANTIZED = "quantized"
    EDGES = "edges"
    BUCKETS = "buckets"


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


class ExtractMode(str, Enum):
    """
    Contour retrieval mode for ``extract_contours``.

    Mirrors view-buffer's ``ExtractMode`` authority:
    - EXTERNAL: Outermost contours only (default).
    - TREE: Full nesting hierarchy.
    - ALL: Every contour, without hierarchy.
    """

    EXTERNAL = "external"
    TREE = "tree"
    ALL = "all"


class ApproxMethod(str, Enum):
    """
    Contour point-approximation method for ``extract_contours``.

    Mirrors view-buffer's ``ApproxMethod`` authority:
    - NONE: Keep every boundary point.
    - SIMPLE: Drop redundant collinear points (default).
    - APPROX: Douglas-Peucker style approximation.
    """

    NONE = "none"
    SIMPLE = "simple"
    APPROX = "approx"


class InterpolationType(str, Enum):
    """
    Interpolation used when sampling an affine warp (``rotate``,
    ``warp_affine``, ``shear``, ``rotate_and_scale``).

    Mirrors view-buffer's ``InterpolationType`` authority:
    - NEAREST: Nearest-neighbour sampling (preserves hard edges/pixel art).
    - BILINEAR: Bilinear sampling (default).
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"


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


class BorderMode(str, Enum):
    """
    Border-handling mode for 2D convolution (``convolve2d``).

    Mirrors view-buffer's ``BorderMode`` authority:
    - REPLICATE: Replicate the nearest edge pixel.
    - ZERO: Treat out-of-bounds pixels as zero.
    - REFLECT: Reflect pixels around the edge (dcba|abcd|dcba).
    """

    REPLICATE = "replicate"
    ZERO = "zero"
    REFLECT = "reflect"


class HistogramClosed(str, Enum):
    """
    Interval inclusiveness for histogram binning.

    Mirrors view-buffer's ``HistogramClosed`` authority:
    - LEFT: Intervals are left-closed ``[a, b)``.
    - RIGHT: Intervals are right-closed ``(a, b]``.
    """

    LEFT = "left"
    RIGHT = "right"


class LabelReduction(str, Enum):
    """
    Reduction over a contour region's pixel values (``label_reduce``).

    Mirrors view-buffer's ``LabelReduction`` authority.
    """

    MAX = "max"
    MEAN = "mean"
    SUM = "sum"


class LabelRegionMode(str, Enum):
    """
    Region selection for ``label_reduce``.

    Mirrors view-buffer's ``LabelRegionMode`` authority:
    - INTERIOR: Pixels strictly inside the contour polygon.
    - BOUNDARY: Interior pixels plus pixels on the contour boundary.
    - BBOX: All pixels within the bounding box.
    """

    INTERIOR = "interior"
    BOUNDARY = "boundary"
    BBOX = "bbox"


class ScaleOrigin(str, Enum):
    """
    Point a contour scale operation is measured from (``.contour.scale``).

    Mirrors view-buffer's ``ScaleOrigin`` authority.
    """

    CENTROID = "centroid"
    BBOX_CENTER = "bbox_center"
    ORIGIN = "origin"


class Winding(str, Enum):
    """
    Winding direction of a contour ring (``.contour.ensure_winding``).

    Mirrors view-buffer's ``Winding`` authority, long spellings included: the
    plugin has always accepted ``"clockwise"``/``"counterclockwise"`` alongside
    the short forms, and the annotation that named only the short ones was the
    reason nobody noticed the parser silently ignored everything else.
    """

    CCW = "ccw"
    COUNTERCLOCKWISE = "counterclockwise"
    CW = "cw"
    CLOCKWISE = "clockwise"


class Domain(str, Enum):
    """
    Data domain for typed pipeline nodes.

    Tracks what type of data is flowing through the pipeline for
    static type inference and sink validation.
    """

    BUFFER = "buffer"  # Image/array data
    CONTOUR = "contour"  # Extracted geometry
    SCALAR = "scalar"  # Single numeric value
    VECTOR = "vector"  # Fixed-length numeric array (incl. histogram buckets,
    # whose List(Struct) schema is selected by the sink encoding, not the domain)


def _reject_expr(value: "Any", what: str) -> None:
    """Reject a Polars expression for a structural parameter.

    Structural parameters fix the output shape, rank, or dtype at planning
    time, so an expression there would desync the lazy schema from the produced
    data. Without this guard the expression fails much later and opaquely —
    inside ``bool()`` ("the truth value of an Expr is ambiguous") or at JSON
    serialization — instead of naming the real problem. Mirrors the message
    ``ParamValue.__post_init__`` raises for scalar structural params.
    """
    if isinstance(value, pl.Expr):
        msg = (
            f"{what} is structural (it fixes the output shape/rank/dtype at "
            "planning time) and must be a literal, not a Polars expression."
        )
        raise TypeError(msg)


def _validate_enum(value: str, enum_cls: type, label: str):
    """Validate a *literal* string against a user-facing enum.

    The single validation shape for every literal enum-valued parameter:
    ``Invalid <label> '<value>'. Valid: [...]``. Enums that may vary per row go
    through ``pipeline._enum_param`` instead, so reaching here with an
    expression means the parameter is structural.

    Lives beside the enums rather than in ``pipeline.py`` because the geometry
    accessors need the same check and importing it from the builder module
    would have meant either a second copy or an import cycle. A second copy is
    how ``.contour.scale(origin=)`` came to accept anything at all.
    """
    _reject_expr(value, f"'{label}'")
    try:
        return enum_cls(value)
    except ValueError as e:
        valid = [v.value for v in enum_cls]
        msg = f"Invalid {label} '{value}'. Valid: {valid}"
        raise ValueError(msg) from e


def _enum_or_expr(value: "Any", enum_cls: type, label: str) -> "Any":
    """Validate a literal enum value, or pass an expression through untouched.

    The geometry accessors' counterpart to ``pipeline._enum_param``: a literal
    is checked here and fails at build time naming the accepted spellings; an
    expression cannot be checked until the row exists, so it is handed to
    ``_ArgBinder``, which appends it as a per-row plugin input. Rust then reads
    it through the same ``NAMED`` table and rejects an unknown value with the
    same "Expected one of [...]" error.

    **Only for enums with no effect on output shape, rank or dtype.** That is
    the eligibility rule for any per-row parameter (root ``CLAUDE.md``), and it
    is what makes the plan-time schema safe to publish before the value is
    known. A structural enum — ``cast(dtype=)``, ``normalize(method=)``,
    ``histogram(output=)`` — must stay on :func:`_validate_enum`, which rejects
    expressions outright.
    """
    if isinstance(value, pl.Expr):
        return value
    return _validate_enum(value, enum_cls, label).value


@dataclass
class ParamValue:
    """
    A parameter value that can be either a literal or an expression reference.

    When serialized, expressions are stored as column references that are
    resolved at execution time per row.
    """

    is_expr: bool
    value: Any  # The literal value or expression

    def __post_init__(self) -> None:
        # A literal parameter can never hold a Polars expression. Structural
        # params (enum tags, axes, kernel shapes, hash_size) are built as
        # literals precisely because they fix the plan-time schema; routing an
        # expression to one lands here with a clear error instead of failing
        # opaquely later at JSON serialization. Dynamic params must set
        # ``is_expr=True`` (see :meth:`from_arg` / ``Pipeline._track_expr``).
        if not self.is_expr and isinstance(self.value, pl.Expr):
            msg = (
                "This parameter is structural (it fixes the output shape/rank "
                "at planning time) and must be a literal, not a Polars "
                "expression."
            )
            raise TypeError(msg)

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
    Remote requests are signed by default, using the provider's standard
    credential chain when explicit keys are not supplied:
    1. Environment variables (AWS_ACCESS_KEY_ID, GOOGLE_APPLICATION_CREDENTIALS, etc.)
    2. Instance metadata / IAM roles

    To read public buckets without credentials, opt into anonymous access
    explicitly with ``anonymous=True`` (honored for S3, GCS, and Azure).

    For anything beyond the named fields below, use ``storage_options`` to pass
    arbitrary configuration straight through to the underlying ``object_store``
    backend, keyed by its native config names. For example, GCS understands
    ``google_service_account`` (path to a service-account JSON file),
    ``google_service_account_key`` (inline service-account JSON), and
    ``google_application_credentials`` (path to an Application Default
    Credentials file); S3 understands ``aws_endpoint``, ``aws_virtual_hosted_style_request``,
    and so on. Keys in ``storage_options`` win over the named fields on collision.

    Federated Google credentials (workforce/workload identity, i.e. GCS ADC of
    type ``external_account`` or ``external_account_authorized_user``) cannot be
    parsed by ``object_store``. They are handled without any extra configuration:
    when the ambient ADC is federated, polars-cv delegates to
    ``gcloud auth application-default print-access-token`` (so the ``gcloud`` CLI
    must be on ``PATH``) and uses the resulting access token. Set the environment
    variable ``POLARS_CV_DISABLE_GCS_FEDERATION=1`` to turn that off.

    To obtain a bearer token some other way — a custom broker, a wrapper script,
    or a different CLI — set ``token_command`` to any shell command that prints
    an access token to stdout. This is provider-agnostic and applies to the
    OAuth-bearer backends, **GCS and Azure**::

        opts = CloudOptions(
            token_command="gcloud auth application-default print-access-token"
        )
        # Azure, equivalently:
        CloudOptions(token_command="az account get-access-token "
                     "--resource https://storage.azure.com/ --query accessToken -o tsv")

    ``token_command`` does not apply to S3, which authenticates with SigV4 rather
    than a bearer token; passing it with an ``s3://`` source raises. Or pass a
    pre-obtained GCS token directly via ``gcs_bearer_token``. Tokens from any of
    these routes are cached until shortly before they expire.

    Attributes:
        aws_region: AWS region (e.g., "us-east-1").
        aws_access_key_id: AWS access key ID.
        aws_secret_access_key: AWS secret access key.
        aws_session_token: AWS session token (for temporary credentials).
        gcs_service_account_key: Path to GCS service account key file.
        azure_storage_account: Azure storage account name.
        azure_storage_access_key: Azure storage access key.
        gcs_bearer_token: Pre-obtained GCS OAuth access token (bearer). Escape
            hatch for credential types object_store cannot load natively.
        token_command: Shell command whose stdout is an OAuth access token, run
            to obtain a bearer credential for federated/brokered setups. Applies
            to GCS and Azure (not S3); takes precedence over the automatic
            ``gcloud`` delegation.
        storage_options: Extra options forwarded verbatim to the object_store
            backend, keyed by its native config names. Wins over named fields.
        anonymous: Set to True to opt into unsigned/anonymous access for public
            buckets. Default None signs requests using the credential chain above.
    """

    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    gcs_service_account_key: str | None = None
    azure_storage_account: str | None = None
    azure_storage_access_key: str | None = None
    anonymous: bool | None = None
    # New fields appended after `anonymous` to preserve positional construction.
    gcs_bearer_token: str | None = None
    storage_options: dict[str, str] | None = None
    token_command: str | None = None

    # Fields that contain sensitive credential data and should be masked
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "aws_secret_access_key",
            "aws_access_key_id",
            "aws_session_token",
            "azure_storage_access_key",
            "gcs_bearer_token",
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
            "gcs_bearer_token",
            "token_command",
        ]:
            value = getattr(self, field_name)
            if value is not None:
                if field_name in self._SENSITIVE_FIELDS:
                    parts.append(f"{field_name}='***'")
                else:
                    parts.append(f"{field_name}={value!r}")
        # Pass-through options may carry secrets (inline keys, SAS tokens); show
        # only the key names with masked values.
        if self.storage_options:
            masked = ", ".join(f"{k!r}: '***'" for k in self.storage_options)
            parts.append(f"storage_options={{{masked}}}")
        if self.anonymous is not None:
            parts.append(f"anonymous={self.anonymous!r}")
        return f"CloudOptions({', '.join(parts)})"

    def to_dict(self) -> dict[str, str]:
        """
        Serialize to dictionary for JSON encoding.

        Named fields are emitted first, then ``storage_options`` is merged in
        (overriding any collisions), matching the precedence documented on the
        class.

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
        if self.gcs_bearer_token is not None:
            result["bearer_token"] = self.gcs_bearer_token
        if self.token_command is not None:
            result["token_command"] = self.token_command
        if self.anonymous is not None:
            result["anonymous"] = str(self.anonymous).lower()
        if self.storage_options:
            for key, value in self.storage_options.items():
                result[key] = str(value)
        return result


def normalize_cloud_options(
    value: "CloudOptions | dict[str, Any] | None",
) -> "CloudOptions | None":
    """Coerce a user-supplied ``cloud_options`` argument to ``CloudOptions``.

    Accepts a ``CloudOptions`` unchanged, or a dict whose keys are either named
    ``CloudOptions`` fields or ``object_store`` config names — the latter are
    routed into ``storage_options`` as pass-through options.

    Shared by every entry point that takes credentials (``Pipeline.source()``
    and ``.cv.read_bytes()``) so they accept exactly the same forms.

    Args:
        value: ``CloudOptions``, a dict of options, or None.

    Returns:
        A ``CloudOptions`` instance, or None when ``value`` is None.

    Raises:
        TypeError: If ``value`` is neither ``CloudOptions``, a dict, nor None.
    """
    if value is None:
        return None
    if isinstance(value, CloudOptions):
        return value
    if not isinstance(value, dict):
        msg = f"cloud_options must be CloudOptions or dict, got {type(value)}"
        raise TypeError(msg)

    known_fields = set(CloudOptions.__dataclass_fields__)
    opts_dict: dict[str, Any] = {}
    passthrough: dict[str, str] = {}
    for key, item in value.items():
        if key in known_fields:
            opts_dict[key] = item
        else:
            passthrough[key] = item
    # Convert "anonymous" from string if present
    if isinstance(opts_dict.get("anonymous"), str):
        opts_dict["anonymous"] = opts_dict["anonymous"].lower() == "true"
    if passthrough:
        merged = dict(opts_dict.get("storage_options") or {})
        merged.update(passthrough)
        opts_dict["storage_options"] = merged
    return CloudOptions(**opts_dict)


@dataclass
class SourceSpec:
    """Specification for pipeline input source."""

    format: SourceFormat
    dtype: DType | None = None  # For "raw" format
    # Contour source parameters
    width: "ParamValue | None" = None
    height: "ParamValue | None" = None
    fill_value: "ParamValue | None" = None
    background: "ParamValue | None" = None
    shape_pipeline: dict | None = (
        None  # Serialized LazyPipelineExpr for shape inference
    )
    # Cloud options for file_path sources
    cloud_options: CloudOptions | None = None
    # Contiguity requirement for list/array sources
    # When True, requires data to be contiguous for zero-copy; errors on jagged data
    # When False (default), allows jagged data with copy-based flattening
    require_contiguous: bool = False
    # Error handling for source decoding:
    #   "raise" (default): propagate decode errors (fails the entire batch)
    #   "null": treat decode errors as null output for that row
    on_error: str = "raise"
    # Explicit decode-scale assertion: the pipeline only needs this many
    # pixels on the decoded image's long side (JPEG uses IDCT scaling).
    decode_max_size: int | None = None
    # Locations this source's path column may read from. None (default) is
    # unrestricted; a tuple restricts reads to those roots. A tuple rather than
    # a list because SourceSpec is hashed for CSE.
    allowed_roots: tuple[str, ...] | None = None

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
            and self.on_error == other.on_error
            and self.decode_max_size == other.decode_max_size
            and self.allowed_roots == other.allowed_roots
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
                self.on_error,
                self.decode_max_size,
                self.allowed_roots,
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
            if self.fill_value is not None:
                result["fill_value"] = self.fill_value.to_dict()
            if self.background is not None:
                result["background"] = self.background.to_dict()
            if self.shape_pipeline is not None:
                result["shape_pipeline"] = self.shape_pipeline
        # Include require_contiguous for list/array sources ("auto" may resolve
        # to a list/array column at runtime).
        if self.format in (SourceFormat.LIST, SourceFormat.ARRAY, SourceFormat.AUTO):
            result["require_contiguous"] = self.require_contiguous
        # Cloud credentials must round-trip for file_path sources so graph
        # execution can authenticate remote reads ("auto" may resolve to
        # file_path from a String column at runtime).
        if (
            self.format in (SourceFormat.FILE_PATH, SourceFormat.AUTO)
            and self.cloud_options is not None
        ):
            result["cloud_options"] = self.cloud_options.to_dict()
        if self.decode_max_size is not None:
            result["decode_max_size"] = self.decode_max_size
        if self.on_error != "raise":
            result["on_error"] = self.on_error
        # A path allowlist rides for the source formats that read paths.
        # Emitted only when set, so an unrestricted source's spec — and the
        # graph-cache key built from it — is byte-identical to before.
        if (
            self.format in (SourceFormat.FILE_PATH, SourceFormat.AUTO)
            and self.allowed_roots is not None
        ):
            result["allowed_roots"] = list(self.allowed_roots)
        return result


# ---------------------------------------------------------------------------
# Which format each spec parameter applies to
# ---------------------------------------------------------------------------

#: Which source formats each :meth:`Pipeline.source` keyword applies to.
#:
#: Each parameter is listed against exactly the formats whose decode path reads
#: it. Set arithmetic where the fact is genuinely "all of them" or "all but
#: one", so a new format does not silently fall outside a parameter that should
#: cover it.
SOURCE_PARAM_APPLIES: "dict[str, frozenset[SourceFormat]]" = {
    # Every source carries an element dtype except the contour one, whose
    # rasterize fixes u8 (`OutputDTypeRule::Fixed(U8)`).
    "dtype": frozenset(SourceFormat) - {SourceFormat.CONTOUR},
    # The canvas and its colours: read only by the contour decode's rasterize.
    "width": frozenset({SourceFormat.CONTOUR}),
    "height": frozenset({SourceFormat.CONTOUR}),
    "shape": frozenset({SourceFormat.CONTOUR}),
    "fill_value": frozenset({SourceFormat.CONTOUR}),
    "background": frozenset({SourceFormat.CONTOUR}),
    # Path reads: `file_path`, and `auto` when a String column resolves to one.
    "cloud_options": frozenset({SourceFormat.FILE_PATH, SourceFormat.AUTO}),
    "allowed_roots": frozenset({SourceFormat.FILE_PATH, SourceFormat.AUTO}),
    # Zero-copy contiguity applies to the nested-column decode.
    "require_contiguous": frozenset(
        {SourceFormat.LIST, SourceFormat.ARRAY, SourceFormat.AUTO}
    ),
    # JPEG IDCT scaling, applied where bytes are decoded as an image.
    "decode_max_size": frozenset(
        {SourceFormat.AUTO, SourceFormat.IMAGE_BYTES, SourceFormat.FILE_PATH}
    ),
    # Every source can fail to decode, including a contour that will not parse.
    "on_error": frozenset(SourceFormat),
}

#: Which sink formats each ``.sink()`` keyword applies to.
#:
#: The same fact for the other end of the pipeline, read from the same place:
#: the Rust encoder's use of the `SinkSpec` field.
#:
#: `quality` is **jpeg only**. `SinkSpec` documents it as "JPEG and WebP" and
#: the sink docstring said "jpeg/webp", but `encode_image` passes it to
#: `encode_jpeg` alone — the WebP arm calls `ImageAdapter::encode`, which has no
#: quality argument. A webp quality is therefore rejected rather than accepted
#: and dropped; supporting it is an encoder change, not a parameter change.
SINK_PARAM_APPLIES: "dict[str, frozenset[SinkFormat]]" = {
    "quality": frozenset({SinkFormat.JPEG}),
    "shape": frozenset({SinkFormat.ARRAY}),
    "dtype": frozenset({SinkFormat.NUMPY, SinkFormat.TORCH}),
}

#: What to do instead, for the parameters where a caller has a real
#: alternative. Keyed by ``(kind, parameter)``.
PARAM_HINTS: "dict[tuple[str, str], str]" = {
    ("source", "dtype"): (
        "rasterizing always produces u8 — use .cast(...) after the source"
    ),
    ("sink", "dtype"): "cast inside the pipeline with .cast(...) instead",
    ("sink", "quality"): (
        "only the JPEG encoder takes a quality; the others encode at their "
        "own fixed settings"
    ),
}


def reject_inapplicable_params(
    *,
    kind: str,
    fmt: "SourceFormat | SinkFormat",
    supplied: "Mapping[str, Any]",
    applies: "Mapping[str, frozenset[Any]]",
) -> None:
    """Reject spec parameters the chosen format never reads.

    **The single answer to "does this parameter do anything here?"**, for both
    ends of the pipeline. Each surface used to answer it per parameter and
    differently: of the source's seven scoped keywords one raised, one warned
    and five were dropped silently, while every scoped sink keyword but
    ``dtype`` was dropped silently. A parameter that does nothing is not a
    harmless no-op — ``source("image_bytes", width=224)`` reads as a decode
    size, and ``sink("png", quality=50)`` reads as compression.

    A name absent from *applies* is rejected as well as one that is present but
    inapplicable. That is what closes an open ``**kwargs`` surface: ``.sink()``
    took any keyword at all and serialized it into the graph, so
    ``sink("jpeg", qualtiy=50)`` silently encoded at the default quality.

    Args:
        kind: ``"source"`` or ``"sink"``, for the message and the hint lookup.
        fmt: The chosen format.
        supplied: Parameter name → value, for what the caller actually passed.
        applies: The authority for this kind (:data:`SOURCE_PARAM_APPLIES` or
            :data:`SINK_PARAM_APPLIES`).
    """
    for name in sorted(supplied):
        formats = applies.get(name)
        if formats is None:
            known = ", ".join(sorted(applies))
            raise ValueError(f"{name} is not a {kind} parameter (known: {known}).")
        if fmt in formats:
            continue
        spelled = ", ".join(sorted(f.value for f in formats))
        hint = PARAM_HINTS.get((kind, name))
        msg = (
            f"{name} does not apply to the '{fmt.value}' {kind} "
            f"(it applies to: {spelled})"
        )
        raise ValueError(f"{msg}; {hint}." if hint else f"{msg}.")


def is_supplied(value: Any, default: Any) -> bool:
    """Did the caller pass ``value``, or is it the parameter's default?

    Identity first so ``None``/``False`` defaults are exact, then equality for
    value defaults (``fill_value=255``). A Polars expression short-circuits: no
    default is an expression, and ``Expr.__ne__`` builds an expression rather
    than answering, so comparing one would raise "the truth value of an Expr is
    ambiguous" instead of reporting it as supplied.

    Only needed where a parameter surface has defaults to compare against —
    ``.sink()`` takes ``**kwargs``, where every key present was passed.
    """
    if value is default:
        return False
    if isinstance(value, pl.Expr):
        return True
    return bool(value != default)


#: The sources whose element dtype and rank are resolved from the Polars column
#: at plan-time-with-input (Rust ``resolved_output_specs``) rather than at build
#: time. The typed ``list``/``array`` sinks defer to that instead of demanding
#: an explicit dtype; spelled once because three separate checks in `lazy.py`
#: carried their own copy of the tuple.
SOURCES_RESOLVED_FROM_COLUMN: "frozenset[SourceFormat]" = frozenset(
    {SourceFormat.LIST, SourceFormat.ARRAY, SourceFormat.AUTO}
)


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
