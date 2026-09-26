"""
Type definitions for polars-cv.

This module contains the core type definitions used throughout the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Union

try:
    from typing import TypeAlias
except ImportError:
    # Python < 3.10 compatibility
    from typing_extensions import TypeAlias

import polars as pl

# The enums are generated from the Rust registries (`scripts/gen_ops.py`);
# re-exported here, where the rest of the package imports them from.
from polars_cv._ops_generated import (
    ApproxMethod as ApproxMethod,
)
from polars_cv._ops_generated import (
    BorderMode as BorderMode,
)
from polars_cv._ops_generated import (
    ColorSpace as ColorSpace,
)
from polars_cv._ops_generated import (
    Domain as Domain,
)
from polars_cv._ops_generated import (
    DType as DType,
)
from polars_cv._ops_generated import (
    ExtractMode as ExtractMode,
)
from polars_cv._ops_generated import (
    FetchErrorPolicy as FetchErrorPolicy,
)
from polars_cv._ops_generated import (
    FilterType as FilterType,
)
from polars_cv._ops_generated import (
    HashAlgorithm as HashAlgorithm,
)
from polars_cv._ops_generated import (
    HistogramClosed as HistogramClosed,
)
from polars_cv._ops_generated import (
    HistogramOutput as HistogramOutput,
)
from polars_cv._ops_generated import (
    InterpolationType as InterpolationType,
)
from polars_cv._ops_generated import (
    LabelReduction as LabelReduction,
)
from polars_cv._ops_generated import (
    LabelRegionMode as LabelRegionMode,
)
from polars_cv._ops_generated import (
    NormalizeMethod as NormalizeMethod,
)
from polars_cv._ops_generated import (
    NullParamPolicy as NullParamPolicy,
)
from polars_cv._ops_generated import (
    PadMode as PadMode,
)
from polars_cv._ops_generated import (
    PadPosition as PadPosition,
)
from polars_cv._ops_generated import (
    RowErrorPolicy as RowErrorPolicy,
)
from polars_cv._ops_generated import (
    ScaleOrigin as ScaleOrigin,
)
from polars_cv._ops_generated import (
    SinkFormat as SinkFormat,
)
from polars_cv._ops_generated import (
    SourceFormat as SourceFormat,
)
from polars_cv._ops_generated import (
    Winding as Winding,
)

from ._dtype_names import NUMPY_TO_SHORT

if TYPE_CHECKING:
    pass

# Type alias for values that can be either literals or expressions
IntOrExpr: TypeAlias = Union[int, pl.Expr]
FloatOrExpr: TypeAlias = Union[float, pl.Expr]
# For non-structural flags only. A flag that changes the output shape — such as
# ``rotate(expand)`` — stays a plain ``bool``.
BoolOrExpr: TypeAlias = Union[bool, pl.Expr]
StrOrExpr: TypeAlias = Union[str, pl.Expr]


#: Polars types that reach a buffer through a *cast* rather than a numpy name.
#:
#: A boolean mask is the documented shape for a ground-truth mask column and
#: the natural output of ``np_mask.astype(bool).tolist()``. It is not a buffer
#: element type, but the source decoder casts rather than reinterprets
#: (``series_to_bytes`` in ``graph/decode.rs``), so ``u8`` gives the 0/1 a mask
#: means. This is the one correspondence not derivable from ``dtype_table!``,
#: so it is written out here and nowhere else.
_CAST_ONLY_NAMES: dict[str, str] = {"boolean": "u8"}


def dtype_name_for(dtype: pl.DataType) -> str:
    """Name *dtype* the way ``source(dtype=)`` and ``cast()`` take it.

    A Polars type spells itself the numpy way (``Float32`` -> ``float32``);
    the engine spells itself the short way (``f32``). This is that hop, and it
    reads the correspondence from :data:`NUMPY_TO_SHORT`, generated from
    ``dtype_table!``. Callers building a pipeline source for a column whose
    dtype they only learn at plan time should use this rather than keeping
    their own table -- which is what the metrics contour matcher did, guarded
    by a test that reached into a private to compare it against the Rust.

    Args:
        dtype: A Polars leaf type. Nested ``List``/``Array`` types are not
            unwrapped; pass the element type.

    Returns:
        The engine's short name, e.g. ``"f32"``.

    Raises:
        ValueError: If *dtype* has no meaningful buffer representation. It does
            not fall back: an unmappable type used to become an ``f32`` source
            and fail later and deeper with a cast error.
    """
    name = getattr(dtype, "__name__", None) or str(dtype)
    key = name.lower()
    if key in _CAST_ONLY_NAMES:
        return _CAST_ONLY_NAMES[key]
    try:
        return NUMPY_TO_SHORT[key]
    except KeyError:
        # Name the *Polars* types, which is what the caller supplied and can
        # change. Listing the engine's names instead says nothing actionable,
        # and repeats "u8" twice since two Polars types map onto it. The
        # spellings are read back off Polars rather than written out here, so
        # this stays a lookup and not a fourth copy of the dtype vocabulary.
        known = {*NUMPY_TO_SHORT, *_CAST_ONLY_NAMES}
        accepted = ", ".join(sorted(n for n in dir(pl) if n.lower() in known))
        raise ValueError(
            f"{dtype} has no meaningful buffer representation. "
            f"Expected one of: {accepted}."
        ) from None


# ImageNet normalization constants
# These are the standard normalization values computed from the ImageNet dataset.
# Use with: normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


#: Maps an expression parameter to the plugin input it binds to.
SlotOf = Callable[[pl.Expr], int]


class SlotTable:
    """The plugin's input columns, in order: the one authority for which input
    an expression parameter binds to.

    Expressions are identified by ``Expr.meta.eq``, never by display text:
    ``str(expr)`` is not an identity (every ``pl.lit(pl.Series(...))`` prints
    alike, as do two Python UDFs — CR-31), and the text-keyed registry that
    papered over that made the graph JSON depend on which *other* expressions
    were alive. A graph builds one table (root columns first, then every
    expression parameter) and serializes each parameter as its position.
    """

    def __init__(self) -> None:
        self._exprs: list[pl.Expr] = []

    def _find(self, expr: pl.Expr) -> int | None:
        for i, known in enumerate(self._exprs):
            if known is expr or known.meta.eq(expr):
                return i
        return None

    def add(self, expr: pl.Expr) -> int:
        """Register *expr* (once, by ``meta.eq``) and return its position."""
        found = self._find(expr)
        if found is not None:
            return found
        self._exprs.append(expr)
        return len(self._exprs) - 1

    def index(self, expr: pl.Expr) -> int:
        """The position of an already-registered *expr*.

        Raises rather than appending: an expression parameter that was never
        registered as a plugin input is a builder bug, and binding it to some
        other column would be a silent wrong answer.
        """
        found = self._find(expr)
        if found is None:
            msg = (
                f"expression {expr} is not a registered plugin input; builders "
                "must register expression parameters via Pipeline._slot"
            )
            raise KeyError(msg)
        return found

    @property
    def columns(self) -> list[pl.Expr]:
        return list(self._exprs)

    def __len__(self) -> int:
        return len(self._exprs)


def _to_python(value: Any) -> Any:
    """A numpy scalar as the Python number it holds; anything else unchanged.

    ``json`` cannot serialize ``np.int64`` (and would silently accept
    ``np.float64`` only because it subclasses ``float``), so a typed field
    converts at the one place values reach the wire.
    """
    if type(value).__module__ == "numpy" and callable(getattr(value, "item", None)):
        return value.item()
    return value


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
