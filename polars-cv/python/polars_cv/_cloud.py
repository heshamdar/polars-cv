"""Checking cloud credentials before any row is read."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._types import normalize_cloud_options

if TYPE_CHECKING:
    from ._types import CloudOptions


class CloudCredentialError(RuntimeError):
    """A remote path could not be reached with the credential in effect.

    The message names that credential (``gcloud ADC (...)``,
    ``token_command [...]``, ``object_store default credential chain (...)``)
    and the backend's own error.
    """


def check_cloud_access(
    path: str,
    *,
    cloud_options: CloudOptions | dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Resolve the credential a read of ``path`` would use and exercise it once.

    Takes the same ``cloud_options`` as ``source("file_path")`` and
    ``.cv.read_bytes()``, and builds the store the same way, so a pass here
    means a pipeline over that bucket starts with working credentials and a
    failure is reported with the credential named rather than as an
    object-store error on the first row.

    An object path is checked with a ``HEAD``; a prefix (a path ending in
    ``/``) with a one-level listing. A missing object is not a failure: the
    credential was accepted, and ``status`` says ``"not_found"``.

    Args:
        path: A remote object or prefix (``gs://``, ``s3://``, ``az://``,
            ``abfs[s]://``).
        cloud_options: Credentials, as for ``source("file_path")``. With none,
            the ambient credentials apply (including a federated gcloud ADC).

    Returns:
        ``{"backend", "credential_source", "status"}``, ``status`` being
        ``"ok"`` or ``"not_found"``.

    Raises:
        CloudCredentialError: The credential could not be obtained, or the
            backend refused it.
        ValueError: ``path`` is not a remote object-store path, or an option
            is invalid.

    Examples:
        >>> polars_cv.check_cloud_access("gs://bucket/prefix/")  # doctest: +SKIP
        {'backend': 'gcp', 'credential_source': 'gcloud ADC (...)', 'status': 'ok'}
    """
    from . import _lib

    opts = normalize_cloud_options(cloud_options)
    report = _lib._check_cloud_access(path, opts.to_dict() if opts else None)
    source = report["credential_source"] or ""
    if report["status"] == "error":
        error = report["error"] or ""
        # A bespoke credential's own failure already names it.
        named = "" if source in error else f" with {source}"
        msg = f"cannot access {path}{named}: {error}"
        raise CloudCredentialError(msg)
    return {k: v for k, v in report.items() if k != "error" and v is not None}
