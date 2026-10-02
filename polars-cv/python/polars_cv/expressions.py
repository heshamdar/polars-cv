"""
Polars expression integration for polars-cv.

This module provides the expression registration and namespace for
applying vision pipelines to Polars DataFrame columns.

All pipelines are converted to graph representation and executed via
the unified vb_graph function. Single-output pipelines return Binary,
multi-output pipelines return Struct.

Additionally, lightweight metadata expressions (width, height, channels,
image_dtype) are available directly on the ``.cv`` namespace without
constructing a full Pipeline. These use header-only decoding. ``read_bytes``
sits alongside them: it reads a path column's bytes without decoding at all,
so originals can be passed through unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from polars_cv._namespace import _PluginNamespace
from polars_cv._types import FetchErrorPolicy, normalize_cloud_options

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars_cv._types import CloudOptions
    from polars_cv.lazy import LazyPipelineExpr
    from polars_cv.pipeline import Pipeline


@pl.api.register_expr_namespace("cv")
class CvNamespace(_PluginNamespace):
    """
    Namespace for computer vision operations on Polars expressions.

    Example:
        >>> pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
        >>> expr = pl.col("image").cv.pipe(pipe).sink("numpy")
        >>> df.with_columns(processed=expr)

    Metadata methods (header-only, no full decode):
        >>> df.with_columns(w=pl.col("image").cv.width())
        >>> df.filter(pl.col("image").cv.height() > 1024)
    """

    def pipe(self, pipe: "Pipeline") -> "LazyPipelineExpr":
        """
        Apply a vision pipeline to this column.

        Returns a LazyPipelineExpr that can be composed with other operations.
        Call .sink(format) to finalize and get a Polars expression.
        """
        from polars_cv.lazy import LazyPipelineExpr

        return LazyPipelineExpr(
            column=self._expr,
            pipeline=pipe,
            # Ops referencing other nodes (rasterize(shape=...)) make those
            # nodes upstream dependencies so they execute first.
            upstream=list(pipe._node_refs),
        )

    # ------------------------------------------------------------------
    # Byte access (no decode)
    # ------------------------------------------------------------------

    def read_bytes(
        self,
        *,
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        on_error: str = "raise",
        allowed_roots: "Sequence[str] | None" = None,
    ) -> pl.Expr:
        """
        Read the bytes each path names, without decoding them.

        This is the first half of the ``"file_path"`` source: that source
        fetches a path's bytes and then decodes them as an image, and this
        stops after the fetch. Bytes are returned verbatim, so an encoded file
        survives the round trip unchanged and can be written back
        byte-for-byte — something a decode cannot offer, since re-encoding a
        decoded JPEG never reproduces the original file and the image sinks
        carry no EXIF/ICC metadata.

        The header-only metadata methods below take a path column directly
        (a local file read only as far as its header needs), so they do not
        need this first.

        Local paths (bare or ``file://``) and remote URIs (``s3://``, ``gs://``,
        ``az://``, ``http://``) are both supported, with the same credential
        handling as ``source("file_path")``. Within one call the distinct
        remote paths are fetched concurrently; local files are read per row.

        Under ``engine="streaming"`` a bytes column produced here is
        morsel-bounded, so it only becomes corpus-resident if you select it in
        the final projection — which is the point when you want the originals.

        Note:
            Without ``allowed_roots`` paths are not sandboxed: whatever the
            column names is read in full, including any local file and any
            ``http://`` address (link-local metadata endpoints among them).
            That is the right default for your own paths and the wrong one for
            paths that came from somewhere else. File size is not capped
            either way.

        Args:
            cloud_options: Credentials/settings for remote reads, as
                ``CloudOptions`` or a dict (see ``Pipeline.source``).
            on_error: ``"raise"`` (default) fails the query on the first
                unreadable path; ``"null"`` yields null for that row only.
            allowed_roots: Restrict which locations may be read, exactly as
                ``Pipeline.source(allowed_roots=...)`` does — one list covering
                local directories and remote URI prefixes, canonicalized and
                matched component-wise, denying anything that matches no entry.
                A refusal follows ``on_error``, so ``on_error="null"`` nulls
                the offending rows rather than failing the query.

        Returns:
            Binary expression with each path's raw file contents.
        """
        valid = tuple(p.value for p in FetchErrorPolicy)
        if on_error not in valid:
            msg = (
                f"Unknown on_error value {on_error!r} for read_bytes() "
                f"(expected one of {valid})"
            )
            raise ValueError(msg)

        kwargs: dict[str, Any] = {"on_error": on_error}
        opts = normalize_cloud_options(cloud_options)
        if opts is not None:
            kwargs["cloud_options"] = opts.to_dict()
        if allowed_roots is not None:
            kwargs["allowed_roots"] = list(allowed_roots)
        return self._plugin("read_file_bytes", kwargs=kwargs)

    # ------------------------------------------------------------------
    # Header-only metadata expressions
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_kwargs(
        cloud_options: "CloudOptions | dict[str, Any] | None",
        on_error: str | None,
        allowed_roots: "Sequence[str] | None",
    ) -> dict[str, Any]:
        """A header-metadata function's kwargs. Every key is sent (an empty kwargs
        map does not reach the plugin); an unset one is ``None``, which the
        plugin reads as absent — it refuses a given path option on bytes."""
        kwargs: dict[str, Any] = {
            "on_error": None,
            "cloud_options": None,
            "allowed_roots": None,
        }
        if on_error is not None:
            valid = tuple(p.value for p in FetchErrorPolicy)
            if on_error not in valid:
                msg = f"Unknown on_error value {on_error!r} (expected one of {valid})"
                raise ValueError(msg)
            kwargs["on_error"] = on_error
        opts = normalize_cloud_options(cloud_options)
        if opts is not None:
            kwargs["cloud_options"] = opts.to_dict()
        if allowed_roots is not None:
            kwargs["allowed_roots"] = list(allowed_roots)
        return kwargs

    def width(
        self,
        *,
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        on_error: str | None = None,
        allowed_roots: "Sequence[str] | None" = None,
    ) -> pl.Expr:
        """
        Get image width from its header, without a full decode.

        The column holds image bytes (``Binary``) or paths (``String``). A
        local path's file is read only as far as its header needs; a remote
        path's object is fetched whole, as ``.cv.read_bytes()`` does. Supports
        encoded images (PNG, JPEG, WebP, TIFF, BMP, GIF) and VIEW protocol
        blobs; ``null`` for an unrecognised format or a null input. For more
        than one field, ``.cv.image_info()`` reads each header once.

        Args:
            cloud_options: Path columns only: credentials for remote reads,
                as for ``.cv.read_bytes()``.
            on_error: Path columns only: ``"raise"`` (default) or ``"null"``
                for an unreadable path.
            allowed_roots: Path columns only: the locations that may be read,
                as for ``.cv.read_bytes()``.

        Returns:
            UInt32 expression with the width of each image.
        """
        return self._plugin(
            "image_width",
            kwargs=self._metadata_kwargs(cloud_options, on_error, allowed_roots),
        )

    def height(
        self,
        *,
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        on_error: str | None = None,
        allowed_roots: "Sequence[str] | None" = None,
    ) -> pl.Expr:
        """
        Get image height from its header, without a full decode.

        Takes bytes or paths, as :meth:`width` does (see it for the options).

        Returns:
            UInt32 expression with the height of each image.
        """
        return self._plugin(
            "image_height",
            kwargs=self._metadata_kwargs(cloud_options, on_error, allowed_roots),
        )

    def channels(
        self,
        *,
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        on_error: str | None = None,
        allowed_roots: "Sequence[str] | None" = None,
    ) -> pl.Expr:
        """
        Get the number of channels from its header, without a full decode.

        Takes bytes or paths, as :meth:`width` does (see it for the options).

        Returns:
            UInt32 expression with the channel count of each image.
        """
        return self._plugin(
            "image_channels",
            kwargs=self._metadata_kwargs(cloud_options, on_error, allowed_roots),
        )

    def image_dtype(
        self,
        *,
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        on_error: str | None = None,
        allowed_roots: "Sequence[str] | None" = None,
    ) -> pl.Expr:
        """
        Get the element dtype from its header, without a full decode.

        Returns dtype names like ``"uint8"``, ``"uint16"``, ``"float32"``.
        Takes bytes or paths, as :meth:`width` does (see it for the options).

        Returns:
            String expression with the dtype name of each image.
        """
        return self._plugin(
            "image_dtype",
            kwargs=self._metadata_kwargs(cloud_options, on_error, allowed_roots),
        )

    def image_info(
        self,
        *,
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        on_error: str | None = None,
        allowed_roots: "Sequence[str] | None" = None,
    ) -> pl.Expr:
        """
        Width, height, channels and dtype from one header read per row.

        Takes bytes or paths, as :meth:`width` does (see it for the options).

        Returns:
            ``Struct{width: UInt32, height: UInt32, channels: UInt32,
            dtype: String}``; null for a null input or an unrecognised format.
        """
        return self._plugin(
            "image_info",
            kwargs=self._metadata_kwargs(cloud_options, on_error, allowed_roots),
        )
