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
from polars_cv._types import normalize_cloud_options

if TYPE_CHECKING:
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
            upstream=list(pipe._shape_refs),
        )

    # ------------------------------------------------------------------
    # Byte access (no decode)
    # ------------------------------------------------------------------

    def read_bytes(
        self,
        *,
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        on_error: str = "raise",
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

        It also lets the header-only metadata methods below reach remote
        files, since they take binary columns::

            >>> raw = pl.col("path").cv.read_bytes()
            >>> df.with_columns(w=raw.cv.width())

        Local paths (bare or ``file://``) and remote URIs (``s3://``, ``gs://``,
        ``az://``, ``http://``) are both supported, with the same credential
        handling as ``source("file_path")``. Within one call the distinct
        remote paths are fetched concurrently; local files are read per row.

        Under ``engine="streaming"`` a bytes column produced here is
        morsel-bounded, so it only becomes corpus-resident if you select it in
        the final projection — which is the point when you want the originals.

        Note:
            Paths are not sandboxed and file size is not capped: whatever the
            column names is read in full, including any local file and any
            ``http://`` address (link-local metadata endpoints among them). Use
            with trusted path columns only.

        Args:
            cloud_options: Credentials/settings for remote reads, as
                ``CloudOptions`` or a dict (see ``Pipeline.source``).
            on_error: ``"raise"`` (default) fails the query on the first
                unreadable path; ``"null"`` yields null for that row only.

        Returns:
            Binary expression with each path's raw file contents.
        """
        if on_error not in ("raise", "null"):
            msg = (
                f"Unknown on_error value {on_error!r} for read_bytes() "
                f"(expected 'raise' or 'null')"
            )
            raise ValueError(msg)

        kwargs: dict[str, Any] = {"on_error": on_error}
        opts = normalize_cloud_options(cloud_options)
        if opts is not None:
            kwargs["cloud_options"] = opts.to_dict()
        return self._plugin("read_file_bytes", kwargs=kwargs)

    # ------------------------------------------------------------------
    # Header-only metadata expressions
    # ------------------------------------------------------------------

    def width(self) -> pl.Expr:
        """
        Get image width from a binary column (header-only, no full decode).

        Supports encoded images (PNG, JPEG, WebP, TIFF, BMP, GIF) and
        VIEW protocol blobs. Returns ``null`` for unrecognised formats or
        null inputs.

        Returns:
            UInt32 expression with the width of each image.
        """
        return self._plugin("image_width")

    def height(self) -> pl.Expr:
        """
        Get image height from a binary column (header-only, no full decode).

        Supports encoded images (PNG, JPEG, WebP, TIFF, BMP, GIF) and
        VIEW protocol blobs. Returns ``null`` for unrecognised formats or
        null inputs.

        Returns:
            UInt32 expression with the height of each image.
        """
        return self._plugin("image_height")

    def channels(self) -> pl.Expr:
        """
        Get number of channels from a binary column (header-only, no full decode).

        Supports encoded images (PNG, JPEG, WebP, TIFF, BMP, GIF) and
        VIEW protocol blobs. Returns ``null`` for unrecognised formats or
        null inputs.

        Returns:
            UInt32 expression with the channel count of each image.
        """
        return self._plugin("image_channels")

    def image_dtype(self) -> pl.Expr:
        """
        Get element dtype from a binary column (header-only, no full decode).

        Returns dtype names like ``"uint8"``, ``"uint16"``, ``"float32"``,
        etc.  Supports encoded images and VIEW protocol blobs. Returns
        ``null`` for unrecognised formats or null inputs.

        Returns:
            String expression with the dtype name of each image.
        """
        return self._plugin("image_dtype")
