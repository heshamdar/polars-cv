"""
Polars expression integration for polars-cv.

This module provides the expression registration and namespace for
applying vision pipelines to Polars DataFrame columns.

All pipelines are converted to graph representation and executed via
the unified vb_graph function. Single-output pipelines return Binary,
multi-output pipelines return Struct.

Additionally, lightweight metadata expressions (width, height, channels,
image_dtype) are available directly on the ``.cv`` namespace without
constructing a full Pipeline. These use header-only decoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_cv._namespace import _PluginNamespace

if TYPE_CHECKING:
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
