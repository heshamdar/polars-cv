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

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars_cv.lazy import LazyPipelineExpr
    from polars_cv.pipeline import Pipeline

LIB_PATH = Path(__file__).parent


@pl.api.register_expr_namespace("cv")
class CvNamespace:
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

    def __init__(self, expr: pl.Expr) -> None:
        """
        Initialize the namespace with an expression.

        Args:
            expr: The Polars expression to extend.
        """
        self._expr = expr

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
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="image_width",
            args=[self._expr],
            is_elementwise=True,
        )

    def height(self) -> pl.Expr:
        """
        Get image height from a binary column (header-only, no full decode).

        Supports encoded images (PNG, JPEG, WebP, TIFF, BMP, GIF) and
        VIEW protocol blobs. Returns ``null`` for unrecognised formats or
        null inputs.

        Returns:
            UInt32 expression with the height of each image.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="image_height",
            args=[self._expr],
            is_elementwise=True,
        )

    def channels(self) -> pl.Expr:
        """
        Get number of channels from a binary column (header-only, no full decode).

        Supports encoded images (PNG, JPEG, WebP, TIFF, BMP, GIF) and
        VIEW protocol blobs. Returns ``null`` for unrecognised formats or
        null inputs.

        Returns:
            UInt32 expression with the channel count of each image.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="image_channels",
            args=[self._expr],
            is_elementwise=True,
        )

    def image_dtype(self) -> pl.Expr:
        """
        Get element dtype from a binary column (header-only, no full decode).

        Returns dtype names like ``"uint8"``, ``"uint16"``, ``"float32"``,
        etc.  Supports encoded images and VIEW protocol blobs. Returns
        ``null`` for unrecognised formats or null inputs.

        Returns:
            String expression with the dtype name of each image.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="image_dtype",
            args=[self._expr],
            is_elementwise=True,
        )
