"""
Polars expression integration for polars-cv.

This module provides the expression registration and namespace for
applying vision pipelines to Polars DataFrame columns.

All pipelines are converted to graph representation and executed via
the unified vb_graph function. Single-output pipelines return Binary,
multi-output pipelines return Struct.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars_cv.lazy import LazyPipelineExpr
    from polars_cv.pipeline import Pipeline


@pl.api.register_expr_namespace("cv")
class CvNamespace:
    """
    Namespace for computer vision operations on Polars expressions.

    Example:
        >>> pipe = Pipeline().source("image_bytes").resize(100, 200)
        >>> expr = pl.col("image").cv.pipe(pipe).sink("numpy")
        >>> df.with_columns(processed=expr)
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
