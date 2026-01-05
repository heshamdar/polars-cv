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
    from polars._typing import IntoExpr

    from polars_cv.lazy import LazyPipelineExpr
    from polars_cv.pipeline import Pipeline


def apply_pipeline(expr: "IntoExpr", pipe: "Pipeline") -> pl.Expr:
    """
    Apply a vision pipeline to a column expression.

    All pipelines are converted to graph representation and executed
    through the unified graph executor (vb_graph / vb_graph_multi).

    This handles:
    - Standard binary sources (image_bytes, blob, etc.)
    - Contour sources (struct-to-binary conversion via rasterization)
    - Expression arguments (dynamic parameters from other columns)
    - Single and multi-output pipelines

    Args:
        expr: The input column expression containing the data.
        pipe: The Pipeline instance defining the operations.

    Returns:
        A Polars expression that will execute the pipeline.
        - For single output: Binary column
        - For multi-output: Struct column with named Binary fields

    Example:
        >>> from polars_cv import Pipeline
        >>> pipe = Pipeline().source("image_bytes").resize(height=224, width=224).sink("numpy")
        >>> result = df.with_columns(processed=apply_pipeline(pl.col("images"), pipe))
        >>>
        >>> # Multi-output example
        >>> pipe = (Pipeline()
        ...     .source("image_bytes")
        ...     .alias("original")
        ...     .resize(height=128, width=128)
        ...     .alias("resized")
        ... ).sink({"original": "png", "resized": "numpy"})
        >>> result = df.with_columns(outputs=apply_pipeline(pl.col("images"), pipe))
    """
    # Validate the pipeline before conversion
    pipe.validate()

    # Ensure expr is a pl.Expr
    if not isinstance(expr, pl.Expr):
        if isinstance(expr, str):
            expr = pl.col(expr)
        else:
            # For other IntoExpr types, wrap in lit or handle appropriately
            msg = f"Expected pl.Expr or column name string, got {type(expr)}"
            raise TypeError(msg)

    # Convert pipeline to graph representation
    # This is the unified path for all pipeline types
    graph = pipe.to_graph(column=expr)

    # Single output - the terminal node alias is "_output"
    assert pipe._sink is not None
    sink_params = {}
    if pipe._sink.quality != 85:
        sink_params["quality"] = pipe._sink.quality
    if pipe._sink.shape is not None:
        sink_params["shape"] = pipe._sink.shape
    graph.set_output("_output", pipe._sink.format.value, **sink_params)

    # Convert to expression via graph
    return graph.to_expr()


@pl.api.register_expr_namespace("cv")
class CvNamespace:
    """
    Polars expression namespace for computer vision operations.

    This namespace provides the `.cv` accessor on Polars expressions
    for applying vision pipelines.

    Two patterns are supported:

    1. **Direct pipeline (eager)**: `.cv.pipeline(pipe)` returns a `pl.Expr` directly.
       Requires the pipeline to have a sink defined.

    2. **Composable pipeline (lazy)**: `.cv.pipe(pipe)` returns a `LazyPipelineExpr`
       that can be composed with other lazy expressions before calling `.sink()`.

    Example (eager):
        >>> pipe = Pipeline().source("image_bytes").resize(height=224, width=224).sink("numpy")
        >>> result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

    Example (composable):
        >>> img_pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
        >>> img = pl.col("image").cv.pipe(img_pipe)  # LazyPipelineExpr
        >>> expr = img.sink("numpy")  # Now a pl.Expr
        >>> result = df.with_columns(processed=expr)
    """

    def __init__(self, expr: pl.Expr) -> None:
        """
        Initialize the namespace with an expression.

        Args:
            expr: The Polars expression to extend.
        """
        self._expr = expr

    def pipeline(self, pipe: "Pipeline") -> pl.Expr:
        """
        Apply a vision pipeline to this column (eager mode).

        The pipeline must have a sink defined. Returns a Polars expression
        directly, suitable for single-pipeline use cases.

        Args:
            pipe: The Pipeline instance with source and sink defined.

        Returns:
            A Polars expression that will execute the pipeline on each element.

        Raises:
            ValueError: If pipeline has no sink. Use `.cv.pipe()` instead
                for composable pipelines without a sink.

        Example:
            ```python
            >>> pipe = (
            ...     Pipeline()
            ...     .source("image_bytes")
            ...     .resize(height=224, width=224)
            ...     .grayscale()
            ...     .sink("numpy")
            ... )
            >>> df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
            ```
        """
        # Validate that sink is present with a helpful error message
        if not pipe.has_sink():
            msg = (
                "Pipeline must have a sink for eager execution with .cv.pipeline(). "
                "Either:\n"
                "  1. Add a sink to the pipeline: pipe.sink('numpy')\n"
                "  2. Use .cv.pipe() for composable pipelines: "
                "pl.col('x').cv.pipe(pipe).sink('numpy')"
            )
            raise ValueError(msg)

        return apply_pipeline(self._expr, pipe)

    def pipe(self, pipe: "Pipeline") -> "LazyPipelineExpr":
        """
        Create a composable lazy pipeline expression (lazy mode).

        Returns a LazyPipelineExpr that can be composed with other lazy
        expressions using methods like `.apply_mask()`, `.add()`, etc.
        Call `.sink(format)` to finalize and get a pl.Expr.

        The pipeline should NOT have a sink defined (sink is specified
        when calling `.sink()` on the returned LazyPipelineExpr).

        Args:
            pipe: The Pipeline instance with source defined (no sink).

        Returns:
            A LazyPipelineExpr that can be composed before execution.

        Raises:
            ValueError: If pipeline has a sink defined. Use `.cv.pipeline()`
                instead for direct execution.

        Example:
            ```python
            >>> img_pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
            >>> mask_pipe = Pipeline().source("contour")
            >>>
            >>> img = pl.col("image").cv.pipe(img_pipe)
            >>> mask = pl.col("contour").cv.pipe(mask_pipe)
            >>>
            >>> result = img.apply_contour_mask(mask).sink("numpy")
            >>> df.with_columns(masked=result)
            ```
        """
        from polars_cv.lazy import LazyPipelineExpr

        # Warn if pipeline has a sink (it will be overwritten by .sink() call)
        if pipe.has_sink():
            import warnings

            warnings.warn(
                "Pipeline has a sink defined, but .cv.pipe() is for composable "
                "pipelines. The sink will be overridden when you call .sink() "
                "on the returned LazyPipelineExpr. Consider using .cv.pipeline() "
                "instead for direct execution.",
                UserWarning,
                stacklevel=2,
            )

        return LazyPipelineExpr(
            column=self._expr,
            pipeline=pipe,
        )
