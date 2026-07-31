# LazyPipelineExpr

The `LazyPipelineExpr` class enables composable, lazy pipeline expressions.

## Overview

```python
import polars as pl
from polars_cv import Pipeline

# Create lazy expression
expr = pl.col("image").cv.pipe(Pipeline().source("image_bytes"))

# Chain operations
gray = expr.pipe(Pipeline().grayscale())

# Materialize with sink
result = df.with_columns(output=gray.sink("png"))
```

## Mirrored `Pipeline` methods

Every operation and policy setter on [`Pipeline`](pipeline.md) — `resize()`,
`grayscale()`, `on_error()`, `on_null_param()` and the rest — is generated onto
`LazyPipelineExpr` at import time, so it can be chained directly after
`.pipe(...)`:

```python
expr = pl.col("image").cv.pipe(pipe).resize(height=pl.col("h")).on_null_param("null")
```

Because those methods are generated rather than written out, they do not appear
in the reference below; their signatures live in `polars_cv/lazy.pyi` (which
your editor reads) and their documentation on the
[`Pipeline`](pipeline.md) page. What is listed below is the hand-written surface
— composition (`pipe()`, `merge_pipe()`), sinks, the binary operators, and the
few operations that take another `LazyPipelineExpr` as an operand.

## API Reference

::: polars_cv.LazyPipelineExpr
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
