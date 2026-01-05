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

## API Reference

::: polars_cv.LazyPipelineExpr
    options:
      members:
        - pipe
        - alias
        - sink
        - merge_pipe
        - add
        - subtract
        - multiply
        - divide
        - blend
        - apply_mask
        - apply_contour_mask
      show_root_heading: true
      show_source: false
      heading_level: 3

