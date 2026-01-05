# Lazy vs Eager Execution

polars-cv supports two execution modes: **eager** and **lazy**. Understanding when to use each is key to getting the most out of the library.

## Eager Mode: `.cv.pipeline()`

Use eager mode for simple, single-output pipelines:

```python
import polars as pl
from polars_cv import Pipeline

# Pipeline must include a sink
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .sink("png")  # Sink is required
)

# Apply eagerly - returns pl.Expr directly
result = df.with_columns(
    resized=pl.col("image").cv.pipeline(pipe)
)
```

**Characteristics:**
- Pipeline must have a `.sink()` call
- Returns a Polars expression directly
- Simple to use for straightforward transformations
- Cannot compose multiple pipelines

## Lazy Mode: `.cv.pipe()`

Use lazy mode for composition, multi-output, and complex pipelines:

```python
# Pipeline WITHOUT sink
pipe = Pipeline().source("image_bytes").resize(height=224, width=224)

# Apply lazily - returns LazyPipelineExpr
lazy_expr = pl.col("image").cv.pipe(pipe)

# Materialize with .sink()
result = df.with_columns(
    resized=lazy_expr.sink("png")
)
```

**Characteristics:**
- Pipeline should NOT have a `.sink()` call
- Returns `LazyPipelineExpr` for composition
- Call `.sink()` when ready to materialize
- Enables chaining, multi-output, and binary operations

## Why Lazy Mode?

### 1. Pipeline Chaining

Chain additional operations onto existing pipelines:

```python
# Base pipeline
base = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(height=128, width=128)
)

# Chain additional operations
gray = base.pipe(Pipeline().grayscale())
thresh = gray.pipe(Pipeline().threshold(128))

# Execute
result = df.with_columns(binary=thresh.sink("png"))
```

### 2. Multi-Output with Aliases

Extract multiple intermediate results:

```python
# Create pipeline with named checkpoints
base = (
    pl.col("image")
    .cv.pipe(Pipeline().source("image_bytes").resize(128, 128))
    .alias("resized")
)
gray = base.pipe(Pipeline().grayscale()).alias("gray")
thresh = gray.pipe(Pipeline().threshold(128)).alias("thresh")

# Merge and sink multiple outputs
merged = thresh.merge_pipe(base)  # Include earlier stages
result = df.with_columns(
    outputs=merged.sink({
        "resized": "png",
        "gray": "png",
        "thresh": "png",
    })
)

# Extract individual outputs
resized = result.select(pl.col("outputs").struct.field("resized"))
```

### 3. Binary Operations

Combine two pipelines element-wise:

```python
img1 = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(128, 128)
)
img2 = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(128, 128).blur(5.0)
)

# Element-wise operations
added = img1.add(img2)
blended = img1.blend(img2)
diff = img1.subtract(img2)

result = df.with_columns(difference=diff.sink("png"))
```

### 4. Mask Application

Apply masks from one pipeline to another:

```python
# Image pipeline
img = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(128, 128)
)

# Mask pipeline (from contour)
mask = pl.col("contour").cv.pipe(
    Pipeline().source("contour", shape=img)  # Infer dimensions
)

# Apply mask
masked = img.apply_mask(mask)
result = df.with_columns(masked=masked.sink("png"))
```

## Common Subexpression Elimination (CSE)

When using lazy mode with multi-output, polars-cv automatically optimizes shared operations:

```python
# Both branches share: resize → grayscale
base = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(100, 100)
)
gray = base.pipe(Pipeline().grayscale()).alias("gray")

# Branch 1: blur from gray
blur = gray.pipe(Pipeline().blur(2.0)).alias("blur")

# Branch 2: threshold from gray
thresh = gray.pipe(Pipeline().threshold(128)).alias("thresh")

# Merge - CSE automatically shares the gray computation
merged = blur.merge_pipe(thresh)
result = df.with_columns(
    outputs=merged.sink({"gray": "png", "blur": "png", "thresh": "png"})
)
```

## Decision Guide

| Use Case | Mode | Why |
|----------|------|-----|
| Simple transformation | Eager | Simpler API |
| Multiple outputs | Lazy | Need `.alias()` and multi-sink |
| Pipeline chaining | Lazy | Need `.pipe()` for chaining |
| Binary operations | Lazy | Need both operands as `LazyPipelineExpr` |
| Mask application | Lazy | Need `apply_mask()` |
| Shared computations | Lazy | CSE optimization |

## Next Steps

- [Multi-Output](../composition/multi-output.md) - Deep dive into multi-output pipelines
- [Binary Operations](../composition/binary-ops.md) - Element-wise operations

