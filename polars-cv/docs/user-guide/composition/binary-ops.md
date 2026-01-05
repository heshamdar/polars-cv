# Binary Operations

Combine two pipelines element-wise for operations like blending, differencing, and masking.

## Overview

Binary operations take two `LazyPipelineExpr` operands and produce a new one:

```python
result = expr_a.add(expr_b)
result = expr_a.blend(expr_b)
result = expr_a.apply_mask(expr_b)
```

## Available Operations

| Operation | Description | U8 Behavior | Float Behavior |
|-----------|-------------|-------------|----------------|
| `add` | Element-wise addition | Saturating (max 255) | Standard |
| `subtract` | Element-wise subtraction | Saturating (min 0) | Standard |
| `multiply` | Element-wise multiplication | Saturating | Standard |
| `divide` | Element-wise division | Integer division | Standard |
| `blend` | Normalized blend | `(a/255)*(b/255)*255` | Standard |
| `ratio` | Scaled ratio | `(a/b)*255` | Standard |

## Basic Usage

```python
import polars as pl
from polars_cv import Pipeline

# Two pipelines processing the same image differently
img1 = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(128, 128)
)
img2 = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(128, 128).blur(5.0)
)

# Binary operations
added = img1.add(img2)
subtracted = img1.subtract(img2)  # Edge detection effect
blended = img1.blend(img2)

result = df.with_columns(
    diff=subtracted.sink("png")
)
```

## Mask Application

Apply a binary mask to an image:

```python
# Image pipeline
img = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(128, 128)
)

# Mask pipeline (from contour)
mask = pl.col("contour").cv.pipe(
    Pipeline().source("contour", width=128, height=128)
)

# Apply mask: pixels where mask=0 become 0
masked = img.apply_mask(mask)
result = df.with_columns(masked=masked.sink("png"))
```

### Shape Inference

Automatically match mask dimensions to image:

```python
# Define image pipeline
img = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(150, 100)  # Non-square
)

# Contour source with shape inference
mask = pl.col("contour").cv.pipe(
    Pipeline().source("contour", shape=img)  # Auto: 150x100
)

# Apply
masked = img.apply_mask(mask)
```

### Convenience Method

Use `apply_contour_mask()` for the common case:

```python
img = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(128, 128)
)
contour = pl.col("contour").cv.pipe(
    Pipeline().source("contour", shape=img)
)

# Convenience method handles shape inference
masked = img.apply_contour_mask(contour)
```

## Multi-Source Operations

Combine data from different DataFrame columns:

```python
# Base image
base = pl.col("base_image").cv.pipe(
    Pipeline().source("image_bytes").resize(128, 128)
)

# Overlay image (different column)
overlay = pl.col("overlay_image").cv.pipe(
    Pipeline().source("image_bytes").resize(128, 128).flip_h()
)

# Blend two different images
blended = base.blend(overlay)
result = df.with_columns(blended=blended.sink("png"))
```

## Chaining Binary Operations

Binary operations return `LazyPipelineExpr`, so you can chain:

```python
# (A + B) * mask
result = img_a.add(img_b).apply_mask(mask)

# Blend A with blurred version, then apply mask
blurred = img_a.pipe(Pipeline().blur(5.0))
result = img_a.blend(blurred).apply_mask(mask)
```

## Use Cases

### Edge Detection

Subtract blurred from original to find edges:

```python
original = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").grayscale()
)
blurred = original.pipe(Pipeline().blur(3.0))

edges = original.subtract(blurred)  # Saturating subtraction
```

### Overlay Composition

Blend two images with alpha:

```python
# For alpha blending, normalize inputs first
img1_norm = pl.col("image1").cv.pipe(
    Pipeline().source("image_bytes").normalize()
)
img2_norm = pl.col("image2").cv.pipe(
    Pipeline().source("image_bytes").normalize()
)

# Blend produces (a * b) effect on normalized inputs
blended = img1_norm.blend(img2_norm)
```

### Attention/Mask Weighting

Apply learned attention weights:

```python
feature_map = pl.col("features").cv.pipe(...)
attention = pl.col("attention").cv.pipe(...)

# Multiply features by attention weights
weighted = feature_map.multiply(attention)
```

## Requirements

Both operands must:

1. Be `LazyPipelineExpr` objects (use `.cv.pipe()`)
2. Produce the same output shape
3. Have compatible dtypes

## Next Steps

- [Multi-Output](multi-output.md) - Extract intermediate results
- [Domains](../concepts/domains.md) - Mask from contours

