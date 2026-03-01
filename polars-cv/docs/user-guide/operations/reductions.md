# Reductions

Reduction operations transform an entire buffer or a specific axis into a single numeric result.

## Global Reductions

Global reductions return a single scalar value per row.

```python
Pipeline().source("image_bytes").grayscale().reduce_sum()
Pipeline().source("image_bytes").grayscale().reduce_mean()
Pipeline().source("image_bytes").grayscale().reduce_max()
Pipeline().source("image_bytes").grayscale().reduce_min()
Pipeline().source("image_bytes").grayscale().reduce_std()
```

## Axis Reductions

Reducing along an axis decreases the dimensionality of the array.

```python
# Compute per-channel mean (axis 2 for HWC images)
Pipeline().source("image_bytes").reduce_mean(axis=2)

# Compute max value for each row of pixels (axis 1)
Pipeline().source("image_bytes").grayscale().reduce_max(axis=1)
```

## Argmax / Argmin

Find the index of the maximum or minimum value along an axis.

```python
# Find the column index with the highest intensity per row
Pipeline().source("image_bytes").grayscale().reduce_argmax(axis=1)

# Find the column index with the lowest intensity per row
Pipeline().source("image_bytes").grayscale().reduce_argmin(axis=1)
```

## Percentile

Compute the q-th percentile of all values using linear interpolation (matching NumPy's default behavior).

```python
# Compute median (50th percentile)
Pipeline().source("image_bytes").grayscale().reduce_percentile(q=50)

# Compute 95th percentile
Pipeline().source("image_bytes").grayscale().reduce_percentile(q=95)
```

## Popcount

Count the number of set bits (1s) in the buffer. Useful for binary masks or Hamming distances.

```python
Pipeline().source("image_bytes").threshold(128).reduce_popcount()
```

## Label Reduce

Score contour regions against a buffer (e.g., extract detection scores from a heatmap). Returns a vector of scores, one per contour.

```python
import polars as pl
from polars_cv import Pipeline

# Score each contour region against a heatmap
pipe = Pipeline().source("image_bytes").grayscale().normalize()
result = df.with_columns(
    scores=pl.col("heatmap").cv.pipe(pipe).label_reduce(
        contours=pl.col("contours"),
        reduction="max",
        region_mode="interior",
    ).sink("list")
)
```

**Reductions:** `"max"`, `"mean"`, `"sum"`.
**Region modes:** `"interior"` (filled area), `"boundary"` (contour edge), `"bbox"` (bounding rectangle).

## Convenience Methods

The `LazyPipelineExpr` provides a `statistics()` method to compute common metrics in a single pass.

```python
df.with_columns(
    stats=pl.col("image").cv.pipe(pipe).statistics()
)
# Returns a struct with {mean, std, min, max}
```
