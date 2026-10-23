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

# Any reduction by name: a statistic, or a percentile spelled "pNN"
Pipeline().source("image_bytes").grayscale().reduce("p90")
Pipeline().source("image_bytes").grayscale().reduce(pl.col("which"))  # per row
Pipeline().source("image_bytes").grayscale().reduce("p90", axis=0)  # per column
```

`reduce("mean")` is `reduce_mean()`'s value, computed by the same kernel, in
f64. The dedicated ops remain for what a per-row reduction cannot be:
`reduce_min`/`reduce_max` along an axis keep the input's dtype, and
`reduce_std` takes `ddof`.

Every reduction parameter (`reduce`, `reduce_masked`, `block_reduce`,
`label_reduce`) takes the same spellings: `"sum"`, `"mean"`, `"min"`, `"max"`,
`"std"` (population), `"median"`, `"count"`, or a percentile `"pNN"` (`"p90"`,
`"p62.5"`, at most two decimals, NumPy's linear interpolation). Over no values
each is NaN, except `"sum"` and `"count"` (0).

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

# Per row (axis=1) or per column (axis=0); reduce_median is q=50
Pipeline().source("image_bytes").grayscale(keepdims=False).reduce_median(axis=1)
```

## Masked Reductions

`apply_mask` zeroes outside a mask, and a reduction then counts the zeros.
`reduce_masked` reduces only the pixels the mask selects (u8 or float, set
where > 0): every masked pixel, or one value per column (`axis=0`) or row
(`axis=1`). A line with no masked pixel is NaN (0 for `"sum"` and `"count"`).

```python
heat = pl.col("heat").cv.pipe(Pipeline().source("array"))
breast = pl.col("breast").cv.pipe(Pipeline().source("array"))

heat.reduce_masked(breast, reduction="median", axis=1).sink("numpy")    # per row
heat.reduce_masked(breast, reduction="p75", axis=0).sink("numpy")
heat.statistics(include=["mean", "median", "p10", "p90", "count"], mask=breast)
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

**Reductions:** `"sum"`, `"mean"`, `"min"`, `"max"` (default), `"std"`
(population), `"median"`, a percentile `"pNN"` and `"count"` — the same
set `reduce_masked` takes.
**Region modes:** `"interior"` (filled area), `"boundary"` (contour edge), `"bbox"` (bounding rectangle).

## Convenience Methods

The `LazyPipelineExpr` provides a `statistics()` method to compute common metrics in a single pass.

```python
df.with_columns(
    stats=pl.col("image").cv.pipe(pipe).statistics()
)
# Returns a struct with {mean, std, min, max}
```

`include=` also takes `"sum"`, `"median"`, `"pNN"` (any percentile, e.g.
`"p97.5"`) and, with `mask=`, `"count"`.

Several maps over each region at once — regionprops for any per-pixel map:

```python
stats = polars_cv.region_stats(pl.col("regions"), {
    "heat_max": (heat, "max"),
    "heat_p90": (heat, "p90"),
    "skin_min": (skin_distance, "min"),
})
df.with_columns(stats=stats).unnest("stats").explode(["heat_max", "heat_p90", "skin_min"])
```
