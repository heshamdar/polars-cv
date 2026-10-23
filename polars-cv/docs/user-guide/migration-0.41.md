# Migrating from 0.40 to 0.41

0.41 gives each normalization method its own op, renames the reduction enum,
and returns image sizes as signed integers. This page lists what to check
coming from 0.40; the [changelog](../changelog.md) has the full list,
including the new ops (`polars_cv.this`, `reduce`, `reduce_masked`,
`masked_select`, `region_stats`, `rank_filter`/`median_filter`,
`background_divide`, `resize_like`, `block_reduce`, `crop_clip`/`crop_pad`,
`normalize_percentile`/`normalize_range`, `flip(when=)` and
`sink("long")`).

## Signatures that change

**`normalize(method="preset", mean=, std=)` is `normalize_preset(mean=,
std=)`.** `normalize` keeps `"minmax"` and `"zscore"`; `"preset"` is refused.
Results are byte-identical.

```python
pipe.normalize_preset(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# was pipe.normalize(method="preset", mean=[...], std=[...])
```

**`LabelReduction` is `Statistic`.** Only code that imports the enum class
needs the new name; the string spellings (`"max"`, `"mean"`, `"sum"`) are
unchanged. Every reduction parameter (`label_reduce`, `.contour.label_reduce`,
the contour matcher's `score_reduction`, and the new `reduce`,
`reduce_masked`, `block_reduce`) now also takes `"min"`, `"std"`,
`"median"`, `"count"` and a percentile spelled `"pNN"` (`"p90"`).

```python
from polars_cv._types import Statistic  # was LabelReduction
```

**`reduce_percentile(q)` refuses `q` outside [0, 100].** It used to clamp
`q` and return the minimum or maximum.

## Results that change

**`.cv.width()`, `.cv.height()`, `.cv.channels()` and `.cv.image_info()`'s
size fields are `Int64`, not `UInt32`.** Arithmetic on a size can now go below
zero and be refused as the negative number it is, rather than wrapping to
about 2^32. A schema pinned to `UInt32`, or a cast added to work around the
wrap, should be updated.

**`label_reduce` over a region with no pixels** gives NaN (0 for `sum` and
`count`), as every other reduction over nothing does; it gave 0.0 for every
reduction.

**The contour matcher drops a detection scored NaN,** as it drops a zero;
NaN sorts above every number in polars, so such a detection was matched as
the most confident one.

## Error messages

A failing row is named within its batch, `(row 11 of 12 in this batch)`,
rather than `(row n)`, which under the streaming engine was the batch's index
presented as the frame's. Code that parses row numbers out of error text
should read the new form; a bad per-row parameter also names its op and
parameter (`crop: width: Parameter column 'l': -1 cannot be negative (row 0
of 1 in this batch)`).

## Persisted graphs

A stored graph that uses `normalize` with `method="preset"` is refused, as
every unknown method is; rebuild it with 0.41. The wire format version is
unchanged.
