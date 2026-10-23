# Parameters & Errors

## Per-row parameters

Most operation parameters accept a Polars expression as well as a literal. An
expression is evaluated against the DataFrame and resolved per row when the
pipeline runs:

```python
import polars as pl
from polars_cv import Pipeline

# Literal: the same for every row
pipe = Pipeline().source("image_bytes").resize(height=224, width=224)

# Per row, from other columns
pipe = Pipeline().source("image_bytes").resize(
    height=pl.col("target_h"), width=pl.col("target_w")
)

# An aggregation is one value, broadcast to every row
pipe = Pipeline().source("image_bytes").crop(
    top=0, left=0, height=pl.col("crop_h").min(), width=pl.col("crop_w").min()
)
```

### Which parameters can be expressions

The rule is not a parameter's type: **a parameter may be per-row if and only
if its value cannot change the output's rank, dtype or domain.** The lazy
schema is computed when the query is planned and must match what executes.

- **Sizes may be per-row.** `resize(height=pl.col("h"))` is fine; that axis is
  then unknown at plan time, which the schema says (a `List` level rather than
  an `Array` one). An `array` sink needs a known shape, so pair a per-row size
  with the `list` or `numpy` sink, or follow it with a fixed-size op.
- **Non-structural enums and flags may be per-row**: a resize `filter`,
  `interpolation`, `pad(mode=)`, `convolve2d(border=, normalize=)`, a
  morphology `footprint`, `distance_transform(signed=)`, and so on.
- **Structural parameters are literal-only**: `cast(dtype=)`,
  `normalize(method=, out_dtype=)`, a reduction `axis`,
  `perceptual_hash(hash_size=, algorithm=)`, `rotate(expand=)`,
  `histogram(closed=, output=)`, the `transpose`/`flip` axis lists and
  `apply_lut(out_dtype=)`. Passing an expression raises
  `TypeError` when the pipeline is built.
- **For a list-valued parameter, the length is structural and the elements are
  not.** A `convolve2d` kernel, a `normalize` mean/std, a `channel_swap` order,
  an affine `matrix`, `reshape`'s sizes and an `apply_lut` table keep a
  literal number of entries, but any entry may be an expression.

Each method's signature in the [API reference](../../api/pipeline.md) says
which applies: a parameter typed `IntOrExpr`, `FloatOrExpr`, `BoolOrExpr` or `... | pl.Expr`
accepts an expression, and one typed `int`, `str` or `list[int]` alone does
not.

`histogram(bins=...)` is the one per-row parameter that fixes a length: a
per-row `bins` cannot feed an `array` sink, which needs that length when the
query is planned. Use the `list` sink, or a literal `bins`.

A per-row value is checked when its row runs. A literal is checked when the
pipeline is built: `blur(sigma=-1)` raises at once, naming the op, while
`blur(sigma=pl.col("s"))` with a negative `s` fails that row (see
[Errors](#errors) for what a failing row does).

### Parameters from the buffer itself: `this`

`polars_cv.this` is the buffer an op reads, per row. Its attributes are
`this.height`, `this.width`, `this.channels` (1 for an `[H, W]` buffer) and
`this.ndim`. Combine them with numbers using `+ - * / //` and pass the result
to any parameter that may be an expression:

```python
from polars_cv import Pipeline, this

# The central 60% of every image, whatever its size
Pipeline().source("image_bytes").crop(
    top=0.2 * this.height,
    left=0.2 * this.width,
    height=0.6 * this.height,
    width=0.6 * this.width,
)

# A blur that scales with the image
Pipeline().source("image_bytes").blur(sigma=this.height / 320)
```

Each op reads the buffer it is given: after `resize(height=512, width=512)`,
`this.height` is 512. An integer parameter takes the value rounded to nearest,
so `0.29 * this.height` at a height of 100 is 29. Where the plan already knows
the sizes an expression reads (after a `resize`, or an `assert_shape`), the
value is known when the query is planned, so the schema states the size and
a fixed-shape `array` sink can carry it; otherwise, as with a column, it is
known per row. A parameter that must be a literal (see the list above) refuses
`this`, as do non-numeric parameters, percentiles and histogram edges (ranks
and points among the values, not sizes), an attribute the buffer does not have
(`this.height` of a 1-D buffer), an op whose input is not a buffer
(`rasterize` reads a contour set), and functions with no buffer such as the
`.contour` methods; each is refused as the pipeline is built wherever the
plan can tell.

### Null parameter values

A parameter column can contain nulls. By default a null parameter fails the
query, naming the column and the row:

```
resize: height: Parameter column 'target_h' has a null value (row 7 of 64 in this batch)
```

A row is named within the batch the plugin was called with: the whole frame
under the in-memory engine, one batch of it under the streaming engine (which
cannot tell the plugin where its batch starts). When the batch is the whole
frame (`7 of 64` in a 64-row frame), the number is the frame's row.

`.on_null_param("null")` opts into null-in/null-out instead: the outputs that
depend on the affected operation are null for that row, exactly as for a null
input image, and every other row and output proceeds.

```python
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=pl.col("h"), width=pl.col("w"))
    .on_null_param("null")
)
```

This is deliberately separate from `.on_error()`: under `"null"` a null
parameter is not an error at all, so decode, encode and genuine operation
failures still raise. You do not have to hide real bugs to tolerate missing
parameter values.

For a **fallback value** rather than a null, fill the null in the expression:
`pl.col("h").fill_null(224)`. There is no policy for this because Polars
already expresses it.

An aggregation broadcasts, so a null in a length-1 parameter (for example
`pl.col("h").max()` over an all-null column) nulls every row.

`.on_null_param()` is a graph-level setting, mirrored on `LazyPipelineExpr`
so you can set it after `.pipe(...)`. When pipelines are composed, a non-default
policy wins: composing a `"null"` pipeline with a `"raise"` one gives the whole
graph `"null"`. The `.contour` / `.point` / `.bbox` namespaces carry the same
policy on the accessor:
`pl.col("c").contour.on_null("null").normalize(pl.col("w"), 100)`.

## Errors

By default, a failure while producing a row raises and aborts the query, with
the earliest failing row's error. Two controls tolerate per-row failures, from
narrowest to broadest.

**Source-level: `source(..., on_error="null")`** covers decode failures only.
A row that fails to decode is null in the outputs that depend on that source;
the rest of the DataFrame still processes.

```python
# Skip undecodable images instead of aborting
pipe = Pipeline().source("image_bytes", on_error="null").resize(height=224, width=224)
```

**Graph-level: `.on_error(policy)`** covers any error that produces a row:
source decode, operation execution or output encoding.

```python
# "raise" (default): the first failing row fails the expression with its error.
pipe = Pipeline().source("image_bytes").resize(height=224, width=224)

# "null": failing rows are null in ALL of the graph's outputs; good rows are unaffected.
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).on_error("null")

# "null_with_message": like "null", but the output becomes a struct with a
# reserved `_error` field holding the failure message (null for good rows).
# A single-output pipeline becomes a two-field struct (`_output` + `_error`).
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .on_error("null_with_message")
)
```

`.on_error()` is mirrored on `LazyPipelineExpr`. When pipelines are composed
(by `merge_pipe` or a binary op) they must agree on it.

Some inputs fail the query under every policy, because they are a mistake in
the query rather than in a row: a column of the wrong type (a `source("contour")`
over a column that is not contours, say), or a literal parameter no row could
use.
