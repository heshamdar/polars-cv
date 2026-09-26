# Migrating to the typed-op API

Every operation, source and sink is now defined once in Rust, and the Python
builder methods are generated from those definitions. The generator applies one
signature rule to every method, which changes how some are called. This page
lists what to change coming from 0.28.

## Keyword-only parameters

An operation with exactly one required parameter takes it positionally or by
name; every other parameter is keyword-only. These calls must now name their
arguments:

| Before | After |
|---|---|
| `.convolve2d(k, 3)` | `.convolve2d(kernel=k, ksize=3)` |
| `.convert_color("rgb", "hsv")` | `.convert_color(from_space="rgb", to_space="hsv")` |
| `.histogram(64)` | `.histogram(bins=64)` |
| `.normalize("zscore")` | `.normalize(method="zscore")` |
| `.reduce_max(0)` (also `reduce_mean`, `reduce_min`) | `.reduce_max(axis=0)` |
| `.reduce_std(0, 1)` | `.reduce_std(axis=0, ddof=1)` |
| `.warp_affine(m, (h, w))` | `.warp_affine(matrix=m, output_size=(h, w))` |
| `.perceptual_hash("average", 64)` | `.perceptual_hash(algorithm="average", hash_size=64)` |

A positional call now raises `TypeError: ... takes 1 positional argument but 3
were given`, so none of these fail silently.

These gained a positional first argument (existing keyword calls still work):
`adjust_contrast(factor)`, `adjust_gamma(gamma)`, `channel_select(index)`,
`channel_swap(order)`, `simplify(tolerance)` and `label_reduce(contours)`.

```python
from polars_cv import Pipeline

pipe = (
    Pipeline()
    .source("image_bytes")
    .convert_color(from_space="rgb", to_space="hsv")
    .channel_select(2)
    .normalize(method="minmax")
)
```

## `source()` keywords default to `None`

Each `source()` keyword now defaults to `None`, meaning "the format's own
default" (`fill_value` 255, `background` 0, `require_contiguous` `False`,
`on_error` `"raise"`). A keyword is passed exactly when it is not `None`, and a
passed keyword the format does not read is refused. That now includes a value
that happens to equal the default: `source("image_bytes",
require_contiguous=False)` raises, because image sources never read
`require_contiguous`. Drop the keyword.

## `assert_shape` is checked

`assert_shape` is an operation that checks every row where it is written: a row
whose data does not match fails naming the assertion, and everything after it
relies on the declared shape. It used to be an unchecked claim, reported (if at
all) when the output's shape disagreed. `dims=` entries may now be per-row
expressions, like `height=`/`width=`/`channels=`.

## Sinks are checked at `.sink()`

`.sink()` compiles and plans the graph with the plugin's own code, so an output
the plugin cannot produce (an unencodable dtype, a `vector` into `numpy`, a
singular `warp_affine` matrix) raises `ValueError` there rather than a
`ComputeError` at `collect()`.

## Geometry accessors

The `.contour`, `.point` and `.bbox` methods are generated from their Rust
definitions too (a method that is also a pipeline operation, such as
`.contour.area` or `.contour.scale`, *is* that operation). Two calls change:

- **`.contour.scale()` scales about the centroid by default**, as
  `Pipeline.scale_contour()` always has. The two used to disagree —
  `.contour.scale(2, 2)` scaled away from `(0, 0)` — and now share one
  definition. To keep the old result, pass `origin="origin"`.
- **`.point.interpolate(other, t)` takes `t` by name**: every accessor
  argument with a default is keyword-only. Write
  `.point.interpolate(other, t=0.25)`.

A misspelled literal (`ensure_winding("CW")`, `scale(origin="top_left")`) still
raises `ValueError` when the expression is built; the message now comes from
the definition and lists every accepted spelling.

## Removed

- `Pipeline.output_encoding()`. The plugin reads whether an output is
  histogram buckets from the ops themselves.

## Hand-built graph JSON

Only relevant if you build the plugin's graph JSON yourself rather than through
`Pipeline`:

- An output is only `{"node", "sink"}`: the plugin plans every output's
  schema from the graph itself. `expected_domain`, `expected_dtype`,
  `expected_shape`, `expected_ndim`, `shape_asserted` and `planned` are refused.
- A shape declaration is an op, `{"op": "assert_shape", "rank": ..., "dims":
  [d0, d1, d2]}`, checked against every row where it appears.
- Op, source and sink fields are the Python parameter names with bare values
  (`"height": 224`) or `{"$slot": n}` for a per-row expression. Unknown fields
  are refused by name.
