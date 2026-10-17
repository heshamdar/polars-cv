# Migrating from 0.28 to 0.29

Every operation, source and sink is now defined once in Rust, and the Python
builder methods are generated from those definitions. The generator applies one
signature rule to every method, which changes how some are called, and many
inputs that used to produce a silently wrong result are now refused. This page
lists what to change coming from 0.28; the
[changelog](../changelog.md) has the full list.

## Keyword-only parameters

An operation with exactly one required parameter takes it positionally or by
name; every other parameter is keyword-only. These calls must now name their
arguments:

| Before | After |
|---|---|
| `.convolve2d(k, 3)` | `.convolve2d(k)` (the side comes from the kernel) |
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
default" (`require_contiguous` `False`, `on_error` `"raise"`). A keyword is passed exactly when it is not `None`, and a
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

## Contour sources

A `contour` source decodes to the contour domain; rasterizing is the
`rasterize()` operation, named explicitly:

| Before | After |
|---|---|
| `source("contour", width=w, height=h)` | `source("contour").rasterize(width=w, height=h)` |
| `source("contour", shape=img, fill_value=v)` | `source("contour").rasterize(shape=img, fill_value=v)` |

`source()` no longer takes `width`, `height`, `shape`, `fill_value` or
`background`. Without `rasterize()`, `source("contour")` is the contour domain,
so geometry operations (`area()`, `simplify()`, ...) apply first. A per-row
canvas value that is invalid for a row is a query error; `source(on_error="null")`
nulls only rows whose contour cannot be decoded.

## Graphs and optimizations

Optimizations are one explicit phase that `.sink()` runs. Choose passes with
`.sink(opt_flags=OptFlags(...))` or the `POLARS_CV_OPTIMIZATIONS` environment
variable, and inspect them with `Pipeline.explain(optimized=True)`.

- **`to_graph()` takes its column**, and `PipelineGraph.set_root_column()` is
  gone: write `pipe.to_graph(pl.col("image"))`.
- **`PipelineGraph.to_expr()` needs an optimized graph.** It raises
  `RuntimeError` otherwise. `.sink()` does this for you; on the low-level path,
  optimize first:

```python
import polars as pl
from polars_cv import OptFlags, Pipeline

pipe = Pipeline().source("image_bytes").grayscale()
expr = pipe.to_graph(pl.col("image")).optimize(OptFlags.all()).to_expr()
```

- **The `affine_fusion` pass is gone.** Consecutive `rotate`/`warp_affine`/
  `shear`/`rotate_and_scale` ops each resample in turn. Composing them changed
  pixels by up to ~185/255, so no optimization may do it. To get one
  interpolation pass, multiply the matrices yourself and call `warp_affine`
  once.

## Results that change

- **`canny` matches `cv2.Canny`** exactly: no Gaussian pre-blur, L1 gradient
  magnitude, and a colour image uses the strongest channel per pixel. It finds
  noticeably more edges than before. Add `.blur(sigma=1.4)` before `.canny(...)`
  to approximate the old smoothing.
- **An image with no contours gives `[]`**, not null, from `extract_contours()`.
  Null now means only "no value" (a null input, or a failed row under
  `on_error="null"`), so test emptiness with `.list.len() == 0`.
- **A null row of `sink("numpy")`/`"torch"`/`"ndarray"` is a null value**, not a
  struct of null fields. Test `row is None` instead of `row["data"] is None`;
  `numpy_from_struct(None)` raises `ValueError`.
- **`Pipeline.output_dtype()` can return `"auto_float"`**: a float whose width
  depends on what the source decodes to.
- **`PlanState.dims` has one entry per dimension** (the planner tracks any
  rank), and `PlanState.DIM_NAMES` is gone.

## Inputs that are now refused

Each of these used to return a wrong or truncated result without complaint.
They now raise when the pipeline is built, if the plan knows enough, or fail
the row (following `on_error`):

| Input | Before | Now |
|---|---|---|
| `crop` with a negative value or a window past the edge | clamped or shrunk | refused |
| `crop` with only one of `height`/`width` | the one given was ignored | honoured |
| `source("raw")` bytes that are not a whole number of elements | remainder dropped | refused |
| `reshape` to a different element count, or after a transpose/flip/crop | unsafe view | refused |
| an image op on anything but `[H, W]` or `[H, W, C]` | passed through or axis dropped | refused |
| `transpose` with too few or repeated axes | failed per row or wrong | refused |
| a jagged or null `List` row (`source("list")`) | misread | refused |
| a null contour coordinate | read as `0.0` | refused |
| a contour struct without an `exterior` field | guessed from `points` or the first list field | refused: rename the field to `exterior` |
| `sobel(axis=)` other than `"x"`/`"y"` | y gradient | refused |
| a per-row `filter="triangle"` or `convert_color` `"grey"`/`"grayscale"` | aliases | refused: use `"bilinear"` / `"gray"` |
| a `list`/`array` sink row whose dtype or length differs from the plan | cast or accepted | refused |

## Removed

- `convolve2d(ksize=)`. The kernel's length decides its side; write
  `.convolve2d(k)` (or `kernel=k`).
- `sobel(ksize=)`, `laplacian(ksize=)`: only `3` was ever accepted; drop the
  argument.
- `Pipeline.output_encoding()`. The plugin reads whether an output is
  histogram buckets from the ops themselves.
- `PipelineGraph.set_root_column()` (see above), `_graph.LIB_PATH`.
- The one-time "ran on one thread" warning and its environment variables
  `POLARS_CV_ENGINE_WARN_SECONDS`, `POLARS_CV_ENGINE_WARN_ROWS` and
  `POLARS_CV_SILENCE_ENGINE_WARNING`: every call now splits its rows over the
  thread pool.

## Dependencies

polars-cv now requires `polars>=1.41.1` and `numpy>=2.0.2`. `networkx`,
`graphviz` and `pydot` are no longer installed by default; install
`polars-cv[viz]` for `PipelineGraph.show_graph()`.

## Hand-built graph JSON

Only relevant if you build the plugin's graph JSON yourself rather than through
`Pipeline`:

- An output is only `{"node", "sink"}`: the plugin plans every output's
  schema from the graph itself. `expected_domain`, `expected_dtype`,
  `expected_shape`, `expected_ndim`, `shape_asserted` and `planned` are refused.
- A shape declaration is an op, `{"op": "assert_shape", "dims": [d0, d1, ...]}`,
  checked against every row where it appears. `dims` is the whole shape, one
  entry per dimension (`null` declares nothing about one), so its length is
  the rank; with `"exact": false` it declares only the leading dimensions
  (what `height=`/`width=`/`channels=` write). The former `rank` field and
  the three-entry `dims` are refused.
- Op, source and sink fields are the Python parameter names with bare values
  (`"height": 224`) or `{"$slot": n}` for a per-row expression. Unknown fields
  are refused by name; a field with a default may be omitted. Renamed:
  `warp_affine`'s `output_size`, `histogram`'s `range`, the binary ops'
  `other`, `apply_mask`'s `mask` and `channel_merge`'s `others`.
- Nodes no longer carry `alias`, `domain` or `output_dtype`, and the
  `expr_column_names` kwarg is refused; both are rejected by name.
- A `contour` source has only `on_error`: it decodes the column to the contour
  domain. Its former `size`/`fill_value`/`background` are a `rasterize` op
  (`{"op": "rasterize", "size": [h, w]}`) as the node's first op.
