# Migrating from 0.29 to 0.30

0.30 changes no method signatures except one deletion (`ratio`). What changes
is results: every op now computes and stores in the dtype it declares,
following NumPy and OpenCV where they define an answer, and many inputs that
used to give a silently wrong value now give the right one or are refused. This
page lists what to check coming from 0.29; the [changelog](../changelog.md) has
the full list.

## Removed

- `ratio`: it executed exactly as `divide`. Use `divide`, and `.scale(255)` for
  the scaling `ratio` documented.

## Dependencies

polars-cv now requires `polars>=1.43.2` (was `>=1.41.1`). Older polars panics
the plugin importing a sliced `Array` column that has nulls.

## Results that change

**Colour conversions read each dtype's value range**, as OpenCV does:

| Dtype | Was | Now |
|-------|-----|-----|
| `f32`, `f64` | 0–255 | [0, 1]; HSV hue in degrees, S and V in [0, 1]; YCbCr chroma centred at 0.5 |
| `u16`, `u32`, `u64` | 0–255 | 0..MAX; YCbCr chroma centred at (MAX + 1) / 2 |
| Lab output → RGB | `f32` in 0–255 | `f32` in [0, 1] |
| signed integers | accepted | refused by `hsv`, `lab` and `ycbcr` (cast first) |

Scale a 0–255 float image by 1/255 before converting. `u8` keeps OpenCV's
8-bit conventions, moving by at most one unit where a tie now rounds
differently.

**Division by zero is IEEE.** `divide` gives `inf` (`-inf` for a negative
numerator) for `x / 0` and `nan` for `0 / 0`; it gave `0`. Replace
non-finite values afterwards where `0` was wanted.

**Dtype promotion follows NumPy.** Mixed operands meet in NumPy's
`result_type` (`u8` + `i8` is `i16`; `u64` with a signed integer is `f64`).
The float-promoting ops (`scale`, `sqrt`, `divide`, `convolve2d`, gamma, the
scalar math family, …) give `f64` for 32/64-bit integer input, where they gave
`f32`. 8/16-bit integers still give `f32`. Add a `cast` if you need the old
dtype.

**NaN propagates.** `relu`, `clamp_min`, `clamp_max`, `maximum`/`minimum`,
`dilate`/`erode`, `morphology_gradient`, MinMax `normalize`, the ordering
reductions (`reduce_max`/`min`/`argmax`/`argmin`/`percentile`) and
`label_reduce(reduction="max")` return NaN when a NaN is involved, as NumPy
does. Use `fill_nan` first where a bound or a skip was wanted.

**`histogram` follows NumPy.** A value outside an explicit `range` or explicit
edges is no longer clamped into the end bin, and a NaN is no longer counted in
bin 0: both are in no bin, and their `quantized` index is one past the last
bin. Open the outer edges to count everything:
`histogram(bins=[-math.inf, 50, 200, math.inf])`.

**`extract_contours` traces pixel edges.** A `w x h` blob's outline bounds
`w x h` (it bounded `(w-1) x (h-1)`), a single pixel is a unit square with area
1, and `extract_contours` → `rasterize` returns the mask unchanged. Vertex
coordinates, `area`, `perimeter`, `bounding_box` and IoU all change, and
`min_area` now filters on the pixel count.

**Detection metrics move.** Through the new outlines, `ContourMatcher` IoU
rises (most for small lesions), so some former misses become true positives
and one-pixel-thick detections now count. Its defaults changed too:
`min_contour_area` is `0.0` (was `1.0`), and `gt_min_contour_area` is `1.0`
and no longer follows `min_contour_area` — pass it explicitly if you relied on
that; `None` is refused.

**Derived resize sizes are exact.** `resize_to_height`, `resize_to_width`,
`resize_max`, `resize_min`, `resize_scale` and `letterbox` round a derived
size half up in exact arithmetic and never to 0. Sizes change only where f32
used to land an exact half below .5, or where the result was 0.

**Smaller shifts:** `normalize(method="zscore")` uses exact sums, so values
can differ in the last bits; 32/64-bit integer and `f64` images keep full
precision through resize, blur and `rgb -> gray`; `invert` of a signed integer
is `-1 - x` (NumPy's `~x`).

## Inputs that are now refused

- Non-finite geometry: a NaN or infinite `rotate`/`rotate_and_scale` angle,
  centre or scale, `warp_affine` coefficient, or a non-finite, zero or
  negative `resize_scale` factor. A literal fails at plan time, a column per
  row.
- A contour, point or bbox with a NaN or infinite coordinate, in every
  geometry function, the `contour` source and `label_reduce`.
- A `histogram` auto range over NaN or infinity, a non-finite `range`, a
  `range` whose min exceeds its max, and explicit edges that do not increase.
- A colour conversion of a channel count other than the space's own or those
  plus alpha (`to_bgr` of a 5-channel image).
- Reductions with no defined value over an empty image
  (`reduce_max`/`min`/`argmax`/`argmin`/`percentile`), edge-type `pad` of an
  empty axis, and an `Array` column with a zero-size dimension.
- A `list`/`array` value the declared dtype cannot hold (`300` or `-1` as
  `u8`, or `NaN`): it stored 0. It now fails naming the value, or nulls the
  row under `on_error="null"`.
- A bitwise op between `u64` and a signed integer (they have no common integer dtype).
