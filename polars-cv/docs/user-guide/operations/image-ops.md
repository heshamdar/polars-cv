# Image Operations

This page documents the primary image processing operations available in polars-cv.

## Resize

Resize images to specified dimensions.

```python
Pipeline().source("image_bytes").resize(height=224, width=224)
Pipeline().source("image_bytes").resize(height=224, width=224, filter="bilinear")
```

**Filters:** `"nearest"`, `"bilinear"`, `"catmullrom"`, `"gaussian"`, `"lanczos3"` (default).

## Grayscale

Convert to grayscale using luminance formula.

```python
Pipeline().source("image_bytes").grayscale()
```

## Blur

Apply Gaussian blur.

```python
Pipeline().source("image_bytes").blur(sigma=3.0)
```

## Threshold

Convert to binary image.

```python
Pipeline().source("image_bytes").threshold(128)
```

## Crop

Extract a rectangular region.

```python
Pipeline().source("image_bytes").crop(top=10, left=10, height=100, width=100)
```

## Rotate

Rotate by an angle in degrees. Part of the [affine transform family](#affine-transforms).

```python
Pipeline().source("image_bytes").rotate(angle=90)
Pipeline().source("image_bytes").rotate(angle=45, expand=True)
Pipeline().source("image_bytes").rotate(angle=30, interpolation="nearest", border_value=128)
```

**Fast path:** 90, 180, and 270 degree rotations use zero-copy view operations (metadata-only, no allocation). The `interpolation` and `border_value` parameters are ignored for these angles.

**Arbitrary angles** are executed via the same affine transform code path as `warp_affine`. When a `rotate` is followed by a `warp_affine` (or vice versa), they are [fused automatically](#pipeline-fusion).

## Pad

Add padding to edges.

```python
Pipeline().source("image_bytes").pad(top=10, bottom=10, value=128)
Pipeline().source("image_bytes").pad_to_size(height=224, width=224)
Pipeline().source("image_bytes").letterbox(height=224, width=224)
```

## Flip

```python
Pipeline().source("image_bytes").flip_h()
Pipeline().source("image_bytes").flip_v()
```

## Histogram

Compute the pixel value histogram. The histogram can return counts, normalized frequencies, bin edges, quantized images, or detailed "buckets" combining counts and edges. The default output is `"buckets"`.

```python
# Detailed buckets output (returns List[Struct] with lower_edge, upper_edge, count, normalized)
Pipeline().source("image_bytes").grayscale().histogram(bins=256)

# Return raw bin counts
Pipeline().source("image_bytes").grayscale().histogram(bins=64, output="counts")

# Return normalized frequencies
Pipeline().source("image_bytes").grayscale().histogram(bins=64, output="normalized")

# Explicit bin edges (custom intervals)
Pipeline().source("image_bytes").grayscale().histogram(bins=[0, 50, 100, 200, 255])

# Left or right closed intervals
Pipeline().source("image_bytes").grayscale().histogram(bins=10, closed="right")
```

**Outputs:** `"buckets"` (default), `"counts"`, `"normalized"`, `"quantized"`, `"edges"`.
**Closed Intervals:** `"left"` (default), `"right"`.

## Color Conversion

Convert between color spaces using `convert_color` or convenience methods.

```python
# Generic conversion
Pipeline().source("image_bytes").convert_color("rgb", "hsv")

# Convenience methods
Pipeline().source("image_bytes").to_hsv()
Pipeline().source("image_bytes").to_lab()    # promotes to f32
Pipeline().source("image_bytes").to_bgr()
Pipeline().source("image_bytes").to_ycbcr()
```

**Supported spaces:** `rgb`, `bgr`, `hsv`, `lab`, `ycbcr`, `gray`.

## Channel Operations

### Channel Select

Extract a single channel from a multi-channel image, producing a 2D `[H, W]` buffer.

```python
# Extract the red channel (index 0 of RGB)
Pipeline().source("image_bytes").channel_select(index=0)
```

### Channel Swap

Reorder channels in a multi-channel image.

```python
# RGB to BGR
Pipeline().source("image_bytes").channel_swap(order=[2, 1, 0])
```

## Intensity Adjustments

### Contrast

Scale pixel deviation from the mean: `(pixel - mean) * factor + mean`.

```python
Pipeline().source("image_bytes").adjust_contrast(factor=1.5)
```

### Gamma Correction

Power-law correction: normalizes to `[0,1]`, applies `pixel^gamma`, then denormalizes.

```python
Pipeline().source("image_bytes").adjust_gamma(gamma=0.5)   # brighter
Pipeline().source("image_bytes").adjust_gamma(gamma=2.0)   # darker
```

### Brightness

Scale pixel values with clamping.

```python
Pipeline().source("image_bytes").adjust_brightness(factor=1.3)
```

### Invert

Invert pixel values (`255 - pixel` for u8, `1.0 - pixel` for float).

```python
Pipeline().source("image_bytes").invert()
```

All intensity parameters accept Polars expressions for per-row dynamic values.

### Preserving the input dtype

Intensity operations compute in `f32`, so by default an integer image is
**promoted to `f32`** on output. Pass `preserve_dtype=True` to cast the result
back to the input dtype instead — the math still runs in `f32`, but the result is
rounded and saturated back into the original storage type. This is available on
`scale`, `clamp`, and `adjust_brightness`.

```python
# Default: u8 in → f32 out
Pipeline().source("image_bytes", dtype="u8").adjust_brightness(factor=1.2)

# preserve_dtype: u8 in → u8 out
Pipeline().source("image_bytes", dtype="u8").adjust_brightness(factor=1.2, preserve_dtype=True)
```

Two constraints, both enforced at build time:

- The pipeline's dtype must be **known** (not `"auto"`). Pin it at the source
  (`source(..., dtype="u8")`) or with a prior dtype-fixing op, otherwise a
  `ValueError` is raised.
- `preserve_dtype=True` is **mutually exclusive** with `out_dtype` on `scale`
  and `clamp` — set one or the other, not both.

## Convolution

Apply 2D convolution with an arbitrary kernel.

```python
# Custom 3x3 emboss kernel
kernel = [-2, -1, 0, -1, 1, 1, 0, 1, 2]
Pipeline().source("image_bytes").convolve2d(kernel, ksize=3)

# Normalize kernel so output values stay in range
Pipeline().source("image_bytes").convolve2d(kernel, ksize=3, normalize=True)
```

**Border modes:** `"replicate"` (default), `"zero"`, `"reflect"`.

### Sobel

Sobel gradient operator (delegates to `convolve2d` with standard kernels).

```python
Pipeline().source("image_bytes").grayscale().sobel(axis="x")
Pipeline().source("image_bytes").grayscale().sobel(axis="y", ksize=3)
```

### Laplacian

Second-derivative operator for edge detection.

```python
Pipeline().source("image_bytes").grayscale().laplacian()
```

### Sharpen

Unsharp-mask-style sharpening. `strength=0` produces the identity.

```python
Pipeline().source("image_bytes").sharpen(strength=1.5)
```

## Edge Detection

### Canny

Multi-stage edge detection (Gaussian blur, Sobel gradients, non-maximum suppression, hysteresis thresholding). Output is a U8 binary edge map.

```python
Pipeline().source("image_bytes").grayscale().canny(low_threshold=50.0, high_threshold=150.0)
```

Thresholds accept Polars expressions for per-row values.

## Histogram Equalization

Contrast enhancement via cumulative histogram remapping. Operates per-channel on multi-channel images. Output is U8.

```python
Pipeline().source("image_bytes").equalize_histogram()
```

## Morphological Operations

Morphological operations for binary mask and segmentation post-processing. All require **single-channel** input (use `.grayscale()` or `.threshold()` first).

### Erode

Shrink bright regions by computing the local minimum over a rectangular neighborhood.

```python
Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3)
Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=5, iterations=2)
```

### Dilate

Grow bright regions by computing the local maximum over a rectangular neighborhood.

```python
Pipeline().source("image_bytes").grayscale().threshold(128).dilate(ksize=3)
Pipeline().source("image_bytes").grayscale().threshold(128).dilate(ksize=5, iterations=2)
```

### Opening

Erode then dilate — removes small bright noise spots while preserving larger structures.

```python
Pipeline().source("image_bytes").grayscale().threshold(128).morphology_open(ksize=3)
```

This is a Python-side composite equivalent to `.erode(ksize=ksize).dilate(ksize=ksize)`.

### Closing

Dilate then erode — fills small dark holes while preserving larger structures.

```python
Pipeline().source("image_bytes").grayscale().threshold(128).morphology_close(ksize=3)
```

This is a Python-side composite equivalent to `.dilate(ksize=ksize).erode(ksize=ksize)`.

### Morphological Gradient

Dilate minus erode — produces an edge outline.

```python
Pipeline().source("image_bytes").grayscale().threshold(128).morphology_gradient(ksize=3)
```

**Common workflow:** `threshold() → erode() → dilate() → extract_contours()`.

## Affine Transforms

Apply arbitrary 2x3 affine transformations. All methods in this family share the same Rust execution code path and can be [fused together](#pipeline-fusion). The matrix uses the **forward-mapping** convention (same as OpenCV `warpAffine`): the kernel inverts it internally for interpolation.

The affine family includes:

- `rotate()` -- simple angle-based rotation ([see above](#rotate))
- `warp_affine()` -- raw 2x3 matrix
- `shear()` -- shear convenience method
- `rotate_and_scale()` -- rotation + uniform scaling around a center

### Warp Affine

```python
# Translate by (tx=30, ty=20), output same size as input
Pipeline().source("image_bytes").warp_affine(
    matrix=[1, 0, 30, 0, 1, 20],
    output_size=(224, 224),
)

# Nearest-neighbor interpolation, white border fill
Pipeline().source("image_bytes").warp_affine(
    matrix=[1, 0, 30, 0, 1, 20],
    output_size=(224, 224),
    interpolation="nearest",
    border_value=255.0,
)
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `matrix` | Six-element `[a, b, tx, c, d, ty]` forward-mapping matrix; each element may be a literal or a per-row Polars expression | — |
| `output_size` | `(height, width)` of the output | — |
| `interpolation` | `"bilinear"` or `"nearest"` | `"bilinear"` |
| `border_value` | Fill value for out-of-bounds pixels | `0.0` |

Because each matrix element accepts an expression, a batch can apply a different
(e.g. random) affine per row in a single call:

```python
Pipeline().source("image_bytes").warp_affine(
    matrix=[pl.col("a"), pl.col("b"), pl.col("tx"),
            pl.col("c"), pl.col("d"), pl.col("ty")],
    output_size=(224, 224),
)
```

### Shear

Convenience wrapper that builds a shear matrix and delegates to `warp_affine`.

```python
Pipeline().source("image_bytes").shear(sx=0.3, output_size=(224, 224))
Pipeline().source("image_bytes").shear(sy=0.2, output_size=(224, 224))
Pipeline().source("image_bytes").shear(sx=0.1, sy=0.15, output_size=(224, 224))
```

### Rotate and Scale

Combined rotation and uniform scaling around a center point.

```python
Pipeline().source("image_bytes").rotate_and_scale(
    angle=45,           # degrees, positive = clockwise
    scale=0.8,
    center=(112, 112),  # (cx, cy) rotation center
    output_size=(224, 224),
)
```

### Pipeline Fusion

Consecutive affine operations (`warp_affine`, `rotate` with static arbitrary angle, `shear`, `rotate_and_scale`) are **automatically fused** into a single matrix multiplication at planning time, eliminating redundant interpolation passes:

```python
pipe = (
    Pipeline()
    .source("image_bytes")
    .warp_affine(matrix=[1, 0, 50, 0, 1, 0], output_size=(224, 224))   # translate X
    .warp_affine(matrix=[1, 0, 0, 0, 1, 30], output_size=(224, 224))   # translate Y
)
# Serializes as a single warp_affine with matrix [1, 0, 50, 0, 1, 30]

pipe = (
    Pipeline()
    .source("image_bytes")
    .assert_shape(height=224, width=224)
    .rotate(45)
    .warp_affine(matrix=[1, 0, 10, 0, 1, 10], output_size=(224, 224))  # translate after rotate
)
# Fused into a single warp_affine
```

**Fusion limitations:** `rotate` with an expression-based angle, or with a zero-copy angle (90/180/270), does not participate in fusion. Non-affine ops between two affine ops break the fusion run.

## Layout

### Transpose

Transpose dimensions.

```python
# HWC to CHW
Pipeline().source("image_bytes").transpose([2, 0, 1])
```

### Reshape

Reshape array to new dimensions.

```python
Pipeline().source("image_bytes").resize(height=224, width=224).reshape([1, 224, 224, 3])
```

## Resize Variants

In addition to `resize(height=..., width=...)`, polars-cv provides aspect-ratio-preserving resize methods.

```python
# Resize by scale factor
Pipeline().source("image_bytes").resize_scale(scale=0.5)
Pipeline().source("image_bytes").resize_scale(scale_x=2.0, scale_y=0.5)

# Resize to target height (width computed from aspect ratio)
Pipeline().source("image_bytes").resize_to_height(512)

# Resize to target width (height computed from aspect ratio)
Pipeline().source("image_bytes").resize_to_width(640)

# Resize so the longest side equals target
Pipeline().source("image_bytes").resize_max(max_size=256)

# Resize so the shortest side equals target
Pipeline().source("image_bytes").resize_min(min_size=128)
```

Every resize variant accepts Polars expressions for per-row dynamic sizes, and
for the `filter` choice. (`thumbnail`, below, is a decode-time assertion rather
than a resize — its `max_size` is literal-only.)

## Thumbnail

Decode only a downscaled *thumbnail* of an image source. This is the explicit,
chainable form of `source(..., decode_max_size=...)`: it asserts the pipeline needs
at most `max_size` pixels on the decoded long side, so JPEG decoding uses IDCT
scaling (1/8, 1/4, 1/2) to skip work — a large CPU and memory win. Non-JPEG formats
(PNG, …) ignore the assertion and decode at full size.

```python
# Cheap decode for a curation pass
Pipeline().source("file_path").thumbnail(256).perceptual_hash()
```

Must be called right after `source(...)` on an `image_bytes`/`file_path` source. A
scaled decode followed by a resize is **not** bit-identical to a full decode then
the same resize (different resampling path), so it is an explicit opt-in. See
[Streaming & Scaling](../concepts/streaming.md#cheaper-decoding-for-curation-passes)
for the two-pass curation pattern.

## Shape Assertion

Declare a shape the planner cannot work out for itself. This is for sources
that do not reveal their shape at plan time — a `list`/`array` column's rank and
sizes are only known once the data arrives. For a source the planner can read
(image bytes, a file path) the shape is already inferred, and an assertion is at
best redundant.

There are two spellings.

`dims=` is positional and complete: entry *i* is the size of dimension *i*, and
`None` leaves one unknown. It also pins the output **rank**, which is what lets
a `list`/`array` source reach a fixed-shape `array` sink:

```python
pipe = Pipeline().source("list", dtype="f32").assert_shape(dims=[8, 8, 3])
df.with_columns(pl.col("arr").cv.pipe(pipe).sink("array"))

# None leaves a dimension unknown while still pinning the rank
Pipeline().source("list", dtype="u8").assert_shape(dims=[None, None, 3])
```

`height=`/`width=`/`channels=` is the `[H, W, C]` spelling of dimensions 0, 1
and 2, for the common image case:

```python
# Assert the decoded image has 4 channels (RGBA)
Pipeline().source("image_bytes").assert_shape(channels=4)

# Assert full shape
Pipeline().source("image_bytes").assert_shape(height=512, width=512, channels=3)
```

Those names describe an `[H, W, C]` buffer, and the hints are positional — so
after a `transpose([2, 0, 1])` dimension 0 is the channel axis and calling it
`height` would be a lie. They are rejected once the rank is known to be anything
but 3; use `dims=` there. Put the assertion *before* a transpose, where the
names mean what they say.

**An assertion states a fact; it does not change one.** A declaration that
contradicts something the pipeline already knows is rejected at the line that
wrote it, naming both values and the operation that established the known one:

```python
Pipeline().source("image_bytes").resize(height=224, width=224).assert_shape(height=999)
# ValueError — resize already fixed dimension 0 at 224
```

Note what an assertion does *not* supply. `.assert_shape()` and `.resize()`
declare or set dimensions of the buffer; neither gives `.sink("array")` its
output shape, which comes from `.sink("array", shape=[...])`.

`dims=` entries are literal `int`s — a rank and its per-dimension sizes are
plan-time schema facts. The `height`/`width`/`channels` keywords do accept
expressions and resolve them per row, but a per-row size is not a plan-time
fact, so it publishes no shape.

## Dynamic Parameters

Most numeric parameters across polars-cv operations accept **Polars expressions** in addition to literal values. When a parameter is an expression, its value is resolved per-row at execution time from the DataFrame.

```python
# Static value (same for all rows)
pipe = Pipeline().source("image_bytes").resize(height=224, width=224)

# Dynamic value (per-row from another column)
pipe = Pipeline().source("image_bytes").resize(
    height=pl.col("target_h"), width=pl.col("target_w")
)

# Expression with aggregation (same value for all rows)
pipe = Pipeline().source("image_bytes").crop(
    top=0, left=0,
    height=pl.col("crop_h").min(),
    width=pl.col("crop_w").min(),
)
```

**Parameters that accept expressions:**

| Category | Parameters |
|----------|-----------|
| Resize | `height`, `width`, `scale`, `scale_x`, `scale_y`, `max_size`, `min_size`, `filter` |
| Crop | `top`, `left`, `height`, `width` |
| Pad | `top`, `bottom`, `left`, `right`, `value`, `mode` |
| Pad to size | `height`, `width`, `value`, `position` |
| Letterbox | `height`, `width`, `value`, `filter` |
| Rotate | `angle`, `border_value`, `interpolation` |
| Warp affine | `matrix` (each of the 6 elements, per-row), `output_size` (height and width), `border_value`, `interpolation` |
| Shear | `sx`, `sy` |
| Rotate and scale | `angle`, `scale`, `center`, `output_size` |
| Scale / Clamp | `factor`, `min_val`, `max_val` |
| Normalize | `mean`, `std` (each element, per-row) |
| Threshold | `value` |
| Blur | `sigma` |
| Canny | `low_threshold`, `high_threshold` |
| Contrast / Gamma / Brightness | `factor`, `gamma` |
| Sharpen | `strength` |
| Morphology | `ksize`, `iterations` |
| Channel select / swap | `index`, `order` (each element, per-row) |
| Convolution | `kernel` (each coefficient, per-row), `ksize`, `border` |
| Flags | `apply_mask(invert)`, `convolve2d(normalize)`, `area(signed)` |
| Reductions | `q` (percentile), `ddof` (std) |
| Histogram | `bins` (integer form), `range` |
| Extract contours | `min_area`, `mode`, `method` |
| Label reduce | `reduction`, `region_mode` |
| Rasterize | `width`, `height`, `fill_value`, `background` |
| Contour source | `width`, `height`, `fill_value`, `background` |

**Planning-time implications:** When a shape-affecting parameter is an expression (e.g., `resize(height=pl.col("h"))`), the pipeline planner cannot determine the output dimensions at planning time. Shape hints will be `None` for those dimensions.

`histogram(bins=...)` is a special case: it is accepted as an expression, but it
fixes the output vector length. Pairing a per-row `bins` with an `array` sink
therefore fails at planning time, since that sink needs a known shape — use the
`list` sink, or pass a literal `bins`.

**Structural parameters remain literal-only**, because they fix the output
shape, rank or dtype at planning time and the lazy schema must match what
executes: reduction `axis`, `perceptual_hash(hash_size)`, the `axes` lists of
`transpose`/`flip`, `reshape`'s element count, `rotate(expand)` (it changes the
output dimensions), `cast(dtype)`, `normalize(method`/`out_dtype)` and
`histogram(closed`/`output)`. Passing an expression to one of these raises
`TypeError` at build time.

Note the distinction for list-valued parameters: the *length* is structural
while the *values* are not. A convolution `kernel`, a `normalize` mean/std pair,
a `channel_swap` order and the affine `matrix` all keep a literal element count
but accept a per-row expression for any individual element.

### Null parameter values

A parameter column can contain nulls. By default a null parameter fails the
whole expression, naming the column and the row:

```
Parameter column 'target_h' has a null value at row 7
```

`.on_null_param("null")` opts into null-in/null-out instead — the rows whose
parameter is null yield null, exactly as a null *input image* already does, and
every other row proceeds:

```python
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=pl.col("h"), width=pl.col("w"))
    .on_null_param("null")
)
```

Only the outputs that depend on the affected operation go null, and unrelated
failures still raise — see
[Error Handling](../concepts/sources.md#error-handling) for how this differs
from `.on_error("null")`.

For a **fallback value** instead of a null, fill it in the expression:
`pl.col("h").fill_null(224)`. There is no policy for this because Polars
already expresses it.

An aggregation broadcasts, so a null in a length-1 parameter column (e.g.
`pl.col("h").max()` over an all-null column) applies to — and nulls — every row.

## Next Steps

- [Geometry Operations](geometry.md)
- [Hashing](hashing.md)
- [Reductions](reductions.md)
