# Image Operations

This page documents the primary image processing operations available in polars-cv.

## Resize

Resize images to specified dimensions.

```python
Pipeline().source("image_bytes").resize(height=224, width=224)
Pipeline().source("image_bytes").resize(height=224, width=224, filter="bilinear")
```

**Filters:** `"nearest"`, `"bilinear"`, `"catmullrom"`, `"gaussian"`, `"lanczos3"` (default),
`"area"`. For a large shrink, `"area"` makes each output pixel the mean of its
footprint, every input pixel weighted by its overlap (OpenCV's `INTER_AREA`);
enlarging, it is bilinear.

### Resize onto another node's grid

`resize_like` resizes an image to another node's height and width, row by
row, keeping its own channels and dtype: a low-resolution heatmap onto the
image it was computed from, so the two compare cell by cell.

```python
img = pl.col("image").cv.pipe(Pipeline().source("image_bytes"))
heat = pl.col("heatmap").cv.pipe(Pipeline().source("array"))
heat.resize_like(img, filter="bilinear").sink("numpy")
```

### Block reduce

`block_reduce` reduces each non-overlapping `height x width` block to one
value (`"mean"`, `"sum"`, `"min"`, `"max"`, `"std"`, `"median"`,
a percentile `"p90"`, `"count"`), per channel. A partial block at the
right or bottom edge reduces only its in-bounds pixels, so the output is
`ceil(H / height) x ceil(W / width)`.

```python
Pipeline().source("array").block_reduce(height=16, width=16, reduction="max")
```

## Grayscale

Convert to grayscale using luminance formula. The result is `[H, W, 1]`;
`keepdims=False` drops the channel axis, so an axis reduction over it gives one
value per row or column rather than one-element lists.

```python
Pipeline().source("image_bytes").grayscale()
Pipeline().source("image_bytes").grayscale(keepdims=False)  # [H, W]
```

## Blur

Apply Gaussian blur.

```python
Pipeline().source("image_bytes").blur(sigma=3.0)
```

## Rank Filters

`rank_filter` replaces each pixel by its window's `q`-th percentile, per
channel: `q=50` is a median, `0`/`100` a minimum/maximum. Unlike a blur, it
ignores the outliers in the window. Both sides must be odd; `border` is
`"replicate"` (default), `"reflect"` or `"zero"`.

```python
Pipeline().source("image_bytes").median_filter(5)
# A running median along each row, 25 pixels wide
Pipeline().source("array").rank_filter(height=1, width=25)
```

`background_divide(sigma)` flat-fields an image by dividing it by its own
Gaussian blur (plus `epsilon`), on a `LazyPipelineExpr`:

```python
img = pl.col("image").cv.pipe(Pipeline().source("image_bytes").grayscale())
img.background_divide(50.0).sink("numpy")
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

Omit `height` or `width` to crop to the end of that axis. By default the window
must lie inside the image: a negative bound, or a window that runs past the
edge, is an error (a per-row error when it comes from a column, so `on_error`
applies) rather than being clamped to a smaller region.

Strips, tiles and sliding windows near a border say what they want instead with
`bounds=` (offsets may then be negative):

```python
# The part of the window inside the image: narrower near the right edge.
Pipeline().source("image_bytes").crop(top=0, left=8, width=24, bounds="clip")
# Always 12 x 24, filled with 0 outside the image.
Pipeline().source("image_bytes").crop(
    top=0, left=8, height=12, width=24, bounds="pad", pad_value=0
)
```

`bounds="clip"` refuses a window with nothing inside the image; `bounds="pad"`
needs both `height` and `width`, and its size is known to the lazy schema.

## Rotate

Rotate by an angle in degrees. Part of the [affine transform family](#affine-transforms).

```python
Pipeline().source("image_bytes").rotate(angle=90)
Pipeline().source("image_bytes").rotate(angle=45, expand=True)
Pipeline().source("image_bytes").rotate(angle=30, interpolation="nearest", border_value=128)
```

**Fast path:** 90, 180, and 270 degree rotations use zero-copy view operations (metadata-only, no allocation). The `interpolation` and `border_value` parameters are ignored for these angles.

**Arbitrary angles** are executed via the same affine transform code path as `warp_affine`.

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
# Flip only the rows whose column says so (e.g. right-side views)
Pipeline().source("image_bytes").flip_h(when=pl.col("laterality") == "R")
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

Values outside the bins and NaN follow NumPy: a value below the first edge or
above the last (of an explicit `range` or explicit edges), like a NaN, is in no
bin, so it is not counted (`normalized` shares sum to 1 over the counted
values), and its `quantized` index is one past the last bin. To count
everything, open the outer edges: `bins=[-math.inf, 50, 200, math.inf]`. An
auto-detected range over NaN or infinity, or a supplied `range` that is not
finite, raises: there are no equal-width bins to make, so pass `range=` or
explicit edges. Explicit edges must increase monotonically.

**Outputs:** `"buckets"` (default), `"counts"`, `"normalized"`, `"quantized"`, `"edges"`.
**Closed Intervals:** `"left"` (default), `"right"`.

## Color Conversion

Convert between color spaces using `convert_color` or convenience methods.

```python
# Generic conversion
Pipeline().source("image_bytes").convert_color(from_space="rgb", to_space="hsv")

# Convenience methods
Pipeline().source("image_bytes").to_hsv()
Pipeline().source("image_bytes").to_lab()    # promotes to f32
Pipeline().source("image_bytes").to_bgr()
Pipeline().source("image_bytes").to_ycbcr()
```

**Supported spaces:** `rgb`, `bgr`, `hsv`, `lab`, `ycbcr`, `gray`.

**Value ranges** follow OpenCV per dtype:

| Dtype | Channel range | HSV hue | YCbCr chroma centre |
|-------|---------------|---------|---------------------|
| `u8` | 0–255 | half-degrees, [0, 180) | 128 |
| `u16`, `u32`, `u64` | 0–MAX | the whole range is one turn, [0, MAX] | (MAX + 1) / 2 |
| `f32`, `f64` | [0, 1] | degrees, [0, 360) | 0.5 |

Lab is always `f32` (L in [0, 100]), and converting from Lab gives `f32` RGB
in [0, 1]. Signed integer images have no colour range: `hsv`, `lab` and
`ycbcr` refuse them (cast first); `rgb`, `bgr` and `gray` accept every dtype.

## Channel Operations

### Channel Select

Extract a single channel from a multi-channel image as `[H, W, 1]` (the shape
`grayscale` and `rasterize` give), or as `[H, W]` with `keepdims=False`.

```python
# Extract the red channel (index 0 of RGB): [H, W, 1]
Pipeline().source("image_bytes").channel_select(index=0)
# ... or as a 2-D [H, W] buffer
Pipeline().source("image_bytes").channel_select(index=0, keepdims=False)
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

### Elementwise Math

Pure per-pixel math primitives. Each promotes to float like `scale`/`relu`:
8/16-bit integers → `f32`, 32/64-bit integers → `f64` (which holds their
values, as NumPy promotes them), floats preserved. On an `f32`-computing input
they **fuse automatically** with adjacent scalar ops — a chain of them runs as a
single kernel pass rather than one pass per op, with no fusion for you to
manage.

```python
p = Pipeline().source("image_bytes", dtype="u8")
p.neg()          # -x
p.abs()          # |x|
p.sqrt()         # sqrt(x)  (NaN for negative input)
p.square()       # x * x
p.reciprocal()   # 1 / x    (±inf at zero)
p.sign()         # -1 / 0 / +1
p.floor()        # round toward -inf
p.ceil()         # round toward +inf
p.round()        # round to nearest, ties to even (matches Polars/numpy)
p.trunc()        # round toward zero (drop the fractional part)
```

`clamp_min`, `clamp_max`, `add_constant` and `subtract_constant` take a bound or
constant that may be a literal **or** a per-row Polars expression:

```python
p.clamp_min(0.0)      # max(x, 0.0)   — one-sided floor
p.clamp_max(255.0)    # min(x, 255.0) — one-sided cap
p.add_constant(10)    # x + 10
p.subtract_constant(pl.col("bias"))  # per-row subtrahend
```

### Preserving the input dtype

Intensity operations compute in a float — `f32` for 8/16-bit integers, `f64`
for 32/64-bit ones — so by default an integer image is **promoted to that
float** on output. Pass `preserve_dtype=True` to cast the result back to the
input dtype instead — the math still runs in the float, but the result is
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
Pipeline().source("image_bytes").convolve2d(kernel=kernel)

# Normalize kernel so output values stay in range
Pipeline().source("image_bytes").convolve2d(kernel=kernel, normalize=True)
```

**Border modes:** `"replicate"` (default), `"zero"`, `"reflect"`.

### Sobel

Sobel gradient operator (delegates to `convolve2d` with standard kernels).

```python
Pipeline().source("image_bytes").grayscale().sobel(axis="x")
Pipeline().source("image_bytes").grayscale().sobel(axis="y")
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

### Footprint

Every morphology op takes `footprint="square"` (the default, a `ksize × ksize`
square) or `footprint="disk"`: the pixels within Euclidean distance
`ksize // 2` of the centre, so shapes grow and shrink round rather than
square.

```python
# Round dilation by 4 pixels
Pipeline().source("image_bytes").grayscale().threshold(128).dilate(ksize=9, footprint="disk")
```

With a disk, `iterations=n` applies the disk `n` times; unlike squares, two
disks do not add up to one larger disk, so for a single round offset of `r`
use `ksize=2 * r + 1` with one iteration. On a mask, a disk dilation by `r`
equals `distance_transform()` of the inverted mask thresholded at `r`.

## Distance Transform

The exact Euclidean distance transform of a single-channel mask, as an `f32`
image of the same shape. A pixel is foreground when it is non-zero.

```python
# Each foreground pixel's distance to the nearest background pixel (0 on the background)
Pipeline().source("image_bytes").grayscale().threshold(128).distance_transform()

# Signed distance to the mask's boundary: negative inside, positive outside
Pipeline().source("image_bytes").grayscale().threshold(128).distance_transform(signed=True)
```

Unsigned distances are centre to centre, as `cv2.distanceTransform(DIST_L2,
DIST_MASK_PRECISE)` and `scipy.ndimage.distance_transform_edt` compute them.
Signed distances are measured to the pixel edges `extract_contours` traces
(`±(d - 0.5)` for the centre distance `d` to the nearest pixel of the other
class), the same convention as `.point.signed_distance_to_contour`. If a class
has no pixel at all, the distance to it is infinite (`±inf`).

Thresholding a distance gives offsets of any radius: `distance_transform()
.threshold(r)` keeps the pixels more than `r` from the background.

## Coordinate Grid

`coordinate_grid(normalized=False)` gives the image's pixel coordinates: an
f32 `[H, W, 2]` buffer holding each pixel's centre `(x + 0.5, y + 0.5)` — the
point `rasterize` samples — or `((x + 0.5) / W, (y + 0.5) / H)` when
`normalized`. It reads only the height and width, never the values, so it
takes any dtype and channel count. Coordinates are in the frame of the buffer
it follows: after a crop they start at the crop's corner.

With ordinary ops it builds fields over positions:

```python
# Squared distance from (32, 32), per axis
dx2_dy2 = (
    Pipeline().source("image_bytes")
    .coordinate_grid()
    .add_constant(-32.0)
    .square()
)
```

## Lookup Tables

`apply_lut(table, out_dtype=)` reads each u8 or u16 value as a row index into
`table`. Rows of one entry map every element in place, on any channel count;
rows of `K` entries turn a single-channel image into `K` channels, a palette.

```python
# Invert through a table (one entry per value, every channel)
Pipeline().source("image_bytes", dtype="u8").apply_lut([[255 - v] for v in range(256)])

# A blue-to-red palette over a grayscale image: [H, W, 1] -> [H, W, 3]
Pipeline().source("image_bytes").grayscale().apply_lut([[v, 0, 255 - v] for v in range(256)])

# Six terraces as float32 values
Pipeline().source("image_bytes").grayscale().apply_lut([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]], out_dtype="f32")
```

A value with no row is an error for its row. To send larger values to the
last row, clamp first while keeping the dtype:
`.clamp(0, len(table) - 1, out_dtype="u8")` (`clamp_max` alone promotes to
`f32`, which `apply_lut` refuses). Float input must be scaled, rounded and cast
to u8 or u16 first.

## Affine Transforms

Apply arbitrary 2x3 affine transformations. All methods in this family share the same Rust execution code path. Each one resamples the image, so consecutive affine ops run one after another (they are not composed into a single warp); build the combined matrix yourself and call `warp_affine` once when you want a single interpolation pass. The matrix uses the **forward-mapping** convention (same as OpenCV `warpAffine`): the kernel inverts it internally for interpolation.

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

### Squeeze and Expand Dims

Remove an axis of size 1, or insert one, without spelling out every size (so
they work where `reshape` cannot: on images whose height and width are only
known per row). Both are zero-copy views, and a negative axis counts from the
end, as in NumPy.

```python
Pipeline().source("image_bytes").grayscale().squeeze(-1)  # [H, W, 1] -> [H, W]
Pipeline().source("image_bytes").expand_dims(0)  # [H, W, C] -> [1, H, W, C]
```

`squeeze` refuses an axis of any other size. Binary ops never broadcast, so
these are how two operands are brought to one rank.

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

## Per-Row Parameters

Most parameters above accept a Polars expression, resolved per row; the
structural ones (`cast(dtype=)`, a reduction `axis`, ...) do not. The rule,
null parameter values and error handling are in
[Parameters & Errors](../concepts/parameters.md).

## Next Steps

- [Geometry Operations](geometry.md)
- [Hashing](hashing.md)
- [Reductions](reductions.md)
