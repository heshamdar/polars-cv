# Image Operations

This page documents the primary image processing operations available in polars-cv.

## Resize

Resize images to specified dimensions.

```python
Pipeline().source("image_bytes").resize(height=224, width=224)
Pipeline().source("image_bytes").resize(height=224, width=224, filter="bilinear")
```

**Filters:** `"nearest"`, `"bilinear"`, `"lanczos3"` (default).

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

Rotate by an angle in degrees.

```python
Pipeline().source("image_bytes").rotate(angle=90)
Pipeline().source("image_bytes").rotate(angle=45, expand=True)
```

**Note:** 90, 180, and 270 degree rotations are zero-copy.

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

Convert between color spaces using `cvt_color` or convenience methods.

```python
# Generic conversion
Pipeline().source("image_bytes").cvt_color("rgb", "hsv")

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

All resize variants accept Polars expressions for per-row dynamic sizes.

## Shape Assertion

Provide shape hints for the pipeline planner. Useful for asserting known dimensions when the source has unknown shape.

```python
# Assert the decoded image has 4 channels (RGBA)
Pipeline().source("image_bytes").assert_shape(channels=4)

# Assert full shape
Pipeline().source("image_bytes").assert_shape(height=512, width=512, channels=3)
```

## Next Steps

- [Geometry Operations](geometry.md)
- [Hashing](hashing.md)
- [Reductions](reductions.md)
