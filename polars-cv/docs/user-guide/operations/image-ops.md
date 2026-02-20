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

## Next Steps

- [Geometry Operations](geometry.md)
- [Hashing](hashing.md)
- [Reductions](reductions.md)
