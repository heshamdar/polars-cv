# Image Operations

This page documents all image processing operations available in polars-vision.

## Resize

Resize images to specified dimensions.

```python
Pipeline().resize(height=224, width=224)
Pipeline().resize(height=224, width=224, filter="lanczos3")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `height` | int or Expr | required | Target height |
| `width` | int or Expr | required | Target width |
| `filter` | str | `"lanczos3"` | Filter type: `"nearest"`, `"bilinear"`, `"lanczos3"` |

**Filter Types:**

| Filter | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `nearest` | Fastest | Lowest | Pixel art, binary masks |
| `bilinear` | Medium | Good | General purpose |
| `lanczos3` | Slowest | Best | Photo quality |

## Grayscale

Convert to grayscale (single channel).

```python
Pipeline().grayscale()
```

Uses standard luminance formula: `0.299R + 0.587G + 0.114B`

## Blur

Apply Gaussian blur.

```python
Pipeline().blur(sigma=3.0)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigma` | float or Expr | required | Blur strength (standard deviation) |

## Threshold

Convert to binary image based on threshold value.

```python
Pipeline().threshold(128)
Pipeline().threshold(pl.col("threshold_value"))  # Dynamic threshold
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | int or Expr | required | Threshold value (0-255) |

Values ≥ threshold become 255, values < threshold become 0.

## Crop

Extract a rectangular region.

```python
Pipeline().crop(top=10, left=10, height=100, width=100)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top` | int or Expr | required | Top edge offset |
| `left` | int or Expr | required | Left edge offset |
| `height` | int or Expr | required | Crop height |
| `width` | int or Expr | required | Crop width |

## Flip

Flip images horizontally or vertically.

```python
Pipeline().flip_h()  # Horizontal flip (left-right)
Pipeline().flip_v()  # Vertical flip (top-bottom)
```

## Normalize

Normalize values to a standard range.

```python
Pipeline().normalize(method="minmax")  # Scale to [0, 1]
Pipeline().normalize(method="zscore")  # Mean=0, std=1
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"minmax"` | `"minmax"` or `"zscore"` |
| `out_dtype` | str | `"f32"` | Output dtype |

**Methods:**

| Method | Formula | Output Range |
|--------|---------|--------------|
| `minmax` | `(x - min) / (max - min)` | [0, 1] |
| `zscore` | `(x - mean) / std` | Unbounded |

## Scale

Multiply all values by a factor.

```python
Pipeline().scale(factor=0.5)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `factor` | float or Expr | required | Scale factor |

## Clamp

Clamp values to a range.

```python
Pipeline().clamp(min_val=0.0, max_val=1.0)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_val` | float or Expr | required | Minimum value |
| `max_val` | float or Expr | required | Maximum value |

## ReLU

Apply rectified linear unit (clamp negative values to 0).

```python
Pipeline().relu()
```

Equivalent to `.clamp(min_val=0)` but optimized.

## Cast

Convert to a different data type.

```python
Pipeline().cast("f32")
Pipeline().cast("u8")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | str | required | Target dtype: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f32`, `f64` |

## Chaining Operations

Operations are designed to chain fluently:

```python
# Common ML preprocessing pipeline
ml_preprocess = (
    Pipeline()
    .source("image_bytes")
    .resize(height=256, width=256)
    .crop(top=16, left=16, height=224, width=224)  # Center crop
    .flip_h()  # Augmentation
    .normalize(method="minmax")
    .sink("numpy")
)
```

## DType Promotion

polars-vision automatically handles type conversions:

| Operation | Input | Output |
|-----------|-------|--------|
| `resize`, `grayscale`, `blur`, `threshold` | Any | U8 |
| `normalize`, `scale`, `clamp`, `relu` | U8/U16 | F32 |
| `normalize`, `scale`, `clamp`, `relu` | F32/F64 | Same |

## Next Steps

- [Geometry Operations](geometry.md) - Contour operations
- [Hashing](hashing.md) - Perceptual hashing

