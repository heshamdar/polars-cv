# Quickstart

This guide will get you started with polars-cv in 5 minutes.

## Your First Pipeline

A polars-cv pipeline has three parts:

1. **Source**: How to interpret input data
2. **Operations**: Transformations to apply
3. **Sink**: Output format

```python
import polars as pl
from polars_cv import Pipeline

# 1. Define a pipeline
pipe = (
    Pipeline()
    .source("image_bytes")  # Decode PNG/JPEG bytes
    .resize(height=224, width=224)
)

# 2. Apply to a DataFrame column and specify sink
df = pl.DataFrame({"image": [png_bytes]})
result = df.with_columns(
    resized=pl.col("image").cv.pipe(pipe).sink("numpy")
)
```

## Common Operations

### Image Processing

```python
# Grayscale conversion
Pipeline().source("image_bytes").grayscale()

# Blur with sigma=3
Pipeline().source("image_bytes").blur(sigma=3.0)

# Threshold to binary
Pipeline().source("image_bytes").grayscale().threshold(128)

# Crop region
Pipeline().source("image_bytes").crop(top=10, left=10, height=100, width=100)

# Flip horizontally
Pipeline().source("image_bytes").flip_h()
```

### Normalization

```python
# MinMax normalization [0, 1]
Pipeline().source("image_bytes").normalize(method="minmax")

# ZScore normalization (mean=0, std=1)
Pipeline().source("image_bytes").normalize(method="zscore")
```

## Dynamic Parameters

Any parameter can be a Polars expression for per-row customization:

```python
# Resize each image to different dimensions
df = pl.DataFrame({
    "image": [img1, img2, img3],
    "target_h": [64, 128, 256],
    "target_w": [64, 128, 256],
})

pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=pl.col("target_h"), width=pl.col("target_w"))
)

result = df.with_columns(
    resized=pl.col("image").cv.pipe(pipe).sink("numpy")
)
```

## Output Formats (Sinks)

| Format | Description | Use Case |
|--------|-------------|----------|
| `numpy` | NumPy-compatible bytes | NumPy, OpenCV, Scikit-image |
| `png` | Re-encode as PNG bytes | Storage, display |
| `jpeg` | Re-encode as JPEG bytes | Web usage |
| `list` | Polars nested List | Internal Polars analysis |
| `native` | Python primitive | Scalars (Area, Mean) |

## Reading from Files

```python
# Read from local paths or URLs
df = pl.DataFrame({"path": ["/path/to/image.png", "https://example.com/img.jpg"]})
pipe = Pipeline().source("file_path").resize(224, 224)

result = df.with_columns(
    processed=pl.col("path").cv.pipe(pipe).sink("numpy")
)
```

