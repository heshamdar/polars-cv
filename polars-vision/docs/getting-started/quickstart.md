# Quickstart

This guide will get you started with polars-vision in 5 minutes.

## Your First Pipeline

A polars-vision pipeline has three parts:

1. **Source**: How to interpret input data
2. **Operations**: Transformations to apply
3. **Sink**: Output format

```python
import polars as pl
from polars_vision import Pipeline

# Create a simple pipeline
pipe = (
    Pipeline()
    .source("image_bytes")  # Input: PNG/JPEG bytes
    .resize(height=224, width=224)  # Resize to 224x224
    .sink("png")  # Output: PNG bytes
)

# Apply to a DataFrame
df = pl.DataFrame({"image": [png_bytes]})
result = df.with_columns(
    resized=pl.col("image").cv.pipeline(pipe)
)
```

## Common Operations

### Image Processing

```python
# Grayscale conversion
Pipeline().source("image_bytes").grayscale().sink("png")

# Blur with sigma=3
Pipeline().source("image_bytes").blur(sigma=3.0).sink("png")

# Threshold to binary
Pipeline().source("image_bytes").grayscale().threshold(128).sink("png")

# Crop region
Pipeline().source("image_bytes").crop(top=10, left=10, height=100, width=100).sink("png")

# Flip horizontally
Pipeline().source("image_bytes").flip_h().sink("png")
```

### Normalization

```python
# MinMax normalization [0, 1]
Pipeline().source("image_bytes").normalize(method="minmax").sink("numpy")

# ZScore normalization (mean=0, std=1)
Pipeline().source("image_bytes").normalize(method="zscore").sink("numpy")
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
    .sink("png")
)

result = df.with_columns(resized=pl.col("image").cv.pipeline(pipe))
```

## Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `png` | PNG bytes | Display, storage |
| `jpeg` | JPEG bytes | Web, compressed |
| `numpy` | NumPy-compatible bytes | ML frameworks |
| `torch` | PyTorch-compatible bytes | Deep learning |
| `list` | Polars List column | Analysis in Polars |
| `array` | Polars Array column | Fixed-shape data |

## Reading from Files

```python
# Local files
pipe = Pipeline().source("file_path").resize(224, 224).sink("numpy")

df = pl.DataFrame({"path": ["/path/to/image.png"]})
result = df.with_columns(tensor=pl.col("path").cv.pipeline(pipe))

# Cloud storage (S3, GCS, Azure)
df = pl.DataFrame({"path": ["s3://bucket/image.png"]})
result = df.with_columns(tensor=pl.col("path").cv.pipeline(pipe))
```

## Next Steps

- [Pipelines](../user-guide/concepts/pipelines.md) - Deep dive into pipeline concepts
- [Composable Pipelines](../user-guide/concepts/lazy-vs-eager.md) - Learn about lazy composition
- [Multi-Output](../user-guide/composition/multi-output.md) - Extract multiple outputs

