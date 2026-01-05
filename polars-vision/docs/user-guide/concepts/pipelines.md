# Pipelines

Pipelines are the core abstraction in polars-vision. They define a sequence of operations to apply to image data.

## Pipeline Structure

Every pipeline has three parts:

```mermaid
flowchart LR
    Source["Source"] --> Operations["Operations"] --> Sink["Sink"]
```

1. **Source**: How to interpret input data (e.g., `image_bytes`, `file_path`, `contour`)
2. **Operations**: Transformations to apply (e.g., `resize`, `grayscale`, `normalize`)
3. **Sink**: Output format (e.g., `png`, `numpy`, `torch`)

## Creating Pipelines

```python
from polars_vision import Pipeline

# Basic pipeline
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .grayscale()
    .sink("numpy")
)
```

## Source Formats

| Format | Input Type | Description |
|--------|-----------|-------------|
| `image_bytes` | Binary | PNG/JPEG bytes (auto-detect) |
| `file_path` | String | Local or cloud file path |
| `blob` | Binary | VIEW protocol binary |
| `raw` | Binary | Raw bytes (requires dtype) |
| `contour` | Struct | Contour geometry to rasterize |

### Example: File Path Source

```python
# Read from local or cloud files
pipe = Pipeline().source("file_path").resize(224, 224).sink("numpy")

df = pl.DataFrame({
    "path": [
        "/local/image.png",
        "s3://bucket/image.jpg",
        "gs://bucket/image.png",
    ]
})
result = df.with_columns(tensor=pl.col("path").cv.pipeline(pipe))
```

### Example: Contour Source

```python
from polars_vision import Pipeline, CONTOUR_SCHEMA

# Rasterize contours to masks
pipe = Pipeline().source("contour", width=200, height=200).sink("numpy")

df = pl.DataFrame({"contour": [contour_struct]}).cast({"contour": CONTOUR_SCHEMA})
result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))
```

## Sink Formats

| Format | Output Type | Description |
|--------|------------|-------------|
| `png` | Binary | PNG bytes |
| `jpeg` | Binary | JPEG bytes |
| `numpy` | Binary | NumPy-compatible bytes |
| `torch` | Binary | PyTorch-compatible bytes |
| `list` | List | Polars nested List |
| `array` | Array | Polars fixed-size Array |
| `native` | Varies | Domain-appropriate type |

## Chaining Operations

Operations are chained fluently:

```python
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=256, width=256)
    .crop(top=16, left=16, height=224, width=224)  # Center crop
    .flip_h()  # Augmentation
    .normalize(method="minmax")
    .sink("numpy")
)
```

## Dynamic Parameters

Any numeric parameter can be a Polars expression:

```python
# Per-row dimensions
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=pl.col("target_h"), width=pl.col("target_w"))
    .crop(
        top=pl.col("bbox_y"),
        left=pl.col("bbox_x"),
        height=pl.col("bbox_h"),
        width=pl.col("bbox_w"),
    )
    .sink("png")
)
```

## Applying Pipelines

### Eager Mode

Use `.cv.pipeline()` for simple, single-output pipelines:

```python
result = df.with_columns(
    output=pl.col("image").cv.pipeline(pipe)
)
```

### Lazy Mode

Use `.cv.pipe()` for composition and multi-output:

```python
# Returns LazyPipelineExpr for composition
expr = pl.col("image").cv.pipe(pipe_without_sink)

# Materialize with .sink()
result = df.with_columns(output=expr.sink("png"))
```

## Best Practices

1. **Reuse Pipelines**: Define pipelines once, apply to many DataFrames
2. **Use Lazy Mode**: For composition and multi-output scenarios
3. **Dynamic Parameters**: Use expressions for per-row customization
4. **Appropriate Sink**: Choose format based on downstream use

## Next Steps

- [Lazy vs Eager](lazy-vs-eager.md) - Understanding composition modes
- [Domains](domains.md) - Multi-domain pipelines

