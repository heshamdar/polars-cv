# Pipelines

Pipelines are the core abstraction in polars-cv. They define a sequence of operations that can be applied to image data.

## Pipeline Structure

A polars-cv pipeline has three parts:

```mermaid
flowchart LR
    Source["Source"] --> Operations["Operations"] --> Sink["Sink"]
```

1. **Source**: How to interpret input data (e.g., `image_bytes`, `file_path`)
2. **Operations**: Transformations to apply (e.g., `resize`, `grayscale`)
3. **Sink**: Final output format (e.g., `numpy`, `png`)

## Modular Style

The recommended way to use polars-cv is to define a pipeline of transformations and then apply it to a column, followed by a sink.

```python
from polars_cv import Pipeline
import polars as pl

# 1. Define operations (reusable)
preprocess = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .grayscale()
)

# 2. Apply to column and choose sink
df = pl.DataFrame({"image": [png_bytes]})
result = df.with_columns(
    processed=pl.col("image").cv.pipe(preprocess).sink("numpy")
)
```

## Source Formats

| Format | Input Type | Description |
|--------|-----------|-------------|
| `image_bytes` | Binary | Decode PNG/JPEG/TIFF bytes (auto format+dtype detect) |
| `file_path` | String | Local/cloud/HTTP path; decodes like `image_bytes` |
| `raw` | Binary | Raw bytes (requires `dtype`) |
| `list` | List | Polars nested List |
| `array` | Array | Polars fixed-size Array |
| `contour` | Struct | Contour geometry to rasterize |

### Auto DType for Image Sources

For `image_bytes` and `file_path`, decoded dtype is determined at runtime:

- PNG/JPEG typically decode to `u8`
- 16-bit PNG decodes to `u16`
- TIFF can decode to `u8`, `u16`, `f32`, or `f64`

Because of this variability, the pipeline tracks dtype as `auto` until it can be resolved by:

- `source(..., dtype="...")`
- a dtype-fixing operation such as `normalize`, `threshold`, or `cast`

If you use `sink("list")` or `sink("array")`, dtype must be known at planning time.

## Sink Formats

| Format | Output Type | Description |
|--------|------------|-------------|
| `numpy` | Struct | NumPy-compatible zero-copy struct (`{data, dtype, shape, strides, offset}`); `dtype="f16"` downcasts |
| `torch` | Struct | PyTorch-compatible zero-copy struct; `dtype="f16"` downcasts |
| `png` | Binary | PNG bytes |
| `jpeg` | Binary | JPEG bytes |
| `webp` | Binary | WebP bytes |
| `tiff` | Binary | TIFF bytes with LZW compression (supports floating-point) |
| `blob` | Binary | Self-describing VIEW binary protocol |
| `list` | List | Polars nested List |
| `array` | Array | Polars fixed-size Array |
| `native` | Varies | Native Python type (for scalars/vectors) |

## Chaining Operations

Operations are chained fluently. Most image operations accept both literal values and Polars expressions.

```python
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=256, width=256)
    .crop(top=pl.col("y_off"), left=pl.col("x_off"), height=100, width=100)
    .normalize(method="minmax")
)
```

## Best Practices

1. **Reuse Pipelines**: Define pipelines once and apply them to many columns.
2. **Dynamic Parameters**: Use Polars expressions for per-row customization.
3. **Modular Sinks**: Keep pipelines generic and choose the sink format at the last step.

## Next Steps

- [Sources](sources.md) - Source formats and auto dtype behavior
- [Domains](domains.md) - Multi-domain pipelines
- [Multi-Output](../composition/multi-output.md) - Extracting multiple results
