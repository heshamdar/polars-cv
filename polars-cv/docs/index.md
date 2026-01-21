# polars-cv

**High-performance vision and array processing for Polars DataFrames.**

polars-cv is a Polars plugin that enables lazy, modular image processing on DataFrame columns.

## Key Features

- **🚀 Performance**: Rust-powered operations with automatic kernel fusion.
- **🔗 Composable**: Modular pipelines that can be branched and merged.
- **🎯 Multi-Domain**: Seamlessly move between images, geometry (contours), and numeric results.
- **📊 Multi-Output**: Extract multiple results from a single execution pass.
- **🔌 Integration**: Direct output to NumPy and other formats.

## Quick Example

```python
import polars as pl
from polars_cv import Pipeline

# Define a reusable preprocessing pipeline
preprocess = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .grayscale()
)

# Apply to a DataFrame column and output as NumPy bytes
df = pl.DataFrame({"image": [image_bytes]})
result = df.with_columns(
    processed=pl.col("image").cv.pipe(preprocess).sink("numpy")
)
```

## Getting Started

1. [Installation](getting-started/installation.md)
2. [Quickstart](getting-started/quickstart.md)
3. [User Guide](user-guide/concepts/pipelines.md)

