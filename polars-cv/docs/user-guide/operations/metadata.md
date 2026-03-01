# Image Metadata & Display

Query image properties directly from binary columns and visualize results in Jupyter notebooks.

## Image Metadata

The `.cv` namespace provides lightweight metadata methods that extract header information from encoded image bytes without decoding the full image.

```python
import polars as pl

result = df.with_columns(
    w=pl.col("image").cv.width(),
    h=pl.col("image").cv.height(),
    ch=pl.col("image").cv.channels(),
    dt=pl.col("image").cv.image_dtype(),
)
```

| Method | Return Type | Description |
|--------|------------|-------------|
| `.cv.width()` | `UInt32` | Image width in pixels |
| `.cv.height()` | `UInt32` | Image height in pixels |
| `.cv.channels()` | `UInt32` | Number of channels (1, 3, 4, etc.) |
| `.cv.image_dtype()` | `String` | Element dtype name (e.g., `"uint8"`, `"float32"`) |

These methods work on binary columns containing encoded images (PNG, JPEG, TIFF, etc.).

### Use Case: Filtering by Size

```python
large_images = df.filter(
    (pl.col("image").cv.width() > 1024) | (pl.col("image").cv.height() > 1024)
)
```

### Use Case: Dynamic Resize

```python
from polars_cv import Pipeline

pipe = Pipeline().source("image_bytes").resize(
    height=pl.col("image").cv.height() // 2,
    width=pl.col("image").cv.width() // 2,
)
```

## Display Utility

`show_images` renders images from a binary column directly in Jupyter notebooks. Outside notebooks, it prints a text summary.

```python
from polars_cv import show_images

show_images(df, "image_column")
show_images(df, "processed", max_rows=5, max_width=300)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame containing the image column |
| `column` | `str` | required | Name of the binary column to display |
| `max_rows` | `int` | `10` | Maximum number of images to show |
| `max_width` | `int` | `200` | Maximum display width per image (pixels) |
| `format` | `str` | `"auto"` | Force a specific format instead of auto-detecting |

Supported formats: PNG, JPEG, WebP, TIFF, BMP, GIF, VIEW protocol blobs, and numpy-sink struct columns.
