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
| `.cv.width()` | `Int64` | Image width in pixels |
| `.cv.height()` | `Int64` | Image height in pixels |
| `.cv.channels()` | `Int64` | Number of channels (1, 3, 4, etc.) |
| `.cv.image_dtype()` | `String` | Element dtype name (e.g., `"uint8"`, `"float32"`) |
| `.cv.image_info()` | `Struct` | All four from one header read: `{width, height, channels, dtype}` |

These methods work on binary columns containing encoded images (PNG, JPEG, TIFF,
etc.) and on **path** columns. A local file is read only as far as its header
needs (a JPEG's frame header may sit behind large EXIF or comment segments, so
the read grows until it is found). A remote object is read the same way: its
first 64 KiB (an HTTP `Range` / object-store ranged read), and the whole object
only when the header is not in them. An empty remote object has no first byte
to range over and is a row error. Path columns take the same `cloud_options`, `on_error` and `allowed_roots` as
`read_bytes()`. Asking for more than one field? `image_info()` reads each
header once:

```python
df.with_columns(info=pl.col("path").cv.image_info()).unnest("info")
```

Sizes are `Int64`, so arithmetic on them can go below zero and be refused as
the negative value it is: `pl.col("path").cv.width() - 32` on a 31-wide image
is -1, not a wrapped `UInt32`.

When the same rows are also decoded, fetch once with
[`read_bytes()`](../concepts/sources.md#reading-bytes-without-decoding) and
read both the header and the pixels from that column, rather than fetching the
object for each expression.

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
