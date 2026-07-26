# Expressions (`.cv`)

The `.cv` accessor is registered on every Polars expression once `polars_cv` is
imported. It is the entry point for pipelines, for raw byte access, and for the
header-only metadata readers.

```python
import polars as pl
import polars_cv  # registers .cv (and .point / .contour / .bbox)
```

## Overview

| Method | Input column | Returns | Purpose |
|--------|--------------|---------|---------|
| [`pipe()`](#polars_cv.expressions.CvNamespace.pipe) | image bytes, paths, lists/arrays, contour structs | `LazyPipelineExpr` | Apply a [`Pipeline`](pipeline.md); finish with `.sink(...)` |
| [`read_bytes()`](#polars_cv.expressions.CvNamespace.read_bytes) | `String` paths | `Binary` | Read the bytes a path names, **without** decoding |
| [`width()`](#polars_cv.expressions.CvNamespace.width) | `Binary` encoded images | `UInt32` | Image width from the header only |
| [`height()`](#polars_cv.expressions.CvNamespace.height) | `Binary` encoded images | `UInt32` | Image height from the header only |
| [`channels()`](#polars_cv.expressions.CvNamespace.channels) | `Binary` encoded images | `UInt32` | Channel count from the header only |
| [`image_dtype()`](#polars_cv.expressions.CvNamespace.image_dtype) | `Binary` encoded images | `String` | Sample dtype from the header only |

Only `pipe()` builds a graph and runs through the `vb_graph` expression. The
others are standalone plugin expressions: they take a column and return a
column, with no pipeline to build.

## Pipelines

```python
pipe = Pipeline().source("image_bytes").resize(height=224, width=224)

df.with_columns(out=pl.col("image").cv.pipe(pipe).sink("numpy"))
```

See [Pipeline](pipeline.md) for the operations and
[LazyPipelineExpr](lazy.md) for composition and sinks.

## Byte access

`read_bytes()` is the fetch half of the `"file_path"` source — that source
fetches a path's bytes and then decodes them as an image, and this stops after
the fetch. Bytes are returned verbatim, so an encoded file survives the round
trip byte-for-byte.

```python
# Read originals, filter on header metadata, decode only the survivors.
lf = (
    pl.scan_parquet("images.parquet")
    .with_columns(raw=pl.col("path").cv.read_bytes())
    .filter(pl.col("raw").cv.width() > 512)
    .with_columns(thumb=pl.col("raw").cv.pipe(pipe).sink("png"))
)
```

It takes the same `cloud_options` and `on_error` values as
`Pipeline.source("file_path")`. See
[Reading Bytes Without Decoding](../user-guide/concepts/sources.md#reading-bytes-without-decoding)
for the memory and streaming behaviour, and for the path-sandboxing caveat.

## Metadata

The metadata readers parse only the image header, so they do not decode pixels.
They take a `Binary` column of encoded images — pair them with `read_bytes()` to
query files named by a path column:

```python
raw = pl.col("path").cv.read_bytes()
df.with_columns(w=raw.cv.width(), h=raw.cv.height())
```

See [Metadata & Display](../user-guide/operations/metadata.md) for details.

## API Reference

::: polars_cv.expressions.CvNamespace
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
