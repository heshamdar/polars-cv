# ![polars-cv](assets/logo/polars-cv-halftone.png)

**Image processing as Polars expressions: decode, transform and analyse images
in a DataFrame column, in Rust, on every core, without a Python loop.**

A `Pipeline` describes the work, `.cv.pipe()` applies it to a column, and
`.sink()` picks the output format:

```python
import polars as pl
from polars_cv import Pipeline

pipe = (
    Pipeline()
    .source("file_path", allowed_roots=["photos"])
    .resize(height=224, width=224)
    .grayscale()
)

df = pl.DataFrame({"path": ["photos/a.jpg", "photos/b.jpg"]})
out = df.with_columns(gray=pl.col("path").cv.pipe(pipe).sink("numpy"))
```

## Features

- **Images are a column.** Paths, bytes, labels and results sit in one
  DataFrame; filtering, joining and `group_by` on image properties is plain
  Polars.
- **Polars runs it.** Pipelines are lazy expressions that run in the streaming
  engine for larger-than-memory data, with each call's rows spread over every
  core.
- **Per-row parameters.** A crop box from an annotation, a size from metadata:
  any parameter that does not change the output's rank, dtype or domain takes
  a Polars expression.
- **One decode, many outputs.** Branches of one image merge into a single
  graph, with shared steps computed once.
- **Beyond pixels.** Contours, points and boxes as columns
  (`.contour` / `.point` / `.bbox`), drawing primitives, perceptual hashes,
  reductions, and detection metrics (COCO AP, FROC, bootstrap intervals).
- **Checked outputs.** Results are held to NumPy, SciPy, OpenCV and
  `imagehash` in CI.

## Where to go next

| If you want to... | Read |
|---|---|
| Install it | [Installation](getting-started/installation.md) |
| Run a first pipeline | [Quickstart](getting-started/quickstart.md) |
| Understand the model | [Pipelines](user-guide/concepts/pipelines.md), [Sources](user-guide/concepts/sources.md), [Parameters & Errors](user-guide/concepts/parameters.md), [Domains](user-guide/concepts/domains.md) |
| Process a large dataset | [Streaming & Scaling](user-guide/concepts/streaming.md) |
| Find an operation | [Image Operations](user-guide/operations/image-ops.md), [Geometry](user-guide/operations/geometry.md), [API Reference](api/pipeline.md) |
| Get several results from one decode | [Multi-Output](user-guide/composition/multi-output.md) |
| Draw shapes, masks and fields | [Drawing](user-guide/drawing.md) |
| Evaluate a detector | [Detection Metrics](user-guide/operations/metrics.md) |
| Upgrade | [Migration Guides](user-guide/migration.md), [Changelog](changelog.md) |
