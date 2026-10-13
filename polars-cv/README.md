# polars-cv

**Image processing as Polars expressions: decode, transform and analyse images
in a DataFrame column, in Rust, on every core, without a Python loop.**

[![PyPI](https://img.shields.io/pypi/v/polars-cv)](https://pypi.org/project/polars-cv/)
[![Downloads](https://img.shields.io/pypi/dm/polars-cv)](https://pypistats.org/packages/polars-cv)
[![CI](https://github.com/heshamdar/polars-cv/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/heshamdar/polars-cv/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-heshamdar.github.io%2Fpolars--cv-blue)](https://heshamdar.github.io/polars-cv/)
[![License](https://img.shields.io/pypi/l/polars-cv)](https://pypi.org/project/polars-cv/)
[![Python](https://img.shields.io/pypi/pyversions/polars-cv)](https://pypi.org/project/polars-cv/)

![Top row: one decoded image feeding five pipelines (source, grayscale, blur, canny, histogram equalisation). Bottom row: one pipeline whose rotation angle and blur sigma come from per-row columns.](https://raw.githubusercontent.com/heshamdar/polars-cv/main/polars-cv/docs/assets/readme-figure.png)

<sub>Every panel above is computed by polars-cv from a CC0 photo
([`scripts/make_readme_figure.py`](https://github.com/heshamdar/polars-cv/blob/main/polars-cv/scripts/make_readme_figure.py)).</sub>

## 30-second example

A folder of photos to centre-cropped 224×224 training tensors plus JPEG
thumbnails. The crop box is different for every image: it is computed from
each file's own width and height, read from the header without decoding.

```python
from pathlib import Path

import polars as pl
from polars_cv import Pipeline

df = pl.DataFrame({"path": sorted(str(p) for p in Path("photos").glob("*.jpg"))})
df = df.with_columns(w=pl.col("path").cv.width(), h=pl.col("path").cv.height())

side = pl.min_horizontal("w", "h")  # a per-row value, used as an op parameter
square = (
    Pipeline()
    .source("file_path", dtype="u8")
    .crop(top=(pl.col("h") - side) // 2, left=(pl.col("w") - side) // 2,
          height=side, width=side)
    .resize(height=224, width=224, filter="bilinear")
)
thumb = Pipeline().source("file_path").resize_max(128)

out = df.with_columns(
    x=pl.col("path").cv.pipe(square).sink("array", shape=[224, 224, 3]),
    thumb=pl.col("path").cv.pipe(thumb).sink("jpeg", quality=85),
)
batch = out["x"].to_numpy()  # (n_images, 224, 224, 3) uint8
```

## Why polars-cv

- **Images are a column.** Paths, bytes, labels, metadata and results sit in one
  DataFrame, so filtering, joining and `group_by` on image properties is plain
  Polars, not a second data structure kept in sync with a list of arrays.
- **Polars runs it.** Pipelines are expressions: they run lazily, in the
  streaming engine for larger-than-memory data, with the rows of each call
  spread over every core by Rust code. There is no Python per image.
- **Parameters can be expressions.** Any parameter that does not change the
  output's rank, dtype or domain takes a Polars expression evaluated per row:
  crop boxes from annotations, sizes from metadata, thresholds from a lookup.

## Benchmarks

A reproducible comparison with PIL, OpenCV, torchvision, Daft and Hugging Face
`datasets` on COCO val2017 lives in
[`benchmarks/`](https://github.com/heshamdar/polars-cv/tree/main/benchmarks):
methodology, parity tests, raw results and the command to reproduce them.

## Install

```bash
pip install polars-cv      # or: uv add polars-cv
```

Wheels are published for Linux (x86_64, aarch64) and macOS (Apple silicon),
for CPython 3.11+ (one `abi3` wheel), and need Polars 2.0 or newer. Other
platforms build from the source distribution, which needs a Rust toolchain
(1.96+).

## Quickstart

A `Pipeline` is a reusable, immutable description of the work. `.cv.pipe()`
binds it to a column, and `.sink()` chooses the output: `"numpy"` (a struct of
bytes, shape and dtype that `polars_cv.numpy_from_column` reads without
copying), `"array"` (a fixed-shape Polars `Array`), `"list"`, `"png"`/`"jpeg"`
bytes, or `"blob"` (polars-cv's own format, to resume a pipeline later).

```python
import io

import numpy as np
import polars as pl
from PIL import Image
from polars_cv import Pipeline, numpy_from_column


def png(h: int, w: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), np.uint8)).save(buf, "PNG")
    return buf.getvalue()


df = pl.DataFrame({
    "image": [png(300, 400), png(480, 640)],
    "target": [128, 256],  # one output size per row
})

pipe = (
    Pipeline()
    .source("image_bytes")  # "auto" (the default) infers this from the dtype
    .resize(height=pl.col("target"), width=pl.col("target"))
    .grayscale()
)
out = df.with_columns(gray=pl.col("image").cv.pipe(pipe).sink("numpy"))
arrays = numpy_from_column(out["gray"])  # [(128, 128, 1), (256, 256, 1)] uint8 views
```

A null in a parameter column fails the query by default, naming the column and
row; `pipe.on_null_param("null")` nulls only the affected outputs instead. For
a fallback value, fill it in the expression: `pl.col("target").fill_null(224)`.

PNG and JPEG decode to `u8`, 16-bit PNG to `u16`, TIFF to whatever it stores,
so an image source's dtype is only known at run time. The `"array"` and
`"list"` sinks need it at planning time: give it with `source(..., dtype="u8")`,
a `.cast(...)`, or an op that fixes it (`normalize`, `threshold`, …).
See the [Sources guide](https://heshamdar.github.io/polars-cv/user-guide/concepts/sources/).

## Scaling and threads

- **Eager calls already use every core.** Each `.cv.pipe(...)` call splits its
  rows over the plugin's own thread pool, so a plain
  `DataFrame.with_columns(...)` is multi-threaded.
- **Both pools are sized by `POLARS_MAX_THREADS`.** The plugin links its own
  copy of Polars, so its pool is separate from the host's, but the threads
  running polars-cv rows share one budget of `POLARS_MAX_THREADS`. Set it
  before importing Polars. If a query with many plugin expressions shows a
  load average well above the core count, lower it.
- **Use the streaming engine for data larger than memory.** A lazy
  `.collect()` uses it by default since Polars 2.0. It hands polars-cv one
  morsel per call, and each call holds its output rows until it returns, so
  peak memory is roughly rows per morsel × output bytes per row × concurrent
  calls. For image tables, write Parquet with row groups sized by output bytes
  or cap the morsel with `pl.Config.set_streaming_chunk_size(n)`.

```python
from pathlib import Path

import polars as pl
from polars_cv import Pipeline

pipe = Pipeline().source("file_path", dtype="u8").resize(height=64, width=64)
lf = pl.LazyFrame({"path": sorted(str(p) for p in Path("photos").glob("*.jpg"))})
for batch in (
    lf.select(x=pl.col("path").cv.pipe(pipe).sink("array", shape=[64, 64, 3]))
    .collect_batches(chunk_size=2)  # streaming engine, one batch at a time
):
    x = batch["x"].to_numpy()  # (rows, 64, 64, 3) uint8, no copy
```

The full story (morsel sizing, spilling, cheap first passes with
`thumbnail()` and header-only metadata) is in
[Streaming & Scaling](https://heshamdar.github.io/polars-cv/user-guide/concepts/streaming/).

## Operations

<details>
<summary>All operations (click to expand)</summary>

- **Image**: `resize`, `resize_scale`, `resize_to_height`, `resize_to_width`, `resize_max`, `resize_min`, `thumbnail`, `grayscale`, `blur`, `threshold`, `crop`, `rotate`, `pad`, `pad_to_size`, `letterbox`, `flip`, `flip_h`, `flip_v`.
- **Color**: `convert_color`, `to_hsv`, `to_lab`, `to_bgr`, `to_ycbcr`.
- **Channels**: `channel_select`, `channel_swap`.
- **Intensity**: `adjust_contrast`, `adjust_gamma`, `adjust_brightness`, `invert`.
- **Convolution & Edge Detection**: `convolve2d`, `sobel`, `laplacian`, `sharpen`, `canny`.
- **Morphology**: `erode`, `dilate`, `morphology_open`, `morphology_close`, `morphology_gradient`.
- **Affine Transforms**: `warp_affine`, `shear`, `rotate_and_scale`.
- **Enhancement**: `equalize_histogram`.
- **Compute**: `normalize`, `scale`, `add_constant`, `subtract_constant`, `clamp`, `clamp_min`, `clamp_max`, `relu`, `cast`.
- **Element-wise math**: `abs`, `neg`, `sign`, `sqrt`, `square`, `reciprocal`, `round`, `floor`, `ceil`, `trunc`.
- **Binary** (on `LazyPipelineExpr`, between two pipelines): `add`, `subtract`, `multiply`, `divide`, `blend`, `maximum`, `minimum`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `apply_mask`, `apply_contour_mask`, `channel_merge`.
- **Layout**: `transpose`, `reshape`, `assert_shape`.
- **Geometry**: `extract_contours`, `rasterize`, `simplify`, `convex_hull`, `scale_contour`, `largest`, `area`, `perimeter`, `centroid`, `bounding_box`; the `.contour` namespace adds `iou`, `dice`, `hausdorff_distance`, `contains_point`, `boundary_distances` and more.
- **Points**: `normalize`, `translate`, `scale`, `rotate`, `distance`, `manhattan_distance`, `distance_to_contour`, `signed_distance_to_contour`, `nearest_point_on_contour`, `angle_to`, `midpoint`, `interpolate`, `within_bbox`.
- **Bounding Boxes**: `pairwise_iou`, `correspond` (via the `.bbox` namespace).
- **Analysis**: `histogram`, `perceptual_hash`, `extract_shape`, `label_reduce`.
- **Reductions**: `reduce_sum`, `reduce_mean`, `reduce_std`, `reduce_max`, `reduce_min`, `reduce_argmax`, `reduce_argmin`, `reduce_percentile`, `reduce_popcount`, and `statistics()` for several in one pass.
- **Metadata** (header only, no decode): `.cv.width()`, `.cv.height()`, `.cv.channels()`, `.cv.image_dtype()`.
- **Byte access**: `.cv.read_bytes()` reads a path column's bytes (local or cloud) without decoding.
- **Display**: `show_images()` for Jupyter notebooks.

Sources: `image_bytes`, `file_path` (local paths, `http(s)://`, `s3://`,
`gs://`, `az://`), `blob`, `list`/`array`, `contour`. See the
[operations guide](https://heshamdar.github.io/polars-cv/user-guide/operations/image-ops/)
and the [API reference](https://heshamdar.github.io/polars-cv/api/pipeline/).

</details>

## Detection metrics

`polars_cv.metrics` evaluates detections from one row per object (boxes,
polygons or masks) to COCO-style AP/mAP, with bootstrap intervals that
resample images:

```python
import polars as pl
from polars_cv.metrics import evaluate_detections

gts = pl.DataFrame({
    "image_id": ["a", "a", "b"], "class_id": [0, 1, 0],
    "x1": [10.0, 50.0, 0.0], "y1": [10.0, 50.0, 0.0],
    "x2": [40.0, 90.0, 30.0], "y2": [40.0, 90.0, 30.0],
})
preds = pl.DataFrame({
    "image_id": ["a", "a", "b", "b"], "class_id": [0, 1, 0, 0],
    "score": [0.9, 0.8, 0.7, 0.3],
    "x1": [12.0, 48.0, 1.0, 60.0], "y1": [11.0, 52.0, 2.0, 60.0],
    "x2": [41.0, 88.0, 29.0, 80.0], "y2": [39.0, 91.0, 31.0, 80.0],
})

report = evaluate_detections(preds, gts, geometry=("x1", "y1", "x2", "y2"), box_format="xyxy")
report.summary    # map, map_50, map_75, mar
report.per_class  # AP and recall per class
report.ci("map")  # bootstrap interval
```

It also covers precision-recall curves, FROC and LROC, confusion counts at a
threshold, heatmap and segmentation evaluation, and per-image weights
(scikit-learn `sample_weight` semantics). See the
[metrics guide](https://heshamdar.github.io/polars-cv/user-guide/operations/metrics/).

## When not to use polars-cv

- **Augmentation inside a training loop.** There are no random transforms and
  no GPU execution. Use torchvision v2, Kornia or DALI there; polars-cv fits the
  deterministic preprocessing before it.
- **One image at a time, latency first.** Each call goes through Polars'
  expression machinery. For a single image in an interactive tool, calling
  OpenCV or PIL directly is simpler and has less overhead.
- **Operations it does not have.** Video decoding, model inference and the
  long tail of OpenCV are out of scope. Check the operations list first.
- **Windows or Intel macOS without a Rust toolchain.** There are no wheels for
  them yet.
- **Production systems that cannot absorb breaking changes.** See below.

## Project status

> **ℹ️ Note:** This is a largely AI-developed project and still in its early
> stages. Use at your own discretion.

What there is to check that against:

- **Tests**: about 6,000 Python tests (5,800 in the per-push lane) and over
  850 Rust unit tests, including reference tests against NumPy, SciPy,
  OpenCV and `imagehash`, a golden corpus, and schema-parity tests that
  check every op's planned output type against what it produces. CI fails
  below 95% line coverage of the Python package.
- **CI**: every pull request runs Linux on Python 3.11, 3.12 and 3.13, both
  Polars engines (streaming and in-memory), the oldest declared dependency
  versions, `clippy`, `rustfmt`, `ruff`, `ty`, `cargo-deny` and a strict docs
  build. Pushes to `main` add macOS.
- **Releases**: 28 minor releases from 0.10.0 (2026-06-13) to 0.37.0
  (2026-10-11), each in the [changelog](https://github.com/heshamdar/polars-cv/blob/main/CHANGELOG.md).
- **Versioning**: the project aims to follow
  [Semantic Versioning](https://semver.org/). While it is 0.x, a minor release
  may break the API. Breaking changes are listed under *Breaking changes* in
  the changelog, and every minor release since 0.29 has a
  [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.37/).
  Pin a minor version (`polars-cv~=0.37.0`) if you depend on it.

## Contributing

Issues and pull requests are welcome. [`CONTRIBUTING.md`](https://github.com/heshamdar/polars-cv/blob/main/CONTRIBUTING.md)
covers the build (`maturin develop` against a Rust 1.96+ toolchain), and
`scripts/verify.sh` runs every check CI runs.

## Citation

If you use polars-cv in research, please cite the version you used:

```bibtex
@software{dar_polars_cv,
  author = {Dar, Hesham},
  title  = {polars-cv: Image processing as Polars expressions},
  url    = {https://github.com/heshamdar/polars-cv},
  year   = {2026}
}
```

Licensed under either of [MIT](https://github.com/heshamdar/polars-cv/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/heshamdar/polars-cv/blob/main/LICENSE-APACHE), at your option.
