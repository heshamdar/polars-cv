<p align="center">
  <img alt="polars-cv" src="https://raw.githubusercontent.com/heshamdar/polars-cv/main/polars-cv/docs/assets/logo/polars-cv-halftone.png" width="600">
</p>

<p align="center">
  <strong>Image processing as Polars expressions: decode, transform and analyse
  images in a DataFrame column, in Rust, on every core, without a Python loop.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/polars-cv/"><img alt="PyPI" src="https://img.shields.io/pypi/v/polars-cv"></a>
  <a href="https://pypistats.org/packages/polars-cv"><img alt="Downloads" src="https://img.shields.io/pypi/dm/polars-cv"></a>
  <a href="https://github.com/heshamdar/polars-cv/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/heshamdar/polars-cv/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://heshamdar.github.io/polars-cv/"><img alt="Docs" src="https://img.shields.io/badge/docs-heshamdar.github.io%2Fpolars--cv-blue"></a>
  <a href="https://pypi.org/project/polars-cv/"><img alt="License" src="https://img.shields.io/pypi/l/polars-cv"></a>
  <a href="https://pypi.org/project/polars-cv/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/polars-cv"></a>
</p>

![Top row: one graph that decodes the image once and produces five outputs (source, grayscale, blur, canny edges, histogram equalisation). Bottom row: one pipeline whose rotation angle and blur sigma come from per-row columns.](https://raw.githubusercontent.com/heshamdar/polars-cv/main/polars-cv/docs/assets/readme-figure.png)

<sub>Every panel is computed by polars-cv from a CC0 photo
([`scripts/make_readme_figure.py`](https://github.com/heshamdar/polars-cv/blob/main/polars-cv/scripts/make_readme_figure.py)).
`equalize_histogram()` is a contrast stretch: it remaps intensities so they
spread evenly over 0–255.</sub>

## Install

```bash
pip install polars-cv      # or: uv add polars-cv
```

Wheels for Linux (x86_64, aarch64) and macOS (Apple silicon), CPython 3.11+,
Polars 2.0+. Other platforms build from source with Rust 1.96+.

## Quickstart

A `Pipeline` describes the work, `.cv.pipe()` applies it to a column, and
`.sink()` picks the output format.

```python
from pathlib import Path

import polars as pl
from polars_cv import Pipeline

df = pl.DataFrame({"path": sorted(str(p) for p in Path("photos").glob("*.jpg"))})

pipe = Pipeline().source("file_path").resize(height=224, width=224).grayscale()
out = df.with_columns(gray=pl.col("path").cv.pipe(pipe).sink("numpy"))
```

Sources include `image_bytes`, local paths and `s3://`/`gs://`/`az://`/`http(s)://`
URLs. Sinks include `numpy`, `array`, `list`, `png` and `jpeg`. See
[Sources](https://heshamdar.github.io/polars-cv/user-guide/concepts/sources/)
and [Pipelines](https://heshamdar.github.io/polars-cv/user-guide/concepts/pipelines/).

## One decode, several outputs, per-row parameters

A folder of photos to centre-cropped 224×224 training tensors plus JPEG
thumbnails. Each file is decoded once and feeds both outputs, and the crop box
is computed per row from each image's header.

```python
from pathlib import Path

import polars as pl
from polars_cv import Pipeline

df = pl.DataFrame({"path": sorted(str(p) for p in Path("photos").glob("*.jpg"))})
df = df.with_columns(pl.col("path").cv.image_info().struct.unnest())  # header only

side = pl.min_horizontal("width", "height")  # a per-row value
img = pl.col("path").cv.pipe(Pipeline().source("file_path", dtype="u8"))
x = img.pipe(
    Pipeline()
    .crop(top=(pl.col("height") - side) // 2, left=(pl.col("width") - side) // 2,
          height=side, width=side)
    .resize(height=224, width=224, filter="bilinear")
).alias("x")
thumb = img.pipe(Pipeline().resize_max(128)).alias("thumb")

out = df.with_columns(
    o=x.merge_pipe(thumb).sink({
        "x": ("array", {"shape": [224, 224, 3]}),
        "thumb": ("jpeg", {"quality": 85}),
    })
).unnest("o")
batch = out["x"].to_numpy()  # (n_images, 224, 224, 3) uint8
```

Any parameter that does not change the output's rank, dtype or domain accepts
a Polars expression. See
[Multi-output pipelines](https://heshamdar.github.io/polars-cv/user-guide/composition/multi-output/).

## Why polars-cv

- **Images are a column.** Paths, bytes, labels, metadata and results sit in one
  DataFrame, so filtering, joining and `group_by` on image properties is plain
  Polars, not a second data structure kept in sync with a list of arrays.
- **Polars runs it.** Pipelines are expressions: they run lazily, in the
  streaming engine for larger-than-memory data, with each call's rows spread
  over every core by Rust code. There is no Python per image.
- **Parameters can be expressions.** Crop boxes from annotations, sizes from
  metadata, thresholds from a lookup, evaluated per row.

## Benchmarks

5,000 COCO val2017 images on 4 cores, against PIL, OpenCV (process pool),
torchvision (DataLoader), Daft and Hugging Face `datasets`. Each is written the
way its users would write it and checked against PIL's output before timing.

![Images per second on four workloads at 4 threads: polars-cv leads ML preprocessing and per-row crop/resize, OpenCV leads thumbnails, and the two are close on perceptual hashing.](https://raw.githubusercontent.com/heshamdar/polars-cv/main/benchmarks/results/summary.svg)

| Workload | polars-cv (streaming) | Fastest alternative | polars-cv speed-up | Peak memory, polars-cv vs that alternative |
|---|---:|---:|---:|---:|
| Decode → resize 224² → gray → NumPy | **1,025 img/s** | torchvision 701 | **1.5×** | **849 MiB** vs 1,094 |
| Per-row crop box + per-row size | **1,132 img/s** | OpenCV 751 | **1.5×** | **686 MiB** vs 2,238 |
| Size + pHash → duplicate groups | 1,048 img/s | OpenCV 1,180 | 0.89× | 143 MiB vs 120 |
| Thumbnail → JPEG | 728 img/s | OpenCV 939 | 0.78× | 250 MiB vs 194 |

polars-cv trails OpenCV on thumbnails because its JPEG encoder is slower than
libjpeg-turbo. Methodology, every table and the command to reproduce them are
in [`benchmarks/`](https://github.com/heshamdar/polars-cv/tree/main/benchmarks).

## What's included

| Area | Examples |
|---|---|
| Resize & geometry | `resize`, `resize_max`, `crop`, `pad`, `letterbox`, `rotate`, `flip`, `warp_affine` |
| Colour & intensity | `grayscale`, `to_hsv`, `to_lab`, `adjust_contrast`, `adjust_gamma`, `equalize_histogram` |
| Filters & morphology | `blur`, `sharpen`, `sobel`, `canny`, `convolve2d`, `erode`/`dilate` (square or disk), `distance_transform` |
| Arithmetic | `normalize`, `cast`, `clamp`, element-wise math, and `add`/`blend`/`apply_mask` between pipelines |
| Analysis | `histogram`, `perceptual_hash`, `reduce_mean`/`max`/`percentile`/…, `statistics()` |
| Contours, points, boxes | `extract_contours`, `area`, `iou`, `hausdorff_distance`, `pairwise_iou` via `.contour`/`.point`/`.bbox` |
| Drawing | `rasterize` (anti-aliased), `stroke`, `offset`, `smooth`, `union`/`intersection`/`difference`, `coordinate_grid`, `apply_lut` palettes |
| Header metadata | `.cv.image_info()`, `.cv.width()`, `.cv.height()` without decoding |
| Detection metrics | COCO-style AP/mAP, PR curves, FROC/LROC, bootstrap intervals in `polars_cv.metrics` |

The full list is in the
[operations guide](https://heshamdar.github.io/polars-cv/user-guide/operations/image-ops/)
and the [API reference](https://heshamdar.github.io/polars-cv/api/pipeline/);
the [drawing guide](https://heshamdar.github.io/polars-cv/user-guide/drawing/)
builds the logo above from these ops.

```python
import polars as pl
from polars_cv.metrics import evaluate_detections

gts = pl.DataFrame({
    "image_id": ["a", "a", "b"], "class_id": [0, 1, 0],
    "x1": [10.0, 50.0, 0.0], "y1": [10.0, 50.0, 0.0],
    "x2": [40.0, 90.0, 30.0], "y2": [40.0, 90.0, 30.0],
})
preds = gts.with_columns(score=pl.Series([0.9, 0.8, 0.7]), x1=pl.col("x1") + 2)

report = evaluate_detections(preds, gts, geometry=("x1", "y1", "x2", "y2"), box_format="xyxy")
report.summary    # map, map_50, map_75, mar
```

## Larger-than-memory data

Eager calls already use every core. For data that does not fit in memory, use
a `LazyFrame`: `.collect()` streams by default, and `.collect_batches()` hands
you one batch at a time.

```python
from pathlib import Path

import polars as pl
from polars_cv import Pipeline

pipe = Pipeline().source("file_path", dtype="u8").resize(height=64, width=64)
lf = pl.LazyFrame({"path": sorted(str(p) for p in Path("photos").glob("*.jpg"))})
for batch in (
    lf.select(x=pl.col("path").cv.pipe(pipe).sink("array", shape=[64, 64, 3]))
    .collect_batches(chunk_size=2)
):
    x = batch["x"].to_numpy()  # (rows, 64, 64, 3) uint8
```

Thread counts, morsel sizing and memory budgets are covered in
[Streaming & Scaling](https://heshamdar.github.io/polars-cv/user-guide/concepts/streaming/).

## When not to use polars-cv

- **Random augmentation in a training loop, or GPU execution.** Use
  torchvision v2, Kornia or DALI there; polars-cv fits the deterministic
  preprocessing before it.
- **One image at a time, latency first.** Calling OpenCV or PIL directly has
  less overhead.
- **Video, model inference, or the long tail of OpenCV.** Out of scope.
- **Windows or Intel macOS without a Rust toolchain.** No wheels yet.

## Project status

> **ℹ️ Note:** This is a largely AI-developed project and still in its early
> stages. Use at your own discretion.

polars-cv is 0.x: a minor release may break the API. Breaking changes are
listed in the [changelog](https://github.com/heshamdar/polars-cv/blob/main/CHANGELOG.md),
each with a [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration/).
Pin a minor version (`polars-cv~=0.38.0`) if you depend on it. Outputs are
checked against NumPy, SciPy, OpenCV and `imagehash` in CI.

## Contributing

Issues and pull requests are welcome. [`CONTRIBUTING.md`](https://github.com/heshamdar/polars-cv/blob/main/CONTRIBUTING.md)
covers the build, and `scripts/verify.sh` runs every check CI runs.

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
