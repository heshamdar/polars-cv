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

## Branch, rejoin, optimise: cutting out the foreground

Pipelines compose into a graph. This one cuts a cup and saucer out of a photo
with a transparent background: two cues find the china (bright pixels, and the
saturated red that the brown table lacks), their union is traced to an outline,
and the outline is rasterized back into an alpha channel for the photo.

![A photo of an espresso cup on a saucer on a wooden table, and the same photo with the table made transparent.](https://raw.githubusercontent.com/heshamdar/polars-cv/main/polars-cv/docs/assets/readme-cutout.png)

```python
from pathlib import Path

import polars as pl
from polars_cv import Pipeline

base = Pipeline().source("file_path").resize_max(512)
img = pl.col("path").cv

# Two cues, each written as if it stood alone. The threshold is a column.
bright = img.pipe(base.grayscale().threshold(190))
saturated = img.pipe(
    base.to_hsv().channel_select(1).threshold(pl.col("min_saturation"))
)

# Rejoin: either cue, cleaned up and traced; keep the largest outline's hull.
outline = bright.maximum(saturated).pipe(
    Pipeline().morphology_open(ksize=5).extract_contours().largest().convex_hull()
)

# The outline as an anti-aliased alpha channel joined to the photo's colours.
photo = img.pipe(base)
alpha = outline.rasterize(shape=photo, supersample=4)
cutout = photo.channel_merge(alpha).alias("cutout")

expr = cutout.merge_pipe(outline.area().alias("area"))
paths = sorted(str(p) for p in Path("photos").glob("*.jpg"))
out = (
    pl.LazyFrame({"path": paths, "min_saturation": 210})
    .with_columns(o=expr.sink({"cutout": "png", "area": "native"}))
    .unnest("o")
    .collect()
)
```

`expr.sink(..., return_expr=False).show_graph()` draws the graph that runs
(it needs `polars-cv[viz]` and Graphviz's `dot`):

![The optimised graph: one source node reads and resizes the image; one branch converts it to HSV and thresholds the saturation; a second node thresholds the brightness, takes the maximum with the saturation branch, cleans it up and traces the outline's hull; the hull feeds an area node and a rasterize node, whose alpha the photo's channels are merged with into the cutout.](https://raw.githubusercontent.com/heshamdar/polars-cv/main/polars-cv/docs/assets/readme-cutout-graph.svg)

Written out, this is eight nodes with three decodes; it runs as six with one.
The optimiser computes each distinct value once: the three pipelines built from `base` share one read
and resize (the `SOURCE` node), and each linear run of steps that nothing else
reads becomes a single node, so the brightness threshold runs straight through
`maximum` to the hull in one. Pass `opt_flags=False` to `sink` to see the
graph as written; the outputs are byte for byte the same. The whole graph is
one Polars expression: it runs in the lazy and streaming engines, per row,
with `min_saturation` read from each row's column, and returns the PNG and the
outline's area together.

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

## Checked against OpenCV, SciPy, NumPy, PIL and imagehash

<!-- parity:start -->
<!-- Generated by scripts/parity_report.py from the test suite; do not edit. -->
789 reference tests hold 114 ops to an independent implementation
(NumPy, OpenCV, SciPy, imagehash, matplotlib, pycocotools) or to a written definition, and all of them pass.
Each test states what it compares and how close the outputs must be, and
the [parity report](https://heshamdar.github.io/polars-cv/user-guide/parity/) lists every comparison with its tolerance.

| Checked against | Tests passing | Ops |
|---|---:|---|
| NumPy | 292 | 60 |
| Definition | 202 | 51 |
| OpenCV | 130 | 37 |
| Hand-computed | 48 | 21 |
| SciPy | 47 | 3 |
| imagehash | 23 | 1 |
| Python | 22 | 2 |
| matplotlib | 20 | 1 |
| pycocotools | 4 | 1 |
| Cross-check | 1 | 3 |

End to end, the [benchmark](https://github.com/heshamdar/polars-cv/tree/main/benchmarks)
holds every library to PIL's output before timing it. Mean / worst difference
from PIL (0–255 levels, or hash bits):

| Workload (40 COCO images) | Unit | polars-cv | OpenCV | torchvision | Daft |
|---|---|---:|---:|---:|---:|
| decode → bilinear 224² → gray | levels | 0.17 / 2 | 4.18 / 127 | 0.66 / 2 | 1.19 / 23 |
| thumbnail → JPEG, re-decoded | levels | 2.00 / 4.32 | 5.70 / 12.27 | 0.03 / 0.44 | n/a |
| 64-bit pHash | bits | 0.15 / 2 | 33.62 / 44 | n/a | 0.50 / 2 |
| per-row crop + bilinear resize | levels | 0.16 / 2 | 4.78 / 136 | 0.00 / 1 | n/a |

OpenCV's larger gap is its `INTER_LINEAR` resize, which doesn't antialias when
downscaling, not a bug. polars-cv and Daft decode JPEGs with pure-Rust
decoders, so they differ from libjpeg-turbo by a level or two at chroma edges.
<!-- parity:end -->

## What's included

| Area | Examples |
|---|---|
| Resize & geometry | `resize` (incl. `filter="area"`), `resize_max`, `resize_like`, `crop`/`crop_clip`/`crop_pad`, `pad`, `letterbox`, `rotate`, `flip`, `warp_affine` |
| Colour & intensity | `grayscale`, `to_hsv`, `to_lab`, `adjust_contrast`, `adjust_gamma`, `equalize_histogram` |
| Filters & morphology | `blur`, `sharpen`, `sobel`, `canny`, `convolve2d`, `median_filter`/`rank_filter`, `background_divide`, `erode`/`dilate` (square or disk), `distance_transform` |
| Arithmetic | `normalize`/`normalize_preset`/`normalize_percentile`, `cast`, `clamp`, element-wise math, and `add`/`blend`/`apply_mask` between pipelines |
| Analysis | `histogram`, `perceptual_hash`, `reduce`/`reduce_mean`/`max`/`percentile`/…, `reduce_masked`, `block_reduce`, `statistics(mask=)`, `region_stats` |
| Contours, points, boxes | `extract_contours`, `area`, `iou`, `hausdorff_distance`, `pairwise_iou` via `.contour`/`.point`/`.bbox` |
| Drawing | `rasterize` (anti-aliased), `stroke`, `offset`, `smooth`, `union`/`intersection`/`difference`, `coordinate_grid`, `apply_lut` palettes |
| Header metadata | `.cv.image_info()`, `.cv.width()`, `.cv.height()` without decoding |
| Detection metrics | COCO-style AP/mAP, PR curves, FROC/LROC, bootstrap intervals in `polars_cv.metrics` |

The full list is in the
[operations guide](https://heshamdar.github.io/polars-cv/user-guide/operations/image-ops/)
and the [API reference](https://heshamdar.github.io/polars-cv/api/pipeline/);
the [drawing guide](https://heshamdar.github.io/polars-cv/user-guide/drawing/)
builds the logo above from these ops.

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

## Project status

> **ℹ️ Note:** This is a largely AI-developed project and still in its early
> stages. Use at your own discretion. What keeps it honest is the
> [parity report](https://heshamdar.github.io/polars-cv/user-guide/parity/):
> every op is checked in CI against an independent implementation.

polars-cv is 0.x: a minor release may break the API. Breaking changes are
listed in the [changelog](https://github.com/heshamdar/polars-cv/blob/main/CHANGELOG.md),
each with a [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration/).
Pin a minor version (`polars-cv~=0.41.0`) if you depend on it. Outputs are
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
