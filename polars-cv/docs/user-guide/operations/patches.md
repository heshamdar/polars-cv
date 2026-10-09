# Patches

Cut images into patches as ordinary Polars rows, one row per patch, and run a
pipeline on each patch: for training on patches of photos, scans or
microscopy, for tiling large images for inference, or for whole-slide images.

The recipe is three steps, all in Polars:

1. **List the patches.** `polars_cv.patch_grid()` gives each image's grid as a
   list of cells; `explode` makes one row per cell.
2. **Process each patch.** A pipeline that starts with a `crop` driven by the
   cell's columns, followed by any ops, runs once per patch row.
3. **Do anything Polars does.** Filter, sample, join, group and pivot the
   patch rows like any other rows.

## The recipe

### Listing patches: `patch_grid`

```python
import polars as pl
import polars_cv as cv

cells = (
    df.with_columns(cell=cv.patch_grid("height", "width", size=256))
    .explode("cell")
    .unnest("cell")
)
```

Each cell is `{row, col, top, left, height, width}` (all `UInt32`), in
row-major order. The field names are `crop`'s keywords.

| Argument | Meaning |
|---|---|
| `height`, `width` | The image size per row: a column name, an expression (such as `pl.col("path").cv.height()`) or an int |
| `size` | Patch size, `n` or `(rows, cols)` |
| `stride` | Distance between patch origins; defaults to `size`. Smaller overlaps patches, larger leaves gaps |
| `edge` | `"drop"` leaves a remainder too small for a whole patch uncovered; `"shift"` adds one patch aligned to the far edge |

Every patch is whole and inside the image. An image smaller than a patch has an
empty list, and a null size gives a null row.

### Processing patches: a crop, then any ops

```python
from polars_cv import Pipeline

pipe = (
    Pipeline()
    .source("image_bytes")  # or "file_path"
    .crop(top=pl.col("top"), left=pl.col("left"), height=256, width=256)
    .resize(height=224, width=224)
    .cast("f32")
    .scale(1 / 255)
)
patches = cells.with_columns(x=pl.col("image").cv.pipe(pipe).sink("torch"))
```

Every op works on a patch exactly as on an image: the crop makes each row a
`[256, 256, C]` patch.

**Each image is decoded once, whatever the number of its patches.** The rows
of one image share one decode within a query. Images are matched by their
bytes, or their path, which is also read once. So 64 patches of a JPEG cost
one JPEG decode and 64 crops, not 64 decodes. A tiled TIFF goes further: each
patch decodes only the tiles under it (see [Large images](#large-images-tiled-tiffs-and-whole-slide-images)).

- The result is exactly the crop of the whole decode.
- **A window outside the image** is that row's error, with `crop`'s usual
  message, under the pipeline's `on_error`.
- **A null window parameter** nulls or fails the row under `on_null_param`.
- Per-row parameters make augmentation a column: random offsets
  (`top=pl.col("top") + jitter`), a per-row `rotate(angle=...)` or contrast.
- Up to 1 GiB of decoded images is kept per query; past that, the least
  recently used are dropped, and decoded again if a later patch needs one.

### Putting patch results back on the grid

Per-patch scores go back onto each image's grid with a plain pivot, which
gives a heatmap:

```python
scores = patches.with_columns(score=...)
heat = scores.filter(pl.col("image_id") == 0).pivot(on="col", index="row", values="score")
```

## One batch per image: `tile`

When you want an image's patches together, as one `[N, height, width, C]`
array per row (a batch for a model), `tile` cuts the decoded image in one
step:

```python
pipe = Pipeline().source("image_bytes", dtype="u8").tile(height=224, width=224, edge="shift")
batches = df.select(pl.col("image").cv.pipe(pipe).sink("numpy"))
```

It uses the same grid as `patch_grid`, so patch `i` is the grid's cell `i`. The
strides and `edge` may be expressions. Ops that read `[H, W, C]` (`resize`,
`grayscale`, ...) refuse its rank-4 output when the pipeline is built: for
per-patch work, use the recipe above, or explode `tile`'s output and run a
second pipeline over it.

### Choosing a route

Measured on 8 images of 2048 x 2048 cut into 512 patches of 256, then resized
to 224 and scaled to `f32` (4 threads; times relative to the lower bound of
decoding each image once plus the ops on already-cut patches):

| Route | JPEG | PNG |
|---|---|---|
| The recipe: `patch_grid` → `explode` → one pipeline (`crop` → ops) | 1.4x | 1.4x |
| `tile` → `sink("array")` → `explode` → `source("array")` → ops | 1.1x | 1.0x |
| `tile` → `sink("numpy")` → split in NumPy → `source("array")` → ops | 1.8x | 1.9x |
| `tile` → `sink("list")` → `explode` → `source("list")` → ops | 3.0x | 3.3x |

The recipe is the general route: any image sizes, per-row windows and
parameters, one pipeline. The `array` route is slightly faster but needs every
image's size known when the pipeline is built (an `assert_shape` before
`tile`), since an `array` sink's shape is fixed. Avoid the `list` route: its
nested lists cost more than the decode.

## Large images: tiled TIFFs and whole-slide images

A crop that is the first op of a pipeline is handed to the decoder (the
`roi_decode` optimization, on by default; spatial-window pushdown also moves a
crop ahead of the pointwise ops before it). For a TIFF it reads only the patch:

- **A tiled or strip TIFF:** only the chunks under the window are decoded. Read
  by path, only the file's header and those chunks are read, locally or from
  S3, GCS, Azure or HTTP (by byte range).
- **A window outside the image, or a null window parameter**, is decided
  from the header: the row fails or nulls without any pixels decoded.
- **The cost of a patch** is the file's header, the patch's tiles and their
  entries in the tile index, whatever the size of the slide.
- **Remote slides** are read ahead. While a row decodes, the windows of the rows
  after it are already being fetched, up to polars' concurrency budget. That
  budget defaults to 10 requests in flight; for a high-latency store, raise it
  with `POLARS_CONCURRENCY_BUDGET` (for example `64`). A slide's header is read
  once and reused by later queries while the object's ETag (or Last-Modified
  time) is unchanged.

### Pyramid levels

A whole-slide image stores reduced copies of itself. `.cv.slide_info()` lists
them from the header:

```python
info = df.with_columns(info=pl.col("path").cv.slide_info())
levels = info.select(pl.col("info").struct.field("levels")).explode("levels").unnest("levels")
```

Each level has `level`, `width`, `height`, `downsample`, `tile_width` and
`tile_height`. The struct also carries `mpp_x` and `mpp_y`, level 0's microns
per pixel, from an Aperio description or a resolution in pixels per
centimetre.

`source(level=k)` decodes level `k`. A crop after it names pixels of *that*
level:

```python
low = Pipeline().source("file_path", level=2).crop(
    top=pl.col("low_top"), left=pl.col("low_left"), height=64, width=64
)
```

`level` may be an expression, so each row can read its own level. A level the
file lacks, or any level above 0 of an image that is not a pyramid, is the
row's decode error.

### Example: tissue patches of a slide

Find tissue on a small level, then read only the full-resolution patches that
hold it. `examples/14_whole_slide_patches.py` runs this end to end.

```python
scale = 4  # the small level's downsample, from slide_info
cells = (
    slides.with_columns(cell=cv.patch_grid(pl.col("h"), pl.col("w"), size=256))
    .explode("cell")
    .unnest("cell")
    .with_columns(low_top=pl.col("top") // scale, low_left=pl.col("left") // scale)
)
tissue = (
    Pipeline()
    .source("file_path", level=2)
    .crop(top=pl.col("low_top"), left=pl.col("low_left"), height=64, width=64)
    .grayscale()
    .threshold(220)
    .invert()
    .reduce_mean()
)
keep = cells.with_columns(
    tissue=pl.col("path").cv.pipe(tissue).sink("native") / 255
).filter(pl.col("tissue") > 0.5)
```

Per-patch scores go back onto the grid with a plain pivot, which gives a
heatmap:

```python
heat = keep.pivot(on="col", index="row", values="tissue")
```

### TIFF formats

| Supported | Not supported |
|---|---|
| Tiled and strip TIFF, including BigTIFF | SubIFD pyramids (OME-TIFF): read as one level |
| Pyramids stored as top-level IFDs (Aperio SVS, generic pyramidal TIFF) | JPEG 2000 tiles |
| Uncompressed, LZW, Deflate, PackBits and JPEG chunks, with horizontal differencing | Vendor formats (MRXS, NDPI, iSyntax, DICOM-WSI) |
| u8/u16 gray, gray + alpha, RGB, RGBA; f32/f64 gray, RGB | |

Other TIFF layouts (palette images, the floating-point predictor) decode whole.
Read by path with a crop or a level, such a file is read whole only when it is
within the 256 MiB decode limit; a larger one is the row's error, naming the
layout.
For a vendor format, read regions with a library such as openslide-python or
tiffslide and hand the bytes or arrays to `source("image_bytes")` or
`source("array")`.
