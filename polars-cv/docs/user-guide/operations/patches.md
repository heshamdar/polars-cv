# Patches & Whole-Slide Images

Cut images into patches with ordinary Polars rows. Each patch is one row. A
crop after the source reads only that patch's pixels, even from a pyramidal
TIFF of tens of gigabytes on S3.

The model is three steps:

1. **List the patches.** `polars_cv.patch_grid()` gives each image's grid as a
   list of cells; `explode` makes one row per cell.
2. **Read each patch.** A `crop` right after the source, driven by the cell's
   columns, decodes only the window. For a tiled TIFF (a whole-slide image) it
   reads only the tiles under the window.
3. **Do anything Polars does.** Filter, join, group and pivot the patch rows
   like any other rows.

## Listing patches: `patch_grid`

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

## Reading patches: a crop after the source

```python
from polars_cv import Pipeline

pipe = (
    Pipeline()
    .source("file_path")
    .crop(top=pl.col("top"), left=pl.col("left"), height=256, width=256)
    .resize(height=224, width=224)
)
patches = cells.with_columns(x=pl.col("path").cv.pipe(pipe).sink("torch"))
```

A crop that is the first op of a pipeline is handed to the decoder (the
`roi_decode` optimization, on by default). Spatial-window pushdown also moves a
crop ahead of the pointwise ops before it.

- **Any image:** the result is exactly the crop of the full decode.
- **A tiled or strip TIFF:** only the chunks under the window are decoded. Read
  by path, only the file's header and those chunks are read, locally or from
  S3, GCS, Azure or HTTP (by byte range).
- **A window outside the image** is that row's error, with `crop`'s usual
  message, under the source's `on_error`.

## Pyramid levels

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

## Example: tissue patches of a slide

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

## An image in memory: `tile`

When the image is already decoded, or small enough to decode whole, `tile` cuts
it into all its patches in one step: `[N, height, width, C]`.

```python
pipe = Pipeline().source("image_bytes", dtype="u8").tile(height=224, width=224, edge="shift")
rows = df.select(pl.col("image").cv.pipe(pipe).sink("list")).explode("image")
```

It uses the same grid as `patch_grid`, so patch `i` is the grid's cell `i`. The
strides and `edge` may be expressions. Ops that read `[H, W, C]` refuse its
rank-4 output when the pipeline is built, so put per-patch work before `tile`,
or explode and run a second pipeline over the patches.

## Formats

| Supported | Not supported |
|---|---|
| Tiled and strip TIFF, including BigTIFF | SubIFD pyramids (OME-TIFF): read as one level |
| Pyramids stored as top-level IFDs (Aperio SVS, generic pyramidal TIFF) | JPEG 2000 tiles |
| Uncompressed, LZW, Deflate, PackBits and JPEG chunks, with horizontal differencing | Vendor formats (MRXS, NDPI, iSyntax, DICOM-WSI) |
| u8/u16 gray, gray + alpha, RGB, RGBA; f32/f64 gray, RGB | |

Other TIFF layouts (palette images, the floating-point predictor) decode whole.
For a vendor format, read regions with a library such as openslide-python or
tiffslide and hand the bytes or arrays to `source("image_bytes")` or
`source("array")`.
