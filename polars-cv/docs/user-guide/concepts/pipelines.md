# Pipelines

Pipelines are the core abstraction in polars-cv. They define a sequence of operations that can be applied to image data.

## Pipeline Structure

A polars-cv pipeline has three parts:

```mermaid
flowchart LR
    Source["Source"] --> Operations["Operations"] --> Sink["Sink"]
```

1. **Source**: How to interpret input data (e.g., `image_bytes`, `file_path`)
2. **Operations**: Transformations to apply (e.g., `resize`, `grayscale`)
3. **Sink**: Final output format (e.g., `tensor`, `png`)

## Modular Style

The recommended way to use polars-cv is to define a pipeline of transformations and then apply it to a column, followed by a sink.

```python
from polars_cv import Pipeline
import polars as pl

# 1. Define operations (reusable)
preprocess = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .grayscale()
)

# 2. Apply to column and choose sink
df = pl.DataFrame({"image": [png_bytes]})
result = df.with_columns(
    processed=pl.col("image").cv.pipe(preprocess).sink("tensor")
)
```

## Source Formats

| Format | Input Type | Description |
|--------|-----------|-------------|
| `auto` (default) | Varies | Infer the decode path from the column dtype (String → `file_path`, List/Array → `list`/`array`, Binary → `blob` if VIEW-tagged else `image_bytes`, the tensor sinks' struct → `tensor`) |
| `image_bytes` | Binary | Decode PNG/JPEG/TIFF bytes (auto format+dtype detect) |
| `file_path` | String | Local/cloud/HTTP path; decodes like `image_bytes`. Needs `allowed_roots` (default-deny; `["*"]` opts in to any path) |
| `raw` | Binary | Raw bytes (requires `dtype`) |
| `blob` | Binary | Self-describing VIEW binary protocol (what `sink("blob")` writes) |
| `tensor` | Struct / `polars_cv.ndarray` | What `sink("tensor")` writes (and the same struct untagged), read at its own strides; a `float16` row reads as `f32` |
| `list` | List | Polars nested List |
| `array` | Array | Polars fixed-size Array |
| `long` | List(Struct) | One struct per pixel, `{row, col, <names>...}`; `names=` one per channel, `where=` keeps a channel's > 0 pixels |
| `contour` | Struct | Contour geometry (contour domain) |

### Auto DType for Image Sources

For `image_bytes` and `file_path`, decoded dtype is determined at runtime:

- PNG/JPEG typically decode to `u8`
- 16-bit PNG decodes to `u16`
- TIFF can decode to `u8`, `u16`, `f32`, or `f64`

Because of this variability, the pipeline tracks dtype as `auto` until it can be resolved by:

- `source(..., dtype="...")`
- a dtype-fixing operation such as `normalize`, `threshold`, or `cast`

If you use `sink("list")` or `sink("array")`, dtype must be known at planning time.

## Sink Formats

| Format | Output Type | Description | Read back by |
|--------|------------|-------------|--------------|
| `tensor` | `polars_cv.ndarray` | One zero-copy tensor per row: the struct `{data, dtype, shape, strides, offset}`, tagged with an [extension type](extension-types.md); `dtype="f16"` downcasts, `compact=True` packs each row | `tensor` |
| `png` | Binary | PNG bytes; preserves bit depth (`u8` → 8-bit, `u16` → 16-bit) | `image_bytes` |
| `jpeg` | Binary | JPEG bytes; 8-bit only, requires `u8` | `image_bytes` |
| `webp` | Binary | WebP bytes; 8-bit only, requires `u8` | `image_bytes` |
| `tiff` | Binary | TIFF bytes with LZW compression (supports floating-point) | `image_bytes` |
| `blob` | Binary | Self-describing VIEW binary protocol | `blob` |
| `list` | List | Polars nested List | `list` |
| `array` | Array | Polars fixed-size Array | `array` |
| `native` | Varies | Native Python type (for scalars/vectors) | (a plain column) |

Every buffer sink's output is some source's input, and `source()` (auto)
routes each of them, so one pipeline's result can feed another, across a
Parquet file too. A tensor row is read where it lies, at its own strides.

A tensor sink hands each row over without copying where it can, so a row
that is a view (a crop, a flip, a transpose) carries the whole storage it reads
from, with the `strides` and `offset` that select the view. That is free in
memory but not on disk: Parquet and IPC write each row's `data` whole. Pass
`compact=True` for a column you will persist, and each row's `data` is exactly
its elements, row-major, at offset 0. Only a row that is not already compact is
copied.

The `long` sink turns maps into a table. Stack them as channels with
`channel_merge` (cast them to one dtype first), name each channel, and name the
mask channel in `where`: only its pixels > 0 are kept, and it is left out.

```python
heat = pl.col("heat").cv.pipe(Pipeline().source("array"))
score = pl.col("score").cv.pipe(Pipeline().source("array"))
mask = pl.col("mask").cv.pipe(Pipeline().source("array"))
pixels = df.select(
    heat.channel_merge(score, mask)
    .sink("long", names=["heat", "score", "mask"], where="mask")
    .alias("p")
).explode("p").unnest("p")  # columns: row, col, heat, score
```

Float buffers cannot be encoded to PNG/JPEG/WebP — `.cast("u8")`/`.cast("u16")`
first, or sink to `tiff`, which stores floating point directly.

### Choosing a sink

The table says what each sink is. Which to use follows from what happens to
the column next. Every block below is a section of
[`examples/15_choosing_a_sink.py`](https://github.com/heshamdar/polars-cv/blob/main/polars-cv/examples/15_choosing_a_sink.py),
which the test suite runs.

| You want | Sink |
|----------|------|
| One array for the whole column (a batch), every row the same shape | `array` |
| Rows of different shapes, kept as Polars values | `list` |
| A tensor per row for a data loader | `tensor` |
| To write tensors to Parquet or IPC and read them back | `tensor` with `compact=True` |
| To write images | `png` (lossless, 8- or 16-bit), `jpeg`, `webp`, or `tiff` (floats too) |
| To hand the result to another pipeline | `blob`, or `tensor` |
| A number or a list per row (after a reduction) | `native` (the default) |

**A batch.** `array` gives a fixed-shape `Array` column, which `to_numpy()`
turns into one `(n, H, W, C)` array without visiting rows in Python. The shape
must be known when the query is planned; a decoded image's channel count is
not, so give the sink its shape.

```python
--8<-- "examples/15_choosing_a_sink.py:array"
```

**Ragged rows.** `list` nests one `List` level per axis and takes any shape
per row. It is the slowest sink to build and to read back: use it when Polars
itself will work on the values, not as a way to reach NumPy.

```python
--8<-- "examples/15_choosing_a_sink.py:list"
```

**A tensor per row.** `tensor` hands each row's buffer over without copying
it; `numpy_from_struct` reads a row and `numpy_from_column` the column. The
column is typed `polars_cv.ndarray`, so it says what it holds after a round
trip; Polars' struct operations need `.ext.storage()` first
([Extension Types](extension-types.md)).

```python
--8<-- "examples/15_choosing_a_sink.py:tensor"
```

**Tensors that persist.** The same sink with `compact=True` (above), so each
row writes only its own elements. The default source reads the column back.

```python
--8<-- "examples/15_choosing_a_sink.py:persist"
```

**Images that persist.** An encoder. A decode followed by an encode is not
the file you started with: to copy files unchanged, use
[`.cv.read_bytes()`](sources.md#reading-bytes-without-decoding).

```python
--8<-- "examples/15_choosing_a_sink.py:encoded"
```

**Another pipeline.** A `blob` carries its dtype and shape, so the pipeline
that reads it declares nothing, and floats survive where an image encoder
would refuse or quantise them.

```python
--8<-- "examples/15_choosing_a_sink.py:blob"
```

**Several at once.** One decode can feed several outputs, each with its own
sink: see [Multi-Output](../composition/multi-output.md).

```python
--8<-- "examples/15_choosing_a_sink.py:multi"
```

## Chaining Operations

Operations are chained fluently. Most operation parameters accept both literal
values and Polars expressions.

```python
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=256, width=256)
    .crop(top=pl.col("y_off"), left=pl.col("x_off"), height=100, width=100)
    .normalize(method="minmax")
)
```

A parameter may be an expression whenever its value does not fix the output
rank or dtype at planning time — the lazy schema has to match what executes. A
size may be per-row; that axis is then unknown at planning time. The same rule governs the geometry namespaces
([`.contour` / `.point` / `.bbox`](../operations/geometry.md#expression-parameters)),
and [Parameters & Errors](parameters.md) gives the rule in full.

## Best Practices

1. **Reuse Pipelines**: Define pipelines once and apply them to many columns.
2. **Dynamic Parameters**: Use Polars expressions for per-row customization.
3. **Modular Sinks**: Keep pipelines generic and choose the sink format at the last step.

## Terminology

Contributors: the words *view*, *plan*, *node*, *step* and *graph* each have one
meaning in the docs and the code's prose. See the
[glossary](https://github.com/heshamdar/polars-cv/blob/main/GLOSSARY.md).

## Next Steps

- [Sources](sources.md) - Source formats and auto dtype behavior
- [Parameters & Errors](parameters.md) - Per-row parameters, nulls and error policies
- [Domains](domains.md) - Multi-domain pipelines
- [Multi-Output](../composition/multi-output.md) - Extracting multiple results
