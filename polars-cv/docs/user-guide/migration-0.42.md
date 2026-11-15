# Migrating from 0.41 to 0.42

0.42 makes path reads default-deny, folds the three array sinks into one
`sink("tensor")`, moves graph inspection onto the expression, and moves the
metrics to their own package. This page lists what to check coming from 0.41;
the [changelog](../changelog.md) has the full list, including the new `.bbox`
accessors, `.bbox.nms`, `Pipeline.crop_box`, `source("tensor")`,
`source(orientation="exif")` and the typed `sink()` keywords.

## Path reads need `allowed_roots`

**Every path-based read fails unless you name where it may read.** This
covers `source("file_path")`, an `auto` source over a `String` column,
`.cv.read_bytes()` and the path-column metadata methods, and remote URLs as
well as local paths. Name the permitted directories and URI prefixes, or pass
`["*"]` for paths you trust:

```python
pipe = Pipeline().source("file_path", allowed_roots=["/data/images", "s3://bucket/train/"])
# was Pipeline().source("file_path")
```

**`CloudOptions.token_command` is a non-empty `list[str]`,** run without a
shell; a string raises `TypeError`. Use `["sh", "-c", "a | b"]` if you need
shell syntax.

## One tensor sink

**`sink("numpy")`, `sink("torch")` and `sink("ndarray")` are `sink("tensor")`**
(likewise `Sink.tensor(...)`, `SinkFormat.TENSOR`, and
`show_images(format="tensor")`). Rows, bytes and strides are unchanged, and
`polars_cv.numpy_from_struct` and friends read the column as before.

**The column is typed `polars_cv.ndarray`,** an extension type over the old
struct. Polars struct operations do not see through the tag, so go through its
storage, and check the dtype with `isinstance`:

```python
pl.col("x").ext.storage().struct.field("shape")  # was pl.col("x").struct.field("shape")
isinstance(df.schema["x"], polars_cv.NdArrayType)
```

To concatenate with a plain-struct column written by 0.41, tag the old one
first: `pl.col("x").ext.to(polars_cv.NdArrayType())`.

## Sink settings and graph inspection

**Sink parameters are typed keywords or a `Sink`.** The `(format, {params})`
tuple in a multi-output `sink()` raises `TypeError`, as does a misspelled
keyword (it was a `ValueError`), and `opt_flags` is keyword-only:

```python
expr.sink({"thumb": Sink.jpeg(quality=85)})  # was {"thumb": ("jpeg", {"quality": 85})}
graph.set_output(node, Sink.jpeg(quality=85))  # was set_output(node, "jpeg", quality=85)
```

**Graphs are inspected from the expression.** `sink(..., return_expr=False)`,
`Pipeline.explain()` and `Pipeline.to_graph(column)` are removed. Build the
expression and ask it for the graph `sink(format)` would run:

```python
expr = pl.col("img").cv.pipe(pipe)
expr.graph("png")
expr.explain("png")
```

`explain(optimized=False, opt_flags=...)` is refused. `show_graph()` returns a
`graphviz.Digraph`, and the `viz` extra installs `graphviz` alone.

## Other signatures that change

- **The metrics live in polars-eval:** `from polars_cv import froc_auc` is
  `from polars_eval import froc_auc`.
- **`Pipeline.thumbnail(max_size)` is removed:** write
  `.source("image_bytes", decode_max_size=64)`.
- **`extract_contours` takes only a single-channel `u8` mask** and raises
  otherwise; apply `.threshold()`, `.cast("u8")`, `.grayscale()` or
  `.channel_select()` first.
- **`polars_cv.lazy.region_stats` is `polars_cv.masks.region_stats`** (still
  importable from `polars_cv`).
- **`image_info()` gains a fifth field, `orientation`;** code that unnests it
  into a fixed set of columns gains a column.

## Results that change

**Scalar-op constants keep full precision on 32/64-bit data.** `scale`,
`clamp*`, `add_constant`, `adjust_contrast`, `adjust_gamma`, `convolve2d`
coefficients and pad/crop fill values were rounded to f32 first: `scale(1/3)`
on `Float64` gave `0.3333333432674408`. Integer data is unaffected.

**A JPEG smaller than twice its `decode_max_size`** now takes the full
decoder, so it decodes as it would with no `decode_max_size`; output for such
images changes slightly.

**A box of negative size is refused,** naming the format to check, instead of
scoring as a 0-IoU false positive.

## Error messages

Error text is plainer: a failing row prints the codec's message, and a failing
pipeline is named (`Pipeline execution failed (column 'img' to the png sink):
...`). Code that matches on error text should match on the part that stayed.
