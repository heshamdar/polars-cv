# Streaming & Scaling

## Every call uses every core

A `.cv.pipe(...)` call splits its rows into ranges and runs them on the
plugin's thread pool, then reassembles them in order. So a plain eager call is
multi-core, whatever the column's chunking:

```python
# Eager: the rows of each call run in parallel
result = df.with_columns(
    processed=pl.col("image").cv.pipe(pipe).sink("numpy")
)
```

The pool is sized by `POLARS_MAX_THREADS`, like Polars' own. Under the
streaming engine several calls run at once; they share that one pool (a caller
waits while its rows run), so the plugin never uses more threads than the
setting allows. Results, error reporting and `on_error` behave exactly as a
row-by-row run would: rows come back in order, and under `on_error="raise"` the
error reported is the earliest failing row's.

## Use the streaming engine for larger-than-memory data

The streaming engine processes the column in *morsels* and can spill
intermediate state to disk when memory is tight:

```python
result = (
    df.lazy()
    .with_columns(processed=pl.col("image").cv.pipe(pipe).sink("blob"))
    .collect(engine="streaming")
)
```

The plugin's graph is compiled once and cached process-wide, so per-morsel
overhead is just a hash lookup.

!!! note
    The detection-metrics APIs already collect with `engine="streaming"`
    internally, so you don't need to opt in when using them.

## Cheaper decoding for curation passes

When scaling over large image sets, you often only need a cheap signal (a
perceptual hash, a mean, a quality score) to *filter* before doing full-resolution
work. Decode a downscaled thumbnail first with
[`thumbnail(max_size)`](../operations/image-ops.md#thumbnail) (JPEG IDCT-scaled
decode), compute the signal, filter, then full-decode only the survivors in a
second pass.

When the filter only needs image *dimensions*, you can skip decoding on the
first pass entirely — and skip the second fetch. Read the bytes once with
[`read_bytes()`](sources.md#reading-bytes-without-decoding), filter on the
header-only metadata methods, and decode only what survives:

```python
lf = (
    pl.scan_parquet("images.parquet")
    .with_columns(raw=pl.col("path").cv.read_bytes())
    .filter(pl.col("raw").cv.width() > 512)
    .with_columns(thumb=pl.col("raw").cv.pipe(pipe).sink("png"))
    .drop("raw")
)
lf.collect(engine="streaming")
```

Dropping `raw` before collecting keeps it morsel-bounded; keep it in the
projection when you want the original bytes as an output.

## Next Steps

- [Pipelines](pipelines.md)
- [Sources](sources.md)
- [Domains](domains.md)
