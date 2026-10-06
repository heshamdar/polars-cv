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

The pool is sized by `POLARS_MAX_THREADS`, like Polars' own. Results, error
reporting and `on_error` behave exactly as a row-by-row run would: rows come
back in order, and under `on_error="raise"` the error reported is the earliest
failing row's.

### Two pools, one setting

The plugin's pool is its **own** — a plugin links its own copy of Polars, so it
cannot join the host's — and both are sized by `POLARS_MAX_THREADS`. A call
spreads its rows over the plugin's pool only when it runs alone; when calls
overlap — the streaming engine running morsels concurrently, or the in-memory
engine evaluating several `.cv`/geometry expressions of one `select` at once —
the overlapping calls run on the Polars thread that made them. So while one
call is spread and others run inline, up to about twice `POLARS_MAX_THREADS`
threads can be runnable at once.

That is usually the fastest arrangement. If a query with many plugin
expressions shows a load average well above the core count and erratic run
times — most likely on machines with mixed performance/efficiency cores, or
when every row holds large buffers and memory is tight — lower
`POLARS_MAX_THREADS` (set it before importing Polars). Measure: on a 4-core
machine, nine large-image pipelines in one streaming query ran fastest at the
default.

## Use the streaming engine for larger-than-memory data

The streaming engine processes the column in *morsels* and spills
intermediate state to disk when memory is tight. Since Polars 2.0 it is what a
lazy `.collect()` uses by default; an eager `DataFrame.with_columns` still runs
in memory.

```python
result = (
    df.lazy()
    .with_columns(processed=pl.col("image").cv.pipe(pipe).sink("blob"))
    .collect()  # the streaming engine (Polars >= 2.0)
)
```

The plugin's graph is compiled once and cached process-wide, so per-morsel
overhead is just a hash lookup.

Spilling applies to the operators that buffer — `sort`, `group_by`, joins and
windows — and every polars-cv output survives it unchanged, including the
`polars_cv.ndarray`/`point`/`contour`/`bbox` extension types. The memory
polars-cv allocates counts toward the same budget, so a query whose decoded
images fill memory spills like any other. Polars' knobs, set before it is
imported:

| Variable | Meaning |
|----------|---------|
| `POLARS_OOC_MEMORY_BUDGET_FRACTION` | Spill past this share of system memory (default 0.8) |
| `POLARS_OOC_MEMORY_BUDGET_MB` | ... or past this many MB, if lower |
| `POLARS_OOC_SPILL_DIR` | Where spill files go (Linux default `/var/tmp/polars-$USER/spill`) |
| `POLARS_OOC_DISK_BUDGET_MB` | Abort the query past this much spilled data (default 64 GB) |

A row is one plugin call's unit of work: an image is decoded, processed and
encoded whole, so a single image must still fit in memory.

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
