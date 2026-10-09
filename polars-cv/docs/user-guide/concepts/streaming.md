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

### Two pools, one budget

The plugin's pool is its **own** — a plugin links its own copy of Polars, so it
cannot join the host's — and both are sized by `POLARS_MAX_THREADS`. Every
call runs its rows on the Polars thread that made it, and the plugin's idle
pool threads join in while the threads running polars-cv rows number fewer
than `POLARS_MAX_THREADS`. How many threads a call gets therefore depends only
on what is running at that moment:

- **A call running alone** — an eager `with_columns`, or one large morsel —
  uses the whole pool.
- **Concurrent calls** — the streaming engine running one call per morsel, or
  the in-memory engine evaluating several `.cv`/geometry expressions of one
  `select` at once — each keep their rows on their own thread once the budget
  is used up, so a row's buffers do not move between threads.
- **A large call beside small ones** (a big Parquet row group next to a small
  one) picks up the threads the small calls free as they finish.

Threads running polars-cv rows stay within `POLARS_MAX_THREADS`, except that
each concurrent call always runs on its own thread, and a pool thread that
joined finishes the range it started before it leaves. If a query with many
plugin expressions still shows a load average well above the core count —
most likely on machines with mixed performance/efficiency cores — lower
`POLARS_MAX_THREADS` (set it before importing Polars).

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

### How big a call is

The streaming engine hands polars-cv one *morsel* per plugin call, and a call
holds every output row of its morsel until it returns (plus, for remote paths,
a [window of fetched files](sources.md#memory-and-streaming)). Spilling cannot
reach inside a call. So a query's peak memory is roughly:

```text
rows per call  x  output bytes per row  x  calls at once (about POLARS_MAX_THREADS)
```

Polars sizes morsels by **rows**, not bytes (`POLARS_IDEAL_MORSEL_SIZE`,
100,000 by default). For image rows that matters:

- A **Parquet scan** gives a call one row group at a time (a row group over
  1.5× the ideal size is split). A row group of 20,000 images sunk to
  224×224×3 `f32` is 20,000 × 602 KB ≈ 12 GB in one call.
- An **in-memory frame** is split into at least one morsel per thread.

Two knobs bound it:

1. **Write image tables with row groups sized by bytes.** Divide a target
   such as 256 MB by the bytes one *output* row will take:

    ```python
    rows = 256 * 2**20 // (224 * 224 * 3 * 4)  # ~445 rows per row group
    df.write_parquet("images.parquet", row_group_size=rows)
    ```

2. **Cap the morsel size**, with `pl.Config.set_streaming_chunk_size(n)` (the
   same setting as `POLARS_IDEAL_MORSEL_SIZE`). It splits large row groups and
   in-memory frames too. It is process-global: it applies to every streaming
   node of every query, not only polars-cv's, which is why polars-cv does not
   set it for you.

A row is still one call's indivisible unit of work: an image is decoded,
processed and encoded whole, so a single image must fit in memory.

!!! note
    The detection-metrics APIs collect with Polars' default engine, the
    streaming engine since Polars 2.0, so you don't need to opt in. They
    follow `pl.Config.set_engine_affinity` like any lazy query: setting an
    in-memory affinity while debugging makes them run in memory too.

## Feeding a training loop

`LazyFrame.collect_batches` runs a query on the streaming engine and hands
Python its result a batch at a time, so a dataset larger than memory can feed
a training loop. With a fixed-shape `array` sink, each batch converts to one
NumPy tensor of shape `(rows, *shape)` that views the batch's memory rather
than copying it:

```python
pipe = Pipeline().source("file_path", dtype="u8").resize(height=224, width=224)
batches = (
    pl.scan_parquet("images.parquet")
    .select(x=pl.col("path").cv.pipe(pipe).sink("array", shape=[224, 224, 3]))
    .collect_batches(chunk_size=256)
)
for batch in batches:
    x = batch["x"].to_numpy()  # (256, 224, 224, 3) uint8, no copy
    train_step(x)
```

`chunk_size` sets the rows per batch you receive; how many rows each plugin
call processes is the morsel size above. A null row (from `on_error="null"`)
makes `to_numpy()` copy the batch. Polars documents `collect_batches` as
unstable.

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
lf.collect()
```

Dropping `raw` before collecting keeps it morsel-bounded; keep it in the
projection when you want the original bytes as an output.

## Next Steps

- [Pipelines](pipelines.md)
- [Sources](sources.md)
- [Domains](domains.md)
