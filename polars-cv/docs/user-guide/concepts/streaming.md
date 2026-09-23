# Streaming & Scaling

The `.cv.pipe(...)` expression is an ordinary elementwise Polars plugin, so the
**parallelism comes from the Polars engine, not from inside the plugin**. This has
one important consequence for large workloads.

## Why eager can run single-threaded

The plugin does not split work inside one call. Its parallelism comes from the
engine calling it for several batches at once. The in-memory engine behind a
plain `DataFrame.with_columns(...)` / `.select(...)` makes one call per
*chunk*, so a single-chunk column (anything built in one piece or rechunked)
runs on one thread:

```python
# Eager: one call per chunk -- a single-chunk column uses one core
result = df.with_columns(
    processed=pl.col("image").cv.pipe(pipe).sink("numpy")
)
```

This is fine for small or interactive use. When a call runs for a while on
one thread with nothing alongside it, polars-cv prints a one-time warning
pointing you here (see [Silencing the warning](#silencing-the-warning)).

## Use the streaming engine for scale

Run through the lazy **streaming** engine instead. Polars splits the column into
*morsels* and processes them across its worker pool, and can spill intermediate
state to disk when memory is tight:

```python
result = (
    df.lazy()
    .with_columns(processed=pl.col("image").cv.pipe(pipe).sink("blob"))
    .collect(engine="streaming")
)
```

This is the recommended path for anything beyond small/interactive use. The
plugin's per-morsel graph is compiled once and cached process-wide, so per-morsel
overhead is just a hash lookup — you get Polars' multi-threaded, larger-than-memory
execution with no extra plumbing.

!!! note
    The detection-metrics APIs already collect with `engine="streaming"`
    internally, so you don't need to opt in when using them.

## Silencing the warning

A single plugin call that runs longer than a threshold (default 2 seconds) with
no other call running alongside it prints a one-time notice. Streaming runs,
whose morsels overlap, do not trigger it. Control it with environment variables:

| Variable | Effect |
| --- | --- |
| `POLARS_CV_SILENCE_ENGINE_WARNING=1` | Suppress the warning entirely |
| `POLARS_CV_ENGINE_WARN_SECONDS=<s>` | How long one call may run alone before the warning fires |

`POLARS_CV_ENGINE_WARN_ROWS` (a row-count threshold) is no longer read; setting
it prints a notice naming its replacement.

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
