# Migrating from 0.37 to 0.38

0.38 fixes two operations whose results were wrong and adds a switch for the
plugin's thread pool. No method or parameter is removed. Two results change;
this page lists them. The [changelog](../changelog.md) has the full list.

## Results that change

**`perceptual_hash()` computes standard pHash.** The default
`algorithm="perceptual"` thresholded its DCT coefficients at their mean, which
the DC term dominates, so unrelated images shared most bits: 236 of COCO
val2017's 5,000 distinct images fell into 64 false duplicate groups. It now
computes what `imagehash.phash` computes (grayscale, a Lanczos resize to
32×32, a 2-D DCT, the 8×8 lowest frequencies against their median), and is
typically within 0–2 bits of imagehash. `hash_size` keeps its meaning, and
`average`, `difference` and `blockhash` are unchanged.

Hashes stored by an earlier version are not comparable with new ones.
Recompute them before comparing, and revisit any Hamming-distance threshold
you tuned against the old hash: distinct images are now about 32 bits apart
rather than 18.

```python
hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
df = df.with_columns(phash=pl.col("image").cv.pipe(hash_pipe).sink("list"))
```

**`blur(sigma=0)` is no blur.** It returned an all-zero image (the kernel was
NaN). It now returns the input unchanged, as PIL and SciPy do. A negative,
NaN or infinite sigma is refused: a literal when the pipeline is built, a
column value when its row runs (subject to
[`on_error`](concepts/sources.md#error-handling) like any other row error). A negative
sigma used to be treated silently as no blur; clamp a column that may hold
one:

```python
pipe = Pipeline().source("image_bytes").blur(sigma=pl.col("sigma").clip(0))
```

## New

**`polars_cv.set_row_parallelism(enabled)`** turns the plugin's own row
splitting off (or back on) for the process, and returns the previous setting;
`polars_cv.row_parallelism()` reads it. Off, each call runs its rows on the
thread Polars called it on, so the streaming engine's morsels are the only
parallelism. Outputs are identical either way. See
[Streaming & Scaling](concepts/streaming.md#turning-the-plugins-pool-off).

```python
import polars_cv

previous = polars_cv.set_row_parallelism(False)
out = lf.collect(engine="streaming")
polars_cv.set_row_parallelism(previous)
```

## Behaviour you get for free

- **Many-core hosts no longer spin the plugin's pool on cheap rows.** A call
  wakes one helper thread up front, then one more per 200 µs of work it
  estimates is left, instead of inviting the whole pool before every row
  range. On a 72-thread host the old behaviour made a contour table take
  19.5 s, against 0.4 s at 16 threads. Results are unchanged.
- **The package ships its license texts** (`LICENSE-MIT`, `LICENSE-APACHE`).
