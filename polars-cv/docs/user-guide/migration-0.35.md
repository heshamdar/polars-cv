# Migrating from 0.34 to 0.35

0.35 is a fixes-and-performance release. No method or parameter is added or
removed, and every output dtype is unchanged. Two things can change a result
you have pinned. This page lists them; the [changelog](../changelog.md) has the
full list.

## Results that change

**Nearest-neighbour resize at exact pixel-centre ties.** Output pixel `i`
takes the source pixel under its centre, `floor((i + 0.5) × src / dst)`. When
that centre falls exactly on a source pixel boundary (2 → 21 rows: row 10;
2 → 7: row 3), the `u8`/`u16`/`f32` path could take the pixel before it. It
now takes the correct one for every dtype. Only those tie rows and columns
change. Re-record any snapshot of a `resize(..., filter="nearest")` output
whose size ratio has ties. No other filter is affected.

**`DetectionTable.collect()` uses Polars' default engine.** Its `engine=`
default is now `"auto"` (it was `"streaming"`), and no metrics function picks
an engine any more. Under Polars 2.0's defaults that is still the streaming
engine, so results are unchanged. What changes is that
`pl.Config.set_engine_affinity(...)` is now respected. Pass
`engine="streaming"` to keep the old pin.

## Behaviour you get for free

- **Rows spread over the thread pool whatever ran before.** Before 0.35, an
  eager call of a pipeline that had just run under the streaming engine could
  run on one thread. Every call now shares one budget of `POLARS_MAX_THREADS`
  threads. See [streaming](concepts/streaming.md#two-pools-one-budget).
- **Remote reads stream through a window.** `file_path` sources,
  `.cv.read_bytes()` and the header-only `.cv.width()` and friends fetch each
  row's file and up to `POLARS_CONCURRENCY_BUDGET` more ahead, instead of every
  file in the call before the first decode. Peak memory for a large remote
  morsel drops, and downloads overlap decoding. See
  [sources](concepts/sources.md#memory-and-streaming).
- **One async runtime.** Remote reads run on Polars' async runtime
  (`POLARS_ASYNC_THREAD_COUNT`), and the plugin no longer starts its own.
- **Every bootstrap plan streams**, including `sample_col=` resampling.
