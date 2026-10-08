# Row splitting from a plugin-wide budget (Phase 1 of the Polars 2.0 plan, PR #124)

Base-vs-head gate for `9e0426f` ("Row splitting decides from what runs now,
never from history").

- **base**: the Rust sources of `a2691d0` (the overlap-history `CallTracker`),
  built from a stash of `polars-cv/src` in the same tree;
- **head**: `9e0426f` (the `BUSY` budget).

Both were built with `maturin develop --profile benchmark` on one 4-core
container. Each side's `_lib.abi3.so` was copied aside and swapped in per
run, so the two ran the same harness, including the new `split_` cases.
That is also why every `.meta.json` here records `git_sha` `9e0426f`: it is
the working tree's SHA, not the extension's.

**Threads: 4** (`--threads 4`). The suite's default of 1 never splits a
call, so it cannot measure this change.

## Full run

`pipelines`, `targeted:geom_*` and `targeted:split_*` at the defaults
(300 × 256², 3 warmup + 10 timed iterations, 3 suite repeats), interleaved
base → head → base → head (`full/`).

At 4 threads this host is far noisier than the 7% gate. Same-binary
self-checks (base1→base2, head1→head2) moved cases by up to +33%, mostly in
one direction (later runs ran faster). So a case counts as changed only
when **both** base→head pairs agree beyond that case's own same-binary
spread and beyond 7%.

| case | Δ throughput pair 1 / pair 2 | same-binary spread | verdict |
|------|------------------------------|--------------------|---------|
| `split_streaming_then_eager` | **+118.6% / +122.1%** | 3.7 / 5.4% | improved, 2.2× |
| `split_streaming_uneven_row_groups` | **+10.6% / +11.7%** | 4.3 / 5.4% | improved |
| `geom_contour_area` | +14.3% / +11.0% | 5.7 / 2.7% | improved |
| every other case | mixed signs or within spread | up to 33% | neutral |

The compare tool flagged eager `medium_pipeline` (−4.8 / −8.2%) and three
other cases with inconsistent signs as regressions. Only eager `medium` was
negative in both pairs, so the pipelines were re-run with more repeats.

## Focused re-run

`pipelines:{light,medium,heavy}_pipeline`, `--repeats 5`, interleaved base
and head three times (`focused/`). Mean throughput, head vs base:

| case | per pair | mean |
|------|----------|------|
| eager heavy | +5.5, +1.6, −3.7 | +1.0% |
| eager light | +1.2, −3.7, +5.9 | +1.0% |
| eager medium | +0.4, +3.3, −5.2 | −0.6% |
| streaming heavy | +4.4, +12.7, 0.0 | +5.5% |
| streaming light | +2.3, +6.6, +10.5 | +6.5% |
| streaming medium | −4.9, −1.7, −4.7 | **−3.8%** |

- **Eager: neutral.** The flagged `medium` dip alternates sign. A lone call
  gets the same four workers as before: the calling thread plus three
  helpers, where it used to be four pool threads.
- **Streaming `medium`: a small, consistent cost (−3.8%), under the 7%
  gate.** The likely cause: in a loop of streaming queries over an
  in-memory frame (4 morsels of 75 rows), the old tracker kept every call
  inline, because the previous query's calls had overlapped. Now the first
  morsel's call starts alone and invites helpers for a moment. Each helper
  runs one ~5-row range before it finds the budget full (the other morsels
  have started) and leaves, so those rows' buffers cross threads.
  Streaming `light` and `heavy` moved the other way (+6.5%, +5.5%).
- That same lack of history is what makes the two `split_` cases 2.2× and
  11% faster, so it is kept as is.

## Reproducing

```bash
cd polars-cv
SEL="pipelines,targeted:geom_*,targeted:split_*"
python -m benchmarks.regression.run_suite --select "$SEL" --threads 4 --out head.json
python -m benchmarks.regression.compare base.json head.json
```
