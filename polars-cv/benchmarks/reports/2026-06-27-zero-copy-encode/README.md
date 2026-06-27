# Zero-extra-copy output encoding (Binary/List/Array sinks)

Date: 2026-06-27

## What changed

Following the Polars 0.54/1.42 upgrade, the plugin's output boundary was cut down
to the minimum number of copies, reusing the zero-copy machinery the numpy sink
already used.

- **Binary/blob/encoded sinks** (`.sink("blob"|"png"|"jpeg"|...)`): each row's
  already-materialised `Vec<u8>` (from `to_blob()` / the image codec) is now
  *moved* into a `polars_buffer::Buffer<u8>` and registered as a backing buffer
  of a `BinaryViewArray` (`src/output.rs::binary_view_series_from_rows`), instead
  of being copied a second time into a `BinaryChunkedBuilder`. Net: **2 copies → 1**
  on the blob path (the remaining copy — `to_blob()`/codec materialisation — is
  inherent).
- **Typed list/array sinks**: `build_series_from_spec` now takes its row results
  **by value** (`src/graph/decode.rs`; call sites in `src/graph/compiled.rs` switched
  from `results.get(...)` to `results.remove(...)`), so the per-row buffers are
  moved rather than cloned while assembling the output.

The numpy/torch struct sink was intentionally left unchanged — see "Note on the
numpy sink" below.

## Result: A/B regression run

Same host, back-to-back builds (baseline = `git stash` of the three source files;
candidate = the change). Both `maturin develop --release`, then
`python -m benchmarks.regression.run_suite --scenarios pipelines --counts 300 --threads 1`.
The `pipelines` scenario exercises `.sink("blob")` (see
`benchmarks/frameworks/polars_cv_adapter.py`), so it measures exactly the changed
path. JSON artifacts: `baseline-zerocopy.json`, `candidate-zerocopy.json`.

```
framework / operation / size / count                           thru%    lat%    mem%  status
--------------------------------------------------------------------------------------------
polars-cv-eager        heavy_pipeline         256x256 n=300      +5.9    -5.6   -46.2  NEUTRAL
polars-cv-eager        imagenet_preprocess    256x256 n=300     +59.7   -37.4   -46.9  IMPROVED
polars-cv-eager        light_pipeline         256x256 n=300     +59.2   -37.2   -47.5  IMPROVED
polars-cv-eager        medical_pipeline       256x256 n=300     +50.5   -33.6   -47.4  IMPROVED
polars-cv-eager        medium_pipeline        256x256 n=300     +41.9   -29.5   -47.0  IMPROVED
polars-cv-streaming    heavy_pipeline         256x256 n=300      +4.2    -4.1   -48.1  NEUTRAL
polars-cv-streaming    imagenet_preprocess    256x256 n=300     +16.7   -14.3   -41.0  IMPROVED
polars-cv-streaming    light_pipeline         256x256 n=300     +17.5   -14.9   -40.6  IMPROVED
polars-cv-streaming    medical_pipeline       256x256 n=300     +26.5   -20.9   -41.6  IMPROVED
polars-cv-streaming    medium_pipeline        256x256 n=300     +11.2   -10.0   -40.5  IMPROVED

Summary: IMPROVED=8, REGRESSED=0, NEUTRAL=2, MISSING=0, NEW=0
PASS: no regressions.
```

### Reading the numbers

- **Peak memory drops ~40–48% across all 10 scenarios.** This is the robust,
  change-attributable signal: eliminating the second per-row image-buffer copy
  roughly halves the resident working set of the blob sink. It is consistent
  across eager and streaming and across every pipeline shape.
- **Throughput improves most on light/medium pipelines (+12% to +60%)**, where
  the eliminated copy is a large fraction of total work, and is neutral (+4–6%)
  on heavy/compute-bound pipelines where the kernels dominate. The largest
  throughput figures carry some host-state tailwind (the prior upgrade report
  documents ±7% host noise on this shared infra), but the direction is
  unambiguous and there are **zero regressions**.

## Correctness

`tests/test_zero_copy_encode.py` (10 tests) pins the new path:
blob round-trip byte-exactness, blob/array/list streaming-vs-in-memory parity,
null and all-null-morsel handling, and typed list round-trips for u8/i16/f32/f64.
Full suite green: `pytest tests/ -m "not network"` → 1422 passed. Rust gate
(`cargo test`, `clippy --all-targets --all-features -D warnings`, `fmt --check`)
and `ruff` all clean.

## Note on the numpy sink (latent bug surfaced, not fixed here)

While wiring the by-value refactor, moving (instead of cloning) the `ViewBuffer`
into the numpy sink made the buffer the *sole* Arc owner, which flips
`into_polars_buffer_strided` onto its **zero-copy strided** branch for
non-contiguous buffers. That branch currently mis-encodes permuted strides:
transpose/flip/rotate outputs came back as the original (un-permuted) layout
(caught by `tests/test_correctness_audit.py`). The clone in today's numpy arm has
been masking this by always forcing the materialise-to-contiguous branch.

This change deliberately **keeps the numpy arm on the clone/materialise path**
(with an explanatory comment in `decode.rs`) so behaviour is unchanged. The
strided zero-copy numpy path is a real latent bug worth a separate fix (correct
the permuted-stride encoding, then it can take the sole-owner zero-copy branch
safely) — out of scope for this copy-reduction.

## Streaming guidance

A "Scaling to Large Datasets (Streaming)" section was added to the README:
`.cv.pipe(...)` is elementwise, so parallelism comes from Polars' streaming
engine (one morsel per worker), not from inside the plugin. Eager
`DataFrame.with_columns(...)` runs single-threaded; large workloads should use
`df.lazy().with_columns(...).collect(engine="streaming")`. No intra-plugin
threading was added — that would duplicate what the streaming engine already does.
