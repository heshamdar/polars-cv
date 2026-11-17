# Optimization log — agentic kernel campaign

Hand-off document between iterations. Each phase appends; nothing is rewritten
except the "Status" line and the candidate table's status column.

**Status:** Phase 3. T1 and T2 accepted; next: PNG decode (now in scope), `Blocked` scratch zeroing, grayscale.

**User decisions (2026-11-17):** prioritise the ops users run most often; outputs
may move within a documented tolerance (any golden re-bless is raised first);
`proptest` may be added.

---

## 1. Architecture summary (what a row goes through)

```
Python Pipeline (generated builder) ──Plan.push──▶ JSON UnifiedGraph
   │  .sink() → _plugin.call("vb_graph", json, cols)
   ▼
polars-cv plugin (Rust)
   graph/compiled/compile.rs   graph parsed + planned once, process-wide cache
   row_split.rs                rows split over the plugin's thread pool
   graph/compiled/run.rs       per row, per node in topo order:
     graph/decode.rs           source → ViewBuffer  (image decode / VIEW blob / list / numpy)
     graph/row.rs::run_step    each GraphStep → view-buffer op
     graph/encode.rs           ViewBuffer → sink (blob / tensor struct / list / array / png …)
   ▼
view-buffer (engine)
   expr.rs ViewExpr → execution/ExecutionPlan (view ops stay zero-copy, scalar chains fuse)
   ops/<family>.rs   contract (shape/dtype/memory/spatial rules) + execute entry
   kernels/*.rs      kernel bodies  ← the optimisation targets
```

- **Kernel registration:** an op is a variant of an engine family enum
  (`ImageOpKind`, `ComputeOp`, `ViewOp`, …) listed in `typed_ops!`
  (`polars-cv/src/catalog/`). Python methods are generated from it. Kernel
  changes don't touch the catalogue, so a pure-kernel optimisation can't
  change the public API.
- **Allocation and copies:** a strided `ViewBuffer` can be a view (crop/flip/
  transpose stay lazy), and a kernel reads it through `to_dense_rows` /
  `map_pixels` / `cast`. Per-pixel scalar chains fuse into one `FusedKernel`
  (f32), with an int-affine and a LUT strategy for small ints
  (`kernels/elementwise/mod.rs::strategy`). The VIEW blob sink/source costs a
  copy in and out per op in single-op benchmarks (2026-10-10 report).
- **Parallelism:** across rows only (`row_split.rs`, one pool). Kernels are
  single-threaded by design, so there's no nested Rayon. Open issue #205: eager
  calls scale poorly across threads.

## 2. Tests and benchmarks (how to run)

All commands run from `polars-cv/` unless marked.

| What | Command | Notes |
|---|---|---|
| Build (tests) | `maturin develop` | debug; required before plugin tests or they **self-skip** |
| Build (benchmarks) | `maturin develop --profile benchmark` | thin-LTO release; `run_suite` refuses an unoptimised build |
| Full CI lane | `../scripts/verify.sh` (`--slow` weekly lane) | the green/red authority |
| Python suite | `uv run pytest tests/ -n auto -m "not network and not slow"` | |
| Reference tests (OpenCV/NumPy/skimage) | `uv run pytest tests/reference/` | 25 files: color, blur, canny, affine, morphology, histogram, LUT, … |
| Golden corpus | `uv run pytest tests/test_golden_corpus.py` | every op's planned state, schema and output digest + samples (`tests/golden/op_corpus.json`) |
| Parity / property (hypothesis) | `uv run pytest tests/parity/` | laws, row/execution invariance |
| Engine unit + integration | `../scripts/with-pyo3-env.sh cargo test -p view-buffer --all-features` | `tests/it/` is one binary; add modules to `main.rs` |
| Kernel micro-bench (criterion) | `../scripts/with-pyo3-env.sh cargo bench -p view-buffer --all-features --bench kernels` | ~55 groups at 256/512/1024²; also run with `RUSTFLAGS="-C target-cpu=x86-64"` (wheel baseline; local config builds for v3) |
| End-to-end regression | `uv run python -m benchmarks.regression.run_suite --out X.json [--select …]` then `… compare base.json head.json` | pipelines @ 256², count 300, 4 threads, best-of-3; 7% gate over ≤5% noise |
| Library comparison | `uv run python -m benchmarks.run_benchmarks …` | vs OpenCV/Pillow/pyvips/torchvision |

**Already in place (Phase 1 needs extending, not building):** criterion kernel
bench, OpenCV reference tests, golden corpus, hypothesis parity laws,
regression harness with a noise-calibrated gate. **Missing:** `proptest`
(not a dependency, and adding one goes through `cargo-deny`), criterion groups
for `to_hsv`/Lab/YCbCr, `canny`, `histogram_equalize`, `warp_affine` and 5×5
`convolve2d`, and per-candidate edge corpora (1×1, odd widths, u16).

## 3. Candidate kernels

Evidence is from `benchmarks/reports/2026-10-10-library-comparison` (polars-cv ÷
OpenCV throughput, 1024² × 50, 4 cores) plus reading the code. Below 1× means
OpenCV is faster. Every row is a hypothesis to measure in Phase 2. None of
them is a ranking yet.

| # | Kernel (file) | vs OpenCV @1024² | What the code does now | Optimisation hypothesis | Falsifiers today | Gaps to fill (Phase 1) |
|---|---|---|---|---|---|---|
| 1 | Colour convert to HSV/Lab/YCbCr (`kernels/color.rs::convert_ranged`) | **0.10×** (slow at every size) | casts the whole image u8→f64 (8× the memory), runs a per-pixel f64 map that matches `from`/`to` enums **per pixel**, then casts f64→u8: three passes and two full-size temporaries | (a) monomorphise the (from, to) pair so the per-pixel `match` disappears; (b) a u8 fast path reading u8 and writing u8 in one pass (f32, or OpenCV-style integer HSV); (c) no f64 intermediate | `reference/test_color_ref.py`, golden corpus | criterion group; u8 exhaustive or round-trip tests; must stay **bit-identical** to current u8 output (golden digests) |
| 2 | Canny (`kernels/canny.rs`) | **0.18×** (slow at every size) | builds four `(h+2)·(w+2)` planes, reads the source through a closure `at(i)` per element, non-u8 input goes through an f64 plane; hysteresis is scalar | a single-plane u8 path (no closure, i16 Sobel, row-ring buffers instead of full planes); possibly fused Sobel and magnitude | `reference/test_canny_ref.py` | criterion group; u16/f32 cases |
| 3 | `convolve2d` 5×5 / Sobel (`kernels/filter.rs::convolve_in`) | **0.28× / 0.26×** | u8→f32 cast of the whole image, then K² shifted-slice multiply-add passes over each output row | (a) accumulate K taps per pass in registers instead of K² read-modify-writes of `out_row`; (b) skip the cast copy by reading u8 rows into the `Rows` ring; (c) a separable fast path changes f32 rounding, so it needs a golden check and probably isn't allowed | `tests/it/convolve_ref.rs`, criterion `convolve2d_k3_*` | 5×5 criterion group |
| 4 | Histogram equalize (`kernels/equalize.rs`) | **0.36×** | per channel: histogram pass, then a remap pass doing one **f64 divide and round per pixel** with strided writes | build a 256-entry LUT per channel once (same formula, so bit-identical), then one interleaved pass over all channels | `reference/test_histogram_ref.py`, golden | criterion group; RGB and cropped-view cases |
| 5 | Warp affine / rotate (`kernels/warp.rs`) | **0.39×** | per output pixel: f64 coordinate math and an f64 bilinear blend | incremental coordinates per row (add a delta instead of a full matrix multiply), fixed-point u8 bilinear like OpenCV's `INTER_BITS` (this changes rounding, so it may only be allowed where it's bit-identical) | `reference/test_affine_ref.py`, `tests/it/affine_ref.rs`, criterion `rotate_30_*` | warp_affine criterion group |
| 6 | Invert u8 (fused scalar, `kernels/elementwise/mod.rs`) | **0.25×** | the int-affine path; the report blames the blob copy in and out, not the kernel | profile first: if the kernel is already memory-bound, the win is in the I/O path (#7), not here | golden, criterion `invert_u8` | none |
| 7 | **Non-kernel:** VIEW blob in/out, per-row allocation, eager thread scaling (`graph/decode.rs`, `graph/encode.rs`, `row_split.rs`; issues #205, #206) | across the board for cheap ops | copy in, op, copy out per row; eager scales poorly across threads (#205); transpose is 8× slower than flip (#206) | buffer reuse across rows, a zero-copy blob source; transpose blocking for #206 | `tests/it/copy_counts.rs`, `exact_data_movement.rs`, regression `zero_copy` scenario | profile before guessing |
| 8 | Gaussian blur, resize (`kernels/blur.rs`, `resize.rs`, `fast_image_resize`) | ≥1× (winning) | already separable and SIMD (fir) | low priority; Phase 2 only reorders this if the e2e profile says so | `reference/test_blur_ref.py`, criterion | none |

**Prior ranking before profiling:** 1 > 4 > 2 > 3 > 5, with 7 likely the
biggest e2e lever on the `pipelines` scenario (resize, normalize and crop are
already competitive). #4 is the cheapest safe win: a pure restructure, with
bit-identity provable by construction.

## 4. Profile ranking (Phase 2)

**Method.** `perf` is not installed and `perf_event_paranoid=2` blocks samply
(lowering it was refused), so profiles are `valgrind --tool=callgrind`
instruction counts over the scratch driver `e2e_drive.py` (100 × 256² PNG,
tensor sink; JPEG 768² × 30), with the extension built `--profile benchmark`
plus symbols. Instruction share ≠ time share: callgrind counts `rep stosb`
per byte (inflating `memset`) and ignores memory stalls, so every candidate
is confirmed by criterion timing before it's accepted.

**End-to-end baseline** (`--profile benchmark`, 4 threads, tensor sink, best
of one 4 s run):

| pipeline | source | img/s | decoded MB/s |
|---|---|---|---|
| decode only | PNG 256² | 7,088 | 1,394 |
| light: resize 224 (lanczos3) → normalize | PNG 256² | 3,200 | 629 |
| imagenet: resize 256 → crop 224 → normalize_preset | PNG 256² | 6,008 | 1,181 |
| heavy: resize → flip → gray → blur 1.5 → normalize → threshold | PNG 256² | 4,141 | 814 |
| medical: gray → normalize → resize 512 | PNG 256² | 2,394 | 471 |
| resize 224 → HSV | PNG 256² | 1,210 | 238 |
| decode only | JPEG 768² | 850 | 1,503 |
| imagenet | JPEG 768² | 711 | 1,258 |

**Instruction share, excluding Python and codec** (decode is
`fdeflate`/`zune-jpeg`, third-party, 25–42% of the total):

| # | Where | light | heavy | medical | imagenet | Notes |
|---|---|---|---|---|---|---|
| T1 | `stats::min_max_f32::int_extremes` (normalize minmax's extremes) | 11% | 5% | 4% | – | two scalar passes, ~10 instr/element, inside a closure that dispatch never reaches |
| – | `memset` (zeroed scratch in `Blocked`, zeroed large allocations) | 8% | 8% | 6% | 7% | inflated by callgrind; confirm by timing |
| T2 | `SeparableBlur<u8,f32>` | – | ~13% | – | – | already dispatched |
| – | Lanczos coefficients (`sin`, `precompute_coefficients`) | 3% | – | 7% | – | inside `fast_image_resize`, recomputed per call, no cache API: upstream idea |
| – | `fast_image_resize` convolution | 15% | – | 20% | – | real work, already AVX2 |
| T3 | `MapPixels<u8,u8,3,1,Grayscale>` | – | 2% | – | – | ~4 instr/pixel |
| T4 | colour conversion (f64 per pixel) | – | – | – | – | 1,210 img/s e2e; not bit-exact-able, so it waits for a golden re-bless decision |

Ranking by (share × how common): **T1 normalize minmax extremes** → `Blocked`
scratch zeroing → T2 blur → T3 grayscale → T4 HSV.

## 5. Hypotheses tried

| Date | Target | Hypothesis | Result (before → after, ±) | Verdict | Commit |
|---|---|---|---|---|---|
| 2026-11-17 | T1 normalize minmax extremes (`stats::min_max_f32`) | Two scalar `Iterator::min/max` passes inside an undispatched closure → one 64-lane fold (`pminub`/`pmaxub`) as a dispatched `SimdKernel` with a `RunSink` for views. Bit-exact. | criterion `normalize_minmax_u8` 4.10/4.34/4.29× (256/512/1024, ±≤13%), wheel 3.9–4.1×; cropped 3.7–4.3×; resize→minmax chain 1.14–1.60×; controls 0.96–1.10×. **e2e light 2,775 → 3,614 img/s (1.30×)**, heavy 1.09×, medical 1.10×, imagenet (control) 1.00× | accepted | `8e41faf` |
| 2026-11-17 | T2 Gaussian blur (`kernels/blur.rs`) | Both passes re-read and re-wrote the output row once per tap → 32 outputs per chunk with sums held in registers across taps, same tap order from zero. Bit-exact. | criterion u8 RGB 1.20/1.18/1.29×, gray 1.05/1.21/1.33×, f32 1.20/1.39/1.30× (wheel 1.05–1.41×); control 0.98–1.10× | accepted | `6e6ae22` |
| 2026-11-17 | PNG decode: swap to `zune-png` | A faster decoder than `png` 0.18 + `fdeflate` (decode is 25–42% of e2e instructions; 96% of PNG decode is inflate) | repo harness `png_decode_eval` (`--profile benchmark`): **0.34× geomean** on RGB/RGBA, slower in every cell (gradient 0.60–0.77×, noise 0.16–0.19×) | rejected | – |
| 2026-11-17 | PNG decode: faster inflater (zlib-rs/libdeflate behind a minimal PNG path) | Inflate headroom on realistic zlib streams | sizing first: C zlib inflates the corpus IDAT streams (PIL level 6, ratio 1.30) at **1.17 ms/img**, vs ~0.56 ms·core for the plugin's *whole* PNG decode with `fdeflate`. fdeflate is already ~2× stock zlib, so a custom PNG decoder at best ties | rejected (not built: sizing shows no headroom) | – |
| 2026-11-17 | T3a `Blocked` scratch (`kernels/elementwise/mod.rs`) | The per-block `[0.0f32; 2048]` scratch (8 KB zeroed per block, per run for views; `memset` 5–8% of callgrind instructions) → uninitialised `MaybeUninit` scratch for the block and the per-channel lane | local x86-64-v3: u8 Blocked users 1.06–1.5× (scale/contrast ±17–23%); **wheel x86-64: mostly noise, `normalize_preset_f32` 0.82× at every size** (±4–8%) | rejected (regression on the shipping build) | – |
| 2026-11-17 | T3b same, block scratch only (lane zeroed as before) | Isolate the suspected lane regression | wheel: every case 0.86–1.12×, the same band as the untouched grayscale controls (0.93–1.10×) | rejected (within noise); two consecutive failures, so the target is stopped | – |

## 6. Remaining ideas (ranked)

_Running list; ranked at the end._

- The `memset` callgrind flagged was mostly its per-byte `rep stosb` count, not time (see T3a/b). Zeroed scratch isn't a measurable cost on the wheel build.
- Lanczos coefficients are recomputed per image inside `fast_image_resize` (3–7% of resize pipelines). It has no cache API, so this needs an upstream patch.
- f32 `normalize` minmax (`float_extremes`) is still a scalar fold with the NaN rule. It's lower priority because pipelines normalize u8 far more often.
- **Measurement method:** base and head bench binaries are built separately (`benchbin.sh`) and alternated three rounds (`ab.py`). End-to-end extensions are built in a git worktree (`e2ebuild.sh`), so edits in the main tree can't contaminate a measurement. An early baseline was lost this way and re-run.

## 7. Correctness issues found in the original code

_None yet. Reported here and to the user, not silently fixed._
