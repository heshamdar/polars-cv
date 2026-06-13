# Streaming-engine performance analysis — main vs other frameworks

**Date:** 2026-06-12
**Code under test:** `main` @ `0ccad81` (baseline), `claude/streaming-engine-benchmark-tqemga` @ `f14c9c1` (dev branch comparison)
**Hardware:** 4-core Intel Xeon @ 2.10 GHz (AVX-512), 15 GB RAM, Linux 6.18
**Software:** Python 3.11.15, Polars 1.37.1, OpenCV 4.11.0, torch 2.9.1 (CPU), Pillow (latest), release builds (`lto=fat`, `codegen-units=1`)

This directory contains the raw results (`*.json`), the comparison/driver scripts,
and this analysis. Headline tables are in `main_tables.md`; the analysis and
recommendations are below.

> **Reading the numbers.** OpenCV/Pillow/torchvision adapters run a
> single-threaded Python loop over pre-decoded images. polars-cv-streaming uses
> all 4 cores via the streaming engine's morsel parallelism; polars-cv-eager is
> effectively single-threaded (one plugin call over the whole column, sequential
> row loop inside). So `stream/ocv = 1.0` means *per-core* polars-cv is ~4×
> slower than OpenCV for that op; `stream/ocv = 4` means per-core parity.

---

## 1. Headline results (main branch)

Geometric mean of `polars-cv-streaming ÷ opencv` throughput over the
count=100 cells (256² and 512²):

| Bucket | Ops | geomean |
|---|---|---|
| Severe deficit | canny **0.10×**, dilate 0.23×, erode 0.25×, sharpen 0.27× | ~0.2× |
| Deficit | crop 0.34×, flip_h 0.40×, flip_v 0.45×, rotate_45 0.46×, blur 0.51×, hist_eq 0.57× | ~0.45× |
| Near parity | sobel 0.70×, threshold 0.73×, grayscale 0.73×, medical_pipeline 0.86×, adjust_brightness 1.01× | ~0.8× |
| Wins | heavy/medium/imagenet pipelines 1.1–1.2×, e2e workflows 1.25–2.5×, normalize 1.56×, invert 1.59×, pad 1.67×, rotate_90 1.98×, adjust_contrast 2.87× | ~1.6× |
| Big wins | light_pipeline 4.5×, **resize 7.9×** | ~6× |

Overall geomean across all 56 cells: **0.88×** — i.e. with 4 cores polars-cv
streaming roughly matches single-threaded OpenCV on this op mix, wins on
realistic multi-op pipelines and end-to-end workloads, and loses heavily on a
handful of individual kernels.

Full tables: [`main_tables.md`](./main_tables.md).

## 2. Streaming-engine behavior (the engine is not the problem)

Dedicated experiments (`streaming_deep_dive.py`, results in `deep_dive_*.json`):

| Experiment | Result |
|---|---|
| ImageNet-style pipeline, 1000×256² blobs, eager | 801–842 rows/s regardless of `POLARS_MAX_THREADS` → **eager is single-threaded** (one plugin call, sequential row loop, no rayon) |
| Same, streaming, threads 1→2→4 | 1 260 → 2 754 → 4 744 rows/s — **near-linear morsel-parallel scaling (3.8× at 4 threads)** |
| Same, streaming@1-thread vs eager | 1 260 vs 801 rows/s — streaming is **1.57× faster even single-threaded**; eager buffers all row results (~150 KB/row × 1000) before building the output column, streaming works morsel-sized |
| Morsel size (default vs 10 vs 1000) at 256² | 4 744 / 5 498 / 4 945 rows/s — **morsel-size tuning is noise at realistic image sizes**; the `POLARS_IDEAL_MORSEL_SIZE=10` hack in the benchmark adapter is unnecessary on Polars 1.37 |
| 20 000 tiny 8×8 rows, trivial op, default vs morsel=10 | 1.81 M vs 0.93 M rows/s — small morsels *hurt* tiny-row workloads; per-plugin-call overhead ≈ **5 µs** (graph-JSON parse + compile + dispatch, on `main`) |
| PNG source vs pre-decoded blob source (resize, 4 threads) | 4 519 vs 16 481 rows/s → **PNG decode is ~72 % of e2e time** |
| blob sink vs numpy struct sink | 17 518 vs 16 481 rows/s — sink encoding choice is immaterial |
| Identity blob→blob round trip, 256² | ~17 µs/row wall (4 threads): fixed per-row protocol cost (zero-copy decode, two output copies) |

The plugin expression is registered `is_elementwise=True` and appears inline in
the streaming physical plan (verified with `explain(engine="streaming")`) — no
fallback to the in-memory engine.

**Conclusion:** the streaming engine delivers what it promises (linear scaling,
better memory behavior than eager). The deficits in §1 are caused by the
*kernels* and by fixed per-row costs, not by engine integration.

## 3. Root causes of the deficits (code-level, `main`)

Ordered by measured impact.

### 3.1 `canny` — 0.10× (per-core ~40×)
`view-buffer/src/execution/runner.rs::apply_canny`:
- per-pixel `gy.atan2(gx).to_degrees()` for direction quantization — an
  expensive libm call per pixel where OpenCV uses sign/ratio comparisons
  against tan(22.5°)/tan(67.5°);
- 5×5 Gaussian smoothing as a naive 25-tap 2-D convolution
  (`gaussian_blur_5x5`) instead of two separable 5-tap passes;
- hysteresis loops whole-image sweeps until fixpoint (`while changed` over all
  pixels) — worst-case O(n·passes); a stack/queue-based flood fill from strong
  seeds is single-pass;
- 6 full-image f32 temporaries (blurred, gx, gy, magnitude, direction, nms).

### 3.2 `erode`/`dilate` — ~0.24×
`morph_minmax_typed`: separable min/max, but the inner loop does
`(x + kx).clamp(0, w-1)` **per tap** and branches on `match kind` **per
element** through a generic `PartialOrd` — none of it vectorizes. OpenCV uses
van Herk/Gil-Werman (O(1) per pixel regardless of ksize) plus SIMD. Splitting
border columns from the interior (branch-free monomorphized min/max over
contiguous slices) typically recovers 4–8× before algorithmic changes.

### 3.3 `sharpen` — 0.27×
Sharpen lowers to the generic `apply_convolve2d` (`ops/filter.rs`):
- calls `sample_pixel` with a border-mode `match` for **every tap of every
  pixel** — no interior fast path;
- channel-outer loop walks memory with stride `c` (cache-hostile for RGB);
- input promoted whole-image u8→f32 and output left f32 (4× the bytes of
  OpenCV's u8 path).
An interior/border split with an unrolled 3×3 kernel over contiguous rows is
the standard fix; OpenCV's `filter2D` is the comparison point.

### 3.4 `crop` 0.34×, `flip` 0.40–0.45× — fixed per-row protocol cost, not kernels
These ops do near-zero compute; the measurement is dominated by the per-row
pipeline: zero-copy blob decode → strided `to_contiguous` copy → `to_blob()`
`Vec<u8>` (copy 1) → buffered in `Vec<RowResult>` → Arrow binary builder
(copy 2). OpenCV's comparison op is a NumPy view (crop) or single memcpy
(flip). Mitigations: encode directly into a pre-sized Arrow builder (eliminates
one copy and the `RowResult` buffering), and stream rows into the builder
instead of accumulating the whole batch (also fixes the eager memory behavior
from §2).

### 3.5 `rotate_45` (affine warp) — 0.46×
`affine_warp_typed`: f64 coordinate math per pixel, `match interpolation`
per pixel, generic `NumCast` per channel sample. Standard fixes: hoist the
interpolation dispatch out of the loop, incremental coordinate stepping
(`x_src += a` per column), f32 or 16.16 fixed-point arithmetic for u8, and a
u8×4-channel specialization.

### 3.6 `blur` — 0.51× *(already fixed on the dev branch)*
`main` delegates to `image::imageops::blur` (generic, slow). The dev branch
replaces it with a separable Gaussian (commit `4344897`) — see §4.

### 3.7 `histogram_equalize` 0.57×, `sobel` 0.70×, `threshold`/`grayscale` 0.73×
Kernels are reasonable; the residual gap is the §3.4 per-row cost plus modest
vectorization headroom (e.g. sobel via the generic convolution path).

### Not deficiencies
`resize` (7.9×, `fast_image_resize` with SIMD beats `cv2.resize` here),
`normalize`/`invert`/`pad`/`adjust_contrast`/`rotate_90` (1.5–2.9×), all
multi-op pipelines and e2e workflows (decode amortizes the per-row overhead and
streaming parallelism dominates).

## 4. Does the dev branch close the gap?

The session branch (`claude/streaming-engine-benchmark-tqemga`) already lands
several of the obvious fixes: separable Gaussian blur, cast-fusion into
`FusedKernel` + fixed loop order for auto-vectorization, a compiled-graph cache
keyed by graph JSON (kills the ~5 µs/call JSON re-parse), `x86-64-v3` codegen
for local builds, and removal of the no-win tiling strategy.

Measured on the identical harness (`branch_*.json`, opencv columns reused from
the main run since OpenCV is unchanged):

Summary (streaming mode, count=100/1000 cells; full table in
[`branch_compare.md`](./branch_compare.md)):

- **Geomean branch/main: 0.98× — flat overall**, decomposing into:
  - real wins of **1.1–1.4×** on the overhead/memory-bound ops
    (adjust_brightness@512 1.42×, histogram_equalize 1.30–1.39×, flips
    1.12–1.35×, crop 1.13–1.29×, normalize 1.18–1.28×, threshold@256 1.22×,
    resize@512 1.24×, rotate_90@512 1.25×) — consistent with the
    compiled-graph cache, fused casts, and `x86-64-v3` codegen;
  - **a 5× regression in `blur`** (2 443 → 452 img/s at 256², 603 → 120 at
    512²) that also drags `heavy_pipeline` to 0.37–0.47× of main;
  - everything else within noise (±10 %).

**The blur regression was diagnosed and fixed in this session.** The branch's
separable Gaussian (commit `4344897`) had the right algorithm but an inner
loop that could not vectorize: per-tap generic `NumCast` conversion of the
source pixel (k·n casts), per-tap index `clamp` even for interior pixels,
per-pixel f64 `clamp_for_dtype` in the vertical pass, and an extra
full-image `to_vec()`. The rewrite (this branch, see
`view-buffer/src/execution/runner.rs::separable_gaussian_blur_typed`)
converts the input to f32 once, accumulates each tap as a contiguous
shifted-slice multiply-add (interior/border split), and does the vertical
pass as whole-row multiply-adds. Post-fix numbers:

| Cell (streaming) | OpenCV | main | branch pre-fix | branch post-fix | post-fix vs main |
|---|---|---|---|---|---|
| blur 100×256² | 3 289 | 2 443 | 452 | **2 703** | 1.11× |
| blur 100×512² | 1 727 | 603 | 120 | **694** | 1.15× |
| heavy_pipeline 100×256² | 3 920 | 3 485 | 1 376 | **4 763** | 1.37× |
| heavy_pipeline 1000×256² | 4 411 | 3 978 | 1 462 | **5 457** | 1.37× |

With the fix, the separable algorithm finally pays off: blur beats `main`
by ~1.1× and `heavy_pipeline` (which is blur-dominated) moves from 0.89× to
**1.22–1.24× of OpenCV**. Blur at 512² remains ~0.4× of OpenCV per this
4-thread-vs-1-thread comparison — OpenCV's u8 fixed-point SIMD Gaussian is
still ~9× faster per core, which is recommendation #2's territory.

Branch deep-dive (same experiments as §2): imagenet-style streaming
4 744 → 5 491 rows/s (+16 %), tiny-row default-morsel 1.81 M → 2.03 M rows/s
(+12 %), tiny-row small-morsel 0.93 M → 1.11 M rows/s (+20 %, the
compiled-graph cache cutting per-call cost), eager +11 %.

The same inner-loop pattern (per-tap clamp + per-element generic dispatch) is
exactly what keeps erode/dilate/sharpen/affine slow on `main` — the fix here
doubles as the template for recommendation #2 below.

## 5. Recommendations (prioritized) — status after the follow-up workstreams

> Status notes added 2026-06-13 after the optimization workstreams landed on
> this branch (commits `1abdd55..`). Per-recommendation outcomes below; every
> kernel change is gated by a naive-reference equivalence test (bit-exact)
> in `view-buffer/tests/*_ref.rs`.

1. **Canny rewrite** — **DONE** (atan2-free direction quantization, blur via
   the vectorized 2-D convolution — the 159-kernel is NOT separable, so the
   "separable 5-tap" idea was wrong — worklist hysteresis, fused
   threshold/NMS). Measured: 2.6–2.9× kernel speedup, 0.10× → 0.36×@256² /
   0.19×@512² of OpenCV. The remaining gap is semantic: `cv2.Canny` uses L1
   magnitude + integer Sobel SIMD by default; matching it would change
   output. Squared-magnitude thresholds were *not* adopted (not bit-exact:
   f32 sqrt rounding flips tie-inclusive NMS comparisons).
2. **Interior/border split for neighborhood kernels** — **DONE**. sharpen
   0.27× → **0.99–1.04×** of OpenCV; sobel 0.70× → **2.45–2.75×**;
   erode/dilate 0.24× → **1.15×**@256² / 0.73×@512². van Herk still deferred.
3. **Affine warp** — **ATTEMPTED, REVERTED** (commit `77a2f8d`): the
   monomorphized interior-fast-path rewrite measured ~10 % *slower* than the
   original in production context despite an identical-structure copy being
   ~10 % faster in the test crate (inlining-context sensitivity). The
   equivalence tests stay as guards; the real win is an f32/16.16 fixed-point
   u8 path like OpenCV's warpAffine — a tolerance-gated semantics change,
   still open.
4. **Output-path copy elimination** — **DONE** (`write_blob_into` + a
   single-output Binary fast path appending straight into the Arrow builder;
   error-policy semantics pinned by `tests/test_binary_fastpath.py`).
   Measured: flips +20–26 %, pad +13–28 %, resize@512² +25 %, memory −14 %
   across the regression suite.
5. **Within-call row parallelism (rayon)** — deferred by decision (streaming
   already scales linearly; eager-only benefit).
6. **PNG decode** — **EVALUATED, NOT ADOPTED.** zune-png measured a geomean
   **0.35×** (i.e. ~3× *slower*) vs `image::load_from_memory` on the 8-bit
   RGB/RGBA gradient+noise corpus at 256²/512²/1024² (adoption gate was
   ≥1.3×); the modern `png`+fdeflate stack has overtaken it. Harness and
   pixel-parity test live in `view-buffer/tests/png_decode_eval.rs`
   (zune-png stays a dev-only dependency). JPEG decode-time downscaling
   (`decode_max_size`) already exists; PNG has no equivalent lever.
7. **Benchmark hygiene** — **DONE** (commit `1abdd55`): morsel-size hack
   removed from the adapter, regression suite, and inference comparison;
   sharpen reference test strengthened to a direct `cv2.filter2D` comparison.

Additionally, the caveat below about `PromoteToFloat` payloads is addressed:
`scale`/`clamp`/`adjust_brightness` now accept opt-in `preserve_dtype=True`
(lowers to a fused trailing cast; see `tests/test_preserve_dtype.py`).

## 6. Caveats

- 4 vCPUs on shared cloud hardware; absolute numbers are noisy (±10–20 % on
  small cells), ratios over count=100 cells are stable.
- The comparison frameworks run single-threaded Python loops by design (the
  suite's methodology) — OpenCV with its own threading or batched torchvision
  would shift the baselines.
- torchvision numbers are CPU-only (no MPS/CUDA available here).
- `pillow`/`torchvision` lack several ops (canny, sobel, erode/dilate on tv) —
  those cells are absent, not zero.
- Scalar ops follow the documented `PromoteToFloat` contract: u8 input →
  **f32 output** (verified for `adjust_brightness`). OpenCV's equivalents stay
  u8, so those cells compare different output payloads (4× the bytes on our
  side); adding an optional u8-preserving fast path would help both semantics
  parity and throughput.
