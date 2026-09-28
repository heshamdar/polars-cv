# Performance plan: kernels, copies and per-row overhead

## Context

The 2026-09-27 assessment found the remaining performance in the **engine's kernels and
materialisation paths**, not the plugin architecture (CR-31–40 already closed that).
Measured on this container (standalone `rustc -O`, 1024² RGB / 3M elements):

| pattern | baseline (wheel) | AVX2 |
|---|---:|---:|
| u8 grayscale as written (`push` loop, `runner.rs:1090`) | 2.45 ms | 2.12 ms |
| same maths into a preallocated slice | 0.98 ms | 0.27 ms |
| f32→u8 `x.round() as u8` (every float→int cast) | 10.3 ms (per-element `roundf` libcall) | 4.3 ms |
| u8 gamma via `powf` | 38 ms | 37 ms |
| u8 gamma via a 256-entry LUT | 2.2 ms | 1.7 ms |

This plan covers the improvements that act on **common operations**: grayscale, threshold,
cast/normalize (ImageNet preprocessing), brightness/contrast/gamma/invert/scalar chains,
resize (including after crop/flip), flips/crop/transpose materialisation, rotation, erode/dilate
with iterations, JPEG encode, list/raw/blob ingestion, f16 tensor sinks, and the per-row
executor overhead that dominates cheap rows.

**Out of scope** (edge cases, or already evaluated): percentile/median, van Herk morphology
for large kernels, `image` feature trimming, a PNG decoder swap (`tests/png_decode_eval.rs`
already ran that gate), 64-byte buffer alignment (≤ a few %), and fixed-point affine
(it would change output).

**Policy (user-approved).** Bit-identical output is the default, and every phase proves it
with tests. Three deliberate output changes are allowed, each with a CHANGELOG entry:
1. The z-score normalize mean/std are computed exactly.
2. The `adjust_contrast` mean is computed exactly.
3. The JPEG sink's bytes change if the encoder swap passes its eval gate (IJG is added to
   `deny.toml` only in that case).

## Unifying mechanisms (new canonical paths, per CLAUDE.md)

Each phase lands one mechanism that callers **cannot step around**. The old ad-hoc code is
deleted in the same change, never left beside it.

- **M1 — CPU dispatch: `simd_dispatch!`** (`view-buffer/src/execution/dispatch.rs`).
  - Generalises the blur pattern at `runner.rs:1755-1790`. The macro emits the portable
    body, an `unsafe #[target_feature(enable = "avx2")]` clone of the *whole* body, and the
    `is_x86_feature_detected!` dispatcher.
  - It enables AVX2 only, never FMA, so results stay bit-identical. AVX2 implies SSE4.1,
    which gives a vectorised `roundps`.
  - Every use registers in one `DISPATCHED_KERNELS` table. A single test,
    `dispatched_kernels_are_bit_identical`, iterates that table, replacing
    `blur_dispatch_is_bit_identical`, so registering a kernel is the same act as testing it.
  - Blur migrates onto the macro. No crate is added: `multiversion` would reintroduce the
    closure/target-feature pitfall CR-35 measured.
- **M2 — Element-wise engine** (`view-buffer/src/ops/elementwise.rs`).
  - `map_values(buf, op: &ValueMap) -> ViewBuffer` is the only route for per-value ops:
    `ComputeOp::scalar()` ops, `Fused`, `Scale`, `Relu`, `Clamp`, `Invert`, `AdjustGamma`,
    `AdjustContrast` (after its mean pass), `Normalize` (after its stats pass) and `Cast`.
  - It chooses the strategy centrally:
    - (a) **LUT** for u8/i8 input, and for u16 input when elements > 65,536. The per-value
      f32 function is evaluated once per possible input value with the *same* code the
      streaming pass uses, so bit-identity holds by construction. `Preset` normalize uses a
      per-channel LUT.
    - (b) **In place** when the buffer is contiguous, solely owned (`Arc::get_mut`, as
      `try_apply_fused_kernel_inplace` does today) and in dtype == out dtype. This extends
      today's f32-only rule to u8→u8 (invert, LUT chains).
    - (c) A **vectorised streaming pass** otherwise, dispatched via M1.
  - It replaces `apply_scalar_op`, `apply_scalar_op_f64`, `apply_scalar_owned_with`,
    `apply_scalar_op_with`, `apply_invert`, `apply_adjust_gamma`, `apply_adjust_contrast`,
    `apply_normalize_f32`, `gather_to_f32` and the `cast_impl!` body in `buffer.rs:283`.
  - The f64 cold path stays: f64 input computes in f64 through `ScalarOp::apply_f64`, the
    existing authority.
- **M3 — Strided traversal** (`view-buffer/src/core/strided.rs`).
  - `RowWalk`: coalesce mergeable dimensions, then iterate the outer index while stepping
    pointers along the inner two axes. The innermost run gets a const-generic element/pixel
    copy for C ∈ {1, 2, 3, 4}, and negative inner strides get a reversed loop.
  - It becomes the single reader of views. `copy_elements_into` (`buffer.rs`), the strided
    arm of `gather_to_f32`, and the strided fallbacks in grayscale and threshold all use it.
    "`to_contiguous()` then process" disappears from the kernels M2 and M5 own.
- **M4 — External adapters through `ExternalView`** (`interop/mod.rs`).
  - `ExternalLayout::FastImageResize` already exists but is bypassed: `resize_typed_*` hand
    raw pointers to fir (`runner.rs:921-1040`).
  - Add `FirViewAdapter<P>: ExternalView`, implementing fir 5.6's `ImageView` trait
    (`iter_rows`) over any dense-rows view, with row stride of either sign.
  - Change `LayoutFacts::compatible_with(FastImageResize)` from `is_contiguous()` to
    `is_dense_rows() && channels_last`. That makes crop and flip_v views resize with
    **zero copies**.
  - The JPEG encoder gets `JpegViewAdapter` the same way (implementing
    `jpeg_encoder::ImageBuffer::fill_buffers`), for the same zero-copy on strided views.
- **M5 — Float→int conversion authority** (`view-buffer/src/core/convert.rs`).
  - `f32_slice_to<T>(src, dst)`: round-then-saturate, M1-dispatched.
  - Used by M2's integer outputs, `finish_fused_output` (`buffer.rs:~1834`), blur's
    output store, affine, and normalize's `out_dtype`. It replaces the 32 scattered
    `.round() as` sites on hot paths.

## Phases (TDD: failing test first, watched failing; then implement; then perf gate)

Each phase is one or more commits on `claude/codebase-performance-assessment-36x2p3`, with:
- `scripts/verify.sh --fast` green;
- `maturin develop --profile benchmark` base-vs-head on `--changed REF`, interleaved per the
  2026-09-27 report because 4-thread noise is ±17%;
- a report in `polars-cv/benchmarks/reports/2026-MM-DD-<phase>/`.

### Phase 0 — Kernel benchmarks

- Add `view-buffer/benches/kernels.rs` (criterion is already a dev-dep; add a `[[bench]]`
  entry). It covers grayscale, threshold, cast f32→u8, gamma/normalize on u8, invert,
  flip_h/crop materialise, resize after crop, affine rotate 30°, erode k=3 ×3,
  JPEG/PNG encode, and a 3-op fused chain.
- Sizes 256², 512², 1024², at baseline and AVX2.
- Record the baseline in `benchmarks/reports/`.

### Phase 1 — M1 dispatch + M5 conversion + the vectorisation fixes

- **Tests first:**
  - `dispatched_kernels_are_bit_identical`, iterating the registry.
  - A `convert.rs` round/saturate/NaN table covering ties, ±inf and out-of-range values,
    matching today's `x.round() as T`.
- **Grayscale** (`runner.rs:1090`): write into a preallocated slice via
  `as_chunks::<3>()`/`::<4>()`, drop the redundant `.min(255)`, and put the u8 path under
  M1. The generic `grayscale_typed<T>` (`runner.rs:1167`) gets the same shape: the channel
  count is fixed at compile time, and it drops the f64 `NumCast::from(..).unwrap_or` per
  element.
- **Threshold** (`runner.rs:1266-1400`): delete `threshold_simd`'s per-row `Vec` and
  fixed-array chunking. Write one `map` into a preallocated output over M3 rows, and let a
  solely owned u8 input threshold in place (via M2's in-place rule).
- **Cast** (`buffer.rs:283`): float→int goes through M5, int→float through a dispatched
  pass. A strided input casts during the M3 walk instead of `to_contiguous` then cast.
- **Guards:**
  - Existing: `tests/fused_ops.rs`, `clamp_fusion.rs`, `blur_dtype.rs`,
    `polars-cv/tests/reference/test_color_ref.py` and `test_scalar_math_ref.py`.
  - New: a grayscale/threshold/cast parity test comparing each new kernel against a
    retained scalar reference implementation in `#[cfg(test)]`, over odd widths and
    strided inputs.

### Phase 2 — M2 element-wise engine (LUT / in-place / pass)

- **Tests first** (`view-buffer/tests/elementwise_engine.rs`):
  - For every `ValueMap` variant and every input dtype, the LUT, in-place and streaming
    strategies agree bit for bit with the scalar reference.
  - A strategy-selection table asserts u8 → LUT, solely owned same-dtype → in place, and
    shared → pass. This is watched failing while the old functions still route directly.
  - An allocation test in the style of `tests/copy_counts.rs`: a solely owned u8 `invert`
    makes 0 image-sized allocations, and u8 → `normalize(Preset)` → f32 makes exactly 1.
- **Normalize** (`runner.rs:214-363`):
  - Stats pass: MinMax min/max in the input dtype (vectorised, which also fixes the
    non-vectorising `fold(f32::min)`). Z-score uses an exact sum (integer accumulator for
    integer input, pairwise/multi-lane f64 for float), a CHANGELOG'd change.
  - Transform pass: M2, so u8 input goes through a per-channel LUT. This removes the
    u8→f32 cast copy, the `i % channels` modulo and the ndarray `view.iter()` path.
- **Contrast:** mean via the same exact-sum helper (CHANGELOG'd), then M2.
- **Gamma:** straight to M2, so u8 input goes through a LUT. This pass is bit-identical.
- **Delete** the functions listed under M2, with a CHANGELOG entry.
- **Guards:**
  - `tests/gamma_dtype_range.rs`, `f64_scalar_contract.rs` and `fused_ops.rs`.
  - `polars-cv/tests/test_dtype_contracts.py`, `reference/test_phase1_ref.py` and
    `reference/test_scalar_math_ref.py`.
  - Update the two z-score/contrast expectations deliberately, stating so in the commit.

### Phase 3 — M3 strided traversal (flips, crop, transpose, rotate-90)

> **Done** (`bf64725`, CR-55; report `polars-cv/benchmarks/reports/2026-09-28-phase3-strided/`).
> A gated tiled transpose was tried (`80d3b52`) and reverted (`c4475b6`): it made
> u8 transpose 1.7x slower at 1024². Transpose remains ~8x a vertical flip.

- **Tests first** (extend `view-buffer/tests/strided_ops.rs`):
  - A property test over random shapes (rank 1–4), slices, negative strides and
    transposes: `RowWalk` copy == the element-by-element reference.
  - Guard `copy_elements_into` itself with the same test before swapping its body.
- **Implement** `core/strided.rs`, then route through it:
  - `copy_elements_into`, which feeds `to_contiguous`, `write_to`/`append_to` and so every
    list/array sink and encode;
  - `gather_to_f32`'s strided arm, now inside M2;
  - the grayscale and threshold strided arms.
- **Guards:**
  - `tests/zero_copy.rs`, `memory_effect_verification.rs` and `stride_contract_fixes.rs`.
  - `polars-cv/src/graph/encode.rs::tensor_sink_tests`.
  - The slow-lane `tests/test_sink_cost_ratio.py`.

### Phase 4 — M4 resize adapter (resize after crop/flip is zero-copy)

- **Tests first:**
  - `resize(crop(x)) == resize(to_contiguous(crop(x)))` byte for byte, for u8/u16/f32 ×
    C ∈ {1, 2, 3, 4}, including flip_v and negative row strides.
  - A `copy_counts`-style test: resize of a crop makes exactly one image-sized allocation
    (the output).
  - The layout predicate change gets a fixture test in `core/layout.rs`: dense-rows views
    are accepted, while channel-strided views and horizontally flipped ones (negative
    pixel stride) are refused.
- **Implement:**
  - `interop/fir.rs`: `FirViewAdapter` / `FirView<'a, P>` implementing
    `fast_image_resize::ImageView` (rows sliced from the buffer by the row stride), with
    `LAYOUT = ExternalLayout::FastImageResize`.
  - `resize_strided` collapses to one generic function over the pixel type:
    `try_view` → resize, else `to_contiguous` → view.
  - `resize_typed_u8/u16/f32` are deleted, and `pixel_type_for`'s panic becomes an
    `ExternalView` error.
  - The destination buffer comes from fir's own `Image::new` (no `vec![0; n]` zero-fill),
    then is moved into `ViewBuffer::from_vec`.
- **Guards:** `polars-cv/tests/test_resize*.py` and the benchmarks' resize/imagenet
  pipelines.

### Phase 5 — Per-row executor overhead (cheap rows, 1–3 µs/row)

- **Tests first:**
  - Keep `static_segments_plan_once_per_source_layout` and `dynamic_segments_are_not_cached`
    (`compiled.rs`).
  - Add `a_cache_hit_takes_no_lock_and_allocates_nothing`, an allocation-counting row loop
    over 1k hits.
- **Implement in `polars-cv/src/graph/compiled.rs:1167-1272`:**
  - Compare the key against `source.shape()`/`strides_bytes()` slices, allocating only on a
    miss.
  - Put a per-range local `Vec<CachedPlan>` (ranges already own their `ParamCtx` and
    scratch) in front of the shared `RwLock`, which is taken only on a local miss.
  - Store `steps: Arc<[PlanStep]>`, and add `ExecutionPlan::execute_steps(source, &[PlanStep])`
    in `view-buffer/src/execution/plan.rs`. `apply_step` clones only the op a kernel
    consumes by value.
- **Chunk lookup:** `get_binary_row_buffer`, `is_array_row_null` and `get_array_row_buffer`
  (`decode.rs`) take a per-batch chunk-offset table (binary search) instead of a linear walk
  per row.
- **Perf gate:** `benchmarks/plugin_overhead.py` (no-op, `invert`, 3-op chain on 8×8 rows)
  plus `zero_copy_*`.

### Phase 6 — Ingestion: `List` source zero-copy, raw/blob alignment

- **Tests first** (`decode.rs` test module, in the style of `array_source_view_tests`):
  - Rectangular `List[List[u8]]` / `List[f32]` rows → the data pointer lies inside the
    column's leaf values buffer.
  - Jagged and null rows keep today's errors (`a_jagged_row_is_refused`,
    `a_null_value_or_inner_list_is_refused`).
  - Dtype-mismatch rows use one typed copy (no Series round trip).
  - For raw u8 rows at an odd address: read in place. For a blob whose payload is aligned
    for its dtype: in place. Misaligned payloads are still copied (keeping CR-41's
    `alignment_soundness.rs`).
- **Implement:**
  - `try_decode_list_zero_copy`: walk `ListArray` offsets level by level for the row, with
    the same rectangularity/null checks as `flatten_nested_series`. The leaf `Buffer<T>` is
    sliced via `try_transmute::<u8>()`, the CR-40 trick.
  - `decode_list_with_copy` keeps only the cast case, written as one typed pass. Delete
    `series_to_bytes`'s iterator-of-`to_ne_bytes`.
  - `get_binary_row_buffer` takes the required alignment: the element size for raw rows,
    and for blobs the header is parsed first and the payload's absolute alignment checked,
    as `decode_blob_zero_copy` already does. The unconditional 8-byte rule goes.
- **Guards:** `test_zero_copy*.py` and `test_list_source*.py`.

### Phase 7 — Rotation / affine (augmentation path), bit-identical

- **Tests first:**
  - `tests/affine_ref.rs` and `rotation.rs` gain a byte-parity test comparing the new
    kernel against the current one, kept as `#[cfg(test)]` reference, over angles
    {1, 30, 45, 137}°, bilinear/nearest, C ∈ {1, 3, 4}, u8/f32 and non-square images.
  - `rotate(0°)` returns a buffer sharing the input's data (`Arc::ptr_eq`), with no warp.
- **Implement in `affine_warp_typed`** (`runner.rs:1457`):
  - The channel count becomes a const generic C (1–4 plus a dynamic fallback).
  - Source coordinates step incrementally per row. They must be recomputed exactly as
    `a*x + b*y + tx`, evaluated in the same order so results stay bit-identical; an
    additive step is only allowed if the parity test proves it.
  - Split each output row into [border | interior | border]. The interior path has no
    bounds checks or `NumCast`, and stores through M5.
  - Dispatch via M1.
  - `ComputeOp::lowered` lowers `Rotation::Identity` to the no-op instead of
    `RotateAffine{0°}` (`compute.rs:456`).
- **Guards:** `reference/test_affine_ref.py`.

### Phase 8 — Morphology iterations, blur input conversion

- `erode/dilate(ksize=k, iterations=n)` becomes one pass with k' = n(k−1)+1. This is exact
  for the rectangular element with replicate border; the plan includes the proof sketch in
  the doc comment.
  - Test first: parity against n repeated passes for k ∈ {3, 5}, n ∈ {1, 2, 4}, on u8/f32,
    including images narrower than the window.
  - Implemented in `apply_erode`/`apply_dilate` (`runner.rs:1953-1995`).
- **Blur:** convert each input row to f32 into a thread-local row scratch inside the
  horizontal pass, instead of allocating a full-image `input_f32`
  (`runner.rs:1837`). The output store uses M5. `blur_dtype.rs` and the dispatch test
  guard it.

### Phase 9 — Sinks: JPEG encoder (eval-gated), f16

- **Eval gate first:** `view-buffer/tests/jpeg_encode_eval.rs`, mirroring
  `png_decode_eval.rs`.
  - It compares `image::codecs::jpeg::JpegEncoder` against `jpeg-encoder 0.7.1`
    (`features = ["simd"]`) on gradient/noise × 256²/512²/1024² × L8/Rgb8, q = 75/90.
  - Decode-back PSNR parity within 0.5 dB always runs.
  - Adopt only if the geomean speedup is ≥ 1.5×. Otherwise stop the JPEG part of this
    phase and record the result in the report.
- **If adopted:**
  - Add the dependency to `view-buffer/Cargo.toml` under the `image_interop` feature.
  - Add `IJG` to `deny.toml` `allow`, with a comment.
  - `ImageAdapter::encode_jpeg` (`interop/image.rs:437`) keeps `encode_in_place`'s
    structure: contiguous u8 goes to `Encoder::encode`, and dense-rows views go through
    `JpegViewAdapter::try_view` → `encode_image`, with zero copies.
  - Alpha/La8 conversion keeps its existing `converted` fallback.
  - The `ImageCodec::check_support` authority is unchanged (JPEG width ≤ 65,535 already
    holds).
  - Update the byte-pinning tests deliberately (CHANGELOG'd). Pixel-tolerance tests must
    pass unchanged.
- **f16 sink** (`polars-cv/src/output.rs:102`):
  - Replace cast → `to_contiguous` → per-element `extend_from_slice` with one M3 walk into
    an f32 row scratch, then `half::slice::HalfFloatSliceExt::convert_from_f32_slice`
    (`half` 2.7.1, already a dependency) into the output.
  - Test first: bit-identical to the existing per-element `f16::from_f32` over all
    special values.

## Critical files

- `view-buffer/src/execution/runner.rs` (most kernels)
- `view-buffer/src/core/buffer.rs` (`to_contiguous`, `copy_elements_into`, `cast_to`, fused)
- `view-buffer/src/ops/scalar.rs`
- `view-buffer/src/core/layout.rs`
- `view-buffer/src/interop/{mod.rs, image.rs, fir.rs (new)}`
- New: `view-buffer/src/{execution/dispatch.rs, ops/elementwise.rs, core/strided.rs, core/convert.rs}`
- `polars-cv/src/graph/{compiled.rs, decode.rs}`
- `polars-cv/src/output.rs`
- `view-buffer/Cargo.toml` and `deny.toml` (phase 9 only)
- `CHANGELOG.md`, and `CODE_REVIEW_FINDINGS.md` (a new "Performance review 2026-09-27"
  section, CR-50+, one entry per phase, closed as each lands)
- `view-buffer/AGENTS.md` and `AGENTS.md` Canonical Paths table (M1–M5 rows and their guards)

## Verification

1. Per phase:
   - `scripts/with-pyo3-env.sh cargo test -p view-buffer --all-features` and
     `cargo test -p polars-cv`;
   - `maturin develop`, then `scripts/verify.sh --fast`, reading its single PASS/FAIL line;
   - every new guard watched failing first, noted in the commit.
2. Bit-identity:
   - the M1 registry test;
   - the per-phase parity tests against retained `#[cfg(test)]` references;
   - the reference suite `polars-cv/tests/reference/` unchanged, except the three
     CHANGELOG'd expectations.
3. Performance:
   - `cargo bench -p view-buffer --bench kernels` at baseline and with
     `RUSTFLAGS="-C target-cpu=x86-64-v3"`;
   - then `maturin develop --profile benchmark` base vs head with
     `python -m benchmarks.regression.run_suite --select "$(python -m benchmarks.regression.relevance REF)"`
     and `compare.py`, interleaving binaries for any 4-thread flag;
   - `benchmarks/plugin_overhead.py` for phase 5.
4. Streaming safety: no kernel adds threads. The existing
   `tests/test_parallel_rows.py` and the streaming lanes of the harness must show no
   regression at 1 and 4 threads.
5. At the end: `scripts/verify.sh` (full, including the slow lane), `cargo deny check`,
   and `scripts/dev-clean.sh`.
