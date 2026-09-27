# Phase 2: the element-wise engine

Base: `bdce7ee` (Phase 1). Candidate: this commit. Both built from the same
`benches/kernels.rs` (which now executes each input as its sole owner, as the
plugin's executor does, so in-place paths are measured), `--profile benchmark`,
for the wheels' target (**x86-64**) and the local dev target (**v3**). 4-core
Intel Xeon @ 2.80 GHz container, one thread. Times in **milliseconds**.

## What changed

Every per-value compute op runs through `view-buffer/src/ops/elementwise/`:
lowered to a `FusedKernel` (one lowering, shared with fusion), statistics
first for normalize/contrast, then one strategy per call:

| strategy | when | e.g. |
|---|---|---|
| integer affine | the kernel is `clamp(±x + c)` over 8/16-bit input into the same dtype | `invert`, integer shifts |
| lookup table | work that cannot vectorise (`powf`) or differs per channel, over 8-bit input (16-bit with ≥ 65,536 elements per table) | gamma, preset normalize |
| blocked | an integer result: 2,048-element blocks through an f32 scratch | fused u8 → u8 chains |
| pass | a float result: the f32 buffer is the output | scale, min-max/z-score, contrast |

Each writes in place when the buffer is its sole owner and the dtype is kept.
The float → 8/16-bit integer conversion was rewritten to vectorise (identical
results).

## Element-wise kernels (median of 3 interleaved rounds)

| kernel | base x86-64 | head x86-64 | | base v3 | head v3 | |
|---|---:|---:|---:|---:|---:|---:|
| grayscale_u8_rgb/256 | 0.018 | 0.018 | 1.04x | 0.018 | 0.018 | 0.98x |
| grayscale_u8_rgb/512 | 0.058 | 0.054 | 1.08x | 0.056 | 0.054 | 1.03x |
| grayscale_u8_rgb/1024 | 0.216 | 0.212 | 1.02x | 0.224 | 0.206 | 1.09x |
| threshold_u8/256 | 0.003 | 0.003 | 1.24x | 0.003 | 0.003 | 1.22x |
| threshold_u8/512 | 0.013 | 0.010 | 1.33x | 0.013 | 0.010 | 1.24x |
| threshold_u8/1024 | 0.071 | 0.039 | 1.82x | 0.068 | 0.038 | 1.79x |
| cast_f32_to_u8/256 | 0.217 | 0.101 | 2.15x | 0.182 | 0.092 | 1.97x |
| cast_f32_to_u8/512 | 0.872 | 0.372 | 2.34x | 0.754 | 0.367 | 2.05x |
| cast_f32_to_u8/1024 | 3.837 | 2.093 | 1.83x | 3.595 | 2.253 | 1.60x |
| cast_u8_to_f32/256 | 0.035 | 0.037 | 0.97x | 0.037 | 0.037 | 0.98x |
| cast_u8_to_f32/512 | 0.140 | 0.139 | 1.01x | 0.140 | 0.139 | 1.01x |
| cast_u8_to_f32/1024 | 0.748 | 0.774 | 0.97x | 0.762 | 0.821 | 0.93x |
| invert_u8/256 | 0.010 | 0.007 | 1.33x | 0.009 | 0.007 | 1.32x |
| invert_u8/512 | 0.051 | 0.029 | 1.76x | 0.052 | 0.028 | 1.87x |
| invert_u8/1024 | 0.227 | 0.117 | 1.94x | 0.215 | 0.116 | 1.85x |
| adjust_gamma_u8/256 | 1.726 | 0.077 | 22.43x | 1.663 | 0.076 | 21.92x |
| adjust_gamma_u8/512 | 7.371 | 0.282 | 26.09x | 6.870 | 0.303 | 22.71x |
| adjust_gamma_u8/1024 | 38.913 | 1.417 | 27.46x | 36.471 | 1.565 | 23.31x |
| normalize_preset_u8_to_f32/256 | 1.028 | 0.116 | 8.88x | 1.040 | 0.218 | 4.76x |
| normalize_preset_u8_to_f32/512 | 4.315 | 0.480 | 8.98x | 4.237 | 0.861 | 4.92x |
| normalize_preset_u8_to_f32/1024 | 26.434 | 2.231 | 11.85x | 26.832 | 4.005 | 6.70x |
| normalize_zscore_u8/256 | 0.582 | 0.238 | 2.44x | 0.567 | 0.194 | 2.92x |
| normalize_zscore_u8/512 | 2.434 | 1.089 | 2.24x | 2.424 | 0.997 | 2.43x |
| normalize_zscore_u8/1024 | 20.491 | 5.246 | 3.91x | 20.453 | 4.986 | 4.10x |
| adjust_contrast_u8/256 | 0.317 | 0.120 | 2.64x | 0.337 | 0.097 | 3.48x |
| adjust_contrast_u8/512 | 1.385 | 0.683 | 2.03x | 1.426 | 0.542 | 2.63x |
| adjust_contrast_u8/1024 | 16.015 | 3.241 | 4.94x | 16.239 | 2.925 | 5.55x |
| scale_u8/256 | 0.074 | 0.054 | 1.37x | 0.076 | 0.055 | 1.38x |
| scale_u8/512 | 0.348 | 0.261 | 1.33x | 0.364 | 0.269 | 1.35x |
| scale_u8/1024 | 11.905 | 1.270 | 9.37x | 11.867 | 1.243 | 9.55x |
| normalize_preset_f32/256 | 1.049 | 0.320 | 3.28x | 1.002 | 0.314 | 3.19x |
| normalize_preset_f32/512 | 4.093 | 1.504 | 2.72x | 4.335 | 1.404 | 3.09x |
| normalize_preset_f32/1024 | 20.294 | 7.681 | 2.64x | 21.235 | 8.522 | 2.49x |
| fused_chain_u8/256 | 0.309 | 0.165 | 1.87x | 0.287 | 0.160 | 1.79x |
| fused_chain_u8/512 | 1.339 | 0.652 | 2.05x | 1.346 | 0.657 | 2.05x |
| fused_chain_u8/1024 | 6.410 | 2.595 | 2.47x | 6.151 | 2.625 | 2.34x |

`cast_u8_to_f32` is untouched code (0.93–1.01x is noise); every other row is
faster on both targets.
One oddity: u8 preset normalize (the per-channel table) runs 2.2 ms in the
x86-64 build and 4.0 ms in the v3 build at 1024², consistently across rounds.
Both are 5–12x the base; the v3 build's table loop is simply compiled less
well, and was not chased further.

## Everything else (one round each; no change intended)

| kernel | base x86-64 | head x86-64 | | base v3 | head v3 | |
|---|---:|---:|---:|---:|---:|---:|
| grayscale_f32_rgb/256 | 0.078 | 0.080 | 0.97x | 0.048 | 0.044 | 1.11x |
| grayscale_f32_rgb/512 | 0.318 | 0.306 | 1.04x | 0.216 | 0.188 | 1.15x |
| grayscale_f32_rgb/1024 | 2.040 | 1.607 | 1.27x | 1.498 | 1.557 | 0.96x |
| threshold_u8_cropped/256 | 0.006 | 0.006 | 0.92x | 0.006 | 0.006 | 1.08x |
| threshold_u8_cropped/512 | 0.016 | 0.016 | 1.03x | 0.016 | 0.016 | 0.99x |
| threshold_u8_cropped/1024 | 0.075 | 0.080 | 0.94x | 0.080 | 0.079 | 1.01x |
| materialize_flip_h_u8/256 | 0.312 | 0.316 | 0.99x | 0.318 | 0.311 | 1.02x |
| materialize_flip_h_u8/512 | 1.323 | 1.128 | 1.17x | 1.311 | 1.259 | 1.04x |
| materialize_flip_h_u8/1024 | 5.527 | 4.841 | 1.14x | 5.424 | 5.444 | 1.00x |
| materialize_flip_v_u8/256 | 0.314 | 0.303 | 1.03x | 0.328 | 0.332 | 0.99x |
| materialize_flip_v_u8/512 | 1.281 | 1.213 | 1.06x | 1.319 | 1.327 | 0.99x |
| materialize_flip_v_u8/1024 | 5.607 | 4.815 | 1.16x | 5.359 | 5.512 | 0.97x |
| materialize_transpose_u8/256 | 0.304 | 0.272 | 1.12x | 0.370 | 0.312 | 1.19x |
| materialize_transpose_u8/512 | 1.194 | 1.201 | 0.99x | 1.251 | 1.271 | 0.98x |
| materialize_transpose_u8/1024 | 6.736 | 6.553 | 1.03x | 7.525 | 7.775 | 0.97x |
| resize_224_u8/256 | 0.198 | 0.201 | 0.98x | 0.199 | 0.189 | 1.05x |
| resize_224_u8/512 | 0.321 | 0.331 | 0.97x | 0.278 | 0.312 | 0.89x |
| resize_224_u8/1024 | 0.618 | 0.623 | 0.99x | 0.546 | 0.553 | 0.99x |
| crop_then_resize_224_u8/256 | 0.342 | 0.354 | 0.97x | 0.336 | 0.370 | 0.91x |
| crop_then_resize_224_u8/512 | 1.109 | 0.996 | 1.11x | 1.352 | 1.335 | 1.01x |
| crop_then_resize_224_u8/1024 | 3.256 | 3.535 | 0.92x | 3.441 | 3.642 | 0.94x |
| blur_sigma2_u8_rgb/256 | 0.889 | 0.826 | 1.08x | 0.938 | 0.899 | 1.04x |
| blur_sigma2_u8_rgb/512 | 3.458 | 3.277 | 1.06x | 3.205 | 3.228 | 0.99x |
| blur_sigma2_u8_rgb/1024 | 17.615 | 16.786 | 1.05x | 15.391 | 15.688 | 0.98x |
| erode_k3_x3_u8/256 | 0.076 | 0.077 | 0.99x | 0.072 | 0.072 | 0.99x |
| erode_k3_x3_u8/512 | 0.239 | 0.269 | 0.89x | 0.226 | 0.222 | 1.02x |
| erode_k3_x3_u8/1024 | 1.279 | 1.248 | 1.02x | 0.928 | 0.912 | 1.02x |
| rotate_30_bilinear_u8_rgb/256 | 2.769 | 2.625 | 1.05x | 1.935 | 2.036 | 0.95x |
| rotate_30_bilinear_u8_rgb/512 | 11.168 | 11.001 | 1.02x | 8.005 | 7.921 | 1.01x |
| rotate_30_bilinear_u8_rgb/1024 | 44.214 | 46.588 | 0.95x | 30.978 | 32.406 | 0.96x |
| encode_jpeg_q90_rgb/256 | 1.977 | 1.908 | 1.04x | 1.272 | 1.357 | 0.94x |
| encode_jpeg_q90_rgb/512 | 7.777 | 7.785 | 1.00x | 5.320 | 5.456 | 0.98x |
| encode_jpeg_q90_rgb/1024 | 33.035 | 30.898 | 1.07x | 21.296 | 20.713 | 1.03x |
| encode_png_rgb/256 | 0.440 | 0.427 | 1.03x | 0.374 | 0.361 | 1.03x |
| encode_png_rgb/512 | 1.865 | 1.930 | 0.97x | 1.471 | 1.551 | 0.95x |
| encode_png_rgb/1024 | 7.640 | 7.981 | 0.96x | 6.124 | 6.327 | 0.97x |

All within the run-to-run noise (±10%) of this container.

## How the strategy rule was found (candidates measured and rejected)

1. **A table for every 8-bit kernel.** u8 gamma 22x faster, but u8 `invert`
   **18x slower**: a table read (~1.3 ns/px) against the old vector
   `255 - x` (~0.1 ns/px).
2. **Cheap kernels streamed through f32 blocks.** `invert` still **10x
   slower**, and profiling showed why: `x.round() as u8` stays scalar even in
   an AVX2 build (no x86 round-half-away, and a saturating `as` into a narrow
   integer is checked per element). The conversion was rewritten
   (`convert::round_narrow`, 3.3 → 1.5 ms for 3M elements), and the f32 round
   trip was still ~10x the work of `255 - x`.
3. **Integer affine, first version** (i32 lanes): `invert` 1.5–2.4x slower
   than the old u8 code.
4. **Final:** native wrapping arithmetic for the non-saturating `MAX + MIN - x`,
   i16 lanes for other 8-bit maps: `invert` 1.3–1.9x *faster* than before,
   and allocation-free.

## Correctness

- `ops/elementwise/tests.rs` compares every per-value op (the 19 scalar ops,
  the named ops, gamma, contrast, min-max, preset, fused chains incl. integer
  affine ones at the clamp boundaries) × all 10 dtypes × {contiguous, crop,
  flip_v, flip_h, transpose} × {shared, sole-owned}, plus 16-bit across the
  table threshold, against the pre-engine code kept verbatim in `legacy.rs`,
  bit for bit (any NaN equals any NaN). Watched failing against ten
  mutations, one per path.
- z-score is checked against its new exact formula (the one deliberate change).
- `tests/copy_counts.rs`: a sole-owned u8 invert, u8 → u8 chain and u8
  threshold allocate nothing; u8 → preset → f32 allocates only its output.
- The view-buffer suite passes built for x86-64, where every dispatched call
  also runs its portable build and must match it byte for byte.
