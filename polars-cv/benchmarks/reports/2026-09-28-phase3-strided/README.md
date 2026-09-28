# Phase 3: one walk over a view's memory

Base: `0ecfe8f` (Phase 2 + CR-53). Candidate: `c4475b6` (`core::strided::Walk`,
`bf64725`, with the transpose tiling reverted). Both built from the same
`benches/kernels.rs`, `--profile benchmark`, for the wheels' target
(**x86-64**) and the local dev target (**v3**). 4-core Intel Xeon @ 2.80 GHz
container, one thread. Times in **milliseconds**, median of three interleaved
rounds (base, head, base, head, …). Each build ran after `cargo clean -p
view-buffer --profile benchmark`, and base and head binaries were compared with
`cmp` so a stale artifact cannot pose as either side.

## What changed

`Walk::of` coalesces a view's layout once into packed **units** (the innermost
contiguous axes), evenly spaced **rows** of units, and **outer** axes walked by
an odometer:

| view (u8 `[H, W, 3]`) | unit | row | per copy |
|---|---|---|---|
| contiguous | the whole buffer | — | one `memcpy` |
| crop / flip_v | one image row | H rows | one `memcpy` per row |
| flip_h | one pixel (3 B) | W pixels, step −3 B | a 3-byte load/store |
| transpose | one pixel (3 B) | H pixels, a source row apart | a 3-byte load/store |

It is behind `to_contiguous`, `append_to`/`write_to` (every list/array sink),
`cast` of a view and the element-wise engine's read of a strided view (both
convert straight from the runs, no packed copy first). Grayscale's per-pixel
strided fallback is gone: other layouts pack, then run the dense kernel.

## Results

Rows marked † come from a separate three-round run of base vs `bf64725`
(`transpose/`): the full run's head binaries (`full/`) were built with the
tiling that was then reverted, which only affects those cases. Every other
row is `full/`.

### x86-64

| kernel | base `0ecfe8f` | head `c4475b6` | speedup |
|---|---:|---:|---:|
| grayscale_u8_rgb/256 | 0.023 | 0.023 | 0.98x |
| grayscale_u8_rgb/512 | 0.067 | 0.065 | 1.03x |
| grayscale_u8_rgb/1024 | 0.290 | 0.286 | 1.01x |
| grayscale_f32_rgb/256 | 0.103 | 0.105 | 0.98x |
| grayscale_f32_rgb/512 | 0.427 | 0.415 | 1.03x |
| grayscale_f32_rgb/1024 | 2.459 | 2.461 | 1.00x |
| threshold_u8/256 | 0.003 | 0.003 | 0.99x |
| threshold_u8/512 | 0.011 | 0.011 | 0.97x |
| threshold_u8/1024 | 0.045 | 0.044 | 1.01x |
| threshold_u8_cropped/256 | 0.008 | 0.007 | 1.03x |
| threshold_u8_cropped/512 | 0.019 | 0.019 | 0.99x |
| threshold_u8_cropped/1024 | 0.092 | 0.091 | 1.01x |
| cast_f32_to_u8/256 | 0.138 | 0.137 | 1.00x |
| cast_f32_to_u8/512 | 0.558 | 0.560 | 1.00x |
| cast_f32_to_u8/1024 | 2.724 | 2.752 | 0.99x |
| cast_u8_to_f32/256 | 0.045 | 0.045 | 1.00x |
| cast_u8_to_f32/512 | 0.179 | 0.175 | 1.03x |
| cast_u8_to_f32/1024 | 1.363 | 1.257 | 1.08x |
| materialize_flip_h_u8/256 | 0.352 | 0.056 | 6.32x |
| materialize_flip_h_u8/512 | 1.468 | 0.230 | 6.38x |
| materialize_flip_h_u8/1024 | 5.808 | 1.126 | 5.16x |
| materialize_flip_v_u8/256 | 0.341 | 0.010 | 33.01x |
| materialize_flip_v_u8/512 | 1.425 | 0.060 | 23.90x |
| materialize_flip_v_u8/1024 | 6.092 | 0.281 | 21.70x |
| materialize_transpose_u8/256 † | 0.357 | 0.093 | 3.83x |
| materialize_transpose_u8/512 † | 1.476 | 0.331 | 4.46x |
| materialize_transpose_u8/1024 † | 13.726 | 2.499 | 5.49x |
| cast_u8_to_f32_flip_h/256 | 0.445 | 0.098 | 4.55x |
| cast_u8_to_f32_flip_h/512 | 1.784 | 0.391 | 4.56x |
| cast_u8_to_f32_flip_h/1024 | 8.276 | 2.462 | 3.36x |
| scale_f32_transposed/256 † | 1.229 | 0.198 | 6.20x |
| scale_f32_transposed/512 † | 9.105 | 2.947 | 3.09x |
| scale_f32_transposed/1024 † | 59.164 | 20.883 | 2.83x |
| grayscale_u8_flip_h/256 | 0.394 | 0.072 | 5.47x |
| grayscale_u8_flip_h/512 | 1.450 | 0.291 | 4.98x |
| grayscale_u8_flip_h/1024 | 6.387 | 1.363 | 4.69x |
| resize_224_u8/256 | 0.265 | 0.271 | 0.98x |
| resize_224_u8/512 | 0.398 | 0.422 | 0.94x |
| resize_224_u8/1024 | 0.891 | 0.856 | 1.04x |
| crop_then_resize_224_u8/256 | 0.425 | 0.235 | 1.80x |
| crop_then_resize_224_u8/512 | 1.154 | 0.397 | 2.91x |
| crop_then_resize_224_u8/1024 | 3.806 | 0.842 | 4.52x |

### v3

| kernel | base `0ecfe8f` | head `c4475b6` | speedup |
|---|---:|---:|---:|
| grayscale_u8_rgb/256 | 0.024 | 0.025 | 0.97x |
| grayscale_u8_rgb/512 | 0.076 | 0.074 | 1.02x |
| grayscale_u8_rgb/1024 | 0.326 | 0.319 | 1.02x |
| grayscale_f32_rgb/256 | 0.067 | 0.069 | 0.98x |
| grayscale_f32_rgb/512 | 0.294 | 0.297 | 0.99x |
| grayscale_f32_rgb/1024 | 2.055 | 2.105 | 0.98x |
| threshold_u8/256 | 0.003 | 0.003 | 1.01x |
| threshold_u8/512 | 0.011 | 0.011 | 1.00x |
| threshold_u8/1024 | 0.046 | 0.046 | 1.00x |
| threshold_u8_cropped/256 | 0.009 | 0.008 | 1.09x |
| threshold_u8_cropped/512 | 0.022 | 0.021 | 1.04x |
| threshold_u8_cropped/1024 | 0.096 | 0.096 | 1.00x |
| cast_f32_to_u8/256 | 0.140 | 0.142 | 0.99x |
| cast_f32_to_u8/512 | 0.573 | 0.575 | 1.00x |
| cast_f32_to_u8/1024 | 2.957 | 2.998 | 0.99x |
| cast_u8_to_f32/256 | 0.045 | 0.045 | 1.00x |
| cast_u8_to_f32/512 | 0.183 | 0.175 | 1.04x |
| cast_u8_to_f32/1024 | 1.378 | 1.334 | 1.03x |
| materialize_flip_h_u8/256 | 0.385 | 0.055 | 7.07x |
| materialize_flip_h_u8/512 | 1.565 | 0.232 | 6.73x |
| materialize_flip_h_u8/1024 | 6.387 | 1.181 | 5.41x |
| materialize_flip_v_u8/256 | 0.376 | 0.011 | 34.83x |
| materialize_flip_v_u8/512 | 1.521 | 0.061 | 25.12x |
| materialize_flip_v_u8/1024 | 6.457 | 0.308 | 20.96x |
| materialize_transpose_u8/256 † | 0.393 | 0.102 | 3.84x |
| materialize_transpose_u8/512 † | 1.660 | 0.329 | 5.05x |
| materialize_transpose_u8/1024 † | 12.857 | 2.522 | 5.10x |
| cast_u8_to_f32_flip_h/256 | 0.502 | 0.103 | 4.87x |
| cast_u8_to_f32_flip_h/512 | 1.911 | 0.435 | 4.39x |
| cast_u8_to_f32_flip_h/1024 | 9.174 | 2.991 | 3.07x |
| scale_f32_transposed/256 † | 1.239 | 0.190 | 6.53x |
| scale_f32_transposed/512 † | 9.374 | 2.604 | 3.60x |
| scale_f32_transposed/1024 † | 60.209 | 21.402 | 2.81x |
| grayscale_u8_flip_h/256 | 0.459 | 0.085 | 5.43x |
| grayscale_u8_flip_h/512 | 1.691 | 0.345 | 4.91x |
| grayscale_u8_flip_h/1024 | 6.984 | 1.440 | 4.85x |
| resize_224_u8/256 | 0.252 | 0.248 | 1.02x |
| resize_224_u8/512 | 0.390 | 0.389 | 1.00x |
| resize_224_u8/1024 | 0.896 | 0.883 | 1.01x |
| crop_then_resize_224_u8/256 | 0.432 | 0.229 | 1.89x |
| crop_then_resize_224_u8/512 | 1.254 | 0.389 | 3.22x |
| crop_then_resize_224_u8/1024 | 4.388 | 0.760 | 5.77x |

Nothing outside the strided cases moves beyond noise.

## Transpose tiling: tried, rejected

The plan gated a tiled transpose on `materialize_transpose_u8` staying above 3×
`materialize_flip_v_u8` at 1024². It did (2.5 vs 0.3 ms, ~8×), so `80d3b52`
added one: tiles of up to 64 output rows × 64 units whenever the innermost
outer axis is adjacent units and the row step is ≥ 256 B. An interleaved
three-round A/B on x86-64 (`tiling-ab/`, `bf64725` vs `80d3b52`):

| case | walk | tiled | |
|---|---:|---:|---:|
| u8 transpose 256² | 0.094 | 0.085 | 1.11x |
| u8 transpose 512² | 0.307 | 0.322 | 0.95x |
| u8 transpose 1024² | 2.346 | 3.971 | **0.59x** |
| f32 scale, transposed 256² | 0.194 | 0.186 | 1.05x |
| f32 scale, transposed 512² | 2.599 | 2.977 | 0.87x |
| f32 scale, transposed 1024² | 20.895 | 16.442 | 1.27x |

With 3-byte units a tile writes to 64 output rows at once, costing more than
the cache misses it saves; only 12-byte units at 1024² gained. Reverted in
`c4475b6`. Transpose remains ~8× a vertical flip; a transpose kernel for small
units is left as a follow-up (CR-55).

## Files

- `full/`: every kernel, base vs head, three rounds per target.
- `transpose/`: the † rows, base vs `bf64725`.
- `tiling-ab/`: `bf64725` (`pretile`) vs `80d3b52` (`head`), x86-64.
