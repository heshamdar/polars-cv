# Kernel baseline (Phase 0 of PERFORMANCE_PLAN.md)

Base: `00a8d13` (0.29.0 + the plan). `view-buffer/benches/kernels.rs`, built with
`--profile benchmark` (release, thin LTO) twice:

- **x86-64**: `RUSTFLAGS="-C target-cpu=x86-64"`, the published wheels' target
  (SSE2 baseline; runtime-dispatched kernels still take AVX2).
- **x86-64-v3**: the local dev config (`.cargo/config.toml`), AVX2 + FMA at
  compile time.

Machine: 4-core Intel Xeon @ 2.80 GHz cloud container; criterion median of 20
samples, one thread. Times in **milliseconds**, `first/last` = x86-64 ÷ v3 (above
1 means the wheels lose that much to the missing AVX2). Raw criterion output:
`base-x86-64.txt`, `base-v3.txt`.

Reproduce:

```bash
scripts/with-pyo3-env.sh cargo bench -p view-buffer --all-features \
    --profile benchmark --bench kernels
```

| kernel | x86-64 | x86-64-v3 | first/last |
|---|---:|---:|---:|
| grayscale_u8_rgb/256 | 0.154 | 0.127 | 1.21x |
| grayscale_u8_rgb/512 | 0.660 | 0.518 | 1.28x |
| grayscale_u8_rgb/1024 | 2.629 | 2.052 | 1.28x |
| grayscale_f32_rgb/256 | 0.203 | 0.221 | 0.92x |
| grayscale_f32_rgb/512 | 0.827 | 0.882 | 0.94x |
| grayscale_f32_rgb/1024 | 4.029 | 4.066 | 0.99x |
| threshold_u8/256 | 0.013 | 0.011 | 1.17x |
| threshold_u8/512 | 0.042 | 0.035 | 1.18x |
| threshold_u8/1024 | 0.140 | 0.133 | 1.05x |
| threshold_u8_cropped/256 | 0.024 | 0.029 | 0.83x |
| threshold_u8_cropped/512 | 0.058 | 0.070 | 0.83x |
| threshold_u8_cropped/1024 | 0.180 | 0.174 | 1.03x |
| cast_f32_to_u8/256 | 0.567 | 0.258 | 2.20x |
| cast_f32_to_u8/512 | 2.314 | 1.042 | 2.22x |
| cast_f32_to_u8/1024 | 9.517 | 4.507 | 2.11x |
| cast_u8_to_f32/256 | 0.044 | 0.049 | 0.90x |
| cast_u8_to_f32/512 | 0.176 | 0.198 | 0.89x |
| cast_u8_to_f32/1024 | 1.493 | 1.917 | 0.78x |
| invert_u8/256 | 0.011 | 0.012 | 0.97x |
| invert_u8/512 | 0.067 | 0.068 | 0.98x |
| invert_u8/1024 | 0.329 | 0.348 | 0.94x |
| adjust_gamma_u8/256 | 2.361 | 2.095 | 1.13x |
| adjust_gamma_u8/512 | 9.125 | 9.356 | 0.98x |
| adjust_gamma_u8/1024 | 54.589 | 54.191 | 1.01x |
| normalize_preset_u8_to_f32/256 | 1.660 | 2.219 | 0.75x |
| normalize_preset_u8_to_f32/512 | 7.141 | 10.149 | 0.70x |
| normalize_preset_u8_to_f32/1024 | 48.146 | 48.101 | 1.00x |
| fused_chain_u8/256 | 0.648 | 0.364 | 1.78x |
| fused_chain_u8/512 | 3.524 | 1.694 | 2.08x |
| fused_chain_u8/1024 | 18.046 | 10.189 | 1.77x |
| materialize_flip_h_u8/256 | 0.391 | 0.409 | 0.96x |
| materialize_flip_h_u8/512 | 1.813 | 1.665 | 1.09x |
| materialize_flip_h_u8/1024 | 6.852 | 7.009 | 0.98x |
| materialize_flip_v_u8/256 | 0.417 | 0.434 | 0.96x |
| materialize_flip_v_u8/512 | 1.558 | 1.798 | 0.87x |
| materialize_flip_v_u8/1024 | 7.172 | 7.215 | 0.99x |
| materialize_transpose_u8/256 | 0.405 | 0.438 | 0.92x |
| materialize_transpose_u8/512 | 1.803 | 1.634 | 1.10x |
| materialize_transpose_u8/1024 | 14.771 | 14.964 | 0.99x |
| resize_224_u8/256 | 0.275 | 0.267 | 1.03x |
| resize_224_u8/512 | 0.474 | 0.440 | 1.08x |
| resize_224_u8/1024 | 1.068 | 0.940 | 1.14x |
| crop_then_resize_224_u8/256 | 0.460 | 0.475 | 0.97x |
| crop_then_resize_224_u8/512 | 1.236 | 1.397 | 0.89x |
| crop_then_resize_224_u8/1024 | 4.654 | 4.804 | 0.97x |
| blur_sigma2_u8_rgb/256 | 1.278 | 1.285 | 0.99x |
| blur_sigma2_u8_rgb/512 | 4.912 | 5.489 | 0.89x |
| blur_sigma2_u8_rgb/1024 | 22.405 | 23.338 | 0.96x |
| erode_k3_x3_u8/256 | 0.082 | 0.086 | 0.94x |
| erode_k3_x3_u8/512 | 0.295 | 0.262 | 1.12x |
| erode_k3_x3_u8/1024 | 1.600 | 1.175 | 1.36x |
| rotate_30_bilinear_u8_rgb/256 | 3.067 | 2.229 | 1.38x |
| rotate_30_bilinear_u8_rgb/512 | 12.952 | 9.161 | 1.41x |
| rotate_30_bilinear_u8_rgb/1024 | 54.311 | 37.645 | 1.44x |
| encode_jpeg_q90_rgb/256 | 2.357 | 1.622 | 1.45x |
| encode_jpeg_q90_rgb/512 | 9.073 | 6.472 | 1.40x |
| encode_jpeg_q90_rgb/1024 | 37.684 | 27.326 | 1.38x |
| encode_png_rgb/256 | 0.684 | 0.563 | 1.21x |
| encode_png_rgb/512 | 2.831 | 2.306 | 1.23x |
| encode_png_rgb/1024 | 12.372 | 10.848 | 1.14x |

## What the baseline says

- **Blur** is the same on both targets: it already dispatches to AVX2 at
  runtime (CR-35). That is the pattern Phase 1 generalises.
- **Float → int casts lose 2.1–2.2x on the wheels** (`cast_f32_to_u8`): on
  SSE2 `f32::round` is a `roundf` call per element. The fused chain with a u8
  output loses 1.8–2.1x for the same reason.
- **u8 grayscale** is 2.6 ms at 1024² on the wheel target, where a
  vectorised loop over the same fixed-point formula measured 0.27–0.98 ms
  standalone (its output `Vec::push` loop does not vectorise).
- **Materialising a flip or transpose** costs ~7 ms (flip) and ~15 ms
  (transpose) for 3 MB at 1024²: the strided copy moves one 3-byte pixel per
  `memcpy` (Phase 3).
- **u8 gamma and normalize** are ~50 ms at 1024² on either target: a `powf`
  or a divide per element over data with 256 distinct values (Phase 2's
  lookup tables).
- **Crop → resize** is 4–5x the plain resize at 1024² because the crop is
  materialised first (Phase 4).
