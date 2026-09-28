# Phase 8: morphology iterations, blur input conversion

Base: `268fb1f` (Phase 7). Candidate: the Phase 8 commit.
`view-buffer/benches/kernels.rs`, `--profile benchmark`, local `x86-64-v3`
config, one thread, criterion median of 20 samples, nothing else running.
New cases: `blur_sigma2_u8_gray`, `blur_sigma2_f32_rgb`, `erode_k3_x1_u8`,
`dilate_k5_x4_u8`, `erode_k3_x3_f32`.

```bash
scripts/with-pyo3-env.sh cargo bench -p view-buffer --all-features \
    --profile benchmark --bench kernels -- "blur|erode|dilate"
```

## Where the time went (measured before changing anything)

Morphology was already vectorised and cheap per pass (0.35 ms for one 3×3
erode at 1024² u8); `iterations=n` cost n full passes (1.05 ms for 3). Blur
(18.5 ms at 1024² RGB u8, 26 ms f32) converted the whole input to f32 into a
fresh image-sized buffer before its first pass.

## What changed

- **Integer erode/dilate run one pass of radius `n * (k / 2)`** instead of
  `n` passes (`morph_iterated`, with the proof in its doc comment). It is
  exact for integers. **Floats keep the passes**: the fold keeps a NaN
  incumbent, ignores a NaN candidate and keeps the incumbent of two equal
  signed zeros, so the result depends on the passes' structure — the parity
  tests showed one wide pass differs (41.0 against 2.0 on a NaN image).
- **Blur converts each input row to f32 as the horizontal pass reaches it**,
  into a row scratch, instead of a whole-image `input_f32`; loads and stores
  go through M5 (`CastFrom`). For the three dtypes the typed blur runs (u8,
  u16, f32; the rest are cast around it) that is the old result.

## Results (ms)

| kernel | base ms | head ms | speedup |
|---|---:|---:|---:|
| blur_sigma2_u8_rgb/256 | 1.012 | 0.856 | 1.18x |
| blur_sigma2_u8_rgb/512 | 3.741 | 3.229 | 1.16x |
| blur_sigma2_u8_rgb/1024 | 18.451 | 15.533 | 1.19x |
| blur_sigma2_u8_gray/256 | 0.301 | 0.260 | 1.16x |
| blur_sigma2_u8_gray/512 | 1.144 | 0.860 | 1.33x |
| blur_sigma2_u8_gray/1024 | 5.358 | 4.240 | 1.26x |
| blur_sigma2_f32_rgb/256 | 0.849 | 0.767 | 1.11x |
| blur_sigma2_f32_rgb/512 | 3.748 | 2.897 | 1.29x |
| blur_sigma2_f32_rgb/1024 | 26.168 | 22.405 | 1.17x |
| erode_k3_x1_u8/256 | 0.027 | 0.028 | 0.98x |
| erode_k3_x1_u8/512 | 0.074 | 0.089 | 0.84x |
| erode_k3_x1_u8/1024 | 0.346 | 0.330 | 1.05x |
| erode_k3_x3_u8/256 | 0.076 | 0.061 | 1.24x |
| erode_k3_x3_u8/512 | 0.212 | 0.155 | 1.37x |
| erode_k3_x3_u8/1024 | 1.045 | 0.612 | 1.71x |
| dilate_k5_x4_u8/256 | 0.170 | 0.151 | 1.13x |
| dilate_k5_x4_u8/512 | 0.473 | 0.384 | 1.23x |
| dilate_k5_x4_u8/1024 | 1.827 | 1.234 | 1.48x |
| erode_k3_x3_f32/256 | 0.181 | 0.180 | 1.01x |
| erode_k3_x3_f32/512 | 1.137 | 1.004 | 1.13x |
| erode_k3_x3_f32/1024 | 5.091 | 4.986 | 1.02x |

Single-iteration and f32 morphology run the same code as before; their
cells are noise (`erode_k3_x1_u8/512` is 74 → 89 µs).

## Bit-identity

- Morphology: `tests/morph_ref.rs` compares with a naive per-pixel reference
  iterated `n` times (it shares no code with the kernel), now over
  iterations 0–4, ksizes 1–5 and 9 (even included), images narrower than the
  collapsed window, and float NaN and signed-zero images iterated. Watched
  failing: using the plan's window size `n(k−1)+1` as the radius.
- Blur: the pre-Phase-8 body is kept verbatim as the oracle in
  `runner.rs`'s blur tests (`blur_matches_the_reference_kernel_bit_for_bit`),
  over u8/u16/f32 with NaN, infinity, −0 and the integer maxima, 1–5
  channels, images narrower than the kernel and four sigmas. Watched failing:
  a truncating store. The wheels' `x86-64` target passes the whole suite with
  M1's portable-vs-AVX2 check.
