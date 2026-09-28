# Phase 7: the affine warp (rotate, warp_affine), bit-identical

Base: `d61a936` (Phase 6). Candidate: the Phase 7 commit.
`view-buffer/benches/kernels.rs`, `--profile benchmark`, local `x86-64-v3`
config, one thread, criterion median of 20 samples; base and head measured
on a quiet machine (a concurrent build had depressed an earlier head run by
~10%). The rotate cases beyond `rotate_30_bilinear_u8_rgb` are new.

```bash
scripts/with-pyo3-env.sh cargo bench -p view-buffer --all-features \
    --profile benchmark --bench kernels -- rotate
```

## Where the time went (measured before changing anything)

`rotate(30°)`, bilinear, 1024² RGB u8: 33 ms, ~36 ns per pixel. Callgrind:
~107 instructions per pixel, all ordinary scalar work — a closure per
sample with its own bounds check, saturating `floor() as i64` conversions
and overflow-checked `+ 1`s, and an unsigned `usize → f64` conversion per
pixel (no single x86 instruction). No libcalls: `round`/`floor` are inline
in the AVX2 build. `rotate(0°)` ran the whole warp (36 ms at 1024²).

## What changed

- **`execution/warp.rs`**: the warp, moved out of `runner.rs`, as a
  `SimdKernel` (M1) compiled per channel count (1–4, and one for any other).
- **An interior test in `f64`**: a bilinear sample is interior exactly when
  `0 <= x < w - 1` (and likewise `y`), which the old integer test was; there
  the four neighbours are two row slices and the conversions need no
  saturation. Edges, off-image pixels and NaN coordinates take the old code.
  Nearest gets the same split (`-0.5 < x < w - 0.5`).
- **Pixel coordinates count in `f64`** (exact below 2^53), so the per-pixel
  `usize → f64` is gone; `a * x + b * y + t` is still added in that order.
- **Loads and stores go through M5** (`core::convert`). That is the old rule
  for every dtype except the 64-bit top (CR-62). M5 gained an f64 form for
  8/16-bit targets (`round().clamp() as T`): `round_narrow`'s form, built for
  f32 bulk casts, kept the 4-channel blend from vectorising (1.8x slower for
  RGBA).
- **0° is the identity** (CR-63): the (packed) input, data shared.

Not done: splitting rows into [border | interior | border] (the per-pixel
test is cheaper than it looks, and a split would re-derive the interior
bounds a second way), and tiling the output (measured: within noise, so the
f32 1024² case is not cache-bound; its fresh 12 MB output is the likely
remainder).

## Results (ms)

| kernel | base ms | head ms | speedup |
|---|---:|---:|---:|
| rotate_30_bilinear_u8_rgb/256 | 2.055 | 1.419 | 1.45x |
| rotate_30_bilinear_u8_rgb/512 | 8.057 | 5.197 | 1.55x |
| rotate_30_bilinear_u8_rgb/1024 | 33.273 | 24.015 | 1.39x |
| rotate_30_bilinear_u8_gray/256 | 1.267 | 0.705 | 1.80x |
| rotate_30_bilinear_u8_gray/512 | 5.430 | 2.791 | 1.95x |
| rotate_30_bilinear_u8_gray/1024 | 20.437 | 12.408 | 1.65x |
| rotate_30_bilinear_u8_rgba/256 | 2.221 | 1.100 | 2.02x |
| rotate_30_bilinear_u8_rgba/512 | 9.875 | 4.584 | 2.15x |
| rotate_30_bilinear_u8_rgba/1024 | 38.405 | 23.678 | 1.62x |
| rotate_30_bilinear_f32_rgb/256 | 1.601 | 0.997 | 1.61x |
| rotate_30_bilinear_f32_rgb/512 | 6.760 | 4.213 | 1.60x |
| rotate_30_bilinear_f32_rgb/1024 | 32.165 | 27.435 | 1.17x |
| rotate_30_nearest_u8_rgb/256 | 0.558 | 0.322 | 1.73x |
| rotate_30_nearest_u8_rgb/512 | 2.295 | 1.459 | 1.57x |
| rotate_30_nearest_u8_rgb/1024 | 11.230 | 5.654 | 1.99x |
| rotate_0_u8_rgb/256 | 2.188 | 0.001 | 2737.51x |
| rotate_0_u8_rgb/512 | 9.019 | 0.001 | 7329.87x |
| rotate_0_u8_rgb/1024 | 35.911 | 0.003 | 11376.48x |

## Bit-identity

`execution/warp.rs`'s tests keep the old kernel verbatim as their oracle and
compare bytes over every dtype, ranks 2 and 3, 1–5 channels, sizes down to
1×1, six angles with and without `expand`, both interpolations, three border
values, and six `warp_affine` matrices (scale, shear, off-image, mirror,
saturating and NaN source coordinates), with NaN and infinities in the float
images. Watched failing: re-associating `a * x + (b * y + t)`, swapping two
neighbours in the interior path, and an off-by-one in the interior bound.
M1's debug check held the portable and AVX2 builds byte-identical on the
wheels' `x86-64` target (the whole view-buffer suite).

The oracle differs from the release kernel it stands for in one place: it
computes `x0 + 1` wrapping, as release builds did; in debug builds the old
code panicked on a saturated coordinate.
