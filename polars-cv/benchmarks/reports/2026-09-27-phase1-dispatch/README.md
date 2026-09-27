# Phase 1: CPU dispatch, one conversion rule, grayscale and threshold kernels

Base: `0aab4db` (Phase 0). Candidate: this commit. `view-buffer/benches/kernels.rs`,
`--profile benchmark`, on each target:

- **x86-64**: `RUSTFLAGS="-C target-cpu=x86-64"`, what the published wheels run;
- **v3**: `x86-64-v3`, the local dev config.

Machine: 4-core Intel Xeon @ 2.80 GHz cloud container, one thread. Base and
candidate binaries were built with identical flags and run back to back.

## What changed

- `core::dispatch` (`SimdKernel` + `dispatch`) compiles a kernel's whole body
  a second time with AVX2 and picks it at runtime. Blur moved onto it; it was
  the one hand-rolled instance (CR-35).
- `core::convert` (`CastFrom` + `convert_slice`) is the one element-conversion
  rule, dispatched, used by `cast` and by the fused kernel's output.
- u8 grayscale and u8 threshold are dispatched kernels over packed rows,
  written without a zero-fill. Contiguous, cropped and vertically flipped inputs
  are read where they lie.
- Non-u8 grayscale fixes its channel count at compile time.

## Results

Kernels this phase touched, median of three interleaved rounds (µs), after the
zero-fill fix below:

| kernel | base x86-64 | head x86-64 | | base v3 | head v3 | |
|---|---:|---:|---:|---:|---:|---:|
| grayscale_u8_rgb/256 | 158.9 | 23.8 | 6.69x | 121.9 | 24.6 | 4.95x |
| grayscale_u8_rgb/512 | 653.0 | 65.3 | 10.00x | 500.4 | 79.1 | 6.32x |
| grayscale_u8_rgb/1024 | 2497.2 | 302.2 | 8.26x | 2102.1 | 348.4 | 6.03x |
| threshold_u8/256 | 13.4 | 4.2 | 3.16x | 11.3 | 4.8 | 2.34x |
| threshold_u8/512 | 44.4 | 15.1 | 2.94x | 33.5 | 16.6 | 2.01x |
| threshold_u8/1024 | 167.8 | 91.7 | 1.83x | 132.3 | 97.8 | 1.35x |

Full suite, one round each (ms; `full-*.txt`), rows this phase moves:

| kernel | base x86-64 | head x86-64 | | base v3 | head v3 | |
|---|---:|---:|---:|---:|---:|---:|
| grayscale_f32_rgb/256 | 0.184 | 0.099 | 1.86x | 0.209 | 0.071 | 2.94x |
| grayscale_f32_rgb/512 | 0.759 | 0.421 | 1.80x | 0.860 | 0.366 | 2.35x |
| grayscale_f32_rgb/1024 | 3.581 | 2.584 | 1.39x | 4.160 | 2.467 | 1.69x |
| threshold_u8_cropped/256 | 0.023 | 0.008 | 2.88x | 0.028 | 0.011 | 2.55x |
| threshold_u8_cropped/512 | 0.056 | 0.023 | 2.43x | 0.062 | 0.029 | 2.14x |
| threshold_u8_cropped/1024 | 0.187 | 0.136 | 1.38x | 0.184 | 0.145 | 1.27x |
| cast_f32_to_u8/256 | 0.577 | 0.317 | 1.82x | 0.248 | 0.257 | 0.96x |
| cast_f32_to_u8/512 | 2.414 | 1.161 | 2.08x | 0.981 | 1.036 | 0.95x |
| cast_f32_to_u8/1024 | 9.054 | 5.084 | 1.78x | 4.341 | 4.289 | 1.01x |
| fused_chain_u8/256 | 0.669 | 0.399 | 1.68x | 0.349 | 0.338 | 1.03x |
| fused_chain_u8/512 | 3.145 | 1.786 | 1.76x | 1.719 | 1.630 | 1.05x |
| fused_chain_u8/1024 | 14.553 | 10.668 | 1.36x | 8.730 | 8.436 | 1.03x |

The wheels' float → int casts now run at the speed of a v3 build: on SSE2
`f32::round` was a `roundf` call per element. Kernels this phase does not
touch (resize, blur, flips, codecs, gamma, normalize) are within the run's
noise in `full-*.txt`.

## Checked and not a regression

- `cast_u8_to_f32/1024` read 0.65x in the single full run. Three interleaved
  rounds put base at 1.43–1.66 ms and head at 1.48–1.82 ms on x86-64, and the
  two overlapping on v3. The case writes a fresh 12 MB buffer, so page faults
  dominate it; `invert_u8/1024`, whose code is unchanged, moved 17% in the same
  run.
- `threshold_u8/1024` **was** a real regression in the first candidate (v3:
  115–128 µs → 141–147 µs, all three rounds). The kernels zero-filled their
  output (`vec![0u8; n]`), which costs a full extra pass once the allocation
  comes from the heap. They now write into spare capacity
  (`runner.rs::map_pixel_rows`); the tables above are after that fix.

## Correctness

- The view-buffer suite passes under `RUSTFLAGS="-C target-cpu=x86-64"`. There,
  every dispatched call in a debug build runs the portable (SSE2) build as well
  and asserts byte-identical output.
- `grayscale_threshold_parity_tests` compare against independent references
  over contiguous, cropped, flipped and transposed inputs, and `luma_u8` is
  checked on all 2^24 RGB triples.
