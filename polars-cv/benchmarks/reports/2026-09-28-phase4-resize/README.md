# Phase 4: resize reads crops and vertical flips where they lie

Base: `0d6997e` (Phases 0–3). Candidate: the Phase 4 commit (`FirViewAdapter`).
Both built from the same `benches/kernels.rs`, `--profile benchmark`, for the
wheels' target (**x86-64**). 4-core Intel Xeon @ 2.80 GHz container, one
thread. Times in **microseconds**, median of three interleaved rounds (base,
head, base, head, …). Each build ran after `cargo clean -p view-buffer
--profile benchmark`, and the two binaries were compared with `cmp` (they
differ, and only head carries `interop/fir.rs`).

## Was it worth doing?

The handover asked for the pack's cost to be measured before building the
adapter. On the base, packing the view was a quarter of the resize at 1024²
(`raw/payoff-head-before.txt`; `packed_crop_resize_224_u8` resizes the same
crop already packed, `resize_224_u8` is the unflipped image):

| 1024² u8 RGB → 224² | view | packed | pack share |
|---|---:|---:|---:|
| crop (768²) | 613 | 464 | 24% |
| flip_v | 843 | 648 | 23% |

(6–17% at 256² and 512².)

## What changed

- `interop/fir.rs`: `FirViewAdapter<P>` (`ExternalView`, `LAYOUT =
  FastImageResize`) gives fast_image_resize a `FirView` whose rows come from
  `ViewBuffer::dense_rows`, so any row stride works, negative included.
- `LayoutFacts::compatible_with(FastImageResize)` is now `is_dense_rows()`
  (was `is_contiguous()`).
- `resize_typed_u8/u16/f32` and `pixel_type_for` are one generic
  `resize_pixels::<P>`; `resize_strided` maps dtype × channels to the pixel
  type once. Other layouts (a transpose, a horizontal flip) pack inside it.
- The resize family's `memory_effect` (`Resize`, `ResizeScale`,
  `ResizeToHeight`/`Width`, `ResizeMax`/`Min`, `Letterbox`) is
  `StridePreserving`: `build_plan` no longer inserts a `MaterializeContiguous`
  in front of them.

The plan's "destination from fir's own `Image::new`" is not done: in 5.6.0
`Image::new` zero-fills too, and returns a `Vec<u8>` that a u16/f32 output
cannot reuse. The output stays one typed `vec![0; n]`.

## Results (x86-64)

| case | base | head | head/base |
|---|---:|---:|---:|
| `crop_then_resize_224_u8/256` | 182 | 185 | 1.01 |
| `crop_then_resize_224_u8/512` | 300 | 281 | 0.94 |
| `crop_then_resize_224_u8/1024` | 658 | 492 | **0.75** |
| `flip_v_then_resize_224_u8/256` | 219 | 201 | 0.92 |
| `flip_v_then_resize_224_u8/512` | 372 | 329 | 0.89 |
| `flip_v_then_resize_224_u8/1024` | 948 | 623 | **0.66** |
| `packed_crop_resize_224_u8/256` | 177 | 179 | 1.01 |
| `packed_crop_resize_224_u8/512` | 271 | 282 | 1.04 |
| `packed_crop_resize_224_u8/1024` | 471 | 485 | 1.03 |
| `resize_224_u8/256` | 201 | 194 | 0.96 |
| `resize_224_u8/512` | 334 | 330 | 0.99 |
| `resize_224_u8/1024` | 650 | 680 | 1.05 |

Contiguous inputs move within ±5% with mixed signs, the noise band of this
container across rounds.

### The row iterator is on fir's hot path

fir's vertical pass asks the view for its rows once per 32-byte column chunk
of every output row, so `iter_rows` is in its innermost loop. The first
version returned `rows.iter().skip(start)`, and `Skip` checks on every
`next()`. That made a contiguous resize ~9% slower over five rounds
(`raw/contiguous-skip-iter.txt`: 516 vs 474 µs median). A subslice
(`rows.get(start..)`) matches fir's own `TypedImageRef` (`expB`, contiguous
input handed to fir's view directly) within noise
(`raw/contiguous-subslice-iter.txt`: 470 vs 459 base, 464 `expB`).

## Files

- `raw/payoff-head-before.txt`: the payoff measurement on the base, three rounds.
- `raw/ab-x86-64.txt`: the results table, base vs head.
- `raw/contiguous-*.txt`: the iterator experiments above.
