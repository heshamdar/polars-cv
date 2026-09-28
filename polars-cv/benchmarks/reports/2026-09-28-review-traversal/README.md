# Review: one traversal for per-value and per-pixel kernels (CR-64, CR-65)

Base: `f223584` (PR 105's head with `main` merged). Candidate: `6229fbe`.
`view-buffer/benches/kernels.rs`, `--profile benchmark`, built for the
**wheels' target** (`RUSTFLAGS="-C target-cpu=x86-64"`, where the runtime
AVX2 dispatch is what users get), one thread, criterion median of 20 samples,
nothing else running. New cases (in both binaries): `fused_chain_f32`,
`normalize_zscore_f32`, `grayscale_u8_transpose`, `grayscale_u16_flip_v`,
`invert_u8_flip_h`, `adjust_gamma_u8_flip_v`, `threshold_u8_flip_h`; the
`f16_*` cases exist only in the candidate (below).

```bash
CARGO_TARGET_DIR=$SCRATCH/target-bench \
RUSTFLAGS="-C target-cpu=x86-64 -C link-arg=-fuse-ld=lld" \
scripts/with-pyo3-env.sh cargo bench -p view-buffer --all-features \
    --profile benchmark --bench kernels --no-run
# then each binary in turn, base / head / base / head ...:
<binary> --bench '^(grayscale|threshold|invert|adjust|normalize|scale|fused|cast|f16)' --noplot
python compare.py raw/   # median over rounds, per side
```

`raw/final-*`: three interleaved rounds per side of every case the change
reaches. `raw/recheck-*`: five more rounds per side of the eight cases the
three-round run put below 1.0 (that run was noisy: the unchanged base's f32
z-score at 1024² measured 13.5 ms in an earlier run and 18.9 ms in this one).

## Results (wheel target, median over rounds, base / head)

| kernel | 256² | 512² | 1024² |
|---|---:|---:|---:|
| **views of per-value ops** | | | |
| `invert` of a horizontal flip (u8) | 1.05x | 1.10x | 1.12x |
| `adjust_gamma` of a vertical flip (u8, table) | 1.00x | 1.12x | 1.02x |
| `scale` of a transpose (f32) | 1.03x | 1.09x | 1.12x |
| `cast` u8 → f32 of a horizontal flip | 1.11x | 1.20x | 1.03x |
| `threshold` of a horizontal flip (u8) | 0.99x | 1.13x | 1.22x |
| **grayscale** | | | |
| u16, vertical flip | 2.04x | 2.04x | 2.11x |
| f32 | 1.96x | 1.77x | 1.52x |
| u8, horizontal flip | 1.06x | 1.24x | 1.17x |
| u8 (contiguous, vertical flip, transpose) | 0.98–1.04x | 0.98–1.08x | 0.94–1.08x |
| **statistics and tables** | | | |
| u8 preset `normalize` → f32 (per-channel table) | 1.78x | 1.86x | 1.53x |
| u8 `adjust_contrast` | 1.08x | 1.51x | 1.51x |
| f32 z-score `normalize` | 1.20x | 1.23x | 3.00x (2.1x in an earlier run) |
| f32 preset `normalize`, in place (recheck) | 1.02x | 0.98x | 1.15x |
| **in-place and streaming chains** | | | |
| f32 scalar chain, in place | 1.13x | 1.74x | 1.97x |
| u8 `scale` → f32 (recheck) | 0.97x | 1.11x | 1.14x |
| **unchanged paths (recheck, 5 rounds)** | | | |
| `cast` f32 → u8 | 1.02x | 1.02x | 1.08x |
| u8 `invert` in place | 0.95x | 0.98x | 1.01x |
| u8 `adjust_gamma` (table, in place) | 1.03x | 1.05x | 1.03x |
| u8 z-score `normalize` | 0.94x | 1.03x | 0.98x |

Every other case is within noise (0.94–1.12x; `raw/final-compare.txt`).

## The half-precision sink's conversion (CR-65)

The old row conversion is reproduced verbatim in the bench
(`f16_per_element_*`: cast to f32, pack, then `f16::from_f32` and a 2-byte
`extend_from_slice` per element) against `ViewBuffer::to_f16_bits`, in the
same binary, three rounds:

| input | size | per element | `to_f16_bits` | speedup |
|---|---:|---:|---:|---:|
| f32 | 256² ×3 | 452 µs | 86 µs | 5.2x |
| f32 | 512² ×3 | 1.71 ms | 399 µs | 4.3x |
| f32 | 1024² ×3 | 8.63 ms | 1.93 ms | 4.5x |
| u8 | 256² ×3 | 464 µs | 73 µs | 6.4x |
| u8 | 512² ×3 | 2.05 ms | 333 µs | 6.2x |
| u8 | 1024² ×3 | 8.61 ms | 1.40 ms | 6.1x |
| f32, transposed | 256² ×3 | 566 µs | 178 µs | 3.2x |
| f32, transposed | 512² ×3 | 2.36 ms | 823 µs | 2.9x |
| f32, transposed | 1024² ×3 | 23.63 ms | 6.93 ms | 3.4x |

In the plugin it also moved from the serial column build onto the row pool,
so on `n` threads the conversion's wall time divides by up to `n` on top of
this. That part was not measured at the plugin level: it would take two
`--profile benchmark` builds of the polars stack, and the A/B above isolates
the conversion. `benchmarks/plugin_overhead.py` now has `static->numpy` and
`static->f16` cases for the next plugin A/B.

## What the first candidates got wrong

The change was measured three times before this run, and each time a
regression showed a code-generation trap (all now in the handover's
"Code-generation traps"):

1. The traversal was wrapped in closures (a spare-capacity helper, the walk's
   consumer). A closure is a function of its own, compiled without the AVX2
   build unless inlined: f32 → u8 `cast` ran 4.5x slower, u8 grayscale 3x.
   Fixed by writing the loops in the dispatched body, with the walk's
   consumer an `#[inline(always)]` `RunSink` method.
2. u8 grayscale of a vertical flip stayed 5x slower. Callgrind at instruction
   level showed the scalar remainder loop running: LLVM had hoisted the
   vector loop's overlap check out of the per-row loop, and the flip's
   negative row step failed the hoisted check for every row. Fixed with an
   opaque (`black_box`) unit address.
3. u8 preset normalize (0.8x): the table map's inner loop over a run-time
   channel count compiled worse in the AVX2 build than in the baseline, and
   `&self.table` (a `&Vec`) reloaded its data pointer after every store.
   Fixed with a slice binding and const-generic 3/4-channel loops (1.5–1.9x).
4. u8 `adjust_contrast` (0.6–0.8x): its mean was switched to the z-score
   helper, which also accumulates Σx² in `u128`. Fixed by summing Σx only.
5. f32 in place (0.84–0.96x): the blocked map copied each f32 block into its
   scratch and back. Now the passes run on the buffer's own blocks.
