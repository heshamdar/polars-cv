# CR-57: grayscale reads a vertical flip where it lies

Base: `e341ff2` (Phase 4). Candidate: the CR-57 commit (`Grayscale` declares
`StridePreserving`; its 1-channel case packs). Same `benches/kernels.rs` on
both sides (`grayscale_u8_flip_v` added), `--profile benchmark`, wheels'
target (**x86-64**), clean builds compared with `cmp`. 4-core Intel Xeon @
2.80 GHz container, one thread. Median of three interleaved rounds, µs.

| case | base µs | head µs | head/base |
|---|---:|---:|---:|
| `grayscale_u8_flip_h/256` | 52.3 | 50.5 | 0.97 |
| `grayscale_u8_flip_h/512` | 242.2 | 228.9 | 0.95 |
| `grayscale_u8_flip_h/1024` | 1154.7 | 1125.6 | 0.97 |
| `grayscale_u8_flip_v/256` | 23.8 | 18.8 | 0.79 |
| `grayscale_u8_flip_v/512` | 109.0 | 72.1 | 0.66 |
| `grayscale_u8_flip_v/1024` | 436.2 | 263.9 | 0.60 |
| `grayscale_u8_rgb/256` | 18.5 | 20.2 | 1.09 |
| `grayscale_u8_rgb/512` | 56.6 | 57.5 | 1.02 |
| `grayscale_u8_rgb/1024` | 233.8 | 223.9 | 0.96 |

`grayscale_u8_flip_h` still packs (a horizontal flip has no packed rows), and
the contiguous kernel is unchanged; both move within this container's noise.

`ab-x86-64.txt` is the raw criterion output.
