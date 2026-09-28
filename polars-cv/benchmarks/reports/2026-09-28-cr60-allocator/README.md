# CR-60: the plugin allocates through polars' allocator

Base (`sys`): the candidate with its `#[global_allocator]` line removed, so the
plugin allocates with the system `malloc` (glibc) as it did through Phase 5.
Candidate (`alloc`): pyo3-polars' `PolarsAllocator`, which relays to the
allocator polars exports (jemalloc in the Linux wheels). Only that line
differs. Both `maturin develop --profile benchmark`, swapped between rounds:
sys, alloc, … three rounds, median of each run's five repeats. 4-core Intel
Xeon @ 2.80 GHz container, 4 threads, `benchmarks/plugin_overhead.py`.

## The regression it was filed for

64×64 u8 → f32 rows, 50,000 per call, one thread (`POLARS_MAX_THREADS=1`),
four calls in one process. Page faults from `getrusage`:

| sink | allocator | call 1 | calls 2–4 | faults, calls 2–4 |
|---|---|---:|---:|---:|
| `array` | system | 1.07–1.14 s | 0.76–1.19 s | 204k–405k |
| `array` | polars | 1.09 s | 0.68–0.89 s | 200k–283k |
| `blob` | system | 0.68–0.72 s | 0.61–0.66 s | ~202k |
| `blob` | polars | 0.71 s | 0.26–0.39 s | 1k–142k |

glibc hands a call's freed rows back to the OS and the next call faults them
in again; jemalloc keeps them for reuse (its decay is time-based). Peak RSS is
~10% higher for it (2.09 vs 1.88 GB for the `array` case).

## The plugin benchmark

### 8×8 u8 (200,000 rows)

| case | sys µs/row | alloc µs/row | speedup |
|---|---:|---:|---:|
| dynamic eager | 1.49 | 1.36 | 1.09x |
| dynamic streaming | 1.12 | 1.14 | 0.98x |
| invert eager | 0.81 | 0.72 | 1.12x |
| invert streaming | 0.46 | 0.47 | 0.98x |
| noop eager | 0.65 | 0.50 | 1.31x |
| noop streaming | 0.38 | 0.37 | 1.03x |
| static eager | 0.93 | 0.72 | 1.28x |
| static streaming | 0.65 | 0.43 | 1.51x |
| static->array eager | 0.74 | 0.67 | 1.10x |
| static->array streaming | 0.36 | 0.37 | 0.99x |

### 64×64 u8 (50,000 rows)

| case | sys µs/row | alloc µs/row | speedup |
|---|---:|---:|---:|
| dynamic eager | 4.40 | 3.84 | 1.15x |
| dynamic streaming | 5.54 | 3.84 | 1.44x |
| invert eager | 2.22 | 1.34 | 1.66x |
| invert streaming | 1.48 | 1.10 | 1.35x |
| noop eager | 1.70 | 1.16 | 1.47x |
| noop streaming | 1.36 | 0.90 | 1.51x |
| static eager | 4.48 | 2.96 | 1.51x |
| static streaming | 5.20 | 3.60 | 1.44x |
| static->array eager | 6.36 | 6.44 | 0.99x |
| static->array streaming | 7.44 | 6.42 | 1.16x |

Nothing slower outside this container's ±5–10% noise.

## Guard

`polars_cv._lib.__allocator__` reports the allocator the plugin's allocations
go to, and `tests/test_allocator.py` requires `"polars"`. It is `"system"`
when the allocator is not the global one (watched: the `#[global_allocator]`
line removed) or polars' capsule cannot be imported, where `PolarsAllocator`
falls back silently (watched: the capsule name changed).
