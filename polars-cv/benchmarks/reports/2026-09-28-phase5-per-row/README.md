# Phase 5: per-row executor overhead

Base: `7134c75` (Phase 4 + CR-57). Candidate: the Phase 5 commit. Both built
with `maturin develop --profile benchmark` in the main tree (x86-64-v3, the
local dev flags), and the two `_lib.abi3.so` files swapped in turn: base,
head, base, head, … three rounds, median of each run's five repeats. 4-core
Intel Xeon @ 2.80 GHz container. `benchmarks/plugin_overhead.py`, blob
source; the cases are in its docstring. Microseconds per row, 4 threads
(polars' default) unless stated.

## Where a cheap row's time went

No sampling profiler works here (no `perf`; `py-spy` sees only threads with
Python state, and the plugin runs on polars' pool). Callgrind on an
unstripped `--profile benchmark` build, one thread, `invert` on 20k 8×8 u8
rows × 4 calls, counted only inside `execute_rows`:

| | base | head |
|---|---:|---:|
| instructions per row | 10,400 | 5,750 |
| `malloc` calls per row | 27 | 12 |
| allocator share of instructions (self) | 58% | 25% |
| the element-wise kernel itself | ~2% | ~3% |

Twenty of the 27 allocations were layout bookkeeping: `Layout` held its
shape and strides in two heap `Vec`s, so every `ViewBuffer::clone`, every
`to_contiguous()` of an already contiguous buffer (a clone), and every
`is_contiguous()` (which rebuilt a `LayoutFacts` from copies, plus a scratch
vector) allocated. A row runs each of those several times. The plan's own
candidates were a small part of it: the cache key (2 allocations), the step
list clone (1), and the shared `RwLock` (not visible to callgrind; see below).

## What changed

- `core::layout::{Dims, Strides}` = `SmallVec<[_; 4]>`: a layout of rank ≤ 4
  is stored inline, so copying one or asking it a question allocates nothing.
  `is_contiguous` is one allocation-free rule (`is_c_contiguous`) for
  `Layout` and `LayoutFacts`. The blob parser and the buffer constructors
  take the inline types too.
- Plan-cache hits compare against the source's own shape/strides slices and
  hand out a shared `Arc<[PlanStep]>`, replayed by
  `ExecutionPlan::execute_steps`.
- `PendingSegment::ops` is inline.
- Row results are no longer concatenated into one call-sized vector before
  the column is built: each output keeps its ranges' parts (`RowParts`) and
  the builder converts them in order. With `RowResult` now 168 bytes (a
  `ViewBuffer` holds its layout inline), that serial copy made the eager
  no-op at 4 threads ~25% slower in the first head build; removing it made it
  equal again, and eager `invert` 25% faster than base rather than 6%.

## Results

### 8×8 u8 (200,000 rows) — the gate

| case | base µs/row | head µs/row | speedup |
|---|---:|---:|---:|
| dynamic eager | 1.60 | 1.43 | 1.12x |
| dynamic streaming | 1.08 | 1.12 | 0.96x |
| invert eager | 1.10 | 0.86 | 1.29x |
| invert streaming | 0.61 | 0.44 | 1.38x |
| noop eager | 0.56 | 0.59 | 0.95x |
| noop streaming | 0.40 | 0.36 | 1.10x |
| static eager | 0.95 | 0.88 | 1.08x |
| static streaming | 0.66 | 0.58 | 1.12x |
| static->array eager | 0.85 | 0.72 | 1.18x |
| static->array streaming | 0.40 | 0.35 | 1.13x |

### 64×64 u8 (50,000 rows)

| case | base µs/row | head µs/row | speedup |
|---|---:|---:|---:|
| dynamic eager | 5.68 | 4.74 | 1.20x |
| dynamic streaming | 6.22 | 5.70 | 1.09x |
| invert eager | 2.26 | 2.20 | 1.03x |
| invert streaming | 1.96 | 1.52 | 1.29x |
| noop eager | 1.60 | 1.58 | 1.01x |
| noop streaming | 2.06 | 1.40 | 1.47x |
| static eager | 5.06 | 5.36 | 0.94x |
| static streaming | 5.58 | 5.12 | 1.09x |
| static->array eager | 6.62 | 6.64 | 1.00x |
| static->array streaming | 5.94 | 7.38 | 0.80x |

The 64×64 rows are dominated by the kernels and by first-touch page faults,
so most cells sit in this container's ±10% noise band.

### A regression that is the allocator, not the code

`static->array` at 64×64, **one thread, eager**, is ~40% slower on head
(1.09–1.18 s vs 0.78–0.83 s) from the second call in a process onwards. The
first call is the same (1.12 vs 1.17 s). Head takes twice the page faults
per call (405k vs 200k, i.e. an extra ~800 MB touched): the call holds 50k
f32 rows (800 MB) until the column is built, and with the stray small layout
allocations gone nothing pins the top of glibc's heap, so glibc hands those
pages back to the OS after the call and the next call faults them in again.
With `MALLOC_TRIM_THRESHOLD_=-1` head is faster than base (0.72–0.74 s vs
0.78–0.80 s). Base kept the memory by accident, and neither side chooses its
allocator: the plugin uses the system `malloc`, while polars uses its own.
CR-60 files adopting pyo3-polars' `PolarsAllocator`.

### Not done, and why

- **A per-range plan cache in front of the shared `RwLock`**: after the
  changes above, streaming `invert` at 8×8 costs ~0.08 µs per row more than a
  no-op, kernel included. There is nothing left there for a lock to be.
- **A chunk-offset table for `decode.rs`'s per-row chunk walk**: a column of
  1,000 chunks runs faster than a single-chunk one (polars evaluates per
  chunk), base and head alike, so the walk is never long.
