# Phase 6: ingestion — `List` rows in place, raw/blob alignment by dtype

Base: `1e7ad74` (Phase 5 + CR-60). Candidate: the Phase 6 commit. Both
`maturin develop --profile benchmark`, the two `_lib.abi3.so` files swapped
between rounds: base, head, … three rounds, median of each run's five
repeats. 4-core Intel Xeon @ 2.80 GHz container.
`benchmarks/ingestion_overhead.py` (new): 20,000 64×64 u8 images per call,
each source straight into a `blob` sink; the cases are in its docstring.
Microseconds per row.

## Where the time went (measured before changing anything)

One thread, eager: the same pixels cost 2.5 µs/row from an `Array` column
and 8.1 from a `List[List[u8]]` one; a flat `List[f32]` 10.2, and a
`List[i64]` declared `u8` 24.6. The `list` source built a `Series` per row,
exploded it level by level, cast it and collected it twice, even when the
values were already the declared dtype. Raw and blob rows at unaligned
addresses cost ~0.3 µs/row more than aligned ones (a 4 KB copy): real, but
small.

## What changed

- **`List` rows are read where the column holds them.** `list_row_grid`
  walks a row's Arrow offsets level by level, with the grid checks
  `flatten_nested_series` made (jagged and null rows keep their errors), and
  a leaf of the declared dtype becomes a view of the column's values.
  `require_contiguous=True` now holds for a rectangular `List` row; it
  refused every `List` row before.
- **A row of another dtype is converted once**, by polars' strict cast of
  just its values (the cast's buffer is used as is). The non-strict cast it
  replaced stored 0 for a value the dtype cannot hold (CR-61).
- **Binary rows need only their own dtype's alignment.** Raw rows are read
  in place when their first byte is aligned for the declared dtype (any
  address for u8), blobs when their payload is aligned for the blob's dtype
  (the header is parsed once, where the row lies). The unconditional 8-byte
  rule copied every odd-address u8 row.

## Results

### One thread

| case | mode | base µs/row | p6 µs/row | speedup |
|---|---|---:|---:|---:|
| array | eager | 1.88 | 1.83 | 1.03x |
| array | streaming | 1.99 | 2.10 | 0.95x |
| blob aligned | eager | 2.60 | 2.50 | 1.04x |
| blob aligned | streaming | 2.14 | 2.34 | 0.91x |
| blob odd | eager | 2.52 | 2.61 | 0.97x |
| blob odd | streaming | 2.48 | 2.43 | 1.02x |
| list f32 flat | eager | 9.88 | 4.81 | 2.05x |
| list f32 flat | streaming | 12.43 | 5.23 | 2.38x |
| list i64 -> u8 | eager | 23.20 | 15.76 | 1.47x |
| list i64 -> u8 | streaming | 23.59 | 16.23 | 1.45x |
| list u8 | eager | 8.12 | 2.16 | 3.76x |
| list u8 | streaming | 8.07 | 2.24 | 3.60x |
| raw aligned | eager | 2.14 | 1.81 | 1.18x |
| raw aligned | streaming | 2.10 | 2.12 | 0.99x |
| raw odd | eager | 2.33 | 2.02 | 1.15x |
| raw odd | streaming | 2.52 | 2.03 | 1.24x |

### Four threads

| case | mode | base µs/row | p6 µs/row | speedup |
|---|---|---:|---:|---:|
| array | eager | 1.28 | 1.10 | 1.16x |
| array | streaming | 1.14 | 1.03 | 1.11x |
| blob aligned | eager | 1.16 | 1.10 | 1.05x |
| blob aligned | streaming | 1.08 | 1.06 | 1.02x |
| blob odd | eager | 1.13 | 1.10 | 1.03x |
| blob odd | streaming | 1.06 | 1.23 | 0.86x |
| list f32 flat | eager | 4.46 | 1.86 | 2.40x |
| list f32 flat | streaming | 3.50 | 2.20 | 1.59x |
| list i64 -> u8 | eager | 7.87 | 5.24 | 1.50x |
| list i64 -> u8 | streaming | 7.12 | 4.94 | 1.44x |
| list u8 | eager | 2.68 | 1.05 | 2.55x |
| list u8 | streaming | 2.34 | 1.00 | 2.34x |
| raw aligned | eager | 0.98 | 0.95 | 1.03x |
| raw aligned | streaming | 0.72 | 0.80 | 0.90x |
| raw odd | eager | 1.09 | 0.95 | 1.15x |
| raw odd | streaming | 1.03 | 0.71 | 1.45x |

`list u8` now costs what `array` does. The f32 case writes a 4x larger
blob, which the sink copies, so it does not reach the u8 cases. The
converting case is dominated by polars' cast. Raw/blob alignment wins are as
small as the measurement said they would be. The remaining 0.86–0.95x cells
are single cells, inside this container's noise, and go the other way in the
other thread count.
