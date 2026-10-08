# Upstream issues for Polars (Phase 7 of `POLARS_2_PLAN.md`)

Ready-to-file drafts for `pola-rs/polars`. Each reproduces with Polars 2.0.0
alone (polars-cv not imported), verified 2026-10-08 unless noted. Each says
what polars-cv does in the meantime, so the workaround can be removed once
the issue is fixed.

---

## 1. Streaming `group_by` cannot build a list (`implode`, implicit list, `head`)

**Title:** Streaming engine falls back to in-memory for list-building
aggregations in `group_by` (`col`, `implode`, `head`)

**Reproduction** (Polars 2.0.0):

```python
import polars as pl

lf = pl.LazyFrame({"g": [1, 1, 2], "x": [3, 1, 2]})
for name, q in {
    "agg(col)": lf.group_by("g").agg(pl.col("x")),
    "agg(col.implode())": lf.group_by("g").agg(pl.col("x").implode()),
    "agg(col.head(2))": lf.group_by("g").agg(pl.col("x").head(2)),
    "agg(col.unique())": lf.group_by("g").agg(pl.col("x").unique()),
}.items():
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True, show=False)
    print(f"{name:20s} in-memory fallback: {'in-memory-map' in dot}")
```

```text
agg(col)             in-memory fallback: True
agg(col.implode())   in-memory fallback: True
agg(col.head(2))     in-memory fallback: True
agg(col.unique())    in-memory fallback: False
```

**Expected:** the plain list aggregation streams, as `unique()` (which also
builds a list per group) already does. The order within a group would be the
input order, or the order of a preceding sort with `maintain_order=True`.

**Why it matters:** per-key lists are the input of row-wise matchers
(detection-to-ground-truth matching takes one row per (image, class) with
both object lists). The fallback collects the whole input of the group-by.

**polars-cv meanwhile:** `group_objects` is the one remaining entry in
`tests/_streaming_guard.py`'s `KNOWN_FALLBACKS`. Two other sites were rewritten
around it: the weight cells became `unique()` then `.list.sort()`, and the
entity expansion became a join.

---

## 2. Feature: let a plugin declare a preferred morsel size in bytes

**Title:** Expression plugins: a hint for the morsel size (bytes or rows) the
streaming engine hands an elementwise plugin

**Problem.** The streaming engine sizes morsels by rows
(`POLARS_IDEAL_MORSEL_SIZE`, 100,000), and a Parquet scan hands an elementwise
plugin call one row group at a time. That is fine for scalar columns. It is
not for a plugin whose rows are large: an image plugin that decodes to
224×224×3 `f32` holds 602 KB per output row, so a 20,000-row group is about
12 GB inside one call. Out-of-core spilling cannot reach inside a plugin call.

**Today's only knob** is `pl.Config.set_streaming_chunk_size` /
`POLARS_IDEAL_MORSEL_SIZE`, which is process-global: it changes the morsels of
every node of every query, so a library cannot set it on its users' behalf.

**Ask:** a per-plugin hint, e.g. `register_plugin_function(...,
max_morsel_rows=...)`, or better a byte target the engine divides by an
estimate of row size, that the streaming engine uses to split morsels feeding
that expression. A row-count hint would already solve it for plugins whose
output row size is known at plan time (it is in polars-cv: the output schema
carries it).

**polars-cv meanwhile:** the streaming guide tells users to write image tables
with row groups sized by bytes, or to set the global chunk size themselves.
`tests/test_streaming_morsels.py` holds the guide's claims about morsel size
to the plugin.

---

## 3. A zero-width `Array` column cannot cross the Arrow C interface

**Title:** Importing a zero-width `Array`/FixedSizeList over the Arrow C
interface panics (length taken as 0, then sliced to the real length)

**Reproduction** (Polars 2.0.0):

```python
import polars as pl

class Stream:
    def __init__(self, obj):
        self.obj = obj
    def __arrow_c_stream__(self, requested_schema=None):
        return self.obj.__arrow_c_stream__(requested_schema)

for width in (0, 2):
    df = pl.DataFrame({"a": pl.Series([[0] * width] * 3, dtype=pl.Array(pl.Int64, width))})
    rt = pl.DataFrame(Stream(df))  # width=0 panics
    print(width, df.height, rt.height)
```

```text
PanicException: the offset of the new Buffer cannot exceed the existing length
```

Width 2 round-trips (3 rows → 3 rows); width 0 panics.

**Cause** (polars-arrow 0.55.2, `array/fixed_size_list/ffi.rs`,
`try_from_ffi`): when the child array is empty the length is set to `0`
(`if values.is_empty() { 0 }`), and the array is then sliced to the imported
`offset`/`length`, which overruns. For a zero-width list an empty child is
expected; the length should come from the C array's `length` (plus
`offset`), not from `values.len() / width`.

**Why it matters:** an empty image (zero rows or columns) is a zero-width
`Array` level, and every plugin input crosses this import.

**polars-cv meanwhile:** `unified_output_dtype` (`src/lib.rs`) refuses a
zero-width `Array` input at planning time with an explanation, before any
data crosses.

---

## 4. Out-of-core: `MemoryManager::do_spill` can livelock under a small budget

**Title:** Streaming `sort` can hang forever in `MemoryManager::do_spill` when
the memory budget cannot be reached

**Reproduction:** `polars-cv/benchmarks/reports/2026-10-06-polars-2.0-upgrade/ooc_livelock_repro.py`
(standalone, polars only). With `POLARS_OOC_MEMORY_BUDGET_MB=1` it hangs in
roughly 1 run in 4–6 on a 4-core machine (reproduced 2026-10-06 on 2.0.0).
gdb shows one executor thread in `do_spill` and every other thread parked.

**Cause (from reading the code):** `do_spill` loops `while should_spill()`.
When the remaining spillables are pinned (in use), it never yields to the
spill tasks it spawned, so the condition never changes.

**Expected:** `do_spill` yields or backs off when no progress is possible, or
gives up and lets the query proceed over budget.

**polars-cv meanwhile:** the forced-spill test uses a 32 MB budget (still
about 85 spills; no hang in 30 runs) and names the livelock if it times out.
