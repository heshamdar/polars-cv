# Polars 2.0 upgrade (1.44.2 → 2.0.0 Python / 0.54.4 → 0.55.2 Rust)

What polars 2.0 changes for polars-cv, what this upgrade did about each item,
and what is left open. Sources: the official upgrade guide
(`docs/source/releases/upgrade/2.md` at tag `py-2.0.0`, 2,361 lines), the
polars source at that tag, and the published `pyo3-polars` 0.27/0.28 crates.

## Versions

| Component | Before | After |
|-----------|--------|-------|
| `polars` (Python) | `>=1.44.2,<2.0` (locked 1.44.2) | `>=2.0.0,<3.0` (locked 2.0.0) |
| `polars` / `polars-arrow` / `polars-buffer` / `polars-utils` (Rust) | 0.54.4 | 0.55.2 |
| `pyo3-polars` | 0.27 | 0.28 |
| `pyo3` | 0.28 | 0.29 |
| `object_store` / `reqwest` | 0.13 / 0.12 | unchanged (what polars-io 0.55.2 uses) |

`pyo3-polars` 0.28's source is identical to 0.27's apart from the version
pins, and the Rust side compiled and passed clippy with no source change. The
MSRV stays 1.96. The `object_store` 0.14 / `reqwest` 0.13 bump seen on
polars' `main` (and its `py-2.0.0` tag) is **not** in the published 0.55.2
crates. polars-cv must match polars-io (it reuses polars' object-store cache), so
both stay. reqwest 0.13 also drops the `rustls-tls` feature this crate relies
on for TLS: the next polars crate release needs a decision between `rustls`
(aws-lc-rs) and `rustls-no-provider` + ring.

FFI note: the 0.55-built plugin still runs under polars 1.44.2 once the new
keyword is removed, so the floor is set by the Python API
(`register_plugin_function(is_deterministic=)`) and by the behaviour the suite
now pins, not by an ABI break.

## The headline features

### Plugin CSE (`is_deterministic`)

`register_plugin_function` gained `is_deterministic: bool = True`. A
deterministic plugin call is eligible for common-subexpression and subplan
elimination; Polars compares calls by library, symbol and **serialized
kwargs**.

- **Before this upgrade it never fired for polars-cv across separately built
  expressions.** Each `LazyPipelineExpr` got a `uuid4` node id, and the graph
  JSON (the kwargs) carried those ids, so two builds of the same pipeline were
  never equal (`expr_a.meta.eq(expr_b)` was `False`). Only reusing the very
  same `Expr` object was merged.
- **Fix:** the wire names nodes by position (`n0`, `n1`, ...). Node references
  inside ops (binary operands, masks, `channel_merge`, `rasterize(shape=node)`)
  are renamed through the typed catalogue: `GraphStep::operands_mut`, used by
  `Plan.to_spec`, which refuses a reference the graph does not name rather than
  emitting a raw id. `operands_mut_reaches_exactly_the_operands` holds it to
  `operands()` over every catalogue sample (watched failing with a mutated
  `channel_merge` arm).
- `_plugin.call` passes `is_deterministic=True` explicitly, so a future default
  flip cannot silently turn it off. Every polars-cv function is pure in its
  inputs (no op draws randomness); path reads see storage as it is during the
  query, as polars' own scans do.
- Side benefit: the Rust compiled-graph cache (keyed by the graph JSON) now hits
  across rebuilt expressions too.
- Limits: Polars merges calls whose pipeline *and sink* are equal. The same
  pipeline sunk to `numpy` in one column and `png` in another is still two
  calls; a multi-output sink remains the way to share a decode across formats.

Tests: `tests/test_plugin_cse.py` (verified on `explain()`: one `vb_graph` call,
results equal to `comm_subexpr_elim=False` under both engines).

### Out-of-core spilling

In polars-ooc 0.54 the `DataFrame` spill backend was an in-memory stub (see
`../2026-06-27-polars-0.54-upgrade`). In 0.55 it is real. Past the memory
budget, `sort`, `group_by`, joins and windows write Arrow IPC to
`POLARS_OOC_SPILL_DIR` and read it back.

- **All polars-cv outputs survive a real spill byte-identically:** Binary,
  the numpy Struct, the `polars_cv.ndarray` extension type (tag kept), nested
  List and fixed Array. Checked against the in-memory engine under 1 MB and
  32 MB budgets; `POLARS_OOC_LOG_METRICS` reported ~85 spills across
  sort/partition/sink contexts.
- **Upstream livelock (polars 2.0.0).** With a budget the query cannot get
  under, a streaming `sort` can spin forever in
  `polars_ooc::MemoryManager::do_spill`: `while should_spill()` keeps finding
  the same pinned spillables and never yields to the spill tasks it spawned.
  Reproduced with polars alone, polars-cv not imported:
  `ooc_livelock_repro.py` here hangs in roughly 1 run in 4–6 at
  `POLARS_OOC_MEMORY_BUDGET_MB=1` (gdb: one executor thread in `do_spill`,
  every other thread parked). The forced-spill test uses 32 MB, which still
  spills and did not hang in 30 runs. Worth reporting upstream.
- **`POLARS_OOC_SPILL_POLICY` no longer exists.** The repo's OOC tests set it,
  so they had been exercising nothing. They are rewritten to force a spill by
  budget and to *prove* it happened from the metrics report; the old canary
  asserting no spill files is gone.
- **The plugin's memory counts toward the budget.** Polars' spill decision
  reads a global allocation counter kept by `polars_ooc::Allocator`. polars-cv
  allocates through polars' capsule (`PolarsAllocator`), so its decoded images
  count. One defect was found and fixed: in **debug** builds view-buffer's
  SIMD parity check made a buffer on a helper thread and freed it on the
  caller's. Polars batches counts per thread (4 MB) and a dead thread's batch is
  lost, so the allocation was never counted but the free was, and the count fell
  to 0 after a few calls (OOC would then never spill). Release wheels were not
  affected. `test_polars_counts_the_plugins_memory` pins it.
- What OOC does **not** do: spill inside a plugin call. A row (one image) is
  decoded, processed and encoded whole.

### Streaming is the default lazy engine

`LazyFrame.collect()` now resolves `engine="auto"` to streaming. polars-cv
already treated streaming as primary (the test suite pins it, with an
in-memory CI lane), so nothing broke. Related wins:

- **`.over()` windows are native streaming nodes.** `grouped_scan`, the
  metrics' one per-group scan, used to fall back to the in-memory engine
  (accepted by the streaming guard only straight after its sort). It now
  streams, and the guard's special case and the edge-walking it needed are
  removed. `test_grouped_scan_stays_streaming` asserts no fallback at all.
- Row order is no longer incidental for joins/group_bys under the default
  engine. polars-cv's metrics already sort or use `maintain_order` where order
  matters; the full suite passes under both engines.

## Breaking changes, item by item

| 2.0 change | Effect on polars-cv | Action |
|------------|---------------------|--------|
| Streaming default for lazy `collect` | None (already streaming-first) | Docs updated; dead `hasattr(set_engine_affinity)` guards removed |
| `read_ipc` → `scan_ipc().collect()`, `memory_map=` removed | Benchmark harness passed `memory_map=False` | Removed |
| `pl.concat(how="horizontal")` strict heights | 3 call sites, all 1-row aggregates | None needed |
| `explode(empty_as_null=False)` default | Both call sites already pass `empty_as_null=True` | None |
| `is_in` strict coercion | **`froc_curve_lazy(thresholds=[1])` raised** | Fixed (Float64 needle) + test |
| `Expr.hash` values changed | Seeded bootstrap draws differ from 1.x | Tests re-pinned; CHANGELOG note |
| Strict struct casts | One test built a 3-field struct and cast to the 5-field schema | Test fixed |
| Unknown Arrow extension → `pl.Extension` | A tagged Parquet read without polars-cv now keeps the tag (was: storage + warning) | Test + docs updated (no data loss either way) |
| Supertype Int/UInt64 → Int128 | No mixed signed/UInt64 arithmetic in the package | None |
| `to_struct` null rows, `rename_fields` strictness, `Map` dtype, SQL, CSV changes | Not used | None |
| Removed deprecated APIs (`melt`, `with_row_count`, `profile`, ...) | None used (grep of package, tests, benchmarks, docs) | None |
| `cut`/`qcut` deprecated | Not used | None |

Test-suite result before fixes: 13 of 5,462 fast tests failed (all accounted
for above). One slow-lane failure is not from this upgrade:
`test_chain_matches_composed_references_deep` finds `resize_to_height(1)` then
`cast("u8")` where the resize lands one ulp below 0.5 (`0x1.fffffffffffffp-2`)
and the engine and the oracle round differently. The same example fails
identically on the pre-upgrade commit with polars 1.44.2.

## Bootstrap re-pin

`_UNIT_WEIGHT_BOUNDS` was frozen from the pre-stratification resampler. It was
re-derived, not just re-pasted: the pre-stratification commit (`1012261`) run
under polars 2.0 reproduces the new values, the lroc g2 upper bound to within
1 ulp (`0.96` vs `0.9600000000000001`). The test keeps exact equality against
the current values. `test_a_replicate_without_positives_nulls_its_groups_bounds`
moved to seed 40, where g1's replicate 43 draws no positive.

## Follow-ups (not done here)

0. **Report the `do_spill` livelock upstream** with `ooc_livelock_repro.py`.
   Until it is fixed, a very low `POLARS_OOC_MEMORY_BUDGET_MB` can hang a
   query; a realistic budget (a share of RAM, the default) is far from this
   regime.

1. **Bootstrap reproducibility across polars versions.** The draw depends on
   `Expr.hash`, which polars does not keep stable across versions. A fixed
   integer mix (e.g. splitmix64 written as polars integer expressions) would make
   seeded intervals stable across upgrades.
2. **Single-row calls hold ~2× their output** (101 MB output counted as 202 MB
   while held; two rows are exact). It is released with the frame. It looks
   pre-existing and unrelated to 2.0; worth a look in the output encoder.
3. **`grouped_scan` could be revisited.** The sort-then-window design was chosen
   because windows fell back; with native windows, an unsorted
   `cum_sum().over()` may now stream without the full sort. Needs measurement.
4. **Next polars crate release:** object_store 0.14 / reqwest 0.13 (TLS
   feature rename, see above), and re-check the `quick-xml` advisory ignores,
   which that bump resolves.
5. **Benchmarks were not re-run** (they need an optimised build). The previous
   upgrade's report shows how to compare back-to-back on one host.
