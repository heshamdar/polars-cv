# Polars upgrade (0.46 → 0.54.4 Rust / 1.37 → 1.42 Python) + OOC/spill verification

Date: 2026-06-27

## Versions landed on

| Component | Before | After |
|---|---|---|
| `polars` / `polars-arrow` (Rust crate) | 0.46 | 0.54.4 |
| `pyo3` | 0.23 | 0.28 |
| `pyo3-polars` | 0.20 | 0.27 |
| `polars` (Python wheel) | 1.37.1 | 1.42.0 |
| `polars-runtime-32` | 1.37.1 | 1.42.0 |

`polars-cv/pyproject.toml`'s Python constraint stays `polars>=1.0,<2.0` (intentionally not raised — see "Decisions" below). `uv.lock` was re-resolved with `uv lock --upgrade-package polars` and now pins 1.42.0.

## Upstream feature-graph gaps hit during the bump

Bumping `pyo3-polars` 0.20→0.27 pulls `polars`/`polars-arrow` from 0.46 straight to 0.54.4 — no usable intermediate stop exists in the resolver. Three distinct upstream feature-wiring gaps surfaced along the way, all fixed on the `polars-cv`/`view-buffer` side without patching polars itself:

1. **`pyo3-polars`'s `derive` feature vs `polars-ops`/`polars-mem-engine`/`polars-stream`.** Enabling `derive` turns on `polars-core/object` and `polars-plan/python` without enabling the matching features on three downstream crates, leaving non-exhaustive `DataType::Object`/`IR::PythonScan`/`IRFunctionExpr::StringExpr` matches that don't compile under `--all-targets` feature unification. Fixed by explicitly requesting `"object"` and `"strings"` on our `polars` dependency and enabling `pyo3-polars`'s `"lazy"` feature (commit `d0d975a`).
2. **A dead `pyo3-extension` Cargo feature**, pre-existing before this upgrade but only now triggered: the `[dependencies]` `pyo3` line hardcoded `extension-module`/`abi3-py39` unconditionally, making the `pyo3-extension` feature gate (which `pyproject.toml`'s maturin config requests) a no-op. With `extension-module` always on, `cargo test -p polars-cv` could never link its test binary (PyO3 omits libpython linkage for extension-module builds). Fixed by removing the hardcoded features from the `pyo3` dependency (commit `91e5198`).
3. **`polars-expr`'s `strings` feature vs the optional `polars-time` dependency.** `polars-expr-0.54.4/src/dispatch/strings.rs` unconditionally does `use polars_time::prelude::StringMethods;`, gated only by `#[cfg(feature = "strings")]`, but `strings` never activates the optional `polars-time` dependency. This only broke `maturin develop --release` (a `--lib`-only build with no dev-dependencies), not `cargo test`/`cargo check --all-targets` — those targets union in enough default features via `[dev-dependencies]` to mask the gap. Fixed by adding `"dtype-datetime"` to both the `[dependencies]` and `[dev-dependencies]` `polars` feature lists in `polars-cv/Cargo.toml` (commit `e07261f`), which (via top-level polars' weak `?/` feature wiring through `polars-lazy`) activates the optional `polars-time` dependency.

Also confirmed: the root `rust-toolchain.toml` already floats `channel = "stable"`, satisfying `polars-ooc-0.54.4`'s use of `std::hint::cold_path` / atomic `try_update` APIs that need a recent stable compiler (installed: rustc 1.96.0). No toolchain file change was needed.

## Verification gate (Step 3)

All green post-upgrade:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test` (workspace), `cargo test -p view-buffer --all-features`, `cargo test -p polars-cv`
- `maturin develop --release`
- `pytest tests/ -m "not network"` — **1402 passed, 0 failed**
- `tests/test_graph_cache.py::TestCacheStreaming` specifically — 2 passed
- `uvx ruff check` / `uvx ruff format --check` — clean

## Performance regression check

The regression suite (`python -m benchmarks.regression.run_suite --scenarios pipelines --counts 300 --threads 1`) initially showed every scenario "REGRESSED" by -29% to -46% throughput comparing the pre-upgrade baseline (captured hours earlier) against the post-upgrade candidate — see `baseline-0.46.json` vs `candidate-0.54.json` in this directory.

This was investigated rather than accepted at face value, per the plan: the regression hit eager and streaming paths near-uniformly, which doesn't fit a polars-engine-level (threading/morsel-scheduling) cause, and this project's own prior report (`benchmarks/reports/2026-06-12-streaming-analysis/README.md`) already documents severe host-level noise on this shared-cloud infrastructure ("the shared host slowed measurably over the session... identical-binary regression-suite runs wobble ±5–13%").

To test that hypothesis directly: used `git worktree` to check out and rebuild the exact pre-upgrade commit (`78f4d34`) in isolation, then re-ran the identical benchmark suite against it *immediately*, on the same host, right before re-measuring the candidate (`old-now.json`). Comparing `old-now.json` against `candidate-0.54.json` — both measured back-to-back on the same host — shows all 10 scenarios **NEUTRAL**, within ±7%:

```
framework / operation / size / count                           thru%    lat%    mem%  status
--------------------------------------------------------------------------------------------
polars-cv-eager        heavy_pipeline         256x256 n=300      +2.8    -2.7    -1.3  NEUTRAL
polars-cv-eager        imagenet_preprocess    256x256 n=300      +6.6    -6.2    -1.1  NEUTRAL
polars-cv-eager        light_pipeline         256x256 n=300      -0.3    +0.3    -1.0  NEUTRAL
polars-cv-eager        medical_pipeline       256x256 n=300      -1.5    +1.5    -0.8  NEUTRAL
polars-cv-eager        medium_pipeline        256x256 n=300      +2.3    -2.3    -1.0  NEUTRAL
polars-cv-streaming    heavy_pipeline         256x256 n=300      +3.4    -3.3    -1.4  NEUTRAL
polars-cv-streaming    imagenet_preprocess    256x256 n=300      +5.7    -5.4    -1.4  NEUTRAL
polars-cv-streaming    light_pipeline         256x256 n=300      +0.5    -0.5    -1.4  NEUTRAL
polars-cv-streaming    medical_pipeline       256x256 n=300      -2.5    +2.5    +0.5  NEUTRAL
polars-cv-streaming    medium_pipeline        256x256 n=300      +2.4    -2.3    -1.4  NEUTRAL

Summary: IMPROVED=0, REGRESSED=0, NEUTRAL=10, MISSING=0, NEW=0
PASS: no regressions.
```

**Conclusion: the polars 0.54.4/1.42.0 upgrade introduces no genuine performance regression.** The apparent -29% to -46% drop was entirely host-level drift between the two measurement times, consistent with documented infrastructure noise — not anything in the dependency bump. All three JSON artifacts (`baseline-0.46.json`, `old-now.json`, `candidate-0.54.json`) are kept in this directory alongside their `.meta.json` sidecars.

## Out-of-core (OOC) / spill-to-disk investigation

### The headline finding: spilling is not yet functionally implemented in 0.54.4

Polars' new streaming engine (landed ~1.40) wires a `polars-ooc` crate into `group_by`, `sort`, and equi-`join` nodes (`polars-stream-0.54.4/src/nodes/{group_by,equi_join,multiplexer,zip,...}.rs` all construct `SpillFrame`/use `SpillToken`), with a real set of tunable config knobs in `polars-config-0.54.4`:

| Env var | Default | Purpose |
|---|---|---|
| `POLARS_OOC_SPILL_POLICY` | `no_spill` | `no_spill` / `spill` — **spilling is opt-in, off by default** |
| `POLARS_OOC_SPILL_FORMAT` | `ipc` | Only `ipc` is a valid value currently |
| `POLARS_OOC_MEMORY_BUDGET_FRACTION` | `0.8` | Fraction of available memory before spill triggers |
| `POLARS_OOC_SPILL_MIN_BYTES` | `102400` | Minimum size worth spilling |
| `POLARS_OOC_SPILL_DIR` | platform default (`/var/tmp/polars-{USER}/spill` on Linux, always real disk) | Spill destination |
| `POLARS_OOC_DRIFT_THRESHOLD` | 4MB | Memory-tracking drift correction |

However, source inspection of `polars-ooc-0.54.4` (the exact version pinned in this repo's `Cargo.lock`, matching the installed `polars` 1.42.0 wheel) shows the actual spill **backend** for `DataFrame` is not implemented yet:

```rust
// polars-ooc-0.54.4/src/spill_frame.rs
impl Spillable for DataFrame {
    // TODO: just a dummy spill for now. Boxed to reduce size.
    type Spilled = Box<DataFrame>;
    async fn spill(&self) -> Self::Spilled {
        Box::new(self.clone())   // <-- clones in memory, never touches disk
    }
    async fn unspill(location: &Self::Spilled) -> Self {
        (**location).clone()
    }
}
```

```rust
// polars-ooc-0.54.4/src/memory_manager.rs
pub async fn spill(&self) {}        // no-op
pub fn spill_blocking(&self) {}     // no-op
```

So `SpillFrame::new()`/`new_blocking()` register cold partitions with a `SpillContext` and then call into `MemoryManager::spill()`/`spill_blocking()` — which do nothing. Even with `POLARS_OOC_SPILL_POLICY=spill` and the most aggressive thresholds (`MEMORY_BUDGET_FRACTION=0.0`, `SPILL_MIN_BYTES=0`), no `DataFrame` is ever actually evicted to disk in this polars version. This is why no spill-event logging exists in `polars-ooc`'s source (confirmed by an empty grep for `verbose|log::|eprintln|tracing::`) — there's nothing to log yet. The token/pin/lock architecture (`SpillToken`, `SpillContext` variants: `MostRecent`/`LeastRecent`/`Random`), the config surface, and the streaming-node integration are all real and already threaded through `group_by`/join/sort — but the disk-backed backend itself is upstream scaffolding for a feature still in development, not a working OOC path yet.

### What was tested given this

New file: `polars-cv/tests/test_streaming_ooc.py` (`@pytest.mark.slow` + `@plugin_required`, register the `slow` marker in `pyproject.toml`; CI's pytest invocation updated to `-m "not network and not slow"`).

Since real spill-to-disk doesn't happen, the tests verify what's actually exercisable today:

1. **Output-side correctness** — polars-cv's Binary (`.sink("blob")`) outputs survive `group_by(...).agg(.first())`, `group_by(...).agg()` (List), `sort()` on a plugin-derived scalar with a blob column carried along, and an equi-`join` of two blob-bearing frames — comparing `engine="streaming"` against `engine="in-memory"` via `pl.DataFrame.equals` (confirmed to do true deep byte-equality on Binary/Struct, not just dtype/identity checks).
2. **Input-side correctness** — `group_by`/`sort` running *upstream* of `.cv.pipe()`, so the plugin's input arrives via a group_by/sort result rather than directly from the source column; compared against direct evaluation.
3. **Compiled-graph-cache stability across morsels** — 2000 rows through a `group_by` + blob pipeline, comparing streaming vs eager. The plugin exposes no spill/recompile counter (`graph/compiled.rs`'s cache key is pure `graph_json` + column names, no data-derived state — by design), so this is verified indirectly via per-row correctness across many morsels, the same approach `test_graph_cache.py::TestCacheStreaming` already uses.
4. **`POLARS_OOC_*` env vars don't change correctness** — subprocess-based (the OOC config is read once into a process-wide `LazyLock`, so it can only be exercised by varying environment *before* the interpreter starts), tried with `no_spill` and with `spill` + the most aggressive thresholds available. Both produce identical, correct results.
5. **Regression canary**: `test_spill_directory_stays_empty` sets `POLARS_OOC_SPILL_DIR` to a fresh temp directory, forces `spill` policy with zero thresholds, runs a 500-row group_by+blob pipeline, and asserts the directory contains zero files afterward — a direct, falsifiable assertion of the dummy-stub finding above. This test is expected to start *failing* the day upstream implements the real disk-backed spill, at which point the other tests in the file should be revisited to additionally check for actual spill files rather than only correctness.

All 10 new tests pass. Run them with `pytest tests/test_streaming_ooc.py -v -m slow`.

## Caveat for future readers

Spilling, as currently implemented in polars 0.54.4/1.42.0, applies (in principle, once the backend lands) to `group_by`/`sort`/equi-`join` — never to the plugin's own elementwise `vb_graph` expression directly, since it's registered as an elementwise plugin function. What this work confirms is that polars-cv's Binary/Struct/List/Array outputs and inputs round-trip correctly through the new streaming-engine machinery (`SpillFrame`/`SpillToken`-wrapped group_by/sort/join nodes) regardless, and that the moment the dummy `Spillable for DataFrame` stub is replaced with a real disk-backed implementation upstream, the existing test suite already has correctness coverage in place across both spill directions — only the "files actually appear on disk" assertion will need updating.
