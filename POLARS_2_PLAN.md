# Plan: leaning on Polars 2.0 (streaming default, 0.55 crates)

## Context

0.34.0 made polars-cv run on Polars 2.0 (Python `>=2.0,<3`, crates 0.55.2,
pyo3-polars 0.28). This plan covers what comes next: where the codebase can be
simplified or sped up now that the streaming engine is the default and the 0.55
crates are in place. It keeps to the project rules: streaming is the main
mode, fix root causes, one authority per fact, and a bypass must fail.

**How this was investigated.** I diffed the 0.54.4 and 0.55.2 crates
(`polars-plan`, `-stream`, `-io`, `-core`, `-arrow`, `-ooc`, `-config`,
`-ffi`) and pyo3-polars 0.27 against 0.28. I read the installed `polars` 2.0.0
Python package. I then ran the questions that mattered against the debug
build on this 4-core container (scripts below, results inline). Debug timings
are only valid as ratios. Every perf phase is gated on the benchmark profile
as usual.

**The headline: 2.0 adds nothing new to the plugin interface.** pyo3-polars
0.28 and pyo3-polars-derive 0.22 are the same source as 0.27/0.21 apart from
version pins. `register_plugin_function` takes the same flags, and
`is_deterministic` was already adopted in 0.34. polars-plan's plugin loader
changed only internally (an `Arc` cache). So the gains below come from how
polars now *runs* plugins: streaming by default, with morsels sized by row
count. Some also come from 0.55's polars-io, and some from the streaming engine
now covering more operators natively.

## Findings

| # | Finding | Evidence |
|---|---------|----------|
| F1 | **A query can make the next run of the same pipeline single-threaded.** `CallTracker` (`row_split.rs`) decides whether a call spreads its rows over the plugin's pool from history ("did the last call overlap another?"). The tracker lives on the `CompiledGraph`, which is cached process-wide, and on a `static` per geometry function. So one query's last morsel decides how the next query's first call runs. Since 0.34's position-based wire ids, rebuilt expressions share a `CompiledGraph`, so they share this state too. | Measured, 360 PNGs, resize+blur → numpy. **Eager after a streaming run: 18.1 s vs 4.7 s fresh (3.9×).** Parquet with 240+120-row groups, streaming: run 0 5.75 s, runs 1–2 13.5 s. Two 180-row groups: 6.1 / 6.2 / **11.2 s** (timing-dependent). |
| F2 | **One plugin call is one morsel, and under 2.0 a morsel is a whole Parquet row group.** Morsels are sized by row count (`POLARS_IDEAL_MORSEL_SIZE`, default 100,000). The Parquet source splits a row group only above 150k rows, or at the last morsel. An in-memory frame splits into `num_pipelines` morsels. polars-cv cannot change the morsel size, and out-of-core does not spill inside a call, so a call's whole output is resident at once. | Measured: 60k rows in 1k-row groups gave 59 calls × 1000 rows + 4 × 250; 20k-row groups gave 2 calls × 20,000 + 4 × 5,000; a 300-row in-memory frame gave 4 × 75. Source: `polars-stream` `split_to_morsels`, `in_memory_source.rs`. |
| F3 | **Remote fetch for a call happens all up front.** `CompiledGraph::execute` → `prefetch_remote_sources` → `fetch::prefetch` fetches every remote path in the call before row 0 decodes. With F2, a 20k-row group of `s3://` paths holds all 20k encoded images in memory, and no network time overlaps with decoding. Eager is worse: one call covers the whole column. | Code: `graph/compiled.rs:300-330, 1111-1130`, `fetch.rs:255-276`, `cloud.rs:533-575`. |
| F4 | **The plugin runs two tokio runtimes.** `cloud::get_runtime()` builds its own `Runtime::new()` (one worker per CPU, ignoring Polars' thread settings). 0.55's object stores now resolve DNS through `CachingResolver`, which spawns onto polars' `ASYNC` runtime (`polars_core::runtime::ASYNC`, sized by `POLARS_MAX_THREADS` capped at 32 and `POLARS_ASYNC_THREAD_COUNT`). That is the plugin's own copy of it, so every S3/GCS/Azure read starts both. | `polars-io-0.55.2/src/cloud/dns.rs:8,175,236`, `options.rs:39`, `cloud.rs:418-430`. |
| F5 | **`cloud.rs` claims sharing that does not exist.** It says "the whole process — our fetches and polars' own scans together — stays inside `POLARS_CONCURRENCY_BUDGET`", and calls the store cache "process-wide". The plugin statically links its own polars-io, as `row_split.rs` already notes for `THREAD_POOL`. So the semaphore, the store cache and `ASYNC` are the plugin's copies, separate from the host's. Each copy reads the env var, so the real cap is up to 2× the budget. | `pl_async.rs` `CONCURRENCY_BUDGET` is a `static OnceLock` in each linked copy. |
| F6 | **`is_elementwise` is a parameter that no caller ever varies.** `_plugin.call` and `_PluginNamespace._plugin` accept it, and every caller passes or defaults `True`. | `grep is_elementwise python/` |
| F7 | **The default engine is now stated in two places.** Polars 2.0's `collect()` already resolves `"auto"` to streaming, yet the package passes `engine="streaming"` at 16 call sites, and `DetectionTable.collect(engine="streaming")` sets its own default. Each site restates the default, and each overrides `pl.Config.set_engine_affinity`. | `grep -rn 'engine="streaming"' python/` |
| F8 | **Two of the three remaining known streaming fallbacks now have exact native rewrites.** `agg(x.unique())` is a native streaming aggregate in 2.0, but `.unique().sort()` inside `agg` is not. A group→list→explode is a join. | Verified on 2.0.0 with `tests/_streaming_guard.py`: `agg(v.unique()).list.sort()` has no fallback and equals `agg(v.unique().sort())` (struct values, nulls). `draws.join(entities)` has no fallback and equals `join(group_by.agg(list)).explode`. |
| F9 | **Ordered windows stream natively.** `cum_sum().over(keys, order_by=...)` has no fallback, so `grouped_scan`'s global sort may be unnecessary wherever the caller does not consume the sorted order. This is the 2.0 report's follow-up 3. | Verified: no fallback for `cum_sum().over("g", order_by="b")`. |
| F10 | **`LazyFrame.collect_batches(chunk_size=)` is a streaming route into Python**, which is how a training loop should consume decoded tensors. The docs never mention it. | `polars/lazyframe/frame.py:4106`. |

### Considered and not proposed

- **`scan_external_reader` / `register_io_source`** (an image-directory scan):
  the reader callbacks are Python, so decoding would be GIL-bound. The
  plugin's `file_path` source plus `read_bytes` is already the native route.
- **`polars.io._expand_paths`** (lazy glob listing): it is private, and it
  returned an empty frame for a local glob in a quick try.
- **0.55's `register_object_store_builder` (hdfs etc.)**: it registers into
  the plugin's own polars-io copy, not the host's. Nothing asks for it yet.
- **0.55's bytes-based `ConcurrencyController`**: it needs each request's size
  before it starts. We read whole small objects without a `head` call (doing
  one would double the requests), so the count-based budget remains the right
  one.
- **FFI export realignment (`to_ffi_aligned`, new in 0.55)**: only the
  validity bitmap is realigned, so the cost is negligible.
- **The zero-width `Array` refusal in `unified_output_dtype`**: the polars
  bug it guards against is unchanged in 0.55
  (`FixedSizeListArray::try_from_ffi` still sets length 0, then slices). The
  refusal stays.
- **HTTP through polars-io**: polars still does not cache `Http` stores, so
  the pooled `reqwest` client remains correct.

## Phases

Each phase follows TDD. Write the failing test first and watch it fail *for
the reason it claims*. Then implement, run `scripts/verify.sh --fast`, and for
perf phases run `maturin develop --profile benchmark` base-vs-head on
`--changed REF`, interleaved, with a report in `benchmarks/reports/`.
Deletions land in the same commit as the mechanism that replaces them.

### Phase 1 — Row splitting decides from the present, never from history (F1)

**Status: implemented** (candidate (a), extended so a running call keeps
inviting idle threads before each range). A call runs its ranges on its own
thread; idle pool threads join while the plugin-wide `BUSY` count is under
the pool size, and leave after a range that finds it over. Tests
`a_lone_call_spreads_whatever_ran_before` and
`a_call_beside_a_small_call_takes_the_idle_threads` were watched failing on
the old tracker; `test_an_eager_call_after_a_streaming_run_uses_the_pool`
was watched failing with helpers disabled. Debug-build timings: eager after
streaming 18.1 s → 4.6 s; uneven row groups 13.5 s → 4.8 s. The
benchmark-profile gate (4 threads, interleaved;
`benchmarks/reports/2026-10-08-row-split-budget`): `split_streaming_then_eager`
2.2×, `split_streaming_uneven_row_groups` +11%, eager pipelines neutral,
streaming `medium` −3.8% (consistent, under the 7% gate; one helper range
per query crossing threads), streaming `light`/`heavy` +6.5%/+5.5%.

**Root cause.** Whether a call spreads is inferred from a `static`/cached
history bit. That bit carries state between unrelated queries, between
engines, and (since 0.34) between rebuilt expressions.

**The property the replacement must have** (and the tests pin): a call's
split depends only on what is running *now*, never on what ran before. Two
candidates, chosen by benchmark:

- (a) **A pool-wide thread budget.** One plugin-wide `static` replaces the
  per-graph and per-function trackers, because all of them share one
  `THREAD_POOL`. A starting call claims `min(free, wanted)` pool threads
  atomically (one if none are free, running inline) and releases them on
  drop. When alone it takes them all; streaming morsels share them.
- (b) **Always split onto the pool and let rayon's work stealing balance
  concurrent calls.** The calling threads block, so CPU use stays at the pool
  size. The CR-32-era measurement that ruled this out (`row_split.rs`: rows'
  buffers moving between threads, "up to half a byte-heavy streaming query's
  throughput") must be re-run under 2.0 before (b) is rejected again.

Either way `CallTracker`, `Call::spreads` and `geom_calls!` are deleted, and
`run_split` remains the only splitter.

**Tests first.**
- A Rust unit test: run a call that overlaps a second, then run a lone call on
  the same tracker, and assert the lone call spreads. This fails today: the
  history bit is set.
- A Python integration test: eager after streaming on the same pipeline
  uses more than one pool thread. Observe this through a debug-only counter
  of ranges per call, exported like `__debug_assertions__`; do not use
  timing.
- Add `streaming_uneven_row_groups` and `eager_after_streaming` to
  `benchmarks/scenarios/targeted.py`.

**Gate.** No regression on the existing streaming and eager cases. The two new
cases must reach the "fresh" timing (in debug, 13.5 s and 18.1 s should drop to
about 5–6 s).

### Phase 2 — Remote fetch is a bounded window, overlapped with decode (F3, F2)

**Status: implemented.** `fetch::Fetcher` (one per call and path column)
replaces `prefetch`/`FetchedBatch` for all three consumers. A row starts its
own fetch and the next `get_concurrency_limit()` rows' fetches, each its own
abortable task on `ASYNC`, and waits only for its own. A body is freed after
its last row. The window is per row thread, with no range plumbing: a
thread's window can reach into the next range, which only means that range
finds its first rows already fetched.

The `bytes`-ahead peak was measured at the user-facing entry point
(`_lib._last_fetch_peak_resident`; `tests/test_fetch_window.py` was watched
failing at 64/64 and 128/128 resident, and now passes at 3–7). Dedup is held
(one request per distinct path per call). The test-design note above about
server-side counting gave way to that hook: a server cannot see consumption.

Gate (`benchmarks/reports/2026-10-08-fetch-window`): remote `file_path`
+51–61% at 1 thread; with 20 ms latency +43–45% / +6% at 1 / 4 threads.
At 4 threads with no latency −4 to −6% (decode now shares the cores with the
downloads). Connection reuse is unchanged.

**Root cause.** "Prefetch the call" was written when a call was assumed to be
small. Under 2.0 a call is a row group (F2), and under eager it is the whole
column.

**Mechanism.** `fetch::prefetch` (the whole call up front) becomes a
**fetch-ahead window** that each row range consumes in row order. While row
*i* decodes, rows *i+1 … i+W* are in flight.
- `W` comes from polars' own budget: `pl_async::get_concurrency_limit()`,
  public since 0.55. It is not a new constant.
- Distinct paths are still fetched once per call. Occurrences are counted up
  front, and a fetched buffer is dropped after its last row reads it, so
  duplicates do not stay resident.
- `read_bytes.rs` takes the same window, so it stays a fetch with the decode
  left out (one mechanism, as today).
- `FetchedBatch` and the "a miss falls back to an inline fetch" arm in
  `row_bytes` are deleted. A row can no longer be missing from the window, so
  that arm would only hide a bug.

**Tests first.**
- Extend the local HTTP server fixture (`test_file_path_prefetch.py`,
  `test_remote_connection_reuse.py`) to record the peak number of fetched
  but not yet consumed bodies, and assert it is at most `W` × ranges. This
  fails today: the peak is the call's row count.
- With injected latency L and a known per-row compute cost, assert that the
  wall time is well under fetch time plus compute time. This fails today
  (no overlap).
- The existing `allowed_roots` tests must still pass unchanged: denied
  paths are never requested.

**Gate.** Run `remote_*` scenarios with `--latency-ms 0` and `20`. The
connections-per-file ratio must not worsen, and peak RSS must drop for a
large single call.

### Phase 3 — One async runtime, and honest docs about what is shared (F4, F5)

**Status: implemented.** `cloud::get_runtime()`, `CloudError::RuntimeError`
and the direct `tokio` dependency are deleted; every future runs on
`ASYNC.block_on`. The root `clippy.toml` disallows
`Runtime::new`/`Builder::new_{multi,current}_thread`. It was watched failing
on the old `cloud.rs`, and again with `tokio` re-added and both constructors
called, so it is dormant, not broken, while no crate names `tokio`.
`test_async_runtime.py` was watched failing (4 threads started, one per CPU,
against `POLARS_ASYNC_THREAD_COUNT=3`) and now passes. `remote` benchmark at
1 and 4 threads: no regression, `remote_http_paths` +4–9%
(`benchmarks/reports/2026-10-08-one-async-runtime`).

- Delete `cloud::get_runtime()` and drive every remote read on
  `polars_core::runtime::ASYNC`, reached the way `THREAD_POOL` already is
  (`pyo3_polars::export::polars_core`). That leaves one runtime, sized by
  polars' own knobs. Phase 2's window is written against it.
- **Make a bypass fail.** List `tokio::runtime::Runtime::new` and
  `tokio::runtime::Builder::new_multi_thread` under `disallowed-methods` in a
  root `clippy.toml`, so clippy (`-D warnings`, already in CI) rejects a
  second runtime. Watch it fail on the current `cloud.rs` before deleting.
- **Test first.** On Linux, after a remote read, count the threads in
  `/proc/self/task` named like tokio workers, and assert there is one pool.
  This fails today, when an `s3://` read against a local mock starts both.
  Use an `http://` read if the mock is unavailable, since that only starts
  ours today. Then assert the count matches `POLARS_ASYNC_THREAD_COUNT`.
- Correct the `cloud.rs` module and function docs, the `Cargo.toml`
  comment, and `docs/user-guide/concepts/sources.md`. The concurrency
  budget, the store cache and the async runtime are **plugin-wide**: the
  plugin and the host each have their own, and each reads the same env
  vars. This is documentation, so no test is added. Say it in the
  CHANGELOG under *Fixed (docs)*.

### Phase 4 — Metrics: fewer fallbacks, one engine decision (F7, F8, F9)

**Status: implemented** (4.3 measured, not adopted).
- 4.1 and 4.2: done. Removing the two `KNOWN_FALLBACKS` entries first made the
  six bootstrap plans fail `test_plan_stays_streaming`; the rewrites make
  them pass. `_UNIT_WEIGHT_BOUNDS` and the slow lane are unchanged.
- 4.3: the ordered window is 15–35% faster than sort-then-window on the scan
  alone (2M rows, 1,000 groups), with equal values. It is not adopted.
  `over()` takes one `descending` for all keys and two callers mix
  directions; negating a key misplaces NaN. `_lazy_resample` consumes the
  sorted order. A second scan path for some callers is not worth it. The
  reasons are recorded in `_grouped_scan.py`.
- 4.4: there is no helper. "auto" is polars' default, so the rule is simply
  that the package never chooses an engine:
  `tests/test_engine_choice.py` (AST scan with fixtures, watched failing on
  the 14 call sites and on `DetectionTable.collect`'s default) refuses a
  literal `engine=` keyword or an `engine` parameter defaulting to anything
  but `"auto"`.

1. **`_sampling_units`.** Change `value.unique().sort()` to
   `value.unique()` in the `agg`, followed by `.list.sort()`. Remove its
   `KnownFallback`. The `test_guard_*` fixtures that use it as "a known
   fallback the guard must accept" switch to `group_objects`, which remains
   a fallback.
2. **`_resolve_bootstrap_samples`.** Replace the
   `group_by(entity).agg(image_id)` → `explode` with a join of draws to
   `(entity, image_id)`. Remove its `KnownFallback`. The oracle is the
   pinned `_UNIT_WEIGHT_BOUNDS`, plus the seeded `sample_col=` tests: the
   bounds must not move by a single ulp. If row order feeds the slot ids,
   keep that order explicitly with `maintain_order` or a sort on the
   existing deterministic key; do not re-pin.
3. **`grouped_scan` (follow-up 3).** For each caller, check whether it
   consumes the sorted frame or only the scan values. Where only the values
   matter, offer `grouped_scan(..., sorted_output=False)` built on
   `.over(keys, order_by=by)`, with no global sort. Adopt it only if
   `benchmarks` shows a win. Callers that need sorted output (the AUC
   integrals, which merge sorted frames) keep the sort.
4. **One engine decision.** Every collect in the package goes through one
   `metrics._collect` / `_collect_all` helper; the 16 literal
   `engine="streaming"` arguments are deleted. Enforce it with a ruff
   `banned-api`-style check, or the existing structural lane, rejecting
   `engine=` in `python/polars_cv` outside the helper (with fixtures).
   The helper passes `"auto"` (see *Decisions*), and
   `DetectionTable.collect(engine=)` defaults to it.

`group_objects` stays a fallback. Polars cannot build an ordered list inside
a streaming group-by. A `struct(row, …).unique()` followed by `list.sort()`
would stream, but it relies on an implementation detail; this is reported
upstream instead (Phase 7).

### Phase 5 — Delete the `is_elementwise` knob (F6)

**Status: implemented.** Checking the declaration first found a gap: with
`is_elementwise=False` a plugin call becomes a whole-column
`columnar-function`, which the streaming guard only flagged for Python UDFs,
and no streaming plan covered the plugin's own entry points, so flipping it
failed nothing. The guard now flags a `columnar-function` over a compiled
plugin (polars' own `int_range` columnar nodes stay native). `PLANS` gains
`cv.pipe`, `cv.read_bytes`, `cv.width`, `contour.area` and `point.translate`,
plus a fixture test of a non-elementwise call. Watched failing: 30 plans
with the flag flipped. The parameter is then deleted.

`_plugin.call` and `_PluginNamespace._plugin` lose the parameter, and the
call always registers `is_elementwise=True`. The docstring gains a fourth
"cannot be skipped" item: every function is row-independent. A future
non-elementwise function needs its own design and streaming tests, not a
flag. A caller passing the keyword now gets a `TypeError` from the
signature, so no tombstone test is added. Record the removal in the commit
message and the CHANGELOG.

### Phase 6 — Document streaming for image-sized rows (F2, F10)

**Status: implemented.** `streaming.md` gains "How big a call is" (rows per
call, the memory estimate, row groups sized by bytes, the chunk-size knob and
why polars-cv does not set it) and "Feeding a training loop"
(`collect_batches` with a fixed-shape `array` sink). That gives
`(rows, *shape)` uint8 batches that view the batch rather than copy it
(`to_numpy()`). `tests/test_streaming_morsels.py` checks each claim at the
plugin through a new `_lib._take_max_split_rows()`:
- a Parquet call is at most one row group (600 rows in groups of 100: ≤ 100);
- the chunk size bounds in-memory frames and splits a large row group;
- the training-loop batches have the documented shape and are views.

Watched failing: a single row group gives 150-row calls against the ≤ 100
bound.

In `docs/user-guide/concepts/streaming.md`:
- **A morsel is a plugin call.** Peak memory is roughly rows per morsel ×
  output size × pipelines, and the Parquet row group sets rows per
  morsel. Write image tables with `row_group_size` chosen by bytes, or set
  `pl.Config.set_streaming_chunk_size` (the same knob as
  `POLARS_IDEAL_MORSEL_SIZE`, process-global). Explain why polars-cv does
  not set it itself: it is a global side effect. Back this with a test that
  writes a Parquet file with row groups of R and asserts no plugin call
  sees more than R rows, read from the Phase 1 debug counter.
- **Feeding a training loop:**
  `lf.collect_batches(chunk_size=...)` with `numpy_from_column` per batch,
  as a doc-tested example.

### Phase 7 — Upstream issues (no code here)

**Status: drafted** in `UPSTREAM_ISSUES.md`, ready to file on
`pola-rs/polars`. That repository is outside this session's GitHub scope,
and filing is an outward-facing action for the user. Each issue was reproduced
on Polars 2.0.0 alone:
- implode / implicit list / head in a streaming group-by;
- the morsel-size hint (feature request);
- the zero-width `Array` across the C interface, which panics while width 2
  round-trips;
- the `do_spill` livelock (repro from 2026-10-06).

- Ordered `implode` in streaming `group_by` (it would remove the last
  `KnownFallback`).
- Let a plugin declare a preferred morsel size in bytes. This is the real
  fix for F2; today only a global row count exists.
- `FixedSizeListArray::try_from_ffi` with zero-width arrays (still present
  in 0.55).
- The `MemoryManager::do_spill` livelock, carried over from the 2.0 report
  with its repro.

## Order and size

| Phase | Kind | Size | Why this position |
|-------|------|------|-------------------|
| 1 | Perf bug, measured 2.3–3.9× | S–M | Largest measured loss; reachable in normal notebook use |
| 2 | Memory + latency, streaming | M | Biggest streaming risk for remote sources |
| 3 | Simplification + correctness of docs | S | Prerequisite for 2's window; do it first if 2 starts first |
| 4 | Streaming coverage, single authority | S–M | Independent; 4.2 needs the bound oracle |
| 5 | Deletion | XS | Independent |
| 6 | Docs + one test | S | After 1 (uses its counter) |
| 7 | Upstream | — | Any time |

## Decisions

**Metrics engine (Phase 4.4): `"auto"`** (user, 2026-10-08). The one collect
helper passes `"auto"`, which is streaming by default in 2.0 and follows
`pl.Config.set_engine_affinity`; `DetectionTable.collect(engine=)` defaults
to `"auto"` too. The no-fallback guards keep collecting with
`engine="streaming"`, so streaming coverage is unaffected.

## Reproducing the measurements

The scripts (scratch, not committed) build 360 random 256×256 PNGs, run
`Pipeline().source("image_bytes").resize(128,128).blur(sigma=2)` →
`sink("numpy")`, and time:
- eager fresh, then streaming, then eager again;
- `scan_parquet` with row groups of 360, 240 and 180, three runs each in one
  process.

They also log morsel sizes with an elementwise `map_batches` probe, and check
the fallback rewrites with `tests/_streaming_guard.in_memory_nodes`. Phase 1
turns the timing cases into committed benchmark scenarios.
