# Performance work: handover

State of the kernel-performance effort planned in
[`PERFORMANCE_PLAN.md`](PERFORMANCE_PLAN.md), for whoever picks it up next.
Read this, then the plan's phase you are starting, then `CLAUDE.md`'s
Working Agreements (they bind every change here).

Branch: `claude/performance-optimization-handover-97vxoc` (Phases 1–5 were on
`claude/codebase-performance-assessment-36x2p3`, which it continues). Last code
commit: `f0428bb` (Phase 6). Everything is committed and pushed; there is no
work in progress.

## Where it stands

| phase | status | commits | report | finding |
|---|---|---|---|---|
| 0: kernel benchmarks | done | `0aab4db` | `2026-09-27-kernel-baseline/` | — |
| 1: CPU dispatch, conversion rule, grayscale/threshold | done | `bdce7ee` | `2026-09-27-phase1-dispatch/` | CR-54 (filed as a duplicate CR-50, renumbered), CR-51 |
| 2: element-wise engine | done | `5fc5fea` | `2026-09-27-phase2-elementwise/` | CR-52 |
| CR-53: integer `invert` keeps its dtype | done | `0ecfe8f` | — | CR-53 |
| 3: strided walk | done | `bf64725`; tiling `80d3b52`, reverted in `c4475b6`; docs in `6c7bc8c` | `2026-09-28-phase3-strided/` | CR-55 |
| 4: resize adapter (resize after crop/flip is zero-copy) | done | `72fd1d1` | `2026-09-28-phase4-resize/` | CR-56 |
| CR-57: grayscale reads crops/flips in place | done | `7134c75` | `2026-09-28-cr57-grayscale/` | CR-57 |
| 5: per-row executor overhead | done | `dbe23ee` | `2026-09-28-phase5-per-row/` | CR-58; CR-59 (blob bug, fixed) |
| CR-60: the plugin allocates through polars' allocator | done | `1e7ad74`, `f0428bb` | `2026-09-28-cr60-allocator/` | CR-60 |
| 6: `List` source zero-copy, raw/blob alignment | done | `f944fc9` | `2026-09-28-phase6-ingestion/` | CR-61 (silent 0 on cast, fixed) |
| 7: rotation / affine | **next** | | | |
| 8: morphology iterations, blur input conversion | not started | | | |
| 9: JPEG encoder (eval-gated), f16 sink | not started | | | |

Reports live under `polars-cv/benchmarks/reports/`; findings are in
`CODE_REVIEW_FINDINGS.md` under "Performance review (2026-09-27)". The next free
finding id is **CR-62**; check with `grep -o '^### CR-[0-9]*' CODE_REVIEW_FINDINGS.md | sort -t- -k2 -n | tail -1`
before filing one. CR-50 was duplicated once.

## What exists now (the mechanisms later phases must use)

Each is a canonical path with a row in `AGENTS.md`'s Canonical Paths table. Do
not write a second one beside it; extend it.

- **`view-buffer/src/core/dispatch.rs`** (`SimdKernel` + `dispatch`,
  `SimdKernelMut` + `dispatch_mut`): the only way a kernel gets an AVX2 build.
  - It compiles the kernel's whole `#[inline(always)]` body with AVX2, never FMA.
  - In a debug build it runs the portable build on a helper thread and asserts
    byte-identical output. It runs on a helper thread so its allocations don't
    count against the `copy_counts` allocation guards.
  - A kernel must not hand its loop to a non-inlined callee, such as a closure
    that stays out of line or a `thread_local!` `with`.
- **`view-buffer/src/core/convert.rs`** (`CastFrom`, `convert_slice`,
  `convert_view`): the one element-conversion rule.
  - Integer sources use `as`.
  - Float → integer rounds half away from zero, then saturates.
  - 8/16-bit targets use the vectorisable `round_narrow!` form.
- **`with_dtype!`** (`core/dtype.rs`): the one runtime `DType` → type match.
- **`view-buffer/src/ops/elementwise/`**: every per-value compute op.
  - `lower_to_scalars` is shared with fusion.
  - `strategy` picks, in order: integer affine, lookup table (only for `powf` or
    per-channel work), blocked, or f32 pass.
  - `unique_contiguous_mut` decides whether a buffer is written in place.
  - `legacy.rs` is the verbatim pre-engine test oracle. Leave it unchanged.
- **`view-buffer/src/core/strided.rs`** (`Walk`): the one walk over a view's
  memory.
  - `copy_to` sits behind `to_contiguous`/`append_to`/`write_to`.
  - `for_each_run` sits behind `convert_view`.
  - A debug-build bound check guards every walk.
- **`ViewBuffer::dense_rows`**: row-wise kernels read crops and vertical flips
  where they lie.
- **`view-buffer/src/interop/fir.rs`** (`FirViewAdapter<P>`): the one way a
  buffer reaches fast_image_resize. `resize_pixels::<P>` (`runner.rs`) is the
  one resize kernel; it packs only a layout the adapter refuses.
- **`core::layout::{Dims, Strides}`** (inline up to rank 4) and
  `is_c_contiguous`: cloning a buffer or asking its layout a question
  allocates nothing. Don't reintroduce a `Vec` copy of a shape or strides on
  a per-row path; `layout_bookkeeping_allocates_nothing` pins it.
- **`ExecutionPlan::execute_steps`** replays a cached `Arc<[PlanStep]>`.
- **Row results reach the column builder as per-range parts** (`RowParts`,
  `decode.rs`); never concatenate them (a `RowResult` is 168 bytes).
- **The global allocator is polars'** (`polars-cv/src/allocator.rs`).
  `_lib.__allocator__` must read `"polars"` (`tests/test_allocator.py`).
  Benchmark A/Bs of the plugin now measure jemalloc, not glibc.
- **`list_row_grid`** (`graph/decode.rs`) is the one way a `List`/`Array` row
  becomes a grid; **`decode_binary_row`** the one binary-row decode, with
  `get_binary_row_buffer`'s `in_place` predicate deciding copy vs view.
- **`benchmarks/ingestion_overhead.py`**: per-row source cost, `array` as the
  in-place reference.
- **`MemoryEffect` is what makes a planned pipeline pack.** A kernel that reads
  views is not enough: `build_plan` inserts `MaterializeContiguous` before any
  op declaring `RequiresContiguous`. Check the plan's steps
  (`expr.plan().steps`), not just the kernel, when claiming "read in place".

The plan text predates these. Where it names `execution/dispatch.rs`,
`simd_dispatch!`, `ops/elementwise.rs`, `RowWalk` or `f32_slice_to`, read them
as the modules above. Line numbers in the plan (`runner.rs:921`, …) are stale;
grep for the symbol.

## Decisions the user made (do not relitigate)

- **Bit-identical output is the default.** The allowed deliberate changes are:
  - z-score statistics exact (done);
  - contrast mean exact (done, and it turned out bit-identical anyway);
  - JPEG bytes, only if Phase 9's encoder passes its eval gate (≥ 1.5× geomean
    speedup, PSNR within 0.5 dB). In that case IJG is added to `deny.toml`, and
    only then.

  Each change gets a CHANGELOG entry.
- **Integer `invert` is `MAX + MIN − x`** (= `!x`) in the input dtype (CR-53).
- **Transpose tiling was tried and reverted.** It made u8 transpose 1.7× slower
  at 1024². Transpose remains ~8× a vertical flip (2.5 vs 0.3 ms); CR-55 lists a
  small-unit transpose kernel as an unplanned follow-up. Don't reintroduce tiling
  without an interleaved A/B that beats `bf64725`'s walk.
- **The user's preferences:** TDD (watch every new guard fail for the reason it
  claims), `uv`, and Polars rather than pandas. Ask before a design choice that
  changes output or scope.

## How to work here (lessons that cost time)

**Build and verify:**
- The Rust MSRV is 1.96: run `rustc --version`, and `rustup update stable` if
  older. Without the toolchain the `.so` never builds and the Python plugin tests
  self-skip, which reads as green.
- Run cargo through `scripts/with-pyo3-env.sh cargo …`.
- Build the extension with
  `cd polars-cv && ../scripts/with-pyo3-env.sh .venv/bin/maturin develop`
  (debug build only).
- The gate is `scripts/verify.sh --fast` from the repo root. Read its final
  PASS/FAIL line, never a filtered view.

**Committing:**
- The pre-commit hook's `test_version_consistency` needs the `.so` built from
  exactly the tree being committed. Run `maturin develop` with nothing
  uncommitted except what you are committing, then commit.
- The end-of-file hook rewrites criterion `.txt` output on the first commit
  attempt. `git add -A` and commit again.

**Test on the wheels' target too.** Local builds target `x86-64-v3`, where the
dispatch parity check compares two identical builds. Also run:
```
CARGO_TARGET_DIR=$SCRATCH/target-x86-64 \
RUSTFLAGS="-C target-cpu=x86-64 -C link-arg=-fuse-ld=lld" \
scripts/with-pyo3-env.sh cargo test -p view-buffer --all-features
```
(`$SCRATCH` is your session's scratchpad; the target dir is ~4.5 GB, and the
container has ~4.5 GB free. Delete it when done.)

**Benchmarking** (`view-buffer/benches/kernels.rs`; add a phase's cases to it
before building either side):
1. Put the base in a worktree (`git worktree add $SCRATCH/base <sha>`) and copy
   the new bench file into it.
2. Build each side with
   `cargo clean -q -p view-buffer --profile benchmark && cargo bench --no-run -p view-buffer --all-features --bench kernels --profile benchmark`
   (through `with-pyo3-env.sh`), and copy the newest `target/benchmark/deps/kernels-*`
   binary aside. **Always `cargo clean -p view-buffer` first and `cmp` base vs head**:
   the cargo fingerprint shares artifacts across worktrees and once handed back
   identical binaries.
3. Run base/head alternately for three rounds per target, filtered with
   `--bench '<regex>'`, and report medians.
4. Never trust a single round. The tiling was nearly kept on a one-round number
   the interleaved runs did not reproduce (4.3 vs 2.35 ms).
5. Keep the CPU otherwise idle while benchmarks run. To commit mid-run, pause
   the bench process with `kill -STOP`, then `kill -CONT` it.

**Guards:**
- Watch every guard fail against a deliberate mutation of the code it covers.
  Distinguish mutations that only lose speed, which fixture tables catch, from
  ones that corrupt output, which property tests catch.
- `copy_counts.rs` counts allocations at least the test view's size, and those
  views are tiny. A growing `Vec::new()` + `push` in a hot helper trips it, so
  size vectors exactly.

**Shell traps:**
- `pkill -f <pattern>` matches your own shell when the pattern is in the
  command line. Kill by PID.
- An unset variable in a path writes to `/`, and the safety check then refuses
  to remove the file. Quote and check variables.
- A stray `/p3-final-table.md` from this mistake may still exist. The user was
  asked to delete it; it is not needed.

## Known open items outside the plan

- **view-buffer doesn't build without the `image_interop` feature**, and it
  didn't at the base either (`separable_gaussian_blur_body` is ungated). CI and
  `verify.sh` build only `--all-features`. This hasn't been filed as a finding.
- **u8 preset normalize is slower in the v3 build than on the wheel target**
  (4.0 vs 2.2 ms at 1024², Phase 2 report). Not investigated.
- **Transpose is still ~8× a vertical flip** (CR-55 follow-up).
- **Eager calls scale poorly across threads**: an 8×8 `invert` over 200k rows
  in one chunk is 1.7x faster on 4 threads than on 1, and the same rows split
  into 100 chunks run ~2x faster than one chunk. Some per-call work is serial
  (the column build is the first suspect). Not investigated.
- **Resize of more than 4 channels panics at run time** (as it did before
  Phase 4): fast_image_resize has no such pixel type and resize's `check()`
  accepts any channel count. Not filed.

- **A converting `list` row goes through polars' `strict_cast`** (Phase 6):
  15.8 µs per 64×64 `i64 → u8` row against 2.2 for an in-place one, most of
  it the Series wrapper and cast. Fine for a path that exists for
  convenience; a typed pass would need its own range checks, a second copy of
  polars' cast rules. Not worth it without a user asking.

## Starting Phase 7

Plan: `PERFORMANCE_PLAN.md`, "Phase 7 — Rotation / affine". Its line numbers
are stale; grep for `affine_warp_typed` and `Rotation::Identity`. It is a
kernel phase, so the kernel benchmarks (`benchmarks/regression`, the
`rotate`/`affine` cases) are the A/B, not the plugin micro-benchmarks. Measure
first, as Phases 4–6 did: each found its real cost somewhere other than the
plan's first guess.

**Profiling the plugin** (what worked in Phase 5):
- There is no `perf`, and `py-spy` sees only threads with Python state (the
  plugin runs on polars' pool). Use callgrind.
- The release and benchmark profiles strip symbols (`strip = true`). Build an
  unstripped copy in its own target dir so the benchmark cache survives:
  `CARGO_TARGET_DIR=$SCRATCH/target-prof CARGO_PROFILE_BENCHMARK_STRIP=false
  CARGO_PROFILE_BENCHMARK_DEBUG=line-tables-only scripts/with-pyo3-env.sh cargo
  build -p polars-cv --lib --profile benchmark --features pyo3-extension`
  (~11 min cold, ~4 incremental, ~5 GB), and copy `libpolars_cv.so` over
  `polars-cv/python/polars_cv/_lib.abi3.so`.
- `POLARS_MAX_THREADS=1 valgrind --tool=callgrind --toggle-collect='*execute_rows*'
  .venv/bin/python script.py`, then `callgrind_annotate --inclusive=yes` and
  `--tree=caller` on `malloc` for who allocates. Use ~20k small rows.
- Callgrind counts instructions: it cannot see lock contention, page faults
  or allocator trimming. Check those with timings (`POLARS_MAX_THREADS=1` vs
  4) and `resource.getrusage` page-fault counts.

**Plugin A/B:** build each side with `maturin develop --profile benchmark` in
the main tree, copy `_lib.abi3.so` aside after each, and swap them between
alternating rounds of `benchmarks/plugin_overhead.py` (`--size`, `--only`).
The Python side must be identical on both sides.

**Disk:** the session allowance is ~38 GB. `target/debug` grew to 20 GB with
stale flag variants and filled it twice. `rm -rf target/debug/incremental`
is always safe; a whole `target/debug` is only build cache. Never build
`-p polars-cv --all-features`: it is a different feature set and rebuilds the
whole polars stack in debug.

Lessons from Phases 4–6 that still apply:
- **Verify with nothing unstaged or untracked.** `verify.sh` passed for the
  CR-60 commit while its new `allocator.rs` was untracked, so the relevance
  guard (which reads tracked files) never saw it; the next run failed. And
  pre-commit stashes unstaged changes, so committing part of a tree whose
  Rust changed fails the source-hash check: commit in an order that leaves
  nothing unstaged.
- **A parity test needs an oracle outside the code under test.** Comparing
  the new path with "the same op on packed input" was blind to both mutations
  tried, because the packed input went through the new adapter too. Compare
  against something that shares no code with the change.
- **This container's benchmark noise is about ±5% between rounds** (±10% for
  the 64×64 plugin cases). Confirm a one-sided shift over more rounds, and
  single-threaded, before acting on it.
- The base worktree has no `.venv`: run the main repo's
  `scripts/with-pyo3-env.sh` by absolute path from inside it.
