# Performance work: handover

State of the kernel-performance effort planned in
[`PERFORMANCE_PLAN.md`](PERFORMANCE_PLAN.md), for whoever picks it up next.
Read this, then the plan's Phase 9, then `CLAUDE.md`'s Working Agreements
(they bind every change here).

Branch: `claude/performance-optimization-handover-97vxoc` (Phases 1–5 were on
`claude/codebase-performance-assessment-36x2p3`, which it continues). Last code
commit: `4cfd1cd` (Phase 8). Everything is committed and pushed; there is no
work in progress. **What is left: Phase 9 (below) and the open items after it.**

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
| 7: rotation / affine | done | `060df58` | `2026-09-28-phase7-warp/` | CR-62 (u64 max stored as 0), CR-63 (0° smeared NaN) |
| 8: morphology iterations, blur input conversion | done | `4cfd1cd` | `2026-09-28-phase8-morph-blur/` | — |
| 9: JPEG encoder (eval-gated), f16 sink | **next** | | | |

Reports live under `polars-cv/benchmarks/reports/`; findings are in
`CODE_REVIEW_FINDINGS.md` under "Performance review (2026-09-27)". The next free
finding id is **CR-64**; check with `grep -o '^### CR-[0-9]*' CODE_REVIEW_FINDINGS.md | sort -t- -k2 -n | tail -1`
before filing one. CR-50 was duplicated once.

## Phase 9: sinks (JPEG encoder, f16)

Plan: `PERFORMANCE_PLAN.md`, "Phase 9 — Sinks". It has two independent halves;
do the f16 sink first (small, certain) and the JPEG eval second (it may end in
"not adopted", which is a valid outcome to record).

### f16 sink

- **Where:** `NumpyRowOutput::from_buffer_f16` (`polars-cv/src/output.rs`),
  behind `.sink("numpy"|"torch", dtype="f16")`. Today it is
  `buffer.cast(F32)` → `to_contiguous()` → one `half::f16::from_f32(v)` and
  `extend_from_slice` per element. `half = "2"` is already a polars-cv
  dependency.
- **Plan:** walk the source once (M3, `core::strided::Walk` — or, simpler,
  `convert_view`/`convert_slice` into an f32 row scratch) and convert each row
  with `half::slice::HalfFloatSliceExt::convert_from_f32_slice` straight into
  the output bytes.
- **Test first:** bit-identical to the per-element `f16::from_f32` over every
  special value (NaN payloads, ±0, ±inf, subnormals, values that round to
  f16's max and overflow to inf, halfway cases) and over strided inputs
  (transposed, flipped, cropped). Check that `half`'s bulk conversion uses the
  same rounding (round-to-nearest-even) as `from_f32` on this target — it may
  use F16C instructions when available; that is why the test must cover
  halfway and NaN cases. Watch it fail against a deliberately wrong
  conversion (e.g. truncating).
- **Measure first:** there is no f16 case in any benchmark. Add one (the
  plugin micro-benchmarks, `benchmarks/plugin_overhead.py`, run through
  `sink("numpy", dtype="f16")` on 64×64 and 512×512) and time it before
  changing anything; a per-element loop over 2-byte outputs may or may not
  matter next to the rest of the sink.

### JPEG encoder (eval-gated)

- **Gate first:** `view-buffer/tests/jpeg_encode_eval.rs`, mirroring
  `view-buffer/tests/png_decode_eval.rs` (read it: a synthesized corpus, a
  parity test that always runs, and an `#[ignore]`d timing test run with
  `--ignored --nocapture` in release).
  - Compare `image::codecs::jpeg::JpegEncoder` (production) with
    `jpeg-encoder` (`features = ["simd"]`; the crates.io index was reachable
    from this container) on gradient/noise × 256²/512²/1024² × L8/Rgb8, at
    quality 75 and 90.
  - Parity: decode both outputs back and require PSNR against the source
    within 0.5 dB of each other. This always runs.
  - **Adopt only if the geomean speedup is ≥ 1.5×.** Otherwise stop, record
    the table in a report and in the plan, and leave the encoder alone.
- **If adopted:**
  - The dependency goes in `view-buffer/Cargo.toml` under `image_interop`.
  - `jpeg-encoder` is IJG-licensed: add `IJG` to `deny.toml`'s `allow` list
    with a comment, and only then (the user's decision, below). Run
    `cargo deny check` (part of `verify.sh`).
  - **The plan's description of `encode_jpeg` is out of date.**
    `ImageAdapter::encode_jpeg` (`view-buffer/src/interop/image.rs`) goes
    through `encode_in_place`, which **packs the buffer** (`to_contiguous()`)
    and hands `native` a contiguous byte slice and colour type; if the encoder
    refuses the colour type (`UnsupportedErrorKind::Color`, e.g. alpha), the
    `converted` closure runs on a `DynamicImage` instead. There is no
    `JpegViewAdapter`. Keep that structure: the new encoder plugs in as the
    `native` closure for `L8`/`Rgb8`, and anything it cannot take falls to the
    existing conversion path. Reading strided views without packing would be
    a separate change (see `ViewBuffer::dense_rows`), not part of the swap.
  - `ImageCodec::check_support` stays the one authority on what is encodable
    (JPEG's 65,535 limit already holds there).
  - Output bytes change: that is a user-visible, CHANGELOG'd change. A grep
    of the tests found **no test pinning exact JPEG bytes** (no hash, golden
    or equality check on encoder output); search again before switching, and
    any pixel-tolerance JPEG test must pass unchanged.

## Remaining open items (outside the plan)

None of these are scheduled. Each says what is known and what to do next; the
ones marked **ask** change behaviour or scope and need the user's go-ahead.

1. **Eager calls scale poorly across threads.** An 8×8 `invert` over 200k rows
   in one chunk is only 1.7× faster on 4 threads than on 1, and the same rows
   split into 100 chunks run ~2× faster than one chunk. Some per-call work is
   serial. First suspect: the column build after the rows are computed
   (`build_series_from_spec`, `graph/decode.rs`; `fill_rows` already runs on
   the pool, but the rest does not). Profile with callgrind
   (`--toggle-collect='*execute_rows*'` excludes it; collect on
   `*vb_graph*` instead) and time 1 vs 4 threads. Likely the biggest
   remaining win for small-image workloads.
2. **Transpose is ~8× a vertical flip** (2.5 vs 0.3 ms at 1024² u8; CR-55
   follow-up). Tiling was tried in Phase 3 and made it 1.7× *slower*
   (reverted). A small-unit transpose kernel (e.g. 8×8 byte blocks with SIMD
   shuffles) is the untried idea. Needs an interleaved A/B against `bf64725`.
3. **u8 preset normalize is slower in the v3 build than on the wheel target**
   (4.0 vs 2.2 ms at 1024², Phase 2 report). Not investigated; start by
   comparing the generated code of the two builds (`objdump`, count `ymm`), as
   Phase 7 did for the warp.
4. **Resize of more than 4 channels panics at run time.** fast_image_resize
   has no such pixel type, and resize's `check()` accepts any channel count
   (as it did before Phase 4). A bug, not filed. The fix belongs in the op's
   contract (`check()` refuses it at plan time, per "a bypass must fail"), or
   in a per-channel-group resize. **Ask** which.
5. **view-buffer does not build without the `image_interop` feature**
   (`separable_gaussian_blur_body` and friends are used ungated). CI and
   `verify.sh` build only `--all-features`, so nothing notices. Not filed.
   Either gate the users or make the feature mandatory; add a
   `cargo check -p view-buffer --no-default-features` to `verify.sh` so it
   stays fixed.
6. **Typed grayscale still stores with `clamp_for_dtype` + `NumCast`**
   (`luma_typed`, `runner.rs`). For u64/i64 that pattern clamps one past the
   range and stores 0 (CR-62), but grayscale's weights sum below 1, so no
   input reaches it (a white u64 image stays white). The blur's store moved
   to M5 in Phase 8. Moving grayscale too would leave `clamp_for_dtype` only
   in the warp's test oracle; do it only with a benchmark, since the M5 form
   can vectorise differently (Phase 7's RGBA lesson below).
7. **The warp's border fill converts with `NumCast`**, truncating: a u8
   `border_value=7.5` fills 7 but blends toward 7.5 at the edge, and an
   out-of-range value fills 0. Inconsistent with M5, left as it was, not
   filed. Changing it changes output: **ask**.
8. **A converting `list` row goes through polars' `strict_cast`** (Phase 6):
   15.8 µs per 64×64 `i64 → u8` row against 2.2 for an in-place one. It is a
   convenience path, and a typed pass would need a second copy of polars'
   cast rules. Not worth doing without a user asking.

## What exists now (the mechanisms later work must use)

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
  `convert_view`): the one element-conversion rule (M5).
  - Integer sources use `as`.
  - Float → integer rounds half away from zero, then saturates.
  - f32 → 8/16-bit uses `round_narrow!` (vectorises in bulk casts); f64 →
    8/16-bit uses `round().clamp() as T` (vectorises across a pixel's
    channels; `round_narrow` did not, RGBA warp 1.8× slower). Same values,
    pinned by `narrow_conversion_equals_round_then_saturate`.
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
- **`execution/warp.rs`** is the affine warp. Its tests keep the pre-Phase-7
  kernel verbatim as a byte-parity oracle over every dtype, channel count,
  angle, matrix and non-finite input: any change to the warp must keep them
  byte-identical, or change the oracle deliberately and say why.
- **Morphology iterations** (`morph_iterated`, `runner.rs`): one pass of
  radius `n * (k / 2)` for integer dtypes, `n` passes for floats (NaN and
  signed-zero handling depends on pass structure). `tests/morph_ref.rs`'s
  naive iterated reference holds both.
- **The blur** (`separable_gaussian_blur_body`) converts input a row at a
  time and stores through M5; its tests keep the pre-Phase-8 body verbatim as
  a byte-parity oracle (`blur_matches_the_reference_kernel_bit_for_bit`).
- **Benchmarks:** `view-buffer/benches/kernels.rs` (kernels; rotate, blur and
  morphology matrices added in Phases 7–8), `benchmarks/plugin_overhead.py`
  (per-row executor cost), `benchmarks/ingestion_overhead.py` (per-row source
  cost, `array` as the in-place reference).
- **`MemoryEffect` is what makes a planned pipeline pack.** A kernel that reads
  views is not enough: `build_plan` inserts `MaterializeContiguous` before any
  op declaring `RequiresContiguous`. Check the plan's steps
  (`expr.plan().steps`), not just the kernel, when claiming "read in place".

The plan text predates these. Where it names `execution/dispatch.rs`,
`simd_dispatch!`, `ops/elementwise.rs`, `RowWalk`, `f32_slice_to` or
`JpegViewAdapter`, read them as the modules above (the last does not exist).
Line numbers in the plan (`runner.rs:921`, …) are stale; grep for the symbol.

## Decisions the user made (do not relitigate)

- **Bit-identical output is the default.** The allowed deliberate changes are:
  - z-score statistics exact (done);
  - contrast mean exact (done, and it turned out bit-identical anyway);
  - JPEG bytes, only if Phase 9's encoder passes its eval gate (≥ 1.5× geomean
    speedup, PSNR within 0.5 dB). In that case IJG is added to `deny.toml`, and
    only then.

  Each change gets a CHANGELOG entry. Bug fixes found on the way (CR-59,
  CR-61, CR-62, CR-63) changed output where the old output was wrong; each is
  a finding, a CHANGELOG "Fixed" entry and a guard.
- **Integer `invert` is `MAX + MIN − x`** (= `!x`) in the input dtype (CR-53).
- **Transpose tiling was tried and reverted** (see open item 2).
- **The user's preferences:** TDD (watch every new guard fail for the reason it
  claims), `uv`, and Polars rather than pandas. Ask before a design choice that
  changes output or scope.

## How to work here (lessons that cost time)

**Measure first.** Every phase from 4 on found its real cost somewhere other
than the plan's first guess (Phase 5: layout bookkeeping, not the plan cache;
Phase 6: `List` decode, not alignment; Phase 7: unsigned conversions, not
bounds checks). Profile, then change.

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
- **Verify with nothing unstaged or untracked** (`git add -A` first).
  `verify.sh` passed for the CR-60 commit while its new `allocator.rs` was
  untracked, so the relevance guard (which reads tracked files) never saw it.
- **Test on the wheels' target too.** Local builds target `x86-64-v3`, where
  the dispatch parity check compares two identical builds. Also run, with
  nothing else building:
  ```
  CARGO_TARGET_DIR=$SCRATCH/target-x86-64 \
  RUSTFLAGS="-C target-cpu=x86-64 -C link-arg=-fuse-ld=lld" \
  scripts/with-pyo3-env.sh cargo test -p view-buffer --all-features
  ```
  (`$SCRATCH` is your session's scratchpad; ~4.5 GB; delete it when done.)

**Committing:**
- The pre-commit hook's `test_version_consistency` needs the `.so` built from
  exactly the tree being committed, and pre-commit stashes unstaged changes:
  commit in an order that leaves nothing unstaged.
- The end-of-file hook rewrites criterion `.txt` output on the first commit
  attempt. `git add -A` and commit again.

**Kernel benchmarks** (`view-buffer/benches/kernels.rs`):
- Add the phase's cases first, run them on the base, then on the head, in the
  same tree (`cargo bench … --profile benchmark --bench kernels -- '<regex>'`),
  and compare criterion medians. For a close call, interleave base and head
  over several rounds (a base worktree with the new bench file copied in,
  `cargo clean -p view-buffer` before each side and `cmp` the two binaries:
  cargo's fingerprint once handed back identical ones).
- **Nothing else may run while benchmarking**: a concurrent test build cost
  Phase 7's head run ~10%. Put background jobs after the bench, not beside it.
- Noise is about ±5% between rounds (±10% for the 64×64 plugin cases).
  Confirm a one-sided shift over more rounds, and single-threaded, before
  acting on it.

**Profiling a kernel:** build the bench unstripped in its own target dir,
`CARGO_TARGET_DIR=$SCRATCH/tprof CARGO_PROFILE_BENCHMARK_STRIP=false
CARGO_PROFILE_BENCHMARK_DEBUG=line-tables-only scripts/with-pyo3-env.sh cargo
bench -p view-buffer --all-features --profile benchmark --bench kernels
--no-run`, then from `view-buffer/`: `valgrind --tool=callgrind <bench binary>
--bench <case> --profile-time 3`, and `callgrind_annotate --auto=yes` for
per-line counts. `objdump -d -C` of the `run_avx2::<…>` function shows whether
a loop vectorised (count `ymm`).

**Profiling the plugin:**
- There is no `perf`, and `py-spy` sees only threads with Python state (the
  plugin runs on polars' pool). Use callgrind.
- The release and benchmark profiles strip symbols. Build an unstripped copy
  in its own target dir: `CARGO_TARGET_DIR=$SCRATCH/target-prof
  CARGO_PROFILE_BENCHMARK_STRIP=false CARGO_PROFILE_BENCHMARK_DEBUG=line-tables-only
  scripts/with-pyo3-env.sh cargo build -p polars-cv --lib --profile benchmark
  --features pyo3-extension` (~11 min cold, ~5 GB), and copy
  `libpolars_cv.so` over `polars-cv/python/polars_cv/_lib.abi3.so`.
- `POLARS_MAX_THREADS=1 valgrind --tool=callgrind --toggle-collect='*execute_rows*'
  .venv/bin/python script.py`, then `callgrind_annotate --inclusive=yes` and
  `--tree=caller` on `malloc` for who allocates. Use ~20k small rows.
- Callgrind counts instructions: it cannot see lock contention, page faults
  or allocator trimming. Check those with timings (`POLARS_MAX_THREADS=1` vs
  4) and `resource.getrusage` page-fault counts.
- **Plugin A/B:** build each side with `maturin develop --profile benchmark`,
  copy `_lib.abi3.so` aside after each, and swap them between alternating
  rounds of `benchmarks/plugin_overhead.py` (`--size`, `--only`). The Python
  side must be identical on both sides. Rebuild debug (`maturin develop`)
  before running tests again.

**Guards:**
- Watch every guard fail against a deliberate mutation of the code it covers.
- **A parity test needs an oracle outside the code under test.** Phases 7–8
  kept the old kernel verbatim in the tests; `morph_ref.rs` uses a naive
  reference. Comparing the new path with "the same op on packed input" was
  blind to both mutations tried in Phase 4, because the packed input went
  through the new adapter too.
- A guard that never failed proves nothing: Phase 7 removed a u64 grayscale/
  blur "fix" test because no input reached the bug.
- `copy_counts.rs` counts allocations at least the test view's size, and those
  views are tiny. A growing `Vec::new()` + `push` in a hot helper trips it, so
  size vectors exactly.

**Code-generation traps:**
- **Unsigned 64-bit ↔ f64 conversions are multi-instruction on x86** (no
  AVX-512): `usize as f64` and `f64 as usize` in a per-pixel loop cost more
  than the arithmetic around them. Count in `f64`, convert through `i64`.
- **How a store is written decides whether a pixel's channels vectorise**
  (Phase 7): the same values, written two ways, measured 1.8× apart for RGBA.
  Check `ymm` counts when a store changes.

**Disk:** the session allowance is ~38 GB. `target/debug` grew to 20 GB with
stale flag variants and filled it twice. `rm -rf target/debug/incremental`
is always safe; a whole `target/debug` is only build cache. Never build
`-p polars-cv --all-features`: it is a different feature set and rebuilds the
whole polars stack in debug.

**Shell traps:**
- A variable set in a command chain that is sent to the background (`… &`) is
  not set in the commands after it: Phase 7 wrote a bench file to `/` that
  way. Set variables in each command, and quote and check them.
- `pkill -f <pattern>` matches your own shell when the pattern is in the
  command line. Kill by PID.
- A stray `/p3-final-table.md` from an earlier session may still exist. The
  user was asked to delete it; it is not needed.
- The base worktree has no `.venv`: run the main repo's
  `scripts/with-pyo3-env.sh` by absolute path from inside it.
