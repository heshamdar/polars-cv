# Open issues

The one ledger of known, unresolved work in polars-cv: defects, performance
gaps, pending decisions and issues to file upstream. It replaces
`CODE_REVIEW_FINDINGS.md`, `FOLLOW_UPS_0.32.md`, `PERFORMANCE_PLAN.md`,
`PERFORMANCE_HANDOVER.md`, `EXTENSION_TYPES_PLAN.md`, `POLARS_2_PLAN.md` and
`UPSTREAM_ISSUES.md` (consolidated 2026-10-08; every item below was re-checked
against `main` at `e788d72` that day, and their full text is in git history).

## How to use this file

- Every item has a stable id, `CR-NN`, continuing the old code-review series so
  ids already cited in code comments and tests keep resolving. A new item takes
  the next free number.
- A verified **defect** (behaviour the code should have and does not) is also
  pinned as an `xfail(strict=True)` in `polars-cv/tests/test_known_gaps.py`, so
  fixing it turns the suite red rather than passing unnoticed (root
  `AGENTS.md`, "Where deferred work is tracked").
- When an item closes, delete its section and add one line for it to
  [Closed findings](#closed-findings) with the outcome and the commit or
  release. The CHANGELOG says what changed for users.
- Items marked **decision** wait on the owner and should not be started
  without one.

## Summary

| ID | Area | Severity | Summary |
|----|------|----------|---------|
| CR-67 | Performance | Medium | Eager `list` sink 15.6% slower at 4 threads since the row-split budget (PR #124) |
| CR-68 | Performance | Medium | Streaming `sobel_x` 11.5% slower at 4 threads since the row-split budget (PR #124) |
| CR-69 | Performance | Medium | Eager calls scale poorly across threads (1.7× on 4 threads for small images) |
| CR-70 | Performance | Low | Transpose is ~8× a vertical flip at 1024² |
| CR-71 | Performance | Low | u8 preset `normalize` in the AVX2 (`x86-64-v3`) build: recheck the CR-64 specialisation |
| CR-72 | Performance | Low | The debug-build parity check spawns a thread per dispatched kernel call |
| CR-73 | Performance | Low | Code size: every element map is instantiated per dtype pair, twice |
| CR-74 | Fetch | Low | A fetched body outlives a row that never reads it (until the call ends) |
| CR-75 | Fetch | Low | `s3a://`, `gcs://`, `azure://`, `adl://` are not recognised as remote |
| CR-76 | Metrics | Medium | Negative-size boxes are accepted and silently score as false positives |
| CR-77 | Metrics | Low | Sweep thresholds compared by exact float equality (`map_75` disappears) |
| CR-78 | Metrics | Low | An image missing from `images=` silently gets weight 1.0 |
| CR-79 | CI | Low | The COCO parity test never runs in CI |
| CR-80 | Extension types | Feature | Opt-in tagged geometry output (`polars_cv.point` / `.contour` / `.bbox`) |
| CR-81 | Extension types | Decision | Whether any output becomes tagged by default (a major version) |
| CR-82 | Extension types | Decision | Whether `polars_cv.ndarray` carries `dtype`/`ndim` metadata |
| CR-83 | Upstream | To file | Streaming `group_by` cannot build a list |
| CR-84 | Upstream | To file | Feature: a plugin's preferred morsel size |
| CR-85 | Upstream | To file | A zero-width `Array` cannot cross the Arrow C interface |
| CR-86 | Upstream | To file | `MemoryManager::do_spill` can livelock under a small budget |
| CR-87 | Upstream | To file | `register_extension_type`'s duplicate check, and `ext_from_params` panics |

## Performance

### CR-67 — Eager `list` sink 15.6% slower at 4 threads since the row-split budget

- **Found:** post-merge assessment of PR #124 (2026-10-08), benchmark profile,
  4-vCPU container, base `f7364f6` vs head `e788d72`.
- **Evidence:** `targeted:sink_list_u8` at `--threads 4`: 1,392 → 1,174 img/s
  (median of 7 interleaved runs per side; no head run reached the base
  median). A direct probe agrees: 300 PNGs, eager, 173–200 ms per call on
  base against 232–258 ms on head. The `numpy` and fixed-shape `array` sinks
  show no consistent difference, and at 1 thread the list sink is +4.9%.
  Eager `invert`, `sobel_x` and `sharpen` drift −4.6 to −5.3% in the same
  runs, under the 7% gate, with +7–10% peak RSS.
- **Likely mechanism (unverified):** the old splitter ran every row range on
  the 4 pool threads with the caller blocked (`THREAD_POOL.scope`); the budget
  runs ranges on the calling thread plus up to 3 helpers (`in_place_scope`).
  The calling thread then builds the list sink's nested offsets alone, about
  19.7M lengths at 300×256×256×3, after producing a quarter of the rows, so
  its allocator and cache state differ from before. `_last_split_workers` also
  showed the list sink on 3 workers at times.
- **Next:** profile `sink("list")` eager at 4 threads on both sides (no
  profiler was available in the container). Candidates: keep the caller out
  of row work while keeping the budget, or build the offsets in one pass into
  an `i64` buffer rather than a `Vec<usize>` of lengths. A splitter change is
  now benchmarked with these cases at `@threads=4`
  (`benchmarks/regression/relevance.py`).

### CR-68 — Streaming `sobel_x` 11.5% slower at 4 threads since the row-split budget

- **Evidence:** as CR-67, `single_ops:sobel_x` streaming at `--threads 4`:
  12,818 → 11,346 img/s, median of 7 runs per side, no head run reaching the
  base median. Other streaming single ops moved the other way (`invert`,
  `sharpen` +4%).
- **Likely cause:** at 300 rows the streaming engine runs few morsels, so
  calls overlap only partly, and that is exactly where "invite idle threads
  while under budget" differs from the old all-or-nothing rule.
- **Next:** rerun at a larger count to see whether it persists once morsels
  saturate the pool; then profile as CR-67.

### CR-69 — Eager calls scale poorly across threads

- **Evidence (2026-09-28, before PR #124):** an 8×8 `invert` over 200k rows in
  one chunk was only 1.7× faster on 4 threads than on 1, and the same rows in
  100 chunks ran about 2× faster than one chunk, so some per-call work is
  serial. CR-65 moved one instance (the f16 sink's conversion) onto the row
  pool.
- **Next:** re-measure on the current splitter first. Then look for other
  O(pixels)-per-row work outside `fill_rows`/`split` in
  `build_series_from_spec` (`graph/decode.rs`) and `output.rs`. Profile with
  callgrind, collecting on `*vb_graph*`, and time 1 against 4 threads.

### CR-70 — Transpose is ~8× a vertical flip

- **Evidence:** 2.5 ms against 0.3 ms at 1024² u8 (follow-up of CR-55).
  A tiled transpose was tried (`80d3b52`, reverted in `c4475b6`): it made u8
  transpose 1.7× slower at 1024².
- **Next:** the untried idea is a small-unit kernel (8×8 byte blocks with
  SIMD shuffles), with an interleaved A/B against `bf64725`.

### CR-71 — Recheck u8 preset `normalize` in the AVX2 build

- The table map's inner loop over a run-time channel count compiled badly in
  the `x86-64-v3` build. CR-64 specialised it for 3 and 4 channels (1.5–1.9×
  on the wheels), but it was never re-measured on v3. Measure, then close.

### CR-72 — The debug parity check spawns a thread per dispatched call

- In a debug build `core::dispatch` runs the portable kernel on a helper
  thread and asserts byte-identical output; it uses a thread so its
  allocations stay out of the `copy_counts` guards. Every debug kernel call,
  the whole Python suite included, pays a spawn and a second run. A
  per-thread "diagnostic" flag the counting allocators skip would do the same
  without the spawn. Measure the suite's time first.

### CR-73 — Code size of the element maps

- Every map is instantiated per (source, destination) dtype pair, twice
  (portable and AVX2); the table strategy alone is 80 copies of about 2,600
  instructions. Nothing measured slower for it. A table lookup gains nothing
  from AVX2 (scalar loads), so it could skip dispatch if size ever matters.

## Fetch

### CR-74 — A fetched body outlives a row that never reads it

- `fetch::Fetcher` frees a body after the last row naming it has read it. A
  row that errors in an earlier source node under `on_error="null"`, in a
  graph with two path sources, never reaches the later source, so that
  body stays resident (and counted in `_last_fetch_peak_resident`) until the
  call ends. It is bounded by the call and freed when the fetcher drops, so
  this is memory only. Fix if a multi-source path pipeline ever shows it.

### CR-75 — Some remote schemes are not recognised as remote

- `cloud::is_remote_path` accepts `s3`, `gs`, `az`, `abfs`, `abfss`, `http`
  and `https`. `s3a://`, `gcs://`, `azure://` and `adl://` are read as local
  paths and fail as missing files, although polars-io can serve them. (Before
  0.34 a second scheme list in `cloud::read_file` named them, but that
  function was only reached for paths `is_remote_path` had already accepted,
  so they never worked.) Extend `is_remote_path`, the one list, with a test
  per scheme, or refuse them with a message naming the supported spellings.

## Metrics

### CR-76 — Negative-size boxes are accepted and silently score as false positives

(was FU-01) Pinned: `test_known_gaps.py::test_a_negative_size_box_is_refused`.

- **Location:** `geometry/coords.py::bbox_from_coords`; the bbox reader in
  `src/geom_columns.rs` (`BBoxColumn`).
- **Evidence:** `bbox_from_coords([[10, 10, 0, 0]], format="xyxy")` gives
  `{x: 10, y: 10, width: -10, height: -10}` with no error, and through
  `match_detections(..., box_format="xyxy")` it scores as a 0-IoU false
  positive. The usual cause is xywh data passed as `"xyxy"`, which the
  required `format=` exists to catch. The reader refuses non-finite fields
  but not negative sizes (re-checked 2026-10-08).
- **Fix:** have the reader refuse `width < 0` / `height < 0` as it refuses
  non-finite fields (decide whether zero is allowed), after checking that no
  internal producer relies on negative sizes. Add the error to the
  `geom_columns.rs` tests and a `match_detections` test with swapped corners.

### CR-77 — Sweep thresholds compared by exact float equality

(was FU-02) Pinned: `test_known_gaps.py::test_a_swept_threshold_reports_its_metrics`.

- **Location:** `metrics/_types.py` (`DetectionTable.at_iou_threshold`,
  `_slice`), `metrics/_reports.py` (`DetectionReport.metrics`, `per_class`).
- **Evidence (reproduced 2026-10-08):** `evaluate_detections(...,
  iou_thresholds=list(np.arange(0.5, 0.96, 0.05)))` reports
  `['map', 'map_50', 'mar']`; the same thresholds rounded to 2 decimals
  report `map_75` too, because 0.75 is stored as `0.7500000000000002`.
  `at_iou_threshold(0.55)` likewise re-thresholds the 0.5 matching instead of
  slicing the 0.55 one. `COCO_IOU_THRESHOLDS` is rounded, so the default path
  is unaffected.
- **Fix:** normalise thresholds once where a sweep is built (round to e.g. 10
  decimals in the matcher and `DetectionTable.stack`), not with tolerant
  comparisons at every reader. Test with `np.arange` thresholds.

### CR-78 — An image missing from `images=` silently gets weight 1.0

(was FU-03) **decision**

- **Location:** `metrics/_inputs.py` (`match_detections`, `_image_columns`);
  the policy in `metrics/_weights.py` (`fill_null(1.0)`).
- **Evidence:** with `images=` covering image `a` only, predictions on `b`
  read `b`'s weight as 1.0, so an incomplete weight table is
  indistinguishable from a deliberate unit weight. The `weight=` docstring
  does not say so (re-checked 2026-10-08).
- **Options:** (a) under `weight=`, require `images=` to cover every image
  and fail the query otherwise; (b) keep 1.0 and document it in the
  `weight=` docstrings of `match_detections` / `evaluate_detections`.

## CI

### CR-79 — The COCO parity test never runs in CI

(was FU-04)

- `tests/reference/test_coco_parity_ref.py` skips unless `pycocotools` is
  importable (`pytest.importorskip`), and nothing installs it, so the
  CHANGELOG's "agrees with COCOeval to 1e-12" is not checked by CI. Run by
  hand (`uv run --with pycocotools pytest tests/reference/test_coco_parity_ref.py`)
  it passed 4/4 at 0.32.0.
- **Fix:** add `pycocotools` to a dependency group the fast lane installs, or
  make the reference lane fail rather than skip when it is missing in CI.

## Extension types

Shipped in 0.29.0: `ExtType`, the four registered types, `sink("ndarray")`,
tagged inputs everywhere. How it works, and why tagging stays opt-in, is in
`polars-cv/python/polars_cv/AGENTS.md` ("Extension Types").

### CR-80 — Opt-in tagged geometry output

- Geometry-producing paths (contour extraction, bbox/point constructors) gain
  a tagged form. Prefer a distinct sink format or constructor over a global
  flag: a flag would reach every plugin call and enter op identity, and
  `CLAUDE.md` rejects per-call keywords that alter the output schema.
- Candidates: `sink("contour")` for tagged contours, and `polars_cv.point(x,
  y)` / `polars_cv.bbox(...)` constructors as zero-copy `.ext.to()` relabels.
  Users can already tag with `.ext.to(PointType())`.
- Tests mirror `tests/test_ndarray_sink.py`, plus metrics end to end on
  tagged inputs.

### CR-81 — Tagged by default? **decision**

- Decide from usage whether `sink("numpy")` and the untagged geometry outputs
  become tagged by default (a major version), or stay as the plain-struct
  interchange formats. Recommendation: keep both permanently, `numpy` plain
  and `ndarray` typed, which avoids a flag day. A flip needs CHANGELOG
  migration notes with the struct-operations table from the AGENTS.md
  section.

### CR-82 — `polars_cv.ndarray` metadata **decision**

- Should the type carry `{"dtype": "uint8", "ndim": 3}`? The planner knows
  both at plan time, enabling type-level checks before data flows. The cost:
  metadata is part of dtype equality, so columns of different dtypes stop
  concatenating, and the factory must parse and validate it.
  Recommendation: no metadata; adding it later is a dtype change, so decide
  before CR-81.

## Upstream (Polars)

Ready-to-file drafts for `pola-rs/polars`. CR-83 to CR-86 reproduce with
Polars 2.0.0 alone (polars-cv not imported), verified 2026-10-08 unless
noted; each says what polars-cv does meanwhile, so the workaround can be
removed once the issue is fixed.

### CR-83 — Streaming `group_by` cannot build a list (`implode`, implicit list, `head`)

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


### CR-84 — Feature: let a plugin declare a preferred morsel size in bytes

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


### CR-85 — A zero-width `Array` column cannot cross the Arrow C interface

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


### CR-86 — Out-of-core: `MemoryManager::do_spill` can livelock under a small budget

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

### CR-87 — Extension-type registration: duplicate check, and a panic

Not yet re-verified on 2.0.0; check before filing.

- `register_extension_type`'s duplicate check tests the literal
  `"ext_name"` rather than the name being registered.
- An `ext_from_params` that raises panics polars instead of surfacing the
  error.

## Decisions (do not relitigate)

Made by the owner; listed so they are not reopened by accident.

- **Bit-identical output is the default.** The allowed deliberate changes
  were: exact z-score statistics and contrast mean (done); the warp's border
  fill stored by the conversion rule (CR-66). Bug fixes that changed wrong
  output (CR-59, CR-61, CR-62, CR-63, CR-66) each got a finding, a CHANGELOG
  "Fixed" entry and a guard.
- **JPEG encoder: image's stays** (2026-09-28). `jpeg-encoder` passed the
  eval gate (1.62–1.65× geomean, ΔPSNR ≤ 0.04 dB) but writes 5–13% larger
  files on smooth content. IJG stays out of `deny.toml`. The eval is
  `view-buffer/tests/jpeg_encode_eval.rs`.
- **A resize of more than 4 channels is refused** by its contract, not split
  into channel groups (`TestResizeChannelLimit`).
- **Integer `invert` is `MAX + MIN − x`** (`!x`) in the input dtype (CR-53).
- **A converting `list` row goes through polars' `strict_cast`** (15.8 µs per
  64×64 `i64 → u8` row against 2.2 for an in-place one). It is a
  convenience path, and a typed pass would need a second copy of polars'
  cast rules: not without a user asking.
- **Metrics collect with `engine="auto"`** (2026-10-08): streaming by default
  in Polars 2.0, following `pl.Config.set_engine_affinity`;
  `tests/test_engine_choice.py` refuses a literal `engine=`.
- **Ordered windows (`over(order_by=)`) are not used in `_grouped_scan.py`**
  (measured 15–35% faster on the scan alone): `over` takes one `descending`
  flag for all keys, callers mix directions, and negating a key misplaces
  NaN. The reasons are in that module.

## Closed findings

One line per closed id, so an id cited in a comment, test or commit still
resolves. The full write-ups are in git history (`CODE_REVIEW_FINDINGS.md`
before 2026-10-08). CR-26 and CR-29 were never assigned.

| ID | Finding | Outcome |
|----|---------|---------|
| CR-01 | `ViewExpr::grayscale()` hardcodes `DType::U8`, producing silent wrong values | Resolved |
| CR-02 | The "known gaps" ledger drifted from its own prose | Resolved |
| CR-03 | `shear`/`rotate_and_scale` advertise unimplemented auto-sizing | Resolved |
| CR-04 | `OutputDTypeRule::Configurable` is declared but never emitted | Resolved |
| CR-05 | Metrics helpers documented as load-bearing are orphaned | Resolved |
| CR-06 | Two all-points-AP implementations kept in sync by hand | Resolved |
| CR-07 | `mean_average_precision` runs an eager Python loop | Resolved |
| CR-08 | `_rotation_matrix` reintroduces rotation trig in Python | Resolved |
| CR-09 | Duplicate integral families (eager vs expression) | Won't fix (premise corrected) |
| CR-10 | PNG-factory guard enforces a subset and is being evaded | Resolved |
| CR-11 | `cloud.rs` / `cloud_auth.rs` have no non-network CI coverage | Resolved (verified 2026-10-08: 27 non-ignored unit tests; `test_async_runtime.py`, `test_path_sandbox.py`, `test_fetch_window.py` read `s3://`/`http://` from local servers in the default lane) |
| CR-12 | Structural parity sweep self-skips without the `.so` | Resolved |
| CR-13 | `TypedBufferData::polars_dtype()` re-enumerated the `DType`→Polars mapping that `polars_dtype_for` owns. | Resolved |
| CR-14 | `GraphNode.alias` carried a false doc comment ("becomes the key in outputs map"). | Resolved |
| CR-15 | Stale TODO pointer at `polars-cv/src/graph/compiled.rs` to an already-shipped path sandbox. | Resolved |
| CR-16 | view-buffer dead code. | Resolved |
| CR-17 | `source()` `BLOB` and `AUTO` branches are byte-identical; collapse. | Resolved |
| CR-18 | `contours.py:376` `label_reduce(heatmap=)` back-compat alias with no caller; delete. | Resolved |
| CR-19 | `metrics/_matching/_contour.py:502` `match(score_col=)` accepted and ignored (Matcher-protocol… | Resolved |
| CR-20 | `geometry/schemas.py` factory helpers used only by tests; move to a test helper or re-export intentionally. | Resolved |
| CR-21 | `metrics/_metrics/_confusion.py` + `f1_at_threshold` issue several separate `.collect()`s instead of… | Resolved |
| CR-22 | ~6 test files use bare `pytest.raises(Exception)` without `match=` (e.g. `test_correctness_audit.py:981`… | Resolved |
| CR-23 | Stale docstring at `polars-cv/tests/test_typed_nodes.py:8` ("marked xfail until implementation complete" —… | Resolved |
| CR-24 | `optimize()` transpose-merge carries a self-admitted "prototype … slightly inaccurate" comment… | Resolved |
| CR-25 | Redundant `_ => None` catch-all after `Invert` in `working_dtype` let a new `ComputeOp` inherit `None`… | Resolved |
| CR-27 | Extend the single-metadata-authority collapse to `Compute` and `View` builders | Resolved |
| CR-28 | `ExprNode::Compute` holds `Vec<Arc<ViewExpr>>` but every site uses exactly one child; narrow to… | Resolved |
| CR-30 | All-points AP has no canonical tie convention, so its two implementations (scalar `_all_points_ap`,… | Resolved |
| CR-31 | Expression params are identified by `str(expr)`, which is not an identity | Resolved |
| CR-32 | Multi-core execution depends on how Polars happens to chunk the input | Resolved |
| CR-33 | `list` sinks of rank ≥ 2, and `array` sinks with any null row, build one `AnyValue` per element | Resolved |
| CR-34 | Panics are the engine's error channel, so `on_error` cannot cover them | Resolved |
| CR-35 | Published wheels ship without the SIMD code paths | Resolved |
| CR-36 | Geometry I/O goes point by point through `AnyValue` | Resolved |
| CR-37 | Per-row executor overhead from stringly-typed dispatch | Resolved |
| CR-38 | Row/sink kind mismatch silently becomes a null row | Resolved |
| CR-39 | A null row in the numpy/ndarray sink is a struct of nulls, not a null | Resolved |
| CR-40 | The "zero-copy" `array` source copied the whole column on every row | Resolved |
| CR-41 | A misaligned blob reaches `slice::from_raw_parts` in release builds | Resolved |
| CR-42 | `crop` and the `raw` source silently accept out-of-range input | Resolved |
| CR-43 | `uv run` builds the release-LTO extension | Resolved |
| CR-44 | Declared dependencies do not describe what the package needs | Resolved |
| CR-45 | Ops cross the boundary as a name plus an untyped param map | Resolved |
| CR-46 | The planner is split across the FFI and folded twice | Resolved |
| CR-47 | Source/sink params are policed by applicability tables | Resolved |
| CR-48 | Geometry accessors carry a second per-row parameter mechanism | Resolved |
| CR-49 | Plan-time shapes are inferred by probing four magic values | Resolved |
| CR-50 | Graph node ids are random, so equal pipelines never share a compiled graph | Resolved in 0.34 (`_to_dict` names nodes by position; verified 2026-10-08: two separately built pipelines are `meta.eq`) |
| CR-51 | Non-u8 gray + alpha grayscale mixed alpha into the intensity | Resolved |
| CR-52 | Per-value ops ran through five near-duplicate paths, each with a full f32 copy | Resolved |
| CR-53 | `invert` on other integer dtypes returns f32 against a `PreserveInput` contract | Resolved |
| CR-54 | Wheels run most kernels without SIMD; float → int casts call `roundf` per element | Resolved |
| CR-55 | Materialising a strided view copied a few bytes at a time | Resolved |
| CR-56 | Resize packed every crop and vertical flip into a new image first | Resolved |
| CR-57 | The planner packed a crop or vertical flip before grayscale, which reads it in place | Resolved |
| CR-58 | A cheap row spent most of its time allocating layout copies | Resolved |
| CR-59 | The blob of a contiguous part of a buffer carried the whole buffer | Resolved |
| CR-60 | The plugin allocated with the system `malloc`, not polars' allocator | Resolved |
| CR-61 | A `list` value the declared dtype cannot hold became 0 | Resolved |
| CR-62 | The affine warp stored a 64-bit maximum as 0 | Resolved |
| CR-63 | `rotate(0)` spread NaN and infinity into neighbouring pixels | Resolved |
| CR-64 | Each kernel re-solved how it reads a view and where it writes | Resolved |
| CR-65 | The half-precision sink converted in the serial column build, one element at a time | Resolved |
| CR-66 | The warp's border fill truncated, and stored 0 for an out-of-range border | Resolved |
| FU-05 | `uv.lock` pinned a yanked numpy (2.4.0) | Resolved (numpy 2.4.6 in `uv.lock`) |
| CR-88 | Nearest resize broke exact pixel-centre ties the wrong way (fir's floating-point steps) | Resolved: one exact integer gather for every dtype; the parity oracle compares nearest exactly |
| CR-89 | The parity oracle's round-half-away used `floor(\|v\| + 0.5)`, rounding 0.49999999999999994 up | Resolved: exact rounding, with fixture tests (the engine was right) |
| CR-90 | The parity framework propagated a bound through a wrapping integer cast (gain 1) | Resolved: `gain_for(x, p)` sees the input; a cast that can wrap is discontinuous |
| CR-91 | A colour conversion's float bound scaled with its alpha (an infinite alpha made `to_hsv`'s unbuildable, `to_ycbcr`'s infinite) | Resolved: `color_magnitude` scales `to_hsv`, `to_ycbcr` and `grayscale` bounds by the colour channels |
| CR-92 | The `to_lab` reference was OpenCV's float Lab, 0.57 off in a* on dark pixels (the engine was exact) | Resolved: Lab by its definition in float64; bound 0.5 -> 0.01 |
