# Handover — view-buffer single-authority refactor

**Branch:** `claude/polars-image-plugin-assessment-JlM54`
**Date:** 2026-06-06
**Scope:** Eliminate Python's parallel re-declarations of the per-op schema
knowledge that the `view-buffer` crate already owns (dtype, output domain, rank
/ndim, channel count, the executable-op set, and the API enum vocabularies), so
there is exactly one authority across the Rust↔Python boundary.

This document summarizes what shipped (workstreams WS-1…WS-5), the design
decisions behind them, the current state, and what is left (WS-5 tail, WS-6,
WS-7) with enough detail to resume cold.

---

## TL;DR

- **WS-1…WS-5 are complete, committed, and pushed.** 8 commits, listed below.
- **Test state:** `1281 passed, 3 xfailed` (Python suite, excluding the
  network-only `test_http_sources.py`); package `ruff` clean; `cargo clippy -p
  polars-cv` clean; new Rust unit tests pass.
- **The core dedup mandate is done:** view-buffer is now the single source for op
  domain, dtype, rank, channel, the executable-op registry, and the API enums.
- **Remaining:** the smaller WS-5 tail (graph format-string guard, boundary
  typing), the heavier WS-6 correctness work (plan==exec for binary promotion
  and image-source `auto`, which needs a change inside `view-buffer`), and the
  WS-7 backlog.

---

## Key invariants (do not regress these)

1. **view-buffer is the authority.** The Python planner must *read* per-op schema
   effects from view-buffer via `_lib.op_contract` / `_lib.op_output_dtype`,
   never re-declare them in a Python table. There is no longer any
   `OPERATION_CONTRACTS` / `DTypeEffect` / `NdimEffect` / `AlphaMode`.
2. **plan == exec.** The dtype/shape the planner infers must equal what execution
   produces. Guarded by the `test_plan_equals_exec_*` tests in
   `polars-cv/tests/test_sanitation.py`. Two cases are still known-divergent and
   tracked as `strict=False` xfails (WS-6).
3. **Enums are guarded, not hand-synced.** API enums duplicated across the
   boundary are checked by `test_enum_parity_*`; the Rust side is the authority.

---

## Build / test / lint (run from `polars-cv/polars-cv/`)

```bash
# Activate the venv FIRST, and always cd into the package dir — the maturin build
# and the venv are both under polars-cv/polars-cv/. (Several CI mishaps in this
# session came from running `source .venv/bin/activate` from the wrong cwd.)
cd polars-cv/polars-cv
source .venv/bin/activate

maturin develop                 # rebuild the Rust extension into the venv
cargo build -p polars-cv        # Rust-only compile check
cargo test  -p polars-cv        # Rust unit tests (incl. known_ops_tests)
cargo clippy -p polars-cv       # lint Rust

python -m pytest tests/ -q --ignore=tests/test_http_sources.py
ruff check python/polars_cv/    # the real Python lint gate (the package only)
```

Notes:
- `tests/` (not the package) has pre-existing `I001` import-sort findings across
  ~40 files; the project does **not** enforce ruff on `tests/`. Don't "fix" those
  in files you happen to touch — it's out of scope and inconsistent with the rest.
- After **any** `.rs` change you must `maturin develop` before the Python tests
  see it. Comment-only `.rs` changes don't need a rebuild for Python tests.

---

## Architecture: how schema inference works now

The Python planner (`polars-cv/polars-cv/python/polars_cv/pipeline.py`,
`_compute_output_domain_dtype_ndim` and `_update_channels_from_rule`) builds the
output schema by asking view-buffer about each op:

- `_lib.op_contract(op_json)` → dict with `output_domain`, `input_domain`,
  `dtype_rule`, `rank_rule`, `channel_rule`.
- `_lib.op_output_dtype(op_json, in_dtype)` → the resolved concrete dtype.

The Rust side of these lives in `polars-cv/polars-cv/src/lib.rs`
(`op_contract`, `op_output_dtype`, `enum_variants`, `known_ops`) and delegates to
view-buffer's `ViewDto` (`output_dtype_rule()`, `output_rank_rule()`,
`output_channel_rule()`, `output_domain()`), which is built on declarative
rule methods on the `Op` trait (WS-1).

A few **param-dependent** cases remain special-cased in the Python planner (they
are genuinely param-driven, not re-declarations): `cast` (dtype from the param),
`histogram` (output mode), and axis-based reductions (`axis` present → reduce one
rank vs. global → scalar).

---

## Completed workstreams

### WS-1 — Declarative rank/channel rules; planner reads them
Commits **6992ee2** (`WS-1a`), **5e522b5** (`WS-1b/c`).

- Added rank/channel rule methods to the view-buffer `Op` trait
  (`OutputRankRule`, `OutputChannelRule` in
  `view-buffer/src/ops/shape_rule.rs`), surfaced through `op_contract`.
- Python planner now derives ndim from `rank_rule` and channels from
  `channel_rule` (`pipeline.py: _update_channels_from_rule`) instead of the old
  `NdimEffect` / `AlphaMode` tables.

### WS-2 — Output domain single-sourced; histogram domain/encoding split
Commits **6cc3776** (`WS-2a`), **17673a9** (`WS-2b`).

- **2a:** Deleted Python's `_OPERATION_OUTPUT_DOMAIN`. Each op's output domain
  now comes from `op_contract()["output_domain"]`. view-buffer's `any` domain is
  an identity (materialize) domain and leaves the pipeline domain unchanged.
- **2b — design decision "separate domain from encoding":** histogram *buckets*
  used to masquerade as a `histogram` **domain**. That string was really a
  *sink-encoding* selector (it routed to a `List(Struct[lower_edge, upper_edge,
  count, normalized])` schema). It is now modelled honestly:
  - buckets are the **`vector` domain**;
  - a new `OutputSpec.expected_encoding` axis (value `"histogram_buckets"`)
    selects the struct schema in Rust encode/decode
    (`src/graph/{encode,decode,types}.rs`);
  - it is set from `Pipeline.output_encoding()` and threaded through the graph
    spec in `_graph.py`.
  - `Domain.HISTOGRAM` / `Pipeline.DOMAIN_HISTOGRAM` removed;
    `test_enum_parity_domain` is green (Python == Rust − internal `any`).
- **Dropped from scope:** `op_infer_shape` / `ViewDto::infer_shape`. Its only
  planned consumer was the buckets `[num_bins, 4]` shape, which the encoding
  approach handles directly. (A H/W-from-params dedup in `_update_shape_hints`
  remains *possible* future work, low priority.)

### WS-3 — Deleted the `OPERATION_CONTRACTS` mirror
Commits **a078112** (code), **87d5431** (docs/comments).

- Ungated `_compute_output_domain_dtype_ndim`: it now applies `op_output_dtype`
  and the Rust `rank_rule` to **every** op (no `OPERATION_CONTRACTS` membership
  gate). The only ops this newly touches are contour-domain ones whose dtype is
  schema-irrelevant; verified safe (full suite stayed green).
- Removed `OPERATION_CONTRACTS`, `OpContract`, `DTypeEffect`, `NdimEffect`,
  `AlphaMode` from `_types.py` and the `AlphaMode` public export from
  `__init__.py`.
- Pruned the obsolete **declaration** tests (they asserted the values of a table
  that no longer exists): `TestContractConsistency`, `TestAlphaModeContracts`,
  `TestAffineContract`, the `test_contract_parity_*` tests, and their
  `_EFFECT_TO_RULE` / `_OP_PARITY_EXCEPTIONS` scaffolding. **All behaviour tests
  were retained** (plan-time `output_dtype`, channel inference, E2E dtype,
  domain/rank-rule parity). Net: 622 lines removed.
- Updated dev docs (`python/polars_cv/AGENTS.md`, `tests/AGENTS.md`,
  `src/AGENTS.md`, `docs/api/functions.md`) and Rust comments to point at the
  view-buffer contract.

### WS-4 — Op registry parity (B1)
Commit **f0ee51d**.

- **Rust:** `execute::KNOWN_OPS` is the single registry of executable op names,
  surfaced via `_lib.known_ops()`. Three unit tests in
  `src/execute.rs::known_ops_tests` keep it honest: every entry resolves (no
  entry without a `resolve_op` arm), a bogus name is rejected by the catch-all,
  and the list stays sorted/unique.
- **Python:** `Pipeline.OP_NAMES` (ops the builders emit, including the
  `lazy.py` binary ops). `test_op_names_covers_all_emitted_ops` scans
  `pipeline.py`/`lazy.py` source so the set can't silently under- or over-list.
- The previously-skipped `test_registry_parity_pipeline_ops_are_executable` now
  runs and asserts `OP_NAMES ⊆ known_ops()`.
- Note: 6 ops are executable but not emitted by any Python builder
  (`channel_merge`, `contour_flip`, `contour_is_convex`, `contour_normalize`,
  `contour_to_absolute`, `contour_winding`) — fine for the ⊆ guarantee.

### WS-5 — Enum parity guards (core)
Commit **e16b9f8**.

- `enum_variants` now answers for `NormalizeMethod`, `ColorSpace`,
  `HashAlgorithm`, `HistogramOutput`, `PadMode`, `PadPosition`, `FilterType`.
  Each maps its variants through an **exhaustive match**, so adding a view-buffer
  variant is a compile error until its canonical string is supplied.
- Parity tests: the first six assert **==** with the Rust set; `FilterType`
  asserts Python **⊆** Rust (Rust also offers `catmullrom`/`gaussian` and
  surfaces `Triangle` as `"bilinear"`; polars-cv intentionally exposes only
  `nearest`/`bilinear`/`lanczos3`).
- **`OutputDType` decision — keep as a strategy enum.** It carries the
  `preserve` strategy (keep input dtype, promote ints→f32) that `DType` cannot
  express, so it is *not* a duplicate. The old `strict=True` xfail expecting its
  deletion was replaced by `test_output_dtype_is_strategy_not_dtype_duplicate`,
  which asserts the intentional distinction. (It was **not** renamed — the name
  already reads as "output dtype option"; a rename is cosmetic and a public-API
  change, left as optional.)

---

## What's left

### WS-5 tail (small, low risk)
- **Graph format-string guard.** `SourceFormat`/`SinkFormat` are a deliberate
  three-way split (view-buffer CamelCase enums; the graph boundary uses plain
  strings; Python defines its own incl. Polars-native `list`/`array`/`native`
  and `file_path`/`raw`). This is documented (comment near the enum-parity tests)
  but not asserted. Add a guard that the graph's accepted format strings are a
  known set (where the graph validates `sink.format` / `source.format`).
- **Optional boundary typing.** In `src/graph/types.rs`, parse the plugin-boundary
  `String` fields into enums once at the edge, and represent `auto` as
  `Option<DType>` rather than the string `"auto"`. Pairs naturally with WS-6/A2.

### WS-6 — plan==exec everywhere (correctness; **heaviest remaining**)
Two `strict=False` xfails in `tests/test_sanitation.py` mark the gaps:

- **A3 — binary-op promotion** (`test_plan_equals_exec_binary_promote`, ~line
  148). Today both layers say binary ops *preserve the left operand*
  (`view-buffer .../binary.rs` = `PreserveInput`; `lazy.py` copies the left
  dtype). Real promotion is missing (e.g. `divide` → float; `add/sub/mul/min/max
  /bitwise` → element promotion of *both* operands). Plan: add a **two-input
  promotion authority** in view-buffer, use it from execution and expose a
  plan-time FFI; have the Python binary-op dtype defer to it. Then flip the xfail
  to `strict=True`.
- **A2 — image-source `auto`** (`test_plan_equals_exec_auto_16bit_image`, ~line
  133). Plan-time can't know a decoded image's dtype (u8 vs u16). Make plan==exec
  *by construction*: have the exec-time dtype guard in `graph/types.rs` back-fill
  the resolved input dtype from the first decoded row (run it even when the plan
  said `auto`), and/or coerce image decode to a deterministic planned dtype. Then
  flip the xfail to `strict=True`.

This is the only remaining work that modifies the upstream **view-buffer** crate,
so budget for a view-buffer change + rebuild + its own unit tests.

### WS-7 — broader backlog (lower priority for the dedup mandate)
- **Plan-time validation (A12, B3, B4):** move column-binding / shape-ref /
  acyclicity checks into `UnifiedGraph::from_json` + `.sink()`; turn the
  `(domain, format)` fallthrough in `decode.rs` into an error; route all
  source-decode through one `on_error` wrapper.
- **`shape_pipeline` (A8):** implement (resolve referenced node's shape at exec)
  or reject at plan time — no declared-but-dead feature.
- **Dead code:** B5 tiling surface, B6 `cloud_options` round-trip, B7
  `PipelineSpec` wrapper + `first_output` hack, B10 `ANNOTATED_POINT_SCHEMA`
  export; add `cargo clippy -D warnings` as a standing CI guard.
- **Zero-copy / perf (A5, A6, A7, C3):** move-not-clone at flush/terminal,
  borrow rather than clone buffers where possible.

---

## Map of the important files

| Concern | File |
|---|---|
| Plan-time schema inference | `polars-cv/polars-cv/python/polars_cv/pipeline.py` (`_compute_output_domain_dtype_ndim`, `_update_channels_from_rule`, `output_encoding`, `OP_NAMES`) |
| Python API enums / types | `polars-cv/polars-cv/python/polars_cv/_types.py` |
| Graph spec build (encoding, domain, dtype) | `polars-cv/polars-cv/python/polars_cv/_graph.py` |
| Binary ops (lazy) | `polars-cv/polars-cv/python/polars_cv/lazy.py` |
| FFI: contract/dtype/enums/known_ops | `polars-cv/polars-cv/src/lib.rs` |
| Op resolution + `KNOWN_OPS` | `polars-cv/polars-cv/src/execute.rs` |
| Sink encode/decode + dtype guard | `polars-cv/polars-cv/src/graph/{encode,decode,types}.rs` |
| Per-op rules (authority) | `view-buffer/src/ops/` (`shape_rule.rs`, `dto.rs`, `compute.rs`, `image.rs`, …) |
| Drift/parity guards | `polars-cv/polars-cv/tests/test_sanitation.py` |

## The full plan
The living plan with per-workstream detail and the original assessment items
(A1–A12, B1–B10, C-series) is at
`~/.claude/plans/can-you-do-a-iterative-shannon.md` (outside the repo). WS-1…WS-5
have `DONE` sections there mirroring this handover.
