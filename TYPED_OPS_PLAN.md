# Plan: migrate polars-cv to a fully typed op protocol

> **Status ledger — keep current.** This file is the working plan for CR-45…CR-49
> (see `CODE_REVIEW_FINDINGS.md`). Update the progress table as phases land; a plan
> that lives only in a session ends with it.
>
> **Integration branch:** `claude/codebase-quality-review-7rtgcb` (the plan's
> "one long-lived branch", continued from `…-wnfwqz` by fast-forward; named by
> the session's branch rules rather than `typed-ops`).
>
> | Phase | Status |
> |---|---|
> | P0 — Safety net and seams | **done** — corpus (214 cases), signature snapshot, pickle pin, removed-symbol gate, `_plan_view` seam (30 files), baselines, dead `GraphNode` fields |
> | P1 — Positional expression slots | **done** — `{"$slot": n}` wire form from a graph-wide `SlotTable`; `expr_key`/`expr_column_names`/name binding deleted; per-call slot bound check; `shape_node` id; CR-50 logged |
> | P2 — Catalogue foundation + spike | **done, one gate item over** — `#[derive(Op)]`/`typed_ops!`, `Param`/`Literal`, name-keyed dispatcher (`LEGACY_OPS` 81), `crop`/`resize`/`warp_affine`/`histogram` typed with generated builders, catalogue ↔ `.so` ↔ generated-module checks, mkdocs inherited members. Gate: corpus ✓, signatures ✓, `mkdocs --strict` ✓ (generated `Args:` render), release `.so` 32,192,392 B (+58 KB, +0.2%), release build 839 s cold (baseline 819 s). Plan build µs/append (release; P0 → P1 → P2): `mixed` 82.8 → 80.7 → **78.0**; `chain` 49.8 → 51.7 → **53.7**; `lazy_continuation` 83.5 → 86.6 → **90.7**. The two legacy-only scenarios are over baseline (~2 µs each from P1's slot table and P2's dispatcher map on the legacy path); both paths shrink in P3 and go in P6/P7 |
> | P3 — Migrate every op | **done** — all 85 ops typed (P3.1 view, P3.2 compute, P3.3 image, P3.4 colour/filter/rotate/reductions/phash/channel, P3.5a geometry, P3.5b binary/mask/merge, P3.5c rasterize/label_reduce/extract_shape); `LEGACY_OPS` empty; name-keyed resolution deleted. Gate: corpus ✓, signatures ✓, full `scripts/verify.sh` PASS at `af41c23` (slow lane, `cargo deny`, `mkdocs --strict` included) |
> | P4 — Typed sources and sinks | **done** — `src/formats/` (`formats!` registry, `io_catalog.json`, `io_check`); both applicability tables, `PARAM_HINTS`, `KNOWN_SOURCE_FORMATS`, `SourceSpec`/`SinkSpec` and the last untyped param readers deleted. Gate: corpus ✓, signatures ✓, full `scripts/verify.sh` PASS at `910afb2` (slow lane, `cargo deny`, `mkdocs --strict` included) |
> | P5 — Geometry namespaces | **done** — `ContourKwargs`/`PointKwargs` are typed (`Param<T>`, `ColumnRef`, `#[derive(Op)]`); `input_slots`, `InputSlots`, `parse_named`/`require_named` deleted. Gate: corpus ✓, signatures ✓, full `scripts/verify.sh` PASS at `f3d7d94` (slow lane, `cargo deny`, `mkdocs --strict` included) |
> | P6 — Delete the legacy protocol | **done** — `pipeline.rs` (`OpSpec`/`LegacyOpSpec`/dispatcher), `LEGACY_OPS`, `resolve_op`, the untyped `ParamValue`, `known_ops`, `OP_NAMES` (P6a); 20 Python enums generated from the registries via `enum_catalog.json` and their parity tests deleted (P6b); `enum_variants`/`enum_names` and the serde-name tests deleted, graph policies parse through `NAMED` (P6c). Python `OpSpec`/`ParamValue` deferred to P7, enum helpers kept (see deviations). Gate: corpus ✓, signatures ✓, full `scripts/verify.sh` PASS at `2ae7651` (slow lane, `cargo deny`, `mkdocs --strict` included) |
> | P7 — Planner into Rust | **done, targets missed (see deviations)** — P7a `a6f0aa1`: `plan_step`, one FFI per append. P7b `98d7592`: per-op entering state and one rewrite primitive, `_replay`. P7c `66ff9bb`, `0df1790`, `9e3b6af`: identity elimination and the spatial pushdown in Rust (`node_pass`, generated `LogicalPass`), the pass catalogue (`OptFlags`/`OptConfig` from one list), `bit_exact` deleted. P7d `2ae8eaf`, `8f7f8fa`: `plan_source`; histogram buckets read off the ops. P7e `cc383db` … `0cef576`: `PlanState` is Rust's `State`, assertions applied by `plan_assert`, sinks checked by `plan_sink`, outputs carry `planned` (the `expected_*` wire fields deleted), binary lazy methods generated, one lineage fold. Gate: corpus ✓, signatures ✓, full `scripts/verify.sh` PASS at `0cef576` |
> | P8 — API reshaping | **done, no version bump (see deviations)** — P8a `65bc3d6`: one signature rule, derived by `gen_ops.positional` (the `#[param(positional)]` marker deleted; 15 ops change kind). P8b `ab13385`: `source()` keywords default to `None` (`is_supplied`, `_source_param_defaults` deleted; contour colour defaults in Rust), `perceptual_hash` generated, `output_encoding()` deleted. P8 exit `6a571ce`: documented `Pipeline()` calls bind against the real signatures, migration page. `signatures.json` re-recorded in P8a/P8b. Gate: full `scripts/verify.sh` PASS at `6a571ce` |
> | P9 — Symbolic shapes | pending |
> | P10 — Final sweep | pending |

## Handover (2026-09-24, P6 closed)

Written so a fresh session can continue without the originating conversation.
Read this section, then the phase text for P7 onwards below.

### State

- Branch `claude/codebase-quality-review-7rtgcb`, pushed. Commits for this
  plan: P0 (up to `5a83210`), P1 `cb6fc46`, P2 `1faec46`, P3.1 `dd52de0`,
  P3.2 `dfa8a94`, P3.3 `b8e23ba`, P3.4 `c774819`, P3.5a `d13274b`, P3.5b
  `0c637e1`, P3.5c `af41c23`, P3 exit `c92c1e9`, P4a (sinks) `0da9163`, P4b
  (sources) `910afb2`, P4 exit `8d79321`, P5 `f3d7d94`, P5 exit `249dcff`,
  P6a `94e9bfa`, P6b `f2a7ea3`, P6c `2ae7651`, then the P6 exit.
- Every op, source and sink is typed, and the untyped protocol is gone:
  `TypedOp` is the wire op (an unregistered name is "Unknown operation").
- Python enums are generated (`enum_catalog` FFI → `tests/golden/
  enum_catalog.json` → `gen_ops.py`) from `named_variants!` + the two
  registries, docstrings included (`named_variants!(Name: "doc" { … })`);
  `gen_ops.NOT_GENERATED` lists the three exceptions with reasons.
- Sources and sinks: one `#[derive(Op)]` struct per format in
  `polars-cv/src/formats/`, a `formats!` registry emitting the enum, its
  tagged wire form (a field the format does not read is refused naming where
  it applies) and `io_catalog.json`; Python validates through `io_check`.
  `is_supplied` stays until P8 (the frozen `source()` signature still carries
  value defaults).
- The geometry namespaces' kwargs are typed the same way (P5): `Param<T>`
  fields and `ColumnRef` operands, `{"$slot": n}` on the wire, checked by
  `GeomParams` against the derived slots.
- P7 is done (`0cef576`): every schema fact comes from `src/plan.rs`
  (`plan_step`/`plan_source`/`plan_assert`/`plan_sink`) and the node passes
  from `node_pass`; Python keeps the op list and immutable `PlanState`
  records, rewriting only through `_replay`. Outputs carry `planned`, the
  node's final state. No `Plan` pyclass (see deviations).
- P8 is done (`6a571ce`): generated methods follow one derived signature
  rule, `source()` keywords default to `None`, and the docs' `Pipeline()`
  calls are bound against the real signatures.
- Next: **P9**, symbolic shapes (`infer_dims` replacing the four-probe
  `infer_shape`).

### Line counts per phase

The end state is meant to be smaller, so each phase exit records where it
stands (lines, by area; `py-gen` is `_ops_generated.py`, generated):

| At | rust-plugin | macros | engine | py-hand | py-gen | tests |
|---|---:|---:|---:|---:|---:|---:|
| P0 end `5a83210` | 16,921 | 0 | 20,809 | 16,068 | 0 | 55,208 |
| P3 exit `c92c1e9` | 17,781 | 209 | 20,991 | 14,299 | 1,575 | 55,468 |
| P4 `910afb2` | 18,193 | 209 | 20,964 | 14,099 | 1,607 | 55,430 |
| P5 `f3d7d94` | 18,085 | 209 | 20,964 | 14,093 | 1,607 | 55,443 |
| P6 `2ae7651` | 17,728 | 209 | 20,956 | 13,752 | 1,862 | 55,266 |
| P7c `9e3b6af` | 18,333 | 209 | 20,969 | 12,985 | 1,930 | 54,890 |
| P7d (part) `8f7f8fa` | 18,479 | 209 | 20,969 | 12,895 | 1,930 | 54,892 |
| P7 exit `0cef576` | 18,967 | 209 | 20,969 | 12,183 | 2,125 | 54,930 |
| P8 `6a571ce` | 18,936 | 197 | 20,969 | 12,062 | 2,127 | 55,140 |

So far the phases have *moved* definitions into typed Rust (each carrying the
docs, defaults and validation Python used to hold) more than they have
removed code. P6 was the first net reduction: plugin −357, hand-written
Python −341 (the enum classes moved to the generated module, +255), tests
−177. Since P0: hand-written Python −2,316, plugin +807.

**P7 target** (the Python planner and its FFIs): hand-written Python ≤ 11,800
(−1,950), plugin ≤ 18,900 (+1,170 for `Plan`/`PlanState` and the passes),
so hand-written code (plugin + engine + py-hand) falls by at least 780.

### How a family is migrated (the recipe every P3 commit followed)

1. **Rust definition** in `polars-cv/src/ops/<family>.rs`: a struct with
   `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]` and
   `#[serde(deny_unknown_fields)]`; fields are `Param<T>` (per-row capable) or
   `Literal<T>` (structural), `Option<_>`, `Vec<_>`, `[_; N]`; every field and
   the struct need a doc comment (they become the Python docstring and
   `Args:`). Markers: `#[param(positional)]`, `#[param(default = <literal>)]`,
   struct-level `#[op(python = "name")]` and `#[op(visibility = "internal")]`.
   Field names and order must reproduce the frozen Python signature
   (`tests/golden/signatures.json`) — the wire follows the signature. An
   `OpDef::resolve` that opens with an exhaustive destructure and returns the
   same `GraphStep` the legacy arm built (port the arm body). Repeated shapes
   use a small local `macro_rules!` (see `compute.rs`, `image.rs`,
   `reduce.rs`).
2. **Register** one line with a valid sample in `typed_ops!` (`ops/mod.rs`),
   kept sorted; add `pub mod` for a new family file.
3. **Delete** the names from `LEGACY_OPS` (only inside that const — the name
   can also appear in test lists) and their `resolve_op_inner` arms; then
   delete every helper the compiler reports dead (`cargo check -p polars-cv`
   *and* `cargo clippy` — clippy sees dead code the `--tests` check hides) and
   the tests that only exercised those helpers.
4. **Port tests**: legacy tests that named a migrated op (`strict_param_tests`,
   `unread_param_tests`, hand-built graph JSON in `graph/compiled.rs` /
   `encode.rs`) move to typed form — rejection cases go into
   `ops::tests::an_invalid_value_is_rejected_naming_its_field`; graph JSON uses
   bare values (`"q": 50.0`, not `{"type":"literal",...}`).
5. **Re-bless + regenerate**: `POLARS_CV_BLESS=1 scripts/with-pyo3-env.sh
   cargo test -p polars-cv` (rewrites `tests/golden/op_catalog.json`), then
   `python scripts/gen_ops.py` and `python scripts/gen_lazy_stub.py` (put
   `.venv/bin` on `PATH` so ruff is found).
6. **Python**: delete the hand-written `Pipeline` methods (they are now
   inherited from `_ops_generated._OpsMixin`). A method whose signature carries
   sugar (extra keywords, enum-member defaults, argument resolution) stays
   hand-written and calls the generated `_<name>` of an `internal` op
   (`scale`, `clamp`, `resize_scale`, `perceptual_hash`). Keep `OP_NAMES`.
7. **Verify**: `maturin develop` (debug), fast lane
   (`pytest tests/ -m "not network and not slow"`), `ruff check`/`format`,
   `uvx ty check --project polars-cv`, clippy, `cargo test -p view-buffer
   --all-features`, `python scripts/check_removed_symbols.py` (add each
   deleted symbol to `REMOVED`), then commit. The pre-commit hook fails on a
   stale `.so` — rebuild after any Rust edit, including doc/fmt changes.

Expected test churn per family: error-message regexes (the Rust definition is
now the validator; same input rejected at the same point — say so in the
commit), wire-shape assertions in `test_serialization.py`, and none in the
golden corpus or signature snapshot (both must stay green unchanged).

### How P3.5 landed (for reference)

- **Geometry** (`ops/geometry.rs`): the eight contour measures/transforms and
  `extract_contours`; `scale_contour` is sugar over an internal op (enum-member
  default).
- **Binary family, `apply_mask`, `channel_merge`** (`ops/binary.rs`): operands
  are a `NodeRef` field (catalogue kind `node`); wire field names follow the
  lazy signatures (`other`, `mask`, `others`). Visibility `lazy_only`:
  `gen_ops.py` emits no `Pipeline` method, and
  `test_every_lazy_only_op_is_a_lazy_method_with_its_fields` pins each to its
  hand-written `LazyPipelineExpr` method. `Pipeline._add_node_op` is the one
  graph hook; `_op_reads_sibling_nodes` reads node-typed fields from
  `OP_FIELDS`.
- **`rasterize`**: internal op, `size: RasterSize::{Fixed([h, w]) |
  FromNode(node)}`; the sugar records the shape reference's upstream edge and
  canvas assertion. `CompiledGraph`'s `RasterizeShapeRef` resolver keys on the
  typed variant and shares `Rasterize::with_size`.
- **`label_reduce`**: `contours: ColumnRef` (slot-only, catalogue kind
  `column`); builder generated.

### Gotchas learned

- The planner's shape probe (`lib.rs::infer_shape_probe`) now runs the op's
  own `validate` and raises only rank-level failures
  (`ValidationError::depends_only_on_rank`); size-level build-time checks were
  deliberately left for P9. `op_infer_shape` returns `None` for "not
  inferable" and raises `ValueError` only for invalid parameters.
- Under a plan-time probe a per-row `Param` reads a placeholder integer; a
  named enum/bool returns `WireScalar::probe_value()`. An op whose validity
  couples a per-row value to a literal (convolve2d `ksize`) must special-case
  `ctx.is_probe()`.
- Parse each op once and never clone the parsed JSON on the append path
  (a clone cost ~13 µs/append; see the P2 row).
- Named enums get their wire form only from `NAMED` (`WireScalar`, emitted by
  `named_variants!`); side-table aliases were unreachable from Python and have
  been deleted (`FilterType`, `ColorSpace`).
- Disk is tight in the container (~4 GB free after `scripts/dev-clean.sh`);
  a release build needs ~2 GB. Benchmarks need `maturin develop --release`,
  then rebuild debug.

## Deviations recorded during execution

Where the implementation departs from the text below, and why. The text is left
as planned so the deviation stays visible.

- **P6 — the enum registries and their source scans stay.** The plan meant to
  reach every enum through `Describe` (op field types) and delete `REGISTRY`/
  `PLUGIN_REGISTRY`, but `Winding`, `RowErrorPolicy` and `NullParamPolicy` are
  not op or io fields, so field types cannot enumerate them. The registries
  are now the enum catalogue's source (not a parity list), and the scans are
  what make declaring a `NAMED` table the same act as generating its class.
  The `enum_variants`/`enum_names` FFIs did go.
- **P6 — the graph policies drop serde rather than `named_variants!` emitting
  it.** Engine enums already derive serde for `ViewDto` in another spelling,
  so the macro cannot emit a second impl; `RowErrorPolicy`/`NullParamPolicy`
  lose their derives and the graph fields read them through
  `ops::param::literal_field`, the path `Literal<T>` takes.
- **P6 — Python `OpSpec`/`ParamValue` move to P7.** They are the planner's op
  representation (`_graph.py`, CSE, `_plan_view`); deleting them is deleting
  the Python planner.
- **P6 — `_validate_enum`, `_enum_or_expr`, `_reject_expr` stay.** They
  validate literals against the *generated* enums (no second vocabulary) for
  `source()`, `out_dtype=` and the geometry accessors, which have no build-time
  Rust check; deleting them would move those errors from build to execution.

- **P8 — no version bump.** The breaking changes are in the CHANGELOG's
  Unreleased section with a migration page; the version moves when that
  section is released, which is a release decision, not a migration step.
- **P8 — hand-written sugar keeps its signatures.** The rule is applied to
  generated methods. `clamp(min_val, max_val)`, `scale(factor, out_dtype=)`,
  `resize_scale` and `rasterize` keep theirs; no renames were agreed.
- **P8 — the binding check reads `Pipeline()`-rooted chains only.** A bare
  `.name(...)` in a doc block may be Polars', so only chains that start at
  `Pipeline()` are bound (396 calls, floor 300).

- **P7 — no `Plan` pyclass; the state is a Rust-computed value instead.**
  What P7 set out to delete, the Python *planning logic*, is gone: every
  schema fact comes from `src/plan.rs` and `node_pass`. What remains in
  Python is bookkeeping — the op list, the entering state per op, the
  assertion map, `_replay` — held as immutable `PlanState` records Rust
  returns. Moving that into a `Plan` pyclass would re-express about as many
  lines in Rust plus a PyO3 surface, a net increase, so it is not done;
  CSE stays in Python over those records, as planned.
- **P7 — Python `OpSpec`/`ParamValue` stay** (deferred from P6). They are
  the builder's record of an op's arguments (literal or expression) until the
  graph assigns slots; with no Python planner left they carry no schema
  logic, and replacing them needs the `Plan` pyclass above.
- **P7 — the line targets are missed.** Hand-written Python is 12,183
  (target ≤ 11,800) and the plugin 18,967 (target ≤ 18,900), because the
  planner moved without the `Plan` pyclass deletions the targets assumed and
  the binary ops' docs moved into Rust doc comments. The remaining reduction
  is P8's: `is_supplied` and the source-default machinery, the
  enum-default wrappers (`perceptual_hash`), `output_encoding` and the
  `out_dtype` sugar all exist to keep signatures P8 is allowed to change.
- **P3 deviation superseded (P7e):** the eleven binary ops' lazy methods
  *are* generated now (`_LazyOpsMixin`, all built by `_binary_op`);
  `apply_mask` and `channel_merge` stay hand-written.

- **P3 — the binary family's lazy methods are not generated.** Each
  `lazy_only` builder constructs a new graph node its own way (a binary op
  starts a BLOB-source pipeline and takes its dtype from
  `binary_output_dtype`; `apply_mask` clones the left operand's pipeline;
  `channel_merge` is variadic), which the generator cannot express. They stay
  hand-written on `LazyPipelineExpr`, held to the catalogue by a guard test
  (method exists, is defined there, parameters equal the op's fields).
- **P3 — one struct per binary op, not a generic `Binary{op}`.** The wire and
  every name-keyed consumer (`binary_output_dtype`, CSE, graph viz) keep the
  op name; `binary_ops_are_exactly_the_named_table` holds the structs to
  `BinaryOp::NAMED`, so the table is still the one list.
- **P3 — the name-keyed resolver and its read-tracker are deleted now, not in
  P6.** With `LEGACY_OPS` empty they had no reachable arm, and the
  no-dead-code rule wins over the phase boundary. P6 keeps the rest of its
  list (`LegacyOpSpec`, the dispatcher's legacy arm, `LEGACY_OPS`).
- **P4 — sources and sinks reuse `#[derive(Op)]` rather than a second
  describing mechanism**, with `Option` fields meaning "absent = the default"
  (applied in one Rust place), so a hand-built `{"format": "numpy"}` stays
  valid. The contour canvas is the `rasterize` op's `RasterSize`, so its error
  names `size` for a misplaced `width`/`height`/`shape` until P8 reshapes the
  signature.
- **P4 — `PARAM_HINTS`' three hints are dropped, not moved.** The typed error
  names the formats a field applies to; only the contour `dtype` → `.cast`
  advice had no equivalent, and one test that pinned it now pins the typed
  rejection.
- **P3 — `ParamCtx` carries the probe value.** A node-sized `rasterize` has
  no slot, so it reads the probe placeholder directly to make its canvas vary
  across probes (unknown at plan time) instead of the deleted slot injection.

- **P2 — `#[derive(Op)]` (crate `polars-cv-macros`) instead of a `define_op!`
  `macro_rules!`.** A declarative macro cannot take doc comments plus per-field
  markers (`#[param(positional)]`, defaults) without a local-ambiguity error,
  and cannot emit a precise error for a missing doc or a forbidden
  `#[serde(...)]`. The derive also *refuses* a struct without
  `#[serde(deny_unknown_fields)]`. The registry half (`OpSpec`-variant list,
  `NAMES`, dispatch, samples) is still one `typed_ops!` line per op.
- **P2 — static vs per-row is computed by the derived slot visitor**
  (`TypedOp::is_static`), not by resolving under `RowCtx::static_only()`. Same
  property (no per-op flag to forget; the visitor is generated from the field
  list), no sentinel error path. `OpDef::resolve` takes `(row, &ParamCtx)`;
  `RowCtx` is not needed yet.
- **P2 — the wire field shape follows the frozen Python signature**, so a
  signature tuple is one field: `warp_affine(output_size=[h, w])` (was
  `output_height`/`output_width`), `histogram(range=[min, max])` (was
  `range_min`/`range_max`, where one without the other was silently ignored).
- **P2 — named enums get their wire form from a `WireScalar` trait** in
  `view_buffer::naming` (emitted by `named_variants!`), not from serde derives:
  the engine enums already derive serde for `ViewDto` with different
  spellings. `NAMED` is the only wire vocabulary, so parser-only side tables
  (`FilterType::ALIASES`' `"triangle"`) are not accepted on typed ops; they go
  with the last legacy op that reads them (P3).
- **P2 — no per-op `plan()` yet.** `OpDef` has `resolve` only; the plan-time
  schema still comes from the existing `op_*` FFIs over the typed op. `plan()`
  arrives with the Rust planner (P7).

## Context

**Why.** Today an operation crosses the Python→Rust boundary as a name string plus
an untyped param map (`OpSpec { op: String, #[serde(flatten)] params:
HashMap<String, ParamValue> }`, `polars-cv/src/pipeline.rs:132`). Nothing structural
ties the two sides together, so the repo keeps them aligned with registries,
parity tests, source scanners, runtime trackers and FFI round-trips:
- `KNOWN_OPS` (85 names, `execute.rs:282`) mirrored by `Pipeline.OP_NAMES`
  (`pipeline.py:449`);
- an ~813-line `match op_name` (`resolve_op_inner`, `execute.rs:421-1234`) whose
  completeness is checked by a test that `include_str!`s its own source;
- a 64-bit read-tracking bitmask (`OpParams`, `params.rs:691`);
- shape inference by probing four magic values (`lib.rs:263`);
- 19 hand-written Python enum mirrors plus parity tests;
- per-format parameter applicability tables;
- a Python planner with 14 state fields that calls five `op_*` FFIs over JSON on
  every append, and a second schema fold in Rust (`compiled.rs:1881/1915`).

**Outcome.** One typed Rust definition per op becomes the only authority:
- **Deserialisation** (serde) rejects unknown ops, unknown or misspelled params,
  wrong types and out-of-range values.
- **The compiler** rejects an unhandled op or field (exhaustive `match` +
  exhaustive destructuring).
- **The Python builder is generated** from the definition.
- **The planner is Rust**, behind one immutable Python object.
- **Shapes are symbolic.**

Every check the types now enforce is **deleted**, in the commit that makes it
structural. The migration ends with no legacy path.

**User decisions:**
- The Python builder is generated.
- The planner moves fully into Rust.
- Symbolic shapes are included.
- The API and wire format may break (pre-1.0).
- Everything lands on **one long-lived branch** merged once at the end.
- There is **no manual checkpoint** after the spike; mechanical exit gates only.

---

## Target architecture (end state)

### 1. Op catalogue: `define_op!` (new, `polars-cv/src/ops/`, one file per family)

A declarative macro is the single definition per op. From one declaration it
emits:
- the struct, always `#[serde(deny_unknown_fields)]`;
- the `OpSpec` newtype variant;
- `OpSpec::NAMES`;
- a `describe() -> OpDesc` entry: doc lines, per-field `ParamKind::{PerRow,
  Literal}`, the Python type, the default, positional/keyword, and visibility
  (`public` / `lazy_only` / `internal`), plus the Python method name when it
  differs from the op name (`cvt_color` → `convert_color`, `contour_area` → `area`,
  …);
- a sample instance, used by generated tests.

It is chosen over schemars because:
- schemars can't express Param vs Literal, value ranges, Python names or
  positional args without custom impls everywhere;
- it isn't a current dependency;
- the macro keeps everything in one place.

**Wire encoding**
- **`OpSpec`** is `#[serde(tag = "op", rename_all = "snake_case")]` with
  **newtype variants only**. Every op, including no-arg ones (`abs`,
  `grayscale`), is `Abs(Abs)` with `struct Abs {}`: serde's internally-tagged
  *unit* variants silently accept extra keys. There is no `flatten` inside op
  structs. Errors go through `serde_path_to_error`, so messages name op and field.
- **`Param<T>`** is `Lit(T) | Slot(usize)`, with a hand-written `Deserialize`: the
  map `{"$slot": n}` is a slot; anything else goes straight to
  `T::deserialize`, so its error message survives. Untagged is forbidden (it is
  ambiguous for integer `T` and destroys errors).
- **`Literal<T>`** is for structural params. The eligibility rule becomes a type:
  a `$slot` there is a serde error.
- **Value domains are types:**
  - `u32` for indices and sizes (so `top=-5` fails at build);
  - newtypes such as `Positive<f32>` (sigma) and `NonSingular` (affine matrix),
    each validating in **one** `TryFrom` shared by serde (literals) and
    `Param::resolve` (per-row);
  - per-element lists `Vec<Param<f32>>` (`warp_affine` matrix, `reshape` shape,
    `normalize` preset), replacing today's `ParamValue::List` sniffing
    (`compiled.rs:1712-1737`).
- **Branch-only params become variants.** A param read only under some branch
  becomes an enum, so a type can't accept it and ignore it:
  - `rotate` at multiples of 90° vs an arbitrary angle;
  - `Bins::{Count | Edges}` for `histogram`;
  - `NormalizeMethod::{MinMax | ZScore | Preset{..}}`;
  - `RasterSize::{Fixed | FromNode}` for `rasterize`.

**Enum spellings.** `named_variants!` (`view-buffer/src/naming.rs:45`) stays the
single spelling authority, aliases included. It is extended to emit
`Serialize`, `Deserialize` (name lookup via `NAMED`, with the valid names in the
error) and `Describe`. Per-row string parsing, literal parsing and codegen all
read `NAMED`. Enums reach the catalogue through the field-type walk.

**Per-op trait `OpDef`** (no default methods):
- `resolve(&self, &RowCtx) -> Result<GraphStep>` builds the *same* `GraphStep` /
  `ViewDto` the engine uses today;
- `plan(&self, &PlanState, &PlanRefs) -> Result<PlanState>`.

Each impl opens with an **exhaustive destructure** (`let Crop { top, left,
height, width } = self;`), so a new or unused field is an `unused_variables`
error under `-D warnings`. `#![deny(clippy::rest_pat_in_fully_bound_structs,
clippy::wildcard_enum_match_arm)]` is scoped to `polars-cv/src/ops/`.

- **Static vs per-row** is *computed*, not declared. Resolving once with
  `RowCtx::static_only()` (slot access returns a `NeedsRow` sentinel) decides it.
  This replaces `OpSpec::is_all_literal`, and there is no per-op flag to forget.
- **Engine contracts are unchanged.** view-buffer's `Op` trait, `ViewDto` and the
  `GraphStep` contract methods stay the authority for dtype, rank, channel,
  spatial and identity rules.

### 2. Planner: Rust `Plan` (`polars-cv/src/plan.rs`)

- **`#[pyclass(frozen)] Plan`** is a persistent cons list,
  `Arc<PlanNode{op, entering_state, prev, len}>`: appends are O(1), and
  per-position state is built in, which replaces `_hint_snapshots` and
  `_POSITION_KEYED_FIELDS`.
- **`PlanState`** (frozen pyclass):
  - domain, dtype, `dims: Vec<Dim>`, channels;
  - assertions, the shape-declared flag, policies.

  `Dim = Known(n) | Unknown | SameAsInput` from day one; P7 changes how dims are
  computed, not this interface.
- **API** (replaces `_append_op`, `_push_op`, `_update_*`, `_rewrite_ops`,
  `_set_ops_slice`, `_STATE_COPIERS`, `binary_output_dtype` and the replay in
  `lazy.py:274-339`):
  - `Plan.push(op_json, n_exprs)`;
  - `push_binary(op, other: PlanState)`;
  - `push` with `refs` for ops that read other nodes (`rasterize(shape=)`,
    `apply_mask`, `channel_merge`);
  - `final_state()`, `rebase(state)`, `slice(a, b)`, `remap_slots(map)`;
  - `op_key(i)`: canonical serde JSON, used for CSE;
  - `common_prefix_len(others)`.
- **Unseeded state.** A source-less `Pipeline` defers the domain check until
  `rebase`.
- **Optimiser passes** move in: identity elimination, spatial pushdown and the
  pass registry (`enum LogicalPass`, exhaustive, exposed through the catalogue).
  They match on enum variants, never names. Python keeps only CSE *grouping*,
  because it needs `pl.Expr` identity; the prefix comparison is in Rust.
- **Graph output schema.** `unified_output_dtype` reads `Plan` state.
- **Python-object behaviour:**
  - `__reduce__` pickles via JSON, and `__copy__` / `__deepcopy__` return `self`;
    today's pickle/deepcopy behaviour is pinned in P0;
  - errors are `PyValueError` with the path-annotated serde message.

### 3. Expressions

- `pl.Expr` objects stay in Python: `Pipeline._exprs`, de-duplicated with
  `Expr.meta.eq`. Params encode as `{"$slot": i}`.
- `PipelineGraph` builds one graph-wide `meta.eq` expression table and calls
  `Plan.remap_slots` on every node **before any logical pass**, so CSE only ever
  compares global slots. That table also replaces `expr_key` for root-column
  grouping and bindings (`_graph.py:424/639/656`).
- `expr_column_names` disappears from `GraphKwargs`. The compiled-graph cache
  keys on graph JSON alone, and each call checks `max_slot < inputs.len()`.

### 4. Python surface (generated)

- **Catalogue flow.** `catalog()` is committed as `tests/golden/op_catalog.json`,
  and a Rust unit test asserts equality, so drift is caught by `cargo test`
  without `maturin develop`. `scripts/gen_ops.py` reads that JSON, never the
  `.so`, and writes the committed `python/polars_cv/_ops_generated.py`:
  - an `_OpsMixin` with one method per public op;
  - generated enums;
  - Google-style docstrings (`Args:` from the field docs, as mkdocs requires).
- **`Pipeline(_OpsMixin)`** keeps: `source`, `assert_shape`, the policies,
  `_push`, explain/repr, and **sugar** that may call generated methods only:
  `flip_h`/`flip_v`, `to_hsv`/`to_lab`/`to_bgr`/`to_ycbcr`,
  `sobel`/`laplacian`/`sharpen`, `morphology_open`/`morphology_close`, `shear`,
  `rotate_and_scale`, `thumbnail`, `adjust_brightness`.
- **Lazy.** The `LazyPipelineExpr` binary family (`lazy_only` ops) is generated
  too. `lazy.py` forwarders and `lazy.pyi` are regenerated as today.
- **Stubs and docs.** `_lib.pyi` is generated. `mkdocs.yml` gains
  `inherited_members: true`: without it the generated methods vanish from the API
  docs.
- **Encoding.** The generated encoder converts numpy scalars to Python numbers.
  NaN and ±inf handling is pinned in the corpus in P0.

### 5. Sources, sinks and geometry

- **Sources and sinks.** `Source` and `Sink` become tagged enums per format,
  using the same `Param<T>` types. The applicability tables disappear.
- **Geometry.** `ContourKwargs`, `PointKwargs` and `GeomParams` use `Param<T>`
  with positional slots: one per-row mechanism for the whole crate.

---

## Transition discipline: no dead code, no half-wiring

1. **One name-keyed dispatcher while ops migrate.** A hand-written
   `Deserialize for WireOp` reads `"op"`:
   - a name in `OpSpec::NAMES` deserialises *strictly* as typed;
   - a name in `LEGACY_OPS` uses the legacy path;
   - anything else errors.

   No fallback is possible, and a typo on a typed op fails loudly (pinned by a
   test). Transitional code is allowed only behind this dispatcher's legacy arm.
2. **A ratchet built from the existing tests.** Rename `KNOWN_OPS` to
   `LEGACY_OPS`:
   - `known_ops_all_resolve` proves each legacy name still has an arm;
   - the existing `include_str!` arm scan fails the moment a migrated op leaves
     an arm behind;
   - one new test: `NAMES ∩ LEGACY_OPS = ∅` and `NAMES ∪ LEGACY_OPS = the frozen
     85-name set` (plus any op added on `main` meanwhile).
3. **The golden corpus is the arbiter for every commit.**
   - Case builders live in `tests/_golden_cases.py`: `case_id → lambda`, grown
     from `_op_cases.py`/`EXTRA_CASES`, plus one expression-param variant per
     `Param` field and the rejection cases taken from every test deleted below.
   - The fixture `tests/golden/op_corpus.json` stores **results only**: the
     Polars `collect_schema()`, a planned view (domain, dtype, rank, dims,
     channels via the helper), and an output digest or small 8×8 array, compared
     with tolerance for float kernels (±1 for u8, 1e-5 relative), since CI also
     runs macOS. Errors are stored as class + stable substring. Each case also
     has a neutral `{"op", "args"}` record for reviewers.
   - The case-id set equals the fixture keys, in both directions.
   - The only allowed diffs are listed bug fixes and API reshaping (Phase 8).
4. **Signatures are frozen until the API phase.** P0 snapshots every public
   method's `inspect.signature` (`tests/golden/signatures.json`). Until Phase 8,
   every generated method must equal it (names, kinds, defaults), so the
   migration doesn't churn ~420 call sites. API reshaping is its own phase.
5. **Every phase exit is a deletion gate:**
   - `scripts/check_removed_symbols.py`: a per-phase list of identifiers that
     must not appear. It skips `CHANGELOG.md`, `CODE_REVIEW_FINDINGS.md` and
     `docs/changelog.md`, and has fixtures for both the hit and the exclusion.
     It runs in `verify.sh`.
   - The full `verify.sh`, including the slow lane.
   - A CHANGELOG entry, plus updates to the ledger (CR-45…CR-49) and the
     `AGENTS.md` canonical-paths table.
   - No "delete later" is left behind, except what the next phase's list names.
6. **Branch hygiene** (one long-lived branch — see the status ledger above):
   - Every commit passes `verify.sh --fast`.
   - `main` is merged in (never rebased) at each phase boundary; new ops or
     params from `main` are migrated in that merge, and the ratchet forces it.
   - The final PR carries the ticked deletion matrix, the full removed-symbol
     list and the benchmark table.

---

## Phases (in execution order)

### P0 — Safety net and seams (no behaviour change)
- Golden corpus: `tests/_golden_cases.py`, `scripts/gen_golden_corpus.py`,
  `tests/golden/op_corpus.json` and `test_golden_corpus.py`. Watch it fail by
  mutating one op.
- Signature snapshot and its test.
- Pin today's pickle/deepcopy of `Pipeline` behaviour.
- `scripts/check_removed_symbols.py` with fixtures, plus its `verify.sh` hook.
- **The test seam `tests/_plan_view.py`**: `planned(p) -> PlanView` and
  `ops_of(p)`. Migrate the ~20 test files that read private planner state onto it
  (e.g. `test_alpha_channel` 50 refs, `test_affine_builder` 39,
  `test_cse_optimization` 37, `test_append_contract` 33, `test_pipeline_builder`
  18, `test_optimize` 15). Rewriting those reads doesn't weaken the tests; each
  test's assertions are unchanged.
- **Baselines:**
  - plan build time (100 appends);
  - `benchmarks/run_benchmarks.py` throughput;
  - `cargo build` time;
  - `.so` size.
- Ledger: open CR-45 (typed ops), CR-46 (Rust planner), CR-47 (typed
  sources/sinks), CR-48 (geometry params), CR-49 (symbolic shapes). Also delete
  the three never-read `GraphNode` fields `alias`, `domain` and `output_dtype`
  (`graph/types.rs:383-410`), with the Python emitter change.

### P1 — Positional expression slots (on the legacy wire, all ops)
- Python `_exprs` with `meta.eq` de-duplication; `{"$slot": i}` encoding.
- The graph-wide expression table, global remap before CSE, and root-column
  grouping on the same table.
- Rust `bind_param` accepts `$slot` directly. `expr_column_names` is removed from
  `GraphKwargs`, and a per-call slot bound check is added.
- **Delete:** `expr_key`, `_EXPR_KEYS` and its lock, `ParamValue.Expr{col}`,
  name→slot binding (`name_to_slot`), and `test_expr_param_identity.py` 39–82.
- **Keep:** its CR-31 behavioural cases (104, 110, 119) as guards of the new
  table.

### P2 — Catalogue foundation and spike (`crop`, `resize`, `warp_affine`, `histogram`)
The spike covers optional extents, a per-row enum, `Vec<Param<f32>>`, and a
variant split with a multi-mode schema.

**Rust:**
- `define_op!`, `Param`/`Literal`/value newtypes, `OpDef`, `RowCtx` (wrapping
  `ParamCtx`: the null policy still flows through `ParamCol::on_null`),
  `static_only`;
- the `WireOp` dispatcher and the ratchet;
- `named_variants!` gains serde + `Describe`;
- `op_catalog.json` plus its equality test.

**Python:** `gen_ops.py`, `_ops_generated.py`, `test_ops_generated_is_current`,
`Pipeline(_OpsMixin)`, the four hand-written methods deleted, and mkdocs
`inherited_members`.

**Serde unit tests, each watched failing first:**
- an unknown field is rejected;
- the tag is not reported as unknown;
- an extra key on a no-arg op is rejected;
- the error names op and field;
- `f32` accepts `1`; `u32` rejects `3.0` and `-5` (recorded as corpus
  changes);
- `$slot` in a `Literal` is rejected.

**Mechanical exit gate (stop and report if any fails):**
- the corpus is green;
- per-append build time ≤ baseline (measure `json.dumps` vs `pythonize`);
- cargo build time and `.so` growth are recorded, with no more than a noted
  budget;
- `mkdocs build --strict` renders the generated methods with `Args:` blocks.

### P3 — Migrate every op (about ten family commits)
**Order:**
1. view
2. compute/scalar
3. image (resize family, pad family, blur, threshold, morphology, canny,
   equalize)
4. colour
5. filter
6. affine/rotate/warp
7. reductions
8. histogram/phash
9. channel ops
10. binary family (generic `Binary{op: BinaryOp}`; lazy methods generated)
11. geometry/contour ops
12. graph-level ops: `rasterize` (`RasterSize`, deleting the `__shape_ref__`
    probe injection in `lib.rs` and `acknowledge("shape_ref")`),
    `extract_contours`, `label_reduce` (typed `ColumnRef`, deleting the
    `bind_param` skip), `extract_shape`, `apply_mask`, `channel_merge`

**Every family commit does all of:**
1. Adds typed structs with docs, and `OpDef` impls that reuse the old arm bodies.
2. Lists every param the old arm read only under a branch, and turns each into a
   variant, pinned by a serde rejection test.
3. **Deletes** those `resolve_op_inner` arms and any `get::*` / `resolve_*`
   helper left unused (the compiler flags them).
4. Shrinks `LEGACY_OPS`, regenerates Python and deletes those hand-written
   methods.
5. Keeps the corpus and the signature snapshot green.

### P4 — Typed sources and sinks
- Tagged `Source` and `Sink` enums per format. `SinkKind::resolve` keeps its
  exhaustive role.
- The Python `source()` / `sink()` signatures stay frozen until P8; the generated
  code validates the format against the tagged enum.
- **Delete:** `KNOWN_SOURCE_FORMATS`, `SourceFormat::parse`, the Python
  `SourceFormat` / `SinkFormat` hand enums (now generated),
  `SOURCE_PARAM_APPLIES`, `SINK_PARAM_APPLIES`, `PARAM_HINTS`,
  `reject_inapplicable_params` and `is_supplied`.

### P5 — Geometry namespaces on the same types
- `ContourKwargs` / `PointKwargs` fields become `Param<T>` with positional slots,
  and `GeomParams` reads via `RowCtx`.
- **Delete:** `InputSlots`, `parse_named` / `require_named`, and `_ArgBinder`'s
  name→slot map.

### P6 — Delete the legacy protocol (now possible: nothing else uses `ParamValue`)
- **Rust deletions:**
  - the `WireOp` dispatcher (the wire op *is* `OpSpec`), `LegacyOpSpec`,
    `resolve_op`, `resolve_op_inner`, `LEGACY_OPS`, the `known_ops` FFI;
  - `OpParams` (`acknowledge`, `unread`), `get_param`, `pub mod get`;
  - the untyped `ParamValue` and all its `resolve_*` / `as_*` accessors,
    `bind_param` / `bind_graph_params`, and `OpSpec::is_all_literal`;
  - the enum registries once everything reaches the catalogue through
    `Describe`: `REGISTRY`, `PLUGIN_REGISTRY`, the `enum_variants` /
    `enum_names` FFI, and the `named_variants!` source scans.
- **Python deletions:**
  - `Pipeline.OP_NAMES`, `_types.OpSpec`, `_types.ParamValue`;
  - `_validate_enum`, `_enum_param`, `_enum_or_expr`, `_reject_expr`,
    `_literal_axes`;
  - the 19 hand-written enum classes, now generated.
- Plus the tests marked P6 in the matrix.

### P7 — Planner into Rust
- Add `Plan` and `PlanState` as specified. The optimiser passes and registry move
  to Rust; `OptFlags` reads its fields from the catalogue.
- `_graph.py` serialises Plans and does CSE grouping only.
- `explain`, `__repr__` and `_graph_viz.py` are ported to `Plan` accessors, and
  `_lib.pyi` is regenerated.
- **Python deletions:**
  - the 14 planner fields and `_STATE_COPIERS` / `_POSITION_KEYED_FIELDS`;
  - `_append_op`, `_push_op`, `_update_*`, `_apply_shape_contract`,
    `_compute_output_domain_dtype_ndim`, `_rewrite_ops`, `_set_ops_slice`;
  - `_create_sub_pipeline`'s fold, the continuation replay, and
    `_op_contract_for`;
  - `_output_shape_equals_input`, `_op_reads_sibling_nodes` (now a variant
    property), and `output_encoding`'s `op == "histogram"` string match;
  - `_pass_handlers`.
- **Rust and FFI deletions:** `op_schema`, `op_contract`, `op_output_channels`,
  `op_infer_shape`, `op_identity_rule`, `binary_output_dtype`,
  `resolve_op_from_json[_probe]`, `fold_output_rank`, `fold_output_dtype`,
  `op_probe_json`, `param_probe_json`, `PRESERVED_DIM`.

### P8 — API reshaping (the only phase allowed to change signatures)
- Apply the generator's signature rule: an op with exactly one required param
  takes it positional-or-keyword (`.cast("f32")`, `.threshold(128)`); everything
  else is keyword-only. Plus any agreed renames.
- Update `examples/01-13`, `benchmarks/`, the README, `docs/user-guide/**` and
  the notebooks. Regenerate the signature snapshot deliberately.
- Extend `test_documented_methods_exist` to **bind** each documented call against
  the real signature (`inspect.signature.bind`).
- Add a user-facing migration page and bump the version.

### P9 — Symbolic shapes
- In view-buffer, the `Op` trait gets a required
  `infer_dims(&[&[Dim]]) -> Result<Vec<Dim>>` (no default) replacing
  `infer_shape`, across 12 impls plus geometry steps. Concrete execution calls it
  with `Known` dims.
- **Delete:** the four-probe machinery, `unknown_dim_probe`, `ParamCtx::probe`
  and the enum-default substitution, the planning `catch_unwind`, and
  `IdentityRule::deciding_params` name strings (now typed field accessors).
- Extend `test_op_schema_rules_are_required_not_defaulted` to cover
  `infer_dims`.

### P10 — Final sweep
- Rewrite the canonical-paths tables and "Adding a New Operation" in CLAUDE.md,
  AGENTS.md, `src/AGENTS.md`, `python/polars_cv/AGENTS.md` and `tests/AGENTS.md`.
  Adding an op becomes: `define_op!` + one `OpDef` impl + `gen_ops.py`.
- Audit:
  - `cargo clippy -D warnings`;
  - `cargo machete`;
  - `vulture` with an allowlist;
  - the complete `check_removed_symbols.py` list.
- `verify.sh` runs `gen_ops.py --check` and the catalogue test in the structural
  lane.
- Close CR-45…CR-49 and open the single PR.

---

## Deletion matrix — every sync guard and its replacement

Legend: **DEL** means the property is now compile- or serde-enforced. **RW**
means rewritten on the new mechanism. **KEEP** means behavioural.

### Rust
| Guard | Where | Verdict / phase |
|---|---|---|
| `known_ops_all_resolve`, `resolve_op_arms_are_all_known_ops` (+`KNOWN_GUARD_ARMS`, `include_str!`), `known_ops_sorted_and_unique` | `execute.rs:1459/1494/1614` | ratchet during P3 → DEL P6 |
| `unknown_op_is_rejected` | `execute.rs:1475` | RW as a serde unit test, P2 |
| `unread_param_tests`, `OpParams` bitmask | `execute.rs:1627`, `params.rs:691` | DEL P6, **only after** every family's branch-read list is empty (a variant or serde test each) |
| `strict_param_tests` | `execute.rs:1252` | ranges → `TryFrom` / serde unit tests; rest DEL, P6 |
| `every_graph_step_variant_is_reachable_from_a_known_op` | `compiled.rs:2224` | RW: catalogue samples → resolve → an exhaustive `all_step_kinds()`, P6 |
| `every_graph_geometry_op_executes` hand `probe_params` | `compiled.rs:2225`, `encode.rs:634` | RW from catalogue samples, P6 |
| `source_format_names_match_the_vocabulary`, `KNOWN_SOURCE_FORMATS` | `compiled.rs:2687/1340` | DEL P4 |
| `every_named_enum_is_registered`, `every_plugin_named_enum_is_registered` (source scans), `registered_enums_have_unique_names` | `naming.rs` (both crates) | DEL P6; RW `*_have_unique_names` over the catalogue (Python class clashes) |
| `row_error_policy_names_match_serde`, `null_param_policy_names_match_serde` | `types.rs:145`, `params.rs:1096` | DEL P6 (`named_variants!` emits serde: one spelling) |

### `test_sanitation.py`
| Line | Verdict |
|---|---|
| 360, 375, 519, 534 (op-name parity and scans) | DEL P6 |
| 569 (removed ops unknown) | DEL P6 → Rust serde "unknown variant" test |
| 399 (accessor ↔ `#[polars_expr]`) | KEEP |
| 476 (lib hooks) | RW P7 (shrunk list + generated `_lib.pyi`) |
| 675 (binary dtype authority) | RW P7 as Rust plan tests; FFI deleted |
| 704 (no defaulted `Op` rules) | KEEP; extend in P9 |
| 752, 779, 824 (planner domain / contract vocab / no second spelling) | DEL P7 |
| 859, 876, 944, 956, 981 (enum parity) | DEL P6 (generated + current-check) |
| 1015 (BinaryOp ↔ lazy names) | DEL P3 (lazy binary methods generated) |
| 1053 (source formats) | DEL P4 |
| 1134, 1181 (lazy parity, stub current) | KEEP; add `_ops_generated.py` and `_lib.pyi` current-checks |
| 1345 (`op_schema` cases) | RW P7 as `OpDef::plan` unit tests, same expectations |
| 1354 (state == batch fold) | DEL P7 |
| 1398 (append cost linear) | RW P7 (count Rust folds) |
| 1454 (histogram dtype declared once) | DEL P3 |
| 1517 (enum validation uniform) | RW P6, catalogue-driven over every enum field |
| 1553 (geometry enums accept Expr) | RW P5 as a catalogue rule |
| 2619 / `_kwargs_scan.py` (closed structs) | KEEP; teach it `define_op!`; drop the `OpSpec` exemption |
| 2662 (entry point rejects unknown kwargs) | KEEP; add a typed-op probe |

### Other Python tests
| Guard | Verdict |
|---|---|
| `test_append_contract.py` 127, 158, 188, 251, 270, 305, 360 | DEL P7 |
| `test_append_contract.py` 404 | KEEP |
| `test_append_contract.py` 412 | RW P7 (message only) |
| `test_append_contract.py` 458 | RW P3 (authority = catalogue public variants) |
| `test_param_strictness.py` `TestParamPolicyRatchet` (269), `TestStructuralParamsRejectExpressions` (356) | DEL P6 → one catalogue test: every `Literal` field rejects `pl.Expr` (Python) and `$slot` (serde) |
| `test_param_strictness.py` `TestEnumValuesExecutable` (119), per-row enum / dtype / literal cases (563/574/585) | KEEP (values from the catalogue) |
| Accept-expression classes (301–850) + `test_every_expression_parameter_has_a_case` | RW P6 → one catalogue-driven sweep: every `Param` field, expr == literal result |
| `TestInputSlotsAreValidated` (866) | RW P5 |
| `test_param_applicability.py` | DEL P4; keep one actionable-error test per surface |
| `test_expr_param_identity.py` 39–82 | DEL P1 |
| `test_expr_param_identity.py` 104/110/119 | KEEP |
| `test_removed_surfaces.py` 42, 64, 274, 293 | DEL P6 |
| `test_removed_surfaces.py` 193 | DEL P4 |
| `test_removed_surfaces.py` 395, 477 | DEL P5 |
| `test_removed_surfaces.py` 245 | → `check_removed_symbols` |
| `test_removed_surfaces.py` 86 | DEL P7 |
| `test_removed_surfaces.py` 118, 159, non-op tombstones | KEEP |
| New generic test (replaces the op-param tombstones) | every generated method raises `TypeError` on an unknown keyword; every op struct rejects an unknown key (catalogue samples) |
| `test_op_case_tables.py` | RW (generated enums, P4 sinks); single-channel / extra checks KEEP |
| `test_spatial_rule.py`, `TestOpInferShapeAuthority` (`plan_matches_data` 90) | DEL P7 (expectations ported to Rust plan tests; pushdown behaviour stays in `test_optimize_equivalence.py`) |
| `test_optimize.py` registry parity (74/83) | RW P7 (Rust `LogicalPass` registry) |
| Schema-parity suites, `test_plan_matches_data`, `test_typed_nodes`, the reference suite, `test_known_gaps`, `test_optimize_equivalence`, `test_parallel_rows`, `test_blob_protocol`, `test_strict_input_bounds` | **KEEP** |

**Rule for every DEL row:** it may only be deleted once the corpus contains its
rejection or behaviour cases. That is what makes deletion safe rather than
coverage loss.

---

## Critical files
- **New:**
  - `polars-cv/src/ops/` (the `define_op!` catalogue) and
    `polars-cv/src/plan.rs`;
  - `scripts/gen_ops.py`, `scripts/gen_golden_corpus.py`,
    `scripts/check_removed_symbols.py`;
  - `python/polars_cv/_ops_generated.py`;
  - `tests/golden/{op_corpus,op_catalog,signatures}.json`,
    `tests/_golden_cases.py`, `tests/_plan_view.py`.
- **Rewritten then shrunk:** `polars-cv/src/{execute.rs (deleted), params.rs,
  lib.rs, pipeline.rs, graph/compiled.rs, graph/types.rs, geom_params.rs,
  contour.rs, point.rs, naming.rs}`, `view-buffer/src/{naming.rs,
  ops/traits.rs}`, `python/polars_cv/{pipeline.py, _types.py, _graph.py,
  lazy.py, _optimize.py, _namespace.py, _lib.pyi}`, `mkdocs.yml`.
- **Reused:**
  - view-buffer `Op`, `ViewDto`, `GraphStep` contracts;
  - `ParamCol::on_null`, `SinkKind`, the compiled-graph cache;
  - `scripts/_format.py` and the `gen_lazy_stub.py` pattern;
  - `_expr_param_runner.py` and `_schema_parity.py` as test harnesses.

## Verification
- Per commit: `scripts/verify.sh --fast` (rebuild with `maturin develop` after
  Rust changes; `build_info()` hashes must match), the golden corpus, and the
  signature snapshot (until P8).
- Per phase: full `scripts/verify.sh`, including the slow lane and examples;
  `check_removed_symbols.py` for that phase; `mkdocs build --strict`.
- At P2 and at the end: benchmarks against the P0 baselines (plan build 100
  appends, `benchmarks/run_benchmarks.py` eager and streaming, cargo build time,
  `.so` size), with no regression beyond noise and the recorded budget.
- Final: a pickle/deepcopy round-trip test, a mixed-expression CSE test (distinct
  exprs at the same local slot must not merge), and the deletion matrix fully
  ticked in the PR.
