# Plan: first-class Arrow extension types in polars-cv

Status: **Phases 1, 2, 3 and 6 implemented** (see §0); Phases 4–5 and the
open decisions remain. Scope: the numpy/torch sink struct (`polars_cv.ndarray`)
and the geometry family (`polars_cv.point` / `.contour` / `.bbox`).

**Recommendation (unchanged):** migrate, in phases. Start by adding a new
opt-in `sink("ndarray")`. Do not change `sink("numpy")`, and do not tag geometry
outputs, until input acceptance and persistence have shipped and been used.
Tagging a column is a breaking change for anyone who touches it with struct
operations (see §2.1), so every step that makes tagging the default must be
explicit and versioned.

## 0. Implementation status

**Shipped:**

| Phase | What landed | Where |
|---|---|---|
| 1 — Foundation | `ExtType` (Rust, one list; storage read from `geom_schema` / `numpy_output_dtype`); `extension_types` FFI; `NdArrayType` / `PointType` / `ContourType` / `BBoxType` registered at `import polars_cv`; `_plugin.call` as the only way into the plugin; `dtype-extension` always on | `src/ext_types.rs`, `python/polars_cv/extension_types.py`, `python/polars_cv/_plugin.py` |
| 2 — `sink("ndarray")` | `SinkKind::NdArray` across the four halves; `SinkFormat.NDARRAY`; `numpy_from_struct` / `show_images` read the tag; `sink("numpy")` unchanged | `src/graph/{sink_kind,decode,encode}.rs`, `tests/test_ndarray_sink.py` |
| 3 — Tagged inputs | Every accessor and `vb_graph` source accepts a tagged column; swept from the accessor case tables | `tests/test_schema_parity_namespaces.py::test_accessors_accept_tagged_inputs` |
| 6 — Spike deleted | `tests/spike_point_ext/`, `src/ext_*.rs`, the `spike-ext-types` feature | — |

**Where the implementation departed from §3, and why.** Building it turned up
a simpler design that is stronger on every axis §3 cared about:

- **No plugin-side registration at all (replaces §3.3's registration role and
  all of §3.4).** `_plugin.call` passes every argument as `.ext.storage()` — a
  no-op on a plain column, a zero-copy relabel on a tagged one. So the plugin
  never *receives* an extension dtype, no Rust input path has to strip tags, and
  a tagged column is validated by exactly the parser that validates its plain
  struct. Rust only *builds* tagged outputs (`ExtType::tag`), which needs no
  registry. The whole §2.2 plugin-side timing problem (registration living in a
  `#[pymodule]` init that polars' `dlopen` never runs) disappears rather than
  being managed.
- **The two-`.so` hazard is removed, not guarded (§3.3).** `_plugin.plugin_path()`
  resolves the file the import system loads and hands *that* to polars, instead
  of the package directory (from which polars took the first `.so` `iterdir()`
  returned). The planned `build_info()` guard became unnecessary.
- **Canonical storage is enforced by the host type, not a Rust check (§3.5).**
  polars *panics* if `ext_from_params` raises, so our classes instead return
  polars' generic `Extension` for our name over any other storage or with
  metadata. `isinstance(dtype, PointType)` is therefore a complete check.
- **No generated Python module (§3.2).** Python declares the four classes over
  the schema constants it already exports, and
  `test_python_types_match_the_rust_declaration` compares names, order and the
  *full* storage dtype over the `extension_types` FFI (an empty Series per type).
  This is the pattern `POINT_SCHEMA` already uses, and it keeps
  `NUMPY_OUTPUT_SCHEMA` — previously pinned only against a literal — honest too.
- **Guard is the case tables (§3.4).** Instead of a per-source-path Rust probe,
  every `.contour` / `.point` / `.bbox` case the completeness check requires is
  re-run on tagged inputs and must equal the untagged result. It caught
  `.point.x` / `.y`, which read the struct in Python and now read
  `.ext.storage()`.

**Remaining:**

- Phase 4 — tagged geometry *outputs*. Users can tag with `.ext.to(PointType())`
  today; a tagged contour sink or constructors are still to decide (§4).
- Phase 5 — defaults (separate, major-version decision).
- Open decision — `polars_cv.ndarray` metadata (dtype/ndim), Phase 2 note.
- Upstream — report polars' `register_extension_type` duplicate check (it tests
  the literal `"ext_name"`) and the `ext_from_params`-raises-panics behaviour.
- Found along the way, not changed: `sink("numpy")` encodes a null input row as
  a struct of null fields, not a null struct. `sink("ndarray")` matches it
  (`test_null_input_rows_encode_as_the_numpy_sink_does`); changing it is its
  own decision.

---

## 1. What the spike established

| Question | Answer | Evidence |
|---|---|---|
| Is the API available on the pinned stack? | Yes: polars 0.54 / py-polars 1.42, feature `dtype-extension`, no lockfile change | spike builds; `test_host_type_identity` |
| Does a tag get through the plugin boundary? | Yes. Inside the plugin `.ext()?.storage()` recovers the storage and `into_extension` re-applies the tag | `test_q2_plugin_op_receives_and_returns_tag`, ndarray/bbox/contour identity tests |
| Can a wrong input be rejected before execution? | Yes, in `output_type_func`: tag **and** storage are checked against the Rust authority | `test_q3_*`, `test_wrong_storage_is_rejected_at_schema_resolution` |
| Does Parquet keep the tag? | Yes, when the reader has the type registered. Otherwise the reader gets **plain storage plus a `UserWarning`** by default, or a generic extension with `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_extension` (the stated Polars 2.0 default) | `test_parquet_persistence`; probe run during review |
| When does the plugin side register? | Only when the Python module init of `_lib` runs. polars' own `dlopen` of the same `.so` for an expression symbol does **not** run it | spike design note; `_ext.ensure_registered` |
| Do both copies of polars-core share one registry? | Only when Python and polars load the **same `.so` file**. A second `.so` (e.g. `_lib.cpython-311*.so` beside `_lib.abi3.so`) gives two registries, and the tag decays silently | found while fixing the spike (`--features` replacing `pyo3-extension`) |

Also found in review: polars' Python `register_extension_type` duplicate check
tests the literal string `"ext_name"`, so it never fires (1.42). Report it
upstream; until it's fixed, our own registry must be what stops a name being
registered twice.

## 2. Constraints

### 2.1 Tagging breaks struct operations

Measured on py-polars 1.42 against a tagged `{x, y}` column:

| Operation | On a tagged column |
|---|---|
| `.struct.field("x")` | **error** (`StructFieldNotFoundError`) |
| `df.unnest(col)` | **error** (`expected 'Struct'`) |
| `.cast(<storage struct>)` | **error** (`cannot cast extension types`) |
| `pl.concat` with an untagged column | **error** (schema mismatch) |
| `.ext.storage().struct.field(...)` | works |
| `.to_list()`, `.rows()` | works (plain dicts) |

This decides the migration strategy. Output that changes from a struct to a
tagged struct breaks user code, including this repo's own tests (e.g.
`test_numpy_helpers.py` casts to `NUMPY_OUTPUT_SCHEMA`). **Emitting a tag must be
opt-in until a major version.** **Accepting a tag** breaks nobody and can ship first.

### 2.2 Registration timing

- **Plugin side:** the Rust registry must be populated before polars hands the
  plugin a tagged dtype. That happens when the input schema crosses the FFI
  boundary, before any of our code in `output_type_func` runs. A lazy `Once`
  inside our functions is therefore too late.
  - Building a `Pipeline` already imports `_lib`: the planner reads the
    `op_schema` FFI. So the `vb_graph` path is covered today.
  - The geometry accessors are **not** covered: `pl.col("c").contour.area()`
    builds an expression without importing `_lib` (verified).
- **Host side:** the Python registry must be populated before a tagged column
  is *read*, e.g. `pl.read_parquet(...)` right after `import polars_cv`.
  Otherwise the column decays to storage on read. So host registration belongs
  at `import polars_cv`, which must stay cheap and must not load the plugin
  (the existing "geometry imports with no compiled extension" invariant).

### 2.3 Unstable upstream API

Polars documents extension types as unstable. Keep every use of
`polars::datatypes::extension` and `pl.datatypes.BaseExtension` behind one
module on each side (§3.1), so an upstream change only has to be fixed in one
place. Pin the behaviour with tests, not comments.

### 2.4 Repository rules that apply

From `CLAUDE.md` Working Agreements:

- One authority per fact: a type's name and storage are declared once.
- Registering is the same act as being checked.
- A bypass must fail, not degrade: no silent decay to storage and no loose
  "looks-like" acceptance.
- Prefer a mechanism callers cannot step around over a ratchet test.

## 3. Architecture

### 3.1 One registry table (Rust), published to Python

Replace the four near-identical `ext_*.rs` modules with one declarative table:

```rust
// polars-cv/src/ext/mod.rs
ext_types! {
    NdArray  => "polars_cv.ndarray", storage = crate::output::numpy_output_dtype(),   display = "ndarray";
    Point    => "polars_cv.point",   storage = crate::geom_schema::point_struct_dtype(), display = "point[xy]";
    Contour  => "polars_cv.contour", storage = DataType::Struct(crate::geom_schema::contour_fields()), display = "contour";
    BBox     => "polars_cv.bbox",    storage = crate::geom_schema::bbox_struct_dtype(),  display = "bbox";
}
```

The macro generates, per row:
- the `ExtensionTypeImpl`;
- the factory. It **checks the storage it is given**. `create_type_instance`
  cannot fail, so on a mismatch it returns a marker type that `expect_ext`
  rejects by name;
- the registration call;
- an arm of `ExtType::ALL`.

Then:
- `register_all()` is the only registration site.
- `expect_ext::<T>()` (already in `src/ext_check.rs`) is the only input check.
- `extension_types()` is a new FFI function returning `[(name, storage), ...]`,
  joining `op_schema` / `enum_variants` / `known_ops`.
- **Guard:** `ExtType` is an enum, so a new type that skips any generated piece
  won't compile.

### 3.2 Host registration from a generated module

Registering at `import polars_cv` without loading the `.so` means Python needs
the names and storages as plain Python. Generate them in the way
`gen_lazy_stub.py` generates `lazy.pyi`:

- `scripts/gen_ext_types.py` writes `python/polars_cv/_ext_types.py`: one
  `BaseExtension` subclass per row, built from the FFI output.
- `polars_cv/__init__.py` registers them at import (pure Python, no `.so`).
- **Guard:** `test_ext_types_module_is_current` compares the generated module
  with `_lib.extension_types()`, so a stale generated file fails CI.
- `NUMPY_OUTPUT_SCHEMA` and the geometry `*_SCHEMA` constants become re-exports
  of the generated storages. That removes today's second copy:
  `NUMPY_OUTPUT_SCHEMA` currently restates `numpy_output_dtype()` and is pinned
  only against a literal in `test_numpy_helpers.py`.
- Remove the swallowed-exception pattern. If a name is already registered by
  someone else with a different class, raise.

### 3.3 One entry point for plugin calls

Exactly two sites call `register_plugin_function`: `_graph.py:620` (`vb_graph`)
and `_namespace.py:59` (every geometry/metadata accessor).

- Route both through one helper, `polars_cv._plugin.call(...)`. It imports
  `_lib` (which runs the Rust registration) and then delegates.
- This makes registration impossible to skip: any expression that reaches the
  plugin has registered the types first.
- **Guard (source scan, limits stated in the docstring):**
  `register_plugin_function` appears only in `_plugin.py`, with a fixture that
  must be rejected. Use `tests/_discovery.py`: `test_scans_go_through_discovery`
  enforces that.
- *Alternative considered:* a load-time constructor (`ctor` crate) that
  registers on `dlopen`, covering even a direct `register_plugin_function` call.
  It's more robust, but it's a new dependency (cargo-deny review) and relies on
  platform-specific init sections. Keep it as the fallback if §3.3 proves leaky.
- **Guard for the two-`.so` hazard:** `build_info()` / `test_version_consistency`
  fails if more than one `_lib*.so` sits in the package directory. This is the
  exact state that made the spike decay silently.

### 3.4 One unwrap at the plugin input boundary

Many existing Rust `match` arms dispatch on `DataType::Struct` / `List` /
`Binary`. An Extension dtype reaching them would either error confusingly or
hit a `_` arm.

- Strip the tag in **one place per entry point**:
  - `vb_graph` input decoding;
  - the `GeomParams` data-input path;
  - `parse_contour` / `geom_arity`.
- Stripping goes through `expect_ext` (a tagged input must carry canonical
  storage) or passes an untagged input through unchanged.
- Ops never see `DataType::Extension`.
- **Guard:** a Rust test feeds a tagged column into every `KNOWN_OPS` source
  path, reusing the per-variant probe pattern of `apply_op_coverage.rs`.

### 3.5 Input acceptance policy

| Input | Accepted? |
|---|---|
| Tagged, canonical storage | yes |
| Tagged, wrong storage | **rejected at schema resolution** (implemented in the spike) |
| Untagged, exactly canonical storage | yes. This is what an unregistered Parquet reader returns and what every existing user has |
| Untagged, "looks-like" (`points` alias, "first list field", ...) | kept through Phase 3, then **removed** with a `test_removed_surfaces.py` entry. Once tags exist, this loose matching is the silent acceptance the rules forbid |

## 4. Phases

Each phase ships on its own and leaves `main` green. Sizes are rough: S under
1 day, M 1–3 days, L over 3 days.

### Phase 0: Decide and prepare (S)
- Record the migrate-or-drop decision in `AGENTS.md` (Canonical Paths gains an
  "extension type" row).
- File upstream issues: the Python duplicate-check bug; ask whether plugin
  `dlopen` could run a registration hook.
- Keep the spike on its feature flag until Phase 1 lands, then delete it (Phase 6).

### Phase 1: Foundation (M)
- `dtype-extension` becomes a normal polars feature. The `spike-ext-types`
  feature is deleted.
- Check wheel size and debug build time against `main`, and record them in the
  PR description.
- Build §3.1 (the `ext_types!` table, `register_all`, `expect_ext`,
  `extension_types()` FFI).
- Build §3.2 (generated `_ext_types.py`, host registration at import,
  `*_SCHEMA` constants derived from it).
- Build §3.3 (`_plugin.call`, both call sites routed through it, source-scan
  guard, two-`.so` guard).
- Tests (write first, watch each fail):
  - `extension_types()` equals the generated module;
  - registration is idempotent;
  - importing `polars_cv` registers host types without loading `_lib`
    (subprocess, like the spike's `test_import_is_lazy`);
  - a geometry accessor call loads `_lib`;
  - registering a name that already belongs to a different class raises.
- No user-visible behaviour change.

### Phase 2: `sink("ndarray")`, the tagged numpy sink (M)
- New `(domain, format)` pair `("buffer", "ndarray")` → new
  `SinkKind::NdArray`. The compiler then names all four halves:
  - `dtype_for_output`: `DataType::Extension(NdArray, numpy_output_dtype())`;
  - `encode_node_output`: the same row encoding as `NumpyStruct`;
  - `null_row_result_for_spec`: tagged typed null;
  - `build_series_from_spec`: builds the struct exactly as today, then calls
    `into_extension` once per column (zero-copy relabel, no per-row cost).
  - Add the pair to the explicit correspondence table in `sink_kind.rs` tests.
- `SinkFormat.NDARRAY` in `_types.py`, then regenerate the lazy stub. The
  source/sink parameter-applicability tables follow automatically if they are
  keyed on the format.
- `numpy_from_struct` accepts a tagged input by unwrapping `.ext.storage()`, so
  there is still one reader. No new `numpy_from_ext`.
- `display.show_images` identifies an ndarray column by its tag. Keep the
  existing field sniffing for `sink("numpy")`.
- A torch helper, if added, also reads the tag. `sink("torch")` keeps its
  current untagged struct.
- Tests:
  - reference-style round trip (contiguous, transposed, flipped: the strided
    cases `numpy_from_struct` already covers);
  - null rows stay tagged nulls;
  - streaming engine plus the dual-engine lane;
  - Parquet round trip, with and without `import polars_cv`;
  - `sink("numpy")` output unchanged (byte-identical schema).
- Docs: a user-guide section on `ndarray` vs `numpy`, plus a §2.1 table of
  which struct operations need `.ext.storage()`.

**Open decision (resolve in Phase 2 review):** should `polars_cv.ndarray` carry
**metadata** (`{"dtype": "uint8", "ndim": 3}`)?
- The planner already knows `expected_dtype` / `expected_ndim` at plan time, so
  the schema could carry them. That enables type-level dispatch and checks
  before any data flows, much like `Array` does for shape.
- Cost: metadata is part of dtype equality. Columns with different dtypes stop
  concatenating, and the factory must parse and validate metadata.
- Recommendation: ship v1 without metadata and keep this as an additive v2.
  Adding metadata later is a dtype change, so decide before Phase 5.

### Phase 3: Geometry accepts tags (M)
- §3.4 at the geometry entry points. `.point`, `.contour`, `.bbox` accessors and
  the metrics subsystem accept tagged inputs (canonical storage enforced) and
  untagged canonical inputs.
- `source("auto")` / `source("contour")` accept a tagged contour column. With a
  tag, auto-detection no longer guesses from the struct's shape.
- Outputs stay untagged, so this is not a breaking change.
- Tests:
  - every accessor with tagged input (parametrize over the accessor registry,
    not a hand list);
  - wrong-storage rejection at schema resolution for each;
  - the contour raster cross-check (`test_contour_raster_crosscheck.py`) run
    with tagged inputs.

### Phase 4: Opt-in tagged geometry output (M)
- Geometry-producing paths (contour extraction `sink("native")`, bbox/point
  constructors) gain a tagged form. Prefer a **distinct sink format or
  constructor** over a global config flag:
  - a flag would have to reach every plugin call and enter op identity;
  - `CLAUDE.md` rejects per-call policy keywords that alter the output schema.
- Candidates: `sink("contour")` for tagged contours, and `polars_cv.point(x, y)`
  / `polars_cv.bbox(...)` constructors as zero-copy `.ext.to()` relabels (the
  spike's `point_ext`).
- Tests mirror Phase 2, plus metrics end-to-end on tagged inputs.

### Phase 5: Defaults (major version; separate decision)
- Decide from Phase 2–4 usage whether `sink("numpy")` and the untagged geometry
  outputs become tagged by default, or stay as the struct interchange formats.
  Recommendation: **keep both permanently**. `numpy` stays the plain-struct
  interchange format and `ndarray` is the typed one. That avoids a flag day.
- If a default flips:
  - CHANGELOG migration notes, with the §2.1 table;
  - retire the loose "looks-like" contour parsing (§3.5) with a
    `test_removed_surfaces.py` entry.

### Phase 6: Delete the spike (S, right after Phase 1)
- Remove `polars-cv/tests/spike_point_ext/` and the four `src/ext_*.rs` spike
  modules (superseded by `src/ext/`).
- Add a `test_removed_surfaces.py` entry naming the spike and pointing at `src/ext/`.

## 5. Guards summary

| Fact / property | Authority | Guard |
|---|---|---|
| A type's name and storage | `ext_types!` table | compiler (`ExtType` enum); generated-module currency test |
| Python host classes, `*_SCHEMA` constants | generated `_ext_types.py` | `test_ext_types_module_is_current` |
| Plugin-side registration before use | `_plugin.call` | source scan (only call site) + fixtures; accessor-loads-`_lib` test |
| One `.so` per package dir | `build_info()` | `test_version_consistency` |
| Wrong storage is rejected | `expect_ext` | per-type schema-resolution tests (exist in the spike) |
| Ops never see `DataType::Extension` | input-boundary unwrap | per-source-path probe test |
| Tagged sink wiring | `SinkKind::NdArray` | compiler across the four halves; correspondence table test |

Every guard gets a known-bad fixture and is watched failing for its stated
reason before it lands (Working Agreements, "Guards must be watched failing").

## 6. Risks

| Risk | Mitigation |
|---|---|
| Upstream API changes (unstable) | one module per side; pinned tests; changes in a polars bump are fixed in one place |
| Tag silently decays (unregistered reader, second `.so`, host/plugin disagreement) | host registration at import; `_plugin.call`; two-`.so` guard; untagged canonical input still accepted, so decay degrades to today's behaviour rather than an error |
| Users break on struct operations | tagging opt-in until Phase 5; §2.1 table in docs; `.ext.storage()` in examples |
| Polars 2.0 flips unknown-type default to `load_as_extension` | test both behaviours via the env var; a generic extension named `polars_cv.*` must be handled by `expect_ext` (clear error or unwrap, decided in Phase 1) |
| `dtype-extension` cost | measured in Phase 1; revert to a feature flag if material |
