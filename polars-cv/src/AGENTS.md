# AGENTS.md — Rust Plugin (`polars-cv/src/`)

> Read the [root AGENTS.md](../../AGENTS.md) first for project-wide context.
> Update this file when you change graph execution, source/sink handling, parameter resolution, or the plugin function interface.

## Purpose

The **Rust plugin layer** bridging the Python API with `view-buffer`. Responsible for:

- **Graph execution** — parsing JSON pipeline graphs and executing them topologically
- **Source decoding** — converting Polars Series data into `ViewBuffer` instances
- **Sink encoding** — converting `ViewBuffer` results back into Polars-compatible output
- **Parameter resolution** — resolving literal vs expression parameters per-row
- **Output dtype inference** — determining Polars output types at planning time
- **Cloud I/O** — reading from S3, GCS, Azure, HTTP/HTTPS

**This is execution-time code.** It runs when Polars evaluates the expression, not when the user constructs the pipeline.

## Build

```bash
cd polars-cv
maturin develop            # Builds cdylib (debug) and installs into .venv
```

Debug, not `--release`: it is what CI and `scripts/verify.sh` build, every test
passes against it, and `--release` re-optimises the whole polars stack for
several minutes. Reach for `--release` only when benchmarking.

## Key Files

| File | Responsibility |
|------|---------------|
| `lib.rs` | PyO3 module entry, `vb_graph` expression function, `unified_output_dtype`, and the planner-facing FFI (`op_schema`, `op_contract`, `binary_output_dtype`, `enum_variants`, `known_ops`) |
| `image_metadata.rs` | Header-only metadata plugin functions (`image_width`, `image_height`, `image_channels`, `image_dtype`) |
| `fetch.rs` | Stage one of every path-based read: path column → bytes (`prefetch`, `row_bytes`, `parse_on_error`). Shared by the `file_path` source and `read_bytes.rs`; owns `PathPolicy`, the `allowed_roots` sandbox both of them check against. Fetch concurrency is **not** a knob here — it is polars' process-wide `POLARS_CONCURRENCY_BUDGET` semaphore, taken one permit per request in `cloud.rs` |
| `read_bytes.rs` | `read_file_bytes` plugin function — `fetch.rs` with the decode omitted, for byte-identical passthrough |
| `graph/types.rs` | `UnifiedGraph`, `GraphNode`, `OutputSpec`, `RowResult` — graph execution engine, `on_error` handling |
| `graph/compiled.rs` | `CompiledGraph` — process-wide compiled-graph cache (parsed spec, topo order, slot-bound params) |
| `graph/decode.rs` | Source decoding, `dtype_for_output` schema inference, reflect/symmetric padding |
| `graph/encode.rs` | Output encoding, geometry op execution |
| `engine_warning.rs` | One-time single-threaded-batch warning pointing users to `engine="streaming"` (env: `POLARS_CV_SILENCE_ENGINE_WARNING`, `POLARS_CV_ENGINE_WARN_ROWS`) |
| `execute.rs` | `resolve_op()` (op-spec to `GraphStep`), decode/encode helpers shared by graph execution |
| `graph/step.rs` | `GraphStep` — the plugin-level step vocabulary: `Buffer(ViewDto)` plus graph-only steps (binary, mask, merge, geometry, reduction, histogram, perceptual_hash, extract_shape, label_reduce); contract methods read by the FFI |
| `pipeline.rs` | `SourceSpec`, `SinkSpec`, `OpSpec` serde types for JSON deserialization |
| `params.rs` | `ParamValue` — literal vs expression parameter resolution (`resolve_*` numerics, `resolve_str`/`resolve_bool` for non-structural enums and flags, `req_*_literal` for structural ones). `ParamCtx::probe` marks the plan-time shape probe, where every expression param is bound to an integer placeholder and the enum/flag accessors fall back to their default; real execution stays strict |
| `output.rs` | Numpy/torch zero-copy struct output (`NumpyRowOutput`, `build_numpy_series`) |
| `cloud.rs` | Cloud storage and HTTP file reads via `object_store` + `reqwest` |
| `cloud_auth.rs` | Bearer-token sourcing for the OAuth backends (GCS/Azure): federated ADC delegation to `gcloud`, `token_command`, expiry-aware caching |
| `contour.rs` | Contour namespace plugin functions (IoU, matching, label_reduce, bbox variants) |
| `point.rs` | Point geometry plugin functions |
| `geom_params.rs` | `GeomParams` — per-row parameter resolution for those namespace functions, reading expression params off the extra inputs named in the graph-free `input_slots` map |

**The graph wire format is closed, struct by struct.** `GraphNode`,
`UnifiedGraph`, `OutputSpec`, `SourceSpec`, `SinkSpec` and `GraphKwargs` each
carry `#[serde(deny_unknown_fields)]`, so anything Python sends must be declared
on the Rust struct — including `domain`/`output_dtype`, which only the Python
visualizer consumes. It was permissive before, which is how node-level
`shape_hints` went on being serialized long after the last reader was removed,
costing a distinct compiled-graph cache entry for pipelines that execute
identically (`graph_json` is the cache key).

**Each of them needs its own attribute**: serde's `deny_unknown_fields` does not
descend into nested types. Closing `GraphNode` alone left everything it holds
wide open, which mattered most for `SourceSpec` — it carries `allowed_roots`, so
a misspelled key deserialized to `None`, i.e. no path sandbox, silently. Adding
a struct to the wire format means adding the attribute to it too; the node's
being closed says nothing about its children.

`OpSpec` cannot be closed the same way: its parameters ride on
`#[serde(flatten)]`, which serde documents as incompatible with
`deny_unknown_fields`. Op names are guarded by the registry-parity tests and
`resolve_op`'s catch-all instead.


## Core Architecture

### The `vb_graph` Expression

Single entry point for all pipeline execution, registered via `#[polars_expr]`:

- `inputs[0..n]` — source column(s)
- `inputs[n..]` — expression parameter columns (dynamic per-row values)
- `kwargs.graph_json` — JSON-serialized `UnifiedGraph`
- `kwargs.expr_column_names` — names mapping expression columns

Returns a single `Series` (typed column for single output, Struct for multi-output).

### Graph Execution Flow

Execution is split into a cacheable **compile** phase and a per-call phase
(`graph/compiled.rs`):

1. `get_or_compile()` fetches a `CompiledGraph` from a process-wide cache
   keyed by `(graph_json, expr_column_names)` — hash plus full string
   equality, never the hash alone. On miss, `CompiledGraph::compile()`
   parses the JSON, computes the topological order, binds every
   `ParamValue::Expr` to an input slot (`ParamValue::Slot`), and resolves
   all-literal ops once (`OpResolver::Static`); ops with dynamic params
   re-resolve per row through typed slot reads (`OpResolver::Dynamic`).
2. Per call: build `ParamCtx` (typed accessors over the input series) and
   `resolved_output_specs` (per-call `"auto"` dtype/ndim resolution from the
   input column type — shared with plan-time `unified_output_dtype`).
3. Per-row: decode sources → apply operations (ViewExpr chain) → encode outputs
4. Collect row results into the appropriate Polars Series type

**Cache-safety invariant:** `CompiledGraph` must contain nothing derived from
the data (no input dtypes, shapes, row counts, or null masks) — one cached
graph must behave identically across heterogeneous inputs. See the module
docs in `graph/compiled.rs` and `tests/test_graph_cache.py`.

### `resolve_op` — Operation Dispatcher

Located in `execute.rs`. Maps operation name strings to `GraphStep` values
(`graph/step.rs`). Buffer ops wrap a view-buffer `ViewDto`
(`GraphStep::Buffer`); steps that involve graph topology (node references,
per-row expression columns) or non-buffer outputs are their own `GraphStep`
variants and never enter view-buffer's vocabulary:

```rust
match op_spec.op.as_str() {
    "resize" => ViewDto::Image(ImageOp { kind: Resize { ... } }).into(),
    "grayscale" | "normalize" | "threshold" => /* ... */,
    "channel_select" | "channel_swap" => /* ... */,
    "cvt_color" => ViewDto::Color(ColorConvertOp { ... }),
    "convolve2d" => ViewDto::Filter(ConvolveOp { ... }),
    "canny" | "equalize_histogram" => /* ... */,
    "erode" | "dilate" | "morphology_gradient" => /* ... */,
    "rotate" => /* 90/180/270 → ViewOp::Rotate{N}, arbitrary → ComputeOp::RotateAffine */,
    "warp_affine" => ViewDto::Compute(ComputeOp::Affine(AffineParams { ... })),
    // ... all supported operations
}
```

**Rotation dispatch:** `rotate` uses zero-copy `ViewOp::Rotate90/180/270` for exact multiples of 90 degrees. All other angles (including 0/360) are routed through `ComputeOp::RotateAffine`, which constructs `AffineParams` at execution time via `AffineParams::from_rotation()` and delegates to `apply_affine_warp()`. The separate `ImageOpKind::Rotate` variant has been removed.

### Source Decoding (`graph/decode.rs`)

| Source Format | Decoding |
|---------------|----------|
| `auto` (the Python default) | Resolved to a concrete format below by `resolve_auto_format` (`graph/compiled.rs`) from the column dtype: String → `file_path`, List/Array → `list`/`array`, Binary → `blob` if the bytes start with `protocol::MAGIC_BYTES` else `image_bytes`. Resolved once per batch (dtype is row-invariant), not per row; an unroutable dtype errors |
| `image_bytes` | Decode PNG/JPEG/TIFF via `ImageAdapter` → `ViewBuffer` (alpha channels preserved) |
| `blob` | VIEW protocol binary (header + data) → `ViewBuffer` |
| `raw` | Raw bytes with explicit dtype → `ViewBuffer` |
| `file_path` | Two stages: `fetch.rs` reads the bytes from local/cloud/HTTP, then they decode as `image_bytes` (alpha channels preserved). The fetch stage is also exposed on its own as `.cv.read_bytes()` (`read_bytes.rs`) — same code, decode omitted |
| `contour` | Parse geometry into `Contour`s and rasterize to a mask. `parse_contour_set` (`contour.rs`) accepts either shape the column takes — one contour per row (`Struct`) or the whole set (`List(Struct)`, what `extract_contours().sink("native")` emits) — dispatching on the list's *element dtype*, since a `List` of point structs is one contour's ring. The set paints as a union via `geometry::rasterize::rasterize`, the same call the `rasterize` op makes |
| `list` / `array` | Zero-copy (when contiguous) or copy from Polars nested types |

Alpha channels are always preserved during image decoding. RGBA → `[H, W, 4]`, GrayA → `[H, W, 2]`. Each op's `ViewDto` contract exposes a channel rule that the Python planner reads for planning-time channel inference; Rust implements the corresponding behavior based on the buffer's actual channel count.

### Sink Encoding (`graph/encode.rs`)

| Sink Format | Output Type | Encoding |
|-------------|-------------|----------|
| `numpy` / `torch` | Struct | Zero-copy struct with `{data, dtype, shape, strides, offset}` |
| `png` / `jpeg` / `webp` / `tiff` | Binary | Encode `ViewBuffer` to image format bytes. `encode_sink` (`execute.rs`) rejects non-`u8` for JPEG/WebP before encoding; PNG carries bit depth through (`ImageAdapter::to_dynamic_image` builds 8-bit variants for `U8`, 16-bit for `U16`, and rejects anything else); TIFF handles float |
| `blob` | Binary | VIEW protocol serialization |
| `list` | List(...) | Typed nested list preserving dtype |
| `array` | Array(..., shape) | Fixed-size array preserving dtype |
| `native` | Varies | Domain-dependent: scalar → Float64, vector → List(Float64), contour → List[Struct] |

### Planning-Time Type Inference (`unified_output_dtype`)

Runs at Polars planning time (NOT execution time). Parses the graph JSON, resolves `auto` dtypes, and returns the Polars `Field` with the correct output `DataType`.

**Critical invariant:** The dtype returned here MUST match what `execute_graph` produces. Divergence causes Polars errors at collect time.

## Contour/Bbox Plugin Functions

`point.rs` and `contour.rs` expose direct plugin expression functions for `.point` / `.contour` / `.bbox` namespaces. These **bypass `vb_graph`** and operate on Struct/List columns directly. `image_metadata.rs` and `read_bytes.rs` bypass it the same way for the `.cv` namespace's non-pipeline methods.

`tests/test_sanitation.py::test_namespace_plugin_symbols_match_registrations` pins both directions of this surface — add any new module carrying a namespace `#[polars_expr]` to the file list it scans, or the symbol silently escapes the check.

Because they bypass `vb_graph`, these functions get no `ParamCtx`. Their per-row parameters ride in as **extra input series**: Python's `_ArgBinder` appends each expression-valued parameter to the inputs and records its name in an `input_slots` name→index map inside the kwargs, which `geom_params.rs` reads back through the same `ParamCol` accessors `params.rs` uses. Names, not positions — several of these functions already take optional data operands (`scores`, `origin`) positionally, where an appended parameter would be indistinguishable from an omitted operand. `GeomParams` rejects an out-of-range index and a map that does not account for every input, so a binder/reader drift fails loudly instead of silently dropping an operand.

Key functions in `contour.rs`:
- `contour_pairwise_iou`, `contour_match_detections`, `contour_label_reduce`
- `bbox_pairwise_iou`, `bbox_match_detections` — rectangle overlap is a two-interval
  intersection, so these stay analytic (`pairwise::bbox_iou`) rather than going
  through general polygon boolean ops. Both share `match_from_matrix` with the
  contour matcher, so the greedy matching policy lives in one place.
- Graph-side `label_reduce` in `resolve_op` (buffer + contour expression parameter → vector)

`contour_label_reduce` and the graph-side `label_reduce` are two entry points onto
**one** implementation: both call `view_buffer::geometry::label::score_contours_on_buffer`
and parse their `reduction`/`region_mode` against that module's `NAMED` tables. Do not
reintroduce a plugin-local scorer — the previous one drifted into a second dialect with
no `boundary` mode and no centroid fallback, and the only parity test covered `bbox`,
the one mode where the two happened to agree.

## Module Notes

### `execute.rs`

Current responsibilities: `resolve_op()` (returns `GraphStep`), `decode_source()`, `decode_contour_source()`, `encode_sink()`. These are shared utilities used by `graph/types.rs` and `graph/encode.rs`.

### `pipeline.rs`

Contains serde types (`SourceSpec`, `SinkSpec`, `OpSpec`) for JSON deserialization. The graph system uses them via `GraphNode`/`OutputSpec`; the decode/encode helpers take `&SourceSpec`/`&SinkSpec` directly (the old `PipelineSpec` wrapper was removed).

## Adding a New Operation (Rust Side)

1. **`view-buffer`**: Implement the op — see [`view-buffer/AGENTS.md`](../../view-buffer/AGENTS.md)
2. **`execute.rs` → `resolve_op()`**: Add a match arm mapping the operation name to a `GraphStep` (`GraphStep::Buffer(dto)` for engine ops)
3. **Test**: Ensure the operation works end-to-end via Python tests

## Error Handling

- Rust panics in `view-buffer` are caught by `std::panic::catch_unwind` and converted to `PolarsResult::Err`
- Source decoding errors produce `polars_err!(ComputeError: ...)` with descriptive messages
- Null inputs produce null outputs (null propagation)
- `on_error="null"` on source spec: decode errors produce `None` for that node instead of propagating (parsed once at compile into `CompiledGraph::source_null_nodes`)
- Graph-level `RowErrorPolicy` (`graph.on_error`: `raise` | `null` | `null_with_message`): any `Result` error while producing a row either fails the expression (raise), nulls all of that row's outputs (null), or additionally records the message in a reserved `_error: String` struct field (null_with_message — forces struct output even for single-output graphs; `unified_output_dtype` mirrors this so plan==exec). Set from Python via `Pipeline.on_error()`. Panics are not covered by the policy — they abort the batch via `catch_unwind`.
- `NullParamPolicy` (`params.rs`; `graph.on_null_param`: `raise` | `null`) — a **null in a per-row expression parameter column**, which is not the same thing as an error. It is a shared mechanism, not per-op: every null reaches `ParamCol::on_null`, the only caller of the null error, which flags the `ParamCtx` (`null_hit: Cell<bool>`) under `Null` and always returns `Err` so resolution short-circuits with no placeholder value reaching an op. Three sites in `compiled.rs` clear the flag before a fallible resolution and test it after — dynamic op resolution, contour-source `resolve_fill`, shape-ref rasterize — and turn a flagged error into `continue 'nodes`, leaving the node out of `node_outputs`. That is the *existing* null-propagation path (`source(on_error="null")`), so nulling is **node-scoped**: only outputs depending on that node go null. Set from Python via `Pipeline.on_null_param()`; the geometry namespaces get it as an `on_null` kwarg applied by `GeomParams::row`. Independent of `RowErrorPolicy`, so it records no `_error` message and does not weaken any other error reporting.
- `CompiledGraph::operand` distinguishes "node is in the graph but produced no output for this row" (→ null this node too) from "node is not in the graph" (→ error), so a null upstream propagates instead of raising "references unknown node". **Every cross-node read of `node_outputs` must go through it** — there are five (`Binary`, `ApplyMask`, `ChannelMerge`, the rasterize shape ref, and the contour source's `shape=` lookup in the decode closure, which returns `Ok(None)` rather than `continue 'nodes`). Enumerating the sites is exactly how one got missed the first time; grep for `node_outputs.get(` when adding a step that reads another node.
- One exception, and it is not a parameter: `GraphStep::LabelReduce` reads its *contours operand* by column name through `ParamCol::get_any` and maps a null to an **empty score vector**, not a null. That is a data operand with pre-existing semantics, deliberately left alone — `get_any` is the one accessor with no `on_null` path.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `view-buffer` | Core tensor engine (path dependency `../view-buffer`) |
| `polars` | DataFrame operations, Series types |
| `pyo3` / `pyo3-polars` | Python bindings, `#[polars_expr]` derive macro |
| `image` | Image decoding/encoding |
| `object_store` | Cloud storage (S3, GCS, Azure) |
| `reqwest` | HTTP file fetching |
| `serde` / `serde_json` | JSON graph deserialization |
| `tokio` | Async runtime for cloud/HTTP ops |
