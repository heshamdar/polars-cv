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
maturin develop --release  # Builds cdylib and installs into .venv
```

## Key Files

| File | Responsibility |
|------|---------------|
| `lib.rs` | PyO3 module entry, `vb_graph` expression function, `unified_output_dtype`, and the planner-facing FFI (`op_schema`, `op_contract`, `op_output_dtype`, `enum_variants`, `known_ops`) |
| `image_metadata.rs` | Header-only metadata plugin functions (`image_width`, `image_height`, `image_channels`, `image_dtype`) |
| `graph/types.rs` | `UnifiedGraph`, `GraphNode`, `OutputSpec`, `RowResult` — graph execution engine, `on_error` handling |
| `graph/decode.rs` | Source decoding, `dtype_for_output` schema inference, reflect/symmetric padding |
| `graph/encode.rs` | Output encoding, geometry op execution |
| `execute.rs` | `resolve_op()` (op-spec to `GraphStep`), decode/encode helpers shared by graph execution |
| `graph/step.rs` | `GraphStep` — the plugin-level step vocabulary: `Buffer(ViewDto)` plus graph-only steps (binary, mask, merge, geometry, reduction, histogram, extract_shape, label_reduce); contract methods read by the FFI |
| `pipeline.rs` | `SourceSpec`, `SinkSpec`, `OpSpec` serde types for JSON deserialization |
| `params.rs` | `ParamValue` — literal vs expression parameter resolution |
| `output.rs` | Numpy/torch zero-copy struct output (`NumpyRowOutput`, `build_numpy_series`) |
| `cloud.rs` | Cloud storage and HTTP file reads via `object_store` + `reqwest` |
| `contour.rs` | Contour namespace plugin functions (IoU, matching, label_reduce, bbox variants) |
| `point.rs` | Point geometry plugin functions |

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
| `image_bytes` | Decode PNG/JPEG/TIFF via `ImageAdapter` → `ViewBuffer` (alpha channels preserved) |
| `blob` | VIEW protocol binary (header + data) → `ViewBuffer` |
| `raw` | Raw bytes with explicit dtype → `ViewBuffer` |
| `file_path` | Read from local/cloud/HTTP, then decode as image (alpha channels preserved) |
| `contour` | Parse Struct column into `Contour`, optionally rasterize to mask |
| `list` / `array` | Zero-copy (when contiguous) or copy from Polars nested types |

Alpha channels are always preserved during image decoding. RGBA → `[H, W, 4]`, GrayA → `[H, W, 2]`. Each op's `ViewDto` contract exposes a channel rule that the Python planner reads for planning-time channel inference; Rust implements the corresponding behavior based on the buffer's actual channel count.

### Sink Encoding (`graph/encode.rs`)

| Sink Format | Output Type | Encoding |
|-------------|-------------|----------|
| `numpy` / `torch` | Struct | Zero-copy struct with `{data, dtype, shape, strides, offset}` |
| `png` / `jpeg` / `webp` / `tiff` | Binary | Encode `ViewBuffer` to image format bytes |
| `blob` | Binary | VIEW protocol serialization |
| `list` | List(...) | Typed nested list preserving dtype |
| `array` | Array(..., shape) | Fixed-size array preserving dtype |
| `native` | Varies | Domain-dependent: scalar → Float64, vector → List(Float64), contour → List[Struct] |

### Planning-Time Type Inference (`unified_output_dtype`)

Runs at Polars planning time (NOT execution time). Parses the graph JSON, resolves `auto` dtypes, and returns the Polars `Field` with the correct output `DataType`.

**Critical invariant:** The dtype returned here MUST match what `execute_graph` produces. Divergence causes Polars errors at collect time.

## Contour/Bbox Plugin Functions

`point.rs` and `contour.rs` expose direct plugin expression functions for `.point` / `.contour` / `.bbox` namespaces. These **bypass `vb_graph`** and operate on Struct/List columns directly.

Key functions in `contour.rs`:
- `contour_pairwise_iou`, `contour_match_detections`, `contour_label_reduce`
- `bbox_pairwise_iou`, `bbox_match_detections` (convert bboxes to rectangular contours internally)
- Graph-side `label_reduce` in `resolve_op` (buffer + contour expression parameter → vector)

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
