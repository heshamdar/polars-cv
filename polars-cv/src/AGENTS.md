# AGENTS.md — Rust Plugin (`polars-cv/src/`)

> Read the [root AGENTS.md](../../AGENTS.md) first for project-wide context.
> Update this file when you change graph execution, source/sink handling, parameter resolution, or the plugin function interface.

## Purpose

This is the **Rust plugin layer** that bridges the Python API with the `view-buffer` engine. It is responsible for:

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
maturin develop --release  # Builds and installs into the .venv
```

The crate produces a `cdylib` (`_lib.abi3.so` / `_lib.pyd`) that Polars loads as a plugin.

## Key Files

| File | Responsibility |
|------|---------------|
| `lib.rs` | PyO3 module entry, `vb_graph` expression function, `unified_output_dtype`, tiling config, dtype helpers |
| `image_metadata.rs` | Header-only image metadata plugin functions (`image_width`, `image_height`, `image_channels`, `image_dtype`) — uses `image` crate `ImageDecoder` trait and VIEW protocol header parsing |
| `graph/mod.rs` | Module re-exports for the graph system |
| `graph/types.rs` | `UnifiedGraph`, `GraphNode`, `OutputSpec`, `RowResult` — the main graph execution engine with topological sort and per-row processing; `on_error` handling via inner closure around source decode |
| `graph/decode.rs` | Source decoding — binary, blob, list/array (zero-copy), raw bytes, contour; also `dtype_for_output` for schema inference; reflect/symmetric padding implementation |
| `graph/encode.rs` | Output encoding — binary, scalar, vector, contour, typed list/array, numpy struct; also geometry op execution |
| `execute.rs` | Shared execution utilities: op resolver (`resolve_op`) + source/sink decode/encode helpers used by graph execution |
| `pipeline.rs` | `PipelineSpec`, `SourceSpec` (with `on_error` field), `SinkSpec` serde types. Used by graph system. |
| `params.rs` | `ParamValue` — literal vs expression parameter resolution |
| `output.rs` | Numpy/torch zero-copy struct output (`NumpyRowOutput`, `build_numpy_series`) |
| `cloud.rs` | Cloud storage (S3, GCS, Azure) and HTTP file reads via `object_store` + `reqwest` |
| `contour.rs` | Contour namespace plugin functions (scalar + set-level primitives, contour parsing/serialization) |
| `point.rs` | Point geometry plugin functions |

## Core Architecture

### The `vb_graph` Expression

This is the **single entry point** for all pipeline execution. It is registered as a Polars expression function via `#[polars_expr]`:

```rust
#[polars_expr(output_type_func_with_kwargs=unified_output_dtype)]
fn vb_graph(inputs: &[Series], kwargs: GraphKwargs) -> PolarsResult<Series> {
    execute_graph(inputs, &kwargs)
}
```

**Inputs:**
- `inputs[0..n]` — source column(s) (the data being processed)
- `inputs[n..]` — expression parameter columns (dynamic per-row values)
- `kwargs.graph_json` — JSON-serialized `UnifiedGraph`
- `kwargs.expr_column_names` — names mapping expression columns

**Output:** A single `Series` — either a typed column (Binary, Float64, List, etc.) for single output, or a Struct column for multi-output.

### Graph Execution Flow

1. **Parse:** `UnifiedGraph::from_json(&graph_json)` deserializes the JSON graph
2. **Resolve auto types:** If output specs have `expected_dtype == "auto"`, resolve from input column type (only for List/Array sources; Binary/String sources stay `auto`)
3. **Topological sort:** `execution_order()` computes processing order
4. **Per-row execution:** For each row in the input:
   a. **Decode sources** — root nodes decode their input column data into `ViewBuffer`
   b. **Apply operations** — each node builds a `ViewExpr` chain from its ops, then executes
   c. **Encode outputs** — terminal nodes encode results based on sink format
5. **Build Series:** Collect all row results into the appropriate Polars Series type

### `resolve_op` — The Operation Dispatcher

Located in `execute.rs`, this function maps operation names to `ViewDto` values:

```rust
match op_spec.op.as_str() {
    "resize" => { /* extract params, build ViewDto::Image(ImageOp { ... }) */ }
    "grayscale" => { /* ... */ }
    "normalize" => { /* ... */ }
    "channel_select" => { /* index -> ViewDto::View(ViewOp::ChannelSelect { index }) */ }
    "channel_swap" => { /* order -> ViewDto::ChannelSwap { order } */ }
    "channel_merge" => { /* other_node_ids -> ViewDto::ChannelMerge { other_node_ids } */ }
    "adjust_contrast" => { /* factor -> ViewDto::Compute(ComputeOp::AdjustContrast(factor)) */ }
    "adjust_gamma" => { /* gamma -> ViewDto::Compute(ComputeOp::AdjustGamma(gamma)) */ }
    "invert" => { /* -> ViewDto::Compute(ComputeOp::Invert) */ }
    "cvt_color" => { /* from_space, to_space -> ViewDto::Color(ColorConvertOp { from, to }) */ }
    "convolve2d" => { /* kernel, ksize, normalize, border -> ViewDto::Filter(ConvolveOp { ... }) */ }
    "canny" => { /* low_threshold, high_threshold -> ViewDto::Image(ImageOp { kind: Canny { ... } }) */ }
    "equalize_histogram" => { /* -> ViewDto::Image(ImageOp { kind: HistogramEqualize }) */ }
    // ... all supported operations
}
```

The legacy row-by-row pipeline executor was removed; `execute.rs` now only contains shared helpers used by the unified graph path.

### Source Decoding (`graph/decode.rs`)

| Source Format | Decoding |
|---------------|----------|
| `image_bytes` | Decode PNG/JPEG/TIFF via `ImageAdapter` → `ViewBuffer` |
| `blob` | VIEW protocol binary (header + data) → `ViewBuffer` |
| `raw` | Raw bytes with explicit dtype → `ViewBuffer` |
| `file_path` | Read from local/cloud/HTTP, then decode as image |
| `contour` | Parse Struct column into `Contour`, optionally rasterize to mask |
| `list` / `array` | Zero-copy (when contiguous) or copy from Polars nested types → `ViewBuffer` |

### Sink Encoding (`graph/encode.rs`)

| Sink Format | Output Type | Encoding |
|-------------|-------------|----------|
| `numpy` / `torch` | Struct | Zero-copy struct with `{data, dtype, shape, strides, offset}` |
| `png` / `jpeg` / `webp` / `tiff` | Binary | Encode `ViewBuffer` to image format bytes |
| `blob` | Binary | VIEW protocol serialization |
| `list` | List(...) | Typed nested list preserving dtype |
| `array` | Array(..., shape) | Fixed-size array preserving dtype |
| `native` | Varies | Domain-dependent: scalar → Float64, vector → List(Float64), contour → List[Struct], histogram → List[Struct] (buckets) |

### Planning-Time Type Inference (`unified_output_dtype`)

This function runs at Polars planning time (NOT execution time). It:
1. Parses the graph JSON
2. Resolves `auto` dtype from input field types
3. Returns the Polars `Field` with the correct output `DataType`

**Critical invariant:** The dtype returned here MUST match what `execute_graph` actually produces. If they diverge, Polars will error at collect time.

## Contour Namespace Plugin Path

`point.rs` and `contour.rs` expose direct plugin expression functions used by
Python `.point` / `.contour` namespaces. These bypass `vb_graph` and operate on
Struct/List columns directly.

Recent additions for detection workflows in `contour.rs`:
- `contour_pairwise_iou` (`List[Contour] x List[Contour] -> List[List[f64]]`)
- `contour_match_detections` (greedy one-to-one matching with deterministic ties)
- `contour_label_reduce` (per-contour scoring from image/array values)
- `bbox_pairwise_iou` (`List[BBOX_SCHEMA] x List[BBOX_SCHEMA] -> List[List[f64]]`)
- `bbox_match_detections` (greedy matching for axis-aligned bounding boxes)

The bbox functions convert bounding boxes to rectangular contours internally and
delegate to the existing contour IoU/matching logic. A `TODO` exists in
`view-buffer/src/geometry/pairwise.rs` for a future direct axis-aligned
optimization.

Recent graph-side primitive additions:
- `label_reduce` in `resolve_op` / graph execution (`buffer -> vector`)
- Uses a buffer input plus contour expression parameter (`contours=pl.col(...)`)
- Produces score lists equivalent to `.contour.label_reduce(...)` for the same inputs

When updating these, keep null propagation and nested list/struct parsing behavior explicit.

## Legacy Code

### `execute.rs`

This module no longer contains the old row-by-row pipeline executor.
Current responsibilities:
- `resolve_op()` (op-spec → `ViewDto`)
- `decode_source()`, `decode_contour_source()`, `decode_contour_source_with_dims()`
- `encode_sink()` for image/blob sink byte encoding

These are shared by `graph/types.rs` and `graph/encode.rs`.

### `pipeline.rs`

Contains serde types (`PipelineSpec`, `SourceSpec`, `SinkSpec`, `OpSpec`) used for JSON deserialization. The graph system (`UnifiedGraph`) uses `SourceSpec`, `SinkSpec`, and `OpSpec` from this module via its `GraphNode` type. So the types themselves are still needed, but the `PipelineSpec` wrapper may be removable once the old execution path is gone.

## Adding a New Operation (Rust Side)

1. **`view-buffer`**: Implement the op — see [`view-buffer/AGENTS.md`](../../view-buffer/AGENTS.md)
2. **`execute.rs` → `resolve_op()`**: Add a match arm mapping the operation name to the corresponding `ViewDto`
3. **Test**: Ensure the operation works end-to-end via Python tests

## Error Handling

- Rust panics in `view-buffer` operations are caught by `std::panic::catch_unwind` in the execution loop and converted to `PolarsResult::Err`
- Source decoding errors produce `polars_err!(ComputeError: ...)` with descriptive messages
- Null inputs produce null outputs (null propagation)

## Dependencies

| Crate | Purpose |
|-------|---------|
| `view-buffer` | Core tensor engine (path dependency `../view-buffer`) |
| `polars` | DataFrame operations, Series types |
| `pyo3` | Python bindings |
| `pyo3-polars` | `#[polars_expr]` derive macro |
| `image` | Image decoding/encoding |
| `object_store` | Cloud storage (S3, GCS, Azure) |
| `reqwest` | HTTP file fetching |
| `serde` / `serde_json` | JSON graph deserialization |
| `tokio` | Async runtime for cloud/HTTP ops |
| `thiserror` | Error types |
