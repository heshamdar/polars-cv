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
| `lib.rs` | PyO3 module entry, `vb_graph` expression function, `unified_output_dtype`, dtype helpers |
| `image_metadata.rs` | Header-only metadata plugin functions (`image_width`, `image_height`, `image_channels`, `image_dtype`) |
| `graph/types.rs` | `UnifiedGraph`, `GraphNode`, `OutputSpec`, `RowResult` — graph execution engine, `on_error` handling |
| `graph/decode.rs` | Source decoding, `dtype_for_output` schema inference, reflect/symmetric padding |
| `graph/encode.rs` | Output encoding, geometry op execution |
| `execute.rs` | `resolve_op()` (op-spec to `ViewDto`), decode/encode helpers shared by graph execution |
| `pipeline.rs` | `PipelineSpec`, `SourceSpec`, `SinkSpec`, `OpSpec` serde types for JSON deserialization |
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

1. `UnifiedGraph::from_json()` deserializes the JSON graph
2. Resolve `auto` dtypes from input column types where possible
3. `execution_order()` computes topological processing order
4. Per-row: decode sources → apply operations (ViewExpr chain) → encode outputs
5. Collect row results into the appropriate Polars Series type

### `resolve_op` — Operation Dispatcher

Located in `execute.rs`. Maps operation name strings to `ViewDto` values:

```rust
match op_spec.op.as_str() {
    "resize" => ViewDto::Image(ImageOp { kind: Resize { ... } }),
    "grayscale" | "normalize" | "threshold" => /* ... */,
    "channel_select" | "channel_swap" => /* ... */,
    "cvt_color" => ViewDto::Color(ColorConvertOp { ... }),
    "convolve2d" => ViewDto::Filter(ConvolveOp { ... }),
    "canny" | "equalize_histogram" => /* ... */,
    // ... all supported operations
}
```

### Source Decoding (`graph/decode.rs`)

| Source Format | Decoding |
|---------------|----------|
| `image_bytes` | Decode PNG/JPEG/TIFF via `ImageAdapter` → `ViewBuffer` (alpha channels preserved) |
| `blob` | VIEW protocol binary (header + data) → `ViewBuffer` |
| `raw` | Raw bytes with explicit dtype → `ViewBuffer` |
| `file_path` | Read from local/cloud/HTTP, then decode as image (alpha channels preserved) |
| `contour` | Parse Struct column into `Contour`, optionally rasterize to mask |
| `list` / `array` | Zero-copy (when contiguous) or copy from Polars nested types |

Alpha channels are always preserved during image decoding. RGBA → `[H, W, 4]`, GrayA → `[H, W, 2]`. The `AlphaMode` contract in Python documents how each operation handles alpha; Rust implements the corresponding behavior based on the buffer's actual channel count.

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

Current responsibilities: `resolve_op()`, `decode_source()`, `decode_contour_source()`, `encode_sink()`. These are shared utilities used by `graph/types.rs` and `graph/encode.rs`.

### `pipeline.rs`

Contains serde types (`PipelineSpec`, `SourceSpec`, `SinkSpec`, `OpSpec`) for JSON deserialization. The graph system uses `SourceSpec`, `SinkSpec`, and `OpSpec` via `GraphNode`. The `PipelineSpec` wrapper itself may be removable.

## Adding a New Operation (Rust Side)

1. **`view-buffer`**: Implement the op — see [`view-buffer/AGENTS.md`](../../view-buffer/AGENTS.md)
2. **`execute.rs` → `resolve_op()`**: Add a match arm mapping the operation name to `ViewDto`
3. **Test**: Ensure the operation works end-to-end via Python tests

## Error Handling

- Rust panics in `view-buffer` are caught by `std::panic::catch_unwind` and converted to `PolarsResult::Err`
- Source decoding errors produce `polars_err!(ComputeError: ...)` with descriptive messages
- Null inputs produce null outputs (null propagation)
- `on_error="null"` on source spec wraps decode in an inner closure; errors produce `None` instead of propagating

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
