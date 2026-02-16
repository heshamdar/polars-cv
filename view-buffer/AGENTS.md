# AGENTS.md — view-buffer (Core Tensor Engine)

> Read the [root AGENTS.md](../AGENTS.md) first for project-wide context.
> Update this file when you change ViewBuffer, ViewExpr, operations, execution planning, or interop.

## Purpose

`view-buffer` is a **zero-copy, stride-aware tensor framework** for Rust. It is the computational engine that powers polars-cv. All actual image/array processing happens here.

While originally designed as an independent crate (published to crates.io), it is currently **tightly coupled** to polars-cv in practice. Agents working on either layer typically need context from both.

### What This Crate Does

- Provides `ViewBuffer` — a strided multi-dimensional array that can represent images, masks, feature maps, etc.
- Implements zero-copy view operations (transpose, flip, crop, reshape) that only modify metadata
- Implements compute operations (scale, normalize, cast, clamp, relu)
- Implements image operations (resize, blur, grayscale, threshold, rotate)
- Implements geometry operations (contour extraction, rasterization, measures)
- Provides `ViewExpr` — a lazy expression graph for building pipelines before execution
- Provides `ExecutionPlan` — optimized execution with kernel fusion
- Provides interop with Arrow, ndarray, image, and Polars-arrow

### What This Crate Does NOT Do

- No Python bindings (those are in `polars-cv`)
- No Polars expression registration
- No JSON graph parsing (that's in `polars-cv/src/graph/`)
- No cloud I/O

## Module Structure

```
src/
├── lib.rs              # Crate root, re-exports
├── core/               # Fundamental types
│   ├── buffer.rs       # ViewBuffer struct — the core data type
│   ├── dtype.rs        # DType enum (U8, I8, U16, ..., F64)
│   ├── layout.rs       # Layout (shape, strides, offset, dtype)
│   └── mod.rs
├── ops/                # Operations
│   ├── mod.rs          # ViewOp, ComputeOp, BinaryOp, ViewDto
│   ├── dto.rs          # ViewDto — serializable operation enum
│   ├── image.rs        # ImageOp, ImageOpKind (resize, blur, grayscale, etc.)
│   ├── compute.rs      # ComputeOp (cast, scale, normalize, clamp, relu)
│   ├── binary.rs       # BinaryOp (add, subtract, multiply, blend, bitwise)
│   ├── histogram.rs    # Histogram computation
│   ├── affine.rs       # Affine transform parameters
│   ├── io.rs           # SourceFormat, SinkFormat
│   ├── cost.rs         # OpCost, OpCostReport for allocation analysis
│   └── mod.rs          # Op trait, ViewOp enum, trait impls
├── expr.rs             # ViewExpr — lazy expression graph builder (~42K)
├── execution/          # Execution engine
│   ├── mod.rs          # Re-exports
│   ├── plan.rs         # ExecutionPlan, PlanStep — compiled execution plan
│   ├── runner.rs       # Execute plans against ViewBuffers
│   └── tiling.rs       # Tiling configuration (currently no-op, see notes)
├── geometry/           # Geometry operations
│   ├── mod.rs          # Contour, Point, BoundingBox types
│   ├── contour.rs      # Contour struct and basic operations
│   ├── extract.rs      # Contour extraction from binary masks
│   ├── rasterize.rs    # Rasterize contours to binary masks
│   ├── measures.rs     # Area, perimeter, centroid
│   ├── transforms.rs   # Translate, scale, rotate contours
│   ├── ops.rs          # GeometryOp enum
│   ├── pairwise.rs     # Pairwise operations (IoU, distance)
│   └── predicates.rs   # Spatial predicates (contains, intersects)
├── protocol.rs         # VIEW binary protocol (header + data serialization)
└── interop/            # External library integration
    ├── mod.rs          # ExternalView, validate_layout
    ├── arrow.rs        # Arrow buffer interop (FromArrow, ToArrow)
    ├── arrow_ffi.rs    # Arrow FFI support
    ├── image.rs        # image crate interop (ImageAdapter, decode/encode)
    ├── ndarray.rs      # ndarray interop (AsNdarray, FromNdarray)
    └── polars.rs       # Polars-arrow interop
```

## Core Concepts

### ViewBuffer

The fundamental data type. A strided multi-dimensional array backed by either a Rust `Vec` or an Arrow buffer.

Key properties:
- **Shape:** `[height, width, channels]` for images, arbitrary for other data
- **Strides:** Byte strides per dimension — enables zero-copy transpose, flip, crop
- **DType:** Element type (U8, F32, etc.)
- **Offset:** Byte offset into the backing buffer

### ViewExpr (Lazy Expression Graph)

A builder for constructing operation pipelines lazily:

```rust
let expr = ViewExpr::new_source(buffer)
    .resize(224, 224, FilterType::Lanczos3)
    .normalize(NormalizeMethod::MinMax, None, None)
    .cast(DType::F32);

let result = expr.plan().execute();
```

Methods on `ViewExpr` correspond to operations. Each method appends a node to the expression graph. `plan()` compiles it into an `ExecutionPlan`, and `execute()` runs it.

### ViewDto (Data Transfer Object)

Serializable enum representing operations. This is the bridge between the JSON graph (from Python) and the actual operation execution:

```rust
pub enum ViewDto {
    Image(ImageOp),           // resize, blur, grayscale, threshold, rotate
    Compute(ComputeOp),       // cast, scale, normalize, clamp, relu
    View(ViewOp),             // transpose, reshape, flip, crop
    Binary(BinaryOp),         // add, subtract, multiply, blend, bitwise
    Geometry(GeometryOp),     // extract_contours, rasterize, measures
    // ... other variants
}
```

The `resolve_op` function in `polars-cv/src/execute.rs` maps operation names from JSON to `ViewDto` variants.

### Operation Categories

| Category | Zero-Copy? | Description |
|----------|-----------|-------------|
| **View** | Yes | Transpose, reshape, flip, crop — only modify metadata |
| **Compute** | No | Cast, scale, normalize, clamp, relu — element-wise, can be fused |
| **Image** | No | Resize, blur, grayscale, threshold — require materialization |
| **Binary** | No | Pixel-wise operations between two buffers |
| **Geometry** | N/A | Contour extraction, rasterization, measures |
| **Reduction** | No | Sum, mean, std, min, max, percentile → scalar/vector |

### Kernel Fusion

Consecutive compute operations (scalar element-wise ops like scale, relu, clamp, cast) can be fused into a single pass over the data. The `ExecutionPlan` handles this optimization.

### Op Trait

Operations implement the `Op` trait:
```rust
pub trait Op {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize>;
    fn memory_effect(&self) -> MemoryEffect;
    // ... other methods
}
```

`infer_shape` is used for shape tracking at planning time. `memory_effect` indicates whether the op is zero-copy (`ViewOnly`), requires contiguous input (`RequiresContiguous`), or allocates (`Allocating`).

## Adding a New Operation

1. **Define the op** in the appropriate `ops/` file (or create a new one):
   - For image ops: `ops/image.rs` — add variant to `ImageOpKind`
   - For compute ops: `ops/compute.rs` — add variant to `ComputeOp`
   - For geometry ops: `geometry/ops.rs` — add variant to `GeometryOp`

2. **Implement the `Op` trait** (if applicable) with `infer_shape` and `memory_effect`

3. **Add to `ViewDto`** in `ops/dto.rs` — this makes it serializable

4. **Add builder method to `ViewExpr`** in `expr.rs` — this exposes it in the lazy API

5. **Add execution logic** in `execution/runner.rs` — the actual compute

6. **Wire into polars-cv**: Add to `resolve_op` in `polars-cv/src/execute.rs`

See `.cursor/polars-cv-contribution-guide.md` for a full walkthrough with the `resize` op as an example.

## Feature Flags

| Feature | Dependencies | Purpose |
|---------|-------------|---------|
| `ndarray_interop` (default) | ndarray | Zero-copy ndarray views |
| `image_interop` | image, fast_image_resize, tiff | Image decode/encode/resize |
| `arrow_interop` | arrow | Arrow buffer interop |
| `polars_interop` | polars-arrow | Polars-specific Arrow interop |
| `perceptual_hash` | image_hasher + image_interop | Perceptual hashing |
| `serde` | serde, serde_json, bytemuck | Serialization support |

## Tiling (Currently No-Op)

Tiling was implemented to improve cache efficiency for large images by processing them in 256x256 tiles. It did not deliver expected performance gains. The `TileConfig` / `TilePolicy` infrastructure exists in `execution/tiling.rs` and is wired up, but the actual tiling path is disabled. This may be revisited for SIMD optimization.

## Performance Notes

- **View operations are O(1)** — they only change metadata
- **Kernel fusion** reduces memory traffic for consecutive scalar ops
- **Zero-copy interop** avoids unnecessary allocations when moving data between Arrow, ndarray, and image
- **Contiguous buffers** enable SIMD-friendly iteration patterns
