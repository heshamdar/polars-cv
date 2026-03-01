# AGENTS.md — view-buffer (Core Tensor Engine)

> Read the [root AGENTS.md](../AGENTS.md) first for project-wide context.
> Update this file when you change ViewBuffer, ViewExpr, operations, execution planning, or interop.

## Purpose

`view-buffer` is a **zero-copy, stride-aware tensor framework** for Rust. It is the computational engine that powers polars-cv. All actual image/array processing happens here.

While originally designed as an independent crate, it is currently **tightly coupled** to polars-cv in practice. Agents working on either layer typically need context from both.

### What This Crate Does

- `ViewBuffer` — strided multi-dimensional array for images, masks, feature maps
- Zero-copy view operations (transpose, flip, crop, reshape) via metadata changes only
- Compute operations (scale, normalize, cast, clamp, relu, contrast, gamma, invert)
- Image operations (resize, blur, grayscale, threshold, rotate, canny, histogram equalize)
- Color space conversions (RGB, BGR, HSV, LAB, YCbCr, Gray)
- Spatial filtering (2D convolution with configurable border modes)
- Geometry operations (contour extraction, rasterization, measures, pairwise matching)
- `ViewExpr` — lazy expression graph builder
- `ExecutionPlan` — optimized execution with kernel fusion
- Interop with Arrow, ndarray, image, and Polars-arrow

### What This Crate Does NOT Do

- No Python bindings (those are in `polars-cv`)
- No Polars expression registration or JSON graph parsing
- No cloud I/O

## Module Structure

```
src/
├── lib.rs              # Crate root, re-exports
├── core/               # ViewBuffer, DType, Layout
├── ops/                # Operations
│   ├── dto.rs          # ViewDto — serializable operation enum
│   ├── image.rs        # ImageOp, ImageOpKind (resize, blur, canny, etc.)
│   ├── color.rs        # ColorConvertOp, ColorSpace
│   ├── filter.rs       # ConvolveOp, BorderMode — 2D convolution
│   ├── compute.rs      # ComputeOp (cast, scale, normalize, clamp, relu, contrast, gamma, invert)
│   ├── binary.rs       # BinaryOp (add, subtract, multiply, blend, bitwise)
│   ├── histogram.rs    # Histogram computation
│   └── mod.rs          # Op trait, ViewOp enum (transpose, reshape, flip, crop, channel_select)
├── expr.rs             # ViewExpr — lazy expression graph builder
├── execution/          # ExecutionPlan, runner, tiling (no-op)
├── geometry/           # Contour, Point, BoundingBox, extraction, rasterization, measures, pairwise
├── protocol.rs         # VIEW binary protocol (header + data serialization)
└── interop/            # Arrow, ndarray, image crate, Polars-arrow integration
```

## Core Concepts

### ViewBuffer

Strided multi-dimensional array backed by a Rust `Vec` or Arrow buffer.

- **Shape:** `[height, width, channels]` for images, arbitrary for other data
- **Strides:** Byte strides per dimension — enables zero-copy transpose, flip, crop
- **DType:** Element type (U8, I8, U16, ..., F64)
- **Offset:** Byte offset into the backing buffer

### ViewExpr (Lazy Expression Graph)

```rust
let result = ViewExpr::new_source(buffer)
    .resize(224, 224, FilterType::Lanczos3)
    .normalize(NormalizeMethod::MinMax, None, None)
    .cast(DType::F32)
    .plan().execute();
```

Each method appends a node. `plan()` compiles into `ExecutionPlan`, `execute()` runs it.

### ViewDto (Data Transfer Object)

Serializable enum bridging JSON graph (from Python) to operation execution:

```rust
pub enum ViewDto {
    Image(ImageOp),         // resize, blur, grayscale, threshold, canny, histogram_equalize
    Compute(ComputeOp),     // cast, scale, normalize, clamp, relu, adjust_contrast, adjust_gamma, invert
    View(ViewOp),           // transpose, reshape, flip, crop, channel_select
    Binary(BinaryOp),       // add, subtract, multiply, blend, bitwise
    Geometry(GeometryOp),   // extract_contours, rasterize, measures
    ChannelSwap { order },  // reorder channels (allocating)
    ChannelMerge { .. },    // merge single-channel buffers (allocating, graph-level)
    Color(ColorConvertOp),  // RGB↔HSV, RGB↔LAB, RGB↔YCbCr, RGB↔BGR, RGB↔Gray
    Filter(ConvolveOp),     // 2D spatial convolution with border handling
}
```

### Operation Categories

| Category | Zero-Copy? | Description |
|----------|-----------|-------------|
| **View** | Yes | Transpose, reshape, flip, crop, channel_select — metadata only |
| **Compute** | No | Element-wise ops (cast, scale, normalize, clamp, contrast, gamma, invert) — can be fused |
| **Image** | No | Resize, blur, grayscale, threshold, canny, histogram equalize — require materialization |
| **Filter** | No | 2D convolution with `Replicate`/`Zero`/`Reflect` border modes — contiguous output, promotes to f32 |
| **Color** | No | Color space conversions — route through f32 RGB internally. LAB uses D65/sRGB. HSV follows OpenCV (H=[0,180] for U8) |
| **Binary** | No | Pixel-wise operations between two buffers |
| **Geometry** | N/A | Contour extraction, rasterization, measures, pairwise matching |
| **Reduction** | No | Sum, mean, std, min, max, percentile → scalar/vector |

### Kernel Fusion

Consecutive compute operations (scalar element-wise: scale, relu, clamp, cast) are fused into a single pass over the data by `ExecutionPlan`.

### Op Trait

```rust
pub trait Op {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize>;
    fn memory_effect(&self) -> MemoryEffect; // ViewOnly, RequiresContiguous, or Allocating
}
```

## Rank Preservation Contracts

- **resize**: Preserves input rank. 2D `[H, W]` → 2D `[H_new, W_new]`; 3D stays 3D.
- **grayscale**: Preserves input rank. 2D passes through; 3D sets channel dim to 1.
- **channel_select**: Reduces rank from 3D `[H, W, C]` to 2D `[H, W]`. Requires `to_contiguous()` + reshape (non-contiguous in HWC layout).

## Implementation Notes

- **Filter** (`ops/filter.rs`): `ConvolveOp` dispatched directly in graph executor (not via ViewExpr/ExecutionPlan), similar to Color.
- **Canny** (`execution/runner.rs`): Fused pipeline (5x5 Gaussian → Sobel gradients → NMS → hysteresis). Outputs U8 binary mask (0/255). `TilePolicy::Global`.
- **HistogramEqualize** (`execution/runner.rs`): 256-bin histogram → CDF remap. U8 output. `TilePolicy::Global`.
- **label_reduce centroid fallback**: When `region_mode="interior"` finds no interior pixels for a contour, falls back to sampling at centroid. Prevents sub-pixel contours from scoring 0.

## Adding a New Operation

1. Define the op in the appropriate `ops/` file (add variant to `ImageOpKind`, `ComputeOp`, `GeometryOp`, etc.)
2. Implement the `Op` trait with `infer_shape` and `memory_effect`
3. Add to `ViewDto` in `ops/dto.rs`
4. Add builder method to `ViewExpr` in `expr.rs`
5. Add execution logic in `execution/runner.rs`
6. Wire into polars-cv: add to `resolve_op` in `polars-cv/src/execute.rs`

See `.cursor/polars-cv-contribution-guide.md` for a full walkthrough.

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

`TileConfig` / `TilePolicy` infrastructure exists in `execution/tiling.rs` but the tiling path is disabled — it did not deliver expected performance gains. May be revisited for SIMD optimization. The Python API still exposes `configure_tiling` / `get_tiling_config` but they are non-functional.

## Performance Notes

- View operations are O(1) — metadata only
- Kernel fusion reduces memory traffic for consecutive scalar ops
- Zero-copy interop avoids unnecessary allocations between Arrow, ndarray, and image
- Contiguous buffers enable SIMD-friendly iteration patterns
