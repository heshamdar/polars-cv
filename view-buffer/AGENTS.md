# AGENTS.md — view-buffer (Core Tensor Engine)

> Read the [root AGENTS.md](../AGENTS.md) first for project-wide context.
> Update this file when you change ViewBuffer, ViewExpr, operations, execution planning, or interop.

## Purpose

`view-buffer` is a **zero-copy, stride-aware tensor framework** for Rust. It is the computational engine that powers polars-cv. All actual image/array processing happens here.

While originally designed as an independent crate, it is currently **tightly coupled** to polars-cv in practice. Agents working on either layer typically need context from both.

### What This Crate Does

- `ViewBuffer` — strided multi-dimensional array for images, masks, feature maps
- Zero-copy view operations (transpose, flip, crop, reshape) via metadata changes only
- Compute operations (scale, normalize, cast, clamp, relu, contrast, gamma, invert, affine warp, rotate-via-affine)
- Image operations (resize, blur, grayscale, threshold, canny, histogram equalize, erode, dilate, morphological gradient)
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
│   ├── image.rs        # ImageOp, ImageOpKind (resize, blur, canny, erode, dilate, morph_gradient, etc.)
│   ├── color.rs        # ColorConvertOp, ColorSpace
│   ├── filter.rs       # ConvolveOp, BorderMode — 2D convolution
│   ├── compute.rs      # ComputeOp (cast, scale, normalize, clamp, relu, contrast, gamma, invert, affine, rotate_affine)
│   ├── affine.rs       # AffineParams, InterpolationType, from_rotation() — affine transform parameters
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
    Compute(ComputeOp),     // cast, scale, normalize, clamp, relu, adjust_contrast, adjust_gamma, invert, affine, rotate_affine
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
| **Compute** | No | Element-wise ops (cast, scale, normalize, clamp, contrast, gamma, invert) — can be fused. Includes `ComputeOp::Affine` and `ComputeOp::RotateAffine` (not fused with scalar ops). |
| **Image** | No | Resize, blur, grayscale, threshold, canny, histogram equalize, erode, dilate, morph gradient — require materialization |
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

## Alpha Channel Support

Alpha channels are **always preserved** during image decoding. `from_dynamic_image()` produces:
- RGBA → `[H, W, 4]`, GrayA → `[H, W, 2]`
- RGB → `[H, W, 3]`, Gray → `[H, W, 1]`

Operations handle alpha via three strategies (aligned with the Python `AlphaMode` contract):

| Strategy | Operations | Behavior |
|----------|-----------|----------|
| **Passthrough** | resize, normalize, crop, flip, pad, etc. | All channels processed uniformly |
| **Strip-Process-Restore** | blur, cvt_color, sobel, laplacian, sharpen | Alpha split off, op on color channels, alpha re-attached |
| **Drop** | grayscale, canny, threshold, erode, dilate, morph_gradient | Alpha discarded, fixed output channels |

Key implementation points:
- `ops/color.rs` provides `split_alpha()` / `merge_alpha()` helpers used by `apply_color_convert()`
- `execution/runner.rs`: `grayscale_u8()` handles 4ch (BT.601 on RGB, ignore alpha) and 2ch (take intensity)
- `execution/runner.rs`: blur dispatches via `ImageBuffer<Rgba<u8>>` and `ImageBuffer<LumaA<u8>>` for 4ch/2ch
- `interop/image.rs`: `to_dynamic_image()` accepts 1–4 channels; `encode_tiff()` supports RGBA/GrayA

## Rank Preservation Contracts

- **resize**: Preserves input rank. 2D `[H, W]` → 2D `[H_new, W_new]`; 3D stays 3D.
- **grayscale**: Preserves input rank. 2D passes through; 3D sets channel dim to 1.
- **channel_select**: Reduces rank from 3D `[H, W, C]` to 2D `[H, W]`. Requires `to_contiguous()` + reshape (non-contiguous in HWC layout).

## Implementation Notes

- **Filter** (`ops/filter.rs`): `ConvolveOp` dispatched directly in graph executor (not via ViewExpr/ExecutionPlan), similar to Color.
- **Canny** (`execution/runner.rs`): Fused pipeline (5x5 Gaussian → Sobel gradients → NMS → hysteresis). Outputs U8 binary mask (0/255). `TilePolicy::Global`.
- **HistogramEqualize** (`execution/runner.rs`): 256-bin histogram → CDF remap. U8 output. `TilePolicy::Global`.
- **Affine** (`execution/runner.rs`): Forward-mapping 2×3 matrix with internal inversion for inverse-mapping interpolation. Supports Nearest and Bilinear interpolation with configurable `border_value`. Parameters in `ops/affine.rs` (`AffineParams`, `InterpolationType`). Two variants: `ComputeOp::Affine` (raw matrix) and `ComputeOp::RotateAffine` (deferred rotation, constructs `AffineParams` via `AffineParams::from_rotation()` at execution time). Both use `apply_affine_warp()`. `MemoryEffect::RequiresContiguous`.
- **Erode/Dilate** (`execution/runner.rs`): Separable row+column min/max filter. Single-channel only. Supports multiple iterations. `TilePolicy::LocalNeighborhood`.
- **MorphGradient** (`execution/runner.rs`): Dilate − Erode (saturating subtract). Single-channel only. `TilePolicy::LocalNeighborhood`.
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
- Kernel fusion reduces memory traffic for consecutive scalar ops. The
  fusable set is `Scale`, `Relu`, `Clamp`, `AdjustGamma`, `Invert`
  (u8/u16/f32 inputs), and `Cast` — casts fold into the kernel itself:
  the kernel reads any numeric input dtype (converting to f32 during the
  gather) and converts its f32 result to `FusedKernel::out_dtype` while
  writing, so `u8 -> cast(f32) -> scale -> clamp -> relu` is a single pass
  with no cast materializations. `out_dtype` is pinned at fusion time to the
  dtype the *unfused* chain would produce (`expr.rs::try_fuse`), so fusion
  can never change the planned schema. f64 inputs are excluded from the
  promote-family lowering (the dtype contract preserves f64 there while the
  unfused runtime computes f32 — a pre-existing divergence fusion must not
  take a side on). Equivalence is guarded by `tests/fused_ops.rs`, which
  compares every fused chain bit-for-bit against per-op execution.
- Zero-copy interop avoids unnecessary allocations between Arrow, ndarray, and image
- Contiguous buffers enable SIMD-friendly iteration patterns
