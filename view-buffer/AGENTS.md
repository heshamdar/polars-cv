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
│   ├── mod.rs          # Module aggregator / re-exports for all op types
│   ├── dto.rs          # ViewDto — serializable operation enum
│   ├── traits.rs       # Op trait and core op types (MemoryEffect — the materialisation authority)
│   ├── image.rs        # ImageOp, ImageOpKind (resize, blur, canny, erode, dilate, morph_gradient, etc.)
│   ├── color.rs        # ColorConvertOp, ColorSpace
│   ├── filter.rs       # ConvolveOp, BorderMode — 2D convolution
│   ├── compute.rs      # ComputeOp (cast, scale, normalize, clamp, relu, contrast, gamma, invert, affine, rotate_affine)
│   ├── scalar.rs       # ScalarOp — elementary f32 ops fusable into a single kernel
│   ├── affine.rs       # AffineParams, InterpolationType, from_rotation() — affine transform parameters
│   ├── binary.rs       # BinaryOp (add, subtract, multiply, blend, bitwise)
│   ├── reduction.rs    # Reduction ops (sum, mean, std, min, max, argmin/argmax, percentile)
│   ├── histogram.rs    # Histogram computation
│   ├── phash.rs        # Perceptual hashing (aHash/pHash/dHash) ops
│   ├── view.rs         # ViewOp enum — zero-copy layout ops (transpose, reshape, flip, crop, channel_select)
│   ├── shape_rule.rs   # OutputRankRule / OutputChannelRule — plan-time structure rules (the authority)
│   ├── validation.rs   # Plan-time shape/dtype constraint checks
│   └── util.rs         # Shared index/coordinate helpers
├── expr.rs             # ViewExpr — lazy expression graph builder
├── execution/          # ExecutionPlan, runner, tiling (no-op)
├── geometry/           # Contour, Point, BoundingBox, extraction, rasterization, measures, pairwise
│                       # Polygon maths is `geo`'s throughout — this layer maps
│                       # Contour <-> geo types and owns degenerate-input conventions.
│                       # `GeometryOp` lists only ops the Pipeline *graph* routes;
│                       # contour-column ops live in the plugin's `.contour` namespace
│                       # and call measures/predicates/pairwise/transforms directly.
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

Serializable enum of exactly the operations `ViewExpr` can execute — every
variant is buffer-in/buffer-out and Op-backed (contracts delegate through
`as_op()`). Graph-level concerns (binary ops between nodes, masks, channel
merge, geometry, reductions, histograms, perceptual hash) live in polars-cv's
`GraphStep` (`polars-cv/src/graph/step.rs`), not here — including anything that
changes the data domain (e.g. `perceptual_hash` → `vector`):

```rust
pub enum ViewDto {
    View(ViewOp),           // transpose, reshape, flip, crop, channel_select, pad, rotate90/180/270
    Compute(ComputeOp),     // cast, scale, normalize, clamp, relu, adjust_contrast, adjust_gamma, invert, affine, rotate_affine
    Image(ImageOp),         // resize, blur, grayscale, threshold, canny, erode, dilate, morph_gradient, equalize
    Color(ColorConvertOp),  // RGB↔HSV, RGB↔LAB, RGB↔YCbCr, RGB↔BGR, RGB↔Gray
    Filter(ConvolveOp),     // 2D spatial convolution with border handling
}
```

`tests/apply_op_coverage.rs` executes one probe per variant against its own
contract and fails to compile when a variant is added without a probe.

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

`Op` (`src/ops/traits.rs`) declares twelve methods, **seven with no default**.
The seven are what a new op cannot skip — it does not compile until it answers
each one:

```rust
pub trait Op {
    fn name(&self) -> &'static str;
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize>;
    fn infer_strides(&self, shape: &[usize], strides: &[isize]) -> Option<Vec<isize>>;

    // The plan-time contract quartet: rank, channels, dtype, memory.
    fn output_rank_rule(&self) -> OutputRankRule;
    fn output_channel_rule(&self) -> OutputChannelRule;
    fn output_dtype_rule(&self) -> OutputDTypeRule;
    fn memory_effect(&self) -> MemoryEffect; // View, StridePreserving, RequiresContiguous
}
```

The remaining five carry defaults: `validate()`, `accepted_input_dtypes()`,
`working_dtype()`, `resolve_output_dtype()` and `validate_output_dtype()`.

The four rules marked as the quartet are the ones the Python planner reads over
FFI, and **adding a default to any of them is a regression** — an op that
declines to declare its dtype rule would silently inherit `PreserveInput` and
publish a schema execution cannot produce. See the Canonical Paths table in the
root `CLAUDE.md`.

`DomainOp` is a *separate* trait (same file), not part of `Op`. It has four
methods, three required — `input_domain()`, `output_domain()` and
`execute_typed()`; `validate_input_domain()` defaults.

## Alpha Channel Support

Alpha channels are **always preserved** during image decoding. `from_dynamic_image()` produces:
- RGBA → `[H, W, 4]`, GrayA → `[H, W, 2]`
- RGB → `[H, W, 3]`, Gray → `[H, W, 1]`

Operations handle alpha via the channel strategy declared by their
`OutputChannelRule` (`ops/shape_rule.rs`):

| `OutputChannelRule` | Operations | Behavior |
|----------|-----------|----------|
| **`PreserveChannels`** | resize, normalize, crop, flip, pad, etc. | All channels processed uniformly |
| **`StripProcessRestore`** | blur, cvt_color, sobel, laplacian, sharpen | Alpha split off, op on color channels, alpha re-attached |
| **`Fixed(n)`** | grayscale, canny, threshold, erode, dilate, morph_gradient | Alpha discarded, fixed output channels |

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
- **label_reduce centroid fallback** (`geometry/label.rs`): When the chosen region catches no pixel centre for a contour, falls back to sampling at the centroid. Prevents sub-pixel contours from scoring 0. `score_contours_on_buffer` is the single implementation behind both `Pipeline.label_reduce` and the `.contour.label_reduce()` accessor — the plugin must not carry its own scorer.

## Adding a New Operation

1. Define the op in the appropriate `ops/` file (add variant to `ImageOpKind`, `ComputeOp`, `GeometryOp`, etc.)
2. Implement the `Op` trait with `infer_shape` and `memory_effect`
3. Add to `ViewDto` in `ops/dto.rs`
4. Add builder method to `ViewExpr` in `expr.rs`
5. Add execution logic in `execution/runner.rs`
6. Wire into polars-cv: add to `resolve_op` in `polars-cv/src/execute.rs` (as `GraphStep::Buffer(dto)`)

## Feature Flags

| Feature | Dependencies | Purpose |
|---------|-------------|---------|
| `ndarray_interop` (default) | ndarray | Zero-copy ndarray views |
| `image_interop` | image, fast_image_resize, tiff | Image decode/encode/resize |
| `arrow_interop` | arrow | Arrow buffer interop |
| `polars_interop` | polars-arrow | Polars-specific Arrow interop |
| `perceptual_hash` | image_hasher + image_interop | Perceptual hashing |
| `serde` | serde, serde_json, bytemuck | Serialization support |

## Removed Subsystems

Two layers from view-buffer's original life as a standalone crate were deleted
once nothing reached them. Both are listed here so the next author does not
reinvent them without a consumer:

- **Pipeline composition (`ops/io.rs`).** `SourceFormat`, `SinkFormat` and
  `PlaceholderMeta`, plus `ExprNode::LazySource` / `::Placeholder` / `::Sink`
  and their constructors. Nothing in the workspace ever called them — the
  plugin builds its own source/sink vocabulary in `polars-cv/src/pipeline.rs`.
  Their only cost was not code size: every `match` over `ExprNode` carried arms
  for them, two of which were `panic!("must be resolved before building plan")`.
  Deleting them also retired the "three-way format representation split" that
  two comments described as a known divergence to live with.
- **Cost reporting (`ops/cost.rs`).** `OpCost`, `OpCostReport`,
  `PipelineCostReport`, `ViewExpr::cost_report()`, `explain_costs()` and
  `Op::intrinsic_cost()`. Exercised only by view-buffer's own tests; no Python
  surface reached it, so every op author maintained a declaration for nobody.
  **`MemoryEffect` stayed** — it is what `build_plan` matches on to insert a
  `MaterializeContiguous`, and cost could never have replaced it because the
  `MemoryEffect -> OpCost` conversion collapsed `StridePreserving` and
  `RequiresContiguous` into one value. Its doc comment claimed the reverse.

  If a cost/allocation explain surface is wanted later, build it against
  `MemoryEffect` and wire it to a user-facing API in the same change.

## Tiling (Removed)

A tiled execution strategy was implemented, benchmarked, and removed — it did not deliver performance gains over the simple per-op full-array passes that LLVM auto-vectorizes (`execution/tiling.rs` and the Python `configure_tiling` surface no longer exist). Treat that history as a prior for future loop-structure micro-optimizations: benchmark first.

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
