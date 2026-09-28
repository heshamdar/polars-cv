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
│   ├── dispatch.rs     # SimdKernel + dispatch(): the one way a kernel gets an AVX2 build
│   │                   # (debug builds assert both builds' outputs are byte-identical)
│   ├── convert.rs      # CastFrom + convert_slice/convert_view: the one element-conversion
│   │                   # rule (cast_to, the engine's f32 read and fused output use it)
│   └── strided.rs      # Walk: the one walk over a view's memory (packing, strided reads)
├── ops/                # Operations
│   ├── mod.rs          # Module aggregator / re-exports for all op types
│   ├── dto.rs          # ViewDto — serializable operation enum
│   ├── traits.rs       # Op trait and core op types (MemoryEffect — the materialisation authority)
│   ├── image.rs        # ImageOp, ImageOpKind (resize, blur, canny, erode, dilate, morph_gradient, etc.)
│   ├── color.rs        # ColorConvertOp, ColorSpace
│   ├── filter.rs       # ConvolveOp, BorderMode — 2D convolution
│   ├── compute.rs      # ComputeOp (cast, scale, normalize, clamp, relu, contrast, gamma, invert, affine, rotate_affine)
│   ├── scalar.rs       # ScalarOp — elementary f32 ops fusable into a single kernel
│   ├── elementwise/    # The engine every per-value compute op runs through: lowering,
│   │                   # statistics, integer / table / blocked / pass strategy, in-place writes;
│   │                   # legacy.rs + tests.rs hold the pre-engine code as its test oracle
│   ├── affine.rs       # AffineParams, InterpolationType, from_rotation() — affine transform parameters
│   ├── binary.rs       # BinaryOp (add, subtract, multiply, blend, bitwise)
│   ├── reduction.rs    # Reduction ops (sum, mean, std, min, max, argmin/argmax, percentile)
│   ├── histogram.rs    # Histogram computation
│   ├── phash.rs        # Perceptual hashing (aHash/pHash/dHash) ops
│   ├── view.rs         # ViewOp enum — zero-copy layout ops (transpose, reshape, flip, crop, channel_select)
│   ├── mask.rs         # apply_mask — mask a buffer by another
│   ├── pad.rs          # PadMode, PadPosition — padding settings
│   ├── shape_rule.rs   # OpShape — shape arithmetic, rank (its length) and channels (axis 2): the authority
│   ├── spatial_rule.rs # SpatialDependency — what an output pixel reads (Pointwise/Neighborhood/Global/Geometric)
│   ├── validation.rs   # Plan-time shape/dtype constraint checks
│   └── util.rs         # Shared index/coordinate helpers
├── expr.rs             # ViewExpr — lazy expression graph builder
├── execution/          # ExecutionPlan (plan.rs), runner (runner.rs)
├── geometry/           # Contour, Point, BoundingBox, extraction, rasterization, measures, pairwise
│                       # Polygon maths is `geo`'s throughout — this layer maps
│                       # Contour <-> geo types and owns degenerate-input conventions.
│                       # `GeometryOp` lists only ops the Pipeline *graph* routes;
│                       # contour-column ops live in the plugin's `.contour` namespace
│                       # and call measures/predicates/pairwise/transforms directly.
├── protocol.rs         # VIEW binary protocol (header + data serialization)
└── interop/            # Arrow, ndarray, image crate, fast_image_resize, Polars-arrow integration
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
    .normalize(Normalization::MinMax, None, None)
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

### Kernels and CPU dispatch

Published wheels target the x86-64 baseline (SSE2). A kernel that should use
AVX2 is a `core::dispatch::SimdKernel` (its whole body in an
`#[inline(always)] fn run`) called through `dispatch()`; do not hand-roll
`is_x86_feature_detected!` + `#[target_feature]` pairs. Only AVX2 is enabled,
never FMA, so both builds are bit-identical, and every debug-build call asserts
it. That check is meaningful under the wheels' flags, so run the suite once as
`RUSTFLAGS="-C target-cpu=x86-64" cargo test -p view-buffer --all-features`
(with its own `CARGO_TARGET_DIR`) after touching a kernel: the local
`.cargo/config.toml` builds for `x86-64-v3`, where the two builds coincide.

Element conversion between dtypes has one rule, `core::convert::CastFrom`
(integer sources `as`; float → integer round-half-away then saturate; float →
float `as`), applied in bulk by `convert_slice` (`convert_view` for a strided
view). `with_dtype!` is the one runtime `DType` → element-type match.

A view's elements are read in logical order only through `core::strided::Walk`:
it coalesces the layout once into packed units, evenly spaced rows and outer
axes, then packs them (`copy_to`, behind `to_contiguous`/`append_to`/
`write_to`) or hands out runs (`for_each_run`, behind `convert_view`). Do not
write another index odometer over strides; a kernel that cannot read a view in
place packs it with `to_contiguous()` and runs its dense path.

Per-value compute ops (the scalar family, scale, relu, clamp, invert, gamma,
contrast, normalize, fused chains) run only through `ops::elementwise::apply`:
it lowers the op to a `FusedKernel` (`lower_to_scalars`, shared with fusion),
picks integer arithmetic for an integer affine kernel over 8/16-bit input into
the same dtype (`invert`, integer shifts), a lookup table (only for work that
cannot vectorise, such as `powf`, or per-channel kernels, over 8/16-bit input),
blocked streaming for an integer result, or one f32 pass, and writes in place when
`ViewBuffer::unique_contiguous_mut` allows. A table read is slower than a
vectorised `255 - x`, so a cheap kernel never takes one. Do not add a per-op
kernel beside it; extend the lowering.

Row-wise kernels read a view where it lies through `ViewBuffer::dense_rows`
(contiguous, crops, vertical flips) instead of calling `to_contiguous()` first.
Resize hands such a view to fast_image_resize through
`interop::fir::FirViewAdapter`, so the resizes declare
`MemoryEffect::StridePreserving` and pack only a layout the adapter refuses.

### Operation Categories

| Category | Zero-Copy? | Description |
|----------|-----------|-------------|
| **View** | Yes | Transpose, reshape, flip, crop, channel_select — metadata only |
| **Compute** | No | Element-wise ops (cast, scale, normalize, clamp, contrast, gamma, invert) — can be fused. Includes `ComputeOp::Affine` and `ComputeOp::RotateAffine` (not fused with scalar ops). |
| **Image** | No | Resize, blur, grayscale, threshold, canny, histogram equalize, erode, dilate, morph gradient — allocate their output; resize and threshold read strided input, the rest require materialization |
| **Filter** | No | 2D convolution with `Replicate`/`Zero`/`Reflect` border modes — contiguous output, promotes to f32 |
| **Color** | No | Color space conversions — route through f32 RGB internally. LAB uses D65/sRGB. HSV follows OpenCV (H=[0,180] for U8) |
| **Binary** | No | Pixel-wise operations between two buffers |
| **Geometry** | N/A | Contour extraction, rasterization, measures, pairwise matching |
| **Reduction** | No | Sum, mean, std, min, max, percentile → scalar/vector |

### Kernel Fusion

Consecutive compute operations (scalar element-wise: scale, relu, clamp, cast) are fused into a single pass over the data by `ExecutionPlan`.

### Op Trait

`Op` (`src/ops/traits.rs`) declares the contract every op must answer. Seven
rule methods carry **no default**, so a new op does not compile until it
states each one — it cannot inherit a lie:

```rust
pub trait Op {
    fn name(&self) -> &'static str;
    fn infer_strides(&self, shape: &[usize], strides: &[isize]) -> Option<Vec<isize>>;

    // The contract — seven required rules, no defaults.
    fn shape(&self) -> OpShape; // the one authority for shape arithmetic
    fn output_dtype_rule(&self) -> OutputDTypeRule;
    fn memory_effect(&self) -> MemoryEffect; // View, StridePreserving, RequiresContiguous
    fn spatial_dependency(&self) -> SpatialDependency; // Global is the safe answer
    fn identity_rule(&self) -> IdentityRule;           // Never is the safe answer
    fn is_spatial_window(&self) -> bool; // a hoistable H/W window (spatial pushdown)
    fn validate(&self, input_shapes: &[&[Dim]], input_dtypes: &[PlannedDType])
        -> Result<(), ValidationError>; // plan time and before every row (CR-34)
}
```

`shape` returns an `OpShape` (`src/ops/shape_rule.rs`): the op's shape
transform as data. Execution evaluates it on known sizes (`OpShape::concrete`);
the plugin's planner evaluates it symbolically (`OpShape::dims`), where a size
is `Dim::Known(n)`, `Dim::Input(k)` (an unknown input axis carried through) or
`Dim::Unknown`, and a shape-deciding parameter is `Sym::Known(v)` or
`Sym::PerRow`. It is total: an unexpected rank yields unknown sizes, never a
panic. `ImageOpKind::shape` is what the runner sizes a deferred resize with.

`identity_rule` answers *under what condition* the op is a removable no-op
(`Never`, `WhenShapePreserved`, `WhenDtypePreserved`), and never depends on a
parameter's value. Whether *these* parameters make a `WhenShapePreserved` op a
no-op is `OpShape::preserves` — a zero pad, a full-frame crop at a known-zero
origin (never at any other origin: it could keep its extent only by running
past the edge), a same-shape reshape — so a per-row parameter, being
`Sym::PerRow`, can never prove one. Shape preservation alone never proves a
no-op for an op that moves pixels, which is why those stay `Never`.

The dtype methods that *do* carry defaults are `accepted_input_dtypes()`,
`working_dtype()`, `resolve_output_dtype()` and `validate_output_dtype()`.

The seven rule methods are what the plugin's Rust planner (`plan::step`) and
its passes read, and **adding a default to any of them is a regression** — an op that declines to
declare its dtype rule would silently inherit `PreserveInput` and publish a
schema execution cannot produce. This matches the required-no-default list in
the root `CLAUDE.md` and the Canonical Paths table in the root `AGENTS.md`.

## Alpha Channel Support

Alpha channels are **always preserved** during image decoding. `from_dynamic_image()` produces:
- RGBA → `[H, W, 4]`, GrayA → `[H, W, 2]`
- RGB → `[H, W, 3]`, Gray → `[H, W, 1]`

Operations handle alpha via their `OpShape` (`ops/shape_rule.rs`), whose axis
2 is the output channel count (the planner reads it; there is no separate
channel rule):

| `OpShape` | Operations | Behavior |
|----------|-----------|----------|
| **`Preserve`** and the H/W-only shapes | resize, normalize, crop, flip, pad, threshold, erode, dilate, etc. | All channels processed uniformly |
| **`ColorChannels`** | cvt_color | Alpha split off, op on color channels, alpha re-attached |
| **`SingleChannel`** | grayscale, canny | Alpha discarded, one output channel |

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

- **Filter** (`ops/filter.rs`): `ConvolveOp` runs as `ViewExpr::Filter` / `PlanStep::Filter` (`apply_convolve2d`).
- **Canny** (`execution/runner.rs`): `cv2.Canny(img, low, high)` exactly (3x3 Sobel with replicated border, L1 magnitude, no pre-blur → OpenCV's fixed-point NMS → 8-connected hysteresis; colour takes the strongest channel per pixel, alpha ignored). `polars-cv/tests/reference/test_canny_ref.py` holds it to OpenCV pixel for pixel. Outputs U8 binary mask (0/255). `SpatialDependency::Global`.
- **HistogramEqualize** (`execution/runner.rs`): 256-bin histogram → CDF remap. U8 output. `SpatialDependency::Global`.
- **Affine** (`execution/runner.rs`): Forward-mapping 2×3 matrix with internal inversion for inverse-mapping interpolation. Supports Nearest and Bilinear interpolation with configurable `border_value`. Parameters in `ops/affine.rs` (`AffineParams`, `InterpolationType`). Two variants: `ComputeOp::Affine` (raw matrix) and `ComputeOp::RotateAffine` (deferred rotation, constructs `AffineParams` via `AffineParams::from_rotation()` at execution time). Both use `apply_affine_warp()`. `MemoryEffect::RequiresContiguous`.
- **Erode/Dilate** (`execution/runner.rs`): Separable row+column min/max filter. Single-channel only. Supports multiple iterations. `SpatialDependency::Neighborhood`.
- **MorphGradient** (`execution/runner.rs`): Dilate − Erode (saturating subtract). Single-channel only. `SpatialDependency::Neighborhood`.
- **label_reduce centroid fallback** (`geometry/label.rs`): When the chosen region catches no pixel centre for a contour, falls back to sampling at the centroid. Prevents sub-pixel contours from scoring 0. `score_contours_on_buffer` is the single implementation behind both `Pipeline.label_reduce` and the `.contour.label_reduce()` accessor — the plugin must not carry its own scorer.

## Adding a New Operation

1. Define the op as a variant of the appropriate mode-generic family in `ops/`
   (`ImageOpKind<M>`, `ComputeOp<M>`, `GeometryOp<M>`, …), with its
   `#[op(name = ..., sample = ...)]` attribute and documented fields
2. Answer the `Op` contract: `shape` (an `OpShape`), `validate`, and the other
   required rules
3. Add to `ViewDto` in `ops/dto.rs` (`tests/apply_op_coverage.rs` requires a probe per variant)
4. Add execution logic in `execution/runner.rs`
5. Expose it to Python through the typed catalogue: re-bless, `gen_ops.py`,
   `maturin develop` — the full procedure is "Adding a New Operation" in the
   root `CLAUDE.md`

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
  plugin builds its own source/sink vocabulary in `polars-cv/src/formats/`.
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
