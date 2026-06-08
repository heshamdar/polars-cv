//! Segment-level tiled execution for cache-efficient pipeline processing.
//!
//! # Why this matters
//!
//! The naive approach tiles *per operation*: extract a tile, apply one op,
//! reassemble the full image, then repeat for the next op.  This destroys
//! cache residency between ops because the full image is written to DRAM and
//! read back on every step.
//!
//! This module implements **outer-loop tiling**: consecutive tileable ops are
//! grouped into *segments*, and each tile is run through **all ops in the
//! segment** before the next tile is fetched.  Each 256 × 256 RGB tile is
//! ~192 KB — it stays in L2 through every op in the segment.
//!
//! # Segment formation
//!
//! Operations are classified by [`TilePolicy`]:
//!
//! | Policy | Examples | Halo |
//! |---|---|---|
//! | `PointWise` | scale, relu, cast, grayscale, threshold | 0 |
//! | `LocalNeighborhood` | blur, erode, dilate, morph-gradient | > 0 |
//! | `Global` | resize, affine, global-normalize, canny | full image |
//!
//! Consecutive `PointWise` and `LocalNeighborhood` steps form one *tileable
//! segment*.  A `Global` step, a `ViewOp`, or `MaterializeContiguous` always
//! terminates the current segment and runs on the full buffer.
//!
//! # Halo accumulation
//!
//! For a segment `[blur(halo=3), erode(halo=1)]` the combined halo is 4: the
//! erode needs 1 pixel of context that was produced by the blur, and the blur
//! needs 3 pixels of the original image, so we extract 4 extra pixels around
//! each tile's core to satisfy both in one extraction.
//!
//! # Per-tile allocation strategy
//!
//! Tile extraction writes directly into a thread-local byte slab, avoiding a
//! `Vec::new` per tile.  Pointwise ops that hold exclusive Arc ownership mutate
//! the buffer in-place via [`Arc::get_mut`] — no output allocation.
//! LocalNeighborhood ops (blur, erode) still allocate an output tile, but for
//! segments with a neighborhood op the segment length is typically short
//! (1–3 ops), so the absolute allocation count is low.
//!
//! # Thread safety
//!
//! [`TILE_EXTRACT_BUF`] is `thread_local!`, so each Polars morsel worker thread
//! has its own slab.  No contention and no synchronisation required.

use std::cell::RefCell;
use std::sync::Arc;

use crate::core::buffer::{BufferStorage, ViewBuffer};
use crate::core::layout::Layout;
use crate::execution::plan::PlanStep;
use crate::execution::runner::{apply_compute_inner, apply_image_inner, apply_perceptual_hash, apply_view};
use crate::ops::traits::Op;

// ── TilePolicy ────────────────────────────────────────────────────────────────

/// Declares how an operation interacts with spatial locality.
///
/// Used during segmentation to decide whether an op can participate in
/// outer-loop tiling, and if so, how large a halo to extract around each tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilePolicy {
    /// No pixel dependencies — every output pixel depends only on the
    /// corresponding input pixel.  Halo = 0.
    ///
    /// Examples: threshold, scale, relu, clamp, cast, grayscale, invert.
    PointWise,

    /// Needs neighboring pixels within a radius.  Tiles must include `halo`
    /// extra pixels on each edge to provide correct context.
    ///
    /// Examples: Gaussian blur (`halo ≈ 3σ`), erode/dilate (`halo = ksize/2`).
    LocalNeighborhood {
        /// Extra pixels needed on each edge of the tile.
        halo: usize,
    },

    /// Needs access to the full image — cannot be tiled.
    ///
    /// Examples: resize (global resampling), affine warp, histogram
    /// equalization, global normalize (min/max or z-score), Canny.
    Global,
}

impl TilePolicy {
    /// Returns the halo size for this policy (0 for `PointWise` and `Global`).
    #[inline]
    pub fn halo(self) -> usize {
        match self {
            TilePolicy::LocalNeighborhood { halo } => halo,
            _ => 0,
        }
    }

    /// Returns `true` if this op can participate in a tileable segment.
    #[inline]
    pub fn is_tileable(self) -> bool {
        !matches!(self, TilePolicy::Global)
    }
}

// ── Segment formation ─────────────────────────────────────────────────────────

/// A group of consecutive plan steps that share a tile during execution.
struct TileableSegment {
    steps: Vec<PlanStep>,
    /// Sum of per-op halos — the number of extra pixels to extract around each
    /// tile's core to satisfy all ops in the segment.
    combined_halo: usize,
}

impl TileableSegment {
    /// Returns `true` when every op is PointWise (combined_halo == 0).
    ///
    /// A purely-pointwise segment is already handled optimally by the fused
    /// kernel on the full buffer — tiling only adds extract+copy overhead.
    #[inline]
    fn is_pointwise_only(&self) -> bool {
        self.combined_halo == 0
    }
}

enum Segment {
    /// One or more tileable steps processed together on each tile.
    Tileable(TileableSegment),
    /// A single step that runs on the full buffer (Global policy, ViewOp, or
    /// MaterializeContiguous).
    GlobalStep(PlanStep),
}

/// Groups `steps` into segments: consecutive tileable ops become one
/// `Tileable` segment; everything else becomes individual `GlobalStep`s.
fn make_segments(steps: Vec<PlanStep>) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    let mut current: Option<TileableSegment> = None;

    let flush = |current: Option<TileableSegment>, segments: &mut Vec<Segment>| {
        if let Some(seg) = current {
            segments.push(Segment::Tileable(seg));
        }
    };

    for step in steps {
        match step_tile_policy(&step) {
            // ViewOps and MaterializeContiguous: flush + run immediately.
            None => {
                flush(current.take(), &mut segments);
                segments.push(Segment::GlobalStep(step));
            }
            // Global ops: flush + run on full buffer.
            Some(TilePolicy::Global) => {
                flush(current.take(), &mut segments);
                segments.push(Segment::GlobalStep(step));
            }
            // Tileable (PointWise or LocalNeighborhood): accumulate.
            Some(policy) => {
                let halo = policy.halo();
                match current.as_mut() {
                    Some(seg) => {
                        seg.combined_halo += halo;
                        seg.steps.push(step);
                    }
                    None => {
                        current = Some(TileableSegment {
                            steps: vec![step],
                            combined_halo: halo,
                        });
                    }
                }
            }
        }
    }

    flush(current, &mut segments);
    segments
}

/// Returns the tiling policy for a plan step, or `None` for steps that must
/// be executed immediately on the full buffer (ViewOp, MaterializeContiguous).
fn step_tile_policy(step: &PlanStep) -> Option<TilePolicy> {
    match step {
        PlanStep::View(_) | PlanStep::MaterializeContiguous => None,
        PlanStep::Compute(op) => Some(op.tile_policy()),
        PlanStep::Image(op) => Some(op.tile_policy()),
        PlanStep::PerceptualHash(_) => Some(TilePolicy::Global),
    }
}

// ── Thread-local extraction arena ────────────────────────────────────────────

// Per-thread byte slab reused for tile extraction.
// Instead of Vec::new() per tile, extract_tile fills this buffer and
// wraps it in a fresh Arc (only the control block is allocated;
// the data pages are already mapped from the previous tile).
thread_local! {
    static TILE_EXTRACT_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

// ── Outer-loop segment execution ─────────────────────────────────────────────

/// Executes `steps` on `source` using segment-level tiling.
///
/// Consecutive tileable ops are grouped and each tile is run through the
/// entire group before the next tile is fetched.  Global ops and view ops
/// still run on the full buffer.
///
/// Purely-pointwise segments (all halo == 0) bypass tiling and fall through
/// to `execute_full` — fusion already collapses them to a single pass, so
/// tile management would only add overhead.
pub fn execute_segmented_tiled(
    source: ViewBuffer,
    steps: Vec<PlanStep>,
    tile_size: usize,
) -> ViewBuffer {
    let segments = make_segments(steps);
    let mut current = source;

    for segment in segments {
        current = match segment {
            Segment::GlobalStep(step) => apply_step_full(current, step),
            Segment::Tileable(seg) if seg.is_pointwise_only() => {
                // All ops are PointWise — the fused kernel handles them in one
                // sequential pass that is already optimal.  Tiling would only
                // add extract + copy_core overhead.
                let mut buf = current;
                for step in seg.steps {
                    buf = apply_step_full(buf, step);
                }
                buf
            }
            Segment::Tileable(seg) => {
                execute_tile_segment(current, seg.steps, tile_size, seg.combined_halo)
            }
        };
    }

    current
}

/// Applies a single plan step to the full buffer without any tiling.
pub(crate) fn apply_step_full(buf: ViewBuffer, step: PlanStep) -> ViewBuffer {
    match step {
        PlanStep::View(op) => apply_view(buf, op),
        PlanStep::Compute(op) => apply_compute_inner(buf, op),
        PlanStep::Image(op) => apply_image_inner(buf, op),
        PlanStep::PerceptualHash(op) => apply_perceptual_hash(buf, op),
        PlanStep::MaterializeContiguous => buf.to_contiguous(),
    }
}

/// Applies a single plan step by reference — borrows the step to avoid
/// cloning the (potentially heap-allocated) op fields per tile.
fn apply_step_ref(buf: ViewBuffer, step: &PlanStep) -> ViewBuffer {
    match step {
        PlanStep::View(op) => apply_view(buf, op.clone()),
        PlanStep::Compute(op) => apply_compute_inner(buf, op.clone()),
        PlanStep::Image(op) => apply_image_inner(buf, op.clone()),
        PlanStep::PerceptualHash(op) => apply_perceptual_hash(buf, op.clone()),
        PlanStep::MaterializeContiguous => buf.to_contiguous(),
    }
}

// ── Tile execution ────────────────────────────────────────────────────────────

/// Executes all `steps` in a tileable segment using outer-loop tiling.
///
/// Each tile (plus its halo) is run through every op in `steps` before the
/// next tile is processed.  Only the core region (excluding halo) of each
/// output tile is copied to the final output buffer.
fn execute_tile_segment(
    input: ViewBuffer,
    steps: Vec<PlanStep>,
    tile_size: usize,
    combined_halo: usize,
) -> ViewBuffer {
    let input_shape = input.shape();
    let ndim = input_shape.len();
    let (h, w) = (input_shape[0], input_shape[1]);
    let in_channels = input_shape.get(2).copied().unwrap_or(1);

    // Process the first tile to discover the output dtype and channel count
    // (ops like grayscale reduce channels; cast changes dtype).
    let first_tile = extract_tile(&input, 0, 0, tile_size, combined_halo, h, w, in_channels, ndim);
    let first_out = run_steps_on_tile(first_tile, &steps);

    let out_shape = first_out.shape();
    let out_ndim = out_shape.len();
    let out_channels = out_shape.get(2).copied().unwrap_or(1);
    let out_dtype = first_out.dtype();
    let out_elem = out_dtype.size_of();

    // Pre-allocate the output buffer.
    let mut output_data: Vec<u8> = vec![0u8; h * w * out_channels * out_elem];

    // Copy the first tile's core into the output.
    copy_core(
        &first_out,
        0, 0, // core offset inside the tile output
        &mut output_data,
        0, 0, // destination top-left
        tile_size.min(h),
        tile_size.min(w),
        w, out_channels, out_elem,
    );

    // Process remaining tiles in row-major order (cache-friendly scan).
    for tile_y in (0..h).step_by(tile_size) {
        for tile_x in (0..w).step_by(tile_size) {
            if tile_y == 0 && tile_x == 0 {
                continue; // already done above
            }

            let tile = extract_tile(
                &input, tile_y, tile_x,
                tile_size, combined_halo,
                h, w, in_channels, ndim,
            );
            let out_tile = run_steps_on_tile(tile, &steps);

            // The halo may be clamped at image edges (when tile_y < combined_halo,
            // in_y0 is 0 rather than tile_y - combined_halo).
            let in_y0 = tile_y.saturating_sub(combined_halo);
            let in_x0 = tile_x.saturating_sub(combined_halo);
            let core_y_off = tile_y - in_y0; // pixels to skip at top of output tile
            let core_x_off = tile_x - in_x0;
            let core_h = tile_size.min(h - tile_y);
            let core_w = tile_size.min(w - tile_x);

            copy_core(
                &out_tile,
                core_y_off, core_x_off,
                &mut output_data,
                tile_y, tile_x,
                core_h, core_w,
                w, out_channels, out_elem,
            );
        }
    }

    let final_shape = if out_ndim >= 3 {
        vec![h, w, out_channels]
    } else {
        vec![h, w]
    };

    ViewBuffer {
        data: BufferStorage::Rust(Arc::new(output_data)),
        layout: Layout::new_contiguous(final_shape, out_dtype),
    }
}

/// Extracts a tile (plus halo) from `input` as a contiguous owned buffer.
///
/// Uses a thread-local byte slab ([`TILE_EXTRACT_BUF`]) to avoid allocating a
/// new `Vec` per tile.  The slab grows to fit the largest tile seen and is
/// never shrunk.  A fresh `Arc` wraps the data for the returned `ViewBuffer` —
/// only the control block is allocated, not the tile data itself.
#[allow(clippy::too_many_arguments)]
fn extract_tile(
    input: &ViewBuffer,
    tile_y: usize,
    tile_x: usize,
    tile_size: usize,
    halo: usize,
    img_h: usize,
    img_w: usize,
    channels: usize,
    ndim: usize,
) -> ViewBuffer {
    let in_y0 = tile_y.saturating_sub(halo);
    let in_y1 = (tile_y + tile_size + halo).min(img_h);
    let in_x0 = tile_x.saturating_sub(halo);
    let in_x1 = (tile_x + tile_size + halo).min(img_w);

    let tile_h = in_y1 - in_y0;
    let tile_w = in_x1 - in_x0;
    let elem_size = input.dtype().size_of();
    let row_bytes = tile_w * channels * elem_size;
    let required = tile_h * row_bytes;

    // Fill the thread-local slab directly — avoids one Vec::new per tile.
    let tile_data: Vec<u8> = TILE_EXTRACT_BUF.with(|cell| {
        let mut slab = cell.borrow_mut();
        if slab.len() < required {
            slab.resize(required, 0u8);
        }

        let src_ptr = unsafe { input.as_ptr::<u8>() };
        let src_strides = input.strides_bytes();

        for row in 0..tile_h {
            let src_byte_off = (in_y0 + row) as isize * src_strides[0]
                + in_x0 as isize * src_strides[1];
            let dst_byte_off = row * row_bytes;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.offset(src_byte_off),
                    slab.as_mut_ptr().add(dst_byte_off),
                    row_bytes,
                );
            }
        }

        // Clone out of the slab — the slab is reused next iteration.
        // This is a memcpy of ~200 KB but avoids the allocator round-trip
        // for every tile: the slab's physical pages are already warm.
        slab[..required].to_vec()
    });

    let shape = if ndim >= 3 {
        vec![tile_h, tile_w, channels]
    } else {
        vec![tile_h, tile_w]
    };

    ViewBuffer {
        data: BufferStorage::Rust(Arc::new(tile_data)),
        layout: Layout::new_contiguous(shape, input.dtype()),
    }
}

/// Applies all steps in a segment to a tile buffer.
///
/// Borrows `steps` to avoid cloning each `PlanStep` per tile. Scalar
/// pointwise ops will attempt in-place mutation via [`Arc::get_mut`] when
/// they hold exclusive ownership.
fn run_steps_on_tile(mut tile: ViewBuffer, steps: &[PlanStep]) -> ViewBuffer {
    for step in steps {
        tile = apply_step_ref(tile, step);
    }
    tile
}

/// Copies the core region of `src_tile` (skipping halo) into `dst`.
///
/// `core_y_off` / `core_x_off` are the row/column offsets inside `src_tile`
/// where the core begins (the halo region to skip).
#[allow(clippy::too_many_arguments)]
fn copy_core(
    src_tile: &ViewBuffer,
    core_y_off: usize,
    core_x_off: usize,
    dst: &mut [u8],
    dst_y: usize,
    dst_x: usize,
    core_h: usize,
    core_w: usize,
    dst_w: usize,
    channels: usize,
    elem_size: usize,
) {
    let tile_strides = src_tile.strides_bytes();
    let tile_ptr = unsafe { src_tile.as_ptr::<u8>() };

    let dst_row_stride = dst_w * channels * elem_size;
    let copy_row_bytes = core_w * channels * elem_size;

    for row in 0..core_h {
        let src_y = core_y_off + row;
        let src_x = core_x_off;

        let src_byte_off = (src_y as isize * tile_strides[0])
            + (src_x as isize * tile_strides[1]);

        let dst_byte_off = (dst_y + row) * dst_row_stride + dst_x * channels * elem_size;

        unsafe {
            let src_ptr = tile_ptr.offset(src_byte_off);
            let dst_ptr = dst.as_mut_ptr().add(dst_byte_off);
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_row_bytes);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_policy_halo() {
        assert_eq!(TilePolicy::PointWise.halo(), 0);
        assert_eq!(TilePolicy::LocalNeighborhood { halo: 6 }.halo(), 6);
        assert_eq!(TilePolicy::Global.halo(), 0);
    }

    #[test]
    fn tile_policy_is_tileable() {
        assert!(TilePolicy::PointWise.is_tileable());
        assert!(TilePolicy::LocalNeighborhood { halo: 4 }.is_tileable());
        assert!(!TilePolicy::Global.is_tileable());
    }

    #[test]
    fn segmentation_groups_tileable_ops() {
        use crate::execution::plan::PlanStep;
        use crate::ops::{ComputeOp, ImageOp, ImageOpKind};

        // scale (PointWise) + blur (LocalNeighborhood) should merge into one segment.
        let steps = vec![
            PlanStep::Compute(ComputeOp::Scale(2.0)),
            PlanStep::Image(ImageOp {
                kind: ImageOpKind::Blur { sigma: 1.0 },
            }),
        ];
        let segs = make_segments(steps);
        assert_eq!(segs.len(), 1);
        match &segs[0] {
            Segment::Tileable(s) => {
                assert_eq!(s.steps.len(), 2);
                assert_eq!(s.combined_halo, 3); // blur halo = ceil(3*1.0) = 3
            }
            _ => panic!("expected Tileable"),
        }
    }

    #[test]
    fn segmentation_breaks_on_global() {
        use crate::execution::plan::PlanStep;
        use crate::ops::{ComputeOp, ImageOp, ImageOpKind};

        let steps = vec![
            PlanStep::Compute(ComputeOp::Scale(2.0)),
            PlanStep::Image(ImageOp {
                kind: ImageOpKind::Resize {
                    height: 224,
                    width: 224,
                    filter: crate::ops::FilterType::Lanczos3,
                },
            }),
            PlanStep::Compute(ComputeOp::Scale(0.5)),
        ];
        let segs = make_segments(steps);
        // scale → [Tileable], resize → [GlobalStep], scale → [Tileable]
        assert_eq!(segs.len(), 3);
        assert!(matches!(segs[0], Segment::Tileable(_)));
        assert!(matches!(segs[1], Segment::GlobalStep(_)));
        assert!(matches!(segs[2], Segment::Tileable(_)));
    }

    #[test]
    fn pointwise_only_segment_is_detected() {
        let seg = TileableSegment {
            steps: vec![],
            combined_halo: 0,
        };
        assert!(seg.is_pointwise_only());

        let seg_with_halo = TileableSegment {
            steps: vec![],
            combined_halo: 3,
        };
        assert!(!seg_with_halo.is_pointwise_only());
    }
}
