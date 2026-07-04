//! Spatial filtering (convolution) operations.
//!
//! Provides generic 2D convolution with arbitrary kernels and configurable
//! border handling. Higher-level operations like Sobel, Laplacian, and
//! sharpen are implemented in the Python layer as predefined kernel wrappers.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::cost::OpCost;
use crate::ops::traits::{MemoryEffect, Op};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Border handling mode for convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BorderMode {
    /// Replicate the nearest edge pixel.
    Replicate,
    /// Treat out-of-bounds pixels as zero.
    Zero,
    /// Reflect pixels around the edge (dcba|abcd|dcba).
    Reflect,
}

crate::naming::named_variants!(BorderMode {
    "replicate" => Replicate,
    "zero" => Zero,
    "reflect" => Reflect,
});

/// Generic 2D convolution operation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConvolveOp {
    /// Flattened kernel values (row-major, ksize×ksize).
    pub kernel: Vec<f32>,
    /// Kernel size (kernel is ksize×ksize, must be odd).
    pub ksize: usize,
    /// If true, divide output by the sum of absolute kernel values.
    pub normalize: bool,
    /// Border handling mode.
    pub border: BorderMode,
}

impl Op for ConvolveOp {
    fn name(&self) -> &'static str {
        "Convolve2D"
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        // Same-size convolution (zero-padded to maintain dimensions)
        inputs[0].to_vec()
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        self.output_dtype_rule().resolve(inputs[0], None)
    }

    fn memory_effect(&self) -> MemoryEffect {
        MemoryEffect::RequiresContiguous
    }

    fn intrinsic_cost(&self) -> OpCost {
        OpCost::Allocating
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        None // Produces contiguous output
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        DTypeCategory::Numeric
    }

    fn working_dtype(&self) -> Option<DType> {
        None // Works in f32 internally regardless of input
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        OutputDTypeRule::PromoteToFloat
    }
}

/// Apply 2D convolution to a buffer.
///
/// Works in f32 internally. For multi-channel images, each channel is
/// convolved independently. Output dtype is f32.
///
/// Structured for auto-vectorization (interior/border split, same pattern as
/// the separable Gaussian blur in `execution/runner.rs`):
/// - interior pixels accumulate each kernel tap as a contiguous shifted-slice
///   multiply-add over the row segment — no bounds handling, no per-tap
///   border dispatch, vectorizes for any channel count;
/// - the border ring keeps the original per-pixel clamped/reflected gather.
///
/// Taps are visited in the same `ky`-outer/`kx`-inner order as the original
/// per-pixel loop and `norm_factor` is applied as a final multiply, so every
/// element accumulates in the identical floating-point order: the result is
/// bit-exact with the pre-split implementation (see tests/convolve_ref.rs).
pub fn apply_convolve2d(buf: &ViewBuffer, op: &ConvolveOp) -> ViewBuffer {
    let work_buf = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = work_buf.to_contiguous();
    let shape = contig.shape();

    let h = shape[0];
    let w = shape[1];
    let c = shape.get(2).copied().unwrap_or(1);

    let count = contig.layout.num_elements();
    let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };

    let kernel = &op.kernel;
    let ksize = op.ksize;
    let half = ksize / 2;

    let norm_factor = if op.normalize {
        let abs_sum: f32 = kernel.iter().map(|k| k.abs()).sum();
        if abs_sum > 0.0 {
            1.0 / abs_sum
        } else {
            1.0
        }
    } else {
        1.0
    };

    let mut output = vec![0.0f32; h * w * c];

    if h > 2 * half && w > 2 * half {
        match ksize {
            3 => convolve_interior::<3>(src, &mut output, h, w, c, kernel, norm_factor),
            5 => convolve_interior::<5>(src, &mut output, h, w, c, kernel, norm_factor),
            7 => convolve_interior::<7>(src, &mut output, h, w, c, kernel, norm_factor),
            _ => convolve_interior_dyn(src, &mut output, h, w, c, kernel, ksize, norm_factor),
        }
        convolve_border_ring(src, &mut output, h, w, c, op, norm_factor);
    } else {
        // Image no larger than the kernel: clamped/reflected gather everywhere.
        convolve_gather_rect(src, &mut output, h, w, c, op, norm_factor, 0, h, 0, w);
    }

    if c == 1 && shape.len() == 2 {
        ViewBuffer::from_vec_with_shape(output, vec![h, w])
    } else {
        ViewBuffer::from_vec_with_shape(output, vec![h, w, c])
    }
}

/// Interior convolution with the kernel size known at compile time.
///
/// Requires `h > 2*(K/2)` and `w > 2*(K/2)`. Writes only the interior
/// rows/columns of `out`; the border ring is left untouched (zero).
fn convolve_interior<const K: usize>(
    src: &[f32],
    out: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    kernel: &[f32],
    norm_factor: f32,
) {
    convolve_interior_dyn(src, out, h, w, c, kernel, K, norm_factor);
}

/// Interior convolution: per-tap shifted-slice multiply-adds over contiguous
/// row segments, then a final `norm_factor` multiply.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn convolve_interior_dyn(
    src: &[f32],
    out: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    kernel: &[f32],
    ksize: usize,
    norm_factor: f32,
) {
    let half = ksize / 2;
    let wc = w * c;
    let lo = half * c;
    let hi = (w - half) * c;
    let seg = hi - lo;

    for y in half..h - half {
        let out_row = &mut out[y * wc + lo..y * wc + hi];
        for ky in 0..ksize {
            let src_row = &src[(y + ky - half) * wc..(y + ky - half + 1) * wc];
            for kx in 0..ksize {
                let kw = kernel[ky * ksize + kx];
                // Column shift of (kx - half) whole pixels within the row.
                let start = lo + kx * c - half * c;
                let src_seg = &src_row[start..start + seg];
                for (o, &v) in out_row.iter_mut().zip(src_seg) {
                    *o += kw * v;
                }
            }
        }
        for o in out_row.iter_mut() {
            *o *= norm_factor;
        }
    }
}

/// Convolve the border ring (top/bottom rows plus left/right columns of the
/// remaining rows) with the original per-pixel gather.
fn convolve_border_ring(
    src: &[f32],
    out: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    op: &ConvolveOp,
    norm_factor: f32,
) {
    let half = op.ksize / 2;
    // Top and bottom rows.
    convolve_gather_rect(src, out, h, w, c, op, norm_factor, 0, half, 0, w);
    convolve_gather_rect(src, out, h, w, c, op, norm_factor, h - half, h, 0, w);
    // Left and right columns of the interior rows.
    convolve_gather_rect(src, out, h, w, c, op, norm_factor, half, h - half, 0, half);
    convolve_gather_rect(
        src,
        out,
        h,
        w,
        c,
        op,
        norm_factor,
        half,
        h - half,
        w - half,
        w,
    );
}

/// Per-pixel gather convolution over the rectangle `[y0,y1) × [x0,x1)`,
/// preserving the original tap order and border handling.
#[allow(clippy::too_many_arguments)]
fn convolve_gather_rect(
    src: &[f32],
    out: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    op: &ConvolveOp,
    norm_factor: f32,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) {
    let half = (op.ksize / 2) as i64;
    let ksize = op.ksize;
    let kernel = &op.kernel;

    for ch in 0..c {
        for y in y0..y1 {
            for x in x0..x1 {
                let mut sum = 0.0f32;
                for ky in 0..ksize {
                    for kx in 0..ksize {
                        let sy = y as i64 + ky as i64 - half;
                        let sx = x as i64 + kx as i64 - half;

                        let pixel = sample_pixel(src, h, w, c, ch, sy, sx, op.border);
                        sum += kernel[ky * ksize + kx] * pixel;
                    }
                }
                out[(y * w + x) * c + ch] = sum * norm_factor;
            }
        }
    }
}

/// Sample a pixel with border handling.
#[inline]
#[allow(clippy::too_many_arguments)]
fn sample_pixel(
    data: &[f32],
    h: usize,
    w: usize,
    c: usize,
    ch: usize,
    y: i64,
    x: i64,
    border: BorderMode,
) -> f32 {
    let (sy, sx) = match border {
        BorderMode::Zero => {
            if y < 0 || y >= h as i64 || x < 0 || x >= w as i64 {
                return 0.0;
            }
            (y as usize, x as usize)
        }
        BorderMode::Replicate => {
            let sy = y.clamp(0, h as i64 - 1) as usize;
            let sx = x.clamp(0, w as i64 - 1) as usize;
            (sy, sx)
        }
        BorderMode::Reflect => {
            let sy = reflect_index(y, h);
            let sx = reflect_index(x, w);
            (sy, sx)
        }
    };
    data[(sy * w + sx) * c + ch]
}

/// Reflect an index about the edge: dcba|abcd|dcba
#[inline]
fn reflect_index(idx: i64, size: usize) -> usize {
    if idx < 0 {
        (-idx - 1).min(size as i64 - 1) as usize
    } else if idx >= size as i64 {
        let reflected = 2 * size as i64 - idx - 1;
        reflected.max(0) as usize
    } else {
        idx as usize
    }
}
