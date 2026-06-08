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

    let half = (op.ksize / 2) as i64;
    let kernel = &op.kernel;
    let ksize = op.ksize;

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

    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                for ky in 0..ksize {
                    for kx in 0..ksize {
                        let sy = y as i64 + ky as i64 - half;
                        let sx = x as i64 + kx as i64 - half;

                        let pixel = sample_pixel(src, h, w, c, ch, sy, sx, op.border);
                        sum += kernel[ky * ksize + kx] * pixel;
                    }
                }
                output[(y * w + x) * c + ch] = sum * norm_factor;
            }
        }
    }

    if c == 1 && shape.len() == 2 {
        ViewBuffer::from_vec_with_shape(output, vec![h, w])
    } else {
        ViewBuffer::from_vec_with_shape(output, vec![h, w, c])
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
