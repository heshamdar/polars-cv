use crate::dtype::DType;
use crate::ops::core::{MemoryEffect, Op};

#[derive(Debug, Clone, PartialEq)]
pub enum ImageOpKind {
    /// Stride-preserving: Can often be done in-place or parallelized easily.
    /// E.g., Thresholding, simple color corrections.
    Threshold(u8),

    /// Layout-breaking: Changes shape or requires neighborhood access.
    /// E.g., Resizing, Convolution/Blur.
    Resize {
        width: u32,
        height: u32,
        filter: FilterType,
    },
    Blur {
        sigma: f32,
    },

    /// Format conversion (often requires contiguous memory for the image crate)
    Grayscale,
}

/// Placeholder for image crate filter types to avoid hard dep in this file if desired,
/// but since we have image dep, we can use it or a simplified enum.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterType {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageOp {
    pub kind: ImageOpKind,
}

impl Op for ImageOp {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input_shape = inputs[0];
        match &self.kind {
            ImageOpKind::Threshold(_) => input_shape.to_vec(),
            ImageOpKind::Blur { .. } => input_shape.to_vec(),
            ImageOpKind::Grayscale => {
                // RGB [H, W, 3] -> Luma [H, W, 1]
                let mut s = input_shape.to_vec();
                if s.len() == 3 {
                    s[2] = 1;
                }
                s
            }
            ImageOpKind::Resize { width, height, .. } => {
                // [H, W, C] -> [new_H, new_W, C]
                let mut s = input_shape.to_vec();
                if s.len() >= 2 {
                    s[0] = *height as usize;
                    s[1] = *width as usize;
                }
                s
            }
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        // Most image ops preserve the input type (u8 -> u8)
        inputs[0]
    }

    fn memory_effect(&self) -> MemoryEffect {
        match &self.kind {
            // Thresholding is pixel-wise and can support strides (assuming iterators)
            ImageOpKind::Threshold(_) => MemoryEffect::StridePreserving,

            // These require the `image` crate which generally expects contiguous buffers
            // or specific layouts that simple strides might not match (e.g. padding).
            // Resize definitely creates a new buffer layout.
            ImageOpKind::Resize { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::Blur { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::Grayscale => MemoryEffect::RequiresContiguous,
        }
    }
}
