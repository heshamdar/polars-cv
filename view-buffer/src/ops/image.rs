use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::execution::tiling::TilePolicy;
use crate::ops::cost::OpCost;
use crate::ops::traits::{MemoryEffect, Op};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageOpKind {
    Threshold(u8),
    Resize {
        width: u32,
        height: u32,
        filter: FilterType,
    },
    Blur {
        sigma: f32,
    },
    Grayscale,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FilterType {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageOp {
    pub kind: ImageOpKind,
}

impl Op for ImageOp {
    fn name(&self) -> &'static str {
        match &self.kind {
            ImageOpKind::Threshold(_) => "Threshold",
            ImageOpKind::Resize { .. } => "Resize",
            ImageOpKind::Blur { .. } => "Blur",
            ImageOpKind::Grayscale => "Grayscale",
        }
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input_shape = inputs[0];
        match &self.kind {
            ImageOpKind::Threshold(_) => input_shape.to_vec(),
            ImageOpKind::Blur { .. } => input_shape.to_vec(),
            ImageOpKind::Grayscale => {
                let mut s = input_shape.to_vec();
                if s.len() == 3 {
                    s[2] = 1;
                } else if s.len() == 2 {
                    s.push(1);
                }
                s
            }
            ImageOpKind::Resize { width, height, .. } => {
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
        match &self.kind {
            ImageOpKind::Grayscale => DType::U8,
            ImageOpKind::Threshold(_) => DType::U8,
            _ => inputs[0],
        }
    }

    fn memory_effect(&self) -> MemoryEffect {
        match &self.kind {
            ImageOpKind::Threshold(_) => MemoryEffect::StridePreserving,
            // Resize uses fast_image_resize which requires contiguous input
            ImageOpKind::Resize { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::Blur { .. } => MemoryEffect::RequiresContiguous,
            // Grayscale has strided implementation
            ImageOpKind::Grayscale => MemoryEffect::StridePreserving,
        }
    }

    fn intrinsic_cost(&self) -> OpCost {
        // All image ops allocate new buffers
        OpCost::Allocating
    }

    fn infer_strides(&self, _input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>> {
        match &self.kind {
            // Threshold preserves shape and strides
            ImageOpKind::Threshold(_) => Some(input_strides.to_vec()),
            // Grayscale changes shape (3 channels -> 1 channel) and always produces
            // contiguous output, so return None to trigger contiguous stride calculation
            ImageOpKind::Grayscale => None,
            // Resize changes shape and produces contiguous output
            ImageOpKind::Resize { .. } => None,
            // Blur preserves shape but produces contiguous output
            ImageOpKind::Blur { .. } => None,
        }
    }

    // --- Dtype Contract Methods ---

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        // Image operations accept all numeric types and handle casting internally
        // This allows pipelines like: normalize(f32) -> threshold to work automatically
        DTypeCategory::Numeric
    }

    fn working_dtype(&self) -> Option<DType> {
        // All image operations work internally with U8
        // For float inputs, we scale and convert to U8 first
        Some(DType::U8)
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        // Image operations always output U8
        OutputDTypeRule::Fixed(DType::U8)
    }

    #[inline]
    fn tile_policy(&self) -> TilePolicy {
        match &self.kind {
            // Point-wise operations - no pixel dependencies
            ImageOpKind::Threshold(_) => TilePolicy::PointWise,
            ImageOpKind::Grayscale => TilePolicy::PointWise,

            // Blur needs neighboring pixels - halo = 3*sigma (rounded up)
            ImageOpKind::Blur { sigma } => TilePolicy::LocalNeighborhood {
                halo: (*sigma * 3.0).ceil() as usize,
            },

            // Resize uses global resampling - cannot be tiled
            ImageOpKind::Resize { .. } => TilePolicy::Global,
        }
    }
}
