use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::cost::OpCost;
use crate::ops::shape_rule::OutputChannelRule;
use crate::ops::traits::{MemoryEffect, Op};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageOpKind {
    Threshold(f64),
    Resize {
        width: u32,
        height: u32,
        filter: FilterType,
    },
    Blur {
        sigma: f32,
    },
    Grayscale,
    /// Canny edge detection (fused Gaussian + Sobel + NMS + hysteresis).
    Canny {
        low_threshold: f32,
        high_threshold: f32,
    },
    /// Histogram equalization for contrast enhancement.
    HistogramEqualize,
    /// Morphological erosion: output = local minimum over ksize×ksize neighborhood.
    /// Requires single-channel input.
    Erode {
        ksize: u32,
        iterations: u32,
    },
    /// Morphological dilation: output = local maximum over ksize×ksize neighborhood.
    /// Requires single-channel input.
    Dilate {
        ksize: u32,
        iterations: u32,
    },
    /// Morphological gradient: dilate − erode (edge outline).
    /// Requires single-channel input.
    MorphGradient {
        ksize: u32,
    },
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
            ImageOpKind::Canny { .. } => "Canny",
            ImageOpKind::HistogramEqualize => "HistogramEqualize",
            ImageOpKind::Erode { .. } => "Erode",
            ImageOpKind::Dilate { .. } => "Dilate",
            ImageOpKind::MorphGradient { .. } => "MorphGradient",
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
                }
                // 2D input stays 2D (already single-channel by definition)
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
            ImageOpKind::Canny { .. } => {
                // Output is single-channel binary edge map
                if input_shape.len() == 3 {
                    vec![input_shape[0], input_shape[1], 1]
                } else {
                    input_shape.to_vec()
                }
            }
            ImageOpKind::HistogramEqualize => input_shape.to_vec(),
            ImageOpKind::Erode { .. } => input_shape.to_vec(),
            ImageOpKind::Dilate { .. } => input_shape.to_vec(),
            ImageOpKind::MorphGradient { .. } => input_shape.to_vec(),
        }
    }

    fn output_channel_rule(&self) -> OutputChannelRule {
        match &self.kind {
            // Grayscale and Canny collapse to a single channel.
            ImageOpKind::Grayscale | ImageOpKind::Canny { .. } => OutputChannelRule::Fixed(1),
            // Threshold, Resize, Blur, HistogramEqualize and the morphological
            // operations are applied per-channel and preserve the channel count.
            ImageOpKind::Threshold(_)
            | ImageOpKind::Resize { .. }
            | ImageOpKind::Blur { .. }
            | ImageOpKind::HistogramEqualize
            | ImageOpKind::Erode { .. }
            | ImageOpKind::Dilate { .. }
            | ImageOpKind::MorphGradient { .. } => OutputChannelRule::PreserveChannels,
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        // Delegate to output_dtype_rule so there is a single source of truth.
        self.output_dtype_rule().resolve(inputs[0], None)
    }

    fn memory_effect(&self) -> MemoryEffect {
        match &self.kind {
            ImageOpKind::Threshold(_) => MemoryEffect::StridePreserving,
            // Resize uses fast_image_resize which requires contiguous input
            ImageOpKind::Resize { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::Blur { .. } => MemoryEffect::RequiresContiguous,
            // Grayscale changes shape (removes channel dim) so needs allocation
            ImageOpKind::Grayscale => MemoryEffect::RequiresContiguous,
            ImageOpKind::Canny { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::HistogramEqualize => MemoryEffect::RequiresContiguous,
            ImageOpKind::Erode { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::Dilate { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::MorphGradient { .. } => MemoryEffect::RequiresContiguous,
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
            ImageOpKind::Canny { .. } => None,
            ImageOpKind::HistogramEqualize => None,
            ImageOpKind::Erode { .. } => None,
            ImageOpKind::Dilate { .. } => None,
            ImageOpKind::MorphGradient { .. } => None,
        }
    }

    // --- Dtype Contract Methods ---

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        // Image operations accept all numeric types and handle casting internally
        // This allows pipelines like: normalize(f32) -> threshold to work automatically
        DTypeCategory::Numeric
    }

    fn working_dtype(&self) -> Option<DType> {
        match &self.kind {
            // Resize operates on the input's native dtype via fast_image_resize.
            ImageOpKind::Resize { .. } => None,
            // Grayscale uses BT.601 channel reduction — generic over dtype.
            ImageOpKind::Grayscale => None,
            // Threshold compares each element against a float threshold — generic.
            ImageOpKind::Threshold(_) => None,
            // Blur operates on the input's native dtype (u8/u16/f32 directly;
            // other dtypes via an f32 round-trip inside the kernel).
            ImageOpKind::Blur { .. } => None,
            // Canny converts internally to grayscale f32
            ImageOpKind::Canny { .. } => None,
            // Histogram equalize works on U8 data
            ImageOpKind::HistogramEqualize => Some(DType::U8),
            // Morphological ops work on native dtype (typically U8 binary masks)
            ImageOpKind::Erode { .. } => None,
            ImageOpKind::Dilate { .. } => None,
            ImageOpKind::MorphGradient { .. } => None,
        }
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match &self.kind {
            // Spatial transformations preserve the input dtype.
            ImageOpKind::Resize { .. } => OutputDTypeRule::PreserveInput,
            // Grayscale is a channel reduction that preserves element dtype.
            ImageOpKind::Grayscale => OutputDTypeRule::PreserveInput,
            // Threshold always produces a U8 binary mask (0 or 255).
            ImageOpKind::Threshold(_) => OutputDTypeRule::Fixed(DType::U8),
            // Blur preserves the input dtype (Gaussian smoothing is value-preserving).
            ImageOpKind::Blur { .. } => OutputDTypeRule::PreserveInput,
            // Canny produces a U8 binary edge map (0 or 255).
            ImageOpKind::Canny { .. } => OutputDTypeRule::Fixed(DType::U8),
            // Histogram equalize produces U8 output.
            ImageOpKind::HistogramEqualize => OutputDTypeRule::Fixed(DType::U8),
            // Morphological ops preserve the input dtype.
            ImageOpKind::Erode { .. } => OutputDTypeRule::PreserveInput,
            ImageOpKind::Dilate { .. } => OutputDTypeRule::PreserveInput,
            ImageOpKind::MorphGradient { .. } => OutputDTypeRule::PreserveInput,
        }
    }

}
