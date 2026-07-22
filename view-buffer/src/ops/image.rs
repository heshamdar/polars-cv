use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::cost::OpCost;
use crate::ops::pad::{PadMode, PadPosition};
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
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
    /// Resize by scale factors — output dimensions derive from the input
    /// shape via [`ImageOpKind::output_hw`].
    ResizeScale {
        scale_x: f32,
        scale_y: f32,
        filter: FilterType,
    },
    /// Resize to a target height, preserving aspect ratio.
    ResizeToHeight {
        height: u32,
        filter: FilterType,
    },
    /// Resize to a target width, preserving aspect ratio.
    ResizeToWidth {
        width: u32,
        filter: FilterType,
    },
    /// Resize so the longer side equals `max_size`, preserving aspect ratio.
    ResizeMax {
        max_size: u32,
        filter: FilterType,
    },
    /// Resize so the shorter side equals `min_size`, preserving aspect ratio.
    ResizeMin {
        min_size: u32,
        filter: FilterType,
    },
    /// Pad with per-side amounts and a border mode.
    Pad {
        top: u32,
        bottom: u32,
        left: u32,
        right: u32,
        value: f32,
        mode: PadMode,
    },
    /// Constant-pad to an exact size at a position (saturating: an input
    /// larger than the target is left unpadded on that axis).
    PadToSize {
        height: u32,
        width: u32,
        position: PadPosition,
        value: f32,
    },
    /// Letterbox: aspect-preserving resize, then center constant-pad to the
    /// exact target size.
    Letterbox {
        height: u32,
        width: u32,
        value: f32,
        filter: FilterType,
    },
    /// Reorder the channels of an `[H, W, C]` buffer (allocating).
    ChannelSwap {
        order: Vec<usize>,
    },
}

impl ImageOpKind {
    /// The output `(height, width)` this op produces for an `in_h × in_w`
    /// input.
    ///
    /// The **single authority** for geometric output dimensions, shared by
    /// [`Op::infer_shape`] (planning) and the execution runner, so planned
    /// and executed dimensions cannot diverge. Returns `None` for kinds that
    /// preserve the input dimensions.
    pub fn output_hw(&self, in_h: usize, in_w: usize) -> Option<(usize, usize)> {
        match self {
            ImageOpKind::Resize { width, height, .. } => Some((*height as usize, *width as usize)),
            ImageOpKind::ResizeScale {
                scale_x, scale_y, ..
            } => Some((
                (in_h as f32 * scale_y).round() as usize,
                (in_w as f32 * scale_x).round() as usize,
            )),
            ImageOpKind::ResizeToHeight { height, .. } => {
                let aspect = in_w as f32 / in_h as f32;
                Some((*height as usize, (*height as f32 * aspect).round() as usize))
            }
            ImageOpKind::ResizeToWidth { width, .. } => {
                let aspect = in_h as f32 / in_w as f32;
                Some(((*width as f32 * aspect).round() as usize, *width as usize))
            }
            ImageOpKind::ResizeMax { max_size, .. } => {
                let scale = *max_size as f32 / in_h.max(in_w) as f32;
                Some((
                    (in_h as f32 * scale).round() as usize,
                    (in_w as f32 * scale).round() as usize,
                ))
            }
            ImageOpKind::ResizeMin { min_size, .. } => {
                let scale = *min_size as f32 / in_h.min(in_w) as f32;
                Some((
                    (in_h as f32 * scale).round() as usize,
                    (in_w as f32 * scale).round() as usize,
                ))
            }
            ImageOpKind::Pad {
                top,
                bottom,
                left,
                right,
                ..
            } => Some((
                in_h + *top as usize + *bottom as usize,
                in_w + *left as usize + *right as usize,
            )),
            ImageOpKind::PadToSize { height, width, .. } => {
                Some((in_h.max(*height as usize), in_w.max(*width as usize)))
            }
            ImageOpKind::Letterbox { height, width, .. } => {
                Some((*height as usize, *width as usize))
            }
            _ => None,
        }
    }
}

/// Aspect-preserving fit of an `in_h × in_w` image inside `height × width`
/// (the intermediate resize dimensions of [`ImageOpKind::Letterbox`]).
pub fn letterbox_fit(in_h: usize, in_w: usize, height: u32, width: u32) -> (usize, usize) {
    let scale_h = height as f32 / in_h as f32;
    let scale_w = width as f32 / in_w as f32;
    let scale = scale_h.min(scale_w);
    (
        (in_h as f32 * scale).round() as usize,
        (in_w as f32 * scale).round() as usize,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FilterType {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

// `Triangle` is surfaced under its API name "bilinear"; the parser-only
// alias "triangle" is kept for backwards compatibility (see `ALIASES`).
crate::naming::named_variants!(FilterType {
    "nearest" => Nearest,
    "bilinear" => Triangle,
    "catmullrom" => CatmullRom,
    "gaussian" => Gaussian,
    "lanczos3" => Lanczos3,
});

impl FilterType {
    /// Additional parser-accepted spellings, not surfaced as canonical names.
    pub const ALIASES: &'static [(&'static str, FilterType)] =
        &[("triangle", FilterType::Triangle)];
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
            ImageOpKind::ResizeScale { .. } => "ResizeScale",
            ImageOpKind::ResizeToHeight { .. } => "ResizeToHeight",
            ImageOpKind::ResizeToWidth { .. } => "ResizeToWidth",
            ImageOpKind::ResizeMax { .. } => "ResizeMax",
            ImageOpKind::ResizeMin { .. } => "ResizeMin",
            ImageOpKind::Pad { .. } => "Pad",
            ImageOpKind::PadToSize { .. } => "PadToSize",
            ImageOpKind::Letterbox { .. } => "Letterbox",
            ImageOpKind::ChannelSwap { .. } => "ChannelSwap",
        }
    }

    fn output_rank_rule(&self) -> OutputRankRule {
        // Every image kind preserves rank: resize/blur/pad/threshold/morph keep
        // [H, W, C]; grayscale/canny keep the rank and set the channel dim via
        // the channel rule; channel_swap permutes within the channel dim.
        OutputRankRule::PreserveRank
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
            // Every geometric kind takes its output H/W from output_hw — the
            // same authority the runner executes with.
            ImageOpKind::Resize { .. }
            | ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Pad { .. }
            | ImageOpKind::PadToSize { .. }
            | ImageOpKind::Letterbox { .. } => {
                let mut s = input_shape.to_vec();
                if s.len() >= 2 {
                    if let Some((h, w)) = self.kind.output_hw(s[0], s[1]) {
                        s[0] = h;
                        s[1] = w;
                    }
                }
                s
            }
            ImageOpKind::ChannelSwap { .. } => input_shape.to_vec(),
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
            | ImageOpKind::MorphGradient { .. }
            | ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Pad { .. }
            | ImageOpKind::PadToSize { .. }
            | ImageOpKind::Letterbox { .. }
            | ImageOpKind::ChannelSwap { .. } => OutputChannelRule::PreserveChannels,
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
            ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Pad { .. }
            | ImageOpKind::PadToSize { .. }
            | ImageOpKind::Letterbox { .. }
            | ImageOpKind::ChannelSwap { .. } => MemoryEffect::RequiresContiguous,
        }
    }

    fn intrinsic_cost(&self) -> OpCost {
        // All image ops allocate new buffers
        OpCost::Allocating
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        // Every image kernel materializes a fresh contiguous buffer —
        // including Threshold, which can consume strided u8 input (hence
        // its StridePreserving memory_effect) but always writes a new
        // contiguous u8 mask, changing the element size for non-u8 input.
        None
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
            // Deferred resizes route through the same resize kernel; padding
            // and channel reorder are dtype-generic.
            ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Pad { .. }
            | ImageOpKind::PadToSize { .. }
            | ImageOpKind::Letterbox { .. }
            | ImageOpKind::ChannelSwap { .. } => None,
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
            // Geometric transforms and channel reorder preserve element dtype
            // (padding is dtype-generic for all ten dtypes).
            ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Pad { .. }
            | ImageOpKind::PadToSize { .. }
            | ImageOpKind::Letterbox { .. }
            | ImageOpKind::ChannelSwap { .. } => OutputDTypeRule::PreserveInput,
        }
    }
}
