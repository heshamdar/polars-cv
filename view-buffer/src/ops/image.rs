use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::pad::{PadMode, PadPosition};
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};

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
    /// shape via [`ImageOpKind::shape`].
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
    /// How this kind's output shape follows from its input: the one
    /// authority the runner executes the deferred resizes with and the
    /// planner reads.
    pub fn shape(&self) -> OpShape {
        let k = |n: u32| Sym::Known(n as usize);
        match *self {
            ImageOpKind::Grayscale | ImageOpKind::Canny { .. } => OpShape::SingleChannel,
            ImageOpKind::Threshold(_)
            | ImageOpKind::Blur { .. }
            | ImageOpKind::ChannelSwap { .. }
            | ImageOpKind::HistogramEqualize
            | ImageOpKind::Erode { .. }
            | ImageOpKind::Dilate { .. }
            | ImageOpKind::MorphGradient { .. } => OpShape::Preserve,
            ImageOpKind::Resize { width, height, .. }
            | ImageOpKind::Letterbox { height, width, .. } => OpShape::SetHw {
                h: k(height),
                w: k(width),
            },
            ImageOpKind::ResizeScale {
                scale_x, scale_y, ..
            } => OpShape::ScaleHw {
                sy: Sym::Known(scale_y),
                sx: Sym::Known(scale_x),
            },
            ImageOpKind::ResizeToHeight { height, .. } => OpShape::HeightTo(k(height)),
            ImageOpKind::ResizeToWidth { width, .. } => OpShape::WidthTo(k(width)),
            ImageOpKind::ResizeMax { max_size, .. } => OpShape::LongSideTo(k(max_size)),
            ImageOpKind::ResizeMin { min_size, .. } => OpShape::ShortSideTo(k(min_size)),
            ImageOpKind::Pad {
                top,
                bottom,
                left,
                right,
                ..
            } => OpShape::Pad {
                top: k(top),
                bottom: k(bottom),
                left: k(left),
                right: k(right),
            },
            ImageOpKind::PadToSize { height, width, .. } => OpShape::AtLeastHw {
                h: k(height),
                w: k(width),
            },
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

// `Triangle` is surfaced under its API name "bilinear".
crate::naming::named_variants!(FilterType: "Image resize filter types." {
    "nearest" => Nearest,
    "bilinear" => Triangle,
    "catmullrom" => CatmullRom,
    "gaussian" => Gaussian,
    "lanczos3" => Lanczos3,
});

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageOp {
    pub kind: ImageOpKind,
}

impl Op for ImageOp {
    fn validate(
        &self,
        input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), crate::ops::validation::ValidationError> {
        use crate::ops::validation::{require_hw_or_hwc, require_single_channel, ValidationError};
        let shape = input_shapes[0];
        match &self.kind {
            ImageOpKind::Threshold(_)
            | ImageOpKind::Erode { .. }
            | ImageOpKind::Dilate { .. }
            | ImageOpKind::MorphGradient { .. } => require_single_channel(shape),
            // Image kernels read axes 0/1 as height/width and axis 2 as channels;
            // anything else would be passed through unchanged or read with an
            // axis silently dropped, so it is refused rather than degraded.
            ImageOpKind::Blur { .. }
            | ImageOpKind::HistogramEqualize
            | ImageOpKind::Pad { .. }
            | ImageOpKind::PadToSize { .. }
            | ImageOpKind::Canny { .. }
            | ImageOpKind::Grayscale => require_hw_or_hwc(shape),
            // The resampler handles one to four interleaved channels.
            ImageOpKind::Resize { .. }
            | ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Letterbox { .. } => {
                require_hw_or_hwc(shape)?;
                match shape.get(2) {
                    Some(&c) if c > 4 => Err(ValidationError::ShapeRequirement {
                        requirement: "at most 4 channels for resampling",
                        got: shape.to_vec(),
                    }),
                    _ => Ok(()),
                }
            }
            ImageOpKind::ChannelSwap { order } => match shape {
                [_, _, c] if order.len() == *c && order.iter().all(|&i| i < *c) => Ok(()),
                _ => Err(ValidationError::ShapeRequirement {
                    requirement: "[H, W, C] with one order entry per channel, each < C",
                    got: shape.to_vec(),
                }),
            },
        }
    }

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

    fn shape(&self) -> OpShape {
        self.kind.shape()
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

    fn identity_rule(&self) -> IdentityRule {
        match &self.kind {
            // A pad that adds nothing, or padding to the current size, copies
            // the input unchanged: `OpShape::preserves` decides whether this
            // one does. (Letterbox resamples first, so shape preservation does
            // *not* imply a no-op — it stays Never.)
            ImageOpKind::Pad { .. } | ImageOpKind::PadToSize { .. } => {
                IdentityRule::WhenShapePreserved
            }
            // Everything else transforms values or coordinates: resamples,
            // reduces/reorders channels, thresholds, filters, or smooths (a
            // blur with sigma 0 is a degenerate NaN kernel, not an identity).
            ImageOpKind::Letterbox { .. }
            | ImageOpKind::Resize { .. }
            | ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Blur { .. }
            | ImageOpKind::Threshold(_)
            | ImageOpKind::Grayscale
            | ImageOpKind::ChannelSwap { .. }
            | ImageOpKind::Erode { .. }
            | ImageOpKind::Dilate { .. }
            | ImageOpKind::MorphGradient { .. }
            | ImageOpKind::Canny { .. }
            | ImageOpKind::HistogramEqualize => IdentityRule::Never,
        }
    }

    fn is_spatial_window(&self) -> bool {
        // Image ops resample, pad, threshold or filter — none is an H/W crop.
        // The only spatial window is `ViewOp::Crop`.
        false
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        match &self.kind {
            // Per-element: threshold compares one pixel, grayscale combines the
            // channels at one pixel, channel_swap reorders channels in place.
            ImageOpKind::Threshold(_)
            | ImageOpKind::Grayscale
            | ImageOpKind::ChannelSwap { .. } => SpatialDependency::Pointwise,
            // Separable Gaussian of radius ceil(3σ) — the radius
            // `gaussian_kernel_1d` builds in the runner.
            ImageOpKind::Blur { sigma } => {
                SpatialDependency::neighborhood((sigma * 3.0).ceil() as usize)
            }
            // A ksize×ksize structuring element applied `iterations` times
            // reaches (ksize / 2) * iterations pixels out.
            ImageOpKind::Erode { ksize, iterations }
            | ImageOpKind::Dilate { ksize, iterations } => {
                SpatialDependency::neighborhood((*ksize as usize / 2) * *iterations as usize)
            }
            // Gradient = one dilate − one erode, each of half-extent ksize / 2.
            ImageOpKind::MorphGradient { ksize } => {
                SpatialDependency::neighborhood(*ksize as usize / 2)
            }
            // Canny's hysteresis links edges via connectivity that can span the
            // whole image, so its support is not bounded — treat as global.
            ImageOpKind::Canny { .. } => SpatialDependency::Global,
            // Histogram equalization builds a global CDF over all pixels.
            ImageOpKind::HistogramEqualize => SpatialDependency::Global,
            // Every resize variant resamples, and pad/letterbox offset the
            // content — all coordinate transforms.
            ImageOpKind::Resize { .. }
            | ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Pad { .. }
            | ImageOpKind::PadToSize { .. }
            | ImageOpKind::Letterbox { .. } => SpatialDependency::geometric(),
        }
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
