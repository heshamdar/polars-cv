use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::mode::{known, size, Exec, Mode};
use crate::ops::pad::{PadMode, PadPosition};
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use polars_cv_macros::{Ops, Resolve};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The image ops: one variant per wire op (see `crate::mode`). Each
/// variant's doc comment is its Python docstring and each field's its `Args:`
/// entry; the kernels read the `Exec` form.
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
pub enum ImageOpKind<M: Mode = Exec> {
    /// Apply binary threshold: a U8 mask, 255 where the element exceeds `value`
    /// and 0 elsewhere (for u8 input typically 0-255; for [0, 1] floats e.g. 0.5).
    #[op(name = "threshold", sample = {"value": 128.0})]
    Threshold {
        /// Threshold value (int or float, or Polars expression).
        value: M::V<f64>,
    },
    /// Resize image to specified dimensions.
    ///
    /// Example:
    ///     >>> Pipeline().source("image_bytes").resize(height=224, width=224)
    #[op(name = "resize", sample = {"height": 4, "width": 4, "filter": "bilinear"})]
    Resize {
        /// Target height.
        height: M::V<u32>,
        /// Target width.
        width: M::V<u32>,
        /// Interpolation: "nearest", "bilinear", "lanczos3" (default).
        #[param(default = "lanczos3")]
        filter: M::V<FilterType>,
    },
    /// Apply Gaussian blur.
    #[op(name = "blur", sample = {"sigma": 1.0})]
    Blur {
        /// Standard deviation for Gaussian kernel.
        sigma: M::V<f32>,
    },
    /// Convert to grayscale (luminance 0.299R + 0.587G + 0.114B).
    #[op(name = "grayscale", sample = {})]
    Grayscale,
    /// Canny edge detection: Gaussian blur, Sobel gradients, non-maximum
    /// suppression and double-threshold hysteresis. Output is a U8 binary edge
    /// map (0 or 255).
    ///
    /// Example:
    ///     >>> edges = Pipeline().source("image_bytes").canny(low_threshold=50, high_threshold=150)
    #[op(name = "canny", sample = {"low_threshold": 50.0, "high_threshold": 150.0})]
    Canny {
        /// Lower hysteresis threshold.
        #[param(default = 50.0)]
        low_threshold: M::V<f32>,
        /// Upper hysteresis threshold.
        #[param(default = 150.0)]
        high_threshold: M::V<f32>,
    },
    /// Apply histogram equalization for contrast enhancement: map each pixel
    /// through the normalized CDF, per channel. Output is U8.
    ///
    /// Example:
    ///     >>> eq = Pipeline().source("image_bytes").grayscale().equalize_histogram()
    #[op(name = "equalize_histogram", sample = {})]
    HistogramEqualize,
    /// Morphological erosion (local minimum over a `ksize × ksize` square). Requires single-channel input (e.g. after `.grayscale()` or `.threshold()`).
    ///
    /// Example:
    ///     >>> mask = Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3)
    #[op(name = "erode", sample = {"ksize": 3, "iterations": 1})]
    Erode {
        /// Size of the square structuring element. Must be odd and >= 1.
        /// Accepts a Polars expression for per-row dynamic values.
        #[param(default = 3)]
        ksize: M::V<u32>,
        /// Number of times the operation is applied. Accepts a Polars
        /// expression for per-row dynamic values.
        #[param(default = 1)]
        iterations: M::V<u32>,
    },
    /// Morphological dilation (local maximum over a `ksize × ksize` square). Requires single-channel input (e.g. after `.grayscale()` or `.threshold()`).
    ///
    /// Example:
    ///     >>> mask = Pipeline().source("image_bytes").grayscale().threshold(128).dilate(ksize=3)
    #[op(name = "dilate", sample = {"ksize": 3, "iterations": 1})]
    Dilate {
        /// Size of the square structuring element. Must be odd and >= 1.
        /// Accepts a Polars expression for per-row dynamic values.
        #[param(default = 3)]
        ksize: M::V<u32>,
        /// Number of times the operation is applied. Accepts a Polars
        /// expression for per-row dynamic values.
        #[param(default = 1)]
        iterations: M::V<u32>,
    },
    /// Morphological gradient (dilate - erode): an edge outline. Requires
    /// single-channel input.
    ///
    /// Example:
    ///     >>> edges = Pipeline().source("image_bytes").grayscale().threshold(128).morphology_gradient(ksize=3)
    #[op(name = "morphology_gradient", sample = {"ksize": 3})]
    MorphGradient {
        /// Size of the square structuring element. Must be odd and >= 1. Accepts a
        /// Polars expression for per-row dynamic values.
        #[param(default = 3)]
        ksize: M::V<u32>,
    },
    /// Resize image by scale factor: `new_width = input_width * scale_x`,
    /// `new_height = input_height * scale_y`, computed at runtime.
    ///
    /// The public `Pipeline.resize_scale` is sugar over this op that also accepts
    /// one uniform `scale`.
    #[op(name = "resize_scale", visibility = "internal",
         sample = {"scale_x": 0.5, "scale_y": 0.5, "filter": "bilinear"})]
    ResizeScale {
        /// X (width) scale factor.
        scale_x: M::V<f32>,
        /// Y (height) scale factor.
        scale_y: M::V<f32>,
        /// Resize filter ("nearest", "bilinear", "lanczos3").
        #[param(default = "lanczos3")]
        filter: M::V<FilterType>,
    },
    /// Resize image to target height, preserving aspect ratio (width is computed at runtime).
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").resize_to_height(224)
    #[op(name = "resize_to_height", sample = {"height": 4, "filter": "bilinear"})]
    ResizeToHeight {
        /// Target height (literal or expression).
        height: M::V<u32>,
        /// Resize filter ("nearest", "bilinear", "lanczos3").
        #[param(default = "lanczos3")]
        filter: M::V<FilterType>,
    },
    /// Resize image to target width, preserving aspect ratio (height is computed at runtime).
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").resize_to_width(224)
    #[op(name = "resize_to_width", sample = {"width": 4, "filter": "bilinear"})]
    ResizeToWidth {
        /// Target width (literal or expression).
        width: M::V<u32>,
        /// Resize filter ("nearest", "bilinear", "lanczos3").
        #[param(default = "lanczos3")]
        filter: M::V<FilterType>,
    },
    /// Resize image so the maximum dimension equals target, preserving aspect ratio (200x100 with max_size=50 gives 50x25).
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").resize_max(224)
    #[op(name = "resize_max", sample = {"max_size": 4, "filter": "bilinear"})]
    ResizeMax {
        /// Target for the maximum dimension (literal or expression).
        max_size: M::V<u32>,
        /// Resize filter ("nearest", "bilinear", "lanczos3").
        #[param(default = "lanczos3")]
        filter: M::V<FilterType>,
    },
    /// Resize image so the minimum dimension equals target, preserving aspect ratio (200x100 with min_size=50 gives 100x50).
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").resize_min(224)
    #[op(name = "resize_min", sample = {"min_size": 4, "filter": "bilinear"})]
    ResizeMin {
        /// Target for the minimum dimension (literal or expression).
        min_size: M::V<u32>,
        /// Resize filter ("nearest", "bilinear", "lanczos3").
        #[param(default = "lanczos3")]
        filter: M::V<FilterType>,
    },
    /// Add padding to the image.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").pad(top=10, bottom=10)
    ///     >>> pipe = Pipeline().source("image_bytes").pad(left=20, right=20, value=128)
    #[op(name = "pad", sample = {"top": 1, "bottom": 1, "left": 1, "right": 1,
                                 "value": 0.0, "mode": "constant"})]
    Pad {
        /// Padding on top edge.
        #[param(default = 0)]
        top: M::V<u32>,
        /// Padding on bottom edge.
        #[param(default = 0)]
        bottom: M::V<u32>,
        /// Padding on left edge.
        #[param(default = 0)]
        left: M::V<u32>,
        /// Padding on right edge.
        #[param(default = 0)]
        right: M::V<u32>,
        /// Fill value for "constant" mode (default 0). Accepts a Polars expression
        /// for per-row dynamic values.
        #[param(default = 0.0)]
        value: M::V<f32>,
        /// Padding mode - "constant", "edge", "reflect", "symmetric".
        #[param(default = "constant")]
        mode: M::V<PadMode>,
    },
    /// Pad image to exact target size (computed at runtime). A larger image is
    /// not cropped - resize first if needed.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").pad_to_size(height=100, width=200)
    #[op(name = "pad_to_size", sample = {"height": 4, "width": 4, "position": "center",
                                         "value": 0.0})]
    PadToSize {
        /// Target height.
        height: M::V<u32>,
        /// Target width.
        width: M::V<u32>,
        /// Where to place original content: "center" (default), "top-left" or
        /// "bottom-right".
        #[param(default = "center")]
        position: M::V<PadPosition>,
        /// Fill value for padding (default 0). Accepts a Polars expression for
        /// per-row dynamic values.
        #[param(default = 0.0)]
        value: M::V<f32>,
    },
    /// Resize image maintaining aspect ratio and pad to exact target size: fit
    /// within the target, then pad with centered positioning.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").letterbox(height=224, width=224)
    #[op(name = "letterbox", sample = {"height": 4, "width": 4, "value": 0.0,
                                       "filter": "bilinear"})]
    Letterbox {
        /// Target height (literal or expression).
        height: M::V<u32>,
        /// Target width (literal or expression).
        width: M::V<u32>,
        /// Fill value for padding (default 0, typically black). Accepts a Polars
        /// expression for per-row dynamic values.
        #[param(default = 0.0)]
        value: M::V<f32>,
        /// Resampling filter for the resize step (default "lanczos3").
        #[param(default = "lanczos3")]
        filter: M::V<FilterType>,
    },
    /// Reorder channels in a multi-channel image.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").channel_swap(order=[2, 1, 0])
    #[op(name = "channel_swap", sample = {"order": [2, 1, 0]})]
    ChannelSwap {
        /// New channel ordering, e.g. [2, 1, 0] for RGB-to-BGR. **Each index may be
        /// a literal or a Polars expression**, so the permutation can vary per row.
        /// The list *length* is the channel count and must be literal.
        order: Vec<M::V<u32>>,
    },
}

impl<M: Mode> ImageOpKind<M> {
    /// Refuse a parameter combination no row can execute. Every image op's
    /// parameters are independent, so there is none.
    pub fn check(&self) -> Result<(), String> {
        Ok(())
    }

    /// How this op's output shape follows from its input — the one
    /// definition, read on the `Wire` op at plan time (a per-row parameter is
    /// `Sym::PerRow`) and on the `Exec` op at execution.
    pub fn shape(&self) -> OpShape {
        match self {
            ImageOpKind::Grayscale | ImageOpKind::Canny { .. } => OpShape::SingleChannel,
            ImageOpKind::Threshold { .. }
            | ImageOpKind::Blur { .. }
            | ImageOpKind::ChannelSwap { .. }
            | ImageOpKind::HistogramEqualize
            | ImageOpKind::Erode { .. }
            | ImageOpKind::Dilate { .. }
            | ImageOpKind::MorphGradient { .. } => OpShape::Preserve,
            ImageOpKind::Resize { width, height, .. }
            | ImageOpKind::Letterbox { height, width, .. } => OpShape::SetHw {
                h: size::<M>(height),
                w: size::<M>(width),
            },
            ImageOpKind::ResizeScale {
                scale_x, scale_y, ..
            } => OpShape::ScaleHw {
                sy: M::sym(scale_y),
                sx: M::sym(scale_x),
            },
            ImageOpKind::ResizeToHeight { height, .. } => OpShape::HeightTo(size::<M>(height)),
            ImageOpKind::ResizeToWidth { width, .. } => OpShape::WidthTo(size::<M>(width)),
            ImageOpKind::ResizeMax { max_size, .. } => OpShape::LongSideTo(size::<M>(max_size)),
            ImageOpKind::ResizeMin { min_size, .. } => OpShape::ShortSideTo(size::<M>(min_size)),
            ImageOpKind::Pad {
                top,
                bottom,
                left,
                right,
                ..
            } => OpShape::Pad {
                top: size::<M>(top),
                bottom: size::<M>(bottom),
                left: size::<M>(left),
                right: size::<M>(right),
            },
            ImageOpKind::PadToSize { height, width, .. } => OpShape::AtLeastHw {
                h: size::<M>(height),
                w: size::<M>(width),
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

/// An image op, as the engine's `ViewDto` carries it.
#[derive(Debug, Clone, PartialEq, Resolve)]
pub struct ImageOp<M: Mode = Exec> {
    pub kind: ImageOpKind<M>,
}

impl<M: Mode> Op for ImageOp<M> {
    fn validate(
        &self,
        input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), crate::ops::validation::ValidationError> {
        use crate::ops::validation::{require_hw_or_hwc, require_single_channel, ValidationError};
        let shape = input_shapes[0];
        match &self.kind {
            ImageOpKind::Threshold { .. }
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
            // A per-row index is checked per row.
            ImageOpKind::ChannelSwap { order } => match shape {
                [_, _, c]
                    if order.len() == *c
                        && order
                            .iter()
                            .all(|i| known::<M, u32>(i).is_none_or(|i| (i as usize) < *c)) =>
                {
                    Ok(())
                }
                _ => Err(ValidationError::ShapeRequirement {
                    requirement: "[H, W, C] with one order entry per channel, each < C",
                    got: shape.to_vec(),
                }),
            },
        }
    }

    fn name(&self) -> &'static str {
        match &self.kind {
            ImageOpKind::Threshold { .. } => "Threshold",
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
            ImageOpKind::Threshold { .. } => MemoryEffect::StridePreserving,
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
            | ImageOpKind::Threshold { .. }
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
            ImageOpKind::Threshold { .. }
            | ImageOpKind::Grayscale
            | ImageOpKind::ChannelSwap { .. } => SpatialDependency::Pointwise,
            // Separable Gaussian of radius ceil(3σ) — the radius
            // `gaussian_kernel_1d` builds in the runner.
            ImageOpKind::Blur { sigma } => SpatialDependency::neighborhood_of(
                M::sym(sigma).map(|sigma| (sigma * 3.0).ceil() as usize),
            ),
            // A ksize×ksize structuring element applied `iterations` times
            // reaches (ksize / 2) * iterations pixels out.
            ImageOpKind::Erode { ksize, iterations }
            | ImageOpKind::Dilate { ksize, iterations } => {
                SpatialDependency::neighborhood_of(match (M::sym(ksize), M::sym(iterations)) {
                    (Sym::Known(k), Sym::Known(n)) => Sym::Known((k as usize / 2) * n as usize),
                    _ => Sym::PerRow,
                })
            }
            // Gradient = one dilate − one erode, each of half-extent ksize / 2.
            ImageOpKind::MorphGradient { ksize } => {
                SpatialDependency::neighborhood_of(M::sym(ksize).map(|k| k as usize / 2))
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
            ImageOpKind::Threshold { .. } => None,
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
            ImageOpKind::Threshold { .. } => OutputDTypeRule::Fixed(DType::U8),
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

#[cfg(test)]
mod rule_tests {
    use super::*;
    use crate::mode::{Param, Wire};
    use crate::ops::spatial_rule::NeighborhoodSupport;

    /// A rule that reads a per-row value says so rather than reading a
    /// stand-in: a blur whose sigma is per-row has a per-row radius.
    #[test]
    fn a_per_row_sigma_plans_a_per_row_radius() {
        let blur = |sigma| ImageOp::<Wire> {
            kind: ImageOpKind::Blur { sigma },
        };
        let radius = |op: ImageOp<Wire>| match op.spatial_dependency() {
            SpatialDependency::Neighborhood(NeighborhoodSupport { radius }) => radius,
            other => panic!("a blur is a neighborhood, got {other:?}"),
        };
        assert_eq!(radius(blur(Param::Slot(1))), Sym::PerRow);
        assert_eq!(radius(blur(Param::Lit(1.0))), Sym::Known(3));
    }
}
