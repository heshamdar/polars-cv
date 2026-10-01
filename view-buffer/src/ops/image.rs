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
    /// Canny edge detection, as ``cv2.Canny(image, low, high)`` computes it:
    /// 3x3 Sobel gradients (L1 magnitude), non-maximum suppression and
    /// double-threshold hysteresis. There is no pre-blur — add ``.blur()``
    /// first for a smoothed edge map, as with OpenCV. A colour image uses, per
    /// pixel, the channel with the strongest gradient; alpha is ignored. Output
    /// is a U8 binary edge map (0 or 255).
    ///
    /// Example:
    ///     >>> edges = Pipeline().source("image_bytes").blur(sigma=1.4).canny(low_threshold=50, high_threshold=150)
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
    #[op(name = "resize_scale", visibility = Internal,
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
        // A scale factor is finite and positive: NaN, infinity and a negative
        // factor have no output size (`shape_rule::scaled_by`).
        if let ImageOpKind::ResizeScale {
            scale_x, scale_y, ..
        } = self
        {
            for (name, s) in [("scale_x", scale_x), ("scale_y", scale_y)] {
                if let Some(s) = M::sym(s).known().filter(|s| !(s.is_finite() && *s > 0.0)) {
                    return Err(format!(
                        "resize_scale: {name} {s} is not a finite positive factor"
                    ));
                }
            }
        }
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
/// (the intermediate resize dimensions of [`ImageOpKind::Letterbox`]): the
/// tighter side meets its target and the other is derived by the shape
/// rule's [`scaled_size`](crate::ops::shape_rule::scaled_size), the two
/// scales compared exactly (`height / in_h` vs `width / in_w`).
pub fn letterbox_fit(in_h: usize, in_w: usize, height: u32, width: u32) -> (usize, usize) {
    use crate::ops::shape_rule::scaled_size;
    let (height, width) = (height as usize, width as usize);
    if height as u128 * in_w as u128 <= width as u128 * in_h as u128 {
        (height, scaled_size(in_w, height, in_h).min(width))
    } else {
        (scaled_size(in_h, width, in_w).min(height), width)
    }
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
        input_shapes: &[&[crate::ops::Dim]],
        _input_dtypes: &[crate::PlannedDType],
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
            // Extending an axis by its own pixels needs one to read (numpy
            // refuses an empty axis for every mode but "constant"). A
            // per-row size or mode is checked when its row runs.
            ImageOpKind::Pad {
                top,
                bottom,
                left,
                right,
                mode,
                ..
            } => {
                require_hw_or_hwc(shape)?;
                let reads_pixels = M::sym(mode).known().is_some_and(|m| m != PadMode::Constant);
                let padded = |a: &M::V<u32>, b: &M::V<u32>| {
                    known::<M, u32>(a).is_some_and(|v| v > 0)
                        || known::<M, u32>(b).is_some_and(|v| v > 0)
                };
                let empty_and_extended = (shape[0].known() == Some(0) && padded(top, bottom))
                    || (shape[1].known() == Some(0) && padded(left, right));
                if reads_pixels && empty_and_extended {
                    return Err(ValidationError::ShapeRequirement {
                        requirement: "a non-empty axis to extend by its pixels (an empty one pads only with mode=\"constant\")",
                        got: shape.to_vec(),
                    });
                }
                Ok(())
            }
            ImageOpKind::Blur { .. }
            | ImageOpKind::HistogramEqualize
            | ImageOpKind::PadToSize { .. }
            | ImageOpKind::Canny { .. }
            | ImageOpKind::Grayscale => require_hw_or_hwc(shape),
            // The resampler handles one to four interleaved channels, and
            // needs a source pixel on each axis: an empty image has nothing
            // to interpolate and no aspect ratio to keep.
            ImageOpKind::Resize { .. }
            | ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Letterbox { .. } => {
                require_hw_or_hwc(shape)?;
                if shape[..2].iter().any(|d| d.known() == Some(0)) {
                    return Err(ValidationError::ShapeRequirement {
                        requirement: "a non-empty image to resample",
                        got: shape.to_vec(),
                    });
                }
                match shape.get(2).and_then(|c| c.known()) {
                    Some(c) if c > 4 => Err(ValidationError::ShapeRequirement {
                        requirement: "at most 4 channels for resampling",
                        got: shape.to_vec(),
                    }),
                    _ => Ok(()),
                }
            }
            // A per-row index is checked per row.
            // An unknown channel count is checked per row.
            ImageOpKind::ChannelSwap { order } => match shape {
                [_, _, c]
                    if c.known().is_none_or(|c| {
                        order.len() == c
                            && order
                                .iter()
                                .all(|i| known::<M, u32>(i).is_none_or(|i| (i as usize) < c))
                    }) =>
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
            // The resizes read rows packed within themselves (a crop, a
            // vertical flip) where they lie and pack any other layout
            // themselves (`resize_pixels`), so a planned materialize would
            // only copy a view they can read.
            ImageOpKind::Resize { .. }
            | ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Letterbox { .. } => MemoryEffect::StridePreserving,
            ImageOpKind::Blur { .. } => MemoryEffect::RequiresContiguous,
            // Reads rows packed within themselves in place and packs any
            // other layout itself (`grayscale_strided`).
            ImageOpKind::Grayscale => MemoryEffect::StridePreserving,
            ImageOpKind::Canny { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::HistogramEqualize => MemoryEffect::RequiresContiguous,
            ImageOpKind::Erode { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::Dilate { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::MorphGradient { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::Pad { .. }
            | ImageOpKind::PadToSize { .. }
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

    /// Extending an empty axis by its own pixels has nothing to read: numpy
    /// refuses it ("can't extend empty axis 0 using modes other than
    /// 'constant'"), and `edge`/`reflect`/`symmetric` panicked the engine. A
    /// constant pad, or no padding on the empty axis, is fine.
    #[test]
    fn pad_refuses_to_extend_an_empty_axis_by_its_pixels() {
        use crate::ops::{Dim, Op};
        let pad = |top: u32, left: u32, mode: PadMode| ImageOp {
            kind: ImageOpKind::<crate::mode::Exec>::Pad {
                top,
                bottom: 0,
                left,
                right: 0,
                value: 0.0,
                mode,
            },
        };
        let check = |op: &ImageOp, shape: [usize; 3]| {
            let dims: Vec<Dim> = shape.iter().map(|&n| Dim::Known(n)).collect();
            op.validate(&[&dims], &[crate::PlannedDType::Known(crate::DType::U8)])
        };
        for mode in [PadMode::Edge, PadMode::Reflect, PadMode::Symmetric] {
            let err = check(&pad(1, 0, mode), [0, 3, 1]).expect_err("rows");
            assert!(err.to_string().contains("empty"), "{err}");
            assert!(check(&pad(0, 1, mode), [3, 0, 1]).is_err(), "{mode:?} cols");
            assert!(
                check(&pad(0, 1, mode), [0, 3, 1]).is_ok(),
                "{mode:?}: rows unpadded"
            );
        }
        assert!(check(&pad(1, 1, PadMode::Constant), [0, 0, 1]).is_ok());
    }

    /// A scale factor must be finite and positive: NaN saturated to a
    /// 1-pixel axis, infinity to `usize::MAX` (an allocation failure), and a
    /// negative factor to the 1-pixel floor.
    #[test]
    fn resize_scale_refuses_a_factor_that_is_not_finite_and_positive() {
        let op = |sx: f32, sy: f32| ImageOpKind::<crate::mode::Exec>::ResizeScale {
            scale_x: sx,
            scale_y: sy,
            filter: FilterType::Triangle,
        };
        for bad in [f32::NAN, f32::INFINITY, 0.0, -0.5] {
            assert!(op(bad, 1.0).check().is_err(), "scale_x {bad}");
            assert!(op(1.0, bad).check().is_err(), "scale_y {bad}");
        }
        assert!(op(0.5, 2.0).check().is_ok());
    }

    /// A resampler needs at least one source pixel on each axis: an empty
    /// image is refused by the contract (at plan time when its size is
    /// known, per row otherwise), not resampled from nothing.
    #[test]
    fn resampling_refuses_an_empty_image() {
        use crate::ops::validation::ValidationError;
        use crate::ops::Dim::Known;
        let filter = FilterType::Triangle;
        for kind in [
            ImageOpKind::Resize {
                width: 4,
                height: 4,
                filter,
            },
            ImageOpKind::ResizeScale {
                scale_x: 2.0,
                scale_y: 2.0,
                filter,
            },
            ImageOpKind::ResizeToHeight { height: 4, filter },
            ImageOpKind::ResizeToWidth { width: 4, filter },
            ImageOpKind::ResizeMax {
                max_size: 4,
                filter,
            },
            ImageOpKind::ResizeMin {
                min_size: 4,
                filter,
            },
            ImageOpKind::Letterbox {
                height: 4,
                width: 4,
                value: 0.0,
                filter,
            },
        ] {
            let op = ImageOp::<crate::mode::Exec> { kind };
            for shape in [
                [Known(0), Known(3), Known(1)],
                [Known(3), Known(0), Known(1)],
            ] {
                let err = op.validate(&[&shape], &[]);
                assert!(
                    matches!(err, Err(ValidationError::ShapeRequirement { .. })),
                    "{op:?} over {shape:?}: {err:?}"
                );
            }
            assert!(op.validate(&[&[Known(3), Known(3), Known(1)]], &[]).is_ok());
        }
    }

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
