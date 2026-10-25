use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{with_dtype, DType, DTypeCategory, OutputDTypeRule};
use crate::kernels::{blur, canny, channel, distance, equalize, grid, morph, pixel, rank, resize};
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
    /// Convert to grayscale (luminance 0.299R + 0.587G + 0.114B): an
    /// ``[H, W, C]`` image becomes ``[H, W, 1]``, or ``[H, W]`` with
    /// ``keepdims=False``. An ``[H, W]`` input is already gray and stays
    /// ``[H, W]`` either way.
    #[op(name = "grayscale", sample = {"keepdims": false})]
    Grayscale {
        /// Keep an ``[H, W, C]`` image's channel axis, as size 1 (default),
        /// or drop it.
        #[param(default = true)]
        keepdims: M::L<bool>,
    },
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
    /// Rank filter: each pixel the ``q``-th percentile of its ``height`` x
    /// ``width`` window (linear interpolation, as every reduction here), per
    /// channel; ``q=50`` is a median filter, 0 and 100 minimum and maximum
    /// filters. Robust to outliers a blur averages in: lesion-robust
    /// backgrounds, artifact detection, a running median along rows with
    /// ``height=1``. The result keeps the input's dtype (a fractional
    /// percentile is rounded).
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("array").rank_filter(height=1, width=25)
    #[op(name = "rank_filter", sample = {"height": 3, "width": 5, "q": 25.0,
                                        "border": "reflect"})]
    RankFilter {
        /// Window height: odd, >= 1. Accepts a Polars expression.
        height: M::V<u32>,
        /// Window width: odd, >= 1. Accepts a Polars expression.
        width: M::V<u32>,
        /// The percentile of each window, in [0, 100] (default 50: a median).
        /// Accepts a Polars expression.
        #[param(default = 50.0)]
        q: M::V<crate::ops::reduction::Percent>,
        /// Border handling (``"replicate"``, ``"zero"``, ``"reflect"``), as
        /// ``convolve2d``'s.
        #[param(default = "replicate")]
        border: M::V<crate::ops::filter::BorderMode>,
    },
    /// Morphological erosion: the local minimum over a footprint (a
    /// `ksize × ksize` square, or a disk), edges replicated. Requires
    /// single-channel input (e.g. after `.grayscale()` or `.threshold()`).
    ///
    /// Example:
    ///     >>> mask = Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3)
    #[op(name = "erode", sample = {"ksize": 3, "iterations": 1})]
    Erode {
        /// Size of the structuring element (the square's side, the disk's
        /// diameter). Must be odd and >= 1. Accepts a Polars expression for
        /// per-row dynamic values.
        #[param(default = 3)]
        ksize: M::V<u32>,
        /// Number of times the operation is applied. Accepts a Polars
        /// expression for per-row dynamic values.
        #[param(default = 1)]
        iterations: M::V<u32>,
        /// The structuring element: ``"square"`` (``ksize x ksize``) or
        /// ``"disk"`` (the pixels within Euclidean distance ``ksize // 2`` of
        /// the centre). Accepts a Polars expression for a per-row choice.
        #[param(default = "square")]
        footprint: M::V<Footprint>,
    },
    /// Morphological dilation: the local maximum over a footprint (a
    /// `ksize × ksize` square, or a disk), edges replicated. Requires
    /// single-channel input (e.g. after `.grayscale()` or `.threshold()`).
    ///
    /// Example:
    ///     >>> mask = Pipeline().source("image_bytes").grayscale().threshold(128).dilate(ksize=3)
    #[op(name = "dilate", sample = {"ksize": 5, "iterations": 1, "footprint": "disk"})]
    Dilate {
        /// Size of the structuring element (the square's side, the disk's
        /// diameter). Must be odd and >= 1. Accepts a Polars expression for
        /// per-row dynamic values.
        #[param(default = 3)]
        ksize: M::V<u32>,
        /// Number of times the operation is applied. Accepts a Polars
        /// expression for per-row dynamic values.
        #[param(default = 1)]
        iterations: M::V<u32>,
        /// The structuring element: ``"square"`` (``ksize x ksize``) or
        /// ``"disk"`` (the pixels within Euclidean distance ``ksize // 2`` of
        /// the centre). Accepts a Polars expression for a per-row choice.
        #[param(default = "square")]
        footprint: M::V<Footprint>,
    },
    /// Morphological gradient (dilate - erode): an edge outline. Requires
    /// single-channel input.
    ///
    /// Example:
    ///     >>> edges = Pipeline().source("image_bytes").grayscale().threshold(128).morphology_gradient(ksize=3)
    #[op(name = "morphology_gradient", sample = {"ksize": 3})]
    MorphGradient {
        /// Size of the structuring element. Must be odd and >= 1. Accepts a
        /// Polars expression for per-row dynamic values.
        #[param(default = 3)]
        ksize: M::V<u32>,
        /// The structuring element: ``"square"`` (``ksize x ksize``) or
        /// ``"disk"`` (the pixels within Euclidean distance ``ksize // 2`` of
        /// the centre). Accepts a Polars expression for a per-row choice.
        #[param(default = "square")]
        footprint: M::V<Footprint>,
    },
    /// Exact Euclidean distance transform of a mask: an f32 image of the same
    /// shape. Requires single-channel input.
    ///
    /// A pixel is foreground when it is non-zero (NaN included). Unsigned, each
    /// foreground pixel holds the distance from its centre to the nearest
    /// background pixel's centre and the background holds 0, as
    /// ``cv2.distanceTransform(DIST_L2, DIST_MASK_PRECISE)`` and
    /// ``scipy.ndimage.distance_transform_edt`` compute. Signed, every pixel
    /// holds its distance to the boundary between the two classes, the pixel
    /// edges ``extract_contours`` traces: negative inside, positive outside
    /// (as ``.point.signed_distance_to_contour``), ``±(d - 0.5)`` for the
    /// centre distance ``d`` to the nearest pixel of the other class. When a
    /// class has no pixel at all, the distance to it is infinite.
    ///
    /// Example:
    ///     >>> sdf = Pipeline().source("image_bytes").grayscale().threshold(128).distance_transform(signed=True)
    #[op(name = "distance_transform", sample = {"signed": true})]
    DistanceTransform {
        /// Signed distance to the mask's boundary (negative inside) instead of
        /// each foreground pixel's distance to the background. Accepts a
        /// Polars expression for a per-row choice.
        #[param(default = false)]
        signed: M::V<bool>,
    },
    /// The pixel coordinates of the image: an f32 ``[H, W, 2]`` buffer holding
    /// each pixel's centre ``(x + 0.5, y + 0.5)`` — the point ``rasterize``
    /// samples — or ``((x + 0.5) / W, (y + 0.5) / H)`` when *normalized*.
    ///
    /// Reads only the input's height and width, never its values, so it takes
    /// any dtype and channel count. The coordinates are in the frame of the
    /// buffer it follows: after a crop they start at the crop's corner. Combine
    /// them with ordinary ops for fields over positions (gradients, radial
    /// falloffs, vignettes).
    ///
    /// Example:
    ///     >>> dx2 = Pipeline().source("image_bytes").coordinate_grid().add_constant(-32.0).square()
    #[op(name = "coordinate_grid", sample = {"normalized": true})]
    CoordinateGrid {
        /// Divide x by the width and y by the height, so each lies in (0, 1).
        /// Accepts a Polars expression for a per-row choice.
        #[param(default = false)]
        normalized: M::V<bool>,
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
        // A rank filter's window has a centre pixel (its q is a `Percent`).
        if let ImageOpKind::RankFilter { height, width, .. } = self {
            for (name, side) in [("height", height), ("width", width)] {
                if let Some(n) = M::sym(side).known().filter(|n| n % 2 == 0) {
                    return Err(format!(
                        "rank_filter: {name} = {n}, but a window's side is odd (1, 3, 5, ...)"
                    ));
                }
            }
        }
        // A Gaussian's sigma is finite and non-negative; 0 is no blur. A
        // negative sigma has no kernel (it used to round to no blur), and NaN
        // or infinity none at all.
        if let ImageOpKind::Blur { sigma } = self {
            if let Some(s) = M::sym(sigma)
                .known()
                .filter(|s| !(s.is_finite() && *s >= 0.0))
            {
                return Err(format!(
                    "blur: sigma {s} is not a finite non-negative value"
                ));
            }
        }
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
            ImageOpKind::Grayscale { keepdims } => match M::lit(keepdims) {
                true => OpShape::SingleChannel,
                false => OpShape::DropChannelAxis,
            },
            ImageOpKind::Canny { .. } => OpShape::SingleChannel,
            ImageOpKind::Threshold { .. }
            | ImageOpKind::Blur { .. }
            | ImageOpKind::ChannelSwap { .. }
            | ImageOpKind::HistogramEqualize
            | ImageOpKind::Erode { .. }
            | ImageOpKind::Dilate { .. }
            | ImageOpKind::MorphGradient { .. }
            | ImageOpKind::RankFilter { .. }
            | ImageOpKind::DistanceTransform { .. } => OpShape::Preserve,
            // The input's own H and W, with an (x, y) pair at each pixel.
            ImageOpKind::CoordinateGrid { .. } => OpShape::Canvas { of: 0, channels: 2 },
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
    /// Area resampling (OpenCV's `INTER_AREA`): shrinking, each output pixel
    /// is the overlap-weighted mean of the input pixels its footprint covers;
    /// enlarging, bilinear.
    Area,
}

// `Triangle` is surfaced under its API name "bilinear".
crate::naming::named_variants!(FilterType: "Image resize filter types.\n\n- AREA: when shrinking, each output pixel is the overlap-weighted mean of the input pixels its footprint covers (OpenCV's ``INTER_AREA``), with no aliasing at large factors; when enlarging, bilinear." {
    "nearest" => Nearest,
    "bilinear" => Triangle,
    "catmullrom" => CatmullRom,
    "gaussian" => Gaussian,
    "lanczos3" => Lanczos3,
    "area" => Area,
});

/// The structuring element of `erode`, `dilate` and `morphology_gradient`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Footprint {
    /// The `ksize × ksize` square.
    Square,
    /// The pixels within Euclidean distance `ksize / 2` of the centre.
    Disk,
}

crate::naming::named_variants!(Footprint: "Structuring element of ``erode``, ``dilate`` and the morphology composites.\n\n- SQUARE: The ``ksize x ksize`` square (default).\n- DISK: The pixels within Euclidean distance ``ksize // 2`` of the centre." {
    "square" => Square,
    "disk" => Disk,
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
            | ImageOpKind::MorphGradient { .. }
            | ImageOpKind::DistanceTransform { .. } => require_single_channel(shape),
            ImageOpKind::RankFilter { .. } => require_hw_or_hwc(shape),
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
            | ImageOpKind::CoordinateGrid { .. }
            | ImageOpKind::Grayscale { .. } => require_hw_or_hwc(shape),
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
            ImageOpKind::Grayscale { .. } => "Grayscale",
            ImageOpKind::Canny { .. } => "Canny",
            ImageOpKind::HistogramEqualize => "HistogramEqualize",
            ImageOpKind::Erode { .. } => "Erode",
            ImageOpKind::RankFilter { .. } => "RankFilter",
            ImageOpKind::Dilate { .. } => "Dilate",
            ImageOpKind::MorphGradient { .. } => "MorphGradient",
            ImageOpKind::DistanceTransform { .. } => "DistanceTransform",
            ImageOpKind::CoordinateGrid { .. } => "CoordinateGrid",
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
            // themselves (`resize_pixels`, `resample::nearest`). The
            // other dtypes read any view: i8/i16 through their cast to f32,
            // the 32/64-bit integers and f64 through `resample::convolve`'s
            // one read into f64. A planned pack would be a second copy.
            ImageOpKind::Resize { .. }
            | ImageOpKind::ResizeScale { .. }
            | ImageOpKind::ResizeToHeight { .. }
            | ImageOpKind::ResizeToWidth { .. }
            | ImageOpKind::ResizeMax { .. }
            | ImageOpKind::ResizeMin { .. }
            | ImageOpKind::Letterbox { .. } => MemoryEffect::PacksOwnRows,
            // Read their input one packed row at a time (`to_dense_rows`).
            ImageOpKind::Blur { .. }
            | ImageOpKind::Erode { .. }
            | ImageOpKind::Dilate { .. }
            | ImageOpKind::MorphGradient { .. }
            | ImageOpKind::DistanceTransform { .. }
            | ImageOpKind::Pad { .. }
            | ImageOpKind::PadToSize { .. }
            | ImageOpKind::ChannelSwap { .. } => MemoryEffect::RequiresDenseRows,
            // Reads a u8 input one packed row at a time and packs one without
            // packed rows itself (`apply_histogram_equalize`); any other
            // dtype is first converted to u8 (`ImageU8`), which reads any
            // view, so a planned pack would only copy the wide input.
            ImageOpKind::HistogramEqualize => MemoryEffect::PacksOwnRows,
            // A per-pixel map (`grayscale_strided`): reads any view where it
            // lies, which measured faster than packing a transpose or a
            // horizontal flip first.
            ImageOpKind::Grayscale { .. } => MemoryEffect::StridePreserving,
            ImageOpKind::Canny { .. } => MemoryEffect::RequiresContiguous,
            // Reads one packed f64 copy of its input (`apply_rank_filter`).
            ImageOpKind::RankFilter { .. } => MemoryEffect::RequiresContiguous,
            // Reads only the shape: any layout, and nothing to pack.
            ImageOpKind::CoordinateGrid { .. } => MemoryEffect::StridePreserving,
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
            | ImageOpKind::Grayscale { .. }
            | ImageOpKind::ChannelSwap { .. }
            | ImageOpKind::Erode { .. }
            | ImageOpKind::Dilate { .. }
            | ImageOpKind::MorphGradient { .. }
            | ImageOpKind::RankFilter { .. }
            | ImageOpKind::DistanceTransform { .. }
            | ImageOpKind::CoordinateGrid { .. }
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
            | ImageOpKind::Grayscale { .. }
            | ImageOpKind::ChannelSwap { .. } => SpatialDependency::Pointwise,
            // Separable Gaussian of radius ceil(3σ) — the radius
            // `gaussian_kernel_1d` builds in `kernels::blur`.
            ImageOpKind::Blur { sigma } => SpatialDependency::neighborhood_of(
                M::sym(sigma).map(|sigma| (sigma * 3.0).ceil() as usize),
            ),
            // A ksize×ksize structuring element applied `iterations` times
            // reaches (ksize / 2) * iterations pixels out.
            // A disk of the same ksize reaches no further than the square.
            ImageOpKind::Erode {
                ksize, iterations, ..
            }
            | ImageOpKind::Dilate {
                ksize, iterations, ..
            } => SpatialDependency::neighborhood_of(match (M::sym(ksize), M::sym(iterations)) {
                (Sym::Known(k), Sym::Known(n)) => Sym::Known((k as usize / 2) * n as usize),
                _ => Sym::PerRow,
            }),
            // A height x width window reaches its half-sides out.
            ImageOpKind::RankFilter { height, width, .. } => {
                SpatialDependency::neighborhood_of(match (M::sym(height), M::sym(width)) {
                    (Sym::Known(h), Sym::Known(w)) => Sym::Known(h.max(w) as usize / 2),
                    _ => Sym::PerRow,
                })
            }
            // Gradient = one dilate − one erode, each of half-extent ksize / 2.
            ImageOpKind::MorphGradient { ksize, .. } => {
                SpatialDependency::neighborhood_of(M::sym(ksize).map(|k| k as usize / 2))
            }
            // Canny's hysteresis links edges via connectivity that can span the
            // whole image, so its support is not bounded — treat as global.
            ImageOpKind::Canny { .. } => SpatialDependency::Global,
            // Histogram equalization builds a global CDF over all pixels.
            ImageOpKind::HistogramEqualize => SpatialDependency::Global,
            // The nearest pixel of the other class can be anywhere in the
            // image, so no window bounds what a pixel reads.
            ImageOpKind::DistanceTransform { .. } => SpatialDependency::Global,
            // A pixel's value is its position in the whole frame: a crop moved
            // across it changes every value, so nothing commutes with it.
            ImageOpKind::CoordinateGrid { .. } => SpatialDependency::Global,
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
            ImageOpKind::Grayscale { .. } => None,
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
            // Reads every dtype as f64 and writes its own back.
            ImageOpKind::RankFilter { .. } => None,
            ImageOpKind::MorphGradient { .. } => None,
            // Reads only whether each element is zero, in its own dtype.
            ImageOpKind::DistanceTransform { .. } => None,
            // Reads no value at all.
            ImageOpKind::CoordinateGrid { .. } => None,
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
            ImageOpKind::Grayscale { .. } => OutputDTypeRule::PreserveInput,
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
            ImageOpKind::RankFilter { .. } => OutputDTypeRule::PreserveInput,
            ImageOpKind::Dilate { .. } => OutputDTypeRule::PreserveInput,
            ImageOpKind::MorphGradient { .. } => OutputDTypeRule::PreserveInput,
            // A distance, whatever the mask's dtype.
            ImageOpKind::DistanceTransform { .. } => OutputDTypeRule::Fixed(DType::F32),
            // A position, whatever the image's dtype.
            ImageOpKind::CoordinateGrid { .. } => OutputDTypeRule::Fixed(DType::F32),
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

/// Applies an image operation to a buffer.
///
/// The conversion strategy depends on the operation's ``working_dtype()``:
///
/// - ``Some(DType::U8)``: the operation requires U8 data (blur).
///   Float inputs in [0.0, 1.0] are scaled to [0, 255].
/// - ``None``: the operation works on the input's native dtype
///   (resize, rotate, grayscale, threshold).
///   The buffer is passed through unchanged.
///
/// Applies an image operation to a buffer.
///
/// Handles dtype promotion (converting to the op's working dtype before
/// dispatch) and validates the output dtype contract in debug builds.
#[inline]
pub(crate) fn apply_image_inner(buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    let input_dtype = buf.dtype();

    // Only convert to U8 when the operation requires it.
    let work_buf = match op.working_dtype() {
        Some(DType::U8) => pixel::convert_to_u8_for_image(buf),
        Some(target) => buf.cast(target),
        None => buf, // Resize: use the input's native dtype
    };

    let result = apply_image_dispatch(work_buf, op.clone());

    debug_assert!(
        op.validate_output_dtype(input_dtype, result.dtype())
            .is_ok(),
        "ImageOp contract violation: {}",
        op.validate_output_dtype(input_dtype, result.dtype())
            .unwrap_err()
    );

    result
}

/// Inner implementation of image operations (the per-op dispatch, without plan sequencing).
#[inline]
fn apply_image_dispatch(work_buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    match op.kind {
        ImageOpKind::Threshold { value } => pixel::threshold_generic(work_buf, value),
        ImageOpKind::Grayscale { keepdims } => {
            let gray = pixel::grayscale_strided(work_buf);
            match (keepdims, gray.shape()) {
                // `[H, W, 1]`, packed: the same elements as `[H, W]`.
                (false, &[h, w, 1]) => gray.reshape(crate::core::layout::Dims::from_slice(&[h, w])),
                _ => gray,
            }
        }
        ImageOpKind::Resize {
            width,
            height,
            filter,
        } => resize::resize_strided(work_buf, width, height, filter),
        ImageOpKind::Canny {
            low_threshold,
            high_threshold,
        } => canny::apply_canny(work_buf, low_threshold, high_threshold),
        ImageOpKind::HistogramEqualize => equalize::apply_histogram_equalize(work_buf),
        ImageOpKind::RankFilter {
            height,
            width,
            q,
            border,
        } => rank::apply_rank_filter(&work_buf, height as usize, width as usize, q.q(), border),
        ImageOpKind::Erode {
            ksize,
            iterations,
            footprint,
        } => morph::apply_erode(work_buf, ksize, iterations, footprint),
        ImageOpKind::Dilate {
            ksize,
            iterations,
            footprint,
        } => morph::apply_dilate(work_buf, ksize, iterations, footprint),
        ImageOpKind::MorphGradient { ksize, footprint } => {
            morph::apply_morph_gradient(work_buf, ksize, footprint)
        }
        ImageOpKind::DistanceTransform { signed } => {
            distance::distance_transform(&work_buf, signed)
        }
        ImageOpKind::CoordinateGrid { normalized } => grid::coordinate_grid(&work_buf, normalized),
        ImageOpKind::Blur { sigma } => {
            let rows_buf = work_buf.to_dense_rows();
            // Every dtype in its own element type, accumulating in its
            // `DType::accumulator`.
            let dtype = rows_buf.dtype();
            with_dtype!(dtype, T => match dtype.accumulator() {
                DType::F64 => blur::separable_gaussian_blur_typed::<T, f64>(&rows_buf, sigma),
                _ => blur::separable_gaussian_blur_typed::<T, f32>(&rows_buf, sigma),
            })
        }
        // Deferred resizes: dimensions come from the kind's shape — the same
        // authority the planner reads — then the shared resize kernel runs.
        ref kind @ (ImageOpKind::ResizeScale { .. }
        | ImageOpKind::ResizeToHeight { .. }
        | ImageOpKind::ResizeToWidth { .. }
        | ImageOpKind::ResizeMax { .. }
        | ImageOpKind::ResizeMin { .. }) => {
            let shape = work_buf.shape();
            let [h, w] = kind
                .shape()
                .concrete(&[&shape[..2]])
                .expect("a resize op's output shape over a known H/W is known")[..]
            else {
                unreachable!("an H/W shape over a rank-2 input is rank 2")
            };
            let filter = match kind {
                ImageOpKind::ResizeScale { filter, .. }
                | ImageOpKind::ResizeToHeight { filter, .. }
                | ImageOpKind::ResizeToWidth { filter, .. }
                | ImageOpKind::ResizeMax { filter, .. }
                | ImageOpKind::ResizeMin { filter, .. } => *filter,
                _ => unreachable!(),
            };
            resize::resize_strided(work_buf, w as u32, h as u32, filter)
        }
        ImageOpKind::Pad {
            top,
            bottom,
            left,
            right,
            value,
            mode,
        } => crate::ops::pad::pad(&work_buf, top, bottom, left, right, value, mode),
        ImageOpKind::PadToSize {
            height,
            width,
            position,
            value,
        } => crate::ops::pad::pad_to_size(&work_buf, height, width, position, value),
        ImageOpKind::Letterbox {
            height,
            width,
            value,
            filter,
        } => {
            let shape = work_buf.shape();
            let (fit_h, fit_w) =
                crate::ops::image::letterbox_fit(shape[0], shape[1], height, width);
            let resized = resize::resize_strided(work_buf, fit_w as u32, fit_h as u32, filter);
            crate::ops::pad::pad_to_size(
                &resized,
                height,
                width,
                crate::ops::pad::PadPosition::Center,
                value,
            )
        }
        ImageOpKind::ChannelSwap { ref order } => {
            let order: Vec<usize> = order.iter().map(|&i| i as usize).collect();
            channel::apply_channel_swap(&work_buf, &order)
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

    /// sigma 0 is no blur; a negative or non-finite sigma has no kernel.
    #[test]
    fn blur_refuses_a_sigma_that_is_not_finite_and_non_negative() {
        let op = |sigma: f32| ImageOpKind::<crate::mode::Exec>::Blur { sigma };
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.5] {
            assert!(op(bad).check().is_err(), "sigma {bad}");
        }
        assert!(op(0.0).check().is_ok());
        assert!(op(1.5).check().is_ok());
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
