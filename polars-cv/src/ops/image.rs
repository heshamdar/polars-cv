//! Image ops: resampling.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::ops::image::{FilterType, ImageOp, ImageOpKind};
use view_buffer::ops::pad::{PadMode, PadPosition};
use view_buffer::ViewDto;

use super::{OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use view_buffer::ops::OpShape;

/// Resize image to specified dimensions.
///
/// Example:
///     >>> Pipeline().source("image_bytes").resize(height=224, width=224)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Resize {
    /// Target height.
    pub height: Param<u32>,
    /// Target width.
    pub width: Param<u32>,
    /// Interpolation: "nearest", "bilinear", "lanczos3" (default).
    #[param(default = "lanczos3")]
    pub filter: Param<FilterType>,
}

impl OpDef for Resize {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::SetHw {
            h: self.height.size(),
            w: self.width.size(),
        })
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Resize {
            height,
            width,
            filter,
        } = self;
        image(ImageOpKind::Resize {
            height: height.resolve(row, ctx)?,
            width: width.resolve(row, ctx)?,
            filter: filter.resolve(row, ctx)?,
        })
    }
}

fn image(kind: ImageOpKind) -> PolarsResult<GraphStep> {
    Ok(GraphStep::Buffer(ViewDto::Image(ImageOp { kind })))
}

/// Resize image by scale factor: `new_width = input_width * scale_x`,
/// `new_height = input_height * scale_y`, computed at runtime.
///
/// The public `Pipeline.resize_scale` is sugar over this op that also accepts
/// one uniform `scale`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(visibility = "internal")]
pub struct ResizeScale {
    /// X (width) scale factor.
    pub scale_x: Param<f32>,
    /// Y (height) scale factor.
    pub scale_y: Param<f32>,
    /// Resize filter ("nearest", "bilinear", "lanczos3").
    #[param(default = "lanczos3")]
    pub filter: Param<FilterType>,
}

impl OpDef for ResizeScale {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::ScaleHw {
            sy: self.scale_y.sym(),
            sx: self.scale_x.sym(),
        })
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ResizeScale {
            scale_x,
            scale_y,
            filter,
        } = self;
        image(ImageOpKind::ResizeScale {
            scale_x: scale_x.resolve(row, ctx)?,
            scale_y: scale_y.resolve(row, ctx)?,
            filter: filter.resolve(row, ctx)?,
        })
    }
}

/// Declare an aspect-preserving resize: one target size plus a filter.
macro_rules! aspect_resizes {
    ($($ty:ident { $field:ident: $field_doc:literal } $doc:literal $example:literal
        => $kind:ident, $shape:ident;)+) => {$(
        #[doc = $doc]
        ///
        /// Example:
        #[doc = $example]
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        pub struct $ty {
            #[doc = $field_doc]
            pub $field: Param<u32>,
            /// Resize filter ("nearest", "bilinear", "lanczos3").
            #[param(default = "lanczos3")]
            pub filter: Param<FilterType>,
        }

        impl OpDef for $ty {
            fn shape(&self) -> Option<OpShape> {
                Some(OpShape::$shape(self.$field.size()))
            }

            fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
                let $ty { $field, filter } = self;
                image(ImageOpKind::$kind {
                    $field: $field.resolve(row, ctx)?,
                    filter: filter.resolve(row, ctx)?,
                })
            }
        }
    )+};
}

aspect_resizes! {
    ResizeToHeight { height: "Target height (literal or expression)." }
        "Resize image to target height, preserving aspect ratio (width is computed at runtime)."
        "    >>> pipe = Pipeline().source(\"image_bytes\").resize_to_height(224)"
        => ResizeToHeight, HeightTo;
    ResizeToWidth { width: "Target width (literal or expression)." }
        "Resize image to target width, preserving aspect ratio (height is computed at runtime)."
        "    >>> pipe = Pipeline().source(\"image_bytes\").resize_to_width(224)"
        => ResizeToWidth, WidthTo;
    ResizeMax { max_size: "Target for the maximum dimension (literal or expression)." }
        "Resize image so the maximum dimension equals target, preserving aspect ratio (200x100 with max_size=50 gives 50x25)."
        "    >>> pipe = Pipeline().source(\"image_bytes\").resize_max(224)"
        => ResizeMax, LongSideTo;
    ResizeMin { min_size: "Target for the minimum dimension (literal or expression)." }
        "Resize image so the minimum dimension equals target, preserving aspect ratio (200x100 with min_size=50 gives 100x50)."
        "    >>> pipe = Pipeline().source(\"image_bytes\").resize_min(224)"
        => ResizeMin, ShortSideTo;
}

/// Add padding to the image.
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").pad(top=10, bottom=10)
///     >>> pipe = Pipeline().source("image_bytes").pad(left=20, right=20, value=128)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Pad {
    /// Padding on top edge.
    #[param(default = 0)]
    pub top: Param<u32>,
    /// Padding on bottom edge.
    #[param(default = 0)]
    pub bottom: Param<u32>,
    /// Padding on left edge.
    #[param(default = 0)]
    pub left: Param<u32>,
    /// Padding on right edge.
    #[param(default = 0)]
    pub right: Param<u32>,
    /// Fill value for "constant" mode (default 0). Accepts a Polars expression
    /// for per-row dynamic values.
    #[param(default = 0.0)]
    pub value: Param<f32>,
    /// Padding mode - "constant", "edge", "reflect", "symmetric".
    #[param(default = "constant")]
    pub mode: Param<PadMode>,
}

impl OpDef for Pad {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Pad {
            top: self.top.size(),
            bottom: self.bottom.size(),
            left: self.left.size(),
            right: self.right.size(),
        })
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Pad {
            top,
            bottom,
            left,
            right,
            value,
            mode,
        } = self;
        image(ImageOpKind::Pad {
            top: top.resolve(row, ctx)?,
            bottom: bottom.resolve(row, ctx)?,
            left: left.resolve(row, ctx)?,
            right: right.resolve(row, ctx)?,
            value: value.resolve(row, ctx)?,
            mode: mode.resolve(row, ctx)?,
        })
    }
}

/// Pad image to exact target size (computed at runtime). A larger image is
/// not cropped - resize first if needed.
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").pad_to_size(height=100, width=200)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct PadToSize {
    /// Target height.
    pub height: Param<u32>,
    /// Target width.
    pub width: Param<u32>,
    /// Where to place original content: "center" (default), "top-left" or
    /// "bottom-right".
    #[param(default = "center")]
    pub position: Param<PadPosition>,
    /// Fill value for padding (default 0). Accepts a Polars expression for
    /// per-row dynamic values.
    #[param(default = 0.0)]
    pub value: Param<f32>,
}

impl OpDef for PadToSize {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::AtLeastHw {
            h: self.height.size(),
            w: self.width.size(),
        })
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let PadToSize {
            height,
            width,
            position,
            value,
        } = self;
        image(ImageOpKind::PadToSize {
            height: height.resolve(row, ctx)?,
            width: width.resolve(row, ctx)?,
            position: position.resolve(row, ctx)?,
            value: value.resolve(row, ctx)?,
        })
    }
}

/// Resize image maintaining aspect ratio and pad to exact target size: fit
/// within the target, then pad with centered positioning.
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").letterbox(height=224, width=224)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Letterbox {
    /// Target height (literal or expression).
    pub height: Param<u32>,
    /// Target width (literal or expression).
    pub width: Param<u32>,
    /// Fill value for padding (default 0, typically black). Accepts a Polars
    /// expression for per-row dynamic values.
    #[param(default = 0.0)]
    pub value: Param<f32>,
    /// Resampling filter for the resize step (default "lanczos3").
    #[param(default = "lanczos3")]
    pub filter: Param<FilterType>,
}

impl OpDef for Letterbox {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::SetHw {
            h: self.height.size(),
            w: self.width.size(),
        })
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Letterbox {
            height,
            width,
            value,
            filter,
        } = self;
        image(ImageOpKind::Letterbox {
            height: height.resolve(row, ctx)?,
            width: width.resolve(row, ctx)?,
            value: value.resolve(row, ctx)?,
            filter: filter.resolve(row, ctx)?,
        })
    }
}

/// Convert to grayscale (luminance 0.299R + 0.587G + 0.114B).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Grayscale {}

impl OpDef for Grayscale {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::SingleChannel)
    }

    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Grayscale {} = self;
        image(ImageOpKind::Grayscale)
    }
}

/// Apply histogram equalization for contrast enhancement: map each pixel
/// through the normalized CDF, per channel. Output is U8.
///
/// Example:
///     >>> eq = Pipeline().source("image_bytes").grayscale().equalize_histogram()
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct EqualizeHistogram {}

impl OpDef for EqualizeHistogram {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Preserve)
    }

    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let EqualizeHistogram {} = self;
        image(ImageOpKind::HistogramEqualize)
    }
}

/// Apply binary threshold: a U8 mask, 255 where the element exceeds `value`
/// and 0 elsewhere (for u8 input typically 0-255; for [0, 1] floats e.g. 0.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Threshold {
    /// Threshold value (int or float, or Polars expression).
    pub value: Param<f64>,
}

impl OpDef for Threshold {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Preserve)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Threshold { value } = self;
        image(ImageOpKind::Threshold(value.resolve(row, ctx)?))
    }
}

/// Apply Gaussian blur.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Blur {
    /// Standard deviation for Gaussian kernel.
    pub sigma: Param<f32>,
}

impl OpDef for Blur {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Preserve)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Blur { sigma } = self;
        image(ImageOpKind::Blur {
            sigma: sigma.resolve(row, ctx)?,
        })
    }
}

/// Declare a repeated square-kernel morphology op (`ksize`, `iterations`).
macro_rules! morphology {
    ($($ty:ident $doc:literal $example:literal => $kind:ident;)+) => {$(
        #[doc = $doc]
        ///
        /// Example:
        #[doc = $example]
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        pub struct $ty {
            /// Size of the square structuring element. Must be odd and >= 1.
            /// Accepts a Polars expression for per-row dynamic values.
            #[param(default = 3)]
            pub ksize: Param<u32>,
            /// Number of times the operation is applied. Accepts a Polars
            /// expression for per-row dynamic values.
            #[param(default = 1)]
            pub iterations: Param<u32>,
        }

        impl OpDef for $ty {
            fn shape(&self) -> Option<OpShape> {
                Some(OpShape::Preserve)
            }

            fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
                let $ty { ksize, iterations } = self;
                image(ImageOpKind::$kind {
                    ksize: ksize.resolve(row, ctx)?,
                    iterations: iterations.resolve(row, ctx)?,
                })
            }
        }
    )+};
}

morphology! {
    Erode "Morphological erosion (local minimum over a `ksize × ksize` square). Requires single-channel input (e.g. after `.grayscale()` or `.threshold()`)."
        "    >>> mask = Pipeline().source(\"image_bytes\").grayscale().threshold(128).erode(ksize=3)"
        => Erode;
    Dilate "Morphological dilation (local maximum over a `ksize × ksize` square). Requires single-channel input (e.g. after `.grayscale()` or `.threshold()`)."
        "    >>> mask = Pipeline().source(\"image_bytes\").grayscale().threshold(128).dilate(ksize=3)"
        => Dilate;
}

/// Morphological gradient (dilate - erode): an edge outline. Requires
/// single-channel input.
///
/// Example:
///     >>> edges = Pipeline().source("image_bytes").grayscale().threshold(128).morphology_gradient(ksize=3)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct MorphologyGradient {
    /// Size of the square structuring element. Must be odd and >= 1. Accepts a
    /// Polars expression for per-row dynamic values.
    #[param(default = 3)]
    pub ksize: Param<u32>,
}

impl OpDef for MorphologyGradient {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Preserve)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let MorphologyGradient { ksize } = self;
        image(ImageOpKind::MorphGradient {
            ksize: ksize.resolve(row, ctx)?,
        })
    }
}

/// Canny edge detection: Gaussian blur, Sobel gradients, non-maximum
/// suppression and double-threshold hysteresis. Output is a U8 binary edge
/// map (0 or 255).
///
/// Example:
///     >>> edges = Pipeline().source("image_bytes").canny(low_threshold=50, high_threshold=150)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Canny {
    /// Lower hysteresis threshold.
    #[param(default = 50.0)]
    pub low_threshold: Param<f32>,
    /// Upper hysteresis threshold.
    #[param(default = 150.0)]
    pub high_threshold: Param<f32>,
}

impl OpDef for Canny {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::SingleChannel)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Canny {
            low_threshold,
            high_threshold,
        } = self;
        image(ImageOpKind::Canny {
            low_threshold: low_threshold.resolve(row, ctx)?,
            high_threshold: high_threshold.resolve(row, ctx)?,
        })
    }
}
