//! Colour-space conversion.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::ops::color::{ColorConvertOp, ColorSpace};
use view_buffer::ViewDto;

use super::{Literal, OpDef};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;

/// Convert between color spaces.
///
/// Domain: buffer → buffer
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").convert_color("rgb", "hsv")
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(python = "convert_color")]
pub struct CvtColor {
    /// Source color space (rgb, bgr, hsv, lab, ycbcr, gray).
    pub from_space: Literal<ColorSpace>,
    /// Target color space (rgb, bgr, hsv, lab, ycbcr, gray).
    pub to_space: Literal<ColorSpace>,
}

impl OpDef for CvtColor {
    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let CvtColor {
            from_space,
            to_space,
        } = self;
        Ok(GraphStep::Buffer(ViewDto::Color(ColorConvertOp {
            from: from_space.get(),
            to: to_space.get(),
        })))
    }
}
