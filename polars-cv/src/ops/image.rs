//! Image ops: resampling.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::ops::image::{FilterType, ImageOp, ImageOpKind};
use view_buffer::ViewDto;

use super::{OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;

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
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Resize {
            height,
            width,
            filter,
        } = self;
        Ok(GraphStep::Buffer(ViewDto::Image(ImageOp {
            kind: ImageOpKind::Resize {
                height: height.resolve(row, ctx)?,
                width: width.resolve(row, ctx)?,
                filter: filter.resolve(row, ctx)?,
            },
        })))
    }
}
