//! Channel selection and reordering.

#[allow(unused_imports)]
use crate::ops::ParamExt as _;
use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::{ViewDto, ViewOp};

use super::{OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use view_buffer::ops::OpShape;

/// Extract a single channel from a multi-channel image: a 2D [H, W] buffer
/// from a [H, W, C] input.
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").channel_select(index=0)  # Red channel
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ChannelSelect {
    /// Channel index to extract (0-based). Accepts a Polars expression for
    /// per-row dynamic selection.
    pub index: Param<u32>,
}

impl OpDef for ChannelSelect {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::DropChannelAxis)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ChannelSelect { index } = self;
        Ok(GraphStep::Buffer(ViewDto::View(ViewOp::ChannelSelect {
            index: index.resolve(row, ctx)? as usize,
        })))
    }
}
