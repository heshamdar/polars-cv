//! Channel selection and reordering.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::{ImageOp, ImageOpKind, ViewDto, ViewOp};

use super::{OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;

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
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ChannelSelect { index } = self;
        Ok(GraphStep::Buffer(ViewDto::View(ViewOp::ChannelSelect {
            index: index.resolve(row, ctx)? as usize,
        })))
    }
}

/// Reorder channels in a multi-channel image.
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").channel_swap(order=[2, 1, 0])
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ChannelSwap {
    /// New channel ordering, e.g. [2, 1, 0] for RGB-to-BGR. **Each index may be
    /// a literal or a Polars expression**, so the permutation can vary per row.
    /// The list *length* is the channel count and must be literal.
    pub order: Vec<Param<u32>>,
}

impl OpDef for ChannelSwap {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ChannelSwap { order } = self;
        let order = order
            .iter()
            .map(|i| i.resolve(row, ctx).map(|i| i as usize))
            .collect::<PolarsResult<_>>()?;
        Ok(GraphStep::Buffer(ViewDto::Image(ImageOp {
            kind: ImageOpKind::ChannelSwap { order },
        })))
    }
}
