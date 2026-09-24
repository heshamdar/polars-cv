//! Label reduction: score contour regions against the current buffer.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::geometry::label::{LabelReduction, LabelRegionMode};

use super::{ColumnRef, OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;

/// Score contour regions against the current buffer values.
///
/// This is the buffer-space variant of label reduction. It accepts contours
/// via a Polars expression and returns one score per contour.
///
/// Domain transition: buffer -> vector
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct LabelReduce {
    /// Contour-set expression (`List[Contour]`) to score.
    pub contours: ColumnRef,
    /// Reduction over contour region values (`"max"`, `"mean"`, `"sum"`).
    #[param(default = "max")]
    pub reduction: Param<LabelReduction>,
    /// Region selection mode. ``"interior"`` — only pixels strictly inside the
    /// contour polygon. ``"boundary"`` — interior pixels *plus* pixels on the
    /// contour boundary (avoids zero-score artifacts for sub-pixel contours).
    /// ``"bbox"`` — all pixels within the bounding box.
    #[param(default = "interior")]
    pub region_mode: Param<LabelRegionMode>,
}

impl OpDef for LabelReduce {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let LabelReduce {
            contours,
            reduction,
            region_mode,
        } = self;
        // The contour set is an operand column, not a value: the step keeps
        // its input position and reads the whole row's list itself.
        Ok(GraphStep::LabelReduce {
            contours_slot: contours.0,
            reduction: reduction.resolve(row, ctx)?,
            region_mode: region_mode.resolve(row, ctx)?,
        })
    }
}
