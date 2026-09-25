//! Reading a buffer's dimensions.

#[allow(unused_imports)]
use crate::ops::ParamExt as _;
use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};

use super::OpDef;
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use view_buffer::ops::OpShape;

/// Extract buffer shape as a struct {height, width, channels}.
///
/// Domain transition: buffer → vector
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ExtractShape {}

impl OpDef for ExtractShape {
    fn shape(&self) -> Option<OpShape> {
        None
    }

    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ExtractShape {} = self;
        Ok(GraphStep::ExtractShape)
    }
}
