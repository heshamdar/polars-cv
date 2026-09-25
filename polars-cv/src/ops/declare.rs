//! Declarations: facts about the data the planner cannot work out itself.

#[allow(unused_imports)]
use crate::ops::ParamExt as _;
use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};

use super::{Literal, OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use view_buffer::ops::OpShape;

/// Declare the shape of the data at this point: its rank and any of the sizes
/// of dimensions 0, 1 and 2.
///
/// The planner applies the declaration (refusing one it contradicts), and
/// execution checks it against every row, so everything downstream rests on a
/// checked fact. The public `Pipeline.assert_shape` is sugar over this op.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(visibility = "internal")]
pub struct AssertShape {
    /// The rank, when declared (`assert_shape(dims=[...])` declares
    /// `len(dims)`).
    pub rank: Option<Literal<u32>>,
    /// The sizes of dimensions 0, 1 and 2; `None` declares nothing about that
    /// dimension. A per-row size is checked per row and is no plan-time fact.
    pub dims: [Option<Param<u32>>; 3],
}

impl OpDef for AssertShape {
    fn shape(&self) -> Option<OpShape> {
        // Declared sizes are applied by the planner itself
        // (`plan::declare`), since they hold even over an unknown rank.
        None
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let AssertShape { rank, dims } = self;
        let mut sizes = [None; 3];
        for (size, declared) in sizes.iter_mut().zip(dims) {
            *size = match declared {
                Some(p) => Some(p.resolve(row, ctx)? as usize),
                None => None,
            };
        }
        Ok(GraphStep::AssertShape {
            rank: rank.map(|r| r.get() as usize),
            dims: sizes,
        })
    }
}
