//! Reductions: buffer → scalar (global) or buffer (along an axis).

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::ops::ReductionOp;

use super::{Literal, OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;

fn reduction(op: ReductionOp) -> PolarsResult<GraphStep> {
    Ok(GraphStep::Reduction(op))
}

fn axis_of(axis: &Option<Literal<u32>>) -> Option<usize> {
    axis.map(|a| a.get() as usize)
}

/// Sum all elements in the buffer.
///
/// Domain transition: buffer → scalar
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ReduceSum {}

impl OpDef for ReduceSum {
    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ReduceSum {} = self;
        reduction(ReductionOp::Sum { axis: None })
    }
}

/// Count set bits (1s) in the buffer.
///
/// Domain transition: buffer → scalar
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ReducePopcount {}

impl OpDef for ReducePopcount {
    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ReducePopcount {} = self;
        reduction(ReductionOp::PopCount)
    }
}

/// Compute the q-th percentile of all values (linear interpolation, as numpy's
/// default).
///
/// Domain transition: buffer -> scalar
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ReducePercentile {
    /// Percentile to compute, in [0, 100]. Accepts a Polars expression for
    /// per-row dynamic values.
    #[param(positional)]
    pub q: Param<f64>,
}

impl OpDef for ReducePercentile {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ReducePercentile { q } = self;
        reduction(ReductionOp::Percentile {
            q: q.resolve(row, ctx)?,
        })
    }
}

/// Declare an optional-axis reduction (`None` = global → scalar).
macro_rules! axis_reductions {
    ($($ty:ident $doc:literal => $variant:ident;)+) => {$(
        #[doc = $doc]
        ///
        /// Domain transition: axis=None: buffer → scalar; axis=N: buffer →
        /// buffer (reduced shape).
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        pub struct $ty {
            /// Axis to reduce along. None for global reduction. It fixes the
            /// output rank, so it is literal-only.
            #[param(positional)]
            pub axis: Option<Literal<u32>>,
        }

        impl OpDef for $ty {
            fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
                let $ty { axis } = self;
                reduction(ReductionOp::$variant { axis: axis_of(axis) })
            }
        }
    )+};
}

axis_reductions! {
    ReduceMax "Reduce buffer by computing the maximum value." => Max;
    ReduceMin "Reduce buffer by computing the minimum value." => Min;
    ReduceMean "Compute arithmetic mean." => Mean;
}

/// Reduce buffer by computing the standard deviation.
///
/// Domain transition: axis=None: buffer -> scalar; axis=N: buffer -> buffer
/// (reduced shape).
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").reduce_std(ddof=1)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ReduceStd {
    /// Axis to reduce along. None for global reduction.
    #[param(positional)]
    pub axis: Option<Literal<u32>>,
    /// Delta degrees of freedom. 0 for population std (default), 1 for sample
    /// std. Accepts a Polars expression for per-row dynamic values.
    #[param(positional, default = 0)]
    pub ddof: Param<u8>,
}

impl OpDef for ReduceStd {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ReduceStd { axis, ddof } = self;
        reduction(ReductionOp::Std {
            axis: axis_of(axis),
            ddof: ddof.resolve(row, ctx)?,
        })
    }
}

/// Declare an arg-reduction, which always needs an axis.
macro_rules! arg_reductions {
    ($($ty:ident $doc:literal => $variant:ident;)+) => {$(
        #[doc = $doc]
        ///
        /// Unlike other reductions it always requires an axis: a global index
        /// is ambiguous for a multi-dimensional array. Domain transition:
        /// buffer → buffer (reduced shape, i64 dtype).
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        pub struct $ty {
            /// Axis along which to find the index.
            #[param(positional)]
            pub axis: Literal<u32>,
        }

        impl OpDef for $ty {
            fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
                let $ty { axis } = self;
                reduction(ReductionOp::$variant {
                    axis: axis.get() as usize,
                })
            }
        }
    )+};
}

arg_reductions! {
    ReduceArgmax "Index of the maximum value along an axis." => ArgMax;
    ReduceArgmin "Index of the minimum value along an axis." => ArgMin;
}

/// Extract buffer shape as a struct {height, width, channels}.
///
/// Domain transition: buffer → vector
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ExtractShape {}

impl OpDef for ExtractShape {
    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ExtractShape {} = self;
        Ok(GraphStep::ExtractShape)
    }
}
