//! Ops that combine this node's buffer with other graph nodes' buffers.
//!
//! All are `lazy_only`: their builders live on `LazyPipelineExpr`, which owns
//! the graph wiring (a new node, its upstreams); the op carries the operand
//! nodes by id ([`NodeRef`]).

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::BinaryOp;

use super::{NodeRef, OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;

/// Declare the two-buffer arithmetic ops, one per `BinaryOp::NAMED` entry
/// (`binary_ops_are_exactly_the_named_table` pins the correspondence).
macro_rules! binary_ops {
    ($($ty:ident $doc:literal => $variant:ident;)+) => {$(
        #[doc = $doc]
        ///
        /// Domain: buffer → buffer (element-wise with another buffer node).
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        #[op(visibility = "lazy_only")]
        pub struct $ty {
            /// The other operand's node.
            #[param(positional)]
            pub other: NodeRef,
        }

        impl OpDef for $ty {
            fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
                let $ty { other } = self;
                Ok(GraphStep::Binary {
                    op: BinaryOp::$variant,
                    other: other.0.clone(),
                })
            }
        }
    )+};
}

binary_ops! {
    Add "Element-wise addition (saturating for integer dtypes)." => Add;
    Subtract "Element-wise subtraction (saturating for integer dtypes)." => Subtract;
    Multiply "Element-wise multiplication (saturating for integer dtypes)." => Multiply;
    Divide "Element-wise division (integer division by zero yields 0)." => Divide;
    Blend "Normalized multiplication, e.g. (a/255) * (b/255) * 255 for u8." => Blend;
    Ratio "Ratio a/b scaled to the dtype's full range." => Ratio;
    Maximum "Element-wise maximum." => Maximum;
    Minimum "Element-wise minimum." => Minimum;
    BitwiseAnd "Element-wise bitwise AND." => BitwiseAnd;
    BitwiseOr "Element-wise bitwise OR." => BitwiseOr;
    BitwiseXor "Element-wise bitwise XOR." => BitwiseXor;
}

/// Apply a binary mask to this image.
///
/// Domain: buffer → buffer
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(visibility = "lazy_only")]
pub struct ApplyMask {
    /// The mask's node.
    #[param(positional)]
    pub mask: NodeRef,
    /// If True, invert the mask (keep exterior, zero interior).
    #[param(default = false)]
    pub invert: Param<bool>,
}

impl OpDef for ApplyMask {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ApplyMask { mask, invert } = self;
        Ok(GraphStep::ApplyMask {
            mask: mask.0.clone(),
            invert: invert.resolve(row, ctx)?,
        })
    }
}

/// Merge single-channel buffers into one multi-channel image.
///
/// Domain: buffer → buffer ([H, W] → [H, W, C])
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(visibility = "lazy_only")]
pub struct ChannelMerge {
    /// The other single-channel operands' nodes, in channel order after this
    /// one.
    #[param(positional)]
    pub others: Vec<NodeRef>,
}

impl OpDef for ChannelMerge {
    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ChannelMerge { others } = self;
        if others.is_empty() {
            polars_bail!(ComputeError:
                "channel_merge requires at least one other channel expression");
        }
        Ok(GraphStep::ChannelMerge {
            others: others.iter().map(|n| n.0.clone()).collect(),
        })
    }
}
