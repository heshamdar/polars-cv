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
/// (`binary_ops_are_exactly_the_named_table` pins the correspondence). Each is
/// buffer → buffer, element-wise with another buffer node; its doc is the
/// generated `LazyPipelineExpr` method's docstring.
macro_rules! binary_ops {
    ($($(#[doc = $doc:literal])+ $ty:ident => $variant:ident;)+) => {$(
        $(#[doc = $doc])+
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        #[op(visibility = "lazy_only")]
        pub struct $ty {
            /// The expression to combine with, element-wise.
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
    /// Element-wise addition with another array.
    ///
    /// For u8/u16: saturating addition (clamps to the maximum, e.g. 255 for
    /// u8). For f32/f64: standard addition.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.add(b).sink("numpy")
    ///     ```
    Add => Add;
    /// Element-wise subtraction.
    ///
    /// For u8/u16: saturating subtraction (clamps to 0). For f32/f64:
    /// standard subtraction.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.subtract(b).sink("numpy")
    ///     ```
    Subtract => Subtract;
    /// Element-wise multiplication.
    ///
    /// For u8/u16: saturating multiplication (clamps to the maximum). For
    /// f32/f64: standard multiplication. For normalized image blending
    /// (values treated as [0, 1]), use ``blend`` instead.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.multiply(b).sink("numpy")
    ///     ```
    Multiply => Multiply;
    /// Element-wise division.
    ///
    /// For u8/u16: integer division, with division by zero yielding 0. For
    /// f32/f64: standard division.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.divide(b).sink("numpy")
    ///     ```
    Divide => Divide;
    /// Normalized blend (element-wise), for image blending/compositing.
    ///
    /// For u8: (a/255) * (b/255) * 255. For u16: (a/65535) * (b/65535) *
    /// 65535. For f32/f64: standard multiplication.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.blend(b).sink("numpy")
    ///     ```
    Blend => Blend;
    /// Scaled ratio: a/b scaled to the full range of the dtype.
    ///
    /// For u8: (a/b) * 255, clamped to [0, 255]. For u16: (a/b) * 65535,
    /// clamped to [0, 65535]. For f32/f64: standard division.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.ratio(b).sink("numpy")
    ///     ```
    Ratio => Ratio;
    /// Element-wise maximum of two arrays, for compositing, clamping and
    /// non-linear image processing.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.maximum(b).sink("numpy")
    ///     ```
    Maximum => Maximum;
    /// Element-wise minimum of two arrays, for compositing, clamping and
    /// non-linear image processing.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.minimum(b).sink("numpy")
    ///     ```
    Minimum => Minimum;
    /// Element-wise bitwise AND: for binary masks (0/255), the intersection.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.bitwise_and(b).sink("numpy")
    ///     ```
    BitwiseAnd => BitwiseAnd;
    /// Element-wise bitwise OR: for binary masks (0/255), the union.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.bitwise_or(b).sink("numpy")
    ///     ```
    BitwiseOr => BitwiseOr;
    /// Element-wise bitwise XOR: for binary masks (0/255), the symmetric
    /// difference.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.bitwise_xor(b).sink("numpy")
    ///     ```
    BitwiseXor => BitwiseXor;
}

/// Apply a binary mask to this image.
///
/// Domain: buffer → buffer
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(visibility = "lazy_only")]
pub struct ApplyMask {
    /// The mask's node.
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
