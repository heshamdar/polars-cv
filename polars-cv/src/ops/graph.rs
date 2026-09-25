//! The graph-level ops: steps that read other graph nodes or input columns,
//! or that produce no buffer the engine's families describe. One variant per
//! wire op, in the same two modes as the engine families (see
//! `view_buffer::mode`); an `Exec` op lowers to its [`GraphStep`].

use view_buffer::geometry::label::{LabelReduction, LabelRegionMode};
use view_buffer::mode::{ColumnRef, Exec, Mode, NodeRef};
use view_buffer::BinaryOp;

use polars_cv_macros::{Ops, Resolve};

use crate::graph::step::GraphStep;

#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
pub enum GraphOp<M: Mode = Exec> {
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
    #[op(name = "add", visibility = "lazy_only", sample = {"other": "n0"})]
    Add {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
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
    #[op(name = "subtract", visibility = "lazy_only", sample = {"other": "n0"})]
    Subtract {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
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
    #[op(name = "multiply", visibility = "lazy_only", sample = {"other": "n0"})]
    Multiply {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
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
    #[op(name = "divide", visibility = "lazy_only", sample = {"other": "n0"})]
    Divide {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
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
    #[op(name = "blend", visibility = "lazy_only", sample = {"other": "n0"})]
    Blend {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
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
    #[op(name = "ratio", visibility = "lazy_only", sample = {"other": "n0"})]
    Ratio {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
    /// Element-wise maximum of two arrays, for compositing, clamping and
    /// non-linear image processing.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.maximum(b).sink("numpy")
    ///     ```
    #[op(name = "maximum", visibility = "lazy_only", sample = {"other": "n0"})]
    Maximum {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
    /// Element-wise minimum of two arrays, for compositing, clamping and
    /// non-linear image processing.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.minimum(b).sink("numpy")
    ///     ```
    #[op(name = "minimum", visibility = "lazy_only", sample = {"other": "n0"})]
    Minimum {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
    /// Element-wise bitwise AND: for binary masks (0/255), the intersection.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.bitwise_and(b).sink("numpy")
    ///     ```
    #[op(name = "bitwise_and", visibility = "lazy_only", sample = {"other": "n0"})]
    BitwiseAnd {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
    /// Element-wise bitwise OR: for binary masks (0/255), the union.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.bitwise_or(b).sink("numpy")
    ///     ```
    #[op(name = "bitwise_or", visibility = "lazy_only", sample = {"other": "n0"})]
    BitwiseOr {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
    /// Element-wise bitwise XOR: for binary masks (0/255), the symmetric
    /// difference.
    ///
    /// Example:
    ///     ```python
    ///     >>> a = pl.col("image1").cv.pipe(pipe1)
    ///     >>> b = pl.col("image2").cv.pipe(pipe2)
    ///     >>> result = a.bitwise_xor(b).sink("numpy")
    ///     ```
    #[op(name = "bitwise_xor", visibility = "lazy_only", sample = {"other": "n0"})]
    BitwiseXor {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
    /// Apply a binary mask to this image.
    ///
    /// Domain: buffer → buffer
    #[op(name = "apply_mask", visibility = "lazy_only", sample = {"mask": "n0", "invert": true})]
    ApplyMask {
        /// The mask's node.
        mask: NodeRef,
        /// If True, invert the mask (keep exterior, zero interior).
        #[param(default = false)]
        invert: M::V<bool>,
    },
    /// Merge single-channel buffers into one multi-channel image.
    ///
    /// Domain: buffer → buffer ([H, W] → [H, W, C])
    #[op(name = "channel_merge", visibility = "lazy_only", sample = {"others": ["n0", "n1"]})]
    ChannelMerge {
        /// The other single-channel operands' nodes, in channel order after this
        /// one.
        others: Vec<NodeRef>,
    },
    /// Declare the shape of the data at this point: its rank and any of the sizes
    /// of dimensions 0, 1 and 2.
    ///
    /// The planner applies the declaration (refusing one it contradicts), and
    /// execution checks it against every row, so everything downstream rests on a
    /// checked fact. The public `Pipeline.assert_shape` is sugar over this op.
    #[op(name = "assert_shape", visibility = "internal", sample = {"rank": 3, "dims": [8, null, 2]})]
    AssertShape {
        /// The rank, when declared (`assert_shape(dims=[...])` declares
        /// `len(dims)`).
        rank: Option<M::L<u32>>,
        /// The sizes of dimensions 0, 1 and 2; `None` declares nothing about that
        /// dimension. A per-row size is checked per row and is no plan-time fact.
        dims: [Option<M::V<u32>>; 3],
    },
    /// Extract buffer shape as a struct {height, width, channels}.
    ///
    /// Domain transition: buffer → vector
    #[op(name = "extract_shape", sample = {})]
    ExtractShape,
    /// Score contour regions against the current buffer values.
    ///
    /// This is the buffer-space variant of label reduction. It accepts contours
    /// via a Polars expression and returns one score per contour.
    ///
    /// Domain transition: buffer -> vector
    #[op(name = "label_reduce",
         sample = {"contours": {"$slot": 1}, "reduction": "mean", "region_mode": "bbox"})]
    LabelReduce {
        /// Contour-set expression (`List[Contour]`) to score.
        contours: ColumnRef,
        /// Reduction over contour region values (`"max"`, `"mean"`, `"sum"`).
        #[param(default = "max")]
        reduction: M::V<LabelReduction>,
        /// Region selection mode. ``"interior"`` — only pixels strictly inside the
        /// contour polygon. ``"boundary"`` — interior pixels *plus* pixels on the
        /// contour boundary (avoids zero-score artifacts for sub-pixel contours).
        /// ``"bbox"`` — all pixels within the bounding box.
        #[param(default = "interior")]
        region_mode: M::V<LabelRegionMode>,
    },
}

impl<M: Mode> GraphOp<M> {
    /// Refuse a parameter combination no row can execute: a channel merge
    /// needs at least one other channel.
    pub fn check(&self) -> Result<(), String> {
        if let GraphOp::ChannelMerge { others } = self {
            if others.is_empty() {
                return Err("channel_merge requires at least one other channel expression".into());
            }
        }
        Ok(())
    }
}

impl GraphOp {
    /// The executor step this op runs as.
    pub(crate) fn step(self) -> GraphStep {
        let (op, other) = match self {
            GraphOp::ApplyMask { mask, invert } => {
                return GraphStep::ApplyMask {
                    mask: mask.0,
                    invert,
                }
            }
            GraphOp::ChannelMerge { others } => {
                return GraphStep::ChannelMerge {
                    others: others.into_iter().map(|n| n.0).collect(),
                }
            }
            GraphOp::AssertShape { rank, dims } => {
                return GraphStep::AssertShape {
                    rank: rank.map(|r| r as usize),
                    dims: dims.map(|d| d.map(|d| d as usize)),
                }
            }
            GraphOp::ExtractShape => return GraphStep::ExtractShape,
            // The contour set is an operand column, not a value: the step
            // keeps its input position and reads the whole row's list itself.
            GraphOp::LabelReduce {
                contours,
                reduction,
                region_mode,
            } => {
                return GraphStep::LabelReduce {
                    contours_slot: contours.0,
                    reduction,
                    region_mode,
                }
            }
            GraphOp::Add { other } => (BinaryOp::Add, other),
            GraphOp::Subtract { other } => (BinaryOp::Subtract, other),
            GraphOp::Multiply { other } => (BinaryOp::Multiply, other),
            GraphOp::Divide { other } => (BinaryOp::Divide, other),
            GraphOp::Blend { other } => (BinaryOp::Blend, other),
            GraphOp::Ratio { other } => (BinaryOp::Ratio, other),
            GraphOp::Maximum { other } => (BinaryOp::Maximum, other),
            GraphOp::Minimum { other } => (BinaryOp::Minimum, other),
            GraphOp::BitwiseAnd { other } => (BinaryOp::BitwiseAnd, other),
            GraphOp::BitwiseOr { other } => (BinaryOp::BitwiseOr, other),
            GraphOp::BitwiseXor { other } => (BinaryOp::BitwiseXor, other),
        };
        GraphStep::Binary { op, other: other.0 }
    }
}
