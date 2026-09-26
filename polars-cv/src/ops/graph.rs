//! The graph-level ops: steps that read other graph nodes or input columns,
//! or that produce no buffer the engine's families describe. One variant per
//! wire op, in the same two modes as the engine families (see
//! `view_buffer::mode`), with the rules the planner reads from it.

use view_buffer::core::dtype::OutputDTypeRule;
use view_buffer::geometry::label::{LabelReduction, LabelRegionMode};
use view_buffer::mode::{ColumnRef, Exec, Mode, NodeRef};
use view_buffer::ops::{Domain, OpShape, SpatialDependency};
use view_buffer::{BinaryOp, IdentityRule, Op};

use polars_cv_macros::{Ops, Resolve};

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
    #[op(name = "add", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "subtract", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "multiply", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "divide", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "blend", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "ratio", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "maximum", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "minimum", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "bitwise_and", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "bitwise_or", visibility = LazyOnly, sample = {"other": "n0"})]
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
    #[op(name = "bitwise_xor", visibility = LazyOnly, sample = {"other": "n0"})]
    BitwiseXor {
        /// The expression to combine with, element-wise.
        other: NodeRef,
    },
    /// Apply a binary mask to this image.
    #[op(name = "apply_mask", visibility = LazyOnly, sample = {"mask": "n0", "invert": true})]
    ApplyMask {
        /// The mask's node.
        mask: NodeRef,
        /// If True, invert the mask (keep exterior, zero interior).
        #[param(default = false)]
        invert: M::V<bool>,
    },
    /// Merge single-channel buffers into one multi-channel image.
    ///
    /// Shape: ``[H, W]`` → ``[H, W, C]``.
    #[op(name = "channel_merge", visibility = LazyOnly, sample = {"others": ["n0", "n1"]})]
    ChannelMerge {
        /// The other single-channel operands' nodes, in channel order after this
        /// one.
        others: Vec<NodeRef>,
    },
    /// Declare the shape of the data at this point: sizes of its dimensions,
    /// and with them its rank when the declaration is exact.
    ///
    /// The planner applies the declaration (refusing one it contradicts), and
    /// execution checks it against every row, so everything downstream rests on a
    /// checked fact. The public `Pipeline.assert_shape` is sugar over this op.
    #[op(name = "assert_shape", visibility = Internal, sample = {"dims": [8, null, 2]})]
    AssertShape {
        /// The size of each dimension from the first; `None` declares nothing
        /// about that dimension. A per-row size is checked per row and is no
        /// plan-time fact.
        dims: Vec<Option<M::V<u32>>>,
        /// Whether `dims` is the whole shape, so the rank is its length
        /// (`assert_shape(dims=[...])`), or only its leading dimensions
        /// (`assert_shape(height=, width=, channels=)`).
        #[param(default = true)]
        exact: M::L<bool>,
    },
    /// Extract buffer shape as a struct {height, width, channels}.
    #[op(name = "extract_shape", sample = {})]
    ExtractShape,
    /// Score contour regions against the current buffer values.
    ///
    /// This is the buffer-space variant of label reduction. It accepts contours
    /// via a Polars expression and returns one score per contour.
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
    /// The element-wise arithmetic and its other operand, for a binary op.
    pub fn binary(&self) -> Option<(BinaryOp, &NodeRef)> {
        match self.role() {
            Role::Binary(op, other) => Some((op, other)),
            _ => None,
        }
    }

    /// What this op does, with its operands: the one exhaustive match
    /// over the ops, which every rule below and the executor read — a new op
    /// must be given a role, and each of those then fails to compile until it
    /// says what that role does.
    pub fn role(&self) -> Role<'_, M> {
        match self {
            GraphOp::Add { other } => Role::Binary(BinaryOp::Add, other),
            GraphOp::Subtract { other } => Role::Binary(BinaryOp::Subtract, other),
            GraphOp::Multiply { other } => Role::Binary(BinaryOp::Multiply, other),
            GraphOp::Divide { other } => Role::Binary(BinaryOp::Divide, other),
            GraphOp::Blend { other } => Role::Binary(BinaryOp::Blend, other),
            GraphOp::Ratio { other } => Role::Binary(BinaryOp::Ratio, other),
            GraphOp::Maximum { other } => Role::Binary(BinaryOp::Maximum, other),
            GraphOp::Minimum { other } => Role::Binary(BinaryOp::Minimum, other),
            GraphOp::BitwiseAnd { other } => Role::Binary(BinaryOp::BitwiseAnd, other),
            GraphOp::BitwiseOr { other } => Role::Binary(BinaryOp::BitwiseOr, other),
            GraphOp::BitwiseXor { other } => Role::Binary(BinaryOp::BitwiseXor, other),
            GraphOp::ApplyMask { mask, invert } => Role::ApplyMask { mask, invert },
            GraphOp::ChannelMerge { others } => Role::ChannelMerge { others },
            GraphOp::AssertShape { dims, exact } => Role::AssertShape { dims, exact },
            GraphOp::ExtractShape => Role::ExtractShape,
            GraphOp::LabelReduce {
                contours,
                reduction,
                region_mode,
            } => Role::LabelReduce {
                contours,
                reduction,
                region_mode,
            },
        }
    }

    /// Every domain this op can consume.
    ///
    /// A set rather than a single domain because binary ops consume any
    /// numeric container, which is `buffer` *and* `vector` (a perceptual hash
    /// is a 1-D u8 buffer encoded as a vector — `hash_a ^ hash_b` is how the
    /// library's own hamming distance starts). Declaring a single `Buffer`
    /// read as "images only" and was wrong; accepting every domain would
    /// stop rejecting `extract_contours()` into an element-wise op.
    pub fn input_domains(&self) -> Vec<Domain> {
        match self.role() {
            // A declaration describes whatever numeric container it follows.
            Role::Binary(..) | Role::AssertShape { .. } => vec![Domain::Buffer, Domain::Vector],
            Role::ApplyMask { .. }
            | Role::ChannelMerge { .. }
            | Role::ExtractShape
            | Role::LabelReduce { .. } => {
                vec![Domain::Buffer]
            }
        }
    }

    /// The domain this op produces from an `input` in its accepted domains.
    pub fn output_domain(&self, input: Domain) -> Domain {
        match self.role() {
            // Same container as its operands (`hash_a ^ hash_b` stays a
            // vector); a declaration describes the data, not its kind.
            Role::Binary(..) | Role::AssertShape { .. } => input,
            Role::ApplyMask { .. } | Role::ChannelMerge { .. } => Domain::Buffer,
            Role::ExtractShape | Role::LabelReduce { .. } => Domain::Vector,
        }
    }

    /// The rule that determines this op's output element dtype.
    pub fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self.role() {
            Role::Binary(op, _) => op.output_dtype_rule(),
            Role::ApplyMask { .. } | Role::ChannelMerge { .. } | Role::AssertShape { .. } => {
                OutputDTypeRule::PreserveInput
            }
            // Dimension reads and region scores are f64 values.
            Role::ExtractShape | Role::LabelReduce { .. } => OutputDTypeRule::ForceF64,
        }
    }

    /// How this op's output depends on the spatial extent of its input.
    pub fn spatial_dependency(&self) -> SpatialDependency {
        match self.role() {
            Role::Binary(op, _) => op.spatial_dependency(),
            // Mask blending and channel merge combine aligned buffers pixel
            // for pixel — spatially per-element.
            Role::ApplyMask { .. } | Role::ChannelMerge { .. } => SpatialDependency::Pointwise,
            // Dimension reads and region reductions aggregate over the whole
            // input, and a declaration is about the whole shape at this
            // point: a window moved across any of them changes what it reads.
            Role::ExtractShape | Role::LabelReduce { .. } | Role::AssertShape { .. } => {
                SpatialDependency::Global
            }
        }
    }

    /// Under what condition this op is a removable no-op.
    pub fn identity_rule(&self) -> IdentityRule {
        match self.role() {
            Role::Binary(op, _) => op.identity_rule(),
            // Masks, merges, dimension reads and region reductions combine or
            // derive from their inputs, and removing a declaration would
            // remove its check.
            Role::ApplyMask { .. }
            | Role::ChannelMerge { .. }
            | Role::ExtractShape
            | Role::LabelReduce { .. }
            | Role::AssertShape { .. } => IdentityRule::Never,
        }
    }

    /// Whether this op reads another graph node's buffer, so a spatial
    /// window hoisted past it would crop only this operand.
    pub fn reads_other_nodes(&self) -> bool {
        match self.role() {
            Role::Binary(..) | Role::ApplyMask { .. } | Role::ChannelMerge { .. } => true,
            Role::AssertShape { .. } | Role::ExtractShape | Role::LabelReduce { .. } => false,
        }
    }

    /// How the op's output shape follows from its inputs.
    pub fn shape(&self) -> OpShape {
        match self.role() {
            Role::Binary(op, _) => op.shape(),
            // The mask is blended into this buffer in place.
            Role::ApplyMask { .. } => OpShape::Preserve,
            // This `[H, W]` buffer and one per merged operand.
            Role::ChannelMerge { others } => OpShape::StackChannels(others.len() + 1),
            Role::ExtractShape => OpShape::InputRank,
            // One score per contour: as many as the row holds.
            Role::LabelReduce { .. } => OpShape::Dynamic,
            // A declaration's sizes are applied by the planner (`plan::declare`).
            Role::AssertShape { .. } => OpShape::Preserve,
        }
    }

    /// Refuse a parameter combination no row can execute: a channel merge
    /// needs at least one other channel.
    pub fn check(&self) -> Result<(), String> {
        match self {
            GraphOp::ChannelMerge { others } if others.is_empty() => {
                Err("channel_merge requires at least one other channel expression".into())
            }
            GraphOp::AssertShape { dims, exact } => {
                let exact = M::lit(exact);
                if dims.iter().all(Option::is_none) && !(exact && !dims.is_empty()) {
                    return Err("assert_shape() declares nothing: give dims=[...] or \
                                height=/width=/channels="
                        .into());
                }
                for (axis, d) in dims.iter().enumerate() {
                    if d.as_ref().and_then(view_buffer::mode::known::<M, u32>) == Some(0) {
                        return Err(format!(
                            "assert_shape({}=0): each size must be a positive int or None",
                            crate::plan::declared_name(axis, exact)
                        ));
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// What a graph op does, with its operands (see [`GraphOp::role`]).
pub enum Role<'a, M: Mode> {
    /// Element-wise arithmetic with another node's buffer.
    Binary(BinaryOp, &'a NodeRef),
    ApplyMask {
        mask: &'a NodeRef,
        invert: &'a M::V<bool>,
    },
    ChannelMerge {
        others: &'a [NodeRef],
    },
    AssertShape {
        dims: &'a [Option<M::V<u32>>],
        exact: &'a M::L<bool>,
    },
    ExtractShape,
    LabelReduce {
        contours: &'a ColumnRef,
        reduction: &'a M::V<LabelReduction>,
        region_mode: &'a M::V<LabelRegionMode>,
    },
}
