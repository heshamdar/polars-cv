//! `GraphStep` — the executor's operation vocabulary.
//!
//! A pipeline op resolves to either a fusable single-buffer engine op
//! (`Buffer(ViewDto)`, executed through `ViewExpr`) or a **graph-level step**:
//! an operation that needs graph wiring (other nodes' buffers, expression
//! columns) or changes the data domain (buffer → contour/scalar/vector).
//!
//! Node references and Polars expression column names live *here*, in the
//! plugin — the engine's `ViewDto` no longer carries graph topology. The
//! step's math still lives in view-buffer (`BinaryOp::execute`, `apply_mask`,
//! `apply_channel_merge`, `score_contours_on_buffer`, `ReductionOp::execute`,
//! `HistogramOp::execute`, geometry ops); the arms in the executor are thin
//! wiring.

use view_buffer::core::dtype::OutputDTypeRule;
use view_buffer::geometry::label::{LabelReduction, LabelRegionMode};
use view_buffer::ops::phash::PerceptualHashOp;
use view_buffer::ops::{Domain, OutputChannelRule, OutputRankRule, SpatialDependency};
use view_buffer::ops::{HistogramOp, ReductionOp};
use view_buffer::{BinaryOp, GeometryOp, IdentityRule, Op, ViewDto};

/// One resolved operation in a compiled graph node.
#[derive(Debug, Clone)]
pub(crate) enum GraphStep {
    /// A fusable single-buffer engine op, executed via `ViewExpr::apply_op`.
    Buffer(ViewDto),
    /// Two-buffer arithmetic; the second operand is another node's output.
    Binary { op: BinaryOp, other: String },
    /// Weighted mask blend; the mask is another node's output.
    ApplyMask { mask: String, invert: bool },
    /// Merge single-channel buffers from other nodes into one `[H, W, C]`.
    ChannelMerge { others: Vec<String> },
    /// Geometry op (extract_contours, rasterize, measures, transforms) —
    /// changes or consumes the contour domain.
    Geometry(GeometryOp),
    /// Reduction: global → scalar, axis → smaller buffer.
    Reduction(ReductionOp),
    /// Histogram: quantized → buffer, other modes → vector.
    Histogram(HistogramOp),
    /// Perceptual hash: image buffer → 1-D u8 fingerprint (vector domain).
    PerceptualHash(PerceptualHashOp),
    /// Read the buffer's dimensions as a vector.
    ExtractShape,
    /// Check a declared shape (`assert_shape`): the data passes through
    /// unchanged, or the row fails naming the declaration.
    AssertShape {
        rank: Option<usize>,
        dims: [Option<usize>; 3],
    },
    /// Score contour regions (from an expression column) over the buffer.
    LabelReduce {
        /// Input position of the contour-list column.
        contours_slot: usize,
        reduction: LabelReduction,
        region_mode: LabelRegionMode,
    },
}

impl GraphStep {
    /// Every domain this step can consume.
    ///
    /// A set rather than a single domain because two families genuinely accept
    /// more than one: binary ops and reductions consume any numeric container,
    /// which is `buffer` *and* `vector` (a perceptual hash is a 1-D u8 buffer
    /// that happens to be encoded as a vector — the library's own
    /// `hamming_distance` is `hash_a ^ hash_b -> reduce_popcount`, with both
    /// operands in `vector`).
    ///
    /// Declaring a single `Buffer` read as "images only" and was wrong; it went
    /// unnoticed because nothing enforced input domains from this contract
    /// until the planner started to. Accepting every domain instead
    /// would have been wrong in the other direction — it would stop
    /// rejecting `extract_contours().reduce_sum()`, which the suite pins.
    ///
    /// Exhaustive on purpose. This was the one contract method on `GraphStep`
    /// with a `_ =>` catch-all, so a new multi-domain variant would silently
    /// have been given a single domain — in the method `CLAUDE.md` names as
    /// *the* authority for accepted input domains, and which the Python
    /// planner validates against. The other five contract methods make a new
    /// variant a compile error; this one now does too.
    pub fn input_domains(&self) -> Vec<Domain> {
        match self {
            GraphStep::Binary { .. } | GraphStep::Reduction(_) | GraphStep::AssertShape { .. } => {
                vec![Domain::Buffer, Domain::Vector]
            }
            GraphStep::Buffer(_)
            | GraphStep::Geometry(_)
            | GraphStep::ApplyMask { .. }
            | GraphStep::ChannelMerge { .. }
            | GraphStep::Histogram(_)
            | GraphStep::PerceptualHash(_)
            | GraphStep::ExtractShape
            | GraphStep::LabelReduce { .. } => vec![self.input_domain()],
        }
    }

    /// The primary domain this step consumes.
    pub fn input_domain(&self) -> Domain {
        match self {
            GraphStep::Buffer(dto) => dto.input_domain(),
            GraphStep::Geometry(op) => op.input_domain(),
            GraphStep::Binary { .. }
            | GraphStep::Reduction(_)
            | GraphStep::ApplyMask { .. }
            | GraphStep::ChannelMerge { .. }
            | GraphStep::Histogram(_)
            | GraphStep::PerceptualHash(_)
            | GraphStep::ExtractShape
            | GraphStep::AssertShape { .. }
            | GraphStep::LabelReduce { .. } => Domain::Buffer,
        }
    }

    /// The domain this step produces from an `input` in its accepted domains.
    pub fn output_domain(&self, input: Domain) -> Domain {
        match self {
            GraphStep::Buffer(dto) => dto.output_domain(),
            GraphStep::Geometry(op) => op.output_domain(),
            GraphStep::Reduction(op) => op.output_domain(),
            GraphStep::Histogram(op) => op.output_domain(),
            // Same container as its operands: `hash_a ^ hash_b` stays a
            // vector, two images stay a buffer.
            GraphStep::Binary { .. } => input,
            // A declaration describes the data; it does not change its kind.
            GraphStep::AssertShape { .. } => input,
            GraphStep::ApplyMask { .. } | GraphStep::ChannelMerge { .. } => Domain::Buffer,
            // Perceptual hash produces a fixed-length 1-D fingerprint.
            GraphStep::PerceptualHash(_)
            | GraphStep::ExtractShape
            | GraphStep::LabelReduce { .. } => Domain::Vector,
        }
    }

    /// The rule that determines this step's output element dtype.
    pub fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self {
            GraphStep::Buffer(dto) => dto.output_dtype_rule(),
            GraphStep::Geometry(op) => op.output_dtype_rule(),
            GraphStep::Binary { op, .. } => op.output_dtype_rule(),
            GraphStep::Reduction(op) => op.output_dtype_rule(),
            GraphStep::Histogram(op) => op.output_dtype_rule(),
            GraphStep::PerceptualHash(op) => op.output_dtype_rule(),
            // Mask blending and channel merge preserve the buffer dtype.
            GraphStep::ApplyMask { .. } | GraphStep::ChannelMerge { .. } => {
                OutputDTypeRule::PreserveInput
            }
            // Dimension reads and region scores are f64 values.
            GraphStep::ExtractShape | GraphStep::LabelReduce { .. } => OutputDTypeRule::ForceF64,
            GraphStep::AssertShape { .. } => OutputDTypeRule::PreserveInput,
        }
    }

    /// The rule that determines how this step transforms the input rank.
    pub fn output_rank_rule(&self) -> OutputRankRule {
        match self {
            GraphStep::Buffer(dto) => dto.output_rank_rule(),
            GraphStep::Geometry(op) => op.output_rank_rule(),
            GraphStep::Binary { op, .. } => op.output_rank_rule(),
            GraphStep::Reduction(op) => op.output_rank_rule(),
            GraphStep::Histogram(op) => op.output_rank_rule(),
            GraphStep::PerceptualHash(op) => op.output_rank_rule(),
            GraphStep::ApplyMask { .. } => OutputRankRule::PreserveRank,
            // Merge always yields an [H, W, C] image.
            GraphStep::ChannelMerge { .. } => OutputRankRule::Fixed(3),
            // Dimension vectors and region scores are 1-D.
            GraphStep::ExtractShape | GraphStep::LabelReduce { .. } => OutputRankRule::Fixed(1),
            GraphStep::AssertShape { rank, .. } => {
                rank.map_or(OutputRankRule::PreserveRank, OutputRankRule::Fixed)
            }
        }
    }

    /// The rule that determines how this step transforms the channel count.
    pub fn output_channel_rule(&self) -> OutputChannelRule {
        match self {
            GraphStep::Buffer(dto) => dto.output_channel_rule(),
            GraphStep::Geometry(op) => op.output_channel_rule(),
            GraphStep::Binary { op, .. } => op.output_channel_rule(),
            GraphStep::Reduction(op) => op.output_channel_rule(),
            GraphStep::Histogram(op) => op.output_channel_rule(),
            GraphStep::PerceptualHash(op) => op.output_channel_rule(),
            GraphStep::ApplyMask { .. } => OutputChannelRule::PreserveChannels,
            // One channel per merged single-channel input (this + others).
            GraphStep::ChannelMerge { others } => OutputChannelRule::Fixed(others.len() + 1),
            GraphStep::ExtractShape | GraphStep::LabelReduce { .. } => {
                OutputChannelRule::NotApplicable
            }
            GraphStep::AssertShape { .. } => OutputChannelRule::PreserveChannels,
        }
    }

    /// How this step's output depends on the spatial extent of its input — the
    /// plan-time authority for whether a spatial window may commute with it.
    pub fn spatial_dependency(&self) -> SpatialDependency {
        match self {
            GraphStep::Buffer(dto) => dto.spatial_dependency(),
            GraphStep::Geometry(op) => op.spatial_dependency(),
            GraphStep::Binary { op, .. } => op.spatial_dependency(),
            GraphStep::Reduction(op) => op.spatial_dependency(),
            GraphStep::Histogram(op) => op.spatial_dependency(),
            GraphStep::PerceptualHash(op) => op.spatial_dependency(),
            // Mask blending and channel merge combine aligned buffers pixel for
            // pixel — spatially per-element.
            GraphStep::ApplyMask { .. } | GraphStep::ChannelMerge { .. } => {
                SpatialDependency::Pointwise
            }
            // Dimension reads and region reductions aggregate over the whole
            // input; neither admits a spatial-window reorder.
            GraphStep::ExtractShape | GraphStep::LabelReduce { .. } => SpatialDependency::Global,
            // A declaration is about the whole shape at this point: a window
            // moved across it would change what it checks.
            GraphStep::AssertShape { .. } => SpatialDependency::Global,
        }
    }

    /// Under what condition this step is a removable no-op — the plan-time
    /// authority an identity-elimination pass reads.
    pub fn identity_rule(&self) -> IdentityRule {
        match self {
            GraphStep::Buffer(dto) => dto.identity_rule(),
            GraphStep::Geometry(op) => op.identity_rule(),
            GraphStep::Binary { op, .. } => op.identity_rule(),
            GraphStep::Reduction(op) => op.identity_rule(),
            GraphStep::Histogram(op) => op.identity_rule(),
            GraphStep::PerceptualHash(op) => op.identity_rule(),
            // Mask blending, channel merge, dimension reads and region
            // reductions all combine or derive from their inputs — never a
            // no-op on a single buffer.
            GraphStep::ApplyMask { .. }
            | GraphStep::ChannelMerge { .. }
            | GraphStep::ExtractShape
            | GraphStep::LabelReduce { .. } => IdentityRule::Never,
            // It checks every row; removing it would remove the check.
            GraphStep::AssertShape { .. } => IdentityRule::Never,
        }
    }

    /// Whether this step reads another graph node's buffer, so a spatial
    /// window hoisted past it would crop only this operand. Exhaustive: a new
    /// multi-input step must say so.
    pub fn reads_other_nodes(&self) -> bool {
        match self {
            GraphStep::Binary { .. }
            | GraphStep::ApplyMask { .. }
            | GraphStep::ChannelMerge { .. } => true,
            GraphStep::Buffer(_)
            | GraphStep::Geometry(_)
            | GraphStep::Reduction(_)
            | GraphStep::Histogram(_)
            | GraphStep::PerceptualHash(_)
            | GraphStep::ExtractShape
            | GraphStep::AssertShape { .. }
            | GraphStep::LabelReduce { .. } => false,
        }
    }

    /// How the step's output shape follows from its input, for the buffer and
    /// geometry steps that have one; `None` for a graph-level step (binary,
    /// reduction, histogram, …), whose output the planner does not size. The
    /// engine's side of `typed_shape_is_the_resolved_steps`: the planner reads
    /// the typed op's symbolic `OpDef::shape`, never a resolved step's.
    #[cfg(test)]
    pub fn shape(&self) -> Option<view_buffer::ops::OpShape> {
        use view_buffer::ops::Op;
        match self {
            GraphStep::Buffer(dto) => Some(dto.as_op().shape()),
            GraphStep::Geometry(op) => Some(op.shape()),
            GraphStep::Binary { .. }
            | GraphStep::ApplyMask { .. }
            | GraphStep::ChannelMerge { .. }
            | GraphStep::Reduction(_)
            | GraphStep::Histogram(_)
            | GraphStep::PerceptualHash(_)
            | GraphStep::ExtractShape
            | GraphStep::AssertShape { .. }
            | GraphStep::LabelReduce { .. } => None,
        }
    }

    /// Whether this step is a hoistable H/W spatial window (a crop/ROI) — the
    /// plan-time authority the spatial-window pushdown reads.
    pub fn is_spatial_window(&self) -> bool {
        match self {
            GraphStep::Buffer(dto) => dto.is_spatial_window(),
            // Geometry, binary, reduction, histogram, perceptual-hash, mask,
            // merge, dimension-read and region-reduction steps are never an
            // H/W crop over a single buffer.
            GraphStep::Geometry(_)
            | GraphStep::Binary { .. }
            | GraphStep::Reduction(_)
            | GraphStep::Histogram(_)
            | GraphStep::PerceptualHash(_)
            | GraphStep::ApplyMask { .. }
            | GraphStep::ChannelMerge { .. }
            | GraphStep::ExtractShape
            | GraphStep::AssertShape { .. }
            | GraphStep::LabelReduce { .. } => false,
        }
    }
}
