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
use view_buffer::ops::{Domain, OutputChannelRule, OutputRankRule};
use view_buffer::ops::{HistogramOp, ReductionOp};
use view_buffer::{BinaryOp, GeometryOp, Op, ViewDto};

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
    /// Score contour regions (from an expression column) over the buffer.
    LabelReduce {
        contours_col: String,
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
    /// until the planner started to. Widening those two to `Domain::Any`
    /// instead would have been wrong in the other direction — it would stop
    /// rejecting `extract_contours().reduce_sum()`, which the suite pins.
    pub fn input_domains(&self) -> Vec<Domain> {
        match self {
            GraphStep::Binary { .. } | GraphStep::Reduction(_) => {
                vec![Domain::Buffer, Domain::Vector]
            }
            _ => vec![self.input_domain()],
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
            | GraphStep::LabelReduce { .. } => Domain::Buffer,
        }
    }

    /// The domain this step produces.
    pub fn output_domain(&self) -> Domain {
        match self {
            GraphStep::Buffer(dto) => dto.output_domain(),
            GraphStep::Geometry(op) => op.output_domain(),
            GraphStep::Reduction(op) => op.output_domain(),
            GraphStep::Histogram(op) => op.output_domain(),
            GraphStep::Binary { .. }
            | GraphStep::ApplyMask { .. }
            | GraphStep::ChannelMerge { .. } => Domain::Buffer,
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
        }
    }
}
