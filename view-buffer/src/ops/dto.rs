#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::geometry::ops::GeometryOp;
use crate::ops::binary::BinaryOp;
use crate::ops::color::ColorConvertOp;
use crate::ops::compute::ComputeOp;
use crate::ops::filter::ConvolveOp;
use crate::ops::histogram::HistogramOp;
use crate::ops::image::ImageOp;
use crate::ops::phash::PerceptualHashOp;
use crate::ops::reduction::ReductionOp;
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
use crate::ops::traits::Op;
use crate::ops::view::ViewOp;
use crate::ops::Domain;

/// A pure Data Transfer Object (DTO) for operation plans.
/// This separates the schema (what to do) from the execution graph (how to do it).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ViewDto {
    View(ViewOp),
    Compute(ComputeOp),
    Image(ImageOp),
    Geometry(GeometryOp),
    /// Perceptual hash operation - computes image fingerprint.
    PerceptualHash(PerceptualHashOp),
    /// Binary operation between two buffers.
    /// The second buffer is referenced by node ID (for graph execution).
    Binary {
        op: BinaryOp,
        other_node_id: String,
    },
    /// Apply a mask to the current buffer.
    /// The mask buffer is referenced by node ID (for graph execution).
    ApplyMask {
        mask_node_id: String,
        invert: bool,
    },
    /// Reduction operation (e.g., sum, mean, max) that reduces array to scalar or along axis.
    Reduction(ReductionOp),
    /// Histogram operation - computes bin counts, normalized histogram, quantized image, or edges.
    Histogram(HistogramOp),
    /// Resize by scale factor - dimensions computed at runtime.
    ResizeScale {
        scale_x: f32,
        scale_y: f32,
        filter: crate::ops::image::FilterType,
    },
    /// Resize to target height, preserving aspect ratio - width computed at runtime.
    ResizeToHeight {
        height: u32,
        filter: crate::ops::image::FilterType,
    },
    /// Resize to target width, preserving aspect ratio - height computed at runtime.
    ResizeToWidth {
        width: u32,
        filter: crate::ops::image::FilterType,
    },
    /// Resize so max dimension equals target, preserving aspect ratio.
    ResizeMax {
        max_size: u32,
        filter: crate::ops::image::FilterType,
    },
    /// Resize so min dimension equals target, preserving aspect ratio.
    ResizeMin {
        min_size: u32,
        filter: crate::ops::image::FilterType,
    },
    /// Pad with specified amounts and mode.
    Pad {
        top: u32,
        bottom: u32,
        left: u32,
        right: u32,
        value: f32,
        mode: PadMode,
    },
    /// Pad to exact size with positioning - dimensions computed at runtime.
    PadToSize {
        height: u32,
        width: u32,
        position: PadPosition,
        value: f32,
    },
    /// Letterbox: resize maintaining aspect ratio, then pad to exact size.
    Letterbox {
        height: u32,
        width: u32,
        value: f32,
    },
    // Helper for plugins to request materialization explicitly
    Materialize,
    /// Extract the shape of the buffer as a vector [height, width, channels].
    /// Returns a Vector domain output with dimension values.
    ExtractShape,
    /// Reorder channels in a [H, W, C] buffer. Allocating (copies data).
    ChannelSwap {
        order: Vec<usize>,
    },
    /// Merge multiple single-channel [H, W] buffers into a [H, W, C] buffer.
    /// Other buffers are referenced by node IDs for graph execution.
    ChannelMerge {
        other_node_ids: Vec<String>,
    },
    /// Reduce an image/array buffer over contour regions.
    /// Contours are provided via an expression column key resolved per-row.
    LabelReduce {
        contours_expr: String,
        reduction: String,
        region_mode: String,
    },
    /// Color space conversion (RGB ↔ HSV, LAB, YCbCr, BGR, Gray).
    Color(ColorConvertOp),
    /// Generic 2D convolution with arbitrary kernel.
    Filter(ConvolveOp),
}

/// Padding mode for Pad operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PadMode {
    /// Fill with constant value.
    Constant,
    /// Replicate edge values.
    Edge,
    /// Reflect without edge.
    Reflect,
    /// Reflect with edge.
    Symmetric,
}

/// Position for PadToSize operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PadPosition {
    /// Center content in padded area.
    Center,
    /// Place at top-left corner.
    TopLeft,
    /// Place at bottom-right corner.
    BottomRight,
}

impl ViewDto {
    /// Get the input domain this operation expects.
    ///
    /// Returns the domain that the predecessor node must output
    /// for this operation to be valid.
    pub fn input_domain(&self) -> Domain {
        match self {
            // View/Compute/Image/PerceptualHash operations work on buffers
            ViewDto::View(_)
            | ViewDto::Compute(_)
            | ViewDto::Image(_)
            | ViewDto::PerceptualHash(_) => Domain::Buffer,
            // Geometry operations have their own domain logic
            ViewDto::Geometry(op) => op.input_domain(),
            // Binary operations work on buffers
            ViewDto::Binary { .. } | ViewDto::ApplyMask { .. } => Domain::Buffer,
            // Reduction operations work on buffers
            ViewDto::Reduction(_) => Domain::Buffer,
            // Histogram operations work on buffers
            ViewDto::Histogram(_) => Domain::Buffer,
            // Deferred resize operations work on buffers
            ViewDto::ResizeScale { .. }
            | ViewDto::ResizeToHeight { .. }
            | ViewDto::ResizeToWidth { .. }
            | ViewDto::ResizeMax { .. }
            | ViewDto::ResizeMin { .. } => Domain::Buffer,
            // Padding operations work on buffers
            ViewDto::Pad { .. } | ViewDto::PadToSize { .. } | ViewDto::Letterbox { .. } => {
                Domain::Buffer
            }
            // Materialize accepts any domain
            ViewDto::Materialize => Domain::Any,
            // ExtractShape works on buffers
            ViewDto::ExtractShape => Domain::Buffer,
            ViewDto::LabelReduce { .. } => Domain::Buffer,
            // Channel operations work on buffers
            ViewDto::ChannelSwap { .. } | ViewDto::ChannelMerge { .. } => Domain::Buffer,
            // Color conversion works on buffers
            ViewDto::Color(_) => Domain::Buffer,
            // Convolution works on buffers
            ViewDto::Filter(_) => Domain::Buffer,
        }
    }

    /// Get the output domain this operation produces.
    ///
    /// Returns the domain that the successor node will receive.
    pub fn output_domain(&self) -> Domain {
        use crate::ops::histogram::HistogramOutput;

        match self {
            // View/Compute/Image operations produce buffers
            ViewDto::View(_) | ViewDto::Compute(_) | ViewDto::Image(_) => Domain::Buffer,
            // PerceptualHash produces a buffer (1D u8 array of hash bytes)
            ViewDto::PerceptualHash(_) => Domain::Buffer,
            // Geometry operations have their own domain logic
            ViewDto::Geometry(op) => op.output_domain(),
            // Binary operations produce buffers
            ViewDto::Binary { .. } | ViewDto::ApplyMask { .. } => Domain::Buffer,
            // Reduction operations: global reduction → Scalar, axis reduction → Buffer
            ViewDto::Reduction(op) => {
                // Global reductions (axis=None) produce a scalar
                // Axis reductions produce a buffer with reduced shape
                match op {
                    ReductionOp::Sum { axis: None }
                    | ReductionOp::Mean { axis: None }
                    | ReductionOp::Max { axis: None }
                    | ReductionOp::Min { axis: None }
                    | ReductionOp::Std { axis: None, .. }
                    | ReductionOp::PopCount => Domain::Scalar,
                    _ => Domain::Buffer, // Axis reductions produce buffers
                }
            }
            // Histogram: Quantized mode produces buffer, other modes produce vector
            ViewDto::Histogram(op) => match op.output {
                HistogramOutput::Quantized => Domain::Buffer,
                _ => Domain::Vector,
            },
            // Deferred resize operations produce buffers
            ViewDto::ResizeScale { .. }
            | ViewDto::ResizeToHeight { .. }
            | ViewDto::ResizeToWidth { .. }
            | ViewDto::ResizeMax { .. }
            | ViewDto::ResizeMin { .. } => Domain::Buffer,
            // Padding operations produce buffers
            ViewDto::Pad { .. } | ViewDto::PadToSize { .. } | ViewDto::Letterbox { .. } => {
                Domain::Buffer
            }
            // Materialize preserves domain
            ViewDto::Materialize => Domain::Any,
            // ExtractShape produces a vector of dimension values
            ViewDto::ExtractShape => Domain::Vector,
            ViewDto::LabelReduce { .. } => Domain::Vector,
            // Channel operations produce buffers
            ViewDto::ChannelSwap { .. } | ViewDto::ChannelMerge { .. } => Domain::Buffer,
            // Color conversion produces buffers
            ViewDto::Color(_) => Domain::Buffer,
            // Convolution produces buffers
            ViewDto::Filter(_) => Domain::Buffer,
        }
    }

    /// Get the rule that determines this operation's output dtype.
    ///
    /// This is the single authority for "what element dtype does this op
    /// produce". Operations backed by an [`Op`] implementation delegate to it;
    /// graph-level variants (deferred resize, padding, channel ops, …) declare
    /// their rule here. Plan-time schema inference and the execution-time dtype
    /// guard both consult this, so the planned and produced dtype cannot diverge.
    pub fn output_dtype_rule(&self) -> crate::core::dtype::OutputDTypeRule {
        use crate::core::dtype::OutputDTypeRule;
        match self {
            ViewDto::View(op) => op.output_dtype_rule(),
            ViewDto::Compute(op) => op.output_dtype_rule(),
            ViewDto::Image(op) => op.output_dtype_rule(),
            ViewDto::Geometry(op) => op.output_dtype_rule(),
            ViewDto::PerceptualHash(op) => op.output_dtype_rule(),
            ViewDto::Binary { op, .. } => op.output_dtype_rule(),
            ViewDto::Reduction(op) => op.output_dtype_rule(),
            ViewDto::Histogram(op) => op.output_dtype_rule(),
            ViewDto::Filter(op) => op.output_dtype_rule(),
            // Color conversion preserves element dtype (routes through f32 internally).
            ViewDto::Color(_) => OutputDTypeRule::PreserveInput,
            // Mask application preserves the buffer dtype.
            ViewDto::ApplyMask { .. } => OutputDTypeRule::PreserveInput,
            // Deferred resize variants only change dimensions, not dtype.
            ViewDto::ResizeScale { .. }
            | ViewDto::ResizeToHeight { .. }
            | ViewDto::ResizeToWidth { .. }
            | ViewDto::ResizeMax { .. }
            | ViewDto::ResizeMin { .. } => OutputDTypeRule::PreserveInput,
            // Padding preserves dtype.
            ViewDto::Pad { .. } | ViewDto::PadToSize { .. } | ViewDto::Letterbox { .. } => {
                OutputDTypeRule::PreserveInput
            }
            // Materialize is a no-op for dtype.
            ViewDto::Materialize => OutputDTypeRule::PreserveInput,
            // ExtractShape yields dimension values as f64.
            ViewDto::ExtractShape => OutputDTypeRule::ForceF64,
            // Channel reorder/merge preserve element dtype.
            ViewDto::ChannelSwap { .. } | ViewDto::ChannelMerge { .. } => {
                OutputDTypeRule::PreserveInput
            }
            // LabelReduce yields f64 region measures.
            ViewDto::LabelReduce { .. } => OutputDTypeRule::ForceF64,
        }
    }

    /// Get the rule that determines how this operation transforms the input
    /// *rank* (number of dimensions).
    ///
    /// The structural, plan-time-inspectable counterpart to `infer_shape`,
    /// alongside [`output_dtype_rule`](ViewDto::output_dtype_rule). Op-backed
    /// variants delegate to the [`Op`] implementation; graph-level variants
    /// declare their rule here.
    pub fn output_rank_rule(&self) -> OutputRankRule {
        match self {
            ViewDto::View(op) => op.output_rank_rule(),
            ViewDto::Compute(op) => op.output_rank_rule(),
            ViewDto::Image(op) => op.output_rank_rule(),
            ViewDto::Geometry(op) => op.output_rank_rule(),
            ViewDto::PerceptualHash(op) => op.output_rank_rule(),
            ViewDto::Binary { op, .. } => op.output_rank_rule(),
            ViewDto::Reduction(op) => op.output_rank_rule(),
            ViewDto::Histogram(op) => op.output_rank_rule(),
            ViewDto::Filter(op) => op.output_rank_rule(),
            ViewDto::Color(op) => op.output_rank_rule(),
            // Mask application and materialize are shape-identity.
            ViewDto::ApplyMask { .. } | ViewDto::Materialize => OutputRankRule::PreserveRank,
            // Deferred resize/padding only change H/W, never the rank.
            ViewDto::ResizeScale { .. }
            | ViewDto::ResizeToHeight { .. }
            | ViewDto::ResizeToWidth { .. }
            | ViewDto::ResizeMax { .. }
            | ViewDto::ResizeMin { .. }
            | ViewDto::Pad { .. }
            | ViewDto::PadToSize { .. }
            | ViewDto::Letterbox { .. } => OutputRankRule::PreserveRank,
            // Channel reorder keeps rank; merge always yields an [H, W, C] image.
            ViewDto::ChannelSwap { .. } => OutputRankRule::PreserveRank,
            ViewDto::ChannelMerge { .. } => OutputRankRule::Fixed(3),
            // ExtractShape and LabelReduce emit 1-D vectors.
            ViewDto::ExtractShape | ViewDto::LabelReduce { .. } => OutputRankRule::Fixed(1),
        }
    }

    /// Get the rule that determines how this operation transforms the input
    /// *channel count* (the trailing dimension of an `[H, W, C]` buffer).
    ///
    /// The single authority for plan-time channel inference, replacing the
    /// former Python-side alpha/channel contract.
    pub fn output_channel_rule(&self) -> OutputChannelRule {
        match self {
            ViewDto::View(op) => op.output_channel_rule(),
            ViewDto::Compute(op) => op.output_channel_rule(),
            ViewDto::Image(op) => op.output_channel_rule(),
            ViewDto::Geometry(op) => op.output_channel_rule(),
            ViewDto::PerceptualHash(op) => op.output_channel_rule(),
            ViewDto::Binary { op, .. } => op.output_channel_rule(),
            ViewDto::Reduction(op) => op.output_channel_rule(),
            ViewDto::Histogram(op) => op.output_channel_rule(),
            ViewDto::Filter(op) => op.output_channel_rule(),
            ViewDto::Color(op) => op.output_channel_rule(),
            // Mask application and materialize preserve channels.
            ViewDto::ApplyMask { .. } | ViewDto::Materialize => {
                OutputChannelRule::PreserveChannels
            }
            // Deferred resize/padding preserve the channel count.
            ViewDto::ResizeScale { .. }
            | ViewDto::ResizeToHeight { .. }
            | ViewDto::ResizeToWidth { .. }
            | ViewDto::ResizeMax { .. }
            | ViewDto::ResizeMin { .. }
            | ViewDto::Pad { .. }
            | ViewDto::PadToSize { .. }
            | ViewDto::Letterbox { .. } => OutputChannelRule::PreserveChannels,
            // Channel reorder preserves the count; merge produces one channel
            // per merged single-channel input (this buffer + the others).
            ViewDto::ChannelSwap { .. } => OutputChannelRule::PreserveChannels,
            ViewDto::ChannelMerge { other_node_ids } => {
                OutputChannelRule::Fixed(other_node_ids.len() + 1)
            }
            // Dimension vectors have no channel concept.
            ViewDto::ExtractShape | ViewDto::LabelReduce { .. } => OutputChannelRule::NotApplicable,
        }
    }

    /// Get the name of this operation for error messages.
    pub fn name(&self) -> &'static str {
        match self {
            ViewDto::View(op) => op.name(),
            ViewDto::Compute(op) => op.name(),
            ViewDto::Image(op) => op.name(),
            ViewDto::Geometry(op) => op.name(),
            ViewDto::PerceptualHash(op) => op.name(),
            ViewDto::Binary { op, .. } => op.name(),
            ViewDto::ApplyMask { .. } => "ApplyMask",
            ViewDto::Reduction(op) => op.name(),
            ViewDto::Histogram(op) => op.name(),
            ViewDto::ResizeScale { .. } => "ResizeScale",
            ViewDto::ResizeToHeight { .. } => "ResizeToHeight",
            ViewDto::ResizeToWidth { .. } => "ResizeToWidth",
            ViewDto::ResizeMax { .. } => "ResizeMax",
            ViewDto::ResizeMin { .. } => "ResizeMin",
            ViewDto::Pad { .. } => "Pad",
            ViewDto::PadToSize { .. } => "PadToSize",
            ViewDto::Letterbox { .. } => "Letterbox",
            ViewDto::Materialize => "Materialize",
            ViewDto::ExtractShape => "ExtractShape",
            ViewDto::LabelReduce { .. } => "LabelReduce",
            ViewDto::ChannelSwap { .. } => "ChannelSwap",
            ViewDto::ChannelMerge { .. } => "ChannelMerge",
            ViewDto::Color(_) => "ColorConvert",
            ViewDto::Filter(_) => "Convolve2D",
        }
    }

    /// Validate that this operation can receive input from the given domain.
    ///
    /// Returns an error with a helpful message if the domains are incompatible.
    pub fn validate_input_domain(&self, input_domain: Domain) -> Result<(), String> {
        let expected = self.input_domain();
        if expected.accepts(input_domain) {
            Ok(())
        } else {
            Err(format!(
                "{}() expects {} input but pipeline is currently in {} domain. \
                 Add a domain-converting operation (e.g., rasterize() or extract_contours()).",
                self.name(),
                expected.name(),
                input_domain.name()
            ))
        }
    }
}
