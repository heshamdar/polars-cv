#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::geometry::ops::GeometryOp;
use crate::ops::binary::BinaryOp;
use crate::ops::compute::ComputeOp;
use crate::ops::histogram::HistogramOp;
use crate::ops::image::ImageOp;
use crate::ops::phash::PerceptualHashOp;
use crate::ops::reduction::ReductionOp;
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
