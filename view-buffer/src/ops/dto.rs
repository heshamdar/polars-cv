#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::geometry::ops::GeometryOp;
use crate::ops::binary::BinaryOp;
use crate::ops::compute::ComputeOp;
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
    // Helper for plugins to request materialization explicitly
    Materialize,
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
            // Materialize accepts any domain
            ViewDto::Materialize => Domain::Any,
        }
    }

    /// Get the output domain this operation produces.
    ///
    /// Returns the domain that the successor node will receive.
    pub fn output_domain(&self) -> Domain {
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
            // Materialize preserves domain
            ViewDto::Materialize => Domain::Any,
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
            ViewDto::Materialize => "Materialize",
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
