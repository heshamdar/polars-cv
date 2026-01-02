#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::geometry::ops::GeometryOp;
use crate::ops::binary::BinaryOp;
use crate::ops::compute::ComputeOp;
use crate::ops::image::ImageOp;
use crate::ops::view::ViewOp;

/// A pure Data Transfer Object (DTO) for operation plans.
/// This separates the schema (what to do) from the execution graph (how to do it).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ViewDto {
    View(ViewOp),
    Compute(ComputeOp),
    Image(ImageOp),
    Geometry(GeometryOp),
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
    // Helper for plugins to request materialization explicitly
    Materialize,
}
