#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
    // Helper for plugins to request materialization explicitly
    Materialize,
}
