#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use crate::ops::{ViewOp, ComputeOp, ImageOp};

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