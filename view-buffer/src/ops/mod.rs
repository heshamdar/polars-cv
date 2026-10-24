//! Operations for the view-buffer framework.
//!
//! This module contains all operation types:
//! - View operations (zero-copy transformations)
//! - Compute operations (data processing)
//! - Image operations (image processing)
//! - Geometry operations (contour, polygon, rasterization)
//! - Binary operations (operations between two arrays)
//! - Reduction operations (statistical aggregations)
//! - Histogram operations (binning and quantization)
//! - Perceptual hash operations (image fingerprinting)
//!
//! # Domains
//!
//! The [`Domain`] enum names the kind of data an op consumes or produces
//! (buffer, contour, scalar, vector); the ops whose output domain depends on
//! their parameters (histogram, reduction, …) report it. The per-node value
//! that carries one of them belongs to the plugin's graph.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub mod affine;
pub mod binary;
pub mod color;
pub mod compute;
pub mod dto;
pub(crate) mod elementwise;
pub mod filter;
pub mod histogram;
pub mod image;
pub mod lut;
pub mod mask;
pub mod pad;
pub mod phash;
pub mod reduction;
pub mod scalar;
pub mod shape_rule;
pub mod spatial_rule;
pub mod traits;
pub(crate) mod util;
pub mod validation;
pub mod view;

pub use binary::BinaryOp;
pub use color::{ColorConvertOp, ColorSpace};
pub use compute::{ComputeOp, NormalizeMethod};
pub use dto::ViewDto;
pub use filter::{BorderMode, ConvolveOp};
pub use histogram::{HistogramOp, HistogramOutput};
pub use image::{FilterType, Footprint, ImageOp, ImageOpKind};
pub use lut::LutOp;
pub use mask::{apply_mask, validate_mask};
pub use pad::{PadMode, PadPosition};
pub use phash::{HashAlgorithm, PerceptualHashOp};
pub use reduction::{
    masked_select, normalize_percentile_masked, reduce_masked, validate_masked_reduce, Masked,
    Percent, Reduction, ReductionOp, Statistic,
};
pub use scalar::{FusedKernel, ScalarOp};
pub use shape_rule::{Dim, OpShape, PlannedInput, Sym};
pub use spatial_rule::{GeometricEffect, NeighborhoodSupport, SpatialDependency};
pub use traits::{IdentityRule, InputLayout, MemoryEffect, Op};
pub use validation::ValidationError;
pub use view::ViewOp;

// Re-export geometry types for convenience
pub use crate::geometry::ops::GeometryOp;

// ============================================================
// Domains
// ============================================================

/// Data domain for type-checking pipeline operations.
///
/// Operations declare their input and output domains, enabling
/// compile-time validation of pipeline structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Domain {
    /// Image or array data (ViewBuffer).
    Buffer,
    /// Geometry data (list of contours/points).
    Contour,
    /// Single scalar value (f64).
    Scalar,
    /// Multiple scalar values (bbox, centroid, etc.).
    Vector,
}

crate::naming::named_variants!(Domain: "The kind of data a pipeline carries at one point: ``buffer`` (an image or array), ``contour`` (geometry), ``scalar`` or ``vector``." {
    "buffer" => Buffer,
    "contour" => Contour,
    "scalar" => Scalar,
    "vector" => Vector,
});

impl Domain {
    /// Get a human-readable name for the domain.
    pub fn name(&self) -> &'static str {
        Self::NAMED
            .iter()
            .find_map(|(n, d)| (d == self).then_some(*n))
            .expect("NAMED covers every Domain variant")
    }
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
