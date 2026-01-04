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
//! - I/O operations (sources and sinks)
//!
//! # Typed Pipeline Nodes
//!
//! The [`NodeOutput`] enum represents outputs from pipeline nodes across different
//! data domains. The [`Domain`] enum categorizes these domains for type-checking
//! during pipeline construction and execution.

use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::geometry::Contour;
use crate::ViewBuffer;

pub mod affine;
pub mod binary;
pub mod compute;
pub mod cost;
pub mod dto;
pub mod histogram;
pub mod image;
pub mod io;
pub mod phash;
pub mod reduction;
pub mod scalar;
pub mod traits;
pub mod validation;
pub mod view;

pub use binary::BinaryOp;
pub use compute::{ComputeOp, NormalizeMethod};
pub use cost::{OpCost, OpCostReport};
pub use dto::ViewDto;
pub use histogram::{HistogramOp, HistogramOutput};
pub use image::{FilterType, ImageOp, ImageOpKind};
pub use io::{PlaceholderMeta, SinkFormat, SourceFormat};
pub use phash::{HashAlgorithm, PerceptualHashOp};
pub use reduction::ReductionOp;
pub use scalar::{FusedKernel, ScalarOp};
pub use traits::{DomainOp, MemoryEffect, Op};
pub use validation::ValidationError;
pub use view::ViewOp;

// Re-export geometry types for convenience
pub use crate::geometry::ops::GeometryOp;

// ============================================================
// Typed Pipeline Node System
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
    /// Accepts any domain (e.g., identity, materialize).
    Any,
}

impl Domain {
    /// Check if this domain can accept input from another domain.
    ///
    /// `Any` domain accepts all inputs.
    pub fn accepts(&self, input: Domain) -> bool {
        *self == Domain::Any || *self == input
    }

    /// Get a human-readable name for the domain.
    pub fn name(&self) -> &'static str {
        match self {
            Domain::Buffer => "buffer",
            Domain::Contour => "contour",
            Domain::Scalar => "scalar",
            Domain::Vector => "vector",
            Domain::Any => "any",
        }
    }
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Output from a pipeline node - supports multiple data domains.
///
/// Uses `Arc` for cheap cloning and efficient graph caching
/// without copying data.
#[derive(Debug, Clone)]
pub enum NodeOutput {
    /// Array/image data (reference-counted for cheap cloning).
    Buffer(Arc<ViewBuffer>),
    /// Extracted contours (geometry domain).
    Contours(Arc<Vec<Contour>>),
    /// Single scalar value.
    Scalar(f64),
    /// Multiple scalar values (bbox, centroid, etc.).
    Vector(Arc<Vec<f64>>),
}

impl NodeOutput {
    /// Get the domain of this output.
    pub fn domain(&self) -> Domain {
        match self {
            NodeOutput::Buffer(_) => Domain::Buffer,
            NodeOutput::Contours(_) => Domain::Contour,
            NodeOutput::Scalar(_) => Domain::Scalar,
            NodeOutput::Vector(_) => Domain::Vector,
        }
    }

    /// Try to extract a buffer from this output.
    pub fn as_buffer(&self) -> Option<&Arc<ViewBuffer>> {
        match self {
            NodeOutput::Buffer(buf) => Some(buf),
            _ => None,
        }
    }

    /// Try to extract contours from this output.
    pub fn as_contours(&self) -> Option<&Arc<Vec<Contour>>> {
        match self {
            NodeOutput::Contours(contours) => Some(contours),
            _ => None,
        }
    }

    /// Try to extract a scalar from this output.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            NodeOutput::Scalar(val) => Some(*val),
            _ => None,
        }
    }

    /// Try to extract a vector from this output.
    pub fn as_vector(&self) -> Option<&Arc<Vec<f64>>> {
        match self {
            NodeOutput::Vector(vec) => Some(vec),
            _ => None,
        }
    }

    /// Create a Buffer output from a ViewBuffer (takes ownership).
    pub fn from_buffer(buf: ViewBuffer) -> Self {
        NodeOutput::Buffer(Arc::new(buf))
    }

    /// Create a Buffer output from an Arc<ViewBuffer>.
    pub fn from_arc_buffer(buf: Arc<ViewBuffer>) -> Self {
        NodeOutput::Buffer(buf)
    }

    /// Create a Contours output from a Vec of contours.
    pub fn from_contours(contours: Vec<Contour>) -> Self {
        NodeOutput::Contours(Arc::new(contours))
    }

    /// Create a Contours output from an Arc.
    pub fn from_arc_contours(contours: Arc<Vec<Contour>>) -> Self {
        NodeOutput::Contours(contours)
    }

    /// Create a Scalar output.
    pub fn from_scalar(val: f64) -> Self {
        NodeOutput::Scalar(val)
    }

    /// Create a Vector output.
    pub fn from_vector(vals: Vec<f64>) -> Self {
        NodeOutput::Vector(Arc::new(vals))
    }
}
