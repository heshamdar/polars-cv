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
//! - I/O operations (sources and sinks)

pub mod affine;
pub mod binary;
pub mod compute;
pub mod cost;
pub mod dto;
pub mod histogram;
pub mod image;
pub mod io;
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
pub use reduction::ReductionOp;
pub use scalar::{FusedKernel, ScalarOp};
pub use traits::{MemoryEffect, Op};
pub use validation::ValidationError;
pub use view::ViewOp;

// Re-export geometry types for convenience
pub use crate::geometry::ops::GeometryOp;
