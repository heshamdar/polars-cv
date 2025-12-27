//! Operations for the view-buffer framework.
//!
//! This module contains all operation types:
//! - View operations (zero-copy transformations)
//! - Compute operations (data processing)
//! - Image operations (image processing)
//! - I/O operations (sources and sinks)

pub mod affine;
pub mod compute;
pub mod cost;
pub mod dto;
pub mod image;
pub mod io;
pub mod scalar;
pub mod traits;
pub mod validation;
pub mod view;

pub use compute::{ComputeOp, NormalizeMethod};
pub use cost::{OpCost, OpCostReport};
pub use dto::ViewDto;
pub use image::{FilterType, ImageOp, ImageOpKind};
pub use io::{PlaceholderMeta, SinkFormat, SourceFormat};
pub use scalar::{FusedKernel, ScalarOp};
pub use traits::{MemoryEffect, Op};
pub use validation::ValidationError;
pub use view::ViewOp;
