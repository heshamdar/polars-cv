pub mod affine;
pub mod core;
pub mod image;
pub mod scalar; // NEW

pub use core::{ViewOp, ComputeOp, Op, MemoryEffect};
pub use image::{ImageOp, ImageOpKind, FilterType};
pub use scalar::{ScalarOp, FusedKernel}; // NEW