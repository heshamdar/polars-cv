pub mod affine;
pub mod core;
pub mod image;
pub mod scalar; // NEW

pub use core::{ComputeOp, MemoryEffect, Op, ViewOp};
pub use image::{FilterType, ImageOp, ImageOpKind};
pub use scalar::{FusedKernel, ScalarOp}; // NEW
