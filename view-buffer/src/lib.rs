//! view-buffer: A zero-copy, stride-aware tensor orchestration framework for Rust.
//!
//! This crate provides a unified interface for working with multi-dimensional
//! arrays (tensors) with zero-copy view operations and efficient compute operations.
//!
//! # Modules
//!
//! - [`core`] - Fundamental types (DType, Layout, ViewBuffer)
//! - [`ops`] - Operations (View, Compute, Image)
//! - [`geometry`] - Geometry operations (Contour, Point, rasterization)
//! - [`expr`] - Expression graph for lazy evaluation
//! - [`execution`] - Execution planning and running
//! - [`protocol`] - Binary serialization format
//! - [`interop`] - External library integrations (optional)
//! - [`naming`] - Canonical name tables for user-facing enums

pub mod core;
pub mod execution;
pub mod expr;
pub mod geometry;
pub mod interop;
pub mod naming;
pub mod ops;
pub mod protocol;

// Re-exports - Core types
pub use core::buffer::{SlicePolicy, ViewBuffer, SIMD_ALIGNMENT};
pub use core::dtype::{DType, OutputDTypeRule, PlannedDType};
pub use core::layout::{ExternalLayout, LayoutFacts};

// Re-exports - Execution
pub use execution::{
    apply_channel_merge, apply_channel_swap, execute_plan, ExecutionPlan, PlanStep,
};

// Re-exports - Expression
pub use expr::ViewExpr;

// Re-exports - Ops
pub use ops::{
    apply_mask, BinaryOp, ColorConvertOp, ColorSpace, ComputeOp, FilterType, ImageOp, ImageOpKind,
    NormalizeMethod, Op, OutputChannelRule, OutputRankRule, ValidationError, ViewDto, ViewOp,
};

// Re-exports - Protocol
pub use protocol::{dtype_to_u8, u8_to_dtype, ViewHeader};

// Re-exports - Interop
pub use interop::{validate_layout, ExternalView};

// Re-exports - Affine
pub use ops::affine::{AffineParams, InterpolationType};

// Re-exports - Geometry
pub use geometry::{BoundingBox, Contour, GeometryOp, Point, Winding};

#[cfg(feature = "image_interop")]
pub use interop::image::{AsImageView, ImageAdapter, ImageCodec, ImageView, ImageViewAdapter};

#[cfg(feature = "ndarray_interop")]
pub use interop::ndarray::{AsNdarray, FromNdarray, NdArrayViewAdapter};

#[cfg(feature = "arrow_interop")]
pub use interop::arrow::{FromArrow, ToArrow};
