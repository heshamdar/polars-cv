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

pub mod core;
pub mod execution;
pub mod expr;
pub mod geometry;
pub mod interop;
pub mod ops;
pub mod protocol;

// Re-exports - Core types
pub use core::buffer::{SlicePolicy, ViewBuffer};
pub use core::dtype::DType;
pub use core::layout::{ExternalLayout, LayoutFacts, LayoutReport};

// Re-exports - Execution
pub use execution::{execute_plan, ExecutionPlan, PlanStep};

// Re-exports - Tiling
pub use execution::{
    configure_tiling, get_tile_config, is_tiling_enabled, set_tile_config, with_tile_config,
    TileConfig, TilePolicy,
};

// Re-exports - Expression
pub use expr::{PipelineCostReport, ViewExpr};

// Re-exports - Ops
pub use ops::{
    BinaryOp, ComputeOp, FilterType, ImageOp, ImageOpKind, NormalizeMethod, Op, OpCost,
    OpCostReport, PlaceholderMeta, SinkFormat, SourceFormat, ValidationError, ViewDto, ViewOp,
};

// Re-exports - Protocol
pub use protocol::{dtype_to_u8, u8_to_dtype, ViewHeader};

// Re-exports - Interop
pub use interop::{validate_layout, ExternalView};

// Re-exports - Affine
pub use ops::affine::AffineParams;

// Re-exports - Geometry
pub use geometry::{BoundingBox, Contour, GeometryOp, Point, Winding};

#[cfg(feature = "image_interop")]
pub use interop::image::{AsImageView, ImageAdapter, ImageView, ImageViewAdapter};

#[cfg(feature = "ndarray_interop")]
pub use interop::ndarray::{AsNdarray, FromNdarray, NdArrayViewAdapter};

#[cfg(feature = "arrow_interop")]
pub use interop::arrow::{FromArrow, ToArrow};
