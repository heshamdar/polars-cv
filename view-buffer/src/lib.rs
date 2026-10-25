//! view-buffer: the kernel library the polars-cv plugin executes with.
//!
//! A strided, zero-copy N-D buffer (`ViewBuffer`) and the ops that run on it —
//! the generic layout and element-wise ops, and the image and CV kernels:
//! codecs, filters, resize, contour extraction, rasterization, histograms and
//! perceptual hashing. It is not a general tensor framework; nothing outside
//! this repository consumes it (`publish = false`).
//!
//! # Modules
//!
//! - [`core`] - Fundamental types (DType, Layout, ViewBuffer)
//! - [`ops`] - Operations (View, Compute, Image)
//! - [`geometry`] - Geometry operations (Contour, Point, rasterization)
//! - [`expr`] - Expression graph for lazy evaluation
//! - [`execution`] - Execution planning and running
//! - `kernels` - The kernel bodies the ops call, one module per kind of computation
//! - [`protocol`] - Binary serialization format
//! - [`interop`] - External library integrations (optional)
//! - [`naming`] - Canonical name tables for user-facing enums

// Lets `#[derive(Ops)]` output name `::view_buffer::…` here as in polars-cv.
extern crate self as view_buffer;

pub mod core;
pub mod execution;
pub mod expr;
pub mod geometry;
pub mod interop;
pub(crate) mod kernels;
pub mod mode;
pub mod naming;
pub mod ops;
pub mod protocol;

// Re-exports - Core types
pub use core::buffer::{SlicePolicy, ViewBuffer, SIMD_ALIGNMENT};
pub use core::dtype::{DType, OutputDTypeRule, PlannedDType};
pub use core::layout::{ExternalLayout, LayoutFacts};

// Re-exports - Execution
pub use execution::{execute_plan, ExecutionPlan, PlanStep};
pub use kernels::channel::{apply_channel_merge, apply_channel_swap};

// Re-exports - Expression
pub use expr::{OptConfig, ViewExpr, ENGINE_PASSES};

// Re-exports - Ops
pub use ops::validation::validate_channel_merge;
pub use ops::{
    apply_mask, validate_mask, BinaryOp, ColorConvertOp, ColorSpace, ComputeOp, FilterType,
    Footprint, GeometricEffect, IdentityRule, ImageOp, ImageOpKind, NeighborhoodSupport,
    NormalizeMethod, Op, ScalarOp, SpatialDependency, ValidationError, ViewDto, ViewOp,
};

// Re-exports - Protocol
pub use protocol::{dtype_to_u8, parse_blob, u8_to_dtype, BlobLayout, ViewHeader};

// Re-exports - Interop
pub use interop::{validate_layout, ExternalView};

// Re-exports - Affine
pub use ops::affine::{AffineParams, InterpolationType};

// Re-exports - Geometry
pub use geometry::{BoundingBox, Contour, GeometryOp, Point, Winding};

pub use interop::image::{AsImageView, ImageAdapter, ImageCodec, ImageView, ImageViewAdapter};

#[cfg(feature = "ndarray_interop")]
pub use interop::ndarray::{AsNdarray, FromNdarray, NdArrayViewAdapter};

#[cfg(feature = "arrow_interop")]
pub use interop::arrow::{FromArrow, ToArrow};
