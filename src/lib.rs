pub mod dtype;
pub mod layout;
pub mod buffer;
pub mod ops;
pub mod expr;
pub mod planner;

// Modules guarded by features
#[cfg(feature = "ndarray_interop")]
pub mod interop;

#[cfg(feature = "arrow_interop")]
pub mod interop_arrow;

#[cfg(feature = "image_interop")]
pub mod io_image;

#[cfg(feature = "image_interop")]
pub mod image_view;

// Re-exports
pub use buffer::TensorBuffer;
pub use dtype::DType;
pub use expr::TensorExpr;
pub use ops::affine::AffineParams;
pub use planner::{ExecutionPlan, PlanStep};
pub use layout::{ExternalLayout, LayoutReport, LayoutFacts};

#[cfg(feature = "image_interop")]
pub use image_view::{ImageView, AsImageView, ImageViewAdapter};

#[cfg(feature = "ndarray_interop")]
pub use interop::{ExternalView, NdArrayViewAdapter, validate_layout};