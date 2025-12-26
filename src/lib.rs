pub mod buffer;
pub mod dtype;
pub mod expr;
pub mod layout;
pub mod ops;
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
pub use layout::{ExternalLayout, LayoutFacts, LayoutReport};
pub use ops::affine::AffineParams;
pub use planner::{ExecutionPlan, PlanStep};

#[cfg(feature = "image_interop")]
pub use image_view::{AsImageView, ImageView, ImageViewAdapter};

#[cfg(feature = "ndarray_interop")]
pub use interop::{validate_layout, ExternalView, NdArrayViewAdapter};
