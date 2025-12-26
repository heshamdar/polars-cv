pub mod dtype;
pub mod layout;
pub mod buffer;
pub mod ops;
pub mod expr;
pub mod planner;
pub mod views;
pub mod protocol;
pub mod engine; // NEW

#[cfg(feature = "ndarray_interop")]
pub mod ndarray_view;

#[cfg(feature = "arrow_interop")]
pub mod interop_arrow;

#[cfg(feature = "image_interop")]
pub mod io_image;

#[cfg(feature = "image_interop")]
pub mod image_view;

// Re-exports
pub use buffer::ViewBuffer;
pub use dtype::DType;
pub use expr::ViewExpr;
pub use ops::affine::AffineParams;
pub use planner::{ExecutionPlan, PlanStep};
pub use layout::{ExternalLayout, LayoutReport, LayoutFacts};
pub use views::{ExternalView, validate_layout};
pub use protocol::{ViewHeader, dtype_to_u8, u8_to_dtype};
pub use engine::execute_plan; // NEW
pub use ops::ViewDto; // NEW: Export DTO for easier access

#[cfg(feature = "image_interop")]
pub use image_view::{ImageView, AsImageView, ImageViewAdapter};

#[cfg(feature = "ndarray_interop")]
pub use ndarray_view::{NdArrayViewAdapter, AsNdarray, FromNdarray};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn view_buffer(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}