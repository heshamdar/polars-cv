pub mod dtype;
pub mod layout;
pub mod buffer;
pub mod ops;
pub mod expr;
pub mod planner;
pub mod views;

// Renamed module
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

#[cfg(feature = "image_interop")]
pub use image_view::{ImageView, AsImageView, ImageViewAdapter};

#[cfg(feature = "ndarray_interop")]
pub use ndarray_view::{NdArrayViewAdapter, AsNdarray, FromNdarray};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn view_buffer(_py: Python, m: &PyModule) -> PyResult<()> {
    Ok(())
}