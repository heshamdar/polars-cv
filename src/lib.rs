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

// Re-exports - Core types
pub use buffer::ViewBuffer;
pub use dtype::DType;
pub use engine::execute_plan;
pub use expr::{PipelineCostReport, ViewExpr};
pub use layout::{ExternalLayout, LayoutFacts, LayoutReport};
pub use ops::affine::AffineParams;
pub use planner::{ExecutionPlan, PlanStep};
pub use protocol::{dtype_to_u8, u8_to_dtype, ViewHeader};
pub use views::{validate_layout, ExternalView};

// Re-exports - Ops
pub use ops::{
    ComputeOp, FilterType, ImageOp, ImageOpKind, NormalizeMethod, Op, OpCost, OpCostReport,
    PlaceholderMeta, SinkFormat, SourceFormat, ValidationError, ViewDto, ViewOp,
};

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