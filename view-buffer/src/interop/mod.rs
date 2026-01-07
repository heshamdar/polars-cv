//! External library interoperability.
//!
//! This module provides adapters for zero-copy integration with external libraries:
//! - ndarray: N-dimensional array support
//! - Arrow: Apache Arrow buffer interop
//! - image: Image processing library support

use crate::core::buffer::{BufferError, ViewBuffer};
use crate::core::layout::ExternalLayout;

/// Unified trait for external view adapters.
/// Enforces compatibility and zero-copy semantics.
pub trait ExternalView<'a>: Sized {
    type View;

    /// Which layout this backend requires.
    const LAYOUT: ExternalLayout;

    /// Attempt zero-copy view construction.
    fn try_view(buf: &'a ViewBuffer) -> Result<Self::View, BufferError>;
}

/// Helper to validate layout against crate requirements.
pub fn validate_layout(buf: &ViewBuffer, target: ExternalLayout) -> Result<(), BufferError> {
    if buf.is_compatible_with(target) {
        Ok(())
    } else {
        Err(BufferError::IncompatibleLayout { target })
    }
}

#[cfg(feature = "ndarray_interop")]
pub mod ndarray;

#[cfg(feature = "arrow_interop")]
pub mod arrow;

#[cfg(feature = "arrow_interop")]
pub mod arrow_ffi;

#[cfg(feature = "image_interop")]
pub mod image;

#[cfg(feature = "polars_interop")]
pub mod polars;
