use crate::buffer::{ViewBuffer, BufferError};
use crate::layout::ExternalLayout;

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
pub fn validate_layout(
    buf: &ViewBuffer,
    target: ExternalLayout,
) -> Result<(), BufferError> {
    if buf.is_compatible_with(target) {
        Ok(())
    } else {
        Err(BufferError::IncompatibleLayout { target })
    }
}