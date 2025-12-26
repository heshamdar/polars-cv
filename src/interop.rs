use crate::buffer::{TensorBuffer, BufferError};
use crate::dtype::TensorType;
use crate::layout::ExternalLayout;
use ndarray::{ArrayD, ArrayView, ArrayViewD, ShapeBuilder};
use std::marker::PhantomData;

/// Unified trait for external view adapters.
/// Enforces compatibility and zero-copy semantics.
pub trait ExternalView<'a>: Sized {
    type View;

    /// Which layout this backend requires.
    const LAYOUT: ExternalLayout;

    /// Attempt zero-copy view construction.
    fn try_view(buf: &'a TensorBuffer) -> Result<Self::View, BufferError>;
}

/// Helper to validate layout against crate requirements.
pub fn validate_layout(
    buf: &TensorBuffer,
    target: ExternalLayout,
) -> Result<(), BufferError> {
    if buf.is_compatible_with(target) {
        Ok(())
    } else {
        Err(BufferError::IncompatibleLayout { target })
    }
}

// --- ndarray Adapter ---

/// Adapter for zero-copy ndarray views.
pub struct NdArrayViewAdapter<T>(PhantomData<T>);

impl<'a, T: TensorType> ExternalView<'a> for NdArrayViewAdapter<T> {
    type View = ArrayViewD<'a, T>;
    const LAYOUT: ExternalLayout = ExternalLayout::NdArray;

    fn try_view(buf: &'a TensorBuffer) -> Result<Self::View, BufferError> {
        // 1. Validate layout compatibility (ndarray is very flexible)
        validate_layout(buf, Self::LAYOUT)?;

        // 2. Delegate to existing AsNdarray logic (which handles stride math)
        buf.as_array_view::<T>()
    }
}

// --- Legacy AsNdarray (Refactored to use new adapter style internally if desired) ---
// Kept for backward compatibility with earlier stages
pub trait AsNdarray {
    fn as_array_view<T: TensorType>(&self) -> Result<ArrayViewD<T>, BufferError>;
}

impl AsNdarray for TensorBuffer {
    fn as_array_view<T: TensorType>(&self) -> Result<ArrayViewD<T>, BufferError> {
        if self.dtype() != T::DTYPE {
            return Err(BufferError::TypeMismatch { 
                expected: T::DTYPE, 
                got: self.dtype() 
            });
        }

        let shape = self.layout.shape.clone();
        let elem_size = std::mem::size_of::<T>() as isize;
        let strides: Vec<usize> = self.layout.strides.iter()
            .map(|&s| {
                if s % elem_size != 0 {
                    panic!("Misaligned stride for type"); 
                }
                (s / elem_size) as usize
            })
            .collect();

        unsafe {
            let ptr = self.as_ptr::<T>();
            Ok(ArrayView::from_shape_ptr(shape.strides(strides), ptr))
        }
    }
}

pub trait FromNdarray {
    fn from_array<T: TensorType>(array: ArrayD<T>) -> TensorBuffer;
}

impl FromNdarray for TensorBuffer {
    fn from_array<T: TensorType>(array: ArrayD<T>) -> TensorBuffer {
        let array = if array.is_standard_layout() {
            array
        } else {
            array.as_standard_layout().into_owned()
        };
        
        let shape = array.shape().to_vec();
        let vec = array.into_raw_vec();
        
        TensorBuffer::from_vec(vec).reshape(shape)
    }
}