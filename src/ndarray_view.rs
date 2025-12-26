use crate::buffer::{BufferError, ViewBuffer};
use crate::dtype::ViewType;
use crate::layout::ExternalLayout;
use crate::views::{validate_layout, ExternalView};
use ndarray::{ArrayD, ArrayView, ArrayViewD, ShapeBuilder};
use std::marker::PhantomData;

// --- Adapter Implementation (The Source of Logic) ---

/// Adapter for zero-copy ndarray views.
pub struct NdArrayViewAdapter<T>(PhantomData<T>);

impl<'a, T: ViewType> ExternalView<'a> for NdArrayViewAdapter<T> {
    type View = ArrayViewD<'a, T>;
    const LAYOUT: ExternalLayout = ExternalLayout::NdArray;

    fn try_view(buf: &'a ViewBuffer) -> Result<Self::View, BufferError> {
        // 1. Validate layout compatibility
        validate_layout(buf, Self::LAYOUT)?;

        // 2. Type Check
        if buf.dtype() != T::DTYPE {
            return Err(BufferError::TypeMismatch {
                expected: T::DTYPE,
                got: buf.dtype(),
            });
        }

        // 3. Logic: Construct strides and shape for ndarray
        // This logic was previously hidden in ViewBuffer::as_array_view
        let shape = buf.layout.shape.clone();
        let elem_size = std::mem::size_of::<T>() as isize;

        let strides: Vec<usize> = buf
            .layout
            .strides
            .iter()
            .map(|&s| {
                // Ensure stride matches element alignment
                if s % elem_size != 0 {
                    // In a robust system, return generic error or handle this.
                    // For now, this invariant should be held by ViewBuffer construction.
                    panic!(
                        "Misaligned stride for type {:?}: stride {} not divisible by {}",
                        T::DTYPE,
                        s,
                        elem_size
                    );
                }
                (s / elem_size) as usize
            })
            .collect();

        // 4. Construct View
        unsafe {
            let ptr = buf.as_ptr::<T>();
            // ndarray handles the raw pointer + strides
            Ok(ArrayView::from_shape_ptr(shape.strides(strides), ptr))
        }
    }
}

// --- Convenience Trait (Thin Wrapper) ---

pub trait AsNdarray {
    fn as_array_view<T: ViewType>(&self) -> Result<ArrayViewD<T>, BufferError>;
}

impl AsNdarray for ViewBuffer {
    fn as_array_view<T: ViewType>(&self) -> Result<ArrayViewD<T>, BufferError> {
        // Delegate to the Adapter, ensuring consistent behavior
        NdArrayViewAdapter::try_view(self)
    }
}

// --- Ownership Transfer (FromNdarray) ---

pub trait FromNdarray {
    fn from_array<T: ViewType>(array: ArrayD<T>) -> ViewBuffer;
}

impl FromNdarray for ViewBuffer {
    fn from_array<T: ViewType>(array: ArrayD<T>) -> ViewBuffer {
        // Ensure standard layout before taking ownership
        let array = if array.is_standard_layout() {
            array
        } else {
            array.as_standard_layout().into_owned()
        };

        let shape = array.shape().to_vec();
        let vec = array.into_raw_vec();

        ViewBuffer::from_vec(vec).reshape(shape)
    }
}
