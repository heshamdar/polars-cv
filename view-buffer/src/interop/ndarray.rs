//! ndarray interoperability.
//!
//! This module provides zero-copy conversion between ViewBuffer and ndarray.
//! It properly handles negative strides (from flip operations) by using
//! ndarray's `invert_axis` method.

use crate::core::buffer::{BufferError, ViewBuffer};
use crate::core::dtype::ViewType;
use crate::core::layout::ExternalLayout;
use crate::interop::{validate_layout, ExternalView};
use ndarray::{ArrayD, ArrayView, ArrayViewD, Axis, ShapeBuilder};
use std::marker::PhantomData;

// --- Adapter Implementation (The Source of Logic) ---

/// Adapter for zero-copy ndarray views.
///
/// This adapter properly handles negative strides (from flip operations)
/// by converting them to positive strides and using `invert_axis`.
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
        // Handle negative strides by tracking flipped axes
        let shape = buf.layout.shape.clone();
        let elem_size = std::mem::size_of::<T>() as isize;

        // Track which axes have negative strides (need to be inverted later)
        let mut flipped_axes: Vec<usize> = Vec::new();

        // Convert byte strides to element strides, using absolute values
        let strides: Vec<usize> = buf
            .layout
            .strides
            .iter()
            .enumerate()
            .map(|(axis, &s)| {
                // Ensure stride matches element alignment
                if s % elem_size != 0 {
                    panic!(
                        "Misaligned stride for type {:?}: stride {} not divisible by {}",
                        T::DTYPE,
                        s,
                        elem_size
                    );
                }

                let elem_stride = s / elem_size;

                if elem_stride < 0 {
                    // Track this axis for later inversion
                    flipped_axes.push(axis);
                    // Use absolute value for ndarray (from_shape_ptr requires non-negative strides)
                    (-elem_stride) as usize
                } else {
                    elem_stride as usize
                }
            })
            .collect();

        // 4. Compute adjusted base pointer for negative strides
        // For each negative stride axis, we need to offset the pointer to point
        // to what will become the first element after inversion
        let mut ptr_offset: isize = 0;
        for &axis in &flipped_axes {
            let axis_len = shape[axis] as isize;
            let byte_stride = buf.layout.strides[axis];
            // Move pointer to the "last" element along this axis (which becomes first after invert)
            ptr_offset += (axis_len - 1) * byte_stride;
        }

        // 5. Construct View with positive strides
        let mut view = unsafe {
            let base_ptr = buf.as_ptr::<u8>();
            // Offset the pointer for negative strides
            let adjusted_ptr = base_ptr.offset(ptr_offset) as *const T;
            ArrayView::from_shape_ptr(shape.strides(strides), adjusted_ptr)
        };

        // 6. Apply invert_axis for each flipped axis to restore correct order
        for &axis in &flipped_axes {
            view.invert_axis(Axis(axis));
        }

        Ok(view)
    }
}

// --- Convenience Trait (Thin Wrapper) ---

/// Trait for converting ViewBuffer to ndarray view.
pub trait AsNdarray {
    /// Attempts to create a zero-copy ndarray view.
    fn as_array_view<T: ViewType>(&self) -> Result<ArrayViewD<'_, T>, BufferError>;
}

impl AsNdarray for ViewBuffer {
    fn as_array_view<T: ViewType>(&self) -> Result<ArrayViewD<'_, T>, BufferError> {
        // Delegate to the Adapter, ensuring consistent behavior
        NdArrayViewAdapter::try_view(self)
    }
}

// --- Ownership Transfer (FromNdarray) ---

/// Trait for creating ViewBuffer from ndarray.
pub trait FromNdarray {
    /// Creates a ViewBuffer from an owned ndarray.
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
        // Use the new API that returns offset (should be 0 for standard layout)
        let (vec, _offset) = array.into_raw_vec_and_offset();

        ViewBuffer::from_vec(vec).reshape(shape)
    }
}
