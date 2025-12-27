//! Apache Arrow interoperability.

use crate::core::buffer::{BufferError, ViewBuffer};
use crate::core::dtype::DType;
use arrow::array::Float32Array;
use arrow::array::{
    Array, BinaryArray, FixedSizeBinaryArray, FixedSizeListArray, LargeBinaryArray, PrimitiveArray,
};
use arrow::datatypes::{DataType, Float32Type, Float64Type, Int32Type, Int64Type, UInt8Type};
use std::sync::Arc;

/// Trait for creating ViewBuffer from Arrow arrays.
pub trait FromArrow {
    /// Creates a ViewBuffer from an Arrow array.
    fn from_arrow_array(array: &dyn Array) -> Result<ViewBuffer, BufferError>;
}

impl FromArrow for ViewBuffer {
    fn from_arrow_array(array: &dyn Array) -> Result<ViewBuffer, BufferError> {
        // 1. Validation: We cannot zero-copy Arrow arrays with nulls into dense tensors
        if array.null_count() > 0 {
            return Err(BufferError::NotContiguous);
        }

        // 2. Resolve Shape and Underlying Primitive Array
        let mut shape = vec![array.len()];
        let mut current_array = array;

        while let DataType::FixedSizeList(_field, size) = current_array.data_type() {
            shape.push(*size as usize);
            let list_array = current_array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or(BufferError::TypeMismatch {
                    expected: DType::U8,
                    got: DType::U8,
                })?;
            current_array = list_array.values().as_ref();
        }

        // 3. Extract Buffer and DType
        let (buffer, dtype) = match current_array.data_type() {
            DataType::Float32 => {
                let a = current_array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Float32Type>>()
                    .unwrap();
                (a.values().inner().clone(), DType::F32)
            }
            DataType::Float64 => {
                let a = current_array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Float64Type>>()
                    .unwrap();
                (a.values().inner().clone(), DType::F64)
            }
            DataType::Int32 => {
                let a = current_array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int32Type>>()
                    .unwrap();
                (a.values().inner().clone(), DType::I32)
            }
            DataType::Int64 => {
                let a = current_array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int64Type>>()
                    .unwrap();
                (a.values().inner().clone(), DType::I64)
            }
            DataType::UInt8 => {
                let a = current_array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt8Type>>()
                    .unwrap();
                (a.values().inner().clone(), DType::U8)
            }
            DataType::FixedSizeBinary(size) => {
                let a = current_array
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .unwrap();
                shape.push(*size as usize);
                (a.to_data().buffers()[0].clone(), DType::U8)
            }
            DataType::Binary => {
                let a = current_array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .unwrap();
                let buf = a.to_data().buffers()[1].clone();
                shape = vec![buf.len()];
                (buf, DType::U8)
            }
            DataType::LargeBinary => {
                let a = current_array
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .unwrap();
                let buf = a.to_data().buffers()[1].clone();
                shape = vec![buf.len()];
                (buf, DType::U8)
            }
            _ => {
                return Err(BufferError::TypeMismatch {
                    expected: DType::F32,
                    got: DType::U8,
                })
            }
        };

        Ok(ViewBuffer::from_arrow_buffer(buffer, shape, dtype))
    }
}

// --- Output Strategy (ViewBuffer -> Arrow) ---

/// Trait for converting ViewBuffer to Arrow arrays.
pub trait ToArrow {
    /// Export as an Arrow BinaryArray (Opaque flat bytes).
    /// Used for Polars Binary Column integration.
    fn to_arrow_binary(&self) -> BinaryArray;

    /// Export as an Arrow FixedSizeListArray (Nested Lists).
    /// Used for structured data.
    fn to_arrow_list(&self) -> Arc<dyn Array>;
}

impl ToArrow for ViewBuffer {
    fn to_arrow_binary(&self) -> BinaryArray {
        // 1. Ensure Contiguous
        let contig = self.to_contiguous();
        let total_bytes = contig.layout.num_elements() * contig.dtype().size_of();

        // 2. Get Raw Slice
        let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), total_bytes) };

        // 3. Wrap in BinaryArray (1 element containing all bytes)
        // Or do we mean "List of Binaries"? Usually Polars plugin returns one Binary array
        // where each row is a tensor.
        // For this prototype, we'll treat the *entire* tensor as a single binary blob
        // suitable for serialization.
        let offsets =
            arrow::buffer::OffsetBuffer::<i32>::from_lengths(std::iter::once(total_bytes));
        let values = arrow::buffer::Buffer::from_slice_ref(slice);

        unsafe {
            // Safety: Offsets are valid for the data length
            BinaryArray::new_unchecked(offsets, values, None)
        }
    }

    fn to_arrow_list(&self) -> Arc<dyn Array> {
        let contig = self.to_contiguous();

        match contig.dtype() {
            DType::F32 => {
                // Flatten to PrimitiveArray
                let count = contig.layout.num_elements();
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };
                let primitive = Arc::new(Float32Array::from(slice.to_vec()));

                // Recursively wrap in FixedSizeList based on shape
                // If shape is [Batch, H, W], we wrap W, then H.
                // Start from the last dimension.
                let mut current_array: Arc<dyn Array> = primitive;

                // We assume dimension 0 is the "Batch" or "Row" dimension in Arrow,
                // so we wrap dimensions 1..N.
                for &dim_size in contig.shape().iter().skip(1).rev() {
                    let field = Arc::new(arrow::datatypes::Field::new(
                        "item",
                        current_array.data_type().clone(),
                        false,
                    ));
                    current_array = Arc::new(FixedSizeListArray::new(
                        field,
                        dim_size as i32,
                        current_array,
                        None,
                    ));
                }
                current_array
            }
            DType::U8 => {
                let count = contig.layout.num_elements();
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), count) };
                let primitive = Arc::new(arrow::array::UInt8Array::from(slice.to_vec()));

                let mut current_array: Arc<dyn Array> = primitive;
                for &dim_size in contig.shape().iter().skip(1).rev() {
                    let field = Arc::new(arrow::datatypes::Field::new(
                        "item",
                        current_array.data_type().clone(),
                        false,
                    ));
                    current_array = Arc::new(FixedSizeListArray::new(
                        field,
                        dim_size as i32,
                        current_array,
                        None,
                    ));
                }
                current_array
            }
            _ => unimplemented!("Arrow export for this dtype not implemented"),
        }
    }
}
