//! Polars interoperability for zero-copy data ingestion.
//!
//! This module provides types and functions for zero-copy data transfer
//! from Polars columns to ViewBuffer, bypassing the Arrow C Data Interface
//! for better performance and simpler lifetime management.
//!
//! # Overview
//!
//! Polars stores data in chunked arrays backed by `polars_arrow::buffer::Buffer<T>`,
//! which is an `Arc`-backed buffer that can be cheaply cloned. This module provides:
//!
//! - [`PolarsBufferRef`]: A reference to a slice within a Polars buffer
//! - [`dtype_from_polars`]: Automatic dtype detection from Polars types
//! - [`shape_from_nested_type`]: Shape inference from nested List/Array types
//! - Contiguity checking for List/Array columns
//!
//! # Example
//!
//! ```ignore
//! use view_buffer::interop::polars::{PolarsBufferRef, dtype_from_polars};
//! use polars::prelude::*;
//!
//! // Get a binary column
//! let binary_ca: &BinaryChunked = series.binary()?;
//!
//! // Extract zero-copy reference to row data
//! if let Some(buf_ref) = PolarsBufferRef::from_binary_chunked(binary_ca, row_idx) {
//!     let view_buffer = buf_ref.to_view_buffer(shape, dtype);
//!     // Process view_buffer...
//! }
//! ```

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;

use polars_arrow::buffer::Buffer;

/// Error type for Polars interop operations.
#[derive(Debug, thiserror::Error)]
pub enum PolarsInteropError {
    /// The requested row index is out of bounds.
    #[error("Row index {row_idx} out of bounds (len={len})")]
    IndexOutOfBounds { row_idx: usize, len: usize },

    /// The row contains a null value.
    #[error("Row {row_idx} is null")]
    NullValue { row_idx: usize },

    /// Data is not contiguous (e.g., jagged nested lists).
    #[error("Data is not contiguous: {reason}")]
    NotContiguous { reason: String },

    /// Unsupported Polars data type.
    #[error("Unsupported Polars dtype: {dtype}")]
    UnsupportedDtype { dtype: String },

    /// Shape inference failed.
    #[error("Shape inference failed: {reason}")]
    ShapeInferenceFailed { reason: String },

    /// Buffer access failed.
    #[error("Buffer access failed: {reason}")]
    BufferAccessFailed { reason: String },
}

/// A zero-copy reference to a slice within a Polars buffer.
///
/// This struct holds an `Arc`-backed reference to a Polars buffer along with
/// the offset and length of the view. Cloning is cheap (Arc clone).
///
/// # Lifetime
///
/// The underlying buffer is kept alive by the `Arc` reference. The `ViewBuffer`
/// created from this reference will also keep the buffer alive.
#[derive(Debug, Clone)]
pub struct PolarsBufferRef {
    /// The underlying polars-arrow buffer.
    pub(crate) buffer: Buffer<u8>,
    /// Byte offset into the buffer where this view starts.
    pub(crate) offset: usize,
    /// Length of this view in bytes.
    pub(crate) len: usize,
}

impl PolarsBufferRef {
    /// Creates a new buffer reference.
    ///
    /// # Arguments
    /// * `buffer` - The polars-arrow buffer.
    /// * `offset` - Byte offset into the buffer.
    /// * `len` - Length of the view in bytes.
    ///
    /// # Returns
    /// `None` if offset + len exceeds buffer length.
    pub fn new(buffer: Buffer<u8>, offset: usize, len: usize) -> Option<Self> {
        if offset + len <= buffer.len() {
            Some(Self {
                buffer,
                offset,
                len,
            })
        } else {
            None
        }
    }

    /// Returns a slice view into the buffer data (no copy).
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer.as_slice()[self.offset..self.offset + self.len]
    }

    /// Returns the length of this view in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if this view is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Converts this reference to a ViewBuffer (zero-copy).
    ///
    /// # Arguments
    /// * `shape` - Shape of the resulting tensor.
    /// * `dtype` - Data type of the elements.
    ///
    /// # Panics
    /// Panics if the byte length doesn't match shape * dtype size.
    pub fn to_view_buffer(self, shape: Vec<usize>, dtype: DType) -> ViewBuffer {
        ViewBuffer::from_polars_buffer(self.buffer, self.offset, shape, dtype)
    }

    /// Converts this reference to a ViewBuffer with explicit length validation.
    ///
    /// # Arguments
    /// * `shape` - Shape of the resulting tensor.
    /// * `dtype` - Data type of the elements.
    ///
    /// # Returns
    /// `Err` if the byte length doesn't match the expected size.
    pub fn try_to_view_buffer(
        self,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<ViewBuffer, PolarsInteropError> {
        let expected_bytes: usize = shape.iter().product::<usize>() * dtype.size_of();
        if self.len != expected_bytes {
            return Err(PolarsInteropError::BufferAccessFailed {
                reason: format!(
                    "Buffer length {} doesn't match expected {} (shape={:?}, dtype={:?})",
                    self.len, expected_bytes, shape, dtype
                ),
            });
        }
        Ok(ViewBuffer::from_polars_buffer_slice(
            self.buffer,
            self.offset,
            self.len,
            shape,
            dtype,
        ))
    }
}

/// Convert a Polars DataType to a view-buffer DType.
///
/// This recursively traverses nested List/Array types to find the innermost
/// primitive type.
///
/// # Arguments
/// * `dt` - The Polars data type.
///
/// # Returns
/// `Some(DType)` if the type is supported, `None` otherwise.
///
/// # Supported Types
/// - Primitive numerics: UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Float32, Float64
/// - Nested containers: List, Array (recursively extracts inner type)
pub fn dtype_from_polars(dt: &polars_arrow::datatypes::ArrowDataType) -> Option<DType> {
    use polars_arrow::datatypes::ArrowDataType;

    match dt {
        ArrowDataType::UInt8 => Some(DType::U8),
        ArrowDataType::Int8 => Some(DType::I8),
        ArrowDataType::UInt16 => Some(DType::U16),
        ArrowDataType::Int16 => Some(DType::I16),
        ArrowDataType::UInt32 => Some(DType::U32),
        ArrowDataType::Int32 => Some(DType::I32),
        ArrowDataType::UInt64 => Some(DType::U64),
        ArrowDataType::Int64 => Some(DType::I64),
        ArrowDataType::Float32 => Some(DType::F32),
        ArrowDataType::Float64 => Some(DType::F64),
        // Nested types: recurse to find inner primitive
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) => {
            dtype_from_polars(field.dtype())
        }
        ArrowDataType::FixedSizeList(field, _) => dtype_from_polars(field.dtype()),
        // Binary is treated as u8
        ArrowDataType::Binary | ArrowDataType::LargeBinary => Some(DType::U8),
        _ => None,
    }
}

/// Check if a nested List/Array column has contiguous (rectangular) data.
///
/// A nested list is contiguous if:
/// 1. All inner lists at each level have the same length
/// 2. The innermost values are a primitive type
///
/// # Arguments
/// * `dt` - The Polars data type to check.
///
/// # Returns
/// `true` if the type structure allows for contiguous storage.
pub fn is_type_potentially_contiguous(dt: &polars_arrow::datatypes::ArrowDataType) -> bool {
    use polars_arrow::datatypes::ArrowDataType;

    match dt {
        // Primitives are always contiguous
        ArrowDataType::UInt8
        | ArrowDataType::Int8
        | ArrowDataType::UInt16
        | ArrowDataType::Int16
        | ArrowDataType::UInt32
        | ArrowDataType::Int32
        | ArrowDataType::UInt64
        | ArrowDataType::Int64
        | ArrowDataType::Float32
        | ArrowDataType::Float64 => true,

        // Fixed-size list is contiguous if inner type is
        ArrowDataType::FixedSizeList(field, _) => is_type_potentially_contiguous(field.dtype()),

        // Variable-size list may or may not be contiguous (depends on data)
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) => {
            is_type_potentially_contiguous(field.dtype())
        }

        // Binary is contiguous within a single row
        ArrowDataType::Binary | ArrowDataType::LargeBinary => true,

        _ => false,
    }
}

/// Infer the nesting depth of a Polars type.
///
/// # Examples
/// - `UInt8` -> 0
/// - `List[UInt8]` -> 1
/// - `List[List[UInt8]]` -> 2
/// - `Array[Array[Float32, 3], 4]` -> 2
pub fn nesting_depth(dt: &polars_arrow::datatypes::ArrowDataType) -> usize {
    use polars_arrow::datatypes::ArrowDataType;

    match dt {
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) => {
            1 + nesting_depth(field.dtype())
        }
        ArrowDataType::FixedSizeList(field, _) => 1 + nesting_depth(field.dtype()),
        _ => 0,
    }
}

/// Extract shape from a FixedSizeList/Array type definition.
///
/// For `Array[Array[UInt8, 3], 4]`, returns `[4, 3]`.
///
/// # Arguments
/// * `dt` - The Polars data type.
///
/// # Returns
/// A vector of dimensions from outer to inner. Empty if not a fixed-size nested type.
pub fn fixed_shape_from_type(dt: &polars_arrow::datatypes::ArrowDataType) -> Vec<usize> {
    use polars_arrow::datatypes::ArrowDataType;

    let mut shape = Vec::new();
    let mut current = dt;

    while let ArrowDataType::FixedSizeList(field, size) = current {
        shape.push(*size);
        current = field.dtype();
    }

    shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polars_buffer_ref_creation() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let buffer = Buffer::from(data);

        // Valid reference
        let buf_ref = PolarsBufferRef::new(buffer.clone(), 2, 4);
        assert!(buf_ref.is_some());
        let buf_ref = buf_ref.unwrap();
        assert_eq!(buf_ref.as_slice(), &[3, 4, 5, 6]);
        assert_eq!(buf_ref.len(), 4);

        // Out of bounds
        let invalid = PolarsBufferRef::new(buffer.clone(), 6, 4);
        assert!(invalid.is_none());

        // Empty is valid
        let empty = PolarsBufferRef::new(buffer, 8, 0);
        assert!(empty.is_some());
        assert!(empty.unwrap().is_empty());
    }

    #[test]
    fn test_dtype_from_polars() {
        use polars_arrow::datatypes::ArrowDataType;

        assert_eq!(dtype_from_polars(&ArrowDataType::UInt8), Some(DType::U8));
        assert_eq!(dtype_from_polars(&ArrowDataType::Float32), Some(DType::F32));
        assert_eq!(dtype_from_polars(&ArrowDataType::Int64), Some(DType::I64));
        assert_eq!(dtype_from_polars(&ArrowDataType::Binary), Some(DType::U8));
    }

    #[test]
    fn test_nesting_depth() {
        use polars_arrow::datatypes::{ArrowDataType, Field};

        let u8_type = ArrowDataType::UInt8;
        assert_eq!(nesting_depth(&u8_type), 0);

        let list_u8 = ArrowDataType::List(Box::new(Field::new(
            "item".into(),
            ArrowDataType::UInt8,
            false,
        )));
        assert_eq!(nesting_depth(&list_u8), 1);

        let list_list_u8 =
            ArrowDataType::List(Box::new(Field::new("item".into(), list_u8.clone(), false)));
        assert_eq!(nesting_depth(&list_list_u8), 2);
    }

    #[test]
    fn test_fixed_shape_from_type() {
        use polars_arrow::datatypes::{ArrowDataType, Field};

        // Array[UInt8, 3] -> [3]
        let arr_u8_3 = ArrowDataType::FixedSizeList(
            Box::new(Field::new("item".into(), ArrowDataType::UInt8, false)),
            3,
        );
        assert_eq!(fixed_shape_from_type(&arr_u8_3), vec![3]);

        // Array[Array[UInt8, 3], 4] -> [4, 3]
        let arr_arr = ArrowDataType::FixedSizeList(
            Box::new(Field::new("item".into(), arr_u8_3.clone(), false)),
            4,
        );
        assert_eq!(fixed_shape_from_type(&arr_arr), vec![4, 3]);

        // Non-fixed type -> []
        let empty: Vec<usize> = vec![];
        assert_eq!(fixed_shape_from_type(&ArrowDataType::UInt8), empty);
    }
}
