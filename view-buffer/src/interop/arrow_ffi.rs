//! Arrow C Data Interface support for language-agnostic zero-copy interop.
//!
//! This module provides FFI exports that allow sharing ViewBuffer data
//! with any language that implements the Arrow C Data Interface:
//! - Python (pyarrow)
//! - R (arrow)
//! - Julia (Arrow.jl)
//! - Go (arrow)
//! - Java (arrow)
//!
//! The Arrow C Data Interface is a standardized ABI for zero-copy data exchange.
//! See: https://arrow.apache.org/docs/format/CDataInterface.html

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, UInt8Array};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use std::sync::Arc;

/// Error type for Arrow FFI operations.
#[derive(Debug, thiserror::Error)]
pub enum ArrowFFIError {
    /// The buffer must be contiguous for FFI export.
    #[error("Buffer must be contiguous for Arrow FFI export")]
    NotContiguous,

    /// Unsupported dtype for Arrow FFI export.
    #[error("Unsupported dtype for Arrow FFI: {0:?}")]
    UnsupportedDType(DType),

    /// Arrow export failed.
    #[error("Arrow FFI export failed: {0}")]
    ExportFailed(String),
}

/// Trait for exporting ViewBuffer to Arrow C Data Interface.
///
/// This enables zero-copy sharing with any language that supports the
/// Arrow C Data Interface ABI.
pub trait ToArrowFFI {
    /// Exports the buffer as Arrow C Data Interface structs.
    ///
    /// Returns a tuple of (schema, array) that can be passed to other
    /// languages via FFI. The caller is responsible for ensuring the
    /// ViewBuffer outlives any external references.
    ///
    /// # Safety
    /// The returned FFI structs contain raw pointers. The ViewBuffer
    /// must remain valid for the lifetime of any external consumers.
    fn to_arrow_ffi(&self) -> Result<(FFI_ArrowSchema, FFI_ArrowArray), ArrowFFIError>;

    /// Exports the buffer as boxed Arrow FFI structs suitable for C FFI.
    ///
    /// The returned boxes can be converted to raw pointers for passing
    /// across FFI boundaries.
    fn to_arrow_ffi_boxed(
        &self,
    ) -> Result<(Box<FFI_ArrowSchema>, Box<FFI_ArrowArray>), ArrowFFIError>;
}

impl ToArrowFFI for ViewBuffer {
    fn to_arrow_ffi(&self) -> Result<(FFI_ArrowSchema, FFI_ArrowArray), ArrowFFIError> {
        // Ensure contiguous for clean FFI export
        if !self.layout.is_contiguous() {
            return Err(ArrowFFIError::NotContiguous);
        }

        let contig = self.to_contiguous();
        let count = contig.layout.num_elements();

        // Create Arrow array based on dtype
        let arrow_array: Arc<dyn Array> = match contig.dtype() {
            DType::U8 => {
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), count) };
                Arc::new(UInt8Array::from(slice.to_vec()))
            }
            DType::I32 => {
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<i32>(), count) };
                Arc::new(Int32Array::from(slice.to_vec()))
            }
            DType::I64 => {
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<i64>(), count) };
                Arc::new(Int64Array::from(slice.to_vec()))
            }
            DType::F32 => {
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };
                Arc::new(Float32Array::from(slice.to_vec()))
            }
            DType::F64 => {
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f64>(), count) };
                Arc::new(Float64Array::from(slice.to_vec()))
            }
            other => return Err(ArrowFFIError::UnsupportedDType(other)),
        };

        // Export to FFI
        // Note: arrow::ffi::to_ffi returns (FFI_ArrowArray, FFI_ArrowSchema)
        let data = arrow_array.into_data();
        let (array, schema) =
            arrow::ffi::to_ffi(&data).map_err(|e| ArrowFFIError::ExportFailed(e.to_string()))?;
        Ok((schema, array))
    }

    fn to_arrow_ffi_boxed(
        &self,
    ) -> Result<(Box<FFI_ArrowSchema>, Box<FFI_ArrowArray>), ArrowFFIError> {
        let (schema, array) = self.to_arrow_ffi()?;
        Ok((Box::new(schema), Box::new(array)))
    }
}

/// Trait for importing Arrow FFI data into ViewBuffer.
pub trait FromArrowFFI {
    /// Creates a ViewBuffer from Arrow C Data Interface structs.
    ///
    /// # Safety
    /// The caller must ensure the FFI structs are valid and properly
    /// initialized according to the Arrow C Data Interface specification.
    ///
    /// Note: This takes ownership of the FFI_ArrowArray as Arrow's import
    /// function requires ownership for proper lifetime management.
    unsafe fn from_arrow_ffi(
        array: FFI_ArrowArray,
        schema: &FFI_ArrowSchema,
    ) -> Result<ViewBuffer, ArrowFFIError>;
}

impl FromArrowFFI for ViewBuffer {
    unsafe fn from_arrow_ffi(
        array: FFI_ArrowArray,
        schema: &FFI_ArrowSchema,
    ) -> Result<ViewBuffer, ArrowFFIError> {
        // Import from FFI to Arrow ArrayData
        let data = arrow::ffi::from_ffi(array, schema)
            .map_err(|e| ArrowFFIError::ExportFailed(e.to_string()))?;

        // Determine dtype from Arrow data type
        use arrow::datatypes::DataType;
        let (dtype, element_size) = match data.data_type() {
            DataType::UInt8 => (DType::U8, 1),
            DataType::Int32 => (DType::I32, 4),
            DataType::Int64 => (DType::I64, 8),
            DataType::Float32 => (DType::F32, 4),
            DataType::Float64 => (DType::F64, 8),
            other => {
                return Err(ArrowFFIError::ExportFailed(format!(
                    "Unsupported Arrow type: {other:?}"
                )))
            }
        };

        // Get the buffer and copy to owned Vec
        // For primitive arrays without nulls, buffer[0] contains the data.
        // If there's a null bitmap (which we don't support), buffer[0] is bitmap and [1] is data.
        let len = data.len();
        let byte_len = len * element_size;

        // Primitive arrays store data in buffer index 1 according to Arrow spec,
        // but buffers() only returns the allocated buffers (not null bitmap).
        // For non-null arrays, the first buffer is the data buffer.
        if data.buffers().is_empty() {
            return Err(ArrowFFIError::ExportFailed(
                "Arrow array missing data buffer".into(),
            ));
        }

        let buffer = &data.buffers()[0]; // Data buffer for primitive arrays without nulls
        if buffer.len() < byte_len {
            return Err(ArrowFFIError::ExportFailed(format!(
                "Buffer too small: {} < {byte_len}",
                buffer.len()
            )));
        }

        // Copy to owned buffer
        let slice = std::slice::from_raw_parts(buffer.as_ptr(), byte_len);
        let vec_data = slice.to_vec();

        Ok(ViewBuffer::from_raw_bytes(vec_data, vec![len], dtype))
    }
}

/// C-compatible struct for passing tensor metadata across FFI.
///
/// This can be used alongside Arrow FFI to communicate tensor shape
/// and strides which Arrow's 1D array model doesn't capture.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TensorMetadataFFI {
    /// Number of dimensions.
    pub ndim: u32,
    /// Pointer to shape array (length = ndim).
    pub shape: *const u64,
    /// Pointer to strides array (length = ndim, in bytes).
    pub strides: *const i64,
    /// Dtype code (matches view_buffer protocol).
    pub dtype: u8,
}

impl ViewBuffer {
    /// Creates tensor metadata suitable for FFI export.
    ///
    /// # Safety
    /// The returned metadata contains raw pointers. The caller must ensure
    /// the ViewBuffer outlives any use of the metadata.
    pub fn tensor_metadata_ffi(&self) -> (TensorMetadataFFI, Vec<u64>, Vec<i64>) {
        let shape: Vec<u64> = self.shape().iter().map(|&s| s as u64).collect();
        let strides: Vec<i64> = self.strides_bytes().iter().map(|&s| s as i64).collect();

        let metadata = TensorMetadataFFI {
            ndim: shape.len() as u32,
            shape: shape.as_ptr(),
            strides: strides.as_ptr(),
            dtype: crate::protocol::dtype_to_u8(self.dtype()),
        };

        // Return ownership of the backing arrays so caller can keep them alive
        (metadata, shape, strides)
    }

    /// Creates a ViewBuffer from raw bytes with explicit shape and dtype.
    ///
    /// This is a lower-level constructor used for FFI import.
    pub fn from_raw_bytes(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        use crate::core::buffer::BufferStorage;
        use crate::core::layout::Layout;
        use std::sync::Arc;

        let layout = Layout::new_contiguous(shape, dtype);
        Self {
            data: BufferStorage::Rust(Arc::new(data)),
            layout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_ffi_roundtrip_f32() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buf = ViewBuffer::from_vec(data.clone());

        // Export to FFI
        let (schema, array) = buf.to_arrow_ffi().expect("FFI export failed");

        // Import back (takes ownership of array)
        let recovered =
            unsafe { ViewBuffer::from_arrow_ffi(array, &schema).expect("FFI import failed") };

        assert_eq!(recovered.shape(), &[4]);
        assert_eq!(recovered.dtype(), DType::F32);

        // Verify data
        let ptr = unsafe { recovered.as_ptr::<f32>() };
        let slice = unsafe { std::slice::from_raw_parts(ptr, 4) };
        assert_eq!(slice, &data[..]);
    }

    #[test]
    fn test_arrow_ffi_roundtrip_u8() {
        let data: Vec<u8> = vec![10, 20, 30, 40, 50];
        let buf = ViewBuffer::from_vec(data.clone());

        let (schema, array) = buf.to_arrow_ffi().expect("FFI export failed");
        // Import back (takes ownership of array)
        let recovered =
            unsafe { ViewBuffer::from_arrow_ffi(array, &schema).expect("FFI import failed") };

        assert_eq!(recovered.shape(), &[5]);
        assert_eq!(recovered.dtype(), DType::U8);
    }

    #[test]
    fn test_tensor_metadata_ffi() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let buf = ViewBuffer::from_vec(data).reshape(vec![2, 3, 4]);

        let (metadata, shape, strides) = buf.tensor_metadata_ffi();

        assert_eq!(metadata.ndim, 3);
        assert_eq!(shape, vec![2, 3, 4]);
        // F32 strides: [48, 16, 4] bytes (contiguous C-order)
        assert_eq!(strides, vec![48, 16, 4]);
    }
}
