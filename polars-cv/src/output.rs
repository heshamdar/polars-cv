//! Output encoding for numpy/torch sinks with zero-copy support.
//!
//! This module provides zero-copy output encoding for numpy and torch sink formats.
//! Instead of prepending a header to binary data (which requires copying), we return
//! a Struct with separate fields that enable strided numpy array creation.
//!
//! # Schema
//!
//! The output schema is:
//! ```text
//! Struct {
//!     data: Binary,       // Raw array bytes (may be larger for strided views)
//!     dtype: String,      // Data type name (e.g., "uint8", "float32")
//!     shape: List[UInt64] // Array dimensions
//!     strides: List[Int64] // Byte strides per dimension (enables strided views)
//!     offset: UInt64      // Byte offset into data buffer
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use polars_cv::output::build_numpy_series;
//!
//! let buffers: Vec<Option<ViewBuffer>> = process_rows(...);
//! let series = build_numpy_series("output".into(), buffers)?;
//! // series has dtype Struct { data, dtype, shape, strides, offset }
//! ```

use polars::prelude::*;
use polars_arrow::array::{BinaryViewArrayGeneric, View};
use polars_arrow::bitmap::MutableBitmap;
use view_buffer::{DType as VbDType, ViewBuffer};

/// Get the Polars DataType for numpy/torch sink output.
///
/// Returns a Struct schema with:
/// - `data`: Binary (raw array bytes, may be larger than needed for strided views)
/// - `dtype`: String (dtype name like "uint8", "float32")
/// - `shape`: List[UInt64] (array dimensions)
/// - `strides`: List[Int64] (byte strides per dimension, enables strided views)
/// - `offset`: UInt64 (byte offset into data buffer)
pub fn numpy_output_dtype() -> DataType {
    DataType::Struct(vec![
        Field::new(PlSmallStr::from_static("data"), DataType::Binary),
        Field::new(PlSmallStr::from_static("dtype"), DataType::String),
        Field::new(
            PlSmallStr::from_static("shape"),
            DataType::List(Box::new(DataType::UInt64)),
        ),
        Field::new(
            PlSmallStr::from_static("strides"),
            DataType::List(Box::new(DataType::Int64)),
        ),
        Field::new(PlSmallStr::from_static("offset"), DataType::UInt64),
    ])
}

/// Convert a view-buffer DType to a string name.
///
/// These names match numpy dtype names for easy conversion on the Python side.
pub fn dtype_to_string(dtype: VbDType) -> &'static str {
    match dtype {
        VbDType::U8 => "uint8",
        VbDType::I8 => "int8",
        VbDType::U16 => "uint16",
        VbDType::I16 => "int16",
        VbDType::U32 => "uint32",
        VbDType::I32 => "int32",
        VbDType::U64 => "uint64",
        VbDType::I64 => "int64",
        VbDType::F32 => "float32",
        VbDType::F64 => "float64",
    }
}

/// Encoded numpy output for a single row.
///
/// This struct holds the components that will become the Struct column fields.
/// Supports strided views for zero-copy output of non-contiguous buffers.
#[derive(Debug, Clone)]
pub struct NumpyRowOutput {
    /// Raw array data as bytes (may be larger than needed for strided views).
    pub data: polars_arrow::buffer::Buffer<u8>,
    /// Data type name (e.g., "uint8", "float32").
    pub dtype: &'static str,
    /// Array shape dimensions.
    pub shape: Vec<u64>,
    /// Byte strides per dimension (enables strided numpy views).
    pub strides: Vec<i64>,
    /// Byte offset into data buffer.
    pub offset: u64,
    /// Whether this was zero-copy (for testing/debugging).
    #[allow(dead_code)]
    pub was_zero_copy: bool,
}

impl NumpyRowOutput {
    /// Create a NumpyRowOutput from a ViewBuffer.
    ///
    /// Uses zero-copy ownership transfer when possible, including for
    /// non-contiguous strided buffers by preserving stride information.
    pub fn from_buffer(buffer: ViewBuffer) -> Self {
        let was_zero_copy = buffer.can_zero_copy_strided();
        let (data, shape, strides, offset, dtype) = buffer.into_polars_buffer_strided();

        Self {
            data,
            dtype: dtype_to_string(dtype),
            shape: shape.into_iter().map(|d| d as u64).collect(),
            strides: strides.into_iter().map(|s| s as i64).collect(),
            offset: offset as u64,
            was_zero_copy,
        }
    }
}

/// Build a numpy output Series from multiple rows.
///
/// This function takes a collection of optional ViewBuffers (one per row) and
/// builds a StructChunked Series with the numpy output schema.
///
/// # Arguments
/// * `name` - Name for the output Series
/// * `rows` - Vector of optional ViewBuffers, one per row (None for null rows)
///
/// # Returns
/// A Series with dtype Struct{data: Binary, dtype: String, shape: List[UInt64], strides: List[Int64], offset: UInt64}
pub fn build_numpy_series(name: PlSmallStr, rows: Vec<Option<ViewBuffer>>) -> PolarsResult<Series> {
    let len = rows.len();

    // Convert each row to NumpyRowOutput
    let encoded: Vec<Option<NumpyRowOutput>> = rows
        .into_iter()
        .map(|opt| opt.map(NumpyRowOutput::from_buffer))
        .collect();

    // Build all five columns
    let data_col = build_data_column(&encoded)?;
    let dtype_col = build_dtype_column(&encoded);
    let shape_col = build_shape_column(&encoded)?;
    let strides_col = build_strides_column(&encoded)?;
    let offset_col = build_offset_column(&encoded);

    // Combine into struct
    StructChunked::from_series(
        name,
        len,
        [data_col, dtype_col, shape_col, strides_col, offset_col].iter(),
    )
    .map(|ca| ca.into_series())
}

/// Build the 'data' column (Binary) from encoded rows using zero-copy buffer registration.
///
/// This implementation uses BinaryViewArray which stores views into external buffers.
/// For data > 12 bytes (all images), the data is stored in registered buffers and
/// views point to those buffers without copying.
///
/// # Zero-Copy Mechanism
///
/// BinaryViewArray uses a two-part structure:
/// 1. Views: 128-bit metadata entries (length, prefix, buffer_idx, offset)
/// 2. Buffers: External Arc-backed memory regions
///
/// By registering our polars_arrow::Buffer<u8> directly, we achieve true zero-copy.
fn build_data_column(rows: &[Option<NumpyRowOutput>]) -> PolarsResult<Series> {
    use polars_arrow::datatypes::ArrowDataType;

    let n_rows = rows.len();
    let mut views: Vec<View> = Vec::with_capacity(n_rows);
    let mut buffers: Vec<polars_arrow::buffer::Buffer<u8>> = Vec::new();
    let mut validity_builder: Option<MutableBitmap> = None;
    let mut total_bytes_len: usize = 0;
    let mut total_buffer_len: usize = 0;

    for (idx, opt) in rows.iter().enumerate() {
        match opt {
            Some(row) => {
                let data_len = row.data.len();
                total_bytes_len += data_len;

                if data_len <= 12 {
                    // Inline small values directly in the view
                    views.push(View::new_inline(row.data.as_slice()));
                } else {
                    // Register buffer and create view pointing to it
                    let buffer_idx = buffers.len() as u32;
                    total_buffer_len += row.data.len();
                    buffers.push(row.data.clone()); // Arc clone, very cheap

                    // Create view with buffer reference
                    views.push(View::new_from_bytes(row.data.as_slice(), buffer_idx, 0));
                }

                // Update validity if we had nulls before
                if let Some(ref mut validity) = validity_builder {
                    validity.push(true);
                }
            }
            None => {
                // Handle null - initialize validity bitmap if first null
                if validity_builder.is_none() {
                    let mut bitmap = MutableBitmap::with_capacity(n_rows);
                    // Set all previous entries as valid
                    for _ in 0..idx {
                        bitmap.push(true);
                    }
                    validity_builder = Some(bitmap);
                }
                validity_builder.as_mut().unwrap().push(false);
                views.push(View::default());
            }
        }
    }

    // Build the BinaryViewArray with registered buffers
    let validity = validity_builder.map(|v| v.into());

    // Safety: We've constructed valid views that reference valid buffer indices
    let array = unsafe {
        BinaryViewArrayGeneric::<[u8]>::new_unchecked(
            ArrowDataType::BinaryView,
            views.into(),
            buffers.into_iter().collect(),
            validity,
            total_bytes_len,
            total_buffer_len,
        )
    };

    let ca = BinaryChunked::with_chunk(PlSmallStr::from_static("data"), array);
    Ok(ca.into_series())
}

/// Build the 'dtype' column (String) from encoded rows.
fn build_dtype_column(rows: &[Option<NumpyRowOutput>]) -> Series {
    let values: Vec<Option<&str>> = rows
        .iter()
        .map(|opt| opt.as_ref().map(|r| r.dtype))
        .collect();

    StringChunked::from_iter_options(PlSmallStr::from_static("dtype"), values.into_iter())
        .into_series()
}

/// Build the 'shape' column (List[UInt64]) from encoded rows.
fn build_shape_column(rows: &[Option<NumpyRowOutput>]) -> PolarsResult<Series> {
    let values: Vec<Option<Series>> = rows
        .iter()
        .map(|opt| {
            opt.as_ref().map(|r| {
                let dims: Vec<u64> = r.shape.clone();
                Series::new(PlSmallStr::from_static(""), dims)
            })
        })
        .collect();

    // Build list column from the series values
    let mut builder = ListPrimitiveChunkedBuilder::<UInt64Type>::new(
        PlSmallStr::from_static("shape"),
        rows.len(),
        8, // Initial capacity per list
        DataType::UInt64,
    );

    for opt_series in values {
        match opt_series {
            Some(s) => {
                let ca = s.u64()?;
                builder.append_slice(ca.cont_slice().unwrap_or(&[]));
            }
            None => {
                builder.append_null();
            }
        }
    }

    Ok(builder.finish().into_series())
}

/// Build the 'strides' column (List[Int64]) from encoded rows.
fn build_strides_column(rows: &[Option<NumpyRowOutput>]) -> PolarsResult<Series> {
    let values: Vec<Option<Series>> = rows
        .iter()
        .map(|opt| {
            opt.as_ref().map(|r| {
                let strides: Vec<i64> = r.strides.clone();
                Series::new(PlSmallStr::from_static(""), strides)
            })
        })
        .collect();

    // Build list column from the series values
    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        PlSmallStr::from_static("strides"),
        rows.len(),
        8, // Initial capacity per list
        DataType::Int64,
    );

    for opt_series in values {
        match opt_series {
            Some(s) => {
                let ca = s.i64()?;
                builder.append_slice(ca.cont_slice().unwrap_or(&[]));
            }
            None => {
                builder.append_null();
            }
        }
    }

    Ok(builder.finish().into_series())
}

/// Build the 'offset' column (UInt64) from encoded rows.
fn build_offset_column(rows: &[Option<NumpyRowOutput>]) -> Series {
    let values: Vec<Option<u64>> = rows
        .iter()
        .map(|opt| opt.as_ref().map(|r| r.offset))
        .collect();

    UInt64Chunked::from_iter_options(PlSmallStr::from_static("offset"), values.into_iter())
        .into_series()
}

/// Check if any rows used zero-copy transfer.
///
/// Useful for testing to verify zero-copy behavior.
#[allow(dead_code)]
pub fn count_zero_copy_rows(rows: &[Option<NumpyRowOutput>]) -> usize {
    rows.iter()
        .filter_map(|opt| opt.as_ref())
        .filter(|r| r.was_zero_copy)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_to_string() {
        assert_eq!(dtype_to_string(VbDType::U8), "uint8");
        assert_eq!(dtype_to_string(VbDType::F32), "float32");
        assert_eq!(dtype_to_string(VbDType::I64), "int64");
    }

    #[test]
    fn test_numpy_output_dtype_schema() {
        let dtype = numpy_output_dtype();
        if let DataType::Struct(fields) = dtype {
            assert_eq!(fields.len(), 5);
            assert_eq!(fields[0].name().as_str(), "data");
            assert_eq!(fields[0].dtype(), &DataType::Binary);
            assert_eq!(fields[1].name().as_str(), "dtype");
            assert_eq!(fields[1].dtype(), &DataType::String);
            assert_eq!(fields[2].name().as_str(), "shape");
            assert_eq!(
                fields[2].dtype(),
                &DataType::List(Box::new(DataType::UInt64))
            );
            assert_eq!(fields[3].name().as_str(), "strides");
            assert_eq!(
                fields[3].dtype(),
                &DataType::List(Box::new(DataType::Int64))
            );
            assert_eq!(fields[4].name().as_str(), "offset");
            assert_eq!(fields[4].dtype(), &DataType::UInt64);
        } else {
            panic!("Expected Struct dtype");
        }
    }

    #[test]
    fn test_numpy_row_output_from_buffer() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let buffer = ViewBuffer::from_vec(data).reshape(vec![2, 3]);

        let output = NumpyRowOutput::from_buffer(buffer);

        assert_eq!(output.dtype, "uint8");
        assert_eq!(output.shape, vec![2, 3]);
        assert_eq!(output.data.len(), 6);
    }

    #[test]
    fn test_build_numpy_series_with_data() {
        let buf1 = ViewBuffer::from_vec(vec![1u8, 2, 3, 4]).reshape(vec![2, 2]);
        let buf2 = ViewBuffer::from_vec(vec![5u8, 6, 7, 8, 9, 10]).reshape(vec![2, 3]);

        let series = build_numpy_series(
            PlSmallStr::from_static("output"),
            vec![Some(buf1), None, Some(buf2)],
        )
        .unwrap();

        assert_eq!(series.len(), 3);
        assert!(matches!(series.dtype(), DataType::Struct(_)));

        // Check null handling
        let struct_ca = series.struct_().unwrap();
        let data_col = struct_ca.field_by_name("data").unwrap();
        assert!(data_col.get(1).unwrap().is_null());
    }
}
