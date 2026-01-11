//! Unified pipeline graph execution engine.
//!
//! This module handles the execution of pipeline graphs (DAGs) where multiple
//! pipelines can be composed and executed as a single fused operation.
//!
//! The graph executor:
//! - Parses a JSON graph specification
//! - Executes nodes in topological order
//! - Passes ViewBuffers between nodes without serialization
//! - Supports binary operations between nodes
//! - Returns Binary for single output ("_output") or Struct for multiple outputs
//! - Supports typed nodes with domain transitions (Buffer → Contour → Buffer)
//!
//! # Optimization Boundaries
//!
//! Each node in the graph represents an optimization boundary. Operations
//! within a node may be fused by view-buffer's optimizer (e.g., scalar ops),
//! but operations across different nodes are never fused. This ensures:
//!
//! - Output nodes produce exactly the buffer state at their alias point
//! - Shared subexpressions are computed once and reused
//! - No mutation safety issues since each node produces a new buffer
//!
//! # Typed Node Outputs
//!
//! The graph supports multiple data domains via [`NodeOutput`]:
//! - Buffer (images/arrays)
//! - Contours (geometry)
//! - Scalar (single values)
//! - Vector (multiple values)
//!
//! Domain transitions are validated at execution time, and the "native" sink
//! format dispatches to the appropriate encoding based on the output domain.

use polars::chunked_array::builder::ListPrimitiveChunkedBuilder;
use polars::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use view_buffer::geometry::{extract::extract_contours, rasterize::rasterize, Contour};
use view_buffer::ops::NodeOutput;
use view_buffer::{BinaryOp, GeometryOp, ImageOp, ImageOpKind, Op, ViewBuffer, ViewDto, ViewExpr};

use crate::execute::{
    decode_contour_source, decode_contour_source_with_dims, decode_source, resolve_op,
};
use crate::pipeline::{PipelineSpec, SinkSpec, SourceSpec};

// ============================================================
// Null Handling Utilities
// ============================================================

/// Compute valid (non-null) row indices for a series.
///
/// Uses the validity bitmap for efficient null checking.
/// Returns a vector of indices where the value is not null.
///
/// This is useful for batch processing optimization where you want to
/// skip null rows entirely rather than checking per-row.
#[allow(dead_code)]
fn compute_valid_indices(series: &Series) -> Vec<usize> {
    let null_mask = series.is_null();
    (0..series.len())
        .filter(|&i| !null_mask.get(i).unwrap_or(true))
        .collect()
}

/// Compute null mask as a boolean vector.
///
/// `true` at index i means the value at i is null.
#[allow(dead_code)]
fn compute_null_mask(series: &Series) -> Vec<bool> {
    let null_mask = series.is_null();
    (0..series.len())
        .map(|i| null_mask.get(i).unwrap_or(true))
        .collect()
}

/// Count non-null values in a series.
#[allow(dead_code)]
fn count_non_null(series: &Series) -> usize {
    series.len() - series.null_count()
}

/// Check if a specific row is null in a series.
///
/// This is a convenience wrapper that handles the ChunkedArray result.
fn is_row_null(series: &Series, row_idx: usize) -> bool {
    series.is_null().get(row_idx).unwrap_or(true)
}

// ============================================================
// Zero-Copy Binary Source Helpers
// ============================================================

/// Extract binary data from a BinaryChunked at a specific row.
///
/// Returns the data as a polars-arrow buffer (involves copy for BinaryViewArray).
///
/// Note: Polars uses BinaryViewArray internally which has a different memory layout
/// than the traditional offset-based BinaryArray. For true zero-copy, we would need
/// to handle the view-based representation. Currently, we copy the data to a buffer
/// for simplicity and compatibility.
///
/// # Arguments
/// * `binary_ca` - The binary chunked array.
/// * `row_idx` - The row index to extract.
///
/// # Returns
/// `Some((buffer, offset, len))` if the row is valid and not null.
/// `None` if the row is null.
fn get_binary_row_buffer(
    binary_ca: &BinaryChunked,
    row_idx: usize,
) -> Option<(polars_arrow::buffer::Buffer<u8>, usize, usize)> {
    // Check for null using the helper
    if is_row_null(&binary_ca.clone().into_series(), row_idx) {
        return None;
    }

    // Get the bytes for this row (this may involve a copy for BinaryViewArray)
    let bytes = binary_ca.get(row_idx)?;
    let len = bytes.len();

    // Create a buffer from the bytes
    let buffer = polars_arrow::buffer::Buffer::from(bytes.to_vec());

    // Offset is 0 since we copied just this row's data
    Some((buffer, 0, len))
}

/// Decode a binary source (blob or raw) with zero-copy when possible.
///
/// For blob format: parses the VIEW protocol header, creates ViewBuffer pointing to data.
/// For raw format: creates ViewBuffer directly from the buffer reference.
///
/// # Arguments
/// * `buffer` - The polars-arrow buffer containing the data.
/// * `offset` - Byte offset into the buffer.
/// * `len` - Length of the data in bytes.
/// * `source_format` - "blob" or "raw".
/// * `dtype_str` - Required for "raw", ignored for "blob" (embedded in header).
fn decode_binary_zero_copy(
    buffer: polars_arrow::buffer::Buffer<u8>,
    offset: usize,
    len: usize,
    source_format: &str,
    dtype_str: Option<&str>,
) -> Result<ViewBuffer, String> {
    match source_format {
        "blob" => {
            // Parse VIEW protocol header from the slice
            // Clone buffer for slice access since we'll move it into decode_blob_zero_copy
            let slice_data: Vec<u8> = buffer.as_slice()[offset..offset + len].to_vec();
            decode_blob_zero_copy(buffer, offset, len, &slice_data)
        }
        "raw" => {
            // Raw bytes - need dtype from source spec
            let dtype_s = dtype_str.ok_or("Raw source format requires dtype")?;
            let dtype = parse_dtype_str(dtype_s)?;

            // For raw format, shape is 1D array
            let element_size = dtype.size_of();
            let num_elements = len / element_size;

            Ok(ViewBuffer::from_polars_buffer(
                buffer,
                offset,
                vec![num_elements],
                dtype,
            ))
        }
        other => Err(format!("Unsupported binary source format: {other}")),
    }
}

/// Decode a blob (VIEW protocol) with zero-copy.
///
/// Parses the header from the slice, then creates a ViewBuffer pointing
/// directly into the data portion of the original buffer.
fn decode_blob_zero_copy(
    buffer: polars_arrow::buffer::Buffer<u8>,
    base_offset: usize,
    total_len: usize,
    slice: &[u8],
) -> Result<ViewBuffer, String> {
    use view_buffer::protocol::{u8_to_dtype, HEADER_SIZE, MAGIC_BYTES, VERSION};

    if total_len < HEADER_SIZE {
        return Err("Blob data too short for header".into());
    }

    // Parse header
    let magic = &slice[0..4];
    if magic != MAGIC_BYTES {
        return Err("Invalid blob magic bytes".into());
    }

    let version = u16::from_le_bytes([slice[4], slice[5]]);
    if version != VERSION {
        return Err(format!("Unsupported blob version: {version}"));
    }

    let dtype_code = slice[6];
    let rank = slice[7] as usize;
    let data_offset = u64::from_le_bytes(slice[8..16].try_into().unwrap()) as usize;

    let dtype =
        u8_to_dtype(dtype_code).ok_or_else(|| format!("Unknown dtype code: {dtype_code}"))?;

    // Read shape
    let shape_start = HEADER_SIZE;
    let mut shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let pos = shape_start + i * 8;
        if pos + 8 > total_len {
            return Err("Blob truncated reading shape".into());
        }
        let dim = u64::from_le_bytes(slice[pos..pos + 8].try_into().unwrap()) as usize;
        shape.push(dim);
    }

    // Validate data portion
    let num_elements: usize = shape.iter().product();
    let expected_data_len = num_elements * dtype.size_of();
    if data_offset + expected_data_len > total_len {
        return Err(format!(
            "Blob data truncated: offset={data_offset}, expected={expected_data_len}, total={total_len}"
        ));
    }

    // Create ViewBuffer pointing to data portion (zero-copy)
    Ok(ViewBuffer::from_polars_buffer_slice(
        buffer,
        base_offset + data_offset,
        expected_data_len,
        shape,
        dtype,
    ))
}

// ============================================================
// Type Inference Helpers
// ============================================================

/// Infer view-buffer DType from Polars DataType.
///
/// Recursively traverses nested List/Array types to find the innermost
/// primitive type.
fn dtype_from_polars_datatype(dt: &DataType) -> Option<view_buffer::DType> {
    match dt {
        DataType::UInt8 => Some(view_buffer::DType::U8),
        DataType::Int8 => Some(view_buffer::DType::I8),
        DataType::UInt16 => Some(view_buffer::DType::U16),
        DataType::Int16 => Some(view_buffer::DType::I16),
        DataType::UInt32 => Some(view_buffer::DType::U32),
        DataType::Int32 => Some(view_buffer::DType::I32),
        DataType::UInt64 => Some(view_buffer::DType::U64),
        DataType::Int64 => Some(view_buffer::DType::I64),
        DataType::Float32 => Some(view_buffer::DType::F32),
        DataType::Float64 => Some(view_buffer::DType::F64),
        // Binary is treated as u8
        DataType::Binary => Some(view_buffer::DType::U8),
        // Nested types: recurse to find inner primitive
        DataType::List(inner) => dtype_from_polars_datatype(inner.as_ref()),
        DataType::Array(inner, _) => dtype_from_polars_datatype(inner.as_ref()),
        _ => None,
    }
}

/// Parse dtype string to view-buffer DType.
fn parse_dtype_str(dtype_str: &str) -> Result<view_buffer::DType, String> {
    match dtype_str {
        "u8" => Ok(view_buffer::DType::U8),
        "i8" => Ok(view_buffer::DType::I8),
        "u16" => Ok(view_buffer::DType::U16),
        "i16" => Ok(view_buffer::DType::I16),
        "u32" => Ok(view_buffer::DType::U32),
        "i32" => Ok(view_buffer::DType::I32),
        "u64" => Ok(view_buffer::DType::U64),
        "i64" => Ok(view_buffer::DType::I64),
        "f32" => Ok(view_buffer::DType::F32),
        "f64" => Ok(view_buffer::DType::F64),
        other => Err(format!("Unknown dtype: {other}")),
    }
}

// ============================================================
// List/Array Source Decoding
// ============================================================

/// Decode a Polars List or Array value at a specific row into a ViewBuffer.
///
/// Uses zero-copy when the data is contiguous (FixedSizeList/Array types),
/// falling back to copy-based flattening for jagged List types.
///
/// If `dtype_str` is provided, it will be used. Otherwise, the dtype will be
/// inferred from the Polars column type.
///
/// If `require_contiguous` is true and zero-copy is not possible, an error is returned.
fn decode_list_or_array_source(
    series: &Series,
    row_idx: usize,
    dtype_str: Option<&str>,
    require_contiguous: bool,
) -> Result<Option<ViewBuffer>, String> {
    // Parse or infer the target dtype
    let dtype = if let Some(dtype_s) = dtype_str {
        parse_dtype_str(dtype_s)?
    } else {
        // Auto-infer from Polars type
        dtype_from_polars_datatype(series.dtype()).ok_or_else(|| {
            format!(
                "Cannot infer dtype from Polars type {:?}. Please specify dtype explicitly.",
                series.dtype()
            )
        })?
    };

    // Try zero-copy path for fixed-size nested arrays first
    if let Some(result) = try_decode_array_zero_copy(series, row_idx, dtype)? {
        return Ok(Some(result));
    }

    // If require_contiguous is set and we didn't get zero-copy, error out
    if require_contiguous {
        return Err(format!(
            "Source 'require_contiguous=true' requires rectangular data with zero-copy access, \
            but row {row_idx} has data that cannot be zero-copied (possibly jagged nested lists or \
            variable-size List type). Use require_contiguous=false to allow copy-based flattening, \
            or use Polars Array type (fixed-size) instead of List."
        ));
    }

    // Fall back to copy-based path for variable-size lists
    decode_list_with_copy(series, row_idx, dtype)
}

/// Try zero-copy decoding for fixed-size Array types.
///
/// Returns `Ok(Some(buffer))` if zero-copy succeeded, `Ok(None)` if not applicable.
fn try_decode_array_zero_copy(
    series: &Series,
    row_idx: usize,
    dtype: view_buffer::DType,
) -> Result<Option<ViewBuffer>, String> {
    // Only attempt zero-copy for Array types (FixedSizeList in Arrow)
    // List types have variable sizes and may be jagged
    if let DataType::Array(inner_dtype, _width) = series.dtype() {
        // Extract shape from the type definition
        let shape = extract_fixed_shape_from_dtype(series.dtype());
        if shape.is_empty() {
            return Ok(None); // Not a fixed-size nested type
        }

        // Check if innermost type is a primitive (not nested)
        if !is_primitive_dtype(get_innermost_dtype(inner_dtype)) {
            return Ok(None);
        }

        // Get the ArrayChunked
        let arr_ca = series
            .array()
            .map_err(|e| format!("Array access error: {e}"))?;

        // Check for null using the helper
        if is_row_null(&arr_ca.clone().into_series(), row_idx) {
            return Ok(None);
        }

        // Try to get zero-copy buffer access
        if let Some((buffer, offset, len)) = get_array_row_buffer(arr_ca, row_idx, dtype) {
            let vb = ViewBuffer::from_polars_buffer_slice(buffer, offset, len, shape, dtype);
            return Ok(Some(vb));
        }
    }

    Ok(None)
}

/// Extract shape from a nested Array type definition.
///
/// For `Array[Array[UInt8, 3], 4]`, returns `[4, 3]`.
fn extract_fixed_shape_from_dtype(dt: &DataType) -> Vec<usize> {
    let mut shape = Vec::new();
    let mut current = dt;

    while let DataType::Array(inner, width) = current {
        shape.push(*width);
        current = inner.as_ref();
    }

    shape
}

/// Get the innermost dtype from nested types.
fn get_innermost_dtype(dt: &DataType) -> &DataType {
    match dt {
        DataType::List(inner) | DataType::Array(inner, _) => get_innermost_dtype(inner),
        _ => dt,
    }
}

/// Check if a dtype is a primitive type.
fn is_primitive_dtype(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::UInt8
            | DataType::Int8
            | DataType::UInt16
            | DataType::Int16
            | DataType::UInt32
            | DataType::Int32
            | DataType::UInt64
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Get zero-copy buffer access for an Array row.
///
/// Returns `(buffer, offset, len)` if zero-copy is possible.
fn get_array_row_buffer(
    arr_ca: &ArrayChunked,
    row_idx: usize,
    dtype: view_buffer::DType,
) -> Option<(polars_arrow::buffer::Buffer<u8>, usize, usize)> {
    // Find which chunk contains this row
    let mut cumulative_len = 0;
    for chunk in arr_ca.downcast_iter() {
        let chunk_len = chunk.len();
        if row_idx < cumulative_len + chunk_len {
            let local_idx = row_idx - cumulative_len;

            // For FixedSizeList, the values are stored contiguously
            // We need to navigate to the innermost values
            return get_fixed_size_list_buffer(chunk, local_idx, dtype);
        }
        cumulative_len += chunk_len;
    }

    None
}

/// Get buffer from a FixedSizeListArray chunk.
fn get_fixed_size_list_buffer(
    chunk: &polars_arrow::array::FixedSizeListArray,
    local_idx: usize,
    dtype: view_buffer::DType,
) -> Option<(polars_arrow::buffer::Buffer<u8>, usize, usize)> {
    let size = chunk.size();
    let values = chunk.values();

    // Recursively navigate to primitive values if nested
    let (primitive_values, elements_per_row) = get_primitive_values(values.as_ref(), size)?;

    // Calculate offset and length for this row
    let element_size = dtype.size_of();
    let offset = local_idx * elements_per_row * element_size;
    let len = elements_per_row * element_size;

    // Get the underlying buffer
    let buffer = get_primitive_buffer(primitive_values, dtype)?;

    Some((buffer, offset, len))
}

/// Recursively get primitive values array from nested FixedSizeList.
fn get_primitive_values(
    array: &dyn polars_arrow::array::Array,
    accumulated_size: usize,
) -> Option<(&dyn polars_arrow::array::Array, usize)> {
    use polars_arrow::array::FixedSizeListArray;

    if let Some(fsl) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let size = fsl.size();
        get_primitive_values(fsl.values().as_ref(), accumulated_size * size)
    } else {
        // Reached primitive array
        Some((array, accumulated_size))
    }
}

/// Get the underlying buffer from a primitive array.
fn get_primitive_buffer(
    array: &dyn polars_arrow::array::Array,
    dtype: view_buffer::DType,
) -> Option<polars_arrow::buffer::Buffer<u8>> {
    use polars_arrow::array::PrimitiveArray;

    macro_rules! try_get_buffer {
        ($array:expr, $type:ty) => {
            if let Some(arr) = $array.as_any().downcast_ref::<PrimitiveArray<$type>>() {
                // Get the values buffer and convert to u8 buffer
                let values = arr.values();
                // PrimitiveArray values is a Buffer<T>, we need to reinterpret as Buffer<u8>
                // This is safe because we're just changing the view, not the data
                let bytes = values.as_slice();
                let u8_slice = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const u8,
                        bytes.len() * std::mem::size_of::<$type>(),
                    )
                };
                return Some(polars_arrow::buffer::Buffer::from(u8_slice.to_vec()));
            }
        };
    }

    // Try each primitive type
    match dtype {
        view_buffer::DType::U8 => try_get_buffer!(array, u8),
        view_buffer::DType::I8 => try_get_buffer!(array, i8),
        view_buffer::DType::U16 => try_get_buffer!(array, u16),
        view_buffer::DType::I16 => try_get_buffer!(array, i16),
        view_buffer::DType::U32 => try_get_buffer!(array, u32),
        view_buffer::DType::I32 => try_get_buffer!(array, i32),
        view_buffer::DType::U64 => try_get_buffer!(array, u64),
        view_buffer::DType::I64 => try_get_buffer!(array, i64),
        view_buffer::DType::F32 => try_get_buffer!(array, f32),
        view_buffer::DType::F64 => try_get_buffer!(array, f64),
    }

    None
}

/// Decode list with copy (fallback path).
fn decode_list_with_copy(
    series: &Series,
    row_idx: usize,
    dtype: view_buffer::DType,
) -> Result<Option<ViewBuffer>, String> {
    // Get the element at this row
    let element_series = match series.dtype() {
        DataType::List(_) => {
            let list_ca = series
                .list()
                .map_err(|e| format!("List access error: {e}"))?;
            list_ca.get_as_series(row_idx)
        }
        DataType::Array(_, _) => {
            let arr_ca = series
                .array()
                .map_err(|e| format!("Array access error: {e}"))?;
            arr_ca.get_as_series(row_idx)
        }
        other => {
            return Err(format!("Expected List or Array column, got {other:?}"));
        }
    };

    let element = match element_series {
        Some(s) => s,
        None => return Ok(None), // Null value
    };

    // Recursively extract shape and flatten data
    let (shape, flat_series) = flatten_nested_series(&element)?;

    if flat_series.is_empty() {
        return Ok(None);
    }

    // Convert flat series to bytes
    let bytes = series_to_bytes(&flat_series, &dtype)?;

    Ok(Some(ViewBuffer::from_raw_bytes(bytes, shape, dtype)))
}

/// Recursively flatten a nested Series and extract shape.
///
/// For a nested list like [[1,2,3], [4,5,6], [7,8,9]]:
/// - First level: 3 lists -> shape starts with [3]
/// - Check first element's length: 3 -> shape = [3, 3]
/// - Final flat primitives: [1,2,3,4,5,6,7,8,9]
///
/// Assumes all inner lists have the same length (rectangular array).
fn flatten_nested_series(series: &Series) -> Result<(Vec<usize>, Series), String> {
    // First pass: determine full shape by traversing first elements
    let shape = infer_nested_shape(series)?;

    // Second pass: flatten to 1D by repeatedly exploding
    let mut current = series.clone();
    while matches!(current.dtype(), DataType::List(_) | DataType::Array(_, _)) {
        current = current
            .explode()
            .map_err(|e| format!("Explode error: {e}"))?;
    }

    Ok((shape, current))
}

/// Infer shape by traversing first elements at each nesting level.
///
/// For List(List(List(Int64))) with 2x2x3 data:
/// 1. Series has 2 elements (outer rows) -> shape = [2]
/// 2. First element has 2 sub-lists (columns) -> shape = [2, 2]  
/// 3. First sub-list has 3 primitives (channels) -> shape = [2, 2, 3]
fn infer_nested_shape(series: &Series) -> Result<Vec<usize>, String> {
    let mut shape = Vec::new();
    let mut current = series.clone();

    loop {
        match current.dtype() {
            DataType::List(_) => {
                let list_ca = current.list().map_err(|e| format!("List error: {e}"))?;
                let len = list_ca.len();
                shape.push(len);

                // Get first element to continue traversing
                if len > 0 {
                    if let Some(first) = list_ca.get_as_series(0) {
                        current = first;
                    } else {
                        break; // Null first element
                    }
                } else {
                    break; // Empty list
                }
            }
            DataType::Array(_, _width) => {
                let len = current.len();
                shape.push(len);

                // Get first element
                let arr_ca = current.array().map_err(|e| format!("Array error: {e}"))?;
                if len > 0 {
                    if let Some(first) = arr_ca.get_as_series(0) {
                        current = first;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            _ => {
                // Reached primitive type
                // Always push the length as the final dimension
                // (this captures the innermost list/array element count)
                shape.push(current.len());
                break;
            }
        }
    }

    Ok(shape)
}

/// Convert a flat primitive Series to raw bytes.
fn series_to_bytes(series: &Series, target_dtype: &view_buffer::DType) -> Result<Vec<u8>, String> {
    macro_rules! convert_series {
        ($series:expr, $method:ident, $rust_type:ty) => {{
            let ca = $series.$method().map_err(|e| format!("Cast error: {e}"))?;
            let values: Vec<$rust_type> = ca.into_no_null_iter().collect();
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            Ok(bytes)
        }};
    }

    // First cast to the target dtype if needed
    let casted = match target_dtype {
        view_buffer::DType::U8 => series.cast(&DataType::UInt8),
        view_buffer::DType::I8 => series.cast(&DataType::Int8),
        view_buffer::DType::U16 => series.cast(&DataType::UInt16),
        view_buffer::DType::I16 => series.cast(&DataType::Int16),
        view_buffer::DType::U32 => series.cast(&DataType::UInt32),
        view_buffer::DType::I32 => series.cast(&DataType::Int32),
        view_buffer::DType::U64 => series.cast(&DataType::UInt64),
        view_buffer::DType::I64 => series.cast(&DataType::Int64),
        view_buffer::DType::F32 => series.cast(&DataType::Float32),
        view_buffer::DType::F64 => series.cast(&DataType::Float64),
    }
    .map_err(|e| format!("Cast to {target_dtype:?} failed: {e}"))?;

    match target_dtype {
        view_buffer::DType::U8 => convert_series!(casted, u8, u8),
        view_buffer::DType::I8 => convert_series!(casted, i8, i8),
        view_buffer::DType::U16 => convert_series!(casted, u16, u16),
        view_buffer::DType::I16 => convert_series!(casted, i16, i16),
        view_buffer::DType::U32 => convert_series!(casted, u32, u32),
        view_buffer::DType::I32 => convert_series!(casted, i32, i32),
        view_buffer::DType::U64 => convert_series!(casted, u64, u64),
        view_buffer::DType::I64 => convert_series!(casted, i64, i64),
        view_buffer::DType::F32 => convert_series!(casted, f32, f32),
        view_buffer::DType::F64 => convert_series!(casted, f64, f64),
    }
}

// ============================================================
// Result Types
// ============================================================

/// Result type for individual row execution.
///
/// Each variant holds the typed data for a single row output.
/// The Option allows null handling - None represents null input or error.
#[derive(Clone)]
enum RowResult {
    /// Binary data (images, blobs, etc.)
    Binary(Option<Vec<u8>>),
    /// Scalar value (reduce operations)
    Scalar(Option<f64>),
    /// Vector of f64 values
    Vector(Option<Vec<f64>>),
    /// Contour geometry data
    Contours(Option<Vec<Contour>>),
    /// Typed list for "list" sink (variable length, preserves dtype).
    TypedList(Option<(TypedBufferData, Vec<usize>)>),
    /// Typed fixed-size array for "array" sink (fixed shape, preserves dtype).
    TypedArray(Option<(TypedBufferData, Vec<usize>)>),
    /// Numpy/Torch struct output (zero-copy ViewBuffer ownership transfer).
    NumpyStruct(Option<ViewBuffer>),
}

// ============================================================
// Static Type Inference Helpers
// ============================================================

/// Convert a dtype string to Polars DataType.
///
/// This is used for static type inference at planning time.
/// Note: Requires dtype-i8/dtype-u8/dtype-i16/dtype-u16 features in polars.
pub fn dtype_str_to_polars(dtype: &str) -> DataType {
    match dtype {
        "u8" => DataType::UInt8,
        "i8" => DataType::Int8,
        "u16" => DataType::UInt16,
        "i16" => DataType::Int16,
        "u32" => DataType::UInt32,
        "i32" => DataType::Int32,
        "u64" => DataType::UInt64,
        "i64" => DataType::Int64,
        "f32" => DataType::Float32,
        "f64" => DataType::Float64,
        _ => DataType::Float64, // Default fallback
    }
}

/// Get the Polars DataType for a given output specification.
///
/// Returns the appropriate dtype based on domain, sink format, and expected dtype.
pub fn dtype_for_output(spec: &OutputSpec) -> DataType {
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();

    match (domain, format) {
        // Buffer domain - numpy/torch use struct format for zero-copy
        ("buffer", "numpy" | "torch") => crate::output::numpy_output_dtype(),
        // Other buffer formats remain binary
        ("buffer", "png" | "jpeg" | "blob") => DataType::Binary,
        ("buffer", "list") => DataType::List(Box::new(dtype_str_to_polars(&spec.expected_dtype))),
        ("buffer", "array") => {
            // For array, we need shape info - for now return List
            // The actual array dtype is built at execution time with shape
            DataType::List(Box::new(dtype_str_to_polars(&spec.expected_dtype)))
        }

        // Scalar domain
        ("scalar", "native") => DataType::Float64,

        // Vector domain (perceptual hash, centroid, bbox)
        ("vector", "native" | "list") => {
            DataType::List(Box::new(dtype_str_to_polars(&spec.expected_dtype)))
        }

        // Contour domain
        ("contour", "native") => {
            // Return contour struct schema
            let point_dtype = DataType::Struct(vec![
                Field::new("x".into(), DataType::Float64),
                Field::new("y".into(), DataType::Float64),
            ]);
            DataType::Struct(vec![
                Field::new(
                    "exterior".into(),
                    DataType::List(Box::new(point_dtype.clone())),
                ),
                Field::new("interiors".into(), DataType::Null),
            ])
        }

        // Fallback
        _ => DataType::Binary,
    }
}

/// Create a null RowResult with the correct type based on OutputSpec.
///
/// This ensures that null values are pushed with the appropriate type variant,
/// allowing the series builder to use static type information.
fn null_row_result_for_spec(spec: &OutputSpec) -> RowResult {
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();

    match (domain, format) {
        // Numpy/Torch use struct format
        ("buffer", "numpy" | "torch") => RowResult::NumpyStruct(None),
        // Other binary formats
        ("buffer", "png" | "jpeg" | "blob") | (_, "binary") => RowResult::Binary(None),
        // List format with buffer domain - use typed list
        ("buffer", "list") | ("vector", "native" | "list") => RowResult::TypedList(None),
        // Array format with buffer domain
        ("buffer", "array") => RowResult::TypedArray(None),
        // Scalar domain
        ("scalar", "native") => RowResult::Scalar(None),
        // Contour domain
        ("contour", "native") => RowResult::Contours(None),
        // Fallback
        _ => RowResult::Binary(None),
    }
}

/// Build a series from row results using the OutputSpec to determine the type.
///
/// This function uses static type information from the OutputSpec rather than
/// inspecting the first row's data. This allows proper handling of null values
/// while preserving the expected output type.
fn build_series_from_spec(
    name: PlSmallStr,
    spec: &OutputSpec,
    data: &[RowResult],
) -> PolarsResult<Series> {
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();
    let dtype = &spec.expected_dtype;

    match (domain, format) {
        // Numpy/Torch formats use struct with zero-copy data
        ("buffer", "numpy" | "torch") => {
            // Extract ViewBuffers from NumpyStruct results
            let buffers: Vec<Option<ViewBuffer>> = data
                .iter()
                .map(|r| match r {
                    RowResult::NumpyStruct(opt) => opt.clone(),
                    _ => None,
                })
                .collect();
            crate::output::build_numpy_series(name, buffers)
        }

        // Other binary formats: png, jpeg, blob
        ("buffer", "png" | "jpeg" | "blob") | (_, "binary") => {
            let binary_data: Vec<Option<Vec<u8>>> = data
                .iter()
                .map(|r| match r {
                    RowResult::Binary(b) => b.clone(),
                    _ => None,
                })
                .collect();
            let output_ca = BinaryChunked::from_iter_options(name, binary_data.into_iter());
            Ok(output_ca.into_series())
        }

        // List format with buffer domain - use typed list builder
        ("buffer", "list") => {
            let rows: Vec<TypedListRow> = data
                .iter()
                .map(|r| match r {
                    RowResult::TypedList(Some((typed_data, shape))) => {
                        Some((typed_data.clone(), shape.clone()))
                    }
                    _ => None,
                })
                .collect();
            build_typed_list_series_from_rows_with_dtype(name, &rows, dtype)
        }

        // Array format with buffer domain - use typed array builder
        ("buffer", "array") => {
            let rows: Vec<TypedListRow> = data
                .iter()
                .map(|r| match r {
                    RowResult::TypedArray(Some((typed_data, shape))) => {
                        Some((typed_data.clone(), shape.clone()))
                    }
                    _ => None,
                })
                .collect();
            build_typed_array_series_from_rows_with_dtype(name, &rows, dtype, &spec.sink.shape)
        }

        // Scalar domain with native format
        ("scalar", "native") => {
            let scalar_data: Vec<Option<f64>> = data
                .iter()
                .map(|r| match r {
                    RowResult::Scalar(s) => *s,
                    _ => None,
                })
                .collect();
            let output_ca = Float64Chunked::from_iter_options(name, scalar_data.into_iter());
            Ok(output_ca.into_series())
        }

        // Vector domain (perceptual hash, centroid, bbox) - typed list
        ("vector", "native" | "list") => {
            // Vectors are stored as TypedList in RowResult
            let rows: Vec<TypedListRow> = data
                .iter()
                .map(|r| {
                    match r {
                        RowResult::TypedList(Some((typed_data, shape))) => {
                            Some((typed_data.clone(), shape.clone()))
                        }
                        RowResult::Vector(Some(vals)) => {
                            // Convert Vec<f64> to TypedBufferData
                            Some((TypedBufferData::F64(vals.clone()), vec![vals.len()]))
                        }
                        _ => None,
                    }
                })
                .collect();
            build_typed_list_series_from_rows_with_dtype(name, &rows, dtype)
        }

        // Vector domain with array sink (fixed-size output like perceptual hash)
        ("vector", "array") => {
            let rows: Vec<TypedListRow> = data
                .iter()
                .map(|r| match r {
                    RowResult::TypedList(Some((typed_data, shape)))
                    | RowResult::TypedArray(Some((typed_data, shape))) => {
                        Some((typed_data.clone(), shape.clone()))
                    }
                    RowResult::Vector(Some(vals)) => {
                        Some((TypedBufferData::F64(vals.clone()), vec![vals.len()]))
                    }
                    _ => None,
                })
                .collect();
            build_typed_array_series_from_rows_with_dtype(name, &rows, dtype, &spec.sink.shape)
        }

        // Contour domain with native format
        ("contour", "native") => {
            let values: PolarsResult<Vec<AnyValue<'static>>> = data
                .iter()
                .map(|r| match r {
                    RowResult::Contours(Some(contours)) => contours_to_polars_value(contours),
                    _ => Ok(AnyValue::Null),
                })
                .collect();
            let values = values?;

            // Use statically known dtype for contours
            let point_dtype = DataType::Struct(vec![
                Field::new("x".into(), DataType::Float64),
                Field::new("y".into(), DataType::Float64),
            ]);
            let contour_dtype = DataType::Struct(vec![
                Field::new(
                    "exterior".into(),
                    DataType::List(Box::new(point_dtype.clone())),
                ),
                Field::new("interiors".into(), DataType::Null),
            ]);

            Series::from_any_values_and_dtype(name, &values, &contour_dtype, true)
        }

        // Fallback to binary
        _ => {
            let binary_data: Vec<Option<Vec<u8>>> = data
                .iter()
                .map(|r| match r {
                    RowResult::Binary(b) => b.clone(),
                    _ => None,
                })
                .collect();
            let output_ca = BinaryChunked::from_iter_options(name, binary_data.into_iter());
            Ok(output_ca.into_series())
        }
    }
}

/// Apply a mask to a buffer.
///
/// The mask should be a single-channel buffer where:
/// - 255 values keep the original pixel (fully visible)
/// - 0 values zero out the pixel (fully hidden)
/// - Intermediate values provide weighted blending
///
/// If `invert` is true, the behavior is reversed:
/// - 0 values keep the original pixel
///
/// Pad a buffer with the specified amounts and mode.
///
/// Supports constant, edge, reflect, and symmetric padding modes.
///
/// COST: Full data copy - O(output_H * output_W * C) - always allocates new buffer.
/// The output dimensions are: (input_H + top + bottom, input_W + left + right, C).
fn pad_buffer(
    buffer: &ViewBuffer,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    value: f32,
    mode: view_buffer::ops::dto::PadMode,
) -> ViewBuffer {
    use view_buffer::ops::dto::PadMode;

    // Dispatch to appropriate padding implementation based on mode
    match mode {
        PadMode::Constant => pad_constant(buffer, top, bottom, left, right, value),
        PadMode::Edge => pad_edge(buffer, top, bottom, left, right),
        PadMode::Reflect | PadMode::Symmetric => {
            // For now, treat as constant padding
            // TODO: Implement proper reflect/symmetric padding
            pad_constant(buffer, top, bottom, left, right, value)
        }
    }
}

/// Pad with constant value.
///
/// COST: Full data copy - O(H*W*C) - allocates new buffer and copies all pixels.
/// Uses row-wise memcpy (copy_from_slice) for efficient copying instead of
/// element-by-element iteration.
fn pad_constant(
    buffer: &ViewBuffer,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    value: f32,
) -> ViewBuffer {
    use view_buffer::DType;

    let shape = buffer.shape();
    let input_h = shape[0];
    let input_w = shape[1];
    let channels = if shape.len() > 2 { shape[2] } else { 1 };

    let output_h = input_h + top as usize + bottom as usize;
    let output_w = input_w + left as usize + right as usize;

    // Row stride in elements (not bytes)
    let input_row_stride = input_w * channels;
    let output_row_stride = output_w * channels;

    match buffer.dtype() {
        DType::U8 => {
            let fill_val = value.clamp(0.0, 255.0) as u8;
            // COST: Full allocation - O(output_h * output_w * channels)
            let mut output = vec![fill_val; output_h * output_w * channels];
            let input = buffer.as_slice::<u8>();

            // COST: Row-wise memcpy - O(input_h) operations instead of O(input_h * input_w * channels)
            // Each copy_from_slice copies an entire row at once using optimized memcpy
            for y in 0..input_h {
                let src_start = y * input_row_stride;
                let src_end = src_start + input_row_stride;
                let dst_y = y + top as usize;
                let dst_start = dst_y * output_row_stride + left as usize * channels;
                let dst_end = dst_start + input_row_stride;
                output[dst_start..dst_end].copy_from_slice(&input[src_start..src_end]);
            }

            ViewBuffer::from_vec_with_shape(output, vec![output_h, output_w, channels])
        }
        DType::F32 => {
            let fill_val = value;
            // COST: Full allocation - O(output_h * output_w * channels)
            let mut output = vec![fill_val; output_h * output_w * channels];
            let input = buffer.as_slice::<f32>();

            // COST: Row-wise memcpy - O(input_h) operations instead of O(input_h * input_w * channels)
            for y in 0..input_h {
                let src_start = y * input_row_stride;
                let src_end = src_start + input_row_stride;
                let dst_y = y + top as usize;
                let dst_start = dst_y * output_row_stride + left as usize * channels;
                let dst_end = dst_start + input_row_stride;
                output[dst_start..dst_end].copy_from_slice(&input[src_start..src_end]);
            }

            ViewBuffer::from_vec_with_shape(output, vec![output_h, output_w, channels])
        }
        _ => {
            // Fallback: convert to u8 and pad
            // COST: to_contiguous() may copy if buffer is not already contiguous
            let contig = buffer.to_contiguous();
            let input = contig.as_slice::<u8>();
            let fill_val = value.clamp(0.0, 255.0) as u8;
            // COST: Full allocation - O(output_h * output_w * channels)
            let mut output = vec![fill_val; output_h * output_w * channels];

            // COST: Row-wise memcpy - O(input_h) operations
            for y in 0..input_h {
                let src_start = y * input_row_stride;
                let src_end = src_start + input_row_stride;
                // Bounds check for safety
                if src_end <= input.len() {
                    let dst_y = y + top as usize;
                    let dst_start = dst_y * output_row_stride + left as usize * channels;
                    let dst_end = dst_start + input_row_stride;
                    output[dst_start..dst_end].copy_from_slice(&input[src_start..src_end]);
                }
            }

            ViewBuffer::from_vec_with_shape(output, vec![output_h, output_w, channels])
        }
    }
}

/// Pad by replicating edge values.
///
/// COST: Full data copy - O(H*W*C) - allocates new buffer.
/// Uses row-wise memcpy for the interior content and optimized edge replication.
fn pad_edge(buffer: &ViewBuffer, top: u32, bottom: u32, left: u32, right: u32) -> ViewBuffer {
    use view_buffer::DType;

    let shape = buffer.shape();
    let input_h = shape[0];
    let input_w = shape[1];
    let channels = if shape.len() > 2 { shape[2] } else { 1 };

    let output_h = input_h + top as usize + bottom as usize;
    let output_w = input_w + left as usize + right as usize;
    // Convert to usize for indexing
    let top_usize = top as usize;
    let left_usize = left as usize;

    // Row strides in elements
    let input_row_stride = input_w * channels;
    let output_row_stride = output_w * channels;

    match buffer.dtype() {
        DType::U8 => {
            // COST: Full allocation - O(output_h * output_w * channels)
            let mut output = vec![0u8; output_h * output_w * channels];
            let input = buffer.as_slice::<u8>();

            // Helper to replicate a single pixel across a range
            let replicate_pixel =
                |output: &mut [u8], dst_start: usize, count: usize, src_pixel: &[u8]| {
                    for i in 0..count {
                        let dst_idx = dst_start + i * channels;
                        output[dst_idx..dst_idx + channels].copy_from_slice(src_pixel);
                    }
                };

            // Process each output row
            for dst_y in 0..output_h {
                // Determine source row (clamped to edges)
                let src_y = if dst_y < top_usize {
                    0
                } else if dst_y >= top_usize + input_h {
                    input_h - 1
                } else {
                    dst_y - top_usize
                };

                let src_row_start = src_y * input_row_stride;
                let dst_row_start = dst_y * output_row_stride;

                // 1. Left padding: replicate first pixel of source row
                if left_usize > 0 {
                    let first_pixel = &input[src_row_start..src_row_start + channels];
                    replicate_pixel(&mut output, dst_row_start, left_usize, first_pixel);
                }

                // 2. Interior: copy entire source row
                // COST: Row-wise memcpy - O(1) per row, much faster than O(input_w * channels) individual copies
                let src_end = src_row_start + input_row_stride;
                let dst_interior_start = dst_row_start + left_usize * channels;
                let dst_interior_end = dst_interior_start + input_row_stride;
                output[dst_interior_start..dst_interior_end]
                    .copy_from_slice(&input[src_row_start..src_end]);

                // 3. Right padding: replicate last pixel of source row
                let right_count = output_w - left_usize - input_w;
                if right_count > 0 {
                    let last_pixel_start = src_row_start + (input_w - 1) * channels;
                    let last_pixel = &input[last_pixel_start..last_pixel_start + channels];
                    let dst_right_start = dst_interior_end;
                    replicate_pixel(&mut output, dst_right_start, right_count, last_pixel);
                }
            }

            ViewBuffer::from_vec_with_shape(output, vec![output_h, output_w, channels])
        }
        _ => {
            // Fallback to constant padding for other dtypes
            // TODO: Add optimized F32 path if needed
            pad_constant(buffer, top, bottom, left, right, 0.0)
        }
    }
}

/// - 255 values zero out the pixel
///
/// Uses normalized blending: pixel * (mask / 255)
fn apply_mask(buffer: &ViewBuffer, mask: &ViewBuffer, invert: bool) -> ViewBuffer {
    // Get shapes
    let buf_shape = buffer.shape();
    let mask_shape = mask.shape();

    // Handle broadcasting: mask might be 2D (H, W) while buffer is 3D (H, W, C)
    // We need to broadcast the mask to match the buffer's channels
    let effective_mask = if mask_shape.len() == 2 && buf_shape.len() == 3 {
        // Need to expand mask from (H, W) to (H, W, C)
        let h = mask_shape[0];
        let w = mask_shape[1];
        let c = buf_shape[2];

        let mask_contig = mask.to_contiguous();
        let mask_data = mask_contig.as_slice::<u8>();

        // Create expanded mask with inversion applied if needed
        let mut expanded: Vec<u8> = Vec::with_capacity(h * w * c);
        for y in 0..h {
            for x in 0..w {
                let raw_val = mask_data[y * w + x];
                let mask_val = if invert { 255 - raw_val } else { raw_val };
                // Replicate across channels
                for _ in 0..c {
                    expanded.push(mask_val);
                }
            }
        }

        ViewBuffer::from_vec_with_shape(expanded, vec![h, w, c])
    } else {
        // Same dimensionality - just use as-is, possibly inverting
        if invert {
            let mask_contig = mask.to_contiguous();
            let mask_data = mask_contig.as_slice::<u8>();
            let inverted: Vec<u8> = mask_data.iter().map(|&v| 255 - v).collect();
            ViewBuffer::from_vec_with_shape(inverted, mask_shape.to_vec())
        } else {
            mask.clone()
        }
    };

    // Apply the mask using normalized blend: pixel * (mask / 255)
    // BinaryOp::Blend computes: (a/255) * (b/255) * 255 = a * b / 255
    // This gives us the desired: pixel * (mask / 255)
    BinaryOp::Blend.execute(buffer, &effective_mask)
}

// ============================================================
// Typed Node Execution Helpers
// ============================================================

/// Execute a geometry operation with typed domain dispatch.
///
/// This handles domain transitions like Buffer → Contour (extract_contours)
/// and Contour → Buffer (rasterize).
fn execute_geometry_op(input: NodeOutput, op: &GeometryOp) -> Result<NodeOutput, String> {
    // Validate input domain
    let expected_domain = op.input_domain();
    let actual_domain = input.domain();

    if !expected_domain.accepts(actual_domain) {
        return Err(format!(
            "{}() expects {} input but received {}. Add a domain-converting operation.",
            op.name(),
            expected_domain.name(),
            actual_domain.name()
        ));
    }

    match op {
        GeometryOp::ExtractContours {
            mode,
            method,
            min_area,
        } => {
            // Buffer → Contour
            let buffer = input
                .as_buffer()
                .ok_or_else(|| "ExtractContours requires Buffer input".to_string())?;
            let contours = extract_contours(buffer, *mode, *method, *min_area);
            Ok(NodeOutput::from_contours(contours))
        }

        GeometryOp::Rasterize {
            width,
            height,
            fill_value,
            background,
            anti_alias,
        } => {
            // Contour → Buffer
            let contours = input
                .as_contours()
                .ok_or_else(|| "Rasterize requires Contour input".to_string())?;

            // Rasterize the first contour (primary contour for mask operations)
            if contours.is_empty() {
                // Empty contours → empty mask (all background)
                let mask = ViewBuffer::from_vec_with_shape(
                    vec![*background; (*height as usize) * (*width as usize)],
                    vec![*height as usize, *width as usize, 1],
                );
                Ok(NodeOutput::from_buffer(mask))
            } else {
                // Use the first contour
                let buffer = rasterize(
                    &contours[0],
                    *width,
                    *height,
                    *fill_value,
                    *background,
                    *anti_alias,
                );
                Ok(NodeOutput::from_buffer(buffer))
            }
        }

        // Contour → Scalar measures
        GeometryOp::Area { signed } => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Area requires Contour input".to_string())?;
            let area = if contours.is_empty() {
                0.0
            } else {
                view_buffer::geometry::measures::area(&contours[0], *signed)
            };
            Ok(NodeOutput::from_scalar(area))
        }

        GeometryOp::Perimeter => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Perimeter requires Contour input".to_string())?;
            let perimeter = if contours.is_empty() {
                0.0
            } else {
                view_buffer::geometry::measures::perimeter(&contours[0])
            };
            Ok(NodeOutput::from_scalar(perimeter))
        }

        GeometryOp::Centroid => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Centroid requires Contour input".to_string())?;
            let (cx, cy) = if contours.is_empty() {
                (0.0, 0.0)
            } else {
                let pt = view_buffer::geometry::measures::centroid(&contours[0]);
                (pt.x, pt.y)
            };
            Ok(NodeOutput::from_vector(vec![cx, cy]))
        }

        GeometryOp::BoundingBox => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "BoundingBox requires Contour input".to_string())?;
            let bbox = if contours.is_empty() || contours[0].bounding_box().is_none() {
                vec![0.0, 0.0, 0.0, 0.0]
            } else {
                let bb = contours[0].bounding_box().unwrap();
                vec![bb.x, bb.y, bb.width, bb.height]
            };
            Ok(NodeOutput::from_vector(bbox))
        }

        // Contour → Contour transforms
        GeometryOp::Translate { dx, dy } => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Translate requires Contour input".to_string())?;
            let translated: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::translate(c, *dx, *dy))
                .collect();
            Ok(NodeOutput::from_contours(translated))
        }

        GeometryOp::Scale { sx, sy, origin } => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Scale requires Contour input".to_string())?;
            let scaled: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::scale(c, *sx, *sy, *origin))
                .collect();
            Ok(NodeOutput::from_contours(scaled))
        }

        GeometryOp::Flip => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Flip requires Contour input".to_string())?;
            let flipped: Vec<Contour> = contours
                .iter()
                .map(view_buffer::geometry::transforms::flip)
                .collect();
            Ok(NodeOutput::from_contours(flipped))
        }

        GeometryOp::Simplify { tolerance } => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Simplify requires Contour input".to_string())?;
            let simplified: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::simplify(c, *tolerance))
                .collect();
            Ok(NodeOutput::from_contours(simplified))
        }

        GeometryOp::ConvexHull => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "ConvexHull requires Contour input".to_string())?;
            let hulls: Vec<Contour> = contours
                .iter()
                .map(view_buffer::geometry::transforms::convex_hull)
                .collect();
            Ok(NodeOutput::from_contours(hulls))
        }

        GeometryOp::Normalize {
            ref_width,
            ref_height,
        } => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Normalize requires Contour input".to_string())?;
            let normalized: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::normalize(c, *ref_width, *ref_height))
                .collect();
            Ok(NodeOutput::from_contours(normalized))
        }

        GeometryOp::ToAbsolute {
            ref_width,
            ref_height,
        } => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "ToAbsolute requires Contour input".to_string())?;
            let absolute: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::to_absolute(c, *ref_width, *ref_height))
                .collect();
            Ok(NodeOutput::from_contours(absolute))
        }

        // For other geometry ops, return an error for now
        // These can be implemented as needed
        _ => Err(format!(
            "Geometry operation {} not yet implemented for typed execution",
            op.name()
        )),
    }
}

/// Build a nested Array AnyValue from flat data and shape.
///
/// For shape [2, 3], builds Array[Array[f64, 3], 2] structure.
#[allow(dead_code)]
fn build_nested_array_value(data: &[f64], shape: &[usize]) -> PolarsResult<AnyValue<'static>> {
    if shape.is_empty() {
        // Scalar case
        return Ok(if data.is_empty() {
            AnyValue::Null
        } else {
            AnyValue::Float64(data[0])
        });
    }

    if shape.len() == 1 {
        // Base case: 1D array -> List of Float64
        let width = shape[0];
        if data.len() != width {
            return Err(
                polars_err!(ComputeError: "Data length {} doesn't match shape {:?}", data.len(), shape),
            );
        }

        // Create a Float64 array
        let values: Vec<AnyValue<'static>> = data.iter().map(|&v| AnyValue::Float64(v)).collect();
        let inner_dtype = DataType::Float64;
        let series =
            Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &values, &inner_dtype, true)?;
        return Ok(AnyValue::Array(series, width));
    }

    // Multi-dimensional: recursively build nested arrays
    let outer_dim = shape[0];
    let inner_shape = &shape[1..];
    let inner_size: usize = inner_shape.iter().product();

    if data.len() != outer_dim * inner_size {
        return Err(
            polars_err!(ComputeError: "Data length {} doesn't match shape {:?}", data.len(), shape),
        );
    }

    // Build each inner array
    let mut inner_values: Vec<AnyValue<'static>> = Vec::with_capacity(outer_dim);
    for i in 0..outer_dim {
        let start = i * inner_size;
        let end = start + inner_size;
        let inner_data = &data[start..end];
        let inner_val = build_nested_array_value(inner_data, inner_shape)?;
        inner_values.push(inner_val);
    }

    // Build inner dtype
    let mut inner_dtype = DataType::Float64;
    for &dim in inner_shape.iter().rev() {
        inner_dtype = DataType::Array(Box::new(inner_dtype), dim);
    }

    let series =
        Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &inner_values, &inner_dtype, true)?;
    Ok(AnyValue::Array(series, outer_dim))
}

/// Extract buffer data as Vec<f64> with type dispatch.
#[allow(dead_code)]
fn extract_buffer_as_f64(buf: &view_buffer::ViewBuffer) -> Vec<f64> {
    match buf.dtype() {
        view_buffer::DType::U8 => buf.as_slice::<u8>().iter().map(|&v| v as f64).collect(),
        view_buffer::DType::I8 => buf.as_slice::<i8>().iter().map(|&v| v as f64).collect(),
        view_buffer::DType::U16 => buf.as_slice::<u16>().iter().map(|&v| v as f64).collect(),
        view_buffer::DType::I16 => buf.as_slice::<i16>().iter().map(|&v| v as f64).collect(),
        view_buffer::DType::U32 => buf.as_slice::<u32>().iter().map(|&v| v as f64).collect(),
        view_buffer::DType::I32 => buf.as_slice::<i32>().iter().map(|&v| v as f64).collect(),
        view_buffer::DType::U64 => buf.as_slice::<u64>().iter().map(|&v| v as f64).collect(),
        view_buffer::DType::I64 => buf.as_slice::<i64>().iter().map(|&v| v as f64).collect(),
        view_buffer::DType::F32 => buf.as_slice::<f32>().iter().map(|&v| v as f64).collect(),
        view_buffer::DType::F64 => buf.as_slice::<f64>().to_vec(),
    }
}

// ============================================================
// Dtype-Preserving List/Array Builders
// ============================================================

/// Helper type for list row data: (TypedBufferData, shape)
type TypedListRow = Option<(TypedBufferData, Vec<usize>)>;

// Macro to generate typed list builders
macro_rules! impl_typed_list_builder {
    ($name:ident, $polars_type:ty, $extract:expr) => {
        fn $name(name: PlSmallStr, rows: &[TypedListRow]) -> PolarsResult<Series> {
            let mut builder = ListPrimitiveChunkedBuilder::<$polars_type>::new(
                name,
                rows.len(),
                64,
                <$polars_type>::get_dtype(),
            );

            for row in rows.iter() {
                if let Some((typed_data, _shape)) = row {
                    let vals = $extract(typed_data);
                    builder.append_slice(&vals);
                } else {
                    builder.append_null();
                }
            }

            Ok(builder.finish().into_series())
        }
    };
}

fn extract_as_u8(data: &TypedBufferData) -> Vec<u8> {
    match data {
        TypedBufferData::U8(v) => v.clone(),
        TypedBufferData::I8(v) => v.iter().map(|&x| x as u8).collect(),
        TypedBufferData::U16(v) => v.iter().map(|&x| x as u8).collect(),
        TypedBufferData::I16(v) => v.iter().map(|&x| x as u8).collect(),
        TypedBufferData::U32(v) => v.iter().map(|&x| x as u8).collect(),
        TypedBufferData::I32(v) => v.iter().map(|&x| x as u8).collect(),
        TypedBufferData::U64(v) => v.iter().map(|&x| x as u8).collect(),
        TypedBufferData::I64(v) => v.iter().map(|&x| x as u8).collect(),
        TypedBufferData::F32(v) => v.iter().map(|&x| x as u8).collect(),
        TypedBufferData::F64(v) => v.iter().map(|&x| x as u8).collect(),
    }
}

fn extract_as_i8(data: &TypedBufferData) -> Vec<i8> {
    match data {
        TypedBufferData::U8(v) => v.iter().map(|&x| x as i8).collect(),
        TypedBufferData::I8(v) => v.clone(),
        TypedBufferData::U16(v) => v.iter().map(|&x| x as i8).collect(),
        TypedBufferData::I16(v) => v.iter().map(|&x| x as i8).collect(),
        TypedBufferData::U32(v) => v.iter().map(|&x| x as i8).collect(),
        TypedBufferData::I32(v) => v.iter().map(|&x| x as i8).collect(),
        TypedBufferData::U64(v) => v.iter().map(|&x| x as i8).collect(),
        TypedBufferData::I64(v) => v.iter().map(|&x| x as i8).collect(),
        TypedBufferData::F32(v) => v.iter().map(|&x| x as i8).collect(),
        TypedBufferData::F64(v) => v.iter().map(|&x| x as i8).collect(),
    }
}

fn extract_as_u16(data: &TypedBufferData) -> Vec<u16> {
    match data {
        TypedBufferData::U8(v) => v.iter().map(|&x| x as u16).collect(),
        TypedBufferData::I8(v) => v.iter().map(|&x| x as u16).collect(),
        TypedBufferData::U16(v) => v.clone(),
        TypedBufferData::I16(v) => v.iter().map(|&x| x as u16).collect(),
        TypedBufferData::U32(v) => v.iter().map(|&x| x as u16).collect(),
        TypedBufferData::I32(v) => v.iter().map(|&x| x as u16).collect(),
        TypedBufferData::U64(v) => v.iter().map(|&x| x as u16).collect(),
        TypedBufferData::I64(v) => v.iter().map(|&x| x as u16).collect(),
        TypedBufferData::F32(v) => v.iter().map(|&x| x as u16).collect(),
        TypedBufferData::F64(v) => v.iter().map(|&x| x as u16).collect(),
    }
}

fn extract_as_i16(data: &TypedBufferData) -> Vec<i16> {
    match data {
        TypedBufferData::U8(v) => v.iter().map(|&x| x as i16).collect(),
        TypedBufferData::I8(v) => v.iter().map(|&x| x as i16).collect(),
        TypedBufferData::U16(v) => v.iter().map(|&x| x as i16).collect(),
        TypedBufferData::I16(v) => v.clone(),
        TypedBufferData::U32(v) => v.iter().map(|&x| x as i16).collect(),
        TypedBufferData::I32(v) => v.iter().map(|&x| x as i16).collect(),
        TypedBufferData::U64(v) => v.iter().map(|&x| x as i16).collect(),
        TypedBufferData::I64(v) => v.iter().map(|&x| x as i16).collect(),
        TypedBufferData::F32(v) => v.iter().map(|&x| x as i16).collect(),
        TypedBufferData::F64(v) => v.iter().map(|&x| x as i16).collect(),
    }
}

fn extract_as_u32(data: &TypedBufferData) -> Vec<u32> {
    match data {
        TypedBufferData::U8(v) => v.iter().map(|&x| x as u32).collect(),
        TypedBufferData::I8(v) => v.iter().map(|&x| x as u32).collect(),
        TypedBufferData::U16(v) => v.iter().map(|&x| x as u32).collect(),
        TypedBufferData::I16(v) => v.iter().map(|&x| x as u32).collect(),
        TypedBufferData::U32(v) => v.clone(),
        TypedBufferData::I32(v) => v.iter().map(|&x| x as u32).collect(),
        TypedBufferData::U64(v) => v.iter().map(|&x| x as u32).collect(),
        TypedBufferData::I64(v) => v.iter().map(|&x| x as u32).collect(),
        TypedBufferData::F32(v) => v.iter().map(|&x| x as u32).collect(),
        TypedBufferData::F64(v) => v.iter().map(|&x| x as u32).collect(),
    }
}

fn extract_as_i32(data: &TypedBufferData) -> Vec<i32> {
    match data {
        TypedBufferData::U8(v) => v.iter().map(|&x| x as i32).collect(),
        TypedBufferData::I8(v) => v.iter().map(|&x| x as i32).collect(),
        TypedBufferData::U16(v) => v.iter().map(|&x| x as i32).collect(),
        TypedBufferData::I16(v) => v.iter().map(|&x| x as i32).collect(),
        TypedBufferData::U32(v) => v.iter().map(|&x| x as i32).collect(),
        TypedBufferData::I32(v) => v.clone(),
        TypedBufferData::U64(v) => v.iter().map(|&x| x as i32).collect(),
        TypedBufferData::I64(v) => v.iter().map(|&x| x as i32).collect(),
        TypedBufferData::F32(v) => v.iter().map(|&x| x as i32).collect(),
        TypedBufferData::F64(v) => v.iter().map(|&x| x as i32).collect(),
    }
}

fn extract_as_u64(data: &TypedBufferData) -> Vec<u64> {
    match data {
        TypedBufferData::U8(v) => v.iter().map(|&x| x as u64).collect(),
        TypedBufferData::I8(v) => v.iter().map(|&x| x as u64).collect(),
        TypedBufferData::U16(v) => v.iter().map(|&x| x as u64).collect(),
        TypedBufferData::I16(v) => v.iter().map(|&x| x as u64).collect(),
        TypedBufferData::U32(v) => v.iter().map(|&x| x as u64).collect(),
        TypedBufferData::I32(v) => v.iter().map(|&x| x as u64).collect(),
        TypedBufferData::U64(v) => v.clone(),
        TypedBufferData::I64(v) => v.iter().map(|&x| x as u64).collect(),
        TypedBufferData::F32(v) => v.iter().map(|&x| x as u64).collect(),
        TypedBufferData::F64(v) => v.iter().map(|&x| x as u64).collect(),
    }
}

fn extract_as_i64(data: &TypedBufferData) -> Vec<i64> {
    match data {
        TypedBufferData::U8(v) => v.iter().map(|&x| x as i64).collect(),
        TypedBufferData::I8(v) => v.iter().map(|&x| x as i64).collect(),
        TypedBufferData::U16(v) => v.iter().map(|&x| x as i64).collect(),
        TypedBufferData::I16(v) => v.iter().map(|&x| x as i64).collect(),
        TypedBufferData::U32(v) => v.iter().map(|&x| x as i64).collect(),
        TypedBufferData::I32(v) => v.iter().map(|&x| x as i64).collect(),
        TypedBufferData::U64(v) => v.iter().map(|&x| x as i64).collect(),
        TypedBufferData::I64(v) => v.clone(),
        TypedBufferData::F32(v) => v.iter().map(|&x| x as i64).collect(),
        TypedBufferData::F64(v) => v.iter().map(|&x| x as i64).collect(),
    }
}

fn extract_as_f32(data: &TypedBufferData) -> Vec<f32> {
    match data {
        TypedBufferData::U8(v) => v.iter().map(|&x| x as f32).collect(),
        TypedBufferData::I8(v) => v.iter().map(|&x| x as f32).collect(),
        TypedBufferData::U16(v) => v.iter().map(|&x| x as f32).collect(),
        TypedBufferData::I16(v) => v.iter().map(|&x| x as f32).collect(),
        TypedBufferData::U32(v) => v.iter().map(|&x| x as f32).collect(),
        TypedBufferData::I32(v) => v.iter().map(|&x| x as f32).collect(),
        TypedBufferData::U64(v) => v.iter().map(|&x| x as f32).collect(),
        TypedBufferData::I64(v) => v.iter().map(|&x| x as f32).collect(),
        TypedBufferData::F32(v) => v.clone(),
        TypedBufferData::F64(v) => v.iter().map(|&x| x as f32).collect(),
    }
}

fn extract_as_f64(data: &TypedBufferData) -> Vec<f64> {
    match data {
        TypedBufferData::U8(v) => v.iter().map(|&x| x as f64).collect(),
        TypedBufferData::I8(v) => v.iter().map(|&x| x as f64).collect(),
        TypedBufferData::U16(v) => v.iter().map(|&x| x as f64).collect(),
        TypedBufferData::I16(v) => v.iter().map(|&x| x as f64).collect(),
        TypedBufferData::U32(v) => v.iter().map(|&x| x as f64).collect(),
        TypedBufferData::I32(v) => v.iter().map(|&x| x as f64).collect(),
        TypedBufferData::U64(v) => v.iter().map(|&x| x as f64).collect(),
        TypedBufferData::I64(v) => v.iter().map(|&x| x as f64).collect(),
        TypedBufferData::F32(v) => v.iter().map(|&x| x as f64).collect(),
        TypedBufferData::F64(v) => v.clone(),
    }
}

fn build_typed_list_u8(name: PlSmallStr, rows: &[TypedListRow]) -> PolarsResult<Series> {
    // Build UInt8 list using the proper builder
    // Requires dtype-u8 feature in polars
    let mut builder =
        ListPrimitiveChunkedBuilder::<UInt8Type>::new(name, rows.len(), 64, DataType::UInt8);

    for row in rows.iter() {
        if let Some((typed_data, _shape)) = row {
            let vals = extract_as_u8(typed_data);
            builder.append_slice(&vals);
        } else {
            builder.append_null();
        }
    }

    Ok(builder.finish().into_series())
}
impl_typed_list_builder!(build_typed_list_i8, Int8Type, extract_as_i8);
impl_typed_list_builder!(build_typed_list_u16, UInt16Type, extract_as_u16);
impl_typed_list_builder!(build_typed_list_i16, Int16Type, extract_as_i16);
impl_typed_list_builder!(build_typed_list_u32, UInt32Type, extract_as_u32);
impl_typed_list_builder!(build_typed_list_i32, Int32Type, extract_as_i32);
impl_typed_list_builder!(build_typed_list_u64, UInt64Type, extract_as_u64);
impl_typed_list_builder!(build_typed_list_i64, Int64Type, extract_as_i64);
impl_typed_list_builder!(build_typed_list_f32, Float32Type, extract_as_f32);
impl_typed_list_builder!(build_typed_list_f64, Float64Type, extract_as_f64);

/// Build a typed list series using a statically known dtype.
///
/// Unlike `build_typed_list_series_from_rows` which infers dtype from data,
/// this function uses the provided dtype string, allowing proper handling
/// of all-null data while preserving the expected output type.
///
/// If shapes indicate multi-dimensional data (shape.len() > 1), builds nested
/// List structures to preserve the shape information.
fn build_typed_list_series_from_rows_with_dtype(
    name: PlSmallStr,
    rows: &[TypedListRow],
    dtype_str: &str,
) -> PolarsResult<Series> {
    // Find the first non-null row to get shape and actual dtype
    let first_row = rows.iter().find_map(|r| r.as_ref());

    // Prefer the actual runtime dtype from the data over the planned dtype
    // This handles cases like "raw" source where dtype is only known at runtime
    let actual_dtype_str = first_row
        .map(|(data, _)| data.dtype_str())
        .unwrap_or(dtype_str);

    // Check if we have shape information indicating nested structure
    let shape = first_row.map(|(_, s)| s.clone());

    if let Some(shape) = shape {
        // If shape has more than 1 dimension (after batch dimension),
        // build nested lists to preserve the structure
        // For shape [H, W] or [H, W, C], we want nested lists
        // Dimension 0 is the batch/row dimension, so we nest dimensions 1..N
        if shape.len() > 1 {
            return build_typed_nested_list_series_from_rows_with_dtype(
                name,
                rows,
                actual_dtype_str,
                &shape,
            );
        }
    }

    // For 1D or no shape info, use flat list builders
    match actual_dtype_str {
        "u8" => build_typed_list_u8(name, rows),
        "i8" => build_typed_list_i8(name, rows),
        "u16" => build_typed_list_u16(name, rows),
        "i16" => build_typed_list_i16(name, rows),
        "u32" => build_typed_list_u32(name, rows),
        "i32" => build_typed_list_i32(name, rows),
        "u64" => build_typed_list_u64(name, rows),
        "i64" => build_typed_list_i64(name, rows),
        "f32" => build_typed_list_f32(name, rows),
        "f64" => build_typed_list_f64(name, rows),
        _ => build_typed_list_u8(name, rows), // Default fallback
    }
}

/// Build a nested List series preserving multi-dimensional shape.
///
/// This function creates nested List types (List[List[...]]) that match
/// the buffer's shape dimensions, preserving the structure of multi-dimensional data.
fn build_typed_nested_list_series_from_rows_with_dtype(
    name: PlSmallStr,
    rows: &[TypedListRow],
    dtype_str: &str,
    shape: &[usize],
) -> PolarsResult<Series> {
    // Build the nested List type from shape
    // The full shape needs to be wrapped into nested Lists
    // For shape [H, W], we want List(List(primitive))
    // For shape [H, W, C], we want List(List(List(primitive)))
    let inner_dtype = dtype_str_to_polars(dtype_str);
    let mut dtype = inner_dtype.clone();
    // Build nested List types for ALL dimensions (in reverse order)
    for _dim in shape.iter().rev() {
        dtype = DataType::List(Box::new(dtype));
    }

    // Build AnyValue lists for each row
    let values: PolarsResult<Vec<AnyValue<'static>>> = rows
        .iter()
        .map(|r| {
            if let Some((typed_data, row_shape)) = r {
                build_typed_nested_list_value(typed_data, row_shape)
            } else {
                Ok(AnyValue::Null)
            }
        })
        .collect();
    let values = values?;

    Series::from_any_values_and_dtype(name, &values, &dtype, true)
}

/// Build a nested List AnyValue from typed data and shape.
///
/// Recursively builds nested List structures matching the shape dimensions.
/// Similar to `build_typed_nested_array_value` but creates variable-length
/// List types instead of fixed-size Array types.
fn build_typed_nested_list_value(
    data: &TypedBufferData,
    shape: &[usize],
) -> PolarsResult<AnyValue<'static>> {
    if shape.is_empty() {
        return Ok(AnyValue::Null);
    }

    if shape.len() == 1 {
        // Base case: 1D list - create a list of primitive values
        let inner_dtype = data.polars_dtype();

        let values: Vec<AnyValue<'static>> = match data {
            TypedBufferData::U8(vals) => vals.iter().map(|&v| AnyValue::UInt8(v)).collect(),
            TypedBufferData::I8(vals) => vals.iter().map(|&v| AnyValue::Int8(v)).collect(),
            TypedBufferData::U16(vals) => vals.iter().map(|&v| AnyValue::UInt16(v)).collect(),
            TypedBufferData::I16(vals) => vals.iter().map(|&v| AnyValue::Int16(v)).collect(),
            TypedBufferData::U32(vals) => vals.iter().map(|&v| AnyValue::UInt32(v)).collect(),
            TypedBufferData::I32(vals) => vals.iter().map(|&v| AnyValue::Int32(v)).collect(),
            TypedBufferData::U64(vals) => vals.iter().map(|&v| AnyValue::UInt64(v)).collect(),
            TypedBufferData::I64(vals) => vals.iter().map(|&v| AnyValue::Int64(v)).collect(),
            TypedBufferData::F32(vals) => vals.iter().map(|&v| AnyValue::Float32(v)).collect(),
            TypedBufferData::F64(vals) => vals.iter().map(|&v| AnyValue::Float64(v)).collect(),
        };

        // Create a Series of primitive values - the Series itself represents the list contents
        let series =
            Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &values, &inner_dtype, true)?;
        // Wrap in AnyValue::List - the Series contains the list elements
        return Ok(AnyValue::List(series));
    }

    // Multi-dimensional: recursively build nested lists
    let outer_dim = shape[0];
    let inner_shape = &shape[1..];
    let inner_size: usize = inner_shape.iter().product();

    // Slice the data for each inner dimension
    let mut inner_values: Vec<AnyValue<'static>> = Vec::with_capacity(outer_dim);
    for i in 0..outer_dim {
        let start = i * inner_size;
        let end = start + inner_size;

        let inner_data = slice_typed_data(data, start, end);
        let inner_val = build_typed_nested_list_value(&inner_data, inner_shape)?;
        inner_values.push(inner_val);
    }

    // Build the dtype for the inner values (what's inside each AnyValue::List)
    // The inner_values are AnyValue::List elements, each containing a Series
    // The dtype should match what those List values contain
    let base_dtype = data.polars_dtype();
    let mut inner_dtype = base_dtype;
    // Build nested List types for inner_shape dimensions
    for _dim in inner_shape.iter().rev() {
        inner_dtype = DataType::List(Box::new(inner_dtype));
    }

    // Create Series from the inner AnyValue::List values
    // The dtype here is the type of each element in inner_values (i.e., List types)
    let series =
        Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &inner_values, &inner_dtype, true)?;
    Ok(AnyValue::List(series))
}

/// Build a typed fixed-size array series using a statically known dtype.
///
/// Unlike `build_typed_array_series_from_rows` which infers dtype from data,
/// this function uses the provided dtype string and shape, allowing proper
/// handling of all-null data while preserving the expected output type.
fn build_typed_array_series_from_rows_with_dtype(
    name: PlSmallStr,
    rows: &[TypedListRow],
    dtype_str: &str,
    sink_shape: &Option<Vec<usize>>,
) -> PolarsResult<Series> {
    // Get shape from sink spec or infer from first non-null row
    let shape = sink_shape
        .clone()
        .or_else(|| rows.iter().find_map(|r| r.as_ref().map(|(_, s)| s.clone())));

    let Some(shape) = shape else {
        // No shape available - fall back to list
        return build_typed_list_series_from_rows_with_dtype(name, rows, dtype_str);
    };

    // Build the nested Array type from shape
    let inner_dtype = dtype_str_to_polars(dtype_str);
    let mut dtype = inner_dtype.clone();
    for &dim in shape.iter().rev() {
        dtype = DataType::Array(Box::new(dtype), dim);
    }

    // Build AnyValue arrays for each row
    let values: PolarsResult<Vec<AnyValue<'static>>> = rows
        .iter()
        .map(|r| {
            if let Some((typed_data, row_shape)) = r {
                build_typed_nested_array_value(typed_data, row_shape)
            } else {
                Ok(AnyValue::Null)
            }
        })
        .collect();
    let values = values?;

    Series::from_any_values_and_dtype(name, &values, &dtype, true)
}

/// Build a nested Array AnyValue from typed data and shape.
fn build_typed_nested_array_value(
    data: &TypedBufferData,
    shape: &[usize],
) -> PolarsResult<AnyValue<'static>> {
    if shape.is_empty() {
        return Ok(AnyValue::Null);
    }

    if shape.len() == 1 {
        // Base case: 1D array
        let width = shape[0];
        let inner_dtype = data.polars_dtype();

        let values: Vec<AnyValue<'static>> = match data {
            TypedBufferData::U8(vals) => vals.iter().map(|&v| AnyValue::UInt8(v)).collect(),
            TypedBufferData::I8(vals) => vals.iter().map(|&v| AnyValue::Int8(v)).collect(),
            TypedBufferData::U16(vals) => vals.iter().map(|&v| AnyValue::UInt16(v)).collect(),
            TypedBufferData::I16(vals) => vals.iter().map(|&v| AnyValue::Int16(v)).collect(),
            TypedBufferData::U32(vals) => vals.iter().map(|&v| AnyValue::UInt32(v)).collect(),
            TypedBufferData::I32(vals) => vals.iter().map(|&v| AnyValue::Int32(v)).collect(),
            TypedBufferData::U64(vals) => vals.iter().map(|&v| AnyValue::UInt64(v)).collect(),
            TypedBufferData::I64(vals) => vals.iter().map(|&v| AnyValue::Int64(v)).collect(),
            TypedBufferData::F32(vals) => vals.iter().map(|&v| AnyValue::Float32(v)).collect(),
            TypedBufferData::F64(vals) => vals.iter().map(|&v| AnyValue::Float64(v)).collect(),
        };

        let series =
            Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &values, &inner_dtype, true)?;
        return Ok(AnyValue::Array(series, width));
    }

    // Multi-dimensional: recursively build nested arrays
    let outer_dim = shape[0];
    let inner_shape = &shape[1..];
    let inner_size: usize = inner_shape.iter().product();

    // Slice the data for each inner dimension
    let mut inner_values: Vec<AnyValue<'static>> = Vec::with_capacity(outer_dim);
    for i in 0..outer_dim {
        let start = i * inner_size;
        let end = start + inner_size;

        let inner_data = slice_typed_data(data, start, end);
        let inner_val = build_typed_nested_array_value(&inner_data, inner_shape)?;
        inner_values.push(inner_val);
    }

    // Build inner dtype
    let base_dtype = data.polars_dtype();
    let mut inner_dtype = base_dtype;
    for &dim in inner_shape.iter().rev() {
        inner_dtype = DataType::Array(Box::new(inner_dtype), dim);
    }

    let series =
        Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &inner_values, &inner_dtype, true)?;
    Ok(AnyValue::Array(series, outer_dim))
}

/// Slice typed buffer data by index range.
fn slice_typed_data(data: &TypedBufferData, start: usize, end: usize) -> TypedBufferData {
    match data {
        TypedBufferData::U8(vals) => TypedBufferData::U8(vals[start..end].to_vec()),
        TypedBufferData::I8(vals) => TypedBufferData::I8(vals[start..end].to_vec()),
        TypedBufferData::U16(vals) => TypedBufferData::U16(vals[start..end].to_vec()),
        TypedBufferData::I16(vals) => TypedBufferData::I16(vals[start..end].to_vec()),
        TypedBufferData::U32(vals) => TypedBufferData::U32(vals[start..end].to_vec()),
        TypedBufferData::I32(vals) => TypedBufferData::I32(vals[start..end].to_vec()),
        TypedBufferData::U64(vals) => TypedBufferData::U64(vals[start..end].to_vec()),
        TypedBufferData::I64(vals) => TypedBufferData::I64(vals[start..end].to_vec()),
        TypedBufferData::F32(vals) => TypedBufferData::F32(vals[start..end].to_vec()),
        TypedBufferData::F64(vals) => TypedBufferData::F64(vals[start..end].to_vec()),
    }
}

/// Encode a NodeOutput to bytes based on sink format.
///
/// Dispatches to the appropriate encoding based on the output domain.
fn encode_node_output(output: &NodeOutput, sink: &SinkSpec) -> Result<OutputValue, String> {
    let format = sink.format.as_str();

    match (output, format) {
        // Buffer outputs - numpy/torch use zero-copy struct format
        (NodeOutput::Buffer(buf), "numpy" | "torch") => {
            // Clone the ViewBuffer from the Arc for ownership transfer
            // ViewBuffer clone is cheap (Arc clone of internal storage)
            // The actual zero-copy happens during series building via into_polars_buffer()
            Ok(OutputValue::NumpyStruct((**buf).clone()))
        }
        (NodeOutput::Buffer(buf), "png" | "jpeg" | "blob") => {
            let pipeline = PipelineSpec {
                source: SourceSpec {
                    format: "blob".to_string(),
                    dtype: None,
                    width: None,
                    height: None,
                    fill_value: 255,
                    background: 0,
                    shape_pipeline: None,
                    require_contiguous: false,
                },
                shape_hints: None,
                ops: vec![],
                sink: sink.clone(),
            };
            crate::execute::encode_sink(buf, &pipeline)
                .map(OutputValue::Binary)
                .map_err(|e| format!("Encode error: {e}"))
        }
        (NodeOutput::Buffer(buf), "list") => {
            // Convert buffer to typed list structure preserving buffer dtype
            let contig = buf.to_contiguous();
            let shape = contig.shape().to_vec();

            // Extract data with original dtype preserved
            let data = TypedBufferData::from_buffer(&contig);

            Ok(OutputValue::TypedList { data, shape })
        }
        (NodeOutput::Buffer(buf), "array") => {
            // Convert buffer to typed fixed-size array preserving buffer dtype
            let contig = buf.to_contiguous();
            let buffer_shape = contig.shape().to_vec();

            // Use provided shape from sink spec, or infer from buffer
            let shape = if let Some(ref spec_shape) = sink.shape {
                // Require exact shape match to avoid dimension confusion
                if spec_shape != &buffer_shape {
                    return Err(format!(
                        "Array sink shape {spec_shape:?} does not match buffer shape {buffer_shape:?}. \
                         Use squeeze() or expand_dims() to adjust dimensions, \
                         or omit shape to infer from buffer."
                    ));
                }
                spec_shape.clone()
            } else {
                // Infer shape from buffer
                buffer_shape
            };

            // Extract data with original dtype preserved
            let data = TypedBufferData::from_buffer(&contig);

            Ok(OutputValue::TypedArray { data, shape })
        }

        // Native format dispatches based on domain
        (NodeOutput::Buffer(_), "native") => {
            Err("Buffer outputs require explicit format (numpy/png/jpeg). Use 'native' for contours/scalars.".to_string())
        }
        (NodeOutput::Contours(contours), "native") => {
            // Return as contour struct data
            Ok(OutputValue::Contours(contours.clone()))
        }
        (NodeOutput::Scalar(val), "native") => {
            Ok(OutputValue::Scalar(*val))
        }
        (NodeOutput::Vector(vals), "native") => {
            Ok(OutputValue::Vector(vals.clone()))
        }
        // List format for vector outputs
        (NodeOutput::Vector(vals), "list") => {
            Ok(OutputValue::Vector(vals.clone()))
        }

        // Type mismatches
        (NodeOutput::Contours(_), "numpy" | "png" | "jpeg") => {
            Err(format!(
                "Cannot encode Contours as {format}. Use 'native' or add .rasterize() first."
            ))
        }
        (NodeOutput::Scalar(_), "numpy" | "png" | "jpeg") => {
            Err(format!(
                "Cannot encode Scalar as {format}. Use 'native' format."
            ))
        }
        (NodeOutput::Vector(_), "numpy" | "png" | "jpeg") => {
            Err(format!(
                "Cannot encode Vector as {format}. Use 'native' format."
            ))
        }

        _ => Err(format!("Unsupported sink format: {format}"))
    }
}

/// Typed buffer data for dtype-preserving list/array outputs.
#[derive(Debug, Clone)]
enum TypedBufferData {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl TypedBufferData {
    /// Extract typed data from a ViewBuffer, preserving its dtype.
    fn from_buffer(buf: &ViewBuffer) -> Self {
        let contig = buf.to_contiguous();
        match contig.dtype() {
            view_buffer::DType::U8 => TypedBufferData::U8(contig.as_slice::<u8>().to_vec()),
            view_buffer::DType::I8 => TypedBufferData::I8(contig.as_slice::<i8>().to_vec()),
            view_buffer::DType::U16 => TypedBufferData::U16(contig.as_slice::<u16>().to_vec()),
            view_buffer::DType::I16 => TypedBufferData::I16(contig.as_slice::<i16>().to_vec()),
            view_buffer::DType::U32 => TypedBufferData::U32(contig.as_slice::<u32>().to_vec()),
            view_buffer::DType::I32 => TypedBufferData::I32(contig.as_slice::<i32>().to_vec()),
            view_buffer::DType::U64 => TypedBufferData::U64(contig.as_slice::<u64>().to_vec()),
            view_buffer::DType::I64 => TypedBufferData::I64(contig.as_slice::<i64>().to_vec()),
            view_buffer::DType::F32 => TypedBufferData::F32(contig.as_slice::<f32>().to_vec()),
            view_buffer::DType::F64 => TypedBufferData::F64(contig.as_slice::<f64>().to_vec()),
        }
    }

    /// Get the Polars DataType for this typed data.
    fn polars_dtype(&self) -> DataType {
        match self {
            TypedBufferData::U8(_) => DataType::UInt8,
            TypedBufferData::I8(_) => DataType::Int8,
            TypedBufferData::U16(_) => DataType::UInt16,
            TypedBufferData::I16(_) => DataType::Int16,
            TypedBufferData::U32(_) => DataType::UInt32,
            TypedBufferData::I32(_) => DataType::Int32,
            TypedBufferData::U64(_) => DataType::UInt64,
            TypedBufferData::I64(_) => DataType::Int64,
            TypedBufferData::F32(_) => DataType::Float32,
            TypedBufferData::F64(_) => DataType::Float64,
        }
    }

    /// Get the dtype string for this typed data.
    fn dtype_str(&self) -> &'static str {
        match self {
            TypedBufferData::U8(_) => "u8",
            TypedBufferData::I8(_) => "i8",
            TypedBufferData::U16(_) => "u16",
            TypedBufferData::I16(_) => "i16",
            TypedBufferData::U32(_) => "u32",
            TypedBufferData::I32(_) => "i32",
            TypedBufferData::U64(_) => "u64",
            TypedBufferData::I64(_) => "i64",
            TypedBufferData::F32(_) => "f32",
            TypedBufferData::F64(_) => "f64",
        }
    }
}

/// Output value from encoding - can be binary, contour struct, scalar, or array.
#[derive(Debug, Clone)]
enum OutputValue {
    Binary(Vec<u8>),
    Contours(Arc<Vec<Contour>>),
    Scalar(f64),
    Vector(Arc<Vec<f64>>),
    /// Typed list representation for "list" sink - preserves buffer dtype.
    TypedList {
        /// Typed data preserving original buffer dtype.
        data: TypedBufferData,
        /// Original shape of the buffer.
        shape: Vec<usize>,
    },
    /// Typed fixed-size array representation for "array" sink.
    TypedArray {
        /// Typed data preserving original buffer dtype.
        data: TypedBufferData,
        /// Fixed shape (validated against buffer).
        shape: Vec<usize>,
    },
    /// Numpy/Torch struct output (zero-copy ViewBuffer for struct encoding).
    NumpyStruct(ViewBuffer),
}

/// Convert contours to Polars AnyValue representation.
fn contours_to_polars_value(contours: &[Contour]) -> PolarsResult<AnyValue<'static>> {
    if contours.is_empty() {
        // Return null for empty contours
        return Ok(AnyValue::Null);
    }

    // Use the first contour (primary contour)
    let contour = &contours[0];

    // Build exterior points as List of Struct {x: f64, y: f64}
    let points: Vec<AnyValue<'static>> = contour
        .exterior
        .iter()
        .map(|p| {
            let values = vec![AnyValue::Float64(p.x), AnyValue::Float64(p.y)];
            let fields = vec![
                Field::new("x".into(), DataType::Float64),
                Field::new("y".into(), DataType::Float64),
            ];
            AnyValue::StructOwned(Box::new((values, fields)))
        })
        .collect();

    // Create exterior as List
    let point_dtype = DataType::Struct(vec![
        Field::new("x".into(), DataType::Float64),
        Field::new("y".into(), DataType::Float64),
    ]);
    let exterior_series =
        Series::from_any_values_and_dtype("exterior".into(), &points, &point_dtype, true)?;

    // Build the contour struct: {exterior: List<{x, y}>, interiors: null}
    let contour_values = vec![
        AnyValue::List(exterior_series),
        AnyValue::Null, // interiors (holes) - not yet implemented
    ];
    let contour_fields = vec![
        Field::new(
            "exterior".into(),
            DataType::List(Box::new(point_dtype.clone())),
        ),
        Field::new("interiors".into(), DataType::Null),
    ];

    Ok(AnyValue::StructOwned(Box::new((
        contour_values,
        contour_fields,
    ))))
}

/// A node in the pipeline graph.
#[derive(Debug, Deserialize)]
pub struct GraphNode {
    /// Source specification for this node's input.
    pub source: SourceSpec,
    /// Operations to apply.
    #[serde(default)]
    pub ops: Vec<crate::pipeline::OpSpec>,
    /// Upstream node IDs this node depends on.
    #[serde(default)]
    pub upstream: Vec<String>,
    /// Optional user-defined alias for multi-output.
    /// Note: Used for deserialization; alias becomes the key in outputs map.
    #[serde(default)]
    #[allow(dead_code)]
    pub alias: Option<String>,
}

/// Output specification for a single output in the graph.
#[derive(Debug, Deserialize)]
pub struct OutputSpec {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: SinkSpec,
    /// Expected output domain for validation and type inference.
    #[serde(default = "default_domain")]
    pub expected_domain: String,
    /// Expected output dtype for list/array sinks.
    #[serde(default = "default_dtype")]
    pub expected_dtype: String,
}

fn default_domain() -> String {
    "buffer".to_string()
}

fn default_dtype() -> String {
    "u8".to_string()
}

/// Unified pipeline graph specification.
///
/// This struct handles all cases:
/// - Single output: `outputs` contains only "_output" key, returns Binary
/// - Multi output: `outputs` contains multiple keys, returns Struct
#[derive(Debug, Deserialize)]
pub struct UnifiedGraph {
    /// Named nodes in the graph.
    pub nodes: HashMap<String, GraphNode>,
    /// Output specifications (alias -> spec).
    /// Single output uses "_output" as key.
    pub outputs: HashMap<String, OutputSpec>,
    /// Mapping from node IDs to input column indices.
    /// Only root nodes (no upstream) have bindings.
    #[serde(default)]
    pub column_bindings: HashMap<String, usize>,
    /// Cached topological order (computed once during parsing).
    /// Not serialized - computed on load.
    #[serde(skip)]
    cached_order: Vec<String>,
}

impl UnifiedGraph {
    /// Parse a graph from JSON.
    ///
    /// This also computes and caches the topological order for efficient
    /// repeated execution.
    pub fn from_json(json: &str) -> PolarsResult<Self> {
        let mut graph: Self = serde_json::from_str(json)
            .map_err(|e| polars_err!(ComputeError: "Failed to parse pipeline graph: {}", e))?;

        // Pre-compute and cache the topological order
        graph.cached_order = graph.compute_topological_order()?;

        Ok(graph)
    }

    /// Check if this is a single-output graph (returns Binary instead of Struct).
    pub fn is_single_output(&self) -> bool {
        self.outputs.len() == 1 && self.outputs.contains_key("_output")
    }

    /// Get all output node IDs.
    #[allow(dead_code)]
    pub fn output_node_ids(&self) -> HashSet<String> {
        self.outputs.values().map(|s| s.node.clone()).collect()
    }

    /// Get cached topological order.
    /// The order is computed once during parsing and reused for all executions.
    fn topological_order(&self) -> &[String] {
        &self.cached_order
    }

    /// Compute nodes in topological order (dependencies first).
    /// Includes all nodes reachable from any output.
    fn compute_topological_order(&self) -> PolarsResult<Vec<String>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();

        fn dfs(
            node_id: &str,
            nodes: &HashMap<String, GraphNode>,
            visited: &mut HashSet<String>,
            order: &mut Vec<String>,
        ) -> PolarsResult<()> {
            if visited.contains(node_id) {
                return Ok(());
            }

            visited.insert(node_id.to_string());

            if let Some(node) = nodes.get(node_id) {
                for upstream_id in &node.upstream {
                    dfs(upstream_id, nodes, visited, order)?;
                }
            }

            order.push(node_id.to_string());
            Ok(())
        }

        // Start from all output nodes
        for spec in self.outputs.values() {
            dfs(&spec.node, &self.nodes, &mut visited, &mut order)?;
        }

        Ok(order)
    }

    /// Execute the graph on input series.
    ///
    /// Returns:
    /// - Binary column if single output ("_output" only)
    /// - Struct column with named fields (Binary/Float64/Struct) if multiple outputs
    ///
    /// # Optimizations
    ///
    /// 1. **Per-node precompilation**: Nodes where all op params are literals
    ///    have their ViewDtos resolved once before the row loop and reused.
    /// 2. **Batch-level panic catching**: A single catch_unwind wraps the
    ///    entire batch for reduced overhead vs per-row catching.
    /// 3. **Cached topological order**: Computed once during from_json().
    ///
    /// # Typed Node Support
    ///
    /// The executor now handles typed nodes via `NodeOutput`, supporting:
    /// - Buffer (images/arrays) → Binary encoding
    /// - Contours (geometry) → Struct encoding with "native" format
    /// - Scalar (single values) → Float64 with "native" format
    /// - Vector (multiple values) → List/Struct with "native" format
    pub fn execute(
        &self,
        inputs: &[Series],
        expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<Series> {
        // Get cached topological order
        let order = self.topological_order();

        // Get length from first input
        let len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            return Err(polars_err!(ComputeError: "No input columns provided"));
        };

        // Get output aliases in deterministic order
        let mut output_aliases: Vec<&String> = self.outputs.keys().collect();
        output_aliases.sort();

        // ============================================================
        // OPTIMIZATION: Per-node precompilation
        // ============================================================
        // For nodes where all op params are literals, precompile ViewDtos
        // once and reuse for all rows. This avoids repeated parameter
        // resolution in the hot loop.
        let precompiled: HashMap<String, Vec<ViewDto>> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.ops.iter().all(|op| op.is_all_literal()))
            .filter_map(|(node_id, node)| {
                // Resolve ops with row_idx=0 and empty expr_columns (all literal anyway)
                let ops: Result<Vec<ViewDto>, _> = node
                    .ops
                    .iter()
                    .map(|op| resolve_op(op, 0, &HashMap::new()))
                    .collect();
                ops.ok().map(|v| (node_id.clone(), v))
            })
            .collect();

        // Prepare result storage for typed outputs
        let mut results: HashMap<String, Vec<RowResult>> = HashMap::new();
        for alias in &output_aliases {
            results.insert((*alias).clone(), Vec::with_capacity(len));
        }

        // ============================================================
        // OPTIMIZATION: Batch-level panic catching
        // ============================================================
        // Wrap the entire row loop in a single catch_unwind to reduce
        // the overhead of setting up unwinding machinery per-row.
        let batch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for row_idx in 0..len {
                // Node output cache for this row (typed outputs)
                let mut node_outputs: HashMap<String, NodeOutput> = HashMap::new();

                // Execute nodes in order
                for node_id in order {
                    let node = match self.nodes.get(node_id) {
                        Some(n) => n,
                        None => continue, // Skip missing nodes (shouldn't happen)
                    };

                    // Determine input source for this node
                    // A node is a "root" if it has a column binding (reads from DataFrame column)
                    // Nodes can have both column bindings AND upstream (e.g., contour with shape inference)
                    let has_column_binding = self.column_bindings.contains_key(node_id);
                    let node_input: Option<NodeOutput> = if has_column_binding {
                        // Root node: get input from column binding
                        let col_idx = self.column_bindings.get(node_id).copied().unwrap_or(0);

                        if col_idx >= inputs.len() {
                            // Return error as None to indicate failure
                            return Err(format!(
                                "Column index {col_idx} out of bounds for node '{node_id}'"
                            ));
                        }

                        let input_series = &inputs[col_idx];

                        // Check if this is a contour source (Struct input) vs binary source
                        let source_format = node.source.format.as_str();
                        if source_format == "contour" {
                            // Contour source: parse struct and rasterize
                            match input_series.get(row_idx) {
                                Ok(value) if !value.is_null() => {
                                    // Check if we have shape_pipeline for dimension inference
                                    if let Some(ref shape_pipeline) = node.source.shape_pipeline {
                                        // Extract node_id from shape_pipeline JSON
                                        let shape_node_id = shape_pipeline
                                            .get("node_id")
                                            .and_then(|v| v.as_str())
                                            .ok_or_else(|| {
                                                "shape_pipeline missing 'node_id'".to_string()
                                            })?;

                                        // Look up the referenced output
                                        let shape_output = node_outputs.get(shape_node_id).ok_or_else(|| {
                                            format!(
                                                "Shape reference '{shape_node_id}' not found. Ensure the shape source is defined before this contour pipeline."
                                            )
                                        })?;

                                        // Get buffer from output
                                        let shape_buffer = shape_output.as_buffer().ok_or_else(|| {
                                            format!("Shape reference '{shape_node_id}' must be a Buffer, not {:?}", shape_output.domain())
                                        })?;

                                        // Get dimensions from buffer shape (HWC layout: [height, width, channels])
                                        let shape = shape_buffer.shape();
                                        if shape.len() < 2 {
                                            return Err(format!(
                                                "Shape buffer has invalid dimensions: expected at least 2D, got {}D",
                                                shape.len()
                                            ));
                                        }
                                        let height = shape[0] as u32;
                                        let width = shape[1] as u32;

                                        // Get fill and background values
                                        let fill_value = node.source.fill_value;
                                        let background = node.source.background;

                                        match decode_contour_source_with_dims(
                                            &value, width, height, fill_value, background,
                                        ) {
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => {
                                                return Err(format!("Contour decode error: {e}"))
                                            }
                                        }
                                    } else {
                                        // Use explicit width/height parameters
                                        let first_output = self.outputs.values().next().unwrap();
                                        let temp_spec = PipelineSpec {
                                            source: node.source.clone(),
                                            shape_hints: None,
                                            ops: vec![],
                                            sink: first_output.sink.clone(),
                                        };
                                        match decode_contour_source(
                                            &value,
                                            row_idx,
                                            &temp_spec,
                                            expr_columns,
                                        ) {
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => {
                                                return Err(format!("Contour decode error: {e}"))
                                            }
                                        }
                                    }
                                }
                                _ => None,
                            }
                        } else if source_format == "file_path" {
                            // File path source: read from local or cloud storage
                            // Handle Null dtype (all nulls) gracefully
                            if input_series.dtype() == &DataType::Null {
                                None
                            } else {
                                let input_ca = match input_series.str() {
                                    Ok(ca) => ca,
                                    Err(_) => {
                                        return Err(format!(
                                            "Expected String column for file_path source '{node_id}', got {:?}",
                                            input_series.dtype()
                                        ))
                                    }
                                };

                                match input_ca.get(row_idx) {
                                    Some(path) => {
                                        // Read the file (local, cloud, or HTTP)
                                        let bytes = if path.starts_with("s3://")
                                            || path.starts_with("gs://")
                                            || path.starts_with("az://")
                                            || path.starts_with("abfs://")
                                            || path.starts_with("abfss://")
                                            || path.starts_with("http://")
                                            || path.starts_with("https://")
                                        {
                                            // Remote path (cloud storage or HTTP URL)
                                            #[cfg(feature = "cloud")]
                                            {
                                                match crate::cloud::read_file(path, None) {
                                                    Ok(b) => b,
                                                    Err(e) => {
                                                        return Err(format!(
                                                        "Failed to read remote file '{path}': {e}"
                                                    ))
                                                    }
                                                }
                                            }
                                            #[cfg(not(feature = "cloud"))]
                                            {
                                                return Err(format!(
                                                    "Remote storage support is not enabled. \
                                                    Install with 'cloud' feature: \
                                                    pip install polars-cv[cloud] or \
                                                    cargo build --features cloud"
                                                ));
                                            }
                                        } else {
                                            // Local file path
                                            std::fs::read(path).map_err(|e| {
                                                format!("Failed to read local file '{path}': {e}")
                                            })?
                                        };

                                        // Decode as image_bytes (auto-detect format)
                                        let first_output = self.outputs.values().next().unwrap();
                                        let mut source_spec = node.source.clone();
                                        source_spec.format = "image_bytes".to_string();
                                        let temp_spec = PipelineSpec {
                                            source: source_spec,
                                            shape_hints: None,
                                            ops: vec![],
                                            sink: first_output.sink.clone(),
                                        };
                                        match decode_source(&bytes, &temp_spec) {
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => {
                                                return Err(format!(
                                                    "Decode error for file '{path}': {e}"
                                                ))
                                            }
                                        }
                                    }
                                    None => None,
                                }
                            }
                        } else if node.source.format == "list" || node.source.format == "array" {
                            // List/Array source: decode from nested Polars List or Array
                            if input_series.dtype() == &DataType::Null {
                                None
                            } else {
                                // dtype is now optional - will be inferred from Polars type if not provided
                                let dtype_opt = node.source.dtype.as_deref();
                                let require_contiguous = node.source.require_contiguous;

                                match decode_list_or_array_source(
                                    input_series,
                                    row_idx,
                                    dtype_opt,
                                    require_contiguous,
                                ) {
                                    Ok(Some(buf)) => Some(NodeOutput::from_buffer(buf)),
                                    Ok(None) => None,
                                    Err(e) => return Err(format!("List/Array decode error: {e}")),
                                }
                            }
                        } else {
                            // Binary source: decode from bytes
                            // Handle Null dtype (all nulls) gracefully
                            if input_series.dtype() == &DataType::Null {
                                None
                            } else {
                                let input_ca = match input_series.binary() {
                                    Ok(ca) => ca,
                                    Err(_) => {
                                        return Err(format!(
                                            "Expected Binary column for node '{node_id}', got {:?}",
                                            input_series.dtype()
                                        ))
                                    }
                                };

                                // Try zero-copy path for blob and raw formats
                                let source_format = node.source.format.as_str();
                                if source_format == "blob" || source_format == "raw" {
                                    // Zero-copy path
                                    if let Some((buffer, offset, len)) =
                                        get_binary_row_buffer(input_ca, row_idx)
                                    {
                                        match decode_binary_zero_copy(
                                            buffer,
                                            offset,
                                            len,
                                            source_format,
                                            node.source.dtype.as_deref(),
                                        ) {
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => {
                                                return Err(format!("Zero-copy decode error: {e}"))
                                            }
                                        }
                                    } else {
                                        None // Null row
                                    }
                                } else {
                                    // Other binary formats (image_bytes, etc.) - use existing path
                                    match input_ca.get(row_idx) {
                                        Some(bytes) => {
                                            // Create temp spec for decoding
                                            let first_output =
                                                self.outputs.values().next().unwrap();
                                            let temp_spec = PipelineSpec {
                                                source: node.source.clone(),
                                                shape_hints: None,
                                                ops: vec![],
                                                sink: first_output.sink.clone(),
                                            };
                                            // Image formats require decoding, can't be zero-copy
                                            match decode_source(bytes, &temp_spec) {
                                                Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                                Err(e) => return Err(format!("Decode error: {e}")),
                                            }
                                        }
                                        None => None,
                                    }
                                }
                            }
                        }
                    } else {
                        // Non-root node: get input from upstream node's output
                        let upstream_id = &node.upstream[0];
                        node_outputs.get(upstream_id).cloned()
                    };

                    if let Some(input) = node_input {
                        // Get ViewDtos - use precompiled if available, otherwise resolve per-row
                        let view_dtos: Vec<ViewDto> = if let Some(cached) = precompiled.get(node_id)
                        {
                            // Fast path: clone precompiled ops
                            cached.clone()
                        } else {
                            // Slow path: resolve per-row for expression parameters
                            let mut dtos = Vec::with_capacity(node.ops.len());
                            for op_spec in &node.ops {
                                match resolve_op(op_spec, row_idx, expr_columns) {
                                    Ok(dto) => dtos.push(dto),
                                    Err(e) => return Err(format!("Op resolution error: {e}")),
                                }
                            }
                            dtos
                        };

                        // Execute operations with typed dispatch
                        // OPTIMIZATION: Batch consecutive buffer ops into a single ViewExpr
                        // to allow view-buffer's optimizer to fuse operations.
                        let mut current_output = input;

                        // Helper: flush pending buffer ops
                        // COST DOCUMENTATION:
                        // - ViewExpr::new_source() clones the Arc<Vec<u8>> (cheap pointer copy, not data)
                        // - expr.plan().execute() produces new buffer (full data copy for the result)
                        // - Batching multiple ops here amortizes the clone cost across all ops
                        fn flush_buffer_ops(
                            output: NodeOutput,
                            pending_ops: &mut Vec<ViewDto>,
                        ) -> Result<NodeOutput, String> {
                            if pending_ops.is_empty() {
                                // COST: Zero-copy - no pending ops means no work
                                return Ok(output);
                            }
                            let buf = output.as_buffer().ok_or_else(|| {
                                format!(
                                    "Expected Buffer for pending ops, got {:?}",
                                    output.domain()
                                )
                            })?;
                            // COST: Arc clone - O(1), just increments reference count
                            let mut expr = ViewExpr::new_source((**buf).clone());
                            for op in pending_ops.drain(..) {
                                expr = expr.apply_op(op);
                            }
                            // COST: Full data copy - O(H*W*C) - produces new buffer with all ops applied
                            let result = expr.plan().execute();
                            Ok(NodeOutput::from_buffer(result))
                        }

                        let mut pending_buffer_ops: Vec<ViewDto> = Vec::new();

                        for view_dto in view_dtos {
                            match &view_dto {
                                ViewDto::Geometry(geo_op) => {
                                    // Flush pending buffer ops first
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    // Use typed geometry execution
                                    current_output = execute_geometry_op(current_output, geo_op)?;
                                }
                                ViewDto::Binary { op, other_node_id } => {
                                    // Flush pending buffer ops first
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    // Binary operation: both inputs must be buffers
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Binary op requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let other_output = node_outputs.get(other_node_id)
                                        .ok_or_else(|| format!("Binary op references unknown node '{other_node_id}'"))?;
                                    let other_buf = other_output.as_buffer().ok_or_else(|| {
                                        format!(
                                            "Binary op other operand must be Buffer, got {:?}",
                                            other_output.domain()
                                        )
                                    })?;
                                    let result = op.execute(current_buf, other_buf);
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::ApplyMask {
                                    mask_node_id,
                                    invert,
                                } => {
                                    // Flush pending buffer ops first
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    // Mask operation: buffer masked by another buffer
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ApplyMask requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let mask_output = node_outputs.get(mask_node_id)
                                        .ok_or_else(|| format!("ApplyMask references unknown node '{mask_node_id}'"))?;
                                    let mask_buf = mask_output.as_buffer().ok_or_else(|| {
                                        format!(
                                            "ApplyMask mask must be Buffer, got {:?}",
                                            mask_output.domain()
                                        )
                                    })?;
                                    let result = apply_mask(current_buf, mask_buf, *invert);
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::Reduction(reduction_op) => {
                                    // Flush pending buffer ops first
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    // Reduction operation: buffer → scalar (for global) or buffer (for axis)
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Reduction requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let result = reduction_op.execute(current_buf);
                                    // Check if this is a global reduction (produces scalar)
                                    if result.shape() == [1] {
                                        // Global reduction: extract scalar value
                                        let scalar_val = result.as_slice::<f64>()[0];
                                        current_output = NodeOutput::Scalar(scalar_val);
                                    } else {
                                        // Axis reduction: still a buffer
                                        current_output = NodeOutput::from_buffer(result);
                                    }
                                }
                                ViewDto::Histogram(histogram_op) => {
                                    use view_buffer::ops::histogram::HistogramOutput;
                                    // Flush pending buffer ops first
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    // Histogram operation: buffer → vector or buffer (for quantized)
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Histogram requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let result = histogram_op.execute(current_buf);
                                    // Check output mode to determine result type
                                    match histogram_op.output {
                                        HistogramOutput::Quantized => {
                                            // Quantized: still a buffer with bin indices
                                            current_output = NodeOutput::from_buffer(result);
                                        }
                                        _ => {
                                            // Counts/Normalized/Edges: vector output
                                            // Extract the 1D array as a vector
                                            current_output = NodeOutput::from_buffer(result);
                                        }
                                    }
                                }
                                ViewDto::ResizeScale {
                                    scale_x,
                                    scale_y,
                                    filter,
                                } => {
                                    // COST: Flush pending ops - may involve data copy if ops need execution
                                    // Flush to get current dimensions, then push concrete resize to pending ops
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeScale requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    // COST: Zero-copy - just reading shape metadata
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let new_height = (input_height * scale_y).round() as u32;
                                    let new_width = (input_width * scale_x).round() as u32;
                                    // Push concrete resize to pending ops - let batching handle execution
                                    // COST: No data copy here - just building op specification
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: new_width,
                                            height: new_height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::ResizeToHeight { height, filter } => {
                                    // COST: Flush pending ops - may involve data copy if ops need execution
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeToHeight requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    // COST: Zero-copy - just reading shape metadata
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let aspect = input_width / input_height;
                                    let new_width = (*height as f32 * aspect).round() as u32;
                                    // Push concrete resize to pending ops - let batching handle execution
                                    // COST: No data copy here - just building op specification
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: new_width,
                                            height: *height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::ResizeToWidth { width, filter } => {
                                    // COST: Flush pending ops - may involve data copy if ops need execution
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeToWidth requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    // COST: Zero-copy - just reading shape metadata
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let aspect = input_height / input_width;
                                    let new_height = (*width as f32 * aspect).round() as u32;
                                    // Push concrete resize to pending ops - let batching handle execution
                                    // COST: No data copy here - just building op specification
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: *width,
                                            height: new_height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::ResizeMax { max_size, filter } => {
                                    // COST: Flush pending ops - may involve data copy if ops need execution
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeMax requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    // COST: Zero-copy - just reading shape metadata
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let scale = *max_size as f32 / input_height.max(input_width);
                                    let new_height = (input_height * scale).round() as u32;
                                    let new_width = (input_width * scale).round() as u32;
                                    // Push concrete resize to pending ops - let batching handle execution
                                    // COST: No data copy here - just building op specification
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: new_width,
                                            height: new_height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::ResizeMin { min_size, filter } => {
                                    // COST: Flush pending ops - may involve data copy if ops need execution
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeMin requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    // COST: Zero-copy - just reading shape metadata
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let scale = *min_size as f32 / input_height.min(input_width);
                                    let new_height = (input_height * scale).round() as u32;
                                    let new_width = (input_width * scale).round() as u32;
                                    // Push concrete resize to pending ops - let batching handle execution
                                    // COST: No data copy here - just building op specification
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: new_width,
                                            height: new_height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::Pad {
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    value,
                                    mode,
                                } => {
                                    // COST: Flush may execute pending ops (full data copy if ops pending)
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Pad requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    // COST: Full data copy - O(H*W*C) - pad_buffer allocates new buffer
                                    // TODO: Add ImageOpKind::Pad to view-buffer for batching support
                                    let result = pad_buffer(
                                        current_buf,
                                        *top,
                                        *bottom,
                                        *left,
                                        *right,
                                        *value,
                                        *mode,
                                    );
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::PadToSize {
                                    height,
                                    width,
                                    position,
                                    value,
                                } => {
                                    use view_buffer::ops::dto::{PadMode, PadPosition};
                                    // COST: Flush may execute pending ops (full data copy if ops pending)
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "PadToSize requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    // COST: Zero-copy - just reading shape metadata
                                    let shape = current_buf.shape();
                                    let current_h = shape[0] as u32;
                                    let current_w = shape[1] as u32;

                                    // Compute padding amounts based on position
                                    let pad_h = height.saturating_sub(current_h);
                                    let pad_w = width.saturating_sub(current_w);

                                    let (top, bottom, left, right) = match position {
                                        PadPosition::Center => {
                                            let t = pad_h / 2;
                                            let b = pad_h - t;
                                            let l = pad_w / 2;
                                            let r = pad_w - l;
                                            (t, b, l, r)
                                        }
                                        PadPosition::TopLeft => (0, pad_h, 0, pad_w),
                                        PadPosition::BottomRight => (pad_h, 0, pad_w, 0),
                                    };

                                    // COST: Full data copy - O(H*W*C) - pad_buffer allocates new buffer
                                    let result = pad_buffer(
                                        current_buf,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        *value,
                                        PadMode::Constant,
                                    );
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::Letterbox {
                                    height,
                                    width,
                                    value,
                                } => {
                                    use view_buffer::ops::dto::PadMode;
                                    use view_buffer::FilterType;
                                    // COST: Flush may execute pending ops (full data copy if ops pending)
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Letterbox requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    // COST: Zero-copy - just reading shape metadata
                                    let shape = current_buf.shape();
                                    let input_h = shape[0] as f32;
                                    let input_w = shape[1] as f32;

                                    // Calculate scale to fit within target
                                    let scale_h = *height as f32 / input_h;
                                    let scale_w = *width as f32 / input_w;
                                    let scale = scale_h.min(scale_w);

                                    let resized_h = (input_h * scale).round() as u32;
                                    let resized_w = (input_w * scale).round() as u32;

                                    // Push resize to pending ops for batching
                                    // COST: No data copy here - just building op specification
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: resized_w,
                                            height: resized_h,
                                            filter: FilterType::Lanczos3,
                                        },
                                    }));

                                    // COST: Flush executes resize (full data copy for resize output)
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let resized = current_output.as_buffer().ok_or_else(|| {
                                        "Letterbox: resized buffer expected".to_string()
                                    })?;

                                    // Then pad to exact size
                                    let pad_h = height.saturating_sub(resized_h);
                                    let pad_w = width.saturating_sub(resized_w);
                                    let top = pad_h / 2;
                                    let bottom = pad_h - top;
                                    let left = pad_w / 2;
                                    let right = pad_w - left;

                                    // COST: Full data copy - O(H*W*C) - pad_buffer allocates new buffer
                                    let result = pad_buffer(
                                        resized,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        *value,
                                        PadMode::Constant,
                                    );
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                _ => {
                                    // Regular buffer operation: accumulate for batching
                                    pending_buffer_ops.push(view_dto.clone());
                                }
                            }
                        }

                        // Flush any remaining pending ops
                        current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;

                        node_outputs.insert(node_id.clone(), current_output);
                    }
                }

                // Encode each output based on its domain and sink format
                for (alias, spec) in &self.outputs {
                    if let Some(output) = node_outputs.get(&spec.node) {
                        match encode_node_output(output, &spec.sink) {
                            Ok(encoded) => {
                                let row_result = match encoded {
                                    OutputValue::Binary(bytes) => RowResult::Binary(Some(bytes)),
                                    OutputValue::Scalar(val) => RowResult::Scalar(Some(val)),
                                    OutputValue::Vector(vals) => {
                                        RowResult::Vector(Some((*vals).clone()))
                                    }
                                    OutputValue::Contours(contours) => {
                                        RowResult::Contours(Some((*contours).clone()))
                                    }
                                    OutputValue::TypedList { data, shape } => {
                                        RowResult::TypedList(Some((data, shape)))
                                    }
                                    OutputValue::TypedArray { data, shape } => {
                                        RowResult::TypedArray(Some((data, shape)))
                                    }
                                    OutputValue::NumpyStruct(buf) => {
                                        RowResult::NumpyStruct(Some(buf))
                                    }
                                };
                                results.get_mut(alias).unwrap().push(row_result);
                            }
                            Err(e) => return Err(format!("Encode error for '{alias}': {e}")),
                        }
                    } else {
                        // No output for this node - push null with correct type based on OutputSpec
                        let null_result = null_row_result_for_spec(spec);
                        results.get_mut(alias).unwrap().push(null_result);
                    }
                }
            }

            Ok(results)
        }));

        // Handle batch result
        let results = match batch_result {
            Ok(Ok(r)) => r,
            Ok(Err(msg)) => {
                return Err(polars_err!(ComputeError: "Pipeline execution failed: {}", msg));
            }
            Err(panic_payload) => {
                // Extract panic message
                let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic during batch execution".to_string()
                };
                return Err(polars_err!(ComputeError: "Pipeline batch failed: {}", panic_msg));
            }
        };

        // Build output based on single vs multi output
        if self.is_single_output() {
            // Single output: use OutputSpec to determine type (not data inspection)
            let spec = self.outputs.get("_output").unwrap();
            let data = results.get("_output").unwrap();

            // Use static type information from OutputSpec
            build_series_from_spec(inputs[0].name().clone(), spec, data)
        } else {
            // Multi output: return Struct column with appropriate field types
            // Use OutputSpec to determine type for each field (not data inspection)
            let mut fields: Vec<Series> = Vec::with_capacity(output_aliases.len());

            for alias in &output_aliases {
                let spec = self.outputs.get(*alias).unwrap();
                let data = results.get(*alias).unwrap();

                let field_series = build_series_from_spec(PlSmallStr::from_str(alias), spec, data)?;

                fields.push(field_series);
            }

            let output_name = inputs[0].name().clone();
            StructChunked::from_series(output_name, len, fields.iter()).map(|sc| sc.into_series())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_unified_single_output() {
        let json = r#"{
            "nodes": {
                "_node_0": {
                    "source": {"format": "image_bytes"},
                    "ops": []
                }
            },
            "outputs": {
                "_output": {"node": "_node_0", "sink": {"format": "numpy"}}
            },
            "column_bindings": {"_node_0": 0}
        }"#;

        let graph = UnifiedGraph::from_json(json).unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.is_single_output());
        assert!(graph.outputs.contains_key("_output"));
    }

    #[test]
    fn test_parse_unified_multi_output() {
        let json = r#"{
            "nodes": {
                "_node_0": {
                    "source": {"format": "image_bytes"},
                    "ops": [],
                    "alias": "original"
                },
                "_node_1": {
                    "source": {"format": "blob"},
                    "ops": [],
                    "upstream": ["_node_0"],
                    "alias": "processed"
                }
            },
            "outputs": {
                "original": {"node": "_node_0", "sink": {"format": "png"}},
                "processed": {"node": "_node_1", "sink": {"format": "numpy"}}
            },
            "column_bindings": {"_node_0": 0}
        }"#;

        let graph = UnifiedGraph::from_json(json).unwrap();
        assert_eq!(graph.nodes.len(), 2);
        assert!(!graph.is_single_output());
        assert!(graph.outputs.contains_key("original"));
        assert!(graph.outputs.contains_key("processed"));
    }

    #[test]
    fn test_unified_topological_order() {
        let json = r#"{
            "nodes": {
                "a": {"source": {"format": "image_bytes"}, "ops": [], "alias": "out_a"},
                "b": {"source": {"format": "blob"}, "ops": [], "upstream": ["a"], "alias": "out_b"}
            },
            "outputs": {
                "out_a": {"node": "a", "sink": {"format": "numpy"}},
                "out_b": {"node": "b", "sink": {"format": "png"}}
            },
            "column_bindings": {"a": 0}
        }"#;

        let graph = UnifiedGraph::from_json(json).unwrap();
        // The order is now cached during from_json, access via private method
        let order = graph.topological_order();

        assert!(order.contains(&"a".to_string()));
        assert!(order.contains(&"b".to_string()));

        let b_pos = order.iter().position(|x| x == "b").unwrap();
        let a_pos = order.iter().position(|x| x == "a").unwrap();
        assert!(b_pos > a_pos);
    }

    #[test]
    fn test_output_node_ids() {
        let json = r#"{
            "nodes": {
                "a": {"source": {"format": "image_bytes"}, "ops": []},
                "b": {"source": {"format": "image_bytes"}, "ops": []}
            },
            "outputs": {
                "out1": {"node": "a", "sink": {"format": "numpy"}},
                "out2": {"node": "b", "sink": {"format": "png"}}
            },
            "column_bindings": {"a": 0, "b": 1}
        }"#;

        let graph = UnifiedGraph::from_json(json).unwrap();
        let output_ids = graph.output_node_ids();

        assert_eq!(output_ids.len(), 2);
        assert!(output_ids.contains("a"));
        assert!(output_ids.contains("b"));
    }
}
