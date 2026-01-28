//! Source decoding and series building utilities.
//!
//! This module contains functions for:
//! - Decoding binary sources (blob, raw, zero-copy)
//! - Decoding list/array sources from Polars
//! - Building output series from row results
//! - Padding and masking operations

use polars::prelude::*;
use view_buffer::{BinaryOp, ViewBuffer};

use super::encode::{
    build_typed_array_series_from_rows_with_dtype, build_typed_list_series_from_rows_with_dtype,
    contours_to_polars_value, TypedListRow,
};
use super::types::{OutputSpec, RowResult, TypedBufferData};

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
pub(crate) fn get_binary_row_buffer(
    binary_ca: &BinaryChunked,
    row_idx: usize,
) -> Option<(polars_arrow::buffer::Buffer<u8>, usize, usize)> {
    if is_row_null(&binary_ca.clone().into_series(), row_idx) {
        return None;
    }
    let bytes = binary_ca.get(row_idx)?;
    let len = bytes.len();
    let buffer = polars_arrow::buffer::Buffer::from(bytes.to_vec());
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
pub(crate) fn decode_binary_zero_copy(
    buffer: polars_arrow::buffer::Buffer<u8>,
    offset: usize,
    len: usize,
    source_format: &str,
    dtype_str: Option<&str>,
) -> Result<ViewBuffer, String> {
    match source_format {
        "blob" => {
            let slice_data: Vec<u8> = buffer.as_slice()[offset..offset + len].to_vec();
            decode_blob_zero_copy(buffer, offset, len, &slice_data)
        }
        "raw" => {
            let dtype_s = dtype_str.ok_or("Raw source format requires dtype")?;
            let dtype = parse_dtype_str(dtype_s)?;
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
    let num_elements: usize = shape.iter().product();
    let expected_data_len = num_elements * dtype.size_of();
    if data_offset + expected_data_len > total_len {
        return Err(
            format!(
                "Blob data truncated: offset={data_offset}, expected={expected_data_len}, total={total_len}"
            ),
        );
    }
    Ok(ViewBuffer::from_polars_buffer_slice(
        buffer,
        base_offset + data_offset,
        expected_data_len,
        shape,
        dtype,
    ))
}
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
        DataType::Binary => Some(view_buffer::DType::U8),
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
/// Decode a Polars List or Array value at a specific row into a ViewBuffer.
///
/// Uses zero-copy when the data is contiguous (FixedSizeList/Array types),
/// falling back to copy-based flattening for jagged List types.
///
/// If `dtype_str` is provided, it will be used. Otherwise, the dtype will be
/// inferred from the Polars column type.
///
/// If `require_contiguous` is true and zero-copy is not possible, an error is returned.
pub(crate) fn decode_list_or_array_source(
    series: &Series,
    row_idx: usize,
    dtype_str: Option<&str>,
    require_contiguous: bool,
) -> Result<Option<ViewBuffer>, String> {
    let dtype = if let Some(dtype_s) = dtype_str {
        parse_dtype_str(dtype_s)?
    } else {
        dtype_from_polars_datatype(series.dtype()).ok_or_else(|| {
            format!(
                "Cannot infer dtype from Polars type {:?}. Please specify dtype explicitly.",
                series.dtype()
            )
        })?
    };
    if let Some(result) = try_decode_array_zero_copy(series, row_idx, dtype)? {
        return Ok(Some(result));
    }
    if require_contiguous {
        return Err(format!(
            "Source 'require_contiguous=true' requires rectangular data with zero-copy access, \
            but row {row_idx} has data that cannot be zero-copied (possibly jagged nested lists or \
            variable-size List type). Use require_contiguous=false to allow copy-based flattening, \
            or use Polars Array type (fixed-size) instead of List."
        ));
    }
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
    if let DataType::Array(inner_dtype, _width) = series.dtype() {
        let shape = extract_fixed_shape_from_dtype(series.dtype());
        if shape.is_empty() {
            return Ok(None);
        }
        if !is_primitive_dtype(get_innermost_dtype(inner_dtype)) {
            return Ok(None);
        }
        let arr_ca = series
            .array()
            .map_err(|e| format!("Array access error: {e}"))?;
        if is_row_null(&arr_ca.clone().into_series(), row_idx) {
            return Ok(None);
        }
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
    let mut cumulative_len = 0;
    for chunk in arr_ca.downcast_iter() {
        let chunk_len = chunk.len();
        if row_idx < cumulative_len + chunk_len {
            let local_idx = row_idx - cumulative_len;
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
    let (primitive_values, elements_per_row) = get_primitive_values(values.as_ref(), size)?;
    let element_size = dtype.size_of();
    let offset = local_idx * elements_per_row * element_size;
    let len = elements_per_row * element_size;
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
                let values = arr.values();
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
        None => return Ok(None),
    };
    let (shape, flat_series) = flatten_nested_series(&element)?;
    if flat_series.is_empty() {
        return Ok(None);
    }
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
    let shape = infer_nested_shape(series)?;
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
                if len > 0 {
                    if let Some(first) = list_ca.get_as_series(0) {
                        current = first;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            DataType::Array(_, _width) => {
                let len = current.len();
                shape.push(len);
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
        _ => DataType::Float64,
    }
}
/// Get the Polars DataType for a given output specification.
///
/// Returns the appropriate dtype based on domain, sink format, and expected dtype.
pub(crate) fn dtype_for_output(spec: &OutputSpec) -> PolarsResult<DataType> {
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();
    match (domain, format) {
        ("buffer", "numpy" | "torch") => Ok(crate::output::numpy_output_dtype()),
        ("buffer", "png" | "jpeg" | "webp" | "blob") => Ok(DataType::Binary),
        ("buffer", "list") => {
            let inner = dtype_str_to_polars(&spec.expected_dtype);
            if let Some(ref shape) = spec.expected_shape {
                let mut dtype = inner;
                for _ in 0..shape.len() {
                    dtype = DataType::List(Box::new(dtype));
                }
                Ok(dtype)
            } else if let Some(ndim) = spec.expected_ndim {
                let mut dtype = inner;
                for _ in 0..ndim {
                    dtype = DataType::List(Box::new(dtype));
                }
                Ok(dtype)
            } else {
                Ok(DataType::List(Box::new(inner)))
            }
        }
        ("buffer", "array") => {
            let inner = dtype_str_to_polars(&spec.expected_dtype);
            let shape = spec.sink.shape.as_ref().or(spec.expected_shape.as_ref());
            if let Some(shape) = shape {
                let mut dtype = inner;
                for &dim in shape.iter().rev() {
                    dtype = DataType::Array(Box::new(dtype), dim);
                }
                Ok(dtype)
            } else {
                polars_bail!(ComputeError:
                    "array sink requires a known shape at planning time. \
                     Provide shape via .sink(shape=[...]) or use .resize()/.assert_shape() \
                     so the planner can determine output dimensions."
                );
            }
        }
        ("scalar", "native") => Ok(DataType::Float64),
        ("vector", "native" | "list") => {
            let inner = dtype_str_to_polars(&spec.expected_dtype);
            if let Some(ref shape) = spec.expected_shape {
                let mut dtype = inner;
                for _ in 0..shape.len() {
                    dtype = DataType::List(Box::new(dtype));
                }
                Ok(dtype)
            } else if let Some(ndim) = spec.expected_ndim {
                let mut dtype = inner;
                for _ in 0..ndim {
                    dtype = DataType::List(Box::new(dtype));
                }
                Ok(dtype)
            } else {
                Ok(DataType::List(Box::new(inner)))
            }
        }
        ("contour", "native") => {
            let point_dtype = DataType::Struct(vec![
                Field::new("x".into(), DataType::Float64),
                Field::new("y".into(), DataType::Float64),
            ]);
            Ok(DataType::Struct(vec![
                Field::new(
                    "exterior".into(),
                    DataType::List(Box::new(point_dtype.clone())),
                ),
                Field::new("interiors".into(), DataType::Null),
            ]))
        }
        _ => Ok(DataType::Binary),
    }
}
/// Create a null RowResult with the correct type based on OutputSpec.
///
/// This ensures that null values are pushed with the appropriate type variant,
/// allowing the series builder to use static type information.
pub(crate) fn null_row_result_for_spec(spec: &OutputSpec) -> RowResult {
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();
    match (domain, format) {
        ("buffer", "numpy" | "torch") => RowResult::NumpyStruct(None),
        ("buffer", "png" | "jpeg" | "webp" | "blob") | (_, "binary") => RowResult::Binary(None),
        ("buffer", "list") | ("vector", "native" | "list") => RowResult::TypedList(None),
        ("buffer", "array") => RowResult::TypedArray(None),
        ("scalar", "native") => RowResult::Scalar(None),
        ("contour", "native") => RowResult::Contours(None),
        _ => RowResult::Binary(None),
    }
}
/// Build a series from row results using the OutputSpec to determine the type.
///
/// This function uses static type information from the OutputSpec rather than
/// inspecting the first row's data. This allows proper handling of null values
/// while preserving the expected output type.
pub(crate) fn build_series_from_spec(
    name: PlSmallStr,
    spec: &OutputSpec,
    data: &[RowResult],
) -> PolarsResult<Series> {
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();
    let dtype = &spec.expected_dtype;
    match (domain, format) {
        ("buffer", "numpy" | "torch") => {
            let buffers: Vec<Option<ViewBuffer>> = data
                .iter()
                .map(|r| match r {
                    RowResult::NumpyStruct(opt) => opt.clone(),
                    _ => None,
                })
                .collect();
            crate::output::build_numpy_series(name, buffers)
        }
        ("buffer", "png" | "jpeg" | "webp" | "blob") | (_, "binary") => {
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
            build_typed_list_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                spec.expected_shape.as_ref(),
                spec.expected_ndim,
            )
        }
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
            build_typed_array_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                &spec.sink.shape,
                spec.expected_shape.as_ref(),
            )
        }
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
        ("vector", "native" | "list") => {
            let rows: Vec<TypedListRow> = data
                .iter()
                .map(|r| match r {
                    RowResult::TypedList(Some((typed_data, shape))) => {
                        Some((typed_data.clone(), shape.clone()))
                    }
                    RowResult::Vector(Some(vals)) => {
                        Some((TypedBufferData::F64(vals.clone()), vec![vals.len()]))
                    }
                    _ => None,
                })
                .collect();
            build_typed_list_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                spec.expected_shape.as_ref(),
                spec.expected_ndim,
            )
        }
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
            build_typed_array_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                &spec.sink.shape,
                spec.expected_shape.as_ref(),
            )
        }
        ("contour", "native") => {
            let values: PolarsResult<Vec<AnyValue<'static>>> = data
                .iter()
                .map(|r| match r {
                    RowResult::Contours(Some(contours)) => contours_to_polars_value(contours),
                    _ => Ok(AnyValue::Null),
                })
                .collect();
            let values = values?;
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
pub(crate) fn pad_buffer(
    buffer: &ViewBuffer,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    value: f32,
    mode: view_buffer::ops::dto::PadMode,
) -> ViewBuffer {
    use view_buffer::ops::dto::PadMode;
    match mode {
        PadMode::Constant => pad_constant(buffer, top, bottom, left, right, value),
        PadMode::Edge => pad_edge(buffer, top, bottom, left, right),
        PadMode::Reflect | PadMode::Symmetric => {
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
    let input_row_stride = input_w * channels;
    let output_row_stride = output_w * channels;
    match buffer.dtype() {
        DType::U8 => {
            let fill_val = value.clamp(0.0, 255.0) as u8;
            let mut output = vec![fill_val; output_h * output_w * channels];
            let contig = buffer.to_contiguous();
            let input = contig.as_slice::<u8>();
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
            let mut output = vec![fill_val; output_h * output_w * channels];
            let contig = buffer.to_contiguous();
            let input = contig.as_slice::<f32>();
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
            let contig = buffer.to_contiguous();
            let input = contig.as_slice::<u8>();
            let fill_val = value.clamp(0.0, 255.0) as u8;
            let mut output = vec![fill_val; output_h * output_w * channels];
            for y in 0..input_h {
                let src_start = y * input_row_stride;
                let src_end = src_start + input_row_stride;
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
    let top_usize = top as usize;
    let left_usize = left as usize;
    let input_row_stride = input_w * channels;
    let output_row_stride = output_w * channels;
    match buffer.dtype() {
        DType::U8 => {
            let mut output = vec![0u8; output_h * output_w * channels];
            let contig = buffer.to_contiguous();
            let input = contig.as_slice::<u8>();
            let replicate_pixel =
                |output: &mut [u8], dst_start: usize, count: usize, src_pixel: &[u8]| {
                    for i in 0..count {
                        let dst_idx = dst_start + i * channels;
                        output[dst_idx..dst_idx + channels].copy_from_slice(src_pixel);
                    }
                };
            for dst_y in 0..output_h {
                let src_y = if dst_y < top_usize {
                    0
                } else if dst_y >= top_usize + input_h {
                    input_h - 1
                } else {
                    dst_y - top_usize
                };
                let src_row_start = src_y * input_row_stride;
                let dst_row_start = dst_y * output_row_stride;
                if left_usize > 0 {
                    let first_pixel = &input[src_row_start..src_row_start + channels];
                    replicate_pixel(&mut output, dst_row_start, left_usize, first_pixel);
                }
                let src_end = src_row_start + input_row_stride;
                let dst_interior_start = dst_row_start + left_usize * channels;
                let dst_interior_end = dst_interior_start + input_row_stride;
                output[dst_interior_start..dst_interior_end]
                    .copy_from_slice(&input[src_row_start..src_end]);
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
        _ => pad_constant(buffer, top, bottom, left, right, 0.0),
    }
}
/// - 255 values zero out the pixel
///
/// Uses normalized blending: pixel * (mask / 255)
pub(crate) fn apply_mask(buffer: &ViewBuffer, mask: &ViewBuffer, invert: bool) -> ViewBuffer {
    let buf_shape = buffer.shape();
    let mask_shape = mask.shape();
    let effective_mask = if mask_shape.len() == 2 && buf_shape.len() == 3 {
        let h = mask_shape[0];
        let w = mask_shape[1];
        let c = buf_shape[2];
        let mask_contig = mask.to_contiguous();
        let mask_data = mask_contig.as_slice::<u8>();
        let mut expanded: Vec<u8> = Vec::with_capacity(h * w * c);
        for y in 0..h {
            for x in 0..w {
                let raw_val = mask_data[y * w + x];
                let mask_val = if invert { 255 - raw_val } else { raw_val };
                for _ in 0..c {
                    expanded.push(mask_val);
                }
            }
        }
        ViewBuffer::from_vec_with_shape(expanded, vec![h, w, c])
    } else if invert {
        let mask_contig = mask.to_contiguous();
        let mask_data = mask_contig.as_slice::<u8>();
        let inverted: Vec<u8> = mask_data.iter().map(|&v| 255 - v).collect();
        ViewBuffer::from_vec_with_shape(inverted, mask_shape.to_vec())
    } else {
        mask.clone()
    };
    BinaryOp::Blend.execute(buffer, &effective_mask)
}
