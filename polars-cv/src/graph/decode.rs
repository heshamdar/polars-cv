//! Source decoding and series building utilities.
//!
//! This module contains functions for:
//! - Decoding binary sources (blob, raw, zero-copy)
//! - Decoding list/array sources from Polars
//! - Building output series from row results
//! - Padding and masking operations

use polars::prelude::*;
use view_buffer::ViewBuffer;

use super::encode::{
    build_typed_array_series_from_rows_with_dtype, build_typed_list_series_from_rows_with_dtype,
    contour_struct_dtype, contours_to_polars_value, histogram_buckets_to_polars_value,
    histogram_struct_dtype, TypedListRow,
};
use super::types::{OutputSpec, RowResult, TypedBufferData};

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
) -> Option<(polars_buffer::Buffer<u8>, usize, usize)> {
    // `get` returns `None` for null (or out-of-bounds) rows, so it doubles as the
    // null check — no need to materialise a validity mask for the whole column.
    let bytes = binary_ca.get(row_idx)?;
    let len = bytes.len();
    let buffer = polars_buffer::Buffer::from(bytes.to_vec());
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
    buffer: polars_buffer::Buffer<u8>,
    offset: usize,
    len: usize,
    source_format: &str,
    dtype_str: Option<&str>,
) -> Result<ViewBuffer, String> {
    match source_format {
        "blob" => decode_blob_zero_copy(buffer, offset, len),
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
/// Parses the header from the slice (including shape and stride arrays),
/// then creates a ViewBuffer pointing directly into the data portion of
/// the original buffer. If the blob has a non-contiguous layout (indicated
/// by the flags field), the stored strides are preserved.
fn decode_blob_zero_copy(
    buffer: polars_buffer::Buffer<u8>,
    base_offset: usize,
    total_len: usize,
) -> Result<ViewBuffer, String> {
    use view_buffer::protocol::{u8_to_dtype, HEADER_SIZE, MAGIC_BYTES, VERSION};
    if total_len < HEADER_SIZE {
        return Err("Blob data too short for header".into());
    }
    // Read the header directly from the shared buffer — no copy needed, the
    // final ViewBuffer references the same `buffer` for its data.
    let slice = &buffer.as_slice()[base_offset..base_offset + total_len];
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
    // Read flags field (bytes 16..24): 1 = contiguous
    let flags = u64::from_le_bytes(slice[16..24].try_into().unwrap());
    let dtype =
        u8_to_dtype(dtype_code).ok_or_else(|| format!("Unknown dtype code: {dtype_code}"))?;

    // Read shape array
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

    // Read stride array (follows shape)
    let stride_start = shape_start + rank * 8;
    let mut strides = Vec::with_capacity(rank);
    for i in 0..rank {
        let pos = stride_start + i * 8;
        if pos + 8 > total_len {
            return Err("Blob truncated reading strides".into());
        }
        let s = i64::from_le_bytes(slice[pos..pos + 8].try_into().unwrap()) as isize;
        strides.push(s);
    }

    let num_elements: usize = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "Shape product overflow: dimensions too large".to_string())?;
    let expected_data_len = num_elements
        .checked_mul(dtype.size_of())
        .ok_or_else(|| "Data length overflow: buffer too large".to_string())?;
    // Every header field is untrusted input: the additions must be checked,
    // or a near-usize::MAX data_offset wraps below total_len and defeats the
    // truncation check (panicking in debug, mis-slicing in release).
    let data_end = data_offset
        .checked_add(expected_data_len)
        .ok_or_else(|| "Blob data offset overflow".to_string())?;
    if data_end > total_len {
        return Err(
            format!(
                "Blob data truncated: offset={data_offset}, expected={expected_data_len}, total={total_len}"
            ),
        );
    }
    let abs_data_offset = base_offset
        .checked_add(data_offset)
        .ok_or_else(|| "Blob data offset overflow".to_string())?;

    // If flags indicate contiguous (1) or strides were not stored, use contiguous layout.
    // Otherwise preserve the stored strides for non-contiguous views.
    if flags == 1 || strides.is_empty() {
        Ok(ViewBuffer::from_polars_buffer_slice(
            buffer,
            abs_data_offset,
            expected_data_len,
            shape,
            dtype,
        ))
    } else {
        // Stored strides are untrusted too. The strided window is every
        // byte from data_offset to the end of the blob (a padded layout may
        // legitimately span more than num_elements * size), and every
        // element the (shape, strides) pair can address must fall inside it.
        let window_len = total_len - data_offset;
        if num_elements > 0 {
            let mut min_reach: i128 = 0;
            let mut max_reach: i128 = 0;
            for (&dim, &stride) in shape.iter().zip(strides.iter()) {
                let reach = (dim as i128 - 1) * stride as i128;
                if reach >= 0 {
                    max_reach += reach;
                } else {
                    min_reach += reach;
                }
            }
            if min_reach < 0 {
                return Err(format!(
                    "Blob strides reach below the data start: shape={shape:?}, strides={strides:?}"
                ));
            }
            let span_end = max_reach + dtype.size_of() as i128;
            if span_end > window_len as i128 {
                return Err(format!(
                    "Blob strides reach outside the data: shape={shape:?}, \
                     strides={strides:?}, span={span_end}, available={window_len}"
                ));
            }
        }
        Ok(ViewBuffer::from_polars_buffer_slice_with_strides(
            buffer,
            abs_data_offset,
            window_len,
            shape,
            strides,
            dtype,
        ))
    }
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
///
/// The names come from `dtype_table!` via `from_short_name`; this wrapper adds
/// the graph layer's error string.
pub(super) fn parse_dtype_str(dtype_str: &str) -> Result<view_buffer::DType, String> {
    view_buffer::DType::from_short_name(dtype_str)
        .ok_or_else(|| format!("Unknown dtype: {dtype_str}"))
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
        if is_array_row_null(arr_ca, row_idx) {
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
/// Check whether the given row of an `ArrayChunked` is null.
///
/// Walks the chunks (cheap) and queries the arrow validity bitmap directly,
/// avoiding the per-row full-column clone that `is_row_null` would incur.
fn is_array_row_null(arr_ca: &ArrayChunked, row_idx: usize) -> bool {
    use polars_arrow::array::Array;
    let mut cumulative_len = 0;
    for chunk in arr_ca.downcast_iter() {
        let chunk_len = chunk.len();
        if row_idx < cumulative_len + chunk_len {
            return chunk.is_null(row_idx - cumulative_len);
        }
        cumulative_len += chunk_len;
    }
    true
}
fn get_array_row_buffer(
    arr_ca: &ArrayChunked,
    row_idx: usize,
    dtype: view_buffer::DType,
) -> Option<(polars_buffer::Buffer<u8>, usize, usize)> {
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
) -> Option<(polars_buffer::Buffer<u8>, usize, usize)> {
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
) -> Option<polars_buffer::Buffer<u8>> {
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
                return Some(polars_buffer::Buffer::from(u8_slice.to_vec()));
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
            .explode(ExplodeOptions {
                empty_as_null: false,
                keep_nulls: true,
            })
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
        // Reachable only for "auto", and only where the value is unused.
        // "auto" means the dtype was not resolved at planning time;
        // `list_array_inner_dtype` bails on it during schema resolution, which
        // Polars runs before execution, so the typed list/array builders in
        // `encode.rs` — the callers that *would* be misled, since they use this
        // to type the output Series — never see it. The remaining callers
        // (binary/struct sinks) ignore the value.
        //
        // Any *other* unmatched string would be silently typed UInt8 here,
        // which is why `test_no_second_dtype_spelling_table` requires every
        // dtype dispatch to name exactly the ten `dtype_table!` declares: the
        // only way to reach this arm with a real dtype is for a table to drift.
        _ => DataType::UInt8,
    }
}

/// Resolve the inner element dtype for a typed list/array sink.
///
/// Refuses the unresolved `"auto"` sentinel: it means the decoded dtype was
/// never pinned down at planning time. The Python sink builder rejects this for
/// list/array sinks up front (requiring an explicit dtype), so reaching here
/// with `"auto"` is an internal error — fail loudly rather than silently
/// materialize a `u8` column that may disagree with execution.
fn list_array_inner_dtype(dtype: &str, sink: &str) -> PolarsResult<DataType> {
    if dtype == "auto" {
        polars_bail!(ComputeError:
            "internal error: '{sink}' sink reached schema resolution with an \
             unresolved 'auto' element dtype. Supply an explicit dtype \
             (e.g. source(..., dtype=\"u16\") or .cast(...)) before the sink."
        );
    }
    Ok(dtype_str_to_polars(dtype))
}
/// Get the Polars DataType for a given output specification.
///
/// Returns the appropriate dtype based on domain, sink format, and expected dtype.
pub(crate) fn dtype_for_output(spec: &OutputSpec) -> PolarsResult<DataType> {
    // Encoding takes precedence over the (domain, format) pair: it selects a
    // distinct Polars schema for outputs that share a domain (e.g. histogram
    // buckets are a vector-domain output encoded as a struct list).
    if spec.expected_encoding.as_deref() == Some("histogram_buckets") {
        return Ok(DataType::List(Box::new(histogram_struct_dtype())));
    }
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();
    match (domain, format) {
        ("buffer", "numpy" | "torch") => Ok(crate::output::numpy_output_dtype()),
        ("buffer", "png" | "jpeg" | "webp" | "tiff" | "blob") => Ok(DataType::Binary),
        ("buffer", "list") => {
            let inner = list_array_inner_dtype(&spec.expected_dtype, "list")?;
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
            let inner = list_array_inner_dtype(&spec.expected_dtype, "array")?;
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
            // Reject an unresolved "auto" element dtype the same way the
            // buffer/list and array arms do, instead of silently mapping it to
            // U8 — a plan/data divergence if a vector output ever reached the
            // sink still "auto". (Today vector dtypes are always concrete.)
            let inner = list_array_inner_dtype(&spec.expected_dtype, "list")?;
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
        // Fixed-size vector outputs (e.g. perceptual hashes) as Array.
        // This pair used to ride the silent Binary fallthrough: execution
        // produced an Array while lazy schema claimed Binary.
        ("vector", "array") => {
            let inner = list_array_inner_dtype(&spec.expected_dtype, "array")?;
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
                     Provide shape via .sink(shape=[...])."
                );
            }
        }
        ("contour", "native") => Ok(DataType::List(Box::new(contour_struct_dtype()))),
        ("buffer", "native") => polars_bail!(ComputeError:
            "'native' sink is not defined for buffer outputs; use an explicit \
             format (numpy, png, list, array, blob, ...)"
        ),
        // Unknown pairs used to silently default to Binary, masking planner
        // bugs (a typo'd format/domain produced a Binary column of garbage).
        (domain, format) => polars_bail!(ComputeError:
            "Unsupported output combination: domain '{}' with sink format '{}'",
            domain, format
        ),
    }
}
/// Create a null RowResult with the correct type based on OutputSpec.
///
/// This ensures that null values are pushed with the appropriate type variant,
/// allowing the series builder to use static type information.
pub(crate) fn null_row_result_for_spec(spec: &OutputSpec) -> RowResult {
    if spec.expected_encoding.as_deref() == Some("histogram_buckets") {
        return RowResult::HistogramBuckets(None);
    }
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();
    match (domain, format) {
        ("buffer", "numpy" | "torch") => RowResult::NumpyStruct(None),
        ("buffer", "png" | "jpeg" | "webp" | "tiff" | "blob") | (_, "binary") => {
            RowResult::Binary(None)
        }
        ("buffer", "list") | ("vector", "native" | "list") => RowResult::TypedList(None),
        ("buffer" | "vector", "array") => RowResult::TypedArray(None),
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
    data: Vec<RowResult>,
) -> PolarsResult<Series> {
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();
    let dtype = &spec.expected_dtype;
    if spec.expected_encoding.as_deref() == Some("histogram_buckets") {
        let values: PolarsResult<Vec<AnyValue<'static>>> = data
            .iter()
            .map(|r| match r {
                RowResult::HistogramBuckets(Some(buckets)) => {
                    histogram_buckets_to_polars_value(buckets)
                }
                _ => Ok(AnyValue::Null),
            })
            .collect();
        let histogram_dtype = DataType::List(Box::new(histogram_struct_dtype()));
        return Series::from_any_values_and_dtype(name, &values?, &histogram_dtype, true);
    }
    match (domain, format) {
        ("buffer", "numpy" | "torch") => {
            // Move the buffers in so each is the sole Arc owner: that lets
            // `into_polars_buffer_strided` take the zero-copy *strided* branch
            // for non-contiguous (transposed/flipped/rotated) outputs. The
            // numpy/torch struct carries shape/strides/offset, and the Python
            // consumers (`numpy_from_struct`, the struct->PNG helper) honor them
            // via `np.lib.stride_tricks.as_strided`, so permuted layouts decode
            // correctly without materialising to contiguous here.
            let buffers: Vec<Option<ViewBuffer>> = data
                .into_iter()
                .map(|r| match r {
                    RowResult::NumpyStruct(opt) => opt,
                    _ => None,
                })
                .collect();
            crate::output::build_numpy_series(name, buffers, spec.sink.out_dtype.as_deref())
        }
        ("buffer", "png" | "jpeg" | "webp" | "tiff" | "blob") | (_, "binary") => {
            // Register each row's already-materialised bytes as a BinaryView
            // backing buffer instead of copying them into a builder — see
            // `crate::output::binary_view_series_from_rows`.
            Ok(crate::output::binary_view_series_from_rows(
                name,
                data.into_iter().map(|r| match r {
                    RowResult::Binary(b) => b,
                    _ => None,
                }),
            ))
        }
        ("buffer", "list") => {
            let rows: Vec<TypedListRow> = data
                .into_iter()
                .map(|r| match r {
                    RowResult::TypedList(Some((typed_data, shape))) => Some((typed_data, shape)),
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
                .into_iter()
                .map(|r| match r {
                    RowResult::TypedArray(Some((typed_data, shape))) => Some((typed_data, shape)),
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
                .into_iter()
                .map(|r| match r {
                    RowResult::Scalar(s) => s,
                    _ => None,
                })
                .collect();
            let output_ca = Float64Chunked::from_iter_options(name, scalar_data.into_iter());
            Ok(output_ca.into_series())
        }
        ("vector", "native" | "list") => {
            let rows: Vec<TypedListRow> = data
                .into_iter()
                .map(|r| match r {
                    RowResult::TypedList(Some((typed_data, shape))) => Some((typed_data, shape)),
                    RowResult::Vector(Some(vals)) => {
                        let len = vals.len();
                        Some((TypedBufferData::F64(vals), vec![len]))
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
                .into_iter()
                .map(|r| match r {
                    RowResult::TypedList(Some((typed_data, shape)))
                    | RowResult::TypedArray(Some((typed_data, shape))) => Some((typed_data, shape)),
                    RowResult::Vector(Some(vals)) => {
                        let len = vals.len();
                        Some((TypedBufferData::F64(vals), vec![len]))
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
            let contour_dtype = DataType::List(Box::new(contour_struct_dtype()));
            Series::from_any_values_and_dtype(name, &values, &contour_dtype, true)
        }
        _ => Ok(crate::output::binary_view_series_from_rows(
            name,
            data.into_iter().map(|r| match r {
                RowResult::Binary(b) => b,
                _ => None,
            }),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::decode_binary_zero_copy;
    use view_buffer::protocol::HEADER_SIZE;

    /// Build a VIEW-protocol blob byte-by-byte so malformed headers can be
    /// crafted (the writer API always emits valid contiguous blobs).
    fn craft_blob(
        dtype_code: u8,
        data_offset: u64,
        flags: u64,
        shape: &[u64],
        strides: &[i64],
        data: &[u8],
    ) -> Vec<u8> {
        let rank = shape.len();
        assert_eq!(rank, strides.len());
        let mut v = vec![0u8; HEADER_SIZE];
        v[0..4].copy_from_slice(b"VIEW");
        v[4..6].copy_from_slice(&1u16.to_le_bytes());
        v[6] = dtype_code;
        v[7] = rank as u8;
        v[8..16].copy_from_slice(&data_offset.to_le_bytes());
        v[16..24].copy_from_slice(&flags.to_le_bytes());
        for dim in shape {
            v.extend_from_slice(&dim.to_le_bytes());
        }
        for s in strides {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v.extend_from_slice(data);
        v
    }

    fn decode(blob: Vec<u8>) -> Result<view_buffer::ViewBuffer, String> {
        let len = blob.len();
        let buffer = polars_buffer::Buffer::from(blob);
        decode_binary_zero_copy(buffer, 0, len, "blob", None)
    }

    /// data_offset for a blob whose payload directly follows shape+strides.
    fn payload_offset(rank: usize) -> u64 {
        (HEADER_SIZE + rank * 16) as u64
    }

    #[test]
    fn valid_contiguous_blob_decodes() {
        let blob = craft_blob(
            1, // u8
            payload_offset(1),
            1, // contiguous
            &[4],
            &[1],
            &[10, 20, 30, 40],
        );
        let buf = decode(blob).expect("valid blob must decode");
        assert_eq!(buf.shape(), &[4]);
        assert_eq!(buf.to_contiguous().as_slice::<u8>(), &[10, 20, 30, 40]);
    }

    #[test]
    fn huge_data_offset_is_rejected_not_wrapped() {
        // data_offset near usize::MAX: the truncation check
        // `data_offset + expected_data_len > total_len` must not overflow
        // (wrap) into acceptance — it must return a clean error.
        let blob = craft_blob(1, u64::MAX - 2, 1, &[4], &[1], &[10, 20, 30, 40]);
        let res = decode(blob);
        assert!(res.is_err(), "wrapping offset must be rejected: {res:?}");
    }

    #[test]
    fn data_offset_past_end_is_rejected() {
        let blob = craft_blob(1, 10_000, 1, &[4], &[1], &[10, 20, 30, 40]);
        assert!(decode(blob).is_err());
    }

    #[test]
    fn hostile_strides_beyond_window_are_rejected() {
        // 4x4 u8 with a row stride pointing 1 MB past the payload: the
        // strided view would read far outside the blob.
        let data = [0u8; 16];
        let blob = craft_blob(1, payload_offset(2), 0, &[4, 4], &[1_000_000, 1], &data);
        assert!(decode(blob).is_err());
    }

    #[test]
    fn negative_stride_reach_below_window_is_rejected() {
        // A negative row stride from element 0 reaches below the payload
        // start.
        let data = [0u8; 16];
        let blob = craft_blob(1, payload_offset(2), 0, &[4, 4], &[-16, 1], &data);
        assert!(decode(blob).is_err());
    }

    #[test]
    fn valid_strided_blob_decodes() {
        // Column-major 2x2 u8: element (i, j) at byte i*1 + j*2.
        let blob = craft_blob(1, payload_offset(2), 0, &[2, 2], &[1, 2], &[10, 20, 30, 40]);
        let buf = decode(blob).expect("in-window strided blob must decode");
        assert_eq!(buf.shape(), &[2, 2]);
        assert_eq!(buf.to_contiguous().as_slice::<u8>(), &[10, 30, 20, 40]);
    }

    #[test]
    fn strided_span_larger_than_logical_size_is_accepted_when_in_window() {
        // A padded row layout: 2x2 u8 with row stride 3 over 7 bytes of
        // payload — spans more than the 4 logical bytes but stays inside
        // the blob.
        let blob = craft_blob(
            1,
            payload_offset(2),
            0,
            &[2, 2],
            &[3, 1],
            &[10, 20, 99, 30, 40, 99, 99],
        );
        let buf = decode(blob).expect("padded strided blob must decode");
        assert_eq!(buf.to_contiguous().as_slice::<u8>(), &[10, 20, 30, 40]);
    }
}
