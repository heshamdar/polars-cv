//! Source decoding and series building utilities.
//!
//! This module contains functions for:
//! - Decoding binary sources (blob, raw, zero-copy)
//! - Decoding list/array sources from Polars
//! - Building output series from row results
//! - Padding and masking operations

use polars::prelude::*;
use view_buffer::{ImageCodec, PlannedDType, ViewBuffer};

use super::encode::{
    build_typed_array_series_from_rows_with_dtype, build_typed_list_series_from_rows_with_dtype,
    contour_set_series, histogram_buckets_to_polars_value, histogram_struct_dtype, TypedListRow,
};
use super::sink_kind::SinkKind;
use super::types::{OutputSpec, RowResult, TypedBufferData};

/// Extract binary data from a BinaryChunked at a specific row.
///
/// Returns the row as a polars-arrow buffer. This is a copy: Polars stores
/// binary as a `BinaryViewArray`, whose short values live inline in the view
/// and whose long ones sit at arbitrary offsets in shared data buffers.
///
/// The copy is placed at an 8-byte-aligned address (the largest element
/// size). `parse_blob` validates a blob's offsets and strides *relative to
/// its first byte*, and the typed views built over it are `&[T]`, so the
/// blob's own start must be aligned for those checks to mean anything. A
/// `Vec<u8>` only promises alignment 1 (CR-41).
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
    Some((aligned_copy(bytes), 0, bytes.len()))
}

/// Copy `bytes` into a buffer whose first byte is 8-byte aligned.
///
/// Backed by a `Vec<u64>` so the alignment comes from the allocation's type,
/// not from allocator behaviour. The tail word is zero-padded; callers carry
/// the true length separately.
fn aligned_copy(bytes: &[u8]) -> polars_buffer::Buffer<u8> {
    let words: Vec<u64> = bytes
        .chunks(8)
        .map(|chunk| {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            u64::from_ne_bytes(word)
        })
        .collect();
    polars_buffer::Buffer::from(words)
        .try_transmute::<u8>()
        .expect("u64 -> u8 reinterpretation cannot fail")
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
            // A remainder is bytes the caller supplied that no element would
            // read; dropping them silently is a truncation, not a decode.
            if !len.is_multiple_of(element_size) {
                return Err(format!(
                    "Raw source: {len} bytes is not a multiple of the {dtype_s} element \
                     size ({element_size} bytes)"
                ));
            }
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
/// The header is parsed and validated by [`view_buffer::parse_blob`] — the
/// same parser `ViewBuffer::from_blob` uses — and the resulting ViewBuffer
/// points directly into `buffer`. A strided layout keeps its stored strides.
fn decode_blob_zero_copy(
    buffer: polars_buffer::Buffer<u8>,
    base_offset: usize,
    total_len: usize,
) -> Result<ViewBuffer, String> {
    let blob = view_buffer::parse_blob(&buffer.as_slice()[base_offset..base_offset + total_len])?;
    let abs_data_offset = base_offset
        .checked_add(blob.data_offset)
        .ok_or_else(|| "Blob data offset overflow".to_string())?;
    // `parse_blob` checked alignment relative to the blob's first byte; this
    // is the absolute address the typed views will actually use.
    let elem = blob.dtype.size_of();
    if !(buffer.as_slice().as_ptr() as usize + abs_data_offset).is_multiple_of(elem) {
        return Err(format!(
            "Blob payload is not aligned to its {:?} element size ({elem} bytes)",
            blob.dtype
        ));
    }
    Ok(match blob.strides {
        None => ViewBuffer::from_polars_buffer_slice(
            buffer,
            abs_data_offset,
            blob.data_len,
            blob.shape,
            blob.dtype,
        ),
        Some(strides) => ViewBuffer::from_polars_buffer_slice_with_strides(
            buffer,
            abs_data_offset,
            blob.data_len,
            blob.shape,
            strides,
            blob.dtype,
        ),
    })
}
/// The buffer element type a Polars *leaf* type holds, if it is one.
///
/// **The single Polars→DType mapping in this crate.** `resolved_output_specs`
/// (graph/compiled.rs) had a second copy of it that returned dtype *strings*,
/// so the two were free to disagree about which Polars types are buffer
/// elements at all — and did: that copy fell back to `"u8"` for everything
/// unmatched, which is how a `List(Decimal)` column came to claim it was a
/// buffer of bytes.
///
/// `None` means "not a buffer element type", which is a fact about the column,
/// not an error to paper over. Callers decide what to do with it.
pub(super) fn dtype_from_polars_leaf(dt: &DataType) -> Option<view_buffer::DType> {
    Some(match dt {
        DataType::UInt8 => view_buffer::DType::U8,
        DataType::Int8 => view_buffer::DType::I8,
        DataType::UInt16 => view_buffer::DType::U16,
        DataType::Int16 => view_buffer::DType::I16,
        DataType::UInt32 => view_buffer::DType::U32,
        DataType::Int32 => view_buffer::DType::I32,
        DataType::UInt64 => view_buffer::DType::U64,
        DataType::Int64 => view_buffer::DType::I64,
        DataType::Float32 => view_buffer::DType::F32,
        DataType::Float64 => view_buffer::DType::F64,
        _ => return None,
    })
}

/// Infer view-buffer DType from Polars DataType.
///
/// Recursively traverses nested List/Array types to find the innermost
/// primitive type, and treats a `Binary` column as the byte buffer it is.
/// The leaf mapping itself is [`dtype_from_polars_leaf`].
fn dtype_from_polars_datatype(dt: &DataType) -> Option<view_buffer::DType> {
    match dt {
        DataType::Binary => Some(view_buffer::DType::U8),
        DataType::List(inner) => dtype_from_polars_datatype(inner.as_ref()),
        DataType::Array(inner, _) => dtype_from_polars_datatype(inner.as_ref()),
        other => dtype_from_polars_leaf(other),
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
/// The primitive array's values as a byte buffer that *shares* its storage.
///
/// A reinterpretation, not a copy: `Buffer<T>` → `Buffer<u8>` keeps the same
/// allocation, so the caller's per-row window is a view into the column. It
/// used to copy the entire values buffer here — once per row — which made the
/// `array` source quadratic in the batch size (CR-40). Sharing is safe because
/// view-buffer never writes through `PolarsArrow` storage (its in-place paths
/// require uniquely-owned `Rust` storage).
///
/// `None` when the array is not a `PrimitiveArray` of exactly `dtype` (the
/// caller then takes the converting copy path).
fn get_primitive_buffer(
    array: &dyn polars_arrow::array::Array,
    dtype: view_buffer::DType,
) -> Option<polars_buffer::Buffer<u8>> {
    use polars_arrow::array::PrimitiveArray;
    macro_rules! view_bytes {
        ($type:ty) => {
            array
                .as_any()
                .downcast_ref::<PrimitiveArray<$type>>()
                .and_then(|arr| arr.values().clone().try_transmute::<u8>().ok())
        };
    }
    match dtype {
        view_buffer::DType::U8 => view_bytes!(u8),
        view_buffer::DType::I8 => view_bytes!(i8),
        view_buffer::DType::U16 => view_bytes!(u16),
        view_buffer::DType::I16 => view_bytes!(i16),
        view_buffer::DType::U32 => view_bytes!(u32),
        view_buffer::DType::I32 => view_bytes!(i32),
        view_buffer::DType::U64 => view_bytes!(u64),
        view_buffer::DType::I64 => view_bytes!(i64),
        view_buffer::DType::F32 => view_bytes!(f32),
        view_buffer::DType::F64 => view_bytes!(f64),
    }
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
/// The Polars spelling of an engine dtype.
///
/// This is the fourth spelling of a dtype (short name / VIEW wire code / numpy
/// name / Polars `DataType`). The first three are generated from
/// `dtype_table!`; this one cannot be, because `view-buffer` does not depend on
/// `polars` and keeping the engine Polars-agnostic is deliberate.
///
/// So it gets the next-best guard instead: the match is **exhaustive over
/// `DType`**, which means an eleventh dtype added to `dtype_table!` fails to
/// compile here rather than being silently typed. That replaced a
/// `_ => DataType::UInt8` catch-all whose own comment conceded "any other
/// unmatched string would be silently typed UInt8 here", guarded only by a
/// source-scanning ratchet — the weakest of the three guard kinds.
pub fn polars_dtype_for(dt: view_buffer::DType) -> DataType {
    match dt {
        view_buffer::DType::U8 => DataType::UInt8,
        view_buffer::DType::I8 => DataType::Int8,
        view_buffer::DType::U16 => DataType::UInt16,
        view_buffer::DType::I16 => DataType::Int16,
        view_buffer::DType::U32 => DataType::UInt32,
        view_buffer::DType::I32 => DataType::Int32,
        view_buffer::DType::U64 => DataType::UInt64,
        view_buffer::DType::I64 => DataType::Int64,
        view_buffer::DType::F32 => DataType::Float32,
        view_buffer::DType::F64 => DataType::Float64,
    }
}

/// Convert a dtype string to a Polars `DataType`.
///
/// Used for static type inference at planning time. The name is parsed through
/// `DType::from_short_name` — the `dtype_table!` authority — so an unresolved
/// sentinel (`"auto"`, `"auto_float"`) or a genuine typo is an **error**, not a
/// `u8` column that execution will contradict.
///
/// Note: requires the dtype-i8/dtype-u8/dtype-i16/dtype-u16 polars features for
/// the narrow integer Series types.
pub fn dtype_str_to_polars(dtype: &str) -> PolarsResult<DataType> {
    view_buffer::DType::from_short_name(dtype)
        .map(polars_dtype_for)
        .ok_or_else(|| {
            polars_err!(ComputeError:
                "cannot map dtype '{dtype}' to a Polars type. Expected one of \
                 the engine's concrete dtypes; '{dtype}' is either an \
                 unresolved planning sentinel (\"auto\"/\"auto_float\") that \
                 should have been resolved before this point, or not a dtype \
                 at all."
            )
        })
}

/// Resolve the inner element dtype for a typed list/array sink.
///
/// Refuses the unresolved `"auto"` sentinel: it means the decoded dtype was
/// never pinned down at planning time. The Python sink builder rejects this for
/// list/array sinks up front (requiring an explicit dtype), so reaching here
/// with `"auto"` is an internal error — fail loudly rather than silently
/// materialize a `u8` column that may disagree with execution.
fn list_array_inner_dtype(dtype: &str, sink: &str) -> PolarsResult<DataType> {
    // Asked of `PlannedDType`, not compared against `"auto"` by hand: there is
    // now more than one way to be unresolved (`"auto_float"` means "a float,
    // but which one depends on the decode"), and a hand-written comparison
    // would let the new one through to `dtype_str_to_polars`'s UInt8 arm.
    if !PlannedDType::parse(dtype).is_some_and(|d| d.is_concrete()) {
        // Not labelled an internal error: the common way to get here is a
        // source column whose element type the planner cannot map to a buffer
        // dtype (a boolean or decimal list), which is the user's input, not a
        // bug. The fix is the same either way — say what it is.
        polars_bail!(ComputeError:
            "the '{sink}' sink needs to know the element dtype at planning \
             time, and it could not be inferred from the input column. \
             Supply it explicitly, e.g. source(..., dtype=\"u16\") or \
             .cast(...) before the sink."
        );
    }
    dtype_str_to_polars(dtype)
}
/// Get the Polars DataType for a given output specification.
///
/// Returns the appropriate dtype based on domain, sink format, and expected dtype.
pub(crate) fn dtype_for_output(spec: &OutputSpec) -> PolarsResult<DataType> {
    match SinkKind::resolve(spec)? {
        SinkKind::HistogramBuckets => Ok(DataType::List(Box::new(histogram_struct_dtype()))),
        SinkKind::NumpyStruct => Ok(crate::output::numpy_output_dtype()),
        SinkKind::NdArray => Ok(crate::ext_types::ExtType::NdArray.dtype()),
        // A VIEW blob is self-describing, so it carries no codec precondition.
        SinkKind::Blob => Ok(DataType::Binary),
        kind @ SinkKind::EncodedImage => {
            // The codec's dtype/rank/channel precondition is decidable now, from
            // the same OutputSpec that produced this dtype — so decide it now.
            // Leaving it to the encoder meant a query planned as `Binary` and
            // then died part-way through `collect()`, which is precisely the
            // "planned one thing, executed another" failure the sink contract
            // exists to prevent. `ImageCodec::check_shape` is the one entry
            // point both halves read, and it treats an unknown as permission,
            // so a source whose dtype is still "auto" is not refused here.
            let format = kind
                .image_codec_format(spec)
                .expect("EncodedImage carries a codec format");
            let codec = ImageCodec::from_sink_format(format)
                .expect("this arm matches exactly the formats from_sink_format parses");
            let dtype = PlannedDType::parse(&spec.expected_dtype).unwrap_or(PlannedDType::Unknown);
            codec
                .check_shape(dtype, spec.expected_shape.as_deref(), spec.expected_ndim)
                .map_err(|msg| polars_err!(ComputeError: "{}", msg))?;
            Ok(DataType::Binary)
        }
        SinkKind::BufferList => {
            let inner = list_array_inner_dtype(&spec.expected_dtype, "list")?;
            let ndim = spec
                .expected_shape
                .as_ref()
                .map(|shape| shape.len())
                .or(spec.expected_ndim);
            // Not a fallback to depth 1: the nesting depth *is* the schema for
            // a list sink, and guessing it is how `source("auto")` on a Binary
            // column came to publish `List(u8)` for data that executes as
            // `List(List(List(u8)))`. `auto` is the one source whose rank
            // Python cannot know, and `resolved_output_specs` can only recover
            // it from a List/Array column — for image bytes the rank is not
            // settled until the decode. Refusing here is what turns that into
            // an error the user sees before any data moves.
            let Some(ndim) = ndim else {
                polars_bail!(ComputeError:
                    "the 'list' sink needs the output rank at planning time, and it \
                     is not knowable for this source. `source(\"auto\")` over a \
                     binary column decides between an image and a VIEW blob by \
                     inspecting the bytes, so the rank is only settled during \
                     execution. Name the source explicitly \
                     (e.g. `source(\"image_bytes\")`), or use a `numpy` sink, \
                     which does not encode the rank in its Polars dtype."
                );
            };
            let mut dtype = inner;
            for _ in 0..ndim {
                dtype = DataType::List(Box::new(dtype));
            }
            Ok(dtype)
        }
        SinkKind::BufferArray => {
            let inner = list_array_inner_dtype(&spec.expected_dtype, "array")?;
            let shape = spec.sink.shape.as_ref().or(spec.expected_shape.as_ref());
            if let Some(shape) = shape {
                let mut dtype = inner;
                for &dim in shape.iter().rev() {
                    dtype = DataType::Array(Box::new(dtype), dim);
                }
                Ok(dtype)
            } else {
                // Names what each remedy actually supplies. The advice this
                // replaces was circular for the source that reaches it most: a
                // list/array column's shape is not knowable until execution, so
                // it lands here — and was told to call `.assert_shape()`, which
                // published nothing without a rank, and `.resize()`, which never
                // supplies the channel count.
                polars_bail!(ComputeError:
                    "an 'array' sink needs the full output shape at planning time, and \
                     this pipeline's is not known. Three ways to supply it:\n  \
                     .sink('array', shape=[8, 8, 3])   — always works; the shape belongs \
                     to the sink\n  \
                     .assert_shape(dims=[8, 8, 3])     — when you know it and the source \
                     does not (a list/array column's shape is only settled during \
                     execution)\n  \
                     .resize(height=8, width=8)        — supplies height and width only"
                );
            }
        }
        SinkKind::Scalar => Ok(DataType::Float64),
        SinkKind::VectorList => {
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
        SinkKind::VectorArray => {
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
        SinkKind::Contours => Ok(DataType::List(Box::new(DataType::Struct(
            crate::geom_schema::contour_fields(),
        )))),
    }
}
/// Create a null RowResult with the correct type based on OutputSpec.
///
/// This ensures that null values are pushed with the appropriate type variant,
/// allowing the series builder to use static type information.
pub(crate) fn null_row_result_for_spec(spec: &OutputSpec) -> PolarsResult<RowResult> {
    let kind = SinkKind::resolve(spec)?;
    Ok(match kind {
        SinkKind::HistogramBuckets => RowResult::HistogramBuckets(None),
        SinkKind::NumpyStruct | SinkKind::NdArray => RowResult::NumpyStruct(None),
        SinkKind::EncodedImage | SinkKind::Blob => RowResult::Binary(None),
        SinkKind::BufferList | SinkKind::VectorList => RowResult::TypedList(None),
        SinkKind::BufferArray | SinkKind::VectorArray => RowResult::TypedArray(None),
        SinkKind::Scalar => RowResult::Scalar(None),
        SinkKind::Contours => RowResult::Contours(None),
    })
}
/// A row whose variant the sink kind does not accept.
///
/// `encode_node_output` and [`SinkKind`] are two halves of one contract, so
/// reaching this means they disagree. Publishing the row as null (the former
/// `_ => None` arms) would pass the bug off as data (CR-38).
fn foreign_row(kind: SinkKind, row: &RowResult) -> PolarsError {
    polars_err!(ComputeError:
        "internal: a {:?} sink received a {} row. The encode half and the sink \
         kind disagree about this output.",
        kind, row.variant_name()
    )
}

/// Convert every row with `accept`, which returns `None` for a variant the
/// kind does not accept; that becomes an error rather than a null.
fn convert_rows<T>(
    kind: SinkKind,
    data: Vec<RowResult>,
    accept: impl Fn(RowResult) -> Result<Option<T>, RowResult>,
) -> PolarsResult<Vec<Option<T>>> {
    data.into_iter()
        .map(|row| accept(row).map_err(|row| foreign_row(kind, &row)))
        .collect()
}

/// A vector row as typed list data.
fn vector_row(vals: Vec<f64>) -> (TypedBufferData, Vec<usize>) {
    let len = vals.len();
    (TypedBufferData::F64(vals), vec![len])
}

/// Build a series from row results using the OutputSpec to determine the type.
///
/// This function uses static type information from the OutputSpec rather than
/// inspecting the first row's data. This allows proper handling of null values
/// while preserving the expected output type. Each kind accepts a fixed set of
/// row variants; `None` of an accepted variant is a null row, and any other
/// variant is an internal error.
pub(crate) fn build_series_from_spec(
    name: PlSmallStr,
    spec: &OutputSpec,
    data: Vec<RowResult>,
) -> PolarsResult<Series> {
    let dtype = &spec.expected_dtype;
    let kind = SinkKind::resolve(spec)?;
    match kind {
        // Every arm below is keyed on the resolved kind, so a new one is a
        // compile error here rather than a row that quietly becomes Binary.
        SinkKind::HistogramBuckets => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::HistogramBuckets(b) => Ok(b),
                other => Err(other),
            })?;
            let values = rows
                .iter()
                .map(|r| match r {
                    Some(buckets) => histogram_buckets_to_polars_value(buckets),
                    None => Ok(AnyValue::Null),
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            let histogram_dtype = DataType::List(Box::new(histogram_struct_dtype()));
            Series::from_any_values_and_dtype(name, &values, &histogram_dtype, true)
        }
        SinkKind::NumpyStruct | SinkKind::NdArray => {
            // Move the buffers in so each is the sole Arc owner: that lets
            // `into_polars_buffer_strided` take the zero-copy *strided* branch
            // for non-contiguous (transposed/flipped/rotated) outputs. The
            // numpy/torch struct carries shape/strides/offset, and the Python
            // consumers (`numpy_from_struct`, the struct->PNG helper) honor them
            // via `np.lib.stride_tricks.as_strided`, so permuted layouts decode
            // correctly without materialising to contiguous here.
            let buffers = convert_rows(kind, data, |r| match r {
                RowResult::NumpyStruct(b) => Ok(b),
                other => Err(other),
            })?;
            let series =
                crate::output::build_numpy_series(name, buffers, spec.sink.out_dtype.as_deref())?;
            match kind {
                SinkKind::NdArray => crate::ext_types::ExtType::NdArray.tag(series),
                _ => Ok(series),
            }
        }
        SinkKind::EncodedImage | SinkKind::Blob => {
            // Register each row's already-materialised bytes as a BinaryView
            // backing buffer instead of copying them into a builder — see
            // `crate::output::binary_view_series_from_rows`.
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::Binary(b) => Ok(b),
                other => Err(other),
            })?;
            Ok(crate::output::binary_view_series_from_rows(
                name,
                rows.into_iter(),
            ))
        }
        SinkKind::BufferList => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::TypedList(t) => Ok(t),
                other => Err(other),
            })?;
            build_typed_list_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                spec.expected_shape.as_ref(),
                spec.expected_ndim,
            )
        }
        SinkKind::BufferArray => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::TypedArray(t) => Ok(t),
                other => Err(other),
            })?;
            build_typed_array_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                &spec.sink.shape,
                spec.expected_shape.as_ref(),
            )
        }
        SinkKind::Scalar => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::Scalar(s) => Ok(s),
                other => Err(other),
            })?;
            Ok(Float64Chunked::from_iter_options(name, rows.into_iter()).into_series())
        }
        SinkKind::VectorList => {
            let rows: Vec<TypedListRow> = convert_rows(kind, data, |r| match r {
                RowResult::TypedList(t) => Ok(t),
                RowResult::Vector(v) => Ok(v.map(vector_row)),
                other => Err(other),
            })?;
            build_typed_list_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                spec.expected_shape.as_ref(),
                spec.expected_ndim,
            )
        }
        SinkKind::VectorArray => {
            let rows: Vec<TypedListRow> = convert_rows(kind, data, |r| match r {
                RowResult::TypedList(t) | RowResult::TypedArray(t) => Ok(t),
                RowResult::Vector(v) => Ok(v.map(vector_row)),
                other => Err(other),
            })?;
            build_typed_array_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                &spec.sink.shape,
                spec.expected_shape.as_ref(),
            )
        }
        SinkKind::Contours => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::Contours(c) => Ok(c),
                other => Err(other),
            })?;
            contour_set_series(name, &rows)
        }
    }
}

/// The `array` source reads a row as a *view* into the column's own values
/// buffer. It used to copy the chunk's entire values buffer on every row to
/// take one row's window, which made a batch quadratic: ~35 µs per 64-byte row
/// at 100k rows (CR-40).
#[cfg(test)]
mod array_source_view_tests {
    use super::decode_list_or_array_source;
    use polars::prelude::*;
    use polars_arrow::array::PrimitiveArray;

    fn array_column<T: NumericNative>(flat: Vec<T>, dims: &[i64]) -> Series
    where
        Series: NamedFrom<Vec<T>, [T]>,
    {
        let mut shape = vec![ReshapeDimension::new(-1)];
        shape.extend(dims.iter().map(|&d| ReshapeDimension::new(d)));
        Series::new("a".into(), flat).reshape_array(&shape).unwrap()
    }

    /// Pointer to element 0 of the (single) chunk's leaf values.
    fn leaf_ptr<T: NumericNative>(s: &Series, chunk: usize) -> *const T {
        let mut arr: &dyn polars_arrow::array::Array =
            s.array().unwrap().downcast_iter().nth(chunk).unwrap();
        while let Some(fsl) = arr
            .as_any()
            .downcast_ref::<polars_arrow::array::FixedSizeListArray>()
        {
            arr = fsl.values().as_ref();
        }
        let prim = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        prim.values().as_ptr()
    }

    #[test]
    fn rows_point_into_the_column_values() {
        let flat: Vec<u8> = (0..12).collect();
        let s = array_column(flat.clone(), &[4]);
        let base = leaf_ptr::<u8>(&s, 0);
        for row in 0..3 {
            let vb = decode_list_or_array_source(&s, row, Some("u8"), true)
                .unwrap()
                .unwrap();
            assert_eq!(vb.as_slice::<u8>(), &flat[row * 4..row * 4 + 4]);
            assert_eq!(vb.as_slice::<u8>().as_ptr(), base.wrapping_add(row * 4));
        }
    }

    #[test]
    fn sliced_and_multi_chunk_columns_read_the_right_rows() {
        let flat: Vec<u8> = (0..12).collect();
        let sliced = array_column(flat.clone(), &[4]).slice(1, 2);
        let vb = decode_list_or_array_source(&sliced, 0, Some("u8"), true)
            .unwrap()
            .unwrap();
        assert_eq!(vb.as_slice::<u8>(), &flat[4..8]);

        let mut chunked = array_column(flat.clone(), &[4]);
        chunked
            .append(&array_column((100..108).collect::<Vec<u8>>(), &[4]))
            .unwrap();
        assert_eq!(chunked.n_chunks(), 2);
        let vb = decode_list_or_array_source(&chunked, 4, Some("u8"), true)
            .unwrap()
            .unwrap();
        assert_eq!(vb.as_slice::<u8>(), &[104, 105, 106, 107]);
        assert_eq!(
            vb.as_slice::<u8>().as_ptr(),
            leaf_ptr::<u8>(&chunked, 1).wrapping_add(4)
        );
    }

    #[test]
    fn nested_f32_rows_are_views_too() {
        let flat: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let s = array_column(flat.clone(), &[2, 4]);
        let vb = decode_list_or_array_source(&s, 1, Some("f32"), true)
            .unwrap()
            .unwrap();
        assert_eq!(vb.shape(), &[2, 4]);
        assert_eq!(vb.as_slice::<f32>(), &flat[8..16]);
        assert_eq!(
            vb.as_slice::<f32>().as_ptr(),
            leaf_ptr::<f32>(&s, 0).wrapping_add(8)
        );
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
    fn misaligned_data_offset_is_rejected() {
        // f32 payload one byte past an aligned offset: in bounds, misaligned.
        let mut data = vec![0u8; 1];
        data.extend_from_slice(&[0u8; 16]);
        let blob = craft_blob(7, payload_offset(1) + 1, 1, &[4], &[4], &data);
        let err = decode(blob).expect_err("misaligned offset must be rejected");
        assert!(err.contains("not aligned"), "{err}");
    }

    #[test]
    fn stride_that_is_not_a_whole_element_is_rejected() {
        // 2x2 f32, row stride 6 bytes: every element stays inside the 16-byte
        // payload, but element (1, 0) starts mid-f32.
        let blob = craft_blob(7, payload_offset(2), 0, &[2, 2], &[6, 4], &[0u8; 16]);
        let err = decode(blob).expect_err("misaligned stride must be rejected");
        assert!(err.contains("not aligned"), "{err}");
    }

    #[test]
    fn binary_rows_are_copied_to_an_aligned_address() {
        use polars::prelude::*;
        // Odd lengths and a sliced column, so no row starts on a natural
        // boundary in the source.
        let ca = BinaryChunked::from_slice(
            "b".into(),
            &[&[1u8, 2, 3][..], &[4u8; 13][..], &[5u8; 1][..]],
        )
        .slice(1, 2);
        for row in 0..ca.len() {
            let (buffer, offset, len) = super::get_binary_row_buffer(&ca, row).unwrap();
            assert_eq!(offset, 0);
            assert_eq!(&buffer.as_slice()[..len], ca.get(row).unwrap());
            assert_eq!(buffer.as_slice().as_ptr() as usize % 8, 0);
        }
    }

    #[test]
    fn from_blob_copies_the_whole_strided_window() {
        // Padded 2x2 u8 rows (stride 3): `from_blob` used to copy only the 4
        // logical bytes and keep stride 3, reading past its own allocation.
        let blob = craft_blob(
            1,
            payload_offset(2),
            0,
            &[2, 2],
            &[3, 1],
            &[10, 20, 99, 30, 40],
        );
        let buf = view_buffer::ViewBuffer::from_blob(&blob).expect("in-window blob decodes");
        assert_eq!(buf.to_contiguous().as_slice::<u8>(), &[10, 20, 30, 40]);
    }

    #[test]
    fn from_blob_rejects_what_the_zero_copy_decode_rejects() {
        let misaligned = craft_blob(7, payload_offset(1) + 1, 1, &[4], &[4], &[0u8; 17]);
        assert!(view_buffer::ViewBuffer::from_blob(&misaligned).is_err());
        let hostile = craft_blob(
            1,
            payload_offset(2),
            0,
            &[4, 4],
            &[1_000_000, 1],
            &[0u8; 16],
        );
        assert!(view_buffer::ViewBuffer::from_blob(&hostile).is_err());
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
