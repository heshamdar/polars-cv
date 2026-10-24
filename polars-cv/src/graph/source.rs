//! Source decoding: a row of an input column to the buffer a graph node
//! starts from (binary blobs, raw bytes and images, in place where the
//! bytes allow; list/array columns).

use polars::prelude::*;
use view_buffer::ViewBuffer;

use crate::execute::decode_image_bytes;
use crate::formats::source::Source;
use crate::graph::node_output::NodeOutput;

/// What a path-based source (`file_path`) reads a row's bytes through: this
/// call's [`Fetcher`](crate::fetch::Fetcher) for the node's column (its cloud
/// options and path sandbox included). `None` when the column holds no paths,
/// which a `file_path` source then refuses.
pub(crate) struct RowFetch<'a> {
    pub(crate) fetcher: Option<&'a crate::fetch::Fetcher<'a>>,
}

/// Decode row `row` of a root node's column through its concrete `source`
/// (an `auto` source is routed per batch before it gets here).
pub(crate) fn decode_source_row(
    node_id: &str,
    source: &Source,
    series: &Series,
    row: usize,
    fetch: RowFetch<'_>,
) -> Result<Option<NodeOutput>, String> {
    if series.dtype() == &DataType::Null {
        return Ok(None);
    }
    let binary = || {
        series.binary().map_err(|_| {
            format!(
                "Expected Binary column for node '{}', got {:?}",
                node_id,
                series.dtype()
            )
        })
    };
    let buffer = |buf| Some(NodeOutput::from_buffer(buf));
    match source {
        // The column's contour set, lines (`is_closed = false`) and regions
        // alike; a mask is the `rasterize` op that follows, if any.
        Source::Contour { .. } => crate::geometry::columns::ContourColumn::new(series)
            .outlines(row)
            .map(|set| set.map(NodeOutput::from_contours))
            .map_err(|e| format!("Contour decode error: {e}")),
        // `file_path` is fetch + decode: `crate::fetch` reads the bytes the
        // path names (applying its `PathPolicy` sandbox), then they decode as
        // image bytes.
        Source::FilePath { .. } => {
            let ca = series.str().map_err(|_| {
                format!(
                    "Expected String column for file_path source '{}', got {:?}",
                    node_id,
                    series.dtype()
                )
            })?;
            let Some(path) = ca.get(row) else {
                return Ok(None);
            };
            // Stage 1: bytes, from the call's fetch window (local files are
            // read inline).
            let fetcher = fetch.fetcher.ok_or_else(|| {
                format!("internal: file_path source '{node_id}' has no fetcher for its column")
            })?;
            let Some(bytes) = fetcher.bytes(row)? else {
                return Ok(None);
            };
            // Stage 2: the contents decode like image bytes.
            decode_image_bytes(&bytes, source)
                .map(buffer)
                .map_err(|e| format!("Decode error for file '{path}': {e}"))
        }
        Source::List { .. } | Source::Array { .. } => {
            decode_list_or_array_source(series, row, source.dtype(), source.require_contiguous())
                .map(|buf| buf.map(NodeOutput::from_buffer))
                .map_err(|e| format!("List/Array decode error: {e}"))
        }
        // Raw bytes take the declared dtype.
        Source::Raw { dtype, .. } => {
            Ok(decode_binary_row(binary()?, row, Some(dtype.get()))?.and_then(buffer))
        }
        // A blob carries its own dtype, which a declared one must match: the
        // planner (and identity elimination) takes the declaration as fact.
        Source::Blob { dtype, .. } => {
            let Some(buf) = decode_binary_row(binary()?, row, None)? else {
                return Ok(None);
            };
            match dtype.map(|d| d.get()) {
                Some(declared) if declared != buf.dtype() => Err(format!(
                    "the blob holds {} elements, but the source declares dtype=\"{}\". A \
                     blob carries its own dtype: drop the declaration, correct it, or \
                     .cast(\"{}\") after the source.",
                    buf.dtype().short_name(),
                    declared.short_name(),
                    declared.short_name()
                )),
                _ => Ok(buffer(buf)),
            }
        }
        Source::ImageBytes { .. } => match binary()?.get(row) {
            Some(bytes) => decode_image_bytes(bytes, source)
                .map(buffer)
                .map_err(|e| format!("Decode error: {e}")),
            None => Ok(None),
        },
        Source::Auto { .. } => Err(format!(
            "internal: auto source '{}' reached decoding unrouted",
            node_id
        )),
    }
}

/// Decode row `row` of a binary column: a raw row as `raw_dtype`, or (with
/// `None`) a VIEW blob. `None` for a null row.
///
/// A row is read where it lies whenever its elements are aligned there for
/// their dtype: a raw row's first byte, a blob's payload. Only a row they
/// are not aligned in (or one stored inline in its view) is copied.
pub(crate) fn decode_binary_row(
    ca: &BinaryChunked,
    row: usize,
    raw_dtype: Option<view_buffer::DType>,
) -> Result<Option<ViewBuffer>, String> {
    let decoded = match raw_dtype {
        Some(dtype) => {
            let elem = dtype.size_of();
            let Some((bytes, offset, len)) =
                get_binary_row_buffer(ca, row, |b| (b.as_ptr() as usize).is_multiple_of(elem))
            else {
                return Ok(None);
            };
            decode_raw(bytes, offset, len, dtype)
        }
        None => {
            // Parsed once, where the row lies. The layout is relative to the
            // blob's first byte, so it holds for a copy of the blob too.
            let mut parsed = None;
            let Some((bytes, offset, _)) = get_binary_row_buffer(ca, row, |b| {
                let blob = view_buffer::parse_blob(b);
                let payload =
                    (b.as_ptr() as usize).wrapping_add(blob.as_ref().map_or(0, |l| l.data_offset));
                let in_place = blob
                    .as_ref()
                    .is_ok_and(|l| payload.is_multiple_of(l.dtype.size_of()));
                parsed = Some(blob);
                in_place
            }) else {
                return Ok(None);
            };
            parsed
                .expect("get_binary_row_buffer asks about every non-null row")
                .and_then(|blob| decode_blob_zero_copy(bytes, offset, blob))
        }
    };
    decoded
        .map(Some)
        .map_err(|e| format!("Zero-copy decode error: {e}"))
}

/// Row `row` of a binary column as a buffer: read where it lies when
/// `in_place` accepts its bytes there, else copied to an 8-byte-aligned
/// address (the largest element size). `in_place` is asked about every
/// non-null row, including one stored inline in its view, which is copied
/// whatever it answers.
///
/// Polars stores binary as a `BinaryViewArray`: a value longer than 12 bytes
/// sits in one of the array's shared data buffers, and reading it in place is
/// that buffer, sliced. The typed views built over a row are `&[T]`, so what
/// `in_place` checks is that `T`'s alignment holds where the row lies: an
/// unconditional 8-byte rule copied every u8 row at an odd address; alignment is a
/// requirement, and this makes it the dtype's.
///
/// # Returns
/// `Some((buffer, offset, len))` if the row is valid and not null.
/// `None` if the row is null.
pub(crate) fn get_binary_row_buffer(
    binary_ca: &BinaryChunked,
    row_idx: usize,
    in_place: impl FnOnce(&[u8]) -> bool,
) -> Option<(polars_buffer::Buffer<u8>, usize, usize)> {
    use polars_arrow::array::{Array, View};

    // The chunk holding the row, and the row's index within it.
    let mut i = row_idx;
    let arr = binary_ca.downcast_iter().find(|arr| {
        let here = i < arr.len();
        if !here {
            i -= arr.len();
        }
        here
    })?;
    if !arr.is_valid(i) {
        return None;
    }
    let bytes = arr.value(i);
    let view = arr.views()[i];
    if in_place(bytes) && view.length > View::MAX_INLINE_SIZE {
        // Sliced to the row: the data buffer is shared by the column's rows,
        // and whatever holds this buffer downstream (a numpy sink's `data`)
        // must hold this row's bytes, not every row's.
        let start = view.offset as usize;
        let buffer = arr.data_buffers()[view.buffer_idx as usize]
            .clone()
            .sliced(start..start + bytes.len());
        return Some((buffer, 0, bytes.len()));
    }
    Some((aligned_copy(bytes), 0, bytes.len()))
}

/// Copy `bytes` into a buffer whose first byte is 8-byte aligned.
///
/// Backed by a `Vec<u64>` so the alignment comes from the allocation's type,
/// not from allocator behaviour. The tail word is zero-padded; callers carry
/// the true length separately. One `copy_from_slice` over the words' bytes:
/// assembling them word by word was ~7x slower than the copy itself, the
/// largest per-row cost of a large blob or raw row.
fn aligned_copy(bytes: &[u8]) -> polars_buffer::Buffer<u8> {
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    bytemuck::cast_slice_mut::<u64, u8>(&mut words)[..bytes.len()].copy_from_slice(bytes);
    polars_buffer::Buffer::from(words)
        .try_transmute::<u8>()
        .expect("u64 -> u8 reinterpretation cannot fail")
}
/// A raw row's bytes as a flat buffer of `dtype`, read where they lie.
fn decode_raw(
    buffer: polars_buffer::Buffer<u8>,
    offset: usize,
    len: usize,
    dtype: view_buffer::DType,
) -> Result<ViewBuffer, String> {
    let element_size = dtype.size_of();
    // A remainder is bytes the caller supplied that no element would
    // read; dropping them silently is a truncation, not a decode.
    if !len.is_multiple_of(element_size) {
        return Err(format!(
            "Raw source: {len} bytes is not a multiple of the {} element \
             size ({element_size} bytes)",
            dtype.short_name()
        ));
    }
    Ok(ViewBuffer::from_polars_buffer(
        buffer,
        offset,
        vec![len / element_size],
        dtype,
    ))
}
/// Decode a blob (VIEW protocol) with zero-copy.
///
/// `blob` is the blob's header as parsed and validated by
/// [`view_buffer::parse_blob`] — the same parser `ViewBuffer::from_blob`
/// uses — from the bytes at `base_offset` in `buffer`, and the resulting
/// ViewBuffer points directly into `buffer`. A strided layout keeps its
/// stored strides.
fn decode_blob_zero_copy(
    buffer: polars_buffer::Buffer<u8>,
    base_offset: usize,
    blob: view_buffer::BlobLayout,
) -> Result<ViewBuffer, String> {
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
/// Decode a Polars List or Array value at a specific row into a ViewBuffer.
///
/// Uses zero-copy when the data is contiguous (FixedSizeList/Array types),
/// falling back to copy-based flattening for jagged List types.
///
/// If `dtype` is provided, it will be used. Otherwise, the dtype will be
/// inferred from the Polars column type.
///
/// If `require_contiguous` is true and zero-copy is not possible, an error is returned.
pub(crate) fn decode_list_or_array_source(
    series: &Series,
    row_idx: usize,
    dtype: Option<view_buffer::DType>,
    require_contiguous: bool,
) -> Result<Option<ViewBuffer>, String> {
    let dtype = if let Some(dtype) = dtype {
        dtype
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
    let Some(grid) = list_row_grid(series, row_idx)? else {
        return Ok(None);
    };
    if grid.values.is_empty() {
        return Ok(None);
    }
    let elem = dtype.size_of();
    if let Some(buffer) = get_primitive_buffer(grid.leaf, dtype) {
        // The row's values where the column holds them.
        return Ok(Some(ViewBuffer::from_polars_buffer_slice(
            buffer,
            grid.values.start * elem,
            grid.values.len() * elem,
            grid.shape,
            dtype,
        )));
    }
    if require_contiguous {
        return Err(format!(
            "Source 'require_contiguous=true' reads rows in place, but this row holds \
             {:?} values, not {}: they would have to be converted. Declare the column's own \
             dtype (and .cast() after the source), cast the column, or use \
             require_contiguous=false.",
            grid.leaf.dtype(),
            dtype.short_name()
        ));
    }
    convert_row_values(grid, dtype).map(Some)
}

/// One row of a `List`/`Array` column as a grid: its shape, and the leaf
/// array range holding its values in row-major order.
struct RowGrid<'a> {
    shape: Vec<usize>,
    leaf: &'a dyn polars_arrow::array::Array,
    values: std::ops::Range<usize>,
}

/// Walk row `row_idx` of a `List`/`Array` column level by level through its
/// Arrow arrays, checking that it is a grid. `None` for a null row.
///
/// Every list at a level must have the same length (that length is the
/// level's size) and nothing may be null: a jagged row has no shape, and a
/// null has no value. Taking each level's size from its first element and
/// then reading the values as that shape, as an earlier version did, read
/// past the end of `[[1, 2], [3]]` and silently re-rowed
/// `[[1, 2], [3], [4, 5, 6]]`. Nothing is copied: a level is its offsets.
fn list_row_grid(series: &Series, row_idx: usize) -> Result<Option<RowGrid<'_>>, String> {
    use polars_arrow::array::{Array, FixedSizeListArray, ListArray};
    // The chunk holding the row, and the row's index within it.
    let mut i = row_idx;
    let Some(chunk) = series.chunks().iter().find(|arr| {
        let here = i < arr.len();
        if !here {
            i -= arr.len();
        }
        here
    }) else {
        return Err("internal: the row is past the end of the column".to_string());
    };
    if chunk.is_null(i) {
        return Ok(None);
    }
    let null_in = |arr: &dyn Array, range: &std::ops::Range<usize>| {
        arr.validity()
            .is_some_and(|v| range.clone().any(|j| !v.get_bit(j)))
    };
    let mut shape = Vec::new();
    let mut arr: &dyn Array = chunk.as_ref();
    // The entries of `arr` that make up the row (at the top, the row itself).
    let mut range = i..i + 1;
    loop {
        if let Some(list) = arr.as_any().downcast_ref::<ListArray<i64>>() {
            if null_in(arr, &range) {
                return Err(null_row_part("list"));
            }
            let offsets = list.offsets().as_slice();
            let len = |j: usize| (offsets[j + 1] - offsets[j]) as usize;
            let size = if range.is_empty() {
                0
            } else {
                len(range.start)
            };
            if range.clone().any(|j| len(j) != size) {
                return Err(
                    "the row is jagged: its lists at one level differ in length, so \
                            it has no array shape"
                        .to_string(),
                );
            }
            shape.push(size);
            range = offsets[range.start] as usize..offsets[range.end] as usize;
            arr = list.values().as_ref();
        } else if let Some(fixed) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
            if null_in(arr, &range) {
                return Err(null_row_part("list"));
            }
            let width = fixed.size();
            shape.push(width);
            range = range.start * width..range.end * width;
            arr = fixed.values().as_ref();
        } else {
            if null_in(arr, &range) {
                return Err(null_row_part("value"));
            }
            return Ok(Some(RowGrid {
                shape,
                leaf: arr,
                values: range,
            }));
        }
    }
}

fn null_row_part(what: &str) -> String {
    format!("the row holds a null {what}; a list/array row must be a grid of values")
}

/// A row whose values are not `dtype`, converted once by polars' strict
/// cast: a value `dtype` cannot hold is an error naming it. Polars'
/// non-strict cast, which this used, left 0 in its place.
fn convert_row_values(grid: RowGrid<'_>, dtype: view_buffer::DType) -> Result<ViewBuffer, String> {
    let values = grid.leaf.sliced(grid.values.start, grid.values.len());
    let converted = Series::from_arrow(PlSmallStr::EMPTY, values)
        .and_then(|s| s.strict_cast(&polars_dtype_for(dtype)))
        .and_then(|s| {
            s.rechunk()
                .chunks()
                .first()
                .cloned()
                .ok_or_else(|| polars_err!(ComputeError: "a cast produced no chunk"))
        })
        .map_err(|e| format!("the row's values cannot all be {}: {e}", dtype.short_name()))?;
    let buffer = get_primitive_buffer(converted.as_ref(), dtype).ok_or_else(|| {
        format!(
            "internal: a cast to {} produced {:?}",
            dtype.short_name(),
            converted.dtype()
        )
    })?;
    Ok(ViewBuffer::from_polars_buffer_slice(
        buffer,
        0,
        grid.values.len() * dtype.size_of(),
        grid.shape,
        dtype,
    ))
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
/// allocation, so the caller's per-row window is a view into the column. Copying
/// the entire values buffer once per row would make the `array` source
/// quadratic in the batch size. Sharing is safe because
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

#[cfg(test)]
mod list_source_tests {
    use polars::prelude::*;

    use super::decode_list_or_array_source;

    fn nested(rows: Vec<Vec<Option<f64>>>) -> Series {
        let inner: Vec<AnyValue> = rows
            .into_iter()
            .map(|r| AnyValue::List(Series::new("".into(), r)))
            .collect();
        let row = Series::from_any_values_and_dtype(
            "".into(),
            &inner,
            &DataType::List(Box::new(DataType::Float64)),
            true,
        )
        .unwrap();
        Series::new("a".into(), &[AnyValue::List(row)])
    }

    fn decode(series: &Series) -> Result<Option<view_buffer::ViewBuffer>, String> {
        decode_list_or_array_source(series, 0, Some(view_buffer::DType::F64), false)
    }

    #[test]
    fn a_rectangular_row_decodes_with_its_shape() {
        let s = nested(vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0), Some(4.0)]]);
        let buf = decode(&s).unwrap().unwrap();
        assert_eq!(buf.shape(), &[2, 2]);
        assert_eq!(buf.as_slice::<f64>(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn a_jagged_row_is_refused() {
        for rows in [
            vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0)]],
            // As many values as a 3 x 2 grid, in rows of 2, 1 and 3.
            vec![
                vec![Some(1.0), Some(2.0)],
                vec![Some(3.0)],
                vec![Some(4.0), Some(5.0), Some(6.0)],
            ],
        ] {
            let err = decode(&nested(rows)).unwrap_err();
            assert!(err.contains("jagged"), "{err}");
        }
    }

    #[test]
    fn a_null_value_or_inner_list_is_refused() {
        let err = decode(&nested(vec![
            vec![Some(1.0), None],
            vec![Some(3.0), Some(4.0)],
        ]))
        .unwrap_err();
        assert!(err.contains("null"), "{err}");

        let row = Series::from_any_values_and_dtype(
            "".into(),
            &[
                AnyValue::List(Series::new("".into(), &[1.0, 2.0])),
                AnyValue::Null,
            ],
            &DataType::List(Box::new(DataType::Float64)),
            true,
        )
        .unwrap();
        let err = decode(&Series::new("a".into(), &[AnyValue::List(row)])).unwrap_err();
        assert!(err.contains("null"), "{err}");
    }

    /// A `List[List[T]]` column of `rows` rows, each `h` lists of `w` values:
    /// `flat` in row-major order.
    fn list_grid<T: NumericNative>(flat: Vec<T>, h: i64, w: i64) -> Series
    where
        Series: NamedFrom<Vec<T>, [T]>,
    {
        let dims = [-1, h, w].map(ReshapeDimension::new);
        let array = Series::new("a".into(), flat).reshape_array(&dims).unwrap();
        let leaf = array.dtype().leaf_dtype().clone();
        array
            .cast(&DataType::List(Box::new(DataType::List(Box::new(leaf)))))
            .unwrap()
    }

    /// Pointer to element 0 of chunk `chunk`'s leaf values, through every
    /// list level.
    fn leaf_ptr<T: NumericNative>(s: &Series, chunk: usize) -> *const T {
        use polars_arrow::array::{Array, ListArray, PrimitiveArray};
        let mut arr: &dyn Array = s.chunks()[chunk].as_ref();
        while let Some(list) = arr.as_any().downcast_ref::<ListArray<i64>>() {
            arr = list.values().as_ref();
        }
        let prim = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        prim.values().as_ptr()
    }

    fn decode_as(
        s: &Series,
        row: usize,
        dtype: view_buffer::DType,
        require_contiguous: bool,
    ) -> Result<Option<view_buffer::ViewBuffer>, String> {
        decode_list_or_array_source(s, row, Some(dtype), require_contiguous)
    }

    #[test]
    fn rectangular_list_rows_point_into_the_column_values() {
        let flat: Vec<u8> = (0..24).collect();
        let s = list_grid(flat.clone(), 3, 4);
        let base = leaf_ptr::<u8>(&s, 0);
        for row in 0..2 {
            let vb = decode_as(&s, row, view_buffer::DType::U8, true)
                .unwrap()
                .unwrap();
            assert_eq!(vb.shape(), &[3, 4]);
            assert_eq!(vb.as_slice::<u8>(), &flat[row * 12..row * 12 + 12]);
            assert_eq!(vb.as_slice::<u8>().as_ptr(), base.wrapping_add(row * 12));
        }
    }

    #[test]
    fn flat_list_rows_and_later_chunks_are_views_too() {
        let flat: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let list = |v: Vec<f32>| {
            Series::new("a".into(), v)
                .reshape_array(&[-1, 4].map(ReshapeDimension::new))
                .unwrap()
                .cast(&DataType::List(Box::new(DataType::Float32)))
                .unwrap()
        };
        let mut s = list(flat.clone());
        s.append(&list((100..108).map(|i| i as f32).collect()))
            .unwrap();
        assert_eq!(s.n_chunks(), 2);
        let vb = decode_as(&s, 1, view_buffer::DType::F32, true)
            .unwrap()
            .unwrap();
        assert_eq!(vb.shape(), &[4]);
        assert_eq!(vb.as_slice::<f32>(), &flat[4..8]);
        assert_eq!(
            vb.as_slice::<f32>().as_ptr(),
            leaf_ptr::<f32>(&s, 0).wrapping_add(4)
        );
        let vb = decode_as(&s, 3, view_buffer::DType::F32, true)
            .unwrap()
            .unwrap();
        assert_eq!(vb.as_slice::<f32>(), &[104.0, 105.0, 106.0, 107.0]);
        assert_eq!(
            vb.as_slice::<f32>().as_ptr(),
            leaf_ptr::<f32>(&s, 1).wrapping_add(4)
        );
    }

    #[test]
    fn require_contiguous_refuses_only_a_jagged_list() {
        let err = decode_as(
            &nested(vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0)]]),
            0,
            view_buffer::DType::F64,
            true,
        )
        .unwrap_err();
        assert!(err.contains("jagged"), "{err}");
    }

    #[test]
    fn another_dtype_is_converted_to_the_declared_one() {
        let s = list_grid((0..12).collect::<Vec<i64>>(), 3, 2);
        let vb = decode_as(&s, 1, view_buffer::DType::U8, false)
            .unwrap()
            .unwrap();
        assert_eq!(vb.shape(), &[3, 2]);
        assert_eq!(vb.as_slice::<u8>(), &[6, 7, 8, 9, 10, 11]);
    }

    /// A value the declared dtype cannot hold is refused, never stored as
    /// the 0 polars' non-strict cast leaves in its place.
    #[test]
    fn a_value_the_declared_dtype_cannot_hold_is_refused() {
        for (values, what) in [(vec![7i64, 300, 1, 2], "300"), (vec![7i64, -1, 1, 2], "-1")] {
            let s = list_grid(values, 2, 2);
            let err = decode_as(&s, 0, view_buffer::DType::U8, false).unwrap_err();
            assert!(err.contains(what) && err.contains("u8"), "{err}");
        }
        let s = nested(vec![vec![Some(1.5), Some(f64::NAN)]]);
        let err = decode_as(&s, 0, view_buffer::DType::U8, false).unwrap_err();
        assert!(err.contains("NaN") && err.contains("u8"), "{err}");
    }

    #[test]
    fn an_empty_row_is_null() {
        let s = Series::new(
            "a".into(),
            &[AnyValue::List(Series::new_empty(
                "".into(),
                &DataType::UInt8,
            ))],
        );
        assert!(decode_as(&s, 0, view_buffer::DType::U8, true)
            .unwrap()
            .is_none());
    }

    #[test]
    fn a_sliced_column_decodes_its_own_row() {
        let a = nested(vec![vec![Some(9.0)]]);
        let mut both = nested(vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0)]]);
        both.append(&a).unwrap();
        let s = both.rechunk().slice(1, 1);
        let buf = decode(&s).unwrap().unwrap();
        assert_eq!(buf.as_slice::<f64>(), &[9.0]);
    }
}

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
            let vb = decode_list_or_array_source(&s, row, Some(view_buffer::DType::U8), true)
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
        let vb = decode_list_or_array_source(&sliced, 0, Some(view_buffer::DType::U8), true)
            .unwrap()
            .unwrap();
        assert_eq!(vb.as_slice::<u8>(), &flat[4..8]);

        let mut chunked = array_column(flat.clone(), &[4]);
        chunked
            .append(&array_column((100..108).collect::<Vec<u8>>(), &[4]))
            .unwrap();
        assert_eq!(chunked.n_chunks(), 2);
        let vb = decode_list_or_array_source(&chunked, 4, Some(view_buffer::DType::U8), true)
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
        let vb = decode_list_or_array_source(&s, 1, Some(view_buffer::DType::F32), true)
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
    use super::decode_blob_zero_copy;
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
        let layout = view_buffer::parse_blob(&blob)?;
        decode_blob_zero_copy(polars_buffer::Buffer::from(blob), 0, layout)
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

    /// A one-chunk Binary column over one 8-byte-aligned data buffer holding
    /// the bytes 0..64, with a row per `(offset, len)` window of it (a window
    /// of 12 bytes or fewer is stored inline, as Arrow requires).
    fn windows_column(windows: &[(usize, usize)]) -> polars::prelude::BinaryChunked {
        use polars_arrow::array::{BinaryViewArrayGeneric, View};
        use polars_arrow::datatypes::ArrowDataType;
        let mut words = vec![0u64; 8];
        for (i, b) in bytemuck::cast_slice_mut::<u64, u8>(&mut words)
            .iter_mut()
            .enumerate()
        {
            *b = i as u8;
        }
        let data = polars_buffer::Buffer::from(words)
            .try_transmute::<u8>()
            .unwrap();
        assert_eq!(data.as_slice().as_ptr() as usize % 8, 0);
        let views: Vec<View> = windows
            .iter()
            .map(|&(offset, len)| {
                let bytes = &data.as_slice()[offset..offset + len];
                if len <= 12 {
                    View::new_inline(bytes)
                } else {
                    View::new_from_bytes(bytes, 0, offset as u32)
                }
            })
            .collect();
        let total: usize = windows.iter().map(|w| w.1).sum();
        // Safety: every view is in bounds of buffer 0 or inline, built above.
        let array = unsafe {
            BinaryViewArrayGeneric::<[u8]>::new_unchecked(
                ArrowDataType::BinaryView,
                views.into(),
                std::iter::once(data).collect(),
                None,
                Some(total),
                64,
            )
        };
        polars::prelude::BinaryChunked::with_chunk("b".into(), array)
    }

    /// The rule these tests hold rows to: in place at an 8-byte boundary,
    /// what an f64 or 64-bit integer row needs.
    fn eight_aligned(bytes: &[u8]) -> bool {
        (bytes.as_ptr() as usize).is_multiple_of(8)
    }

    /// A row stored at an 8-byte-aligned address in the column's data is read
    /// where it lies: the decoded buffer is the column's memory, not a copy.
    #[test]
    fn aligned_binary_rows_are_read_in_place() {
        // Two chunks, so the rows of the second are found by their index
        // within it.
        let mut ca = windows_column(&[(0, 24), (16, 40)]);
        ca.append(&windows_column(&[(8, 13), (24, 16)])).unwrap();
        assert_eq!(ca.chunks().len(), 2);
        for row in 0..ca.len() {
            let (buffer, offset, len) =
                super::get_binary_row_buffer(&ca, row, eight_aligned).unwrap();
            let bytes = ca.get(row).unwrap();
            assert_eq!(len, bytes.len());
            assert_eq!(
                buffer.as_slice()[offset..].as_ptr(),
                bytes.as_ptr(),
                "row {row} was copied"
            );
            // Only the row: the rest of the shared data buffer is other rows'.
            assert_eq!(
                (offset, buffer.len()),
                (0, len),
                "row {row} carries other rows"
            );
        }
    }

    /// A row that is inline in its view, or misaligned in the data, is copied
    /// to an aligned address, which is what the typed views need.
    #[test]
    fn unaligned_and_inline_binary_rows_are_copied_aligned() {
        let ca = windows_column(&[(3, 20), (0, 5), (9, 12)]);
        for row in 0..ca.len() {
            let (buffer, offset, len) =
                super::get_binary_row_buffer(&ca, row, eight_aligned).unwrap();
            let bytes = ca.get(row).unwrap();
            let got = &buffer.as_slice()[offset..offset + len];
            assert_eq!(got, bytes, "row {row}");
            assert_eq!(got.as_ptr() as usize % 8, 0, "row {row} is unaligned");
            assert_ne!(got.as_ptr(), bytes.as_ptr(), "row {row}");
        }
    }

    #[test]
    fn binary_rows_are_read_at_an_aligned_address() {
        use polars::prelude::*;
        // Odd lengths and a sliced column, so no row starts on a natural
        // boundary in the source.
        let ca = BinaryChunked::from_slice(
            "b".into(),
            &[
                &[1u8, 2, 3][..],
                &[4u8; 13][..],
                &[5u8; 1][..],
                &[][..],
                &[6u8; 8][..],
            ],
        )
        .slice(1, 4);
        for row in 0..ca.len() {
            let (buffer, offset, len) =
                super::get_binary_row_buffer(&ca, row, eight_aligned).unwrap();
            let got = &buffer.as_slice()[offset..offset + len];
            assert_eq!(got, ca.get(row).unwrap());
            assert_eq!(got.as_ptr() as usize % 8, 0);
        }
    }

    /// A one-row binary column whose row is `bytes`, stored `offset` bytes
    /// into an 8-byte-aligned data buffer.
    fn row_at(bytes: &[u8], offset: usize) -> polars::prelude::BinaryChunked {
        use polars_arrow::array::{BinaryViewArrayGeneric, View};
        use polars_arrow::datatypes::ArrowDataType;
        assert!(bytes.len() > 12, "a row of 12 bytes or fewer is inline");
        let mut words = vec![0u64; (offset + bytes.len()).div_ceil(8)];
        bytemuck::cast_slice_mut::<u64, u8>(&mut words)[offset..offset + bytes.len()]
            .copy_from_slice(bytes);
        let data = polars_buffer::Buffer::from(words)
            .try_transmute::<u8>()
            .unwrap();
        let view = View::new_from_bytes(bytes, 0, offset as u32);
        // Safety: the view is in bounds of buffer 0, built above.
        let array = unsafe {
            BinaryViewArrayGeneric::<[u8]>::new_unchecked(
                ArrowDataType::BinaryView,
                vec![view].into(),
                std::iter::once(data).collect(),
                None,
                Some(bytes.len()),
                bytes.len(),
            )
        };
        polars::prelude::BinaryChunked::with_chunk("b".into(), array)
    }

    /// A raw row needs only its own dtype's alignment: u8 rows anywhere, f32
    /// rows at a multiple of 4, are read where they lie.
    #[test]
    fn raw_rows_aligned_for_their_dtype_are_read_in_place() {
        use view_buffer::DType;
        let bytes: Vec<u8> = (0..16).collect();
        for (offset, dtype, in_place) in [
            (3, DType::U8, true),
            (4, DType::F32, true),
            (2, DType::F32, false),
            (4, DType::F64, false),
        ] {
            let ca = row_at(&bytes, offset);
            let buf = super::decode_binary_row(&ca, 0, Some(dtype))
                .unwrap()
                .unwrap();
            // SAFETY: the pointer is only compared, never read.
            let start = unsafe { buf.as_ptr::<u8>() };
            assert_eq!(
                start == ca.get(0).unwrap().as_ptr(),
                in_place,
                "{dtype:?} row at offset {offset}"
            );
            assert!((start as usize).is_multiple_of(dtype.size_of()));
            assert_eq!(buf.shape(), &[16 / dtype.size_of()]);
        }
    }

    /// A blob is read where it lies when its payload is aligned for the
    /// blob's dtype, wherever the blob itself starts; otherwise it is copied.
    #[test]
    fn blobs_with_an_aligned_payload_are_read_in_place() {
        let u8_blob = craft_blob(1, payload_offset(1), 1, &[20], &[1], &[7u8; 20]);
        let ca = row_at(&u8_blob, 3);
        let buf = super::decode_binary_row(&ca, 0, None).unwrap().unwrap();
        let row = ca.get(0).unwrap();
        assert_eq!(buf.as_slice::<u8>(), &[7u8; 20]);
        assert_eq!(
            buf.as_slice::<u8>().as_ptr(),
            row[payload_offset(1) as usize..].as_ptr()
        );

        let values: Vec<u8> = bytemuck::cast_slice(&[1.5f32, 2.5, 3.5, 4.5]).to_vec();
        let f32_blob = craft_blob(7, payload_offset(1), 1, &[4], &[4], &values);
        for (offset, in_place) in [(4, true), (2, false)] {
            let ca = row_at(&f32_blob, offset);
            let buf = super::decode_binary_row(&ca, 0, None).unwrap().unwrap();
            let row = ca.get(0).unwrap();
            assert_eq!(buf.as_slice::<f32>(), &[1.5, 2.5, 3.5, 4.5]);
            assert_eq!(
                buf.as_slice::<f32>().as_ptr().cast::<u8>()
                    == row[payload_offset(1) as usize..].as_ptr(),
                in_place,
                "f32 blob at offset {offset}"
            );
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
