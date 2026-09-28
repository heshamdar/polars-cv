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
use pyo3::prelude::*;
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

/// Encoded numpy output for a single row.
///
/// This struct holds the components that will become the Struct column fields.
/// Supports strided views for zero-copy output of non-contiguous buffers.
#[derive(Debug, Clone)]
pub struct NumpyRowOutput {
    /// Raw array data as bytes (may be larger than needed for strided views).
    pub data: polars_buffer::Buffer<u8>,
    /// Data type name (e.g., "uint8", "float32").
    pub dtype: &'static str,
    /// Array shape dimensions.
    pub shape: Vec<u64>,
    /// Byte strides per dimension (enables strided numpy views).
    pub strides: Vec<i64>,
    /// Byte offset into data buffer.
    pub offset: u64,
}

impl NumpyRowOutput {
    /// Create a NumpyRowOutput from a ViewBuffer.
    ///
    /// Uses zero-copy ownership transfer when possible, including for
    /// non-contiguous strided buffers by preserving stride information.
    pub fn from_buffer(buffer: ViewBuffer) -> Self {
        let (data, shape, strides, offset, dtype) = buffer.into_polars_buffer_strided();

        Self {
            data,
            dtype: dtype.numpy_name(),
            shape: shape.into_iter().map(|d| d as u64).collect(),
            strides: strides.into_iter().map(|s| s as i64).collect(),
            offset: offset as u64,
        }
    }

    /// A half-precision (`float16`) row from its f16 bits, as
    /// `ViewBuffer::to_f16_bits` gives them: zero-copy like any other row,
    /// labelled `float16`. That conversion, which backs
    /// `.sink("numpy"|"torch", dtype="f16")` (the engine has no f16 dtype),
    /// runs on the row's own thread in the encode half
    /// (`graph::encode::encode_node_output`), not here in the serial column
    /// build.
    ///
    /// # Panics
    /// Panics unless `bits` is a `U16` buffer.
    pub fn from_f16_bits(bits: ViewBuffer) -> Self {
        assert_eq!(
            bits.dtype(),
            VbDType::U16,
            "internal: a float16 row holds its f16 bits as u16"
        );
        Self {
            dtype: "float16",
            ..Self::from_buffer(bits)
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
pub fn build_numpy_series(
    name: PlSmallStr,
    rows: Vec<Option<ViewBuffer>>,
    as_f16: bool,
) -> PolarsResult<Series> {
    let len = rows.len();

    // A half-precision request (`formats::sink::SinkDType`) was converted
    // on each row's thread (`NumpyRowOutput::from_f16_bits`): its rows are
    // f16 bits, labelled here.

    // Convert each row to NumpyRowOutput
    let encoded: Vec<Option<NumpyRowOutput>> = rows
        .into_iter()
        .map(|opt| {
            opt.map(|b| {
                if as_f16 {
                    NumpyRowOutput::from_f16_bits(b)
                } else {
                    NumpyRowOutput::from_buffer(b)
                }
            })
        })
        .collect();

    // Build all five columns
    let data_col = build_data_column(&encoded)?;
    let dtype_col = build_dtype_column(&encoded);
    let shape_col = build_shape_column(&encoded)?;
    let strides_col = build_strides_column(&encoded)?;
    let offset_col = build_offset_column(&encoded);

    // A null row is a null *struct*, like every other sink's null row, so
    // `is_null()`/`drop_nulls()` see it; the fields under it are null too.
    // Field-level nulls alone read as a present row of nulls (CR-39).
    let validity: Option<polars_arrow::bitmap::Bitmap> = encoded
        .iter()
        .any(Option::is_none)
        .then(|| encoded.iter().map(Option::is_some).collect());

    StructChunked::from_series(
        name,
        len,
        [data_col, dtype_col, shape_col, strides_col, offset_col].iter(),
    )
    .map(|ca| ca.with_outer_validity(validity).into_series())
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
    let mut buffers: Vec<polars_buffer::Buffer<u8>> = Vec::new();
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
            Some(total_bytes_len),
            total_buffer_len,
        )
    };

    let ca = BinaryChunked::with_chunk(PlSmallStr::from_static("data"), array);
    Ok(ca.into_series())
}

/// An Arrow buffer of a `Binary` column, kept alive for the numpy arrays that
/// view it (`polars_cv.numpy_from_column`).
///
/// Opaque to Python: an array built over a row's address holds one of these
/// (through its `.base`), and the buffer lives as long as that array does.
#[pyclass(frozen, module = "polars_cv._lib")]
pub(crate) struct ArrowBytes {
    _owner: ArrowBytesOwner,
}

/// Where a `BinaryView` row's bytes live: in the views buffer itself (a value
/// of 12 bytes or fewer is stored inline) or in one of the data buffers.
#[expect(
    dead_code,
    reason = "held, never read: owning the buffer is what keeps the memory alive"
)]
enum ArrowBytesOwner {
    Views(polars_buffer::Buffer<View>),
    Data(polars_buffer::Buffer<u8>),
}

/// One row of a `Binary` column as `(owner, address, length)`: the address of
/// its first byte in the column's own memory and the buffer that holds it,
/// or `None` for a null row. Nothing is copied.
type BinaryRow = Option<(Py<ArrowBytes>, usize, usize)>;

/// Every row of a `Binary` column, as the address of its bytes in the
/// column's Arrow memory (see [`BinaryRow`]).
///
/// The column crosses from Python through polars' own series export, which
/// shares its buffers rather than copying them, so the addresses are the
/// Python column's. One owner is created per buffer and shared by the rows
/// that live in it.
#[pyfunction]
pub(crate) fn binary_rows(
    py: Python<'_>,
    series: pyo3_polars::PySeries,
) -> PyResult<Vec<BinaryRow>> {
    let ca = series.0.binary().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "binary_rows reads a Binary column, got {}",
            series.0.dtype()
        ))
    })?;
    let mut rows: Vec<BinaryRow> = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        let mut views_owner: Option<Py<ArrowBytes>> = None;
        let mut data_owners: Vec<Option<Py<ArrowBytes>>> =
            (0..arr.data_buffers().len()).map(|_| None).collect();
        for i in 0..arr.len() {
            if !polars_arrow::array::Array::is_valid(arr, i) {
                rows.push(None);
                continue;
            }
            let bytes = arr.value(i);
            let view = arr.views()[i];
            let owner = if view.length <= View::MAX_INLINE_SIZE {
                views_owner.get_or_insert_with(|| {
                    Py::new(
                        py,
                        ArrowBytes {
                            _owner: ArrowBytesOwner::Views(arr.views().clone()),
                        },
                    )
                    .expect("allocating a Python object")
                })
            } else {
                let idx = view.buffer_idx as usize;
                data_owners[idx].get_or_insert_with(|| {
                    Py::new(
                        py,
                        ArrowBytes {
                            _owner: ArrowBytesOwner::Data(arr.data_buffers()[idx].clone()),
                        },
                    )
                    .expect("allocating a Python object")
                })
            };
            rows.push(Some((
                owner.clone_ref(py),
                bytes.as_ptr() as usize,
                bytes.len(),
            )));
        }
    }
    Ok(rows)
}

/// Build a `Binary` series from owned per-row blobs without copying the bytes
/// into the Arrow buffer.
///
/// Each row's `Vec<u8>` (already materialised by `to_blob()` / image encode) is
/// *moved* into a `polars_buffer::Buffer<u8>` and registered as a backing buffer
/// of a `BinaryViewArray`, with a view pointing at it — the same zero-extra-copy
/// registration `build_data_column` uses for the numpy `data` field. This
/// replaces a `BinaryChunkedBuilder`, which copies every row's bytes a second
/// time (on top of the inherent `to_blob()`/codec materialisation).
///
/// Values of 12 bytes or fewer are stored inline in the view (no buffer), which
/// is also how Arrow's BinaryView format avoids tiny allocations.
pub(crate) fn binary_view_series_from_rows(
    name: PlSmallStr,
    rows: impl ExactSizeIterator<Item = Option<Vec<u8>>>,
) -> Series {
    use polars_arrow::datatypes::ArrowDataType;

    let n_rows = rows.len();
    let mut views: Vec<View> = Vec::with_capacity(n_rows);
    let mut buffers: Vec<polars_buffer::Buffer<u8>> = Vec::new();
    let mut validity_builder: Option<MutableBitmap> = None;
    let mut total_bytes_len: usize = 0;
    let mut total_buffer_len: usize = 0;

    for (idx, opt) in rows.enumerate() {
        match opt {
            Some(data) => {
                let data_len = data.len();
                total_bytes_len += data_len;

                if data_len <= 12 {
                    views.push(View::new_inline(data.as_slice()));
                } else {
                    let buffer_idx = buffers.len() as u32;
                    total_buffer_len += data_len;
                    // Build the view from the borrowed bytes first, then move the
                    // Vec into a Buffer (Vec -> Buffer is a zero-copy handoff).
                    let view = View::new_from_bytes(data.as_slice(), buffer_idx, 0);
                    buffers.push(polars_buffer::Buffer::from(data));
                    views.push(view);
                }

                if let Some(ref mut validity) = validity_builder {
                    validity.push(true);
                }
            }
            None => {
                if validity_builder.is_none() {
                    let mut bitmap = MutableBitmap::with_capacity(n_rows);
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

    let validity = validity_builder.map(|v| v.into());

    // Safety: views reference valid buffer indices and lengths built above.
    let array = unsafe {
        BinaryViewArrayGeneric::<[u8]>::new_unchecked(
            ArrowDataType::BinaryView,
            views.into(),
            buffers.into_iter().collect(),
            validity,
            Some(total_bytes_len),
            total_buffer_len,
        )
    };

    BinaryChunked::with_chunk(name, array).into_series()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_row_output_names_its_dtype_from_the_dtype_table() {
        // `dtype_to_string` used to sit here as a second name for
        // `DType::numpy_name`. The struct field is the only thing that ever
        // needed it, so it reads the authority directly.
        let buf = ViewBuffer::from_vec_with_shape(vec![0.0f32; 4], vec![2, 2]);
        assert_eq!(NumpyRowOutput::from_buffer(buf).dtype, "float32");
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
    fn test_numpy_row_output_f16_downcast() {
        // A float buffer downcast to f16: 2 bytes/element, "float16", contiguous.
        let buffer = ViewBuffer::from_vec(vec![0.0f32, 1.0, 2.0, 3.0]).reshape(vec![2, 2]);

        let output = NumpyRowOutput::from_f16_bits(buffer.to_f16_bits());

        assert_eq!(output.dtype, "float16");
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.offset, 0);
        // 4 elements * 2 bytes.
        assert_eq!(output.data.len(), 8);
        // Row-major f16 strides: [row=2*2 bytes, col=2 bytes].
        assert_eq!(output.strides, vec![4, 2]);
        // First element round-trips through half.
        let first =
            half::f16::from_le_bytes([output.data.as_slice()[0], output.data.as_slice()[1]]);
        assert_eq!(first.to_f32(), 0.0);
        let second =
            half::f16::from_le_bytes([output.data.as_slice()[2], output.data.as_slice()[3]]);
        assert_eq!(second.to_f32(), 1.0);
    }

    #[test]
    fn test_build_numpy_series_with_data() {
        let buf1 = ViewBuffer::from_vec(vec![1u8, 2, 3, 4]).reshape(vec![2, 2]);
        let buf2 = ViewBuffer::from_vec(vec![5u8, 6, 7, 8, 9, 10]).reshape(vec![2, 3]);

        let series = build_numpy_series(
            PlSmallStr::from_static("output"),
            vec![Some(buf1), None, Some(buf2)],
            false,
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
