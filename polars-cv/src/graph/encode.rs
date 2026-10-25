//! Sinks: what an output publishes and how its rows become a Series — the
//! planned output dtype, each row's encoded value, and the column the rows
//! assemble into.

use polars::prelude::*;
use std::mem::MaybeUninit;
use std::ops::Range;
use view_buffer::geometry::contour::Outline;
use view_buffer::{DType, PlannedDType, ViewBuffer};

use super::decode::polars_dtype_for;
use super::sink_kind::SinkKind;
use super::types::{LongRow, OutputSpec, OutputValue, RowResult};
use crate::formats::sink::LongLayout;
use crate::formats::Format as _;
use crate::graph::node_output::NodeOutput;
use crate::row_split::Split;

/// The Polars element type of a typed `list`/`array` sink.
///
/// Refuses a dtype the planner never pinned down (`auto`, `auto_float`): a
/// typed column cannot be planned from it, and mapping it to anything would be
/// a column execution may contradict.
fn list_array_inner_dtype(
    dtype: PlannedDType,
    sink: &str,
    facts: ColumnFacts,
) -> PolarsResult<DataType> {
    match dtype {
        PlannedDType::Known(dtype) => Ok(polars_dtype_for(dtype)),
        // The column will supply it; this schema is only checked, never
        // published (`check_output_before_column`).
        PlannedDType::SomeFloat | PlannedDType::Unknown
            if matches!(facts, ColumnFacts::Pending { .. }) =>
        {
            Ok(DataType::Null)
        }
        // Not labelled an internal error: the common way to get here is a
        // source column whose element type the planner cannot map to a buffer
        // dtype (a boolean or decimal list), which is the user's input, not a
        // bug. The fix is the same either way — say what it is.
        PlannedDType::SomeFloat | PlannedDType::Unknown => polars_bail!(ComputeError:
            "the '{sink}' sink needs to know the element dtype at planning \
             time, and it could not be inferred from the input column. \
             Supply it explicitly, e.g. source(..., dtype=\"u16\") or \
             .cast(...) before the sink."
        ),
    }
}
/// Whether the input column has been seen when an output's schema is decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColumnFacts {
    /// Polars is planning the query: the plan already holds everything the
    /// column reveals.
    Resolved,
    /// `.sink()` is checking a graph whose root takes its element type and
    /// rank — and, when `sizes`, its sizes — from a column not yet seen. Those
    /// facts are the column's to supply, so an unknown one is not refused;
    /// everything else is decided now.
    Pending { sizes: bool },
}

/// Get the Polars DataType for a given output specification.
///
/// Returns the appropriate dtype based on domain, sink format, and expected dtype.
pub(crate) fn dtype_for_output(spec: &OutputSpec) -> PolarsResult<DataType> {
    output_schema(spec, ColumnFacts::Resolved)
}

/// Check an output's sink before its column is seen (see
/// [`ColumnFacts::Pending`]). The schema is not returned: what the column
/// will supply is not known yet.
pub(crate) fn check_output_before_column(
    spec: &OutputSpec,
    facts: ColumnFacts,
) -> PolarsResult<()> {
    output_schema(spec, facts).map(|_| ())
}

/// The one sink-schema decision, for [`dtype_for_output`] and
/// [`check_output_before_column`].
fn output_schema(spec: &OutputSpec, facts: ColumnFacts) -> PolarsResult<DataType> {
    let inner = |sink: &str| list_array_inner_dtype(spec.expected_dtype, sink, facts);
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
            let codec = kind
                .image_codec(spec)
                .expect("EncodedImage carries a codec");
            let dtype = spec.expected_dtype;
            codec
                .check_shape(dtype, spec.expected_shape.as_deref(), spec.expected_ndim)
                .map_err(|msg| polars_err!(ComputeError: "{}", msg))?;
            Ok(DataType::Binary)
        }
        SinkKind::BufferList => {
            let inner = inner("list")?;
            let ndim = spec
                .expected_shape
                .as_ref()
                .map(|shape| shape.len())
                .or(spec.expected_ndim)
                .or(matches!(facts, ColumnFacts::Pending { .. }).then_some(1));
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
            let inner = inner("array")?;
            let sink_shape = spec.sink.shape();
            let shape = sink_shape.as_ref().or(spec.expected_shape.as_ref());
            if let Some(shape) = shape {
                let mut dtype = inner;
                for &dim in shape.iter().rev() {
                    dtype = DataType::Array(Box::new(dtype), dim);
                }
                Ok(dtype)
            } else if facts == (ColumnFacts::Pending { sizes: true }) {
                // A fixed-size `Array` column's type states every size, so
                // this is decided when the query is planned with the column.
                // The dtype is not returned for `Pending`.
                Ok(inner)
            } else {
                // Names what each remedy actually supplies. The advice this
                // replaces was circular for the source that reaches it most: a
                // list column's sizes vary per row, so it lands here — and was
                // told to call `.assert_shape()`, which
                // published nothing without a rank, and `.resize()`, which never
                // supplies the channel count.
                polars_bail!(ComputeError:
                    "an 'array' sink needs the full output shape at planning time, and \
                     this pipeline's is not known. Three ways to supply it:\n  \
                     .sink('array', shape=[8, 8, 3])   — always works; the shape belongs \
                     to the sink\n  \
                     .assert_shape(dims=[8, 8, 3])     — when you know it and the source \
                     does not (a list column's sizes are only settled per row)\n  \
                     .resize(height=8, width=8)        — supplies height and width only"
                );
            }
        }
        SinkKind::Long => {
            let layout = long_layout(spec);
            if let Some(ndim) = spec.expected_ndim.filter(|n| !matches!(n, 2 | 3)) {
                polars_bail!(ComputeError:
                    "the 'long' sink takes an [H, W] or [H, W, C] buffer, and this \
                     one has rank {ndim}"
                );
            }
            if let Some(channels) = spec.expected_channels {
                layout
                    .check_channels(channels)
                    .map_err(|msg| polars_err!(ComputeError: "{}", msg))?;
            }
            Ok(super::encode::long_dtype(&layout, inner("long")?))
        }
        SinkKind::Scalar => Ok(DataType::Float64),
        SinkKind::VectorList => {
            // Reject an unresolved "auto" element dtype the same way the
            // buffer/list and array arms do, instead of silently mapping it to
            // U8 — a plan/data divergence if a vector output ever reached the
            // sink still "auto". (Today vector dtypes are always concrete.)
            let inner = inner("list")?;
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
            let inner = inner("array")?;
            let sink_shape = spec.sink.shape();
            let shape = sink_shape.as_ref().or(spec.expected_shape.as_ref());
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
            crate::namespaces::schema::contour_fields(),
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
        SinkKind::Long => RowResult::Long(None),
    })
}

/// The `long` sink's layout, for a spec that resolved to [`SinkKind::Long`]
/// (which only a `long` sink resolves to).
pub(crate) fn long_layout(spec: &OutputSpec) -> crate::formats::sink::LongLayout<'_> {
    spec.sink
        .long_layout()
        .expect("SinkKind::Long is resolved from a long sink")
}
/// A row whose variant the sink kind does not accept.
///
/// `encode_node_output` and [`SinkKind`] are two halves of one contract, so
/// reaching this means they disagree. Publishing the row as null (the former
/// `_ => None` arms) would pass the bug off as data.
fn foreign_row(kind: SinkKind, variant: &str) -> PolarsError {
    polars_err!(ComputeError:
        "internal: a {:?} sink received a {} row. The encode half and the sink \
         kind disagree about this output.",
        kind, variant
    )
}

/// Convert every row with `accept`, which returns the variant's name for a
/// variant the kind does not accept; that becomes an error rather than a null.
fn convert_rows<T>(
    kind: SinkKind,
    data: RowParts,
    accept: impl Fn(RowResult) -> Result<Option<T>, &'static str>,
) -> PolarsResult<Vec<Option<T>>> {
    let mut rows = Vec::with_capacity(data.iter().map(Vec::len).sum());
    for row in data.into_iter().flatten() {
        rows.push(accept(row).map_err(|variant| foreign_row(kind, variant))?);
    }
    Ok(rows)
}

/// A column's row results in row order, as the row ranges that computed
/// them left them: converted straight from the parts, never concatenated
/// into one call-sized vector of (large) `RowResult`s first.
pub(crate) type RowParts = Vec<Vec<RowResult>>;

/// A vector row as typed list data.
fn vector_row(vals: Vec<f64>) -> ViewBuffer {
    ViewBuffer::from_vec(vals)
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
    data: RowParts,
    split: Option<&crate::row_split::Split>,
) -> PolarsResult<Series> {
    let dtype = spec.expected_dtype;
    let kind = SinkKind::resolve(spec)?;
    match kind {
        // Every arm below is keyed on the resolved kind, so a new one is a
        // compile error here rather than a row that quietly becomes Binary.
        SinkKind::HistogramBuckets => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::HistogramBuckets(b) => Ok(b),
                other => Err(other.variant_name()),
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
                other => Err(other.variant_name()),
            })?;
            let series = crate::output::build_numpy_series(name, buffers, spec.sink.as_f16())?;
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
                other => Err(other.variant_name()),
            })?;
            Ok(crate::output::binary_view_series_from_rows(
                name,
                rows.into_iter(),
            ))
        }
        SinkKind::BufferList => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::TypedList(t) => Ok(t),
                other => Err(other.variant_name()),
            })?;
            build_typed_list_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                spec.expected_shape.as_ref(),
                spec.expected_ndim,
                split,
            )
        }
        SinkKind::BufferArray => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::TypedArray(t) => Ok(t),
                other => Err(other.variant_name()),
            })?;
            build_typed_array_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                &spec.sink.shape(),
                spec.expected_shape.as_ref(),
                split,
            )
        }
        SinkKind::Scalar => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::Scalar(s) => Ok(s),
                other => Err(other.variant_name()),
            })?;
            Ok(Float64Chunked::from_iter_options(name, rows.into_iter()).into_series())
        }
        SinkKind::VectorList => {
            let rows: Vec<TypedListRow> = convert_rows(kind, data, |r| match r {
                RowResult::TypedList(t) => Ok(t),
                RowResult::Vector(v) => Ok(v.map(vector_row)),
                other => Err(other.variant_name()),
            })?;
            build_typed_list_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                spec.expected_shape.as_ref(),
                spec.expected_ndim,
                split,
            )
        }
        SinkKind::VectorArray => {
            let rows: Vec<TypedListRow> = convert_rows(kind, data, |r| match r {
                RowResult::TypedList(t) | RowResult::TypedArray(t) => Ok(t),
                RowResult::Vector(v) => Ok(v.map(vector_row)),
                other => Err(other.variant_name()),
            })?;
            build_typed_array_series_from_rows_with_dtype(
                name,
                &rows,
                dtype,
                &spec.sink.shape(),
                spec.expected_shape.as_ref(),
                split,
            )
        }
        SinkKind::Contours => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::Contours(c) => Ok(c),
                other => Err(other.variant_name()),
            })?;
            contour_set_series(name, &rows)
        }
        SinkKind::Long => {
            let rows = convert_rows(kind, data, |r| match r {
                RowResult::Long(l) => Ok(l),
                other => Err(other.variant_name()),
            })?;
            super::encode::long_series(name, &rows, &long_layout(spec), dtype, split)
        }
    }
}

/// The `array` source reads a row as a *view* into the column's own values
/// buffer. Copying the chunk's entire values buffer on every row to take one
/// row's window would make a batch quadratic: ~35 µs per 64-byte row at 100k
/// rows.
/// A `List` row decodes to a buffer only when it is a rectangular grid of
/// values: a jagged row, a null value or a null inner list is refused. The
/// decoder used to take each level's size from its first element and read the
/// row's values as that shape — past the end of them for `[[1, 2], [3]]`,
/// silently re-rowed for `[[1, 2], [3], [4, 5, 6]]` — and read a null as 0.
/// One row of a tensor sink: the row's buffer, in any layout. Its shape is
/// the row's shape in the column.
pub(crate) type TypedListRow = Option<ViewBuffer>;

/// The number of elements a row's buffer holds.
fn element_count(buf: &ViewBuffer) -> usize {
    buf.shape().iter().product()
}

// The tensor sinks (`list`, `array`) are built straight into Arrow: one flat
// primitive values buffer holding every row, wrapped in one offsets (list) or
// fixed-size (array) level per dimension, with the row nulls as the outermost
// validity. Each value is copied once, from the row's buffer — through its
// strides, whatever its layout — into the flat buffer.
//
// They never build one `AnyValue` per *element*: that is ~34-40x slower than
// the numpy sink, and an irregular column (every rank >= 2 list sink, any array
// sink with a null row) must not need it.
// There is no slow path to fall back to: an irregular row is either
// representable here (ragged list rows, null rows) or an error.

/// The element dtype a tensor sink column is built with.
///
/// The planner's dtype when it declared one; only for an unresolved one —
/// which `dtype_for_output` refuses for a planned typed sink, so only direct
/// callers of the executor reach it — the first row's. Every row must then
/// carry exactly this dtype (see [`flat_values`]).
fn element_dtype(rows: &[TypedListRow], dtype: PlannedDType) -> PolarsResult<DType> {
    let first_row = || rows.iter().find_map(|r| r.as_ref()).map(ViewBuffer::dtype);
    match dtype {
        PlannedDType::Known(dtype) => Ok(dtype),
        PlannedDType::Unknown | PlannedDType::SomeFloat => first_row().ok_or_else(|| {
            polars_err!(ComputeError:
                "a typed sink's element dtype was never planned ({}) and there is no \
                 row to take it from", dtype.as_str())
        }),
    }
}

/// Concatenate every row's values into one primitive Arrow array.
///
/// `null_fill` is the number of placeholder elements a null row occupies:
/// `None` for a list (a null row is a zero-length slot), the fixed element
/// count for an array (every slot has the same size, so a null row is a
/// validity bit over zeroed values).
///
/// A row whose values are not `dtype` is an error, never a cast: the column's
/// dtype is the plan's promise, and a row that breaks it is a contract bug.
fn flat_values(
    rows: &[TypedListRow],
    dtype: DType,
    null_fill: Option<usize>,
    split: Option<&Split>,
) -> PolarsResult<Box<dyn polars_arrow::array::Array>> {
    use polars_arrow::array::PrimitiveArray;
    // Checked before anything is written, so a bad row cannot leave the
    // values half-filled.
    for (i, row) in rows.iter().enumerate() {
        if let Some(other) = row.as_ref().filter(|data| data.dtype() != dtype) {
            polars_bail!(ComputeError: "{}", crate::row_ref::at_row(format!(
                "produced {} but the column was planned as {}. The planner's dtype \
                 contract disagrees with the Rust implementation.",
                other.dtype().short_name(), dtype.short_name()
            ), i, rows.len()));
        }
    }
    // Each row's place in the values: where it starts, and how many it takes
    // (a null row takes `null_fill` placeholders).
    let mut starts = Vec::with_capacity(rows.len() + 1);
    let mut total = 0usize;
    for row in rows {
        starts.push(total);
        total += match row {
            Some(data) => element_count(data),
            None => null_fill.unwrap_or(0),
        };
    }
    starts.push(total);
    macro_rules! flat {
        ($t:ty) => {{
            let mut flat: Vec<$t> = Vec::with_capacity(total);
            fill_rows::<$t>(
                rows,
                &starts,
                &mut flat.spare_capacity_mut()[..total],
                split,
            );
            // SAFETY: `fill_rows` wrote every one of the `total` slots.
            unsafe { flat.set_len(total) };
            Box::new(PrimitiveArray::<$t>::from_vec(flat)) as Box<dyn polars_arrow::array::Array>
        }};
    }
    Ok(match dtype {
        DType::U8 => flat!(u8),
        DType::I8 => flat!(i8),
        DType::U16 => flat!(u16),
        DType::I16 => flat!(i16),
        DType::U32 => flat!(u32),
        DType::I32 => flat!(i32),
        DType::U64 => flat!(u64),
        DType::I64 => flat!(i64),
        DType::F32 => flat!(f32),
        DType::F64 => flat!(f64),
    })
}

/// Write every row into its slot of `out` (`starts[i]..starts[i + 1]`): a
/// row's elements through its strides, once, or a null row's placeholders.
///
/// The rows are split as the call's rows were ([`Split::run`]), so the one
/// copy each row makes — a strided gather, for a transposed row — runs on the
/// pool, as it did when each row was copied while it was encoded.
fn fill_rows<T: view_buffer::core::ViewType + Default + Send>(
    rows: &[TypedListRow],
    starts: &[usize],
    out: &mut [MaybeUninit<T>],
    split: Option<&Split>,
) {
    /// `out`, shared by the row ranges. Each writes only its own rows'
    /// slots, which do not overlap.
    struct Slots<T>(*mut MaybeUninit<T>);
    // SAFETY: ranges write disjoint slots (below), so no slot is accessed by
    // two threads; the values written are moved to the owner of `out`, which
    // needs `T: Send`.
    unsafe impl<T: Send> Send for Slots<T> {}
    // SAFETY: as `Send`: sharing `&Slots` only lets threads form the pointer
    // of slots the `starts` partition gives each of them alone.
    unsafe impl<T: Send> Sync for Slots<T> {}
    impl<T> Slots<T> {
        /// Slot `start` of `out`. A method, so a closure captures the whole
        /// `Slots` (which is `Sync`) rather than its raw pointer.
        fn at(&self, start: usize) -> *mut MaybeUninit<T> {
            self.0.wrapping_add(start)
        }
    }
    let slots = Slots(out.as_mut_ptr());
    let write = |range: Range<usize>| {
        for i in range {
            // SAFETY: `starts` is non-decreasing and ends at `out.len()`, so
            // row `i`'s slot is in bounds and no other row's.
            let slot = unsafe {
                std::slice::from_raw_parts_mut(slots.at(starts[i]), starts[i + 1] - starts[i])
            };
            match &rows[i] {
                Some(data) => data.write_to(slot),
                None => slot.fill(MaybeUninit::new(T::default())),
            }
        }
    };
    match split {
        Some(split) => {
            split.run(rows.len(), |_, range| write(range));
        }
        None => write(0..rows.len()),
    }
}

/// Row validity as a bitmap, or `None` when no row is null.
fn row_validity(rows: &[TypedListRow]) -> Option<polars_arrow::bitmap::Bitmap> {
    if rows.iter().all(Option::is_some) {
        return None;
    }
    Some(rows.iter().map(Option::is_some).collect())
}

/// Build a typed list series from the planner's declared dtype and rank.
///
/// The `_with_dtype` suffix is historical: it distinguished this from a
/// sibling that inferred the dtype from the data, and that sibling is gone.
/// Nothing here infers anything — the element dtype and the nesting depth both
/// come from the `OutputSpec` the lazy schema was published from, which is the
/// only way the produced column can be guaranteed to match it. Rows may be
/// ragged (each carries its own shape) but every row must have the planned rank.
pub(super) fn build_typed_list_series_from_rows_with_dtype(
    name: PlSmallStr,
    rows: &[TypedListRow],
    dtype: PlannedDType,
    expected_shape: Option<&Vec<usize>>,
    expected_ndim: Option<usize>,
    split: Option<&Split>,
) -> PolarsResult<Series> {
    use polars_arrow::array::ListArray;
    use polars_arrow::offset::{Offsets, OffsetsBuffer};

    let dtype = element_dtype(rows, dtype)?;
    // `dtype_for_output` refuses a list sink whose rank it cannot name, so a
    // planned query always reaches here with one. The row fallback keeps a
    // direct (unplanned) caller working; only a genuinely rankless call fails.
    let ndim = expected_shape
        .map(|shape| shape.len())
        .or(expected_ndim)
        .or_else(|| {
            rows.iter()
                .find_map(|r| r.as_ref())
                .map(|b| b.shape().len())
        });
    let Some(ndim) = ndim.filter(|&n| n > 0) else {
        polars_bail!(ComputeError: "cannot build a list series without a known output rank");
    };
    for (i, row) in rows.iter().enumerate() {
        if let Some(data) = row {
            polars_ensure!(
                data.shape().len() == ndim,
                ComputeError:
                "{}", crate::row_ref::at_row(format!(
                    "shape {:?}, but the list column was planned with rank {}",
                    data.shape(), ndim
                ), i, rows.len())
            );
        }
    }

    // Innermost level first: level `k` holds, for every row, prod(shape[..k])
    // lists of length shape[k]. Level 0 is one list per row and carries the
    // row nulls.
    let mut array = flat_values(rows, dtype, None, split)?;
    for level in (0..ndim).rev() {
        let mut lengths: Vec<usize> = Vec::new();
        for row in rows {
            match row {
                Some(data) => {
                    let shape = data.shape();
                    let repeats: usize = shape[..level].iter().product();
                    lengths.extend(std::iter::repeat_n(shape[level], repeats));
                }
                None if level == 0 => lengths.push(0),
                None => {}
            }
        }
        let offsets: OffsetsBuffer<i64> =
            Offsets::<i64>::try_from_lengths(lengths.into_iter())?.into();
        let validity = if level == 0 { row_validity(rows) } else { None };
        let arrow_dtype = ListArray::<i64>::default_datatype(array.dtype().clone());
        array = Box::new(ListArray::<i64>::try_new(
            arrow_dtype,
            offsets,
            array,
            validity,
        )?);
    }
    Series::from_arrow(name, array)
}

/// Build a typed fixed-size array series from the planner's dtype and shape.
///
/// As above, the `_with_dtype` suffix names a distinction that no longer
/// exists. The shape comes from the sink or the `OutputSpec` and never from
/// the rows: a fixed-size column whose dimensions depend on which row arrived
/// first is exactly what this sink exists to rule out.
pub(super) fn build_typed_array_series_from_rows_with_dtype(
    name: PlSmallStr,
    rows: &[TypedListRow],
    dtype: PlannedDType,
    sink_shape: &Option<Vec<usize>>,
    expected_shape: Option<&Vec<usize>>,
    split: Option<&Split>,
) -> PolarsResult<Series> {
    use polars_arrow::array::FixedSizeListArray;

    // Spec first, and no data fallback: an `array` sink's whole point is a
    // fixed shape published at plan time. Taking it from the first non-null row
    // would make the column's dtype depend on which row happened to arrive
    // first — and `dtype_for_output` has already refused any array sink whose
    // shape it could not name, so a planned query always supplies one here.
    let shape = sink_shape.clone().or_else(|| expected_shape.cloned());
    let Some(shape) = shape.filter(|s| !s.is_empty()) else {
        // Not user-facing advice: `dtype_for_output` reads the same two fields
        // and refuses first, so reaching here means the schema half and the
        // encode half disagreed about the same `OutputSpec`. Restating the
        // "how to supply a shape" guidance here made this a second copy of it,
        // which would drift — and would tell the user to fix something they
        // cannot, because a query that got this far already passed the check.
        polars_bail!(ComputeError:
            "internal: the array sink reached encoding with no shape, which \
             dtype_for_output refuses. The schema and encode halves of the sink \
             contract disagree about this output."
        );
    };
    let dtype = element_dtype(rows, dtype)?;
    let expected_len: usize = shape.iter().product();
    for (i, row) in rows.iter().enumerate() {
        if let Some(data) = row {
            polars_ensure!(
                element_count(data) == expected_len,
                ComputeError:
                "{}", crate::row_ref::at_row(format!(
                    "{} values, but the array column was planned with shape {:?} ({} values)",
                    element_count(data), shape, expected_len
                ), i, rows.len())
            );
        }
    }

    // Innermost dimension first; level `k` holds rows * prod(shape[..k]) slots
    // of size shape[k]. Lengths are explicit so a zero-sized dimension works.
    let mut array = flat_values(rows, dtype, Some(expected_len), split)?;
    for level in (0..shape.len()).rev() {
        let length = rows.len() * shape[..level].iter().product::<usize>();
        let validity = if level == 0 { row_validity(rows) } else { None };
        let arrow_dtype = FixedSizeListArray::default_datatype(array.dtype().clone(), shape[level]);
        array = Box::new(FixedSizeListArray::try_new(
            arrow_dtype,
            length,
            array,
            validity,
        )?);
    }
    Series::from_arrow(name, array)
}
/// The buffer behind a node output, or an error naming what was there instead.
fn require_buffer<'a>(
    output: &'a NodeOutput,
    domain: &str,
    format: &str,
) -> Result<&'a ViewBuffer, String> {
    output.as_buffer().map(|b| &**b).ok_or_else(|| {
        format!(
            "the '{format}' sink planned a {domain} output, but execution produced \
             {:?}. This is a planner/executor disagreement, not a usage error.",
            output.domain()
        )
    })
}

/// Fixed-shape array encoding of a buffer, validated against the sink's shape.
/// The buffer is shared, not copied: its values are copied once, when the
/// column is built.
fn typed_array_of(
    buf: &ViewBuffer,
    spec_shape: Option<&Vec<usize>>,
) -> Result<OutputValue, String> {
    if let Some(s) = spec_shape.filter(|s| s.as_slice() != buf.shape()) {
        return Err(format!(
            "Array sink shape {s:?} does not match buffer shape {:?}. \
             Use squeeze() or expand_dims() to adjust dimensions, \
             or omit shape to infer from buffer.",
            buf.shape()
        ));
    }
    Ok(OutputValue::TypedArray(buf.clone()))
}

/// Encode a NodeOutput to an output value, keyed on the resolved [`SinkKind`].
///
/// `dtype_for_output` decides the Polars *dtype*; this
/// decides the *value*. Both halves of one contract, so both key on the same
/// resolved kind. They used to key on the `(expected_domain, format)` string
/// pair separately and "mirror each other one for one, deliberately, so a
/// reader can check the correspondence" — a correspondence whose only guard
/// was a reader, and which the two null/build halves did not keep at all.
///
/// This used to match on the `NodeOutput` *variant* instead, which is a
/// different fact: a domain can arrive in more than one representation. A
/// perceptual hash is a `vector`-domain output that rides as a `Buffer`
/// (`apply_perceptual_hash` returns a 1-D `u8` buffer), while `extract_shape`
/// produces a real `Vector`. Keying the two halves differently meant they
/// disagreed wherever those diverged, always as plan-says-one-thing,
/// execution-does-another:
///
/// - `perceptual_hash().sink("native")` planned `List(UInt8)` and failed with
///   "Buffer outputs require explicit format".
/// - `extract_shape().sink("array", shape=[3])` planned `Array(Float64, 3)`
///   and failed with "Unsupported sink format: array" — the schema arm for
///   `("vector", "array")` was added to fix an earlier divergence without the
///   encode arm that makes it real.
/// - The pairs that did work did so by coincidence of the two dispatches
///   agreeing, not by construction.
///
/// The `NodeOutput` variant is now used only to *get at the data*, which is
/// what it actually tells you.
pub(crate) fn encode_node_output(
    output: &NodeOutput,
    spec: &OutputSpec,
) -> Result<OutputValue, String> {
    let sink = &spec.sink;
    let format = sink.name();
    let domain = spec.expected_domain.name();
    let kind = SinkKind::resolve(spec).map_err(|e| e.to_string())?;

    match kind {
        SinkKind::HistogramBuckets => {
            let contig = require_buffer(output, domain, format)?.to_contiguous();
            Ok(OutputValue::HistogramBuckets(
                contig.as_slice::<f64>().to_vec(),
            ))
        }
        SinkKind::NumpyStruct | SinkKind::NdArray => {
            let buf = require_buffer(output, domain, format)?;
            // A half-precision sink converts here, on the row's thread,
            // not in the serial column build; the row then holds the f16
            // bits (`crate::output::NumpyRowOutput::from_f16_bits`). So
            // does a compacting one, whose row then holds only its elements
            // and is handed to the column as it is.
            let row = if sink.as_f16() {
                buf.to_f16_bits()
            } else {
                buf.clone()
            };
            Ok(OutputValue::NumpyStruct(if sink.compact() {
                row.compact()
            } else {
                row
            }))
        }
        SinkKind::EncodedImage | SinkKind::Blob => {
            crate::execute::encode_sink(require_buffer(output, domain, format)?, sink)
                .map(OutputValue::Binary)
                .map_err(|e| format!("Encode error: {e}"))
        }
        SinkKind::BufferList => Ok(OutputValue::TypedList(
            require_buffer(output, domain, format)?.clone(),
        )),
        SinkKind::BufferArray => typed_array_of(
            require_buffer(output, domain, format)?,
            sink.shape().as_ref(),
        ),
        // A vector arrives either as a real `Vector` or as the 1-D buffer a
        // hash/histogram produces. Both are the same domain to the planner, so
        // both encode the same way here.
        SinkKind::VectorList => match output {
            NodeOutput::Vector(vals) => Ok(OutputValue::Vector(vals.clone())),
            _ => Ok(OutputValue::TypedList(
                require_buffer(output, domain, format)?.clone(),
            )),
        },
        SinkKind::VectorArray => match output {
            NodeOutput::Vector(vals) => {
                let values = vals.as_ref().clone();
                let shape = sink.shape().unwrap_or_else(|| vec![values.len()]);
                let planned: usize = shape.iter().product();
                if planned != values.len() {
                    return Err(format!(
                        "Array sink shape {shape:?} holds {planned} elements but the \
                         vector has {}.",
                        values.len()
                    ));
                }
                Ok(OutputValue::TypedArray(
                    ViewBuffer::from_vec(values).reshape(shape),
                ))
            }
            _ => typed_array_of(
                require_buffer(output, domain, format)?,
                sink.shape().as_ref(),
            ),
        },
        SinkKind::Scalar => match output {
            NodeOutput::Scalar(val) => Ok(OutputValue::Scalar(*val)),
            _ => Err(format!(
                "the 'native' sink planned a scalar output, but execution produced {:?}.",
                output.domain()
            )),
        },
        SinkKind::Long => {
            let layout = super::encode::long_layout(spec);
            long_row(require_buffer(output, domain, format)?, &layout).map(OutputValue::Long)
        }
        SinkKind::Contours => match output {
            NodeOutput::Contours(contours) => Ok(OutputValue::Contours(contours.clone())),
            _ => Err(format!(
                "the 'native' sink planned a contour output, but execution produced {:?}.",
                output.domain()
            )),
        },
    }
}
/// A buffer's kept pixels as a `long` sink row: their positions, row-major,
/// and each value field's channel, gathered once on the row's thread.
fn long_row(buf: &ViewBuffer, layout: &LongLayout) -> Result<LongRow, String> {
    let (height, width, channels) = match *buf.shape() {
        [h, w] => (h, w, 1),
        [h, w, c] => (h, w, c),
        ref other => {
            return Err(format!(
                "the 'long' sink takes an [H, W] or [H, W, C] buffer, got shape {other:?}"
            ))
        }
    };
    layout.check_channels(channels)?;
    if u32::try_from(height.max(width)).is_err() {
        return Err(format!(
            "the 'long' sink's positions are u32, and this buffer is {height}x{width}"
        ));
    }
    let contig = buf.to_contiguous();
    macro_rules! gather {
        ($t:ty) => {{
            let data = contig.as_slice::<$t>();
            let picked: Vec<usize> = match layout.select {
                Some(s) => (0..height * width)
                    .filter(|p| data[p * channels + s] > <$t>::default())
                    .collect(),
                None => (0..height * width).collect(),
            };
            let fields = layout
                .fields()
                .map(|(c, _)| {
                    ViewBuffer::from_vec(
                        picked
                            .iter()
                            .map(|p| data[p * channels + c])
                            .collect::<Vec<$t>>(),
                    )
                })
                .collect();
            (picked, fields)
        }};
    }
    let (picked, fields): (Vec<usize>, Vec<ViewBuffer>) = match buf.dtype() {
        DType::U8 => gather!(u8),
        DType::I8 => gather!(i8),
        DType::U16 => gather!(u16),
        DType::I16 => gather!(i16),
        DType::U32 => gather!(u32),
        DType::I32 => gather!(i32),
        DType::U64 => gather!(u64),
        DType::I64 => gather!(i64),
        DType::F32 => gather!(f32),
        DType::F64 => gather!(f64),
    };
    // In range: both are below `height.max(width)`, checked above.
    Ok(LongRow {
        rows: picked.iter().map(|p| (p / width) as u32).collect(),
        cols: picked.iter().map(|p| (p % width) as u32).collect(),
        fields,
    })
}

/// The `long` sink's dtype: `List(Struct{row, col, <fields>})`, each value
/// field `value`.
pub(super) fn long_dtype(layout: &LongLayout, value: DataType) -> DataType {
    let mut fields = vec![
        Field::new("row".into(), DataType::UInt32),
        Field::new("col".into(), DataType::UInt32),
    ];
    fields.extend(
        layout
            .fields()
            .map(|(_, name)| Field::new(name.into(), value.clone())),
    );
    DataType::List(Box::new(DataType::Struct(fields)))
}

/// The `long` sink's column, built straight into Arrow: one struct per kept
/// pixel, one list per row; a null row is null, a row that kept no pixel `[]`.
pub(super) fn long_series(
    name: PlSmallStr,
    rows: &[Option<LongRow>],
    layout: &LongLayout,
    dtype: PlannedDType,
    split: Option<&Split>,
) -> PolarsResult<Series> {
    use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
    use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
    use polars_arrow::offset::Offsets;

    let positions = |pick: fn(&LongRow) -> &[u32]| -> Box<dyn Array> {
        let flat: Vec<u32> = rows.iter().flatten().flat_map(pick).copied().collect();
        PrimitiveArray::<u32>::from_vec(flat).boxed()
    };
    let mut values = vec![positions(|r| &r.rows), positions(|r| &r.cols)];
    for k in 0..layout.fields().count() {
        let field: Vec<TypedListRow> = rows
            .iter()
            .map(|r| r.as_ref().map(|r| r.fields[k].clone()))
            .collect();
        let dtype = element_dtype(&field, dtype)?;
        values.push(flat_values(&field, dtype, None, split)?);
    }
    let names = ["row", "col"]
        .into_iter()
        .chain(layout.fields().map(|(_, n)| n));
    let arrow_fields = names
        .zip(&values)
        .map(|(n, v)| ArrowField::new(n.into(), v.dtype().clone(), true))
        .collect();
    let length = rows.iter().flatten().map(|r| r.rows.len()).sum();
    let structs = StructArray::try_new(ArrowDataType::Struct(arrow_fields), length, values, None)?;
    let offsets = Offsets::<i64>::try_from_lengths(
        rows.iter().map(|r| r.as_ref().map_or(0, |r| r.rows.len())),
    )?;
    let validity = rows
        .iter()
        .any(Option::is_none)
        .then(|| rows.iter().map(Option::is_some).collect());
    let list = ListArray::<i64>::try_new(
        ListArray::<i64>::default_datatype(structs.dtype().clone()),
        offsets.into(),
        structs.boxed(),
        validity,
    )?;
    Series::from_arrow(name, list.boxed())
}

/// A `List[Contour]` column, one contour set per row, built straight into Arrow
/// (see [`crate::namespaces::schema::contour_array`]).
///
/// A null row (no input, or a failed row) is null; a row whose image simply
/// contains no contours is the empty set `[]`. The two used to be collapsed
/// into null, which made `list.len()` read null for "found nothing" and
/// disagreed with the `.contour` transforms, which keep `[]`.
pub(super) fn contour_set_series(
    name: PlSmallStr,
    rows: &[Option<Vec<Outline>>],
) -> PolarsResult<Series> {
    use polars_arrow::array::ListArray;
    use polars_arrow::offset::Offsets;

    // Each member written with its own `is_closed`: a line stays a line.
    let all: Vec<&Outline> = rows.iter().flatten().flatten().collect();
    let values = crate::namespaces::schema::contour_array(all.iter().copied())?;
    let lengths = rows.iter().map(|r| r.as_ref().map_or(0, Vec::len));
    let offsets = Offsets::<i64>::try_from_lengths(lengths)?;
    let validity = rows
        .iter()
        .any(Option::is_none)
        .then(|| rows.iter().map(Option::is_some).collect());
    let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
    let array = ListArray::<i64>::try_new(dtype, offsets.into(), values, validity)?;
    Series::from_arrow(name, array.boxed())
}

/// Convert flat f64 histogram buckets [lower_edge, upper_edge, count, normalized] to Polars List(Struct).
pub(super) fn histogram_buckets_to_polars_value(
    buckets: &[f64],
) -> PolarsResult<AnyValue<'static>> {
    if buckets.is_empty() {
        return Ok(AnyValue::Null);
    }
    let num_bins = buckets.len() / 4;
    let mut lowers = Vec::with_capacity(num_bins);
    let mut uppers = Vec::with_capacity(num_bins);
    let mut counts = Vec::with_capacity(num_bins);
    let mut norms = Vec::with_capacity(num_bins);

    for i in 0..num_bins {
        lowers.push(buckets[i * 4]);
        uppers.push(buckets[i * 4 + 1]);
        counts.push(buckets[i * 4 + 2] as u64);
        norms.push(buckets[i * 4 + 3]);
    }

    let lowers_s = Series::new("lower_edge".into(), lowers);
    let uppers_s = Series::new("upper_edge".into(), uppers);
    let counts_s = Series::new("count".into(), counts);
    let norms_s = Series::new("normalized".into(), norms);

    let struct_chunked = StructChunked::from_series(
        "".into(),
        num_bins,
        [&lowers_s, &uppers_s, &counts_s, &norms_s].iter().copied(),
    )?;

    let series = struct_chunked.into_series();
    Ok(AnyValue::List(series))
}

/// Shared histogram bucket struct dtype.
pub(super) fn histogram_struct_dtype() -> DataType {
    DataType::Struct(vec![
        Field::new("lower_edge".into(), DataType::Float64),
        Field::new("upper_edge".into(), DataType::Float64),
        Field::new("count".into(), DataType::UInt64),
        Field::new("normalized".into(), DataType::Float64),
    ])
}

/// The tensor sinks (`list`, `array`) are built straight from one flat values
/// buffer — never element by element through `AnyValue`, which made a rank-3
/// `list` sink ~34x, and an `array` sink with a single null row ~40x, slower
/// than the zero-copy numpy sink.
#[cfg(test)]
mod tensor_sink_tests {
    use super::{
        build_typed_array_series_from_rows_with_dtype,
        build_typed_list_series_from_rows_with_dtype, TypedListRow,
    };
    use polars::prelude::*;
    use view_buffer::ViewBuffer;

    fn u8_row(start: u8, shape: &[usize]) -> TypedListRow {
        let n: usize = shape.iter().product();
        Some(
            ViewBuffer::from_vec(
                (0..n)
                    .map(|i| start.wrapping_add(i as u8))
                    .collect::<Vec<u8>>(),
            )
            .reshape(shape.to_vec()),
        )
    }

    const EXPLODE: ExplodeOptions = ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    };

    /// Explode every nesting level, returning the flat leaf values.
    fn leaves(series: &Series, depth: usize) -> Vec<Option<u8>> {
        let mut s = series.clone();
        for _ in 0..depth {
            s = match s.dtype() {
                DataType::Array(_, _) => s.array().unwrap().explode(EXPLODE).unwrap(),
                _ => s.list().unwrap().explode(EXPLODE).unwrap(),
            };
        }
        s.u8().unwrap().iter().collect()
    }

    fn nested(dtype: DataType, depth: usize, array_dims: Option<&[usize]>) -> DataType {
        let mut dt = dtype;
        for level in (0..depth).rev() {
            dt = match array_dims {
                Some(dims) => DataType::Array(Box::new(dt), dims[level]),
                None => DataType::List(Box::new(dt)),
            };
        }
        dt
    }

    use view_buffer::{DType, PlannedDType};

    const U8: PlannedDType = PlannedDType::Known(DType::U8);
    const F32: PlannedDType = PlannedDType::Known(DType::F32);

    #[test]
    fn array_sink_with_a_null_row_keeps_values_and_the_null() {
        let shape = vec![2, 2, 3];
        let rows = vec![u8_row(0, &shape), None, u8_row(100, &shape)];
        let s = build_typed_array_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            U8,
            &Some(shape.clone()),
            None,
            None,
        )
        .unwrap();
        assert_eq!(s.dtype(), &nested(DataType::UInt8, 3, Some(&shape)));
        assert_eq!(s.len(), 3);
        assert_eq!(s.null_count(), 1);
        assert!(s.get(1).unwrap().is_null());
        let non_null = s.drop_nulls();
        let flat: Vec<u8> = leaves(&non_null, 3)
            .into_iter()
            .map(Option::unwrap)
            .collect();
        let expected: Vec<u8> = (0..12u8).chain(100..112u8).collect();
        assert_eq!(flat, expected);
    }

    #[test]
    fn list_sink_rank3_ragged_rows_with_a_null() {
        let rows = vec![u8_row(0, &[2, 1, 3]), None, u8_row(50, &[1, 2, 3])];
        let s = build_typed_list_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            U8,
            None,
            Some(3),
            None,
        )
        .unwrap();
        assert_eq!(s.dtype(), &nested(DataType::UInt8, 3, None));
        assert_eq!(s.len(), 3);
        assert!(s.get(1).unwrap().is_null());
        // Outer lengths are each row's own first dimension.
        let outer: Vec<Option<u32>> = s.list().unwrap().lst_lengths().iter().collect();
        assert_eq!(outer[0], Some(2));
        assert_eq!(outer[2], Some(1));
        let flat: Vec<u8> = leaves(&s.drop_nulls(), 3)
            .into_iter()
            .map(Option::unwrap)
            .collect();
        let expected: Vec<u8> = (0..6u8).chain(50..56u8).collect();
        assert_eq!(flat, expected);
    }

    #[test]
    fn list_sink_rank1_with_a_null() {
        let rows = vec![u8_row(1, &[3]), None, u8_row(9, &[2])];
        let s = build_typed_list_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            U8,
            None,
            Some(1),
            None,
        )
        .unwrap();
        assert_eq!(s.dtype(), &DataType::List(Box::new(DataType::UInt8)));
        assert!(s.get(1).unwrap().is_null());
        let flat: Vec<u8> = leaves(&s.drop_nulls(), 1)
            .into_iter()
            .map(Option::unwrap)
            .collect();
        assert_eq!(flat, vec![1, 2, 3, 9, 10]);
    }

    #[test]
    fn a_later_row_with_another_dtype_is_an_error_not_a_cast() {
        let rows = vec![
            u8_row(0, &[2]),
            Some(ViewBuffer::from_vec(vec![0.5f32, 1.5])),
        ];
        let list = build_typed_list_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            U8,
            None,
            Some(1),
            None,
        );
        assert!(list.is_err(), "list sink cast a f32 row to u8: {list:?}");
        let array = build_typed_array_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            U8,
            &Some(vec![2]),
            None,
            None,
        );
        assert!(
            array.is_err(),
            "array sink accepted a f32 row as u8: {array:?}"
        );
    }

    #[test]
    fn array_row_with_the_wrong_element_count_is_an_error() {
        let rows = vec![u8_row(0, &[2, 3]), u8_row(0, &[2, 2])];
        let r = build_typed_array_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            U8,
            &Some(vec![2, 3]),
            None,
            None,
        );
        assert!(r.is_err());
    }

    #[test]
    fn list_row_with_the_wrong_rank_is_an_error() {
        let rows = vec![u8_row(0, &[2, 3]), u8_row(0, &[6])];
        let r = build_typed_list_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            U8,
            None,
            Some(2),
            None,
        );
        assert!(r.is_err());
    }

    #[test]
    fn all_null_rows_keep_the_planned_nesting() {
        let rows: Vec<TypedListRow> = vec![None, None];
        let list = build_typed_list_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            F32,
            None,
            Some(3),
            None,
        )
        .unwrap();
        assert_eq!(list.dtype(), &nested(DataType::Float32, 3, None));
        assert_eq!(list.null_count(), 2);
        let shape = vec![2, 2, 1];
        let array = build_typed_array_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            F32,
            &Some(shape.clone()),
            None,
            None,
        )
        .unwrap();
        assert_eq!(array.dtype(), &nested(DataType::Float32, 3, Some(&shape)));
        assert_eq!(array.null_count(), 2);
    }
}

/// The contour sink builds its column straight into Arrow, and must
/// publish exactly what the per-contour `AnyValue` construction published.
#[cfg(test)]
mod contour_sink_tests {
    use crate::graph::encode::build_series_from_spec;
    use crate::graph::types::{OutputSpec, RowResult};
    use polars::prelude::*;
    use view_buffer::geometry::{Contour, Point};

    fn spec() -> OutputSpec {
        OutputSpec {
            node: "n".to_string(),
            sink: serde_json::from_value(serde_json::json!({"format": "native"})).unwrap(),
            expected_domain: view_buffer::ops::Domain::Contour,
            expected_dtype: view_buffer::PlannedDType::Unknown,
            expected_shape: None,
            expected_ndim: None,
            expected_channels: None,
            histogram_buckets: false,
        }
    }

    fn p(x: f64, y: f64) -> Point {
        Point::new(x, y)
    }

    fn rows() -> Vec<Option<Vec<Contour>>> {
        let with_hole = Contour::with_holes(
            vec![p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0), p(0.0, 10.0)],
            vec![
                vec![p(4.0, 4.0), p(6.0, 4.0), p(6.0, 6.0)],
                vec![p(1.0, 1.0), p(2.0, 1.0), p(2.0, 2.0), p(1.0, 2.0)],
            ],
        );
        let triangle = Contour::new(vec![p(20.0, 20.0), p(30.0, 20.0), p(30.0, 30.0)]);
        vec![
            Some(vec![with_hole, triangle.clone()]),
            None,
            Some(vec![]),
            Some(vec![triangle]),
        ]
    }

    /// The per-value construction, kept here as the oracle. An empty set is
    /// `[]` and only a null row is null.
    fn oracle(rows: &[Option<Vec<Contour>>]) -> Series {
        let element = crate::namespaces::schema::contour_struct_dtype();
        let values: Vec<AnyValue<'static>> = rows
            .iter()
            .map(|row| match row {
                Some(contours) => {
                    let items: Vec<AnyValue<'static>> = contours
                        .iter()
                        .map(crate::namespaces::contour::contour_to_anyvalue)
                        .collect();
                    AnyValue::List(
                        Series::from_any_values_and_dtype(
                            PlSmallStr::EMPTY,
                            &items,
                            &element,
                            true,
                        )
                        .unwrap(),
                    )
                }
                None => AnyValue::Null,
            })
            .collect();
        Series::from_any_values_and_dtype(
            "o".into(),
            &values,
            &DataType::List(Box::new(element)),
            true,
        )
        .unwrap()
    }

    #[test]
    fn contour_sink_matches_the_anyvalue_construction() {
        let rows = rows();
        let expected = oracle(&rows);
        let data: Vec<RowResult> = rows
            .into_iter()
            .map(|r| {
                RowResult::Contours(r.map(|v| {
                    v.into_iter()
                        .map(view_buffer::geometry::contour::Outline::Closed)
                        .collect()
                }))
            })
            .collect();
        let got = build_series_from_spec("o".into(), &spec(), vec![data], None).unwrap();
        assert_eq!(got.dtype(), expected.dtype());
        assert!(
            got.equals_missing(&expected),
            "got {got:?}\nexpected {expected:?}"
        );
        assert_eq!(
            got.is_null().iter().collect::<Vec<_>>(),
            vec![Some(false), Some(true), Some(false), Some(false)]
        );
    }
}
