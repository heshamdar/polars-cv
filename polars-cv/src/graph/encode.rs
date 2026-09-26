//! Output encoding and geometry execution utilities.
//!
//! This module contains functions for:
//! - Encoding node outputs to various formats (numpy, png, list, array)
//! - Executing geometry operations (extract_contours, rasterize, transforms)
//! - Building typed list/array series from row data
//! - Converting contours to Polars representations

use polars::prelude::*;
use view_buffer::geometry::{extract::extract_contours, rasterize::rasterize, Contour};
use view_buffer::ops::NodeOutput;
use view_buffer::{DType, GeometryOp, Op, PlannedDType, ViewBuffer};

use super::sink_kind::SinkKind;
use super::types::{OutputSpec, OutputValue, TypedBufferData};
use crate::formats::Format as _;

/// Execute a geometry operation with typed domain dispatch.
///
/// This handles domain transitions like Buffer → Contour (extract_contours)
/// and Contour → Buffer (rasterize).
pub(crate) fn execute_geometry_op(
    input: NodeOutput,
    op: &GeometryOp,
) -> Result<NodeOutput, String> {
    let shapes: Vec<&[usize]> = input.as_buffer().map(|b| b.shape()).into_iter().collect();
    let dtypes: Vec<view_buffer::DType> =
        input.as_buffer().map(|b| b.dtype()).into_iter().collect();
    op.validate(&shapes, &dtypes)
        .map_err(|e| format!("{}: {e}", op.name()))?;
    let expected_domain = op.input_domain();
    let actual_domain = input.domain();
    if expected_domain != actual_domain {
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
            let buffer = input
                .as_buffer()
                .ok_or_else(|| "ExtractContours requires Buffer input".to_string())?;
            let contours = extract_contours(buffer, *mode, *method, *min_area);
            Ok(NodeOutput::from_contours(contours))
        }
        GeometryOp::Rasterize {
            fill_value,
            background,
            ..
        } => {
            let (height, width) = op
                .canvas()
                .ok_or_else(|| "rasterize reached execution without its canvas".to_string())?;
            let contours = input
                .as_contours()
                .ok_or_else(|| "Rasterize requires Contour input".to_string())?;
            // The whole set in one call: `rasterize` paints their union, which
            // is order-independent and honours an inverted fill/background pair.
            // Folding per-contour masks with `max` here did neither.
            Ok(NodeOutput::from_buffer(rasterize(
                contours,
                width,
                height,
                *fill_value,
                *background,
            )))
        }
        GeometryOp::Area { signed } => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Area requires Contour input".to_string())?;
            let areas: Vec<f64> = contours
                .iter()
                .map(|c| view_buffer::geometry::measures::area(c, *signed))
                .collect();
            Ok(NodeOutput::from_vector(areas))
        }
        GeometryOp::Perimeter => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Perimeter requires Contour input".to_string())?;
            let perimeters: Vec<f64> = contours
                .iter()
                .map(view_buffer::geometry::measures::perimeter)
                .collect();
            Ok(NodeOutput::from_vector(perimeters))
        }
        GeometryOp::Centroid => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Centroid requires Contour input".to_string())?;
            // Flat interleaved: [cx₀, cy₀, cx₁, cy₁, ...]
            let mut coords = Vec::with_capacity(contours.len() * 2);
            for c in contours.iter() {
                let pt = view_buffer::geometry::measures::centroid(c);
                coords.push(pt.x);
                coords.push(pt.y);
            }
            Ok(NodeOutput::from_vector(coords))
        }
        GeometryOp::BoundingBox => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "BoundingBox requires Contour input".to_string())?;
            // Flat interleaved: [x₀, y₀, w₀, h₀, x₁, y₁, w₁, h₁, ...]
            let mut coords = Vec::with_capacity(contours.len() * 4);
            for c in contours.iter() {
                match c.bounding_box() {
                    Some(bb) => coords.extend_from_slice(&[bb.x, bb.y, bb.width, bb.height]),
                    None => coords.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]),
                }
            }
            Ok(NodeOutput::from_vector(coords))
        }
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
    }
}
/// Helper type for list row data: (TypedBufferData, shape)
pub(crate) type TypedListRow = Option<(TypedBufferData, Vec<usize>)>;

// The tensor sinks (`list`, `array`) are built straight into Arrow: one flat
// primitive values buffer holding every row, wrapped in one offsets (list) or
// fixed-size (array) level per dimension, with the row nulls as the outermost
// validity. Each value is copied once, from the row into the flat buffer.
//
// They used to build one `AnyValue` per *element* whenever the column was not
// perfectly regular — every rank >= 2 list sink, and any array sink with a single
// null row — which made those sinks ~34-40x slower than the numpy sink (CR-33).
// There is no slow path to fall back to: an irregular row is either
// representable here (ragged list rows, null rows) or an error.

/// The element dtype a tensor sink column is built with.
///
/// The planner's dtype when it declared one; only for an unresolved one —
/// which `dtype_for_output` refuses for a planned typed sink, so only direct
/// callers of the executor reach it — the first row's. Every row must then
/// carry exactly this dtype (see [`flat_values`]).
fn element_dtype(rows: &[TypedListRow], dtype: PlannedDType) -> PolarsResult<DType> {
    let first_row = || rows.iter().find_map(|r| r.as_ref()).map(|(d, _)| d.dtype());
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
) -> PolarsResult<Box<dyn polars_arrow::array::Array>> {
    use polars_arrow::array::PrimitiveArray;
    let total: usize = rows
        .iter()
        .map(|r| match r {
            Some((data, _)) => data.len(),
            None => null_fill.unwrap_or(0),
        })
        .sum();
    macro_rules! flat {
        ($variant:ident, $t:ty) => {{
            let mut flat: Vec<$t> = Vec::with_capacity(total);
            for (i, row) in rows.iter().enumerate() {
                match row {
                    Some((TypedBufferData::$variant(values), _)) => flat.extend_from_slice(values),
                    Some((other, _)) => polars_bail!(ComputeError:
                        "row {} produced {} but the column was planned as {}. The \
                         planner's dtype contract disagrees with the Rust implementation.",
                        i, other.dtype_str(), dtype.short_name()
                    ),
                    None => flat.resize(flat.len() + null_fill.unwrap_or(0), <$t>::default()),
                }
            }
            Box::new(PrimitiveArray::<$t>::from_vec(flat)) as Box<dyn polars_arrow::array::Array>
        }};
    }
    Ok(match dtype {
        DType::U8 => flat!(U8, u8),
        DType::I8 => flat!(I8, i8),
        DType::U16 => flat!(U16, u16),
        DType::I16 => flat!(I16, i16),
        DType::U32 => flat!(U32, u32),
        DType::I32 => flat!(I32, i32),
        DType::U64 => flat!(U64, u64),
        DType::I64 => flat!(I64, i64),
        DType::F32 => flat!(F32, f32),
        DType::F64 => flat!(F64, f64),
    })
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
        .or_else(|| rows.iter().find_map(|r| r.as_ref()).map(|(_, s)| s.len()));
    let Some(ndim) = ndim.filter(|&n| n > 0) else {
        polars_bail!(ComputeError: "cannot build a list series without a known output rank");
    };
    for (i, row) in rows.iter().enumerate() {
        if let Some((data, shape)) = row {
            polars_ensure!(
                shape.len() == ndim && shape.iter().product::<usize>() == data.len(),
                ComputeError:
                "row {} has shape {:?} ({} values) but the list column was planned with rank {}",
                i, shape, data.len(), ndim
            );
        }
    }

    // Innermost level first: level `k` holds, for every row, prod(shape[..k])
    // lists of length shape[k]. Level 0 is one list per row and carries the
    // row nulls.
    let mut array = flat_values(rows, dtype, None)?;
    for level in (0..ndim).rev() {
        let mut lengths: Vec<usize> = Vec::new();
        for row in rows {
            match row {
                Some((_, shape)) => {
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
        if let Some((data, _)) = row {
            polars_ensure!(
                data.len() == expected_len,
                ComputeError:
                "row {} has {} values but the array column was planned with shape {:?} ({} values)",
                i, data.len(), shape, expected_len
            );
        }
    }

    // Innermost dimension first; level `k` holds rows * prod(shape[..k]) slots
    // of size shape[k]. Lengths are explicit so a zero-sized dimension works.
    let mut array = flat_values(rows, dtype, Some(expected_len))?;
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

/// `[H, W, …]`-shaped list encoding of a buffer.
fn typed_list_of(buf: &ViewBuffer) -> OutputValue {
    let contig = buf.to_contiguous();
    let shape = contig.shape().to_vec();
    OutputValue::TypedList {
        data: TypedBufferData::from_contiguous_buffer(&contig),
        shape,
    }
}

/// Fixed-shape array encoding of a buffer, validated against the sink's shape.
fn typed_array_of(
    buf: &ViewBuffer,
    spec_shape: Option<&Vec<usize>>,
) -> Result<OutputValue, String> {
    let contig = buf.to_contiguous();
    let buffer_shape = contig.shape().to_vec();
    let shape = match spec_shape {
        Some(s) if s != &buffer_shape => {
            return Err(format!(
                "Array sink shape {s:?} does not match buffer shape {buffer_shape:?}. \
                 Use squeeze() or expand_dims() to adjust dimensions, \
                 or omit shape to infer from buffer."
            ));
        }
        Some(s) => s.clone(),
        None => buffer_shape,
    };
    Ok(OutputValue::TypedArray {
        data: TypedBufferData::from_contiguous_buffer(&contig),
        shape,
    })
}

/// Encode a NodeOutput to an output value, keyed on the resolved [`SinkKind`].
///
/// `dtype_for_output` (graph/decode.rs) decides the Polars *dtype*; this
/// decides the *value*. Both halves of one contract, so both key on the same
/// resolved kind. They used to key on the `(expected_domain, format)` string
/// pair separately and "mirror each other one for one, deliberately, so a
/// reader can check the correspondence" — a correspondence whose only guard
/// was a reader, and which the two null/build halves in `decode.rs` did not
/// keep at all.
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
        SinkKind::NumpyStruct | SinkKind::NdArray => Ok(OutputValue::NumpyStruct(
            require_buffer(output, domain, format)?.clone(),
        )),
        SinkKind::EncodedImage | SinkKind::Blob => {
            crate::execute::encode_sink(require_buffer(output, domain, format)?, sink)
                .map(OutputValue::Binary)
                .map_err(|e| format!("Encode error: {e}"))
        }
        SinkKind::BufferList => Ok(typed_list_of(require_buffer(output, domain, format)?)),
        SinkKind::BufferArray => typed_array_of(
            require_buffer(output, domain, format)?,
            sink.shape().as_ref(),
        ),
        // A vector arrives either as a real `Vector` or as the 1-D buffer a
        // hash/histogram produces. Both are the same domain to the planner, so
        // both encode the same way here.
        SinkKind::VectorList => match output {
            NodeOutput::Vector(vals) => Ok(OutputValue::Vector(vals.clone())),
            _ => Ok(typed_list_of(require_buffer(output, domain, format)?)),
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
                Ok(OutputValue::TypedArray {
                    data: TypedBufferData::F64(values),
                    shape,
                })
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
        SinkKind::Contours => match output {
            NodeOutput::Contours(contours) => Ok(OutputValue::Contours(contours.clone())),
            _ => Err(format!(
                "the 'native' sink planned a contour output, but execution produced {:?}.",
                output.domain()
            )),
        },
    }
}
/// A `List[Contour]` column, one contour set per row, built straight into Arrow
/// (see [`crate::geom_schema::contour_array`]).
///
/// A null row (no input, or a failed row) is null; a row whose image simply
/// contains no contours is the empty set `[]`. The two used to be collapsed
/// into null, which made `list.len()` read null for "found nothing" and
/// disagreed with the `.contour` transforms, which keep `[]`.
pub(super) fn contour_set_series(
    name: PlSmallStr,
    rows: &[Option<Vec<Contour>>],
) -> PolarsResult<Series> {
    use polars_arrow::array::ListArray;
    use polars_arrow::offset::Offsets;

    let all: Vec<&Contour> = rows.iter().flatten().flatten().collect();
    let values = crate::geom_schema::contour_array(all.iter().copied())?;
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

#[cfg(test)]
mod tests {
    use super::super::types::UnifiedGraph;
    use super::execute_geometry_op;

    /// Structural coverage: every geometry op the graph builder can construct
    /// by resolving must actually execute. This is the geometry analog of
    /// view-buffer's `apply_op_coverage` probe.
    ///
    /// `GeometryOp` carries only variants the graph routes, so a variant
    /// `execute_geometry_op` cannot handle is a non-exhaustive-match compile
    /// error rather than a runtime string. What remains for this test is the
    /// other direction: that resolving and running each op *works*. Every
    /// registered op carries a sample, so a new geometry op is covered by
    /// registering it.
    #[test]
    fn every_graph_geometry_op_executes() {
        use crate::graph::step::GraphStep;
        use crate::params::ParamCtx;
        use view_buffer::geometry::Contour;
        use view_buffer::ops::{Domain, NodeOutput};
        use view_buffer::ViewBuffer;

        let sample_contours = || {
            NodeOutput::from_contours(vec![Contour::from_tuples(&[
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
            ])])
        };
        let sample_buffer = || {
            NodeOutput::from_buffer(ViewBuffer::from_vec_with_shape(
                vec![0u8, 255, 255, 0],
                vec![2, 2, 1],
            ))
        };

        let mut executed = 0;
        for op in crate::ops::TypedOp::samples() {
            let name = op.name();
            let step = op
                .resolve(0, &ParamCtx::empty())
                .expect("a registered sample resolves");
            if let GraphStep::Geometry(geo) = step {
                let input = if geo.input_domain() == Domain::Buffer {
                    sample_buffer()
                } else {
                    sample_contours()
                };
                if let Err(err) = execute_geometry_op(input, &geo) {
                    panic!("op '{name}' resolves to {geo:?} but does not execute: {err}");
                }
                executed += 1;
            }
        }
        // extract_contours, rasterize, four measures, four transforms.
        assert!(executed >= 10, "only {executed} geometry ops executed");
    }

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
                    "ops": []
                },
                "_node_1": {
                    "source": {"format": "blob"},
                    "ops": [],
                    "upstream": ["_node_0"]
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
                "a": {"source": {"format": "image_bytes"}, "ops": []},
                "b": {"source": {"format": "blob"}, "ops": [], "upstream": ["a"]}
            },
            "outputs": {
                "out_a": {"node": "a", "sink": {"format": "numpy"}},
                "out_b": {"node": "b", "sink": {"format": "png"}}
            },
            "column_bindings": {"a": 0}
        }"#;
        let graph = UnifiedGraph::from_json(json).unwrap();
        let order = graph.topological_order();
        assert!(order.contains(&"a".to_string()));
        assert!(order.contains(&"b".to_string()));
        let b_pos = order.iter().position(|x| x == "b").unwrap();
        let a_pos = order.iter().position(|x| x == "a").unwrap();
        assert!(b_pos > a_pos);
    }
}

/// The tensor sinks (`list`, `array`) are built straight from one flat values
/// buffer — never element by element through `AnyValue`, which made a rank-3
/// `list` sink ~34x, and an `array` sink with a single null row ~40x, slower
/// than the zero-copy numpy sink (CR-33).
#[cfg(test)]
mod tensor_sink_tests {
    use super::{
        build_typed_array_series_from_rows_with_dtype,
        build_typed_list_series_from_rows_with_dtype, TypedListRow,
    };
    use crate::graph::types::TypedBufferData;
    use polars::prelude::*;

    fn u8_row(start: u8, shape: &[usize]) -> TypedListRow {
        let n: usize = shape.iter().product();
        Some((
            TypedBufferData::U8((0..n).map(|i| start.wrapping_add(i as u8)).collect()),
            shape.to_vec(),
        ))
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
        let s = build_typed_list_series_from_rows_with_dtype("o".into(), &rows, U8, None, Some(3))
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
        let s = build_typed_list_series_from_rows_with_dtype("o".into(), &rows, U8, None, Some(1))
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
            Some((TypedBufferData::F32(vec![0.5, 1.5]), vec![2])),
        ];
        let list =
            build_typed_list_series_from_rows_with_dtype("o".into(), &rows, U8, None, Some(1));
        assert!(list.is_err(), "list sink cast a f32 row to u8: {list:?}");
        let array = build_typed_array_series_from_rows_with_dtype(
            "o".into(),
            &rows,
            U8,
            &Some(vec![2]),
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
        );
        assert!(r.is_err());
    }

    #[test]
    fn list_row_with_the_wrong_rank_is_an_error() {
        let rows = vec![u8_row(0, &[2, 3]), u8_row(0, &[6])];
        let r = build_typed_list_series_from_rows_with_dtype("o".into(), &rows, U8, None, Some(2));
        assert!(r.is_err());
    }

    #[test]
    fn all_null_rows_keep_the_planned_nesting() {
        let rows: Vec<TypedListRow> = vec![None, None];
        let list =
            build_typed_list_series_from_rows_with_dtype("o".into(), &rows, F32, None, Some(3))
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
        )
        .unwrap();
        assert_eq!(array.dtype(), &nested(DataType::Float32, 3, Some(&shape)));
        assert_eq!(array.null_count(), 2);
    }
}

/// The contour sink builds its column straight into Arrow (CR-36), and must
/// publish exactly what the per-contour `AnyValue` construction published.
#[cfg(test)]
mod contour_sink_tests {
    use crate::graph::decode::build_series_from_spec;
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
        let element = DataType::Struct(crate::geom_schema::contour_fields());
        let values: Vec<AnyValue<'static>> = rows
            .iter()
            .map(|row| match row {
                Some(contours) => {
                    let items: Vec<AnyValue<'static>> = contours
                        .iter()
                        .map(crate::contour::contour_to_anyvalue)
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
        let data: Vec<RowResult> = rows.into_iter().map(RowResult::Contours).collect();
        let got = build_series_from_spec("o".into(), &spec(), data).unwrap();
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
