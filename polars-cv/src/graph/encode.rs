//! Output encoding and geometry execution utilities.
//!
//! This module contains functions for:
//! - Encoding node outputs to various formats (numpy, png, list, array)
//! - Executing geometry operations (extract_contours, rasterize, transforms)
//! - Building typed list/array series from row data
//! - Converting contours to Polars representations

use polars::chunked_array::builder::ListPrimitiveChunkedBuilder;
use polars::prelude::*;
use view_buffer::geometry::{extract::extract_contours, rasterize::rasterize, Contour};
use view_buffer::ops::NodeOutput;
use view_buffer::{BinaryOp, GeometryOp, Op, ViewBuffer};

use crate::contour::contour_to_anyvalue;

use super::decode::dtype_str_to_polars;
use super::types::{OutputSpec, OutputValue, TypedBufferData};

/// Execute a geometry operation with typed domain dispatch.
///
/// This handles domain transitions like Buffer → Contour (extract_contours)
/// and Contour → Buffer (rasterize).
pub(crate) fn execute_geometry_op(
    input: NodeOutput,
    op: &GeometryOp,
) -> Result<NodeOutput, String> {
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
        } => {
            let contours = input
                .as_contours()
                .ok_or_else(|| "Rasterize requires Contour input".to_string())?;
            if contours.is_empty() {
                let mask = ViewBuffer::from_vec_with_shape(
                    vec![*background; (*height as usize) * (*width as usize)],
                    vec![*height as usize, *width as usize, 1],
                );
                Ok(NodeOutput::from_buffer(mask))
            } else {
                // Render all contours onto the same canvas by folding with max.
                let mut canvas = rasterize(&contours[0], *width, *height, *fill_value, *background);
                for c in &contours[1..] {
                    let overlay = rasterize(c, *width, *height, *fill_value, *background);
                    canvas = BinaryOp::Maximum.execute(&canvas, &overlay);
                }
                Ok(NodeOutput::from_buffer(canvas))
            }
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
macro_rules! impl_typed_list_builder {
    ($name:ident, $polars_type:ty, $extract:expr) => {
        fn $name(name: PlSmallStr, rows: &[TypedListRow]) -> PolarsResult<Series> {
            let mut builder = ListPrimitiveChunkedBuilder::<$polars_type>::new(
                name,
                rows.len(),
                64,
                <$polars_type>::get_static_dtype(),
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
macro_rules! impl_extract_as {
    ($name:ident, $target:ty, $variant:ident) => {
        #[allow(unreachable_patterns)]
        fn $name(data: &TypedBufferData) -> Vec<$target> {
            match data {
                TypedBufferData::$variant(v) => v.clone(),
                TypedBufferData::U8(v) => v.iter().map(|&x| x as $target).collect(),
                TypedBufferData::I8(v) => v.iter().map(|&x| x as $target).collect(),
                TypedBufferData::U16(v) => v.iter().map(|&x| x as $target).collect(),
                TypedBufferData::I16(v) => v.iter().map(|&x| x as $target).collect(),
                TypedBufferData::U32(v) => v.iter().map(|&x| x as $target).collect(),
                TypedBufferData::I32(v) => v.iter().map(|&x| x as $target).collect(),
                TypedBufferData::U64(v) => v.iter().map(|&x| x as $target).collect(),
                TypedBufferData::I64(v) => v.iter().map(|&x| x as $target).collect(),
                TypedBufferData::F32(v) => v.iter().map(|&x| x as $target).collect(),
                TypedBufferData::F64(v) => v.iter().map(|&x| x as $target).collect(),
            }
        }
    };
}
impl_extract_as!(extract_as_u8, u8, U8);
impl_extract_as!(extract_as_i8, i8, I8);
impl_extract_as!(extract_as_u16, u16, U16);
impl_extract_as!(extract_as_i16, i16, I16);
impl_extract_as!(extract_as_u32, u32, U32);
impl_extract_as!(extract_as_i32, i32, I32);
impl_extract_as!(extract_as_u64, u64, U64);
impl_extract_as!(extract_as_i64, i64, I64);
impl_extract_as!(extract_as_f32, f32, F32);
impl_extract_as!(extract_as_f64, f64, F64);
fn build_typed_list_u8(name: PlSmallStr, rows: &[TypedListRow]) -> PolarsResult<Series> {
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
/// Build a typed list series from the planner's declared dtype and rank.
///
/// The `_with_dtype` suffix is historical: it distinguished this from a
/// sibling that inferred the dtype from the data, and that sibling is gone.
/// Nothing here infers anything — the element dtype and the nesting depth both
/// come from the `OutputSpec` the lazy schema was published from, which is the
/// only way the produced column can be guaranteed to match it.
pub(super) fn build_typed_list_series_from_rows_with_dtype(
    name: PlSmallStr,
    rows: &[TypedListRow],
    dtype_str: &str,
    expected_shape: Option<&Vec<usize>>,
    expected_ndim: Option<usize>,
) -> PolarsResult<Series> {
    // The *spec* is the authority here, not the data.
    //
    // This function used to take the element dtype and the nesting depth from
    // the first non-null row, falling back to the spec only when every row was
    // null. That inverts the contract: `dtype_str`, `expected_shape` and
    // `expected_ndim` are what the planner already published in the lazy
    // schema, and a column built to match the data instead is exactly how a
    // query comes to collect to something other than what `collect_schema()`
    // promised. It also made the outcome depend on *where the nulls fall* —
    // an all-null column honoured the plan while the same pipeline with one
    // value in it did not.
    //
    // A row whose data contradicts the spec is a bug to surface, not to
    // follow, so the disagreement is an error rather than a silent
    // reinterpretation.
    let first_row = rows.iter().find_map(|r| r.as_ref());
    if let Some((data, _)) = first_row {
        if data.dtype_str() != dtype_str {
            return Err(polars_err!(
                ComputeError:
                "planned element dtype {} but execution produced {}. The planner's \
                 dtype contract disagrees with the Rust implementation.",
                dtype_str,
                data.dtype_str()
            ));
        }
    }

    let ndim = expected_shape
        .map(|shape| shape.len())
        .or(expected_ndim)
        .or_else(|| first_row.map(|(_, s)| s.len()));
    // `dtype_for_output` refuses a list sink whose rank it cannot name, so a
    // planned query always reaches here with one. The row fallback above keeps
    // any non-graph caller working; only a genuinely rankless call fails.
    let Some(ndim) = ndim else {
        return Err(polars_err!(
            ComputeError: "cannot build a list series without a known output rank"
        ));
    };

    if ndim > 1 {
        // The nested builder uses shape.len() for recursion depth; the actual
        // sizes only matter for non-null rows, which carry their own shape.
        let effective_shape = expected_shape
            .cloned()
            .or_else(|| first_row.map(|(_, s)| s.clone()))
            .unwrap_or_else(|| vec![0; ndim]);
        return build_typed_nested_list_series_from_rows_with_dtype(
            name,
            rows,
            dtype_str,
            &effective_shape,
        );
    }
    match dtype_str {
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
        // Not a fallback: building a u8 list for an unrecognised dtype would
        // reinterpret every element and hand back a plausible-looking wrong
        // answer. The dtype string is produced upstream by `dtype_table!`, so
        // reaching this arm means the two have drifted.
        other => Err(polars_err!(
            ComputeError: "unknown dtype {} when building a list series", other
        )),
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
    let inner_dtype = dtype_str_to_polars(dtype_str);
    let mut dtype = inner_dtype.clone();
    for _dim in shape.iter().rev() {
        dtype = DataType::List(Box::new(dtype));
    }
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
        return Ok(AnyValue::List(series));
    }
    let outer_dim = shape[0];
    let inner_shape = &shape[1..];
    let inner_size: usize = inner_shape.iter().product();
    let mut inner_values: Vec<AnyValue<'static>> = Vec::with_capacity(outer_dim);
    for i in 0..outer_dim {
        let start = i * inner_size;
        let end = start + inner_size;
        let inner_data = slice_typed_data(data, start, end);
        let inner_val = build_typed_nested_list_value(&inner_data, inner_shape)?;
        inner_values.push(inner_val);
    }
    let base_dtype = data.polars_dtype();
    let mut inner_dtype = base_dtype;
    for _dim in inner_shape.iter().rev() {
        inner_dtype = DataType::List(Box::new(inner_dtype));
    }
    let series =
        Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &inner_values, &inner_dtype, true)?;
    Ok(AnyValue::List(series))
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
    dtype_str: &str,
    sink_shape: &Option<Vec<usize>>,
    expected_shape: Option<&Vec<usize>>,
) -> PolarsResult<Series> {
    // Spec first, and no data fallback: an `array` sink's whole point is a
    // fixed shape published at plan time. Taking it from the first non-null row
    // would make the column's dtype depend on which row happened to arrive
    // first — and `dtype_for_output` has already refused any array sink whose
    // shape it could not name, so a planned query always supplies one here.
    let shape = sink_shape.clone().or_else(|| expected_shape.cloned());
    let Some(shape) = shape else {
        return Err(
            polars_err!(ComputeError: "Cannot determine shape for array sink. Provide shape via .sink(shape=[...]) or use .resize()/.assert_shape()."),
        );
    };
    // Fast path: every row present with exactly the target element count —
    // concatenate the flat values once and reshape into the nested Array.
    // The fallback builds one AnyValue per ELEMENT (a 224x224x3 tensor is
    // ~150k enum values plus recursive sub-Series per row), which dominated
    // sink time for tensor outputs.
    if let Some(series) = try_build_array_series_flat(name.clone(), rows, dtype_str, &shape)? {
        return Ok(series);
    }
    let inner_dtype = dtype_str_to_polars(dtype_str);
    let mut dtype = inner_dtype.clone();
    for &dim in shape.iter().rev() {
        dtype = DataType::Array(Box::new(dtype), dim);
    }
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

/// Flat construction of an Array-sink series: one values buffer + reshape.
///
/// Applies only when no row is null and every row's data length matches the
/// target shape's element count (any irregularity falls back to the
/// per-element `AnyValue` path, which handles nulls and per-row validation).
fn try_build_array_series_flat(
    name: PlSmallStr,
    rows: &[TypedListRow],
    dtype_str: &str,
    shape: &[usize],
) -> PolarsResult<Option<Series>> {
    let expected_len: usize = shape.iter().product();
    if rows.is_empty() || expected_len == 0 {
        return Ok(None);
    }
    let all_regular = rows.iter().all(|r| match r {
        Some((data, _)) => data.len() == expected_len && data.dtype_str() == dtype_str,
        None => false,
    });
    if !all_regular {
        return Ok(None);
    }

    macro_rules! concat_rows {
        ($variant:ident) => {{
            let mut flat = Vec::with_capacity(rows.len() * expected_len);
            for row in rows {
                let Some((TypedBufferData::$variant(values), _)) = row else {
                    return Ok(None);
                };
                flat.extend_from_slice(values);
            }
            Series::new(name, flat)
        }};
    }
    let first_variant = rows[0].as_ref().map(|(d, _)| d.dtype_str()).unwrap_or("");
    let flat_series = match first_variant {
        "u8" => concat_rows!(U8),
        "i8" => concat_rows!(I8),
        "u16" => concat_rows!(U16),
        "i16" => concat_rows!(I16),
        "u32" => concat_rows!(U32),
        "i32" => concat_rows!(I32),
        "u64" => concat_rows!(U64),
        "i64" => concat_rows!(I64),
        "f32" => concat_rows!(F32),
        "f64" => concat_rows!(F64),
        _ => return Ok(None),
    };

    let mut dims = Vec::with_capacity(shape.len() + 1);
    dims.push(ReshapeDimension::new(rows.len() as i64));
    dims.extend(shape.iter().map(|&d| ReshapeDimension::new(d as i64)));
    flat_series.reshape_array(&dims).map(Some)
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
    let outer_dim = shape[0];
    let inner_shape = &shape[1..];
    let inner_size: usize = inner_shape.iter().product();
    let mut inner_values: Vec<AnyValue<'static>> = Vec::with_capacity(outer_dim);
    for i in 0..outer_dim {
        let start = i * inner_size;
        let end = start + inner_size;
        let inner_data = slice_typed_data(data, start, end);
        let inner_val = build_typed_nested_array_value(&inner_data, inner_shape)?;
        inner_values.push(inner_val);
    }
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
///
/// # Panics
/// Panics if `start > end` or `end > data.len()`.
fn slice_typed_data(data: &TypedBufferData, start: usize, end: usize) -> TypedBufferData {
    let len = data.len();
    assert!(
        start <= end && end <= len,
        "slice_typed_data: bounds check failed: start={start}, end={end}, len={len}"
    );
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

/// Encode a NodeOutput to an output value, keyed on the **planned** domain.
///
/// `dtype_for_output` (graph/decode.rs) decides the Polars *dtype* from
/// `(expected_domain, format)`; this decides the *value*. Both halves of one
/// contract, so both read the same key — and the arms below mirror that
/// function's arms one for one, deliberately, so a reader can check the
/// correspondence.
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
    let format = sink.format.as_str();
    let domain = spec.expected_domain.as_str();

    // Encoding outranks the (domain, format) pair, exactly as it does in
    // `dtype_for_output`: histogram buckets are a vector-domain output with
    // their own struct schema.
    if spec.expected_encoding.as_deref() == Some("histogram_buckets") {
        let contig = require_buffer(output, domain, format)?.to_contiguous();
        return Ok(OutputValue::HistogramBuckets(
            contig.as_slice::<f64>().to_vec(),
        ));
    }

    match (domain, format) {
        ("buffer", "numpy" | "torch") => Ok(OutputValue::NumpyStruct(
            require_buffer(output, domain, format)?.clone(),
        )),
        ("buffer", "png" | "jpeg" | "webp" | "tiff" | "blob") => {
            crate::execute::encode_sink(require_buffer(output, domain, format)?, sink)
                .map(OutputValue::Binary)
                .map_err(|e| format!("Encode error: {e}"))
        }
        ("buffer", "list") => Ok(typed_list_of(require_buffer(output, domain, format)?)),
        ("buffer", "array") => {
            typed_array_of(require_buffer(output, domain, format)?, sink.shape.as_ref())
        }
        // A vector arrives either as a real `Vector` or as the 1-D buffer a
        // hash/histogram produces. Both are the same domain to the planner, so
        // both encode the same way here.
        ("vector", "native" | "list") => match output {
            NodeOutput::Vector(vals) => Ok(OutputValue::Vector(vals.clone())),
            _ => Ok(typed_list_of(require_buffer(output, domain, format)?)),
        },
        ("vector", "array") => match output {
            NodeOutput::Vector(vals) => {
                let values = vals.as_ref().clone();
                let shape = sink.shape.clone().unwrap_or_else(|| vec![values.len()]);
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
            _ => typed_array_of(require_buffer(output, domain, format)?, sink.shape.as_ref()),
        },
        ("scalar", "native") => match output {
            NodeOutput::Scalar(val) => Ok(OutputValue::Scalar(*val)),
            _ => Err(format!(
                "the 'native' sink planned a scalar output, but execution produced {:?}.",
                output.domain()
            )),
        },
        ("contour", "native") => match output {
            NodeOutput::Contours(contours) => Ok(OutputValue::Contours(contours.clone())),
            _ => Err(format!(
                "the 'native' sink planned a contour output, but execution produced {:?}.",
                output.domain()
            )),
        },
        // Mirrors `dtype_for_output`'s arm of the same name. Unreachable in
        // practice — Polars resolves the schema before executing, so this pair
        // is rejected there first — but kept so the two match arm for arm.
        ("buffer", "native") => Err(
            "'native' sink is not defined for buffer outputs; use an explicit \
             format (numpy, png, list, array, blob, ...)"
                .to_string(),
        ),
        _ => Err(format!(
            "Unsupported output combination: domain '{domain}' with sink format '{format}'"
        )),
    }
}
/// Convert contours to Polars AnyValue representation.
pub(super) fn contours_to_polars_value(contours: &[Contour]) -> PolarsResult<AnyValue<'static>> {
    if contours.is_empty() {
        return Ok(AnyValue::Null);
    }
    let contour_values: Vec<AnyValue<'static>> = contours.iter().map(contour_to_anyvalue).collect();
    let contour_series = Series::from_any_values_and_dtype(
        PlSmallStr::EMPTY,
        &contour_values,
        &contour_struct_dtype(),
        true,
    )?;
    Ok(AnyValue::List(contour_series))
}

/// Shared contour struct dtype used by native contour encoding.
pub(super) fn contour_struct_dtype() -> DataType {
    let point_dtype = DataType::Struct(vec![
        Field::new("x".into(), DataType::Float64),
        Field::new("y".into(), DataType::Float64),
    ]);
    let hole_dtype = DataType::List(Box::new(point_dtype.clone()));
    DataType::Struct(vec![
        Field::new("exterior".into(), DataType::List(Box::new(point_dtype))),
        Field::new("holes".into(), DataType::List(Box::new(hole_dtype))),
        Field::new("is_closed".into(), DataType::Boolean),
    ])
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

pub(crate) fn default_domain() -> String {
    "buffer".to_string()
}
pub(crate) fn default_dtype() -> String {
    "auto".to_string()
}
#[cfg(test)]
mod tests {
    use super::super::types::UnifiedGraph;
    use super::execute_geometry_op;

    /// Structural coverage: every geometry op the graph builder can construct
    /// via `resolve_op` must actually execute. This is the geometry analog of
    /// view-buffer's `apply_op_coverage` probe.
    ///
    /// `GeometryOp` now carries only variants the graph routes, so a variant
    /// `execute_geometry_op` cannot handle is a non-exhaustive-match compile
    /// error rather than a runtime string. What remains for this test is the
    /// other direction: that resolving and running each op *works*, and that the
    /// `probe_params` table lists exactly the geometry ops `resolve_op` produces,
    /// so registering a new one without a probe fails here rather than silently
    /// escaping coverage.
    #[test]
    fn every_graph_geometry_op_executes() {
        use crate::execute::{resolve_op, KNOWN_OPS};
        use crate::graph::step::GraphStep;
        use crate::params::{ParamCtx, ParamValue};
        use crate::pipeline::OpSpec;
        use serde_json::json;
        use std::collections::{BTreeSet, HashMap};
        use view_buffer::geometry::Contour;
        use view_buffer::ops::{Domain, NodeOutput};
        use view_buffer::ViewBuffer;

        // Representative params for every geometry-producing op.
        fn probe_params(op: &str) -> Option<Vec<(&'static str, serde_json::Value)>> {
            Some(match op {
                "contour_area" => vec![],
                "contour_perimeter" => vec![],
                "contour_centroid" => vec![],
                "contour_bounding_box" => vec![],
                "contour_convex_hull" => vec![],
                "contour_translate" => vec![("dx", json!(1.0)), ("dy", json!(2.0))],
                "contour_scale" => vec![("sx", json!(2.0)), ("sy", json!(2.0))],
                "contour_simplify" => vec![("tolerance", json!(0.5))],
                "extract_contours" => vec![],
                "rasterize" => vec![("width", json!(8)), ("height", json!(8))],
                _ => return None,
            })
        }

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

        let mut executed: BTreeSet<&str> = BTreeSet::new();
        for &op_name in KNOWN_OPS {
            let params: HashMap<String, ParamValue> = probe_params(op_name)
                .unwrap_or_default()
                .into_iter()
                .map(|(k, v)| (k.to_string(), ParamValue::Literal { value: v }))
                .collect();
            let spec = OpSpec {
                op: op_name.to_string(),
                params,
            };
            // Non-geometry ops may need params we didn't supply — not our concern.
            let step = match resolve_op(&spec, 0, &ParamCtx::empty()) {
                Ok(step) => step,
                Err(_) => continue,
            };
            let GraphStep::Geometry(geo) = step else {
                continue;
            };
            executed.insert(op_name);

            let input = if geo.input_domain() == Domain::Buffer {
                sample_buffer()
            } else {
                sample_contours()
            };
            if let Err(err) = execute_geometry_op(input, &geo) {
                panic!(
                    "graph op '{op_name}' resolves to GeometryOp::{geo:?} but does \
                     not execute: {err}"
                );
            }
        }

        // Ratchet: the probe table must match exactly the geometry ops that
        // `resolve_op` actually produces, so a newly-registered graph geometry
        // op cannot be added without a probe (and a removed one cannot leave a
        // stale probe behind).
        let probed: BTreeSet<&str> = KNOWN_OPS
            .iter()
            .copied()
            .filter(|n| probe_params(n).is_some())
            .collect();
        assert_eq!(
            probed, executed,
            "geometry probe table out of sync with the graph's geometry ops"
        );
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
        let order = graph.topological_order();
        assert!(order.contains(&"a".to_string()));
        assert!(order.contains(&"b".to_string()));
        let b_pos = order.iter().position(|x| x == "b").unwrap();
        let a_pos = order.iter().position(|x| x == "a").unwrap();
        assert!(b_pos > a_pos);
    }
}
