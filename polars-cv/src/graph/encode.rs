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
            anti_alias,
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
                let mut canvas = rasterize(
                    &contours[0],
                    *width,
                    *height,
                    *fill_value,
                    *background,
                    *anti_alias,
                );
                for c in &contours[1..] {
                    let overlay =
                        rasterize(c, *width, *height, *fill_value, *background, *anti_alias);
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
        // Contour transforms/measures/predicates that the graph path does not
        // execute. Their user-facing behaviour is served by the dedicated
        // `.contour` namespace (`src/contour.rs`), which calls the view-buffer
        // geometry helpers directly rather than through `GeometryOp`. This arm is
        // written out explicitly (rather than a catch-all `_`) so the match stays
        // exhaustive: adding a new `GeometryOp` variant is a compile error here
        // until it is either implemented above or classified as unsupported.
        GeometryOp::Winding
        | GeometryOp::IsConvex
        | GeometryOp::Flip
        | GeometryOp::Normalize { .. }
        | GeometryOp::ToAbsolute { .. }
        | GeometryOp::EnsureWinding { .. }
        | GeometryOp::ContainsPoint { .. }
        | GeometryOp::IoU
        | GeometryOp::Dice
        | GeometryOp::HausdorffDistance => Err(format!(
            "Geometry operation {} is not supported in the pipeline graph; \
             use the .contour namespace accessor instead",
            op.name()
        )),
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
/// Build a typed list series using a statically known dtype.
///
/// Unlike `build_typed_list_series_from_rows` which infers dtype from data,
/// this function uses the provided dtype string, allowing proper handling
/// of all-null data while preserving the expected output type.
///
/// If shapes indicate multi-dimensional data (shape.len() > 1), builds nested
/// List structures to preserve the shape information.
pub(super) fn build_typed_list_series_from_rows_with_dtype(
    name: PlSmallStr,
    rows: &[TypedListRow],
    dtype_str: &str,
    expected_shape: Option<&Vec<usize>>,
    expected_ndim: Option<usize>,
) -> PolarsResult<Series> {
    let first_row = rows.iter().find_map(|r| r.as_ref());
    let actual_dtype_str = first_row
        .map(|(data, _)| data.dtype_str())
        .unwrap_or(dtype_str);
    let shape = first_row
        .map(|(_, s)| s.clone())
        .or_else(|| expected_shape.cloned());

    // Determine if we need nested List structure
    let needs_nesting = shape.as_ref().map(|s| s.len() > 1).unwrap_or(false)
        || expected_ndim.map(|n| n > 1).unwrap_or(false);

    if needs_nesting {
        let ndim = shape
            .as_ref()
            .map(|s| s.len())
            .or(expected_ndim)
            .unwrap_or(1);
        // Use shape if available, otherwise synthesize a dummy shape with correct ndim.
        // The nested builder uses shape.len() for recursion depth; actual sizes only
        // matter for non-null rows (which carry their own shape).
        let effective_shape = shape.unwrap_or_else(|| vec![0; ndim]);
        return build_typed_nested_list_series_from_rows_with_dtype(
            name,
            rows,
            actual_dtype_str,
            &effective_shape,
        );
    }
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
        _ => build_typed_list_u8(name, rows),
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
/// Build a typed fixed-size array series using a statically known dtype.
///
/// Unlike `build_typed_array_series_from_rows` which infers dtype from data,
/// this function uses the provided dtype string and shape, allowing proper
/// handling of all-null data while preserving the expected output type.
pub(super) fn build_typed_array_series_from_rows_with_dtype(
    name: PlSmallStr,
    rows: &[TypedListRow],
    dtype_str: &str,
    sink_shape: &Option<Vec<usize>>,
    expected_shape: Option<&Vec<usize>>,
) -> PolarsResult<Series> {
    let shape = sink_shape
        .clone()
        .or_else(|| expected_shape.cloned())
        .or_else(|| rows.iter().find_map(|r| r.as_ref().map(|(_, s)| s.clone())));
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
/// Encode a NodeOutput to bytes based on sink format.
///
/// Dispatches to the appropriate encoding based on the output domain.
pub(crate) fn encode_node_output(
    output: &NodeOutput,
    spec: &OutputSpec,
) -> Result<OutputValue, String> {
    let sink = &spec.sink;
    let format = sink.format.as_str();
    match (output, format) {
        (NodeOutput::Buffer(buf), "numpy" | "torch") => {
            Ok(OutputValue::NumpyStruct((**buf).clone()))
        }
        (NodeOutput::Buffer(buf), "native")
            if spec.expected_encoding.as_deref() == Some("histogram_buckets") =>
        {
            let contig = buf.to_contiguous();
            Ok(OutputValue::HistogramBuckets(contig.as_slice::<f64>().to_vec()))
        }
        (NodeOutput::Buffer(buf), "png" | "jpeg" | "webp" | "tiff" | "blob") => {
            crate::execute::encode_sink(buf, sink)
                .map(OutputValue::Binary)
                .map_err(|e| format!("Encode error: {e}"))
        }
        (NodeOutput::Buffer(buf), "list") => {
            let contig = buf.to_contiguous();
            let shape = contig.shape().to_vec();
            let data = TypedBufferData::from_contiguous_buffer(&contig);
            Ok(OutputValue::TypedList {
                data,
                shape,
            })
        }
        (NodeOutput::Buffer(buf), "array") => {
            let contig = buf.to_contiguous();
            let buffer_shape = contig.shape().to_vec();
            let shape = if let Some(ref spec_shape) = sink.shape {
                if spec_shape != &buffer_shape {
                    return Err(
                        format!(
                            "Array sink shape {spec_shape:?} does not match buffer shape {buffer_shape:?}. \
                         Use squeeze() or expand_dims() to adjust dimensions, \
                         or omit shape to infer from buffer."
                        ),
                    );
                }
                spec_shape.clone()
            } else {
                buffer_shape
            };
            let data = TypedBufferData::from_contiguous_buffer(&contig);
            Ok(OutputValue::TypedArray {
                data,
                shape,
            })
        }
        (NodeOutput::Buffer(_), "native") => {
            Err(
                "Buffer outputs require explicit format (numpy/png/jpeg). Use 'native' for contours/scalars."
                    .to_string(),
            )
        }
        (NodeOutput::Contours(contours), "native") => {
            Ok(OutputValue::Contours(contours.clone()))
        }
        (NodeOutput::Scalar(val), "native") => Ok(OutputValue::Scalar(*val)),
        (NodeOutput::Vector(vals), "native") => Ok(OutputValue::Vector(vals.clone())),
        (NodeOutput::Vector(vals), "list") => Ok(OutputValue::Vector(vals.clone())),
        (NodeOutput::Contours(_), "numpy" | "png" | "jpeg" | "webp" | "tiff") => {
            Err(
                format!(
                    "Cannot encode Contours as {format}. Use 'native' or add .rasterize() first."
                ),
            )
        }
        (NodeOutput::Scalar(_), "numpy" | "png" | "jpeg" | "webp" | "tiff") => {
            Err(format!("Cannot encode Scalar as {format}. Use 'native' format."))
        }
        (NodeOutput::Vector(_), "numpy" | "png" | "jpeg" | "webp" | "tiff") => {
            Err(format!("Cannot encode Vector as {format}. Use 'native' format."))
        }
        _ => Err(format!("Unsupported sink format: {format}")),
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
    /// via `resolve_op` must be executable by `execute_geometry_op` — none may
    /// fall through to the "not supported in the pipeline graph" arm. This is
    /// the geometry analog of view-buffer's `apply_op_coverage` probe: it pins
    /// the class of bug where a `resolve_op` arm builds a `GeometryOp` variant
    /// that `execute_geometry_op` never handles (e.g. the former
    /// `Winding`/`IsConvex` gap that resolved but errored at runtime).
    ///
    /// The `probe_params` table is asserted to list *exactly* the geometry ops
    /// `resolve_op` produces, so registering a new graph geometry op without a
    /// probe here fails the test rather than silently escaping coverage.
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
                assert!(
                    !err.contains("not supported in the pipeline graph"),
                    "graph op '{op_name}' resolves to GeometryOp::{geo:?} but \
                     execute_geometry_op has no arm for it: {err}"
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
