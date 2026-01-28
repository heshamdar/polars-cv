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
use view_buffer::{GeometryOp, Op, ViewBuffer};

use crate::pipeline::{PipelineSpec, SinkSpec, SourceSpec};

use super::decode::dtype_str_to_polars;
use super::types::{OutputValue, TypedBufferData};

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
        return Ok(if data.is_empty() {
            AnyValue::Null
        } else {
            AnyValue::Float64(data[0])
        });
    }
    if shape.len() == 1 {
        let width = shape[0];
        if data.len() != width {
            return Err(polars_err!(
                ComputeError : "Data length {} doesn't match shape {:?}", data.len(),
                shape
            ));
        }
        let values: Vec<AnyValue<'static>> = data.iter().map(|&v| AnyValue::Float64(v)).collect();
        let inner_dtype = DataType::Float64;
        let series =
            Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &values, &inner_dtype, true)?;
        return Ok(AnyValue::Array(series, width));
    }
    let outer_dim = shape[0];
    let inner_shape = &shape[1..];
    let inner_size: usize = inner_shape.iter().product();
    if data.len() != outer_dim * inner_size {
        return Err(polars_err!(
            ComputeError : "Data length {} doesn't match shape {:?}", data.len(),
            shape
        ));
    }
    let mut inner_values: Vec<AnyValue<'static>> = Vec::with_capacity(outer_dim);
    for i in 0..outer_dim {
        let start = i * inner_size;
        let end = start + inner_size;
        let inner_data = &data[start..end];
        let inner_val = build_nested_array_value(inner_data, inner_shape)?;
        inner_values.push(inner_val);
    }
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
/// Helper type for list row data: (TypedBufferData, shape)
pub(crate) type TypedListRow = Option<(TypedBufferData, Vec<usize>)>;
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
pub(crate) fn encode_node_output(
    output: &NodeOutput,
    sink: &SinkSpec,
) -> Result<OutputValue, String> {
    let format = sink.format.as_str();
    match (output, format) {
        (NodeOutput::Buffer(buf), "numpy" | "torch") => {
            Ok(OutputValue::NumpyStruct((**buf).clone()))
        }
        (NodeOutput::Buffer(buf), "png" | "jpeg" | "webp" | "tiff" | "blob") => {
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
            let contig = buf.to_contiguous();
            let shape = contig.shape().to_vec();
            let data = TypedBufferData::from_buffer(&contig);
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
            let data = TypedBufferData::from_buffer(&contig);
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
    let contour = &contours[0];
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
    let point_dtype = DataType::Struct(vec![
        Field::new("x".into(), DataType::Float64),
        Field::new("y".into(), DataType::Float64),
    ]);
    let exterior_series =
        Series::from_any_values_and_dtype("exterior".into(), &points, &point_dtype, true)?;
    let contour_values = vec![AnyValue::List(exterior_series), AnyValue::Null];
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
pub(crate) fn default_domain() -> String {
    "buffer".to_string()
}
pub(crate) fn default_dtype() -> String {
    "u8".to_string()
}
#[cfg(test)]
mod tests {
    use super::super::types::UnifiedGraph;

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
