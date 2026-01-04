//! Unified pipeline graph execution engine.
//!
//! This module handles the execution of pipeline graphs (DAGs) where multiple
//! pipelines can be composed and executed as a single fused operation.
//!
//! The graph executor:
//! - Parses a JSON graph specification
//! - Executes nodes in topological order
//! - Passes ViewBuffers between nodes without serialization
//! - Supports binary operations between nodes
//! - Returns Binary for single output ("_output") or Struct for multiple outputs
//! - Supports typed nodes with domain transitions (Buffer → Contour → Buffer)
//!
//! # Optimization Boundaries
//!
//! Each node in the graph represents an optimization boundary. Operations
//! within a node may be fused by view-buffer's optimizer (e.g., scalar ops),
//! but operations across different nodes are never fused. This ensures:
//!
//! - Output nodes produce exactly the buffer state at their alias point
//! - Shared subexpressions are computed once and reused
//! - No mutation safety issues since each node produces a new buffer
//!
//! # Typed Node Outputs
//!
//! The graph supports multiple data domains via [`NodeOutput`]:
//! - Buffer (images/arrays)
//! - Contours (geometry)
//! - Scalar (single values)
//! - Vector (multiple values)
//!
//! Domain transitions are validated at execution time, and the "native" sink
//! format dispatches to the appropriate encoding based on the output domain.

use polars::prelude::*;
use polars::chunked_array::builder::ListPrimitiveChunkedBuilder;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use view_buffer::geometry::{extract::extract_contours, rasterize::rasterize, Contour};
use view_buffer::ops::NodeOutput;
use view_buffer::{BinaryOp, GeometryOp, Op, ViewBuffer, ViewDto, ViewExpr};

use crate::execute::{
    decode_contour_source, decode_contour_source_with_dims, decode_source, resolve_op,
};
use crate::pipeline::{PipelineSpec, SinkSpec, SourceSpec};

// ============================================================
// Static Type Inference Helpers
// ============================================================

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
        _ => DataType::Float64, // Default fallback
    }
}

/// Get the Polars DataType for a given output specification.
///
/// Returns the appropriate dtype based on domain, sink format, and expected dtype.
pub fn dtype_for_output(spec: &OutputSpec) -> DataType {
    let format = spec.sink.format.as_str();
    let domain = spec.expected_domain.as_str();
    
    match (domain, format) {
        // Buffer domain
        ("buffer", "numpy" | "torch" | "png" | "jpeg" | "blob") => DataType::Binary,
        ("buffer", "list") => DataType::List(Box::new(dtype_str_to_polars(&spec.expected_dtype))),
        ("buffer", "array") => {
            // For array, we need shape info - for now return List
            // The actual array dtype is built at execution time with shape
            DataType::List(Box::new(dtype_str_to_polars(&spec.expected_dtype)))
        }
        
        // Scalar domain
        ("scalar", "native") => DataType::Float64,
        
        // Vector domain (perceptual hash, centroid, bbox)
        ("vector", "native" | "list") => {
            DataType::List(Box::new(dtype_str_to_polars(&spec.expected_dtype)))
        }
        
        // Contour domain
        ("contour", "native") => {
            // Return contour struct schema
            let point_dtype = DataType::Struct(vec![
                Field::new("x".into(), DataType::Float64),
                Field::new("y".into(), DataType::Float64),
            ]);
            DataType::Struct(vec![
                Field::new("exterior".into(), DataType::List(Box::new(point_dtype.clone()))),
                Field::new("interiors".into(), DataType::Null),
            ])
        }
        
        // Fallback
        _ => DataType::Binary,
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
/// - 255 values zero out the pixel
///
/// Uses normalized blending: pixel * (mask / 255)
fn apply_mask(buffer: &ViewBuffer, mask: &ViewBuffer, invert: bool) -> ViewBuffer {
    // Get shapes
    let buf_shape = buffer.shape();
    let mask_shape = mask.shape();

    // Handle broadcasting: mask might be 2D (H, W) while buffer is 3D (H, W, C)
    // We need to broadcast the mask to match the buffer's channels
    let effective_mask = if mask_shape.len() == 2 && buf_shape.len() == 3 {
        // Need to expand mask from (H, W) to (H, W, C)
        let h = mask_shape[0];
        let w = mask_shape[1];
        let c = buf_shape[2];

        let mask_contig = mask.to_contiguous();
        let mask_data = mask_contig.as_slice::<u8>();

        // Create expanded mask with inversion applied if needed
        let mut expanded: Vec<u8> = Vec::with_capacity(h * w * c);
        for y in 0..h {
            for x in 0..w {
                let raw_val = mask_data[y * w + x];
                let mask_val = if invert { 255 - raw_val } else { raw_val };
                // Replicate across channels
                for _ in 0..c {
                    expanded.push(mask_val);
                }
            }
        }

        ViewBuffer::from_vec_with_shape(expanded, vec![h, w, c])
    } else {
        // Same dimensionality - just use as-is, possibly inverting
        if invert {
            let mask_contig = mask.to_contiguous();
            let mask_data = mask_contig.as_slice::<u8>();
            let inverted: Vec<u8> = mask_data.iter().map(|&v| 255 - v).collect();
            ViewBuffer::from_vec_with_shape(inverted, mask_shape.to_vec())
        } else {
            mask.clone()
        }
    };

    // Apply the mask using normalized blend: pixel * (mask / 255)
    // BinaryOp::Blend computes: (a/255) * (b/255) * 255 = a * b / 255
    // This gives us the desired: pixel * (mask / 255)
    BinaryOp::Blend.execute(buffer, &effective_mask)
}

// ============================================================
// Typed Node Execution Helpers
// ============================================================

/// Execute a geometry operation with typed domain dispatch.
///
/// This handles domain transitions like Buffer → Contour (extract_contours)
/// and Contour → Buffer (rasterize).
fn execute_geometry_op(
    input: NodeOutput,
    op: &GeometryOp,
) -> Result<NodeOutput, String> {
    // Validate input domain
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
        GeometryOp::ExtractContours { mode, method, min_area } => {
            // Buffer → Contour
            let buffer = input.as_buffer()
                .ok_or_else(|| "ExtractContours requires Buffer input".to_string())?;
            let contours = extract_contours(buffer, *mode, *method, *min_area);
            Ok(NodeOutput::from_contours(contours))
        }
        
        GeometryOp::Rasterize { width, height, fill_value, background, anti_alias } => {
            // Contour → Buffer
            let contours = input.as_contours()
                .ok_or_else(|| "Rasterize requires Contour input".to_string())?;
            
            // Rasterize the first contour (primary contour for mask operations)
            if contours.is_empty() {
                // Empty contours → empty mask (all background)
                let mask = ViewBuffer::from_vec_with_shape(
                    vec![*background; (*height as usize) * (*width as usize)],
                    vec![*height as usize, *width as usize, 1],
                );
                Ok(NodeOutput::from_buffer(mask))
            } else {
                // Use the first contour
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
        
        // Contour → Scalar measures
        GeometryOp::Area { signed } => {
            let contours = input.as_contours()
                .ok_or_else(|| "Area requires Contour input".to_string())?;
            let area = if contours.is_empty() {
                0.0
            } else {
                view_buffer::geometry::measures::area(&contours[0], *signed)
            };
            Ok(NodeOutput::from_scalar(area))
        }
        
        GeometryOp::Perimeter => {
            let contours = input.as_contours()
                .ok_or_else(|| "Perimeter requires Contour input".to_string())?;
            let perimeter = if contours.is_empty() {
                0.0
            } else {
                view_buffer::geometry::measures::perimeter(&contours[0])
            };
            Ok(NodeOutput::from_scalar(perimeter))
        }
        
        GeometryOp::Centroid => {
            let contours = input.as_contours()
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
            let contours = input.as_contours()
                .ok_or_else(|| "BoundingBox requires Contour input".to_string())?;
            let bbox = if contours.is_empty() || contours[0].bounding_box().is_none() {
                vec![0.0, 0.0, 0.0, 0.0]
            } else {
                let bb = contours[0].bounding_box().unwrap();
                vec![bb.x, bb.y, bb.width, bb.height]
            };
            Ok(NodeOutput::from_vector(bbox))
        }
        
        // Contour → Contour transforms
        GeometryOp::Translate { dx, dy } => {
            let contours = input.as_contours()
                .ok_or_else(|| "Translate requires Contour input".to_string())?;
            let translated: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::translate(c, *dx, *dy))
                .collect();
            Ok(NodeOutput::from_contours(translated))
        }
        
        GeometryOp::Scale { sx, sy, origin } => {
            let contours = input.as_contours()
                .ok_or_else(|| "Scale requires Contour input".to_string())?;
            let scaled: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::scale(c, *sx, *sy, *origin))
                .collect();
            Ok(NodeOutput::from_contours(scaled))
        }
        
        GeometryOp::Flip => {
            let contours = input.as_contours()
                .ok_or_else(|| "Flip requires Contour input".to_string())?;
            let flipped: Vec<Contour> = contours
                .iter()
                .map(view_buffer::geometry::transforms::flip)
                .collect();
            Ok(NodeOutput::from_contours(flipped))
        }
        
        GeometryOp::Simplify { tolerance } => {
            let contours = input.as_contours()
                .ok_or_else(|| "Simplify requires Contour input".to_string())?;
            let simplified: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::simplify(c, *tolerance))
                .collect();
            Ok(NodeOutput::from_contours(simplified))
        }
        
        GeometryOp::ConvexHull => {
            let contours = input.as_contours()
                .ok_or_else(|| "ConvexHull requires Contour input".to_string())?;
            let hulls: Vec<Contour> = contours
                .iter()
                .map(view_buffer::geometry::transforms::convex_hull)
                .collect();
            Ok(NodeOutput::from_contours(hulls))
        }
        
        GeometryOp::Normalize { ref_width, ref_height } => {
            let contours = input.as_contours()
                .ok_or_else(|| "Normalize requires Contour input".to_string())?;
            let normalized: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::normalize(c, *ref_width, *ref_height))
                .collect();
            Ok(NodeOutput::from_contours(normalized))
        }
        
        GeometryOp::ToAbsolute { ref_width, ref_height } => {
            let contours = input.as_contours()
                .ok_or_else(|| "ToAbsolute requires Contour input".to_string())?;
            let absolute: Vec<Contour> = contours
                .iter()
                .map(|c| view_buffer::geometry::transforms::to_absolute(c, *ref_width, *ref_height))
                .collect();
            Ok(NodeOutput::from_contours(absolute))
        }
        
        // For other geometry ops, return an error for now
        // These can be implemented as needed
        _ => Err(format!("Geometry operation {} not yet implemented for typed execution", op.name()))
    }
}

/// Build a nested Array AnyValue from flat data and shape.
///
/// For shape [2, 3], builds Array[Array[f64, 3], 2] structure.
#[allow(dead_code)]
fn build_nested_array_value(data: &[f64], shape: &[usize]) -> PolarsResult<AnyValue<'static>> {
    if shape.is_empty() {
        // Scalar case
        return Ok(if data.is_empty() {
            AnyValue::Null
        } else {
            AnyValue::Float64(data[0])
        });
    }
    
    if shape.len() == 1 {
        // Base case: 1D array -> List of Float64
        let width = shape[0];
        if data.len() != width {
            return Err(polars_err!(ComputeError: "Data length {} doesn't match shape {:?}", data.len(), shape));
        }
        
        // Create a Float64 array
        let values: Vec<AnyValue<'static>> = data.iter().map(|&v| AnyValue::Float64(v)).collect();
        let inner_dtype = DataType::Float64;
        let series = Series::from_any_values_and_dtype(
            PlSmallStr::EMPTY,
            &values,
            &inner_dtype,
            true,
        )?;
        return Ok(AnyValue::Array(series, width));
    }
    
    // Multi-dimensional: recursively build nested arrays
    let outer_dim = shape[0];
    let inner_shape = &shape[1..];
    let inner_size: usize = inner_shape.iter().product();
    
    if data.len() != outer_dim * inner_size {
        return Err(polars_err!(ComputeError: "Data length {} doesn't match shape {:?}", data.len(), shape));
    }
    
    // Build each inner array
    let mut inner_values: Vec<AnyValue<'static>> = Vec::with_capacity(outer_dim);
    for i in 0..outer_dim {
        let start = i * inner_size;
        let end = start + inner_size;
        let inner_data = &data[start..end];
        let inner_val = build_nested_array_value(inner_data, inner_shape)?;
        inner_values.push(inner_val);
    }
    
    // Build inner dtype
    let mut inner_dtype = DataType::Float64;
    for &dim in inner_shape.iter().rev() {
        inner_dtype = DataType::Array(Box::new(inner_dtype), dim);
    }
    
    let series = Series::from_any_values_and_dtype(
        PlSmallStr::EMPTY,
        &inner_values,
        &inner_dtype,
        true,
    )?;
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

// ============================================================
// Dtype-Preserving List/Array Builders
// ============================================================

/// Helper type for list row data: (TypedBufferData, shape)
type TypedListRow = Option<(TypedBufferData, Vec<usize>)>;

/// Build a typed list series from row results, preserving the original buffer dtype.
///
/// This creates a List column with the correct inner dtype (UInt8, Float32, etc.)
/// instead of always using Float64.
fn build_typed_list_series_from_rows(
    name: PlSmallStr,
    rows: &[TypedListRow],
) -> PolarsResult<Series> {
    // Get dtype from first non-null element
    let first_typed = rows.iter().find_map(|r| r.as_ref().map(|(d, _)| d));
    
    let Some(first) = first_typed else {
        // All nulls - return list column of nulls with UInt8 inner (reasonable default)
        let mut builder = ListPrimitiveChunkedBuilder::<UInt8Type>::new(
            name,
            rows.len(),
            0,
            DataType::UInt8,
        );
        for _ in 0..rows.len() {
            builder.append_null();
        }
        return Ok(builder.finish().into_series());
    };
    
    // Match on dtype and build with appropriate builder type
    match first {
        TypedBufferData::U8(_) => build_typed_list_u8(name, rows),
        TypedBufferData::I8(_) => build_typed_list_i8(name, rows),
        TypedBufferData::U16(_) => build_typed_list_u16(name, rows),
        TypedBufferData::I16(_) => build_typed_list_i16(name, rows),
        TypedBufferData::U32(_) => build_typed_list_u32(name, rows),
        TypedBufferData::I32(_) => build_typed_list_i32(name, rows),
        TypedBufferData::U64(_) => build_typed_list_u64(name, rows),
        TypedBufferData::I64(_) => build_typed_list_i64(name, rows),
        TypedBufferData::F32(_) => build_typed_list_f32(name, rows),
        TypedBufferData::F64(_) => build_typed_list_f64(name, rows),
    }
}

// Macro to generate typed list builders
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
    // Build UInt8 list using the proper builder
    // Requires dtype-u8 feature in polars
    let mut builder = ListPrimitiveChunkedBuilder::<UInt8Type>::new(
        name,
        rows.len(),
        64,
        DataType::UInt8,
    );
    
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

/// Build a typed fixed-size array series from row results.
fn build_typed_array_series_from_rows(
    name: PlSmallStr,
    rows: &[TypedListRow],
) -> PolarsResult<Series> {
    // Get shape and dtype from first non-null element
    let first_result = rows.iter().find_map(|r| r.as_ref());
    
    let Some((first_data, first_shape)) = first_result else {
        // All nulls - return list column of nulls with UInt8 inner (reasonable default)
        let mut builder = ListPrimitiveChunkedBuilder::<UInt8Type>::new(
            name,
            rows.len(),
            0,
            DataType::UInt8,
        );
        for _ in 0..rows.len() {
            builder.append_null();
        }
        return Ok(builder.finish().into_series());
    };
    
    let shape = first_shape.clone();
    let inner_dtype = first_data.polars_dtype();
    
    // Build the nested Array type from shape
    let mut dtype = inner_dtype.clone();
    for &dim in shape.iter().rev() {
        dtype = DataType::Array(Box::new(dtype), dim);
    }
    
    // Build AnyValue arrays for each row
    let values: PolarsResult<Vec<AnyValue<'static>>> = rows.iter().map(|r| {
        if let Some((typed_data, row_shape)) = r {
            build_typed_nested_array_value(typed_data, row_shape)
        } else {
            Ok(AnyValue::Null)
        }
    }).collect();
    let values = values?;
    
    Series::from_any_values_and_dtype(name, &values, &dtype, true)
}

/// Build a nested Array AnyValue from typed data and shape.
fn build_typed_nested_array_value(data: &TypedBufferData, shape: &[usize]) -> PolarsResult<AnyValue<'static>> {
    if shape.is_empty() {
        return Ok(AnyValue::Null);
    }
    
    if shape.len() == 1 {
        // Base case: 1D array
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
        
        let series = Series::from_any_values_and_dtype(
            PlSmallStr::EMPTY,
            &values,
            &inner_dtype,
            true,
        )?;
        return Ok(AnyValue::Array(series, width));
    }
    
    // Multi-dimensional: recursively build nested arrays
    let outer_dim = shape[0];
    let inner_shape = &shape[1..];
    let inner_size: usize = inner_shape.iter().product();
    
    // Slice the data for each inner dimension
    let mut inner_values: Vec<AnyValue<'static>> = Vec::with_capacity(outer_dim);
    for i in 0..outer_dim {
        let start = i * inner_size;
        let end = start + inner_size;
        
        let inner_data = slice_typed_data(data, start, end);
        let inner_val = build_typed_nested_array_value(&inner_data, inner_shape)?;
        inner_values.push(inner_val);
    }
    
    // Build inner dtype
    let base_dtype = data.polars_dtype();
    let mut inner_dtype = base_dtype;
    for &dim in inner_shape.iter().rev() {
        inner_dtype = DataType::Array(Box::new(inner_dtype), dim);
    }
    
    let series = Series::from_any_values_and_dtype(
        PlSmallStr::EMPTY,
        &inner_values,
        &inner_dtype,
        true,
    )?;
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
fn encode_node_output(
    output: &NodeOutput,
    sink: &SinkSpec,
) -> Result<OutputValue, String> {
    let format = sink.format.as_str();
    
    match (output, format) {
        // Buffer outputs
        (NodeOutput::Buffer(buf), "numpy" | "torch") => {
            let pipeline = PipelineSpec {
                source: SourceSpec {
                    format: "blob".to_string(),
                    dtype: None,
                    width: None,
                    height: None,
                    fill_value: 255,
                    background: 0,
                    shape_pipeline: None,
                },
                shape_hints: None,
                ops: vec![],
                sink: sink.clone(),
            };
            crate::execute::encode_sink(buf, &pipeline)
                .map(OutputValue::Binary)
                .map_err(|e| format!("Encode error: {e}"))
        }
        (NodeOutput::Buffer(buf), "png" | "jpeg" | "blob") => {
            let pipeline = PipelineSpec {
                source: SourceSpec {
                    format: "blob".to_string(),
                    dtype: None,
                    width: None,
                    height: None,
                    fill_value: 255,
                    background: 0,
                    shape_pipeline: None,
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
            // Convert buffer to typed list structure preserving buffer dtype
            let contig = buf.to_contiguous();
            let shape = contig.shape().to_vec();
            
            // Extract data with original dtype preserved
            let data = TypedBufferData::from_buffer(&contig);
            
            Ok(OutputValue::TypedList { data, shape })
        }
        (NodeOutput::Buffer(buf), "array") => {
            // Convert buffer to typed fixed-size array preserving buffer dtype
            let contig = buf.to_contiguous();
            let buffer_shape = contig.shape().to_vec();
            
            // Use provided shape from sink spec, or infer from buffer
            let shape = if let Some(ref spec_shape) = sink.shape {
                // Require exact shape match to avoid dimension confusion
                if spec_shape != &buffer_shape {
                    return Err(format!(
                        "Array sink shape {spec_shape:?} does not match buffer shape {buffer_shape:?}. \
                         Use squeeze() or expand_dims() to adjust dimensions, \
                         or omit shape to infer from buffer."
                    ));
                }
                spec_shape.clone()
            } else {
                // Infer shape from buffer
                buffer_shape
            };
            
            // Extract data with original dtype preserved
            let data = TypedBufferData::from_buffer(&contig);
            
            Ok(OutputValue::TypedArray { data, shape })
        }
        
        // Native format dispatches based on domain
        (NodeOutput::Buffer(_), "native") => {
            Err("Buffer outputs require explicit format (numpy/png/jpeg). Use 'native' for contours/scalars.".to_string())
        }
        (NodeOutput::Contours(contours), "native") => {
            // Return as contour struct data
            Ok(OutputValue::Contours(contours.clone()))
        }
        (NodeOutput::Scalar(val), "native") => {
            Ok(OutputValue::Scalar(*val))
        }
        (NodeOutput::Vector(vals), "native") => {
            Ok(OutputValue::Vector(vals.clone()))
        }
        // List format for vector outputs
        (NodeOutput::Vector(vals), "list") => {
            Ok(OutputValue::Vector(vals.clone()))
        }
        
        // Type mismatches
        (NodeOutput::Contours(_), "numpy" | "png" | "jpeg") => {
            Err(format!(
                "Cannot encode Contours as {format}. Use 'native' or add .rasterize() first."
            ))
        }
        (NodeOutput::Scalar(_), "numpy" | "png" | "jpeg") => {
            Err(format!(
                "Cannot encode Scalar as {format}. Use 'native' format."
            ))
        }
        (NodeOutput::Vector(_), "numpy" | "png" | "jpeg") => {
            Err(format!(
                "Cannot encode Vector as {format}. Use 'native' format."
            ))
        }
        
        _ => Err(format!("Unsupported sink format: {format}"))
    }
}

/// Typed buffer data for dtype-preserving list/array outputs.
#[derive(Debug, Clone)]
enum TypedBufferData {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl TypedBufferData {
    /// Extract typed data from a ViewBuffer, preserving its dtype.
    fn from_buffer(buf: &ViewBuffer) -> Self {
        let contig = buf.to_contiguous();
        match contig.dtype() {
            view_buffer::DType::U8 => TypedBufferData::U8(contig.as_slice::<u8>().to_vec()),
            view_buffer::DType::I8 => TypedBufferData::I8(contig.as_slice::<i8>().to_vec()),
            view_buffer::DType::U16 => TypedBufferData::U16(contig.as_slice::<u16>().to_vec()),
            view_buffer::DType::I16 => TypedBufferData::I16(contig.as_slice::<i16>().to_vec()),
            view_buffer::DType::U32 => TypedBufferData::U32(contig.as_slice::<u32>().to_vec()),
            view_buffer::DType::I32 => TypedBufferData::I32(contig.as_slice::<i32>().to_vec()),
            view_buffer::DType::U64 => TypedBufferData::U64(contig.as_slice::<u64>().to_vec()),
            view_buffer::DType::I64 => TypedBufferData::I64(contig.as_slice::<i64>().to_vec()),
            view_buffer::DType::F32 => TypedBufferData::F32(contig.as_slice::<f32>().to_vec()),
            view_buffer::DType::F64 => TypedBufferData::F64(contig.as_slice::<f64>().to_vec()),
        }
    }
    
    /// Get the Polars DataType for this typed data.
    fn polars_dtype(&self) -> DataType {
        match self {
            TypedBufferData::U8(_) => DataType::UInt8,
            TypedBufferData::I8(_) => DataType::Int8,
            TypedBufferData::U16(_) => DataType::UInt16,
            TypedBufferData::I16(_) => DataType::Int16,
            TypedBufferData::U32(_) => DataType::UInt32,
            TypedBufferData::I32(_) => DataType::Int32,
            TypedBufferData::U64(_) => DataType::UInt64,
            TypedBufferData::I64(_) => DataType::Int64,
            TypedBufferData::F32(_) => DataType::Float32,
            TypedBufferData::F64(_) => DataType::Float64,
        }
    }
}

/// Output value from encoding - can be binary, contour struct, scalar, or array.
#[derive(Debug, Clone)]
enum OutputValue {
    Binary(Vec<u8>),
    Contours(Arc<Vec<Contour>>),
    Scalar(f64),
    Vector(Arc<Vec<f64>>),
    /// Typed list representation for "list" sink - preserves buffer dtype.
    TypedList {
        /// Typed data preserving original buffer dtype.
        data: TypedBufferData,
        /// Original shape of the buffer.
        shape: Vec<usize>,
    },
    /// Typed fixed-size array representation for "array" sink.
    TypedArray {
        /// Typed data preserving original buffer dtype.
        data: TypedBufferData,
        /// Fixed shape (validated against buffer).
        shape: Vec<usize>,
    },
}

/// Convert contours to Polars AnyValue representation.
fn contours_to_polars_value(contours: &[Contour]) -> PolarsResult<AnyValue<'static>> {
    if contours.is_empty() {
        // Return null for empty contours
        return Ok(AnyValue::Null);
    }
    
    // Use the first contour (primary contour)
    let contour = &contours[0];
    
    // Build exterior points as List of Struct {x: f64, y: f64}
    let points: Vec<AnyValue<'static>> = contour.exterior.iter()
        .map(|p| {
            let values = vec![AnyValue::Float64(p.x), AnyValue::Float64(p.y)];
            let fields = vec![
                Field::new("x".into(), DataType::Float64),
                Field::new("y".into(), DataType::Float64),
            ];
            AnyValue::StructOwned(Box::new((values, fields)))
        })
        .collect();
    
    // Create exterior as List
    let point_dtype = DataType::Struct(vec![
        Field::new("x".into(), DataType::Float64),
        Field::new("y".into(), DataType::Float64),
    ]);
    let exterior_series = Series::from_any_values_and_dtype(
        "exterior".into(),
        &points,
        &point_dtype,
        true,
    )?;
    
    // Build the contour struct: {exterior: List<{x, y}>, interiors: null}
    let contour_values = vec![
        AnyValue::List(exterior_series),
        AnyValue::Null, // interiors (holes) - not yet implemented
    ];
    let contour_fields = vec![
        Field::new("exterior".into(), DataType::List(Box::new(point_dtype.clone()))),
        Field::new("interiors".into(), DataType::Null),
    ];
    
    Ok(AnyValue::StructOwned(Box::new((contour_values, contour_fields))))
}

/// A node in the pipeline graph.
#[derive(Debug, Deserialize)]
pub struct GraphNode {
    /// Source specification for this node's input.
    pub source: SourceSpec,
    /// Operations to apply.
    #[serde(default)]
    pub ops: Vec<crate::pipeline::OpSpec>,
    /// Upstream node IDs this node depends on.
    #[serde(default)]
    pub upstream: Vec<String>,
    /// Optional user-defined alias for multi-output.
    /// Note: Used for deserialization; alias becomes the key in outputs map.
    #[serde(default)]
    #[allow(dead_code)]
    pub alias: Option<String>,
}

/// Output specification for a single output in the graph.
#[derive(Debug, Deserialize)]
pub struct OutputSpec {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: SinkSpec,
    /// Expected output domain for validation and type inference.
    #[serde(default = "default_domain")]
    pub expected_domain: String,
    /// Expected output dtype for list/array sinks.
    #[serde(default = "default_dtype")]
    pub expected_dtype: String,
}

fn default_domain() -> String {
    "buffer".to_string()
}

fn default_dtype() -> String {
    "u8".to_string()
}

/// Unified pipeline graph specification.
///
/// This struct handles all cases:
/// - Single output: `outputs` contains only "_output" key, returns Binary
/// - Multi output: `outputs` contains multiple keys, returns Struct
#[derive(Debug, Deserialize)]
pub struct UnifiedGraph {
    /// Named nodes in the graph.
    pub nodes: HashMap<String, GraphNode>,
    /// Output specifications (alias -> spec).
    /// Single output uses "_output" as key.
    pub outputs: HashMap<String, OutputSpec>,
    /// Mapping from node IDs to input column indices.
    /// Only root nodes (no upstream) have bindings.
    #[serde(default)]
    pub column_bindings: HashMap<String, usize>,
    /// Cached topological order (computed once during parsing).
    /// Not serialized - computed on load.
    #[serde(skip)]
    cached_order: Vec<String>,
}

impl UnifiedGraph {
    /// Parse a graph from JSON.
    ///
    /// This also computes and caches the topological order for efficient
    /// repeated execution.
    pub fn from_json(json: &str) -> PolarsResult<Self> {
        let mut graph: Self = serde_json::from_str(json)
            .map_err(|e| polars_err!(ComputeError: "Failed to parse pipeline graph: {}", e))?;
        
        // Pre-compute and cache the topological order
        graph.cached_order = graph.compute_topological_order()?;
        
        Ok(graph)
    }

    /// Check if this is a single-output graph (returns Binary instead of Struct).
    pub fn is_single_output(&self) -> bool {
        self.outputs.len() == 1 && self.outputs.contains_key("_output")
    }

    /// Get all output node IDs.
    #[allow(dead_code)]
    pub fn output_node_ids(&self) -> HashSet<String> {
        self.outputs.values().map(|s| s.node.clone()).collect()
    }

    /// Get cached topological order.
    /// The order is computed once during parsing and reused for all executions.
    fn topological_order(&self) -> &[String] {
        &self.cached_order
    }

    /// Compute nodes in topological order (dependencies first).
    /// Includes all nodes reachable from any output.
    fn compute_topological_order(&self) -> PolarsResult<Vec<String>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();

        fn dfs(
            node_id: &str,
            nodes: &HashMap<String, GraphNode>,
            visited: &mut HashSet<String>,
            order: &mut Vec<String>,
        ) -> PolarsResult<()> {
            if visited.contains(node_id) {
                return Ok(());
            }

            visited.insert(node_id.to_string());

            if let Some(node) = nodes.get(node_id) {
                for upstream_id in &node.upstream {
                    dfs(upstream_id, nodes, visited, order)?;
                }
            }

            order.push(node_id.to_string());
            Ok(())
        }

        // Start from all output nodes
        for spec in self.outputs.values() {
            dfs(&spec.node, &self.nodes, &mut visited, &mut order)?;
        }

        Ok(order)
    }

    /// Execute the graph on input series.
    ///
    /// Returns:
    /// - Binary column if single output ("_output" only)
    /// - Struct column with named fields (Binary/Float64/Struct) if multiple outputs
    ///
    /// # Optimizations
    ///
    /// 1. **Per-node precompilation**: Nodes where all op params are literals
    ///    have their ViewDtos resolved once before the row loop and reused.
    /// 2. **Batch-level panic catching**: A single catch_unwind wraps the
    ///    entire batch for reduced overhead vs per-row catching.
    /// 3. **Cached topological order**: Computed once during from_json().
    ///
    /// # Typed Node Support
    ///
    /// The executor now handles typed nodes via `NodeOutput`, supporting:
    /// - Buffer (images/arrays) → Binary encoding
    /// - Contours (geometry) → Struct encoding with "native" format
    /// - Scalar (single values) → Float64 with "native" format
    /// - Vector (multiple values) → List/Struct with "native" format
    pub fn execute(
        &self,
        inputs: &[Series],
        expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<Series> {
        // Get cached topological order
        let order = self.topological_order();

        // Get length from first input
        let len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            return Err(polars_err!(ComputeError: "No input columns provided"));
        };

        // Get output aliases in deterministic order
        let mut output_aliases: Vec<&String> = self.outputs.keys().collect();
        output_aliases.sort();

        // ============================================================
        // OPTIMIZATION: Per-node precompilation
        // ============================================================
        // For nodes where all op params are literals, precompile ViewDtos
        // once and reuse for all rows. This avoids repeated parameter
        // resolution in the hot loop.
        let precompiled: HashMap<String, Vec<ViewDto>> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.ops.iter().all(|op| op.is_all_literal()))
            .filter_map(|(node_id, node)| {
                // Resolve ops with row_idx=0 and empty expr_columns (all literal anyway)
                let ops: Result<Vec<ViewDto>, _> = node
                    .ops
                    .iter()
                    .map(|op| resolve_op(op, 0, &HashMap::new()))
                    .collect();
                ops.ok().map(|v| (node_id.clone(), v))
            })
            .collect();

        // Prepare result storage for typed outputs
        // Each output can be Binary, Scalar, or Contour data
        #[derive(Clone)]
        enum RowResult {
            Binary(Option<Vec<u8>>),
            Scalar(Option<f64>),
            Vector(Option<Vec<f64>>),
            Contours(Option<Vec<Contour>>),
            /// Typed list for "list" sink (variable length, preserves dtype).
            TypedList(Option<(TypedBufferData, Vec<usize>)>),
            /// Typed fixed-size array for "array" sink (fixed shape, preserves dtype).
            TypedArray(Option<(TypedBufferData, Vec<usize>)>),
        }
        
        let mut results: HashMap<String, Vec<RowResult>> = HashMap::new();
        for alias in &output_aliases {
            results.insert((*alias).clone(), Vec::with_capacity(len));
        }

        // ============================================================
        // OPTIMIZATION: Batch-level panic catching
        // ============================================================
        // Wrap the entire row loop in a single catch_unwind to reduce
        // the overhead of setting up unwinding machinery per-row.
        let batch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for row_idx in 0..len {
                // Node output cache for this row (typed outputs)
                let mut node_outputs: HashMap<String, NodeOutput> = HashMap::new();

                // Execute nodes in order
                for node_id in order {
                    let node = match self.nodes.get(node_id) {
                        Some(n) => n,
                        None => continue, // Skip missing nodes (shouldn't happen)
                    };

                    // Determine input source for this node
                    // A node is a "root" if it has a column binding (reads from DataFrame column)
                    // Nodes can have both column bindings AND upstream (e.g., contour with shape inference)
                    let has_column_binding = self.column_bindings.contains_key(node_id);
                    let node_input: Option<NodeOutput> = if has_column_binding {
                        // Root node: get input from column binding
                        let col_idx = self.column_bindings.get(node_id).copied().unwrap_or(0);

                        if col_idx >= inputs.len() {
                            // Return error as None to indicate failure
                            return Err(format!(
                                "Column index {col_idx} out of bounds for node '{node_id}'"
                            ));
                        }

                        let input_series = &inputs[col_idx];

                        // Check if this is a contour source (Struct input) vs binary source
                        let source_format = node.source.format.as_str();
                        if source_format == "contour" {
                            // Contour source: parse struct and rasterize
                            match input_series.get(row_idx) {
                                Ok(value) if !value.is_null() => {
                                    // Check if we have shape_pipeline for dimension inference
                                    if let Some(ref shape_pipeline) = node.source.shape_pipeline {
                                        // Extract node_id from shape_pipeline JSON
                                        let shape_node_id = shape_pipeline
                                            .get("node_id")
                                            .and_then(|v| v.as_str())
                                            .ok_or_else(|| {
                                                "shape_pipeline missing 'node_id'".to_string()
                                            })?;

                                        // Look up the referenced output
                                        let shape_output = node_outputs.get(shape_node_id).ok_or_else(|| {
                                            format!(
                                                "Shape reference '{shape_node_id}' not found. Ensure the shape source is defined before this contour pipeline."
                                            )
                                        })?;
                                        
                                        // Get buffer from output
                                        let shape_buffer = shape_output.as_buffer().ok_or_else(|| {
                                            format!("Shape reference '{shape_node_id}' must be a Buffer, not {:?}", shape_output.domain())
                                        })?;

                                        // Get dimensions from buffer shape (HWC layout: [height, width, channels])
                                        let shape = shape_buffer.shape();
                                        if shape.len() < 2 {
                                            return Err(format!(
                                                "Shape buffer has invalid dimensions: expected at least 2D, got {}D",
                                                shape.len()
                                            ));
                                        }
                                        let height = shape[0] as u32;
                                        let width = shape[1] as u32;

                                        // Get fill and background values
                                        let fill_value = node.source.fill_value;
                                        let background = node.source.background;

                                        match decode_contour_source_with_dims(
                                            &value, width, height, fill_value, background,
                                        ) {
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => return Err(format!("Contour decode error: {e}")),
                                        }
                                    } else {
                                        // Use explicit width/height parameters
                                        let first_output = self.outputs.values().next().unwrap();
                                        let temp_spec = PipelineSpec {
                                            source: node.source.clone(),
                                            shape_hints: None,
                                            ops: vec![],
                                            sink: first_output.sink.clone(),
                                        };
                                        match decode_contour_source(
                                            &value,
                                            row_idx,
                                            &temp_spec,
                                            expr_columns,
                                        ) {
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => return Err(format!("Contour decode error: {e}")),
                                        }
                                    }
                                }
                                _ => None,
                            }
                        } else {
                            // Binary source: decode from bytes
                            let input_ca = match input_series.binary() {
                                Ok(ca) => ca,
                                Err(_) => {
                                    return Err(format!(
                                        "Expected Binary column for node '{node_id}'"
                                    ))
                                }
                            };

                            match input_ca.get(row_idx) {
                                Some(bytes) => {
                                    // Create temp spec for decoding
                                    let first_output = self.outputs.values().next().unwrap();
                                    let temp_spec = PipelineSpec {
                                        source: node.source.clone(),
                                        shape_hints: None,
                                        ops: vec![],
                                        sink: first_output.sink.clone(),
                                    };
                                    // Copy the bytes to avoid any lifetime issues
                                    let bytes_owned = bytes.to_vec();
                                    match decode_source(&bytes_owned, &temp_spec) {
                                        Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                        Err(e) => return Err(format!("Decode error: {e}")),
                                    }
                                }
                                None => None,
                            }
                        }
                    } else {
                        // Non-root node: get input from upstream node's output
                        let upstream_id = &node.upstream[0];
                        node_outputs.get(upstream_id).cloned()
                    };

                    if let Some(input) = node_input {
                        // Get ViewDtos - use precompiled if available, otherwise resolve per-row
                        let view_dtos: Vec<ViewDto> = if let Some(cached) = precompiled.get(node_id)
                        {
                            // Fast path: clone precompiled ops
                            cached.clone()
                        } else {
                            // Slow path: resolve per-row for expression parameters
                            let mut dtos = Vec::with_capacity(node.ops.len());
                            for op_spec in &node.ops {
                                match resolve_op(op_spec, row_idx, expr_columns) {
                                    Ok(dto) => dtos.push(dto),
                                    Err(e) => return Err(format!("Op resolution error: {e}")),
                                }
                            }
                            dtos
                        };

                        // Execute operations with typed dispatch
                        // OPTIMIZATION: Batch consecutive buffer ops into a single ViewExpr
                        // to allow view-buffer's optimizer to fuse operations.
                        let mut current_output = input;

                        // Helper: flush pending buffer ops
                        fn flush_buffer_ops(
                            output: NodeOutput,
                            pending_ops: &mut Vec<ViewDto>,
                        ) -> Result<NodeOutput, String> {
                            if pending_ops.is_empty() {
                                return Ok(output);
                            }
                            let buf = output.as_buffer()
                                .ok_or_else(|| format!("Expected Buffer for pending ops, got {:?}", output.domain()))?;
                            let mut expr = ViewExpr::new_source((**buf).clone());
                            for op in pending_ops.drain(..) {
                                expr = expr.apply_op(op);
                            }
                            let result = expr.plan().execute();
                            Ok(NodeOutput::from_buffer(result))
                        }

                        let mut pending_buffer_ops: Vec<ViewDto> = Vec::new();

                        for view_dto in view_dtos {
                            match &view_dto {
                                ViewDto::Geometry(geo_op) => {
                                    // Flush pending buffer ops first
                                    current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    // Use typed geometry execution
                                    current_output = execute_geometry_op(current_output, geo_op)?;
                                }
                                ViewDto::Binary { op, other_node_id } => {
                                    // Flush pending buffer ops first
                                    current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    // Binary operation: both inputs must be buffers
                                    let current_buf = current_output.as_buffer()
                                        .ok_or_else(|| format!("Binary op requires Buffer, got {:?}", current_output.domain()))?;
                                    let other_output = node_outputs.get(other_node_id)
                                        .ok_or_else(|| format!("Binary op references unknown node '{other_node_id}'"))?;
                                    let other_buf = other_output.as_buffer()
                                        .ok_or_else(|| format!("Binary op other operand must be Buffer, got {:?}", other_output.domain()))?;
                                    let result = op.execute(current_buf, other_buf);
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::ApplyMask { mask_node_id, invert } => {
                                    // Flush pending buffer ops first
                                    current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    // Mask operation: buffer masked by another buffer
                                    let current_buf = current_output.as_buffer()
                                        .ok_or_else(|| format!("ApplyMask requires Buffer, got {:?}", current_output.domain()))?;
                                    let mask_output = node_outputs.get(mask_node_id)
                                        .ok_or_else(|| format!("ApplyMask references unknown node '{mask_node_id}'"))?;
                                    let mask_buf = mask_output.as_buffer()
                                        .ok_or_else(|| format!("ApplyMask mask must be Buffer, got {:?}", mask_output.domain()))?;
                                    let result = apply_mask(current_buf, mask_buf, *invert);
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::Reduction(reduction_op) => {
                                    // Flush pending buffer ops first
                                    current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    // Reduction operation: buffer → scalar (for global) or buffer (for axis)
                                    let current_buf = current_output.as_buffer()
                                        .ok_or_else(|| format!("Reduction requires Buffer, got {:?}", current_output.domain()))?;
                                    let result = reduction_op.execute(current_buf);
                                    // Check if this is a global reduction (produces scalar)
                                    if result.shape() == [1] {
                                        // Global reduction: extract scalar value
                                        let scalar_val = result.as_slice::<f64>()[0];
                                        current_output = NodeOutput::Scalar(scalar_val);
                                    } else {
                                        // Axis reduction: still a buffer
                                        current_output = NodeOutput::from_buffer(result);
                                    }
                                }
                                _ => {
                                    // Regular buffer operation: accumulate for batching
                                    pending_buffer_ops.push(view_dto.clone());
                                }
                            }
                        }

                        // Flush any remaining pending ops
                        current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;

                        node_outputs.insert(node_id.clone(), current_output);
                    }
                }

                // Encode each output based on its domain and sink format
                for (alias, spec) in &self.outputs {
                    if let Some(output) = node_outputs.get(&spec.node) {
                        match encode_node_output(output, &spec.sink) {
                            Ok(encoded) => {
                                let row_result = match encoded {
                                    OutputValue::Binary(bytes) => RowResult::Binary(Some(bytes)),
                                    OutputValue::Scalar(val) => RowResult::Scalar(Some(val)),
                                    OutputValue::Vector(vals) => RowResult::Vector(Some((*vals).clone())),
                                    OutputValue::Contours(contours) => RowResult::Contours(Some((*contours).clone())),
                                    OutputValue::TypedList { data, shape } => RowResult::TypedList(Some((data, shape))),
                                    OutputValue::TypedArray { data, shape } => RowResult::TypedArray(Some((data, shape))),
                                };
                                results.get_mut(alias).unwrap().push(row_result);
                            }
                            Err(e) => return Err(format!("Encode error for '{alias}': {e}")),
                        }
                    } else {
                        // No output for this node - push null based on expected type
                        // Default to Binary null for now
                        results.get_mut(alias).unwrap().push(RowResult::Binary(None));
                    }
                }
            }

            Ok(results)
        }));

        // Handle batch result
        let results = match batch_result {
            Ok(Ok(r)) => r,
            Ok(Err(msg)) => {
                return Err(polars_err!(ComputeError: "Pipeline execution failed: {}", msg));
            }
            Err(panic_payload) => {
                // Extract panic message
                let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic during batch execution".to_string()
                };
                return Err(polars_err!(ComputeError: "Pipeline batch failed: {}", panic_msg));
            }
        };

        // Build output based on single vs multi output
        if self.is_single_output() {
            // Single output: determine type and return appropriate column
            let data = results.get("_output").unwrap();
            
            // Check what type of results we have
            if data.is_empty() {
                let output_ca = BinaryChunked::from_iter_options(
                    inputs[0].name().clone(),
                    std::iter::empty::<Option<Vec<u8>>>(),
                );
                return Ok(output_ca.into_series());
            }
            
            match &data[0] {
                RowResult::Binary(_) => {
                    let binary_data: Vec<Option<Vec<u8>>> = data.iter().map(|r| {
                        match r {
                            RowResult::Binary(b) => b.clone(),
                            _ => None,
                        }
                    }).collect();
                    let output_ca = BinaryChunked::from_iter_options(
                        inputs[0].name().clone(),
                        binary_data.into_iter(),
                    );
                    Ok(output_ca.into_series())
                }
                RowResult::Scalar(_) => {
                    let scalar_data: Vec<Option<f64>> = data.iter().map(|r| {
                        match r {
                            RowResult::Scalar(s) => *s,
                            _ => None,
                        }
                    }).collect();
                    let output_ca = Float64Chunked::from_iter_options(
                        inputs[0].name().clone(),
                        scalar_data.into_iter(),
                    );
                    Ok(output_ca.into_series())
                }
                RowResult::Contours(_) => {
                    // Build contour struct series
                    let values: PolarsResult<Vec<AnyValue<'static>>> = data.iter().map(|r| {
                        match r {
                            RowResult::Contours(Some(contours)) => contours_to_polars_value(contours),
                            _ => Ok(AnyValue::Null),
                        }
                    }).collect();
                    let values = values?;
                    
                    // Infer dtype from first non-null value
                    let dtype = values.iter()
                        .find(|v| !matches!(v, AnyValue::Null))
                        .map(|v| v.dtype())
                        .unwrap_or(DataType::Null);
                    
                    let series = Series::from_any_values_and_dtype(
                        inputs[0].name().clone(),
                        &values,
                        &dtype,
                        true,
                    )?;
                    Ok(series)
                }
                RowResult::Vector(_) => {
                    // Build list of f64 series
                    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                        inputs[0].name().clone(),
                        data.len(),
                        4,  // Initial capacity for each list
                        DataType::Float64,
                    );
                    
                    for r in data.iter() {
                        match r {
                            RowResult::Vector(Some(vals)) => {
                                builder.append_slice(vals);
                            }
                            _ => {
                                builder.append_null();
                            }
                        }
                    }
                    
                    Ok(builder.finish().into_series())
                }
                RowResult::TypedList(_) => {
                    // Build typed list series preserving buffer dtype
                    let rows: Vec<TypedListRow> = data.iter().map(|r| {
                        match r {
                            RowResult::TypedList(Some((typed_data, shape))) => {
                                Some((typed_data.clone(), shape.clone()))
                            }
                            _ => None,
                        }
                    }).collect();
                    build_typed_list_series_from_rows(inputs[0].name().clone(), &rows)
                }
                RowResult::TypedArray(_) => {
                    // Build typed fixed-size array series preserving buffer dtype
                    let rows: Vec<TypedListRow> = data.iter().map(|r| {
                        match r {
                            RowResult::TypedArray(Some((typed_data, shape))) => {
                                Some((typed_data.clone(), shape.clone()))
                            }
                            _ => None,
                        }
                    }).collect();
                    build_typed_array_series_from_rows(inputs[0].name().clone(), &rows)
                }
            }
        } else {
            // Multi output: return Struct column with appropriate field types
            let mut fields: Vec<Series> = Vec::with_capacity(output_aliases.len());
            
            for alias in &output_aliases {
                let data = results.get(*alias).unwrap();
                
                if data.is_empty() {
                    let ca = BinaryChunked::from_iter_options(
                        PlSmallStr::from_str(alias),
                        std::iter::empty::<Option<Vec<u8>>>(),
                    );
                    fields.push(ca.into_series());
                    continue;
                }
                
                // Determine type from first element
                let field_series = match &data[0] {
                    RowResult::Binary(_) => {
                        let binary_data: Vec<Option<Vec<u8>>> = data.iter().map(|r| {
                            match r {
                                RowResult::Binary(b) => b.clone(),
                                _ => None,
                            }
                        }).collect();
                        let ca = BinaryChunked::from_iter_options(
                            PlSmallStr::from_str(alias),
                            binary_data.into_iter(),
                        );
                        ca.into_series()
                    }
                    RowResult::Scalar(_) => {
                        let scalar_data: Vec<Option<f64>> = data.iter().map(|r| {
                            match r {
                                RowResult::Scalar(s) => *s,
                                _ => None,
                            }
                        }).collect();
                        let ca = Float64Chunked::from_iter_options(
                            PlSmallStr::from_str(alias),
                            scalar_data.into_iter(),
                        );
                        ca.into_series()
                    }
                    RowResult::Contours(_) => {
                        let values: PolarsResult<Vec<AnyValue<'static>>> = data.iter().map(|r| {
                            match r {
                                RowResult::Contours(Some(contours)) => contours_to_polars_value(contours),
                                _ => Ok(AnyValue::Null),
                            }
                        }).collect();
                        let values = values?;
                        
                        let dtype = values.iter()
                            .find(|v| !matches!(v, AnyValue::Null))
                            .map(|v| v.dtype())
                            .unwrap_or(DataType::Null);
                        
                        Series::from_any_values_and_dtype(
                            PlSmallStr::from_str(alias),
                            &values,
                            &dtype,
                            true,
                        )?
                    }
                    RowResult::Vector(_) => {
                        let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                            PlSmallStr::from_str(alias),
                            data.len(),
                            4,  // Initial capacity for each list
                            DataType::Float64,
                        );
                        
                        for r in data.iter() {
                            match r {
                                RowResult::Vector(Some(vals)) => {
                                    builder.append_slice(vals);
                                }
                                _ => {
                                    builder.append_null();
                                }
                            }
                        }
                        
                        builder.finish().into_series()
                    }
                    RowResult::TypedList(_) => {
                        // Build typed list series preserving buffer dtype
                        let rows: Vec<TypedListRow> = data.iter().map(|r| {
                            match r {
                                RowResult::TypedList(Some((typed_data, shape))) => {
                                    Some((typed_data.clone(), shape.clone()))
                                }
                                _ => None,
                            }
                        }).collect();
                        build_typed_list_series_from_rows(PlSmallStr::from_str(alias), &rows)?
                    }
                    RowResult::TypedArray(_) => {
                        // Build typed fixed-size array series preserving buffer dtype
                        let rows: Vec<TypedListRow> = data.iter().map(|r| {
                            match r {
                                RowResult::TypedArray(Some((typed_data, shape))) => {
                                    Some((typed_data.clone(), shape.clone()))
                                }
                                _ => None,
                            }
                        }).collect();
                        build_typed_array_series_from_rows(PlSmallStr::from_str(alias), &rows)?
                    }
                };
                
                fields.push(field_series);
            }

            let output_name = inputs[0].name().clone();
            StructChunked::from_series(output_name, len, fields.iter()).map(|sc| sc.into_series())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // The order is now cached during from_json, access via private method
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
