//! Pipeline execution engine.
//!
//! This module handles the execution of vision pipelines on Polars Series,
//! including parameter resolution and view-buffer integration.

// Some functions are currently unused (graph.rs handles execution) but kept for potential future use
#![allow(dead_code)]

use polars::prelude::*;
use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};

use view_buffer::{
    geometry::{rasterize::rasterize, Contour, Point},
    BinaryOp, ComputeOp, DType, FilterType, GeometryOp, ImageAdapter, ImageOp, ImageOpKind,
    NormalizeMethod, ViewBuffer, ViewDto, ViewExpr, ViewOp,
};

use crate::params::ParamValue;
use crate::pipeline::{OpSpec, PipelineSpec};

/// Execute a pipeline on a Series, returning a new Series with results.
pub fn execute_pipeline(
    data_series: &Series,
    pipeline: &PipelineSpec,
    expr_columns: &HashMap<String, &Series>,
) -> PolarsResult<Series> {
    // Validate pipeline configuration first
    pipeline.validate()?;

    // Check if this is a contour source
    if pipeline.source_format() == "contour" {
        return execute_contour_pipeline(data_series, pipeline, expr_columns);
    }

    // Get the binary data for non-contour sources
    let data_ca = data_series
        .binary()
        .map_err(|_| polars_err!(ComputeError: "Expected Binary column for pipeline input"))?;

    let len = data_ca.len();
    let mut results: Vec<Option<Vec<u8>>> = Vec::with_capacity(len);

    // Process each row
    for row_idx in 0..len {
        match data_ca.get(row_idx) {
            Some(bytes) => {
                let result = execute_row(bytes, row_idx, pipeline, expr_columns)?;
                results.push(Some(result));
            }
            None => {
                results.push(None);
            }
        }
    }

    // Build the output series
    let output_ca =
        BinaryChunked::from_iter_options(data_series.name().clone(), results.into_iter());
    Ok(output_ca.into_series())
}

/// Execute a contour pipeline on a Struct Series.
fn execute_contour_pipeline(
    data_series: &Series,
    pipeline: &PipelineSpec,
    expr_columns: &HashMap<String, &Series>,
) -> PolarsResult<Series> {
    let len = data_series.len();
    let mut results: Vec<Option<Vec<u8>>> = Vec::with_capacity(len);

    // Process each row
    for row_idx in 0..len {
        let value = data_series.get(row_idx)?;
        match value {
            AnyValue::Null => {
                results.push(None);
            }
            _ => {
                let result = execute_contour_row(&value, row_idx, pipeline, expr_columns)?;
                results.push(Some(result));
            }
        }
    }

    // Build the output series
    let output_ca =
        BinaryChunked::from_iter_options(data_series.name().clone(), results.into_iter());
    Ok(output_ca.into_series())
}

/// Execute the contour pipeline on a single row.
fn execute_contour_row(
    value: &AnyValue,
    row_idx: usize,
    pipeline: &PipelineSpec,
    expr_columns: &HashMap<String, &Series>,
) -> PolarsResult<Vec<u8>> {
    // 1. Decode the contour source (parse struct and rasterize)
    let buffer = decode_contour_source(value, row_idx, pipeline, expr_columns)?;

    // 2. Resolve all operations first
    let mut view_dtos = Vec::with_capacity(pipeline.ops.len());
    for op_spec in &pipeline.ops {
        let view_dto = resolve_op(op_spec, row_idx, expr_columns)?;
        view_dtos.push(view_dto);
    }

    // 3. Build expression and execute with panic catching
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut expr = ViewExpr::new_source(buffer);
        for view_dto in view_dtos {
            expr = expr.apply_op(view_dto);
        }
        let plan = expr.plan();
        plan.execute()
    }));

    let result_buffer = match result {
        Ok(buf) => buf,
        Err(panic_payload) => {
            let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic during pipeline execution".to_string()
            };
            return Err(polars_err!(ComputeError: "Pipeline execution failed: {}", panic_msg));
        }
    };

    // 4. Encode the sink
    encode_sink(&result_buffer, pipeline)
}

/// Decode a contour source by parsing the struct and rasterizing to ViewBuffer.
pub fn decode_contour_source(
    value: &AnyValue,
    row_idx: usize,
    pipeline: &PipelineSpec,
    expr_columns: &HashMap<String, &Series>,
) -> PolarsResult<ViewBuffer> {
    // Parse the contour from the struct
    let contour = parse_contour_from_anyvalue(value)?;

    // Resolve dimensions
    let (width, height) = resolve_contour_dimensions(row_idx, pipeline, expr_columns)?;

    // Get fill and background values
    let fill_value = pipeline.source.fill_value;
    let background = pipeline.source.background;

    // Rasterize the contour to a ViewBuffer
    Ok(rasterize(
        &contour, width, height, fill_value, background, false, // anti_alias not yet supported
    ))
}

/// Decode a contour source with explicit dimensions (for graph execution with shape inference).
///
/// This variant is used when dimensions are resolved from a shape reference (another node's buffer)
/// rather than from explicit width/height parameters.
pub fn decode_contour_source_with_dims(
    value: &AnyValue,
    width: u32,
    height: u32,
    fill_value: u8,
    background: u8,
) -> PolarsResult<ViewBuffer> {
    // Parse the contour from the struct
    let contour = parse_contour_from_anyvalue(value)?;

    // Rasterize the contour to a ViewBuffer
    Ok(rasterize(
        &contour, width, height, fill_value, background, false, // anti_alias not yet supported
    ))
}

/// Parse a contour from an AnyValue (struct or list).
fn parse_contour_from_anyvalue(value: &AnyValue) -> PolarsResult<Contour> {
    match value {
        AnyValue::StructOwned(boxed) => {
            let (values, fields) = boxed.as_ref();

            // Find the exterior field
            for (i, field) in fields.iter().enumerate() {
                if field.name().as_str() == "exterior" || field.name().as_str() == "points" {
                    if let Some(AnyValue::List(series)) = values.get(i) {
                        let points = extract_points_from_series(series)?;
                        // Look for holes field
                        let holes = extract_holes_from_struct(values, fields)?;
                        return Ok(Contour::with_holes(points, holes));
                    }
                }
            }

            // If no named field found, try to use the first list field
            for av in values.iter() {
                if let AnyValue::List(series) = av {
                    let points = extract_points_from_series(series)?;
                    return Ok(Contour::new(points));
                }
            }

            Err(polars_err!(ComputeError: "Could not find contour points in struct"))
        }
        AnyValue::Struct(idx, array, fields) => {
            // Handle struct array reference - convert to owned for easier handling
            let owned = value.clone().into_static();
            if let AnyValue::StructOwned(boxed) = owned {
                let (values, flds) = boxed.as_ref();
                for (i, field) in flds.iter().enumerate() {
                    if field.name().as_str() == "exterior" || field.name().as_str() == "points" {
                        if let Some(AnyValue::List(series)) = values.get(i) {
                            let points = extract_points_from_series(series)?;
                            let holes = extract_holes_from_struct(values, flds)?;
                            return Ok(Contour::with_holes(points, holes));
                        }
                    }
                }
            }
            // Suppress unused variable warnings
            let _ = (idx, array, fields);
            Err(polars_err!(ComputeError: "Could not extract contour from struct array"))
        }
        _ => Err(polars_err!(ComputeError: "Expected Struct for contour, got {:?}", value.dtype())),
    }
}

/// Extract holes from a struct's holes field.
fn extract_holes_from_struct(
    values: &[AnyValue],
    fields: &[Field],
) -> PolarsResult<Vec<Vec<Point>>> {
    for (i, field) in fields.iter().enumerate() {
        if field.name().as_str() == "holes" {
            if let Some(AnyValue::List(holes_series)) = values.get(i) {
                let mut holes = Vec::new();
                for j in 0..holes_series.len() {
                    if let Ok(AnyValue::List(hole_points_series)) = holes_series.get(j) {
                        let points = extract_points_from_series(&hole_points_series)?;
                        if !points.is_empty() {
                            holes.push(points);
                        }
                    }
                }
                return Ok(holes);
            }
        }
    }
    Ok(Vec::new())
}

/// Extract points from a Series containing point structs.
fn extract_points_from_series(series: &Series) -> PolarsResult<Vec<Point>> {
    let mut points = Vec::new();

    for i in 0..series.len() {
        let value = series.get(i)?;
        match value {
            AnyValue::StructOwned(boxed) => {
                let (vals, flds) = boxed.as_ref();
                let (mut x, mut y) = (0.0, 0.0);
                for (j, fld) in flds.iter().enumerate() {
                    match fld.name().as_str() {
                        "x" => x = extract_f64(&vals[j])?,
                        "y" => y = extract_f64(&vals[j])?,
                        _ => {}
                    }
                }
                points.push(Point::new(x, y));
            }
            AnyValue::Struct(idx, array, fields) => {
                // Convert to owned for easier handling
                let owned = value.clone().into_static();
                if let AnyValue::StructOwned(boxed) = owned {
                    let (vals, flds) = boxed.as_ref();
                    let (mut x, mut y) = (0.0, 0.0);
                    for (j, fld) in flds.iter().enumerate() {
                        match fld.name().as_str() {
                            "x" => x = extract_f64(&vals[j])?,
                            "y" => y = extract_f64(&vals[j])?,
                            _ => {}
                        }
                    }
                    points.push(Point::new(x, y));
                }
                // Suppress unused variable warnings
                let _ = (idx, array, fields);
            }
            AnyValue::Null => {
                // Skip null points
            }
            _ => {
                return Err(
                    polars_err!(ComputeError: "Expected struct for point, got {:?}", value.dtype()),
                );
            }
        }
    }

    Ok(points)
}

/// Extract f64 from various numeric AnyValue types.
fn extract_f64(value: &AnyValue) -> PolarsResult<f64> {
    match value {
        AnyValue::Float64(v) => Ok(*v),
        AnyValue::Float32(v) => Ok(*v as f64),
        AnyValue::Int64(v) => Ok(*v as f64),
        AnyValue::Int32(v) => Ok(*v as f64),
        AnyValue::Int16(v) => Ok(*v as f64),
        AnyValue::Int8(v) => Ok(*v as f64),
        AnyValue::UInt64(v) => Ok(*v as f64),
        AnyValue::UInt32(v) => Ok(*v as f64),
        AnyValue::UInt16(v) => Ok(*v as f64),
        AnyValue::UInt8(v) => Ok(*v as f64),
        _ => {
            Err(polars_err!(ComputeError: "Expected numeric value for coordinate, got {:?}", value))
        }
    }
}

/// Resolve contour dimensions from pipeline source spec.
fn resolve_contour_dimensions(
    row_idx: usize,
    pipeline: &PipelineSpec,
    expr_columns: &HashMap<String, &Series>,
) -> PolarsResult<(u32, u32)> {
    // Check for shape_pipeline first (not yet implemented - just error)
    if pipeline.source.shape_pipeline.is_some() {
        return Err(
            polars_err!(ComputeError: "Shape inference from pipeline not yet implemented. Use explicit width/height."),
        );
    }

    // Get explicit width and height
    let width = pipeline
        .source
        .width
        .as_ref()
        .ok_or_else(|| polars_err!(ComputeError: "Contour source requires 'width' parameter"))?
        .resolve_usize(row_idx, expr_columns)? as u32;

    let height = pipeline
        .source
        .height
        .as_ref()
        .ok_or_else(|| polars_err!(ComputeError: "Contour source requires 'height' parameter"))?
        .resolve_usize(row_idx, expr_columns)? as u32;

    Ok((width, height))
}

/// Execute the pipeline on a single row.
fn execute_row(
    bytes: &[u8],
    row_idx: usize,
    pipeline: &PipelineSpec,
    expr_columns: &HashMap<String, &Series>,
) -> PolarsResult<Vec<u8>> {
    // 1. Decode the source
    let buffer = decode_source(bytes, pipeline)?;

    // 2. Resolve all operations first (outside of catch_unwind for better error messages)
    let mut view_dtos = Vec::with_capacity(pipeline.ops.len());
    for op_spec in &pipeline.ops {
        let view_dto = resolve_op(op_spec, row_idx, expr_columns)?;
        view_dtos.push(view_dto);
    }

    // 3. Build expression and execute with panic catching
    //    This catches panics from view-buffer operations that use panic!() for errors
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut expr = ViewExpr::new_source(buffer);
        for view_dto in view_dtos {
            expr = expr.apply_op(view_dto);
        }
        let plan = expr.plan();
        plan.execute()
    }));

    let result_buffer = match result {
        Ok(buf) => buf,
        Err(panic_payload) => {
            // Extract panic message if possible
            let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic during pipeline execution".to_string()
            };
            return Err(polars_err!(ComputeError: "Pipeline execution failed: {}", panic_msg));
        }
    };

    // 4. Encode the sink (also wrap in catch_unwind for safety)
    encode_sink(&result_buffer, pipeline)
}

/// Decode the source bytes into a ViewBuffer.
pub fn decode_source(bytes: &[u8], pipeline: &PipelineSpec) -> PolarsResult<ViewBuffer> {
    match pipeline.source_format() {
        "image_bytes" => {
            // Use image crate to decode
            ImageAdapter::decode(bytes)
                .map_err(|e| polars_err!(ComputeError: "Failed to decode image: {:?}", e))
        }
        "blob" => {
            // Decode from VIEW protocol
            ViewBuffer::from_blob(bytes)
                .map_err(|e| polars_err!(ComputeError: "Failed to decode blob: {:?}", e))
        }
        "raw" => {
            // Raw bytes - need dtype from source spec
            let dtype_str = pipeline
                .source
                .dtype
                .as_ref()
                .ok_or_else(|| polars_err!(ComputeError: "Raw source format requires dtype"))?;
            let dtype = parse_dtype(dtype_str)?;

            // For raw format, we need shape from shape_hints
            // For now, treat as 1D array
            let element_size = dtype.size_of();
            let num_elements = bytes.len() / element_size;

            Ok(ViewBuffer::from_raw_bytes(
                bytes.to_vec(),
                vec![num_elements],
                dtype,
            ))
        }
        other => Err(polars_err!(ComputeError: "Unknown source format: {}", other)),
    }
}

/// Encode the result buffer to the sink format.
///
/// Note: numpy/torch sinks are now handled by the output module for zero-copy support.
/// This function handles png, jpeg, blob, and raw binary formats.
pub fn encode_sink(buffer: &ViewBuffer, pipeline: &PipelineSpec) -> PolarsResult<Vec<u8>> {
    match pipeline.sink_format() {
        "numpy" | "torch" => {
            // numpy/torch now use struct-based output via crate::output module
            // This path should not be reached in normal operation
            Err(polars_err!(ComputeError:
                "numpy/torch sinks should use output module for zero-copy struct encoding"))
        }
        "blob" => {
            // VIEW protocol
            Ok(buffer.to_blob())
        }
        "png" => ImageAdapter::encode(buffer, image::ImageFormat::Png)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode PNG: {:?}", e)),
        "jpeg" => {
            let quality = pipeline.sink.quality;
            ImageAdapter::encode_jpeg(buffer, quality)
                .map_err(|e| polars_err!(ComputeError: "Failed to encode JPEG: {:?}", e))
        }
        "webp" => ImageAdapter::encode(buffer, image::ImageFormat::WebP)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode WebP: {:?}", e)),
        "tiff" => ImageAdapter::encode_tiff(buffer)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode TIFF: {:?}", e)),
        "array" | "list" => {
            // For array/list, we return raw bytes that Polars will interpret
            // The actual type conversion happens in the output dtype
            //
            // Optimization: Check if already contiguous to avoid unnecessary copy
            let num_elements: usize = buffer.shape().iter().product();
            let data_len = num_elements * buffer.dtype().size_of();

            if buffer.layout_facts().is_contiguous() {
                // Already contiguous - avoid copy
                let data_slice =
                    unsafe { std::slice::from_raw_parts(buffer.as_ptr::<u8>(), data_len) };
                Ok(data_slice.to_vec())
            } else {
                // Need to materialize to contiguous layout
                let contig = buffer.to_contiguous();
                let data_slice =
                    unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len) };
                Ok(data_slice.to_vec())
            }
        }
        other => Err(polars_err!(ComputeError: "Unknown sink format: {}", other)),
    }
}

/// Resolve an operation specification to a ViewDto.
pub fn resolve_op(
    op_spec: &OpSpec,
    row_idx: usize,
    expr_columns: &HashMap<String, &Series>,
) -> PolarsResult<ViewDto> {
    match op_spec.op.as_str() {
        // View operations
        "transpose" => {
            let axes = get_param(&op_spec.params, "axes")?.as_int_list()?;
            Ok(ViewDto::View(ViewOp::Transpose(axes)))
        }
        "reshape" => {
            let shape_params = get_param(&op_spec.params, "shape")?.as_param_list()?;
            let shape: Vec<usize> = shape_params
                .iter()
                .map(|p| p.resolve_usize(row_idx, expr_columns))
                .collect::<PolarsResult<_>>()?;
            Ok(ViewDto::View(ViewOp::Reshape(shape)))
        }
        "flip" => {
            let axes = get_param(&op_spec.params, "axes")?.as_int_list()?;
            Ok(ViewDto::View(ViewOp::Flip(axes)))
        }
        "crop" => {
            // Allow negative values for top/left and clamp to 0
            // This makes the API more forgiving and follows NumPy/OpenCV conventions
            let top_raw = get_param(&op_spec.params, "top")?.resolve_i64(row_idx, expr_columns)?;
            let left_raw =
                get_param(&op_spec.params, "left")?.resolve_i64(row_idx, expr_columns)?;

            // Clamp negative values to 0
            let top = top_raw.max(0) as usize;
            let left = left_raw.max(0) as usize;

            // Height and width might be optional - these should still be non-negative
            let height = op_spec
                .params
                .get("height")
                .map(|p| {
                    let h = p.resolve_i64(row_idx, expr_columns)?;
                    // Clamp negative height to 0 (will result in empty crop)
                    Ok::<usize, PolarsError>(h.max(0) as usize)
                })
                .transpose()?;
            let width = op_spec
                .params
                .get("width")
                .map(|p| {
                    let w = p.resolve_i64(row_idx, expr_columns)?;
                    // Clamp negative width to 0 (will result in empty crop)
                    Ok::<usize, PolarsError>(w.max(0) as usize)
                })
                .transpose()?;

            // For crop, we need start and end vectors
            // Assuming HWC layout: start = [top, left, 0], end = [top+height, left+width, C]
            // The slice operation in ViewBuffer will further clamp these to valid bounds
            let start = vec![top, left, 0];
            let end = match (height, width) {
                (Some(h), Some(w)) => {
                    vec![top.saturating_add(h), left.saturating_add(w), usize::MAX]
                }
                _ => vec![usize::MAX, usize::MAX, usize::MAX], // Full extent
            };

            Ok(ViewDto::View(ViewOp::Crop { start, end }))
        }

        // Compute operations
        "cast" => {
            let dtype_str = get_param(&op_spec.params, "dtype")?.resolve_string()?;
            let dtype = parse_dtype(&dtype_str)?;
            Ok(ViewDto::Compute(ComputeOp::Cast(dtype)))
        }
        "scale" => {
            let factor =
                get_param(&op_spec.params, "factor")?.resolve_f32(row_idx, expr_columns)?;
            Ok(ViewDto::Compute(ComputeOp::Scale(factor)))
        }
        "normalize" => {
            let method_str = get_param(&op_spec.params, "method")?.resolve_string()?;
            let method = match method_str.as_str() {
                "minmax" => NormalizeMethod::MinMax,
                "zscore" => NormalizeMethod::ZScore,
                "preset" => {
                    // Extract mean and std arrays from parameters
                    let mean_param = get_param(&op_spec.params, "mean")?;
                    let std_param = get_param(&op_spec.params, "std")?;

                    // Parse mean array from ParamValue
                    let mean = mean_param.as_f32_vec().ok_or_else(|| {
                        polars_err!(ComputeError: "normalize preset requires 'mean' as array of floats")
                    })?;

                    // Parse std array from ParamValue
                    let std = std_param.as_f32_vec().ok_or_else(|| {
                        polars_err!(ComputeError: "normalize preset requires 'std' as array of floats")
                    })?;

                    NormalizeMethod::Preset { mean, std }
                }
                other => {
                    return Err(polars_err!(ComputeError: "Unknown normalize method: {}", other))
                }
            };
            Ok(ViewDto::Compute(ComputeOp::Normalize(method)))
        }
        "clamp" => {
            let min = get_param(&op_spec.params, "min")?.resolve_f32(row_idx, expr_columns)?;
            let max = get_param(&op_spec.params, "max")?.resolve_f32(row_idx, expr_columns)?;
            Ok(ViewDto::Compute(ComputeOp::Clamp { min, max }))
        }
        "relu" => Ok(ViewDto::Compute(ComputeOp::Relu)),

        // Image operations
        "resize" => {
            let height =
                get_param(&op_spec.params, "height")?.resolve_u32(row_idx, expr_columns)?;
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, expr_columns)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(&filter_str)?;

            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Resize {
                    width,
                    height,
                    filter,
                },
            }))
        }
        "resize_scale" => {
            let scale_x =
                get_param(&op_spec.params, "scale_x")?.resolve_f32(row_idx, expr_columns)?;
            let scale_y =
                get_param(&op_spec.params, "scale_y")?.resolve_f32(row_idx, expr_columns)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(&filter_str)?;

            Ok(ViewDto::ResizeScale {
                scale_x,
                scale_y,
                filter,
            })
        }
        "resize_to_height" => {
            let height =
                get_param(&op_spec.params, "height")?.resolve_u32(row_idx, expr_columns)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(&filter_str)?;

            Ok(ViewDto::ResizeToHeight { height, filter })
        }
        "resize_to_width" => {
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, expr_columns)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(&filter_str)?;

            Ok(ViewDto::ResizeToWidth { width, filter })
        }
        "resize_max" => {
            let max_size =
                get_param(&op_spec.params, "max_size")?.resolve_u32(row_idx, expr_columns)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(&filter_str)?;

            Ok(ViewDto::ResizeMax { max_size, filter })
        }
        "resize_min" => {
            let min_size =
                get_param(&op_spec.params, "min_size")?.resolve_u32(row_idx, expr_columns)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(&filter_str)?;

            Ok(ViewDto::ResizeMin { min_size, filter })
        }

        // Padding operations
        "pad" => {
            use view_buffer::ops::dto::PadMode;

            let top = get_param(&op_spec.params, "top")?.resolve_u32(row_idx, expr_columns)?;
            let bottom =
                get_param(&op_spec.params, "bottom")?.resolve_u32(row_idx, expr_columns)?;
            let left = get_param(&op_spec.params, "left")?.resolve_u32(row_idx, expr_columns)?;
            let right = get_param(&op_spec.params, "right")?.resolve_u32(row_idx, expr_columns)?;
            let value = get_param(&op_spec.params, "value")?.resolve_f32(row_idx, expr_columns)?;
            let mode_str = get_param(&op_spec.params, "mode")?.resolve_string()?;
            let mode = match mode_str.as_str() {
                "constant" => PadMode::Constant,
                "edge" => PadMode::Edge,
                "reflect" => PadMode::Reflect,
                "symmetric" => PadMode::Symmetric,
                other => return Err(polars_err!(ComputeError: "Unknown pad mode: {}", other)),
            };

            Ok(ViewDto::Pad {
                top,
                bottom,
                left,
                right,
                value,
                mode,
            })
        }
        "pad_to_size" => {
            use view_buffer::ops::dto::PadPosition;

            let height =
                get_param(&op_spec.params, "height")?.resolve_u32(row_idx, expr_columns)?;
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, expr_columns)?;
            let value = get_param(&op_spec.params, "value")?.resolve_f32(row_idx, expr_columns)?;
            let position_str = get_param(&op_spec.params, "position")?.resolve_string()?;
            let position = match position_str.as_str() {
                "center" => PadPosition::Center,
                "top-left" => PadPosition::TopLeft,
                "bottom-right" => PadPosition::BottomRight,
                other => return Err(polars_err!(ComputeError: "Unknown pad position: {}", other)),
            };

            Ok(ViewDto::PadToSize {
                height,
                width,
                position,
                value,
            })
        }
        "letterbox" => {
            let height =
                get_param(&op_spec.params, "height")?.resolve_u32(row_idx, expr_columns)?;
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, expr_columns)?;
            let value = get_param(&op_spec.params, "value")?.resolve_f32(row_idx, expr_columns)?;

            Ok(ViewDto::Letterbox {
                height,
                width,
                value,
            })
        }
        "grayscale" => Ok(ViewDto::Image(ImageOp {
            kind: ImageOpKind::Grayscale,
        })),
        "threshold" => {
            let value =
                get_param(&op_spec.params, "value")?.resolve_usize(row_idx, expr_columns)? as u8;
            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Threshold(value),
            }))
        }
        "blur" => {
            let sigma = get_param(&op_spec.params, "sigma")?.resolve_f32(row_idx, expr_columns)?;
            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Blur { sigma },
            }))
        }
        "rotate" => {
            let angle = get_param(&op_spec.params, "angle")?.resolve_f32(row_idx, expr_columns)?;
            let expand = op_spec
                .params
                .get("expand")
                .map(|p| {
                    matches!(
                        p,
                        ParamValue::Literal {
                            value: serde_json::Value::Bool(true)
                        }
                    )
                })
                .unwrap_or(false);

            // Normalize angle to [0, 360)
            let normalized_angle = angle % 360.0;
            let normalized_angle = if normalized_angle < 0.0 {
                normalized_angle + 360.0
            } else {
                normalized_angle
            };

            // Check for zero-copy rotations (90, 180, 270)
            // Use a small epsilon for floating point comparison
            const EPSILON: f32 = 0.001;
            if (normalized_angle - 90.0).abs() < EPSILON {
                Ok(ViewDto::View(ViewOp::Rotate90))
            } else if (normalized_angle - 180.0).abs() < EPSILON {
                Ok(ViewDto::View(ViewOp::Rotate180))
            } else if (normalized_angle - 270.0).abs() < EPSILON
                || (normalized_angle - (-90.0)).abs() < EPSILON
            {
                Ok(ViewDto::View(ViewOp::Rotate270))
            } else if normalized_angle.abs() < EPSILON || (normalized_angle - 360.0).abs() < EPSILON
            {
                // 0 or 360 degrees - no-op, but we'll use ViewOp for consistency
                // Actually, we can just return the identity, but for simplicity use Rotate180 twice
                // Or better: use a no-op view. But since we don't have that, we'll use ImageOp with 0 angle
                Ok(ViewDto::Image(ImageOp {
                    kind: ImageOpKind::Rotate {
                        angle: normalized_angle,
                        expand,
                    },
                }))
            } else {
                // Arbitrary angle - use ImageOp
                Ok(ViewDto::Image(ImageOp {
                    kind: ImageOpKind::Rotate {
                        angle: normalized_angle,
                        expand,
                    },
                }))
            }
        }

        // Perceptual hash operation
        "perceptual_hash" => {
            use view_buffer::ops::phash::{HashAlgorithm, PerceptualHashOp};

            let algorithm = op_spec
                .params
                .get("algorithm")
                .and_then(|p| match p {
                    ParamValue::Literal { value } => value.as_str(),
                    _ => None,
                })
                .unwrap_or("perceptual");

            let hash_algorithm = match algorithm {
                "average" => HashAlgorithm::Average,
                "difference" => HashAlgorithm::Difference,
                "perceptual" => HashAlgorithm::Perceptual,
                "blockhash" => HashAlgorithm::Blockhash,
                _ => HashAlgorithm::Perceptual,
            };

            let hash_size = op_spec
                .params
                .get("hash_size")
                .map(|p| p.resolve_usize(row_idx, expr_columns).unwrap_or(64) as u32)
                .unwrap_or(64);

            Ok(ViewDto::PerceptualHash(
                PerceptualHashOp::new(hash_algorithm).with_hash_size(hash_size),
            ))
        }

        // Geometry operations
        "rasterize" => {
            let width =
                get_param(&op_spec.params, "width")?.resolve_usize(row_idx, expr_columns)? as u32;
            let height =
                get_param(&op_spec.params, "height")?.resolve_usize(row_idx, expr_columns)? as u32;
            let fill_value = op_spec
                .params
                .get("fill_value")
                .map(|p| p.resolve_usize(row_idx, expr_columns).unwrap_or(255) as u8)
                .unwrap_or(255);
            let background = op_spec
                .params
                .get("background")
                .map(|p| p.resolve_usize(row_idx, expr_columns).unwrap_or(0) as u8)
                .unwrap_or(0);
            let anti_alias = op_spec
                .params
                .get("anti_alias")
                .map(|p| {
                    matches!(
                        p,
                        ParamValue::Literal {
                            value: serde_json::Value::Bool(true)
                        }
                    )
                })
                .unwrap_or(false);
            Ok(ViewDto::Geometry(GeometryOp::Rasterize {
                width,
                height,
                fill_value,
                background,
                anti_alias,
            }))
        }
        "extract_contours" => {
            use view_buffer::geometry::ops::{ApproxMethod, ExtractMode};

            let mode = op_spec
                .params
                .get("mode")
                .and_then(|p| match p {
                    ParamValue::Literal {
                        value: serde_json::Value::String(s),
                    } => Some(s.as_str()),
                    _ => None,
                })
                .map(|s| match s {
                    "external" => ExtractMode::External,
                    "tree" => ExtractMode::Tree,
                    _ => ExtractMode::All,
                })
                .unwrap_or(ExtractMode::External);

            let method = op_spec
                .params
                .get("method")
                .and_then(|p| match p {
                    ParamValue::Literal {
                        value: serde_json::Value::String(s),
                    } => Some(s.as_str()),
                    _ => None,
                })
                .map(|s| match s {
                    "none" => ApproxMethod::None,
                    "approx" => ApproxMethod::Approx,
                    _ => ApproxMethod::Simple,
                })
                .unwrap_or(ApproxMethod::Simple);

            let min_area = op_spec.params.get("min_area").and_then(|p| match p {
                ParamValue::Literal {
                    value: serde_json::Value::Number(n),
                } => n.as_f64(),
                _ => None,
            });

            Ok(ViewDto::Geometry(GeometryOp::ExtractContours {
                mode,
                method,
                min_area,
            }))
        }

        // Geometry measure operations
        "contour_area" => {
            let signed = op_spec
                .params
                .get("signed")
                .map(|p| {
                    matches!(
                        p,
                        ParamValue::Literal {
                            value: serde_json::Value::Bool(true)
                        }
                    )
                })
                .unwrap_or(false);
            Ok(ViewDto::Geometry(GeometryOp::Area { signed }))
        }
        "contour_perimeter" => Ok(ViewDto::Geometry(GeometryOp::Perimeter)),
        "contour_centroid" => Ok(ViewDto::Geometry(GeometryOp::Centroid)),
        "contour_bounding_box" => Ok(ViewDto::Geometry(GeometryOp::BoundingBox)),
        "contour_winding" => Ok(ViewDto::Geometry(GeometryOp::Winding)),
        "contour_is_convex" => Ok(ViewDto::Geometry(GeometryOp::IsConvex)),
        "contour_convex_hull" => Ok(ViewDto::Geometry(GeometryOp::ConvexHull)),

        // Geometry transforms
        "contour_translate" => {
            let dx = get_param(&op_spec.params, "dx")?.resolve_f64(row_idx, expr_columns)?;
            let dy = get_param(&op_spec.params, "dy")?.resolve_f64(row_idx, expr_columns)?;
            Ok(ViewDto::Geometry(GeometryOp::Translate { dx, dy }))
        }
        "contour_scale" => {
            let sx = get_param(&op_spec.params, "sx")?.resolve_f64(row_idx, expr_columns)?;
            let sy = get_param(&op_spec.params, "sy")?.resolve_f64(row_idx, expr_columns)?;
            Ok(ViewDto::Geometry(GeometryOp::Scale {
                sx,
                sy,
                origin: view_buffer::geometry::ops::ScaleOrigin::Centroid,
            }))
        }
        "contour_flip" => Ok(ViewDto::Geometry(GeometryOp::Flip)),
        "contour_simplify" => {
            let tolerance =
                get_param(&op_spec.params, "tolerance")?.resolve_f64(row_idx, expr_columns)?;
            Ok(ViewDto::Geometry(GeometryOp::Simplify { tolerance }))
        }
        "contour_normalize" => {
            let ref_width =
                get_param(&op_spec.params, "ref_width")?.resolve_f64(row_idx, expr_columns)?;
            let ref_height =
                get_param(&op_spec.params, "ref_height")?.resolve_f64(row_idx, expr_columns)?;
            Ok(ViewDto::Geometry(GeometryOp::Normalize {
                ref_width,
                ref_height,
            }))
        }
        "contour_to_absolute" => {
            let ref_width =
                get_param(&op_spec.params, "ref_width")?.resolve_f64(row_idx, expr_columns)?;
            let ref_height =
                get_param(&op_spec.params, "ref_height")?.resolve_f64(row_idx, expr_columns)?;
            Ok(ViewDto::Geometry(GeometryOp::ToAbsolute {
                ref_width,
                ref_height,
            }))
        }

        // Binary operations
        "add" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::Add,
                other_node_id,
            })
        }
        "subtract" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::Subtract,
                other_node_id,
            })
        }
        "multiply" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::Multiply,
                other_node_id,
            })
        }
        "divide" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::Divide,
                other_node_id,
            })
        }
        "blend" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::Blend,
                other_node_id,
            })
        }
        "ratio" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::Ratio,
                other_node_id,
            })
        }
        "maximum" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::Maximum,
                other_node_id,
            })
        }
        "minimum" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::Minimum,
                other_node_id,
            })
        }
        "bitwise_and" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::BitwiseAnd,
                other_node_id,
            })
        }
        "bitwise_or" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::BitwiseOr,
                other_node_id,
            })
        }
        "bitwise_xor" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            Ok(ViewDto::Binary {
                op: BinaryOp::BitwiseXor,
                other_node_id,
            })
        }

        // Reduction operations
        "reduce_sum" => {
            use view_buffer::ops::ReductionOp;
            // Global reduction: axis = None means reduce entire array to scalar
            Ok(ViewDto::Reduction(ReductionOp::Sum { axis: None }))
        }
        "reduce_popcount" => {
            use view_buffer::ops::ReductionOp;
            // Count set bits across entire buffer (for Hamming distance)
            Ok(ViewDto::Reduction(ReductionOp::PopCount))
        }
        "reduce_max" => {
            use view_buffer::ops::ReductionOp;
            let axis = op_spec
                .params
                .get("axis")
                .and_then(|p| p.resolve_usize(row_idx, expr_columns).ok());
            Ok(ViewDto::Reduction(ReductionOp::Max { axis }))
        }
        "reduce_min" => {
            use view_buffer::ops::ReductionOp;
            let axis = op_spec
                .params
                .get("axis")
                .and_then(|p| p.resolve_usize(row_idx, expr_columns).ok());
            Ok(ViewDto::Reduction(ReductionOp::Min { axis }))
        }
        "reduce_mean" => {
            use view_buffer::ops::ReductionOp;
            let axis = op_spec
                .params
                .get("axis")
                .and_then(|p| p.resolve_usize(row_idx, expr_columns).ok());
            Ok(ViewDto::Reduction(ReductionOp::Mean { axis }))
        }
        "reduce_std" => {
            use view_buffer::ops::ReductionOp;
            let axis = op_spec
                .params
                .get("axis")
                .and_then(|p| p.resolve_usize(row_idx, expr_columns).ok());
            let ddof = op_spec
                .params
                .get("ddof")
                .map(|p| p.resolve_usize(row_idx, expr_columns).unwrap_or(0) as u8)
                .unwrap_or(0);
            Ok(ViewDto::Reduction(ReductionOp::Std { axis, ddof }))
        }
        "reduce_argmax" => {
            use view_buffer::ops::ReductionOp;
            let axis = get_param(&op_spec.params, "axis")?.resolve_usize(row_idx, expr_columns)?;
            Ok(ViewDto::Reduction(ReductionOp::ArgMax { axis }))
        }
        "reduce_argmin" => {
            use view_buffer::ops::ReductionOp;
            let axis = get_param(&op_spec.params, "axis")?.resolve_usize(row_idx, expr_columns)?;
            Ok(ViewDto::Reduction(ReductionOp::ArgMin { axis }))
        }
        "extract_shape" => {
            // Extract shape returns buffer dimensions as a vector
            Ok(ViewDto::ExtractShape)
        }

        // Histogram operation
        "histogram" => {
            use view_buffer::ops::histogram::{HistogramOp, HistogramOutput};

            let bins = get_param(&op_spec.params, "bins")?.resolve_usize(row_idx, expr_columns)?;

            // Parse output mode
            let output_str = get_param(&op_spec.params, "output")?.resolve_string()?;
            let output = match output_str.as_str() {
                "counts" => HistogramOutput::Counts,
                "normalized" => HistogramOutput::Normalized,
                "quantized" => HistogramOutput::Quantized,
                "edges" => HistogramOutput::Edges,
                other => {
                    return Err(
                        polars_err!(ComputeError: "Unknown histogram output mode: {}", other),
                    )
                }
            };

            // Parse optional range
            let range = if op_spec.params.contains_key("range_min") {
                let range_min =
                    get_param(&op_spec.params, "range_min")?.resolve_f64(row_idx, expr_columns)?;
                let range_max =
                    get_param(&op_spec.params, "range_max")?.resolve_f64(row_idx, expr_columns)?;
                Some((range_min, range_max))
            } else {
                None
            };

            let mut op = HistogramOp::new(bins).with_output(output);
            if let Some((min, max)) = range {
                op = op.with_range(min, max);
            }

            Ok(ViewDto::Histogram(op))
        }

        // Mask operation
        "apply_mask" => {
            let mask_node_id = get_param(&op_spec.params, "other_node")?.resolve_string()?;
            let invert = op_spec
                .params
                .get("invert")
                .map(|p| {
                    matches!(
                        p,
                        ParamValue::Literal {
                            value: serde_json::Value::Bool(true)
                        }
                    )
                })
                .unwrap_or(false);
            Ok(ViewDto::ApplyMask {
                mask_node_id,
                invert,
            })
        }

        other => Err(polars_err!(ComputeError: "Unknown operation: {}", other)),
    }
}

/// Get a required parameter from the params map.
fn get_param<'a>(
    params: &'a HashMap<String, ParamValue>,
    name: &str,
) -> PolarsResult<&'a ParamValue> {
    params
        .get(name)
        .ok_or_else(|| polars_err!(ComputeError: "Missing required parameter: {}", name))
}

/// Parse a dtype string to DType.
fn parse_dtype(s: &str) -> PolarsResult<DType> {
    match s {
        "u8" => Ok(DType::U8),
        "i8" => Ok(DType::I8),
        "u16" => Ok(DType::U16),
        "i16" => Ok(DType::I16),
        "u32" => Ok(DType::U32),
        "i32" => Ok(DType::I32),
        "u64" => Ok(DType::U64),
        "i64" => Ok(DType::I64),
        "f32" => Ok(DType::F32),
        "f64" => Ok(DType::F64),
        other => Err(polars_err!(ComputeError: "Unknown dtype: {}", other)),
    }
}

/// Parse a filter type string.
fn parse_filter(s: &str) -> PolarsResult<FilterType> {
    match s {
        "nearest" => Ok(FilterType::Nearest),
        "bilinear" | "triangle" => Ok(FilterType::Triangle),
        "lanczos3" => Ok(FilterType::Lanczos3),
        "catmullrom" => Ok(FilterType::CatmullRom),
        "gaussian" => Ok(FilterType::Gaussian),
        other => Err(polars_err!(ComputeError: "Unknown filter type: {}", other)),
    }
}
