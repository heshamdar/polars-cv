//! Pipeline execution engine.
//!
//! This module handles the execution of vision pipelines on Polars Series,
//! including parameter resolution and view-buffer integration.

use polars::prelude::*;
use std::collections::HashMap;

use view_buffer::{
    geometry::{rasterize::rasterize, Contour, Point},
    AffineParams, BinaryOp, ComputeOp, DType, FilterType, GeometryOp, ImageAdapter, ImageOp,
    ImageOpKind, InterpolationType, NormalizeMethod, ViewBuffer, ViewDto, ViewOp,
};

use crate::params::{ParamCtx, ParamValue};
use crate::pipeline::{OpSpec, PipelineSpec};

/// Decode a contour source by parsing the struct and rasterizing to ViewBuffer.
pub fn decode_contour_source(
    value: &AnyValue,
    row_idx: usize,
    pipeline: &PipelineSpec,
    ctx: &ParamCtx,
) -> PolarsResult<ViewBuffer> {
    // Parse the contour from the struct
    let contour = parse_contour_from_anyvalue(value)?;

    // Resolve dimensions
    let (width, height) = resolve_contour_dimensions(row_idx, pipeline, ctx)?;

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
    ctx: &ParamCtx,
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
        .resolve_usize(row_idx, ctx)? as u32;

    let height = pipeline
        .source
        .height
        .as_ref()
        .ok_or_else(|| polars_err!(ComputeError: "Contour source requires 'height' parameter"))?
        .resolve_usize(row_idx, ctx)? as u32;

    Ok((width, height))
}

/// Decode the source bytes into a ViewBuffer.
pub fn decode_source(bytes: &[u8], pipeline: &PipelineSpec) -> PolarsResult<ViewBuffer> {
    match pipeline.source_format() {
        "image_bytes" => {
            // Use image crate to decode
            let buf = ImageAdapter::decode(bytes)
                .map_err(|e| polars_err!(ComputeError: "Failed to decode image: {:?}", e))?;
            // If source spec declares an expected dtype, cast to it.
            // This is a no-op when the decoded dtype already matches.
            if let Some(ref dtype_str) = pipeline.source.dtype {
                let target = parse_dtype(dtype_str)?;
                if buf.dtype() != target {
                    return Ok(buf.cast(target));
                }
            }
            Ok(buf)
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

/// The complete set of operation names `resolve_op` can execute.
///
/// This is the single registry of executable ops, surfaced to Python via
/// `_lib.known_ops()` so the planner/tests can check that every op a `Pipeline`
/// emits is executable (B1). It must list exactly the top-level match arms in
/// [`resolve_op`]; the `known_ops_all_resolve` unit test guards the forward
/// direction (every entry resolves), and `unknown_op_is_rejected` guards that
/// the catch-all still rejects names that are not arms.
pub const KNOWN_OPS: &[&str] = &[
    "add",
    "adjust_contrast",
    "adjust_gamma",
    "apply_mask",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "blend",
    "blur",
    "canny",
    "cast",
    "channel_merge",
    "channel_select",
    "channel_swap",
    "clamp",
    "contour_area",
    "contour_bounding_box",
    "contour_centroid",
    "contour_convex_hull",
    "contour_flip",
    "contour_is_convex",
    "contour_normalize",
    "contour_perimeter",
    "contour_scale",
    "contour_simplify",
    "contour_to_absolute",
    "contour_translate",
    "contour_winding",
    "convolve2d",
    "crop",
    "cvt_color",
    "dilate",
    "divide",
    "equalize_histogram",
    "erode",
    "extract_contours",
    "extract_shape",
    "flip",
    "grayscale",
    "histogram",
    "invert",
    "label_reduce",
    "letterbox",
    "maximum",
    "minimum",
    "morphology_gradient",
    "multiply",
    "normalize",
    "pad",
    "pad_to_size",
    "perceptual_hash",
    "rasterize",
    "ratio",
    "reduce_argmax",
    "reduce_argmin",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_percentile",
    "reduce_popcount",
    "reduce_std",
    "reduce_sum",
    "relu",
    "reshape",
    "resize",
    "resize_max",
    "resize_min",
    "resize_scale",
    "resize_to_height",
    "resize_to_width",
    "rotate",
    "scale",
    "subtract",
    "threshold",
    "transpose",
    "warp_affine",
];

/// Resolve an operation specification to a ViewDto.
pub fn resolve_op(op_spec: &OpSpec, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<ViewDto> {
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
                .map(|p| p.resolve_usize(row_idx, ctx))
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
            let top_raw = get_param(&op_spec.params, "top")?.resolve_i64(row_idx, ctx)?;
            let left_raw = get_param(&op_spec.params, "left")?.resolve_i64(row_idx, ctx)?;

            // Clamp negative values to 0
            let top = top_raw.max(0) as usize;
            let left = left_raw.max(0) as usize;

            // Height and width might be optional - these should still be non-negative
            let height = op_spec
                .params
                .get("height")
                .map(|p| {
                    let h = p.resolve_i64(row_idx, ctx)?;
                    // Clamp negative height to 0 (will result in empty crop)
                    Ok::<usize, PolarsError>(h.max(0) as usize)
                })
                .transpose()?;
            let width = op_spec
                .params
                .get("width")
                .map(|p| {
                    let w = p.resolve_i64(row_idx, ctx)?;
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
            let dtype = parse_dtype(dtype_str)?;
            Ok(ViewDto::Compute(ComputeOp::Cast(dtype)))
        }
        "scale" => {
            let factor = get_param(&op_spec.params, "factor")?.resolve_f32(row_idx, ctx)?;
            Ok(ViewDto::Compute(ComputeOp::Scale(factor)))
        }
        "normalize" => {
            let method_str = get_param(&op_spec.params, "method")?.resolve_string()?;
            let method = match method_str {
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
            let min = get_param(&op_spec.params, "min")?.resolve_f32(row_idx, ctx)?;
            let max = get_param(&op_spec.params, "max")?.resolve_f32(row_idx, ctx)?;
            Ok(ViewDto::Compute(ComputeOp::Clamp { min, max }))
        }
        "relu" => Ok(ViewDto::Compute(ComputeOp::Relu)),

        // Image operations
        "resize" => {
            let height = get_param(&op_spec.params, "height")?.resolve_u32(row_idx, ctx)?;
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Resize {
                    width,
                    height,
                    filter,
                },
            }))
        }
        "resize_scale" => {
            let scale_x = get_param(&op_spec.params, "scale_x")?.resolve_f32(row_idx, ctx)?;
            let scale_y = get_param(&op_spec.params, "scale_y")?.resolve_f32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            Ok(ViewDto::ResizeScale {
                scale_x,
                scale_y,
                filter,
            })
        }
        "resize_to_height" => {
            let height = get_param(&op_spec.params, "height")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            Ok(ViewDto::ResizeToHeight { height, filter })
        }
        "resize_to_width" => {
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            Ok(ViewDto::ResizeToWidth { width, filter })
        }
        "resize_max" => {
            let max_size = get_param(&op_spec.params, "max_size")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            Ok(ViewDto::ResizeMax { max_size, filter })
        }
        "resize_min" => {
            let min_size = get_param(&op_spec.params, "min_size")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            Ok(ViewDto::ResizeMin { min_size, filter })
        }

        // Padding operations
        "pad" => {
            use view_buffer::ops::dto::PadMode;

            let top = get_param(&op_spec.params, "top")?.resolve_u32(row_idx, ctx)?;
            let bottom = get_param(&op_spec.params, "bottom")?.resolve_u32(row_idx, ctx)?;
            let left = get_param(&op_spec.params, "left")?.resolve_u32(row_idx, ctx)?;
            let right = get_param(&op_spec.params, "right")?.resolve_u32(row_idx, ctx)?;
            let value = get_param(&op_spec.params, "value")?.resolve_f32(row_idx, ctx)?;
            let mode_str = get_param(&op_spec.params, "mode")?.resolve_string()?;
            let mode = match mode_str {
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

            let height = get_param(&op_spec.params, "height")?.resolve_u32(row_idx, ctx)?;
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, ctx)?;
            let value = get_param(&op_spec.params, "value")?.resolve_f32(row_idx, ctx)?;
            let position_str = get_param(&op_spec.params, "position")?.resolve_string()?;
            let position = match position_str {
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
            let height = get_param(&op_spec.params, "height")?.resolve_u32(row_idx, ctx)?;
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, ctx)?;
            let value = get_param(&op_spec.params, "value")?.resolve_f32(row_idx, ctx)?;

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
            let value = get_param(&op_spec.params, "value")?.resolve_f64(row_idx, ctx)?;
            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Threshold(value),
            }))
        }
        "blur" => {
            let sigma = get_param(&op_spec.params, "sigma")?.resolve_f32(row_idx, ctx)?;
            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Blur { sigma },
            }))
        }
        "rotate" => {
            let angle = get_param(&op_spec.params, "angle")?.resolve_f32(row_idx, ctx)?;
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

            let normalized_angle = angle % 360.0;
            let normalized_angle = if normalized_angle < 0.0 {
                normalized_angle + 360.0
            } else {
                normalized_angle
            };

            const EPSILON: f32 = 0.001;
            if (normalized_angle - 90.0).abs() < EPSILON {
                Ok(ViewDto::View(ViewOp::Rotate90))
            } else if (normalized_angle - 180.0).abs() < EPSILON {
                Ok(ViewDto::View(ViewOp::Rotate180))
            } else if (normalized_angle - 270.0).abs() < EPSILON {
                Ok(ViewDto::View(ViewOp::Rotate270))
            } else if normalized_angle.abs() < EPSILON || (normalized_angle - 360.0).abs() < EPSILON
            {
                Ok(ViewDto::Compute(ComputeOp::RotateAffine {
                    angle_deg: 0.0,
                    expand: false,
                    interpolation: InterpolationType::Bilinear,
                    border_value: 0.0,
                }))
            } else {
                // Route arbitrary angles through AffineParams for unified code path.
                // The affine matrix is built at execution time from the current
                // buffer dimensions (handled by RotateToAffine).
                let interp_str = op_spec
                    .params
                    .get("interpolation")
                    .and_then(|p| match p {
                        ParamValue::Literal { value } => value.as_str(),
                        _ => None,
                    })
                    .unwrap_or("bilinear");
                let interpolation = match interp_str {
                    "nearest" => InterpolationType::Nearest,
                    "bilinear" => InterpolationType::Bilinear,
                    other => {
                        return Err(polars_err!(
                            ComputeError: "rotate: unknown interpolation '{}', expected 'nearest' or 'bilinear'",
                            other
                        ));
                    }
                };
                let border_value = op_spec
                    .params
                    .get("border_value")
                    .and_then(|p| match p {
                        ParamValue::Literal { value } => value.as_f64(),
                        _ => None,
                    })
                    .unwrap_or(0.0);

                Ok(ViewDto::Compute(ComputeOp::RotateAffine {
                    angle_deg: normalized_angle,
                    expand,
                    interpolation,
                    border_value,
                }))
            }
        }

        // Affine warp operation
        "warp_affine" => {
            let matrix_val = get_param(&op_spec.params, "matrix")?;
            let matrix_vec: Vec<f64> = match matrix_val {
                ParamValue::Literal { value } => {
                    value
                        .as_array()
                        .ok_or_else(|| {
                            polars_err!(ComputeError: "warp_affine: matrix must be an array")
                        })?
                        .iter()
                        .map(|v| {
                            v.as_f64().ok_or_else(|| {
                                polars_err!(ComputeError: "warp_affine: matrix elements must be numbers")
                            })
                        })
                        .collect::<PolarsResult<Vec<f64>>>()?
                }
                _ => {
                    return Err(polars_err!(
                        ComputeError: "warp_affine: matrix must be a literal array"
                    ));
                }
            };
            if matrix_vec.len() != 6 {
                return Err(polars_err!(
                    ComputeError: "warp_affine: matrix must have exactly 6 elements, got {}",
                    matrix_vec.len()
                ));
            }
            let matrix: [f64; 6] = matrix_vec
                .try_into()
                .map_err(|_| polars_err!(ComputeError: "warp_affine: matrix conversion failed"))?;

            let output_height =
                get_param(&op_spec.params, "output_height")?.resolve_u32(row_idx, ctx)?;
            let output_width =
                get_param(&op_spec.params, "output_width")?.resolve_u32(row_idx, ctx)?;

            let interp_str = op_spec
                .params
                .get("interpolation")
                .and_then(|p| match p {
                    ParamValue::Literal { value } => value.as_str(),
                    _ => None,
                })
                .unwrap_or("bilinear");
            let interpolation = match interp_str {
                "nearest" => InterpolationType::Nearest,
                "bilinear" => InterpolationType::Bilinear,
                other => {
                    return Err(polars_err!(
                        ComputeError: "warp_affine: unknown interpolation '{}', expected 'nearest' or 'bilinear'",
                        other
                    ));
                }
            };

            let border_value = op_spec
                .params
                .get("border_value")
                .and_then(|p| match p {
                    ParamValue::Literal { value } => value.as_f64(),
                    _ => None,
                })
                .unwrap_or(0.0);

            Ok(ViewDto::Compute(ComputeOp::Affine(AffineParams {
                matrix,
                output_height,
                output_width,
                interpolation,
                border_value,
            })))
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
                .map(|p| p.resolve_usize(row_idx, ctx).unwrap_or(64) as u32)
                .unwrap_or(64);

            Ok(ViewDto::PerceptualHash(
                PerceptualHashOp::new(hash_algorithm).with_hash_size(hash_size),
            ))
        }

        // Geometry operations
        "rasterize" => {
            let width = get_param(&op_spec.params, "width")?.resolve_usize(row_idx, ctx)? as u32;
            let height = get_param(&op_spec.params, "height")?.resolve_usize(row_idx, ctx)? as u32;
            let fill_value = op_spec
                .params
                .get("fill_value")
                .map(|p| p.resolve_usize(row_idx, ctx).unwrap_or(255) as u8)
                .unwrap_or(255);
            let background = op_spec
                .params
                .get("background")
                .map(|p| p.resolve_usize(row_idx, ctx).unwrap_or(0) as u8)
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
            let dx = get_param(&op_spec.params, "dx")?.resolve_f64(row_idx, ctx)?;
            let dy = get_param(&op_spec.params, "dy")?.resolve_f64(row_idx, ctx)?;
            Ok(ViewDto::Geometry(GeometryOp::Translate { dx, dy }))
        }
        "contour_scale" => {
            let sx = get_param(&op_spec.params, "sx")?.resolve_f64(row_idx, ctx)?;
            let sy = get_param(&op_spec.params, "sy")?.resolve_f64(row_idx, ctx)?;
            Ok(ViewDto::Geometry(GeometryOp::Scale {
                sx,
                sy,
                origin: view_buffer::geometry::ops::ScaleOrigin::Centroid,
            }))
        }
        "contour_flip" => Ok(ViewDto::Geometry(GeometryOp::Flip)),
        "contour_simplify" => {
            let tolerance = get_param(&op_spec.params, "tolerance")?.resolve_f64(row_idx, ctx)?;
            Ok(ViewDto::Geometry(GeometryOp::Simplify { tolerance }))
        }
        "contour_normalize" => {
            let ref_width = get_param(&op_spec.params, "ref_width")?.resolve_f64(row_idx, ctx)?;
            let ref_height = get_param(&op_spec.params, "ref_height")?.resolve_f64(row_idx, ctx)?;
            Ok(ViewDto::Geometry(GeometryOp::Normalize {
                ref_width,
                ref_height,
            }))
        }
        "contour_to_absolute" => {
            let ref_width = get_param(&op_spec.params, "ref_width")?.resolve_f64(row_idx, ctx)?;
            let ref_height = get_param(&op_spec.params, "ref_height")?.resolve_f64(row_idx, ctx)?;
            Ok(ViewDto::Geometry(GeometryOp::ToAbsolute {
                ref_width,
                ref_height,
            }))
        }

        // Binary operations
        "add" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::Add,
                other_node_id,
            })
        }
        "subtract" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::Subtract,
                other_node_id,
            })
        }
        "multiply" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::Multiply,
                other_node_id,
            })
        }
        "divide" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::Divide,
                other_node_id,
            })
        }
        "blend" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::Blend,
                other_node_id,
            })
        }
        "ratio" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::Ratio,
                other_node_id,
            })
        }
        "maximum" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::Maximum,
                other_node_id,
            })
        }
        "minimum" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::Minimum,
                other_node_id,
            })
        }
        "bitwise_and" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::BitwiseAnd,
                other_node_id,
            })
        }
        "bitwise_or" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(ViewDto::Binary {
                op: BinaryOp::BitwiseOr,
                other_node_id,
            })
        }
        "bitwise_xor" => {
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
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
                .and_then(|p| p.resolve_usize(row_idx, ctx).ok());
            Ok(ViewDto::Reduction(ReductionOp::Max { axis }))
        }
        "reduce_min" => {
            use view_buffer::ops::ReductionOp;
            let axis = op_spec
                .params
                .get("axis")
                .and_then(|p| p.resolve_usize(row_idx, ctx).ok());
            Ok(ViewDto::Reduction(ReductionOp::Min { axis }))
        }
        "reduce_mean" => {
            use view_buffer::ops::ReductionOp;
            let axis = op_spec
                .params
                .get("axis")
                .and_then(|p| p.resolve_usize(row_idx, ctx).ok());
            Ok(ViewDto::Reduction(ReductionOp::Mean { axis }))
        }
        "reduce_std" => {
            use view_buffer::ops::ReductionOp;
            let axis = op_spec
                .params
                .get("axis")
                .and_then(|p| p.resolve_usize(row_idx, ctx).ok());
            let ddof = op_spec
                .params
                .get("ddof")
                .map(|p| p.resolve_usize(row_idx, ctx).unwrap_or(0) as u8)
                .unwrap_or(0);
            Ok(ViewDto::Reduction(ReductionOp::Std { axis, ddof }))
        }
        "reduce_percentile" => {
            use view_buffer::ops::ReductionOp;
            let q = get_param(&op_spec.params, "q")?.resolve_f64(row_idx, ctx)?;
            Ok(ViewDto::Reduction(ReductionOp::Percentile { q }))
        }
        "reduce_argmax" => {
            use view_buffer::ops::ReductionOp;
            let axis = get_param(&op_spec.params, "axis")?.resolve_usize(row_idx, ctx)?;
            Ok(ViewDto::Reduction(ReductionOp::ArgMax { axis }))
        }
        "reduce_argmin" => {
            use view_buffer::ops::ReductionOp;
            let axis = get_param(&op_spec.params, "axis")?.resolve_usize(row_idx, ctx)?;
            Ok(ViewDto::Reduction(ReductionOp::ArgMin { axis }))
        }
        "extract_shape" => {
            // Extract shape returns buffer dimensions as a vector
            Ok(ViewDto::ExtractShape)
        }
        "label_reduce" => {
            let contours_param = get_param(&op_spec.params, "contours")?;
            // The contour column is referenced by *name* inside the ViewDto
            // (a view-buffer type), so graph compilation deliberately leaves
            // this param unbound; the executor maps the name to its input
            // slot via `CompiledGraph::name_to_slot`.
            let contours_expr = match contours_param {
                ParamValue::Expr { col: Some(col), .. } => col.clone(),
                ParamValue::Expr { col: None, .. } => {
                    return Err(polars_err!(
                        ComputeError: "label_reduce requires a contour expression with a column key"
                    ))
                }
                ParamValue::Literal { .. } | ParamValue::Slot { .. } => {
                    return Err(polars_err!(
                        ComputeError: "label_reduce contours parameter must be a Polars expression"
                    ))
                }
            };
            let reduction = op_spec
                .params
                .get("reduction")
                .map(|p| p.resolve_string())
                .transpose()?
                .unwrap_or("max");
            let region_mode = op_spec
                .params
                .get("region_mode")
                .map(|p| p.resolve_string())
                .transpose()?
                .unwrap_or("interior");
            Ok(ViewDto::LabelReduce {
                contours_expr,
                reduction: reduction.to_string(),
                region_mode: region_mode.to_string(),
            })
        }

        // Histogram operation
        "histogram" => {
            use view_buffer::ops::histogram::{HistogramClosed, HistogramOp, HistogramOutput};

            let bins_param = get_param(&op_spec.params, "bins")?;
            let (bins_count, edges) = if let Some(edges) = bins_param.as_f64_vec() {
                // If it's a vector, those are the edges
                (edges.len().saturating_sub(1), Some(edges))
            } else {
                (bins_param.resolve_usize(row_idx, ctx)?, None)
            };

            // Parse closed mode
            let closed_str = get_param(&op_spec.params, "closed")?.resolve_string()?;
            let closed = match closed_str {
                "left" => HistogramClosed::Left,
                "right" => HistogramClosed::Right,
                other => {
                    return Err(
                        polars_err!(ComputeError: "Unknown histogram closed mode: {}", other),
                    )
                }
            };

            // Parse output mode
            let output_str = get_param(&op_spec.params, "output")?.resolve_string()?;
            let output = match output_str {
                "counts" => HistogramOutput::Counts,
                "normalized" => HistogramOutput::Normalized,
                "quantized" => HistogramOutput::Quantized,
                "edges" => HistogramOutput::Edges,
                "buckets" => HistogramOutput::Buckets,
                other => {
                    return Err(
                        polars_err!(ComputeError: "Unknown histogram output mode: {}", other),
                    )
                }
            };

            // Parse optional range
            let range = if op_spec.params.contains_key("range_min")
                && op_spec.params.contains_key("range_max")
            {
                let range_min =
                    get_param(&op_spec.params, "range_min")?.resolve_f64(row_idx, ctx)?;
                let range_max =
                    get_param(&op_spec.params, "range_max")?.resolve_f64(row_idx, ctx)?;
                Some((range_min, range_max))
            } else {
                None
            };

            let mut op = HistogramOp::new(bins_count)
                .with_output(output)
                .with_closed(closed);
            if let Some(e) = edges {
                op = op.with_edges(e);
            }
            if let Some((min, max)) = range {
                op = op.with_range(min, max);
            }

            Ok(ViewDto::Histogram(op))
        }

        // Channel operations
        "channel_select" => {
            let index = get_param(&op_spec.params, "index")?.resolve_usize(row_idx, ctx)?;
            Ok(ViewDto::View(ViewOp::ChannelSelect { index }))
        }
        "channel_swap" => {
            let order = get_param(&op_spec.params, "order")?.as_int_list()?;
            Ok(ViewDto::ChannelSwap { order })
        }
        "channel_merge" => {
            let other_nodes_param = get_param(&op_spec.params, "other_nodes")?;
            let other_node_ids = match other_nodes_param {
                ParamValue::Literal {
                    value: serde_json::Value::Array(arr),
                } => arr
                    .iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect(),
                _ => {
                    return Err(
                        polars_err!(ComputeError: "channel_merge other_nodes must be an array of node IDs"),
                    )
                }
            };
            Ok(ViewDto::ChannelMerge { other_node_ids })
        }

        // Intensity operations
        "adjust_contrast" => {
            let factor = get_param(&op_spec.params, "factor")?.resolve_f32(row_idx, ctx)?;
            Ok(ViewDto::Compute(ComputeOp::AdjustContrast(factor)))
        }
        "adjust_gamma" => {
            let gamma = get_param(&op_spec.params, "gamma")?.resolve_f32(row_idx, ctx)?;
            Ok(ViewDto::Compute(ComputeOp::AdjustGamma(gamma)))
        }
        "invert" => Ok(ViewDto::Compute(ComputeOp::Invert)),

        // Color space conversion
        "cvt_color" => {
            use view_buffer::ops::color::{ColorConvertOp, ColorSpace};

            let from_str = get_param(&op_spec.params, "from_space")?.resolve_string()?;
            let to_str = get_param(&op_spec.params, "to_space")?.resolve_string()?;
            let from = ColorSpace::from_str_name(from_str)
                .ok_or_else(|| polars_err!(ComputeError: "Unknown color space: {}", from_str))?;
            let to = ColorSpace::from_str_name(to_str)
                .ok_or_else(|| polars_err!(ComputeError: "Unknown color space: {}", to_str))?;
            Ok(ViewDto::Color(ColorConvertOp { from, to }))
        }

        // Convolution / filter operations
        "convolve2d" => {
            use view_buffer::ops::filter::{BorderMode, ConvolveOp};

            let kernel = get_param(&op_spec.params, "kernel")?
                .as_f32_vec()
                .ok_or_else(
                    || polars_err!(ComputeError: "convolve2d requires 'kernel' as array of floats"),
                )?;
            let ksize = get_param(&op_spec.params, "ksize")?.resolve_usize(row_idx, ctx)?;
            let normalize = op_spec
                .params
                .get("normalize")
                .map(|p| {
                    matches!(
                        p,
                        ParamValue::Literal {
                            value: serde_json::Value::Bool(true)
                        }
                    )
                })
                .unwrap_or(false);
            let border_str = op_spec
                .params
                .get("border")
                .map(|p| p.resolve_string())
                .transpose()?
                .unwrap_or("replicate");
            let border = match border_str {
                "replicate" => BorderMode::Replicate,
                "zero" => BorderMode::Zero,
                "reflect" => BorderMode::Reflect,
                other => return Err(polars_err!(ComputeError: "Unknown border mode: {}", other)),
            };

            Ok(ViewDto::Filter(ConvolveOp {
                kernel,
                ksize,
                normalize,
                border,
            }))
        }
        "erode" => {
            let ksize = get_param(&op_spec.params, "ksize")?.resolve_u32(row_idx, ctx)?;
            let iterations = op_spec
                .params
                .get("iterations")
                .map(|p| p.resolve_u32(row_idx, ctx))
                .transpose()?
                .unwrap_or(1);
            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Erode { ksize, iterations },
            }))
        }
        "dilate" => {
            let ksize = get_param(&op_spec.params, "ksize")?.resolve_u32(row_idx, ctx)?;
            let iterations = op_spec
                .params
                .get("iterations")
                .map(|p| p.resolve_u32(row_idx, ctx))
                .transpose()?
                .unwrap_or(1);
            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Dilate { ksize, iterations },
            }))
        }
        "morphology_gradient" => {
            let ksize = get_param(&op_spec.params, "ksize")?.resolve_u32(row_idx, ctx)?;
            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::MorphGradient { ksize },
            }))
        }

        "canny" => {
            let low_threshold =
                get_param(&op_spec.params, "low_threshold")?.resolve_f32(row_idx, ctx)?;
            let high_threshold =
                get_param(&op_spec.params, "high_threshold")?.resolve_f32(row_idx, ctx)?;
            Ok(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Canny {
                    low_threshold,
                    high_threshold,
                },
            }))
        }
        "equalize_histogram" => Ok(ViewDto::Image(ImageOp {
            kind: ImageOpKind::HistogramEqualize,
        })),

        // Mask operation
        "apply_mask" => {
            let mask_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
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

#[cfg(test)]
mod known_ops_tests {
    use super::*;
    use std::collections::HashMap;

    /// Build an OpSpec with no params (enough to exercise the name dispatch).
    fn op(name: &str) -> OpSpec {
        OpSpec {
            op: name.to_string(),
            params: HashMap::new(),
        }
    }

    /// Every name in KNOWN_OPS must be a real resolve_op arm: with empty params
    /// most arms fail with a missing-param error, but none may fall through to
    /// the "Unknown operation" catch-all.
    #[test]
    fn known_ops_all_resolve() {
        let ctx = ParamCtx::empty();
        for name in KNOWN_OPS {
            if let Err(e) = resolve_op(&op(name), 0, &ctx) {
                let msg = e.to_string();
                assert!(
                    !msg.contains("Unknown operation"),
                    "KNOWN_OPS lists '{name}' but resolve_op has no arm for it: {msg}"
                );
            }
        }
    }

    /// A name that is not an arm must be rejected by the catch-all, so the
    /// registry can't silently accept bogus ops.
    #[test]
    fn unknown_op_is_rejected() {
        let ctx = ParamCtx::empty();
        let err = resolve_op(&op("definitely_not_a_real_op"), 0, &ctx)
            .expect_err("bogus op must not resolve");
        assert!(err.to_string().contains("Unknown operation"));
    }

    /// KNOWN_OPS must be sorted and unique so the registry is easy to scan and
    /// diff against the Python OP_NAMES set.
    #[test]
    fn known_ops_sorted_and_unique() {
        for pair in KNOWN_OPS.windows(2) {
            assert!(
                pair[0] < pair[1],
                "KNOWN_OPS must be sorted/unique; '{}' !< '{}'",
                pair[0],
                pair[1]
            );
        }
    }
}
