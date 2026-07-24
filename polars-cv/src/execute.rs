//! Pipeline execution engine.
//!
//! This module handles the execution of vision pipelines on Polars Series,
//! including parameter resolution and view-buffer integration.

use polars::prelude::*;
use std::collections::HashMap;

use view_buffer::{
    geometry::rasterize::rasterize, AffineParams, BinaryOp, ComputeOp, DType, FilterType,
    GeometryOp, ImageAdapter, ImageOp, ImageOpKind, InterpolationType, NormalizeMethod, ViewBuffer,
    ViewDto, ViewOp,
};

use crate::graph::step::GraphStep;
use crate::params::{get, ParamCtx, ParamValue};
use crate::pipeline::{OpSpec, SinkSpec, SourceSpec};
use view_buffer::geometry::label::{LabelReduction, LabelRegionMode};
use view_buffer::naming;

/// The Python-facing name of every two-buffer binary operation.
///
/// Single authority consumed by `resolve_op` (one match arm for the whole
/// family) and by `lib.rs::parse_binary_op` (the planner's two-input dtype
/// query), so the two cannot drift.
pub(crate) const BINARY_OPS: &[(&str, BinaryOp)] = &[
    ("add", BinaryOp::Add),
    ("subtract", BinaryOp::Subtract),
    ("multiply", BinaryOp::Multiply),
    ("divide", BinaryOp::Divide),
    ("blend", BinaryOp::Blend),
    ("ratio", BinaryOp::Ratio),
    ("maximum", BinaryOp::Maximum),
    ("minimum", BinaryOp::Minimum),
    ("bitwise_and", BinaryOp::BitwiseAnd),
    ("bitwise_or", BinaryOp::BitwiseOr),
    ("bitwise_xor", BinaryOp::BitwiseXor),
];

/// Parse the optional `interpolation` parameter (shared by `rotate` and
/// `warp_affine`; defaults to bilinear).
fn resolve_interpolation(params: &HashMap<String, ParamValue>) -> PolarsResult<InterpolationType> {
    get::opt_enum(
        params,
        "interpolation",
        InterpolationType::NAMED,
        &[],
        InterpolationType::Bilinear,
    )
}

/// Parse the optional `border_value` parameter (shared by `rotate` and
/// `warp_affine`; defaults to 0.0).
fn resolve_border_value(
    params: &HashMap<String, ParamValue>,
    row_idx: usize,
    ctx: &ParamCtx,
) -> PolarsResult<f64> {
    get::opt_f64(params, "border_value", 0.0, row_idx, ctx)
}

/// Parse rasterize's optional style parameters `(fill_value, background,
/// anti_alias)` — shared with the graph executor's rasterize-by-shape-ref
/// path so the two sites cannot diverge.
pub(crate) fn resolve_rasterize_style(
    params: &HashMap<String, ParamValue>,
    row_idx: usize,
    ctx: &ParamCtx,
) -> PolarsResult<(u8, u8, bool)> {
    Ok((
        get::opt_u8(params, "fill_value", 255, row_idx, ctx)?,
        get::opt_u8(params, "background", 0, row_idx, ctx)?,
        get::opt_bool(params, "anti_alias", false)?,
    ))
}

/// Decode a contour source by parsing the struct and rasterizing to ViewBuffer.
pub fn decode_contour_source(
    value: &AnyValue,
    row_idx: usize,
    source: &SourceSpec,
    ctx: &ParamCtx,
) -> PolarsResult<ViewBuffer> {
    // Parse via the plugin's single contour parser (contour.rs).
    let contour = crate::contour::parse_contour(value)?;

    // Resolve dimensions
    let (width, height) = resolve_contour_dimensions(row_idx, source, ctx)?;

    // Get fill and background values
    let fill_value = source.fill_value;
    let background = source.background;

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
    // Parse via the plugin's single contour parser (contour.rs).
    let contour = crate::contour::parse_contour(value)?;

    // Rasterize the contour to a ViewBuffer
    Ok(rasterize(
        &contour, width, height, fill_value, background, false, // anti_alias not yet supported
    ))
}

/// Resolve contour dimensions from pipeline source spec.
fn resolve_contour_dimensions(
    row_idx: usize,
    source: &SourceSpec,
    ctx: &ParamCtx,
) -> PolarsResult<(u32, u32)> {
    // shape_pipeline sources never reach this function: the graph executor
    // resolves the referenced node's dimensions and calls
    // `decode_contour_source_with_dims` instead (see compiled.rs).

    // Get explicit width and height
    let width = source
        .width
        .as_ref()
        .ok_or_else(|| polars_err!(ComputeError: "Contour source requires 'width' parameter"))?
        .resolve_usize(row_idx, ctx)? as u32;

    let height = source
        .height
        .as_ref()
        .ok_or_else(|| polars_err!(ComputeError: "Contour source requires 'height' parameter"))?
        .resolve_usize(row_idx, ctx)? as u32;

    Ok((width, height))
}

/// Decode a JPEG at a reduced IDCT scale sufficient for `max_size` pixels on
/// the long side.
///
/// Picks the smallest of the decoder's supported scale factors (1/8, 1/4,
/// 1/2, 1) whose output is >= `max_size` on at least one axis, so the long
/// side never drops below `min(max_size, original)` — downstream resizes
/// down to `max_size` never upscale. Returns `None` for non-JPEG bytes or
/// pixel formats the scaled path does not cover (16-bit, CMYK); the caller
/// falls back to the full decoder.
fn decode_jpeg_scaled(bytes: &[u8], max_size: u32) -> Option<ViewBuffer> {
    // JPEG SOI marker; anything else takes the regular decode path.
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let requested = max_size.min(u16::MAX as u32) as u16;
    let (width, height) = decoder.scale(requested, requested).ok()?;
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let (h, w) = (height as usize, width as usize);
    // Shapes mirror ImageAdapter::decode: grayscale is [H, W, 1].
    match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => {
            Some(ViewBuffer::from_vec_with_shape(pixels, vec![h, w, 1]))
        }
        jpeg_decoder::PixelFormat::RGB24 => {
            Some(ViewBuffer::from_vec_with_shape(pixels, vec![h, w, 3]))
        }
        _ => None,
    }
}

/// Decode image-format source bytes into a ViewBuffer.
///
/// Only `image_bytes` is handled here (`file_path` sources are rewritten to
/// `image_bytes` after the file is read). `blob`/`raw` sources never reach
/// this function: the graph executor decodes them zero-copy via
/// `graph::decode::decode_binary_zero_copy` (pinned by the
/// `blob_and_raw_sources_decode_via_zero_copy` test).
pub fn decode_source(bytes: &[u8], source: &SourceSpec) -> PolarsResult<ViewBuffer> {
    match source.format.as_str() {
        "image_bytes" => {
            // An explicit decode-scale assertion lets JPEG decode skip work
            // via IDCT scaling; other formats fall through to a full decode.
            let scaled = source
                .decode_max_size
                .and_then(|max_size| decode_jpeg_scaled(bytes, max_size));
            let buf = match scaled {
                Some(buf) => buf,
                None => ImageAdapter::decode(bytes)
                    .map_err(|e| polars_err!(ComputeError: "Failed to decode image: {:?}", e))?,
            };
            // If source spec declares an expected dtype, cast to it.
            // This is a no-op when the decoded dtype already matches.
            if let Some(ref dtype_str) = source.dtype {
                let target = parse_dtype(dtype_str)?;
                if buf.dtype() != target {
                    return Ok(buf.cast(target));
                }
            }
            Ok(buf)
        }
        other => Err(polars_err!(ComputeError: "Unknown source format: {}", other)),
    }
}

/// Encode the result buffer to a binary sink format.
///
/// Handles the byte-producing sinks only: `png`/`jpeg`/`webp`/`tiff`/`blob`.
/// The other sink formats never reach this function — `numpy`/`torch` are
/// encoded as zero-copy structs (`crate::output`) and `list`/`array` as typed
/// nested values, both directly in `graph::encode::encode_node_output` (the
/// sole caller).
pub fn encode_sink(buffer: &ViewBuffer, sink: &SinkSpec) -> PolarsResult<Vec<u8>> {
    match sink.format.as_str() {
        "blob" => {
            // VIEW protocol
            Ok(buffer.to_blob())
        }
        "png" => ImageAdapter::encode(buffer, image::ImageFormat::Png)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode PNG: {:?}", e)),
        "jpeg" => {
            if buffer.dtype() != DType::U8 {
                return Err(polars_err!(
                    ComputeError:
                    "JPEG is an 8-bit format but the image is {:?}; cast to u8 first \
                     (.cast(\"u8\")) or sink to PNG/TIFF to preserve higher bit depth.",
                    buffer.dtype()
                ));
            }
            let quality = sink.quality;
            ImageAdapter::encode_jpeg(buffer, quality)
                .map_err(|e| polars_err!(ComputeError: "Failed to encode JPEG: {:?}", e))
        }
        "webp" => {
            if buffer.dtype() != DType::U8 {
                return Err(polars_err!(
                    ComputeError:
                    "WebP is an 8-bit format but the image is {:?}; cast to u8 first \
                     (.cast(\"u8\")) or sink to PNG/TIFF to preserve higher bit depth.",
                    buffer.dtype()
                ));
            }
            ImageAdapter::encode(buffer, image::ImageFormat::WebP)
                .map_err(|e| polars_err!(ComputeError: "Failed to encode WebP: {:?}", e))
        }
        "tiff" => ImageAdapter::encode_tiff(buffer)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode TIFF: {:?}", e)),
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
    "contour_perimeter",
    "contour_scale",
    "contour_simplify",
    "contour_translate",
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

/// A fusable single-buffer engine op, as a resolved step.
fn buffer_step(dto: ViewDto) -> PolarsResult<GraphStep> {
    Ok(GraphStep::Buffer(dto))
}

/// Resolve an operation specification to a [`GraphStep`].
///
/// Single-buffer ops become `GraphStep::Buffer(ViewDto)` (executed via the
/// engine's `ViewExpr`); multi-input and domain-changing ops become typed
/// graph-level steps. Node references and expression column names enter the
/// step here — they never reach the engine's `ViewDto`.
pub fn resolve_op(op_spec: &OpSpec, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
    match op_spec.op.as_str() {
        // View operations
        "transpose" => {
            let axes = get_param(&op_spec.params, "axes")?.as_int_list()?;
            buffer_step(ViewDto::View(ViewOp::Transpose(axes)))
        }
        "reshape" => {
            // Borrow the compiled `List` on the hot path; parse once otherwise.
            let shape_param = get_param(&op_spec.params, "shape")?;
            let owned_shape;
            let shape_params: &[ParamValue] = match shape_param.as_param_slice() {
                Some(slice) => slice,
                None => {
                    owned_shape = shape_param.as_param_list()?;
                    &owned_shape
                }
            };
            let shape: Vec<usize> = shape_params
                .iter()
                .map(|p| p.resolve_usize(row_idx, ctx))
                .collect::<PolarsResult<_>>()?;
            buffer_step(ViewDto::View(ViewOp::Reshape(shape)))
        }
        "flip" => {
            let axes = get_param(&op_spec.params, "axes")?.as_int_list()?;
            buffer_step(ViewDto::View(ViewOp::Flip(axes)))
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

            buffer_step(ViewDto::View(ViewOp::Crop { start, end }))
        }

        // Compute operations
        "cast" => {
            let dtype_str = get_param(&op_spec.params, "dtype")?.resolve_string()?;
            let dtype = parse_dtype(dtype_str)?;
            buffer_step(ViewDto::Compute(ComputeOp::Cast(dtype)))
        }
        "scale" => {
            let factor = get_param(&op_spec.params, "factor")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Compute(ComputeOp::Scale(factor)))
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
                    return Err(polars_err!(ComputeError:
                        "parameter 'method': unknown value '{}', expected one of {:?}",
                        other, NormalizeMethod::NAMES))
                }
            };
            // Honor the configured output dtype so the produced buffer matches
            // the planner's `Configurable(F32)` resolution (default f32). Without
            // this, `normalize(out_dtype=...)` planned one dtype and executed
            // another — a plan/execution contract violation.
            let out_dtype = match op_spec.params.get("out_dtype") {
                Some(p) => parse_dtype(p.resolve_string()?)?,
                None => DType::F32,
            };
            buffer_step(ViewDto::Compute(ComputeOp::Normalize(method, out_dtype)))
        }
        "clamp" => {
            let min = get_param(&op_spec.params, "min")?.resolve_f32(row_idx, ctx)?;
            let max = get_param(&op_spec.params, "max")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Compute(ComputeOp::Clamp { min, max }))
        }
        "relu" => buffer_step(ViewDto::Compute(ComputeOp::Relu)),

        // Image operations
        "resize" => {
            let height = get_param(&op_spec.params, "height")?.resolve_u32(row_idx, ctx)?;
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            buffer_step(ViewDto::Image(ImageOp {
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

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeScale {
                    scale_x,
                    scale_y,
                    filter,
                },
            }))
        }
        "resize_to_height" => {
            let height = get_param(&op_spec.params, "height")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeToHeight { height, filter },
            }))
        }
        "resize_to_width" => {
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeToWidth { width, filter },
            }))
        }
        "resize_max" => {
            let max_size = get_param(&op_spec.params, "max_size")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeMax { max_size, filter },
            }))
        }
        "resize_min" => {
            let min_size = get_param(&op_spec.params, "min_size")?.resolve_u32(row_idx, ctx)?;
            let filter_str = get_param(&op_spec.params, "filter")?.resolve_string()?;
            let filter = parse_filter(filter_str)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeMin { min_size, filter },
            }))
        }

        // Padding operations
        "pad" => {
            use view_buffer::ops::dto::PadMode;

            let top = get_param(&op_spec.params, "top")?.resolve_u32(row_idx, ctx)?;
            let bottom = get_param(&op_spec.params, "bottom")?.resolve_u32(row_idx, ctx)?;
            let left = get_param(&op_spec.params, "left")?.resolve_u32(row_idx, ctx)?;
            let right = get_param(&op_spec.params, "right")?.resolve_u32(row_idx, ctx)?;
            let value = get_param(&op_spec.params, "value")?.resolve_f32(row_idx, ctx)?;
            let mode = get::req_enum(&op_spec.params, "mode", PadMode::NAMED, &[])?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Pad {
                    top,
                    bottom,
                    left,
                    right,
                    value,
                    mode,
                },
            }))
        }
        "pad_to_size" => {
            use view_buffer::ops::dto::PadPosition;

            let height = get_param(&op_spec.params, "height")?.resolve_u32(row_idx, ctx)?;
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, ctx)?;
            let value = get_param(&op_spec.params, "value")?.resolve_f32(row_idx, ctx)?;
            let position = get::req_enum(&op_spec.params, "position", PadPosition::NAMED, &[])?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::PadToSize {
                    height,
                    width,
                    position,
                    value,
                },
            }))
        }
        "letterbox" => {
            let height = get_param(&op_spec.params, "height")?.resolve_u32(row_idx, ctx)?;
            let width = get_param(&op_spec.params, "width")?.resolve_u32(row_idx, ctx)?;
            let value = get_param(&op_spec.params, "value")?.resolve_f32(row_idx, ctx)?;

            // Letterbox has always resized with lanczos3; kept as the default
            // (now overridable per-op if the builder ever exposes it).
            let filter = get::opt_enum(
                &op_spec.params,
                "filter",
                FilterType::NAMED,
                FilterType::ALIASES,
                FilterType::Lanczos3,
            )?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Letterbox {
                    height,
                    width,
                    value,
                    filter,
                },
            }))
        }
        "grayscale" => buffer_step(ViewDto::Image(ImageOp {
            kind: ImageOpKind::Grayscale,
        })),
        "threshold" => {
            let value = get_param(&op_spec.params, "value")?.resolve_f64(row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Threshold(value),
            }))
        }
        "blur" => {
            let sigma = get_param(&op_spec.params, "sigma")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Blur { sigma },
            }))
        }
        "rotate" => {
            let angle = get_param(&op_spec.params, "angle")?.resolve_f32(row_idx, ctx)?;
            let expand = get::opt_bool(&op_spec.params, "expand", false)?;

            let normalized_angle = angle % 360.0;
            let normalized_angle = if normalized_angle < 0.0 {
                normalized_angle + 360.0
            } else {
                normalized_angle
            };

            const EPSILON: f32 = 0.001;
            if (normalized_angle - 90.0).abs() < EPSILON {
                buffer_step(ViewDto::View(ViewOp::Rotate90))
            } else if (normalized_angle - 180.0).abs() < EPSILON {
                buffer_step(ViewDto::View(ViewOp::Rotate180))
            } else if (normalized_angle - 270.0).abs() < EPSILON {
                buffer_step(ViewDto::View(ViewOp::Rotate270))
            } else if normalized_angle.abs() < EPSILON || (normalized_angle - 360.0).abs() < EPSILON
            {
                buffer_step(ViewDto::Compute(ComputeOp::RotateAffine {
                    angle_deg: 0.0,
                    expand: false,
                    interpolation: InterpolationType::Bilinear,
                    border_value: 0.0,
                }))
            } else {
                // Route arbitrary angles through AffineParams for unified code path.
                // The affine matrix is built at execution time from the current
                // buffer dimensions (handled by RotateToAffine).
                let interpolation = resolve_interpolation(&op_spec.params)?;
                let border_value = resolve_border_value(&op_spec.params, row_idx, ctx)?;

                buffer_step(ViewDto::Compute(ComputeOp::RotateAffine {
                    angle_deg: normalized_angle,
                    expand,
                    interpolation,
                    border_value,
                }))
            }
        }

        // Affine warp operation
        "warp_affine" => {
            // Each matrix element is its own ParamValue so any of the six can be
            // a per-row expression (a different affine per row in one call). The
            // compiled graph pre-parses this into a `List`, borrowed here with no
            // per-row allocation; the JSON-introspection path parses once.
            let matrix_param = get_param(&op_spec.params, "matrix")?;
            let owned_matrix;
            let matrix_params: &[ParamValue] = match matrix_param.as_param_slice() {
                Some(slice) => slice,
                None => {
                    owned_matrix = matrix_param.as_param_list()?;
                    &owned_matrix
                }
            };
            if matrix_params.len() != 6 {
                return Err(polars_err!(
                    ComputeError: "warp_affine: matrix must have exactly 6 elements, got {}",
                    matrix_params.len()
                ));
            }
            let matrix_vec: Vec<f64> = matrix_params
                .iter()
                .map(|p| p.resolve_f64(row_idx, ctx))
                .collect::<PolarsResult<Vec<f64>>>()?;
            let matrix: [f64; 6] = matrix_vec
                .try_into()
                .map_err(|_| polars_err!(ComputeError: "warp_affine: matrix conversion failed"))?;

            let output_height =
                get_param(&op_spec.params, "output_height")?.resolve_u32(row_idx, ctx)?;
            let output_width =
                get_param(&op_spec.params, "output_width")?.resolve_u32(row_idx, ctx)?;

            let interpolation = resolve_interpolation(&op_spec.params)?;
            let border_value = resolve_border_value(&op_spec.params, row_idx, ctx)?;

            buffer_step(ViewDto::Compute(ComputeOp::Affine(AffineParams {
                matrix,
                output_height,
                output_width,
                interpolation,
                border_value,
            })))
        }

        // Perceptual hash operation — a graph-level vector producer (image
        // buffer → 1-D u8 fingerprint), executed via `apply_perceptual_hash`.
        "perceptual_hash" => {
            use view_buffer::ops::phash::{HashAlgorithm, PerceptualHashOp};

            let algorithm = get::opt_enum(
                &op_spec.params,
                "algorithm",
                HashAlgorithm::NAMED,
                &[],
                HashAlgorithm::Perceptual,
            )?;
            // `hash_size` fixes the output vector length, so it is a structural
            // (literal-only) param — reject a bound expression slot.
            let hash_size = get::opt_u32_literal(&op_spec.params, "hash_size", 64)?;

            Ok(GraphStep::PerceptualHash(
                PerceptualHashOp::new(algorithm).with_hash_size(hash_size),
            ))
        }

        // Geometry operations
        "rasterize" => {
            let width = get_param(&op_spec.params, "width")?.resolve_usize(row_idx, ctx)? as u32;
            let height = get_param(&op_spec.params, "height")?.resolve_usize(row_idx, ctx)? as u32;
            let (fill_value, background, anti_alias) =
                resolve_rasterize_style(&op_spec.params, row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Rasterize {
                width,
                height,
                fill_value,
                background,
                anti_alias,
            }))
        }
        "extract_contours" => {
            use view_buffer::geometry::ops::{ApproxMethod, ExtractMode};

            let mode = get::opt_enum(
                &op_spec.params,
                "mode",
                ExtractMode::NAMED,
                &[],
                ExtractMode::External,
            )?;
            let method = get::opt_enum(
                &op_spec.params,
                "method",
                ApproxMethod::NAMED,
                &[],
                ApproxMethod::Simple,
            )?;
            let min_area = get::maybe_f64(&op_spec.params, "min_area", row_idx, ctx)?;

            Ok(GraphStep::Geometry(GeometryOp::ExtractContours {
                mode,
                method,
                min_area,
            }))
        }

        // Geometry measure operations
        "contour_area" => {
            let signed = get::opt_bool(&op_spec.params, "signed", false)?;
            Ok(GraphStep::Geometry(GeometryOp::Area { signed }))
        }
        "contour_perimeter" => Ok(GraphStep::Geometry(GeometryOp::Perimeter)),
        "contour_centroid" => Ok(GraphStep::Geometry(GeometryOp::Centroid)),
        "contour_bounding_box" => Ok(GraphStep::Geometry(GeometryOp::BoundingBox)),
        "contour_convex_hull" => Ok(GraphStep::Geometry(GeometryOp::ConvexHull)),

        // Geometry transforms
        "contour_translate" => {
            let dx = get_param(&op_spec.params, "dx")?.resolve_f64(row_idx, ctx)?;
            let dy = get_param(&op_spec.params, "dy")?.resolve_f64(row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Translate { dx, dy }))
        }
        "contour_scale" => {
            let sx = get_param(&op_spec.params, "sx")?.resolve_f64(row_idx, ctx)?;
            let sy = get_param(&op_spec.params, "sy")?.resolve_f64(row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Scale {
                sx,
                sy,
                origin: view_buffer::geometry::ops::ScaleOrigin::Centroid,
            }))
        }
        "contour_simplify" => {
            let tolerance = get_param(&op_spec.params, "tolerance")?.resolve_f64(row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Simplify { tolerance }))
        }

        // Binary operations (two-buffer): one arm for the whole family,
        // dispatched through the BINARY_OPS name table.
        name if naming::lookup(BINARY_OPS, name).is_some() => {
            let op = naming::lookup(BINARY_OPS, name).expect("guard checked membership");
            let other_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            Ok(GraphStep::Binary {
                op,
                other: other_node_id,
            })
        }

        // Reduction operations
        "reduce_sum" => {
            use view_buffer::ops::ReductionOp;
            // Global reduction: axis = None means reduce entire array to scalar
            Ok(GraphStep::Reduction(ReductionOp::Sum { axis: None }))
        }
        "reduce_popcount" => {
            use view_buffer::ops::ReductionOp;
            // Count set bits across entire buffer (for Hamming distance)
            Ok(GraphStep::Reduction(ReductionOp::PopCount))
        }
        "reduce_max" => {
            use view_buffer::ops::ReductionOp;
            // `axis` is structural — it fixes the output rank at plan time, so
            // it must be a literal (a bound expression slot is rejected).
            let axis = get::maybe_usize_literal(&op_spec.params, "axis")?;
            Ok(GraphStep::Reduction(ReductionOp::Max { axis }))
        }
        "reduce_min" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(&op_spec.params, "axis")?;
            Ok(GraphStep::Reduction(ReductionOp::Min { axis }))
        }
        "reduce_mean" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(&op_spec.params, "axis")?;
            Ok(GraphStep::Reduction(ReductionOp::Mean { axis }))
        }
        "reduce_std" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(&op_spec.params, "axis")?;
            let ddof = get::opt_u8(&op_spec.params, "ddof", 0, row_idx, ctx)?;
            Ok(GraphStep::Reduction(ReductionOp::Std { axis, ddof }))
        }
        "reduce_percentile" => {
            use view_buffer::ops::ReductionOp;
            let q = get_param(&op_spec.params, "q")?.resolve_f64(row_idx, ctx)?;
            Ok(GraphStep::Reduction(ReductionOp::Percentile { q }))
        }
        "reduce_argmax" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(&op_spec.params, "axis")?
                .ok_or_else(|| polars_err!(ComputeError: "Missing required parameter: axis"))?;
            Ok(GraphStep::Reduction(ReductionOp::ArgMax { axis }))
        }
        "reduce_argmin" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(&op_spec.params, "axis")?
                .ok_or_else(|| polars_err!(ComputeError: "Missing required parameter: axis"))?;
            Ok(GraphStep::Reduction(ReductionOp::ArgMin { axis }))
        }
        "extract_shape" => {
            // Extract shape returns buffer dimensions as a vector
            Ok(GraphStep::ExtractShape)
        }
        "label_reduce" => {
            let contours_param = get_param(&op_spec.params, "contours")?;
            // The contour column is referenced by *name* inside the step, so
            // graph compilation deliberately leaves this param unbound; the
            // executor maps the name to its input slot via
            // `CompiledGraph::name_to_slot`.
            let contours_col = match contours_param {
                ParamValue::Expr { col: Some(col), .. } => col.clone(),
                ParamValue::Expr { col: None, .. } => {
                    return Err(polars_err!(
                        ComputeError: "label_reduce requires a contour expression with a column key"
                    ))
                }
                ParamValue::Literal { .. } | ParamValue::Slot { .. } | ParamValue::List(_) => {
                    return Err(polars_err!(
                        ComputeError: "label_reduce contours parameter must be a Polars expression"
                    ))
                }
            };
            let reduction = get::opt_enum(
                &op_spec.params,
                "reduction",
                LabelReduction::NAMED,
                &[],
                LabelReduction::Max,
            )?;
            let region_mode = get::opt_enum(
                &op_spec.params,
                "region_mode",
                LabelRegionMode::NAMED,
                &[],
                LabelRegionMode::Interior,
            )?;
            Ok(GraphStep::LabelReduce {
                contours_col,
                reduction,
                region_mode,
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

            let closed = get::req_enum(&op_spec.params, "closed", HistogramClosed::NAMED, &[])?;
            let output = get::req_enum(&op_spec.params, "output", HistogramOutput::NAMED, &[])?;

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

            Ok(GraphStep::Histogram(op))
        }

        // Channel operations
        "channel_select" => {
            let index = get_param(&op_spec.params, "index")?.resolve_usize(row_idx, ctx)?;
            buffer_step(ViewDto::View(ViewOp::ChannelSelect { index }))
        }
        "channel_swap" => {
            let order = get_param(&op_spec.params, "order")?.as_int_list()?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ChannelSwap { order },
            }))
        }
        "channel_merge" => {
            let other_nodes_param = get_param(&op_spec.params, "other_nodes")?;
            let other_node_ids = match other_nodes_param {
                ParamValue::Literal {
                    value: serde_json::Value::Array(arr),
                } => arr
                    .iter()
                    .map(|v| {
                        v.as_str().map(str::to_string).ok_or_else(|| {
                            polars_err!(ComputeError:
                                "parameter 'other_nodes' must be an array of node-ID strings, got {}", v)
                        })
                    })
                    .collect::<PolarsResult<Vec<_>>>()?,
                _ => {
                    return Err(
                        polars_err!(ComputeError: "parameter 'other_nodes' must be an array of node IDs"),
                    )
                }
            };
            Ok(GraphStep::ChannelMerge {
                others: other_node_ids,
            })
        }

        // Intensity operations
        "adjust_contrast" => {
            let factor = get_param(&op_spec.params, "factor")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Compute(ComputeOp::AdjustContrast(factor)))
        }
        "adjust_gamma" => {
            let gamma = get_param(&op_spec.params, "gamma")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Compute(ComputeOp::AdjustGamma(gamma)))
        }
        "invert" => buffer_step(ViewDto::Compute(ComputeOp::Invert)),

        // Color space conversion
        "cvt_color" => {
            use view_buffer::ops::color::{ColorConvertOp, ColorSpace};

            let from_str = get_param(&op_spec.params, "from_space")?.resolve_string()?;
            let to_str = get_param(&op_spec.params, "to_space")?.resolve_string()?;
            let from = ColorSpace::from_str_name(from_str).ok_or_else(|| {
                polars_err!(ComputeError:
                    "parameter 'from_space': unknown color space '{}', expected one of {:?}",
                    from_str, naming::names(ColorSpace::NAMED))
            })?;
            let to = ColorSpace::from_str_name(to_str).ok_or_else(|| {
                polars_err!(ComputeError:
                    "parameter 'to_space': unknown color space '{}', expected one of {:?}",
                    to_str, naming::names(ColorSpace::NAMED))
            })?;
            buffer_step(ViewDto::Color(ColorConvertOp { from, to }))
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
            let normalize = get::opt_bool(&op_spec.params, "normalize", false)?;
            let border = get::opt_enum(
                &op_spec.params,
                "border",
                BorderMode::NAMED,
                &[],
                BorderMode::Replicate,
            )?;

            buffer_step(ViewDto::Filter(ConvolveOp {
                kernel,
                ksize,
                normalize,
                border,
            }))
        }
        "erode" => {
            let ksize = get_param(&op_spec.params, "ksize")?.resolve_u32(row_idx, ctx)?;
            let iterations = get::opt_u32(&op_spec.params, "iterations", 1, row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Erode { ksize, iterations },
            }))
        }
        "dilate" => {
            let ksize = get_param(&op_spec.params, "ksize")?.resolve_u32(row_idx, ctx)?;
            let iterations = get::opt_u32(&op_spec.params, "iterations", 1, row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Dilate { ksize, iterations },
            }))
        }
        "morphology_gradient" => {
            let ksize = get_param(&op_spec.params, "ksize")?.resolve_u32(row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::MorphGradient { ksize },
            }))
        }

        "canny" => {
            let low_threshold =
                get_param(&op_spec.params, "low_threshold")?.resolve_f32(row_idx, ctx)?;
            let high_threshold =
                get_param(&op_spec.params, "high_threshold")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Canny {
                    low_threshold,
                    high_threshold,
                },
            }))
        }
        "equalize_histogram" => buffer_step(ViewDto::Image(ImageOp {
            kind: ImageOpKind::HistogramEqualize,
        })),

        // Mask operation
        "apply_mask" => {
            let mask_node_id = get_param(&op_spec.params, "other_node")?
                .resolve_string()?
                .to_string();
            let invert = get::opt_bool(&op_spec.params, "invert", false)?;
            Ok(GraphStep::ApplyMask {
                mask: mask_node_id,
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

/// Parse a dtype string to DType (canonical short names from `DType::NAMED`).
fn parse_dtype(s: &str) -> PolarsResult<DType> {
    DType::from_short_name(s).ok_or_else(|| {
        polars_err!(ComputeError:
            "Unknown dtype: {}, expected one of {:?}", s, naming::names(DType::NAMED))
    })
}

/// Parse a filter type string (canonical names from `FilterType::NAMED`,
/// plus parser-only aliases).
fn parse_filter(s: &str) -> PolarsResult<FilterType> {
    naming::lookup(FilterType::NAMED, s)
        .or_else(|| naming::lookup(FilterType::ALIASES, s))
        .ok_or_else(|| {
            polars_err!(ComputeError:
                "Unknown filter type: {}, expected one of {:?}",
                s, naming::names(FilterType::NAMED))
        })
}

#[cfg(test)]
mod strict_param_tests {
    //! One failure policy for operation parameters: an *absent* optional
    //! parameter takes its documented default, but a parameter that is
    //! *present and invalid* (unknown enum string, wrong type, out of range)
    //! must be an error — never silently coerced to a default. These tests
    //! pin that policy for every parameter that historically swallowed
    //! errors.

    use super::*;
    use crate::params::ParamValue;
    use serde_json::json;
    use std::collections::HashMap;

    fn op_with(name: &str, params: &[(&str, serde_json::Value)]) -> OpSpec {
        OpSpec {
            op: name.to_string(),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), ParamValue::Literal { value: v.clone() }))
                .collect::<HashMap<_, _>>(),
        }
    }

    fn resolve_err(spec: &OpSpec) -> String {
        resolve_op(spec, 0, &ParamCtx::empty())
            .expect_err("invalid parameter must be rejected")
            .to_string()
    }

    #[test]
    fn perceptual_hash_unknown_algorithm_errors() {
        let err = resolve_err(&op_with(
            "perceptual_hash",
            &[("algorithm", json!("phash"))],
        ));
        assert!(err.contains("algorithm"), "{err}");
        assert!(
            err.contains("perceptual"),
            "error must list valid names: {err}"
        );
    }

    #[test]
    fn perceptual_hash_invalid_hash_size_errors() {
        let err = resolve_err(&op_with(
            "perceptual_hash",
            &[("hash_size", json!("large"))],
        ));
        assert!(err.contains("hash_size"), "{err}");
    }

    #[test]
    fn perceptual_hash_defaults_apply_when_params_absent() {
        resolve_op(&op_with("perceptual_hash", &[]), 0, &ParamCtx::empty())
            .expect("absent optional params must take their defaults");
    }

    #[test]
    fn extract_contours_unknown_mode_errors() {
        let err = resolve_err(&op_with("extract_contours", &[("mode", json!("outer"))]));
        assert!(err.contains("mode"), "{err}");
    }

    #[test]
    fn extract_contours_unknown_method_errors() {
        let err = resolve_err(&op_with("extract_contours", &[("method", json!("fancy"))]));
        assert!(err.contains("method"), "{err}");
    }

    #[test]
    fn rasterize_invalid_fill_value_errors() {
        let base = [("width", json!(8)), ("height", json!(8))];
        let mut params = base.to_vec();
        params.push(("fill_value", json!("red")));
        let err = resolve_err(&op_with("rasterize", &params));
        assert!(err.contains("fill_value"), "{err}");

        let mut params = base.to_vec();
        params.push(("fill_value", json!(300)));
        let err = resolve_err(&op_with("rasterize", &params));
        assert!(
            err.contains("fill_value"),
            "out-of-range u8 must error: {err}"
        );

        let mut params = base.to_vec();
        params.push(("background", json!(-1)));
        let err = resolve_err(&op_with("rasterize", &params));
        assert!(err.contains("background"), "{err}");
    }

    #[test]
    fn reduce_axis_invalid_value_errors() {
        for op in ["reduce_max", "reduce_min", "reduce_mean"] {
            let err = resolve_err(&op_with(op, &[("axis", json!("rows"))]));
            assert!(err.contains("axis"), "{op}: {err}");
            // Absent axis must still mean a global reduction, not an error.
            resolve_op(&op_with(op, &[]), 0, &ParamCtx::empty())
                .expect("absent axis means global reduction");
        }
    }

    #[test]
    fn reduce_std_invalid_ddof_errors() {
        let err = resolve_err(&op_with("reduce_std", &[("ddof", json!("one"))]));
        assert!(err.contains("ddof"), "{err}");
        let err = resolve_err(&op_with("reduce_std", &[("ddof", json!(300))]));
        assert!(err.contains("ddof"), "out-of-range u8 must error: {err}");
    }

    #[test]
    fn bool_param_rejects_non_bool() {
        // Booleans are structural literals: a string/number must error, not
        // silently read as `false`.
        #[allow(clippy::type_complexity)]
        let cases: &[(&str, &str, &[(&str, serde_json::Value)])] = &[
            ("rotate", "expand", &[("angle", json!(45.0))]),
            (
                "rasterize",
                "anti_alias",
                &[("width", json!(8)), ("height", json!(8))],
            ),
            ("contour_area", "signed", &[]),
            (
                "convolve2d",
                "normalize",
                &[("kernel", json!(vec![0.0; 9])), ("ksize", json!(3))],
            ),
            ("apply_mask", "invert", &[("other_node", json!("m"))]),
        ];
        for (op, bool_param, base) in cases {
            let mut params = base.to_vec();
            params.push((bool_param, json!("yes")));
            let err = resolve_err(&op_with(op, &params));
            assert!(err.contains(bool_param), "{op}.{bool_param}: {err}");
        }
    }

    #[test]
    fn channel_merge_rejects_non_string_node_ids() {
        let err = resolve_err(&op_with("channel_merge", &[("other_nodes", json!([1, 2]))]));
        assert!(err.contains("other_nodes"), "{err}");
    }

    #[test]
    fn interpolation_shared_between_rotate_and_warp_affine() {
        let rotate_err = resolve_err(&op_with(
            "rotate",
            &[("angle", json!(45.0)), ("interpolation", json!("cubic"))],
        ));
        // The matrix is a list of per-element ParamValue dicts (each element may
        // be a per-row expression), matching what the Python planner emits.
        let ident = json!([
            {"type": "literal", "value": 1.0},
            {"type": "literal", "value": 0.0},
            {"type": "literal", "value": 0.0},
            {"type": "literal", "value": 0.0},
            {"type": "literal", "value": 1.0},
            {"type": "literal", "value": 0.0},
        ]);
        let warp_err = resolve_err(&op_with(
            "warp_affine",
            &[
                ("matrix", ident),
                ("output_height", json!(8)),
                ("output_width", json!(8)),
                ("interpolation", json!("cubic")),
            ],
        ));
        for err in [&rotate_err, &warp_err] {
            assert!(err.contains("interpolation"), "{err}");
            assert!(
                err.contains("nearest") && err.contains("bilinear"),
                "error must list valid names: {err}"
            );
        }
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

    /// Reverse guard: every top-level match arm in `resolve_op` must be listed
    /// in KNOWN_OPS, so a new arm cannot silently bypass the registry (the
    /// forward direction is covered by `known_ops_all_resolve`).
    ///
    /// The scan reads this file's source between the `resolve_op` header and
    /// its "Unknown operation" catch-all. Top-level arm patterns sit at one
    /// match-nesting level (8-space indent under rustfmt, which CI enforces);
    /// deeper string arms (e.g. normalize's method match) are excluded by the
    /// indent check.
    #[test]
    fn resolve_op_arms_are_all_known_ops() {
        let src = include_str!("execute.rs");
        let start = src.find("pub fn resolve_op").expect("resolve_op not found");
        let end = start
            + src[start..]
                .find("Unknown operation")
                .expect("resolve_op catch-all not found");
        let mut arm_names: Vec<&str> = Vec::new();
        for line in src[start..end].lines() {
            let trimmed = line.trim_start();
            let indent = line.len() - trimmed.len();
            if indent != 8 || !trimmed.starts_with('"') {
                continue;
            }
            // Arm patterns look like `"name" => {` or `"a" | "b" => ...`;
            // collect every string literal before the `=>`.
            let pattern = trimmed.split("=>").next().unwrap_or(trimmed);
            for (i, piece) in pattern.split('"').enumerate() {
                if i % 2 == 1 {
                    arm_names.push(piece);
                }
            }
        }
        // Sanity floor so the scan can't silently rot to zero. The binary-op
        // family dispatches through one BINARY_OPS-guarded arm (not string
        // patterns), so the floor is below KNOWN_OPS.len().
        assert!(
            arm_names.len() >= 60,
            "arm scan found only {} arms — the source scan has rotted, fix the test",
            arm_names.len()
        );
        // The guarded binary-op family must still be fully registered.
        for (name, _) in BINARY_OPS {
            assert!(
                KNOWN_OPS.contains(name),
                "BINARY_OPS entry '{name}' is missing from KNOWN_OPS"
            );
        }
        for name in &arm_names {
            assert!(
                KNOWN_OPS.contains(name),
                "resolve_op has an arm for '{name}' that is missing from KNOWN_OPS"
            );
        }
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
