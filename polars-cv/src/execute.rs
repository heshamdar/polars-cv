//! Pipeline execution engine.
//!
//! This module handles the execution of vision pipelines on Polars Series,
//! including parameter resolution and view-buffer integration.

use polars::prelude::*;

use view_buffer::{
    geometry::rasterize::rasterize, AffineParams, BinaryOp, ComputeOp, DType, FilterType,
    GeometryOp, ImageAdapter, ImageCodec, ImageOp, ImageOpKind, InterpolationType, NormalizeMethod,
    ViewBuffer, ViewDto, ViewOp,
};

use crate::graph::step::GraphStep;
use crate::params::{get, OpParams, ParamCtx, ParamValue};
use crate::pipeline::{OpSpec, SinkSpec, SourceSpec};
use view_buffer::geometry::label::{LabelReduction, LabelRegionMode};
use view_buffer::naming;

/// The name Python queries [`BINARY_OPS`] under via `enum_variants`.
///
/// This family is the one enum-shaped vocabulary view-buffer's registry cannot
/// hold, because the table below lives in this crate. Naming it here keeps the
/// string next to what it names rather than loose in the FFI.
pub(crate) const BINARY_OP_ENUM: &str = "BinaryOp";

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
/// `warp_affine`; defaults to bilinear). Resolved per row: the choice of
/// interpolation affects pixel values, never output shape or dtype.
fn resolve_interpolation(
    params: &OpParams<'_>,
    row_idx: usize,
    ctx: &ParamCtx,
) -> PolarsResult<InterpolationType> {
    get::opt_enum(
        params,
        "interpolation",
        InterpolationType::NAMED,
        &[],
        InterpolationType::Bilinear,
        row_idx,
        ctx,
    )
}

/// Parse the `filter` parameter shared by every resize variant.
///
/// Resolved per row — a resampling filter changes pixel values, never the
/// output geometry, which the resize dimensions alone determine.
///
/// Required, not defaulted: every builder path emits `filter`, so an absent one
/// means the spec was built wrong and must error rather than silently resample
/// with some other filter. The `Lanczos3` argument is only consumed under a
/// plan-time probe context, where no concrete value exists yet and the choice
/// cannot affect the inferred schema.
fn resolve_filter(
    params: &OpParams<'_>,
    row_idx: usize,
    ctx: &ParamCtx,
) -> PolarsResult<FilterType> {
    get::req_enum(
        params,
        "filter",
        FilterType::NAMED,
        FilterType::ALIASES,
        FilterType::Lanczos3,
        row_idx,
        ctx,
    )
}

/// Parse the optional `border_value` parameter (shared by `rotate` and
/// `warp_affine`; defaults to 0.0).
fn resolve_border_value(
    params: &OpParams<'_>,
    row_idx: usize,
    ctx: &ParamCtx,
) -> PolarsResult<f64> {
    get::opt_f64(params, "border_value", 0.0, row_idx, ctx)
}

/// Parse rasterize's optional style parameters `(fill_value, background)` —
/// shared with the graph executor's rasterize-by-shape-ref path so the two
/// sites cannot diverge.
pub(crate) fn resolve_rasterize_style(
    params: &OpParams<'_>,
    row_idx: usize,
    ctx: &ParamCtx,
) -> PolarsResult<(u8, u8)> {
    Ok((
        get::opt_u8(params, "fill_value", 255, row_idx, ctx)?,
        get::opt_u8(params, "background", 0, row_idx, ctx)?,
    ))
}

/// Decode a contour source by parsing the geometry and rasterizing to ViewBuffer.
///
/// The column may hold one contour per row or a whole set (`List[Contour]`) —
/// `parse_contour_set` accepts both, and the set is painted as a union, exactly
/// as the `rasterize` op paints the set `extract_contours` produces.
pub fn decode_contour_source(
    value: &AnyValue,
    row_idx: usize,
    source: &SourceSpec,
    ctx: &ParamCtx,
) -> PolarsResult<ViewBuffer> {
    // Parse via the plugin's single contour parser (contour.rs).
    let contours = crate::contour::parse_contour_set(value)?;

    // Resolve dimensions
    let (width, height) = resolve_contour_dimensions(row_idx, source, ctx)?;

    // Get fill and background values (both per-row capable)
    let (fill_value, background) = source.resolve_fill(row_idx, ctx)?;

    // Rasterize the contours to a ViewBuffer
    Ok(rasterize(&contours, width, height, fill_value, background))
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
    let contours = crate::contour::parse_contour_set(value)?;

    // Rasterize the contours to a ViewBuffer
    Ok(rasterize(&contours, width, height, fill_value, background))
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
    if sink.format.as_str() == "blob" {
        // VIEW protocol: self-describing, so no codec precondition applies.
        return Ok(buffer.to_blob());
    }

    let Some(codec) = ImageCodec::from_sink_format(sink.format.as_str()) else {
        return Err(polars_err!(ComputeError: "Unknown sink format: {}", sink.format));
    };

    // The same check the planner ran before publishing this query's schema
    // (`dtype_for_output`). Reaching a failure here means the planner had less
    // information than we do now — a source whose dtype was still "auto", or a
    // shape only the data could settle — not that the two disagree.
    let shape = buffer.shape();
    let channels = match shape.len() {
        3 => Some(shape[2]),
        2 => Some(1),
        _ => None,
    };
    codec
        .check_support(Some(buffer.dtype()), Some(shape.len()), channels)
        .map_err(|msg| polars_err!(ComputeError: "{}", msg))?;

    match codec {
        ImageCodec::Png => ImageAdapter::encode(buffer, image::ImageFormat::Png)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode PNG: {:?}", e)),
        ImageCodec::Jpeg => ImageAdapter::encode_jpeg(buffer, sink.quality)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode JPEG: {:?}", e)),
        ImageCodec::WebP => ImageAdapter::encode(buffer, image::ImageFormat::WebP)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode WebP: {:?}", e)),
        ImageCodec::Tiff => ImageAdapter::encode_tiff(buffer)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode TIFF: {:?}", e)),
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
///
/// Every parameter on the spec must be read by the arm that handles it. See
/// [`resolve_op_inner`] for why, and [`OpParams`] for how.
pub fn resolve_op(op_spec: &OpSpec, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
    let params = OpParams::new(&op_spec.params);
    let step = resolve_op_inner(&op_spec.op, &params, row_idx, ctx)?;

    // A parameter the arm never looked at is a parameter that did nothing. It
    // still rode in the op's identity — so two pipelines that behave
    // identically hash differently for CSE and compile to separate graph-cache
    // entries — and it read, to whoever passed it, as a request that was
    // honoured. `scale`/`clamp` carried an `out_dtype` like that.
    //
    // This runs at both ends of the boundary with no extra wiring:
    // `resolve_op_from_json` backs the `op_schema` FFI, so a stray parameter
    // fails in Python while the `Pipeline` is being built, and
    // `CompiledGraph::compile` resolves every spec before any row executes.
    let unread = params.unread()?;
    if !unread.is_empty() {
        let mut names: Vec<&str> = unread;
        names.sort_unstable();
        polars_bail!(ComputeError:
            "operation '{}': parameter(s) {:?} are not read by this operation. \
             A parameter that reaches no code path is not a no-op — it enters \
             the op's identity and silently discards what the caller asked for.",
            op_spec.op, names);
    }
    Ok(step)
}

/// The per-operation dispatch behind [`resolve_op`].
///
/// Split out so the parameter-use check has exactly one place to run and the
/// arms have no way to return past it.
///
/// It takes the op *name* and an [`OpParams`], never the `OpSpec` — an arm that
/// could still reach `op_spec.params` could read a parameter without recording
/// it, which is not a tracker so much as a suggestion. Two of `crop`'s reads
/// were exactly that shape before the signature was narrowed, and the compiler
/// is what found them.
fn resolve_op_inner(
    op_name: &str,
    params: &OpParams<'_>,
    row_idx: usize,
    ctx: &ParamCtx,
) -> PolarsResult<GraphStep> {
    match op_name {
        // View operations
        "transpose" => {
            let axes = get_param(params, "axes")?.as_int_list()?;
            buffer_step(ViewDto::View(ViewOp::Transpose(axes)))
        }
        "reshape" => {
            // Borrow the compiled `List` on the hot path; parse once otherwise.
            let shape_param = get_param(params, "shape")?;
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
            let axes = get_param(params, "axes")?.as_int_list()?;
            buffer_step(ViewDto::View(ViewOp::Flip(axes)))
        }
        "crop" => {
            // Allow negative values for top/left and clamp to 0
            // This makes the API more forgiving and follows NumPy/OpenCV conventions
            let top_raw = get_param(params, "top")?.resolve_i64(row_idx, ctx)?;
            let left_raw = get_param(params, "left")?.resolve_i64(row_idx, ctx)?;

            // Clamp negative values to 0
            let top = top_raw.max(0) as usize;
            let left = left_raw.max(0) as usize;

            // Height and width might be optional - these should still be non-negative
            let height = params
                .get("height")
                .map(|p| {
                    let h = p.resolve_i64(row_idx, ctx)?;
                    // Clamp negative height to 0 (will result in empty crop)
                    Ok::<usize, PolarsError>(h.max(0) as usize)
                })
                .transpose()?;
            let width = params
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
            let dtype_str = get_param(params, "dtype")?.resolve_string()?;
            let dtype = parse_dtype(dtype_str)?;
            buffer_step(ViewDto::Compute(ComputeOp::Cast(dtype)))
        }
        "scale" => {
            let factor = get_param(params, "factor")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Compute(ComputeOp::Scale(factor)))
        }
        "normalize" => {
            let method_str = get_param(params, "method")?.resolve_string()?;
            let method = match method_str {
                "minmax" => NormalizeMethod::MinMax,
                "zscore" => NormalizeMethod::ZScore,
                "preset" => {
                    // Per-element ParamValues, so a per-row mean/std (dataset
                    // statistics joined in as columns) resolves per row. The
                    // element count is the channel count and stays structural.
                    let mean = get_param(params, "mean")?.resolve_f32_list(row_idx, ctx)?;
                    let std = get_param(params, "std")?.resolve_f32_list(row_idx, ctx)?;

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
            let out_dtype = match params.get("out_dtype") {
                Some(p) => parse_dtype(p.resolve_string()?)?,
                None => DType::F32,
            };
            buffer_step(ViewDto::Compute(ComputeOp::Normalize(method, out_dtype)))
        }
        "clamp" => {
            let min = get_param(params, "min")?.resolve_f32(row_idx, ctx)?;
            let max = get_param(params, "max")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Compute(ComputeOp::Clamp { min, max }))
        }
        "relu" => buffer_step(ViewDto::Compute(ComputeOp::Relu)),

        // Image operations
        "resize" => {
            let height = get_param(params, "height")?.resolve_u32(row_idx, ctx)?;
            let width = get_param(params, "width")?.resolve_u32(row_idx, ctx)?;
            let filter = resolve_filter(params, row_idx, ctx)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Resize {
                    width,
                    height,
                    filter,
                },
            }))
        }
        "resize_scale" => {
            let scale_x = get_param(params, "scale_x")?.resolve_f32(row_idx, ctx)?;
            let scale_y = get_param(params, "scale_y")?.resolve_f32(row_idx, ctx)?;
            let filter = resolve_filter(params, row_idx, ctx)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeScale {
                    scale_x,
                    scale_y,
                    filter,
                },
            }))
        }
        "resize_to_height" => {
            let height = get_param(params, "height")?.resolve_u32(row_idx, ctx)?;
            let filter = resolve_filter(params, row_idx, ctx)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeToHeight { height, filter },
            }))
        }
        "resize_to_width" => {
            let width = get_param(params, "width")?.resolve_u32(row_idx, ctx)?;
            let filter = resolve_filter(params, row_idx, ctx)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeToWidth { width, filter },
            }))
        }
        "resize_max" => {
            let max_size = get_param(params, "max_size")?.resolve_u32(row_idx, ctx)?;
            let filter = resolve_filter(params, row_idx, ctx)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeMax { max_size, filter },
            }))
        }
        "resize_min" => {
            let min_size = get_param(params, "min_size")?.resolve_u32(row_idx, ctx)?;
            let filter = resolve_filter(params, row_idx, ctx)?;

            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ResizeMin { min_size, filter },
            }))
        }

        // Padding operations
        "pad" => {
            use view_buffer::ops::dto::PadMode;

            let top = get_param(params, "top")?.resolve_u32(row_idx, ctx)?;
            let bottom = get_param(params, "bottom")?.resolve_u32(row_idx, ctx)?;
            let left = get_param(params, "left")?.resolve_u32(row_idx, ctx)?;
            let right = get_param(params, "right")?.resolve_u32(row_idx, ctx)?;
            let value = get_param(params, "value")?.resolve_f32(row_idx, ctx)?;
            let mode = get::req_enum(
                params,
                "mode",
                PadMode::NAMED,
                &[],
                PadMode::Constant,
                row_idx,
                ctx,
            )?;

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

            let height = get_param(params, "height")?.resolve_u32(row_idx, ctx)?;
            let width = get_param(params, "width")?.resolve_u32(row_idx, ctx)?;
            let value = get_param(params, "value")?.resolve_f32(row_idx, ctx)?;
            let position = get::req_enum(
                params,
                "position",
                PadPosition::NAMED,
                &[],
                PadPosition::Center,
                row_idx,
                ctx,
            )?;

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
            let height = get_param(params, "height")?.resolve_u32(row_idx, ctx)?;
            let width = get_param(params, "width")?.resolve_u32(row_idx, ctx)?;
            let value = get_param(params, "value")?.resolve_f32(row_idx, ctx)?;

            // Letterbox has always resized with lanczos3, so that stays the
            // default; the builder now exposes it, per row like every other
            // resize variant's filter.
            let filter = resolve_filter(params, row_idx, ctx)?;
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
            let value = get_param(params, "value")?.resolve_f64(row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Threshold(value),
            }))
        }
        "blur" => {
            let sigma = get_param(params, "sigma")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Blur { sigma },
            }))
        }
        "rotate" => {
            let angle = get_param(params, "angle")?.resolve_f32(row_idx, ctx)?;
            let expand = get::opt_bool(params, "expand", false)?;

            // The lattice rotations (90/180/270) and the 0° no-op below are
            // exact permutations of the input pixels: nothing is resampled and
            // no out-of-bounds region is exposed, so `interpolation` and
            // `border_value` have no effect on those branches — they are
            // inapplicable there, not discarded. Declaring them here rather
            // than in whichever branch happens to read them also means a new
            // branch cannot silently drop them.
            params.acknowledge("interpolation");
            params.acknowledge("border_value");

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
                let interpolation = resolve_interpolation(params, row_idx, ctx)?;
                let border_value = resolve_border_value(params, row_idx, ctx)?;

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
            let matrix_param = get_param(params, "matrix")?;
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

            let output_height = get_param(params, "output_height")?.resolve_u32(row_idx, ctx)?;
            let output_width = get_param(params, "output_width")?.resolve_u32(row_idx, ctx)?;

            let interpolation = resolve_interpolation(params, row_idx, ctx)?;
            let border_value = resolve_border_value(params, row_idx, ctx)?;

            let affine = AffineParams {
                matrix,
                output_height,
                output_width,
                interpolation,
                border_value,
            };
            // Warping is inverse mapping — for each output pixel, ask where it
            // came from — so a matrix that collapses the plane onto a line or a
            // point has no answer. The runner used to substitute the identity
            // for one, handing back the input as though the transform had been
            // applied. Reject it here, where the user supplied it, and name the
            // determinant so the offending coefficients are findable.
            //
            // Not under a plan-time probe: every expression parameter is bound
            // to the *same* placeholder there, so a per-row matrix arrives as
            // six equal coefficients and is singular by construction. Its real
            // values only exist per row, which is where this same arm runs for
            // a dynamic op — so the check still covers them, just later.
            if !ctx.is_probe() && !affine.is_invertible() {
                return Err(polars_err!(ComputeError:
                    "warp_affine: matrix {:?} is singular (determinant {}), so it \
                     has no inverse and the warp is undefined. A row of zeros, a \
                     zero scale factor on an axis, or two proportional rows will \
                     do this.",
                    affine.matrix, affine.determinant()));
            }
            buffer_step(ViewDto::Compute(ComputeOp::Affine(affine)))
        }

        // Perceptual hash operation — a graph-level vector producer (image
        // buffer → 1-D u8 fingerprint), executed via `apply_perceptual_hash`.
        "perceptual_hash" => {
            use view_buffer::ops::phash::{HashAlgorithm, PerceptualHashOp};

            // Paired with the structural `hash_size` below; kept literal so
            // the fingerprint's identity is fixed at planning time.
            let algorithm = get::opt_enum_literal(
                params,
                "algorithm",
                HashAlgorithm::NAMED,
                &[],
                HashAlgorithm::Perceptual,
            )?;
            // `hash_size` fixes the output vector length, so it is a structural
            // (literal-only) param — reject a bound expression slot.
            let hash_size = get::opt_u32_literal(params, "hash_size", 64)?;

            Ok(GraphStep::PerceptualHash(
                PerceptualHashOp::new(algorithm).with_hash_size(hash_size),
            ))
        }

        // Geometry operations
        "rasterize" => {
            // `rasterize(shape=<node>)` names another graph node to take the
            // mask dimensions from. `CompiledGraph::compile` resolves it before
            // this arm is reached and substitutes the concrete width/height, so
            // the parameter belongs to the op but is consumed a layer up.
            params.acknowledge("shape_ref");
            let width = get_param(params, "width")?.resolve_usize(row_idx, ctx)? as u32;
            let height = get_param(params, "height")?.resolve_usize(row_idx, ctx)? as u32;
            let (fill_value, background) = resolve_rasterize_style(params, row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Rasterize {
                width,
                height,
                fill_value,
                background,
            }))
        }
        "extract_contours" => {
            use view_buffer::geometry::ops::{ApproxMethod, ExtractMode};

            let mode = get::opt_enum(
                params,
                "mode",
                ExtractMode::NAMED,
                &[],
                ExtractMode::External,
                row_idx,
                ctx,
            )?;
            let method = get::opt_enum(
                params,
                "method",
                ApproxMethod::NAMED,
                &[],
                ApproxMethod::Simple,
                row_idx,
                ctx,
            )?;
            let min_area = get::maybe_f64(params, "min_area", row_idx, ctx)?;

            Ok(GraphStep::Geometry(GeometryOp::ExtractContours {
                mode,
                method,
                min_area,
            }))
        }

        // Geometry measure operations
        "contour_area" => {
            let signed = get::opt_bool_dyn(params, "signed", false, row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Area { signed }))
        }
        "contour_perimeter" => Ok(GraphStep::Geometry(GeometryOp::Perimeter)),
        "contour_centroid" => Ok(GraphStep::Geometry(GeometryOp::Centroid)),
        "contour_bounding_box" => Ok(GraphStep::Geometry(GeometryOp::BoundingBox)),
        "contour_convex_hull" => Ok(GraphStep::Geometry(GeometryOp::ConvexHull)),

        // Geometry transforms
        "contour_translate" => {
            let dx = get_param(params, "dx")?.resolve_f64(row_idx, ctx)?;
            let dy = get_param(params, "dy")?.resolve_f64(row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Translate { dx, dy }))
        }
        "contour_scale" => {
            let sx = get_param(params, "sx")?.resolve_f64(row_idx, ctx)?;
            let sy = get_param(params, "sy")?.resolve_f64(row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Scale {
                sx,
                sy,
                origin: view_buffer::geometry::ops::ScaleOrigin::Centroid,
            }))
        }
        "contour_simplify" => {
            let tolerance = get_param(params, "tolerance")?.resolve_f64(row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Simplify { tolerance }))
        }

        // Binary operations (two-buffer): one arm for the whole family,
        // dispatched through the BINARY_OPS name table.
        name if naming::lookup(BINARY_OPS, name).is_some() => {
            let op = naming::lookup(BINARY_OPS, name).expect("guard checked membership");
            let other_node_id = get_param(params, "other_node")?
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
            let axis = get::maybe_usize_literal(params, "axis")?;
            Ok(GraphStep::Reduction(ReductionOp::Max { axis }))
        }
        "reduce_min" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(params, "axis")?;
            Ok(GraphStep::Reduction(ReductionOp::Min { axis }))
        }
        "reduce_mean" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(params, "axis")?;
            Ok(GraphStep::Reduction(ReductionOp::Mean { axis }))
        }
        "reduce_std" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(params, "axis")?;
            let ddof = get::opt_u8(params, "ddof", 0, row_idx, ctx)?;
            Ok(GraphStep::Reduction(ReductionOp::Std { axis, ddof }))
        }
        "reduce_percentile" => {
            use view_buffer::ops::ReductionOp;
            let q = get_param(params, "q")?.resolve_f64(row_idx, ctx)?;
            Ok(GraphStep::Reduction(ReductionOp::Percentile { q }))
        }
        "reduce_argmax" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(params, "axis")?
                .ok_or_else(|| polars_err!(ComputeError: "Missing required parameter: axis"))?;
            Ok(GraphStep::Reduction(ReductionOp::ArgMax { axis }))
        }
        "reduce_argmin" => {
            use view_buffer::ops::ReductionOp;
            let axis = get::maybe_usize_literal(params, "axis")?
                .ok_or_else(|| polars_err!(ComputeError: "Missing required parameter: axis"))?;
            Ok(GraphStep::Reduction(ReductionOp::ArgMin { axis }))
        }
        "extract_shape" => {
            // Extract shape returns buffer dimensions as a vector
            Ok(GraphStep::ExtractShape)
        }
        "label_reduce" => {
            let contours_param = get_param(params, "contours")?;
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
                params,
                "reduction",
                LabelReduction::NAMED,
                &[],
                LabelReduction::Max,
                row_idx,
                ctx,
            )?;
            let region_mode = get::opt_enum(
                params,
                "region_mode",
                LabelRegionMode::NAMED,
                &[],
                LabelRegionMode::Interior,
                row_idx,
                ctx,
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

            let bins_param = get_param(params, "bins")?;
            let (bins_count, edges) = if let Some(edges) = bins_param.as_f64_vec() {
                // If it's a vector, those are the edges
                (edges.len().saturating_sub(1), Some(edges))
            } else {
                (bins_param.resolve_usize(row_idx, ctx)?, None)
            };

            // `closed` and `output` shape the histogram's dtype/semantics at plan
            // time, so both stay literal-only.
            let closed = get::req_enum_literal(params, "closed", HistogramClosed::NAMED, &[])?;
            let output = get::req_enum_literal(params, "output", HistogramOutput::NAMED, &[])?;

            // Parse optional range
            let range = if params.contains_key("range_min") && params.contains_key("range_max") {
                let range_min = get_param(params, "range_min")?.resolve_f64(row_idx, ctx)?;
                let range_max = get_param(params, "range_max")?.resolve_f64(row_idx, ctx)?;
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
            let index = get_param(params, "index")?.resolve_usize(row_idx, ctx)?;
            buffer_step(ViewDto::View(ViewOp::ChannelSelect { index }))
        }
        "channel_swap" => {
            // A permutation: the element count is structural (channel count is
            // preserved) but the indices themselves may be per-row.
            let order = get_param(params, "order")?.resolve_usize_list(row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::ChannelSwap { order },
            }))
        }
        "channel_merge" => {
            let other_nodes_param = get_param(params, "other_nodes")?;
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
            let factor = get_param(params, "factor")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Compute(ComputeOp::AdjustContrast(factor)))
        }
        "adjust_gamma" => {
            let gamma = get_param(params, "gamma")?.resolve_f32(row_idx, ctx)?;
            buffer_step(ViewDto::Compute(ComputeOp::AdjustGamma(gamma)))
        }
        "invert" => buffer_step(ViewDto::Compute(ComputeOp::Invert)),

        // Color space conversion
        "cvt_color" => {
            use view_buffer::ops::color::{ColorConvertOp, ColorSpace};

            let from_str = get_param(params, "from_space")?.resolve_string()?;
            let to_str = get_param(params, "to_space")?.resolve_string()?;
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

            // Each coefficient is its own ParamValue, so a kernel whose values
            // derive from a column (an unsharp mask with a per-row strength)
            // resolves per row. The kernel *length* stays structural.
            let kernel = get_param(params, "kernel")?.resolve_f32_list(row_idx, ctx)?;
            let ksize = get_param(params, "ksize")?.resolve_usize(row_idx, ctx)?;
            let normalize = get::opt_bool_dyn(params, "normalize", false, row_idx, ctx)?;
            let border = get::opt_enum(
                params,
                "border",
                BorderMode::NAMED,
                &[],
                BorderMode::Replicate,
                row_idx,
                ctx,
            )?;

            buffer_step(ViewDto::Filter(ConvolveOp {
                kernel,
                ksize,
                normalize,
                border,
            }))
        }
        "erode" => {
            let ksize = get_param(params, "ksize")?.resolve_u32(row_idx, ctx)?;
            let iterations = get::opt_u32(params, "iterations", 1, row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Erode { ksize, iterations },
            }))
        }
        "dilate" => {
            let ksize = get_param(params, "ksize")?.resolve_u32(row_idx, ctx)?;
            let iterations = get::opt_u32(params, "iterations", 1, row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::Dilate { ksize, iterations },
            }))
        }
        "morphology_gradient" => {
            let ksize = get_param(params, "ksize")?.resolve_u32(row_idx, ctx)?;
            buffer_step(ViewDto::Image(ImageOp {
                kind: ImageOpKind::MorphGradient { ksize },
            }))
        }

        "canny" => {
            let low_threshold = get_param(params, "low_threshold")?.resolve_f32(row_idx, ctx)?;
            let high_threshold = get_param(params, "high_threshold")?.resolve_f32(row_idx, ctx)?;
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
            let mask_node_id = get_param(params, "other_node")?
                .resolve_string()?
                .to_string();
            let invert = get::opt_bool_dyn(params, "invert", false, row_idx, ctx)?;
            Ok(GraphStep::ApplyMask {
                mask: mask_node_id,
                invert,
            })
        }

        other => Err(polars_err!(ComputeError: "Unknown operation: {}", other)),
    }
}

/// Get a required parameter, recording the read.
fn get_param<'a>(params: &OpParams<'a>, name: &str) -> PolarsResult<&'a ParamValue> {
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

#[cfg(test)]
mod strict_param_tests {
    //! One failure policy for operation parameters: an *absent* optional
    //! parameter takes its documented default, but a parameter that is
    //! *present and invalid* (unknown enum string, wrong type, out of range)
    //! must be an error — never silently coerced to a default. These tests
    //! pin that policy for every parameter that historically swallowed
    //! errors.
    //!
    //! A **null** per-row value is a separate axis and is deliberately not
    //! covered here: it is neither absent nor invalid, and what it means is
    //! chosen by `NullParamPolicy` (`params.rs`) — raise, or null the affected
    //! rows. It is never coerced to a default under either policy, so the rule
    //! above still holds; see `params::tests::test_null_policy_*`.

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

    /// Build the per-element encoding a list-valued param now uses: an array of
    /// serialized `ParamValue`s rather than raw numbers, so any element can be a
    /// per-row expression (see `ParamValue::resolve_f32_list`).
    fn param_list(values: &[f64]) -> serde_json::Value {
        json!(values
            .iter()
            .map(|v| json!({"type": "literal", "value": v}))
            .collect::<Vec<_>>())
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
            ("contour_area", "signed", &[]),
            (
                "convolve2d",
                "normalize",
                &[("kernel", param_list(&[0.0; 9])), ("ksize", json!(3))],
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

    /// Reverse guard: every top-level match arm in `resolve_op_inner` must be
    /// listed in KNOWN_OPS, so a new arm cannot silently bypass the registry
    /// (the forward direction is covered by `known_ops_all_resolve`).
    ///
    /// The scan reads this file's source between the `resolve_op_inner` header
    /// and its "Unknown operation" catch-all. That is where the arms live —
    /// `resolve_op` is the wrapper that runs the parameter-use check around
    /// them, and anchoring here rather than on it keeps the scanned region to
    /// the match itself. Top-level arm patterns sit at one match-nesting level
    /// (8-space indent under rustfmt, which CI enforces); deeper string arms
    /// (e.g. normalize's method match) are excluded by the indent check.
    #[test]
    fn resolve_op_arms_are_all_known_ops() {
        let src = include_str!("execute.rs");
        let start = src
            .find("fn resolve_op_inner")
            .expect("resolve_op_inner not found");
        let end = start
            + src[start..]
                .find("Unknown operation")
                .expect("resolve_op catch-all not found");
        let mut arm_names: Vec<&str> = Vec::new();
        let mut guard_arms: Vec<&str> = Vec::new();
        for line in src[start..end].lines() {
            let trimmed = line.trim_start();
            let indent = line.len() - trimmed.len();
            if indent != 8 {
                continue;
            }
            // Continuation lines of a wrapped arm, and the closing brace of a
            // block-bodied one, carry no pattern. Everything else at this
            // indent starts an arm and must be classified.
            if trimmed.is_empty()
                || trimmed.starts_with("//")
                || trimmed.starts_with('}')
                || trimmed.starts_with("=>")
                || trimmed.starts_with("&&")
                || trimmed.starts_with("||")
                || trimmed.starts_with('|')
            {
                continue;
            }
            if trimmed.starts_with('"') {
                // Fall through to the string-literal handling below.
            } else {
                // Anything else registers ops without naming them in a form
                // this scan can read: a guard arm, an `@` binding, a bare
                // binder. Record the whole pattern and require it to be
                // explicitly known below.
                //
                // The previous version only recognised a guard arm when the
                // pattern and its `=>` shared a line, and silently skipped
                // every other shape. rustfmt moves `=>` to the next line once
                // the condition is long enough, and an `@` binding never had
                // one -- both let a whole op family become executable with no
                // KNOWN_OPS entry. "Anything I do not recognise is ignored"
                // was the bug; "anything I do not recognise fails" is the
                // guard.
                let pattern = trimmed
                    .split("=>")
                    .next()
                    .unwrap_or(trimmed)
                    .trim()
                    .trim_end_matches('{')
                    .trim();
                guard_arms.push(pattern);
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
        // Every op in KNOWN_OPS is either a string arm found above or covered
        // by one of the known guard arms below, so the scan cannot rot to a
        // subset without this failing. A count floor was used here before; it
        // was both too weak (10 arms could drop out of indent 8 unnoticed) and
        // too brittle (deprecating an op tripped it), so the relationship is
        // pinned instead of a magic number.
        let guarded: Vec<&str> = BINARY_OPS.iter().map(|(n, _)| *n).collect();
        let unaccounted: Vec<&&str> = KNOWN_OPS
            .iter()
            .filter(|n| !arm_names.contains(n) && !guarded.contains(n))
            .collect();
        assert!(
            unaccounted.is_empty(),
            "these KNOWN_OPS have no string arm and are not in a known guarded \
             family: {unaccounted:?} — either resolve_op changed shape or the \
             source scan has rotted"
        );
        // Guard arms register a whole family at once. Each one needs a rule
        // above tying its table to KNOWN_OPS; a new one has none, so fail
        // until it is given one rather than let it register ops invisibly.
        const KNOWN_GUARD_ARMS: &[&str] = &[
            // Registers the whole binary-op family; its table is checked
            // against KNOWN_OPS below.
            "name if naming::lookup(BINARY_OPS, name).is_some()",
            // The catch-all that produces the "Unknown operation" error this
            // scan terminates on. Registers nothing.
            "other",
        ];
        for arm in &guard_arms {
            assert!(
                KNOWN_GUARD_ARMS.contains(arm),
                "resolve_op has an unrecognised guard arm '{arm}'. Guard arms \
                 register ops without naming them, so add it to \
                 KNOWN_GUARD_ARMS here along with a check that its table is \
                 fully listed in KNOWN_OPS (see BINARY_OPS below)."
            );
        }
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

#[cfg(test)]
mod unread_param_tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    /// Build an `OpSpec` from a name and literal params.
    fn op_with(name: &str, params: &[(&str, serde_json::Value)]) -> OpSpec {
        OpSpec {
            op: name.to_string(),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), ParamValue::Literal { value: v.clone() }))
                .collect::<HashMap<_, _>>(),
        }
    }

    /// `(op name, base params, the parameter under test)`.
    type UnreadCase<'a> = (&'a str, &'a [(&'a str, serde_json::Value)], &'a str);

    /// `(op name, params that must all be consumed)`.
    type AcceptedCase<'a> = (&'a str, &'a [(&'a str, serde_json::Value)]);

    fn resolve_err(spec: &OpSpec) -> String {
        resolve_op(spec, 0, &ParamCtx::empty())
            .expect_err("expected resolve_op to reject the spec")
            .to_string()
    }

    /// The known-bad half: a parameter no arm reads must be rejected, by name.
    ///
    /// `scale`/`clamp` are the two ops that actually shipped this — they
    /// accepted an `out_dtype` that entered the op's identity and reached no
    /// code path. The fabricated cases alongside them keep the check honest for
    /// ops that never had the bug.
    #[test]
    fn a_parameter_no_arm_reads_is_rejected() {
        let cases: &[UnreadCase<'_>] = &[
            ("scale", &[("factor", json!(2.0))], "out_dtype"),
            (
                "clamp",
                &[("min", json!(0.0)), ("max", json!(1.0))],
                "out_dtype",
            ),
            ("grayscale", &[], "sigma"),
            (
                "resize",
                &[
                    ("height", json!(4)),
                    ("width", json!(4)),
                    ("filter", json!("nearest")),
                ],
                "antialias",
            ),
        ];
        for (op, base, stray) in cases {
            let mut params = base.to_vec();
            params.push((stray, json!("u8")));
            let err = resolve_err(&op_with(op, &params));
            assert!(
                err.contains(stray) && err.contains(op),
                "{op}: error must name the operation and the unread parameter \
                 '{stray}', got: {err}"
            );
        }
    }

    /// A singular matrix must be refused where the user supplies it.
    ///
    /// The runner substituted the identity for one, so a caller who asked for a
    /// degenerate transform got their input back and no signal that nothing had
    /// happened. `AffineParams::is_invertible` is the single authority; this
    /// pins the boundary that consults it.
    #[test]
    fn warp_affine_rejects_a_singular_matrix() {
        let elements = |m: [f64; 6]| {
            serde_json::Value::Array(
                m.iter()
                    .map(|v| json!({"type": "literal", "value": v}))
                    .collect(),
            )
        };
        let spec = |m: [f64; 6]| {
            op_with(
                "warp_affine",
                &[
                    ("matrix", elements(m)),
                    ("output_height", json!(8)),
                    ("output_width", json!(8)),
                ],
            )
        };

        let err = resolve_err(&spec([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
        assert!(
            err.contains("singular") && err.contains("determinant"),
            "the error must say what is wrong and name the determinant: {err}"
        );

        // An extreme but invertible transform is still accepted — the check is
        // for degeneracy, not for poor conditioning.
        assert!(
            resolve_op(
                &spec([1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0]),
                0,
                &ParamCtx::empty()
            )
            .is_ok(),
            "a heavily stretched but invertible matrix must resolve"
        );
    }

    /// The known-good half: a checker that rejects everything proves nothing.
    ///
    /// These specs carry parameters read through *helpers* rather than a
    /// literal `get_param` call in the arm (`resolve_filter`,
    /// `resolve_border_value`, `resolve_rasterize_style`) plus `rasterize`'s
    /// `shape_ref`, which a layer above the arm consumes. All must resolve.
    #[test]
    fn parameters_read_through_helpers_are_accepted() {
        let cases: &[AcceptedCase<'_>] = &[
            (
                "resize",
                &[
                    ("height", json!(4)),
                    ("width", json!(4)),
                    ("filter", json!("nearest")),
                ],
            ),
            (
                "rotate",
                &[
                    ("angle", json!(45.0)),
                    ("expand", json!(false)),
                    ("interpolation", json!("nearest")),
                    ("border_value", json!(7.0)),
                ],
            ),
            (
                "rasterize",
                &[
                    ("width", json!(8)),
                    ("height", json!(8)),
                    ("fill_value", json!(255)),
                    ("background", json!(0)),
                    ("shape_ref", json!("other_node")),
                ],
            ),
            ("scale", &[("factor", json!(2.0))]),
            ("clamp", &[("min", json!(0.0)), ("max", json!(1.0))]),
        ];
        for (op, params) in cases {
            let spec = op_with(op, params);
            assert!(
                resolve_op(&spec, 0, &ParamCtx::empty()).is_ok(),
                "{op}: every parameter here is consumed, so it must resolve; \
                 got: {}",
                resolve_err(&spec)
            );
        }
    }
}
