//! Pipeline execution engine.
//!
//! This module handles the execution of vision pipelines on Polars Series,
//! including parameter resolution and view-buffer integration.

use polars::prelude::*;

use view_buffer::{
    geometry::rasterize::rasterize, DType, ImageAdapter, ImageCodec, PlannedDType, ViewBuffer,
};

use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use crate::pipeline::{OpSpec, SinkSpec, SourceSpec};
use view_buffer::naming;

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
    // shape_node sources never reach this function: the graph executor
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

/// Decode encoded image bytes (PNG/JPEG/TIFF/…) into a ViewBuffer, honouring
/// the source's decode-scale and dtype settings.
///
/// Reached for `image_bytes` sources, `file_path` sources once their bytes are
/// read, and `auto` sources that resolved to image bytes. The executor
/// dispatches on its `SourceFormat` before calling, so this takes no format.
/// It used to take one, which made `file_path` and `auto` rows clone their
/// whole `SourceSpec` to overwrite the format string first (CR-37). `blob`/`raw`
/// sources never reach it: they decode zero-copy via
/// `graph::decode::decode_binary_zero_copy`.
pub fn decode_image_bytes(bytes: &[u8], source: &SourceSpec) -> PolarsResult<ViewBuffer> {
    // An explicit decode-scale assertion lets JPEG decode skip work via IDCT
    // scaling; other formats fall through to a full decode.
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
    codec
        .check_shape(
            PlannedDType::Known(buffer.dtype()),
            Some(buffer.shape()),
            None,
        )
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

/// The operations resolved by name through the untyped legacy protocol.
///
/// Empty: every op is typed (`crate::ops::TypedOp`, typed-op plan P3). The
/// dispatcher in `pipeline.rs` still consults it, so a name in neither set is
/// an error; P6 deletes it with the rest of the legacy protocol.
pub const LEGACY_OPS: &[&str] = &[];

/// Resolve an operation specification to a [`GraphStep`].
///
/// Single-buffer ops become `GraphStep::Buffer(ViewDto)` (executed via the
/// engine's `ViewExpr`); multi-input and domain-changing ops become typed
/// graph-level steps. Node references and expression column names enter the
/// step here — they never reach the engine's `ViewDto`.
///
/// Serde has already rejected any field an op does not declare, and each op's
/// `OpDef` destructures every one it does.
pub fn resolve_op(op_spec: &OpSpec, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
    match op_spec {
        OpSpec::Typed(op) => op.resolve(row_idx, ctx),
        // Unreachable from the wire (`LEGACY_OPS` is empty), and refused
        // rather than guessed at if a spec is built by hand.
        OpSpec::Legacy(spec) => polars_bail!(ComputeError: "Unknown operation: {}", spec.op),
    }
}

/// Parse a dtype string to DType (canonical short names from `DType::NAMED`).
fn parse_dtype(s: &str) -> PolarsResult<DType> {
    DType::from_short_name(s).ok_or_else(|| {
        polars_err!(ComputeError:
            "Unknown dtype: {}, expected one of {:?}", s, naming::names(DType::NAMED))
    })
}
