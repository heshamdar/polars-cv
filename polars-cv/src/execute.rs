//! Pipeline execution engine.
//!
//! This module handles the execution of vision pipelines on Polars Series,
//! including parameter resolution and view-buffer integration.

use polars::prelude::*;

use view_buffer::{ImageAdapter, PlannedDType, ViewBuffer};

use crate::formats::sink::Sink;
use crate::formats::source::Source;
use crate::formats::Format as _;

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
/// whole source spec to overwrite the format string first (CR-37). `blob`/`raw`
/// sources never reach it: they decode zero-copy via
/// `graph::decode::decode_binary_zero_copy`.
pub fn decode_image_bytes(bytes: &[u8], source: &Source) -> PolarsResult<ViewBuffer> {
    // An explicit decode-scale assertion lets JPEG decode skip work via IDCT
    // scaling; other formats fall through to a full decode.
    let scaled = source
        .decode_max_size()
        .and_then(|max_size| decode_jpeg_scaled(bytes, max_size));
    let buf = match scaled {
        Some(buf) => buf,
        None => ImageAdapter::decode(bytes)
            .map_err(|e| polars_err!(ComputeError: "Failed to decode image: {:?}", e))?,
    };
    // If source spec declares an expected dtype, cast to it.
    // This is a no-op when the decoded dtype already matches.
    if let Some(target) = source.dtype() {
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
pub fn encode_sink(buffer: &ViewBuffer, sink: &Sink) -> PolarsResult<Vec<u8>> {
    if let Sink::Blob = sink {
        // VIEW protocol: self-describing, so no codec precondition applies.
        return Ok(buffer.to_blob());
    }

    let Some(codec) = sink.image_codec() else {
        return Err(polars_err!(ComputeError: "the '{}' sink is not a byte encoding", sink.name()));
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

    // Each codec's settings are its sink's fields.
    let encoded = match sink {
        Sink::Png => ImageAdapter::encode(buffer, image::ImageFormat::Png),
        Sink::Jpeg { quality } => ImageAdapter::encode_jpeg(buffer, quality.get()),
        Sink::WebP => ImageAdapter::encode(buffer, image::ImageFormat::WebP),
        Sink::Tiff => ImageAdapter::encode_tiff(buffer),
        Sink::Array { .. }
        | Sink::Blob
        | Sink::List
        | Sink::Native
        | Sink::NdArray { .. }
        | Sink::Numpy { .. }
        | Sink::Torch { .. } => unreachable!("only an image sink has a codec"),
    };
    encoded.map_err(|e| polars_err!(ComputeError: "Failed to encode {}: {:?}", codec.name(), e))
}
