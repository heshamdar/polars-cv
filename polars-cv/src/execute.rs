//! Pipeline execution engine.
//!
//! This module handles the execution of vision pipelines on Polars Series,
//! including parameter resolution and view-buffer integration.

use polars::prelude::*;

use view_buffer::interop::image::{RegionDecode, TiffRegion};
use view_buffer::interop::tiff_region::{self, TiffImage, TiffSource};
use view_buffer::ops::validation::ValidationError;
use view_buffer::{ImageAdapter, PlannedDType, ViewBuffer, ViewOp};

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

/// A node's leading crop as one row reaches the decoder (the `roi_decode`
/// pass): none, its window, or a window whose parameters did not resolve for
/// this row (a null or refused value), which the executor raises or nulls
/// as the crop would.
#[derive(Clone, Copy)]
pub enum RowCrop<'a> {
    None,
    Window(&'a ViewOp),
    Unresolved,
}

/// What decoding one row's image gave.
pub enum ImageDecode {
    /// The image (or the source's level), whole.
    Whole(ViewBuffer),
    /// The leading crop's output: exactly the crop of the whole decode.
    Window(ViewBuffer),
    /// The leading crop refuses the image's shape: the crop's own error.
    Refused(ValidationError),
    /// The leading crop's parameters did not resolve (`RowCrop::Unresolved`),
    /// and the image is a TIFF whose header reads: its pixels were left
    /// unread, since the row can only fail or null on its parameter.
    Unread,
}

impl ImageDecode {
    /// A [`RegionDecode`] cast to the dtype the source declares.
    fn region(decoded: RegionDecode, source: &Source) -> Self {
        match decoded {
            RegionDecode::Whole(buf) => ImageDecode::Whole(with_declared_dtype(buf, source)),
            RegionDecode::Window(buf) => ImageDecode::Window(with_declared_dtype(buf, source)),
            RegionDecode::Refused(e) => ImageDecode::Refused(e),
        }
    }
}

fn decode_failed(e: image::ImageError) -> PolarsError {
    polars_err!(ComputeError: "Failed to decode image: {:?}", e)
}

/// Decode encoded image bytes (PNG/JPEG/TIFF/…) into a ViewBuffer, honouring
/// the source's decode-scale and dtype settings.
///
/// Reached for `image_bytes` sources, `file_path` sources once their bytes are
/// read, and `auto` sources routed to image bytes. The executor dispatches on
/// the (routed) `Source` before calling and passes it for its `dtype` and
/// `decode_max_size`. `blob`/`raw`
/// sources never reach it: they decode zero-copy via
/// `graph::decode::decode_binary_row`.
///
/// With a `crop` window (the node's leading crop, under the `roi_decode`
/// pass) only the window is decoded; a window the crop refuses is its error.
/// A declared dtype then casts the window: a cast is per element, so this is
/// the cast of the full decode, cropped.
///
/// `level` is the row's pyramid level (`source(level=)`, resolved by the
/// executor): 0 is the image; above 0 only a pyramidal TIFF has one.
pub fn decode_image_bytes(
    bytes: &[u8],
    source: &Source,
    crop: RowCrop<'_>,
    level: u32,
) -> PolarsResult<ImageDecode> {
    // An explicit decode-scale assertion lets JPEG decode skip work via IDCT
    // scaling; other formats fall through to a full decode. A scaled decode
    // is never given a crop (`compiled::roi_decodable`).
    if let Some(buf) = source
        .decode_max_size()
        .and_then(|max_size| decode_jpeg_scaled(bytes, max_size))
    {
        return Ok(ImageDecode::Whole(with_declared_dtype(buf, source)));
    }
    match crop {
        RowCrop::Window(op) => ImageAdapter::decode_region(bytes, op, level)
            .map(|d| ImageDecode::region(d, source))
            .map_err(decode_failed),
        RowCrop::Unresolved if tiff_region::is_tiff(bytes) => {
            let image = ImageAdapter::open_tiff(std::io::Cursor::new(bytes), level)
                .map_err(decode_failed)?;
            match image.unsupported() {
                None => Ok(ImageDecode::Unread),
                Some(_) => whole(bytes, source, level),
            }
        }
        RowCrop::None | RowCrop::Unresolved => whole(bytes, source, level),
    }
}

/// `bytes` decoded whole, at `level`.
fn whole(bytes: &[u8], source: &Source, level: u32) -> PolarsResult<ImageDecode> {
    ImageAdapter::decode_level(bytes, level)
        .map(|buf| ImageDecode::Whole(with_declared_dtype(buf, source)))
        .map_err(decode_failed)
}

/// An opened TIFF (a file or object read by range) decoded as
/// [`decode_image_bytes`] decodes its bytes, reading only the chunks a crop's
/// window or the level needs. `Ok(None)` for a layout the chunk decoder
/// does not carry ([`TiffImage::unsupported`]).
pub fn decode_tiff_image<S: TiffSource>(
    image: &mut TiffImage<S>,
    source: &Source,
    crop: RowCrop<'_>,
) -> PolarsResult<Option<ImageDecode>> {
    if image.unsupported().is_some() {
        return Ok(None);
    }
    let crop = match crop {
        RowCrop::Unresolved => return Ok(Some(ImageDecode::Unread)),
        RowCrop::Window(op) => Some(op),
        RowCrop::None => None,
    };
    match ImageAdapter::decode_tiff_region(image, crop).map_err(decode_failed)? {
        TiffRegion::Decoded(decoded) => Ok(Some(ImageDecode::region(decoded, source))),
        TiffRegion::Unsupported { .. } => Ok(None),
    }
}

/// `buf` cast to the dtype the source declares, if it declares one (a no-op
/// when it already has it).
fn with_declared_dtype(buf: ViewBuffer, source: &Source) -> ViewBuffer {
    match source.dtype() {
        Some(target) if buf.dtype() != target => buf.cast(target),
        _ => buf,
    }
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
