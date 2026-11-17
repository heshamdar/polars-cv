//! Pipeline execution engine.
//!
//! This module handles the execution of vision pipelines on Polars Series,
//! including parameter resolution and view-buffer integration.

use polars::prelude::*;

use view_buffer::{ImageAdapter, PlannedDType, ViewBuffer};

use crate::formats::sink::Sink;
use crate::formats::source::Source;
use crate::formats::Format as _;

/// What a node's leading steps let its image decode skip, settled once when
/// the graph compiles (`graph::compiled::compile`) and passed to every row's
/// [`decode_image_bytes`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeHint {
    /// Exact: the rows a leading literal `crop` reads (its last row). A PNG
    /// decodes only those, bit-identical to their full decode.
    pub rows_needed: Option<u32>,
    /// Approximate (`decode_from_plan="approximate"`): a leading literal
    /// `resize`'s `(width, height)`, as displayed. A JPEG decodes at the
    /// smallest IDCT scale whose output still covers both.
    pub scale_to: Option<(u32, u32)>,
    /// Approximate: the node starts with `grayscale` and the decode stays u8.
    /// A YCbCr JPEG no `scale_to` reduces decodes its luma (Y) channel alone,
    /// as `[H, W, 1]`.
    pub luma: bool,
}

/// How far a JPEG may be scaled down: the one choice of IDCT scale.
#[derive(Debug, Clone, Copy)]
enum JpegScale {
    /// `decode_max_size`: the long side stays `>= min(max_size, original)`.
    LongSide(u32),
    /// `decode_from_plan` before a resize: both stored sides stay `>=` these.
    Cover { width: u32, height: u32 },
}

/// A JPEG's stored size at an IDCT scale of `scale / 8`, as the decoder
/// computes it (rounded up).
fn idct_scaled(len: u16, scale: u32) -> u32 {
    (u32::from(len) * scale).div_ceil(8)
}

/// The size to request from `jpeg_decoder`'s `scale` so it picks the
/// smallest IDCT scale covering `width` x `height` on both sides (its own
/// rule needs only one side), or `None` when only the full size does.
fn cover_request(full: (u16, u16), width: u32, height: u32) -> Option<(u16, u16)> {
    [1, 2, 4].into_iter().find_map(|scale| {
        let (w, h) = (idct_scaled(full.0, scale), idct_scaled(full.1, scale));
        (w >= width && h >= height).then_some((w as u16, h as u16))
    })
}

/// Whether `bytes` start with the JPEG SOI marker.
fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0xD8])
}

/// Decode a JPEG at a reduced IDCT scale, as `scale` allows.
///
/// Picks the smallest of the decoder's scale factors (1/8, 1/4, 1/2) that
/// `scale` admits. Returns `None` for pixel formats the scaled path does not
/// cover (16-bit, CMYK), on any decode error, and when the factor is 1: the
/// caller falls back to the full decoder.
///
/// A factor of 1 has no work to skip, and this decoder is not the full one:
/// it is slower at full size (about 30%) and its pixels differ. So a
/// `scale` the image already meets is the full decode, byte for byte.
fn decode_jpeg_scaled(bytes: &[u8], scale: JpegScale) -> Option<ViewBuffer> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    // The stored size, read before `scale` replaces it with the scaled one.
    decoder.read_info().ok()?;
    let full = decoder.info()?;
    let (request_w, request_h) = match scale {
        JpegScale::LongSide(max_size) => {
            let requested = max_size.min(u16::MAX as u32) as u16;
            (requested, requested)
        }
        JpegScale::Cover { width, height } => {
            cover_request((full.width, full.height), width, height)?
        }
    };
    let (width, height) = decoder.scale(request_w, request_h).ok()?;
    if (width, height) == (full.width, full.height) {
        return None;
    }
    if let JpegScale::Cover {
        width: need_w,
        height: need_h,
    } = scale
    {
        if u32::from(width) < need_w || u32::from(height) < need_h {
            return None;
        }
    }
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

/// A zune-jpeg decoder (the one `image` decodes JPEG with, configured as it
/// does) over `bytes` with its headers read, if the file is YCbCr: the only
/// colour model whose first component is the luma.
fn ycbcr_decoder(
    bytes: &[u8],
) -> Option<zune_jpeg::JpegDecoder<zune_jpeg::zune_core::bytestream::ZCursor<&[u8]>>> {
    use zune_jpeg::zune_core::{
        bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions,
    };
    let options = DecoderOptions::default()
        .set_strict_mode(false)
        .set_max_width(usize::MAX)
        .set_max_height(usize::MAX)
        .jpeg_set_out_colorspace(ColorSpace::Luma);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder.decode_headers().ok()?;
    (decoder.input_colorspace()? == ColorSpace::YCbCr).then_some(decoder)
}

/// The decode a JPEG's leading steps allow in place of the full one, or
/// `None` for the full decode: not a JPEG, nothing to skip, or a file the
/// shortcut does not reproduce the shape of.
///
/// A scaled decode, when `scale` admits one, comes first and is RGB: no
/// decoder at hand does both (zune-jpeg has no IDCT scaling, and
/// jpeg-decoder's unconverted-components output overruns its buffer), and
/// the scale is the larger saving. Otherwise `luma` decodes the Y channel
/// alone with zune-jpeg, honoured only for a YCbCr file, and lands as
/// `[H, W, 1]`: the
/// shape `ImageAdapter::decode` gives a gray JPEG, which `grayscale` maps
/// exactly as it maps the full decode's `[H, W, 3]` under either `keepdims`
/// (`[H, W, 1]` packed, or reshaped to `[H, W]`), so the node's output shape
/// is the full decode's.
fn decode_jpeg_reduced(bytes: &[u8], scale: Option<JpegScale>, luma: bool) -> Option<ViewBuffer> {
    if !is_jpeg(bytes) {
        return None;
    }
    if let Some(buf) = scale.and_then(|scale| decode_jpeg_scaled(bytes, scale)) {
        return Some(buf);
    }
    if !luma {
        return None;
    }
    let mut decoder = ycbcr_decoder(bytes)?;
    let pixels = decoder.decode().ok()?;
    let (w, h) = decoder.dimensions()?;
    (pixels.len() == w * h).then(|| ViewBuffer::from_vec_with_shape(pixels, vec![h, w, 1]))
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
/// `hint` is what the node's leading steps let the decode skip
/// ([`DecodeHint`]): a PNG decodes only the rows a leading crop reads (not
/// with an orientation applied, where stored rows are not displayed rows),
/// and under `decode_from_plan="approximate"` a JPEG decodes scaled or as
/// luma.
pub fn decode_image_bytes(
    bytes: &[u8],
    source: &Source,
    hint: &DecodeHint,
) -> PolarsResult<ViewBuffer> {
    let tag = match source.applies_orientation() {
        true => ImageAdapter::orientation(bytes),
        false => None,
    };
    // How far a JPEG may scale down: an explicit assertion, or a leading
    // resize's target. The target is as displayed, so under a tag that
    // transposes (5..=8) its width is a stored height.
    let scale = match (source.decode_max_size(), hint.scale_to) {
        (Some(max_size), _) => Some(JpegScale::LongSide(max_size)),
        (None, Some((width, height))) => Some(match tag {
            Some(5..=8) => JpegScale::Cover {
                width: height,
                height: width,
            },
            _ => JpegScale::Cover { width, height },
        }),
        (None, None) => None,
    };
    let reduced = || decode_jpeg_reduced(bytes, scale, hint.luma);
    let partial = || {
        let rows = hint.rows_needed.filter(|_| !source.applies_orientation())?;
        ImageAdapter::decode_png_rows(bytes, rows as usize)
    };
    let buf = match reduced().or_else(partial) {
        Some(buf) => buf,
        None => ImageAdapter::decode(bytes)
            .map_err(|e| polars_err!(ComputeError: "Failed to decode image: {}", e))?,
    };
    // The file's orientation tag, when the source asks for the displayed
    // image: one step after either decoder, so a scaled decode (whose scale
    // was chosen from the stored size) is turned the same way.
    let buf = match tag {
        Some(tag) => ImageAdapter::oriented(buf, tag),
        None => buf,
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
/// The other sink formats never reach this function — `tensor` is
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
        | Sink::Long { .. }
        | Sink::Native
        | Sink::Tensor { .. } => unreachable!("only an image sink has a codec"),
    };
    encoded.map_err(|e| polars_err!(ComputeError: "Failed to encode {}: {}", codec.name(), e))
}

#[cfg(test)]
mod tests {
    use super::cover_request;

    /// `decode_from_plan` covers the target on *both* sides; the decoder's own
    /// `scale` rule (and `decode_max_size`'s) needs only one.
    #[test]
    fn a_cover_scale_keeps_both_sides_at_least_the_target() {
        // 1600x1200 -> 224x224: 1/4 (400x300); 1/8 (200x150) is too small.
        assert_eq!(cover_request((1600, 1200), 224, 224), Some((400, 300)));
        // A wide target the long side alone would accept at 1/8 (200 >= 150)
        // but whose short side then falls short (150 < 160): 1/4.
        assert_eq!(cover_request((1600, 1200), 150, 160), Some((400, 300)));
        // Odd sizes round up, as the decoder does: 1001/8 is 126.
        assert_eq!(cover_request((1001, 1001), 126, 126), Some((126, 126)));
        // Nothing smaller than full size covers it.
        assert_eq!(cover_request((400, 300), 224, 224), None);
    }
}
