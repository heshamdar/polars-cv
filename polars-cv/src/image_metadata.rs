//! Header-only image metadata extraction.
//!
//! Provides Polars expression functions that extract width, height, channels,
//! and dtype from image binary data without performing a full decode. Supports
//! both standard image formats (PNG, JPEG, WebP, TIFF, BMP, GIF) via the
//! `image` crate's header-only reader, and the VIEW binary protocol via
//! direct header parsing.

use image::ImageDecoder;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::io::Cursor;
use view_buffer::protocol::{u8_to_dtype, HEADER_SIZE, MAGIC_BYTES};
use view_buffer::DType as VbDType;

/// Parsed metadata from an image or VIEW blob header.
struct ImageMeta {
    width: u32,
    height: u32,
    channels: u32,
    dtype: &'static str,
}

/// Try to extract metadata from VIEW protocol header (first 64 bytes).
fn try_view_header(bytes: &[u8]) -> Option<ImageMeta> {
    if bytes.len() < HEADER_SIZE || bytes[..4] != MAGIC_BYTES {
        return None;
    }
    let dtype_code = bytes[6];
    let rank = bytes[7] as usize;
    let dt = u8_to_dtype(dtype_code)?;
    let dtype_str = dt.numpy_name();

    // Shape dimensions are stored after the 64-byte header, each as u64 LE
    let shape_start = HEADER_SIZE;
    let shape_bytes_needed = shape_start + rank * 8;
    if bytes.len() < shape_bytes_needed {
        return None;
    }

    let mut shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let offset = shape_start + i * 8;
        let dim = u64::from_le_bytes(bytes[offset..offset + 8].try_into().ok()?) as u32;
        shape.push(dim);
    }

    let (height, width, channels) = match shape.len() {
        0 => (0, 0, 0),
        1 => (1, shape[0], 1),
        2 => (shape[0], shape[1], 1),
        _ => (shape[0], shape[1], shape[2]),
    };

    Some(ImageMeta {
        width,
        height,
        channels,
        dtype: dtype_str,
    })
}

/// Try to extract metadata from an encoded image (PNG, JPEG, etc.) using
/// header-only decoding via the `image` crate.
fn try_image_header(bytes: &[u8]) -> Option<ImageMeta> {
    let cursor = Cursor::new(bytes);
    let reader = image::ImageReader::new(cursor).with_guessed_format().ok()?;

    reader.format()?;

    let decoder = reader.into_decoder().ok()?;
    let (width, height) = decoder.dimensions();
    let color = decoder.color_type();

    // Channel count and dtype are two facts about one colour type, so they are
    // read in one match: splitting them let the two arms disagree about which
    // variants they covered, and each carried its own `_` guess (1 channel,
    // "uint8") for the ones it did not. `image::ColorType` is `#[non_exhaustive]`,
    // so a catch-all is mandatory — but an unrecognised colour type means we do
    // not know the metadata, and `None` says that. Reporting a confident
    // "uint8"/1-channel answer for a format we failed to recognise is worse
    // than a null: it is indistinguishable from a real greyscale image.
    let (channels, dtype) = match color {
        image::ColorType::L8 => (1, VbDType::U8),
        image::ColorType::L16 => (1, VbDType::U16),
        image::ColorType::La8 => (2, VbDType::U8),
        image::ColorType::La16 => (2, VbDType::U16),
        image::ColorType::Rgb8 => (3, VbDType::U8),
        image::ColorType::Rgb16 => (3, VbDType::U16),
        image::ColorType::Rgb32F => (3, VbDType::F32),
        image::ColorType::Rgba8 => (4, VbDType::U8),
        image::ColorType::Rgba16 => (4, VbDType::U16),
        image::ColorType::Rgba32F => (4, VbDType::F32),
        _ => return None,
    };

    Some(ImageMeta {
        width,
        height,
        channels,
        dtype: dtype.numpy_name(),
    })
}

/// Extract metadata by trying VIEW protocol first, then image format.
fn extract_metadata(bytes: &[u8]) -> Option<ImageMeta> {
    try_view_header(bytes).or_else(|| try_image_header(bytes))
}

/// Get image width from binary column (header-only, no full decode).
#[polars_expr(output_type=UInt32)]
fn image_width(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].binary()?;
    let out: UInt32Chunked = ca
        .iter()
        .map(|opt_bytes| opt_bytes.and_then(|b| extract_metadata(b).map(|m| m.width)))
        .collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}

/// Get image height from binary column (header-only, no full decode).
#[polars_expr(output_type=UInt32)]
fn image_height(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].binary()?;
    let out: UInt32Chunked = ca
        .iter()
        .map(|opt_bytes| opt_bytes.and_then(|b| extract_metadata(b).map(|m| m.height)))
        .collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}

/// Get image channel count from binary column (header-only, no full decode).
#[polars_expr(output_type=UInt32)]
fn image_channels(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].binary()?;
    let out: UInt32Chunked = ca
        .iter()
        .map(|opt_bytes| opt_bytes.and_then(|b| extract_metadata(b).map(|m| m.channels)))
        .collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}

/// Get image element dtype from binary column (header-only, no full decode).
#[polars_expr(output_type=String)]
fn image_dtype(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].binary()?;
    let out: StringChunked = ca
        .iter()
        .map(|opt_bytes| opt_bytes.and_then(|b| extract_metadata(b).map(|m| m.dtype)))
        .collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}
