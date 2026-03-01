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
    let dtype_str = match dt {
        view_buffer::DType::U8 => "uint8",
        view_buffer::DType::I8 => "int8",
        view_buffer::DType::U16 => "uint16",
        view_buffer::DType::I16 => "int16",
        view_buffer::DType::U32 => "uint32",
        view_buffer::DType::I32 => "int32",
        view_buffer::DType::U64 => "uint64",
        view_buffer::DType::I64 => "int64",
        view_buffer::DType::F32 => "float32",
        view_buffer::DType::F64 => "float64",
    };

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

    let channels = match color {
        image::ColorType::L8 | image::ColorType::L16 => 1,
        image::ColorType::La8 | image::ColorType::La16 => 2,
        image::ColorType::Rgb8 | image::ColorType::Rgb16 | image::ColorType::Rgb32F => 3,
        image::ColorType::Rgba8 | image::ColorType::Rgba16 | image::ColorType::Rgba32F => 4,
        _ => 1,
    };

    let dtype_str = match color {
        image::ColorType::L8
        | image::ColorType::La8
        | image::ColorType::Rgb8
        | image::ColorType::Rgba8 => "uint8",
        image::ColorType::L16
        | image::ColorType::La16
        | image::ColorType::Rgb16
        | image::ColorType::Rgba16 => "uint16",
        image::ColorType::Rgb32F | image::ColorType::Rgba32F => "float32",
        _ => "uint8",
    };

    Some(ImageMeta {
        width,
        height,
        channels,
        dtype: dtype_str,
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
        .into_iter()
        .map(|opt_bytes| opt_bytes.and_then(|b| extract_metadata(b).map(|m| m.width)))
        .collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}

/// Get image height from binary column (header-only, no full decode).
#[polars_expr(output_type=UInt32)]
fn image_height(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].binary()?;
    let out: UInt32Chunked = ca
        .into_iter()
        .map(|opt_bytes| opt_bytes.and_then(|b| extract_metadata(b).map(|m| m.height)))
        .collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}

/// Get image channel count from binary column (header-only, no full decode).
#[polars_expr(output_type=UInt32)]
fn image_channels(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].binary()?;
    let out: UInt32Chunked = ca
        .into_iter()
        .map(|opt_bytes| opt_bytes.and_then(|b| extract_metadata(b).map(|m| m.channels)))
        .collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}

/// Get image element dtype from binary column (header-only, no full decode).
#[polars_expr(output_type=String)]
fn image_dtype(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].binary()?;
    let out: StringChunked = ca
        .into_iter()
        .map(|opt_bytes| opt_bytes.and_then(|b| extract_metadata(b).map(|m| m.dtype)))
        .collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}
