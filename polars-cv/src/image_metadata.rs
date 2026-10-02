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
    /// `None` when the colour type was not recognised. Dimensions are read
    /// from the header before the colour type is consulted and stay valid,
    /// so an unknown colour type nulls these two fields only.
    channels: Option<u32>,
    dtype: Option<&'static str>,
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

    // Deliberately *not* `ImageCodec::channels_from_shape`, which the sink
    // contract's two halves share. That one answers "what may this be encoded
    // as", so a rank outside 2/3 has no channel count and it says so with
    // `None`, leaving the rank check to do the rejecting.
    //
    // This answers `.cv.width()`/`height()`/`channels()` for whatever buffer is
    // in the column, including ranks no encoder accepts, and its callers need a
    // number rather than a maybe. So it interprets every rank: a 1-D buffer is
    // one row of pixels, a rank-4+ buffer is described by its leading three
    // dimensions. Merging the two would force one of them to lie.
    let (height, width, channels) = match shape.len() {
        0 => (0, 0, 0),
        1 => (1, shape[0], 1),
        2 => (shape[0], shape[1], 1),
        _ => (shape[0], shape[1], shape[2]),
    };

    Some(ImageMeta {
        width,
        height,
        channels: Some(channels),
        dtype: Some(dtype_str),
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
    // not know these two facts, and `None` says that. Reporting a confident
    // "uint8"/1-channel answer for a format we failed to recognise is worse
    // than a null: it is indistinguishable from a real greyscale image.
    //
    // Only these two are nulled. `width`/`height` come off the header above,
    // before the colour type is consulted, and are equally valid whether or
    // not we recognise it — failing them too would make a future `image`
    // release that adds a variant silently regress `.cv.width()`/`.cv.height()`
    // on files whose dimensions we read perfectly well.
    let channels_and_dtype = match color {
        image::ColorType::L8 => Some((1, VbDType::U8)),
        image::ColorType::L16 => Some((1, VbDType::U16)),
        image::ColorType::La8 => Some((2, VbDType::U8)),
        image::ColorType::La16 => Some((2, VbDType::U16)),
        image::ColorType::Rgb8 => Some((3, VbDType::U8)),
        image::ColorType::Rgb16 => Some((3, VbDType::U16)),
        image::ColorType::Rgb32F => Some((3, VbDType::F32)),
        image::ColorType::Rgba8 => Some((4, VbDType::U8)),
        image::ColorType::Rgba16 => Some((4, VbDType::U16)),
        image::ColorType::Rgba32F => Some((4, VbDType::F32)),
        _ => None,
    };

    Some(ImageMeta {
        width,
        height,
        channels: channels_and_dtype.map(|(c, _)| c),
        dtype: channels_and_dtype.map(|(_, d)| d.numpy_name()),
    })
}

/// Extract metadata by trying VIEW protocol first, then image format.
fn extract_metadata(bytes: &[u8]) -> Option<ImageMeta> {
    try_view_header(bytes).or_else(|| try_image_header(bytes))
}

/// Static kwargs of the metadata functions: how to reach a *path* column's
/// files, exactly as `.cv.read_bytes()` takes them. Closed, and refused on a
/// binary column, where there is no path for them to apply to.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetaKwargs {
    #[serde(default)]
    cloud_options: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    on_error: Option<String>,
    #[serde(default)]
    allowed_roots: Option<Vec<String>>,
}

/// Each row's header metadata: from the bytes of a `Binary` column, or from
/// the files a `String` column of paths names — read only as far as the
/// header needs ([`crate::fetch::row_header`]). `None` for a null row, an
/// unrecognised format, or (with `on_error="null"`) an unreadable path.
fn metas(
    inputs: &[Series],
    kwargs: &MetaKwargs,
    name: &str,
) -> PolarsResult<Vec<Option<ImageMeta>>> {
    let input = &inputs[0];
    match input.dtype() {
        DataType::Binary => {
            if kwargs.cloud_options.is_some()
                || kwargs.on_error.is_some()
                || kwargs.allowed_roots.is_some()
            {
                polars_bail!(ComputeError:
                    "{}: cloud_options, on_error and allowed_roots apply to a path \
                     column; this column holds bytes",
                    name
                );
            }
            Ok(input
                .binary()?
                .iter()
                .map(|b| b.and_then(extract_metadata))
                .collect())
        }
        DataType::String | DataType::Null => {
            if input.dtype() == &DataType::Null {
                return Ok((0..input.len()).map(|_| None).collect());
            }
            let ca = input.str()?;
            let null_on_error =
                crate::fetch::parse_on_error(kwargs.on_error.as_deref().unwrap_or("raise"), name)?;
            let options = kwargs
                .cloud_options
                .as_ref()
                .map(crate::cloud::CloudOptions::from_map);
            let policy = kwargs
                .allowed_roots
                .as_deref()
                .map(crate::fetch::PathPolicy::new)
                .unwrap_or_default();
            let batch = crate::fetch::prefetch(ca, options.as_ref(), &policy);
            ca.iter()
                .map(|path| {
                    let Some(path) = path else { return Ok(None) };
                    match crate::fetch::row_header(
                        &batch,
                        path,
                        options.as_ref(),
                        &policy,
                        extract_metadata,
                    ) {
                        Ok(meta) => Ok(meta),
                        Err(_) if null_on_error => Ok(None),
                        Err(e) => Err(polars_err!(ComputeError: "{}: {}", name, e)),
                    }
                })
                .collect()
        }
        other => polars_bail!(ComputeError:
            "{} takes a Binary column of image bytes or a String column of paths, got {}",
            name, other
        ),
    }
}

fn u32_column(
    inputs: &[Series],
    kwargs: MetaKwargs,
    name: &'static str,
    field: impl Fn(&ImageMeta) -> Option<u32>,
) -> PolarsResult<Series> {
    let out: UInt32Chunked = metas(inputs, &kwargs, name)?
        .iter()
        .map(|m| m.as_ref().and_then(&field))
        .collect();
    Ok(out.with_name(inputs[0].name().clone()).into_series())
}

/// Image width (header-only, no full decode).
#[polars_expr(output_type=UInt32)]
fn image_width(inputs: &[Series], kwargs: MetaKwargs) -> PolarsResult<Series> {
    u32_column(inputs, kwargs, "width()", |m| Some(m.width))
}

/// Image height (header-only, no full decode).
#[polars_expr(output_type=UInt32)]
fn image_height(inputs: &[Series], kwargs: MetaKwargs) -> PolarsResult<Series> {
    u32_column(inputs, kwargs, "height()", |m| Some(m.height))
}

/// Image channel count (header-only, no full decode).
#[polars_expr(output_type=UInt32)]
fn image_channels(inputs: &[Series], kwargs: MetaKwargs) -> PolarsResult<Series> {
    u32_column(inputs, kwargs, "channels()", |m| m.channels)
}

/// Image element dtype (header-only, no full decode).
#[polars_expr(output_type=String)]
fn image_dtype(inputs: &[Series], kwargs: MetaKwargs) -> PolarsResult<Series> {
    let out: StringChunked = metas(inputs, &kwargs, "image_dtype()")?
        .iter()
        .map(|m| m.as_ref().and_then(|m| m.dtype))
        .collect();
    Ok(out.with_name(inputs[0].name().clone()).into_series())
}

/// The fields of `image_info`, in order: the four metadata functions' values.
fn image_info_fields() -> Vec<Field> {
    vec![
        Field::new(PlSmallStr::from_static("width"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("height"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("channels"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("dtype"), DataType::String),
    ]
}

fn image_info_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let name = input_fields.first().map_or_else(
        || PlSmallStr::from_static("image_info"),
        |f| f.name().clone(),
    );
    Ok(Field::new(name, DataType::Struct(image_info_fields())))
}

/// Width, height, channels and dtype from one header read per row.
#[polars_expr(output_type_func=image_info_output_type)]
fn image_info(inputs: &[Series], kwargs: MetaKwargs) -> PolarsResult<Series> {
    let metas = metas(inputs, &kwargs, "image_info()")?;
    let [w, h, c, d] = ["width", "height", "channels", "dtype"].map(PlSmallStr::from_static);
    let width: UInt32Chunked = metas.iter().map(|m| m.as_ref().map(|m| m.width)).collect();
    let height: UInt32Chunked = metas.iter().map(|m| m.as_ref().map(|m| m.height)).collect();
    let channels: UInt32Chunked = metas
        .iter()
        .map(|m| m.as_ref().and_then(|m| m.channels))
        .collect();
    let dtype: StringChunked = metas
        .iter()
        .map(|m| m.as_ref().and_then(|m| m.dtype))
        .collect();
    let fields = [
        width.with_name(w).into_series(),
        height.with_name(h).into_series(),
        channels.with_name(c).into_series(),
        dtype.with_name(d).into_series(),
    ];
    let mut out = StructChunked::from_series(inputs[0].name().clone(), metas.len(), fields.iter())?;
    if metas.iter().any(Option::is_none) {
        let validity: polars_arrow::bitmap::Bitmap = metas.iter().map(Option::is_some).collect();
        out = out.with_outer_validity(Some(validity));
    }
    Ok(out.into_series())
}
