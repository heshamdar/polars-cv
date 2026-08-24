//! Image crate interoperability.
//!
//! This module provides:
//! - Zero-copy image views via [`ImageView`] and [`ImageViewAdapter`]
//! - Image I/O via [`ImageAdapter`]

use crate::core::buffer::{BufferError, ViewBuffer};
use crate::core::dtype::{DType, PlannedDType, ViewType};
use crate::core::layout::ExternalLayout;
use crate::interop::{validate_layout, ExternalView};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, LumaA, Pixel, Rgb, Rgba};
use std::marker::PhantomData;
use std::path::Path;

// --- Image codec support ---

/// The image codecs a buffer can be encoded to.
///
/// Each carries a restriction on the buffer's dtype and channel count that has
/// nothing to do with the pixels: JPEG is an 8-bit format, PNG carries 8- or
/// 16-bit samples, TIFF has a fixed table of colour types. Those facts are
/// knowable from a buffer's *description* — they never require looking at the
/// data — which is why [`ImageCodec::check_support`] exists and why the query
/// planner can call it before a byte moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageCodec {
    Png,
    Jpeg,
    WebP,
    Tiff,
}

impl ImageCodec {
    /// Parse the plugin's sink-format spelling.
    pub fn from_sink_format(format: &str) -> Option<Self> {
        match format {
            "png" => Some(Self::Png),
            "jpeg" => Some(Self::Jpeg),
            "webp" => Some(Self::WebP),
            "tiff" => Some(Self::Tiff),
            _ => None,
        }
    }

    /// The codec's name, for error messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::WebP => "WebP",
            Self::Tiff => "TIFF",
        }
    }

    /// Can this codec encode a buffer with the given description?
    ///
    /// **This is the single authority for every image-codec precondition.** The
    /// encoders below call it, and so does the plugin's planner, so a query
    /// that cannot possibly encode is refused while it is still being planned
    /// rather than part-way through `collect()`. A second copy of this table
    /// would be a second answer to "can this be a JPEG".
    ///
    /// Every argument is optional because the planner does not always know:
    /// `dtype` is unknown while a source's decode dtype is still `"auto"`, and
    /// `channels` is unknown whenever the shape hints are incomplete. **An
    /// unknown is never a rejection** — this returns `Err` only for a
    /// combination that could not encode under any completion of the unknowns,
    /// so a plan-time caller passing less information can only be more
    /// permissive, never wrong.
    pub fn check_support(
        self,
        dtype: Option<DType>,
        rank: Option<usize>,
        channels: Option<usize>,
    ) -> Result<(), String> {
        self.check_planned(
            dtype.map_or(PlannedDType::Unknown, PlannedDType::Known),
            rank,
            channels,
        )
    }

    /// The channel count an image-shaped buffer of this `shape` carries.
    ///
    /// `[H, W, C]` carries `C`; `[H, W]` is single-channel; any other rank has
    /// no channel count, which is *unknown* rather than zero — the rank check
    /// is what rejects it, and reporting `Some(0)` here would misattribute
    /// that to the channel range.
    ///
    /// Private on purpose. This mapping was written out twice, character for
    /// character, at the two sites that call into this check — one on the
    /// planning side and one on the executing side of the same contract — so
    /// [`check_shape`](Self::check_shape) takes the shape and applies it here
    /// rather than leaving each caller to derive it and pass it in.
    fn channels_from_shape(shape: &[usize]) -> Option<usize> {
        match shape.len() {
            3 => Some(shape[2]),
            2 => Some(1),
            _ => None,
        }
    }

    /// [`check_planned`](Self::check_planned) against a *shape*, deriving the
    /// rank and channel count instead of trusting a caller to.
    ///
    /// This is the entry point for both halves of the sink contract: the
    /// planner checks a shape it expects, and the encoder re-checks the shape
    /// it actually got. Neither computes channels, so neither can compute them
    /// differently.
    ///
    /// `shape` is what the caller knows of the buffer's dimensions, and
    /// `rank_hint` covers the planner's case of knowing the rank without the
    /// shape (an `expected_ndim` with no `expected_shape`). A known `shape`
    /// wins, since its length *is* the rank.
    pub fn check_shape(
        self,
        dtype: PlannedDType,
        shape: Option<&[usize]>,
        rank_hint: Option<usize>,
    ) -> Result<(), String> {
        self.check_planned(
            dtype,
            shape.map(<[usize]>::len).or(rank_hint),
            shape.and_then(Self::channels_from_shape),
        )
    }

    /// [`check_support`](Self::check_support) for a dtype the planner has only
    /// partially resolved.
    ///
    /// Rejects only when *every* dtype the input could still turn out to be is
    /// unsupported, so a less-resolved state is always more permissive. This is
    /// what lets the planner refuse `scale(...)` into a JPEG sink over a source
    /// whose decode dtype is unknown: the op promotes to a float, and no float
    /// is an 8-bit sample, whichever float it is.
    pub fn check_planned(
        self,
        dtype: PlannedDType,
        rank: Option<usize>,
        channels: Option<usize>,
    ) -> Result<(), String> {
        if let Some(rank) = rank {
            if rank != 2 && rank != 3 {
                return Err(format!(
                    "{} encoding needs an image-shaped buffer ([H, W] or [H, W, C]), \
                     but this is {rank}-dimensional. Reshape before the sink, or use \
                     a `numpy`/`list`/`array` sink for non-image data.",
                    self.name()
                ));
            }
        }
        if let Some(channels) = channels {
            if !(1..=4).contains(&channels) {
                return Err(format!(
                    "{} encoding supports 1 to 4 channels, but this buffer has \
                     {channels}.",
                    self.name()
                ));
            }
        }

        let candidates = dtype.candidates();
        if candidates.is_empty() {
            return Ok(());
        }
        // Every candidate must fail before this is a refusal.
        let mut last_err = None;
        for &candidate in candidates {
            match self.check_one_dtype(candidate, channels) {
                Ok(()) => return Ok(()),
                Err(msg) => last_err = Some(msg),
            }
        }
        // For a partially-resolved dtype, naming one candidate would be
        // arbitrary — the user's pipeline does not say "F64", it says "some
        // float". Describe the constraint that actually rules the sink out.
        if !dtype.is_concrete() {
            return Err(format!(
                "{} cannot encode this pipeline's output. The source's decode \
                 dtype is not known at planning time, but the operations \
                 promote it to floating point, and no floating-point buffer is \
                 encodable here{}. Cast before the sink (e.g. `.cast(\"u8\")`), \
                 or sink to a format that carries float data.",
                self.name(),
                match (self, channels) {
                    (Self::Tiff, Some(c)) if c != 1 =>
                        format!(" at {c} channels (TIFF stores floats greyscale-only)"),
                    _ => String::new(),
                }
            ));
        }
        Err(last_err.expect("candidates is non-empty"))
    }

    /// The dtype half of the table, for one concrete dtype.
    fn check_one_dtype(self, dtype: DType, channels: Option<usize>) -> Result<(), String> {
        match self {
            // Both go through `to_dynamic_image`, which carries 8- and 16-bit
            // samples only; WebP is additionally 8-bit.
            Self::Png => {
                if !matches!(dtype, DType::U8 | DType::U16) {
                    return Err(format!(
                        "Image export requires U8 or U16 dtype, got {dtype:?}. \
                         Cast to an integer dtype first (e.g. `.cast(\"u8\")` or \
                         `.cast(\"u16\")`), or sink to TIFF to preserve float data."
                    ));
                }
            }
            Self::Jpeg | Self::WebP => {
                if dtype != DType::U8 {
                    return Err(format!(
                        "{} is an 8-bit format but the image is {dtype:?}; cast to u8 \
                         first (.cast(\"u8\")) or sink to PNG/TIFF to preserve higher \
                         bit depth.",
                        self.name()
                    ));
                }
            }
            // The tiff crate has a colour type per (dtype, channels) pair, and
            // the float ones are greyscale only. Mirrors the match in
            // `encode_tiff` — which is why that match's fallback arm defers
            // here for its message rather than writing a second one.
            Self::Tiff => {
                let ok = match dtype {
                    DType::U8 | DType::U16 => true,
                    DType::F32 | DType::F64 => channels.is_none_or(|c| c == 1),
                    _ => false,
                };
                if !ok {
                    return Err(match (dtype, channels) {
                        (DType::F32 | DType::F64, Some(c)) => format!(
                            "TIFF stores floating-point samples as greyscale only, but \
                             this buffer is {dtype:?} with {c} channels; reduce to one \
                             channel or cast to u8/u16."
                        ),
                        _ => format!(
                            "TIFF encoding does not support {dtype:?}; cast to u8, u16, \
                             or a single-channel f32/f64."
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

// --- Image View Types ---

/// A zero-copy view over a ViewBuffer interpreted as an image.
#[derive(Debug, Clone)]
pub struct ImageView<'a, P: Pixel> {
    pub data: &'a [P::Subpixel],
    pub width: u32,
    pub height: u32,
    pub row_stride: usize,
    _marker: PhantomData<P>,
}

impl<'a, P> ImageView<'a, P>
where
    P: Pixel,
    P::Subpixel: ViewType + 'static,
{
    /// Returns the pixel data at the given coordinates.
    ///
    /// Kept despite having no caller inside this workspace: it is the only
    /// accessor on [`ImageView`], which `AsImageView` hands to downstream users
    /// of the crate. Deleting it would leave the view type with no way to read
    /// what it borrows.
    pub fn get_pixel(&self, x: u32, y: u32) -> &[P::Subpixel] {
        let start = (y as usize * self.row_stride) + (x as usize * P::CHANNEL_COUNT as usize);
        &self.data[start..start + P::CHANNEL_COUNT as usize]
    }
}

// --- Image Adapter ---

/// Adapter for zero-copy image views.
pub struct ImageViewAdapter<P>(PhantomData<P>);

impl<'a, P> ExternalView<'a> for ImageViewAdapter<P>
where
    P: Pixel,
    P::Subpixel: ViewType + 'static,
{
    type View = ImageView<'a, P>;
    const LAYOUT: ExternalLayout = ExternalLayout::ImageCrate;

    fn try_view(buf: &'a ViewBuffer) -> Result<Self::View, BufferError> {
        validate_layout(buf, Self::LAYOUT)?;

        if buf.dtype() != P::Subpixel::DTYPE {
            return Err(BufferError::TypeMismatch {
                expected: P::Subpixel::DTYPE,
                got: buf.dtype(),
            });
        }

        let shape = buf.shape();
        let (h, w) = (shape[0], shape[1]);
        let stride_bytes = buf.strides_bytes()[0];
        let elem_size = std::mem::size_of::<P::Subpixel>() as isize;
        let row_stride_elems = (stride_bytes / elem_size) as usize;

        let total_elems = row_stride_elems * h;
        let ptr = unsafe { buf.as_ptr::<P::Subpixel>() };

        let data = unsafe { std::slice::from_raw_parts(ptr, total_elems) };

        Ok(ImageView {
            data,
            width: w as u32,
            height: h as u32,
            row_stride: row_stride_elems,
            _marker: PhantomData,
        })
    }
}

// --- Convenience Trait ---

/// Trait for converting ViewBuffer to image view.
pub trait AsImageView {
    /// Attempts to create a zero-copy image view.
    fn as_image_view<P>(&self) -> Result<ImageView<'_, P>, BufferError>
    where
        P: Pixel,
        P::Subpixel: ViewType + 'static;
}

impl AsImageView for ViewBuffer {
    fn as_image_view<P>(&self) -> Result<ImageView<'_, P>, BufferError>
    where
        P: Pixel,
        P::Subpixel: ViewType + 'static,
    {
        ImageViewAdapter::try_view(self)
    }
}

// --- Image I/O Adapter ---

/// Adapter for image file I/O operations.
pub struct ImageAdapter;

impl ImageAdapter {
    /// Decodes raw image bytes (PNG, JPEG, etc.) into a ViewBuffer [H, W, C].
    pub fn decode(encoded_bytes: &[u8]) -> Result<ViewBuffer, image::ImageError> {
        // Check if it's a TIFF file by magic bytes
        if encoded_bytes.len() >= 4
            && (
                &encoded_bytes[0..4] == b"II*\x00" ||  // Little-endian TIFF
            &encoded_bytes[0..4] == b"MM\x00*"
                // Big-endian TIFF
            )
        {
            // Use our custom TIFF decoder for floating-point support
            Self::decode_tiff(encoded_bytes)
        } else {
            // Use image crate for other formats
            let img = image::load_from_memory(encoded_bytes)?;
            Ok(Self::from_dynamic_image(img))
        }
    }

    /// Opens an image from disk and decodes it into a ViewBuffer.
    /// Routes through `decode()` to use the custom TIFF decoder for float TIFF support.
    pub fn open(path: impl AsRef<Path>) -> Result<ViewBuffer, image::ImageError> {
        let bytes = std::fs::read(path.as_ref()).map_err(image::ImageError::IoError)?;
        Self::decode(&bytes)
    }

    /// Converts a loaded DynamicImage into a ViewBuffer.
    ///
    /// Preserves native dtype (u8, u16, f32) and always produces 3D `[H, W, C]`.
    /// Alpha channels are preserved: RGBA produces `[H, W, 4]`, LumaA produces `[H, W, 2]`.
    pub fn from_dynamic_image(img: DynamicImage) -> ViewBuffer {
        let (w, h) = img.dimensions();

        match &img {
            // 16-bit RGBA
            DynamicImage::ImageRgba16(_) => {
                let rgba16 = img.to_rgba16();
                let shape = vec![h as usize, w as usize, 4];
                ViewBuffer::from_vec(rgba16.into_raw()).reshape(shape)
            }
            // 16-bit RGB
            DynamicImage::ImageRgb16(_) => {
                let rgb16 = img.to_rgb16();
                let shape = vec![h as usize, w as usize, 3];
                ViewBuffer::from_vec(rgb16.into_raw()).reshape(shape)
            }
            // 16-bit grayscale+alpha
            DynamicImage::ImageLumaA16(_) => {
                let lumaa16 = img.to_luma_alpha16();
                let shape = vec![h as usize, w as usize, 2];
                ViewBuffer::from_vec(lumaa16.into_raw()).reshape(shape)
            }
            // 16-bit grayscale
            DynamicImage::ImageLuma16(_) => {
                let luma16 = img.to_luma16();
                let shape = vec![h as usize, w as usize, 1];
                ViewBuffer::from_vec(luma16.into_raw()).reshape(shape)
            }
            // 8-bit grayscale+alpha
            DynamicImage::ImageLumaA8(_) => {
                let lumaa8 = img.to_luma_alpha8();
                let shape = vec![h as usize, w as usize, 2];
                ViewBuffer::from_vec(lumaa8.into_raw()).reshape(shape)
            }
            // 8-bit grayscale
            DynamicImage::ImageLuma8(_) => {
                let luma8 = img.to_luma8();
                let shape = vec![h as usize, w as usize, 1];
                ViewBuffer::from_vec(luma8.into_raw()).reshape(shape)
            }
            // 32-bit float RGB
            DynamicImage::ImageRgb32F(_) => {
                let rgb32f = img.to_rgb32f();
                let shape = vec![h as usize, w as usize, 3];
                ViewBuffer::from_vec(rgb32f.into_raw()).reshape(shape)
            }
            // 32-bit float RGBA
            DynamicImage::ImageRgba32F(_) => {
                let rgba32f = img.to_rgba32f();
                let shape = vec![h as usize, w as usize, 4];
                ViewBuffer::from_vec(rgba32f.into_raw()).reshape(shape)
            }
            // 8-bit RGBA
            DynamicImage::ImageRgba8(_) => {
                let rgba8 = img.to_rgba8();
                let shape = vec![h as usize, w as usize, 4];
                ViewBuffer::from_vec(rgba8.into_raw()).reshape(shape)
            }
            // 8-bit RGB (common case)
            _ => {
                let rgb_img = img.to_rgb8();
                let shape = vec![h as usize, w as usize, 3];
                ViewBuffer::from_vec(rgb_img.into_raw()).reshape(shape)
            }
        }
    }

    /// Encodes a ViewBuffer into bytes (PNG/JPEG/etc).
    ///
    /// Note: For JPEG quality control, use `encode_jpeg` instead.
    pub fn encode(
        buffer: &ViewBuffer,
        format: image::ImageFormat,
    ) -> Result<Vec<u8>, image::ImageError> {
        let dynamic_image = Self::to_dynamic_image(buffer)?;
        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);
        dynamic_image.write_to(&mut cursor, format)?;
        Ok(bytes)
    }

    /// Encodes a ViewBuffer as JPEG with specified quality (1-100).
    pub fn encode_jpeg(buffer: &ViewBuffer, quality: u8) -> Result<Vec<u8>, image::ImageError> {
        use image::codecs::jpeg::JpegEncoder;

        let dynamic_image = Self::to_dynamic_image(buffer)?;
        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);

        let encoder = JpegEncoder::new_with_quality(&mut cursor, quality);
        dynamic_image.write_with_encoder(encoder)?;
        Ok(bytes)
    }

    /// Encodes a ViewBuffer as TIFF with native support for floating-point data.
    ///
    /// This method supports both integer and floating-point data types,
    /// making it suitable for medical imaging and scientific data.
    /// Uses the tiff crate directly for native floating-point support with LZW compression.
    pub fn encode_tiff(buffer: &ViewBuffer) -> Result<Vec<u8>, image::ImageError> {
        use std::io::Cursor;
        use tiff::encoder::{colortype, Compression, TiffEncoder};

        // Ensure buffer is contiguous
        let contiguous = buffer.to_contiguous();
        let shape = contiguous.shape();

        let (h, w, channels) = match shape.len() {
            2 => (shape[0] as u32, shape[1] as u32, 1),
            3 if matches!(shape[2], 1..=4) => (shape[0] as u32, shape[1] as u32, shape[2]),
            _ => {
                return Err(image::ImageError::Parameter(
                    image::error::ParameterError::from_kind(
                        image::error::ParameterErrorKind::Generic(
                            "TIFF encoder supports [H, W], [H, W, 1], [H, W, 2], [H, W, 3], or [H, W, 4]".to_string(),
                        ),
                    ),
                ));
            }
        };

        // Choose compression based on data type and characteristics
        // LZW works well for most data types and provides good compression
        // For floating-point data, LZW is preferred over Deflate as it handles
        // the bit patterns in IEEE 754 floats more efficiently
        let compression = Compression::Lzw;

        // Setup encoder with lossless compression
        let mut bytes = Vec::new();
        let mut cursor = Cursor::new(&mut bytes);
        let mut encoder = TiffEncoder::new(&mut cursor)
            .map_err(|e| {
                image::ImageError::IoError(std::io::Error::other(format!(
                    "TIFF encoder creation failed: {e}"
                )))
            })?
            .with_compression(compression);

        // Helper to map tiff encoding errors
        let tiff_err = |e: tiff::TiffError| -> image::ImageError {
            image::ImageError::IoError(std::io::Error::other(format!("TIFF encoding failed: {e}")))
        };

        match (contiguous.dtype(), channels) {
            (crate::core::dtype::DType::U8, 1) => {
                encoder
                    .write_image::<colortype::Gray8>(w, h, contiguous.as_slice::<u8>())
                    .map_err(tiff_err)?;
            }
            (crate::core::dtype::DType::U8, 2) => {
                // tiff crate has no GrayA encoder; expand to RGBA for encoding
                let src = contiguous.as_slice::<u8>();
                let mut rgba = Vec::with_capacity(src.len() * 2);
                for pixel in src.as_chunks::<2>().0 {
                    rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], pixel[1]]);
                }
                encoder
                    .write_image::<colortype::RGBA8>(w, h, &rgba)
                    .map_err(tiff_err)?;
            }
            (crate::core::dtype::DType::U8, 3) => {
                encoder
                    .write_image::<colortype::RGB8>(w, h, contiguous.as_slice::<u8>())
                    .map_err(tiff_err)?;
            }
            (crate::core::dtype::DType::U8, 4) => {
                encoder
                    .write_image::<colortype::RGBA8>(w, h, contiguous.as_slice::<u8>())
                    .map_err(tiff_err)?;
            }
            (crate::core::dtype::DType::U16, 1) => {
                encoder
                    .write_image::<colortype::Gray16>(w, h, contiguous.as_slice::<u16>())
                    .map_err(tiff_err)?;
            }
            (crate::core::dtype::DType::U16, 2) => {
                // tiff crate has no GrayA encoder; expand to RGBA for encoding
                let src = contiguous.as_slice::<u16>();
                let mut rgba = Vec::with_capacity(src.len() * 2);
                for pixel in src.as_chunks::<2>().0 {
                    rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], pixel[1]]);
                }
                encoder
                    .write_image::<colortype::RGBA16>(w, h, &rgba)
                    .map_err(tiff_err)?;
            }
            (crate::core::dtype::DType::U16, 3) => {
                encoder
                    .write_image::<colortype::RGB16>(w, h, contiguous.as_slice::<u16>())
                    .map_err(tiff_err)?;
            }
            (crate::core::dtype::DType::U16, 4) => {
                encoder
                    .write_image::<colortype::RGBA16>(w, h, contiguous.as_slice::<u16>())
                    .map_err(tiff_err)?;
            }
            (crate::core::dtype::DType::F32, 1) => {
                encoder
                    .write_image::<colortype::Gray32Float>(w, h, contiguous.as_slice::<f32>())
                    .map_err(tiff_err)?;
            }
            (crate::core::dtype::DType::F64, 1) => {
                encoder
                    .write_image::<colortype::Gray64Float>(w, h, contiguous.as_slice::<f64>())
                    .map_err(tiff_err)?;
            }
            (dtype, channels) => {
                // Unreachable for anything `ImageCodec::Tiff` admits: the arms
                // above cover exactly its table. Ask it for the message rather
                // than writing a second one that could describe a different
                // rule than the planner enforced.
                let msg = ImageCodec::Tiff
                    .check_support(Some(dtype), Some(shape.len()), Some(channels))
                    .err()
                    .unwrap_or_else(|| {
                        format!(
                            "TIFF encoding has no colour type for {dtype:?} with \
                             {channels} channels"
                        )
                    });
                return Err(image::ImageError::Parameter(
                    image::error::ParameterError::from_kind(
                        image::error::ParameterErrorKind::Generic(msg),
                    ),
                ));
            }
        }

        Ok(bytes)
    }

    /// Decodes TIFF bytes with native support for floating-point data.
    ///
    /// This method supports both integer and floating-point TIFF files,
    /// making it suitable for reading medical imaging and scientific data.
    /// Uses the tiff crate directly for native floating-point support.
    pub fn decode_tiff(encoded_bytes: &[u8]) -> Result<ViewBuffer, image::ImageError> {
        use std::io::Cursor;
        use tiff::decoder::{Decoder, DecodingResult};

        let mut cursor = Cursor::new(encoded_bytes);
        let mut decoder = Decoder::new(&mut cursor).map_err(|e| {
            image::ImageError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("TIFF decoder creation failed: {e}"),
            ))
        })?;

        // Get image dimensions and format info
        let (width, height) = decoder.dimensions().map_err(|e| {
            image::ImageError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to get TIFF dimensions: {e}"),
            ))
        })?;

        let colortype = decoder.colortype().map_err(|e| {
            image::ImageError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to get TIFF color type: {e}"),
            ))
        })?;

        // Decode the image data
        let decoding_result = decoder.read_image().map_err(|e| {
            image::ImageError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("TIFF decoding failed: {e}"),
            ))
        })?;

        // Helper: determine channel count from TIFF color type
        let channels_for =
            |ct: &tiff::ColorType, allow_alpha: bool| -> Result<usize, image::ImageError> {
                match ct {
                    tiff::ColorType::Gray(_) => Ok(1),
                    tiff::ColorType::RGB(_) => Ok(3),
                    tiff::ColorType::Palette(_) if allow_alpha => Ok(3),
                    tiff::ColorType::GrayA(_) if allow_alpha => Ok(2),
                    tiff::ColorType::RGBA(_) if allow_alpha => Ok(4),
                    _ => Err(image::ImageError::IoError(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unsupported TIFF color type: {ct:?}"),
                    ))),
                }
            };

        // Helper: build shape from dimensions and channels.
        // Always produces 3D [H, W, C] for consistency with PNG/JPEG decoding.
        let make_shape =
            |h: u32, w: u32, c: usize| -> Vec<usize> { vec![h as usize, w as usize, c] };

        // Convert the decoded data to ViewBuffer based on the data type
        match decoding_result {
            DecodingResult::U8(data) => {
                let channels = channels_for(&colortype, true)?;
                Ok(ViewBuffer::from_vec(data).reshape(make_shape(height, width, channels)))
            }
            DecodingResult::U16(data) => {
                let channels = channels_for(&colortype, true)?;
                Ok(ViewBuffer::from_vec(data).reshape(make_shape(height, width, channels)))
            }
            DecodingResult::F32(data) => {
                let channels = channels_for(&colortype, false)?;
                Ok(ViewBuffer::from_vec(data).reshape(make_shape(height, width, channels)))
            }
            DecodingResult::F64(data) => {
                let channels = channels_for(&colortype, false)?;
                Ok(ViewBuffer::from_vec(data).reshape(make_shape(height, width, channels)))
            }
            _ => Err(image::ImageError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unsupported TIFF data type",
            ))),
        }
    }

    /// Saves a ViewBuffer to a file.
    pub fn save(buffer: &ViewBuffer, path: impl AsRef<Path>) -> Result<(), image::ImageError> {
        let dynamic_image = Self::to_dynamic_image(buffer)?;
        dynamic_image.save(path)
    }

    /// Convert ViewBuffer -> DynamicImage.
    ///
    /// This is useful for interoperating with the image crate's APIs.
    /// The buffer must have U8 or U16 dtype and be in `[H, W, C]` where C is 1,
    /// 2, 3, or 4, or `[H, W]` (treated as single-channel). U8 buffers produce
    /// 8-bit `DynamicImage` variants; U16 buffers produce 16-bit variants (so a
    /// 16-bit buffer round-trips to a 16-bit PNG).
    pub fn to_dynamic_image(buffer: &ViewBuffer) -> Result<DynamicImage, image::ImageError> {
        let dtype = buffer.dtype();
        // The dtype/channel rule is ImageCodec's, not this function's: the
        // planner refuses unencodable queries from the same table, and a second
        // copy here is how the two would come to disagree. PNG is the widest of
        // the `to_dynamic_image` consumers (8- or 16-bit), so it is the right
        // gate for the shared conversion; JPEG and WebP narrow it further at
        // their own entry points.
        ImageCodec::Png
            .check_shape(PlannedDType::Known(dtype), Some(buffer.shape()), None)
            .map_err(|msg| {
                image::ImageError::Parameter(image::error::ParameterError::from_kind(
                    image::error::ParameterErrorKind::Generic(msg),
                ))
            })?;

        let shape = buffer.shape();
        let channels = if shape.len() == 3 {
            shape[2]
        } else if shape.len() == 2 {
            1
        } else {
            0
        };

        if !matches!(channels, 1..=4) {
            return Err(image::ImageError::Parameter(
                image::error::ParameterError::from_kind(
                    image::error::ParameterErrorKind::DimensionMismatch,
                ),
            ));
        }

        let (h, w) = (shape[0] as u32, shape[1] as u32);
        let contiguous = buffer.to_contiguous();

        let make_err = |label: &str| {
            image::ImageError::Parameter(image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::Generic(format!(
                    "Failed to create {label} ImageBuffer"
                )),
            ))
        };

        match dtype {
            DType::U8 => {
                let slice = contiguous.as_slice::<u8>();
                match channels {
                    4 => {
                        let img_buf =
                            ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                                .ok_or_else(|| make_err("RGBA"))?;
                        Ok(DynamicImage::ImageRgba8(img_buf))
                    }
                    3 => {
                        let img_buf =
                            ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                                .ok_or_else(|| make_err("RGB"))?;
                        Ok(DynamicImage::ImageRgb8(img_buf))
                    }
                    2 => {
                        let img_buf =
                            ImageBuffer::<LumaA<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                                .ok_or_else(|| make_err("LumaA"))?;
                        Ok(DynamicImage::ImageLumaA8(img_buf))
                    }
                    _ => {
                        let img_buf =
                            ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                                .ok_or_else(|| make_err("Luma"))?;
                        Ok(DynamicImage::ImageLuma8(img_buf))
                    }
                }
            }
            _ => {
                let slice = contiguous.as_slice::<u16>();
                match channels {
                    4 => {
                        let img_buf =
                            ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(w, h, slice.to_vec())
                                .ok_or_else(|| make_err("RGBA16"))?;
                        Ok(DynamicImage::ImageRgba16(img_buf))
                    }
                    3 => {
                        let img_buf =
                            ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(w, h, slice.to_vec())
                                .ok_or_else(|| make_err("RGB16"))?;
                        Ok(DynamicImage::ImageRgb16(img_buf))
                    }
                    2 => {
                        let img_buf =
                            ImageBuffer::<LumaA<u16>, Vec<u16>>::from_raw(w, h, slice.to_vec())
                                .ok_or_else(|| make_err("LumaA16"))?;
                        Ok(DynamicImage::ImageLumaA16(img_buf))
                    }
                    _ => {
                        let img_buf =
                            ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(w, h, slice.to_vec())
                                .ok_or_else(|| make_err("Luma16"))?;
                        Ok(DynamicImage::ImageLuma16(img_buf))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every codec, every dtype, at the channel counts an image can have.
    const ALL_DTYPES: &[DType] = &[
        DType::U8,
        DType::I8,
        DType::U16,
        DType::I16,
        DType::U32,
        DType::I32,
        DType::U64,
        DType::I64,
        DType::F32,
        DType::F64,
    ];
    const ALL_CODECS: &[ImageCodec] = &[
        ImageCodec::Png,
        ImageCodec::Jpeg,
        ImageCodec::WebP,
        ImageCodec::Tiff,
    ];

    #[test]
    fn unknowns_are_never_a_rejection() {
        // The planner calls `check_support` with whatever it knows, which is
        // often less than the encoder will know. That is only sound if adding
        // information can turn an Ok into an Err but never the reverse —
        // otherwise the planner would refuse queries that execute perfectly.
        for &codec in ALL_CODECS {
            assert!(
                codec.check_support(None, None, None).is_ok(),
                "{codec:?} rejected a fully-unknown buffer"
            );
            for &dtype in ALL_DTYPES {
                for channels in 1..=4usize {
                    if codec
                        .check_support(Some(dtype), Some(3), Some(channels))
                        .is_ok()
                    {
                        // Every less-informed call must also be Ok.
                        assert!(codec.check_support(Some(dtype), Some(3), None).is_ok());
                        assert!(codec.check_support(None, Some(3), Some(channels)).is_ok());
                        assert!(codec.check_support(None, None, None).is_ok());
                    }
                }
            }
        }
    }

    /// Cells where the table is deliberately stricter than the raw encoder.
    ///
    /// `image`'s JPEG and WebP writers accept a 16-bit `DynamicImage` and
    /// quietly downconvert it to 8 bits. Silently halving a user's bit depth is
    /// worse than refusing, so the plugin has always rejected u16 into those
    /// two — see `test_jpeg_rejects_u16_with_actionable_error`. The table
    /// carries that product decision, which is why it says "no" where the
    /// adapter would have said "yes".
    const DELIBERATELY_STRICTER: &[(ImageCodec, DType)] = &[
        (ImageCodec::Jpeg, DType::U16),
        (ImageCodec::WebP, DType::U16),
    ];

    #[test]
    fn the_table_never_promises_what_an_encoder_cannot_deliver() {
        // The direction that matters. A table that admits something the
        // encoder then rejects is precisely the plan-then-fail bug this whole
        // mechanism exists to prevent: the planner would publish a schema and
        // `collect()` would die. The reverse (table stricter than the encoder)
        // is safe, and where it is intentional it is listed above.
        for &codec in ALL_CODECS {
            for &dtype in ALL_DTYPES {
                for channels in 1..=4usize {
                    let buf = ViewBuffer::from_vec_with_shape(
                        vec![0u8; 4 * channels],
                        vec![2, 2, channels],
                    )
                    .cast_to(dtype);
                    let permitted = codec
                        .check_support(Some(dtype), Some(3), Some(channels))
                        .is_ok();
                    let encodes = match codec {
                        ImageCodec::Png => {
                            ImageAdapter::encode(&buf, image::ImageFormat::Png).is_ok()
                        }
                        ImageCodec::Jpeg => ImageAdapter::encode_jpeg(&buf, 85).is_ok(),
                        ImageCodec::WebP => {
                            ImageAdapter::encode(&buf, image::ImageFormat::WebP).is_ok()
                        }
                        ImageCodec::Tiff => ImageAdapter::encode_tiff(&buf).is_ok(),
                    };
                    if permitted {
                        assert!(
                            encodes,
                            "{codec:?} with {dtype:?} x {channels}ch: the table \
                             permits it but the encoder rejects it — a query \
                             would plan and then fail at collect()"
                        );
                    } else if encodes {
                        assert!(
                            DELIBERATELY_STRICTER.contains(&(codec, dtype)),
                            "{codec:?} with {dtype:?} x {channels}ch: the table \
                             refuses something the encoder handles. If that is \
                             intended, add it to DELIBERATELY_STRICTER with the \
                             reason; otherwise the table is over-rejecting."
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_deliberate_narrowings_are_still_narrowings() {
        // Guards the exemption list itself: if `image` ever starts rejecting
        // 16-bit JPEG on its own, the entry becomes dead and should go, and if
        // the plugin ever stops refusing it the entry is actively wrong.
        for &(codec, dtype) in DELIBERATELY_STRICTER {
            assert!(
                codec.check_support(Some(dtype), Some(3), Some(1)).is_err(),
                "{codec:?}/{dtype:?} is listed as deliberately refused but the \
                 table admits it"
            );
        }
    }

    #[test]
    fn rank_one_is_not_an_image_for_any_codec() {
        for &codec in ALL_CODECS {
            assert!(codec.check_support(Some(DType::U8), Some(1), None).is_err());
            assert!(codec.check_support(Some(DType::U8), Some(4), None).is_err());
            assert!(codec.check_support(Some(DType::U8), Some(2), None).is_ok());
            assert!(codec.check_support(Some(DType::U8), Some(3), None).is_ok());
        }
    }

    #[test]
    fn from_sink_format_covers_exactly_the_codec_sinks() {
        for &codec in ALL_CODECS {
            let spelling = match codec {
                ImageCodec::Png => "png",
                ImageCodec::Jpeg => "jpeg",
                ImageCodec::WebP => "webp",
                ImageCodec::Tiff => "tiff",
            };
            assert_eq!(ImageCodec::from_sink_format(spelling), Some(codec));
        }
        // `blob` is a VIEW dump, not a codec, and must not be parsed as one.
        assert_eq!(ImageCodec::from_sink_format("blob"), None);
        assert_eq!(ImageCodec::from_sink_format("numpy"), None);
    }

    #[test]
    fn test_image_roundtrip() {
        let data: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
        let tb = ViewBuffer::from_vec(data).reshape(vec![2, 2, 3]);

        let encoded = ImageAdapter::encode(&tb, image::ImageFormat::Png).unwrap();
        assert!(!encoded.is_empty());

        let decoded = ImageAdapter::decode(&encoded).unwrap();
        assert_eq!(decoded.shape(), &[2, 2, 3]);
    }

    #[test]
    fn test_u16_png_round_trip_gray() {
        // 16-bit grayscale values that would be clipped/lost at 8 bits.
        let original_data: Vec<u16> = vec![65535, 30000, 12345, 1];
        let tb = ViewBuffer::from_vec(original_data.clone()).reshape(vec![2, 2, 1]);

        let encoded = ImageAdapter::encode(&tb, image::ImageFormat::Png).unwrap();
        assert!(!encoded.is_empty());

        let decoded = ImageAdapter::decode(&encoded).unwrap();
        assert_eq!(decoded.shape(), &[2, 2, 1]);
        assert_eq!(decoded.dtype(), DType::U16);
        assert_eq!(decoded.as_slice::<u16>(), &original_data);
    }

    #[test]
    fn test_u16_png_round_trip_rgb() {
        // Distinct 16-bit values across an RGB image.
        let original_data: Vec<u16> = vec![
            65535, 0, 30000, // px (0,0)
            12345, 54321, 100, // px (0,1)
            1, 2, 3, // px (1,0)
            40000, 41000, 42000, // px (1,1)
        ];
        let tb = ViewBuffer::from_vec(original_data.clone()).reshape(vec![2, 2, 3]);

        let encoded = ImageAdapter::encode(&tb, image::ImageFormat::Png).unwrap();
        assert!(!encoded.is_empty());

        let decoded = ImageAdapter::decode(&encoded).unwrap();
        assert_eq!(decoded.shape(), &[2, 2, 3]);
        assert_eq!(decoded.dtype(), DType::U16);
        assert_eq!(decoded.as_slice::<u16>(), &original_data);
    }

    #[test]
    fn test_to_dynamic_image_rejects_float_with_actionable_error() {
        let data: Vec<f32> = vec![1.0, 0.5, 0.25, 0.125];
        let tb = ViewBuffer::from_vec(data).reshape(vec![2, 2, 1]);

        let err = ImageAdapter::to_dynamic_image(&tb).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("U8 or U16") && msg.contains("cast"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_jpeg_roundtrip() {
        let data: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
        let tb = ViewBuffer::from_vec(data).reshape(vec![2, 2, 3]);

        let encoded = ImageAdapter::encode_jpeg(&tb, 85).unwrap();
        assert!(!encoded.is_empty());

        let decoded = ImageAdapter::decode(&encoded).unwrap();
        assert_eq!(decoded.shape(), &[2, 2, 3]);
    }

    #[test]
    fn test_tiff_u8_encoding() {
        let data: Vec<u8> = vec![255, 128, 64, 32];
        let tb = ViewBuffer::from_vec(data).reshape(vec![2, 2]);

        let encoded = ImageAdapter::encode_tiff(&tb).unwrap();
        assert!(!encoded.is_empty());

        // Verify TIFF magic bytes
        assert_eq!(&encoded[0..4], b"II*\0"); // Little-endian TIFF header
    }

    #[test]
    fn test_tiff_f32_encoding() {
        let data: Vec<f32> = vec![1.0, 0.5, 0.25, 0.125];
        let tb = ViewBuffer::from_vec(data).reshape(vec![2, 2]);

        let encoded = ImageAdapter::encode_tiff(&tb).unwrap();
        assert!(!encoded.is_empty());

        // Verify TIFF magic bytes
        assert_eq!(&encoded[0..4], b"II*\0"); // Little-endian TIFF header
    }

    #[test]
    fn test_tiff_f32_round_trip() {
        let original_data: Vec<f32> = vec![1.0, 0.5, 0.25, 0.125];
        let original_buffer = ViewBuffer::from_vec(original_data.clone()).reshape(vec![2, 2]);

        // Encode to TIFF
        let encoded = ImageAdapter::encode_tiff(&original_buffer).unwrap();
        assert!(!encoded.is_empty());
        assert_eq!(&encoded[0..4], b"II*\0");

        // Decode back from TIFF
        let decoded_buffer = ImageAdapter::decode_tiff(&encoded).unwrap();

        // decode_tiff always returns 3D [H, W, C] (consistent with PNG/JPEG decode);
        // a 2x2 grayscale image decodes to [2, 2, 1].
        assert_eq!(decoded_buffer.shape(), &[2, 2, 1]);
        assert_eq!(decoded_buffer.dtype(), crate::core::dtype::DType::F32);

        // Verify data is preserved (floating-point precision)
        let decoded_data = decoded_buffer.as_slice::<f32>();
        assert_eq!(decoded_data.len(), original_data.len());

        for (original, decoded) in original_data.iter().zip(decoded_data.iter()) {
            assert!(
                (original - decoded).abs() < f32::EPSILON,
                "Original: {original}, Decoded: {decoded}"
            );
        }
    }

    #[test]
    fn test_tiff_u8_round_trip() {
        let original_data: Vec<u8> = vec![255, 128, 64, 32];
        let original_buffer = ViewBuffer::from_vec(original_data.clone()).reshape(vec![2, 2]);

        // Encode to TIFF
        let encoded = ImageAdapter::encode_tiff(&original_buffer).unwrap();
        assert!(!encoded.is_empty());
        assert_eq!(&encoded[0..4], b"II*\0");

        // Decode back from TIFF
        let decoded_buffer = ImageAdapter::decode_tiff(&encoded).unwrap();

        // decode_tiff always returns 3D [H, W, C] (consistent with PNG/JPEG decode);
        // a 2x2 grayscale image decodes to [2, 2, 1].
        assert_eq!(decoded_buffer.shape(), &[2, 2, 1]);
        assert_eq!(decoded_buffer.dtype(), crate::core::dtype::DType::U8);

        // Verify data is preserved exactly
        let decoded_data = decoded_buffer.as_slice::<u8>();
        assert_eq!(decoded_data, &original_data);
    }
}
