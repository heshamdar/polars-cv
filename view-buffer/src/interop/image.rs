//! Image crate interoperability.
//!
//! This module provides:
//! - Zero-copy image views via [`ImageView`] and [`ImageViewAdapter`]
//! - Image I/O via [`ImageAdapter`]

use crate::core::buffer::{BufferError, ViewBuffer};
use crate::core::dtype::{DType, ViewType};
use crate::core::layout::ExternalLayout;
use crate::interop::{validate_layout, ExternalView};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, LumaA, Pixel, Rgb, Rgba};
use std::marker::PhantomData;
use std::path::Path;

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
                for pixel in src.chunks_exact(2) {
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
                for pixel in src.chunks_exact(2) {
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
                return Err(image::ImageError::Parameter(
                    image::error::ParameterError::from_kind(
                        image::error::ParameterErrorKind::Generic(
                            format!("Unsupported combination for TIFF encoding: {dtype:?} with {channels} channels"),
                        ),
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
    /// The buffer must have U8 dtype and be in `[H, W, C]` where C is 1, 2, 3, or 4,
    /// or `[H, W]` (treated as single-channel).
    pub fn to_dynamic_image(buffer: &ViewBuffer) -> Result<DynamicImage, image::ImageError> {
        if buffer.dtype() != DType::U8 {
            return Err(image::ImageError::Parameter(
                image::error::ParameterError::from_kind(image::error::ParameterErrorKind::Generic(
                    "Image export requires U8 dtype".to_string(),
                )),
            ));
        }

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
        let slice = unsafe {
            std::slice::from_raw_parts(contiguous.as_ptr::<u8>(), contiguous.layout.num_elements())
        };

        let make_err = |label: &str| {
            image::ImageError::Parameter(image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::Generic(format!(
                    "Failed to create {label} ImageBuffer"
                )),
            ))
        };

        match channels {
            4 => {
                let img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                    .ok_or_else(|| make_err("RGBA"))?;
                Ok(DynamicImage::ImageRgba8(img_buf))
            }
            3 => {
                let img_buf = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                    .ok_or_else(|| make_err("RGB"))?;
                Ok(DynamicImage::ImageRgb8(img_buf))
            }
            2 => {
                let img_buf = ImageBuffer::<LumaA<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                    .ok_or_else(|| make_err("LumaA"))?;
                Ok(DynamicImage::ImageLumaA8(img_buf))
            }
            _ => {
                let img_buf = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                    .ok_or_else(|| make_err("Luma"))?;
                Ok(DynamicImage::ImageLuma8(img_buf))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // Verify shape is preserved
        assert_eq!(decoded_buffer.shape(), &[2, 2]);
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

        // Verify shape and data type are preserved
        assert_eq!(decoded_buffer.shape(), &[2, 2]);
        assert_eq!(decoded_buffer.dtype(), crate::core::dtype::DType::U8);

        // Verify data is preserved exactly
        let decoded_data = decoded_buffer.as_slice::<u8>();
        assert_eq!(decoded_data, &original_data);
    }
}
