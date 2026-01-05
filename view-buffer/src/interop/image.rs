//! Image crate interoperability.
//!
//! This module provides:
//! - Zero-copy image views via [`ImageView`] and [`ImageViewAdapter`]
//! - Image I/O via [`ImageAdapter`]

use crate::core::buffer::{BufferError, ViewBuffer};
use crate::core::dtype::{DType, ViewType};
use crate::core::layout::ExternalLayout;
use crate::interop::{validate_layout, ExternalView};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Pixel, Rgb};
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
        let img = image::load_from_memory(encoded_bytes)?;
        Ok(Self::from_dynamic_image(img))
    }

    /// Opens an image from disk and decodes it into a ViewBuffer.
    pub fn open(path: impl AsRef<Path>) -> Result<ViewBuffer, image::ImageError> {
        let img = image::open(path)?;
        Ok(Self::from_dynamic_image(img))
    }

    /// Converts a loaded DynamicImage into a ViewBuffer.
    pub fn from_dynamic_image(img: DynamicImage) -> ViewBuffer {
        let (w, h) = img.dimensions();
        let shape = vec![h as usize, w as usize, 3];

        let rgb_img = img.to_rgb8();
        let raw_bytes = rgb_img.into_raw();

        ViewBuffer::from_vec(raw_bytes).reshape(shape)
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

    /// Saves a ViewBuffer to a file.
    pub fn save(buffer: &ViewBuffer, path: impl AsRef<Path>) -> Result<(), image::ImageError> {
        let dynamic_image = Self::to_dynamic_image(buffer)?;
        dynamic_image.save(path)
    }

    /// Convert ViewBuffer -> DynamicImage.
    ///
    /// This is useful for interoperating with the image crate's APIs.
    /// The buffer must have U8 dtype and be in [H, W, 3] (RGB) or [H, W] / [H, W, 1] (Luma) format.
    pub fn to_dynamic_image(buffer: &ViewBuffer) -> Result<DynamicImage, image::ImageError> {
        // 1. Validation
        if buffer.dtype() != DType::U8 {
            return Err(image::ImageError::Parameter(
                image::error::ParameterError::from_kind(image::error::ParameterErrorKind::Generic(
                    "Image export requires U8 dtype".to_string(),
                )),
            ));
        }

        let shape = buffer.shape();
        // Support [H, W, 3] (RGB) or [H, W, 1] / [H, W] (Luma)
        let channels = if shape.len() == 3 {
            shape[2]
        } else if shape.len() == 2 {
            1
        } else {
            0
        };

        if channels != 1 && channels != 3 {
            return Err(image::ImageError::Parameter(
                image::error::ParameterError::from_kind(
                    image::error::ParameterErrorKind::DimensionMismatch,
                ),
            ));
        }

        let (h, w) = (shape[0] as u32, shape[1] as u32);

        // 2. Ensure Contiguous
        // We need a standard contiguous buffer for the image crate to consume
        let contiguous = buffer.to_contiguous();

        // 3. Construct ImageBuffer
        let slice = unsafe {
            std::slice::from_raw_parts(contiguous.as_ptr::<u8>(), contiguous.layout.num_elements())
        };

        if channels == 3 {
            // RGB
            let img_buf = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                .ok_or_else(|| {
                    image::ImageError::Parameter(image::error::ParameterError::from_kind(
                        image::error::ParameterErrorKind::Generic(
                            "Failed to create RGB ImageBuffer".to_string(),
                        ),
                    ))
                })?;
            Ok(DynamicImage::ImageRgb8(img_buf))
        } else {
            // Grayscale (Luma)
            let img_buf = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                .ok_or_else(|| {
                    image::ImageError::Parameter(image::error::ParameterError::from_kind(
                        image::error::ParameterErrorKind::Generic(
                            "Failed to create Luma ImageBuffer".to_string(),
                        ),
                    ))
                })?;
            Ok(DynamicImage::ImageLuma8(img_buf))
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
}
