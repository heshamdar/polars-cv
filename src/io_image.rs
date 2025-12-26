use crate::buffer::TensorBuffer;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use std::path::Path;

pub struct ImageAdapter;

impl ImageAdapter {
    /// Decodes raw image bytes (PNG, JPEG, etc.) into a TensorBuffer [H, W, C].
    /// This performs the "unavoidable copy" to decompress the image into memory.
    pub fn decode(encoded_bytes: &[u8]) -> Result<TensorBuffer, image::ImageError> {
        let img = image::load_from_memory(encoded_bytes)?;
        Ok(Self::from_dynamic_image(img))
    }

    /// Opens an image from disk and decodes it into a TensorBuffer.
    pub fn open(path: impl AsRef<Path>) -> Result<TensorBuffer, image::ImageError> {
        let img = image::open(path)?;
        Ok(Self::from_dynamic_image(img))
    }

    /// Converts a loaded DynamicImage into a TensorBuffer.
    /// Standardizes to RGB8 (3 channels) for now.
    pub fn from_dynamic_image(img: DynamicImage) -> TensorBuffer {
        let (w, h) = img.dimensions();
        // Shape: [Height, Width, Channels]
        let shape = vec![h as usize, w as usize, 3]; 
        
        // Convert to RGB8 (contiguous bytes)
        // This is where the copy/conversion happens.
        let rgb_img = img.to_rgb8();
        let raw_bytes = rgb_img.into_raw();

        // Create TensorBuffer directly from the bytes
        // We use the internal reshape helper from buffer.rs
        TensorBuffer::from_vec(raw_bytes)
            .reshape(shape)
    }

    /// Encodes a TensorBuffer into bytes (PNG/JPEG/etc).
    /// Requires the buffer to be [H, W, 3] and U8.
    pub fn encode(buffer: &TensorBuffer, format: image::ImageOutputFormat) -> Result<Vec<u8>, image::ImageError> {
        let dynamic_image = Self::to_dynamic_image(buffer)?;
        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);
        dynamic_image.write_to(&mut cursor, format)?;
        Ok(bytes)
    }

    /// Saves a TensorBuffer to a file.
    pub fn save(buffer: &TensorBuffer, path: impl AsRef<Path>) -> Result<(), image::ImageError> {
        let dynamic_image = Self::to_dynamic_image(buffer)?;
        dynamic_image.save(path)
    }

    /// Helper to convert TensorBuffer -> DynamicImage
    fn to_dynamic_image(buffer: &TensorBuffer) -> Result<DynamicImage, image::ImageError> {
        // 1. Validation
        if buffer.dtype() != crate::dtype::DType::U8 {
            return Err(image::ImageError::Parameter(image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::Generic("Image export requires U8 dtype".to_string())
            )));
        }
        let shape = buffer.shape();
        if shape.len() != 3 || shape[2] != 3 {
             return Err(image::ImageError::Parameter(image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::DimensionMismatch
            )));
        }
        let (h, w) = (shape[0] as u32, shape[1] as u32);

        // 2. Ensure Contiguous
        // We need a standard contiguous buffer for the image crate to consume
        let contiguous = buffer.to_contiguous();
        
        // 3. Construct ImageBuffer
        // Safety: We checked dtype is U8.
        let slice = unsafe { 
            std::slice::from_raw_parts(contiguous.as_ptr::<u8>(), contiguous.layout.num_elements()) 
        };
        
        // ImageBuffer::from_raw takes Vec<u8>, so we create a copy into image container
        let img_buf = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
            .ok_or_else(|| image::ImageError::Parameter(image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::Generic("Failed to create ImageBuffer".to_string())
            )))?;

        Ok(DynamicImage::ImageRgb8(img_buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_roundtrip() {
        // 1. Create Synthetic Image (2x2 RGB)
        let data: Vec<u8> = vec![
            255, 0, 0,   0, 255, 0,    // Row 1
            0, 0, 255,   255, 255, 0   // Row 2
        ];
        let tb = TensorBuffer::from_vec(data).reshape(vec![2, 2, 3]);

        // 2. Encode to PNG
        let encoded = ImageAdapter::encode(&tb, image::ImageOutputFormat::Png).unwrap();
        assert!(encoded.len() > 0);

        // 3. Decode back
        let decoded = ImageAdapter::decode(&encoded).unwrap();
        assert_eq!(decoded.shape(), &[2, 2, 3]);
    }
}