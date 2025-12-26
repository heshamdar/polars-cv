use crate::buffer::ViewBuffer;
use crate::dtype::DType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Luma};
use std::path::Path;

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

        ViewBuffer::from_vec(raw_bytes)
            .reshape(shape)
    }

    /// Encodes a ViewBuffer into bytes (PNG/JPEG/etc).
    pub fn encode(buffer: &ViewBuffer, format: image::ImageOutputFormat) -> Result<Vec<u8>, image::ImageError> {
        let dynamic_image = Self::to_dynamic_image(buffer)?;
        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);
        dynamic_image.write_to(&mut cursor, format)?;
        Ok(bytes)
    }

    /// Saves a ViewBuffer to a file.
    pub fn save(buffer: &ViewBuffer, path: impl AsRef<Path>) -> Result<(), image::ImageError> {
        let dynamic_image = Self::to_dynamic_image(buffer)?;
        dynamic_image.save(path)
    }

    /// Helper to convert ViewBuffer -> DynamicImage
    fn to_dynamic_image(buffer: &ViewBuffer) -> Result<DynamicImage, image::ImageError> {
        // 1. Validation
        if buffer.dtype() != DType::U8 {
            return Err(image::ImageError::Parameter(image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::Generic("Image export requires U8 dtype".to_string())
            )));
        }
        
        let shape = buffer.shape();
        // Support [H, W, 3] (RGB) or [H, W, 1] / [H, W] (Luma)
        let channels = if shape.len() == 3 { shape[2] } else if shape.len() == 2 { 1 } else { 0 };

        if channels != 1 && channels != 3 {
            return Err(image::ImageError::Parameter(image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::DimensionMismatch
            )));
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
                .ok_or_else(|| image::ImageError::Parameter(image::error::ParameterError::from_kind(
                    image::error::ParameterErrorKind::Generic("Failed to create RGB ImageBuffer".to_string())
                )))?;
            Ok(DynamicImage::ImageRgb8(img_buf))
        } else {
            // Grayscale (Luma)
            let img_buf = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(w, h, slice.to_vec())
                .ok_or_else(|| image::ImageError::Parameter(image::error::ParameterError::from_kind(
                    image::error::ParameterErrorKind::Generic("Failed to create Luma ImageBuffer".to_string())
                )))?;
            Ok(DynamicImage::ImageLuma8(img_buf))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_roundtrip() {
        let data: Vec<u8> = vec![
            255, 0, 0,   0, 255, 0,    
            0, 0, 255,   255, 255, 0   
        ];
        let tb = ViewBuffer::from_vec(data).reshape(vec![2, 2, 3]);

        let encoded = ImageAdapter::encode(&tb, image::ImageOutputFormat::Png).unwrap();
        assert!(encoded.len() > 0);

        let decoded = ImageAdapter::decode(&encoded).unwrap();
        assert_eq!(decoded.shape(), &[2, 2, 3]);
    }
}