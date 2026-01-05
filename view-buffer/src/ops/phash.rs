//! Perceptual hashing operations.
//!
//! This module provides perceptual image hashing using various algorithms:
//! - Average Hash (aHash): Fastest, least robust
//! - Difference Hash (dHash): Gradient-based, good balance
//! - Perceptual Hash (pHash): DCT-based, most robust to transformations
//! - Blockhash: Block-based, good for cropped images
//!
//! The output is a fixed-length binary hash (as u8 bytes) that can be
//! compared using Hamming distance to determine image similarity.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::cost::OpCost;
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Perceptual hash algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum HashAlgorithm {
    /// Average Hash (aHash) - fastest, least robust.
    /// Computes average pixel value and compares each pixel to the mean.
    Average,

    /// Difference Hash (dHash) - gradient-based.
    /// Compares adjacent pixels for gradient direction.
    Difference,

    /// Perceptual Hash (pHash) - DCT-based, most robust.
    /// Uses Discrete Cosine Transform for frequency analysis.
    /// Best for detecting similar images under resize/compression.
    #[default]
    Perceptual,

    /// Blockhash.io algorithm - block-based.
    /// Divides image into blocks and compares block averages.
    /// More robust to cropping than other algorithms.
    Blockhash,
}

/// Perceptual hashing operation.
///
/// Computes a fixed-length perceptual hash of an image that can be used
/// to detect similar images even after transformations like resize,
/// compression, or minor edits.
///
/// # Output
///
/// Returns a 1D u8 array of shape `[hash_size / 8]` containing the hash bytes.
/// For a 64-bit hash (default), this is shape `[8]`.
///
/// # Example
///
/// ```ignore
/// let op = PerceptualHashOp::new(HashAlgorithm::Perceptual);
/// let hash = op.execute(&image_buffer);
/// // hash.shape() == [8] for 64-bit hash
/// // hash.dtype() == DType::U8
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerceptualHashOp {
    /// Hash algorithm to use.
    pub algorithm: HashAlgorithm,

    /// Hash size in bits (must be a power of 2: 64, 128, 256).
    /// Default is 64 bits (8 bytes).
    pub hash_size: u32,
}

impl PerceptualHashOp {
    /// Create a new perceptual hash operation with the given algorithm.
    ///
    /// Uses default hash size of 64 bits.
    pub fn new(algorithm: HashAlgorithm) -> Self {
        Self {
            algorithm,
            hash_size: 64,
        }
    }

    /// Create a new perceptual hash operation with default algorithm (pHash).
    pub fn phash() -> Self {
        Self::new(HashAlgorithm::Perceptual)
    }

    /// Set the hash size in bits.
    ///
    /// Must be a power of 2 (64, 128, 256, etc.).
    /// Larger hash sizes provide more precision but require more storage.
    pub fn with_hash_size(mut self, bits: u32) -> Self {
        self.hash_size = bits;
        self
    }

    /// Get the number of bytes in the output hash.
    pub fn hash_bytes(&self) -> usize {
        (self.hash_size / 8) as usize
    }

    /// Execute the perceptual hash operation on a buffer.
    ///
    /// The buffer should be an image in [H, W, C] or [H, W] format.
    /// Returns a 1D u8 buffer containing the hash bytes.
    #[cfg(feature = "perceptual_hash")]
    pub fn execute(&self, buffer: &ViewBuffer) -> ViewBuffer {
        use crate::interop::image::ImageAdapter;
        use image_hasher::{HashAlg, HasherConfig};

        // Convert ViewBuffer to DynamicImage
        let dynamic_image = ImageAdapter::to_dynamic_image(buffer)
            .expect("Failed to convert buffer to image for hashing");

        // Calculate hash_size parameter for image_hasher
        // image_hasher uses hash_size to mean the side length of the hash grid
        // For a 64-bit hash, we need hash_size=8 (8x8=64)
        let hash_width = (self.hash_size as f64).sqrt() as u32;

        // Configure the hasher based on algorithm
        // Note: image_hasher algorithm mappings:
        // - Mean: Average hash (aHash)
        // - Mean + dct: Perceptual hash (pHash) - DCT-based
        // - Gradient: Difference hash (dHash)
        // - Blockhash: Block-based hash
        let hasher = match self.algorithm {
            HashAlgorithm::Average => HasherConfig::new()
                .hash_alg(HashAlg::Mean)
                .hash_size(hash_width, hash_width)
                .to_hasher(),
            HashAlgorithm::Difference => HasherConfig::new()
                .hash_alg(HashAlg::Gradient)
                .hash_size(hash_width, hash_width)
                .to_hasher(),
            HashAlgorithm::Perceptual => {
                // pHash uses Mean with DCT preprocessing
                HasherConfig::new()
                    .hash_alg(HashAlg::Mean)
                    .preproc_dct()
                    .hash_size(hash_width, hash_width)
                    .to_hasher()
            }
            HashAlgorithm::Blockhash => HasherConfig::new()
                .hash_alg(HashAlg::Blockhash)
                .hash_size(hash_width, hash_width)
                .to_hasher(),
        };

        // Compute the hash
        let hash = hasher.hash_image(&dynamic_image);

        // Get hash as bytes
        let hash_bytes = hash.as_bytes().to_vec();

        // Return as 1D u8 buffer
        let len = hash_bytes.len();
        ViewBuffer::from_vec(hash_bytes).reshape(vec![len])
    }

    /// Fallback when perceptual_hash feature is not enabled.
    #[cfg(not(feature = "perceptual_hash"))]
    pub fn execute(&self, _buffer: &ViewBuffer) -> ViewBuffer {
        panic!("PerceptualHashOp requires the 'perceptual_hash' feature to be enabled");
    }
}

impl Default for PerceptualHashOp {
    fn default() -> Self {
        Self::phash()
    }
}

impl Op for PerceptualHashOp {
    fn name(&self) -> &'static str {
        "PerceptualHash"
    }

    fn infer_shape(&self, _inputs: &[&[usize]]) -> Vec<usize> {
        // Output is always a 1D array of hash bytes
        vec![self.hash_bytes()]
    }

    fn infer_dtype(&self, _inputs: &[DType]) -> DType {
        // Hash is always output as u8 bytes
        DType::U8
    }

    fn memory_effect(&self) -> MemoryEffect {
        // Hashing requires the full image to be loaded
        MemoryEffect::RequiresContiguous
    }

    fn intrinsic_cost(&self) -> OpCost {
        // Hash computation allocates a new buffer for the result
        OpCost::Allocating
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        // Output is always contiguous
        None
    }

    fn validate(
        &self,
        input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        // Validate input is an image-like shape
        let shape = input_shapes[0];
        if shape.len() < 2 || shape.len() > 3 {
            return Err(ValidationError::InvalidParameter {
                param: "input_shape".to_string(),
                reason: format!("Expected 2D or 3D image shape [H, W] or [H, W, C], got {shape:?}"),
            });
        }

        // Validate hash_size is a power of 2 and reasonable
        if !self.hash_size.is_power_of_two() {
            return Err(ValidationError::InvalidParameter {
                param: "hash_size".to_string(),
                reason: format!("hash_size must be a power of 2, got {}", self.hash_size),
            });
        }

        if self.hash_size < 16 || self.hash_size > 1024 {
            return Err(ValidationError::InvalidParameter {
                param: "hash_size".to_string(),
                reason: format!(
                    "hash_size must be between 16 and 1024 bits, got {}",
                    self.hash_size
                ),
            });
        }

        Ok(())
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        // Accept any numeric input - will be converted to U8 for image processing
        DTypeCategory::Numeric
    }

    fn working_dtype(&self) -> Option<DType> {
        // Internally works with U8 (image format)
        Some(DType::U8)
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        // Hash is always U8 bytes
        OutputDTypeRule::Fixed(DType::U8)
    }
}

#[cfg(all(test, feature = "perceptual_hash"))]
mod tests {
    use super::*;

    fn create_test_image(width: usize, height: usize, seed: u8) -> ViewBuffer {
        // Create a simple test image with a pattern
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                // Create a gradient pattern
                let r = ((x + seed as usize) % 256) as u8;
                let g = ((y + seed as usize) % 256) as u8;
                let b = (((x + y) + seed as usize) % 256) as u8;
                data.push(r);
                data.push(g);
                data.push(b);
            }
        }
        ViewBuffer::from_vec(data).reshape(vec![height, width, 3])
    }

    #[test]
    fn test_phash_output_shape() {
        let img = create_test_image(64, 64, 0);
        let op = PerceptualHashOp::new(HashAlgorithm::Perceptual);
        let hash = op.execute(&img);

        // Default 64-bit hash = 8 bytes
        assert_eq!(hash.shape(), &[8]);
        assert_eq!(hash.dtype(), DType::U8);
    }

    #[test]
    fn test_ahash_output_shape() {
        let img = create_test_image(64, 64, 0);
        let op = PerceptualHashOp::new(HashAlgorithm::Average);
        let hash = op.execute(&img);

        assert_eq!(hash.shape(), &[8]);
        assert_eq!(hash.dtype(), DType::U8);
    }

    #[test]
    fn test_dhash_output_shape() {
        let img = create_test_image(64, 64, 0);
        let op = PerceptualHashOp::new(HashAlgorithm::Difference);
        let hash = op.execute(&img);

        assert_eq!(hash.shape(), &[8]);
        assert_eq!(hash.dtype(), DType::U8);
    }

    #[test]
    fn test_blockhash_output_shape() {
        let img = create_test_image(64, 64, 0);
        let op = PerceptualHashOp::new(HashAlgorithm::Blockhash);
        let hash = op.execute(&img);

        assert_eq!(hash.shape(), &[8]);
        assert_eq!(hash.dtype(), DType::U8);
    }

    #[test]
    fn test_same_image_same_hash() {
        let img = create_test_image(64, 64, 42);
        let op = PerceptualHashOp::phash();

        let hash1 = op.execute(&img);
        let hash2 = op.execute(&img);

        let bytes1 = hash1.as_slice::<u8>();
        let bytes2 = hash2.as_slice::<u8>();

        assert_eq!(bytes1, bytes2, "Same image should produce same hash");
    }

    #[test]
    fn test_different_images_different_hash() {
        // Create two fundamentally different images
        // img1: bright gradient
        let img1 = create_test_image(64, 64, 0);

        // img2: completely different - dark with inverted pattern
        let mut data2 = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64 {
            for x in 0..64 {
                // Create an inverted and shifted pattern
                let r = 255 - ((x * 4) % 256) as u8;
                let g = 255 - ((y * 4) % 256) as u8;
                let b = ((x * y) % 256) as u8;
                data2.push(r);
                data2.push(g);
                data2.push(b);
            }
        }
        let img2 = ViewBuffer::from_vec(data2).reshape(vec![64, 64, 3]);

        let op = PerceptualHashOp::phash();
        let hash1 = op.execute(&img1);
        let hash2 = op.execute(&img2);

        let bytes1 = hash1.as_slice::<u8>();
        let bytes2 = hash2.as_slice::<u8>();

        // At least some bytes should differ
        assert_ne!(
            bytes1, bytes2,
            "Different images should produce different hashes"
        );
    }

    #[test]
    fn test_hash_size_validation() {
        let op = PerceptualHashOp::new(HashAlgorithm::Perceptual).with_hash_size(64);
        let result = op.validate(&[&[64, 64, 3]], &[DType::U8]);
        assert!(result.is_ok());

        // Invalid: not power of 2
        let op_invalid = PerceptualHashOp::new(HashAlgorithm::Perceptual).with_hash_size(100);
        let result = op_invalid.validate(&[&[64, 64, 3]], &[DType::U8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_shape() {
        let op = PerceptualHashOp::new(HashAlgorithm::Perceptual).with_hash_size(64);
        let shape = op.infer_shape(&[&[256, 256, 3]]);
        assert_eq!(shape, vec![8]); // 64 bits = 8 bytes

        let op_large = PerceptualHashOp::new(HashAlgorithm::Perceptual).with_hash_size(256);
        let shape_large = op_large.infer_shape(&[&[256, 256, 3]]);
        assert_eq!(shape_large, vec![32]); // 256 bits = 32 bytes
    }
}
