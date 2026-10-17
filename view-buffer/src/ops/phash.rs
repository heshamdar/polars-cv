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
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;

use crate::mode::{Exec, Mode};
use polars_cv_macros::{Ops, Resolve};

/// Perceptual hash algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

crate::naming::named_variants!(HashAlgorithm: "Perceptual hash algorithm selection.\n\nDifferent algorithms trade off speed vs robustness to transformations:\n- AVERAGE: Fastest, least robust. Good for exact/near-exact matches.\n- DIFFERENCE: Gradient-based, good balance of speed and robustness.\n- PERCEPTUAL: DCT-based, most robust to resize/compression. Recommended default.\n- BLOCKHASH: Block-based, good resistance to cropping." {
    "average" => Average,
    "difference" => Difference,
    "perceptual" => Perceptual,
    "blockhash" => Blockhash,
});

/// Compute a perceptual hash fingerprint.
///
/// Example:
///     >>> Pipeline().source("image_bytes").perceptual_hash()
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
#[op(name = "perceptual_hash", sample = {"algorithm": "average", "hash_size": 64})]
pub struct PerceptualHashOp<M: Mode = Exec> {
    /// "perceptual" (pHash), "average" (aHash), "difference" (dHash).
    #[param(default = "perceptual")]
    pub algorithm: M::L<HashAlgorithm>,
    /// Number of bits in the hash (must be power of 2). It fixes the output
    /// vector length, so it is literal-only.
    #[param(default = 64)]
    pub hash_size: M::L<u32>,
}

impl<M: Mode> PerceptualHashOp<M> {
    /// A hash needs at least one bit.
    pub fn check(&self) -> Result<(), String> {
        if M::lit(&self.hash_size) == 0 {
            return Err("perceptual_hash: hash_size must be a positive integer".into());
        }
        Ok(())
    }

    /// Output is always a 1D array of hash bytes.
    pub fn shape(&self) -> OpShape {
        OpShape::Fixed(vec![Sym::Known((M::lit(&self.hash_size) / 8) as usize)])
    }
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

        // The hash grid's side: 64 bits is an 8x8 grid.
        let hash_width = (self.hash_size as f64).sqrt() as u32;
        let n_bytes = self.hash_bytes();

        let hash_bytes = match self.algorithm {
            HashAlgorithm::Perceptual => phash(buffer, hash_width as usize, n_bytes),
            alg => {
                let dynamic_image = ImageAdapter::to_dynamic_image(buffer)
                    .expect("Failed to convert buffer to image for hashing");
                // image_hasher: Mean is aHash, Gradient dHash.
                let hash_alg = match alg {
                    HashAlgorithm::Average => HashAlg::Mean,
                    HashAlgorithm::Difference => HashAlg::Gradient,
                    HashAlgorithm::Blockhash => HashAlg::Blockhash,
                    HashAlgorithm::Perceptual => unreachable!(),
                };
                HasherConfig::new()
                    .hash_alg(hash_alg)
                    .hash_size(hash_width, hash_width)
                    .to_hasher()
                    .hash_image(&dynamic_image)
                    .as_bytes()
                    .to_vec()
            }
        };

        let len = hash_bytes.len();
        ViewBuffer::from_vec(hash_bytes).reshape(vec![len])
    }

    /// Fallback when perceptual_hash feature is not enabled.
    #[cfg(not(feature = "perceptual_hash"))]
    pub fn execute(&self, _buffer: &ViewBuffer) -> ViewBuffer {
        panic!("PerceptualHashOp requires the 'perceptual_hash' feature to be enabled");
    }
}

/// Standard pHash, the steps of `imagehash.phash`: BT.601 grayscale, a
/// Lanczos3 resize to a `4 * side` square, a 2-D DCT-II, its `side x side`
/// lowest frequencies thresholded at their median, and the bits row-major,
/// most significant first, in `n_bytes` bytes. The grayscale and the resize
/// are the engine's own (`grayscale` and `resize` ops), so the hash sees the
/// pixels a pipeline would.
///
/// It replaced image_hasher's `Mean` + `preproc_dct`, which thresholds a
/// 16x16-input DCT at its *mean*: the DC coefficient dominates that mean, so
/// most bits agreed between unrelated images (distinct COCO images ~18 bits
/// apart instead of ~32).
#[cfg(feature = "perceptual_hash")]
fn phash(buffer: &ViewBuffer, side: usize, n_bytes: usize) -> Vec<u8> {
    use crate::execution::runner::{grayscale_strided, resize_strided};
    use crate::ops::FilterType;

    let n = 4 * side;
    let gray = if buffer.shape().len() == 2 {
        let (h, w) = (buffer.shape()[0], buffer.shape()[1]);
        buffer.to_contiguous().reshape(vec![h, w, 1])
    } else {
        grayscale_strided(buffer.clone())
    };
    let small = resize_strided(gray, n as u32, n as u32, FilterType::Lanczos3).to_contiguous();
    let pixels: Vec<f64> = small
        .as_slice::<u8>()
        .iter()
        .map(|&v| f64::from(v))
        .collect();

    // DCT-II along axis 0 then axis 1, keeping the `side` lowest
    // frequencies of each. The constant factor is dropped: it scales every
    // coefficient alike, so the median threshold is unchanged.
    let basis: Vec<f64> = (0..side * n)
        .map(|i| {
            let (k, x) = (i / n, i % n);
            (std::f64::consts::PI * k as f64 * (2 * x + 1) as f64 / (2 * n) as f64).cos()
        })
        .collect();
    let mut cols = vec![0.0f64; side * n]; // [k][x]
    for k in 0..side {
        for y in 0..n {
            let b = basis[k * n + y];
            for x in 0..n {
                cols[k * n + x] += b * pixels[y * n + x];
            }
        }
    }
    let mut coeffs = vec![0.0f64; side * side]; // [k][l], row-major
    for k in 0..side {
        for l in 0..side {
            coeffs[k * side + l] = (0..n).map(|x| basis[l * n + x] * cols[k * n + x]).sum();
        }
    }

    // A coefficient that is mathematically 0 (e.g. every vertical frequency
    // of an image constant down its columns) comes out of the direct sums as
    // +-1e-13 rounding noise, and its sign would decide its bit; scipy's FFT
    // DCT, which imagehash uses, returns those as exact zeros. Snap them, so
    // they compare as not above the median.
    let scale = coeffs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for v in &mut coeffs {
        if v.abs() <= scale * 1e-9 {
            *v = 0.0;
        }
    }
    let mut sorted = coeffs.clone();
    sorted.sort_by(f64::total_cmp);
    let m = sorted.len();
    let median = if m % 2 == 1 {
        sorted[m / 2]
    } else {
        (sorted[m / 2 - 1] + sorted[m / 2]) / 2.0
    };

    let mut out = vec![0u8; n_bytes];
    for (i, &v) in coeffs.iter().enumerate() {
        if v > median {
            out[i / 8] |= 0x80 >> (i % 8);
        }
    }
    out
}

impl Default for PerceptualHashOp {
    fn default() -> Self {
        Self::phash()
    }
}

impl<M: Mode> Op for PerceptualHashOp<M> {
    fn name(&self) -> &'static str {
        "PerceptualHash"
    }

    fn shape(&self) -> OpShape {
        PerceptualHashOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        // Hashing requires the full image to be loaded
        MemoryEffect::RequiresContiguous
    }

    fn identity_rule(&self) -> IdentityRule {
        // Computes / combines / reduces — never a removable no-op.
        IdentityRule::Never
    }

    fn is_spatial_window(&self) -> bool {
        false // A perceptual hash is not an H/W crop window.
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        // The hash is a global fingerprint of the whole image (downscale + DCT).
        SpatialDependency::Global
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
        input_shapes: &[&[crate::ops::Dim]],
        _input_dtypes: &[crate::PlannedDType],
    ) -> Result<(), ValidationError> {
        // Validate input is an image-like shape
        let shape = input_shapes[0];
        let show = crate::ops::shape_rule::show_dims(shape);
        if shape.len() < 2 || shape.len() > 3 {
            return Err(ValidationError::InvalidParameter {
                param: "input_shape".to_string(),
                reason: format!("Expected 2D or 3D image shape [H, W] or [H, W, C], got {show}"),
            });
        }

        // An unknown channel count is checked per row.
        if shape.len() == 3 && shape[2].known().is_some_and(|c| c > 4) {
            return Err(ValidationError::InvalidParameter {
                param: "input_shape".to_string(),
                reason: format!("at most 4 channels can be hashed as an image, got {show}"),
            });
        }

        // Validate hash_size is a power of 2 and reasonable
        let hash_size = M::lit(&self.hash_size);
        if !hash_size.is_power_of_two() {
            return Err(ValidationError::InvalidParameter {
                param: "hash_size".to_string(),
                reason: format!("hash_size must be a power of 2, got {hash_size}"),
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

    /// A flat image has energy only at DC, so only the first bit (the DC
    /// coefficient, row-major first, most significant) is above the median.
    #[test]
    fn phash_of_a_flat_image_sets_only_the_dc_bit() {
        let buf = ViewBuffer::from_vec(vec![90u8; 50 * 40 * 3]).reshape(vec![50, 40, 3]);
        let out = PerceptualHashOp::phash().execute(&buf);
        let mut expected = vec![0u8; 8];
        expected[0] = 0x80;
        assert_eq!(out.as_slice::<u8>(), expected.as_slice());
    }

    /// Two unrelated patterns differ in about half their bits, the property
    /// the mean-threshold version lost.
    #[test]
    fn phash_separates_unrelated_patterns() {
        let a = create_test_image(96, 64, 3);
        let blocks: Vec<u8> = (0..64 * 96)
            .flat_map(|i| {
                let (y, x) = (i / 96, i % 96);
                let v = if (y / 16 + x / 24) % 2 == 0 {
                    220u8
                } else {
                    30
                };
                [v, v, v]
            })
            .collect();
        let b = ViewBuffer::from_vec(blocks).reshape(vec![64, 96, 3]);
        let ha = PerceptualHashOp::phash().execute(&a);
        let hb = PerceptualHashOp::phash().execute(&b);
        let bits: u32 = ha
            .as_slice::<u8>()
            .iter()
            .zip(hb.as_slice::<u8>())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();
        assert!(bits >= 16, "{bits} differing bits");
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
        let result = crate::ops::validation::validate_concrete(&op, &[&[64, 64, 3]], &[DType::U8]);
        assert!(result.is_ok());

        // Invalid: not power of 2
        let op_invalid = PerceptualHashOp::new(HashAlgorithm::Perceptual).with_hash_size(100);
        let result =
            crate::ops::validation::validate_concrete(&op_invalid, &[&[64, 64, 3]], &[DType::U8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape() {
        let op = PerceptualHashOp::new(HashAlgorithm::Perceptual).with_hash_size(64);
        let shape = op.shape().concrete(&[&[256, 256, 3]]);
        assert_eq!(shape, vec![8]); // 64 bits = 8 bytes

        let op_large = PerceptualHashOp::new(HashAlgorithm::Perceptual).with_hash_size(256);
        let shape_large = op_large.shape().concrete(&[&[256, 256, 3]]);
        assert_eq!(shape_large, vec![32]); // 256 bits = 32 bytes
    }
}
