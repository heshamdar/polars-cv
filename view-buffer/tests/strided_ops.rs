//! Tests for strided operation support.
//!
//! These tests verify that operations work correctly on non-contiguous buffers
//! (e.g., after flip, crop, transpose) without requiring materialization.

#![cfg(all(feature = "ndarray_interop", feature = "image_interop"))]

use view_buffer::{FilterType, NormalizeMethod, ViewBuffer, ViewExpr};

/// Helper to create a test RGB image buffer.
fn make_rgb_image(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<u8> = (0..(h * w * 3)).map(|i| ((i * 7) % 256) as u8).collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w, 3])
}

/// Helper to create a test grayscale image buffer.
fn make_gray_image(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<u8> = (0..(h * w)).map(|i| ((i * 7) % 256) as u8).collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w, 1])
}

/// Helper to create a test f32 buffer.
fn make_f32_buffer(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<f32> = (0..(h * w)).map(|i| i as f32).collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w])
}

/// Assert that two buffers share the same underlying storage (zero-copy).
fn assert_zero_copy(a: &ViewBuffer, b: &ViewBuffer) {
    assert_eq!(
        a.storage_id(),
        b.storage_id(),
        "Expected zero-copy view, but storage differs"
    );
}

// ============================================================
// Flip + Grayscale Tests
// ============================================================

#[test]
fn test_flip_then_grayscale() {
    let buf = make_rgb_image(100, 100);

    // Flip is a zero-copy view operation
    let flipped = buf.flip(&[0]); // Vertical flip
    assert_zero_copy(&buf, &flipped);

    // Grayscale should work on the flipped (strided) buffer
    let gray = ViewExpr::new_source(flipped).grayscale().plan().execute();

    assert_eq!(gray.shape(), &[100, 100, 1]);
    assert_eq!(gray.dtype(), view_buffer::DType::U8);
}

#[test]
fn test_horizontal_flip_then_grayscale() {
    let buf = make_rgb_image(50, 80);

    // Horizontal flip
    let flipped = buf.flip(&[1]);
    assert_zero_copy(&buf, &flipped);

    let gray = ViewExpr::new_source(flipped).grayscale().plan().execute();

    assert_eq!(gray.shape(), &[50, 80, 1]);
}

#[test]
fn test_double_flip_then_grayscale() {
    let buf = make_rgb_image(64, 64);

    // Double flip (both axes)
    let flipped = buf.flip(&[0, 1]);
    assert_zero_copy(&buf, &flipped);

    let gray = ViewExpr::new_source(flipped).grayscale().plan().execute();

    assert_eq!(gray.shape(), &[64, 64, 1]);
}

// ============================================================
// Crop + Grayscale Tests
// ============================================================

#[test]
fn test_crop_then_grayscale() {
    let buf = make_rgb_image(100, 100);

    // Crop to 50x50 region
    let cropped = buf.slice(&[25, 25, 0], &[75, 75, 3]);
    assert_zero_copy(&buf, &cropped);
    assert_eq!(cropped.shape(), &[50, 50, 3]);

    // Grayscale on cropped buffer
    let gray = ViewExpr::new_source(cropped).grayscale().plan().execute();

    assert_eq!(gray.shape(), &[50, 50, 1]);
}

// ============================================================
// Flip + Resize Tests
// ============================================================

#[test]
fn test_flip_then_resize() {
    let buf = make_rgb_image(100, 100);

    let flipped = buf.flip(&[0]);
    assert_zero_copy(&buf, &flipped);

    // Resize should work on flipped buffer
    let resized = ViewExpr::new_source(flipped)
        .resize(64, 64, FilterType::Lanczos3)
        .plan()
        .execute();

    assert_eq!(resized.shape(), &[64, 64, 3]);
}

#[test]
fn test_crop_then_resize() {
    let buf = make_rgb_image(200, 200);

    let cropped = buf.slice(&[50, 50, 0], &[150, 150, 3]);
    assert_zero_copy(&buf, &cropped);
    assert_eq!(cropped.shape(), &[100, 100, 3]);

    // Resize on cropped buffer
    let resized = ViewExpr::new_source(cropped)
        .resize(64, 64, FilterType::Triangle)
        .plan()
        .execute();

    assert_eq!(resized.shape(), &[64, 64, 3]);
}

#[test]
fn test_flip_resize_grayscale() {
    let buf = make_rgb_image(100, 100);

    // Chain: flip -> resize -> grayscale
    let flipped = buf.flip(&[1]);

    let result = ViewExpr::new_source(flipped)
        .resize(64, 64, FilterType::CatmullRom)
        .grayscale()
        .plan()
        .execute();

    assert_eq!(result.shape(), &[64, 64, 1]);
}

// ============================================================
// Normalize on Strided Buffers
// ============================================================

#[test]
fn test_normalize_on_transposed_buffer() {
    let buf = make_f32_buffer(100, 100);

    // Transpose makes it non-contiguous
    let transposed = buf.permute(&[1, 0]);
    assert_zero_copy(&buf, &transposed);
    assert_eq!(transposed.shape(), &[100, 100]);

    // Normalize should work on transposed buffer via ndarray
    let normalized = ViewExpr::new_source(transposed)
        .normalize(NormalizeMethod::MinMax)
        .plan()
        .execute();

    assert_eq!(normalized.shape(), &[100, 100]);

    // Check normalization worked
    let slice = normalized.as_slice::<f32>();
    let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!((min - 0.0).abs() < 1e-6, "Min should be ~0, got {min}");
    assert!((max - 1.0).abs() < 1e-6, "Max should be ~1, got {max}");
}

#[test]
fn test_normalize_on_flipped_buffer() {
    let buf = make_f32_buffer(50, 50);

    let flipped = buf.flip(&[0]);
    assert_zero_copy(&buf, &flipped);

    let normalized = ViewExpr::new_source(flipped)
        .normalize(NormalizeMethod::ZScore)
        .plan()
        .execute();

    assert_eq!(normalized.shape(), &[50, 50]);

    // Check z-score normalization (mean ~= 0, std ~= 1)
    let slice = normalized.as_slice::<f32>();
    let n = slice.len() as f32;
    let mean: f32 = slice.iter().sum::<f32>() / n;
    assert!(mean.abs() < 1e-4, "Mean should be ~0, got {mean}");
}

// ============================================================
// Scalar Ops on Strided Buffers
// ============================================================

#[test]
fn test_scale_on_transposed_buffer() {
    let buf = make_f32_buffer(32, 32);

    let transposed = buf.permute(&[1, 0]);
    assert_zero_copy(&buf, &transposed);

    let scaled = ViewExpr::new_source(transposed).scale(2.0).plan().execute();

    assert_eq!(scaled.shape(), &[32, 32]);
}

#[test]
fn test_relu_on_flipped_buffer() {
    // Create buffer with some negative values
    let data: Vec<f32> = (-50..50).map(|i| i as f32).collect();
    let buf = ViewBuffer::from_vec(data).reshape(vec![10, 10]);

    let flipped = buf.flip(&[0]);
    assert_zero_copy(&buf, &flipped);

    let result = ViewExpr::new_source(flipped).relu().plan().execute();

    assert_eq!(result.shape(), &[10, 10]);

    // Check ReLU worked - no negative values
    let slice = result.as_slice::<f32>();
    let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        min >= 0.0,
        "ReLU should have no negative values, got min={min}"
    );
}

// ============================================================
// Grayscale Image Resize Tests
// ============================================================

#[test]
fn test_grayscale_resize() {
    let buf = make_gray_image(100, 100);

    let resized = ViewExpr::new_source(buf)
        .resize(50, 50, FilterType::Nearest)
        .plan()
        .execute();

    assert_eq!(resized.shape(), &[50, 50, 1]);
}

#[test]
fn test_flip_grayscale_resize() {
    let buf = make_gray_image(80, 80);

    let flipped = buf.flip(&[0]);

    let result = ViewExpr::new_source(flipped)
        .resize(40, 40, FilterType::Triangle)
        .plan()
        .execute();

    assert_eq!(result.shape(), &[40, 40, 1]);
}

// ============================================================
// Strided Output Tests (into_polars_buffer_strided)
// ============================================================

#[cfg(feature = "polars_interop")]
mod strided_output_tests {
    use super::*;
    use view_buffer::SlicePolicy;

    #[test]
    fn test_strided_output_contiguous() {
        let buf = make_rgb_image(50, 50);

        // Contiguous buffer should return matching strides
        let (data, shape, strides, offset, dtype) = buf.into_polars_buffer_strided();

        assert_eq!(shape, vec![50, 50, 3]);
        // Contiguous strides for [50, 50, 3]: [150, 3, 1]
        assert_eq!(strides, vec![150, 3, 1]);
        assert_eq!(offset, 0);
        assert_eq!(dtype, view_buffer::DType::U8);
        assert_eq!(data.len(), 50 * 50 * 3);
    }

    #[test]
    fn test_strided_output_after_flip_copies_when_shared() {
        let buf = make_rgb_image(50, 50);
        let original_storage_id = buf.storage_id();

        // Flip creates non-contiguous view but SHARES the underlying Arc
        let flipped = buf.flip(&[0]);
        assert_eq!(flipped.storage_id(), original_storage_id);

        // Since Arc is shared (refcount > 1), AlwaysZeroCopy won't work
        // and it will fall back to materializing (which makes it contiguous)
        let (data, shape, strides, offset, dtype) =
            flipped.into_polars_buffer_strided_with_policy(SlicePolicy::AlwaysZeroCopy);

        assert_eq!(shape, vec![50, 50, 3]);
        // Due to Arc sharing, it materializes to contiguous strides
        assert_eq!(strides, vec![150, 3, 1]);
        assert_eq!(offset, 0);
        assert_eq!(dtype, view_buffer::DType::U8);
        assert_eq!(data.len(), 50 * 50 * 3);
    }

    #[test]
    fn test_strided_output_after_flip_sole_owner() {
        // Create buffer and immediately flip (sole owner)
        let flipped = make_rgb_image(50, 50).flip(&[0]);

        // Now flipped is the sole owner - strided output should work
        let (data, shape, strides, offset, dtype) =
            flipped.into_polars_buffer_strided_with_policy(SlicePolicy::AlwaysZeroCopy);

        assert_eq!(shape, vec![50, 50, 3]);
        // After vertical flip with sole ownership, should have negative row stride
        assert_eq!(strides[0], -150);
        assert_eq!(strides[1], 3);
        assert_eq!(strides[2], 1);
        // Offset should point to last row
        assert_eq!(offset, (50 - 1) * 150); // 49 * 150 = 7350
        assert_eq!(dtype, view_buffer::DType::U8);
        assert_eq!(data.len(), 50 * 50 * 3);
    }

    #[test]
    fn test_strided_output_after_crop_copies_when_shared() {
        let buf = make_rgb_image(100, 100);
        let original_storage_id = buf.storage_id();

        // Crop creates non-contiguous view but SHARES the Arc
        let cropped = buf.slice(&[10, 10, 0], &[60, 60, 3]);
        assert_eq!(cropped.storage_id(), original_storage_id);
        assert_eq!(cropped.shape(), &[50, 50, 3]);

        // Since Arc is shared, it will materialize to contiguous
        let (data, shape, strides, offset, dtype) =
            cropped.into_polars_buffer_strided_with_policy(SlicePolicy::AlwaysZeroCopy);

        assert_eq!(shape, vec![50, 50, 3]);
        // Materialized to contiguous
        assert_eq!(strides, vec![150, 3, 1]);
        assert_eq!(offset, 0);
        assert_eq!(dtype, view_buffer::DType::U8);
        assert_eq!(data.len(), 50 * 50 * 3);
    }

    #[test]
    fn test_strided_output_after_crop_sole_owner() {
        // Create and immediately crop (sole owner)
        let cropped = make_rgb_image(100, 100).slice(&[10, 10, 0], &[60, 60, 3]);
        assert_eq!(cropped.shape(), &[50, 50, 3]);

        // Now cropped is sole owner - should get strided output
        let (data, shape, strides, offset, dtype) =
            cropped.into_polars_buffer_strided_with_policy(SlicePolicy::AlwaysZeroCopy);

        assert_eq!(shape, vec![50, 50, 3]);
        // Original strides preserved (from 100x100 image): [300, 3, 1]
        assert_eq!(strides, vec![300, 3, 1]);
        assert_eq!(dtype, view_buffer::DType::U8);
        // Offset should be 10*300 + 10*3 = 3030
        assert_eq!(offset, 10 * 300 + 10 * 3);
        // Full original buffer returned
        assert_eq!(data.len(), 100 * 100 * 3);
    }

    #[test]
    fn test_strided_output_copy_policy() {
        // Sole owner crop
        let cropped = make_rgb_image(100, 100).slice(&[40, 40, 0], &[60, 60, 3]);
        assert_eq!(cropped.shape(), &[20, 20, 3]);

        // With AlwaysCopy policy, should materialize to contiguous
        let (data, shape, strides, offset, _dtype) =
            cropped.into_polars_buffer_strided_with_policy(SlicePolicy::AlwaysCopy);

        assert_eq!(shape, vec![20, 20, 3]);
        // Should be materialized to contiguous strides
        assert_eq!(strides, vec![60, 3, 1]); // 20*3, 3, 1
        assert_eq!(offset, 0);
        // Only the cropped data
        assert_eq!(data.len(), 20 * 20 * 3);
    }

    #[test]
    fn test_can_zero_copy_strided() {
        let buf = make_rgb_image(50, 50);

        // Contiguous buffer with sole ownership can zero-copy
        assert!(buf.can_zero_copy_strided());
    }

    #[test]
    fn test_can_zero_copy_strided_shared() {
        let buf = make_rgb_image(50, 50);

        // After flip, the Arc is shared so can_zero_copy_strided returns false
        let flipped = buf.flip(&[0]);
        assert!(!flipped.can_zero_copy_strided());

        // Original still exists, so refcount > 1
        drop(buf);
        // Now flipped could be sole owner, but we already checked
    }
}
