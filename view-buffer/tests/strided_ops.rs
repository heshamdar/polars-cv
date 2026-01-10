//! Tests for strided operation support.
//!
//! These tests verify that operations work correctly on non-contiguous buffers
//! (e.g., after flip, crop, transpose) without requiring materialization.

#![cfg(all(feature = "ndarray_interop", feature = "image_interop"))]

use view_buffer::{FilterType, NormalizeMethod, ViewBuffer, ViewExpr};

/// Helper to create a test RGB image buffer.
fn make_rgb_image(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<u8> = (0..(h * w * 3))
        .map(|i| ((i * 7) % 256) as u8)
        .collect();
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
    let gray = ViewExpr::new_source(flipped)
        .grayscale()
        .plan()
        .execute();
    
    assert_eq!(gray.shape(), &[100, 100, 1]);
    assert_eq!(gray.dtype(), view_buffer::DType::U8);
}

#[test]
fn test_horizontal_flip_then_grayscale() {
    let buf = make_rgb_image(50, 80);
    
    // Horizontal flip
    let flipped = buf.flip(&[1]);
    assert_zero_copy(&buf, &flipped);
    
    let gray = ViewExpr::new_source(flipped)
        .grayscale()
        .plan()
        .execute();
    
    assert_eq!(gray.shape(), &[50, 80, 1]);
}

#[test]
fn test_double_flip_then_grayscale() {
    let buf = make_rgb_image(64, 64);
    
    // Double flip (both axes)
    let flipped = buf.flip(&[0, 1]);
    assert_zero_copy(&buf, &flipped);
    
    let gray = ViewExpr::new_source(flipped)
        .grayscale()
        .plan()
        .execute();
    
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
    let gray = ViewExpr::new_source(cropped)
        .grayscale()
        .plan()
        .execute();
    
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
    assert!((min - 0.0).abs() < 1e-6, "Min should be ~0, got {}", min);
    assert!((max - 1.0).abs() < 1e-6, "Max should be ~1, got {}", max);
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
    assert!(mean.abs() < 1e-4, "Mean should be ~0, got {}", mean);
}

// ============================================================
// Scalar Ops on Strided Buffers
// ============================================================

#[test]
fn test_scale_on_transposed_buffer() {
    let buf = make_f32_buffer(32, 32);
    
    let transposed = buf.permute(&[1, 0]);
    assert_zero_copy(&buf, &transposed);
    
    let scaled = ViewExpr::new_source(transposed)
        .scale(2.0)
        .plan()
        .execute();
    
    assert_eq!(scaled.shape(), &[32, 32]);
}

#[test]
fn test_relu_on_flipped_buffer() {
    // Create buffer with some negative values
    let data: Vec<f32> = (-50..50).map(|i| i as f32).collect();
    let buf = ViewBuffer::from_vec(data).reshape(vec![10, 10]);
    
    let flipped = buf.flip(&[0]);
    assert_zero_copy(&buf, &flipped);
    
    let result = ViewExpr::new_source(flipped)
        .relu()
        .plan()
        .execute();
    
    assert_eq!(result.shape(), &[10, 10]);
    
    // Check ReLU worked - no negative values
    let slice = result.as_slice::<f32>();
    let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(min >= 0.0, "ReLU should have no negative values, got min={}", min);
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
