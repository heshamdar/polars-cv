//! Tests for tiled execution consistency.
//!
//! These tests verify that tiled execution produces identical results
//! to non-tiled execution for all tileable operations.
//!
//! The testing strategy is:
//! 1. Create a large test image
//! 2. Run the operation without tiling
//! 3. Run the operation with tiling
//! 4. Compare the results for exact equality (or approximate for floats)

#![cfg(all(feature = "ndarray_interop", feature = "image_interop"))]

use view_buffer::{with_tile_config, DType, NormalizeMethod, TileConfig, ViewBuffer, ViewExpr};

// ============================================================
// Helper Functions
// ============================================================

/// Create a large RGB image buffer with deterministic content.
///
/// The image content is deterministic based on pixel position,
/// which is important for verifying tiled execution consistency.
fn make_large_rgb(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<u8> = (0..(h * w * 3))
        .map(|i| {
            // Create a pattern that varies spatially
            let pixel_idx = i / 3;
            let channel = i % 3;
            let row = pixel_idx / w;
            let col = pixel_idx % w;
            // Mix row, col, and channel to create varied pattern
            ((row * 7 + col * 13 + channel * 37) % 256) as u8
        })
        .collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w, 3])
}

/// Create a large grayscale image buffer with deterministic content.
fn make_large_gray(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<u8> = (0..(h * w))
        .map(|i| {
            let row = i / w;
            let col = i % w;
            ((row * 7 + col * 13) % 256) as u8
        })
        .collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w, 1])
}

/// Create a large f32 buffer for testing compute operations.
fn make_large_f32(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<f32> = (0..(h * w))
        .map(|i| {
            let row = i / w;
            let col = i % w;
            // Values in range [-1.0, 1.0] for testing relu, clamp, etc.
            ((row * 7 + col * 13) % 200) as f32 / 100.0 - 1.0
        })
        .collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w])
}

/// Assert that two buffers have identical content.
fn assert_buffers_equal(a: &ViewBuffer, b: &ViewBuffer) {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch");
    assert_eq!(a.dtype(), b.dtype(), "Dtype mismatch");

    let a_contig = a.to_contiguous();
    let b_contig = b.to_contiguous();

    let a_slice = a_contig.as_slice::<u8>();
    let b_slice = b_contig.as_slice::<u8>();

    assert_eq!(a_slice.len(), b_slice.len(), "Length mismatch");

    for (i, (&av, &bv)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        assert_eq!(
            av, bv,
            "Mismatch at byte index {i}: expected {av}, got {bv}"
        );
    }
}

/// Assert that two f32 buffers are approximately equal.
fn assert_buffers_approx_equal(a: &ViewBuffer, b: &ViewBuffer, tolerance: f32) {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch");
    assert_eq!(a.dtype(), b.dtype(), "Dtype mismatch");
    assert_eq!(a.dtype(), DType::F32, "Expected f32 dtype");

    let a_contig = a.to_contiguous();
    let b_contig = b.to_contiguous();

    let a_slice = a_contig.as_slice::<f32>();
    let b_slice = b_contig.as_slice::<f32>();

    assert_eq!(a_slice.len(), b_slice.len(), "Length mismatch");

    for (i, (&av, &bv)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        let diff = (av - bv).abs();
        assert!(
            diff <= tolerance,
            "Mismatch at index {i}: {av} vs {bv}, diff {diff} > tolerance {tolerance}"
        );
    }
}

/// Tile config for testing: small tiles to ensure multiple tiles are used.
fn test_tile_config() -> TileConfig {
    TileConfig::with_min_size(128, 64)
}

// ============================================================
// Point-wise Image Operation Consistency Tests
// ============================================================

#[test]
fn test_threshold_tiled_vs_non_tiled() {
    let input = make_large_gray(1024, 1024);

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .threshold(128)
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .threshold(128)
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

#[test]
fn test_grayscale_tiled_vs_non_tiled() {
    let input = make_large_rgb(1024, 1024);

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .grayscale()
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .grayscale()
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

// ============================================================
// Point-wise Compute Operation Consistency Tests
// ============================================================

#[test]
fn test_scale_tiled_vs_non_tiled() {
    let input = make_large_f32(1024, 1024);

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .scale(2.5)
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .scale(2.5)
            .plan()
            .execute()
    });

    assert_buffers_approx_equal(&result_no_tile, &result_tiled, 1e-6);
}

#[test]
fn test_relu_tiled_vs_non_tiled() {
    let input = make_large_f32(1024, 1024);

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone()).relu().plan().execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone()).relu().plan().execute()
    });

    assert_buffers_approx_equal(&result_no_tile, &result_tiled, 1e-6);
}

#[test]
fn test_clamp_tiled_vs_non_tiled() {
    let input = make_large_f32(1024, 1024);

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .clamp(-0.5, 0.5)
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .clamp(-0.5, 0.5)
            .plan()
            .execute()
    });

    assert_buffers_approx_equal(&result_no_tile, &result_tiled, 1e-6);
}

#[test]
fn test_normalize_preset_tiled_vs_non_tiled() {
    // Create RGB f32 image
    let data: Vec<f32> = (0..(1024 * 1024 * 3))
        .map(|i| (i % 256) as f32 / 255.0)
        .collect();
    let input = ViewBuffer::from_vec(data).reshape(vec![1024, 1024, 3]);

    // ImageNet-style preset values
    let mean = vec![0.485, 0.456, 0.406];
    let std = vec![0.229, 0.224, 0.225];

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .normalize(NormalizeMethod::Preset {
                mean: mean.clone(),
                std: std.clone(),
            })
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .normalize(NormalizeMethod::Preset {
                mean: mean.clone(),
                std: std.clone(),
            })
            .plan()
            .execute()
    });

    assert_buffers_approx_equal(&result_no_tile, &result_tiled, 1e-5);
}

// ============================================================
// Neighborhood Operation Consistency Tests
// ============================================================

#[test]
fn test_blur_tiled_vs_non_tiled() {
    let input = make_large_gray(1024, 1024);

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .blur(2.0)
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .blur(2.0)
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

#[test]
fn test_blur_rgb_tiled_vs_non_tiled() {
    let input = make_large_rgb(512, 512);

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .blur(1.5)
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .blur(1.5)
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

// ============================================================
// Strided Input Tests
// ============================================================

#[test]
fn test_tiled_on_flipped_buffer() {
    let input = make_large_gray(1024, 1024);

    // Flip the buffer (creates non-contiguous strided view with negative strides)
    let flipped = input.flip(&[0]);

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(flipped.clone())
            .threshold(128)
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(flipped.clone())
            .threshold(128)
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

#[test]
fn test_tiled_on_cropped_buffer() {
    let input = make_large_rgb(1024, 1024);

    // Crop the buffer (creates non-contiguous strided view)
    let cropped = input.slice(&[100, 100, 0], &[900, 900, 3]);

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(cropped.clone())
            .grayscale()
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(cropped.clone())
            .grayscale()
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

// ============================================================
// Edge Cases
// ============================================================

#[test]
fn test_tiling_skipped_for_small_images() {
    // Small image that shouldn't be tiled
    let input = make_large_gray(64, 64);

    // With tiling enabled but image too small
    let result = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .threshold(128)
            .plan()
            .execute()
    });

    // Just verify it runs without error and produces correct shape
    assert_eq!(result.shape(), &[64, 64, 1]);
}

#[test]
fn test_tiling_with_non_tile_aligned_dimensions() {
    // Image dimensions that don't divide evenly by tile size
    let input = make_large_gray(1000, 1000); // Not divisible by 128

    // Run without tiling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .threshold(100)
            .plan()
            .execute()
    });

    // Run with tiling
    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .threshold(100)
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

#[test]
fn test_halo_handling_at_image_edges() {
    // Test blur near edges where halo extends beyond image bounds
    let input = make_large_gray(512, 512);

    // Large sigma means larger halo, tests edge handling
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .blur(5.0)
            .plan()
            .execute()
    });

    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .blur(5.0)
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

// ============================================================
// Pipeline Tests
// ============================================================

#[test]
fn test_chained_ops_tiled_consistency() {
    let input = make_large_rgb(1024, 1024);

    // Chain of operations: grayscale -> blur -> threshold
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .grayscale()
            .blur(1.5)
            .threshold(128)
            .plan()
            .execute()
    });

    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .grayscale()
            .blur(1.5)
            .threshold(128)
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

#[test]
fn test_mixed_tileable_and_global_ops() {
    // This tests a pipeline with both tileable and global operations
    // The global operations (like resize) should break tiling boundaries
    let input = make_large_rgb(1024, 1024);

    // Grayscale (tileable) followed by resize (global)
    let result_no_tile = with_tile_config(None, || {
        ViewExpr::new_source(input.clone())
            .grayscale()
            .resize(512, 512, view_buffer::FilterType::Lanczos3)
            .plan()
            .execute()
    });

    let result_tiled = with_tile_config(Some(test_tile_config()), || {
        ViewExpr::new_source(input.clone())
            .grayscale()
            .resize(512, 512, view_buffer::FilterType::Lanczos3)
            .plan()
            .execute()
    });

    assert_buffers_equal(&result_no_tile, &result_tiled);
}

// ============================================================
// Configuration Tests
// ============================================================

#[test]
fn test_with_tile_config_restores_previous() {
    // Get initial state
    let initial = view_buffer::get_tile_config();

    // Nest configurations
    let result = with_tile_config(Some(TileConfig::new(256)), || {
        let outer = view_buffer::get_tile_config();
        assert!(outer.is_some());
        assert_eq!(outer.unwrap().tile_size, 256);

        with_tile_config(Some(TileConfig::new(128)), || {
            let inner = view_buffer::get_tile_config();
            assert!(inner.is_some());
            assert_eq!(inner.unwrap().tile_size, 128);
            42
        })
    });

    assert_eq!(result, 42);

    // Should be restored to initial
    assert_eq!(view_buffer::get_tile_config(), initial);
}

#[test]
fn test_tile_policy_for_operations() {
    use view_buffer::ops::traits::Op;
    use view_buffer::{ComputeOp, ImageOp, ImageOpKind};

    // Point-wise compute ops
    assert!(ComputeOp::Scale(2.0).tile_policy().is_tileable());
    assert!(ComputeOp::Relu.tile_policy().is_tileable());
    assert!(ComputeOp::Clamp { min: 0.0, max: 1.0 }
        .tile_policy()
        .is_tileable());

    // Global compute ops
    assert!(!ComputeOp::Normalize(NormalizeMethod::MinMax)
        .tile_policy()
        .is_tileable());
    assert!(!ComputeOp::Normalize(NormalizeMethod::ZScore)
        .tile_policy()
        .is_tileable());

    // Preset normalize is tileable
    assert!(ComputeOp::Normalize(NormalizeMethod::Preset {
        mean: vec![0.5],
        std: vec![0.5],
    })
    .tile_policy()
    .is_tileable());

    // Point-wise image ops
    assert!(ImageOp {
        kind: ImageOpKind::Threshold(128)
    }
    .tile_policy()
    .is_tileable());
    assert!(ImageOp {
        kind: ImageOpKind::Grayscale
    }
    .tile_policy()
    .is_tileable());

    // Neighborhood image ops (tileable with halo)
    let blur_policy = ImageOp {
        kind: ImageOpKind::Blur { sigma: 2.0 },
    }
    .tile_policy();
    assert!(blur_policy.is_tileable());
    assert_eq!(blur_policy.halo(), 6); // 3 * sigma = 6

    // Global image ops
    assert!(!ImageOp {
        kind: ImageOpKind::Resize {
            width: 100,
            height: 100,
            filter: view_buffer::FilterType::Nearest
        }
    }
    .tile_policy()
    .is_tileable());
}
