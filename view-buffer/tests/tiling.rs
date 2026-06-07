//! Tests for segment-level tiled execution consistency.
//!
//! These tests verify that segment-level tiled execution produces identical
//! results to full-image execution for all operations, and that the
//! `ExecutionStrategy` API works correctly.

#![cfg(all(feature = "ndarray_interop", feature = "image_interop"))]

use view_buffer::{
    with_execution_strategy, DType, ExecutionStrategy, NormalizeMethod, ViewBuffer, ViewExpr,
    get_execution_strategy,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_large_rgb(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<u8> = (0..(h * w * 3))
        .map(|i| {
            let pixel_idx = i / 3;
            let channel = i % 3;
            let row = pixel_idx / w;
            let col = pixel_idx % w;
            ((row * 7 + col * 13 + channel * 37) % 256) as u8
        })
        .collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w, 3])
}

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

fn make_large_f32(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<f32> = (0..(h * w))
        .map(|i| {
            let row = i / w;
            let col = i % w;
            ((row * 7 + col * 13) % 200) as f32 / 100.0 - 1.0
        })
        .collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w])
}

fn assert_buffers_equal(a: &ViewBuffer, b: &ViewBuffer) {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch");
    assert_eq!(a.dtype(), b.dtype(), "Dtype mismatch");
    let a_c = a.to_contiguous();
    let b_c = b.to_contiguous();
    let a_sl = a_c.as_slice::<u8>();
    let b_sl = b_c.as_slice::<u8>();
    assert_eq!(a_sl.len(), b_sl.len(), "Length mismatch");
    for (i, (&av, &bv)) in a_sl.iter().zip(b_sl.iter()).enumerate() {
        assert_eq!(av, bv, "Mismatch at byte {i}: expected {av}, got {bv}");
    }
}

fn assert_buffers_approx_equal(a: &ViewBuffer, b: &ViewBuffer, tol: f32) {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch");
    assert_eq!(a.dtype(), b.dtype(), "Dtype mismatch");
    assert_eq!(a.dtype(), DType::F32);
    let a_c = a.to_contiguous();
    let b_c = b.to_contiguous();
    let a_sl = a_c.as_slice::<f32>();
    let b_sl = b_c.as_slice::<f32>();
    for (i, (&av, &bv)) in a_sl.iter().zip(b_sl.iter()).enumerate() {
        let diff = (av - bv).abs();
        assert!(diff <= tol, "Mismatch at {i}: {av} vs {bv}, diff {diff} > {tol}");
    }
}

/// Strategy that always tiles with 128-pixel tiles (threshold_bytes = 0 → always active).
fn always_tiled() -> ExecutionStrategy {
    ExecutionStrategy::Tiled {
        tile_size: 128,
        threshold_bytes: 0,
    }
}

// ── Point-wise image ops ───────────────────────────────────────────────────────

#[test]
fn test_threshold_tiled_vs_full() {
    let input = make_large_gray(1024, 1024);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).threshold(128.0).plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone()).threshold(128.0).plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

#[test]
fn test_grayscale_tiled_vs_full() {
    let input = make_large_rgb(1024, 1024);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).grayscale().plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone()).grayscale().plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

// ── Point-wise compute ops ─────────────────────────────────────────────────────

#[test]
fn test_scale_tiled_vs_full() {
    let input = make_large_f32(1024, 1024);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).scale(2.5).plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone()).scale(2.5).plan().execute()
    });
    assert_buffers_approx_equal(&full, &tiled, 1e-6);
}

#[test]
fn test_relu_tiled_vs_full() {
    let input = make_large_f32(1024, 1024);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).relu().plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone()).relu().plan().execute()
    });
    assert_buffers_approx_equal(&full, &tiled, 1e-6);
}

#[test]
fn test_clamp_tiled_vs_full() {
    let input = make_large_f32(1024, 1024);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).clamp(-0.5, 0.5).plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone()).clamp(-0.5, 0.5).plan().execute()
    });
    assert_buffers_approx_equal(&full, &tiled, 1e-6);
}

#[test]
fn test_normalize_preset_tiled_vs_full() {
    let data: Vec<f32> = (0..(1024 * 1024 * 3))
        .map(|i| (i % 256) as f32 / 255.0)
        .collect();
    let input = ViewBuffer::from_vec(data).reshape(vec![1024, 1024, 3]);
    let mean = vec![0.485, 0.456, 0.406];
    let std = vec![0.229, 0.224, 0.225];
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone())
            .normalize(NormalizeMethod::Preset { mean: mean.clone(), std: std.clone() })
            .plan()
            .execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone())
            .normalize(NormalizeMethod::Preset { mean: mean.clone(), std: std.clone() })
            .plan()
            .execute()
    });
    assert_buffers_approx_equal(&full, &tiled, 1e-5);
}

// ── Neighborhood ops ──────────────────────────────────────────────────────────

#[test]
fn test_blur_tiled_vs_full() {
    let input = make_large_gray(1024, 1024);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).blur(2.0).plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone()).blur(2.0).plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

#[test]
fn test_blur_rgb_tiled_vs_full() {
    let input = make_large_rgb(512, 512);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).blur(1.5).plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone()).blur(1.5).plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

// ── Strided input ─────────────────────────────────────────────────────────────

#[test]
fn test_tiled_on_flipped_buffer() {
    let input = make_large_gray(1024, 1024);
    let flipped = input.flip(&[0]);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(flipped.clone()).threshold(128.0).plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(flipped.clone()).threshold(128.0).plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

#[test]
fn test_tiled_on_cropped_buffer() {
    let input = make_large_rgb(1024, 1024);
    let cropped = input.slice(&[100, 100, 0], &[900, 900, 3]);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(cropped.clone()).grayscale().plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(cropped.clone()).grayscale().plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[test]
fn test_tiling_skipped_below_threshold() {
    let input = make_large_gray(64, 64); // 4096 bytes
    // Use a threshold just above this image's size — tiling should not activate.
    let result = with_execution_strategy(
        ExecutionStrategy::Tiled { tile_size: 128, threshold_bytes: 4097 },
        || {
            ViewExpr::new_source(input.clone()).threshold(128.0).plan().execute()
        },
    );
    assert_eq!(result.shape(), &[64, 64, 1]);
}

#[test]
fn test_tiling_with_non_aligned_dimensions() {
    let input = make_large_gray(1000, 1000); // 1 000 000 bytes, not divisible by 128
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).threshold(100.0).plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone()).threshold(100.0).plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

#[test]
fn test_large_halo_at_image_edges() {
    let input = make_large_gray(512, 512);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).blur(5.0).plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone()).blur(5.0).plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

// ── Pipeline (segment crossing) ───────────────────────────────────────────────

#[test]
fn test_chained_ops_consistency() {
    let input = make_large_rgb(1024, 1024);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone())
            .grayscale().blur(1.5).threshold(128.0)
            .plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone())
            .grayscale().blur(1.5).threshold(128.0)
            .plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

#[test]
fn test_tileable_then_global_op() {
    let input = make_large_rgb(1024, 1024);
    let full = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone())
            .grayscale()
            .resize(512, 512, view_buffer::FilterType::Lanczos3)
            .plan().execute()
    });
    let tiled = with_execution_strategy(always_tiled(), || {
        ViewExpr::new_source(input.clone())
            .grayscale()
            .resize(512, 512, view_buffer::FilterType::Lanczos3)
            .plan().execute()
    });
    assert_buffers_equal(&full, &tiled);
}

// ── Strategy API ──────────────────────────────────────────────────────────────

#[test]
fn test_with_execution_strategy_restores_previous() {
    let initial = get_execution_strategy();

    let result = with_execution_strategy(
        ExecutionStrategy::Tiled { tile_size: 256, threshold_bytes: 0 },
        || {
            assert!(matches!(
                get_execution_strategy(),
                ExecutionStrategy::Tiled { tile_size: 256, .. }
            ));
            with_execution_strategy(
                ExecutionStrategy::Tiled { tile_size: 128, threshold_bytes: 0 },
                || {
                    assert!(matches!(
                        get_execution_strategy(),
                        ExecutionStrategy::Tiled { tile_size: 128, .. }
                    ));
                    42
                },
            )
        },
    );

    assert_eq!(result, 42);
    // Strategy should be restored to what it was before.
    assert_eq!(get_execution_strategy(), initial);
}

#[test]
fn test_plan_with_strategy_override() {
    let input = make_large_rgb(1024, 1024);
    // Build plan under FullImage strategy, then override to tiled.
    let plan = with_execution_strategy(ExecutionStrategy::FullImage, || {
        ViewExpr::new_source(input.clone()).grayscale().plan()
    });
    // Override strategy on the plan itself.
    let result = plan.with_strategy(always_tiled()).execute();
    assert_eq!(result.shape()[0], 1024);
    assert_eq!(result.shape()[1], 1024);
}

// ── TilePolicy queries ────────────────────────────────────────────────────────

#[test]
fn test_tile_policy_for_operations() {
    use view_buffer::ops::traits::Op;
    use view_buffer::{ComputeOp, ImageOp, ImageOpKind};

    assert!(ComputeOp::Scale(2.0).tile_policy().is_tileable());
    assert!(ComputeOp::Relu.tile_policy().is_tileable());
    assert!(ComputeOp::Clamp { min: 0.0, max: 1.0 }.tile_policy().is_tileable());

    assert!(!ComputeOp::Normalize(NormalizeMethod::MinMax).tile_policy().is_tileable());
    assert!(!ComputeOp::Normalize(NormalizeMethod::ZScore).tile_policy().is_tileable());

    assert!(ComputeOp::Normalize(NormalizeMethod::Preset {
        mean: vec![0.5],
        std: vec![0.5],
    }).tile_policy().is_tileable());

    assert!(ImageOp { kind: ImageOpKind::Threshold(128.0) }.tile_policy().is_tileable());
    assert!(ImageOp { kind: ImageOpKind::Grayscale }.tile_policy().is_tileable());

    let blur_policy = ImageOp { kind: ImageOpKind::Blur { sigma: 2.0 } }.tile_policy();
    assert!(blur_policy.is_tileable());
    assert_eq!(blur_policy.halo(), 6); // ceil(3 * 2.0) = 6

    assert!(!ImageOp {
        kind: ImageOpKind::Resize { width: 100, height: 100, filter: view_buffer::FilterType::Nearest }
    }.tile_policy().is_tileable());
}
