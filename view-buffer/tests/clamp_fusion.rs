//! Tests for Clamp fusion into FusedKernel.
//!
//! These tests verify that:
//! 1. Clamp alone produces correct results
//! 2. Clamp fused with other operations produces identical results to sequential execution
//! 3. Edge cases at clamp boundaries are handled correctly

use view_buffer::ops::scalar::{FusedKernel, ScalarOp};
use view_buffer::{DType, ViewBuffer};

/// Helper to extract f32 slice from ViewBuffer.
fn extract_f32_slice(buf: &ViewBuffer) -> Vec<f32> {
    let (ptr, _, _, _) = buf.as_raw_parts();
    let len = buf.shape().iter().product::<usize>();
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, len) };
    slice.to_vec()
}

#[test]
fn test_clamp_alone_matches_manual() {
    // Input: [-2.0, -0.5, 0.0, 0.5, 1.5, 2.0]
    let input_data = vec![-2.0f32, -0.5, 0.0, 0.5, 1.5, 2.0];
    let buf = ViewBuffer::from_vec(input_data);

    // Clamp to [0.0, 1.0]
    // Expected: [0.0, 0.0, 0.0, 0.5, 1.0, 1.0]
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Clamp(0.0, 1.0));

    let result = buf.apply_fused_kernel(&kernel);

    assert_eq!(result.dtype(), DType::F32);
    let result_slice = extract_f32_slice(&result);
    assert_eq!(result_slice, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0]);
}

#[test]
fn test_scale_then_clamp_fused_equals_sequential() {
    // Input: [0.0, 0.25, 0.5, 0.75, 1.0]
    let input_data = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];

    // Sequential: scale(2.0) then clamp(0.0, 1.0)
    // Step 1: [0.0, 0.5, 1.0, 1.5, 2.0]
    // Step 2: [0.0, 0.5, 1.0, 1.0, 1.0]
    let buf1 = ViewBuffer::from_vec(input_data.clone());
    let step1 = {
        let mut k = FusedKernel::new();
        k.push(ScalarOp::Mul(2.0));
        buf1.apply_fused_kernel(&k)
    };
    let sequential_result = {
        let mut k = FusedKernel::new();
        k.push(ScalarOp::Clamp(0.0, 1.0));
        step1.apply_fused_kernel(&k)
    };

    // Fused: scale(2.0) and clamp(0.0, 1.0) in one pass
    let buf2 = ViewBuffer::from_vec(input_data);
    let mut fused_kernel = FusedKernel::new();
    fused_kernel.push(ScalarOp::Mul(2.0));
    fused_kernel.push(ScalarOp::Clamp(0.0, 1.0));
    let fused_result = buf2.apply_fused_kernel(&fused_kernel);

    // Results must be identical
    let seq_slice = extract_f32_slice(&sequential_result);
    let fused_slice = extract_f32_slice(&fused_result);
    assert_eq!(seq_slice, fused_slice);
    assert_eq!(fused_slice, vec![0.0, 0.5, 1.0, 1.0, 1.0]);
}

#[test]
fn test_clamp_then_scale_fused_equals_sequential() {
    // Input: [-1.0, 0.0, 0.5, 1.0, 2.0]
    let input_data = vec![-1.0f32, 0.0, 0.5, 1.0, 2.0];

    // Sequential: clamp(0.0, 1.0) then scale(10.0)
    // Step 1: [0.0, 0.0, 0.5, 1.0, 1.0]
    // Step 2: [0.0, 0.0, 5.0, 10.0, 10.0]
    let buf1 = ViewBuffer::from_vec(input_data.clone());
    let step1 = {
        let mut k = FusedKernel::new();
        k.push(ScalarOp::Clamp(0.0, 1.0));
        buf1.apply_fused_kernel(&k)
    };
    let sequential_result = {
        let mut k = FusedKernel::new();
        k.push(ScalarOp::Mul(10.0));
        step1.apply_fused_kernel(&k)
    };

    // Fused: clamp(0.0, 1.0) and scale(10.0) in one pass
    let buf2 = ViewBuffer::from_vec(input_data);
    let mut fused_kernel = FusedKernel::new();
    fused_kernel.push(ScalarOp::Clamp(0.0, 1.0));
    fused_kernel.push(ScalarOp::Mul(10.0));
    let fused_result = buf2.apply_fused_kernel(&fused_kernel);

    let seq_slice = extract_f32_slice(&sequential_result);
    let fused_slice = extract_f32_slice(&fused_result);
    assert_eq!(seq_slice, fused_slice);
    assert_eq!(fused_slice, vec![0.0, 0.0, 5.0, 10.0, 10.0]);
}

#[test]
fn test_triple_fusion_scale_clamp_relu() {
    // Input: [-2.0, -1.0, 0.0, 1.0, 2.0]
    let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    // Sequential: scale(2.0), clamp(-1.0, 1.0), relu
    // Step 1: [-4.0, -2.0, 0.0, 2.0, 4.0]
    // Step 2: [-1.0, -1.0, 0.0, 1.0, 1.0]
    // Step 3: [0.0, 0.0, 0.0, 1.0, 1.0]
    let buf1 = ViewBuffer::from_vec(input_data.clone());
    let step1 = {
        let mut k = FusedKernel::new();
        k.push(ScalarOp::Mul(2.0));
        buf1.apply_fused_kernel(&k)
    };
    let step2 = {
        let mut k = FusedKernel::new();
        k.push(ScalarOp::Clamp(-1.0, 1.0));
        step1.apply_fused_kernel(&k)
    };
    let sequential_result = {
        let mut k = FusedKernel::new();
        k.push(ScalarOp::Relu);
        step2.apply_fused_kernel(&k)
    };

    // Fused: all three in one pass
    let buf2 = ViewBuffer::from_vec(input_data);
    let mut fused_kernel = FusedKernel::new();
    fused_kernel.push(ScalarOp::Mul(2.0));
    fused_kernel.push(ScalarOp::Clamp(-1.0, 1.0));
    fused_kernel.push(ScalarOp::Relu);
    let fused_result = buf2.apply_fused_kernel(&fused_kernel);

    let seq_slice = extract_f32_slice(&sequential_result);
    let fused_slice = extract_f32_slice(&fused_result);
    assert_eq!(seq_slice, fused_slice);
    assert_eq!(fused_slice, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_fusion_preserves_edge_values() {
    // Test exactly at clamp bounds
    // Input: [0.0, 1.0] (exactly at bounds)
    let input_data = vec![0.0f32, 1.0];
    let buf = ViewBuffer::from_vec(input_data);

    // Clamp to [0.0, 1.0] should leave values unchanged
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Clamp(0.0, 1.0));

    let result = buf.apply_fused_kernel(&kernel);
    let result_slice = extract_f32_slice(&result);
    assert_eq!(result_slice, vec![0.0, 1.0]);
}

#[test]
fn test_clamp_with_negative_range() {
    // Input: [-3.0, -2.0, -1.0, 0.0, 1.0]
    let input_data = vec![-3.0f32, -2.0, -1.0, 0.0, 1.0];
    let buf = ViewBuffer::from_vec(input_data);

    // Clamp to [-2.0, -0.5]
    // Expected: [-2.0, -2.0, -1.0, -0.5, -0.5]
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Clamp(-2.0, -0.5));

    let result = buf.apply_fused_kernel(&kernel);
    let result_slice = extract_f32_slice(&result);
    assert_eq!(result_slice, vec![-2.0, -2.0, -1.0, -0.5, -0.5]);
}

#[test]
fn test_add_clamp_fused() {
    // Input: [0.0, 0.5, 1.0]
    let input_data = vec![0.0f32, 0.5, 1.0];
    let buf = ViewBuffer::from_vec(input_data);

    // Add 0.6, then clamp to [0.0, 1.0]
    // Step 1: [0.6, 1.1, 1.6]
    // Step 2: [0.6, 1.0, 1.0]
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Add(0.6));
    kernel.push(ScalarOp::Clamp(0.0, 1.0));

    let result = buf.apply_fused_kernel(&kernel);
    let result_slice = extract_f32_slice(&result);

    // Use approximate comparison for floating point
    assert!((result_slice[0] - 0.6).abs() < 1e-6);
    assert!((result_slice[1] - 1.0).abs() < 1e-6);
    assert!((result_slice[2] - 1.0).abs() < 1e-6);
}
