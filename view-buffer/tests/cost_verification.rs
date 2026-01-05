//! Cost verification tests.
//!
//! These tests verify that the declared OpCost matches the actual runtime behavior:
//! - ZeroCopy operations should preserve storage_id
//! - Allocating operations should create new storage
//! - DType should be preserved unless explicitly changed

use view_buffer::ops::{ComputeOp, NormalizeMethod, Op, OpCost, ViewOp};
use view_buffer::{DType, ViewBuffer, ViewExpr};

// --- Helper Functions ---

/// Asserts that two buffers share the same underlying storage (zero-copy).
fn assert_zero_copy(original: &ViewBuffer, transformed: &ViewBuffer) {
    assert_eq!(
        original.storage_id(),
        transformed.storage_id(),
        "Expected zero-copy (same storage_id), but buffers have different storage"
    );
}

/// Asserts that two buffers have different storage (allocation occurred).
fn assert_allocated(original: &ViewBuffer, transformed: &ViewBuffer) {
    assert_ne!(
        original.storage_id(),
        transformed.storage_id(),
        "Expected new allocation (different storage_id), but buffers share storage"
    );
}

fn make_2d_buffer() -> ViewBuffer {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let buf = ViewBuffer::from_vec(data);
    ViewExpr::new_source(buf)
        .reshape(vec![10, 10])
        .plan()
        .execute()
}

#[allow(dead_code)]
fn make_3d_buffer() -> ViewBuffer {
    let data: Vec<u8> = (0..300).map(|i| (i % 256) as u8).collect();
    let buf = ViewBuffer::from_vec(data);
    ViewExpr::new_source(buf)
        .reshape(vec![10, 10, 3])
        .plan()
        .execute()
}

// --- OpCost Declaration Tests ---

#[test]
fn test_view_ops_declare_zero_copy() {
    let ops = [
        ViewOp::Transpose(vec![1, 0]),
        ViewOp::Reshape(vec![100]),
        ViewOp::Flip(vec![0]),
        ViewOp::Crop {
            start: vec![0, 0],
            end: vec![5, 5],
        },
    ];

    for op in &ops {
        assert_eq!(
            op.intrinsic_cost(),
            OpCost::ZeroCopy,
            "ViewOp {op:?} should declare ZeroCopy cost"
        );
    }
}

#[test]
fn test_compute_ops_declare_allocating() {
    let ops = [
        ComputeOp::Cast(DType::F32),
        ComputeOp::Scale(2.0),
        ComputeOp::Relu,
        ComputeOp::Normalize(NormalizeMethod::MinMax),
        ComputeOp::Clamp { min: 0.0, max: 1.0 },
    ];

    for op in &ops {
        assert_eq!(
            op.intrinsic_cost(),
            OpCost::Allocating,
            "ComputeOp {op:?} should declare Allocating cost"
        );
    }
}

// --- Runtime Behavior Tests ---

#[test]
fn test_transpose_is_zero_copy() {
    let original = make_2d_buffer();
    let transposed = original.permute(&[1, 0]);
    assert_zero_copy(&original, &transposed);
}

#[test]
fn test_flip_is_zero_copy() {
    let original = make_2d_buffer();
    let flipped = original.flip(&[0]);
    assert_zero_copy(&original, &flipped);
}

#[test]
fn test_crop_is_zero_copy() {
    let original = make_2d_buffer();
    let cropped = original.slice(&[0, 0], &[5, 5]);
    assert_zero_copy(&original, &cropped);
}

#[test]
fn test_clone_is_zero_copy() {
    let original = make_2d_buffer();
    let cloned = original.clone();
    assert_zero_copy(&original, &cloned);
}

#[test]
fn test_to_contiguous_on_contiguous_is_zero_copy() {
    let original = make_2d_buffer();
    assert!(original.layout_facts().is_contiguous());
    let contig = original.to_contiguous();
    assert_zero_copy(&original, &contig);
}

#[test]
fn test_to_contiguous_on_strided_allocates() {
    let original = make_2d_buffer();
    let transposed = original.permute(&[1, 0]);
    // Transposed buffer is not contiguous
    assert!(!transposed.layout_facts().is_contiguous());
    let contig = transposed.to_contiguous();
    assert_allocated(&transposed, &contig);
}

#[test]
fn test_fused_kernel_allocates() {
    use view_buffer::ops::scalar::{FusedKernel, ScalarOp};

    let original = make_2d_buffer();
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Mul(2.0));

    let result = original.apply_fused_kernel(&kernel);
    assert_allocated(&original, &result);
}

// --- DType Preservation Tests ---

#[test]
fn test_scale_preserves_dtype() {
    let original = make_2d_buffer();
    assert_eq!(original.dtype(), DType::F32);

    let expr = ViewExpr::new_source(original);
    let scaled = expr.scale(2.0).plan().execute();
    assert_eq!(
        scaled.dtype(),
        DType::F32,
        "Scale should preserve F32 dtype"
    );
}

#[test]
fn test_relu_preserves_dtype() {
    let original = make_2d_buffer();
    assert_eq!(original.dtype(), DType::F32);

    let expr = ViewExpr::new_source(original);
    let relued = expr.relu().plan().execute();
    assert_eq!(relued.dtype(), DType::F32, "Relu should preserve F32 dtype");
}

#[test]
fn test_clamp_preserves_dtype() {
    let original = make_2d_buffer();
    assert_eq!(original.dtype(), DType::F32);

    let expr = ViewExpr::new_source(original);
    let clamped = expr.clamp(0.0, 1.0).plan().execute();
    assert_eq!(
        clamped.dtype(),
        DType::F32,
        "Clamp should preserve F32 dtype"
    );
}

#[test]
fn test_normalize_preserves_dtype() {
    let original = make_2d_buffer();
    assert_eq!(original.dtype(), DType::F32);

    let expr = ViewExpr::new_source(original);
    let normalized = expr.normalize(NormalizeMethod::MinMax).plan().execute();
    assert_eq!(
        normalized.dtype(),
        DType::F32,
        "Normalize should preserve F32 dtype"
    );
}

#[test]
fn test_cast_changes_dtype() {
    let original = make_2d_buffer();
    assert_eq!(original.dtype(), DType::F32);

    let expr = ViewExpr::new_source(original);
    let casted = expr.cast(DType::U8).plan().execute();
    assert_eq!(casted.dtype(), DType::U8, "Cast should change dtype to U8");
}

// --- Validation Tests ---

#[test]
fn test_normalize_validation_accepts_2d() {
    let buf_2d = make_2d_buffer(); // [10, 10] F32
    let op = ComputeOp::Normalize(NormalizeMethod::MinMax);
    let result = op.validate(&[buf_2d.shape()], &[buf_2d.dtype()]);
    assert!(result.is_ok(), "Normalize should accept 2D F32 buffer");
}

#[test]
fn test_normalize_validation_accepts_hw1() {
    // Create HW1 buffer
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let buf = ViewBuffer::from_vec(data);
    let buf_hw1 = ViewExpr::new_source(buf)
        .reshape(vec![10, 10, 1])
        .plan()
        .execute();

    let op = ComputeOp::Normalize(NormalizeMethod::MinMax);
    let result = op.validate(&[buf_hw1.shape()], &[buf_hw1.dtype()]);
    assert!(result.is_ok(), "Normalize should accept HW1 F32 buffer");
}

#[test]
fn test_normalize_validation_rejects_hwc() {
    // Create HWC buffer with C=3
    let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let buf = ViewBuffer::from_vec(data);
    let buf_hwc = ViewExpr::new_source(buf)
        .reshape(vec![10, 10, 3])
        .plan()
        .execute();

    let op = ComputeOp::Normalize(NormalizeMethod::MinMax);
    let result = op.validate(&[buf_hwc.shape()], &[buf_hwc.dtype()]);
    assert!(
        result.is_err(),
        "Normalize should reject HWC buffer with C>1"
    );
}

#[test]
fn test_normalize_validation_accepts_all_numeric_types() {
    // With the dtype promotion system, normalize now accepts all numeric types
    // and handles casting internally. This test verifies that behavior.
    let op = ComputeOp::Normalize(NormalizeMethod::MinMax);

    // Test that all numeric types are accepted
    let numeric_dtypes = [
        DType::U8,
        DType::I8,
        DType::U16,
        DType::I16,
        DType::U32,
        DType::I32,
        DType::U64,
        DType::I64,
        DType::F32,
        DType::F64,
    ];

    for dtype in numeric_dtypes {
        let result = op.validate(&[&[10, 10]], &[dtype]);
        assert!(
            result.is_ok(),
            "Normalize should accept {dtype:?} dtype with dtype promotion"
        );
    }
}

// --- Cost Report Tests ---

#[test]
fn test_cost_report_counts_allocations() {
    let buf = make_2d_buffer();
    let expr = ViewExpr::new_source(buf);

    // Pipeline with 2 allocating ops and 1 zero-copy op
    let pipeline = expr
        .flip(vec![0]) // ZeroCopy
        .scale(2.0) // Allocating
        .relu(); // Allocating

    let report = pipeline.cost_report();

    assert_eq!(report.total_allocations, 2, "Should have 2 allocating ops");
    assert_eq!(report.operations.len(), 3, "Should have 3 operations total");
}

#[test]
fn test_cost_report_tracks_dtype_changes() {
    let buf = make_2d_buffer();
    let expr = ViewExpr::new_source(buf);

    let pipeline = expr.cast(DType::U8);

    let report = pipeline.cost_report();

    assert_eq!(report.dtype_changes.len(), 1, "Should have 1 dtype change");
    let (op_name, from, to) = &report.dtype_changes[0];
    assert_eq!(op_name, "Cast");
    assert_eq!(*from, DType::F32);
    assert_eq!(*to, DType::U8);
}

/// Tests that the explain_costs output contains expected information.
#[test]
fn test_explain_costs_output() {
    let buf = make_2d_buffer();
    let expr = ViewExpr::new_source(buf);

    let pipeline = expr.flip(vec![0]).scale(2.0);

    let explanation = pipeline.explain_costs();

    // Check the summary header
    assert!(explanation.contains("Pipeline Cost Summary"));
    // The new format shows "Operations: X (Y zero-copy, Z allocating)"
    assert!(explanation.contains("1 zero-copy, 1 allocating"));
    // Check individual ops are listed with their cost symbols and dtype info
    assert!(explanation.contains("Flip [0]")); // ZeroCopy symbol
    assert!(explanation.contains("Scale [A]")); // Allocating symbol
                                                // Check dtype flow is shown
    assert!(explanation.contains("F32"));
}
