//! Validation system tests.
//!
//! Tests for the plan-time validation framework.

use view_buffer::ops::validation::{is_2d_like, is_float_dtype, is_image_like, is_integer_dtype};
use view_buffer::ops::{ComputeOp, NormalizeMethod, Op};
use view_buffer::DType;

// --- Shape Predicate Tests ---

#[test]
fn test_is_2d_like_accepts_2d() {
    assert!(is_2d_like(&[10, 10]));
    assert!(is_2d_like(&[100, 200]));
    assert!(is_2d_like(&[1, 1]));
}

#[test]
fn test_is_2d_like_accepts_hw1() {
    assert!(is_2d_like(&[10, 10, 1]));
    assert!(is_2d_like(&[100, 200, 1]));
}

#[test]
fn test_is_2d_like_rejects_hwc() {
    assert!(!is_2d_like(&[10, 10, 3]));
    assert!(!is_2d_like(&[10, 10, 4]));
    assert!(!is_2d_like(&[10, 10, 2]));
}

#[test]
fn test_is_2d_like_rejects_other_ranks() {
    assert!(!is_2d_like(&[10]));
    assert!(!is_2d_like(&[10, 10, 3, 1]));
    assert!(!is_2d_like(&[]));
}

#[test]
fn test_is_image_like() {
    assert!(is_image_like(&[10, 10, 1])); // Grayscale
    assert!(is_image_like(&[10, 10, 3])); // RGB
    assert!(is_image_like(&[10, 10, 4])); // RGBA

    assert!(!is_image_like(&[10, 10])); // 2D
    assert!(!is_image_like(&[10, 10, 2])); // Invalid channel count
    assert!(!is_image_like(&[10, 10, 5])); // Invalid channel count
}

// --- DType Predicate Tests ---

#[test]
fn test_is_float_dtype() {
    assert!(is_float_dtype(DType::F32));
    assert!(is_float_dtype(DType::F64));

    assert!(!is_float_dtype(DType::U8));
    assert!(!is_float_dtype(DType::I32));
}

#[test]
fn test_is_integer_dtype() {
    assert!(is_integer_dtype(DType::U8));
    assert!(is_integer_dtype(DType::I8));
    assert!(is_integer_dtype(DType::U16));
    assert!(is_integer_dtype(DType::I16));
    assert!(is_integer_dtype(DType::U32));
    assert!(is_integer_dtype(DType::I32));
    assert!(is_integer_dtype(DType::U64));
    assert!(is_integer_dtype(DType::I64));

    assert!(!is_integer_dtype(DType::F32));
    assert!(!is_integer_dtype(DType::F64));
}

// --- Op Validation Tests ---

#[test]
fn test_normalize_validates_shape() {
    let op = ComputeOp::Normalize(NormalizeMethod::MinMax);

    // Valid shapes
    assert!(op.validate(&[&[10, 10]], &[DType::F32]).is_ok());
    assert!(op.validate(&[&[100, 200]], &[DType::F32]).is_ok());
    assert!(op.validate(&[&[10, 10, 1]], &[DType::F32]).is_ok());

    // Invalid shapes
    assert!(op.validate(&[&[10, 10, 3]], &[DType::F32]).is_err());
    assert!(op.validate(&[&[10, 10, 4]], &[DType::F32]).is_err());
    assert!(op.validate(&[&[10]], &[DType::F32]).is_err());
}

#[test]
fn test_normalize_accepts_all_numeric_dtypes() {
    let op = ComputeOp::Normalize(NormalizeMethod::MinMax);

    // With dtype promotion, all numeric types are valid
    // The operation internally casts to f32 for computation
    assert!(op.validate(&[&[10, 10]], &[DType::F32]).is_ok());
    assert!(op.validate(&[&[10, 10]], &[DType::U8]).is_ok());
    assert!(op.validate(&[&[10, 10]], &[DType::I32]).is_ok());
    assert!(op.validate(&[&[10, 10]], &[DType::F64]).is_ok());
    assert!(op.validate(&[&[10, 10]], &[DType::U16]).is_ok());
    assert!(op.validate(&[&[10, 10]], &[DType::I16]).is_ok());
}

#[test]
fn test_normalize_error_message_contains_shape() {
    let op = ComputeOp::Normalize(NormalizeMethod::MinMax);
    let result = op.validate(&[&[10, 10, 3]], &[DType::F32]);

    let err = result.unwrap_err();
    let msg = format!("{err}");

    assert!(msg.contains("10"), "Error should contain shape dimension");
    assert!(
        msg.contains("HW") || msg.contains("2D"),
        "Error should mention required shape"
    );
}

#[test]
fn test_normalize_dtype_promotion_behavior() {
    // With dtype promotion, normalize accepts all numeric types
    // This test verifies the working dtype is used correctly
    let op = ComputeOp::Normalize(NormalizeMethod::MinMax);

    // All numeric types should be accepted - the operation handles casting internally
    assert!(op.validate(&[&[10, 10]], &[DType::U8]).is_ok());
    assert!(op.validate(&[&[10, 10]], &[DType::F32]).is_ok());

    // The working dtype should be F32
    assert_eq!(op.working_dtype(), Some(DType::F32));
}

#[test]
fn test_other_compute_ops_have_no_validation() {
    let ops = [
        ComputeOp::Cast(DType::U8),
        ComputeOp::Scale(2.0),
        ComputeOp::Relu,
        ComputeOp::Clamp { min: 0.0, max: 1.0 },
    ];

    // These should all pass validation with any input
    for op in &ops {
        assert!(
            op.validate(&[&[10, 10, 3]], &[DType::U8]).is_ok(),
            "Op {op:?} should have no special validation requirements"
        );
    }
}

#[test]
fn test_zscore_normalize_validates_same_as_minmax() {
    let op = ComputeOp::Normalize(NormalizeMethod::ZScore);

    // Should have same requirements as MinMax
    // With dtype promotion, all numeric types are accepted
    assert!(op.validate(&[&[10, 10]], &[DType::F32]).is_ok());
    assert!(op.validate(&[&[10, 10, 3]], &[DType::F32]).is_err()); // Shape still matters
    assert!(op.validate(&[&[10, 10]], &[DType::U8]).is_ok()); // Now accepts U8
}
