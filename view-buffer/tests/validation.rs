//! Validation system tests.
//!
//! Tests for the plan-time validation framework.

use view_buffer::ops::validation::{is_2d_like, is_float_dtype, is_image_like, is_integer_dtype};
use view_buffer::ops::{ComputeOp, Normalization, Op};
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

/// MinMax/ZScore take global statistics over every element (as documented on
/// `Pipeline.normalize`), so any shape is valid. These tests used to pin a
/// "2-D or single-channel only" rule that execution never enforced — it never
/// called `validate` — and that contradicts the kernel; enforcing it (CR-34)
/// would have rejected every minmax normalize of an RGB image.
#[test]
fn test_normalize_accepts_any_shape() {
    let op = ComputeOp::from_normalization(Normalization::MinMax, DType::F32);
    for shape in [
        &[10, 10][..],
        &[100, 200],
        &[10, 10, 1],
        &[10, 10, 3],
        &[10, 10, 4],
        &[10],
    ] {
        assert!(op.validate(&[shape], &[DType::F32]).is_ok(), "{shape:?}");
    }
}

#[test]
fn test_normalize_accepts_all_numeric_dtypes() {
    let op = ComputeOp::from_normalization(Normalization::MinMax, DType::F32);

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
fn test_normalize_preset_error_message_names_the_mismatch() {
    // The per-channel method is the one that constrains the channel count.
    let op = ComputeOp::from_normalization(
        Normalization::Preset {
            mean: vec![0.5, 0.5],
            std: vec![0.2, 0.2],
        },
        DType::F32,
    );
    let msg = format!(
        "{}",
        op.validate(&[&[10, 10, 3]], &[DType::F32]).unwrap_err()
    );
    assert!(msg.contains("channel"), "{msg}");
}

#[test]
fn test_normalize_dtype_promotion_behavior() {
    // With dtype promotion, normalize accepts all numeric types
    // This test verifies the working dtype is used correctly
    let op = ComputeOp::from_normalization(Normalization::MinMax, DType::F32);

    // All numeric types should be accepted - the operation handles casting internally
    assert!(op.validate(&[&[10, 10]], &[DType::U8]).is_ok());
    assert!(op.validate(&[&[10, 10]], &[DType::F32]).is_ok());

    // The working dtype should be F32
    assert_eq!(op.working_dtype(), Some(DType::F32));
}

#[test]
fn test_other_compute_ops_have_no_validation() {
    let ops: [ComputeOp; 4] = [
        ComputeOp::Cast { dtype: DType::U8 },
        ComputeOp::Scale { factor: 2.0 },
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
    let op = ComputeOp::from_normalization(Normalization::ZScore, DType::F32);

    // Same requirements as MinMax: a global statistic, any shape, any numeric dtype.
    assert!(op.validate(&[&[10, 10]], &[DType::F32]).is_ok());
    assert!(op.validate(&[&[10, 10, 3]], &[DType::F32]).is_ok());
    assert!(op.validate(&[&[10, 10]], &[DType::U8]).is_ok());
}
