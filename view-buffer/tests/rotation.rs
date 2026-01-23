//! Tests for rotation operations.

#![cfg(all(feature = "ndarray_interop", feature = "image_interop"))]

use view_buffer::{ViewBuffer, ViewExpr, ViewOp};

/// Helper to create a test grayscale image buffer with known pattern.
fn make_test_image(h: usize, w: usize) -> ViewBuffer {
    let mut data = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            // Create a pattern: value increases with x, different per row
            data.push(((y * 10 + x) % 256) as u8);
        }
    }
    ViewBuffer::from_vec(data).reshape(vec![h, w])
}

/// Helper to create a test RGB image buffer.
fn make_rgb_image(h: usize, w: usize) -> ViewBuffer {
    let mut data = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                data.push(((y * 10 + x * 3 + c) % 256) as u8);
            }
        }
    }
    ViewBuffer::from_vec(data).reshape(vec![h, w, 3])
}

#[test]
fn test_rotate90_zero_copy() {
    let buf = make_test_image(4, 6);
    let original_id = buf.storage_id();

    // Rotate90 should be zero-copy
    let rotated = ViewExpr::new_source(buf)
        .apply_op(view_buffer::ViewDto::View(ViewOp::Rotate90))
        .plan()
        .execute();

    // Check shape: [4, 6] -> [6, 4]
    assert_eq!(rotated.shape(), &[6, 4]);

    // Verify zero-copy (should share storage)
    // Note: Rotate90 uses transpose + flip, which may create new views
    // but the underlying data should be shared
    let rotated_id = rotated.storage_id();
    // For zero-copy operations, storage_id should match or be related
    // Since we're doing transpose + flip, the storage might be the same
    assert_eq!(original_id, rotated_id, "Rotate90 should be zero-copy");
}

#[test]
fn test_rotate180_zero_copy() {
    let buf = make_test_image(4, 6);
    let original_id = buf.storage_id();

    // Rotate180 should be zero-copy
    let rotated = ViewExpr::new_source(buf)
        .apply_op(view_buffer::ViewDto::View(ViewOp::Rotate180))
        .plan()
        .execute();

    // Check shape: [4, 6] -> [4, 6] (same)
    assert_eq!(rotated.shape(), &[4, 6]);

    // Verify zero-copy
    assert_eq!(
        original_id,
        rotated.storage_id(),
        "Rotate180 should be zero-copy"
    );
}

#[test]
fn test_rotate270_zero_copy() {
    let buf = make_test_image(4, 6);
    let original_id = buf.storage_id();

    // Rotate270 should be zero-copy
    let rotated = ViewExpr::new_source(buf)
        .apply_op(view_buffer::ViewDto::View(ViewOp::Rotate270))
        .plan()
        .execute();

    // Check shape: [4, 6] -> [6, 4]
    assert_eq!(rotated.shape(), &[6, 4]);

    // Verify zero-copy
    assert_eq!(
        original_id,
        rotated.storage_id(),
        "Rotate270 should be zero-copy"
    );
}

#[test]
fn test_rotate90_rgb() {
    let buf = make_rgb_image(3, 5);

    // Rotate90 on RGB image
    let rotated = ViewExpr::new_source(buf)
        .apply_op(view_buffer::ViewDto::View(ViewOp::Rotate90))
        .plan()
        .execute();

    // Check shape: [3, 5, 3] -> [5, 3, 3]
    assert_eq!(rotated.shape(), &[5, 3, 3]);
}

#[test]
fn test_rotate_double_180() {
    // Rotating 180 twice should return to original
    let buf = make_test_image(4, 6);
    let original_data = buf.to_contiguous().as_slice::<u8>().to_vec();

    let rotated = ViewExpr::new_source(buf)
        .apply_op(view_buffer::ViewDto::View(ViewOp::Rotate180))
        .apply_op(view_buffer::ViewDto::View(ViewOp::Rotate180))
        .plan()
        .execute();

    let rotated_data = rotated.to_contiguous().as_slice::<u8>().to_vec();
    assert_eq!(
        original_data, rotated_data,
        "Double 180 rotation should return to original"
    );
}

#[test]
fn test_rotate_arbitrary_angle() {
    use view_buffer::ops::{ImageOp, ImageOpKind};

    let buf = make_test_image(10, 10);

    // Test arbitrary angle rotation (45 degrees)
    let rotated = ViewExpr::new_source(buf)
        .apply_op(view_buffer::ViewDto::Image(ImageOp {
            kind: ImageOpKind::Rotate {
                angle: 45.0,
                expand: true,
            },
        }))
        .plan()
        .execute();

    // With expand=true, output should be larger
    assert!(rotated.shape()[0] >= 10);
    assert!(rotated.shape()[1] >= 10);
}

#[test]
fn test_rotate_arbitrary_no_expand() {
    use view_buffer::ops::{ImageOp, ImageOpKind};

    let buf = make_test_image(10, 10);
    let original_shape = buf.shape().to_vec();

    // Test arbitrary angle rotation without expansion
    let rotated = ViewExpr::new_source(buf)
        .apply_op(view_buffer::ViewDto::Image(ImageOp {
            kind: ImageOpKind::Rotate {
                angle: 30.0,
                expand: false,
            },
        }))
        .plan()
        .execute();

    // Without expand, output should have same dimensions
    assert_eq!(rotated.shape(), original_shape.as_slice());
}
