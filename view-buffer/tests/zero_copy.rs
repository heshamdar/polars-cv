//! Zero-copy verification tests.
//!
//! These tests require the `image_interop` and `ndarray_interop` features.

#![cfg(all(feature = "image_interop", feature = "ndarray_interop"))]

use image::Rgb;
use view_buffer::{ExternalView, ImageViewAdapter, NdArrayViewAdapter, ViewBuffer, ViewExpr};

fn assert_zero_copy(a: &ViewBuffer, b: &ViewBuffer) {
    assert_eq!(
        a.storage_id(),
        b.storage_id(),
        "Expected zero-copy view, but storage differs (Allocated new buffer)"
    );
}

fn assert_copy(a: &ViewBuffer, b: &ViewBuffer) {
    assert_ne!(
        a.storage_id(),
        b.storage_id(),
        "Expected new allocation (copy), but storage ID is identical"
    );
}

fn make_image_view() -> ViewBuffer {
    let data = vec![0u8; 100 * 100 * 3];
    let buf = ViewBuffer::from_vec(data);
    ViewExpr::new_source(buf)
        .reshape(vec![100, 100, 3])
        .plan()
        .execute()
}

#[test]
fn case_contiguous_ndarray() {
    let base = make_image_view();
    let _view = NdArrayViewAdapter::<u8>::try_view(&base).expect("Contiguous ndarray view failed");
}

#[test]
fn case_contiguous_image() {
    let base = make_image_view();
    let _view = ImageViewAdapter::<Rgb<u8>>::try_view(&base).expect("Contiguous image view failed");
}

#[test]
fn case_crop_image_zero_copy() {
    let base = make_image_view();
    let crop = base.slice(&[10, 10, 0], &[90, 90, 3]);
    let _img = ImageViewAdapter::<Rgb<u8>>::try_view(&crop).expect("Crop image view failed");
    assert_zero_copy(&base, &crop);
}

#[test]
fn case_transpose_ndarray_ok() {
    let base = make_image_view();
    let t = base.permute(&[1, 0, 2]);
    NdArrayViewAdapter::<u8>::try_view(&t).expect("Transpose ndarray view failed");
    assert_zero_copy(&base, &t);
}

#[test]
fn case_transpose_image_rejected() {
    let base = make_image_view();
    let t = base.permute(&[1, 0, 2]);
    let err = ImageViewAdapter::<Rgb<u8>>::try_view(&t).unwrap_err();
    match err {
        view_buffer::core::buffer::BufferError::IncompatibleLayout { .. } => {}
        _ => panic!("Expected IncompatibleLayout error, got {err:?}"),
    }
}

#[test]
fn case_transpose_then_materialize_image() {
    let base = make_image_view();
    let t = base.permute(&[1, 0, 2]);
    let m = t.to_contiguous();
    let _img = ImageViewAdapter::<Rgb<u8>>::try_view(&m).expect("Materialized image view failed");
    assert_zero_copy(&base, &t);
    assert_copy(&t, &m);
}

#[test]
fn test_storage_ids() {
    let a = make_image_view();
    let b = a.clone();
    assert_zero_copy(&a, &b);
    let c = a.to_contiguous();
    assert_zero_copy(&a, &c);
}

#[test]
fn test_slice_clamps_end_to_shape() {
    // Test that slice clamps end values exceeding the shape
    let base = make_image_view(); // 100x100x3
    let crop = base.slice(&[10, 10, 0], &[90, 90, usize::MAX]);
    // The usize::MAX should be clamped to the actual channel dimension (3)
    assert_eq!(crop.shape(), &[80, 80, 3]);
    assert_zero_copy(&base, &crop);

    // Verify we can materialize without overflow
    let contiguous = crop.to_contiguous();
    assert_eq!(contiguous.shape(), &[80, 80, 3]);
}

#[test]
fn test_slice_clamps_start_and_end() {
    let base = make_image_view(); // 100x100x3
                                  // Test with start > dimension size (should clamp to shape, resulting in 0-size dim)
    let crop = base.slice(&[200, 0, 0], &[300, 50, 3]);
    assert_eq!(crop.shape(), &[0, 50, 3]);

    // Test with both start and end beyond bounds
    let crop2 = base.slice(&[0, 0, 0], &[150, 150, 5]);
    assert_eq!(crop2.shape(), &[100, 100, 3]);
}

#[test]
fn test_slice_handles_reversed_bounds() {
    let base = make_image_view(); // 100x100x3
                                  // Test with start > end (should produce zero-size dimension via saturating_sub)
    let crop = base.slice(&[50, 0, 0], &[30, 50, 3]);
    assert_eq!(crop.shape(), &[0, 50, 3]);
}
