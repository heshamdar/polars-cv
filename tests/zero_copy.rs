use image::Rgb;
use view_buffer::{
    ExternalView, ImageViewAdapter, NdArrayViewAdapter, ViewBuffer, ViewExpr,
};

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
        view_buffer::buffer::BufferError::IncompatibleLayout { .. } => {}
        _ => panic!("Expected IncompatibleLayout error, got {:?}", err),
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
