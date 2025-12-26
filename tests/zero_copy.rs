use tensor_buffer::{TensorBuffer, ImageViewAdapter, ExternalView, DType, NdArrayViewAdapter, TensorExpr};
use image::Rgb;

// 3.1 Canonical test helper
fn assert_zero_copy(a: &TensorBuffer, b: &TensorBuffer) {
    assert_eq!(
        a.storage_id(),
        b.storage_id(),
        "Expected zero-copy view, but storage differs (Allocated new buffer)"
    );
}

fn assert_copy(a: &TensorBuffer, b: &TensorBuffer) {
    assert_ne!(
        a.storage_id(),
        b.storage_id(),
        "Expected new allocation (copy), but storage ID is identical"
    );
}

// Helper to make a standard 100x100 RGB image
fn make_image_tensor() -> TensorBuffer {
    // 100x100x3 = 30,000 bytes
    let data = vec![0u8; 100 * 100 * 3];
    let buf = TensorBuffer::from_vec(data);
    
    // Use TensorExpr to access the planner's ability to reshape (which accesses internal APIs)
    TensorExpr::new_source(buf)
        .reshape(vec![100, 100, 3])
        .plan()
        .execute()
}

// 3.2 Layout × Backend test table

#[test]
fn case_contiguous_ndarray() {
    let base = make_image_tensor();
    
    // Attempt view
    let _view = NdArrayViewAdapter::<u8>::try_view(&base).expect("Contiguous ndarray view failed");
    
    // NdArray adapter doesn't return a TensorBuffer, but the success implies compatibility.
    // If we had a hypothetical method `base.as_ndarray_tensor()`, we would assert_zero_copy.
    // Here we trust the try_view didn't panic and checks logic in src.
}

#[test]
fn case_contiguous_image() {
    let base = make_image_tensor();
    // Attempt view
    let _view = ImageViewAdapter::<Rgb<u8>>::try_view(&base).expect("Contiguous image view failed");
}

#[test]
fn case_crop_image_zero_copy() {
    let base = make_image_tensor();
    // Crop: 10..90 in H, 10..90 in W
    let crop = base.slice(&[10, 10, 0], &[90, 90, 3]);

    // This should work because it preserves row density (just offsets start/end, keeps strides)
    // Wait, standard slicing in TensorBuffer preserves original strides.
    // [100, 100, 3] -> strides [300, 3, 1]
    // Slice 80x80x3.
    // Row stride is still 300.
    // Width * Channels * Elem = 80 * 3 * 1 = 240.
    // 300 >= 240. Padded rows are supported by `ExternalLayout::ImageCrate`.
    
    let _img = ImageViewAdapter::<Rgb<u8>>::try_view(&crop).expect("Crop image view failed");
    
    assert_zero_copy(&base, &crop);
}

#[test]
fn case_transpose_ndarray_ok() {
    let base = make_image_tensor();
    // Transpose H and W: [100, 100, 3] -> [100, 100, 3]
    let t = base.permute(&[1, 0, 2]);
    
    // ndarray supports arbitrary strides
    NdArrayViewAdapter::<u8>::try_view(&t).expect("Transpose ndarray view failed");
    
    assert_zero_copy(&base, &t);
}

#[test]
fn case_transpose_image_rejected() {
    let base = make_image_tensor();
    // Transpose H and W
    let t = base.permute(&[1, 0, 2]);

    // This breaks "Dense Rows" or "Contiguous Channels" depending on exact permutation.
    // Strides become [3, 300, 1].
    // Channel stride (1) is fine.
    // Pixel stride (stride[1]) is 300. 
    // Expected Pixel stride = C * 1 = 3.
    // 300 != 3. Not dense pixels.
    
    let err = ImageViewAdapter::<Rgb<u8>>::try_view(&t).unwrap_err();
    
    // Verify it failed due to incompatible layout
    match err {
        tensor_buffer::buffer::BufferError::IncompatibleLayout { .. } => {},
        _ => panic!("Expected IncompatibleLayout error, got {:?}", err),
    }
}

#[test]
fn case_transpose_then_materialize_image() {
    let base = make_image_tensor();
    let t = base.permute(&[1, 0, 2]); // Zero copy, weird strides

    // Force copy to make it compatible
    let m = t.to_contiguous(); // Allocates new buffer
    
    // Now it should work
    let _img = ImageViewAdapter::<Rgb<u8>>::try_view(&m).expect("Materialized image view failed");

    // Assert identity
    assert_zero_copy(&base, &t); // Transpose is view
    assert_copy(&t, &m);         // Materialize is copy
}

#[test]
fn test_storage_ids() {
    let a = make_image_tensor();
    let b = a.clone(); // Shallow clone of Arc
    assert_zero_copy(&a, &b);
    
    let c = a.to_contiguous(); // Is contiguous, so returns self (cheap clone)
    assert_zero_copy(&a, &c);
}