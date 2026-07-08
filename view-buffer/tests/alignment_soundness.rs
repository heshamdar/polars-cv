//! Allocator-contract soundness of ViewBuffer's owned byte storage.
//!
//! Kernel outputs are built as `Vec<T>` and stored as raw bytes. The
//! storage must deallocate with the same `Layout` (size AND alignment)
//! the allocation was created with; rebuilding a `Vec<u8>` (align 1)
//! over a `Vec<f32>`/`Vec<f64>` allocation and dropping it is undefined
//! behavior per the allocator contract. These tests are ordinary
//! roundtrips in a normal build, but under Miri
//! (`cargo +nightly miri test --test alignment_soundness`) they fail on
//! the mismatched dealloc.

use view_buffer::{DType, ViewBuffer};

#[test]
fn from_vec_f32_drops_with_original_alignment() {
    let buf = ViewBuffer::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
    assert_eq!(buf.dtype(), DType::F32);
    assert_eq!(buf.to_contiguous().as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
    drop(buf);
}

#[test]
fn from_vec_with_shape_f64_drops_with_original_alignment() {
    let buf = ViewBuffer::from_vec_with_shape(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(buf.dtype(), DType::F64);
    drop(buf);
}

#[test]
fn from_vec_u16_clone_and_drop() {
    let buf = ViewBuffer::from_vec(vec![1u16, 2, 3]);
    let clone = buf.clone();
    drop(buf);
    assert_eq!(clone.to_contiguous().as_slice::<u16>(), &[1, 2, 3]);
}

#[test]
fn from_slice_aligned_keeps_custom_alignment() {
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let buf = ViewBuffer::from_slice_aligned(&data, 64);
    assert_eq!(buf.shape(), &[64]);
    // The allocation really is 64-byte aligned (SIMD contract).
    assert_eq!(unsafe { buf.as_ptr::<f32>() } as usize % 64, 0);
    assert_eq!(buf.to_contiguous().as_slice::<f32>(), &data[..]);
}

#[test]
fn empty_typed_vec_is_sound() {
    let buf = ViewBuffer::from_vec(Vec::<f64>::new());
    assert_eq!(buf.shape(), &[0]);
    drop(buf);
}

#[test]
fn owned_bytes_roundtrip_still_works() {
    let buf = ViewBuffer::from_vec(vec![1u8, 2, 3, 4]);
    let owned = buf.try_into_owned_bytes().expect("sole-owner u8 unwrap");
    assert_eq!(owned, vec![1, 2, 3, 4]);
}
