//! Unit tests for polars interoperability module.
//!
//! These tests verify zero-copy buffer extraction, dtype detection,
//! shape inference, and contiguity validation from Polars types.

#![cfg(feature = "polars_interop")]

use polars_arrow::buffer::Buffer;
use polars_arrow::datatypes::{ArrowDataType, Field};
use view_buffer::core::dtype::DType;
use view_buffer::interop::polars::{
    dtype_from_polars, fixed_shape_from_type, is_type_potentially_contiguous, nesting_depth,
    PolarsBufferRef,
};

// ============================================================
// PolarsBufferRef Tests
// ============================================================

#[test]
fn test_polars_buffer_ref_new_valid() {
    let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let buffer = Buffer::from(data);

    // Full buffer reference
    let buf_ref = PolarsBufferRef::new(buffer.clone(), 0, 10);
    assert!(buf_ref.is_some());
    let buf_ref = buf_ref.unwrap();
    assert_eq!(buf_ref.len(), 10);
    assert!(!buf_ref.is_empty());
    assert_eq!(buf_ref.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    // Partial buffer reference
    let buf_ref = PolarsBufferRef::new(buffer.clone(), 3, 4);
    assert!(buf_ref.is_some());
    let buf_ref = buf_ref.unwrap();
    assert_eq!(buf_ref.as_slice(), &[3, 4, 5, 6]);

    // Empty reference is valid
    let buf_ref = PolarsBufferRef::new(buffer.clone(), 5, 0);
    assert!(buf_ref.is_some());
    assert!(buf_ref.unwrap().is_empty());

    // Reference at end of buffer
    let buf_ref = PolarsBufferRef::new(buffer, 10, 0);
    assert!(buf_ref.is_some());
}

#[test]
fn test_polars_buffer_ref_new_out_of_bounds() {
    let data: Vec<u8> = vec![0, 1, 2, 3, 4];
    let buffer = Buffer::from(data);

    // Offset exceeds buffer
    let buf_ref = PolarsBufferRef::new(buffer.clone(), 6, 0);
    assert!(buf_ref.is_none());

    // Length exceeds remaining buffer
    let buf_ref = PolarsBufferRef::new(buffer.clone(), 3, 5);
    assert!(buf_ref.is_none());

    // Combined overflow
    let buf_ref = PolarsBufferRef::new(buffer, 10, 10);
    assert!(buf_ref.is_none());
}

#[test]
fn test_polars_buffer_ref_to_view_buffer() {
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let buffer = Buffer::from(data);

    let buf_ref = PolarsBufferRef::new(buffer, 0, 8).unwrap();

    // Create a 2x4 u8 array
    let view_buffer = buf_ref.to_view_buffer(vec![2, 4], DType::U8);

    assert_eq!(view_buffer.shape(), &[2, 4]);
    assert_eq!(view_buffer.dtype(), DType::U8);
    assert_eq!(view_buffer.as_slice::<u8>(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_polars_buffer_ref_try_to_view_buffer_size_mismatch() {
    let data: Vec<u8> = vec![1, 2, 3, 4];
    let buffer = Buffer::from(data);

    let buf_ref = PolarsBufferRef::new(buffer, 0, 4).unwrap();

    // Shape requires 8 bytes but buffer only has 4
    let result = buf_ref.try_to_view_buffer(vec![2, 4], DType::U8);
    assert!(result.is_err());
}

// ============================================================
// dtype_from_polars Tests
// ============================================================

#[test]
fn test_dtype_from_polars_primitives() {
    assert_eq!(dtype_from_polars(&ArrowDataType::UInt8), Some(DType::U8));
    assert_eq!(dtype_from_polars(&ArrowDataType::Int8), Some(DType::I8));
    assert_eq!(dtype_from_polars(&ArrowDataType::UInt16), Some(DType::U16));
    assert_eq!(dtype_from_polars(&ArrowDataType::Int16), Some(DType::I16));
    assert_eq!(dtype_from_polars(&ArrowDataType::UInt32), Some(DType::U32));
    assert_eq!(dtype_from_polars(&ArrowDataType::Int32), Some(DType::I32));
    assert_eq!(dtype_from_polars(&ArrowDataType::UInt64), Some(DType::U64));
    assert_eq!(dtype_from_polars(&ArrowDataType::Int64), Some(DType::I64));
    assert_eq!(dtype_from_polars(&ArrowDataType::Float32), Some(DType::F32));
    assert_eq!(dtype_from_polars(&ArrowDataType::Float64), Some(DType::F64));
}

#[test]
fn test_dtype_from_polars_binary() {
    assert_eq!(dtype_from_polars(&ArrowDataType::Binary), Some(DType::U8));
    assert_eq!(
        dtype_from_polars(&ArrowDataType::LargeBinary),
        Some(DType::U8)
    );
}

#[test]
fn test_dtype_from_polars_nested_list() {
    // List[UInt8] -> U8
    let list_u8 = ArrowDataType::List(Box::new(Field::new(
        "item".into(),
        ArrowDataType::UInt8,
        false,
    )));
    assert_eq!(dtype_from_polars(&list_u8), Some(DType::U8));

    // List[List[Float32]] -> F32
    let list_f32 = ArrowDataType::List(Box::new(Field::new(
        "item".into(),
        ArrowDataType::Float32,
        false,
    )));
    let list_list_f32 =
        ArrowDataType::List(Box::new(Field::new("item".into(), list_f32.clone(), false)));
    assert_eq!(dtype_from_polars(&list_list_f32), Some(DType::F32));
}

#[test]
fn test_dtype_from_polars_fixed_size_list() {
    // FixedSizeList[Int32, 3] -> I32
    let fixed_i32 = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), ArrowDataType::Int32, false)),
        3,
    );
    assert_eq!(dtype_from_polars(&fixed_i32), Some(DType::I32));
}

#[test]
fn test_dtype_from_polars_unsupported() {
    assert_eq!(dtype_from_polars(&ArrowDataType::Utf8), None);
    assert_eq!(dtype_from_polars(&ArrowDataType::Boolean), None);
    assert_eq!(dtype_from_polars(&ArrowDataType::Date32), None);
}

// ============================================================
// nesting_depth Tests
// ============================================================

#[test]
fn test_nesting_depth_primitives() {
    assert_eq!(nesting_depth(&ArrowDataType::UInt8), 0);
    assert_eq!(nesting_depth(&ArrowDataType::Float64), 0);
    assert_eq!(nesting_depth(&ArrowDataType::Binary), 0);
}

#[test]
fn test_nesting_depth_lists() {
    // List[UInt8] -> depth 1
    let list_u8 = ArrowDataType::List(Box::new(Field::new(
        "item".into(),
        ArrowDataType::UInt8,
        false,
    )));
    assert_eq!(nesting_depth(&list_u8), 1);

    // List[List[UInt8]] -> depth 2
    let list_list_u8 =
        ArrowDataType::List(Box::new(Field::new("item".into(), list_u8.clone(), false)));
    assert_eq!(nesting_depth(&list_list_u8), 2);

    // List[List[List[UInt8]]] -> depth 3
    let list_list_list_u8 = ArrowDataType::List(Box::new(Field::new(
        "item".into(),
        list_list_u8.clone(),
        false,
    )));
    assert_eq!(nesting_depth(&list_list_list_u8), 3);
}

#[test]
fn test_nesting_depth_fixed_size() {
    // FixedSizeList[UInt8, 3] -> depth 1
    let fixed_u8 = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), ArrowDataType::UInt8, false)),
        3,
    );
    assert_eq!(nesting_depth(&fixed_u8), 1);

    // FixedSizeList[FixedSizeList[UInt8, 3], 4] -> depth 2
    let nested_fixed = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), fixed_u8.clone(), false)),
        4,
    );
    assert_eq!(nesting_depth(&nested_fixed), 2);
}

// ============================================================
// fixed_shape_from_type Tests
// ============================================================

#[test]
fn test_fixed_shape_from_type_primitives() {
    // Primitives have no shape
    let empty: Vec<usize> = vec![];
    assert_eq!(fixed_shape_from_type(&ArrowDataType::UInt8), empty);
    assert_eq!(fixed_shape_from_type(&ArrowDataType::Float32), empty);
}

#[test]
fn test_fixed_shape_from_type_single_level() {
    // FixedSizeList[UInt8, 3] -> [3]
    let fixed_u8_3 = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), ArrowDataType::UInt8, false)),
        3,
    );
    assert_eq!(fixed_shape_from_type(&fixed_u8_3), vec![3]);
}

#[test]
fn test_fixed_shape_from_type_nested() {
    // FixedSizeList[FixedSizeList[UInt8, 3], 4] -> [4, 3]
    let fixed_u8_3 = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), ArrowDataType::UInt8, false)),
        3,
    );
    let nested = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), fixed_u8_3.clone(), false)),
        4,
    );
    assert_eq!(fixed_shape_from_type(&nested), vec![4, 3]);

    // FixedSizeList[FixedSizeList[FixedSizeList[UInt8, 3], 4], 2] -> [2, 4, 3]
    let triple_nested = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), nested.clone(), false)),
        2,
    );
    assert_eq!(fixed_shape_from_type(&triple_nested), vec![2, 4, 3]);
}

#[test]
fn test_fixed_shape_from_type_variable_list() {
    // Variable-size List doesn't contribute to fixed shape
    let list_u8 = ArrowDataType::List(Box::new(Field::new(
        "item".into(),
        ArrowDataType::UInt8,
        false,
    )));
    let empty: Vec<usize> = vec![];
    assert_eq!(fixed_shape_from_type(&list_u8), empty);
}

// ============================================================
// is_type_potentially_contiguous Tests
// ============================================================

#[test]
fn test_is_contiguous_primitives() {
    assert!(is_type_potentially_contiguous(&ArrowDataType::UInt8));
    assert!(is_type_potentially_contiguous(&ArrowDataType::Float32));
    assert!(is_type_potentially_contiguous(&ArrowDataType::Int64));
}

#[test]
fn test_is_contiguous_binary() {
    assert!(is_type_potentially_contiguous(&ArrowDataType::Binary));
    assert!(is_type_potentially_contiguous(&ArrowDataType::LargeBinary));
}

#[test]
fn test_is_contiguous_fixed_size_list() {
    let fixed_u8 = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), ArrowDataType::UInt8, false)),
        3,
    );
    assert!(is_type_potentially_contiguous(&fixed_u8));

    // Nested FixedSizeList
    let nested = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), fixed_u8.clone(), false)),
        4,
    );
    assert!(is_type_potentially_contiguous(&nested));
}

#[test]
fn test_is_contiguous_variable_list() {
    // Variable-size list can be contiguous if data happens to be rectangular
    let list_u8 = ArrowDataType::List(Box::new(Field::new(
        "item".into(),
        ArrowDataType::UInt8,
        false,
    )));
    assert!(is_type_potentially_contiguous(&list_u8));
}

#[test]
fn test_is_contiguous_unsupported() {
    // String types are not contiguous in the tensor sense
    assert!(!is_type_potentially_contiguous(&ArrowDataType::Utf8));
    assert!(!is_type_potentially_contiguous(&ArrowDataType::Boolean));
}

// ============================================================
// ViewBuffer integration tests
// ============================================================

#[test]
fn test_view_buffer_from_polars_buffer_basic() {
    use view_buffer::ViewBuffer;

    let data: Vec<u8> = (0..24).collect();
    let buffer = Buffer::from(data);

    // Create a 2x3x4 u8 tensor
    let view = ViewBuffer::from_polars_buffer(buffer.clone(), 0, vec![2, 3, 4], DType::U8);

    assert_eq!(view.shape(), &[2, 3, 4]);
    assert_eq!(view.dtype(), DType::U8);

    // Verify data integrity
    let slice = view.as_slice::<u8>();
    assert_eq!(slice.len(), 24);
    assert_eq!(slice[0], 0);
    assert_eq!(slice[23], 23);
}

#[test]
fn test_view_buffer_from_polars_buffer_with_offset() {
    use view_buffer::ViewBuffer;

    let data: Vec<u8> = (0..100).collect();
    let buffer = Buffer::from(data);

    // Create a 4x4 u8 tensor starting at offset 10
    let view = ViewBuffer::from_polars_buffer(buffer.clone(), 10, vec![4, 4], DType::U8);

    assert_eq!(view.shape(), &[4, 4]);
    assert_eq!(view.dtype(), DType::U8);

    // Verify data starts at offset
    let slice = view.as_slice::<u8>();
    assert_eq!(slice.len(), 16);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[15], 25);
}

#[test]
fn test_view_buffer_from_polars_buffer_f32() {
    use view_buffer::ViewBuffer;

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let buffer = Buffer::from(bytes);

    // Create a 2x3 f32 tensor
    let view = ViewBuffer::from_polars_buffer(buffer, 0, vec![2, 3], DType::F32);

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.dtype(), DType::F32);

    let slice = view.as_slice::<f32>();
    assert_eq!(slice.len(), 6);
    assert!((slice[0] - 1.0).abs() < 1e-6);
    assert!((slice[5] - 6.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "Polars buffer too small")]
fn test_view_buffer_from_polars_buffer_too_small() {
    use view_buffer::ViewBuffer;

    let data: Vec<u8> = vec![1, 2, 3, 4];
    let buffer = Buffer::from(data);

    // Trying to create 10 element tensor from 4 byte buffer should panic
    let _ = ViewBuffer::from_polars_buffer(buffer, 0, vec![10], DType::U8);
}

#[test]
fn test_view_buffer_storage_id_consistency() {
    use view_buffer::ViewBuffer;

    let data: Vec<u8> = (0..100).collect();
    let buffer = Buffer::from(data);

    // Two views into the same buffer with different offsets
    let view1 = ViewBuffer::from_polars_buffer(buffer.clone(), 0, vec![10], DType::U8);
    let view2 = ViewBuffer::from_polars_buffer(buffer.clone(), 10, vec![10], DType::U8);
    let view3 = ViewBuffer::from_polars_buffer(buffer.clone(), 0, vec![10], DType::U8);

    // Different offsets should give different storage IDs
    assert_ne!(view1.storage_id(), view2.storage_id());

    // Same offset should give same storage ID (for zero-copy verification)
    assert_eq!(view1.storage_id(), view3.storage_id());
}

// ============================================================
// SlicePolicy and Zero-Copy Transfer Tests
// ============================================================

#[test]
fn test_slice_policy_default() {
    use view_buffer::SlicePolicy;

    let policy = SlicePolicy::default();
    assert_eq!(policy, SlicePolicy::Heuristic { threshold: 0.5 });
}

#[test]
fn test_zero_copy_transfer_full_buffer() {
    use view_buffer::ViewBuffer;

    // Full buffer with offset 0 should always zero-copy
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let buffer = ViewBuffer::from_vec(data);

    // Should be eligible for zero-copy
    assert!(buffer.can_zero_copy_transfer());

    // Transfer and verify
    let (polars_buf, shape, dtype) = buffer.into_polars_buffer();
    assert_eq!(polars_buf.len(), 8);
    assert_eq!(shape, vec![8]);
    assert_eq!(dtype, DType::U8);
}

#[test]
fn test_zero_copy_transfer_with_offset_always_zero_copy_policy() {
    use view_buffer::{SlicePolicy, ViewBuffer};

    // Create a 1D buffer and slice it (introduces offset but stays contiguous)
    // 1D slices remain contiguous because there's no row stride to worry about
    let data: Vec<u8> = (0..100).collect();
    let buffer = ViewBuffer::from_vec(data);

    // Slice to get offset > 0 - 1D slice stays contiguous
    let sliced = buffer.slice(&[20], &[80]);
    assert_eq!(sliced.shape(), &[60]);

    // Note: slice() clones the Arc, so there are now 2 references.
    // We need to drop the original to get sole ownership.
    drop(buffer);

    // With AlwaysZeroCopy policy, contiguous buffer with offset should be zero-copy eligible
    assert!(sliced.can_zero_copy_transfer_with_policy(SlicePolicy::AlwaysZeroCopy));

    // Transfer with AlwaysZeroCopy policy
    let (polars_buf, shape, dtype) =
        sliced.into_polars_buffer_with_policy(SlicePolicy::AlwaysZeroCopy);
    assert_eq!(polars_buf.len(), 60);
    assert_eq!(shape, vec![60]);
    assert_eq!(dtype, DType::U8);

    // Verify the data starts at the right offset
    assert_eq!(polars_buf.as_slice()[0], 20);
    assert_eq!(polars_buf.as_slice()[59], 79);
}

#[test]
fn test_zero_copy_transfer_with_offset_always_copy_policy() {
    use view_buffer::{SlicePolicy, ViewBuffer};

    // Create a 1D buffer and slice it (1D slices stay contiguous)
    let data: Vec<u8> = (0..100).collect();
    let buffer = ViewBuffer::from_vec(data);

    // Slice to get offset > 0
    let sliced = buffer.slice(&[20], &[80]);
    assert_eq!(sliced.shape(), &[60]);

    // With AlwaysCopy policy, offset > 0 means NOT zero-copy eligible
    // (even though the buffer is contiguous)
    assert!(!sliced.can_zero_copy_transfer_with_policy(SlicePolicy::AlwaysCopy));

    // But full buffer should still be zero-copy with AlwaysCopy
    let full_data: Vec<u8> = (0..100).collect();
    let full_buffer = ViewBuffer::from_vec(full_data);
    assert!(full_buffer.can_zero_copy_transfer_with_policy(SlicePolicy::AlwaysCopy));
}

#[test]
fn test_zero_copy_transfer_heuristic_policy() {
    use view_buffer::{SlicePolicy, ViewBuffer};

    // Test 1: Large slice (60% of buffer) - should be zero-copy with 50% threshold
    {
        let data: Vec<u8> = (0..100).collect();
        let buffer = ViewBuffer::from_vec(data);
        let large_slice = buffer.slice(&[20], &[80]);
        assert_eq!(large_slice.shape(), &[60]);

        // Drop original to get sole ownership of Arc
        drop(buffer);

        assert!(large_slice
            .can_zero_copy_transfer_with_policy(SlicePolicy::Heuristic { threshold: 0.5 }));
    }

    // Test 2: Small slice (30% of buffer) - should NOT be zero-copy with 50% threshold
    {
        let data: Vec<u8> = (0..100).collect();
        let buffer = ViewBuffer::from_vec(data);
        let small_slice = buffer.slice(&[10], &[40]);
        assert_eq!(small_slice.shape(), &[30]);

        // Drop original to get sole ownership of Arc
        drop(buffer);

        // 30% < 50% threshold, so should NOT be zero-copy
        assert!(!small_slice
            .can_zero_copy_transfer_with_policy(SlicePolicy::Heuristic { threshold: 0.5 }));

        // But should be zero-copy with 25% threshold (30% >= 25%)
        assert!(small_slice
            .can_zero_copy_transfer_with_policy(SlicePolicy::Heuristic { threshold: 0.25 }));
    }
}

#[test]
fn test_into_polars_buffer_preserves_data_with_offset() {
    use view_buffer::{SlicePolicy, ViewBuffer};

    // Create a 1D buffer with known data - 1D slices remain contiguous
    let data: Vec<u8> = (0..100).collect();
    let buffer = ViewBuffer::from_vec(data);

    // Slice from [25] to [50] - a contiguous 25-element region
    let sliced = buffer.slice(&[25], &[50]);
    assert_eq!(sliced.shape(), &[25]);

    // Transfer with AlwaysZeroCopy - should work because 1D slice is contiguous
    let (polars_buf, shape, _dtype) =
        sliced.into_polars_buffer_with_policy(SlicePolicy::AlwaysZeroCopy);

    assert_eq!(shape, vec![25]);
    assert_eq!(polars_buf.len(), 25);

    // Verify data integrity - should start at 25 and go to 49
    assert_eq!(polars_buf.as_slice()[0], 25);
    assert_eq!(polars_buf.as_slice()[24], 49);
}

#[test]
fn test_into_polars_buffer_with_non_contiguous_slice() {
    use view_buffer::ViewBuffer;

    // Create a 2D buffer and slice it - 2D slice is NOT contiguous
    let data: Vec<u8> = (0..100).collect();
    let buffer = ViewBuffer::from_vec(data).reshape(vec![10, 10]);

    // Slice from [2,2] to [5,5] - a 3x3 region (NOT contiguous due to row strides)
    let sliced = buffer.slice(&[2, 2], &[5, 5]);
    assert_eq!(sliced.shape(), &[3, 3]);

    // This should NOT be zero-copy eligible because it's not contiguous
    assert!(!sliced.can_zero_copy_transfer());

    // But into_polars_buffer should still work (via copy/materialization)
    let (polars_buf, shape, _dtype) = sliced.into_polars_buffer();

    assert_eq!(shape, vec![3, 3]);
    assert_eq!(polars_buf.len(), 9);
}

#[test]
fn test_polars_arrow_storage_zero_copy() {
    use view_buffer::ViewBuffer;

    // Create a ViewBuffer from a polars buffer
    let data: Vec<u8> = (0..100).collect();
    let polars_buffer = Buffer::from(data);

    let view = ViewBuffer::from_polars_buffer(polars_buffer, 10, vec![20], DType::U8);

    // PolarsArrow storage should always be zero-copy eligible (when contiguous)
    assert!(view.can_zero_copy_transfer());

    // Transfer back to polars buffer
    let (result_buf, shape, dtype) = view.into_polars_buffer();
    assert_eq!(result_buf.len(), 20);
    assert_eq!(shape, vec![20]);
    assert_eq!(dtype, DType::U8);

    // Verify the data is correct (starts at offset 10)
    assert_eq!(result_buf.as_slice()[0], 10);
    assert_eq!(result_buf.as_slice()[19], 29);
}

#[test]
fn test_zero_copy_with_reshape() {
    use view_buffer::ViewBuffer;

    // Reshape doesn't change offset, should still be zero-copy
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
    let buffer = ViewBuffer::from_vec(data).reshape(vec![2, 3]);

    assert!(buffer.can_zero_copy_transfer());

    let (polars_buf, shape, dtype) = buffer.into_polars_buffer();
    assert_eq!(polars_buf.len(), 6);
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(dtype, DType::U8);
}

#[test]
fn test_non_contiguous_buffer_copies() {
    use view_buffer::ViewBuffer;

    // Create a buffer and permute it to make it non-contiguous
    let data: Vec<u8> = (0..12).collect();
    let buffer = ViewBuffer::from_vec(data).reshape(vec![3, 4]);

    // Permute dimensions - makes it non-contiguous
    let permuted = buffer.permute(&[1, 0]);
    assert_eq!(permuted.shape(), &[4, 3]);

    // Non-contiguous should NOT be zero-copy eligible
    assert!(!permuted.can_zero_copy_transfer());

    // But into_polars_buffer should still work (via copy)
    let (polars_buf, shape, dtype) = permuted.into_polars_buffer();
    assert_eq!(polars_buf.len(), 12);
    assert_eq!(shape, vec![4, 3]);
    assert_eq!(dtype, DType::U8);
}

#[test]
fn test_slice_policy_with_polars_arrow_storage() {
    use view_buffer::{SlicePolicy, ViewBuffer};

    // PolarsArrow storage should always use zero-copy regardless of policy
    let data: Vec<u8> = (0..100).collect();
    let polars_buffer = Buffer::from(data);

    let view = ViewBuffer::from_polars_buffer(polars_buffer, 10, vec![20], DType::U8);

    // All policies should report zero-copy for PolarsArrow storage
    assert!(view.can_zero_copy_transfer_with_policy(SlicePolicy::AlwaysZeroCopy));
    assert!(view.can_zero_copy_transfer_with_policy(SlicePolicy::AlwaysCopy));
    assert!(view.can_zero_copy_transfer_with_policy(SlicePolicy::Heuristic { threshold: 0.99 }));
}

// ============================================================
// Buffer Registration / BinaryViewArray Tests
// ============================================================

#[test]
fn test_buffer_can_be_registered_for_binary_view() {
    use view_buffer::ViewBuffer;

    // Create a buffer that would be used as image data
    let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let buffer = ViewBuffer::from_vec(data);

    // Get the polars buffer
    let (polars_buf, _shape, _dtype) = buffer.into_polars_buffer();

    // Verify the buffer can be cloned (Arc-based) for registration
    let cloned = polars_buf.clone();
    assert_eq!(polars_buf.len(), cloned.len());
    assert_eq!(polars_buf.as_slice(), cloned.as_slice());

    // Verify the data pointer is the same (true zero-copy)
    assert_eq!(
        polars_buf.as_ptr(),
        cloned.as_ptr(),
        "Buffer clone should share memory"
    );
}

#[test]
fn test_multiple_buffers_for_registration() {
    use view_buffer::ViewBuffer;

    // Create multiple buffers that would represent different rows
    let buffers: Vec<Buffer<u8>> = (0..5)
        .map(|i| {
            let data: Vec<u8> = (0..100).map(|j| ((i * 100 + j) % 256) as u8).collect();
            let vb = ViewBuffer::from_vec(data);
            let (polars_buf, _, _) = vb.into_polars_buffer();
            polars_buf
        })
        .collect();

    // Each buffer should be independent
    assert_eq!(buffers.len(), 5);
    for buf in &buffers {
        assert_eq!(buf.len(), 100);
    }

    // Different buffers should have different data pointers
    let ptrs: Vec<*const u8> = buffers.iter().map(|b| b.as_ptr()).collect();
    for i in 0..ptrs.len() {
        for j in (i + 1)..ptrs.len() {
            assert_ne!(
                ptrs[i], ptrs[j],
                "Different buffers should have different pointers"
            );
        }
    }
}

#[test]
fn test_buffer_len_for_view_creation() {
    use view_buffer::ViewBuffer;

    // Test various sizes including edge cases
    let test_sizes = vec![0, 1, 12, 13, 100, 1000, 10000];

    for size in test_sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let buffer = ViewBuffer::from_vec(data);
        let (polars_buf, _, _) = buffer.into_polars_buffer();

        assert_eq!(
            polars_buf.len(),
            size,
            "Buffer size mismatch for size {size}"
        );

        // Verify data integrity
        for (i, &byte) in polars_buf.as_slice().iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data mismatch at index {i}");
        }
    }
}

#[test]
fn test_large_buffer_for_view_registration() {
    use view_buffer::ViewBuffer;

    // Create a large buffer simulating a high-resolution image (1920x1080x3)
    let width = 1920;
    let height = 1080;
    let channels = 3;
    let size = width * height * channels;

    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    let buffer = ViewBuffer::from_vec(data).reshape(vec![height, width, channels]);

    assert_eq!(buffer.shape(), &[height, width, channels]);

    let (polars_buf, shape, dtype) = buffer.into_polars_buffer();

    assert_eq!(polars_buf.len(), size);
    assert_eq!(shape, vec![height, width, channels]);
    assert_eq!(dtype, DType::U8);
}

#[test]
fn test_view_buffer_to_polars_buffer_arc_count() {
    use view_buffer::ViewBuffer;

    // Create a buffer
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let buffer = ViewBuffer::from_vec(data);

    // Get polars buffer and clone it
    let (polars_buf, _, _) = buffer.into_polars_buffer();
    let cloned1 = polars_buf.clone();
    let cloned2 = polars_buf.clone();

    // All clones should share the same underlying data
    assert_eq!(polars_buf.as_ptr(), cloned1.as_ptr());
    assert_eq!(polars_buf.as_ptr(), cloned2.as_ptr());
}
