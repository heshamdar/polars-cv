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
    let list_list_f32 = ArrowDataType::List(Box::new(Field::new(
        "item".into(),
        list_f32.clone(),
        false,
    )));
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
    let list_list_u8 = ArrowDataType::List(Box::new(Field::new(
        "item".into(),
        list_u8.clone(),
        false,
    )));
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

