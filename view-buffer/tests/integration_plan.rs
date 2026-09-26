//! Integration tests for pipeline planning and execution.

use view_buffer::{DType, ViewBuffer, ViewExpr};

#[test]
fn test_blob_roundtrip() {
    // 1. Create a non-contiguous buffer (slice) to verify normalization
    // Data: [0, 1, 10, 11, 20, 21, 30, 31]
    let data: Vec<f32> = vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
    // Shape [4, 2]
    let buf = ViewBuffer::from_vec(data);

    // We will use the public API via ViewExpr to create a reshaped result.
    let buf = ViewExpr::new_source(buf)
        .reshape(vec![4, 2])
        .plan()
        .execute();

    // Slice first column: [0, 10, 20, 30]. Strides will be non-default.
    let slice = buf.slice(&[0, 0], &[4, 1]);

    // 2. Serialize to Blob
    // This forces materialization to contiguous memory in the blob
    let blob = slice.to_blob();

    // Basic Header Checks
    assert!(blob.len() > 64);
    assert_eq!(&blob[0..4], b"VIEW");

    // 3. Deserialize
    let recovered = ViewBuffer::from_blob(&blob).expect("Failed to deserialize blob");

    // 4. Verify
    assert!(recovered.layout_facts().is_contiguous());
    assert_eq!(recovered.shape(), &[4, 1]);
    assert_eq!(recovered.dtype(), DType::F32);

    let (ptr, _, _, _) = recovered.as_raw_parts();
    let result_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, 4) };
    assert_eq!(result_slice, &[0.0, 10.0, 20.0, 30.0]);
}

#[test]
fn test_plan_execution() {
    use view_buffer::ops::ComputeOp;
    use view_buffer::ViewDto;

    // Source [1.0, 2.0, 3.0, 4.0]; plan: Scale(2.0) -> Relu.
    let source = ViewBuffer::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0]);
    let ops = vec![
        ViewDto::Compute(ComputeOp::Scale { factor: 2.0 }),
        ViewDto::Compute(ComputeOp::Relu),
    ];
    let result = view_buffer::execute_plan(source, ops);

    let (ptr, _, _, _) = result.as_raw_parts();
    let res_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, 4) };
    assert_eq!(res_slice, &[2.0, 4.0, 6.0, 8.0]);
}
