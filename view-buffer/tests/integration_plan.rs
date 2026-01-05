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

#[cfg(feature = "serde")]
#[test]
fn test_plan_execution_from_json() {
    use view_buffer::ops::{ComputeOp, FusedKernel, ScalarOp};
    use view_buffer::ViewDto;

    // 1. Source View: [1.0, 2.0, 3.0, 4.0]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let source = ViewBuffer::from_vec(data);

    // 2. Define Plan using serializable DTOs
    // Plan: Scale(2.0) -> Relu -> Materialize
    let ops = vec![
        ViewDto::Compute(ComputeOp::Scale(2.0)),
        ViewDto::Compute(ComputeOp::Relu),
        ViewDto::Materialize,
    ];

    // Simulate JSON Transport
    let json_plan = serde_json::to_string(&ops).expect("Failed to serialize plan");
    println!("JSON Plan: {json_plan}");

    // 3. Deserialize Plan
    let deserialized_ops: Vec<ViewDto> =
        serde_json::from_str(&json_plan).expect("Failed to deserialize plan");

    // 4. Execute Engine
    let result = view_buffer::execute_plan(source, deserialized_ops);

    // 5. Verify Result: [2.0, 4.0, 6.0, 8.0]
    let (ptr, _, _, _) = result.as_raw_parts();
    let res_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, 4) };
    assert_eq!(res_slice, &[2.0, 4.0, 6.0, 8.0]);

    // Suppress unused warnings for types used in serde tests
    let _ = FusedKernel::new();
    let _ = ScalarOp::Relu;
}

#[cfg(feature = "serde")]
#[test]
fn test_fused_op_serialization() {
    use view_buffer::ops::{ComputeOp, FusedKernel, ScalarOp};
    use view_buffer::ViewDto;

    // Verify that the FusedKernel struct serializes correctly
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Add(10.0));
    kernel.push(ScalarOp::Mul(0.5));

    let op = ViewDto::Compute(ComputeOp::Fused(kernel));

    let json = serde_json::to_string(&op).unwrap();

    let deserialized: ViewDto = serde_json::from_str(&json).unwrap();

    if let ViewDto::Compute(ComputeOp::Fused(k)) = deserialized {
        assert_eq!(k.ops.len(), 2);
        assert_eq!(k.ops[0], ScalarOp::Add(10.0));
    } else {
        panic!("Deserialization mismatch");
    }
}
