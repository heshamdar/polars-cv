use view_buffer::ops::scalar::{FusedKernel, ScalarOp};
use view_buffer::{DType, ViewBuffer, ViewExpr};

#[test]
fn test_fused_execution_f32() {
    // 1. Setup Input: [1.0, -2.0, 3.0, 4.0]
    let input_data = vec![1.0f32, -2.0, 3.0, 4.0];
    let buf = ViewBuffer::from_vec(input_data);

    // 2. Define Kernel: (x * 2.0) + 1.0 -> Relu
    // Expected:
    // 1.0 -> 2.0 -> 3.0 -> 3.0
    // -2.0 -> -4.0 -> -3.0 -> 0.0 (Relu)
    // 3.0 -> 6.0 -> 7.0 -> 7.0
    // 4.0 -> 8.0 -> 9.0 -> 9.0
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Mul(2.0));
    kernel.push(ScalarOp::Add(1.0));
    kernel.push(ScalarOp::Relu);

    // 3. Execute
    let result = buf.apply_fused_kernel(&kernel);

    // 4. Verify
    assert_eq!(result.dtype(), DType::F32);
    assert!(result.layout_facts().is_contiguous());

    // We need to inspect values.
    // Since as_slice is not exposed directly for generic types safely yet,
    // we use a little unsafe helper or cast.
    // For this test, let's use the raw pointer since we know it's contiguous F32.
    let (ptr, _, _, _) = result.as_raw_parts();
    let result_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, 4) };

    assert_eq!(result_slice, &[3.0, 0.0, 7.0, 9.0]);
}

#[test]
fn test_fused_on_strided_input() {
    // 1. Input 2x2: [[1.0, 2.0], [3.0, 4.0]]
    //    Strides: [8, 4] bytes
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buf = ViewExpr::new_source(ViewBuffer::from_vec(input_data))
        .reshape(vec![2, 2])
        .plan()
        .execute();

    // 2. Transpose -> [[1.0, 3.0], [2.0, 4.0]]
    //    Strides: [4, 8] bytes.
    let transposed = buf.permute(&[1, 0]);

    // 3. Define Kernel: Add(10.0)
    // Expected Output (Contiguous): [11.0, 13.0, 12.0, 14.0]
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Add(10.0));

    // 4. Execute
    let result = transposed.apply_fused_kernel(&kernel);

    // 5. Verify
    assert!(result.layout_facts().is_contiguous());
    assert_eq!(result.shape(), &[2, 2]);

    let (ptr, _, _, _) = result.as_raw_parts();
    let result_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, 4) };

    // Row-major output of the transposed input
    assert_eq!(result_slice, &[11.0, 13.0, 12.0, 14.0]);
}
