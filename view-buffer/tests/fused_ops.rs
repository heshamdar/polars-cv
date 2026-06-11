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

// ============================================================================
// Fusion-equivalence tests
//
// The optimizer may fuse adjacent scalar/cast ops into a single FusedKernel
// pass (with the input conversion folded into the read and the output
// conversion folded into the write). Whatever it decides, the fused plan
// must produce exactly the same values AND dtype as executing every op as
// its own single-op plan (which can never fuse). The kernel's out_dtype is
// pinned to the chain's planned dtype, so these tests also guard the
// plan == exec contract for fused chains.
// ============================================================================

use std::sync::Arc;
use view_buffer::execution::PlanStep;
use view_buffer::ComputeOp;

/// Execute `build` as ONE expression (fusion allowed).
fn run_fused(
    buf: &ViewBuffer,
    build: impl Fn(Arc<ViewExpr>) -> Arc<ViewExpr>,
) -> (ViewBuffer, usize) {
    let expr = build(ViewExpr::new_source(buf.clone()));
    let plan = expr.plan();
    let fused_kernels = plan
        .steps
        .iter()
        .filter(|s| matches!(s, PlanStep::Compute(ComputeOp::Fused(_))))
        .count();
    (plan.execute(), fused_kernels)
}

/// Execute each op as its own single-op plan (fusion impossible).
fn run_stepwise(buf: &ViewBuffer, steps: &[&dyn Fn(Arc<ViewExpr>) -> Arc<ViewExpr>]) -> ViewBuffer {
    let mut current = buf.clone();
    for step in steps {
        current = step(ViewExpr::new_source(current)).plan().execute();
    }
    current
}

fn assert_buffers_equal(fused: &ViewBuffer, stepwise: &ViewBuffer) {
    assert_eq!(fused.dtype(), stepwise.dtype(), "fusion changed the dtype");
    assert_eq!(fused.shape(), stepwise.shape(), "fusion changed the shape");
    let a = fused.to_contiguous();
    let b = stepwise.to_contiguous();
    let n: usize = a.shape().iter().product();
    let elem = a.dtype().size_of();
    let (pa, _, _, _) = a.as_raw_parts();
    let (pb, _, _, _) = b.as_raw_parts();
    let ba = unsafe { std::slice::from_raw_parts(pa, n * elem) };
    let bb = unsafe { std::slice::from_raw_parts(pb, n * elem) };
    assert_eq!(ba, bb, "fused values differ from unfused values");
}

fn u8_ramp() -> ViewBuffer {
    // Covers extremes and mid values.
    ViewBuffer::from_vec((0..=255u8).collect::<Vec<u8>>())
}

#[test]
fn canonical_ml_chain_fuses_into_one_kernel() {
    // u8 -> cast(f32) -> scale -> clamp -> relu: the cast folds into the
    // kernel's read, so the whole chain is ONE pass.
    let buf = u8_ramp();
    let (fused, kernels) = run_fused(&buf, |e| {
        e.cast(DType::F32).scale(2.0).clamp(0.0, 300.0).relu()
    });
    assert_eq!(kernels, 1, "expected the whole chain in one fused kernel");

    let stepwise = run_stepwise(
        &buf,
        &[
            &|e: Arc<ViewExpr>| e.cast(DType::F32),
            &|e: Arc<ViewExpr>| e.scale(2.0),
            &|e: Arc<ViewExpr>| e.clamp(0.0, 300.0),
            &|e: Arc<ViewExpr>| e.relu(),
        ],
    );
    assert_eq!(fused.dtype(), DType::F32);
    assert_buffers_equal(&fused, &stepwise);
}

#[test]
fn trailing_cast_fuses_as_output_conversion() {
    // u8 -> scale(0.5) -> cast(u8): rounding/saturation through the kernel's
    // write conversion must match a standalone cast exactly.
    let buf = u8_ramp();
    let (fused, kernels) = run_fused(&buf, |e| e.scale(0.5).cast(DType::U8));
    assert_eq!(kernels, 1);
    assert_eq!(fused.dtype(), DType::U8);

    let stepwise = run_stepwise(
        &buf,
        &[&|e: Arc<ViewExpr>| e.scale(0.5), &|e: Arc<ViewExpr>| {
            e.cast(DType::U8)
        }],
    );
    assert_buffers_equal(&fused, &stepwise);
    // Spot-check round-to-nearest: 3 * 0.5 = 1.5 -> 2.
    assert_eq!(fused.to_contiguous().as_slice::<u8>()[3], 2);
}

#[test]
fn invert_fuses_per_dtype() {
    // Invert's lowering depends on its input dtype (255 / 65535 / 1.0).
    let (fused, kernels) = run_fused(&u8_ramp(), |e| e.invert().scale(1.0));
    assert_eq!(kernels, 1);
    assert_eq!(fused.dtype(), DType::F32); // scale promotes
    let stepwise = run_stepwise(
        &u8_ramp(),
        &[&|e: Arc<ViewExpr>| e.invert(), &|e: Arc<ViewExpr>| {
            e.scale(1.0)
        }],
    );
    assert_buffers_equal(&fused, &stepwise);

    let buf16 = ViewBuffer::from_vec(vec![0u16, 1, 1000, 65534, 65535]);
    let (fused, _) = run_fused(&buf16, |e| e.invert().relu());
    let stepwise = run_stepwise(
        &buf16,
        &[&|e: Arc<ViewExpr>| e.invert(), &|e: Arc<ViewExpr>| e.relu()],
    );
    assert_buffers_equal(&fused, &stepwise);

    let buff32 = ViewBuffer::from_vec(vec![0.0f32, 0.25, 0.5, 1.0, 2.0, -1.0]);
    let (fused, _) = run_fused(&buff32, |e| e.scale(1.5).invert());
    let stepwise = run_stepwise(
        &buff32,
        &[&|e: Arc<ViewExpr>| e.scale(1.5), &|e: Arc<ViewExpr>| {
            e.invert()
        }],
    );
    assert_buffers_equal(&fused, &stepwise);
}

#[test]
fn invert_preserving_int_dtype_through_kernel() {
    // u8 -> invert -> invert: planned dtype stays u8, so the fused kernel
    // must convert its f32 result back to u8 — exactly (integers <= 255 are
    // exact in f32).
    let buf = u8_ramp();
    let (fused, kernels) = run_fused(&buf, |e| e.invert().invert());
    assert_eq!(kernels, 1);
    assert_eq!(fused.dtype(), DType::U8);
    // Double inversion is the identity.
    assert_eq!(
        fused.to_contiguous().as_slice::<u8>(),
        buf.to_contiguous().as_slice::<u8>()
    );
}

#[test]
fn gamma_fuses_bit_identically() {
    let buf = u8_ramp();
    let (fused, kernels) = run_fused(&buf, |e| e.adjust_gamma(2.2).scale(1.0));
    assert_eq!(kernels, 1);
    let stepwise = run_stepwise(
        &buf,
        &[&|e: Arc<ViewExpr>| e.adjust_gamma(2.2), &|e: Arc<
            ViewExpr,
        >| {
            e.scale(1.0)
        }],
    );
    assert_buffers_equal(&fused, &stepwise);

    // Float input uses the [0, 1] range.
    let buff32 = ViewBuffer::from_vec(vec![0.0f32, 0.1, 0.5, 0.9, 1.0, 1.5]);
    let (fused, _) = run_fused(&buff32, |e| e.adjust_gamma(0.45).relu());
    let stepwise = run_stepwise(
        &buff32,
        &[&|e: Arc<ViewExpr>| e.adjust_gamma(0.45), &|e: Arc<
            ViewExpr,
        >| {
            e.relu()
        }],
    );
    assert_buffers_equal(&fused, &stepwise);
}

#[test]
fn f64_invert_does_not_fuse_but_stays_correct() {
    // f64 invert is excluded from fusion (f32 compute would lose precision);
    // the chain must still execute correctly, just unfused.
    let buf = ViewBuffer::from_vec(vec![0.0f64, 0.123456789012345, 1.0]);
    let (fused, kernels) = run_fused(&buf, |e| e.invert().scale(2.0));
    assert_eq!(kernels, 0, "f64 invert must not fuse");
    let stepwise = run_stepwise(
        &buf,
        &[&|e: Arc<ViewExpr>| e.invert(), &|e: Arc<ViewExpr>| {
            e.scale(2.0)
        }],
    );
    assert_buffers_equal(&fused, &stepwise);
}

#[test]
fn mid_chain_int_cast_is_a_fusion_barrier() {
    // u8 -> scale(0.4) -> cast(u8) -> scale(10): the mid-chain cast
    // quantizes; folding through it would skip the rounding. The two sides
    // fuse separately and the quantization must be observable.
    let buf = ViewBuffer::from_vec(vec![1u8, 2, 3, 4, 5]);
    let (fused, _) = run_fused(&buf, |e| e.scale(0.4).cast(DType::U8).scale(10.0));
    let stepwise = run_stepwise(
        &buf,
        &[
            &|e: Arc<ViewExpr>| e.scale(0.4),
            &|e: Arc<ViewExpr>| e.cast(DType::U8),
            &|e: Arc<ViewExpr>| e.scale(10.0),
        ],
    );
    assert_buffers_equal(&fused, &stepwise);
    // 2 * 0.4 = 0.8 -> rounds to 1 -> * 10 = 10.0 (not 8.0): quantized.
    assert_eq!(fused.to_contiguous().as_slice::<f32>()[1], 10.0);
}

#[test]
fn fused_chain_on_strided_input_gathers_any_dtype() {
    // Transposed (strided) u8 input through a fused chain: the kernel's
    // strided gather handles non-f32 dtypes directly.
    let buf = ViewExpr::new_source(ViewBuffer::from_vec(vec![10u8, 20, 30, 40]))
        .reshape(vec![2, 2])
        .plan()
        .execute()
        .permute(&[1, 0]);
    let (fused, _) = run_fused(&buf, |e| e.cast(DType::F32).scale(2.0).relu());
    let stepwise = run_stepwise(
        &buf,
        &[
            &|e: Arc<ViewExpr>| e.cast(DType::F32),
            &|e: Arc<ViewExpr>| e.scale(2.0),
            &|e: Arc<ViewExpr>| e.relu(),
        ],
    );
    assert_buffers_equal(&fused, &stepwise);
    assert_eq!(
        fused.to_contiguous().as_slice::<f32>(),
        &[20.0, 60.0, 40.0, 80.0]
    );
}
