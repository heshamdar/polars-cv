//! Every registered buffer op executes in the dtype its contract declares,
//! on every dtype and channel count the contract admits.
//!
//! The axis is the op registry (`TypedOp::samples()`), so a new op is swept
//! the day it is registered. For each sample, each of the ten dtypes and each
//! input layout (`[H, W]` and `[H, W, C]` for C in 1..=4) the op is applied
//! through `ViewExpr::try_apply_op` — the executor's own entry, which tracks
//! the dtype the op's contract declares and refuses what the contract does
//! not admit (those cases are skipped) — and then executed. The executed
//! dtype must be the tracked one.
//!
//! This is the plan == exec dtype property checked *per op*, where the
//! plugin's output guard (`compiled.rs`) only sees a graph's final output
//! and the engine's `debug_assert` only image ops. A kernel that converts
//! through f32 and forgets to convert back (`channel_swap` on u16) fails
//! here, as does one that panics against its contract.

use std::panic::{catch_unwind, AssertUnwindSafe};

use view_buffer::{DType, ViewBuffer, ViewExpr};

use crate::graph::step::GraphStep;
use crate::ops::TypedOp;
use crate::params::ParamCtx;

/// The input layouts swept: rank 2, then rank 3 with 1-4 channels.
fn shapes() -> Vec<Vec<usize>> {
    let mut shapes = vec![vec![6, 7]];
    shapes.extend((1..=4).map(|c| vec![6, 7, c]));
    shapes
}

/// A small pattern every dtype holds exactly, as `dtype`.
fn image(shape: &[usize], dtype: DType) -> ViewBuffer {
    let n: usize = shape.iter().product();
    let values: Vec<u8> = (0..n).map(|i| (i * 7 % 100) as u8).collect();
    ViewBuffer::from_vec_with_shape(values, shape.to_vec()).cast(dtype)
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "<non-string panic>".to_string())
}

#[test]
fn every_sample_op_executes_in_its_declared_dtype() {
    let mut failures: Vec<String> = Vec::new();
    let mut executed = 0usize;
    for sample in TypedOp::samples() {
        let name = sample.name();
        let step = sample
            .resolve(0, &ParamCtx::empty())
            .expect("a registered sample resolves");
        let GraphStep::Buffer(dto) = step else {
            continue;
        };
        for &dtype in DType::ALL {
            for shape in shapes() {
                let source = ViewExpr::new_source(image(&shape, dtype));
                let Ok(expr) = source.try_apply_op(dto.clone()) else {
                    continue; // the contract does not admit this input
                };
                let declared = expr.dtype;
                let what = format!("{name} on {dtype:?}{shape:?}");
                match catch_unwind(AssertUnwindSafe(|| expr.plan().execute())) {
                    Ok(out) if out.dtype() == declared => executed += 1,
                    Ok(out) => failures.push(format!(
                        "{what}: declared {declared:?}, executed {:?}",
                        out.dtype()
                    )),
                    Err(payload) => {
                        let message = panic_message(payload.as_ref());
                        let first = message.lines().next().unwrap_or("");
                        failures.push(format!("{what}: panicked: {first}"));
                    }
                }
            }
        }
    }
    // A sweep that admits nothing checks nothing.
    assert!(executed > 500, "only {executed} cases executed");
    assert!(
        failures.is_empty(),
        "{} op/dtype/layout cases do not execute as declared:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Runs `f`, recording a failure unless it returns a buffer of `declared`.
fn expect_dtype(
    failures: &mut Vec<String>,
    what: String,
    declared: DType,
    f: impl FnOnce() -> ViewBuffer,
) {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(out) if out.dtype() == declared => {}
        Ok(out) => failures.push(format!(
            "{what}: planned {declared:?}, executed {:?}",
            out.dtype()
        )),
        Err(payload) => {
            let message = panic_message(payload.as_ref());
            let first = message.lines().next().unwrap_or("").to_string();
            failures.push(format!("{what}: panicked: {first}"));
        }
    }
}

/// The graph-level kernels the registry sweep cannot reach through
/// `ViewExpr` (they read two or more buffers): every binary op executes in the
/// dtype `plan::step` plans for it (`BinaryOp::output_dtype`), and
/// `channel_merge` in its operands' (`PreserveInput`), on every dtype.
#[test]
fn every_multi_operand_kernel_executes_in_its_planned_dtype() {
    let mut failures: Vec<String> = Vec::new();
    for &dtype in DType::ALL {
        for &(name, op) in view_buffer::BinaryOp::NAMED {
            for shape in shapes() {
                let (a, b) = (image(&shape, dtype), image(&shape, dtype));
                expect_dtype(
                    &mut failures,
                    format!("{name} on {dtype:?}{shape:?}"),
                    op.output_dtype(dtype, dtype),
                    || op.execute(&a, &b),
                );
            }
        }
        for planes in 1..=4 {
            let inputs: Vec<ViewBuffer> = (0..planes).map(|_| image(&[6, 7], dtype)).collect();
            let refs: Vec<&ViewBuffer> = inputs.iter().collect();
            expect_dtype(
                &mut failures,
                format!("channel_merge of {planes} {dtype:?} planes"),
                dtype,
                || view_buffer::apply_channel_merge(&refs),
            );
        }
    }
    assert!(
        failures.is_empty(),
        "{} multi-operand cases do not execute as planned:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
