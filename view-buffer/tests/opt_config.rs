//! Engine-tier (Tier-2) optimization toggles and their on/off equivalence.
//!
//! Every rewrite in `ViewExpr::optimize_with` is guarded by an `OptConfig`
//! toggle so it can be A/B differential tested: the output with the optimization
//! enabled must equal the output with it disabled. These tests exercise each
//! toggle and pin the consecutive-cast-collapse soundness fix (a narrowing
//! intermediate cast must not be dropped).

use std::sync::Arc;

use view_buffer::{ComputeOp, DType, OptConfig, ViewBuffer, ViewDto, ViewExpr};

/// All engine optimizations disabled — the "without optimization" baseline.
fn all_off() -> OptConfig {
    OptConfig {
        view_flip_involution: false,
        view_transpose_merge: false,
        cast_identity: false,
        cast_chain_collapse: false,
        scalar_fusion: false,
    }
}

/// Execute `expr` under `cfg` and read the first `n` elements as f32.
fn run(expr: &Arc<ViewExpr>, cfg: &OptConfig, n: usize) -> Vec<f32> {
    let out = expr.plan_with(cfg).execute();
    let (ptr, _, _, _) = out.as_raw_parts();
    unsafe { std::slice::from_raw_parts(ptr as *const f32, n) }.to_vec()
}

fn steps(expr: &Arc<ViewExpr>, cfg: &OptConfig) -> usize {
    expr.plan_with(cfg).steps.len()
}

#[test]
fn cast_chain_narrowing_intermediate_is_preserved() {
    // f32 -> cast(u8) -> cast(f32): the u8 step quantizes and MUST run. Dropping
    // it (the old unconditional collapse) returned the raw f32 unchanged.
    let data = vec![0.7f32, 2.3, 3.9, 0.1];
    let expr = ViewExpr::new_source(ViewBuffer::from_vec(data))
        .cast(DType::U8)
        .cast(DType::F32);

    let on = run(&expr, &OptConfig::default(), 4);
    let off = run(&expr, &all_off(), 4);

    // The mandate: output with optimization == output without.
    assert_eq!(on, off, "cast_chain_collapse changed the output");
    // And the u8 quantization actually happened (every value is now integral),
    // whatever the cast's rounding mode — i.e. the intermediate was not dropped.
    assert!(
        on.iter().all(|v| v.fract() == 0.0),
        "u8 quantization was skipped: {on:?}"
    );
}

#[test]
fn cast_chain_widening_intermediate_collapses_but_preserves_output() {
    // u8 -> cast(u16) -> cast(f32): u16 losslessly holds u8, so the intermediate
    // is safe to drop; output must be unchanged and the collapse must fire.
    let data: Vec<u8> = vec![10, 20, 30, 40];
    let expr = ViewExpr::new_source(ViewBuffer::from_vec(data))
        .cast(DType::U16)
        .cast(DType::F32);

    assert_eq!(run(&expr, &OptConfig::default(), 4), vec![10.0, 20.0, 30.0, 40.0]);
    assert_eq!(run(&expr, &all_off(), 4), vec![10.0, 20.0, 30.0, 40.0]);
    // The collapse fired: one fewer step than the unoptimized chain.
    assert!(
        steps(&expr, &OptConfig::default()) < steps(&expr, &all_off()),
        "widening cast chain did not collapse"
    );
}

#[test]
fn flip_involution_toggle_is_output_preserving() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let expr = ViewExpr::new_source(ViewBuffer::from_vec(data))
        .reshape(vec![2, 2])
        .flip(vec![0])
        .flip(vec![0]);

    assert_eq!(run(&expr, &OptConfig::default(), 4), run(&expr, &all_off(), 4));
    assert_eq!(run(&expr, &OptConfig::default(), 4), vec![1.0, 2.0, 3.0, 4.0]);
    // With the toggle on the two flips cancel to nothing; off keeps them.
    assert!(steps(&expr, &OptConfig::default()) < steps(&expr, &all_off()));
}

#[test]
fn transpose_merge_toggle_is_output_preserving() {
    // Two transposes that compose to the identity permutation.
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expr = ViewExpr::new_source(ViewBuffer::from_vec(data))
        .reshape(vec![2, 3])
        .transpose(vec![1, 0])
        .transpose(vec![1, 0]);

    assert_eq!(run(&expr, &OptConfig::default(), 6), run(&expr, &all_off(), 6));
    assert_eq!(
        run(&expr, &OptConfig::default(), 6),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn scalar_fusion_toggle_is_output_preserving() {
    let data = vec![1.0f32, -2.0, 3.0, 4.0];
    let expr = ViewExpr::new_source(ViewBuffer::from_vec(data))
        .apply_op(ViewDto::Compute(ComputeOp::Scale(2.0)))
        .apply_op(ViewDto::Compute(ComputeOp::Relu));

    assert_eq!(run(&expr, &OptConfig::default(), 4), run(&expr, &all_off(), 4));
    assert_eq!(run(&expr, &OptConfig::default(), 4), vec![2.0, 0.0, 6.0, 8.0]);
}
