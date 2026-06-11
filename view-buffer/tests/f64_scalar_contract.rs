//! The float-promoting scalar family (`PromoteToFloat`) preserves f64:
//! f64 inputs must compute in f64 and return f64, exactly as the dtype
//! contract (`OutputDTypeRule::resolve`) declares. These ops previously
//! computed everything in f32, silently returning f32 for f64 inputs —
//! a contract/runtime divergence that surfaced as a plan!=exec dtype
//! guard error in polars-cv.

use std::sync::Arc;
use view_buffer::{DType, ViewBuffer, ViewExpr};

/// A value that loses precision when round-tripped through f32.
const PRECISE: f64 = 0.123_456_789_012_345_67;

fn run(buf: &ViewBuffer, build: impl Fn(Arc<ViewExpr>) -> Arc<ViewExpr>) -> ViewBuffer {
    build(ViewExpr::new_source(buf.clone())).plan().execute()
}

fn f64_vals(buf: &ViewBuffer) -> Vec<f64> {
    assert_eq!(buf.dtype(), DType::F64, "output must stay f64");
    buf.to_contiguous().as_slice::<f64>().to_vec()
}

#[test]
fn scale_preserves_f64_and_precision() {
    let buf = ViewBuffer::from_vec(vec![PRECISE, 1.0, -2.5]);
    let out = run(&buf, |e| e.scale(2.0));
    let vals = f64_vals(&out);
    assert_eq!(vals[0], PRECISE * 2.0);
    // The f32 path would have collapsed the low-order bits.
    assert_ne!(vals[0], (PRECISE as f32 * 2.0) as f64);
}

#[test]
fn relu_preserves_f64() {
    let buf = ViewBuffer::from_vec(vec![PRECISE, -PRECISE, 0.0]);
    let out = run(&buf, |e| e.relu());
    assert_eq!(f64_vals(&out), vec![PRECISE, 0.0, 0.0]);
}

#[test]
fn clamp_preserves_f64() {
    let buf = ViewBuffer::from_vec(vec![PRECISE, 5.0, -5.0]);
    let out = run(&buf, |e| e.clamp(-1.0, 1.0));
    assert_eq!(f64_vals(&out), vec![PRECISE, 1.0, -1.0]);
}

#[test]
fn adjust_gamma_preserves_f64() {
    let buf = ViewBuffer::from_vec(vec![0.25_f64, PRECISE, 1.0]);
    let out = run(&buf, |e| e.adjust_gamma(2.0));
    let vals = f64_vals(&out);
    assert_eq!(vals[0], 0.25_f64.powf(2.0));
    assert_eq!(vals[1], PRECISE.powf(2.0));
}

#[test]
fn adjust_contrast_preserves_f64() {
    let buf = ViewBuffer::from_vec(vec![0.0_f64, 0.5, 1.0]);
    let out = run(&buf, |e| e.adjust_contrast(2.0));
    let vals = f64_vals(&out);
    // mean = 0.5; (x - 0.5) * 2 + 0.5
    assert_eq!(vals, vec![-0.5, 0.5, 1.5]);
}

#[test]
fn f32_inputs_still_compute_in_f32() {
    // The promotion path for everything else is unchanged.
    let buf = ViewBuffer::from_vec(vec![1.5_f32, -1.0]);
    let out = run(&buf, |e| e.scale(2.0));
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.to_contiguous().as_slice::<f32>(), &[3.0, -2.0]);

    let buf_u8 = ViewBuffer::from_vec(vec![10u8, 20]);
    let out = run(&buf_u8, |e| e.relu());
    assert_eq!(out.dtype(), DType::F32);
}

#[test]
fn f64_chain_stays_unfused_but_correct() {
    // Fusion computes in f32, so f64 chains are excluded from it — the
    // chain must still execute correctly per-op in f64.
    let buf = ViewBuffer::from_vec(vec![PRECISE, -1.0, 2.0]);
    let out = run(&buf, |e| e.scale(2.0).clamp(-1.0, 1.0).relu());
    let vals = f64_vals(&out);
    assert_eq!(vals[0], (PRECISE * 2.0).clamp(-1.0, 1.0).max(0.0));
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 1.0);
}
