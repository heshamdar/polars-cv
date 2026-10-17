//! `adjust_gamma` must normalize integer inputs by their dtype's value
//! range, not a hardcoded 255. With the 255 constant, every u16/u32 pixel
//! above 255 clamps to 1.0 and the whole image collapses to a constant.

use std::sync::Arc;
use view_buffer::{DType, ViewBuffer, ViewExpr};

fn run(buf: ViewBuffer, build: impl Fn(Arc<ViewExpr>) -> Arc<ViewExpr>) -> ViewBuffer {
    build(ViewExpr::new_source(buf)).plan().execute()
}

fn assert_gamma_matches_range(got: &[f32], input: &[f32], max: f32, gamma: f32) {
    for (g, x) in got.iter().zip(input) {
        let expected = (x / max).clamp(0.0, 1.0).powf(gamma) * max;
        assert!(
            (g - expected).abs() <= expected.abs() * 1e-4 + 1e-2,
            "got {g}, want {expected} (input {x}, max {max})"
        );
    }
}

#[test]
fn adjust_gamma_u16_normalizes_by_dtype_range() {
    let vals: Vec<u16> = vec![0, 255, 6553, 32768, 65535];
    let buf = ViewBuffer::from_vec(vals.clone());
    let out = run(buf, |e| e.adjust_gamma(2.0));
    assert_eq!(out.dtype(), DType::F32);
    let got = out.to_contiguous().as_slice::<f32>().to_vec();

    // Values above 255 must not saturate: the curve stays strictly
    // increasing across the u16 range.
    assert!(got[3] > got[2], "u16 gamma collapsed above 255: {got:?}");
    assert!(got[4] > got[3], "u16 gamma collapsed above 255: {got:?}");

    let input_f32: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
    assert_gamma_matches_range(&got, &input_f32, 65535.0, 2.0);
}

#[test]
fn adjust_gamma_i16_normalizes_by_dtype_range() {
    let vals: Vec<i16> = vec![-100, 0, 300, 16384, 32767];
    let buf = ViewBuffer::from_vec(vals.clone());
    let out = run(buf, |e| e.adjust_gamma(2.0));
    assert_eq!(out.dtype(), DType::F32);
    let got = out.to_contiguous().as_slice::<f32>().to_vec();

    // Negative inputs clamp to 0; positive values normalize by i16::MAX.
    let input_f32: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
    assert_gamma_matches_range(&got, &input_f32, 32767.0, 2.0);
    assert!(got[3] > got[2]);
    assert!(got[4] > got[3]);
}

#[test]
fn adjust_gamma_u8_unchanged() {
    // The u8 path keeps its 255 normalization — regression guard.
    let vals: Vec<u8> = vec![0, 64, 128, 255];
    let buf = ViewBuffer::from_vec(vals.clone());
    let out = run(buf, |e| e.adjust_gamma(2.2));
    assert_eq!(out.dtype(), DType::F32);
    let got = out.to_contiguous().as_slice::<f32>().to_vec();
    let input_f32: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
    assert_gamma_matches_range(&got, &input_f32, 255.0, 2.2);
}

#[test]
fn adjust_gamma_u16_fused_matches_unfused() {
    // The f32 fusion lowering must use the same dtype-derived range as
    // the unfused kernel.
    let vals: Vec<u16> = (0..64).map(|i| (i * 1024) as u16).collect();
    let buf = ViewBuffer::from_vec(vals);

    let unfused = {
        let a = run(buf.clone(), |e| e.adjust_gamma(2.2));
        run(a, |e| e.scale(1.0))
    };
    let fused = run(buf, |e| e.adjust_gamma(2.2).scale(1.0));

    assert_eq!(fused.dtype(), unfused.dtype());
    let f = fused.to_contiguous().as_slice::<f32>().to_vec();
    let u = unfused.to_contiguous().as_slice::<f32>().to_vec();
    for (a, b) in f.iter().zip(&u) {
        assert!(
            (a - b).abs() <= b.abs() * 1e-5 + 1e-3,
            "fused {a} != unfused {b}"
        );
    }
}
