//! Every op that orders values propagates NaN, whichever operand or
//! neighbour holds it.
//!
//! One rule (numpy's `maximum`/`minimum`, `nanmax` aside): a NaN compared
//! against anything is the result. Each op used to decide with its own
//! comparison, so the answer depended on position — binary `maximum(NaN, 1)`
//! was 1 but `maximum(1, NaN)` NaN, a dilate kept a NaN centre but skipped a
//! NaN neighbour, the morphological gradient turned NaN into 0, MinMax
//! normalize skipped NaN while ZScore propagated it, and HSV's `max(r, g, b)`
//! skipped a NaN channel. The reference here is `is_nan` on the inputs, not a
//! restatement of the kernel's comparison.

#![cfg(feature = "image_interop")]

use view_buffer::{
    BinaryOp, ColorConvertOp, ColorSpace, ComputeOp, ImageOp, ImageOpKind, NormalizeMethod,
    ViewBuffer, ViewDto, ViewExpr,
};

fn run(buf: ViewBuffer, dto: ViewDto) -> ViewBuffer {
    ViewExpr::new_source(buf).apply_op(dto).plan().execute()
}

fn f32s(buf: &ViewBuffer) -> Vec<f32> {
    buf.to_contiguous().as_slice::<f32>().to_vec()
}

#[test]
fn binary_maximum_and_minimum_propagate_nan_from_either_side() {
    let vals = [f32::NAN, -1.0, 0.0, 2.5, f32::INFINITY];
    let n = vals.len();
    let a: Vec<f32> = vals
        .iter()
        .flat_map(|&x| std::iter::repeat_n(x, n))
        .collect();
    let b: Vec<f32> = (0..n).flat_map(|_| vals.iter().copied()).collect();
    let (ba, bb) = (
        ViewBuffer::from_vec_with_shape(a.clone(), vec![n, n]),
        ViewBuffer::from_vec_with_shape(b.clone(), vec![n, n]),
    );
    for (op, pick) in [
        (BinaryOp::Maximum, f32::max as fn(f32, f32) -> f32),
        (BinaryOp::Minimum, f32::min),
    ] {
        let got = f32s(&op.execute(&ba, &bb));
        for i in 0..n * n {
            if a[i].is_nan() || b[i].is_nan() {
                assert!(got[i].is_nan(), "{op:?}({}, {}) = {}", a[i], b[i], got[i]);
            } else {
                assert_eq!(got[i], pick(a[i], b[i]), "{op:?}({}, {})", a[i], b[i]);
            }
        }
    }
}

#[test]
fn morphology_propagates_a_nan_anywhere_in_the_window() {
    // A NaN at each position of a 1x3 row: an output pixel is NaN exactly
    // when its 3-wide window (clamped at the edges) holds the NaN.
    for at in 0..3 {
        let mut row = vec![1.0f32, 2.0, 3.0];
        row[at] = f32::NAN;
        for kind in [
            ImageOpKind::Dilate {
                ksize: 3,
                iterations: 1,
            },
            ImageOpKind::Erode {
                ksize: 3,
                iterations: 1,
            },
            ImageOpKind::MorphGradient { ksize: 3 },
        ] {
            let buf = ViewBuffer::from_vec_with_shape(row.clone(), vec![1, 3, 1]);
            let out = f32s(&run(buf, ViewDto::Image(ImageOp { kind: kind.clone() })));
            for (x, v) in out.iter().enumerate() {
                if x.abs_diff(at) <= 1 {
                    assert!(v.is_nan(), "{kind:?} NaN at {at}: pixel {x} = {v}");
                } else {
                    assert!(!v.is_nan(), "{kind:?} NaN at {at}: pixel {x} = {v}");
                }
            }
        }
    }
}

#[test]
fn minmax_normalize_of_input_with_nan_is_nan() {
    let buf = ViewBuffer::from_vec_with_shape(vec![1.0f32, f32::NAN, 3.0, 5.0], vec![2, 2, 1]);
    let op = ViewDto::Compute(ComputeOp::Normalize {
        method: NormalizeMethod::MinMax,
        mean: None,
        std: None,
        out_dtype: None,
    });
    let out = f32s(&run(buf, op));
    assert!(out.iter().all(|v| v.is_nan()), "{out:?}");
}

#[test]
fn hsv_of_a_nan_channel_is_nan() {
    for at in 0..3 {
        let mut px = vec![0.25f32, 0.5, 0.75];
        px[at] = f32::NAN;
        let buf = ViewBuffer::from_vec_with_shape(px, vec![1, 1, 3]);
        let op = ViewDto::Color(ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Hsv,
        });
        let out = f32s(&run(buf, op));
        assert!(out[2].is_nan(), "NaN in channel {at}: V = {}", out[2]);
    }
}
