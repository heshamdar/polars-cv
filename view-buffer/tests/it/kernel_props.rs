//! Property tests for the hot per-pixel kernels: each kernel against a scalar
//! oracle written here, over random shapes (1×1, odd widths, rows that are not
//! a multiple of any SIMD width), dtypes and views (crop, flips, transpose).
//!
//! The oracles restate the ops' documented arithmetic, element by element and
//! in the same IEEE order, so the comparisons are exact: a fast path that
//! reorders, skips or mis-strides an element fails here, whatever it does to
//! performance.

use proptest::prelude::*;
use std::sync::Arc;
use view_buffer::execution::ExecutionPlan;
use view_buffer::ops::scalar::{FusedKernel, ScalarOp};
use view_buffer::{DType, NormalizeMethod, ViewBuffer, ViewExpr};

/// Plan from a clone and execute with the source moved in, as the plugin does.
fn exec(buf: ViewBuffer, build: impl Fn(&Arc<ViewExpr>) -> Arc<ViewExpr>) -> ViewBuffer {
    let steps = build(&ViewExpr::new_source(buf.clone())).plan().steps;
    ExecutionPlan { source: buf, steps }.execute()
}

/// The buffer's elements in logical (row-major) order.
fn logical<T: view_buffer::core::dtype::ViewType>(buf: &ViewBuffer) -> Vec<T> {
    buf.to_contiguous().as_slice::<T>().to_vec()
}

/// A view of `buf` (shape `[h, w, c]`): 0 none, 1 an interior crop, 2 flip_h,
/// 3 flip_v, 4 a transpose of the spatial axes.
fn view(buf: ViewBuffer, kind: u8) -> ViewBuffer {
    let (h, w, c) = (buf.shape()[0], buf.shape()[1], buf.shape()[2]);
    match kind {
        1 if h > 2 && w > 2 => buf.slice(&[1, 1, 0], &[h - 1, w - 1, c]),
        2 => buf.flip(&[1]),
        3 => buf.flip(&[0]),
        4 => buf.permute(&[1, 0, 2]),
        _ => buf,
    }
}

fn shape() -> impl Strategy<Value = (usize, usize, usize)> {
    (
        1usize..48,
        1usize..48,
        prop::sample::select(vec![1usize, 3, 4]),
    )
}

/// MinMax's arithmetic: `(x - lo) / (hi - lo)` in f32, with `lo`/`hi` the
/// extremes as f32; zeros when there is no spread.
fn minmax_oracle(values: &[f32]) -> Vec<f32> {
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = hi - lo;
    if range == 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|&x| (x - lo) / range).collect()
}

fn check_minmax(buf: ViewBuffer, values: Vec<f32>) -> Result<(), TestCaseError> {
    let out = exec(buf, |e| e.normalize(NormalizeMethod::MinMax, DType::F32));
    prop_assert_eq!(out.dtype(), DType::F32);
    let got = logical::<f32>(&out);
    let want = minmax_oracle(&values);
    prop_assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        prop_assert!(g.to_bits() == w.to_bits(), "element {i}: {g} != {w}");
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn normalize_minmax_u8((h, w, c) in shape(), kind in 0u8..5, seed in any::<u64>(), narrow in any::<bool>(), at in any::<prop::sample::Index>()) {
        let mut data: Vec<u8> = lcg(seed, h * w * c)
            .map(|v| if narrow { 100 + (v % 7) as u8 } else { 1 + (v % 254) as u8 })
            .collect();
        plant(&mut data, at, 0, 255);
        let buf = view(ViewBuffer::from_vec_with_shape(data, vec![h, w, c]), kind);
        let values = logical::<u8>(&buf).into_iter().map(f32::from).collect();
        check_minmax(buf, values)?;
    }

    #[test]
    fn normalize_minmax_u16((h, w, c) in shape(), kind in 0u8..5, seed in any::<u64>()) {
        let data: Vec<u16> = lcg(seed, h * w * c).map(|v| v as u16).collect();
        let buf = view(ViewBuffer::from_vec_with_shape(data, vec![h, w, c]), kind);
        let values = logical::<u16>(&buf).into_iter().map(f32::from).collect();
        check_minmax(buf, values)?;
    }

    #[test]
    fn normalize_minmax_i16((h, w, c) in shape(), kind in 0u8..5, seed in any::<u64>(), at in any::<prop::sample::Index>()) {
        let mut data: Vec<i16> = lcg(seed, h * w * c).map(|v| (v as i16).clamp(-32767, 32766)).collect();
        plant(&mut data, at, i16::MIN, i16::MAX);
        let buf = view(ViewBuffer::from_vec_with_shape(data, vec![h, w, c]), kind);
        let values = logical::<i16>(&buf).into_iter().map(f32::from).collect();
        check_minmax(buf, values)?;
    }

    #[test]
    fn normalize_minmax_f32((h, w, c) in shape(), kind in 0u8..5, seed in any::<u64>()) {
        let data: Vec<f32> = lcg(seed, h * w * c)
            .map(|v| (v as i32 as f32) / 1000.0)
            .collect();
        let buf = view(ViewBuffer::from_vec_with_shape(data, vec![h, w, c]), kind);
        let values = logical::<f32>(&buf);
        check_minmax(buf, values)?;
    }

    /// A fused affine chain (the normalize family's `Blocked` map) over
    /// lengths either side of its 2048-element block, on u8 and f32 input,
    /// against the chain applied one f32 operation at a time.
    #[test]
    fn fused_affine_chain(
        len in 1usize..5000,
        seed in any::<u64>(),
        ops in prop::collection::vec((0u8..4, -50.0f32..50.0), 1..5),
        is_u8 in any::<bool>(),
    ) {
        let mut kernel = FusedKernel::new();
        for &(op, c) in &ops {
            // A divisor away from zero keeps the chain finite.
            let c = if op == 3 { c.abs() + 0.5 } else { c };
            kernel.push(match op {
                0 => ScalarOp::Add(c.into()),
                1 => ScalarOp::Sub(c.into()),
                2 => ScalarOp::Mul(c.into()),
                _ => ScalarOp::Div(c.into()),
            });
        }
        let (buf, values): (ViewBuffer, Vec<f32>) = if is_u8 {
            let data: Vec<u8> = lcg(seed, len).map(|v| v as u8).collect();
            let values = data.iter().map(|&v| f32::from(v)).collect();
            (ViewBuffer::from_vec_with_shape(data, vec![len]), values)
        } else {
            let data: Vec<f32> = lcg(seed, len).map(|v| (v as i32 as f32) / 977.0).collect();
            (ViewBuffer::from_vec_with_shape(data.clone(), vec![len]), data)
        };
        let out = exec(buf, |e| e.fused(kernel.clone()));
        let got = logical::<f32>(&out);
        for (i, (&g, &x)) in got.iter().zip(&values).enumerate() {
            let mut want = x;
            for op in &kernel.ops {
                want = match *op {
                    ScalarOp::Add(c) => want + c as f32,
                    ScalarOp::Sub(c) => want - c as f32,
                    ScalarOp::Mul(c) => want * c as f32,
                    ScalarOp::Div(c) => want / c as f32,
                    _ => unreachable!(),
                };
            }
            prop_assert!(g.to_bits() == want.to_bits(), "element {i}: {g} != {want}");
        }
    }

    /// Blur keeps shape and dtype, keeps a constant image constant, and
    /// stays within the input's range (its weights are a convex combination).
    #[test]
    fn blur_u8_invariants((h, w, c) in shape(), kind in 0u8..5, seed in any::<u64>(), sigma in 0.3f32..4.0, constant in any::<bool>()) {
        let data: Vec<u8> = lcg(seed, h * w * c)
            .map(|v| if constant { 77 } else { v as u8 })
            .collect();
        let buf = view(ViewBuffer::from_vec_with_shape(data, vec![h, w, c]), kind);
        let in_shape = buf.shape().to_vec();
        let values = logical::<u8>(&buf);
        let (lo, hi) = (*values.iter().min().unwrap(), *values.iter().max().unwrap());
        let out = exec(buf, |e| e.blur(sigma));
        prop_assert_eq!(out.dtype(), DType::U8);
        prop_assert_eq!(out.shape(), &in_shape[..]);
        let got = logical::<u8>(&out);
        if constant {
            prop_assert!(got.iter().all(|&v| v == 77));
        }
        prop_assert!(got.iter().all(|&v| (lo..=hi).contains(&v)));
    }

    /// Grayscale of a pixel whose colour channels agree is that value, as
    /// `[h, w, 1]` (an alpha channel is dropped).
    #[test]
    fn grayscale_u8_neutral((h, w) in (1usize..48, 1usize..48), kind in 0u8..5, seed in any::<u64>(), alpha in any::<bool>()) {
        let c = if alpha { 4 } else { 3 };
        let data: Vec<u8> = lcg(seed, h * w)
            .flat_map(|v| {
                let g = v as u8;
                [g, g, g, 255][..c].to_vec()
            })
            .collect();
        let buf = view(ViewBuffer::from_vec_with_shape(data, vec![h, w, c]), kind);
        let want: Vec<u8> = logical::<u8>(&buf).chunks(c).map(|px| px[0]).collect();
        let out = exec(buf, |e| e.grayscale());
        let got = logical::<u8>(&out);
        prop_assert_eq!(&out.shape()[2..], &[1][..]); // alpha is dropped
        let luma: Vec<u8> = got.chunks(out.shape()[2]).map(|px| px[0]).collect();
        prop_assert_eq!(luma, want);
    }
}

/// Plant a unique minimum and maximum (values `data` otherwise avoids): at
/// the first and last elements, or the reverse, or at a random index — the
/// places a run-chunking or lane-folding fast path drops an element.
fn plant<T: Copy>(data: &mut [T], at: prop::sample::Index, lo: T, hi: T) {
    let n = data.len();
    match at.index(4) {
        0 => {
            data[0] = lo;
            data[n - 1] = hi;
        }
        1 => {
            data[n - 1] = lo;
            data[0] = hi;
        }
        _ => {
            let i = at.index(n);
            data[i] = lo;
            data[(i + n / 2) % n] = hi;
        }
    }
}

/// `n` deterministic pseudo-random words from `seed`.
fn lcg(seed: u64, n: usize) -> impl Iterator<Item = u64> {
    let mut state = seed | 1;
    (0..n).map(move |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    })
}
