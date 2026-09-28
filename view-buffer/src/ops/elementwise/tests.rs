//! The engine against the legacy oracle (`legacy.rs`, the pre-engine code
//! verbatim), bit for bit, over every per-value op, every dtype and every
//! layout a view reaches an op in, shared and solely owned.
//!
//! "Bit for bit" treats any NaN as equal to any NaN: IEEE leaves a NaN
//! result's sign and payload unspecified, and nothing downstream reads them.

use super::{apply, legacy, strategy, Strategy};
use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{with_dtype, DType};
use crate::ops::compute::{ComputeOp, Normalization};
use crate::ops::scalar::{FusedKernel, ScalarOp};

/// Pseudo-random `u64`s (an LCG), deterministic per call.
fn lcg(n: usize, seed: u64) -> Vec<u64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        })
        .collect()
}

/// Float edge values placed at the start of every float sample.
const SPECIAL: [f64; 9] = [
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    -0.0,
    0.0,
    1e30,
    -1e-30,
    255.0,
    65535.5,
];

/// A `[h, w, c]` buffer of `dtype`: integers over their whole range, floats
/// mostly in `[-300, 300)` (where gamma and normalize are interesting) with
/// the special values first.
fn sample(dtype: DType, h: usize, w: usize, c: usize) -> ViewBuffer {
    let n = h * w * c;
    let raw = lcg(n, 0x9E37_79B9_7F4A_7C15 ^ dtype.wire_code() as u64);
    macro_rules! ints {
        ($t:ty) => {
            ViewBuffer::from_vec_with_shape(
                raw.iter().map(|&r| (r >> 17) as $t).collect::<Vec<$t>>(),
                vec![h, w, c],
            )
        };
    }
    macro_rules! floats {
        ($t:ty) => {{
            let mut v: Vec<$t> = raw
                .iter()
                .map(|&r| ((r >> 11) as f64 / (1u64 << 53) as f64 * 600.0 - 300.0) as $t)
                .collect();
            for (slot, &s) in v.iter_mut().zip(SPECIAL.iter()) {
                *slot = s as $t;
            }
            ViewBuffer::from_vec_with_shape(v, vec![h, w, c])
        }};
    }
    match dtype {
        DType::U8 => ints!(u8),
        DType::I8 => ints!(i8),
        DType::U16 => ints!(u16),
        DType::I16 => ints!(i16),
        DType::U32 => ints!(u32),
        DType::I32 => ints!(i32),
        DType::U64 => ints!(u64),
        DType::I64 => ints!(i64),
        DType::F32 => floats!(f32),
        DType::F64 => floats!(f64),
    }
}

/// The views of `parent` an op can receive, each sharing its data.
fn layouts(parent: &ViewBuffer, all: bool) -> Vec<(&'static str, ViewBuffer)> {
    let s = parent.shape().to_vec();
    let mut out = vec![
        ("contiguous", parent.clone()),
        ("flip_v", parent.flip(&[0])),
    ];
    if all {
        out.push(("flip_h", parent.flip(&[1])));
        out.push((
            "crop",
            parent.slice(&[1, 1, 0], &[s[0] - 1, s[1] - 1, s[2]]),
        ));
        out.push(("transpose", parent.permute(&[1, 0, 2])));
    }
    out
}

/// A contiguous buffer with the same elements as `buf` that nothing else
/// holds.
fn sole_owned(buf: &ViewBuffer) -> ViewBuffer {
    let packed = buf.to_contiguous();
    with_dtype!(buf.dtype(), T => ViewBuffer::from_vec_with_shape(
        packed.as_slice::<T>().to_vec(),
        buf.shape().to_vec(),
    ))
}

/// Panic unless `got` and `want` have one dtype, one shape and the same
/// elements, bit for bit, in logical order (any NaN equals any NaN).
fn assert_same(got: &ViewBuffer, want: &ViewBuffer, label: &str) {
    assert_eq!(got.dtype(), want.dtype(), "{label}: dtype");
    assert_eq!(got.shape(), want.shape(), "{label}: shape");
    let (g, w) = (got.to_contiguous(), want.to_contiguous());
    macro_rules! compare {
        ($t:ty, $is_nan:expr) => {{
            let is_nan = $is_nan;
            for (i, (a, b)) in g
                .as_slice::<$t>()
                .iter()
                .zip(w.as_slice::<$t>())
                .enumerate()
            {
                let same = a.to_ne_bytes() == b.to_ne_bytes() || (is_nan(*a) && is_nan(*b));
                assert!(
                    same,
                    "{label}: element {i} is {a:?}, the legacy code gave {b:?}"
                );
            }
        }};
    }
    match got.dtype() {
        DType::F32 => compare!(f32, |x: f32| x.is_nan()),
        DType::F64 => compare!(f64, |x: f64| x.is_nan()),
        other => with_dtype!(other, T => compare!(T, |_: T| false)),
    }
}

/// Every per-value op the engine runs (z-score is tested separately: its
/// statistics deliberately changed).
fn per_value_ops() -> Vec<ComputeOp> {
    let scalars = [
        ScalarOp::Add(3.5),
        ScalarOp::Sub(-2.25),
        ScalarOp::Mul(1.7),
        ScalarOp::Div(3.0),
        ScalarOp::Pow(0.5),
        ScalarOp::Neg,
        ScalarOp::Abs,
        ScalarOp::Sqrt,
        ScalarOp::Square,
        ScalarOp::Recip,
        ScalarOp::Min(100.0),
        ScalarOp::Max(-5.0),
        ScalarOp::Sign,
        ScalarOp::Floor,
        ScalarOp::Ceil,
        ScalarOp::Round,
        ScalarOp::Trunc,
        ScalarOp::Relu,
        ScalarOp::Clamp(-10.0, 200.0),
    ];
    let mut ops: Vec<ComputeOp> = scalars.into_iter().map(ComputeOp::Scalar).collect();
    ops.extend([
        ComputeOp::Neg,
        ComputeOp::Abs,
        ComputeOp::Sqrt,
        ComputeOp::Square,
        ComputeOp::Reciprocal,
        ComputeOp::Sign,
        ComputeOp::Floor,
        ComputeOp::Ceil,
        ComputeOp::Round,
        ComputeOp::Trunc,
        ComputeOp::ClampMin { value: 12.5 },
        ComputeOp::ClampMax { value: 99.0 },
        ComputeOp::AddConstant { value: 4.0 },
        ComputeOp::SubtractConstant { value: 1.5 },
        ComputeOp::Scale { factor: 0.37 },
        ComputeOp::Relu,
        ComputeOp::Clamp {
            min: -3.0,
            max: 150.0,
        },
        ComputeOp::Invert,
        ComputeOp::AdjustGamma { gamma: 0.45 },
        ComputeOp::AdjustGamma { gamma: 2.2 },
        ComputeOp::AdjustContrast { factor: 0.5 },
        ComputeOp::AdjustContrast { factor: 1.7 },
    ]);
    for out in [DType::F32, DType::U8, DType::I16, DType::F64] {
        ops.push(ComputeOp::from_normalization(Normalization::MinMax, out));
    }
    for out in [DType::F32, DType::U8] {
        ops.push(ComputeOp::from_normalization(
            Normalization::Preset {
                mean: vec![100.5, -3.25, 17.0],
                std: vec![57.1, 3.0, 0.5],
            },
            out,
        ));
    }
    // Integer affine chains (the integer strategy when the output dtype is
    // the input's): a shifted invert, and a large shift that saturates.
    for out in [DType::U8, DType::I16, DType::U16, DType::I8] {
        ops.push(ComputeOp::Fused(FusedKernel {
            ops: vec![
                ScalarOp::Mul(-1.0),
                ScalarOp::Add(200.0),
                ScalarOp::Sub(-60.0),
            ],
            out_dtype: out,
        }));
        ops.push(ComputeOp::Fused(FusedKernel {
            ops: vec![ScalarOp::Neg, ScalarOp::Add(40_000.0)],
            out_dtype: out,
        }));
        // `MAX + MIN - x` for each 8/16-bit dtype (the non-saturating form),
        // and offsets just beyond the clamp span, on both sides.
        for c in [255.0, -1.0, 65_535.0, 511.0, -511.0, 131_071.0, -131_071.0] {
            ops.push(ComputeOp::Fused(FusedKernel {
                ops: vec![ScalarOp::Mul(-1.0), ScalarOp::Add(c)],
                out_dtype: out,
            }));
            ops.push(ComputeOp::Fused(FusedKernel {
                ops: vec![ScalarOp::Add(c)],
                out_dtype: out,
            }));
        }
    }
    for out in [DType::U8, DType::F32, DType::I16] {
        ops.push(ComputeOp::Fused(FusedKernel {
            ops: vec![
                ScalarOp::Mul(1.2),
                ScalarOp::Add(-10.0),
                ScalarOp::Clamp(0.0, 255.0),
            ],
            out_dtype: out,
        }));
    }
    ops
}

/// Run `op` through the engine and the oracle on `buf` (shared) and on a
/// solely owned copy, and require the same result.
fn check(op: &ComputeOp, buf: &ViewBuffer, label: &str) {
    if let Some(want) = changed_from_legacy(op, buf) {
        assert_same(&apply(buf.clone(), op), &want, &format!("{label} shared"));
        assert_same(
            &apply(sole_owned(buf), op),
            &want,
            &format!("{label} owned"),
        );
        return;
    }
    let want = legacy::apply(buf.clone(), op.clone());
    assert_same(&apply(buf.clone(), op), &want, &format!("{label} shared"));
    let want = legacy::apply(sole_owned(buf), op.clone());
    assert_same(
        &apply(sole_owned(buf), op),
        &want,
        &format!("{label} owned"),
    );
}

/// The bitwise complement of every element of an integer buffer, in logical
/// order and the buffer's own dtype; `None` for a float buffer.
fn complement(buf: &ViewBuffer) -> Option<ViewBuffer> {
    let packed = buf.to_contiguous();
    macro_rules! not {
        ($t:ty) => {
            ViewBuffer::from_vec_with_shape(
                packed
                    .as_slice::<$t>()
                    .iter()
                    .map(|&x| !x)
                    .collect::<Vec<$t>>(),
                buf.shape().to_vec(),
            )
        };
    }
    Some(match buf.dtype() {
        DType::U8 => not!(u8),
        DType::I8 => not!(i8),
        DType::U16 => not!(u16),
        DType::I16 => not!(i16),
        DType::U32 => not!(u32),
        DType::I32 => not!(i32),
        DType::U64 => not!(u64),
        DType::I64 => not!(i64),
        DType::F32 | DType::F64 => return None,
    })
}

/// The expected result where the engine deliberately departs from the legacy
/// code (CR-53): `invert` on an integer dtype other than u8/u16 used to
/// return f32 `1 - x`; it now keeps the dtype as `MAX + MIN - x`, which is
/// `!x`.
fn changed_from_legacy(op: &ComputeOp, buf: &ViewBuffer) -> Option<ViewBuffer> {
    match (op, buf.dtype()) {
        (ComputeOp::Invert, DType::U8 | DType::U16) => None,
        (ComputeOp::Invert, _) => complement(buf),
        _ => None,
    }
}

/// `invert` maps every integer dtype's range onto itself as
/// `MAX + MIN - x` (`255 - x` for u8, `-1 - x` for a signed dtype): the
/// bitwise complement, in the input dtype, as its `PreserveInput` contract
/// says. Extremes swap.
#[test]
fn integer_invert_keeps_its_dtype() {
    for dtype in DType::ALL
        .iter()
        .copied()
        .filter(|d| !matches!(d, DType::F32 | DType::F64))
    {
        let parent = sample(dtype, 5, 7, 3);
        let want = complement(&parent).expect("an integer dtype");
        for owned in [false, true] {
            let input = if owned {
                sole_owned(&parent)
            } else {
                parent.clone()
            };
            let got = apply(input, &ComputeOp::Invert);
            assert_same(&got, &want, &format!("invert {dtype:?} owned={owned}"));
        }
        macro_rules! extremes {
            ($t:ty) => {{
                let buf = ViewBuffer::from_vec_with_shape(vec![<$t>::MIN, <$t>::MAX], vec![2]);
                let got = apply(buf, &ComputeOp::Invert);
                assert_eq!(got.as_slice::<$t>(), &[<$t>::MAX, <$t>::MIN], "{dtype:?}");
            }};
        }
        with_dtype!(dtype, T => extremes!(T));
    }
}

#[test]
fn every_per_value_op_matches_the_legacy_code_on_every_dtype_and_layout() {
    for dtype in DType::ALL.iter().copied() {
        let parent = sample(dtype, 7, 13, 3);
        for (layout, view) in layouts(&parent, true) {
            for op in per_value_ops() {
                check(&op, &view, &format!("{op:?} on {dtype:?} {layout}"));
            }
        }
    }
}

/// `parent`'s elements stored channel-first (`[c, h, w]`) and viewed
/// channels-last again: the same logical buffer with its channels
/// `h · w` elements apart, so nothing but single elements is packed.
fn channel_first(parent: &ViewBuffer) -> ViewBuffer {
    let s = parent.shape().to_vec();
    let packed = parent.to_contiguous();
    with_dtype!(parent.dtype(), T => {
        let src = packed.as_slice::<T>();
        let mut chw = Vec::with_capacity(src.len());
        for c in 0..s[2] {
            chw.extend(src.iter().skip(c).step_by(s[2]).copied());
        }
        ViewBuffer::from_vec_with_shape(chw, vec![s[2], s[0], s[1]]).permute(&[1, 2, 0])
    })
}

/// Per-channel kernels (a preset normalize: a table for 8-bit input, blocks
/// of passes otherwise) read an element's channel from its position in the
/// buffer. A block or a run of a view can start inside a pixel: blocks are
/// 2,048 elements, not a whole number of 3-channel pixels, and a
/// channel-first view is handed out in scratch-sized runs. Images big enough
/// for several blocks and runs, on every layout, must agree with the oracle.
#[test]
fn per_channel_kernels_match_the_legacy_code_across_blocks_and_runs() {
    let ops: Vec<ComputeOp> = [DType::F32, DType::U8, DType::I16]
        .into_iter()
        .map(|out| {
            ComputeOp::from_normalization(
                Normalization::Preset {
                    mean: vec![100.5, -3.25, 17.0],
                    std: vec![57.1, 3.0, 0.5],
                },
                out,
            )
        })
        .collect();
    for dtype in [DType::U8, DType::I16, DType::U32, DType::F32] {
        let parent = sample(dtype, 61, 53, 3);
        let mut views = layouts(&parent, true);
        views.push(("channel_first", channel_first(&parent)));
        for (layout, view) in views {
            for op in &ops {
                check(op, &view, &format!("{op:?} on {dtype:?} 61x53x3 {layout}"));
            }
        }
    }
}

/// 16-bit input switches to a lookup table at 65,536 elements per table;
/// both sides of the threshold must agree with the oracle.
#[test]
fn sixteen_bit_ops_match_the_legacy_code_through_the_lookup_table() {
    for dtype in [DType::U16, DType::I16] {
        let parent = sample(dtype, 260, 260, 3);
        let n = 260 * 260 * 3;
        let gamma = FusedKernel {
            ops: vec![ScalarOp::Pow(0.45)],
            out_dtype: DType::F32,
        };
        let preset = vec![gamma.clone(); 3];
        assert_eq!(
            strategy(dtype, n, std::slice::from_ref(&gamma)),
            Strategy::Lut
        );
        assert_eq!(strategy(dtype, n, &preset), Strategy::Lut);
        for (layout, view) in layouts(&parent, false) {
            for op in per_value_ops() {
                check(
                    &op,
                    &view,
                    &format!("{op:?} on {dtype:?} 260x260x3 {layout}"),
                );
            }
        }
    }
}

#[test]
fn the_strategy_tables_only_costly_kernels_over_enumerable_input() {
    let k = |ops: Vec<ScalarOp>, out| FusedKernel {
        ops,
        out_dtype: out,
    };
    let cheap_u8 = || {
        vec![k(
            vec![ScalarOp::Mul(-1.0), ScalarOp::Add(255.0)],
            DType::U8,
        )]
    };
    let cheap_f32 = || vec![k(vec![ScalarOp::Sub(3.0), ScalarOp::Div(2.0)], DType::F32)];
    let gamma = || {
        vec![k(
            vec![ScalarOp::Clamp(0.0, 1.0), ScalarOp::Pow(0.5)],
            DType::F32,
        )]
    };
    let preset = || vec![k(vec![ScalarOp::Sub(1.0)], DType::F32); 3];
    let table = [
        (
            "u8 invert",
            DType::U8,
            10,
            cheap_u8(),
            Strategy::IntAffine {
                sign: -1,
                offset: 255,
            },
        ),
        (
            "u16 shift",
            DType::U16,
            10,
            vec![k(vec![ScalarOp::Add(7.0), ScalarOp::Neg], DType::U16)],
            Strategy::IntAffine {
                sign: -1,
                offset: -7,
            },
        ),
        (
            "u8 half shift",
            DType::U8,
            10,
            vec![k(vec![ScalarOp::Add(0.5)], DType::U8)],
            Strategy::Blocked,
        ),
        (
            "u8 scale 2",
            DType::U8,
            10,
            vec![k(vec![ScalarOp::Mul(2.0)], DType::U8)],
            Strategy::Blocked,
        ),
        (
            "u8 invert to i16",
            DType::U8,
            10,
            vec![k(vec![ScalarOp::Neg], DType::I16)],
            Strategy::Blocked,
        ),
        (
            "u8 minmax -> f32",
            DType::U8,
            10,
            cheap_f32(),
            Strategy::Blocked,
        ),
        ("u8 gamma", DType::U8, 10, gamma(), Strategy::Lut),
        ("i8 preset", DType::I8, 12, preset(), Strategy::Lut),
        (
            "small u16 gamma",
            DType::U16,
            65_535,
            gamma(),
            Strategy::Blocked,
        ),
        (
            "large u16 gamma",
            DType::U16,
            65_536,
            gamma(),
            Strategy::Lut,
        ),
        (
            "small i16 preset",
            DType::I16,
            3 * 65_536 - 3,
            preset(),
            Strategy::Blocked,
        ),
        (
            "large i16 preset",
            DType::I16,
            3 * 65_536,
            preset(),
            Strategy::Lut,
        ),
        ("u32 gamma", DType::U32, 1 << 20, gamma(), Strategy::Blocked),
        (
            "f32 -> u8 chain",
            DType::F32,
            7,
            cheap_u8(),
            Strategy::Blocked,
        ),
        ("f64 preset", DType::F64, 9, preset(), Strategy::Blocked),
    ];
    for (label, dtype, n, kernels, want) in table {
        assert_eq!(strategy(dtype, n, &kernels), want, "{label}");
    }
}

/// z-score statistics are exact (the one deliberate change): 8/16-bit input
/// from integer sums, other input from f64 sums of the elements as f32. The
/// transform is `(x - mean) / std` in f32, as before.
#[test]
fn zscore_uses_exact_statistics() {
    let op = |out| ComputeOp::from_normalization(Normalization::ZScore, out);
    for dtype in DType::ALL.iter().copied() {
        let parent = sample(dtype, 9, 11, 3);
        for (layout, view) in layouts(&parent, true) {
            let values: Vec<f32> = {
                let packed = view.to_contiguous();
                with_dtype!(dtype, T => crate::core::convert::convert_slice::<T, f32>(packed.as_slice::<T>()))
            };
            let n = values.len() as f64;
            let (mean, var) = if matches!(dtype, DType::U8 | DType::I8 | DType::U16 | DType::I16) {
                let sum: i128 = values.iter().map(|&x| x as i128).sum();
                let sum_sq: i128 = values.iter().map(|&x| (x as i128) * (x as i128)).sum();
                let numerator = (values.len() as i128) * sum_sq - sum * sum;
                (sum as f64 / n, numerator as f64 / (n * n))
            } else {
                let mean = values.iter().map(|&x| x as f64).sum::<f64>() / n;
                let var = values
                    .iter()
                    .map(|&x| (x as f64 - mean).powi(2))
                    .sum::<f64>()
                    / n;
                (mean, var)
            };
            let (mean, std) = (mean as f32, var.sqrt() as f32);
            for out in [DType::F32, DType::U8] {
                let got = apply(view.clone(), &op(out));
                let expected = with_dtype!(out, T => {
                    let f: Vec<f32> = values.iter().map(|&x| (x - mean) / std).collect();
                    ViewBuffer::from_vec_with_shape(
                        crate::core::convert::convert_slice::<f32, T>(&f),
                        view.shape().to_vec(),
                    )
                });
                assert_same(
                    &got,
                    &expected,
                    &format!("zscore {dtype:?} {layout} -> {out:?}"),
                );
            }
        }
    }
}

/// A constant buffer has no spread: min-max and z-score give zeros, as before.
#[test]
fn a_constant_buffer_normalizes_to_zeros() {
    for method in [Normalization::MinMax, Normalization::ZScore] {
        let buf = ViewBuffer::from_vec_with_shape(vec![42u8; 24], vec![2, 4, 3]);
        let got = apply(
            buf,
            &ComputeOp::from_normalization(method.clone(), DType::F32),
        );
        assert!(
            got.as_slice::<f32>().iter().all(|&x| x == 0.0),
            "{method:?}"
        );
    }
}
