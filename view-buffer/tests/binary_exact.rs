//! The two-buffer ops are exact on every dtype.
//!
//! Each integer op is checked against an exact reference in `i128` (`u128`
//! where a u64 product needs it) over values at the edges of every integer
//! dtype and beyond f32's integer range; float ops against the native float
//! operation. The semantics, per integer dtype:
//!
//! - `add`/`subtract`/`multiply` saturate to the dtype's range;
//! - `blend` is `round(a * b / MAX)` (MAX is odd for every dtype, so there
//!   are no ties), saturated: the normalized product `(a/MAX)(b/MAX)MAX`;
//! - `maximum`/`minimum` and the bitwise ops are exact.
//!
//! They used to hold only for u8 and u16, which had native kernels; every
//! other pair was computed in f32 (u32 `16777219 ^ 16777221` came back 0),
//! and `blend` was a plain multiply.

use view_buffer::{BinaryOp, DType, ViewBuffer};

/// `round(num / max)` for an odd `max`, rounding half away from zero.
fn div_round(num: i128, max: i128) -> i128 {
    let half = max / 2;
    if num >= 0 {
        (num + half) / max
    } else {
        (num - half) / max
    }
}

fn reference(op: BinaryOp, a: i128, b: i128, min: i128, max: i128) -> i128 {
    let sat = |v: Option<i128>, sign_positive: bool| match v {
        Some(v) => v.clamp(min, max),
        None if sign_positive => max,
        None => min,
    };
    match op {
        BinaryOp::Add => sat(a.checked_add(b), a > 0),
        BinaryOp::Subtract => sat(a.checked_sub(b), a > 0),
        BinaryOp::Multiply => sat(a.checked_mul(b), (a > 0) == (b > 0)),
        BinaryOp::Blend => match a.checked_mul(b) {
            Some(p) => div_round(p, max).clamp(min, max),
            // Only u64 * u64 overflows i128; it is non-negative.
            None => {
                let (a, b, max) = (a as u128, b as u128, max as u128);
                let p = a * b;
                ((p + max / 2) / max).min(max) as i128
            }
        },
        BinaryOp::Maximum => a.max(b),
        BinaryOp::Minimum => a.min(b),
        BinaryOp::BitwiseAnd => a & b,
        BinaryOp::BitwiseOr => a | b,
        BinaryOp::BitwiseXor => a ^ b,
        BinaryOp::Divide => unreachable!("true division is float"),
    }
}

const INTEGER_OPS: [BinaryOp; 9] = [
    BinaryOp::Add,
    BinaryOp::Subtract,
    BinaryOp::Multiply,
    BinaryOp::Blend,
    BinaryOp::Maximum,
    BinaryOp::Minimum,
    BinaryOp::BitwiseAnd,
    BinaryOp::BitwiseOr,
    BinaryOp::BitwiseXor,
];

macro_rules! check_int {
    ($t:ty) => {{
        let (min, max) = (<$t>::MIN as i128, <$t>::MAX as i128);
        let mut values: Vec<i128> = vec![min, min + 1, -1, 0, 1, 2, 3, 81, max / 2, max - 1, max];
        // Beyond f32's 24-bit integers, where the dtype reaches.
        values.extend([16_777_217, 16_777_219, 16_777_221, -16_777_217]);
        values.retain(|v| (min..=max).contains(v));
        values.sort();
        values.dedup();
        let n = values.len();
        // Every pair: a row of `a`s against a row of `b`s.
        let a: Vec<$t> = values
            .iter()
            .flat_map(|&x| std::iter::repeat_n(x as $t, n))
            .collect();
        let b: Vec<$t> = (0..n)
            .flat_map(|_| values.iter().map(|&y| y as $t))
            .collect();
        let (ba, bb) = (
            ViewBuffer::from_vec_with_shape(a.clone(), vec![n, n]),
            ViewBuffer::from_vec_with_shape(b.clone(), vec![n, n]),
        );
        for op in INTEGER_OPS {
            let out = op.execute(&ba, &bb);
            let got = out.to_contiguous();
            let got = got.as_slice::<$t>();
            for i in 0..n * n {
                let want = reference(op, a[i] as i128, b[i] as i128, min, max);
                assert_eq!(
                    got[i] as i128,
                    want,
                    "{} {op:?}({}, {})",
                    stringify!($t),
                    a[i],
                    b[i]
                );
            }
        }
    }};
}

#[test]
fn integer_ops_are_exact_on_every_integer_dtype() {
    check_int!(u8);
    check_int!(i8);
    check_int!(u16);
    check_int!(i16);
    check_int!(u32);
    check_int!(i32);
    check_int!(u64);
    check_int!(i64);
}

/// `maximum`/`minimum` by NumPy's rule: NaN if either side is NaN.
fn ordered(x: f64, y: f64, larger: bool) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if (x > y) == larger {
        x
    } else {
        y
    }
}

/// Bit-identical, any NaN matching any NaN.
fn same(got: f64, want: f64) -> bool {
    (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits()
}

const FLOAT_OPS: [BinaryOp; 6] = [
    BinaryOp::Add,
    BinaryOp::Subtract,
    BinaryOp::Multiply,
    BinaryOp::Blend,
    BinaryOp::Maximum,
    BinaryOp::Minimum,
];

/// Every pair of `values`: a row of `a`s against a row of `b`s.
fn pairs<T: view_buffer::core::dtype::ViewType>(
    values: &[T],
) -> (Vec<T>, Vec<T>, ViewBuffer, ViewBuffer) {
    let n = values.len();
    let a: Vec<T> = values
        .iter()
        .flat_map(|&x| std::iter::repeat_n(x, n))
        .collect();
    let b: Vec<T> = (0..n).flat_map(|_| values.iter().copied()).collect();
    let (ba, bb) = (
        ViewBuffer::from_vec_with_shape(a.clone(), vec![n, n]),
        ViewBuffer::from_vec_with_shape(b.clone(), vec![n, n]),
    );
    (a, b, ba, bb)
}

/// f32 against f64 arithmetic rounded once to f32: f64 carries more than
/// twice f32's precision, so that is the correctly rounded f32 result of
/// `+`, `-` and `*` -- a reference that does not share the kernel's f32 path.
#[test]
fn f32_ops_are_correctly_rounded() {
    let values: Vec<f32> = vec![
        -2.5,
        -0.1,
        0.0,
        0.1,
        1.0 / 3.0,
        7.0,
        1e30,
        16_777_217.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let (a, b, ba, bb) = pairs(&values);
    for op in FLOAT_OPS {
        let out = op.execute(&ba, &bb);
        let got = out.to_contiguous();
        let got = got.as_slice::<f32>();
        for i in 0..a.len() {
            let (x, y) = (a[i] as f64, b[i] as f64);
            let want = match op {
                BinaryOp::Add => x + y,
                BinaryOp::Subtract => x - y,
                BinaryOp::Multiply | BinaryOp::Blend => x * y,
                BinaryOp::Maximum => ordered(x, y, true),
                BinaryOp::Minimum => ordered(x, y, false),
                _ => unreachable!(),
            } as f32;
            assert!(
                same(got[i] as f64, want as f64),
                "f32 {op:?}({}, {}): got {}, want {want}",
                a[i],
                b[i],
                got[i]
            );
        }
    }
}

/// f64 on quarter-integers below 2^24, where every sum, difference and
/// product is exactly an f64: the reference is exact integer arithmetic on
/// the values times four.
#[test]
fn f64_ops_are_exact_where_the_result_is_representable() {
    let quarters: Vec<i128> = vec![-10_000_001, -10, -1, 0, 1, 3, 4, 13, 9_999_999, 33_554_431];
    let values: Vec<f64> = quarters.iter().map(|&q| q as f64 / 4.0).collect();
    let (a, b, ba, bb) = pairs(&values);
    for op in FLOAT_OPS {
        let out = op.execute(&ba, &bb);
        let got = out.to_contiguous();
        let got = got.as_slice::<f64>();
        for i in 0..a.len() {
            let (x, y) = ((a[i] * 4.0) as i128, (b[i] * 4.0) as i128);
            // (value, its denominator): x/4 + y/4 = (x + y)/4, x/4 * y/4 = xy/16.
            let (num, den) = match op {
                BinaryOp::Add => (x + y, 4),
                BinaryOp::Subtract => (x - y, 4),
                BinaryOp::Multiply | BinaryOp::Blend => (x * y, 16),
                BinaryOp::Maximum => (x.max(y), 4),
                BinaryOp::Minimum => (x.min(y), 4),
                _ => unreachable!(),
            };
            assert_eq!(
                got[i] * den as f64,
                num as f64,
                "f64 {op:?}({}, {})",
                a[i],
                b[i]
            );
        }
    }
    // NaN by NumPy's rule, on the dtype the exact check cannot reach.
    let (a, b, ba, bb) = pairs(&[f64::NAN, 1.0, f64::NEG_INFINITY]);
    for (op, larger) in [(BinaryOp::Maximum, true), (BinaryOp::Minimum, false)] {
        let got = op.execute(&ba, &bb).to_contiguous();
        for (i, &g) in got.as_slice::<f64>().iter().enumerate() {
            assert!(
                same(g, ordered(a[i], b[i], larger)),
                "f64 {op:?}({}, {})",
                a[i],
                b[i]
            );
        }
    }
}

/// A value of any dtype, exactly: integers in `i128`, floats in `f64`.
#[derive(Clone, Copy, Debug)]
enum Val {
    Int(i128),
    Float(f64),
}

impl Val {
    fn as_f64(self) -> f64 {
        match self {
            Val::Int(i) => i as f64,
            Val::Float(f) => f,
        }
    }
}

/// Edge values of `dtype`, and values f32 cannot hold where it reaches them.
fn edge_values(dtype: DType) -> Vec<Val> {
    let int = |min: i128, max: i128| {
        let mut v: Vec<i128> = vec![min, min + 1, -1, 0, 1, 3, 200, max - 1, max];
        v.extend([16_777_217, -16_777_217, 9_007_199_254_740_993]);
        v.retain(|x| (min..=max).contains(x));
        v.sort();
        v.dedup();
        v.into_iter().map(Val::Int).collect::<Vec<_>>()
    };
    match dtype {
        DType::U8 => int(0, u8::MAX as i128),
        DType::I8 => int(i8::MIN as i128, i8::MAX as i128),
        DType::U16 => int(0, u16::MAX as i128),
        DType::I16 => int(i16::MIN as i128, i16::MAX as i128),
        DType::U32 => int(0, u32::MAX as i128),
        DType::I32 => int(i32::MIN as i128, i32::MAX as i128),
        DType::U64 => int(0, u64::MAX as i128),
        DType::I64 => int(i64::MIN as i128, i64::MAX as i128),
        DType::F32 | DType::F64 => [-2.5, 0.0, 0.1, 7.0, 16_777_217.0, -3.0e9]
            .into_iter()
            .map(|f| {
                Val::Float(if dtype == DType::F32 {
                    f as f32 as f64
                } else {
                    f
                })
            })
            .collect(),
    }
}

fn buffer(dtype: DType, values: &[Val]) -> ViewBuffer {
    let n = values.len();
    macro_rules! ints {
        ($t:ty) => {
            ViewBuffer::from_vec_with_shape(
                values
                    .iter()
                    .map(|v| match v {
                        Val::Int(i) => *i as $t,
                        Val::Float(_) => unreachable!(),
                    })
                    .collect::<Vec<$t>>(),
                vec![n],
            )
        };
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
        DType::F32 => ViewBuffer::from_vec_with_shape(
            values
                .iter()
                .map(|v| v.as_f64() as f32)
                .collect::<Vec<f32>>(),
            vec![n],
        ),
        DType::F64 => ViewBuffer::from_vec_with_shape(
            values.iter().map(|v| v.as_f64()).collect::<Vec<f64>>(),
            vec![n],
        ),
    }
}

/// Every element of a buffer, exactly.
fn values_of(buf: &ViewBuffer) -> Vec<Val> {
    let b = buf.to_contiguous();
    macro_rules! ints {
        ($t:ty) => {
            b.as_slice::<$t>()
                .iter()
                .map(|&x| Val::Int(x as i128))
                .collect()
        };
    }
    match b.dtype() {
        DType::U8 => ints!(u8),
        DType::I8 => ints!(i8),
        DType::U16 => ints!(u16),
        DType::I16 => ints!(i16),
        DType::U32 => ints!(u32),
        DType::I32 => ints!(i32),
        DType::U64 => ints!(u64),
        DType::I64 => ints!(i64),
        DType::F32 => b
            .as_slice::<f32>()
            .iter()
            .map(|&x| Val::Float(x as f64))
            .collect(),
        DType::F64 => b.as_slice::<f64>().iter().map(|&x| Val::Float(x)).collect(),
    }
}

fn int_range(dtype: DType) -> Option<(i128, i128)> {
    Some(match dtype {
        DType::U8 => (0, u8::MAX as i128),
        DType::I8 => (i8::MIN as i128, i8::MAX as i128),
        DType::U16 => (0, u16::MAX as i128),
        DType::I16 => (i16::MIN as i128, i16::MAX as i128),
        DType::U32 => (0, u32::MAX as i128),
        DType::I32 => (i32::MIN as i128, i32::MAX as i128),
        DType::U64 => (0, u64::MAX as i128),
        DType::I64 => (i64::MIN as i128, i64::MAX as i128),
        DType::F32 | DType::F64 => return None,
    })
}

/// Two operands of different dtypes combine in NumPy's promoted dtype
/// (`DType::promote`), each value converted exactly, then the op's
/// semantics in that dtype (decision D3, finding F1). The larger-integer
/// rule lost values: u8 200 with i8 gave i8, u64 with i64 gave i64, and an
/// i64 with an f32 was computed in f32.
#[test]
fn mixed_dtype_operands_combine_in_numpys_promoted_dtype() {
    let ops = [
        BinaryOp::Add,
        BinaryOp::Subtract,
        BinaryOp::Multiply,
        BinaryOp::Blend,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
        BinaryOp::BitwiseAnd,
        BinaryOp::BitwiseOr,
        BinaryOp::BitwiseXor,
        BinaryOp::Divide,
    ];
    let mut failures = Vec::new();
    for &da in DType::ALL {
        for &db in DType::ALL {
            if da == db {
                continue;
            }
            let common = da.promote(db);
            let (va, vb) = (edge_values(da), edge_values(db));
            let n = va.len() * vb.len();
            let a: Vec<Val> = va
                .iter()
                .flat_map(|&x| std::iter::repeat_n(x, vb.len()))
                .collect();
            let b: Vec<Val> = (0..va.len()).flat_map(|_| vb.iter().copied()).collect();
            let (ba, bb) = (buffer(da, &a), buffer(db, &b));
            for op in ops {
                let bitwise = matches!(
                    op,
                    BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor
                );
                if bitwise && int_range(common).is_none() {
                    continue; // refused by the contract (no common integer)
                }
                let out_dtype = op.output_dtype(da, db);
                let want_dtype = if op == BinaryOp::Divide {
                    common.accumulator()
                } else {
                    common
                };
                if out_dtype != want_dtype {
                    failures.push(format!(
                        "{da:?} {op:?} {db:?}: dtype {out_dtype:?}, want {want_dtype:?}"
                    ));
                    continue;
                }
                let got = values_of(&op.execute(&ba, &bb));
                for i in 0..n {
                    let want = match (op, int_range(common)) {
                        (BinaryOp::Divide, _) => {
                            let q = a[i].as_f64() / b[i].as_f64();
                            Val::Float(if want_dtype == DType::F32 {
                                q as f32 as f64
                            } else {
                                q
                            })
                        }
                        (_, Some((min, max))) => {
                            let (Val::Int(x), Val::Int(y)) = (a[i], b[i]) else {
                                unreachable!()
                            };
                            Val::Int(reference(op, x, y, min, max))
                        }
                        (_, None) => {
                            let (x, y) = (a[i].as_f64(), b[i].as_f64());
                            let r = match op {
                                BinaryOp::Add => x + y,
                                BinaryOp::Subtract => x - y,
                                BinaryOp::Multiply | BinaryOp::Blend => x * y,
                                BinaryOp::Maximum => ordered(x, y, true),
                                BinaryOp::Minimum => ordered(x, y, false),
                                _ => unreachable!(),
                            };
                            Val::Float(if common == DType::F32 {
                                r as f32 as f64
                            } else {
                                r
                            })
                        }
                    };
                    let same = match (got[i], want) {
                        (Val::Int(g), Val::Int(w)) => g == w,
                        (Val::Float(g), Val::Float(w)) => self::same(g, w),
                        _ => false,
                    };
                    if !same {
                        failures.push(format!(
                            "{da:?} {op:?} {db:?}: ({:?}, {:?}) -> {:?}, want {want:?}",
                            a[i], b[i], got[i]
                        ));
                        break;
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mixed cases wrong:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
