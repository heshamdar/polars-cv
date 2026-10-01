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

use view_buffer::{BinaryOp, ViewBuffer};

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
        BinaryOp::Divide | BinaryOp::Ratio => unreachable!("true division is float"),
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

macro_rules! check_float {
    ($t:ty) => {{
        let values: Vec<$t> = vec![-2.5, -0.1, 0.0, 0.1, 1.0 / 3.0, 7.0, 1e30, 16_777_217.0];
        let n = values.len();
        let a: Vec<$t> = values
            .iter()
            .flat_map(|&x| std::iter::repeat_n(x, n))
            .collect();
        let b: Vec<$t> = (0..n).flat_map(|_| values.iter().copied()).collect();
        let (ba, bb) = (
            ViewBuffer::from_vec_with_shape(a.clone(), vec![n, n]),
            ViewBuffer::from_vec_with_shape(b.clone(), vec![n, n]),
        );
        let cases: [(BinaryOp, fn($t, $t) -> $t); 6] = [
            (BinaryOp::Add, |x, y| x + y),
            (BinaryOp::Subtract, |x, y| x - y),
            (BinaryOp::Multiply, |x, y| x * y),
            (BinaryOp::Blend, |x, y| x * y),
            (BinaryOp::Maximum, |x, y| if x > y { x } else { y }),
            (BinaryOp::Minimum, |x, y| if x < y { x } else { y }),
        ];
        for (op, f) in cases {
            let out = op.execute(&ba, &bb);
            let got = out.to_contiguous();
            let got = got.as_slice::<$t>();
            for i in 0..n * n {
                assert_eq!(
                    got[i].to_bits(),
                    f(a[i], b[i]).to_bits(),
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
fn float_ops_are_native_float_arithmetic() {
    check_float!(f32);
    check_float!(f64);
}
