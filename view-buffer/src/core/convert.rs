//! Element conversion between dtypes: **the one rule for turning a value of
//! one dtype into another**, and the one bulk kernel that applies it.
//!
//! The rule ([`CastFrom`]):
//! - integer → any: plain `as` (integer narrowing wraps; → float is exact or
//!   nearest);
//! - float → integer: **round to nearest (ties away from zero), then
//!   saturate** — `x.round() as T`, so `140.75 → 141` rather than 140, NaN →
//!   0 and out-of-range values clamp to the target's bounds;
//! - float → float: plain `as`.
//!
//! [`ViewBuffer::cast_to`](crate::ViewBuffer::cast_to) and the fused scalar
//! kernel's output conversion both go through [`convert_slice`], so a cast
//! and a fused trailing cast cannot round differently. The bulk loop is a
//! [`SimdKernel`]: on the wheels' SSE2 baseline `f32::round` is a `roundf`
//! call per element, while the AVX2 build rounds a vector at a time.

use crate::core::dispatch::{dispatch, SimdKernel};
use crate::core::dtype::ViewType;

/// `Self` from an `S`, by the crate's conversion rule (module docs).
pub trait CastFrom<S>: Sized {
    fn cast_from(value: S) -> Self;
}

macro_rules! cast_from {
    // Integer source: plain `as` to every target.
    (int $src:ty) => {
        cast_from!(@plain $src => u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
    };
    // Float source: round-then-saturate to integers, plain `as` to floats.
    (float $src:ty) => {
        cast_from!(@round $src => u8, i8, u16, i16, u32, i32, u64, i64);
        cast_from!(@plain $src => f32, f64);
    };
    (@plain $src:ty => $($dst:ty),+) => {
        $(impl CastFrom<$src> for $dst {
            #[inline(always)]
            fn cast_from(value: $src) -> $dst {
                value as $dst
            }
        })+
    };
    (@round $src:ty => $($dst:ty),+) => {
        $(impl CastFrom<$src> for $dst {
            #[inline(always)]
            fn cast_from(value: $src) -> $dst {
                value.round() as $dst
            }
        })+
    };
}

cast_from!(int u8);
cast_from!(int i8);
cast_from!(int u16);
cast_from!(int i16);
cast_from!(int u32);
cast_from!(int i32);
cast_from!(int u64);
cast_from!(int i64);
cast_from!(float f32);
cast_from!(float f64);

/// Every element of `src` converted to `D` by the crate's rule.
pub fn convert_slice<S: ViewType, D: ViewType + CastFrom<S>>(src: &[S]) -> Vec<D> {
    dispatch(Convert::<S, D> {
        src,
        _to: std::marker::PhantomData,
    })
}

struct Convert<'a, S, D> {
    src: &'a [S],
    _to: std::marker::PhantomData<D>,
}

// Derived `Clone` would demand `D: Clone` of the marker's parameter.
impl<S, D> Clone for Convert<'_, S, D> {
    fn clone(&self) -> Self {
        Convert {
            src: self.src,
            _to: std::marker::PhantomData,
        }
    }
}

impl<S: ViewType, D: ViewType + CastFrom<S>> SimdKernel for Convert<'_, S, D> {
    type Output = Vec<D>;

    #[inline(always)]
    fn run(self) -> Vec<D> {
        self.src.iter().map(|&x| D::cast_from(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_to_int_rounds_half_away_from_zero_then_saturates() {
        let src = [
            0.5f32,
            1.5,
            2.5,
            -0.5,
            -1.5,
            140.75,
            0.49999997,
            254.5,
            255.49,
            255.5,
            300.0,
            -3.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        assert_eq!(
            convert_slice::<f32, u8>(&src),
            vec![1, 2, 3, 0, 0, 141, 0, 255, 255, 255, 255, 0, 0, 255, 0]
        );
        assert_eq!(
            convert_slice::<f32, i8>(&src),
            vec![1, 2, 3, -1, -2, 127, 0, 127, 127, 127, 127, -3, 0, 127, -128]
        );
        assert_eq!(
            convert_slice::<f64, i64>(&[2.5, -2.5, 1e300, -1e300, f64::NAN]),
            vec![3, -3, i64::MAX, i64::MIN, 0]
        );
        assert_eq!(
            convert_slice::<f32, u16>(&[65535.4, 65535.6, -0.6, 1e9]),
            vec![65535, 65535, 0, 65535]
        );
    }

    #[test]
    fn integer_sources_use_plain_as() {
        // Narrowing wraps: only float sources round and saturate.
        assert_eq!(convert_slice::<u16, u8>(&[10, 250, 300]), vec![10, 250, 44]);
        assert_eq!(convert_slice::<i16, u8>(&[-1, 256]), vec![255, 0]);
        assert_eq!(
            convert_slice::<u8, f32>(&[0, 7, 255]),
            vec![0.0, 7.0, 255.0]
        );
        assert_eq!(
            convert_slice::<i64, f32>(&[16_777_217]),
            vec![16_777_216.0],
            "int → float is nearest, not exact"
        );
    }

    #[test]
    fn float_to_float_does_not_round() {
        assert_eq!(convert_slice::<f32, f64>(&[1.25, -2.75]), vec![1.25, -2.75]);
        assert_eq!(convert_slice::<f64, f32>(&[0.1]), vec![0.1f64 as f32]);
    }

    /// Every length a vector loop splits differently: empty, shorter than a
    /// vector, and every remainder past a few full vectors.
    #[test]
    fn every_length_converts_every_element() {
        for len in 0..70usize {
            let src: Vec<f32> = (0..len).map(|i| i as f32 * 3.7 - 40.5).collect();
            let expected: Vec<u8> = src.iter().map(|&x| x.round() as u8).collect();
            assert_eq!(convert_slice::<f32, u8>(&src), expected, "len {len}");
        }
    }
}
