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
//! [`ViewBuffer::cast_to`](crate::ViewBuffer::cast_to) and every kernel that
//! stores a computed value (the element-wise engine, the warp, the blur,
//! grayscale) convert by [`CastFrom`], so a cast and a fused trailing cast
//! cannot round differently. The bulk conversion is the element map
//! `Convert`, run by the one traversal (`core::map`), which dispatches it:
//! on the wheels' SSE2 baseline `f32::round` is a `roundf` call per element,
//! while the AVX2 build rounds a vector at a time.

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::ViewType;
use crate::core::map::{map_new, map_slice, ElementMap};

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
    // The 8/16-bit targets take the vectorisable form (see `round_narrow`).
    (float $src:ty) => {
        cast_from!(@narrow $src => u8, i8, u16, i16);
        cast_from!(@round $src => u32, i32, u64, i64);
        cast_from!(@plain $src => f32, f64);
    };
    // f64 source: as `float`, but the 8/16-bit targets clamp (`clamp_narrow`).
    (double $src:ty) => {
        cast_from!(@clamp $src => u8, i8, u16, i16);
        cast_from!(@round $src => u32, i32, u64, i64);
        cast_from!(@plain $src => f32, f64);
    };
    (@clamp $src:ty => $($dst:ty),+) => {
        $(impl CastFrom<$src> for $dst {
            /// `value.round() as $dst`, clamped to the target first: the
            /// same result (the clamped value is in range, and NaN passes
            /// the clamp to become 0), in the form the compiler vectorises
            /// across the values of a per-pixel loop, such as the four
            /// channels a warp blends (`round_narrow`'s did not: 1.8x slower
            /// for RGBA there).
            #[inline(always)]
            fn cast_from(value: $src) -> $dst {
                value.round().clamp(<$dst>::MIN as $src, <$dst>::MAX as $src) as $dst
            }
        })+
    };
    (@narrow $src:ty => $($dst:ty),+) => {
        $(impl CastFrom<$src> for $dst {
            #[inline(always)]
            fn cast_from(value: $src) -> $dst {
                round_narrow!($src, value, $dst)
            }
        })+
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

/// `value.round() as $dst` for an 8/16-bit `$dst`, written so it vectorises.
///
/// `f32::round` (half away from zero) has no x86 instruction, and `as` into
/// a narrow integer saturates through per-element checks, so the plain form
/// ran one element at a time even in the AVX2 build (3.3 ms for 3M f32 → u8,
/// against 1.5 ms for this). Here rounding is `trunc` plus a step away from
/// zero at `|frac| >= 0.5`, NaN becomes 0 and the value is clamped to the
/// target's bounds (exact in f32 for 8/16-bit targets), after which the
/// conversion is exact and unchecked. The result equals `value.round() as
/// $dst` for every input (`narrow_conversion_equals_round_then_saturate`).
macro_rules! round_narrow {
    ($src:ty, $value:expr, $dst:ty) => {{
        let x: $src = $value;
        let t = x.trunc();
        let r = t + if (x - t).abs() >= 0.5 {
            (1.0 as $src).copysign(x)
        } else {
            0.0
        };
        let clamped = if r.is_nan() {
            0.0
        } else {
            r.max(<$dst>::MIN as $src).min(<$dst>::MAX as $src)
        };
        // SAFETY: `clamped` is finite, integral and inside `$dst`'s range,
        // so it is representable as i32 and the conversion is exact.
        (unsafe { clamped.to_int_unchecked::<i32>() }) as $dst
    }};
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
cast_from!(double f64);

/// The conversion rule as an element map: each value of `S` to `D` by
/// [`CastFrom`], through the one traversal (`core::map`).
pub(crate) struct Convert<S, D>(PhantomData<fn(S) -> D>);

impl<S, D> Convert<S, D> {
    pub(crate) const fn new() -> Self {
        Convert(PhantomData)
    }
}

// SAFETY: `map_into` writes every element of `dst`.
unsafe impl<S: ViewType, D: ViewType + CastFrom<S>> ElementMap<S, D> for Convert<S, D> {
    #[inline(always)]
    fn map_into(&self, src: &[S], dst: &mut [MaybeUninit<D>], _at: usize) {
        for (d, &x) in dst.iter_mut().zip(src) {
            d.write(D::cast_from(x));
        }
    }
}

/// Every element of `src` converted to `D` by the crate's rule.
pub fn convert_slice<S: ViewType, D: ViewType + CastFrom<S>>(src: &[S]) -> Vec<D> {
    map_slice(src, &Convert::<S, D>::new())
}

/// Every element of `view`, in logical order, converted to `D` by the
/// crate's rule, as a new packed buffer of the view's shape: a view is read
/// in the runs its layout has, with no packed intermediate.
///
/// # Panics
/// Panics if `S` is not the view's dtype.
pub(crate) fn convert_view<S: ViewType, D: ViewType + CastFrom<S>>(
    view: &ViewBuffer,
) -> ViewBuffer {
    map_new(view, &Convert::<S, D>::new())
}

/// Elements per block of [`HalfBits`]: an 8 KiB f32 scratch on the stack.
const HALF_BLOCK: usize = 2048;

/// Each value as an IEEE-754 binary16 bit pattern: read as f32 by the
/// conversion rule, then rounded to nearest-even by `half` in bulk (F16C
/// eight at a time when the CPU has it, detected once per block, else its
/// software conversion). Per value, exactly `f16::from_f32(f32::cast_from(x))`.
pub(crate) struct HalfBits<S>(PhantomData<fn(S)>);

// SAFETY: every block of `dst` is written from the scratch of its block.
unsafe impl<S: ViewType> ElementMap<S, u16> for HalfBits<S>
where
    f32: CastFrom<S>,
{
    #[inline(always)]
    fn map_into(&self, src: &[S], dst: &mut [MaybeUninit<u16>], _at: usize) {
        use half::slice::HalfFloatSliceExt;
        for (src, dst) in src.chunks(HALF_BLOCK).zip(dst.chunks_mut(HALF_BLOCK)) {
            let mut wide = [0.0f32; HALF_BLOCK];
            let wide = &mut wide[..src.len()];
            for (w, &x) in wide.iter_mut().zip(src) {
                *w = f32::cast_from(x);
            }
            let mut narrow = [half::f16::ZERO; HALF_BLOCK];
            let narrow = &mut narrow[..src.len()];
            narrow.convert_from_f32_slice(wide);
            for (d, h) in dst.iter_mut().zip(narrow.iter()) {
                d.write(h.to_bits());
            }
        }
    }
}

/// [`ViewBuffer::to_f16_bits`]: `view`'s elements as binary16 bit patterns
/// in a new packed u16 buffer of its shape.
pub(crate) fn f16_bits<S: ViewType>(view: &ViewBuffer) -> ViewBuffer
where
    f32: CastFrom<S>,
{
    map_new(view, &HalfBits::<S>(PhantomData))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32 values at every edge of binary16: NaN payloads (quiet and
    /// signalling, both signs), ±0, ±inf, f16's largest finite value and
    /// the values either side of where rounding overflows to infinity,
    /// f16 subnormals and the underflow edge, and ties to even.
    fn half_edge_values() -> Vec<f32> {
        let mut v = vec![
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7F80_0001), // signalling NaN, low payload
            f32::from_bits(0x7FC0_1234),
            f32::from_bits(0xFFBF_FFFF),
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            65504.0,  // f16::MAX
            65519.99, // rounds to f16::MAX
            65520.0,  // the tie above f16::MAX: rounds to infinity
            -65520.0,
            1e10,
            6.103_515_6e-5, // smallest normal f16
            5.960_464_5e-8, // smallest subnormal f16
            2.980_232_2e-8, // half of it: ties to zero (even)
            2.980_233e-8,   // just above: rounds up to the smallest
            1e-30,
            1.0 + 1.0 / 2048.0, // tie between 1 and the next f16: to even
            1.0 + 3.0 / 2048.0, // tie: rounds up to even
            0.1,
            -1234.567,
        ];
        let mut state = 0x1234_5678u32;
        for _ in 0..5000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push(f32::from_bits(state));
        }
        v
    }

    fn per_element_f16(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect()
    }

    /// The half-precision bits of a buffer are what `f16::from_f32` gives
    /// for each element read as f32 (the conversion rule), for every edge
    /// value, over more than one block.
    #[test]
    fn f16_bits_are_the_per_element_conversion() {
        let values = half_edge_values();
        let buf = ViewBuffer::from_vec_with_shape(values.clone(), vec![values.len()]);
        let got = buf.to_f16_bits();
        assert_eq!(got.dtype(), crate::core::dtype::DType::U16);
        assert_eq!(got.shape(), buf.shape());
        assert_eq!(got.as_slice::<u16>(), &per_element_f16(&values)[..]);
    }

    /// Any dtype is read as f32 first (so f64 rounds to f32, then to f16,
    /// as `cast(F32)` then `from_f32` always did), and any layout is read
    /// in logical order.
    #[test]
    fn f16_bits_read_every_dtype_and_layout() {
        let (h, w, c) = (37, 29, 3);
        let n = h * w * c;
        let seed: Vec<f64> = (0..n).map(|i| (i as f64 * 7.37).sin() * 70_000.0).collect();
        let parents = [
            ViewBuffer::from_vec_with_shape(seed.clone(), vec![h, w, c]),
            ViewBuffer::from_vec_with_shape(
                seed.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                vec![h, w, c],
            ),
            ViewBuffer::from_vec_with_shape(
                seed.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                vec![h, w, c],
            ),
            ViewBuffer::from_vec_with_shape(
                seed.iter().map(|&x| x.abs() as u16).collect::<Vec<_>>(),
                vec![h, w, c],
            ),
        ];
        for parent in parents {
            for (layout, view) in [
                ("contiguous", parent.clone()),
                ("flip_h", parent.flip(&[1])),
                ("transpose", parent.permute(&[1, 0, 2])),
                ("crop", parent.slice(&[2, 3, 0], &[30, 20, 3])),
            ] {
                let as_f32 = view.cast_to(crate::core::dtype::DType::F32).to_contiguous();
                let want = per_element_f16(as_f32.as_slice::<f32>());
                let got = view.to_f16_bits();
                assert_eq!(got.shape(), view.shape(), "{:?} {layout}", parent.dtype());
                assert_eq!(
                    got.as_slice::<u16>(),
                    &want[..],
                    "{:?} {layout}",
                    parent.dtype()
                );
            }
        }
    }

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

    /// The 8/16-bit forms equal `x.round() as T` on a spread of every f32 and
    /// f64 bit pattern (and every special value).
    #[test]
    fn narrow_conversion_equals_round_then_saturate() {
        let specials = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.5,
            -0.5,
            1.5,
            -2.5,
            254.5,
            255.5,
            -128.5,
            127.5,
            32767.5,
            -32768.5,
            65535.5,
            0.49999997,
            8_388_609.0,
        ];
        let spread = (0..=u32::MAX).step_by(9973).map(f32::from_bits);
        for x in specials.into_iter().chain(spread) {
            assert_eq!(u8::cast_from(x), x.round() as u8, "u8 {x:e}");
            assert_eq!(i8::cast_from(x), x.round() as i8, "i8 {x:e}");
            assert_eq!(u16::cast_from(x), x.round() as u16, "u16 {x:e}");
            assert_eq!(i16::cast_from(x), x.round() as i16, "i16 {x:e}");
            let d = f64::from(x) * 1.000_000_1;
            assert_f64_narrow_rule(d);
        }
        let f64_specials = [
            0.499_999_999_999_999_94,
            -0.499_999_999_999_999_94,
            4_503_599_627_370_497.0,
            f64::MAX,
            f64::MIN_POSITIVE,
            255.499_999_999_999_97,
            -32_768.500_000_000_01,
        ];
        let spread = (0..=u64::MAX)
            .step_by(0x0000_7FF3_9A5B_C001)
            .map(f64::from_bits);
        for d in f64_specials.into_iter().chain(spread) {
            assert_f64_narrow_rule(d);
        }
    }

    fn assert_f64_narrow_rule(d: f64) {
        assert_eq!(u8::cast_from(d), d.round() as u8, "u8 {d:e}");
        assert_eq!(i8::cast_from(d), d.round() as i8, "i8 {d:e}");
        assert_eq!(u16::cast_from(d), d.round() as u16, "u16 {d:e}");
        assert_eq!(i16::cast_from(d), d.round() as i16, "i16 {d:e}");
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
