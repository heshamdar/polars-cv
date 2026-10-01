//! Where an element falls against a float boundary, exactly.
//!
//! **The one comparison of a pixel against a float boundary** — a
//! `threshold`, a histogram edge. Rounding an integer pixel to `f64` to
//! compare it answers for a neighbour above 2**53 (`2**53 + 3` rounds to the
//! edge `2**53 + 4`), so an integer is compared exactly instead: `p > t` iff
//! `p > floor(t)`, `p >= t` iff `p >= ceil(t)`, and every integer dtype fits
//! `i128`. A float compares in `f64`, which holds both float dtypes exactly.

/// A boundary `t`, with the integer cuts that decide it for an integer
/// element, computed once per boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cut {
    t: f64,
    /// `floor(t)`, saturated to ±2**100: `p > t` iff `p > floor`.
    floor: i128,
    /// `ceil(t)`, saturated to ±2**100: `p >= t` iff `p >= ceil`.
    ceil: i128,
}

impl Cut {
    pub(crate) fn of(t: f64) -> Self {
        // Beyond ±2**100 no integer dtype reaches, so a saturated cut decides
        // the same; NaN orders against nothing, so it is above everything.
        const BOUND: f64 = 1.2676506002282294e30; // 2**100
        let saturate = |v: f64| {
            if v.is_nan() || v >= BOUND {
                BOUND as i128
            } else if v <= -BOUND {
                -(BOUND as i128)
            } else {
                v as i128
            }
        };
        Cut {
            t,
            floor: saturate(t.floor()),
            ceil: saturate(t.ceil()),
        }
    }
}

/// An element type compared against a [`Cut`]: exact for every dtype. A NaN
/// element is neither above nor at any boundary.
pub(crate) trait AgainstCut: Copy {
    /// `self > t`.
    fn above(self, cut: &Cut) -> bool;
    /// `self >= t`.
    fn reaches(self, cut: &Cut) -> bool;
    fn is_nan(self) -> bool;
}

macro_rules! against_cut_int {
    ($($t:ty),+) => {$(
        impl AgainstCut for $t {
            #[inline(always)]
            fn above(self, cut: &Cut) -> bool {
                i128::from(self) > cut.floor
            }
            #[inline(always)]
            fn reaches(self, cut: &Cut) -> bool {
                i128::from(self) >= cut.ceil
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                false
            }
        }
    )+};
}
against_cut_int!(u8, i8, u16, i16, u32, i32, u64, i64);

macro_rules! against_cut_float {
    ($($t:ty),+) => {$(
        impl AgainstCut for $t {
            #[inline(always)]
            fn above(self, cut: &Cut) -> bool {
                f64::from(self) > cut.t
            }
            #[inline(always)]
            fn reaches(self, cut: &Cut) -> bool {
                f64::from(self) >= cut.t
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
        }
    )+};
}
against_cut_float!(f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every integer is compared exactly, where its f64 rounding is not:
    /// 2**53 + 3 rounds to 2**53 + 4 but is below it.
    #[test]
    fn an_integer_is_compared_exactly() {
        let e = Cut::of(9_007_199_254_740_996.0); // 2**53 + 4
        let below = 9_007_199_254_740_995u64;
        assert!(!below.reaches(&e) && !below.above(&e));
        assert!((below + 1).reaches(&e) && !(below + 1).above(&e));
        assert!((below + 2).above(&e));
        let half = Cut::of(2.5);
        assert!(!2i8.reaches(&half) && 3i8.reaches(&half) && 3i8.above(&half));
        assert!(!(-3i8).reaches(&Cut::of(-2.5)) && (-2i8).above(&Cut::of(-2.5)));
    }

    /// Infinite and NaN boundaries, and boundaries beyond every dtype.
    #[test]
    fn non_finite_and_far_boundaries() {
        for p in [i64::MIN, 0, i64::MAX] {
            assert!(p.above(&Cut::of(f64::NEG_INFINITY)) && p.reaches(&Cut::of(-1e300)));
            assert!(!p.reaches(&Cut::of(f64::INFINITY)) && !p.above(&Cut::of(1e300)));
            assert!(!p.reaches(&Cut::of(f64::NAN)) && !p.above(&Cut::of(f64::NAN)));
        }
        assert!(u64::MAX.reaches(&Cut::of(u64::MAX as f64 - 4096.0)));
        assert!(!u64::MAX.reaches(&Cut::of(18_446_744_073_709_551_616.0))); // 2**64
        assert!(f64::INFINITY.reaches(&Cut::of(f64::INFINITY)));
        assert!(!f64::NAN.reaches(&Cut::of(0.0)) && !f64::NAN.above(&Cut::of(0.0)));
    }
}
