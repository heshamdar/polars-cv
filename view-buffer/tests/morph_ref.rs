//! Naive-reference equivalence tests for the vectorized erode/dilate
//! (separable min/max) kernels.
//!
//! The production kernel uses an interior/border split with per-tap
//! shifted-slice elementwise min/max. These tests preserve the original
//! per-element clamped-gather implementation verbatim as
//! `naive_morph_reference` and assert **exact equality** — including float
//! inputs laced with NaN, where the original's strict-comparison semantics
//! (a NaN incumbent is kept, a NaN candidate is ignored) must be preserved
//! bit for bit.

#![cfg(feature = "image_interop")]

use view_buffer::{DType, ViewBuffer, ViewExpr};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Min,
    Max,
}

// ---------------------------------------------------------------------------
// Reference implementation (the pre-optimization code, verbatim semantics)
// ---------------------------------------------------------------------------

fn naive_pass<T>(src: &[T], h: usize, w: usize, ksize: u32, kind: Kind) -> Vec<T>
where
    T: Default + Copy + PartialOrd,
{
    let radius = (ksize / 2) as i64;

    // Row pass
    let mut row_out: Vec<T> = vec![T::default(); h * w];
    for y in 0..h {
        for x in 0..w {
            let mut val = src[y * w + x];
            for kx in -radius..=radius {
                let sx = (x as i64 + kx).clamp(0, w as i64 - 1) as usize;
                let candidate = src[y * w + sx];
                val = match kind {
                    Kind::Min => {
                        if candidate < val {
                            candidate
                        } else {
                            val
                        }
                    }
                    Kind::Max => {
                        if candidate > val {
                            candidate
                        } else {
                            val
                        }
                    }
                };
            }
            row_out[y * w + x] = val;
        }
    }

    // Column pass
    let mut col_out: Vec<T> = vec![T::default(); h * w];
    for y in 0..h {
        for x in 0..w {
            let mut val = row_out[y * w + x];
            for ky in -radius..=radius {
                let sy = (y as i64 + ky).clamp(0, h as i64 - 1) as usize;
                let candidate = row_out[sy * w + x];
                val = match kind {
                    Kind::Min => {
                        if candidate < val {
                            candidate
                        } else {
                            val
                        }
                    }
                    Kind::Max => {
                        if candidate > val {
                            candidate
                        } else {
                            val
                        }
                    }
                };
            }
            col_out[y * w + x] = val;
        }
    }

    col_out
}

fn naive_morph_reference<T>(
    src: &[T],
    h: usize,
    w: usize,
    ksize: u32,
    iterations: u32,
    kind: Kind,
) -> Vec<T>
where
    T: Default + Copy + PartialOrd + Clone,
{
    let mut cur: Vec<T> = src.to_vec();
    for _ in 0..iterations {
        if ksize <= 1 {
            continue; // morph_minmax_pass clones the input unchanged
        }
        cur = naive_pass(&cur, h, w, ksize, kind);
    }
    cur
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_morph(buf: ViewBuffer, ksize: u32, iterations: u32, kind: Kind) -> ViewBuffer {
    let expr = ViewExpr::new_source(buf);
    let expr = match kind {
        Kind::Min => expr.erode(ksize, iterations),
        Kind::Max => expr.dilate(ksize, iterations),
    };
    expr.plan().execute()
}

/// Deterministic pseudo-random values in [0, 256) (LCG; no rand dep).
fn seeded_values(n: usize, seed: u64) -> Vec<u64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % 256
        })
        .collect()
}

/// 100x100 binary mask: center square with noise pixels and holes (mirrors
/// the Python reference fixture's structure).
fn binary_mask(h: usize, w: usize) -> Vec<u64> {
    let mut img = vec![0u64; h * w];
    for y in h / 4..3 * h / 4 {
        for x in w / 4..3 * w / 4 {
            img[y * w + x] = 255;
        }
    }
    for (i, v) in seeded_values(h * w, 7).iter().enumerate() {
        if *v < 6 {
            img[i] = 255 - img[i]; // sparse noise + holes
        }
    }
    img
}

macro_rules! check_case {
    ($T:ty, $dtype:expr, $vals:expr, $h:expr, $w:expr, $ksize:expr, $iters:expr, $kind:expr) => {{
        let data: Vec<$T> = $vals.iter().map(|&v| v as $T).collect();
        let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![$h, $w]);
        let got = run_morph(buf, $ksize, $iters, $kind);
        let want = naive_morph_reference(&data, $h, $w, $ksize, $iters, $kind);
        assert_eq!(got.dtype(), $dtype);
        let got_slice = got.as_slice::<$T>();
        assert_eq!(
            got_slice,
            &want[..],
            "mismatch: dtype={:?} h={} w={} ksize={} iters={} kind={}",
            $dtype,
            $h,
            $w,
            $ksize,
            $iters,
            if $kind == Kind::Min {
                "erode"
            } else {
                "dilate"
            },
        );
    }};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn morph_matches_naive_reference_all_dtypes() {
    let sizes: &[(usize, usize)] = &[(1, 1), (2, 3), (5, 5), (64, 64)];
    let ksizes: &[u32] = &[1, 2, 3, 5, 9];

    for &(h, w) in sizes {
        let vals = seeded_values(h * w, (h * 100 + w) as u64);
        for &ksize in ksizes {
            for iters in [1u32, 2] {
                for kind in [Kind::Min, Kind::Max] {
                    check_case!(u8, DType::U8, vals, h, w, ksize, iters, kind);
                    check_case!(u16, DType::U16, vals, h, w, ksize, iters, kind);
                    check_case!(i32, DType::I32, vals, h, w, ksize, iters, kind);
                    check_case!(f32, DType::F32, vals, h, w, ksize, iters, kind);
                    check_case!(f64, DType::F64, vals, h, w, ksize, iters, kind);
                }
            }
        }
    }
}

#[test]
fn morph_matches_naive_reference_binary_mask() {
    let (h, w) = (100, 100);
    let vals = binary_mask(h, w);
    for &ksize in &[3u32, 5] {
        for iters in [1u32, 2] {
            for kind in [Kind::Min, Kind::Max] {
                check_case!(u8, DType::U8, vals, h, w, ksize, iters, kind);
            }
        }
    }
}

#[test]
fn morph_matches_naive_reference_with_nans() {
    // NaN incumbents must be kept and NaN candidates ignored, exactly like
    // the original strict-comparison fold. Compare bit patterns so NaN slots
    // are checked too.
    let (h, w) = (16, 16);
    let mut data: Vec<f32> = seeded_values(h * w, 3).iter().map(|&v| v as f32).collect();
    for i in (0..h * w).step_by(13) {
        data[i] = f32::NAN;
    }

    for &ksize in &[3u32, 5] {
        for kind in [Kind::Min, Kind::Max] {
            let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![h, w]);
            let got = run_morph(buf, ksize, 1, kind);
            let want = naive_morph_reference(&data, h, w, ksize, 1, kind);
            let got_slice = got.as_slice::<f32>();
            for (i, (g, e)) in got_slice.iter().zip(&want).enumerate() {
                assert!(
                    g.to_bits() == e.to_bits(),
                    "NaN-case bit mismatch at {i}: {g:?} vs {e:?} \
                     (ksize={ksize}, kind={})",
                    if kind == Kind::Min { "erode" } else { "dilate" },
                );
            }
        }
    }
}
