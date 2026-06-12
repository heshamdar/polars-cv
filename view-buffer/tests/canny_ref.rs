//! Naive-reference equivalence tests for the optimized Canny implementation.
//!
//! The production kernel replaces the per-pixel `atan2` direction
//! quantization, the naive 25-tap blur loop, and the fixpoint-sweep
//! hysteresis with vectorizable equivalents. This file preserves the
//! original implementation verbatim as `naive_canny_reference` and asserts
//! **exact equality** of the u8 edge maps.
//!
//! The only theoretically permitted divergence is a gradient direction whose
//! angle sits within ~1 ulp of a 22.5°/67.5°/112.5°/157.5° bin boundary
//! (where the old atan2 answer was itself rounding-determined); the
//! direction-grid unit test in `execution/runner.rs` bounds that set. If this
//! file ever fails, that is the only acceptable root cause — investigate, do
//! not loosen the assertion.

#![cfg(feature = "image_interop")]

use view_buffer::{DType, ImageOp, ImageOpKind, ViewBuffer, ViewDto, ViewExpr};

// ---------------------------------------------------------------------------
// Reference implementation (the pre-optimization code, verbatim)
// ---------------------------------------------------------------------------

fn ref_gaussian_blur_5x5(src: &[f32], h: usize, w: usize) -> Vec<f32> {
    #[rustfmt::skip]
    let kernel: [f32; 25] = [
        2.0/159.0,  4.0/159.0,  5.0/159.0,  4.0/159.0,  2.0/159.0,
        4.0/159.0,  9.0/159.0, 12.0/159.0,  9.0/159.0,  4.0/159.0,
        5.0/159.0, 12.0/159.0, 15.0/159.0, 12.0/159.0,  5.0/159.0,
        4.0/159.0,  9.0/159.0, 12.0/159.0,  9.0/159.0,  4.0/159.0,
        2.0/159.0,  4.0/159.0,  5.0/159.0,  4.0/159.0,  2.0/159.0,
    ];
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for ky in 0..5i64 {
                for kx in 0..5i64 {
                    let sy = (y as i64 + ky - 2).clamp(0, h as i64 - 1) as usize;
                    let sx = (x as i64 + kx - 2).clamp(0, w as i64 - 1) as usize;
                    sum += kernel[(ky * 5 + kx) as usize] * src[sy * w + sx];
                }
            }
            out[y * w + x] = sum;
        }
    }
    out
}

fn ref_sobel_gradients(src: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; h * w];
    let mut gy = vec![0.0f32; h * w];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let tl = src[(y - 1) * w + x - 1];
            let tc = src[(y - 1) * w + x];
            let tr = src[(y - 1) * w + x + 1];
            let ml = src[y * w + x - 1];
            let mr = src[y * w + x + 1];
            let bl = src[(y + 1) * w + x - 1];
            let bc = src[(y + 1) * w + x];
            let br = src[(y + 1) * w + x + 1];

            gx[y * w + x] = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
            gy[y * w + x] = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
        }
    }
    (gx, gy)
}

/// The original `apply_canny` body on a single-channel f32 plane.
fn naive_canny_reference(src: &[f32], gh: usize, gw: usize, low: f32, high: f32) -> Vec<u8> {
    let count = gh * gw;

    let blurred = ref_gaussian_blur_5x5(src, gh, gw);
    let (gx, gy) = ref_sobel_gradients(&blurred, gh, gw);

    let mut magnitude = vec![0.0f32; count];
    let mut direction = vec![0u8; count];
    for i in 0..count {
        magnitude[i] = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
        let angle = gy[i].atan2(gx[i]).to_degrees();
        let angle = if angle < 0.0 { angle + 180.0 } else { angle };
        direction[i] = if !(22.5..157.5).contains(&angle) {
            0
        } else if angle < 67.5 {
            1
        } else if angle < 112.5 {
            2
        } else {
            3
        };
    }

    let mut nms = vec![0.0f32; count];
    for y in 1..gh - 1 {
        for x in 1..gw - 1 {
            let idx = y * gw + x;
            let mag = magnitude[idx];
            let (n1, n2) = match direction[idx] {
                0 => (magnitude[idx - 1], magnitude[idx + 1]),
                1 => (
                    magnitude[(y - 1) * gw + x + 1],
                    magnitude[(y + 1) * gw + x - 1],
                ),
                2 => (magnitude[(y - 1) * gw + x], magnitude[(y + 1) * gw + x]),
                _ => (
                    magnitude[(y - 1) * gw + x - 1],
                    magnitude[(y + 1) * gw + x + 1],
                ),
            };
            if mag >= n1 && mag >= n2 {
                nms[idx] = mag;
            }
        }
    }

    let mut edges = vec![0u8; count];
    const STRONG: u8 = 255;
    const WEAK: u8 = 128;
    for i in 0..count {
        if nms[i] >= high {
            edges[i] = STRONG;
        } else if nms[i] >= low {
            edges[i] = WEAK;
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..gh - 1 {
            for x in 1..gw - 1 {
                let idx = y * gw + x;
                if edges[idx] == WEAK {
                    let has_strong_neighbor = edges[(y - 1) * gw + x - 1] == STRONG
                        || edges[(y - 1) * gw + x] == STRONG
                        || edges[(y - 1) * gw + x + 1] == STRONG
                        || edges[y * gw + x - 1] == STRONG
                        || edges[y * gw + x + 1] == STRONG
                        || edges[(y + 1) * gw + x - 1] == STRONG
                        || edges[(y + 1) * gw + x] == STRONG
                        || edges[(y + 1) * gw + x + 1] == STRONG;
                    if has_strong_neighbor {
                        edges[idx] = STRONG;
                        changed = true;
                    }
                }
            }
        }
    }
    for e in &mut edges {
        if *e == WEAK {
            *e = 0;
        }
    }

    edges
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_canny(plane: &[f32], h: usize, w: usize, low: f32, high: f32) -> Vec<u8> {
    let buf = ViewBuffer::from_vec_with_shape(plane.to_vec(), vec![h, w]);
    let out = ViewExpr::new_source(buf)
        .apply_op(ViewDto::Image(ImageOp {
            kind: ImageOpKind::Canny {
                low_threshold: low,
                high_threshold: high,
            },
        }))
        .plan()
        .execute();
    assert_eq!(out.dtype(), DType::U8);
    out.as_slice::<u8>().to_vec()
}

/// Deterministic pseudo-random u8-valued f32 plane (LCG; no rand dep).
fn seeded_plane(h: usize, w: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..h * w)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 256) as f32
        })
        .collect()
}

/// Bright rectangle on a noisy background (mirrors the pytest fixture).
fn rectangle_on_noise(h: usize, w: usize) -> Vec<f32> {
    let mut img = seeded_plane(h, w, 42);
    for v in img.iter_mut() {
        *v = 20.0 + *v * (180.0 / 255.0); // background noise in [20, 200)
    }
    for y in h / 3..2 * h / 3 {
        for x in w / 3..2 * w / 3 {
            img[y * w + x] = 200.0;
        }
    }
    img
}

fn linear_ramp(h: usize, w: usize) -> Vec<f32> {
    (0..h * w)
        .map(|i| ((i % w) as f32 / w as f32) * 255.0)
        .collect()
}

fn diagonal_stripes(h: usize, w: usize) -> Vec<f32> {
    (0..h * w)
        .map(|i| {
            let (y, x) = (i / w, i % w);
            if ((y + x) / 8) % 2 == 0 {
                30.0
            } else {
                220.0
            }
        })
        .collect()
}

fn check_image(name: &str, plane: &[f32], h: usize, w: usize) {
    for (low, high) in [(50.0f32, 150.0f32), (30.0, 100.0), (10.0, 20.0), (0.0, 0.0)] {
        let got = run_canny(plane, h, w, low, high);
        let want = naive_canny_reference(plane, h, w, low, high);
        assert_eq!(
            got, want,
            "edge map mismatch: image={name} h={h} w={w} low={low} high={high}"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn canny_matches_naive_reference_random() {
    check_image("random64", &seeded_plane(64, 64, 1), 64, 64);
    check_image("random256", &seeded_plane(256, 256, 2), 256, 256);
}

#[test]
fn canny_matches_naive_reference_structured() {
    check_image("rectangle", &rectangle_on_noise(96, 96), 96, 96);
    check_image("ramp", &linear_ramp(64, 64), 64, 64);
    check_image("stripes", &diagonal_stripes(64, 64), 64, 64);
    check_image("constant", &vec![128.0f32; 64 * 64], 64, 64);
}

#[test]
fn canny_tiny_images_degrade_gracefully() {
    // h or w < 3 means no interior pixels: the original loops were empty
    // ranges; the optimized version must produce the same (all
    // zero-classified) maps without panicking.
    for (h, w) in [(2usize, 2usize), (1, 5), (5, 1), (2, 8)] {
        let plane = seeded_plane(h, w, 9);
        check_image("tiny", &plane, h, w);
    }
}

#[test]
fn canny_u8_input_matches_reference() {
    // u8 input exercises the cast prologue inside apply_canny.
    let (h, w) = (64usize, 64usize);
    let plane = seeded_plane(h, w, 5);
    let u8_buf = ViewBuffer::from_vec_with_shape(plane.clone(), vec![h, w]).cast(DType::U8);
    let out = ViewExpr::new_source(u8_buf)
        .apply_op(ViewDto::Image(ImageOp {
            kind: ImageOpKind::Canny {
                low_threshold: 50.0,
                high_threshold: 150.0,
            },
        }))
        .plan()
        .execute();
    let want = naive_canny_reference(&plane, h, w, 50.0, 150.0);
    assert_eq!(out.as_slice::<u8>(), &want[..]);
}
