//! Naive-reference equivalence tests for the optimized affine warp.
//!
//! The production kernel monomorphizes the interpolation dispatch out of the
//! per-pixel loop and adds an interior fast path that skips per-tap bounds
//! checks. All arithmetic (f64 coordinates with the original expression
//! association, f64 bilinear weights, round-then-saturate stores) is
//! unchanged, so these tests preserve the original implementation verbatim
//! as `naive_affine_reference` and assert **bit-exact** equality.

use view_buffer::{AffineParams, ComputeOp, InterpolationType, ViewBuffer, ViewDto, ViewExpr};

// ---------------------------------------------------------------------------
// Reference implementation (the pre-optimization code, verbatim semantics)
// ---------------------------------------------------------------------------

fn ref_clamp_u8(v: f64) -> f64 {
    v.round().clamp(0.0, u8::MAX as f64)
}

/// The original `affine_warp_typed` body, specialized to one element type via
/// closures for load/store conversion.
#[allow(clippy::too_many_arguments)]
fn naive_affine_reference<T: Copy + Default>(
    src_data: &[T],
    in_h: usize,
    in_w: usize,
    channels: usize,
    params: &AffineParams,
    to_f64: impl Fn(T) -> f64,
    from_f64: impl Fn(f64) -> T,
    is_float: bool,
) -> Vec<T> {
    let out_h = params.output_height as usize;
    let out_w = params.output_width as usize;

    // Every matrix reaching here is invertible (see `test_matrices`), so the
    // reference inverts unconditionally — exactly as the runner does.
    let [a_fwd, b_fwd, tx_fwd, c_fwd, d_fwd, ty_fwd] = params.matrix;
    let inv_det = 1.0 / (a_fwd * d_fwd - b_fwd * c_fwd);
    let a = d_fwd * inv_det;
    let b = -b_fwd * inv_det;
    let c = -c_fwd * inv_det;
    let d = a_fwd * inv_det;
    let tx = -(a * tx_fwd + b * ty_fwd);
    let ty = -(c * tx_fwd + d * ty_fwd);

    let border_val: T = from_f64(params.border_value);
    let mut dst_data: Vec<T> = vec![border_val; out_h * out_w * channels];

    for y_dst in 0..out_h {
        for x_dst in 0..out_w {
            let x_src = a * x_dst as f64 + b * y_dst as f64 + tx;
            let y_src = c * x_dst as f64 + d * y_dst as f64 + ty;

            match params.interpolation {
                InterpolationType::Nearest => {
                    let sx = x_src.round() as i64;
                    let sy = y_src.round() as i64;
                    if sx >= 0 && sy >= 0 && (sx as usize) < in_w && (sy as usize) < in_h {
                        let src_idx = (sy as usize * in_w + sx as usize) * channels;
                        let dst_idx = (y_dst * out_w + x_dst) * channels;
                        dst_data[dst_idx..dst_idx + channels]
                            .copy_from_slice(&src_data[src_idx..src_idx + channels]);
                    }
                }
                InterpolationType::Bilinear => {
                    let x0 = x_src.floor() as i64;
                    let y0 = y_src.floor() as i64;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;

                    if x1 < 0 || y1 < 0 || x0 >= in_w as i64 || y0 >= in_h as i64 {
                        continue;
                    }

                    let dx = x_src - x0 as f64;
                    let dy = y_src - y0 as f64;
                    let bv: f64 = params.border_value;
                    let in_bounds = |px: i64, py: i64| -> bool {
                        px >= 0 && py >= 0 && (px as usize) < in_w && (py as usize) < in_h
                    };

                    let dst_idx = (y_dst * out_w + x_dst) * channels;
                    for ch in 0..channels {
                        let sample = |px: i64, py: i64| -> f64 {
                            if in_bounds(px, py) {
                                let idx = (py as usize * in_w + px as usize) * channels + ch;
                                to_f64(src_data[idx])
                            } else {
                                bv
                            }
                        };

                        let v00 = sample(x0, y0);
                        let v10 = sample(x1, y0);
                        let v01 = sample(x0, y1);
                        let v11 = sample(x1, y1);

                        let v0 = v00 * (1.0 - dx) + v10 * dx;
                        let v1 = v01 * (1.0 - dx) + v11 * dx;
                        let v = v0 * (1.0 - dy) + v1 * dy;

                        let clamped = if is_float { v } else { ref_clamp_u8(v) };
                        dst_data[dst_idx + ch] = from_f64(clamped);
                    }
                }
            }
        }
    }

    dst_data
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_affine(buf: ViewBuffer, params: AffineParams) -> ViewBuffer {
    ViewExpr::new_source(buf)
        .apply_op(ViewDto::Compute(ComputeOp::Affine(params)))
        .plan()
        .execute()
}

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

/// Test matrices: (name, forward 2x3 matrix, out_h, out_w).
fn test_matrices(h: usize, w: usize) -> Vec<(&'static str, [f64; 6], u32, u32)> {
    let (hf, wf) = (h as f64, w as f64);
    let (cx, cy) = (wf / 2.0, hf / 2.0);
    let th = 45.0f64.to_radians();
    let (cos, sin) = (th.cos(), th.sin());
    vec![
        (
            "identity",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            h as u32,
            w as u32,
        ),
        (
            "translate_int",
            [1.0, 0.0, 3.0, 0.0, 1.0, -2.0],
            h as u32,
            w as u32,
        ),
        (
            "translate_frac",
            [1.0, 0.0, 1.5, 0.0, 1.0, 0.25],
            h as u32,
            w as u32,
        ),
        (
            "rotate45_center",
            [
                cos,
                -sin,
                cx - cos * cx + sin * cy,
                sin,
                cos,
                cy - sin * cx - cos * cy,
            ],
            h as u32,
            w as u32,
        ),
        (
            "scale_up",
            [2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            2 * h as u32,
            2 * w as u32,
        ),
        (
            "scale_down",
            [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            (h / 2).max(1) as u32,
            (w / 2).max(1) as u32,
        ),
        ("shear", [1.0, 0.3, 0.0, 0.1, 1.0, 0.0], h as u32, w as u32),
        // A singular matrix is deliberately absent: it has no inverse, so
        // there is no reference result to match. It used to sit here pinning
        // the runner's identity fallback — see `singular_matrices_are_not_
        // invertible` below, and the plugin's `warp_affine` arm, which rejects
        // one before it can reach execution.
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn affine_bilinear_matches_naive_reference_u8() {
    for &(h, w) in &[(16usize, 16usize), (33, 17), (64, 64)] {
        for &c in &[1usize, 3, 4] {
            let vals = seeded_values(h * w * c, (h * 31 + w * 7 + c) as u64);
            let data: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
            let shape = if c == 1 { vec![h, w] } else { vec![h, w, c] };
            for (name, matrix, out_h, out_w) in test_matrices(h, w) {
                let params = AffineParams {
                    matrix,
                    output_height: out_h,
                    output_width: out_w,
                    interpolation: InterpolationType::Bilinear,
                    border_value: 0.0,
                };
                let buf = ViewBuffer::from_vec_with_shape(data.clone(), shape.clone());
                let got = run_affine(buf, params.clone());
                let want = naive_affine_reference(
                    &data,
                    h,
                    w,
                    c,
                    &params,
                    |v: u8| v as f64,
                    |f: f64| f as u8,
                    false,
                );
                assert_eq!(
                    got.as_slice::<u8>(),
                    &want[..],
                    "u8 bilinear mismatch: {name} h={h} w={w} c={c}"
                );
            }
        }
    }
}

#[test]
fn affine_bilinear_matches_naive_reference_f32() {
    let (h, w, c) = (32usize, 24usize, 3usize);
    let data: Vec<f32> = seeded_values(h * w * c, 11)
        .iter()
        .map(|&v| v as f32)
        .collect();
    for (name, matrix, out_h, out_w) in test_matrices(h, w) {
        let params = AffineParams {
            matrix,
            output_height: out_h,
            output_width: out_w,
            interpolation: InterpolationType::Bilinear,
            border_value: -1.0,
        };
        let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![h, w, c]);
        let got = run_affine(buf, params.clone());
        let want = naive_affine_reference(
            &data,
            h,
            w,
            c,
            &params,
            |v: f32| v as f64,
            |f: f64| f as f32,
            true,
        );
        let got_slice = got.as_slice::<f32>();
        for (i, (g, e)) in got_slice.iter().zip(&want).enumerate() {
            assert!(
                g.to_bits() == e.to_bits(),
                "f32 bilinear bit mismatch at {i}: {g:?} vs {e:?} ({name})"
            );
        }
    }
}

#[test]
fn affine_nearest_matches_naive_reference() {
    let (h, w, c) = (24usize, 24usize, 3usize);
    let data: Vec<u8> = seeded_values(h * w * c, 23)
        .iter()
        .map(|&v| v as u8)
        .collect();
    for (name, matrix, out_h, out_w) in test_matrices(h, w) {
        let params = AffineParams {
            matrix,
            output_height: out_h,
            output_width: out_w,
            interpolation: InterpolationType::Nearest,
            border_value: 7.0,
        };
        let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![h, w, c]);
        let got = run_affine(buf, params.clone());
        let want = naive_affine_reference(
            &data,
            h,
            w,
            c,
            &params,
            |v: u8| v as f64,
            |f: f64| f as u8,
            false,
        );
        assert_eq!(got.as_slice::<u8>(), &want[..], "nearest mismatch: {name}");
    }
}

#[test]
fn affine_interior_span_edge_cases() {
    // Translations chosen so the in-bounds interior span of a row is empty,
    // a single pixel, or the full row — exercising the fast/slow path
    // boundary in the optimized kernel.
    let (h, w) = (8usize, 8usize);
    let data: Vec<u8> = seeded_values(h * w, 3).iter().map(|&v| v as u8).collect();
    for (name, tx, ty) in [
        ("all_out", 100.0, 100.0),
        ("one_col", -(w as f64) + 1.5, 0.5),
        ("full_row", 0.25, 0.25),
        ("half_out_left", (w as f64) / 2.0, 0.0),
        ("half_out_top", 0.0, (h as f64) / 2.0),
    ] {
        let params = AffineParams {
            matrix: [1.0, 0.0, tx, 0.0, 1.0, ty],
            output_height: h as u32,
            output_width: w as u32,
            interpolation: InterpolationType::Bilinear,
            border_value: 9.0,
        };
        let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![h, w]);
        let got = run_affine(buf, params.clone());
        let want = naive_affine_reference(
            &data,
            h,
            w,
            1,
            &params,
            |v: u8| v as f64,
            |f: f64| f as u8,
            false,
        );
        assert_eq!(
            got.as_slice::<u8>(),
            &want[..],
            "span case mismatch: {name}"
        );
    }
}

/// The matrices the equivalence sweep above must never be handed.
///
/// Inverse mapping has no answer for a transform that collapses the plane, so
/// these are rejected where a matrix is accepted rather than silently replaced
/// with the identity — which is what the runner used to do, reporting a warp
/// that had not happened. `is_invertible` is the single authority for the
/// question; both the plugin's rejection and the runner's `debug_assert` read
/// it.
#[test]
fn singular_matrices_are_not_invertible() {
    let singular: &[(&str, [f64; 6])] = &[
        ("zero scale on both axes", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("zero scale on x", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ("zero scale on y", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("proportional rows", [1.0, 2.0, 0.0, 2.0, 4.0, 0.0]),
        ("underflows to zero", [1e-20, 0.0, 0.0, 0.0, 1e-20, 0.0]),
    ];
    for (name, matrix) in singular {
        let params = AffineParams {
            matrix: *matrix,
            output_height: 8,
            output_width: 8,
            interpolation: InterpolationType::Bilinear,
            border_value: 0.0,
        };
        assert!(
            !params.is_invertible(),
            "{name}: {matrix:?} has determinant {} and must be refused",
            params.determinant()
        );
    }

    // The counterpart: an extreme but genuine transform stays usable. A
    // conditioning test rather than a zero test would have failed here.
    let stretched = AffineParams {
        matrix: [1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0],
        output_height: 8,
        output_width: 8,
        interpolation: InterpolationType::Bilinear,
        border_value: 0.0,
    };
    assert!(
        stretched.is_invertible(),
        "a heavily stretched but invertible matrix must still be accepted"
    );
}
