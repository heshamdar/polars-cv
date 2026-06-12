//! Naive-reference equivalence tests for the vectorized `apply_convolve2d`.
//!
//! The production kernel uses an interior/border split with per-tap
//! shifted-slice accumulation. These tests preserve the original per-pixel
//! gather implementation verbatim as `naive_convolve_reference` and assert
//! **bit-exact** equality: the optimized kernel visits taps in the same
//! `ky`-outer/`kx`-inner order and applies `norm_factor` as the same final
//! multiply, so no floating-point divergence is permitted.

use view_buffer::ops::filter::{apply_convolve2d, BorderMode, ConvolveOp};
use view_buffer::{DType, ViewBuffer};

// ---------------------------------------------------------------------------
// Reference implementation (the pre-optimization code, verbatim semantics)
// ---------------------------------------------------------------------------

fn ref_reflect_index(idx: i64, size: usize) -> usize {
    if idx < 0 {
        (-idx - 1).min(size as i64 - 1) as usize
    } else if idx >= size as i64 {
        let reflected = 2 * size as i64 - idx - 1;
        reflected.max(0) as usize
    } else {
        idx as usize
    }
}

#[allow(clippy::too_many_arguments)]
fn ref_sample_pixel(
    data: &[f32],
    h: usize,
    w: usize,
    c: usize,
    ch: usize,
    y: i64,
    x: i64,
    border: BorderMode,
) -> f32 {
    let (sy, sx) = match border {
        BorderMode::Zero => {
            if y < 0 || y >= h as i64 || x < 0 || x >= w as i64 {
                return 0.0;
            }
            (y as usize, x as usize)
        }
        BorderMode::Replicate => {
            let sy = y.clamp(0, h as i64 - 1) as usize;
            let sx = x.clamp(0, w as i64 - 1) as usize;
            (sy, sx)
        }
        BorderMode::Reflect => {
            let sy = ref_reflect_index(y, h);
            let sx = ref_reflect_index(x, w);
            (sy, sx)
        }
    };
    data[(sy * w + sx) * c + ch]
}

fn naive_convolve_reference(buf: &ViewBuffer, op: &ConvolveOp) -> ViewBuffer {
    let work_buf = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = work_buf.to_contiguous();
    let shape = contig.shape();

    let h = shape[0];
    let w = shape[1];
    let c = shape.get(2).copied().unwrap_or(1);

    let src = contig.as_slice::<f32>();

    let half = (op.ksize / 2) as i64;
    let kernel = &op.kernel;
    let ksize = op.ksize;

    let norm_factor = if op.normalize {
        let abs_sum: f32 = kernel.iter().map(|k| k.abs()).sum();
        if abs_sum > 0.0 {
            1.0 / abs_sum
        } else {
            1.0
        }
    } else {
        1.0
    };

    let mut output = vec![0.0f32; h * w * c];

    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                for ky in 0..ksize {
                    for kx in 0..ksize {
                        let sy = y as i64 + ky as i64 - half;
                        let sx = x as i64 + kx as i64 - half;
                        let pixel = ref_sample_pixel(src, h, w, c, ch, sy, sx, op.border);
                        sum += kernel[ky * ksize + kx] * pixel;
                    }
                }
                output[(y * w + x) * c + ch] = sum * norm_factor;
            }
        }
    }

    if c == 1 && shape.len() == 2 {
        ViewBuffer::from_vec_with_shape(output, vec![h, w])
    } else {
        ViewBuffer::from_vec_with_shape(output, vec![h, w, c])
    }
}

// ---------------------------------------------------------------------------
// Test inputs
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random f32 image in [0, 255] (LCG; no rand dep).
fn seeded_image(h: usize, w: usize, c: usize, seed: u64) -> ViewBuffer {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) % 256) as f32
    };
    let data: Vec<f32> = (0..h * w * c).map(|_| next()).collect();
    let shape = if c == 1 { vec![h, w] } else { vec![h, w, c] };
    ViewBuffer::from_vec_with_shape(data, shape)
}

/// Same image quantized to u8 (exercises the cast-to-f32 prologue).
fn seeded_image_u8(h: usize, w: usize, c: usize, seed: u64) -> ViewBuffer {
    seeded_image(h, w, c, seed).cast(DType::U8)
}

fn test_kernels(ksize: usize) -> Vec<(&'static str, Vec<f32>)> {
    let n = ksize * ksize;
    let mut kernels: Vec<(&'static str, Vec<f32>)> = Vec::new();

    // Identity (center 1).
    let mut identity = vec![0.0f32; n];
    identity[n / 2] = 1.0;
    kernels.push(("identity", identity));

    // Box.
    kernels.push(("box", vec![1.0; n]));

    // Seeded "random" kernel with mixed signs.
    let vals: Vec<f32> = (0..n).map(|i| ((i * 37 + 11) % 17) as f32 - 8.0).collect();
    kernels.push(("mixed", vals));

    if ksize == 3 {
        kernels.push(("sobel_x", vec![-1., 0., 1., -2., 0., 2., -1., 0., 1.]));
        kernels.push(("sobel_y", vec![-1., -2., -1., 0., 0., 0., 1., 2., 1.]));
        kernels.push(("sharpen", vec![-1., -1., -1., -1., 9., -1., -1., -1., -1.]));
    }

    kernels
}

fn assert_bit_exact(a: &ViewBuffer, b: &ViewBuffer, context: &str) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch: {context}");
    assert_eq!(a.dtype(), b.dtype(), "dtype mismatch: {context}");
    let av = a.as_slice::<f32>();
    let bv = b.as_slice::<f32>();
    for (i, (x, y)) in av.iter().zip(bv).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "bit mismatch at flat index {i}: {x:?} vs {y:?} ({context})"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn convolve_matches_naive_reference_bit_exact() {
    let sizes: &[(usize, usize)] = &[(1, 1), (3, 3), (4, 5), (5, 4), (17, 31), (64, 64)];
    let channels: &[usize] = &[1, 3, 4];
    let ksizes: &[usize] = &[3, 5, 7];
    let borders = [BorderMode::Replicate, BorderMode::Zero, BorderMode::Reflect];

    for &(h, w) in sizes {
        for &c in channels {
            let img = seeded_image(h, w, c, (h * 1000 + w * 10 + c) as u64);
            for &ksize in ksizes {
                for (kname, kernel) in test_kernels(ksize) {
                    for border in borders {
                        for normalize in [false, true] {
                            let op = ConvolveOp {
                                kernel: kernel.clone(),
                                ksize,
                                normalize,
                                border,
                            };
                            let got = apply_convolve2d(&img, &op);
                            let want = naive_convolve_reference(&img, &op);
                            assert_bit_exact(
                                &got,
                                &want,
                                &format!(
                                    "h={h} w={w} c={c} ksize={ksize} kernel={kname} \
                                     border={border:?} normalize={normalize}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn convolve_matches_naive_reference_u8_input() {
    // u8 input exercises the PromoteToFloat cast prologue.
    for &(h, w) in &[(8, 8), (33, 17)] {
        for &c in &[1usize, 3] {
            let img = seeded_image_u8(h, w, c, 7);
            for &ksize in &[3usize, 5] {
                let op = ConvolveOp {
                    kernel: test_kernels(ksize).pop().unwrap().1,
                    ksize,
                    normalize: false,
                    border: BorderMode::Replicate,
                };
                let got = apply_convolve2d(&img, &op);
                let want = naive_convolve_reference(&img, &op);
                assert_bit_exact(&got, &want, &format!("u8 h={h} w={w} c={c} ksize={ksize}"));
            }
        }
    }
}

#[test]
fn convolve_identity_kernel_is_identity_on_interior() {
    // Sanity beyond equivalence: identity kernel must reproduce the input
    // exactly (everywhere — borders too, since the center tap is in-bounds).
    let img = seeded_image(16, 16, 3, 99);
    let mut kernel = vec![0.0f32; 9];
    kernel[4] = 1.0;
    let op = ConvolveOp {
        kernel,
        ksize: 3,
        normalize: false,
        border: BorderMode::Replicate,
    };
    let got = apply_convolve2d(&img, &op);
    let src = img.as_slice::<f32>();
    let dst = got.as_slice::<f32>();
    for (i, (a, b)) in src.iter().zip(dst).enumerate() {
        // identity: sum = 0*… + 1*src + 0*… ; +0.0 terms keep the value.
        assert_eq!(a.to_bits(), b.to_bits(), "identity mismatch at {i}");
    }
}
