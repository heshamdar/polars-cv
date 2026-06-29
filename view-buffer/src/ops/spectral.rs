//! Frequency-domain transforms: 2D FFT/IFFT and 2D DCT/IDCT.
//!
//! These are the foundational spectral primitives. The forward FFT maps a
//! single-channel real image `[H, W, 1]` to a complex `[H, W, 2]` buffer
//! (trailing axis `[real, imag]`); the helpers in [`crate::ops::complex`] then
//! derive magnitude / phase / power spectra, and [`crate::ops::complex::apply_complex_mul`]
//! enables frequency-domain filtering and FFT-based convolution. The DCT-II /
//! DCT-III pair is a real-to-real transform (no complex representation needed),
//! foundational for energy-compaction and JPEG-style analysis.
//!
//! The actual transforms are backed by `rustfft` / `rustdct` and gated behind
//! the `spectral` feature. The [`SpectralOp`] type and its [`Op`] contract are
//! always available so plan-time schema inference compiles without the feature;
//! only the `apply_*` execution functions require it (mirroring `phash`).

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::cost::OpCost;
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Frequency-domain transform selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SpectralOp {
    /// Forward 2D FFT: real `[H, W, 1]` → complex `[H, W, 2]`.
    Fft2,
    /// Inverse 2D FFT: complex `[H, W, 2]` → real `[H, W, 1]` (real part).
    Ifft2,
    /// Forward 2D DCT-II (per channel): real `[H, W, C]` → real `[H, W, C]`.
    Dct2,
    /// Inverse 2D DCT (DCT-III, per channel): real `[H, W, C]` → real `[H, W, C]`.
    Idct2,
}

impl Op for SpectralOp {
    fn name(&self) -> &'static str {
        match self {
            SpectralOp::Fft2 => "Fft2",
            SpectralOp::Ifft2 => "Ifft2",
            SpectralOp::Dct2 => "Dct2",
            SpectralOp::Idct2 => "Idct2",
        }
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input = inputs[0];
        let h = input[0];
        let w = *input.get(1).unwrap_or(&1);
        match self {
            // Forward FFT always produces a 2-channel complex buffer.
            SpectralOp::Fft2 => vec![h, w, 2],
            // Inverse FFT returns a single real channel.
            SpectralOp::Ifft2 => vec![h, w, 1],
            // DCT/IDCT preserve shape (per-channel real transform).
            SpectralOp::Dct2 | SpectralOp::Idct2 => input.to_vec(),
        }
    }

    fn output_rank_rule(&self) -> OutputRankRule {
        OutputRankRule::PreserveRank
    }

    fn output_channel_rule(&self) -> OutputChannelRule {
        match self {
            SpectralOp::Fft2 => OutputChannelRule::Fixed(2),
            SpectralOp::Ifft2 => OutputChannelRule::Fixed(1),
            SpectralOp::Dct2 | SpectralOp::Idct2 => OutputChannelRule::PreserveChannels,
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        self.output_dtype_rule().resolve(inputs[0], None)
    }

    fn memory_effect(&self) -> MemoryEffect {
        MemoryEffect::RequiresContiguous
    }

    fn intrinsic_cost(&self) -> OpCost {
        OpCost::Allocating
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        None
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        DTypeCategory::Numeric
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self {
            // FFT results are always f32 (real and imaginary parts).
            SpectralOp::Fft2 | SpectralOp::Ifft2 => OutputDTypeRule::Fixed(DType::F32),
            // DCT promotes integers to f32, preserves floats.
            SpectralOp::Dct2 | SpectralOp::Idct2 => OutputDTypeRule::PromoteToFloat,
        }
    }

    fn validate(
        &self,
        input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        let shape = input_shapes[0];
        let channels = shape.get(2).copied().unwrap_or(1);
        match self {
            SpectralOp::Fft2 => {
                // Require an unambiguous single color channel: a complex buffer
                // is `[H, W, 2]`, so feeding multi-channel input is rejected.
                if !(shape.len() == 2 || (shape.len() == 3 && channels == 1)) {
                    return Err(ValidationError::ShapeRequirement {
                        requirement: "single-channel [H, W] or [H, W, 1] (call grayscale() first)",
                        got: shape.to_vec(),
                    });
                }
            }
            SpectralOp::Ifft2 => {
                if !(shape.len() == 3 && channels == 2) {
                    return Err(ValidationError::ShapeRequirement {
                        requirement: "complex [H, W, 2] buffer (real/imag channels)",
                        got: shape.to_vec(),
                    });
                }
            }
            SpectralOp::Dct2 | SpectralOp::Idct2 => {}
        }
        Ok(())
    }
}

/// Apply a [`SpectralOp`] to a buffer.
///
/// Requires the `spectral` feature; without it this panics, exactly as
/// perceptual hashing does without `perceptual_hash`.
#[cfg(feature = "spectral")]
pub fn apply_spectral(buf: &ViewBuffer, op: SpectralOp) -> ViewBuffer {
    match op {
        SpectralOp::Fft2 => fft2_forward(buf),
        SpectralOp::Ifft2 => fft2_inverse(buf),
        SpectralOp::Dct2 => dct2(buf, false),
        SpectralOp::Idct2 => dct2(buf, true),
    }
}

/// Stub used when the `spectral` feature is disabled.
#[cfg(not(feature = "spectral"))]
pub fn apply_spectral(_buf: &ViewBuffer, _op: SpectralOp) -> ViewBuffer {
    panic!("spectral operations require the 'spectral' feature (rustfft/rustdct) to be enabled");
}

#[cfg(feature = "spectral")]
use rustfft::num_complex::Complex;

/// Run an in-place separable 2D FFT over a row-major `h × w` complex grid.
#[cfg(feature = "spectral")]
fn fft2_inplace(data: &mut [Complex<f32>], h: usize, w: usize, inverse: bool) {
    use rustfft::FftPlanner;
    let mut planner = FftPlanner::<f32>::new();

    // Row transforms: rows are contiguous slices of length `w`.
    let fft_w = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    for r in 0..h {
        fft_w.process(&mut data[r * w..(r + 1) * w]);
    }

    // Column transforms: gather the strided column, transform, scatter back.
    let fft_h = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    let mut col = vec![Complex::new(0.0f32, 0.0f32); h];
    for c in 0..w {
        for (r, slot) in col.iter_mut().enumerate() {
            *slot = data[r * w + c];
        }
        fft_h.process(&mut col);
        for (r, &v) in col.iter().enumerate() {
            data[r * w + c] = v;
        }
    }
}

#[cfg(feature = "spectral")]
fn fft2_forward(buf: &ViewBuffer) -> ViewBuffer {
    let work = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = work.to_contiguous();
    let shape = contig.shape();
    let h = shape[0];
    let w = *shape.get(1).unwrap_or(&1);
    let src = contig.as_slice::<f32>();

    let mut data: Vec<Complex<f32>> = src.iter().map(|&re| Complex::new(re, 0.0)).collect();
    fft2_inplace(&mut data, h, w, false);

    let mut out = vec![0.0f32; h * w * 2];
    for (i, c) in data.iter().enumerate() {
        out[i * 2] = c.re;
        out[i * 2 + 1] = c.im;
    }
    ViewBuffer::from_vec_with_shape(out, vec![h, w, 2])
}

#[cfg(feature = "spectral")]
fn fft2_inverse(buf: &ViewBuffer) -> ViewBuffer {
    let work = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = work.to_contiguous();
    let shape = contig.shape();
    let h = shape[0];
    let w = shape[1];
    let src = contig.as_slice::<f32>();

    let mut data: Vec<Complex<f32>> = (0..h * w)
        .map(|i| Complex::new(src[i * 2], src[i * 2 + 1]))
        .collect();
    fft2_inplace(&mut data, h, w, true);

    // rustfft's inverse is unnormalized; divide by N to recover the original.
    let norm = 1.0f32 / (h * w) as f32;
    let mut out = vec![0.0f32; h * w];
    for (o, c) in out.iter_mut().zip(data.iter()) {
        *o = c.re * norm;
    }
    ViewBuffer::from_vec_with_shape(out, vec![h, w, 1])
}

/// Separable 2D DCT-II (forward) or DCT-III (inverse), applied per channel.
#[cfg(feature = "spectral")]
fn dct2(buf: &ViewBuffer, inverse: bool) -> ViewBuffer {
    use rustdct::DctPlanner;

    let work = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = work.to_contiguous();
    let shape = contig.shape().to_vec();
    let h = shape[0];
    let w = shape[1];
    let c = shape.get(2).copied().unwrap_or(1);
    let src = contig.as_slice::<f32>();

    let mut planner = DctPlanner::<f32>::new();
    let dct_w = if inverse {
        planner.plan_dct3(w)
    } else {
        planner.plan_dct2(w)
    };
    let dct_h = if inverse {
        planner.plan_dct3(h)
    } else {
        planner.plan_dct2(h)
    };
    // rustdct's DCT-III is the inverse of DCT-II scaled by len/2 per axis, so the
    // 2D round trip picks up a factor of (W/2)(H/2) = WH/4; undo it on inverse.
    let norm = if inverse {
        4.0f32 / (h as f32 * w as f32)
    } else {
        1.0
    };

    let mut out = vec![0.0f32; h * w * c];
    let mut plane = vec![0.0f32; h * w];
    let mut col = vec![0.0f32; h];
    for ch in 0..c {
        // Extract the channel plane.
        for i in 0..h * w {
            plane[i] = src[i * c + ch];
        }
        // Row pass.
        for r in 0..h {
            let row = &mut plane[r * w..(r + 1) * w];
            if inverse {
                dct_w.process_dct3(row);
            } else {
                dct_w.process_dct2(row);
            }
        }
        // Column pass.
        for cc in 0..w {
            for (r, slot) in col.iter_mut().enumerate() {
                *slot = plane[r * w + cc];
            }
            if inverse {
                dct_h.process_dct3(&mut col);
            } else {
                dct_h.process_dct2(&mut col);
            }
            for (r, &v) in col.iter().enumerate() {
                plane[r * w + cc] = v;
            }
        }
        // Write the channel plane back (normalized on inverse).
        for i in 0..h * w {
            out[i * c + ch] = plane[i] * norm;
        }
    }
    ViewBuffer::from_vec_with_shape(out, shape)
}

#[cfg(all(test, feature = "spectral"))]
mod tests {
    use super::*;

    #[test]
    fn fft_of_constant_has_dc_only() {
        // A constant image: all energy concentrates in the DC (0,0) bin.
        let buf = ViewBuffer::from_vec_with_shape(vec![2.0f32; 16], vec![4, 4, 1]);
        let spec = apply_spectral(&buf, SpectralOp::Fft2);
        assert_eq!(spec.shape(), &[4, 4, 2]);
        let s = spec.as_slice::<f32>();
        // DC bin = sum of all pixels = 2*16 = 32 (real), 0 imag.
        assert!((s[0] - 32.0).abs() < 1e-3);
        assert!(s[1].abs() < 1e-3);
        // Every other bin is ~0.
        for i in 1..16 {
            assert!(s[i * 2].abs() < 1e-2, "bin {i} re not zero");
            assert!(s[i * 2 + 1].abs() < 1e-2, "bin {i} im not zero");
        }
    }

    #[test]
    fn fft_ifft_round_trip() {
        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 1.5 - 3.0).collect();
        let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![4, 4, 1]);
        let spec = apply_spectral(&buf, SpectralOp::Fft2);
        let back = apply_spectral(&spec, SpectralOp::Ifft2);
        assert_eq!(back.shape(), &[4, 4, 1]);
        let s = back.as_slice::<f32>();
        for (i, &v) in s.iter().enumerate() {
            assert!((v - data[i]).abs() < 1e-3, "idx {i}: {v} != {}", data[i]);
        }
    }

    #[test]
    fn dct_idct_round_trip() {
        let data: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
        let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![4, 4, 1]);
        let fwd = apply_spectral(&buf, SpectralOp::Dct2);
        let back = apply_spectral(&fwd, SpectralOp::Idct2);
        let s = back.as_slice::<f32>();
        for (i, &v) in s.iter().enumerate() {
            assert!((v - data[i]).abs() < 1e-3, "idx {i}: {v} != {}", data[i]);
        }
    }

    #[test]
    fn fft_known_single_frequency() {
        // A horizontal cosine of frequency 1 over width 4 has energy in bins ±1.
        let mut data = vec![0.0f32; 16];
        for r in 0..4 {
            for cc in 0..4 {
                data[r * 4 + cc] = (2.0 * std::f32::consts::PI * cc as f32 / 4.0).cos();
            }
        }
        let buf = ViewBuffer::from_vec_with_shape(data, vec![4, 4, 1]);
        let spec = apply_spectral(&buf, SpectralOp::Fft2);
        let mag =
            crate::ops::complex::apply_complex(&spec, crate::ops::complex::ComplexOp::Magnitude);
        let m = mag.as_slice::<f32>();
        // Row 0 (DC in vertical), columns 1 and 3 carry the energy.
        assert!(m[1] > 1.0, "bin (0,1) should carry energy, got {}", m[1]);
        assert!(m[3] > 1.0, "bin (0,3) should carry energy, got {}", m[3]);
        assert!(m[2].abs() < 1e-2, "bin (0,2) should be ~0, got {}", m[2]);
    }
}
