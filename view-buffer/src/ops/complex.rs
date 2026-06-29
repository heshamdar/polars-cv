//! Complex-buffer helper operations.
//!
//! Frequency-domain transforms ([`crate::ops::spectral`]) represent a complex
//! image as a two-channel `[H, W, 2]` `f32` buffer whose trailing axis holds
//! `[real, imag]`. These operations turn such a buffer into the real-valued
//! quantities used for spectral *analysis* — magnitude, phase and power spectra
//! — or manipulate it in the complex domain (conjugate, complex multiply).
//!
//! Keeping these as small, composable primitives means downstream analyses fall
//! out of composition (e.g. a centered log-magnitude spectrum is
//! `fft2 → roll → magnitude`), rather than each being a bespoke operation.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::cost::OpCost;
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Element-wise operations over a complex `[H, W, 2]` buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComplexOp {
    /// Magnitude `sqrt(re² + im²)` → real `[H, W, 1]` (the magnitude spectrum).
    Magnitude,
    /// Phase angle `atan2(im, re)` in radians → real `[H, W, 1]`.
    Phase,
    /// Power `re² + im²` → real `[H, W, 1]` (the power spectrum).
    Power,
    /// Complex conjugate `(re, -im)` → complex `[H, W, 2]`.
    Conj,
}

impl ComplexOp {
    /// Whether this op collapses the complex pair to a single real channel.
    fn is_reducing(&self) -> bool {
        matches!(self, ComplexOp::Magnitude | ComplexOp::Phase | ComplexOp::Power)
    }
}

impl Op for ComplexOp {
    fn name(&self) -> &'static str {
        match self {
            ComplexOp::Magnitude => "ComplexMagnitude",
            ComplexOp::Phase => "ComplexPhase",
            ComplexOp::Power => "ComplexPower",
            ComplexOp::Conj => "ComplexConj",
        }
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input = inputs[0];
        if self.is_reducing() && input.len() == 3 {
            vec![input[0], input[1], 1]
        } else {
            input.to_vec()
        }
    }

    fn output_rank_rule(&self) -> OutputRankRule {
        // The real/imag axis is fixed at length 2, so reducing ops keep rank 3
        // (`[H, W, 2]` → `[H, W, 1]`); conjugate is shape-identity.
        OutputRankRule::PreserveRank
    }

    fn output_channel_rule(&self) -> OutputChannelRule {
        if self.is_reducing() {
            OutputChannelRule::Fixed(1)
        } else {
            OutputChannelRule::PreserveChannels
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
        DTypeCategory::Float
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        OutputDTypeRule::PromoteToFloat
    }

    fn validate(
        &self,
        input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        let shape = input_shapes[0];
        let channels = shape.get(2).copied().unwrap_or(1);
        if shape.len() != 3 || channels != 2 {
            return Err(ValidationError::ShapeRequirement {
                requirement: "complex [H, W, 2] buffer (real/imag channels)",
                got: shape.to_vec(),
            });
        }
        Ok(())
    }
}

/// Apply a [`ComplexOp`] to a complex `[H, W, 2]` buffer.
///
/// Input is cast to `f32`; the trailing axis is interpreted as `[real, imag]`.
/// Reducing ops (`Magnitude`/`Phase`/`Power`) yield `[H, W, 1]`; `Conj` yields
/// `[H, W, 2]`.
pub fn apply_complex(buf: &ViewBuffer, op: ComplexOp) -> ViewBuffer {
    let work = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = work.to_contiguous();
    let shape = contig.shape();
    assert!(
        shape.len() == 3 && shape[2] == 2,
        "complex op requires [H, W, 2] input, got {shape:?}"
    );
    let h = shape[0];
    let w = shape[1];
    let src = contig.as_slice::<f32>();

    match op {
        ComplexOp::Conj => {
            let mut out = vec![0.0f32; h * w * 2];
            for i in 0..h * w {
                out[i * 2] = src[i * 2];
                out[i * 2 + 1] = -src[i * 2 + 1];
            }
            ViewBuffer::from_vec_with_shape(out, vec![h, w, 2])
        }
        _ => {
            let mut out = vec![0.0f32; h * w];
            for (i, o) in out.iter_mut().enumerate() {
                let re = src[i * 2];
                let im = src[i * 2 + 1];
                *o = match op {
                    ComplexOp::Magnitude => (re * re + im * im).sqrt(),
                    ComplexOp::Power => re * re + im * im,
                    ComplexOp::Phase => im.atan2(re),
                    ComplexOp::Conj => unreachable!(),
                };
            }
            ViewBuffer::from_vec_with_shape(out, vec![h, w, 1])
        }
    }
}

/// Element-wise complex multiply of two `[H, W, 2]` buffers.
///
/// `(a_re + i·a_im)·(b_re + i·b_im)`, producing a complex `[H, W, 2]` result.
/// This is the frequency-domain building block for fast convolution
/// (`fft2(a) · fft2(b)`) and cross-correlation (`fft2(a) · conj(fft2(b))`).
/// Both operands must share the same `[H, W, 2]` shape.
pub fn apply_complex_mul(a: &ViewBuffer, b: &ViewBuffer) -> ViewBuffer {
    let a = if a.dtype() != DType::F32 {
        a.cast(DType::F32)
    } else {
        a.clone()
    };
    let b = if b.dtype() != DType::F32 {
        b.cast(DType::F32)
    } else {
        b.clone()
    };
    let a = a.to_contiguous();
    let b = b.to_contiguous();
    let sa = a.shape();
    let sb = b.shape();
    assert!(
        sa.len() == 3 && sa[2] == 2,
        "complex_mul requires [H, W, 2] inputs, got {sa:?}"
    );
    assert!(
        sa == sb,
        "complex_mul operands must have matching shapes, got {sa:?} and {sb:?}"
    );
    let h = sa[0];
    let w = sa[1];
    let ad = a.as_slice::<f32>();
    let bd = b.as_slice::<f32>();
    let mut out = vec![0.0f32; h * w * 2];
    for i in 0..h * w {
        let (ar, ai) = (ad[i * 2], ad[i * 2 + 1]);
        let (br, bi) = (bd[i * 2], bd[i * 2 + 1]);
        out[i * 2] = ar * br - ai * bi;
        out[i * 2 + 1] = ar * bi + ai * br;
    }
    ViewBuffer::from_vec_with_shape(out, vec![h, w, 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complex_buf(h: usize, w: usize, vals: &[(f32, f32)]) -> ViewBuffer {
        let mut data = Vec::with_capacity(h * w * 2);
        for &(re, im) in vals {
            data.push(re);
            data.push(im);
        }
        ViewBuffer::from_vec_with_shape(data, vec![h, w, 2])
    }

    #[test]
    fn magnitude_phase_power() {
        // (3, 4) -> magnitude 5, power 25, phase atan2(4,3)
        let buf = complex_buf(1, 1, &[(3.0, 4.0)]);
        let mag = apply_complex(&buf, ComplexOp::Magnitude);
        assert_eq!(mag.shape(), &[1, 1, 1]);
        assert!((mag.as_slice::<f32>()[0] - 5.0).abs() < 1e-5);

        let pow = apply_complex(&buf, ComplexOp::Power);
        assert!((pow.as_slice::<f32>()[0] - 25.0).abs() < 1e-4);

        let ph = apply_complex(&buf, ComplexOp::Phase);
        assert!((ph.as_slice::<f32>()[0] - 4.0f32.atan2(3.0)).abs() < 1e-6);
    }

    #[test]
    fn conj_negates_imag() {
        let buf = complex_buf(1, 1, &[(3.0, 4.0)]);
        let c = apply_complex(&buf, ComplexOp::Conj);
        assert_eq!(c.shape(), &[1, 1, 2]);
        let s = c.as_slice::<f32>();
        assert_eq!(s[0], 3.0);
        assert_eq!(s[1], -4.0);
    }

    #[test]
    fn complex_mul_matches_hand_computation() {
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i^2 = -5 + 10i
        let a = complex_buf(1, 1, &[(1.0, 2.0)]);
        let b = complex_buf(1, 1, &[(3.0, 4.0)]);
        let r = apply_complex_mul(&a, &b);
        let s = r.as_slice::<f32>();
        assert!((s[0] - (-5.0)).abs() < 1e-5);
        assert!((s[1] - 10.0).abs() < 1e-5);
    }
}
