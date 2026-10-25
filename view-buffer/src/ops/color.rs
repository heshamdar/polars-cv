//! Color space conversion operations.
//!
//! Supports conversions between RGB, BGR, HSV, LAB, YCbCr, and Grayscale color spaces.
//! Follows OpenCV's value ranges per dtype (`ColorRange`): u8 0..255 with
//! H in [0, 180); u16/u32/u64 0..MAX with a full-range hue; floats [0, 1]
//! with H in degrees. Signed integers have no colour range.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::shape_rule::OpShape;
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};

use crate::kernels::color::{apply_color_convert_core, merge_alpha, split_alpha};
use crate::mode::{Exec, Mode};
use polars_cv_macros::{Ops, Resolve};

/// Supported color spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Rgb,
    Bgr,
    Hsv,
    Lab,
    YCbCr,
    Gray,
}

crate::naming::named_variants!(ColorSpace: "Supported color spaces for ``convert_color``." {
    "rgb" => Rgb,
    "bgr" => Bgr,
    "hsv" => Hsv,
    "lab" => Lab,
    "ycbcr" => YCbCr,
    "gray" => Gray,
});

impl ColorSpace {
    /// Number of channels for this color space.
    pub fn channels(&self) -> usize {
        match self {
            Self::Gray => 1,
            _ => 3,
        }
    }
}

/// Convert between color spaces.
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").convert_color(from_space="rgb", to_space="hsv")
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
#[op(name = "cvt_color", python = "convert_color",
     sample = {"from_space": "rgb", "to_space": "hsv"})]
pub struct ColorConvertOp<M: Mode = Exec> {
    /// Source color space (rgb, bgr, hsv, lab, ycbcr, gray).
    pub from_space: M::L<ColorSpace>,
    /// Target color space (rgb, bgr, hsv, lab, ycbcr, gray).
    pub to_space: M::L<ColorSpace>,
}

impl<M: Mode> ColorConvertOp<M> {
    /// Both spaces are structural, and any pair converts.
    pub fn check(&self) -> Result<(), String> {
        Ok(())
    }

    /// The target's colour channels, plus a carried alpha.
    pub fn shape(&self) -> OpShape {
        let to = M::lit(&self.to_space);
        OpShape::ColorChannels {
            channels: to.channels(),
            to_gray: to == ColorSpace::Gray,
        }
    }
}

impl<M: Mode> ColorConvertOp<M> {
    /// Whether the conversion promotes dtype to f32.
    ///
    /// LAB conversions require float math and output f32.
    /// All other conversions preserve the input dtype.
    pub fn promotes_to_float(&self) -> bool {
        M::lit(&self.from_space) == ColorSpace::Lab || M::lit(&self.to_space) == ColorSpace::Lab
    }
}

impl<M: Mode> Op for ColorConvertOp<M> {
    fn validate(
        &self,
        input_shapes: &[&[crate::ops::Dim]],
        input_dtypes: &[crate::PlannedDType],
    ) -> Result<(), crate::ops::validation::ValidationError> {
        let shape = input_shapes[0];
        crate::ops::validation::require_hw_or_hwc(shape)?;
        crate::ops::validation::require_channels_at_least(
            shape,
            M::lit(&self.from_space).channels(),
            "[H, W, C] with the source color space's channels (and optionally alpha)",
        )?;
        // Exactly the space's channels, or those plus alpha (`color_channels`
        // is the one alpha rule): every kernel reads a pixel as that many
        // values, so a fifth channel would be read as the next pixel's.
        // A ranged conversion reads values against a colour range, which a
        // signed integer does not have (`ColorRange`).
        if is_ranged(M::lit(&self.from_space)) || is_ranged(M::lit(&self.to_space)) {
            if let Some(crate::PlannedDType::Known(dtype)) = input_dtypes.first().copied() {
                if !ranged_dtypes().accepts(dtype) {
                    return Err(crate::ops::validation::ValidationError::DTypeRequirement {
                        expected: RANGED_DTYPES.to_vec(),
                        got: dtype,
                    });
                }
            }
        }
        let space = M::lit(&self.from_space).channels();
        match shape.get(2).and_then(|c| c.known()) {
            Some(c) if color_channels(c) != space => {
                Err(crate::ops::validation::ValidationError::ShapeRequirement {
                    requirement: "[H, W, C] with the source color space's channels \
                                  (and optionally alpha)",
                    got: shape.to_vec(),
                })
            }
            _ => Ok(()),
        }
    }

    fn name(&self) -> &'static str {
        "ColorConvert"
    }

    fn shape(&self) -> OpShape {
        ColorConvertOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        // Reads any layout: each conversion packs its input at most once
        // itself, and a ranged one (HSV, Lab, YCbCr) starts by casting to f64
        // (`convert_ranged`), which walks a view without a separate pack.
        MemoryEffect::StridePreserving
    }

    fn identity_rule(&self) -> IdentityRule {
        // Computes / combines / reduces — never a removable no-op.
        IdentityRule::Never
    }

    fn is_spatial_window(&self) -> bool {
        false // Colour conversion is not an H/W crop window.
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        // Color-space conversion maps each pixel's channels independently of
        // any neighbor.
        SpatialDependency::Pointwise
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        None
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        if is_ranged(M::lit(&self.from_space)) || is_ranged(M::lit(&self.to_space)) {
            return ranged_dtypes();
        }
        DTypeCategory::Numeric
    }

    /// `None`: `apply_color_convert` reads the input in its native dtype and
    /// does its own f64 arithmetic per dtype's `ColorRange`, so the execution
    /// layer must not pre-cast (a pre-cast to u8 would corrupt float and wide
    /// integer ranges).
    fn working_dtype(&self) -> Option<DType> {
        None
    }

    /// Truthful dtype contract for `apply_color_convert`: conversions
    /// involving Lab compute in f32 and stay f32 for every input dtype;
    /// every other conversion preserves the element dtype (the math may
    /// run in f32 internally, but the result is cast back to the input
    /// dtype for all inputs, not just u8).
    fn output_dtype_rule(&self) -> OutputDTypeRule {
        if self.promotes_to_float() {
            OutputDTypeRule::Fixed(DType::F32)
        } else {
            OutputDTypeRule::PreserveInput
        }
    }
}

/// How many leading channels of a `C`-channel image are colour: all but the
/// last when `C` is 2 (GrayA) or 4 (RGBA), where the last is alpha; else all.
/// The one rule for "which channels are alpha", read by every op that sets
/// alpha aside.
///
/// **The declaration** that a 2-channel image is gray + alpha and a 4-channel
/// one RGBA: the resamplers' premultiply, the shape rule's carried alpha,
/// grayscale's gray channel and the colour conversions all read it (through
/// this or [`has_alpha`]) rather than testing the channel count themselves.
pub const fn color_channels(channels: usize) -> usize {
    channels - has_alpha(channels) as usize
}

/// Whether a `C`-channel image's last channel is alpha: 2 (gray + alpha) and
/// 4 (RGBA). The rule itself; see [`color_channels`].
pub const fn has_alpha(channels: usize) -> bool {
    matches!(channels, 2 | 4)
}

/// Apply color conversion to a ViewBuffer.
///
/// The buffer must be contiguous `[H, W, C]` or `[H, W]` for grayscale input.
/// Alpha channels (C=2 for GrayA, C=4 for RGBA) are handled via
/// strip-process-restore: the alpha is separated, the conversion is applied
/// to the color channels, and then the alpha is re-attached.
pub fn apply_color_convert(buf: &ViewBuffer, op: &ColorConvertOp) -> ViewBuffer {
    if op.from_space == op.to_space {
        return buf.clone();
    }

    let shape = buf.shape();
    let channels = if shape.len() == 3 { shape[2] } else { 1 };
    if has_alpha(channels) {
        let (color_buf, alpha_buf) = split_alpha(buf);
        let converted = apply_color_convert_core(&color_buf, op);
        // Alpha is a value in the image's range too: a conversion that
        // changes dtype (into or out of Lab) rescales it to the output's
        // (u8 255 is f32 1.0).
        let (from, to) = (buf.dtype(), converted.dtype());
        let alpha_buf = if from != to {
            let k = to.value_range_max() / from.value_range_max();
            let alpha = alpha_buf.cast(DType::F64).to_contiguous();
            let scaled: Vec<f64> = alpha.as_slice::<f64>().iter().map(|a| a * k).collect();
            ViewBuffer::from_vec_with_shape(scaled, alpha_buf.shape().to_vec()).cast(to)
        } else {
            alpha_buf
        };
        merge_alpha(&converted, &alpha_buf)
    } else {
        apply_color_convert_core(buf, op)
    }
}

/// The dtypes with a colour range (`ColorRange::of`, in `kernels/color.rs`).
const RANGED_DTYPES: [DType; 6] = [
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
    DType::F32,
    DType::F64,
];

fn ranged_dtypes() -> DTypeCategory {
    DTypeCategory::Specific(RANGED_DTYPES.to_vec())
}

/// Whether a conversion between these spaces reads or writes a colour range
/// (HSV, Lab, YCbCr); RGB, BGR and gray only move or mix channel values.
fn is_ranged(space: ColorSpace) -> bool {
    matches!(space, ColorSpace::Hsv | ColorSpace::Lab | ColorSpace::YCbCr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb_u8(r: u8, g: u8, b: u8) -> ViewBuffer {
        ViewBuffer::from_vec_with_shape(vec![r, g, b], vec![1, 1, 3])
    }

    #[test]
    fn test_rgb_to_bgr_roundtrip() {
        let rgb = make_rgb_u8(100, 150, 200);
        let op_fwd = ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Bgr,
        };
        let bgr = apply_color_convert(&rgb, &op_fwd);
        assert_eq!(bgr.as_slice::<u8>(), &[200, 150, 100]);

        let op_bwd = ColorConvertOp {
            from_space: ColorSpace::Bgr,
            to_space: ColorSpace::Rgb,
        };
        let back = apply_color_convert(&bgr, &op_bwd);
        assert_eq!(back.as_slice::<u8>(), &[100, 150, 200]);
    }

    #[test]
    fn test_rgb_to_gray() {
        let rgb = make_rgb_u8(255, 0, 0);
        let op = ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Gray,
        };
        let gray = apply_color_convert(&rgb, &op);
        assert_eq!(gray.shape(), &[1, 1, 1]);
        // BT.601: 0.299*255 = 76.2 → 76 (fixed-point)
        let val = gray.as_slice::<u8>()[0];
        assert!((val as i32 - 76).abs() <= 1);
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let rgb = make_rgb_u8(100, 150, 200);
        let to_hsv = ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Hsv,
        };
        let hsv = apply_color_convert(&rgb, &to_hsv);

        let to_rgb = ColorConvertOp {
            from_space: ColorSpace::Hsv,
            to_space: ColorSpace::Rgb,
        };
        let back = apply_color_convert(&hsv, &to_rgb);
        let back_vals = back.as_slice::<u8>();
        // Allow ±2 for rounding through f32 intermediates
        assert!((back_vals[0] as i32 - 100).abs() <= 2);
        assert!((back_vals[1] as i32 - 150).abs() <= 2);
        assert!((back_vals[2] as i32 - 200).abs() <= 2);
    }

    #[test]
    fn test_rgb_ycbcr_roundtrip() {
        let rgb = make_rgb_u8(100, 150, 200);
        let to_ycbcr = ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::YCbCr,
        };
        let ycbcr = apply_color_convert(&rgb, &to_ycbcr);

        let to_rgb = ColorConvertOp {
            from_space: ColorSpace::YCbCr,
            to_space: ColorSpace::Rgb,
        };
        let back = apply_color_convert(&ycbcr, &to_rgb);
        let back_vals = back.as_slice::<u8>();
        assert!((back_vals[0] as i32 - 100).abs() <= 2);
        assert!((back_vals[1] as i32 - 150).abs() <= 2);
        assert!((back_vals[2] as i32 - 200).abs() <= 2);
    }

    #[test]
    fn test_rgb_lab_roundtrip() {
        let rgb = make_rgb_u8(100, 150, 200);
        let to_lab = ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Lab,
        };
        let lab = apply_color_convert(&rgb, &to_lab);
        // LAB always outputs f32
        assert_eq!(lab.dtype(), DType::F32);

        let to_rgb = ColorConvertOp {
            from_space: ColorSpace::Lab,
            to_space: ColorSpace::Rgb,
        };
        let back = apply_color_convert(&lab, &to_rgb);
        // LAB->RGB also stays f32
        assert_eq!(back.dtype(), DType::F32);
        // A float image's values are in [0, 1].
        let back_vals = back.as_slice::<f32>();
        assert!((back_vals[0] - 100.0 / 255.0).abs() <= 2.0 / 255.0);
        assert!((back_vals[1] - 150.0 / 255.0).abs() <= 2.0 / 255.0);
        assert!((back_vals[2] - 200.0 / 255.0).abs() <= 2.0 / 255.0);
    }

    #[test]
    fn test_noop_conversion() {
        let rgb = make_rgb_u8(100, 150, 200);
        let op = ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Rgb,
        };
        let result = apply_color_convert(&rgb, &op);
        assert_eq!(result.as_slice::<u8>(), rgb.as_slice::<u8>());
    }

    /// Non-Lab conversions must honor the PreserveInput contract for EVERY
    /// input dtype, not just u8 — execution matches `output_dtype_rule`.
    #[test]
    fn test_non_lab_conversions_preserve_dtype() {
        let cases = [
            (ColorSpace::Rgb, ColorSpace::Hsv),
            (ColorSpace::Rgb, ColorSpace::YCbCr),
            (ColorSpace::Rgb, ColorSpace::Gray),
            (ColorSpace::Gray, ColorSpace::Rgb),
        ];
        for (from, to) in cases {
            let op = ColorConvertOp {
                from_space: from,
                to_space: to,
            };
            let channels = if from == ColorSpace::Gray { 1 } else { 3 };
            // (Signed integers have no colour range and are refused by the
            // contract for HSV/YCbCr; `signed_integers_are_refused`.)
            for dtype in [DType::U16, DType::U32, DType::F64] {
                let src =
                    ViewBuffer::from_vec_with_shape(vec![100u8; channels], vec![1, 1, channels])
                        .cast(dtype);
                let out = apply_color_convert(&src, &op);
                assert_eq!(
                    out.dtype(),
                    dtype,
                    "{from:?}->{to:?} must preserve {dtype:?}, got {:?}",
                    out.dtype()
                );
                assert_eq!(
                    out.dtype(),
                    op.resolve_output_dtype(src.dtype()),
                    "{from:?}->{to:?} execution dtype diverges from the contract"
                );
            }
        }
    }

    /// The preserve contract holds through the alpha split/process/restore
    /// path too (its helpers compute in u8/u16/f32 internally).
    #[test]
    fn test_alpha_path_preserves_dtype() {
        let op = ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Hsv,
        };
        for dtype in [DType::U16, DType::F64] {
            let src = ViewBuffer::from_vec_with_shape(vec![100u8, 150, 200, 255], vec![1, 1, 4])
                .cast(dtype);
            let out = apply_color_convert(&src, &op);
            assert_eq!(out.shape(), &[1, 1, 4]);
            assert_eq!(
                out.dtype(),
                dtype,
                "alpha path must preserve {dtype:?}, got {:?}",
                out.dtype()
            );
        }
    }

    fn convert(src: ViewBuffer, from: ColorSpace, to: ColorSpace) -> ViewBuffer {
        apply_color_convert(
            &src,
            &ColorConvertOp {
                from_space: from,
                to_space: to,
            },
        )
    }

    /// 8-bit hue is half-degrees in [0, 180): 359.8° rounds to 180, which is
    /// 0°. It used to be stored as 180, outside the range.
    #[test]
    fn u8_hue_wraps_at_180() {
        let out = convert(make_rgb_u8(255, 0, 1), ColorSpace::Rgb, ColorSpace::Hsv);
        assert_eq!(out.as_slice::<u8>(), &[0, 255, 255]);
    }

    /// A gray pixel has no chroma: Cb = Cr = the middle of the dtype's range
    /// (32768 for u16; it was 128, the u8 value, on every dtype).
    #[test]
    fn chroma_is_centred_in_each_dtypes_range() {
        let gray = ViewBuffer::from_vec_with_shape(vec![32768u16; 3], vec![1, 1, 3]);
        let out = convert(gray, ColorSpace::Rgb, ColorSpace::YCbCr);
        assert_eq!(out.as_slice::<u16>(), &[32768, 32768, 32768]);
        let gray = ViewBuffer::from_vec_with_shape(vec![0.5f32; 3], vec![1, 1, 3]);
        let out = convert(gray, ColorSpace::Rgb, ColorSpace::YCbCr);
        assert_eq!(out.as_slice::<f32>(), &[0.5, 0.5, 0.5]);
    }

    /// Floats are in [0, 1] and their hue in degrees, as OpenCV's float HSV.
    #[test]
    fn float_hsv_is_degrees_and_unit_range() {
        let blue = ViewBuffer::from_vec_with_shape(vec![0.0f32, 0.0, 1.0], vec![1, 1, 3]);
        let out = convert(blue, ColorSpace::Rgb, ColorSpace::Hsv);
        assert_eq!(out.as_slice::<f32>(), &[240.0, 1.0, 1.0]);
        let half = ViewBuffer::from_vec_with_shape(vec![0.5f32, 0.25, 0.25], vec![1, 1, 3]);
        let out = convert(half, ColorSpace::Rgb, ColorSpace::Hsv);
        assert_eq!(out.as_slice::<f32>(), &[0.0, 0.5, 0.5]);
    }

    /// A wider unsigned integer uses its whole range: V and S at full scale
    /// are MAX, and the hue covers 0..MAX for one turn.
    #[test]
    fn u16_hsv_uses_the_full_range_and_round_trips() {
        let rgb = ViewBuffer::from_vec_with_shape(vec![65535u16, 0, 0, 0, 0, 65535], vec![1, 2, 3]);
        let hsv = convert(rgb.clone(), ColorSpace::Rgb, ColorSpace::Hsv);
        // red: 0°, blue: 240° = 65536 * 2/3 = 43690.67 -> 43691.
        assert_eq!(
            hsv.as_slice::<u16>(),
            &[0, 65535, 65535, 43691, 65535, 65535]
        );
        // Back to RGB within the hue's quantum (one 65536th of a turn leaks
        // a few units into a neighbouring channel).
        let back = convert(hsv, ColorSpace::Hsv, ColorSpace::Rgb);
        for (a, b) in back.as_slice::<u16>().iter().zip(rgb.as_slice::<u16>()) {
            assert!(a.abs_diff(*b) <= 4, "{a} vs {b}");
        }
    }

    /// Lab output is f32, so a carried alpha is rescaled to the float range.
    #[test]
    fn alpha_follows_the_output_range() {
        let rgba = ViewBuffer::from_vec_with_shape(vec![100u8, 150, 200, 255], vec![1, 1, 4]);
        let out = convert(rgba, ColorSpace::Rgb, ColorSpace::Lab);
        assert_eq!(out.as_slice::<f32>()[3], 1.0);
    }

    /// HSV, Lab and YCbCr read values against a colour range; a signed
    /// integer has none, so the contract refuses it (RGB, BGR and gray do
    /// not need one).
    #[test]
    fn signed_integers_are_refused() {
        use crate::ops::traits::Op;
        let known = |d| [crate::PlannedDType::Known(d)];
        let shape = [
            crate::ops::Dim::Known(2),
            crate::ops::Dim::Known(2),
            crate::ops::Dim::Known(3),
        ];
        for (from, to) in [
            (ColorSpace::Rgb, ColorSpace::Hsv),
            (ColorSpace::YCbCr, ColorSpace::Rgb),
            (ColorSpace::Rgb, ColorSpace::Lab),
        ] {
            let op: ColorConvertOp = ColorConvertOp {
                from_space: from,
                to_space: to,
            };
            assert!(
                op.validate(&[&shape], &known(DType::I16)).is_err(),
                "{from:?}->{to:?}"
            );
            assert!(
                op.validate(&[&shape], &known(DType::U16)).is_ok(),
                "{from:?}->{to:?}"
            );
        }
        let to_bgr: ColorConvertOp = ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Bgr,
        };
        assert!(to_bgr.validate(&[&shape], &known(DType::I16)).is_ok());
    }

    /// Lab-involving conversions are f32 for every input dtype, including f64.
    #[test]
    fn test_lab_is_f32_for_all_input_dtypes() {
        let to_lab = ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Lab,
        };
        for dtype in [DType::U8, DType::U16, DType::F64] {
            let src =
                ViewBuffer::from_vec_with_shape(vec![100u8, 150, 200], vec![1, 1, 3]).cast(dtype);
            let out = apply_color_convert(&src, &to_lab);
            assert_eq!(out.dtype(), DType::F32, "{dtype:?}->lab must yield f32");
        }

        // With an alpha channel: LabA output stays f32 (alpha re-attached).
        let rgba = ViewBuffer::from_vec_with_shape(vec![100u8, 150, 200, 255], vec![1, 1, 4]);
        let out = apply_color_convert(&rgba, &to_lab);
        assert_eq!(out.shape(), &[1, 1, 4]);
        assert_eq!(out.dtype(), DType::F32, "u8 rgba->lab must yield f32 LabA");
    }
}
