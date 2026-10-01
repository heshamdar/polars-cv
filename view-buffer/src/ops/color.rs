//! Color space conversion operations.
//!
//! Supports conversions between RGB, BGR, HSV, LAB, YCbCr, and Grayscale color spaces.
//! Follows OpenCV's value ranges per dtype (`ColorRange`): u8 0..255 with
//! H in [0, 180); u16/u32/u64 0..MAX with a full-range hue; floats [0, 1]
//! with H in degrees. Signed integers have no colour range.

use crate::core::buffer::ViewBuffer;
use crate::core::convert::CastFrom;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::shape_rule::OpShape;
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};

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
        MemoryEffect::RequiresContiguous
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
pub fn color_channels(channels: usize) -> usize {
    if matches!(channels, 2 | 4) {
        channels - 1
    } else {
        channels
    }
}

/// Split alpha channel from a buffer.
///
/// `[H, W, 4]` -> `([H, W, 3], [H, W, 1])` (color, alpha)
/// `[H, W, 2]` -> `([H, W, 1], [H, W, 1])` (gray, alpha)
pub fn split_alpha(buf: &ViewBuffer) -> (ViewBuffer, ViewBuffer) {
    let contig = buf.to_contiguous();
    let shape = contig.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    let color_c = color_channels(c);

    crate::core::dtype::with_dtype!(buf.dtype(), T => {
        split_alpha_typed::<T>(&contig, h, w, c, color_c)
    })
}

fn split_alpha_typed<T: crate::core::dtype::ViewType>(
    buf: &ViewBuffer,
    h: usize,
    w: usize,
    total_c: usize,
    color_c: usize,
) -> (ViewBuffer, ViewBuffer) {
    let src = buf.as_slice::<T>();
    let mut color_data: Vec<T> = Vec::with_capacity(h * w * color_c);
    let mut alpha_data: Vec<T> = Vec::with_capacity(h * w);

    for pixel in src.chunks_exact(total_c) {
        color_data.extend_from_slice(&pixel[..color_c]);
        alpha_data.push(pixel[color_c]);
    }

    let color = ViewBuffer::from_vec(color_data).reshape(vec![h, w, color_c]);
    let alpha = ViewBuffer::from_vec(alpha_data).reshape(vec![h, w, 1]);
    (color, alpha)
}

/// Merge an alpha channel back onto a color buffer.
///
/// `([H, W, C], [H, W, 1])` -> `[H, W, C+1]`
pub fn merge_alpha(color: &ViewBuffer, alpha: &ViewBuffer) -> ViewBuffer {
    let contig_color = color.to_contiguous();
    // The conversion may have changed the color dtype while alpha kept the
    // source's (u8 RGBA -> f32 Lab): align alpha before the typed merge,
    // which reads both slices as the color's element type.
    let mut contig_alpha = alpha.to_contiguous();
    if contig_alpha.dtype() != contig_color.dtype() {
        contig_alpha = contig_alpha.cast(contig_color.dtype());
    }
    let shape = contig_color.shape();
    let (h, w) = (shape[0], shape[1]);
    let color_c = if shape.len() == 3 { shape[2] } else { 1 };

    crate::core::dtype::with_dtype!(color.dtype(), T => {
        merge_alpha_typed::<T>(&contig_color, &contig_alpha, h, w, color_c)
    })
}

fn merge_alpha_typed<T: crate::core::dtype::ViewType>(
    color: &ViewBuffer,
    alpha: &ViewBuffer,
    h: usize,
    w: usize,
    color_c: usize,
) -> ViewBuffer {
    let color_src = color.as_slice::<T>();
    let alpha_src = alpha.as_slice::<T>();
    let out_c = color_c + 1;
    let mut out: Vec<T> = Vec::with_capacity(h * w * out_c);

    for (color_pixel, &a) in color_src.chunks_exact(color_c).zip(alpha_src.iter()) {
        out.extend_from_slice(color_pixel);
        out.push(a);
    }

    ViewBuffer::from_vec(out).reshape(vec![h, w, out_c])
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
    let has_alpha = color_channels(channels) < channels;

    if has_alpha {
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

/// Core color conversion logic operating on color channels only (no alpha).
fn apply_color_convert_core(buf: &ViewBuffer, op: &ColorConvertOp) -> ViewBuffer {
    if op.from_space == op.to_space {
        return buf.clone();
    }

    // BGR is just a channel reorder — delegate to swap
    if op.from_space == ColorSpace::Rgb && op.to_space == ColorSpace::Bgr {
        return channel_reorder(buf, &[2, 1, 0]);
    }
    if op.from_space == ColorSpace::Bgr && op.to_space == ColorSpace::Rgb {
        return channel_reorder(buf, &[2, 1, 0]);
    }

    // Grayscale from RGB
    if op.from_space == ColorSpace::Rgb && op.to_space == ColorSpace::Gray {
        return rgb_to_gray(buf);
    }
    if op.from_space == ColorSpace::Bgr && op.to_space == ColorSpace::Gray {
        let rgb = channel_reorder(buf, &[2, 1, 0]);
        return rgb_to_gray(&rgb);
    }

    // Gray to RGB/BGR: replicate single channel
    if op.from_space == ColorSpace::Gray
        && (op.to_space == ColorSpace::Rgb || op.to_space == ColorSpace::Bgr)
    {
        return gray_to_rgb(buf);
    }

    // Every other pair: decode to RGB, encode the target, in each dtype's
    // colour range (`ColorRange`).
    convert_ranged(buf, op, op.resolve_output_dtype(buf.dtype()))
}

// =============================================================================
// Colour ranges: how each dtype holds a colour value (OpenCV's conventions)
// =============================================================================

/// How a hue is stored.
#[derive(Debug, Clone, Copy, PartialEq)]
enum HueScale {
    /// u8: half-degrees, `[0, 180)` (OpenCV's `COLOR_RGB2HSV`).
    HalfDegrees,
    /// Floats: degrees, `[0, 360)`.
    Degrees,
    /// The wider unsigned integers: the whole range is one turn, `[0, MAX]`,
    /// the period `MAX + 1`. OpenCV has no HSV above 8 bits; this extends its
    /// 8-bit `HSV_FULL` scheme and is polars-cv's own convention.
    Full(f64),
}

/// How a dtype holds colour values — the one table the ranged conversions
/// (HSV, Lab, YCbCr) read, after OpenCV: an unsigned integer spans
/// `0..=MAX` ([`DType::value_range_max`]) with chroma centred at
/// `(MAX + 1) / 2`; a float spans `[0, 1]` with chroma centred at `0.5`.
/// Signed integers have no colour range, and are refused
/// ([`ColorConvertOp::validate`]).
#[derive(Debug, Clone, Copy)]
struct ColorRange {
    /// Full scale: the value of a saturated channel.
    full: f64,
    /// The centre of a chroma channel (YCbCr's Cb and Cr).
    chroma_offset: f64,
    hue: HueScale,
}

impl ColorRange {
    fn of(dtype: DType) -> Self {
        let full = dtype.value_range_max();
        match dtype {
            DType::U8 => ColorRange {
                full,
                chroma_offset: 128.0,
                hue: HueScale::HalfDegrees,
            },
            DType::U16 | DType::U32 | DType::U64 => ColorRange {
                full,
                chroma_offset: (full + 1.0) / 2.0,
                hue: HueScale::Full(full + 1.0),
            },
            DType::F32 | DType::F64 => ColorRange {
                full,
                chroma_offset: 0.5,
                hue: HueScale::Degrees,
            },
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => unreachable!(
                "signed integers have no colour range; ColorConvertOp::validate refuses them"
            ),
        }
    }

    fn encode_hue(&self, degrees: f64) -> f64 {
        match self.hue {
            HueScale::HalfDegrees => degrees / 2.0,
            HueScale::Degrees => degrees,
            HueScale::Full(period) => degrees * period / 360.0,
        }
    }

    fn decode_hue(&self, stored: f64) -> f64 {
        match self.hue {
            HueScale::HalfDegrees => stored * 2.0,
            HueScale::Degrees => stored,
            HueScale::Full(period) => stored * 360.0 / period,
        }
    }

    /// The stored hue's period, for an integer dtype: a hue that rounds up
    /// to it is the same angle as 0 (OpenCV wraps 359.8° to 0, not 180).
    fn hue_period(&self) -> Option<f64> {
        match self.hue {
            HueScale::HalfDegrees => Some(180.0),
            HueScale::Degrees => None,
            HueScale::Full(period) => Some(period),
        }
    }
}

/// The dtypes with a colour range ([`ColorRange::of`]).
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

/// A pixel's RGB, in units of `scale` (a channel at full scale is `scale`).
fn decode_rgb(from: ColorSpace, px: &[f64], range: &ColorRange, scale: f64) -> [f64; 3] {
    let v = |i: usize| px[i];
    match from {
        ColorSpace::Rgb => [v(0), v(1), v(2)],
        ColorSpace::Bgr => [v(2), v(1), v(0)],
        ColorSpace::Gray => [v(0), v(0), v(0)],
        ColorSpace::Hsv => {
            let hue = range.decode_hue(v(0));
            let s = v(1) / range.full;
            let val = v(2) * scale / range.full;
            hsv_to_rgb(hue, s, val)
        }
        ColorSpace::YCbCr => {
            let k = scale / range.full;
            let y = v(0) * k;
            let cb = (v(1) - range.chroma_offset) * k;
            let cr = (v(2) - range.chroma_offset) * k;
            [
                (y + 1.402 * cr).clamp(0.0, scale),
                (y - 0.344136 * cb - 0.714136 * cr).clamp(0.0, scale),
                (y + 1.772 * cb).clamp(0.0, scale),
            ]
        }
        ColorSpace::Lab => {
            let [r, g, b] = lab_to_rgb(v(0), v(1), v(2));
            [r * scale, g * scale, b * scale]
        }
    }
}

/// A pixel's target-space values from its RGB in units of `scale`, stored
/// in `range`.
fn encode(to: ColorSpace, [r, g, b]: [f64; 3], scale: f64, range: &ColorRange) -> [f64; 3] {
    let k = range.full / scale;
    match to {
        ColorSpace::Rgb => [r * k, g * k, b * k],
        ColorSpace::Bgr => [b * k, g * k, r * k],
        ColorSpace::Gray => {
            let y = luma_f64(r, g, b) * k;
            [y, y, y]
        }
        ColorSpace::Hsv => {
            use crate::ops::util::{maximum, minimum};
            let max = maximum(maximum(r, g), b);
            let min = minimum(minimum(r, g), b);
            let diff = max - min;
            let s = if max == 0.0 { 0.0 } else { diff / max };
            let hue = if diff == 0.0 {
                0.0
            } else if max == r {
                let h = 60.0 * (g - b) / diff;
                if h < 0.0 {
                    h + 360.0
                } else {
                    h
                }
            } else if max == g {
                60.0 * (b - r) / diff + 120.0
            } else {
                60.0 * (r - g) / diff + 240.0
            };
            [range.encode_hue(hue), s * range.full, max * k]
        }
        ColorSpace::YCbCr => [
            luma_f64(r, g, b) * k,
            range.chroma_offset + (-0.168736 * r - 0.331264 * g + 0.5 * b) * k,
            range.chroma_offset + (0.5 * r - 0.418688 * g - 0.081312 * b) * k,
        ],
        ColorSpace::Lab => rgb_to_lab(r / scale, g / scale, b / scale),
    }
}

/// HSV (hue in degrees, saturation in [0, 1]) to RGB in the value's units.
fn hsv_to_rgb(hue: f64, s: f64, v: f64) -> [f64; 3] {
    if s == 0.0 {
        return [v, v, v];
    }
    let sector = hue / 60.0;
    let sector_int = sector.floor() as i64;
    let f = sector - sector_int as f64;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match sector_int.rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Convert colour channels (no alpha) between spaces at least one of which
/// is ranged: decode each pixel to RGB in the input's range, encode the
/// target in the output's, in f64 (which holds every dtype's values the
/// math needs; f32 rounded u32/u64 channels), then store in `out_dtype`. An
/// integer hue that rounds up to its period wraps to 0.
fn convert_ranged(buf: &ViewBuffer, op: &ColorConvertOp, out_dtype: DType) -> ViewBuffer {
    let shape = buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let in_c = op.from_space.channels();
    let out_c = op.to_space.channels();
    let in_range = ColorRange::of(buf.dtype());
    let out_range = ColorRange::of(out_dtype);
    // RGB is carried in the input's units, or the output's when the input
    // is Lab (which has none).
    let scale = if op.from_space == ColorSpace::Lab {
        out_range.full
    } else {
        in_range.full
    };
    let contig = buf.cast(DType::F64).to_contiguous();
    let src = contig.as_slice::<f64>();
    let wrap = (op.to_space == ColorSpace::Hsv && DTypeCategory::Integer.accepts(out_dtype))
        .then(|| out_range.hue_period())
        .flatten();
    let mut out: Vec<f64> = Vec::with_capacity(h * w * out_c);
    for px in src.chunks_exact(in_c) {
        let rgb = decode_rgb(op.from_space, px, &in_range, scale);
        let mut v = encode(op.to_space, rgb, scale, &out_range);
        if let Some(period) = wrap {
            let rounded = v[0].round();
            v[0] = if rounded >= period {
                rounded - period
            } else {
                rounded
            };
        }
        out.extend_from_slice(&v[..out_c]);
    }
    ViewBuffer::from_vec_with_shape(out, vec![h, w, out_c]).cast(out_dtype)
}

// =============================================================================
// Luma: the one BT.601 authority
// =============================================================================

/// BT.601 luma, `Y = 0.299R + 0.587G + 0.114B`, of values in any units.
///
/// **The one luma formula**: `grayscale()` (through [`Luma`]),
/// `convert_color` to gray and the YCbCr encoder all read it, so no two can
/// weigh the channels differently.
#[inline(always)]
pub(crate) fn luma_f64(r: f64, g: f64, b: f64) -> f64 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// An element type's luma, stored in its own dtype: u8 in fixed point
/// (`(77R + 150G + 29B + 128) >> 8`), every other dtype as [`luma_f64`]
/// stored by the conversion rule (rounded and saturated for an integer).
/// `grayscale()` and `convert_color(rgb -> gray)` both run it.
pub(crate) trait Luma: crate::core::dtype::ViewType {
    fn luma(r: Self, g: Self, b: Self) -> Self;
}

/// Fixed-point BT.601 luma of one u8 pixel. The sum peaks at
/// `256 * 255 + 128 = 65408`, so it fits in `u16`, which gives the vector
/// loop twice the lanes of `u32`.
impl Luma for u8 {
    #[inline(always)]
    fn luma(r: u8, g: u8, b: u8) -> u8 {
        ((77 * u16::from(r) + 150 * u16::from(g) + 29 * u16::from(b) + 128) >> 8) as u8
    }
}

macro_rules! luma_by_f64 {
    ($($t:ty),+) => {$(
        impl Luma for $t {
            #[inline(always)]
            fn luma(r: $t, g: $t, b: $t) -> $t {
                let f = f64::cast_from;
                <$t>::cast_from(luma_f64(f(r), f(g), f(b)))
            }
        }
    )+};
}
luma_by_f64!(i8, u16, i16, u32, i32, u64, i64, f32, f64);

// =============================================================================
// RGB ↔ Grayscale
// =============================================================================

/// The luma of each pixel of a packed `[H, W, 3]` buffer, in its dtype.
fn rgb_to_gray(buf: &ViewBuffer) -> ViewBuffer {
    let contig = buf.to_contiguous();
    let shape = contig.shape();
    let (h, w) = (shape[0], shape[1]);
    crate::core::dtype::with_dtype!(buf.dtype(), T => {
        let out: Vec<T> = contig
            .as_slice::<T>()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| T::luma(p[0], p[1], p[2]))
            .collect();
        ViewBuffer::from_vec_with_shape(out, vec![h, w, 1])
    })
}

/// Replicate a gray plane into three channels: data movement, in the
/// input's own dtype.
fn gray_to_rgb(buf: &ViewBuffer) -> ViewBuffer {
    let contig = buf.to_contiguous();
    let shape = contig.shape();
    let (h, w) = (shape[0], shape[1]);
    crate::core::dtype::with_dtype!(buf.dtype(), T => {
        let src = contig.as_slice::<T>();
        let mut out: Vec<T> = Vec::with_capacity(h * w * 3);
        for &val in src {
            out.extend([val, val, val]);
        }
        ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
    })
}

// =============================================================================
// Channel reorder helpers
// =============================================================================

fn channel_reorder(buf: &ViewBuffer, order: &[usize]) -> ViewBuffer {
    crate::execution::runner::apply_channel_swap(buf, order)
}

// =============================================================================
// RGB ↔ CIE LAB (D65 illuminant, sRGB transfer)
// =============================================================================

// D65 reference white point
const D65_XN: f64 = 0.950456;
const D65_YN: f64 = 1.0;
const D65_ZN: f64 = 1.088754;

const LAB_DELTA: f64 = 6.0 / 29.0;
const LAB_DELTA_SQ: f64 = LAB_DELTA * LAB_DELTA;
const LAB_DELTA_CU: f64 = LAB_DELTA * LAB_DELTA * LAB_DELTA;

/// sRGB gamma removal (linearize).
#[inline]
fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB gamma application.
#[inline]
fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Lab f(t) function.
#[inline]
fn lab_f(t: f64) -> f64 {
    if t > LAB_DELTA_CU {
        t.cbrt()
    } else {
        t / (3.0 * LAB_DELTA_SQ) + 4.0 / 29.0
    }
}

/// sRGB in [0, 1] to CIE Lab (D65): L in [0, 100], a and b about [-128, 127].
fn rgb_to_lab(r: f64, g: f64, b: f64) -> [f64; 3] {
    let (r, g, b) = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    let fx = lab_f(x / D65_XN);
    let fy = lab_f(y / D65_YN);
    let fz = lab_f(z / D65_ZN);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIE Lab (D65) to sRGB in [0, 1], clamped.
fn lab_to_rgb(l: f64, a: f64, b: f64) -> [f64; 3] {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let x = D65_XN * lab_f_inv(fx);
    let y = D65_YN * lab_f_inv(fy);
    let z = D65_ZN * lab_f_inv(fz);
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    [r, g, b].map(|v| linear_to_srgb(v).clamp(0.0, 1.0))
}

/// Lab f_inv(t) function.
#[inline]
fn lab_f_inv(t: f64) -> f64 {
    if t > LAB_DELTA {
        t * t * t
    } else {
        3.0 * LAB_DELTA_SQ * (t - 4.0 / 29.0)
    }
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
