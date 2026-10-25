//! Colour kernels: conversion between colour spaces and the alpha split and
//! merge around it, in each dtype's colour range; the BT.601 luma authority.

use std::mem::MaybeUninit;

use crate::core::buffer::ViewBuffer;
use crate::core::convert::CastFrom;
use crate::core::dtype::{DType, DTypeCategory};
use crate::core::map::{map_pixels, PixelMap};
use crate::ops::color::{color_channels, ColorConvertOp, ColorSpace};
use crate::ops::traits::Op;

/// Split alpha channel from a buffer.
///
/// `[H, W, 4]` -> `([H, W, 3], [H, W, 1])` (color, alpha)
/// `[H, W, 2]` -> `([H, W, 1], [H, W, 1])` (gray, alpha)
pub(crate) fn split_alpha(buf: &ViewBuffer) -> (ViewBuffer, ViewBuffer) {
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
pub(crate) fn merge_alpha(color: &ViewBuffer, alpha: &ViewBuffer) -> ViewBuffer {
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

/// Core color conversion logic operating on color channels only (no alpha).
pub(crate) fn apply_color_convert_core(buf: &ViewBuffer, op: &ColorConvertOp) -> ViewBuffer {
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
    let wrap = (op.to_space == ColorSpace::Hsv && DTypeCategory::Integer.accepts(out_dtype))
        .then(|| out_range.hue_period())
        .flatten();
    let map = Ranged {
        from: op.from_space,
        to: op.to_space,
        in_range,
        out_range,
        scale,
        wrap,
    };
    // A pixel map (`core::map`), so a view is read where it lies: an f64
    // input's cast is the input itself.
    let src = buf.cast(DType::F64);
    let out = match (in_c, out_c) {
        (1, 1) => map_pixels::<f64, f64, 1, 1, _>(&src, &map),
        (1, 3) => map_pixels::<f64, f64, 1, 3, _>(&src, &map),
        (3, 1) => map_pixels::<f64, f64, 3, 1, _>(&src, &map),
        (3, 3) => map_pixels::<f64, f64, 3, 3, _>(&src, &map),
        (i, o) => unreachable!("a colour space has 1 or 3 channels, not {i} or {o}"),
    };
    debug_assert_eq!(out.shape(), [h, w, out_c]);
    out.cast(out_dtype)
}

/// [`convert_ranged`]'s per-pixel conversion: decode to RGB in the input's
/// range, encode the target in the output's, in f64.
struct Ranged {
    from: ColorSpace,
    to: ColorSpace,
    in_range: ColorRange,
    out_range: ColorRange,
    scale: f64,
    /// The hue period an integer HSV output wraps at.
    wrap: Option<f64>,
}

// SAFETY: `map_into` writes all `O` slots of every output pixel.
unsafe impl<const C: usize, const O: usize> PixelMap<f64, f64, C, O> for Ranged {
    #[inline(always)]
    fn map_into(&self, src: &[[f64; C]], dst: &mut [MaybeUninit<f64>]) {
        for (px, out) in src.iter().zip(dst.as_chunks_mut::<O>().0) {
            let rgb = decode_rgb(self.from, px, &self.in_range, self.scale);
            let mut v = encode(self.to, rgb, self.scale, &self.out_range);
            if let Some(period) = self.wrap {
                let rounded = v[0].round();
                v[0] = if rounded >= period {
                    rounded - period
                } else {
                    rounded
                };
            }
            for (o, &x) in out.iter_mut().zip(&v[..O]) {
                o.write(x);
            }
        }
    }
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
    crate::kernels::channel::apply_channel_swap(buf, order)
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
