//! Color space conversion operations.
//!
//! Supports conversions between RGB, BGR, HSV, LAB, YCbCr, and Grayscale color spaces.
//! Follows OpenCV conventions for U8 ranges (e.g. H=[0,180] for HSV).

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Supported color spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ColorSpace {
    Rgb,
    Bgr,
    Hsv,
    Lab,
    YCbCr,
    Gray,
}

impl ColorSpace {
    /// Parse a color space from a string.
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rgb" => Some(Self::Rgb),
            "bgr" => Some(Self::Bgr),
            "hsv" => Some(Self::Hsv),
            "lab" => Some(Self::Lab),
            "ycbcr" => Some(Self::YCbCr),
            "gray" | "grey" | "grayscale" => Some(Self::Gray),
            _ => None,
        }
    }

    /// Number of channels for this color space.
    pub fn channels(&self) -> usize {
        match self {
            Self::Gray => 1,
            _ => 3,
        }
    }
}

/// Color conversion operation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ColorConvertOp {
    pub from: ColorSpace,
    pub to: ColorSpace,
}

impl ColorConvertOp {
    /// Infer the output shape given an input shape.
    pub fn infer_shape(&self, input: &[usize]) -> Vec<usize> {
        let out_c = self.to.channels();
        match input.len() {
            2 if self.to == ColorSpace::Gray => input.to_vec(),
            2 => vec![input[0], input[1], out_c],
            3 if self.to == ColorSpace::Gray => vec![input[0], input[1], 1],
            3 => vec![input[0], input[1], out_c],
            _ => input.to_vec(),
        }
    }

    /// Whether the conversion promotes dtype to f32.
    ///
    /// LAB conversions require float math and output f32.
    /// All other conversions preserve the input dtype.
    pub fn promotes_to_float(&self) -> bool {
        matches!(self.from, ColorSpace::Lab) || matches!(self.to, ColorSpace::Lab)
    }
}

/// Apply color conversion to a ViewBuffer.
///
/// The buffer must be contiguous [H, W, C] or [H, W] for grayscale input.
pub fn apply_color_convert(buf: &ViewBuffer, op: &ColorConvertOp) -> ViewBuffer {
    if op.from == op.to {
        return buf.clone();
    }

    // BGR is just a channel reorder — delegate to swap
    if op.from == ColorSpace::Rgb && op.to == ColorSpace::Bgr {
        return channel_reorder(buf, &[2, 1, 0]);
    }
    if op.from == ColorSpace::Bgr && op.to == ColorSpace::Rgb {
        return channel_reorder(buf, &[2, 1, 0]);
    }

    // Grayscale from RGB (delegate to existing grayscale if available, but inline BT.601 here)
    if op.from == ColorSpace::Rgb && op.to == ColorSpace::Gray {
        return rgb_to_gray(buf);
    }
    if op.from == ColorSpace::Bgr && op.to == ColorSpace::Gray {
        let rgb = channel_reorder(buf, &[2, 1, 0]);
        return rgb_to_gray(&rgb);
    }

    // Gray to RGB/BGR: replicate single channel
    if op.from == ColorSpace::Gray && (op.to == ColorSpace::Rgb || op.to == ColorSpace::Bgr) {
        return gray_to_rgb(buf);
    }

    // For remaining conversions, route through f32 RGB intermediate
    let rgb_f32 = to_rgb_f32(buf, op.from);
    let result = from_rgb_f32(&rgb_f32, op.to);

    // If the input was u8 and the target doesn't require float, cast back to u8
    if buf.dtype() == DType::U8 && !op.promotes_to_float() {
        result.cast(DType::U8)
    } else {
        result
    }
}

/// Convert any supported color space to f32 RGB.
fn to_rgb_f32(buf: &ViewBuffer, from: ColorSpace) -> ViewBuffer {
    let f32_buf = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = f32_buf.to_contiguous();

    match from {
        ColorSpace::Rgb => contig,
        ColorSpace::Bgr => channel_reorder_f32(&contig, &[2, 1, 0]),
        ColorSpace::Hsv => hsv_to_rgb_f32(&contig),
        ColorSpace::Lab => lab_to_rgb_f32(&contig),
        ColorSpace::YCbCr => ycbcr_to_rgb_f32(&contig),
        ColorSpace::Gray => gray_to_rgb_f32(&contig),
    }
}

/// Convert f32 RGB to any supported color space.
fn from_rgb_f32(rgb: &ViewBuffer, to: ColorSpace) -> ViewBuffer {
    match to {
        ColorSpace::Rgb => rgb.clone(),
        ColorSpace::Bgr => channel_reorder_f32(rgb, &[2, 1, 0]),
        ColorSpace::Hsv => rgb_to_hsv_f32(rgb),
        ColorSpace::Lab => rgb_to_lab_f32(rgb),
        ColorSpace::YCbCr => rgb_to_ycbcr_f32(rgb),
        ColorSpace::Gray => {
            // BT.601 in float: Y = 0.299*R + 0.587*G + 0.114*B
            let shape = rgb.shape();
            let (h, w) = (shape[0], shape[1]);
            let src = rgb.as_slice::<f32>();
            let mut out = Vec::with_capacity(h * w);
            for pix in src.chunks_exact(3) {
                out.push(0.299 * pix[0] + 0.587 * pix[1] + 0.114 * pix[2]);
            }
            ViewBuffer::from_vec_with_shape(out, vec![h, w, 1])
        }
    }
}

// =============================================================================
// RGB ↔ Grayscale
// =============================================================================

fn rgb_to_gray(buf: &ViewBuffer) -> ViewBuffer {
    let contig = buf.to_contiguous();
    let shape = contig.shape();
    let (h, w) = (shape[0], shape[1]);

    match buf.dtype() {
        DType::U8 => {
            let src = contig.as_slice::<u8>();
            let mut out = Vec::with_capacity(h * w);
            for pix in src.chunks_exact(3) {
                let r = pix[0] as u32;
                let g = pix[1] as u32;
                let b = pix[2] as u32;
                out.push(((77 * r + 150 * g + 29 * b + 128) >> 8).min(255) as u8);
            }
            ViewBuffer::from_vec_with_shape(out, vec![h, w, 1])
        }
        _ => {
            let f32_buf = contig.cast(DType::F32);
            let src = f32_buf.as_slice::<f32>();
            let mut out = Vec::with_capacity(h * w);
            for pix in src.chunks_exact(3) {
                out.push(0.299 * pix[0] + 0.587 * pix[1] + 0.114 * pix[2]);
            }
            ViewBuffer::from_vec_with_shape(out, vec![h, w, 1])
        }
    }
}

fn gray_to_rgb(buf: &ViewBuffer) -> ViewBuffer {
    let contig = buf.to_contiguous();
    let shape = contig.shape();
    let (h, w) = (shape[0], shape[1]);

    match buf.dtype() {
        DType::U8 => {
            let src = contig.as_slice::<u8>();
            let mut out = Vec::with_capacity(h * w * 3);
            for &val in src {
                out.push(val);
                out.push(val);
                out.push(val);
            }
            ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
        }
        _ => {
            let f32_buf = contig.cast(DType::F32);
            let src = f32_buf.as_slice::<f32>();
            let mut out = Vec::with_capacity(h * w * 3);
            for &val in src {
                out.push(val);
                out.push(val);
                out.push(val);
            }
            ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
        }
    }
}

fn gray_to_rgb_f32(buf: &ViewBuffer) -> ViewBuffer {
    let shape = buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let src = buf.as_slice::<f32>();
    let mut out = Vec::with_capacity(h * w * 3);
    for &val in src {
        out.push(val);
        out.push(val);
        out.push(val);
    }
    ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
}

// =============================================================================
// Channel reorder helpers
// =============================================================================

fn channel_reorder(buf: &ViewBuffer, order: &[usize]) -> ViewBuffer {
    let contig = buf.to_contiguous();
    let shape = contig.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);

    match buf.dtype() {
        DType::U8 => {
            let src = contig.as_slice::<u8>();
            let mut out = vec![0u8; h * w * c];
            for i in 0..(h * w) {
                for (dst_c, &src_c) in order.iter().enumerate() {
                    out[i * c + dst_c] = src[i * c + src_c];
                }
            }
            ViewBuffer::from_vec_with_shape(out, vec![h, w, c])
        }
        _ => {
            let f32_buf = contig.cast(DType::F32);
            let reordered = channel_reorder_f32(&f32_buf, order);
            if buf.dtype() != DType::F32 {
                reordered.cast(buf.dtype())
            } else {
                reordered
            }
        }
    }
}

fn channel_reorder_f32(buf: &ViewBuffer, order: &[usize]) -> ViewBuffer {
    let shape = buf.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    let src = buf.as_slice::<f32>();
    let mut out = vec![0.0f32; h * w * c];
    for i in 0..(h * w) {
        for (dst_c, &src_c) in order.iter().enumerate() {
            out[i * c + dst_c] = src[i * c + src_c];
        }
    }
    ViewBuffer::from_vec_with_shape(out, vec![h, w, c])
}

// =============================================================================
// RGB ↔ HSV (OpenCV convention: H=[0,180] for U8, H=[0,360] for float)
// =============================================================================

/// Convert f32 RGB [0..255] or [0..1] to f32 HSV.
///
/// For U8 compatibility, input is assumed [0,255] range.
/// Output: H=[0,180], S=[0,255], V=[0,255] (OpenCV U8 convention).
fn rgb_to_hsv_f32(rgb: &ViewBuffer) -> ViewBuffer {
    let shape = rgb.shape();
    let (h, w) = (shape[0], shape[1]);
    let src = rgb.as_slice::<f32>();
    let mut out = vec![0.0f32; h * w * 3];

    for i in 0..(h * w) {
        let r = src[i * 3];
        let g = src[i * 3 + 1];
        let b = src[i * 3 + 2];

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let diff = max - min;

        // Value
        let v = max;

        // Saturation
        let s = if max == 0.0 { 0.0 } else { diff / max * 255.0 };

        // Hue (mapped to [0, 180] like OpenCV)
        let hue = if diff == 0.0 {
            0.0
        } else if max == r {
            let mut h = 60.0 * (g - b) / diff;
            if h < 0.0 {
                h += 360.0;
            }
            h
        } else if max == g {
            60.0 * (b - r) / diff + 120.0
        } else {
            60.0 * (r - g) / diff + 240.0
        };

        out[i * 3] = hue / 2.0; // [0, 360] -> [0, 180]
        out[i * 3 + 1] = s;
        out[i * 3 + 2] = v;
    }

    ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
}

/// Convert f32 HSV (H=[0,180], S=[0,255], V=[0,255]) back to f32 RGB.
fn hsv_to_rgb_f32(hsv: &ViewBuffer) -> ViewBuffer {
    let shape = hsv.shape();
    let (h, w) = (shape[0], shape[1]);
    let src = hsv.as_slice::<f32>();
    let mut out = vec![0.0f32; h * w * 3];

    for i in 0..(h * w) {
        let hue = src[i * 3] * 2.0; // [0, 180] -> [0, 360]
        let s = src[i * 3 + 1] / 255.0; // [0, 255] -> [0, 1]
        let v = src[i * 3 + 2]; // Keep as-is (pixel range)

        if s == 0.0 {
            out[i * 3] = v;
            out[i * 3 + 1] = v;
            out[i * 3 + 2] = v;
            continue;
        }

        let sector = hue / 60.0;
        let sector_int = sector.floor() as i32;
        let f = sector - sector_int as f32;

        let p = v * (1.0 - s);
        let q = v * (1.0 - s * f);
        let t = v * (1.0 - s * (1.0 - f));

        let (r, g, b) = match sector_int % 6 {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };

        out[i * 3] = r;
        out[i * 3 + 1] = g;
        out[i * 3 + 2] = b;
    }

    ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
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

/// Lab f_inv(t) function.
#[inline]
fn lab_f_inv(t: f64) -> f64 {
    if t > LAB_DELTA {
        t * t * t
    } else {
        3.0 * LAB_DELTA_SQ * (t - 4.0 / 29.0)
    }
}

/// Convert f32 RGB [0,255] to f32 LAB.
///
/// Output: L=[0,100], a~[-128,127], b~[-128,127].
fn rgb_to_lab_f32(rgb: &ViewBuffer) -> ViewBuffer {
    let shape = rgb.shape();
    let (h, w) = (shape[0], shape[1]);
    let src = rgb.as_slice::<f32>();
    let mut out = vec![0.0f32; h * w * 3];

    for i in 0..(h * w) {
        let r = srgb_to_linear(src[i * 3] as f64 / 255.0);
        let g = srgb_to_linear(src[i * 3 + 1] as f64 / 255.0);
        let b = srgb_to_linear(src[i * 3 + 2] as f64 / 255.0);

        // Linear RGB to XYZ (sRGB D65 matrix)
        let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
        let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
        let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

        // XYZ to Lab
        let fx = lab_f(x / D65_XN);
        let fy = lab_f(y / D65_YN);
        let fz = lab_f(z / D65_ZN);

        let l = 116.0 * fy - 16.0;
        let a = 500.0 * (fx - fy);
        let b_val = 200.0 * (fy - fz);

        out[i * 3] = l as f32;
        out[i * 3 + 1] = a as f32;
        out[i * 3 + 2] = b_val as f32;
    }

    ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
}

/// Convert f32 LAB to f32 RGB [0,255].
fn lab_to_rgb_f32(lab: &ViewBuffer) -> ViewBuffer {
    let shape = lab.shape();
    let (h, w) = (shape[0], shape[1]);
    let src = lab.as_slice::<f32>();
    let mut out = vec![0.0f32; h * w * 3];

    for i in 0..(h * w) {
        let l = src[i * 3] as f64;
        let a = src[i * 3 + 1] as f64;
        let b_val = src[i * 3 + 2] as f64;

        // Lab to XYZ
        let fy = (l + 16.0) / 116.0;
        let fx = a / 500.0 + fy;
        let fz = fy - b_val / 200.0;

        let x = D65_XN * lab_f_inv(fx);
        let y = D65_YN * lab_f_inv(fy);
        let z = D65_ZN * lab_f_inv(fz);

        // XYZ to linear RGB (inverse sRGB D65 matrix)
        let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
        let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
        let b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

        // Linear RGB to sRGB, then scale to [0,255] and clamp
        out[i * 3] = (linear_to_srgb(r) * 255.0).clamp(0.0, 255.0) as f32;
        out[i * 3 + 1] = (linear_to_srgb(g) * 255.0).clamp(0.0, 255.0) as f32;
        out[i * 3 + 2] = (linear_to_srgb(b) * 255.0).clamp(0.0, 255.0) as f32;
    }

    ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
}

// =============================================================================
// RGB ↔ YCbCr (ITU-R BT.601)
// =============================================================================

/// Convert f32 RGB [0,255] to f32 YCbCr.
///
/// Y = 0.299*R + 0.587*G + 0.114*B
/// Cb = 128 - 0.169*R - 0.331*G + 0.500*B
/// Cr = 128 + 0.500*R - 0.419*G - 0.081*B
fn rgb_to_ycbcr_f32(rgb: &ViewBuffer) -> ViewBuffer {
    let shape = rgb.shape();
    let (h, w) = (shape[0], shape[1]);
    let src = rgb.as_slice::<f32>();
    let mut out = vec![0.0f32; h * w * 3];

    for i in 0..(h * w) {
        let r = src[i * 3];
        let g = src[i * 3 + 1];
        let b = src[i * 3 + 2];

        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
        let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;

        out[i * 3] = y;
        out[i * 3 + 1] = cb;
        out[i * 3 + 2] = cr;
    }

    ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
}

/// Convert f32 YCbCr to f32 RGB [0,255].
fn ycbcr_to_rgb_f32(ycbcr: &ViewBuffer) -> ViewBuffer {
    let shape = ycbcr.shape();
    let (h, w) = (shape[0], shape[1]);
    let src = ycbcr.as_slice::<f32>();
    let mut out = vec![0.0f32; h * w * 3];

    for i in 0..(h * w) {
        let y = src[i * 3];
        let cb = src[i * 3 + 1] - 128.0;
        let cr = src[i * 3 + 2] - 128.0;

        let r = y + 1.402 * cr;
        let g = y - 0.344136 * cb - 0.714136 * cr;
        let b = y + 1.772 * cb;

        out[i * 3] = r.clamp(0.0, 255.0);
        out[i * 3 + 1] = g.clamp(0.0, 255.0);
        out[i * 3 + 2] = b.clamp(0.0, 255.0);
    }

    ViewBuffer::from_vec_with_shape(out, vec![h, w, 3])
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
            from: ColorSpace::Rgb,
            to: ColorSpace::Bgr,
        };
        let bgr = apply_color_convert(&rgb, &op_fwd);
        assert_eq!(bgr.as_slice::<u8>(), &[200, 150, 100]);

        let op_bwd = ColorConvertOp {
            from: ColorSpace::Bgr,
            to: ColorSpace::Rgb,
        };
        let back = apply_color_convert(&bgr, &op_bwd);
        assert_eq!(back.as_slice::<u8>(), &[100, 150, 200]);
    }

    #[test]
    fn test_rgb_to_gray() {
        let rgb = make_rgb_u8(255, 0, 0);
        let op = ColorConvertOp {
            from: ColorSpace::Rgb,
            to: ColorSpace::Gray,
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
            from: ColorSpace::Rgb,
            to: ColorSpace::Hsv,
        };
        let hsv = apply_color_convert(&rgb, &to_hsv);

        let to_rgb = ColorConvertOp {
            from: ColorSpace::Hsv,
            to: ColorSpace::Rgb,
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
            from: ColorSpace::Rgb,
            to: ColorSpace::YCbCr,
        };
        let ycbcr = apply_color_convert(&rgb, &to_ycbcr);

        let to_rgb = ColorConvertOp {
            from: ColorSpace::YCbCr,
            to: ColorSpace::Rgb,
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
            from: ColorSpace::Rgb,
            to: ColorSpace::Lab,
        };
        let lab = apply_color_convert(&rgb, &to_lab);
        // LAB always outputs f32
        assert_eq!(lab.dtype(), DType::F32);

        let to_rgb = ColorConvertOp {
            from: ColorSpace::Lab,
            to: ColorSpace::Rgb,
        };
        let back = apply_color_convert(&lab, &to_rgb);
        // LAB->RGB also stays f32
        assert_eq!(back.dtype(), DType::F32);
        let back_vals = back.as_slice::<f32>();
        assert!((back_vals[0] - 100.0).abs() <= 2.0);
        assert!((back_vals[1] - 150.0).abs() <= 2.0);
        assert!((back_vals[2] - 200.0).abs() <= 2.0);
    }

    #[test]
    fn test_noop_conversion() {
        let rgb = make_rgb_u8(100, 150, 200);
        let op = ColorConvertOp {
            from: ColorSpace::Rgb,
            to: ColorSpace::Rgb,
        };
        let result = apply_color_convert(&rgb, &op);
        assert_eq!(result.as_slice::<u8>(), rgb.as_slice::<u8>());
    }
}
