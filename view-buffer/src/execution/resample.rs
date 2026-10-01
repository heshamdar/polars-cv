//! Resampling for the dtypes fast_image_resize has no pixel type for.
//!
//! fast_image_resize (fir) resamples u8, u16 and f32 images, which is every
//! dtype whose [`DType::accumulator`] is f32 once i8/i16 are widened to f32
//! exactly. The 32/64-bit integers and f64 hold values f32 cannot, so they
//! are resampled here instead, with fir's own algorithm in f64:
//!
//! - **nearest** is a gather in the image's own dtype ([`nearest`]), so it is
//!   exact on every dtype; its source positions are fir's
//!   (`resample_nearest`: the column table by multiplication, the rows by
//!   accumulated steps);
//! - the **convolution** filters ([`convolve`]) use fir's filter functions
//!   and `precompute_coefficients` (adaptive kernel width, normalized
//!   weights, zero weights trimmed from each bound), horizontal pass first as
//!   fir does for non-u8 pixels, with alpha premultiplied for 2- and
//!   4-channel images as fir does. The result is stored in the input dtype
//!   by the crate's conversion rule (round, saturate).
//!
//! `nearest_matches_fast_image_resize` and
//! `convolution_matches_fast_image_resize` hold both to fir on the inputs
//! fir accepts.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{with_dtype, DType};
use crate::ops::FilterType;

/// Resample `[H, W]` or `[H, W, C]` to `target_h × target_w`.
pub(crate) fn resample(
    buf: &ViewBuffer,
    target_w: usize,
    target_h: usize,
    filter: FilterType,
) -> ViewBuffer {
    match filter {
        FilterType::Nearest => nearest(buf, target_w, target_h),
        FilterType::Triangle => convolve(buf, target_w, target_h, bilinear, 1.0),
        FilterType::CatmullRom => convolve(buf, target_w, target_h, catmull_rom, 2.0),
        FilterType::Gaussian => convolve(buf, target_w, target_h, gaussian, 3.0),
        FilterType::Lanczos3 => convolve(buf, target_w, target_h, lanczos3, 3.0),
    }
}

/// `(height, width, channels)` and the output shape (same rank as the input).
fn dims(buf: &ViewBuffer, target_w: usize, target_h: usize) -> ((usize, usize, usize), Vec<usize>) {
    let shape = buf.shape();
    let c = shape.get(2).copied().unwrap_or(1);
    let mut out = shape.to_vec();
    out[0] = target_h;
    out[1] = target_w;
    ((shape[0], shape[1], c), out)
}

/// Nearest-neighbour resampling: a gather, in the input's own dtype.
fn nearest(buf: &ViewBuffer, target_w: usize, target_h: usize) -> ViewBuffer {
    let ((h, w, c), out_shape) = dims(buf, target_w, target_h);
    if target_w == 0 || target_h == 0 || h == 0 || w == 0 {
        return with_dtype!(buf.dtype(), T => ViewBuffer::from_vec_with_shape(Vec::<T>::new(), out_shape));
    }
    // fir's positions: the centre of each output pixel, mapped back.
    let x_scale = w as f64 / target_w as f64;
    let y_scale = h as f64 / target_h as f64;
    let x_start = x_scale * 0.5;
    let cols: Vec<usize> = (0..target_w)
        .map(|x| ((x_start + x_scale * x as f64) as usize).min(w - 1))
        .collect();
    let mut rows = Vec::with_capacity(target_h);
    let mut y = y_scale * 0.5;
    for _ in 0..target_h {
        rows.push((y as usize).min(h - 1));
        y += y_scale;
    }
    let contig = buf.to_contiguous();
    with_dtype!(buf.dtype(), T => {
        let src = contig.as_slice::<T>();
        let mut out: Vec<T> = Vec::with_capacity(target_w * target_h * c);
        for &r in &rows {
            for &col in &cols {
                let at = (r * w + col) * c;
                out.extend_from_slice(&src[at..at + c]);
            }
        }
        ViewBuffer::from_vec_with_shape(out, out_shape)
    })
}

/// One output position's taps: the first input index and its weights.
struct Taps {
    start: usize,
    weights: Vec<f64>,
}

/// fir's `precompute_coefficients` with `adaptive_kernel_size` (its
/// convolution mode): the filter widens by the downscale factor, weights are
/// normalized to sum to 1, and zero weights at either end are dropped.
fn coefficients(
    in_size: usize,
    out_size: usize,
    filter: fn(f64) -> f64,
    support: f64,
) -> Vec<Taps> {
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let radius = support * filter_scale;
    let recip = 1.0 / filter_scale;
    (0..out_size)
        .map(|out_x| {
            let in_center = (out_x as f64 + 0.5) * scale;
            let x_min = (in_center - radius).floor().max(0.0) as usize;
            let x_max = (in_center + radius).ceil().min(in_size as f64) as usize;
            let center = in_center - 0.5;
            let mut start = x_min;
            let mut weights: Vec<f64> = Vec::with_capacity(x_max - x_min);
            for x in x_min..x_max {
                let wgt = filter((x as f64 - center) * recip);
                if x == start && wgt == 0.0 {
                    start += 1;
                } else {
                    weights.push(wgt);
                }
            }
            while weights.last() == Some(&0.0) {
                weights.pop();
            }
            let sum: f64 = weights.iter().sum();
            if sum != 0.0 {
                weights.iter_mut().for_each(|w| *w /= sum);
            }
            Taps { start, weights }
        })
        .collect()
}

/// Separable convolution in f64 along one axis of a packed `[rows, cols, c]`
/// plane: `axis_len` is the resampled axis (rows when `vertical`).
fn pass(
    src: &[f64],
    (rows, cols, c): (usize, usize, usize),
    taps: &[Taps],
    vertical: bool,
) -> Vec<f64> {
    let (out_rows, out_cols) = if vertical {
        (taps.len(), cols)
    } else {
        (rows, taps.len())
    };
    let mut out = vec![0.0f64; out_rows * out_cols * c];
    for r in 0..out_rows {
        for col in 0..out_cols {
            let t = if vertical { &taps[r] } else { &taps[col] };
            for ch in 0..c {
                let mut acc = 0.0f64;
                for (k, &wgt) in t.weights.iter().enumerate() {
                    let (sr, sc) = if vertical {
                        (t.start + k, col)
                    } else {
                        (r, t.start + k)
                    };
                    acc += wgt * src[(sr * cols + sc) * c + ch];
                }
                out[(r * out_cols + col) * c + ch] = acc;
            }
        }
    }
    out
}

/// Convolution resampling in f64 (module docs).
fn convolve(
    buf: &ViewBuffer,
    target_w: usize,
    target_h: usize,
    filter: fn(f64) -> f64,
    support: f64,
) -> ViewBuffer {
    let ((h, w, c), out_shape) = dims(buf, target_w, target_h);
    let dtype = buf.dtype();
    if target_w == 0 || target_h == 0 || h == 0 || w == 0 {
        return ViewBuffer::from_vec_with_shape(Vec::<f64>::new(), out_shape).cast(dtype);
    }
    let packed = buf.cast(DType::F64).to_contiguous();
    let mut data: Vec<f64> = packed.as_slice::<f64>().to_vec();
    let alpha = matches!(c, 2 | 4);
    if alpha {
        for px in data.chunks_exact_mut(c) {
            let a = px[c - 1];
            px[..c - 1].iter_mut().for_each(|v| *v *= a);
        }
    }
    let (mut rows, mut cols) = (h, w);
    if target_w != w {
        data = pass(
            &data,
            (rows, cols, c),
            &coefficients(w, target_w, filter, support),
            false,
        );
        cols = target_w;
    }
    if target_h != h {
        data = pass(
            &data,
            (rows, cols, c),
            &coefficients(h, target_h, filter, support),
            true,
        );
        rows = target_h;
    }
    debug_assert_eq!((rows, cols), (target_h, target_w));
    if alpha {
        for px in data.chunks_exact_mut(c) {
            let a = px[c - 1];
            let recip = if a == 0.0 { 0.0 } else { 1.0 / a };
            px[..c - 1].iter_mut().for_each(|v| *v *= recip);
        }
    }
    ViewBuffer::from_vec_with_shape(data, out_shape).cast(dtype)
}

// fir's filter functions (`convolution/filters.rs`), verbatim.

fn bilinear(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

fn catmull_rom(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.) * x - (A + 3.)) * x * x + 1.
    } else if x < 2.0 {
        (((x - 5.) * x + 8.) * x - 4.) * A
    } else {
        0.0
    }
}

fn gaussian(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        let r = 0.5f64;
        ((2.0 * std::f64::consts::PI).sqrt() * r).recip() * (-x.powi(2) / (2.0 * r.powi(2))).exp()
    } else {
        0.0
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x = x * std::f64::consts::PI;
        x.sin() / x
    }
}

fn lanczos3(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageOp, ImageOpKind, ViewDto, ViewExpr};

    /// fir's own resize, through the engine's u8/f32 path.
    fn fir_resize(buf: ViewBuffer, w: u32, h: u32, filter: FilterType) -> ViewBuffer {
        let op = ViewDto::Image(ImageOp {
            kind: ImageOpKind::Resize {
                width: w,
                height: h,
                filter,
            },
        });
        ViewExpr::new_source(buf).apply_op(op).plan().execute()
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + 11) % 251) as u8).collect()
    }

    const SIZES: [(usize, usize); 4] = [(1, 1), (3, 5), (7, 4), (9, 9)];
    const TARGETS: [(usize, usize); 5] = [(1, 1), (2, 3), (5, 8), (13, 6), (16, 16)];

    #[test]
    fn nearest_matches_fast_image_resize() {
        for (h, w) in SIZES {
            for c in 1..=4 {
                let buf = ViewBuffer::from_vec_with_shape(pattern(h * w * c), vec![h, w, c]);
                for (th, tw) in TARGETS {
                    let ours = nearest(&buf, tw, th);
                    let fir = fir_resize(buf.clone(), tw as u32, th as u32, FilterType::Nearest);
                    assert_eq!(
                        ours.as_slice::<u8>(),
                        fir.to_contiguous().as_slice::<u8>(),
                        "{h}x{w}x{c} -> {th}x{tw}"
                    );
                }
            }
        }
    }

    /// On f32 input the f64 convolution agrees with fir's f32 one to f32
    /// rounding (fir accumulates in f32).
    #[test]
    fn convolution_matches_fast_image_resize() {
        for filter in [
            FilterType::Triangle,
            FilterType::CatmullRom,
            FilterType::Gaussian,
            FilterType::Lanczos3,
        ] {
            for (h, w) in SIZES {
                for c in 1..=4 {
                    let values: Vec<f32> = pattern(h * w * c)
                        .iter()
                        .map(|&v| f32::from(v) / 7.0 + 1.0)
                        .collect();
                    let buf = ViewBuffer::from_vec_with_shape(values, vec![h, w, c]);
                    for (th, tw) in TARGETS {
                        let ours = resample(&buf, tw, th, filter);
                        let fir = fir_resize(buf.clone(), tw as u32, th as u32, filter);
                        let fir = fir.to_contiguous();
                        for (i, (a, b)) in ours
                            .as_slice::<f32>()
                            .iter()
                            .zip(fir.as_slice::<f32>())
                            .enumerate()
                        {
                            assert!(
                                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                                "{filter:?} {h}x{w}x{c} -> {th}x{tw} at {i}: {a} vs {b}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A wide integer survives a resize that does not change it: f32 would
    /// round 16777217 to 16777216.
    #[test]
    fn a_constant_wide_image_stays_constant() {
        for filter in [
            FilterType::Nearest,
            FilterType::Triangle,
            FilterType::Lanczos3,
        ] {
            let buf = ViewBuffer::from_vec_with_shape(vec![16_777_217u32; 12], vec![3, 4, 1]);
            let out = resample(&buf, 7, 5, filter);
            assert!(
                out.as_slice::<u32>().iter().all(|&v| v == 16_777_217),
                "{filter:?}: {:?}",
                out.as_slice::<u32>()
            );
        }
    }
}
