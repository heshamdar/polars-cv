//! Resampling for the dtypes fast_image_resize has no pixel type for.
//!
//! fast_image_resize (fir) resamples u8, u16 and f32 images, which is every
//! dtype whose [`DType::accumulator`] is f32 once i8/i16 are widened to f32
//! exactly. The 32/64-bit integers and f64 hold values f32 cannot, so they
//! are resampled here instead, with fir's own algorithm in f64:
//!
//! - **nearest** is a gather in the image's own dtype ([`nearest`]), so it is
//!   exact on every dtype, fir's included: its source positions are computed
//!   in integers ([`nearest_indices`]), where fir's accumulate floating-point
//!   steps and break an exact pixel-centre tie either way;
//! - the **convolution** filters ([`convolve`]) use fir's filter functions
//!   and `precompute_coefficients` (adaptive kernel width, normalized
//!   weights, zero weights trimmed from each bound), horizontal pass first as
//!   fir does for non-u8 pixels, with alpha premultiplied for 2- and
//!   4-channel images as fir does. The result is stored in the input dtype
//!   by the crate's conversion rule (round, saturate).
//!
//! `nearest_takes_the_pixel_under_each_centre_exactly` holds nearest to the
//! pixel-centre rule on every path, and `convolution_matches_fast_image_resize`
//! holds the convolution to fir on the inputs fir accepts.

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

/// The source index each of `dst` output positions samples: the pixel under
/// its centre, `floor((i + 1/2) * src / dst) = floor((2i + 1) * src / (2 * dst))`,
/// in integers. A centre exactly on a boundary (`2 -> 21`, row 10 at 1.0)
/// takes the pixel after it, as exact arithmetic does; a floating-point step
/// lands on either side of it. Always `< src`, since `2i + 1 < 2 * dst`.
pub(crate) fn nearest_indices(src: usize, dst: usize) -> Vec<usize> {
    let (src, dst) = (src as u128, dst as u128);
    (0..dst)
        .map(|i| ((2 * i + 1) * src / (2 * dst)) as usize)
        .collect()
}

/// Nearest-neighbour resampling of any dtype: a gather of whole pixels at
/// [`nearest_indices`], in the input's own dtype.
///
/// Pure data movement, so it copies each pixel's bytes: a fixed-size copy per
/// pixel ([`gather`]), and an output row that samples the same source row as
/// the one before is a copy of it. A view whose pixels are packed within each
/// row (a crop, a vertical flip) is read where it lies, as fir reads one; any
/// other layout is packed first.
pub(crate) fn nearest(buf: &ViewBuffer, target_w: usize, target_h: usize) -> ViewBuffer {
    let ((h, w, c), out_shape) = dims(buf, target_w, target_h);
    if target_w == 0 || target_h == 0 || h == 0 || w == 0 {
        return with_dtype!(buf.dtype(), T => ViewBuffer::from_vec_with_shape(Vec::<T>::new(), out_shape));
    }
    let rows = nearest_indices(h, target_h);
    let packed;
    let src = if buf.layout_facts().is_dense_rows() {
        buf
    } else {
        packed = buf.to_contiguous();
        &packed
    };
    let pixel = c * src.dtype().size_of();
    let offsets: Vec<usize> = nearest_indices(w, target_w)
        .into_iter()
        .map(|x| x * pixel)
        .collect();
    let row_bytes = target_w * pixel;
    with_dtype!(src.dtype(), T => {
        let src_rows = src
            .dense_rows::<T>()
            .expect("a contiguous buffer has packed rows");
        let len = target_h * target_w * c;
        let mut out: Vec<T> = Vec::with_capacity(len);
        // Written row by row below, every byte once, before `set_len`: no
        // zeroing pass over an output this loop overwrites entirely.
        let base: *mut u8 = out.as_mut_ptr().cast();
        for (y, &r) in rows.iter().enumerate() {
            // SAFETY: row `y` is `row_bytes` inside the `len * size_of::<T>()`
            // bytes reserved; each branch writes all of it, from bytes it
            // reads in bounds (the previous output row, already written, or a
            // source row checked by slicing).
            unsafe {
                let dst = base.add(y * row_bytes);
                if y > 0 && rows[y - 1] == r {
                    std::ptr::copy_nonoverlapping(dst.sub(row_bytes), dst, row_bytes);
                } else if target_w == w {
                    let row = &bytes(src_rows[r])[..row_bytes];
                    std::ptr::copy_nonoverlapping(row.as_ptr(), dst, row_bytes);
                } else {
                    gather(bytes(src_rows[r]), &offsets, pixel, dst);
                }
            }
        }
        // SAFETY: every one of the `len` elements was written above.
        unsafe { out.set_len(len) };
        ViewBuffer::from_vec_with_shape(out, out_shape)
    })
}

/// Write the `pixel`-byte pixels of `row` starting at `offsets` to `dst`,
/// one after another, one fixed-size copy each for the common pixel sizes.
///
/// # Safety
/// `dst` must be valid for writes of `offsets.len() * pixel` bytes. Every
/// read is bounds-checked against `row`.
unsafe fn gather(row: &[u8], offsets: &[usize], pixel: usize, dst: *mut u8) {
    #[inline(always)]
    unsafe fn fixed<const N: usize>(row: &[u8], offsets: &[usize], dst: *mut u8) {
        for (i, &at) in offsets.iter().enumerate() {
            let px: &[u8; N] = row[at..at + N].try_into().expect("N bytes");
            // SAFETY: pixel `i` of `offsets.len()` is inside `dst` (caller).
            unsafe { std::ptr::copy_nonoverlapping(px.as_ptr(), dst.add(i * N), N) };
        }
    }
    // SAFETY: forwarded from the caller.
    unsafe {
        match pixel {
            1 => fixed::<1>(row, offsets, dst),
            2 => fixed::<2>(row, offsets, dst),
            3 => fixed::<3>(row, offsets, dst),
            4 => fixed::<4>(row, offsets, dst),
            6 => fixed::<6>(row, offsets, dst),
            8 => fixed::<8>(row, offsets, dst),
            12 => fixed::<12>(row, offsets, dst),
            16 => fixed::<16>(row, offsets, dst),
            _ => {
                for (i, &at) in offsets.iter().enumerate() {
                    let px = &row[at..at + pixel];
                    std::ptr::copy_nonoverlapping(px.as_ptr(), dst.add(i * pixel), pixel);
                }
            }
        }
    }
}

/// The bytes of `values`.
fn bytes<T: crate::core::dtype::ViewType>(values: &[T]) -> &[u8] {
    // SAFETY: a `ViewType` is a plain numeric type (no padding, every bit
    // pattern a value), and `u8` needs no alignment.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
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
    // Written once from the input where it lies (`append_to` walks a view):
    // an f64 input's cast is the input itself, so packing it first was a
    // second copy of a crop or flip.
    let mut data: Vec<f64> = Vec::with_capacity(h * w * c);
    buf.cast(DType::F64).append_to(&mut data);
    let alpha = crate::ops::color::has_alpha(c);
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
    use crate::core::convert::CastFrom;
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

    /// Output pixel `i` of `dst` takes the source pixel under its centre,
    /// `floor((i + 1/2) * src / dst)`, exactly: a centre on a boundary (a
    /// tie, e.g. row 10 of 2 -> 21 at exactly 1.0) takes the pixel after it.
    /// Through the public resize, on a native fir dtype (u8), on f32, and on
    /// i32 (the f64 resampler), with each pixel's value its source index.
    #[test]
    fn nearest_takes_the_pixel_under_each_centre_exactly() {
        let exact = |i: usize, src: usize, dst: usize| (2 * i + 1) * src / (2 * dst);
        for src in 1..=24 {
            let rows: Vec<usize> = (0..src).collect();
            let column = |dtype: DType| -> ViewBuffer {
                with_dtype!(dtype, T => ViewBuffer::from_vec_with_shape(
                    rows.iter().map(|&r| T::cast_from(r as f64)).collect::<Vec<T>>(),
                    vec![src, 1, 1],
                ))
            };
            for dst in 1..=24 {
                for dtype in [DType::U8, DType::F32, DType::I32] {
                    // Rows: an `src x 1` column of row indices; columns: its transpose.
                    let tall = fir_resize(column(dtype), 1, dst as u32, FilterType::Nearest);
                    let wide = fir_resize(
                        column(dtype).reshape(vec![1, src, 1]),
                        dst as u32,
                        1,
                        FilterType::Nearest,
                    );
                    let expect: Vec<f64> = (0..dst).map(|i| exact(i, src, dst) as f64).collect();
                    for (axis, out) in [("rows", tall), ("cols", wide)] {
                        let got: Vec<f64> = with_dtype!(dtype, T => out
                            .to_contiguous()
                            .as_slice::<T>()
                            .iter()
                            .map(|&v| f64::cast_from(v))
                            .collect());
                        assert_eq!(got, expect, "{dtype:?} {axis}: {src} -> {dst}");
                    }
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
