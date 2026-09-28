//! The affine warp: `rotate` by a non-lattice angle and `warp_affine`.
//!
//! Inverse mapping: each output pixel asks where it came from, and samples
//! the input there (nearest, or bilinear in `f64`). The kernel is compiled per
//! channel count (1–4, and one for any other count) and through the CPU
//! dispatch (M1), and a bilinear sample whose four neighbours all lie inside
//! the image reads two row slices instead of bounds-checking every one.
//!
//! **Bit-identical to the kernel it replaced**, which the tests keep as their
//! oracle: every source coordinate is `a * x + b * y + t`, added in that
//! order, and every sample and store converts exactly as before.

use crate::core::buffer::ViewBuffer;
use crate::core::convert::CastFrom;
use crate::core::dispatch::{dispatch, SimdKernel};
use crate::core::dtype::{DType, ViewType};
use crate::ops::affine::{AffineParams, InterpolationType};
use num_traits::NumCast;

/// Rotate `buf` about its centre, as `ComputeOp::RotateAffine` lowers it.
///
/// 0° is how `ComputeOp::lowered` spells the identity (a near-0° angle
/// lowers to exactly 0, never to a warp), and it is the input, packed: the op
/// declares `RequiresContiguous`, so a planned input already is, and its data
/// is shared rather than warped. A 0° warp returned the same values for
/// finite pixels, but smeared a NaN or infinity into its left and upper
/// neighbours (`NaN * 0.0` is NaN).
pub(crate) fn rotate(
    buf: ViewBuffer,
    angle_deg: f32,
    expand: bool,
    interpolation: InterpolationType,
    border_value: f64,
) -> ViewBuffer {
    if angle_deg == 0.0 && !expand {
        return buf.to_contiguous();
    }
    let h = buf.shape()[0] as u32;
    let w = buf.shape()[1] as u32;
    let params = AffineParams::from_rotation(angle_deg, h, w, expand, interpolation, border_value);
    apply_affine_warp(buf, params)
}

/// Apply a 2D affine warp to a `ViewBuffer`, in its own dtype.
pub(crate) fn apply_affine_warp(buf: ViewBuffer, params: AffineParams) -> ViewBuffer {
    if buf.shape().len() < 2 {
        return buf;
    }
    match buf.dtype() {
        DType::U8 => warp_typed::<u8>(buf, &params),
        DType::I8 => warp_typed::<i8>(buf, &params),
        DType::U16 => warp_typed::<u16>(buf, &params),
        DType::I16 => warp_typed::<i16>(buf, &params),
        DType::U32 => warp_typed::<u32>(buf, &params),
        DType::I32 => warp_typed::<i32>(buf, &params),
        DType::F32 => warp_typed::<f32>(buf, &params),
        DType::F64 => warp_typed::<f64>(buf, &params),
        DType::U64 => warp_typed::<u64>(buf, &params),
        DType::I64 => warp_typed::<i64>(buf, &params),
    }
}

fn warp_typed<T: WarpElem>(buf: ViewBuffer, params: &AffineParams) -> ViewBuffer
where
    f64: CastFrom<T>,
{
    let shape = buf.shape();
    let (in_h, in_w) = (shape[0], shape[1]);
    let channels = shape.get(2).copied().unwrap_or(1);
    // Whether the input carried an explicit channel axis, as distinct from a
    // channel *count* of 1: `[H, W]` and `[H, W, 1]` have the same count but
    // different ranks, and the output must keep whichever the input had.
    let has_channel_axis = shape.len() >= 3;
    let (out_h, out_w) = (params.output_height as usize, params.output_width as usize);

    // The user-facing matrix follows OpenCV convention (forward mapping);
    // inverse mapping needs its inverse. A singular matrix is rejected where
    // the user supplies it (`AffineParams::is_invertible`), so it cannot reach
    // here: substituting the identity, as this once did, reported a transform
    // that had not happened.
    debug_assert!(
        params.is_invertible(),
        "affine warp reached the runner with a singular matrix (det = {}); \
         invertibility is enforced when the matrix is accepted",
        params.determinant()
    );
    let [a_fwd, b_fwd, tx_fwd, c_fwd, d_fwd, ty_fwd] = params.matrix;
    let inv_det = 1.0 / params.determinant();
    let a = d_fwd * inv_det;
    let b = -b_fwd * inv_det;
    let c = -c_fwd * inv_det;
    let d = a_fwd * inv_det;
    let tx = -(a * tx_fwd + b * ty_fwd);
    let ty = -(c * tx_fwd + d * ty_fwd);

    let contiguous = buf.to_contiguous();
    let src: &[T] = contiguous.as_slice::<T>();
    let kernel = |ch| Warp {
        src,
        in_h,
        in_w,
        ch,
        out_h,
        out_w,
        inverse: [a, b, tx, c, d, ty],
        interpolation: params.interpolation,
        border_value: params.border_value,
    };
    let dst = match channels {
        1 => dispatch(kernel(1).channels::<1>()),
        2 => dispatch(kernel(2).channels::<2>()),
        3 => dispatch(kernel(3).channels::<3>()),
        4 => dispatch(kernel(4).channels::<4>()),
        other => dispatch(kernel(other)),
    };
    // Mirror `ComputeOp::Affine`'s `shape`, which replaces H and W and leaves
    // the rest of the input shape alone.
    let output_shape = if has_channel_axis {
        vec![out_h, out_w, channels]
    } else {
        vec![out_h, out_w]
    };
    ViewBuffer::from_vec(dst).reshape(output_shape)
}

/// One warp of a packed `[in_h, in_w, ch]` image into a fresh
/// `[out_h, out_w, ch]` one. `C` is the channel count when it is 1–4 (so the
/// per-channel loops unroll), and 0 for any other, read from `ch`.
#[derive(Clone)]
struct Warp<'a, T, const C: usize> {
    src: &'a [T],
    in_h: usize,
    in_w: usize,
    ch: usize,
    out_h: usize,
    out_w: usize,
    /// The inverse matrix `[a, b, tx, c, d, ty]`: output `(x, y)` samples
    /// input `(a x + b y + tx, c x + d y + ty)`.
    inverse: [f64; 6],
    interpolation: InterpolationType,
    border_value: f64,
}

impl<'a, T> Warp<'a, T, 0> {
    /// The same warp, compiled for exactly `K` channels.
    fn channels<const K: usize>(self) -> Warp<'a, T, K> {
        debug_assert_eq!(self.ch, K);
        Warp {
            src: self.src,
            in_h: self.in_h,
            in_w: self.in_w,
            ch: self.ch,
            out_h: self.out_h,
            out_w: self.out_w,
            inverse: self.inverse,
            interpolation: self.interpolation,
            border_value: self.border_value,
        }
    }
}

/// The element types a warp reads as `f64` and stores back from it, both
/// through the crate's one conversion rule (M5, `core::convert`): exact to
/// `f64`, and round-then-saturate back to an integer.
pub(crate) trait WarpElem: ViewType + Default + NumCast + CastFrom<f64>
where
    f64: CastFrom<Self>,
{
}

impl<T> WarpElem for T
where
    T: ViewType + Default + NumCast + CastFrom<f64>,
    f64: CastFrom<T>,
{
}

/// A sample in the `f64` the interpolation computes in.
#[inline(always)]
fn load<T: WarpElem>(v: T) -> f64
where
    f64: CastFrom<T>,
{
    f64::cast_from(v)
}

/// An interpolated value stored as `T`: rounded (half away from zero) and
/// saturated for an integer dtype, as is for a float one.
#[inline(always)]
fn store<T: WarpElem>(v: f64) -> T
where
    f64: CastFrom<T>,
{
    T::cast_from(v)
}

impl<T: WarpElem, const C: usize> SimdKernel for Warp<'_, T, C>
where
    f64: CastFrom<T>,
{
    type Output = Vec<T>;

    #[inline(always)]
    fn run(self) -> Vec<T> {
        let ch = if C == 0 { self.ch } else { C };
        let border: T = NumCast::from(self.border_value).unwrap_or(T::default());
        let mut dst = vec![border; self.out_h * self.out_w * ch];
        if dst.is_empty() {
            return dst;
        }
        match self.interpolation {
            InterpolationType::Nearest => self.nearest(&mut dst, ch),
            InterpolationType::Bilinear => self.bilinear(&mut dst, ch),
        }
        dst
    }
}

impl<T: WarpElem, const C: usize> Warp<'_, T, C>
where
    f64: CastFrom<T>,
{
    /// Every output pixel as its source coordinates: `each(px, x_src, y_src)`.
    /// `a * x + b * y + tx`, added in that order: `b * y` is the same value
    /// for the whole row, so it is computed once per row, not re-associated.
    /// `x` and `y` count in `f64` (exact below 2^53): the same values as
    /// `x_dst as f64`, without an unsigned 64-bit conversion per pixel.
    #[inline(always)]
    fn for_each_pixel(&self, dst: &mut [T], ch: usize, mut each: impl FnMut(&mut [T], f64, f64)) {
        let [a, b, tx, c, d, ty] = self.inverse;
        let mut y = 0.0f64;
        for row in dst.chunks_exact_mut(self.out_w * ch) {
            let by = b * y;
            let dy = d * y;
            let mut x = 0.0f64;
            for px in row.chunks_exact_mut(ch) {
                each(px, a * x + by + tx, c * x + dy + ty);
                x += 1.0;
            }
            y += 1.0;
        }
    }

    #[inline(always)]
    fn nearest(&self, dst: &mut [T], ch: usize) {
        let (in_w, in_h, src) = (self.in_w, self.in_h, self.src);
        // `round(v)` (half away from zero) lands in `0..n` exactly when
        // `-0.5 < v < n - 0.5`; NaN fails both and takes the general path.
        let (x_hi, y_hi) = (in_w as f64 - 0.5, in_h as f64 - 0.5);
        self.for_each_pixel(dst, ch, |px, x_src, y_src| {
            let at = if x_src > -0.5 && x_src < x_hi && y_src > -0.5 && y_src < y_hi {
                // SAFETY: both rounded coordinates are integers in
                // `0..in_w` / `0..in_h` (the test above), so finite and
                // representable. Through `i64`: one signed conversion.
                let (sx, sy): (i64, i64) = unsafe {
                    (
                        x_src.round().to_int_unchecked(),
                        y_src.round().to_int_unchecked(),
                    )
                };
                (sy as usize * in_w + sx as usize) * ch
            } else {
                let sx = x_src.round() as i64;
                let sy = y_src.round() as i64;
                if !(sx >= 0 && sy >= 0 && (sx as usize) < in_w && (sy as usize) < in_h) {
                    return;
                }
                (sy as usize * in_w + sx as usize) * ch
            };
            px.copy_from_slice(&src[at..at + ch]);
        });
    }

    #[inline(always)]
    fn bilinear(&self, dst: &mut [T], ch: usize) {
        let (in_w, in_h, src) = (self.in_w, self.in_h, self.src);
        let (w, h) = (in_w as i64, in_h as i64);
        let bv = self.border_value;
        let row_len = in_w * ch;
        // All four neighbours lie inside exactly when `0 <= floor(v)` and
        // `floor(v) + 1 < n`, that is `0 <= v < n - 1` (`n - 1` is an exact
        // integer). NaN fails the test and takes the general path.
        let (x_hi, y_hi) = (in_w as f64 - 1.0, in_h as f64 - 1.0);
        self.for_each_pixel(dst, ch, |px, x_src, y_src| {
            let lerp = |dx: f64, dy: f64, v00: f64, v10: f64, v01: f64, v11: f64| {
                let v0 = v00 * (1.0 - dx) + v10 * dx;
                let v1 = v01 * (1.0 - dx) + v11 * dx;
                v0 * (1.0 - dy) + v1 * dy
            };
            if x_src >= 0.0 && y_src >= 0.0 && x_src < x_hi && y_src < y_hi {
                // Two rows of two pixels. `floor` is what `x0 as f64` was.
                let (xf, yf) = (x_src.floor(), y_src.floor());
                let (dx, dy) = (x_src - xf, y_src - yf);
                // SAFETY: `xf`, `yf` are integers in `0..in_w - 1` /
                // `0..in_h - 1` (the test above), so finite and representable.
                // Through `i64`: one signed conversion.
                let (x0, y0): (i64, i64) =
                    unsafe { (xf.to_int_unchecked(), yf.to_int_unchecked()) };
                let at = (y0 as usize * in_w + x0 as usize) * ch;
                let top = &src[at..at + 2 * ch];
                let bottom = &src[at + row_len..at + row_len + 2 * ch];
                for k in 0..ch {
                    px[k] = store(lerp(
                        dx,
                        dy,
                        load(top[k]),
                        load(top[ch + k]),
                        load(bottom[k]),
                        load(bottom[ch + k]),
                    ));
                }
                return;
            }
            // On the edge, off the image, or NaN: the general form.
            let x0 = x_src.floor() as i64;
            let y0 = y_src.floor() as i64;
            // Saturating: a huge source coordinate saturates `as i64` to
            // `i64::MAX`, which is off the image either way.
            let x1 = x0.saturating_add(1);
            let y1 = y0.saturating_add(1);
            // Fully off the image: the pixel keeps the border value.
            if x1 < 0 || y1 < 0 || x0 >= w || y0 >= h {
                return;
            }
            let dx = x_src - x0 as f64;
            let dy = y_src - y0 as f64;
            // A neighbour outside reads the border value.
            let sample = |x: i64, y: i64, k: usize| {
                if x >= 0 && y >= 0 && x < w && y < h {
                    load(src[(y as usize * in_w + x as usize) * ch + k])
                } else {
                    bv
                }
            };
            for (k, out) in px.iter_mut().enumerate() {
                *out = store(lerp(
                    dx,
                    dy,
                    sample(x0, y0, k),
                    sample(x1, y0, k),
                    sample(x0, y1, k),
                    sample(x1, y1, k),
                ));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dispatch::KernelOutput;

    /// The warp as it was before Phase 7, kept verbatim as the parity oracle.
    pub(super) fn reference_warp<T>(
        buf: ViewBuffer,
        params: &crate::ops::affine::AffineParams,
    ) -> ViewBuffer
    where
        T: crate::core::dtype::ViewType + Default + num_traits::NumCast,
    {
        use crate::ops::affine::InterpolationType;
        use num_traits::NumCast;

        let shape = buf.shape();
        let in_h = shape[0];
        let in_w = shape[1];
        let channels = shape.get(2).copied().unwrap_or(1);
        // Whether the input carried an explicit channel axis, as distinct from a
        // channel *count* of 1: `[H, W]` and `[H, W, 1]` have the same count but
        // different ranks, and the output must keep whichever the input had.
        let has_channel_axis = shape.len() >= 3;

        let out_h = params.output_height as usize;
        let out_w = params.output_width as usize;

        // The user-facing matrix follows OpenCV convention (forward mapping).
        // Invert the 2x3 matrix for inverse-mapping interpolation.
        let [a_fwd, b_fwd, tx_fwd, c_fwd, d_fwd, ty_fwd] = params.matrix;
        // A singular matrix is rejected where the user supplies it (the plugin's
        // `warp_affine` arm, via `AffineParams::is_invertible`), so it cannot reach
        // here. This used to substitute the identity instead, which handed back the
        // input and reported a transform that had not happened — a fallback that
        // hid the one case inverse mapping cannot express. `debug_assert` pins the
        // invariant at its consumer without a release-build branch; the arithmetic
        // below has no other way to fail.
        debug_assert!(
            params.is_invertible(),
            "affine warp reached the runner with a singular matrix (det = {}); \
             invertibility is enforced when the matrix is accepted",
            params.determinant()
        );
        let inv_det = 1.0 / params.determinant();
        let a = d_fwd * inv_det;
        let b = -b_fwd * inv_det;
        let c = -c_fwd * inv_det;
        let d = a_fwd * inv_det;
        let tx = -(a * tx_fwd + b * ty_fwd);
        let ty = -(c * tx_fwd + d * ty_fwd);

        let contig_buf = if buf.layout.is_contiguous() {
            buf
        } else {
            buf.to_contiguous()
        };
        let src_data: &[T] = contig_buf.as_slice::<T>();

        let output_size = out_h * out_w * channels;
        let border_val: T = NumCast::from(params.border_value).unwrap_or(T::default());
        let mut dst_data: Vec<T> = vec![border_val; output_size];

        let is_float = matches!(T::DTYPE, crate::DType::F32 | crate::DType::F64);

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
                        // Wrapping, as the release build this oracle stands
                        // for behaved; `+ 1` panicked in debug builds at a
                        // saturated coordinate.
                        let x1 = x0.wrapping_add(1);
                        let y1 = y0.wrapping_add(1);

                        // Fully out of bounds — dst already filled with border_val
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
                                    NumCast::from(src_data[idx]).unwrap_or(bv)
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

                            let clamped = if is_float {
                                v
                            } else {
                                crate::execution::runner::clamp_for_dtype(v, T::DTYPE)
                            };
                            dst_data[dst_idx + ch] = NumCast::from(clamped).unwrap_or(T::default());
                        }
                    }
                }
            }
        }

        // Mirror `ComputeOp::Affine`'s `shape`, which replaces H and W and
        // leaves the rest of the input shape alone. Collapsing a `[H, W, 1]` input
        // to `[H, W]` here contradicted that contract, so a single-channel affine
        // planned rank 3 and produced rank 2.
        let output_shape = if has_channel_axis {
            vec![out_h, out_w, channels]
        } else {
            vec![out_h, out_w]
        };

        ViewBuffer::from_vec(dst_data).reshape(output_shape)
    }

    /// Every dtype, as a pattern with its extremes, and NaN and infinities
    /// for the floats, so saturation and non-finite arithmetic are compared too.
    /// The 64-bit integers stay below 2^63: at the top of their range the
    /// reference stored 0 (`sixty_four_bit_values_saturate_rather_than_become_zero`).
    fn image(dtype: DType, shape: &[usize]) -> ViewBuffer {
        let n: usize = shape.iter().product();
        let wave = |i: usize| ((i * 37 + 11) % 256) as f64;
        macro_rules! make {
            ($t:ty, $f:expr) => {
                ViewBuffer::from_vec_with_shape((0..n).map($f).collect::<Vec<$t>>(), shape.to_vec())
            };
        }
        match dtype {
            DType::U8 => make!(u8, |i| wave(i) as u8),
            DType::I8 => make!(i8, |i| (wave(i) - 128.0) as i8),
            DType::U16 => make!(u16, |i| (wave(i) * 257.0) as u16),
            DType::I16 => make!(i16, |i| ((wave(i) - 128.0) * 255.0) as i16),
            DType::U32 => make!(u32, |i| (wave(i) * 16_000_000.0) as u32),
            DType::I32 => make!(i32, |i| ((wave(i) - 128.0) * 16_000_000.0) as i32),
            DType::U64 => make!(u64, |i| if i % 5 == 0 { 1 << 62 } else { wave(i) as u64 }),
            DType::I64 => make!(i64, |i| match i % 5 {
                0 => i64::MIN,
                3 => 1 << 62,
                _ => wave(i) as i64,
            }),
            DType::F32 => make!(f32, |i| match i % 23 {
                7 => f32::NAN,
                13 => f32::INFINITY,
                17 => f32::NEG_INFINITY,
                _ => (wave(i) * 1.37 - 40.0) as f32,
            }),
            DType::F64 => make!(f64, |i| match i % 19 {
                5 => f64::NAN,
                11 => f64::INFINITY,
                _ => wave(i) * 0.731 - 20.0,
            }),
        }
    }

    fn reference(buf: ViewBuffer, params: &AffineParams) -> ViewBuffer {
        match buf.dtype() {
            DType::U8 => reference_warp::<u8>(buf, params),
            DType::I8 => reference_warp::<i8>(buf, params),
            DType::U16 => reference_warp::<u16>(buf, params),
            DType::I16 => reference_warp::<i16>(buf, params),
            DType::U32 => reference_warp::<u32>(buf, params),
            DType::I32 => reference_warp::<i32>(buf, params),
            DType::F32 => reference_warp::<f32>(buf, params),
            DType::F64 => reference_warp::<f64>(buf, params),
            DType::U64 => reference_warp::<u64>(buf, params),
            DType::I64 => reference_warp::<i64>(buf, params),
        }
    }

    const DTYPES: [DType; 10] = [
        DType::U8,
        DType::I8,
        DType::U16,
        DType::I16,
        DType::U32,
        DType::I32,
        DType::U64,
        DType::I64,
        DType::F32,
        DType::F64,
    ];

    /// Input shapes: rank 2 and 3, 1–5 channels, square and not, down to 1x1.
    const SHAPES: [&[usize]; 8] = [
        &[1, 1, 3],
        &[2, 3],
        &[7, 5, 1],
        &[9, 16, 2],
        &[16, 9, 3],
        &[13, 21, 4],
        &[6, 8, 5],
        &[20, 20],
    ];

    fn assert_same(got: &ViewBuffer, want: &ViewBuffer, what: &str) {
        assert_eq!(got.shape(), want.shape(), "{what}");
        assert_eq!(got.dtype(), want.dtype(), "{what}");
        assert!(
            got.output_bytes() == want.output_bytes(),
            "{what}: the warp differs from the reference kernel"
        );
    }

    #[test]
    fn rotations_match_the_reference_kernel_bit_for_bit() {
        for dtype in DTYPES {
            for shape in SHAPES {
                let buf = image(dtype, shape);
                for angle in [1.0f32, 30.0, 45.0, 137.0, 200.5, -17.0] {
                    for expand in [false, true] {
                        for interpolation in
                            [InterpolationType::Bilinear, InterpolationType::Nearest]
                        {
                            for border in [0.0, 7.5, -3.0] {
                                let params = AffineParams::from_rotation(
                                    angle,
                                    shape[0] as u32,
                                    shape[1] as u32,
                                    expand,
                                    interpolation,
                                    border,
                                );
                                let what = format!(
                                    "{dtype:?} {shape:?} {angle}° expand={expand} \
                                     {interpolation:?} border={border}"
                                );
                                let got = rotate(buf.clone(), angle, expand, interpolation, border);
                                assert_same(&got, &reference(buf.clone(), &params), &what);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn affine_warps_match_the_reference_kernel_bit_for_bit() {
        // Scale, shear, a translation off the image, a mirror, a huge scale
        // whose source coordinates saturate the i64 conversions, and NaN ones.
        let matrices = [
            [0.5, 0.0, 1.25, 0.0, 0.5, -0.75],
            [1.0, 0.3, 0.0, -0.2, 1.0, 2.0],
            [1.0, 0.0, 400.0, 0.0, 1.0, -400.0],
            [-1.0, 0.0, 8.0, 0.0, 1.0, 0.0],
            [1e-7, 0.0, 5e20, 0.0, 1e-7, -5e20],
            // Inverse translation `inf - inf`: every source coordinate is NaN.
            [1e-7, 1e-7, 1e305, 0.0, 1e-7, 1e305],
        ];
        for dtype in DTYPES {
            for shape in SHAPES {
                let buf = image(dtype, shape);
                for matrix in matrices {
                    for (out_h, out_w) in [(5, 7), (shape[0] as u32, shape[1] as u32)] {
                        for interpolation in
                            [InterpolationType::Bilinear, InterpolationType::Nearest]
                        {
                            let params = AffineParams {
                                matrix,
                                output_height: out_h,
                                output_width: out_w,
                                interpolation,
                                border_value: 2.5,
                            };
                            let what = format!("{dtype:?} {shape:?} {matrix:?} {interpolation:?}");
                            let got = apply_affine_warp(buf.clone(), params.clone());
                            assert_same(&got, &reference(buf.clone(), &params), &what);
                        }
                    }
                }
            }
        }
    }

    /// A 64-bit value at the top of its range is stored saturated, as every
    /// float→integer conversion in the crate is (M5). The warp's own store
    /// clamped to `u64::MAX as f64` (2^64, one past the range), which the
    /// conversion after it refused, and stored 0 instead.
    #[test]
    fn sixty_four_bit_values_saturate_rather_than_become_zero() {
        // A mirror: every source coordinate is an exact pixel centre.
        let params = AffineParams {
            matrix: [-1.0, 0.0, 3.0, 0.0, 1.0, 0.0],
            output_height: 4,
            output_width: 4,
            interpolation: InterpolationType::Bilinear,
            border_value: 0.0,
        };
        let u = ViewBuffer::from_vec_with_shape(vec![u64::MAX; 16], vec![4, 4]);
        let got = apply_affine_warp(u, params.clone());
        assert_eq!(got.as_slice::<u64>(), &[u64::MAX; 16]);
        let i = ViewBuffer::from_vec_with_shape(vec![i64::MAX; 16], vec![4, 4]);
        let got = apply_affine_warp(i, params);
        assert_eq!(got.as_slice::<i64>(), &[i64::MAX; 16]);
    }

    /// A 0° rotation is its input: the data is shared, not warped.
    #[test]
    fn a_zero_degree_rotation_shares_its_input() {
        let buf = image(DType::U8, &[16, 9, 3]);
        let before = buf.as_slice::<u8>().as_ptr();
        let got = rotate(buf.clone(), 0.0, false, InterpolationType::Bilinear, 0.0);
        assert_eq!(got.as_slice::<u8>().as_ptr(), before);
        assert_eq!(got.shape(), buf.shape());
    }

    /// Through the engine, as the plugin runs it: `rotate(0)` and a near-0
    /// angle lower to the identity, which keeps a NaN where it was. The warp
    /// it replaced turned its left and upper neighbours into NaN too.
    #[test]
    fn a_zero_degree_rotation_leaves_every_value_where_it_was() {
        use crate::ops::compute::ComputeOp;
        use crate::{ViewDto, ViewExpr};
        let mut values: Vec<f32> = (0..20).map(|v| v as f32).collect();
        values[12] = f32::NAN;
        let buf = ViewBuffer::from_vec_with_shape(values.clone(), vec![4, 5]);
        for angle in [0.0f32, 360.0, 0.0004] {
            let op = ViewDto::Compute(ComputeOp::Rotate {
                angle,
                expand: false,
                interpolation: InterpolationType::Bilinear,
                border_value: 0.0,
            });
            let got = ViewExpr::new_source(buf.clone())
                .apply_op(op)
                .plan()
                .execute();
            let got = got.to_contiguous();
            assert_eq!(
                got.as_slice::<f32>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{angle}°"
            );
        }
    }
}
