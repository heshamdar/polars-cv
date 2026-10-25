//! 2-D convolution: the interior/border split, accumulating in f32 or f64.

use crate::core::buffer::ViewBuffer;
use crate::core::strided::Walk;
use crate::ops::filter::{BorderMode, ConvolveOp};

/// The float a convolution accumulates in: f32 or f64
/// ([`DType::accumulator`]).
pub(crate) trait Acc:
    crate::core::dtype::ViewType
    + num_traits::Float
    + std::ops::AddAssign
    + std::ops::MulAssign
    + num_traits::FromPrimitive
{
}
impl Acc for f32 {}
impl Acc for f64 {}

/// The rows the convolution reads, `w * c` packed elements each, from the
/// top down: the view's own rows where they are packed (contiguous, a crop, a
/// vertical flip), or else each row packed once, through its [`Walk`], into
/// a ring of the `ksize` rows a tap can reach. No copy of the image is made.
///
/// Output row `y` reads source rows `y - half ..= y + half` only, each
/// replicated or reflected into `0..h` (which lands no further from `y`), so
/// rows `..min(y + half + 1, h)` loaded and the last `ksize` of them kept
/// are every row it needs.
enum Rows<'a, F> {
    Packed(Vec<&'a [F]>),
    Ring {
        view: &'a ViewBuffer,
        ring: Vec<F>,
        row_elems: usize,
        /// Rows `0..loaded` have been packed; the last `ksize` are kept.
        loaded: usize,
    },
}

impl<'a, F: Acc> Rows<'a, F> {
    fn of(view: &'a ViewBuffer, ksize: usize) -> Self {
        match view.dense_rows::<F>() {
            Some(rows) => Rows::Packed(rows),
            None => {
                let shape = view.shape();
                let row_elems = shape[1..].iter().product();
                Rows::Ring {
                    view,
                    ring: vec![F::zero(); ksize * row_elems],
                    row_elems,
                    loaded: 0,
                }
            }
        }
    }

    /// Make rows `..upto` readable.
    fn load(&mut self, upto: usize) {
        if let Rows::Ring {
            view,
            ring,
            row_elems,
            loaded,
        } = self
        {
            let slots = ring.len() / *row_elems;
            let rank = view.shape().len();
            let mut end = view.shape().to_vec();
            while *loaded < upto {
                let y = *loaded;
                let mut start = vec![0; rank];
                (start[0], end[0]) = (y, y + 1);
                let row = view.slice(&start, &end);
                let slot = &mut ring[(y % slots) * *row_elems..][..*row_elems];
                // SAFETY: `slot` holds exactly the row's elements, of the
                // row's dtype `F`, and is not the view's data.
                unsafe { Walk::of(&row).copy_to(slot.as_mut_ptr().cast()) };
                *loaded += 1;
            }
        }
    }

    /// Row `y`, loaded and among the last `ksize` loaded.
    #[inline(always)]
    fn row(&self, y: usize) -> &[F] {
        match self {
            Rows::Packed(rows) => rows[y],
            Rows::Ring {
                ring, row_elems, ..
            } => {
                let slot = y % (ring.len() / row_elems);
                &ring[slot * row_elems..][..*row_elems]
            }
        }
    }
}

pub(crate) fn convolve_in<F: Acc>(buf: &ViewBuffer, op: &ConvolveOp) -> ViewBuffer {
    // The input itself when already in the accumulator's dtype, else its
    // conversion, which walks a view once: read where it lies either way.
    let view = buf.cast(F::DTYPE);
    let shape = view.shape();

    let h = shape[0];
    let w = shape[1];
    let c = shape.get(2).copied().unwrap_or(1);

    // The coefficients in the accumulator's precision (an f32 kernel widens
    // exactly).
    let kernel: Vec<F> = op
        .kernel
        .iter()
        .map(|&k| F::from_f32(k).expect("an f32 coefficient converts"))
        .collect();
    let kernel = &kernel[..];
    let ksize = op.side();
    let half = ksize / 2;

    let norm_factor = if op.normalize {
        let abs_sum: F = kernel.iter().fold(F::zero(), |acc, k| acc + k.abs());
        if abs_sum > F::zero() {
            F::one() / abs_sum
        } else {
            F::one()
        }
    } else {
        F::one()
    };

    let mut output = vec![F::zero(); h * w * c];
    let out_shape = if c == 1 && shape.len() == 2 {
        vec![h, w]
    } else {
        vec![h, w, c]
    };
    if output.is_empty() {
        return ViewBuffer::from_vec_with_shape(output, out_shape);
    }
    let mut rows = Rows::of(&view, ksize);
    // An image no larger than the kernel is gathered everywhere; otherwise
    // the interior of each row is shifted-slice multiply-adds and its two
    // ends, and the top and bottom `half` rows, are gathered.
    let interior = h > 2 * half && w > 2 * half;
    let gather = |rows: &Rows<'_, F>, out: &mut [F], y, x0, x1| {
        convolve_gather_row(rows, out, h, w, c, op, kernel, norm_factor, y, x0, x1)
    };
    for y in 0..h {
        rows.load((y + half + 1).min(h));
        if interior && (half..h - half).contains(&y) {
            match ksize {
                3 => convolve_interior::<F, 3>(&rows, &mut output, y, w, c, kernel, norm_factor),
                5 => convolve_interior::<F, 5>(&rows, &mut output, y, w, c, kernel, norm_factor),
                7 => convolve_interior::<F, 7>(&rows, &mut output, y, w, c, kernel, norm_factor),
                _ => convolve_interior_dyn(&rows, &mut output, y, w, c, kernel, ksize, norm_factor),
            }
            gather(&rows, &mut output, y, 0, half);
            gather(&rows, &mut output, y, w - half, w);
        } else {
            gather(&rows, &mut output, y, 0, w);
        }
    }

    ViewBuffer::from_vec_with_shape(output, out_shape)
}

/// [`convolve_interior_dyn`] with the kernel side known at compile time.
#[allow(clippy::too_many_arguments)]
fn convolve_interior<F: Acc, const K: usize>(
    rows: &Rows<'_, F>,
    out: &mut [F],
    y: usize,
    w: usize,
    c: usize,
    kernel: &[F],
    norm_factor: F,
) {
    convolve_interior_dyn(rows, out, y, w, c, kernel, K, norm_factor);
}

/// The interior of output row `y` (columns `half..w - half`; requires
/// `half <= y < h - half` and `w > 2 * half`): per-tap shifted-slice
/// multiply-adds over packed row segments, then a final `norm_factor`
/// multiply.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn convolve_interior_dyn<F: Acc>(
    rows: &Rows<'_, F>,
    out: &mut [F],
    y: usize,
    w: usize,
    c: usize,
    kernel: &[F],
    ksize: usize,
    norm_factor: F,
) {
    let half = ksize / 2;
    let wc = w * c;
    let lo = half * c;
    let hi = (w - half) * c;
    let seg = hi - lo;

    let out_row = &mut out[y * wc + lo..y * wc + hi];
    for ky in 0..ksize {
        let src_row = rows.row(y + ky - half);
        for kx in 0..ksize {
            let kw = kernel[ky * ksize + kx];
            // Column shift of (kx - half) whole pixels within the row.
            let start = lo + kx * c - half * c;
            let src_seg = &src_row[start..start + seg];
            for (o, &v) in out_row.iter_mut().zip(src_seg) {
                *o += kw * v;
            }
        }
    }
    for o in out_row.iter_mut() {
        *o *= norm_factor;
    }
}

/// Per-pixel gather convolution over columns `x0..x1` of output row `y`,
/// with the border handling: the taps in `ky`-outer, `kx`-inner order, as
/// the interior visits them.
#[allow(clippy::too_many_arguments)]
fn convolve_gather_row<F: Acc>(
    rows: &Rows<'_, F>,
    out: &mut [F],
    h: usize,
    w: usize,
    c: usize,
    op: &ConvolveOp,
    kernel: &[F],
    norm_factor: F,
    y: usize,
    x0: usize,
    x1: usize,
) {
    let ksize = op.side();
    let half = (ksize / 2) as i64;

    for ch in 0..c {
        for x in x0..x1 {
            let mut sum = F::zero();
            for ky in 0..ksize {
                for kx in 0..ksize {
                    let sy = y as i64 + ky as i64 - half;
                    let sx = x as i64 + kx as i64 - half;

                    let pixel = sample_pixel(rows, h, w, c, ch, sy, sx, op.border);
                    sum += kernel[ky * ksize + kx] * pixel;
                }
            }
            out[(y * w + x) * c + ch] = sum * norm_factor;
        }
    }
}

/// Sample a pixel with border handling.
#[inline]
#[allow(clippy::too_many_arguments)]
fn sample_pixel<F: Acc>(
    rows: &Rows<'_, F>,
    h: usize,
    w: usize,
    c: usize,
    ch: usize,
    y: i64,
    x: i64,
    border: BorderMode,
) -> F {
    let (sy, sx) = match border {
        BorderMode::Zero => {
            if y < 0 || y >= h as i64 || x < 0 || x >= w as i64 {
                return F::zero();
            }
            (y as usize, x as usize)
        }
        BorderMode::Replicate => {
            let sy = y.clamp(0, h as i64 - 1) as usize;
            let sx = x.clamp(0, w as i64 - 1) as usize;
            (sy, sx)
        }
        BorderMode::Reflect => {
            let sy = reflect_index(y, h);
            let sx = reflect_index(x, w);
            (sy, sx)
        }
    };
    rows.row(sy)[sx * c + ch]
}

/// Reflect an index about the edge: dcba|abcd|dcba
#[inline]
fn reflect_index(idx: i64, size: usize) -> usize {
    if idx < 0 {
        (-idx - 1).min(size as i64 - 1) as usize
    } else if idx >= size as i64 {
        let reflected = 2 * size as i64 - idx - 1;
        reflected.max(0) as usize
    } else {
        idx as usize
    }
}
