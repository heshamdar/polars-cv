//! Morphology: erode, dilate and the gradient, over a square or a disk.

use super::{get_channel_count, is_single_channel};
use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{with_dtype, DType};
use crate::ops::Footprint;

// ============================================================
// Morphological Operations (Erode / Dilate / Gradient)
// ============================================================

/// Morphological erosion: output = local minimum over ksize×ksize neighborhood.
///
/// Uses replicate border handling (edge pixels are replicated).
/// Operates on single-channel data only; panics for multi-channel input.
/// Dtype-generic: dispatches per element type.
pub(crate) fn apply_erode(
    buf: ViewBuffer,
    ksize: u32,
    iterations: u32,
    footprint: Footprint,
) -> ViewBuffer {
    let shape = buf.shape();
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "Erode requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() or .threshold() first."
        );
    }
    morph_iterated(buf, ksize, iterations, MorphKind::Min, footprint)
}

/// Morphological dilation: output = local maximum over ksize×ksize neighborhood.
///
/// Uses replicate border handling (edge pixels are replicated).
/// Operates on single-channel data only; panics for multi-channel input.
/// Dtype-generic: dispatches per element type.
pub(crate) fn apply_dilate(
    buf: ViewBuffer,
    ksize: u32,
    iterations: u32,
    footprint: Footprint,
) -> ViewBuffer {
    let shape = buf.shape();
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "Dilate requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() or .threshold() first."
        );
    }
    morph_iterated(buf, ksize, iterations, MorphKind::Max, footprint)
}

/// `iterations` min (or max) passes of a `ksize` window.
///
/// For an integer dtype, **one** pass of radius `iterations * (ksize / 2)`.
/// Growing a window by radius `r` and then by `r` again reaches every pixel
/// within `2r` and nothing further, and the min of mins is the min over the
/// union. That holds at the border too — replicating the edge clamps each
/// step's index, and a clamp of a clamped interval is the clamp of the wider
/// interval — and for the separable passes, since row and column minima
/// commute. The taps are about the same in number; the passes over the whole
/// image are what it saves.
///
/// A float dtype keeps the passes: its fold keeps a NaN incumbent, ignores a
/// NaN candidate and keeps the incumbent of two equal signed zeros, so its
/// result depends on the passes' structure, and one wide pass differs
/// (`tests/it/morph_ref.rs`, NaN and signed-zero cases).
///
/// A disk keeps the passes for every dtype: two discrete disks do not compose
/// into one larger disk (the radius-2 disk twice reaches `(4, 0)`, `(3, 1)`
/// and `(2, 2)` but not the radius-4 disk's `(3, 2)`), so `iterations` means
/// that many disk passes.
fn morph_iterated(
    buf: ViewBuffer,
    ksize: u32,
    iterations: u32,
    kind: MorphKind,
    footprint: Footprint,
) -> ViewBuffer {
    let radius = (ksize / 2) as usize;
    if footprint == Footprint::Disk || matches!(buf.dtype(), DType::F32 | DType::F64) {
        let mut result = buf;
        for _ in 0..iterations {
            result = morph_pass(&result, radius, kind, footprint);
        }
        return result;
    }
    morph_pass(&buf, radius * iterations as usize, kind, footprint)
}

/// Morphological gradient: dilate(input) − erode(input), clamped to valid range.
///
/// Produces an edge outline by computing the difference between dilation and
/// erosion. Output dtype matches input.
pub(crate) fn apply_morph_gradient(
    buf: ViewBuffer,
    ksize: u32,
    footprint: Footprint,
) -> ViewBuffer {
    let shape = buf.shape();
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "MorphGradient requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() or .threshold() first."
        );
    }
    let radius = (ksize / 2) as usize;
    let dilated = morph_pass(&buf, radius, MorphKind::Max, footprint);
    let eroded = morph_pass(&buf, radius, MorphKind::Min, footprint);

    morph_subtract(&dilated, &eroded)
}

#[derive(Clone, Copy)]
enum MorphKind {
    Min,
    Max,
}

/// Single-pass min or max filter over a footprint of radius `radius`: the
/// `(2 * radius + 1)`-square, or the disk of that radius.
///
/// Border handling: replicate edge pixels.
fn morph_pass(
    buf: &ViewBuffer,
    radius: usize,
    kind: MorphKind,
    footprint: Footprint,
) -> ViewBuffer {
    if radius == 0 {
        return buf.clone();
    }
    with_dtype!(buf.dtype(), T => match (kind, footprint) {
        (MorphKind::Min, Footprint::Square) => morph_minmax_typed::<T, true>(buf, radius),
        (MorphKind::Max, Footprint::Square) => morph_minmax_typed::<T, false>(buf, radius),
        (MorphKind::Min, Footprint::Disk) => morph_disk_typed::<T, true>(buf, radius),
        (MorphKind::Max, Footprint::Disk) => morph_disk_typed::<T, false>(buf, radius),
    })
}

/// One Min/Max fold step with the comparison kind monomorphized out of the
/// loop.
///
/// The window's running extreme: [`crate::ops::util`]'s NaN rule, so a NaN
/// anywhere in the window is the result (it used to be kept as the centre
/// but skipped as a neighbour). A tie keeps the incumbent `val`.
#[inline(always)]
fn morph_select<T: PartialOrd + Copy, const IS_MIN: bool>(val: T, candidate: T) -> T {
    if IS_MIN {
        crate::ops::util::minimum(val, candidate)
    } else {
        crate::ops::util::maximum(val, candidate)
    }
}

/// Typed separable min/max filter: row pass then column pass.
///
/// Structured for auto-vectorization (interior/border split, same pattern as
/// the separable Gaussian blur above):
/// - row pass: the output row is initialized from the source row (the
///   original code's `val = center` seed), then each tap `kx ∈ [-r, r]` is a
///   contiguous shifted-slice elementwise min/max over the in-bounds segment,
///   with the clamped (replicate) border lanes folding the constant edge
///   pixel — identical comparison order to the original per-element loop;
/// - column pass: whole-row elementwise min/max against the clamped source
///   row per tap (the index clamp happens once per row, not per element).
///
/// Bit-exact vs the pre-split implementation (see tests/it/morph_ref.rs).
fn morph_minmax_typed<T, const IS_MIN: bool>(buf: &ViewBuffer, radius: usize) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + Copy + PartialOrd,
{
    // Read in place, at the view's own row stride (a crop, a vertical flip).
    let rows_buf = buf.to_dense_rows();
    let shape = rows_buf.shape();
    let h = shape[0];
    let w = shape[1];
    let src_rows: Vec<&[T]> = rows_buf
        .dense_rows::<T>()
        .expect("a rank-2 or rank-3 buffer has packed rows after to_dense_rows");

    let row_out = row_extreme::<T, IS_MIN>(&src_rows, w, radius);

    // ── Column pass: whole-row folds with a per-row index clamp ──────────
    // Seeded from the row-pass output (the original code's `val = center`),
    // then folded in ascending ky order; src rows live in `row_out`, the
    // destination in `col_out`, so the borrows are disjoint.
    let mut col_out: Vec<T> = row_out.clone();
    for y in 0..h {
        let dst_row = &mut col_out[y * w..(y + 1) * w];
        for ky in -(radius as i64)..=radius as i64 {
            let sy = (y as i64 + ky).clamp(0, h as i64 - 1) as usize;
            let src_row = &row_out[sy * w..(sy + 1) * w];
            for (dst, &cand) in dst_row.iter_mut().zip(src_row) {
                *dst = morph_select::<T, IS_MIN>(*dst, cand);
            }
        }
    }

    ViewBuffer::from_vec_with_shape(col_out, shape.to_vec())
}

/// The row pass of a min/max filter: each row's extreme over `x ± radius`,
/// edges replicated, folded in ascending tap order from the centre pixel.
/// The square's first pass.
fn row_extreme<T, const IS_MIN: bool>(src_rows: &[&[T]], w: usize, radius: usize) -> Vec<T>
where
    T: crate::core::dtype::ViewType + Default + Copy + PartialOrd,
{
    let h = src_rows.len();
    let mut row_out: Vec<T> = vec![T::default(); h * w];
    for y in 0..h {
        let row_in = src_rows[y];
        let row_dst = &mut row_out[y * w..(y + 1) * w];

        if w <= 2 * radius {
            // Row narrower than the kernel: clamped gather everywhere.
            for (x, dst) in row_dst.iter_mut().enumerate() {
                let mut val = row_in[x];
                for kx in -(radius as i64)..=radius as i64 {
                    let sx = (x as i64 + kx).clamp(0, w as i64 - 1) as usize;
                    val = morph_select::<T, IS_MIN>(val, row_in[sx]);
                }
                *dst = val;
            }
            continue;
        }

        // Seed with the center pixel, then fold taps in ascending kx order.
        row_dst.copy_from_slice(row_in);
        for kx in -(radius as i64)..=radius as i64 {
            // In-bounds segment for this tap: x ∈ [lo, hi) ⇒ x+kx ∈ [0, w).
            let lo = (-kx).max(0) as usize;
            let hi = w - kx.max(0) as usize;
            let src_seg = &row_in[(lo as i64 + kx) as usize..(hi as i64 + kx) as usize];
            for (dst, &cand) in row_dst[lo..hi].iter_mut().zip(src_seg) {
                *dst = morph_select::<T, IS_MIN>(*dst, cand);
            }
            // Border lanes: the clamped index degenerates to the edge pixel.
            let left_edge = row_in[0];
            for dst in row_dst[..lo].iter_mut() {
                *dst = morph_select::<T, IS_MIN>(*dst, left_edge);
            }
            let right_edge = row_in[w - 1];
            for dst in row_dst[hi..].iter_mut() {
                *dst = morph_select::<T, IS_MIN>(*dst, right_edge);
            }
        }
    }
    row_out
}

/// Typed min/max filter over the disk of radius `radius`: the pixels within
/// Euclidean distance `radius` of the centre, edges replicated.
///
/// The disk is a stack of horizontal spans, row offset `dy` spanning
/// `±floor(sqrt(radius² − dy²))`. Its result is fixed by the fold order the
/// square uses (`tests/it/morph_ref.rs` pins it bit for bit, which decides
/// which of two equal signed zeros, or NaNs, survives): each span from its
/// centre pixel then its taps ascending, the spans from the centre row's
/// full span then ascending `dy`. `morph_select` keeps the first extreme met,
/// which is associative though not commutative, so the fold may be
/// regrouped as long as the order is kept. That buys O(r) work per pixel
/// instead of O(r²), with five images live instead of one per width:
///
/// - a span of half-width `k` is `pick(pick(x − k, span_{k−1}), x + k)`, so
///   the half-widths grow one step at a time ([`DiskSpans`]);
/// - the column fold runs `dy` outermost: above the centre (`dy = −r..=0`)
///   the half-widths only grow, in fold order; below it they shrink, so
///   that half is folded from `dy = r` down to 1 by *prepending* each term,
///   which keeps the order while the half-widths grow again;
/// - the result is the seed, then the upper half, then the lower half.
fn morph_disk_typed<T, const IS_MIN: bool>(buf: &ViewBuffer, radius: usize) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + Copy + PartialOrd,
{
    let rows_buf = buf.to_dense_rows();
    let shape = rows_buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let src_rows: Vec<&[T]> = rows_buf
        .dense_rows::<T>()
        .expect("a rank-2 or rank-3 buffer has packed rows after to_dense_rows");
    if h == 0 || w == 0 {
        // No pixel to fold; and a zero-width row has no chunks to walk.
        return rows_buf.to_contiguous();
    }
    let half = |d: usize| (radius * radius - d * d).isqrt();
    let select = morph_select::<T, IS_MIN>;

    // Above the centre, in fold order: `upper[y] = fold(t(-r), …, t(0))`,
    // `t(dy)` the span of half-width `half(|dy|)` on row `y + dy`.
    let mut spans = DiskSpans::<T, IS_MIN>::new(&src_rows, w);
    let mut upper: Vec<T> = vec![T::default(); h * w];
    for d in (0..=radius).rev() {
        spans.grow_to(half(d));
        for (y, dst) in upper.chunks_exact_mut(w).enumerate() {
            let sy = y.saturating_sub(d);
            let terms = src_rows[sy].iter().zip(spans.window(sy));
            for (u, (&c, &v)) in dst.iter_mut().zip(terms) {
                let t = select(c, v);
                *u = if d == radius { t } else { select(*u, t) };
            }
        }
    }
    // The seed, the centre row's full span (what `spans` now holds), then
    // the upper half.
    let mut out: Vec<T> = Vec::with_capacity(h * w);
    for (y, upper_row) in upper.chunks_exact(w).enumerate() {
        let seeds = src_rows[y].iter().zip(spans.window(y));
        out.extend(
            seeds
                .zip(upper_row)
                .map(|((&c, &v), &u)| select(select(c, v), u)),
        );
    }

    // Below the centre, folded from `dy = r` down to 1 by prepending: the
    // lower half is `fold(t(1), …, t(r))`, built as `t(d) · (…)`.
    let mut spans = DiskSpans::<T, IS_MIN>::new(&src_rows, w);
    let mut lower = upper;
    for d in (1..=radius).rev() {
        spans.grow_to(half(d));
        for (y, dst) in lower.chunks_exact_mut(w).enumerate() {
            let sy = (y + d).min(h - 1);
            let terms = src_rows[sy].iter().zip(spans.window(sy));
            for (l, (&c, &v)) in dst.iter_mut().zip(terms) {
                let t = select(c, v);
                *l = if d == radius { t } else { select(t, *l) };
            }
        }
    }
    if radius > 0 {
        for (o, &l) in out.iter_mut().zip(&lower) {
            *o = select(*o, l);
        }
    }
    ViewBuffer::from_vec_with_shape(out, shape.to_vec())
}

/// Every row's spans of one half-width `k`, grown a step at a time:
/// `window[x]` folds the clamped taps `x − k..=x + k` in ascending order,
/// and a span is its centre pixel then that window (`row_extreme`'s order).
struct DiskSpans<'a, T, const IS_MIN: bool> {
    src_rows: &'a [&'a [T]],
    w: usize,
    k: usize,
    window: Vec<T>,
}

impl<'a, T, const IS_MIN: bool> DiskSpans<'a, T, IS_MIN>
where
    T: crate::core::dtype::ViewType + Default + Copy + PartialOrd,
{
    /// Half-width 0: each window is its own pixel.
    fn new(src_rows: &'a [&'a [T]], w: usize) -> Self {
        DiskSpans {
            src_rows,
            w,
            k: 0,
            window: src_rows.concat(),
        }
    }

    /// Widen every window to half-width `k`, one tap on each side per step:
    /// `pick(pick(x − k, window), x + k)` keeps the ascending tap order.
    fn grow_to(&mut self, k: usize) {
        debug_assert!(k >= self.k, "the half-widths only grow");
        let w = self.w;
        while self.k < k {
            self.k += 1;
            let k = self.k;
            let step = |left: T, v: &mut T, right: T| {
                *v = morph_select::<T, IS_MIN>(morph_select::<T, IS_MIN>(left, *v), right);
            };
            for (src, win) in self.src_rows.iter().zip(self.window.chunks_exact_mut(w)) {
                if w > 2 * k {
                    // Interior: both taps in bounds, three aligned slices (so
                    // the loop vectorizes, as the square's row pass does).
                    let lefts = &src[..w - 2 * k];
                    let rights = &src[2 * k..];
                    for ((&l, v), &r) in lefts.iter().zip(&mut win[k..w - k]).zip(rights) {
                        step(l, v, r);
                    }
                }
                // Border lanes: a tap past an edge reads the edge pixel.
                for x in (0..k.min(w)).chain(w.saturating_sub(k).max(k.min(w))..w) {
                    step(
                        src[x.saturating_sub(k)],
                        &mut win[x],
                        src[(x + k).min(w - 1)],
                    );
                }
            }
        }
    }

    /// Row `y`'s windows at the current half-width; a span is
    /// `pick(centre, window)`.
    fn window(&self, y: usize) -> &[T] {
        &self.window[y * self.w..(y + 1) * self.w]
    }
}

/// Element-wise saturating subtraction: result = a − b, clamped to valid range.
fn morph_subtract(a: &ViewBuffer, b: &ViewBuffer) -> ViewBuffer {
    let ca = a.to_contiguous();
    let cb = b.to_contiguous();
    let shape = crate::core::layout::Dims::from_slice(ca.shape());
    with_dtype!(a.dtype(), T => {
        let out: Vec<T> = ca
            .as_slice::<T>()
            .iter()
            .zip(cb.as_slice::<T>())
            .map(|(&a, &b)| a.sub_floor_zero(b))
            .collect();
        ViewBuffer::from_vec_with_shape(out, shape)
    })
}

/// `max(a - b, 0)`, saturating at the dtype's maximum, in the dtype itself:
/// the morphological gradient's dilate − erode (never negative, but an i8
/// 127 − (−128) exceeds i8).
trait SubFloorZero: Copy {
    fn sub_floor_zero(self, other: Self) -> Self;
}

macro_rules! sub_floor_zero_int {
    ($($t:ty),+) => {$(
        impl SubFloorZero for $t {
            #[inline(always)]
            fn sub_floor_zero(self, other: Self) -> Self {
                self.saturating_sub(other).max(0)
            }
        }
    )+};
}
sub_floor_zero_int!(u8, i8, u16, i16, u32, i32, u64, i64);

macro_rules! sub_floor_zero_float {
    ($($t:ty),+) => {$(
        impl SubFloorZero for $t {
            #[inline(always)]
            fn sub_floor_zero(self, other: Self) -> Self {
                // NaN stays NaN (`f32::max` would make it 0).
                crate::ops::util::maximum(self - other, 0.0)
            }
        }
    )+};
}
sub_floor_zero_float!(f32, f64);
