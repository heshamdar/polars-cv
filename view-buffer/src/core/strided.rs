//! Reading a view's elements in logical order: **the one walk over a view's
//! memory.**
//!
//! A view (a crop, a flip, a transpose, any mix) is an offset and a byte
//! stride per axis. Walking it one element at a time recomputes an N-d offset
//! per element; [`Walk::of`] instead coalesces the layout once into the
//! largest runs of packed bytes it has and the fewest loops over them:
//!
//! - **unit**: the innermost axes whose elements are packed, merged into one
//!   run of bytes (a whole contiguous buffer is a single unit);
//! - **row**: the next axis, `row_len` units `step` bytes apart, merged
//!   outward while the axes are evenly spaced (a flip's negative step
//!   included);
//! - **outer**: whatever remains, walked by an odometer that moves the row
//!   pointer by adding strides (no per-row multiplication).
//!
//! Every reader of a view's elements goes through it: packing
//! ([`ViewBuffer::to_contiguous`], `append_to`/`write_to` and so every
//! list/array sink), and converting a view's elements to another dtype
//! (`cast`, the element-wise engine's f32 read), through
//! `convert::convert_view`.

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, ViewType};

/// A view's layout coalesced into units, rows and outer axes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Geometry {
    /// Bytes in one packed run.
    pub(crate) unit: usize,
    /// Units per row.
    pub(crate) row_len: usize,
    /// Bytes from one unit of a row to the next (any sign).
    pub(crate) step: isize,
    /// The remaining axes, outermost first, as `(extent, stride in bytes)`.
    pub(crate) outer: Vec<(usize, isize)>,
}

impl Geometry {
    /// Coalesce `shape`/`strides` (bytes) of `elem`-byte elements.
    fn of(shape: &[usize], strides: &[isize], elem: usize) -> Self {
        // Innermost first. A size-1 axis is never stepped along, so its
        // stride says nothing.
        let mut dims = shape
            .iter()
            .copied()
            .zip(strides.iter().copied())
            .filter(|&(n, _)| n != 1)
            .rev()
            .peekable();
        // The unit: packed innermost axes.
        let mut unit = elem;
        while let Some((n, _)) = dims.next_if(|&(_, stride)| stride == unit as isize) {
            unit *= n;
        }
        // The row: the next axis, extended outward while evenly spaced.
        let (mut row_len, mut step) = (1usize, 0isize);
        if let Some(axis) = dims.next() {
            (row_len, step) = axis;
            while let Some((n, _)) = dims.next_if(|&(_, stride)| stride == step * row_len as isize)
            {
                row_len *= n;
            }
        }
        // The outer axes, each merged into the one inside it when evenly
        // spaced; outermost first.
        // Sized exactly: a packing copy makes no allocation near a small
        // view's own size (`copy_counts::append_to_copies_each_element_once`).
        let mut outer: Vec<(usize, isize)> = Vec::with_capacity(dims.clone().count());
        for (n, stride) in dims {
            match outer.last_mut() {
                Some(inner) if stride == inner.1 * inner.0 as isize => inner.0 *= n,
                _ => outer.push((n, stride)),
            }
        }
        outer.reverse();
        Geometry {
            unit,
            row_len,
            step,
            outer,
        }
    }
}

#[cfg(debug_assertions)]
impl Geometry {
    /// Panic unless every byte the walk reads from `offset` lies in
    /// `0..data_len`: a coalescing fault fails here, not as a wild read.
    fn assert_inside(&self, offset: usize, data_len: usize) {
        let axes = self
            .outer
            .iter()
            .copied()
            .chain(std::iter::once((self.row_len, self.step)));
        let (mut low, mut high) = (offset as isize, offset as isize);
        for (n, stride) in axes {
            let reach = (n as isize - 1) * stride;
            if reach < 0 {
                low += reach;
            } else {
                high += reach;
            }
        }
        assert!(
            low >= 0 && high + self.unit as isize <= data_len as isize,
            "internal: the walk {self:?} from byte {offset} reads bytes {low}..{} of {data_len}",
            high + self.unit as isize
        );
    }
}

/// A view's elements in logical (row-major) order, read where they lie.
pub(crate) struct Walk<'a> {
    /// The first logical element.
    base: *const u8,
    geometry: Geometry,
    dtype: DType,
    /// No element: nothing to walk.
    empty: bool,
    _view: PhantomData<&'a ViewBuffer>,
}

/// Bytes of the stack scratch [`Walk::for_each_run`] packs short units into:
/// 8 KiB, inside L1.
const SCRATCH_BYTES: usize = 8192;

/// A unit at least this many elements long is handed out where it lies;
/// shorter ones are packed into the scratch first, so a consumer never sees
/// runs of a pixel or two.
const DIRECT_ELEMS: usize = 64;

impl<'a> Walk<'a> {
    /// The walk over `buf`'s elements.
    pub(crate) fn of(buf: &'a ViewBuffer) -> Self {
        let layout = &buf.layout;
        let empty = layout.shape.contains(&0);
        // SAFETY: the offset is inside the data (layouts are validated when
        // built); for an empty view it is never dereferenced.
        let base = unsafe { buf.data.as_ptr().add(layout.offset) };
        let geometry = Geometry::of(&layout.shape, &layout.strides, layout.dtype.size_of());
        #[cfg(debug_assertions)]
        if !empty {
            geometry.assert_inside(layout.offset, buf.data.len());
        }
        Walk {
            base,
            geometry,
            dtype: layout.dtype,
            empty,
            _view: PhantomData,
        }
    }

    #[cfg(test)]
    pub(crate) fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// Call `f` with the first unit of every row, in logical order.
    #[inline(always)]
    fn for_each_row(&self, mut f: impl FnMut(*const u8)) {
        if self.empty {
            return;
        }
        let outer = &self.geometry.outer;
        match outer.as_slice() {
            [] => f(self.base),
            &[(n, stride)] => {
                let mut row = self.base;
                for _ in 0..n {
                    f(row);
                    row = row.wrapping_offset(stride);
                }
            }
            _ => {
                let rows: usize = outer.iter().map(|&(n, _)| n).product();
                let mut index = vec![0usize; outer.len()];
                let mut row = self.base;
                for _ in 0..rows {
                    f(row);
                    // The odometer: step the innermost outer axis, carrying.
                    for (i, &(n, stride)) in outer.iter().enumerate().rev() {
                        index[i] += 1;
                        row = row.wrapping_offset(stride);
                        if index[i] < n {
                            break;
                        }
                        index[i] = 0;
                        row = row.wrapping_offset(-stride * n as isize);
                    }
                }
            }
        }
    }

    /// Copy every element, packed row-major, to `dst`.
    ///
    /// # Safety
    /// `dst` must be valid for writes of the view's element count × element
    /// size bytes and must not overlap the view's data.
    pub(crate) unsafe fn copy_to(&self, dst: *mut u8) {
        let Geometry {
            unit,
            row_len,
            step,
            ..
        } = self.geometry;
        let row_bytes = unit * row_len;
        let mut out = dst;
        // One match per call, not per row: each arm is its own loop.
        macro_rules! rows {
            ($copy:expr) => {
                self.for_each_row(|row| {
                    // SAFETY: `row` is a row of the view, whose `row_len`
                    // units lie inside its data; `out` advances by one
                    // packed row per row, inside what the caller provides.
                    unsafe {
                        $copy(row, out, row_len, step);
                        out = out.add(row_bytes);
                    }
                })
            };
        }
        match unit {
            1 => rows!(copy_units::<1>),
            2 => rows!(copy_units::<2>),
            3 => rows!(copy_units::<3>),
            4 => rows!(copy_units::<4>),
            6 => rows!(copy_units::<6>),
            8 => rows!(copy_units::<8>),
            12 => rows!(copy_units::<12>),
            16 => rows!(copy_units::<16>),
            _ => rows!(|src, dst, n, step| copy_any(src, dst, n, step, unit)),
        }
    }

    /// Call `f` with runs of `T` whose concatenation is every element in
    /// logical order. Long units are handed out where they lie; short ones
    /// are packed into a stack scratch first.
    ///
    /// # Panics
    /// Panics if `T` is not the view's dtype.
    #[inline(always)]
    pub(crate) fn for_each_run<T: ViewType>(&self, mut f: impl FnMut(&[T])) {
        assert_eq!(
            T::DTYPE,
            self.dtype,
            "for_each_run: asked for {:?} runs of a {:?} view",
            T::DTYPE,
            self.dtype
        );
        if self.empty {
            return;
        }
        let Geometry {
            unit,
            row_len,
            step,
            ..
        } = self.geometry;
        let elem = std::mem::size_of::<T>();
        let unit_elems = unit / elem;
        if unit_elems >= DIRECT_ELEMS {
            self.for_each_row(|row| {
                for k in 0..row_len {
                    // SAFETY: unit `k` of a row is `unit_elems` packed
                    // elements inside the view's data, aligned for `T`
                    // (offsets and strides are whole elements, CR-41).
                    f(unsafe {
                        std::slice::from_raw_parts(
                            row.wrapping_offset(k as isize * step).cast::<T>(),
                            unit_elems,
                        )
                    });
                }
            });
            return;
        }
        // Pack short units into an aligned stack scratch, handing it out
        // whenever it fills.
        let mut scratch = [MaybeUninit::<u64>::uninit(); SCRATCH_BYTES / 8];
        let scratch = scratch.as_mut_ptr().cast::<u8>();
        let per_block = SCRATCH_BYTES / unit;
        let mut filled = 0usize;
        let flush = |filled: usize, f: &mut dyn FnMut(&[T])| {
            // SAFETY: the first `filled` units of the scratch were written
            // with whole elements of `T`, and the scratch is 8-byte aligned.
            f(unsafe { std::slice::from_raw_parts(scratch.cast::<T>(), filled * unit_elems) });
        };
        self.for_each_row(|row| {
            let mut done = 0;
            while done < row_len {
                let take = (row_len - done).min(per_block - filled);
                // SAFETY: units `done..done + take` of the row lie inside the
                // view's data, and `filled + take <= per_block` units fit the
                // scratch.
                unsafe {
                    copy_any(
                        row.wrapping_offset(done as isize * step),
                        scratch.add(filled * unit),
                        take,
                        step,
                        unit,
                    );
                }
                filled += take;
                done += take;
                if filled == per_block {
                    flush(filled, &mut f);
                    filled = 0;
                }
            }
        });
        if filled > 0 {
            flush(filled, &mut f);
        }
    }
}

/// Copy `n` units of `N` bytes, `step` bytes apart from `src`, packed to
/// `dst`: a constant-size copy is a load and a store, not a `memcpy` call.
///
/// # Safety
/// The units must be readable and `dst` writable for `n * N` bytes, not
/// overlapping them.
#[inline(always)]
unsafe fn copy_units<const N: usize>(src: *const u8, dst: *mut u8, n: usize, step: isize) {
    if step == N as isize {
        // SAFETY: the caller's contract, the units being adjacent.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, n * N) };
        return;
    }
    for k in 0..n {
        // SAFETY: the caller's contract, unit by unit.
        unsafe {
            std::ptr::copy_nonoverlapping(src.wrapping_offset(k as isize * step), dst.add(k * N), N)
        };
    }
}

/// [`copy_units`] for a unit size known only at run time.
///
/// # Safety
/// As [`copy_units`], with `unit` in place of `N`.
#[inline(always)]
unsafe fn copy_any(src: *const u8, dst: *mut u8, n: usize, step: isize, unit: usize) {
    // SAFETY (every arm): the caller's contract.
    unsafe {
        match unit {
            1 => copy_units::<1>(src, dst, n, step),
            2 => copy_units::<2>(src, dst, n, step),
            3 => copy_units::<3>(src, dst, n, step),
            4 => copy_units::<4>(src, dst, n, step),
            6 => copy_units::<6>(src, dst, n, step),
            8 => copy_units::<8>(src, dst, n, step),
            12 => copy_units::<12>(src, dst, n, step),
            16 => copy_units::<16>(src, dst, n, step),
            _ => {
                for k in 0..n {
                    std::ptr::copy_nonoverlapping(
                        src.wrapping_offset(k as isize * step),
                        dst.add(k * unit),
                        unit,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dtype::with_dtype;

    fn geometry(buf: &ViewBuffer) -> Geometry {
        Walk::of(buf).geometry().clone()
    }

    fn g(unit: usize, row_len: usize, step: isize, outer: &[(usize, isize)]) -> Geometry {
        Geometry {
            unit,
            row_len,
            step,
            outer: outer.to_vec(),
        }
    }

    /// The coalescing table: each view's runs are as long, and its loops as
    /// few, as its layout allows.
    #[test]
    fn views_coalesce_into_the_longest_runs_their_layout_has() {
        let u8_img = ViewBuffer::from_vec_with_shape(vec![0u8; 4 * 5 * 3], vec![4, 5, 3]);
        let f32_img = ViewBuffer::from_vec_with_shape(vec![0f32; 4 * 5 * 3], vec![4, 5, 3]);
        let cases: Vec<(&str, ViewBuffer, Geometry)> = vec![
            // One memcpy.
            ("contiguous u8", u8_img.clone(), g(60, 1, 0, &[])),
            ("contiguous f32", f32_img.clone(), g(240, 1, 0, &[])),
            // Rows of whole pixels: each cropped row is one unit.
            (
                "crop",
                u8_img.slice(&[1, 1, 0], &[3, 4, 3]),
                g(9, 2, 15, &[]),
            ),
            // A vertical flip: whole rows, walked backwards.
            ("flip_v", u8_img.flip(&[0]), g(15, 4, -15, &[])),
            // A horizontal flip: pixels walked backwards within each row.
            ("flip_h", u8_img.flip(&[1]), g(3, 5, -3, &[(4, 15)])),
            ("flip_h f32", f32_img.flip(&[1]), g(12, 5, -12, &[(4, 60)])),
            // Both flips: every pixel backwards, as one row.
            ("flip_hv", u8_img.flip(&[0, 1]), g(3, 20, -3, &[])),
            // A transpose: pixels a whole source row apart.
            (
                "transpose",
                u8_img.permute(&[1, 0, 2]),
                g(3, 4, 15, &[(5, 3)]),
            ),
            // A channel flip: single elements, each pixel a reversed row of
            // three; the pixels themselves are evenly spaced, one outer axis.
            ("flip_c", u8_img.flip(&[2]), g(1, 3, -1, &[(20, 3)])),
            // Size-1 axes carry no stride information and are dropped.
            (
                "one row",
                u8_img.slice(&[2, 0, 0], &[3, 5, 3]).flip(&[1]),
                g(3, 5, -3, &[]),
            ),
            // Outer axes that are evenly spaced merge.
            (
                "channel planes",
                ViewBuffer::from_vec_with_shape(vec![0u8; 2 * 3 * 4 * 5], vec![2, 3, 4, 5])
                    .flip(&[3]),
                g(1, 5, -1, &[(24, 5)]),
            ),
        ];
        for (name, view, want) in cases {
            assert_eq!(geometry(&view), want, "{name}");
        }
    }

    /// Deterministic pseudo-random `u64`s.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 17
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// A random view: a rank 1–4 buffer of distinct elements, then a few
    /// random permutes, flips and slices.
    fn random_view(rng: &mut Lcg, dtype: DType) -> ViewBuffer {
        let rank = 1 + rng.below(4);
        let shape: Vec<usize> = (0..rank).map(|_| 1 + rng.below(7)).collect();
        let n: usize = shape.iter().product();
        let mut view = with_dtype!(dtype, T => ViewBuffer::from_vec_with_shape(
            (0..n).map(|i| i as T).collect::<Vec<T>>(),
            shape.clone(),
        ));
        for _ in 0..rng.below(5) {
            let rank = view.shape().len();
            match rng.below(3) {
                0 => {
                    let mut dims: Vec<usize> = (0..rank).collect();
                    for i in (1..rank).rev() {
                        dims.swap(i, rng.below(i + 1));
                    }
                    view = view.permute(&dims);
                }
                1 => {
                    let axes: Vec<usize> = (0..rank).filter(|_| rng.below(2) == 0).collect();
                    view = view.flip(&axes);
                }
                _ => {
                    let s = view.shape().to_vec();
                    let start: Vec<usize> = s.iter().map(|&d| rng.below(d)).collect();
                    let end: Vec<usize> = s
                        .iter()
                        .zip(&start)
                        .map(|(&d, &a)| a + 1 + rng.below(d - a))
                        .collect();
                    view = view.slice(&start, &end);
                }
            }
        }
        view
    }

    /// The reference: every element's bytes, read one at a time at
    /// `offset + Σ index · stride`.
    fn naive_bytes(view: &ViewBuffer) -> Vec<u8> {
        let shape = view.shape();
        let strides = view.strides_bytes();
        let elem = view.dtype().size_of();
        let data = view.data.as_ptr();
        let mut out = Vec::new();
        let n: usize = shape.iter().product();
        let mut idx = vec![0usize; shape.len()];
        for _ in 0..n {
            let offset = view.layout.offset as isize
                + idx
                    .iter()
                    .zip(strides)
                    .map(|(&i, &s)| i as isize * s)
                    .sum::<isize>();
            for b in 0..elem {
                // SAFETY: every element of the view lies inside its data.
                out.push(unsafe { *data.offset(offset + b as isize) });
            }
            for d in (0..shape.len()).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    }

    fn walk_bytes(view: &ViewBuffer) -> Vec<u8> {
        let len = view.shape().iter().product::<usize>() * view.dtype().size_of();
        let mut out = Vec::<u8>::with_capacity(len);
        // SAFETY: `out` has room for every element's bytes.
        unsafe {
            Walk::of(view).copy_to(out.as_mut_ptr());
            out.set_len(len);
        }
        out
    }

    fn run_bytes(view: &ViewBuffer) -> Vec<u8> {
        let mut out = Vec::new();
        with_dtype!(view.dtype(), T => Walk::of(view).for_each_run::<T>(|run: &[T]| {
            for x in run {
                out.extend_from_slice(&x.to_ne_bytes());
            }
        }));
        out
    }

    /// Every way of reading a view agrees with the per-element reference,
    /// over random views of every rank the engine sees.
    #[test]
    fn the_walk_reads_every_random_view_in_logical_order() {
        let mut rng = Lcg(0x5EED_CAFE);
        for dtype in [DType::U8, DType::U16, DType::F32, DType::F64] {
            for case in 0..400 {
                let view = random_view(&mut rng, dtype);
                let label = format!(
                    "{dtype:?} case {case}: shape {:?} strides {:?}",
                    view.shape(),
                    view.strides_bytes()
                );
                let want = naive_bytes(&view);
                assert_eq!(walk_bytes(&view), want, "copy_to, {label}");
                assert_eq!(run_bytes(&view), want, "for_each_run, {label}");
                let packed = view.to_contiguous();
                assert_eq!(naive_bytes(&packed), want, "to_contiguous, {label}");
                for target in [DType::F32, DType::U8, DType::I16] {
                    let cast = view.cast_to(target);
                    let via_packed = with_dtype!(dtype, S => with_dtype!(target, D => {
                        crate::core::convert::convert_slice::<S, D>(packed.as_slice::<S>())
                            .iter()
                            .flat_map(|x| x.to_ne_bytes())
                            .collect::<Vec<u8>>()
                    }));
                    assert_eq!(
                        naive_bytes(&cast),
                        via_packed,
                        "cast to {target:?}, {label}"
                    );
                }
            }
        }
    }

    /// Long units are handed out in place, short ones packed: both orders
    /// are logical, across scratch refills.
    #[test]
    fn runs_cross_the_scratch_boundary_in_order() {
        for (h, w, c) in [(3, 5000, 3), (2, 70, 64), (4000, 3, 1)] {
            let buf = ViewBuffer::from_vec_with_shape(
                (0..h * w * c).map(|i| i as u16).collect::<Vec<u16>>(),
                vec![h, w, c],
            );
            for view in [buf.flip(&[1]), buf.permute(&[1, 0, 2]), buf.flip(&[0])] {
                assert_eq!(run_bytes(&view), naive_bytes(&view), "{h}x{w}x{c}");
            }
        }
    }

    #[test]
    fn an_empty_view_walks_nothing() {
        let buf = ViewBuffer::from_vec_with_shape(vec![0u8; 12], vec![4, 3]);
        let empty = buf.slice(&[2, 0], &[2, 3]);
        assert_eq!(walk_bytes(&empty), Vec::<u8>::new());
        assert_eq!(run_bytes(&empty), Vec::<u8>::new());
    }
}
