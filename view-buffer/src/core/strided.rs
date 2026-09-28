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

/// The outer axes of a [`Geometry`]: at most rank − 1 of them.
pub(crate) type Outer = smallvec::SmallVec<[(usize, isize); 4]>;

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
    /// Inline for any image or batch of images: walking a view allocates
    /// nothing.
    pub(crate) outer: Outer,
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
        let mut outer = Outer::new();
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

    /// The first unit of every row, in logical order.
    ///
    /// An iterator rather than a function taking a closure: a loop over it
    /// keeps its body in the caller, so inside a dispatched kernel the body
    /// is compiled with the kernel's instruction set. A closure is a function
    /// of its own that does not inherit it unless LLVM chooses to inline it,
    /// which it declines for a large body (measured: a map over a view ran
    /// its baseline build, 4x slower on the wheels).
    #[inline(always)]
    fn rows(&self) -> Rows<'_> {
        let outer = self.geometry.outer.as_slice();
        Rows {
            row: self.base,
            left: if self.empty {
                0
            } else {
                outer.iter().map(|&(n, _)| n).product()
            },
            outer,
            index: smallvec::smallvec![0; outer.len()],
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
                for row in self.rows() {
                    // SAFETY: `row` is a row of the view, whose `row_len`
                    // units lie inside its data; `out` advances by one
                    // packed row per row, inside what the caller provides.
                    unsafe {
                        $copy(row, out, row_len, step);
                        out = out.add(row_bytes);
                    }
                }
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

    /// Hand `sink` runs of `T` whose concatenation is every element in
    /// logical order, each run a whole number of `grain`s (a pixel's
    /// channels, say; 1 for any run). Long units are handed out where they
    /// lie; short ones are packed into a stack scratch first.
    ///
    /// `sink` is called directly, never through a `dyn` pointer, and a
    /// dispatched consumer implements [`RunSink`] with an
    /// `#[inline(always)]` method rather than passing a closure: it is the
    /// consumer's kernel, and inside a dispatched build it must inline to
    /// get that build's instruction set.
    ///
    /// # Panics
    /// Panics if `T` is not the view's dtype, or if the view's element count
    /// is not a whole number of `grain`s.
    #[inline(always)]
    pub(crate) fn for_each_run<T: ViewType>(&self, grain: usize, sink: &mut impl RunSink<T>) {
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
        if unit_elems >= DIRECT_ELEMS && unit_elems.is_multiple_of(grain) {
            for row in self.rows() {
                for k in 0..row_len {
                    // Opaque to the optimiser: otherwise LLVM hoists the
                    // consumer's vector-loop overlap check out of this loop,
                    // over every unit at once, and gives up whenever `step`
                    // is negative (a vertical flip), so each unit ran the
                    // scalar loop (5x slower grayscale). Made per unit, the
                    // check passes.
                    let unit_ptr = std::hint::black_box(row.wrapping_offset(k as isize * step));
                    // SAFETY: unit `k` of a row is `unit_elems` packed
                    // elements inside the view's data, aligned for `T`
                    // (offsets and strides are whole elements, CR-41).
                    sink.take(unsafe {
                        std::slice::from_raw_parts(unit_ptr.cast::<T>(), unit_elems)
                    });
                }
            }
            return;
        }
        // Pack short units into an aligned stack scratch, handing it out
        // whenever it fills. A full scratch holds a whole number of grains:
        // `per_block` units is a multiple of the units a grain spans (a unit
        // is either a whole number of grains, or divides one, since the
        // packed innermost axes either include a pixel's channels or are
        // one element; asserted).
        let per_grain = grain / gcd(grain, unit_elems);
        assert!(
            grain.is_multiple_of(unit_elems) || unit_elems.is_multiple_of(grain),
            "for_each_run: {unit_elems}-element units cannot form {grain}-element grains"
        );
        let per_block = (SCRATCH_BYTES / unit) / per_grain * per_grain;
        assert!(
            per_block > 0,
            "for_each_run: a {grain}-element grain exceeds the scratch"
        );
        let mut scratch = [MaybeUninit::<u64>::uninit(); SCRATCH_BYTES / 8];
        let scratch = scratch.as_mut_ptr().cast::<u8>();
        let mut filled = 0usize;
        for row in self.rows() {
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
                    // SAFETY: the scratch's first `per_block` units were
                    // written with whole elements of `T`; it is 8-byte
                    // aligned.
                    sink.take(unsafe {
                        std::slice::from_raw_parts(scratch.cast::<T>(), per_block * unit_elems)
                    });
                    filled = 0;
                }
            }
        }
        if filled > 0 {
            let len = filled * unit_elems;
            assert!(
                len.is_multiple_of(grain),
                "for_each_run: {len} trailing elements are not whole {grain}-element grains"
            );
            // SAFETY: as for a full scratch, over its first `filled` units.
            sink.take(unsafe { std::slice::from_raw_parts(scratch.cast::<T>(), len) });
        }
    }
}

/// The consumer of [`Walk::for_each_run`]'s runs.
///
/// A dispatched kernel implements it on a struct with an `#[inline(always)]`
/// [`take`](Self::take), so its body is compiled into the dispatched build;
/// any `FnMut(&[T])` closure also is one, for code that is not dispatched.
pub(crate) trait RunSink<T> {
    fn take(&mut self, run: &[T]);
}

impl<T, F: FnMut(&[T])> RunSink<T> for F {
    #[inline(always)]
    fn take(&mut self, run: &[T]) {
        self(run)
    }
}

/// The rows of a [`Walk`]: each row's first unit, the outer axes stepped
/// by an odometer that moves the row pointer by adding strides.
struct Rows<'w> {
    row: *const u8,
    left: usize,
    outer: &'w [(usize, isize)],
    index: smallvec::SmallVec<[usize; 4]>,
}

impl Iterator for Rows<'_> {
    type Item = *const u8;

    #[inline(always)]
    fn next(&mut self) -> Option<*const u8> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        let row = self.row;
        // Step the innermost outer axis, carrying.
        for (i, &(n, stride)) in self.outer.iter().enumerate().rev() {
            self.index[i] += 1;
            self.row = self.row.wrapping_offset(stride);
            if self.index[i] < n {
                break;
            }
            self.index[i] = 0;
            self.row = self.row.wrapping_offset(-stride * n as isize);
        }
        Some(row)
    }
}

/// The greatest common divisor of two positive counts.
const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
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
            outer: outer.iter().copied().collect(),
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
        with_dtype!(view.dtype(), T => Walk::of(view).for_each_run::<T>(1, &mut |run: &[T]| {
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

    /// Runs of `grain`-element pieces (a pixel's channels) never split one,
    /// whatever the layout, and still concatenate to every element in order.
    /// The channel-strided case (an image stored channel-first, viewed
    /// channels-last) packs one element at a time into a scratch that holds
    /// no whole number of 3-channel pixels.
    #[test]
    fn a_run_holds_whole_grains_in_every_layout() {
        let mut views: Vec<(String, ViewBuffer)> = Vec::new();
        let chw = ViewBuffer::from_vec_with_shape(
            (0..3 * 40 * 700).map(|i| i as u8).collect::<Vec<u8>>(),
            vec![3, 40, 700],
        );
        views.push((
            "channel-first as channels-last".into(),
            chw.permute(&[1, 2, 0]),
        ));
        let mut rng = Lcg(0x6A1E);
        for case in 0..400 {
            let view = random_view(&mut rng, DType::U16);
            views.push((format!("random case {case}"), view));
        }
        for (label, view) in views {
            let grain = view.shape().last().copied().unwrap_or(1).max(1);
            let mut out = Vec::new();
            with_dtype!(view.dtype(), T => Walk::of(&view).for_each_run::<T>(grain, &mut |run: &[T]| {
                assert!(
                    run.len().is_multiple_of(grain),
                    "{label}: a run of {} elements splits a {grain}-element grain",
                    run.len()
                );
                for x in run {
                    out.extend_from_slice(&x.to_ne_bytes());
                }
            }));
            assert_eq!(out, naive_bytes(&view), "{label}");
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
