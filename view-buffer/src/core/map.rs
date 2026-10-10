//! Running a map over a buffer's elements: **the one traversal of a
//! per-value or per-pixel kernel.**
//!
//! A kernel whose output value depends on one input element (a cast,
//! `invert`, a scalar chain, `threshold`) or on one pixel's channels
//! (`grayscale`) states only *what* it computes, as an [`ElementMap`] or a
//! [`PixelMap`] over a run of values. How a buffer is read and written is
//! decided here, once, for all of them:
//!
//! - **in place** when the buffer is its data's only owner, packed, and the
//!   map keeps its dtype ([`map_owned`], through
//!   [`ViewBuffer::unique_contiguous_mut`]);
//! - otherwise **into a new packed buffer**, reading a contiguous input as
//!   one run and any other view (a crop, a flip, a transpose) in the runs its
//!   layout has ([`Walk`]), never packing the input first;
//! - always through the CPU dispatch ([`dispatch`]), with the map inlined
//!   into the traversal, so both get the AVX2 build.
//!
//! The output is written into the vector's spare capacity, never zero-filled
//! first: a zeroed image-sized buffer costs a whole extra pass once its
//! memory comes from the heap rather than fresh pages (~15% of a u8
//! threshold at 1024²).

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::core::buffer::ViewBuffer;
use crate::core::dispatch::{dispatch, dispatch_mut, SimdKernel, SimdKernelMut};
use crate::core::dtype::ViewType;
use crate::core::layout::Dims;
use crate::core::strided::{RunSink, Walk};

/// A per-element map from `S` to `D`.
///
/// # Safety
/// [`map_into`](Self::map_into) must initialise every element of `dst`: the
/// traversal marks its output initialised on the strength of it.
pub(crate) unsafe trait ElementMap<S: ViewType, D: ViewType>: Sync {
    /// Write the map of `src[i]` to `dst[i]` for every `i` (`dst` is exactly
    /// as long as `src`). `at` is the logical index of `src[0]` in the
    /// buffer, for a map that depends on an element's channel
    /// (`(at + i) % C`); a run may start anywhere.
    ///
    /// Implementations are `#[inline(always)]`, so the map is compiled into
    /// each dispatched build of the traversal.
    fn map_into(&self, src: &[S], dst: &mut [MaybeUninit<D>], at: usize);
}

/// An [`ElementMap`] from a dtype to itself that can also rewrite a buffer's
/// elements where they lie.
pub(crate) trait ElementMapInPlace<T: ViewType>: ElementMap<T, T> {
    /// Replace every element of `data`, a whole buffer from logical index 0,
    /// by its map. `#[inline(always)]`, as [`ElementMap::map_into`].
    fn map_in_place(&self, data: &mut [T]);
}

/// A map from each `C`-element pixel of `S` to an `O`-element pixel of `D`
/// (grayscale: 3 to 1; a colour space conversion: 3 to 3, 1 to 3, …).
///
/// # Safety
/// As [`ElementMap`]: [`map_into`](Self::map_into) initialises all of `dst`.
pub(crate) unsafe trait PixelMap<S: ViewType, D: ViewType, const C: usize, const O: usize>:
    Sync
{
    /// Write the map of pixel `src[i]` to `dst[i * O..(i + 1) * O]` for every
    /// pixel `i` (`dst` is exactly `O` times as long as `src`).
    /// `#[inline(always)]`, as [`ElementMap::map_into`].
    fn map_into(&self, src: &[[S; C]], dst: &mut [MaybeUninit<D>]);
}

/// `map` of every element of `buf`, in a new packed buffer of `buf`'s shape.
pub(crate) fn map_new<S, D, M>(buf: &ViewBuffer, map: &M) -> ViewBuffer
where
    S: ViewType,
    D: ViewType,
    M: ElementMap<S, D>,
{
    ViewBuffer::from_vec_with_shape(map_vec(buf, map), Dims::from_slice(buf.shape()))
}

/// [`map_new`]'s elements, row-major, as the vector itself: for a kernel that
/// goes on working in it (the f64 resampler's working buffer).
pub(crate) fn map_vec<S, D, M>(buf: &ViewBuffer, map: &M) -> Vec<D>
where
    S: ViewType,
    D: ViewType,
    M: ElementMap<S, D>,
{
    dispatch(MapNew {
        buf,
        map,
        _types: PhantomData,
    })
}

/// `map` of every element of `buf`: rewritten where they lie when `buf` is
/// its data's only owner and packed, else into a new buffer ([`map_new`]).
pub(crate) fn map_owned<T, M>(mut buf: ViewBuffer, map: &M) -> ViewBuffer
where
    T: ViewType,
    M: ElementMapInPlace<T>,
{
    if let Some(data) = buf.unique_contiguous_mut::<T>() {
        dispatch_mut(&InPlace(map), data);
        return buf;
    }
    map_new(&buf, map)
}

/// `map` of every element of the slice `src`, for values that are not a
/// buffer (a lookup table's domain, say).
pub(crate) fn map_slice<S, D, M>(src: &[S], map: &M) -> Vec<D>
where
    S: ViewType,
    D: ViewType,
    M: ElementMap<S, D>,
{
    dispatch(MapSlice {
        src,
        map,
        _to: PhantomData,
    })
}

/// `map` of every `C`-element pixel of `buf`, whose last axis holds its `C`
/// channels, in a new packed buffer of `buf`'s shape with that axis of size
/// `O`.
///
/// # Panics
/// Panics unless `buf`'s last axis has `C` elements.
pub(crate) fn map_pixels<S, D, const C: usize, const O: usize, M>(
    buf: &ViewBuffer,
    map: &M,
) -> ViewBuffer
where
    S: ViewType,
    D: ViewType,
    M: PixelMap<S, D, C, O>,
{
    let mut shape = Dims::from_slice(buf.shape());
    let channels = shape.last_mut().expect("a pixel map reads a channel axis");
    assert_eq!(
        *channels, C,
        "map_pixels: a {C}-channel map over {channels} channels"
    );
    *channels = O;
    let out = dispatch(MapPixels::<S, D, C, O, M> {
        buf,
        map,
        _types: PhantomData,
    });
    ViewBuffer::from_vec_with_shape(out, shape)
}

/// Call `f` with runs of `buf`'s elements whose concatenation is every
/// element in logical order: a contiguous buffer as one run, any other view
/// in the runs its layout has. For a reduction (a statistic) outside any
/// dispatched kernel, so a closure is fine here.
///
/// # Panics
/// Panics if `T` is not `buf`'s dtype.
pub(crate) fn for_each_run<T: ViewType>(buf: &ViewBuffer, mut f: impl FnMut(&[T])) {
    if buf.layout.is_contiguous() {
        f(buf.as_slice::<T>());
    } else {
        Walk::of(buf).for_each_run::<T>(1, &mut f);
    }
}

// The kernels below are dispatched bodies: every loop is written in them, or
// in an `#[inline(always)]` method, never in a closure. A closure is its own
// function, compiled without the dispatched build's instruction set unless
// LLVM chooses to inline it, and it declines for a large body: an earlier
// version of this module wrapped the traversal in one and ran a view's map
// 4x slower on the wheels.

/// The runs of a walk written through an element map into consecutive
/// output slots.
struct MapRuns<'a, S, D, M> {
    map: &'a M,
    dst: &'a mut [MaybeUninit<D>],
    at: usize,
    _source: PhantomData<fn(S)>,
}

impl<S: ViewType, D: ViewType, M: ElementMap<S, D>> RunSink<S> for MapRuns<'_, S, D, M> {
    #[inline(always)]
    fn take(&mut self, run: &[S]) {
        let at = self.at;
        self.map
            .map_into(run, &mut self.dst[at..at + run.len()], at);
        self.at = at + run.len();
    }
}

/// The runs of a walk, whole `C`-element pixels, written through a pixel
/// map into consecutive `O`-element output pixels.
struct PixelRuns<'a, S, D, const C: usize, const O: usize, M> {
    map: &'a M,
    dst: &'a mut [MaybeUninit<D>],
    at: usize,
    _source: PhantomData<fn(S)>,
}

impl<S, D, const C: usize, const O: usize, M> RunSink<S> for PixelRuns<'_, S, D, C, O, M>
where
    S: ViewType,
    D: ViewType,
    M: PixelMap<S, D, C, O>,
{
    #[inline(always)]
    fn take(&mut self, run: &[S]) {
        let run = pixels::<S, C>(run);
        let at = self.at;
        self.map
            .map_into(run, &mut self.dst[at * O..(at + run.len()) * O]);
        self.at = at + run.len();
    }
}

/// `run` as whole `C`-element pixels: every run a pixel map is handed is a
/// whole number of them.
#[inline(always)]
fn pixels<S, const C: usize>(run: &[S]) -> &[[S; C]] {
    let (pixels, rest) = run.as_chunks::<C>();
    assert!(
        rest.is_empty(),
        "internal: a run splits a {C}-element pixel"
    );
    pixels
}

struct MapNew<'a, S, D, M> {
    buf: &'a ViewBuffer,
    map: &'a M,
    _types: PhantomData<(S, D)>,
}

// A derived `Clone` would demand `Clone` of every type parameter.
impl<S, D, M> Clone for MapNew<'_, S, D, M> {
    fn clone(&self) -> Self {
        MapNew {
            buf: self.buf,
            map: self.map,
            _types: PhantomData,
        }
    }
}

impl<S: ViewType, D: ViewType, M: ElementMap<S, D>> SimdKernel for MapNew<'_, S, D, M> {
    type Output = Vec<D>;

    #[inline(always)]
    fn run(self) -> Vec<D> {
        let (buf, map) = (self.buf, self.map);
        let n = buf.shape().iter().product();
        let mut out = Vec::with_capacity(n);
        let dst = &mut out.spare_capacity_mut()[..n];
        if buf.layout.is_contiguous() {
            map.map_into(buf.as_slice::<S>(), dst, 0);
        } else {
            let mut runs = MapRuns {
                map,
                dst,
                at: 0,
                _source: PhantomData,
            };
            Walk::of(buf).for_each_run::<S>(1, &mut runs);
            // Moved out, not borrowed: a panic message borrowing a field of
            // `runs` would keep the struct in memory, and its output pointer
            // with it, which the map's stores might then alias; the loop
            // stayed scalar (17 instructions a pixel for a flipped image).
            let MapRuns { at, .. } = runs;
            assert_eq!(at, n, "internal: the walk read {at} of {n} elements");
        }
        // SAFETY: every one of the `n` slots was handed to `map_into`, which
        // initialises what it is given (`ElementMap`): all at once for a
        // contiguous input, else in runs whose lengths add up to the view's
        // element count (asserted).
        unsafe { out.set_len(n) };
        out
    }
}

struct MapSlice<'a, S, D, M> {
    src: &'a [S],
    map: &'a M,
    _to: PhantomData<D>,
}

impl<S, D, M> Clone for MapSlice<'_, S, D, M> {
    fn clone(&self) -> Self {
        MapSlice {
            src: self.src,
            map: self.map,
            _to: PhantomData,
        }
    }
}

impl<S: ViewType, D: ViewType, M: ElementMap<S, D>> SimdKernel for MapSlice<'_, S, D, M> {
    type Output = Vec<D>;

    #[inline(always)]
    fn run(self) -> Vec<D> {
        let n = self.src.len();
        let mut out = Vec::with_capacity(n);
        self.map
            .map_into(self.src, &mut out.spare_capacity_mut()[..n], 0);
        // SAFETY: `map_into` initialises all of its `dst` (`ElementMap`).
        unsafe { out.set_len(n) };
        out
    }
}

struct InPlace<'a, M>(&'a M);

impl<T: ViewType, M: ElementMapInPlace<T>> SimdKernelMut<T> for InPlace<'_, M> {
    #[inline(always)]
    fn run_mut(&self, data: &mut [T]) {
        self.0.map_in_place(data);
    }
}

struct MapPixels<'a, S, D, const C: usize, const O: usize, M> {
    buf: &'a ViewBuffer,
    map: &'a M,
    _types: PhantomData<(S, D)>,
}

impl<S, D, const C: usize, const O: usize, M> Clone for MapPixels<'_, S, D, C, O, M> {
    fn clone(&self) -> Self {
        MapPixels {
            buf: self.buf,
            map: self.map,
            _types: PhantomData,
        }
    }
}

impl<S, D, const C: usize, const O: usize, M> SimdKernel for MapPixels<'_, S, D, C, O, M>
where
    S: ViewType,
    D: ViewType,
    M: PixelMap<S, D, C, O>,
{
    type Output = Vec<D>;

    #[inline(always)]
    fn run(self) -> Vec<D> {
        let (buf, map) = (self.buf, self.map);
        let n = buf.shape().iter().product::<usize>() / C;
        let mut out = Vec::with_capacity(n * O);
        let dst = &mut out.spare_capacity_mut()[..n * O];
        if buf.layout.is_contiguous() {
            map.map_into(pixels::<S, C>(buf.as_slice::<S>()), dst);
        } else {
            let mut runs = PixelRuns::<S, D, C, O, M> {
                map,
                dst,
                at: 0,
                _source: PhantomData,
            };
            Walk::of(buf).for_each_run::<S>(C, &mut runs);
            // Moved out, not borrowed: see `MapNew`.
            let PixelRuns { at, .. } = runs;
            assert_eq!(at, n, "internal: the walk read {at} of {n} pixels");
        }
        // SAFETY: as in `MapNew`, counting pixels: every one of the `n`
        // output pixels' `O` slots was handed to `map_into` exactly once
        // (asserted).
        unsafe { out.set_len(n * O) };
        out
    }
}
