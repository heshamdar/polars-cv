//! Separable Gaussian blur.

use crate::core::buffer::ViewBuffer;
use crate::core::convert::CastFrom;
use crate::core::dispatch::{dispatch, SimdKernel};

thread_local! {
    /// Scratch buffer for the horizontal pass of separable Gaussian blur.
    /// Holds f32 values for the current image (h × w × channels). Grows to
    /// fit the largest image seen per thread and is never shrunk — amortised
    /// O(1) allocation.
    static BLUR_HORIZ_BUF: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// The same slab for an f64-accumulating blur (`BlurAcc for f64`).
    static BLUR_HORIZ_BUF_F64: std::cell::RefCell<Vec<f64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Builds a normalised 1-D Gaussian kernel of radius `ceil(3σ)`.
/// The radius formula matches the halo declaration in `ops/image.rs`.
/// σ = 0 is the one-tap identity kernel: the Gaussian formula would divide
/// 0 by 0 there (`ImageOpKind::check` refuses a negative σ).
fn gaussian_kernel_1d<F: BlurAcc>(sigma: f32) -> Vec<F> {
    if sigma == 0.0 {
        return vec![F::from_f32(1.0)];
    }
    let radius = (sigma * 3.0).ceil() as usize;
    let size = 2 * radius + 1;
    let sigma = F::from_f32(sigma);
    let two = F::from_f32(2.0);
    let mut k: Vec<F> = (0..size)
        .map(|i| {
            let x = F::from_usize(i) - F::from_usize(radius);
            (-x * x / (two * sigma * sigma)).exp()
        })
        .collect();
    let s: F = k.iter().fold(F::zero(), |acc, &v| acc + v);
    k.iter_mut().for_each(|v| *v = *v / s);
    k
}

/// The float a blur accumulates in ([`DType::accumulator`]): f32, or f64 for
/// f64 and the 32/64-bit integers. Each has its own per-thread scratch slab
/// for the horizontal pass.
pub(crate) trait BlurAcc:
    Copy
    + Default
    + std::ops::AddAssign
    + std::ops::Neg<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Add<Output = Self>
    + 'static
{
    fn zero() -> Self;
    fn exp(self) -> Self;
    fn from_f32(v: f32) -> Self;
    fn from_usize(v: usize) -> Self;
    /// The thread's scratch slab, taken (and given back by [`Self::give_slab`])
    /// so the passes stay in the dispatched function's own body.
    fn take_slab() -> Vec<Self>;
    fn give_slab(slab: Vec<Self>);
}

impl BlurAcc for f32 {
    fn zero() -> Self {
        0.0
    }
    fn exp(self) -> Self {
        f32::exp(self)
    }
    fn from_f32(v: f32) -> Self {
        v
    }
    fn from_usize(v: usize) -> Self {
        v as f32
    }
    fn take_slab() -> Vec<Self> {
        BLUR_HORIZ_BUF.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
    }
    fn give_slab(slab: Vec<Self>) {
        BLUR_HORIZ_BUF.with(|cell| *cell.borrow_mut() = slab);
    }
}

impl BlurAcc for f64 {
    fn zero() -> Self {
        0.0
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn from_f32(v: f32) -> Self {
        f64::from(v)
    }
    fn from_usize(v: usize) -> Self {
        v as f64
    }
    fn take_slab() -> Vec<Self> {
        BLUR_HORIZ_BUF_F64.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
    }
    fn give_slab(slab: Vec<Self>) {
        BLUR_HORIZ_BUF_F64.with(|cell| *cell.borrow_mut() = slab);
    }
}

/// Separable Gaussian blur: one 1-D horizontal pass followed by one 1-D
/// vertical pass.  Mathematically equivalent to 2-D Gaussian convolution but
/// O(k) per pixel instead of O(k²) — ~6.5× fewer multiply-adds for σ=2.
///
/// Border handling: replicate (edge pixels clamped to image boundary), matching
/// the behaviour of `morph_minmax_typed` (`kernels::morph`).
///
/// Structured for auto-vectorization:
/// - the input is converted to f32 once up front (n casts instead of k·n
///   per-tap `NumCast` calls);
/// - the horizontal pass accumulates each kernel tap as a contiguous
///   shifted-slice multiply-add over the row interior, with clamped-index
///   gathers only for the `radius` columns at each border;
/// - the vertical pass accumulates whole rows (`row_out += k·row_src`), so
///   border handling is a per-row clamp, not a per-element one;
/// - the f32 → T conversion happens once per element after the passes.
///
/// The horizontal-pass intermediate is stored in a thread-local f32 slab
/// (`BLUR_HORIZ_BUF`) that grows to fit the largest image seen per thread and
/// is never shrunk — zero allocator round-trip on warm paths.
/// The separable Gaussian blur, run through [`dispatch`] so the whole body
/// (conversion, both passes, the output clamp) has an AVX2 build
/// (~1.4x over the wheels' SSE2 baseline; dispatching only the row axpy
/// measured slower).
pub(crate) fn separable_gaussian_blur_typed<T, F>(contig_buf: &ViewBuffer, sigma: f32) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + Copy + CastFrom<F>,
    F: BlurAcc + CastFrom<T>,
{
    dispatch(SeparableBlur::<T, F> {
        buf: contig_buf,
        sigma,
        _elem: std::marker::PhantomData,
    })
}

struct SeparableBlur<'a, T, F> {
    buf: &'a ViewBuffer,
    sigma: f32,
    _elem: std::marker::PhantomData<fn() -> (T, F)>,
}

// Derived `Clone` would demand `T: Clone` of the marker's parameter.
impl<T, F> Clone for SeparableBlur<'_, T, F> {
    fn clone(&self) -> Self {
        SeparableBlur {
            buf: self.buf,
            sigma: self.sigma,
            _elem: std::marker::PhantomData,
        }
    }
}

impl<T, F> SimdKernel for SeparableBlur<'_, T, F>
where
    T: crate::core::dtype::ViewType + Default + Copy + CastFrom<F>,
    F: BlurAcc + CastFrom<T>,
{
    type Output = ViewBuffer;

    #[inline(always)]
    fn run(self) -> ViewBuffer {
        separable_gaussian_blur_body::<T, F>(self.buf, self.sigma)
    }
}

#[cfg(test)]
mod blur_dispatch_tests {
    use super::{separable_gaussian_blur_body, separable_gaussian_blur_typed};
    use crate::core::buffer::ViewBuffer;

    use super::{gaussian_kernel_1d, BLUR_HORIZ_BUF};
    use crate::core::dtype::DType;
    use crate::kernels::clamp_for_dtype;
    use crate::ops::{ImageOp, ImageOpKind};

    /// The blur as it was before Phase 8, kept verbatim as the parity oracle.
    pub(super) fn reference_blur_body<T>(contig_buf: &ViewBuffer, sigma: f32) -> ViewBuffer
    where
        T: crate::core::dtype::ViewType + Default + Copy + num_traits::NumCast,
    {
        use num_traits::NumCast;

        let shape = contig_buf.shape();
        let h = shape[0];
        let w = shape[1];
        let c = shape.get(2).copied().unwrap_or(1);
        let n = h * w * c;
        let wc = w * c;

        let kernel = gaussian_kernel_1d::<f32>(sigma);
        let radius = kernel.len() / 2;
        let src: &[T] = contig_buf.as_slice::<T>();

        // ── Convert input to f32 once ────────────────────────────────────────
        let mut input_f32: Vec<f32> = Vec::with_capacity(n);
        input_f32.extend(src.iter().map(|&v| {
            let v: f32 = NumCast::from(v).unwrap_or(0.0);
            v
        }));

        // The scratch slab is taken out of the thread-local for the duration of
        // the call rather than used inside a `with` closure: the passes must be in
        // this function's own body to be compiled with its target features (a
        // closure is a separate function and does not inherit them).
        let mut slab = BLUR_HORIZ_BUF.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
        if slab.len() < n {
            slab.resize(n, 0.0f32);
        }
        let result = {
            let horiz = &mut slab[..n];

            // ── Horizontal pass (f32 → f32) ──────────────────────────────────
            for y in 0..h {
                let row_in = &input_f32[y * wc..(y + 1) * wc];
                let row_out = &mut horiz[y * wc..(y + 1) * wc];

                if w <= 2 * radius {
                    // Image narrower than the kernel: clamped gather everywhere.
                    for x in 0..w {
                        for ch in 0..c {
                            let mut sum = 0.0f32;
                            for (ki, &kw) in kernel.iter().enumerate() {
                                let sx = (x as i64 + ki as i64 - radius as i64)
                                    .clamp(0, w as i64 - 1)
                                    as usize;
                                sum += kw * row_in[sx * c + ch];
                            }
                            row_out[x * c + ch] = sum;
                        }
                    }
                    continue;
                }

                // Interior: per-tap shifted-slice multiply-add over contiguous
                // memory — the inner zip vectorizes.
                let lo = radius * c;
                let hi = (w - radius) * c;
                row_out[lo..hi].fill(0.0);
                for (ki, &kw) in kernel.iter().enumerate() {
                    let shift = (ki as i64 - radius as i64) * c as i64;
                    let src_start = (lo as i64 + shift) as usize;
                    let src_slice = &row_in[src_start..src_start + (hi - lo)];
                    for (o, &v) in row_out[lo..hi].iter_mut().zip(src_slice) {
                        *o += kw * v;
                    }
                }
                // Borders: clamped gather for `radius` columns on each side.
                for x in (0..radius).chain(w - radius..w) {
                    for ch in 0..c {
                        let mut sum = 0.0f32;
                        for (ki, &kw) in kernel.iter().enumerate() {
                            let sx = (x as i64 + ki as i64 - radius as i64).clamp(0, w as i64 - 1)
                                as usize;
                            sum += kw * row_in[sx * c + ch];
                        }
                        row_out[x * c + ch] = sum;
                    }
                }
            }

            // ── Vertical pass (f32 → f32 row accumulation) → T ───────────────
            let is_float = matches!(T::DTYPE, DType::F32 | DType::F64);
            let mut acc_row = vec![0.0f32; wc];
            let mut out: Vec<T> = Vec::with_capacity(n);
            for y in 0..h {
                acc_row.fill(0.0);
                for (ki, &kw) in kernel.iter().enumerate() {
                    let sy = (y as i64 + ki as i64 - radius as i64).clamp(0, h as i64 - 1) as usize;
                    let src_row = &horiz[sy * wc..(sy + 1) * wc];
                    for (o, &v) in acc_row.iter_mut().zip(src_row) {
                        *o += kw * v;
                    }
                }
                if is_float {
                    out.extend(
                        acc_row
                            .iter()
                            .map(|&v| NumCast::from(v).unwrap_or(T::default())),
                    );
                } else {
                    out.extend(acc_row.iter().map(|&v| {
                        NumCast::from(clamp_for_dtype(v as f64, T::DTYPE)).unwrap_or(T::default())
                    }));
                }
            }

            ViewBuffer::from_vec_with_shape(out, shape.to_vec())
        };
        BLUR_HORIZ_BUF.with(|cell| *cell.borrow_mut() = slab);
        result
    }

    /// Every dtype the typed blur runs (others cast to f32 around it), with
    /// NaN, infinities and the integer extremes, over 1-5 channels, sizes
    /// down to narrower than the kernel, and several sigmas: the blur equals
    /// the pre-Phase-8 kernel byte for byte.
    #[test]
    fn blur_matches_the_reference_kernel_bit_for_bit() {
        let shapes: [&[usize]; 7] = [
            &[1, 1],
            &[3, 2, 3],
            &[9, 5, 1],
            &[16, 23],
            &[20, 17, 2],
            &[11, 30, 4],
            &[7, 13, 5],
        ];
        let wave = |i: usize| ((i * 7919 + 13) % 256) as f32;
        for shape in shapes {
            let n: usize = shape.iter().product();
            let u8s: Vec<u8> = (0..n)
                .map(|i| if i % 11 == 0 { 255 } else { wave(i) as u8 })
                .collect();
            let u16s: Vec<u16> = (0..n)
                .map(|i| {
                    if i % 7 == 0 {
                        u16::MAX
                    } else {
                        (wave(i) * 251.0) as u16
                    }
                })
                .collect();
            let f32s: Vec<f32> = (0..n)
                .map(|i| match i % 17 {
                    3 => f32::NAN,
                    8 => f32::INFINITY,
                    12 => -0.0,
                    _ => wave(i) / 7.0 - 11.0,
                })
                .collect();
            let bufs = [
                ViewBuffer::from_vec_with_shape(u8s, shape.to_vec()),
                ViewBuffer::from_vec_with_shape(u16s, shape.to_vec()),
                ViewBuffer::from_vec_with_shape(f32s, shape.to_vec()),
            ];
            for buf in &bufs {
                for sigma in [0.3f32, 0.8, 2.0, 5.5] {
                    let got = crate::ops::image::apply_image_inner(
                        buf.clone(),
                        ImageOp {
                            kind: ImageOpKind::Blur { sigma },
                        },
                    );
                    let want = match buf.dtype() {
                        DType::U8 => reference_blur_body::<u8>(buf, sigma),
                        DType::U16 => reference_blur_body::<u16>(buf, sigma),
                        _ => reference_blur_body::<f32>(buf, sigma),
                    };
                    let bytes = |b: &ViewBuffer| {
                        use crate::core::dispatch::KernelOutput;
                        b.output_bytes().to_vec()
                    };
                    assert_eq!(got.shape(), want.shape());
                    assert!(
                        bytes(&got) == bytes(&want),
                        "{:?} {shape:?} sigma {sigma}: the blur differs from the reference",
                        buf.dtype()
                    );
                }
            }
        }
    }

    /// sigma 0 is the identity: one tap of weight 1, not exp(-0/0) = NaN,
    /// which cast every u8 pixel to 0.
    #[test]
    fn zero_sigma_blur_is_the_identity() {
        assert_eq!(gaussian_kernel_1d::<f32>(0.0), vec![1.0f32]);
        assert_eq!(gaussian_kernel_1d::<f64>(0.0), vec![1.0f64]);
        let data: Vec<u8> = (0..5 * 4 * 3).map(|i| (i * 37 % 256) as u8).collect();
        let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![5, 4, 3]);
        let out = separable_gaussian_blur_typed::<u8, f32>(&buf, 0.0);
        assert_eq!(out.as_slice::<u8>(), data.as_slice());
    }

    /// The dispatched blur (AVX2 build when the CPU has it) and the portable
    /// build agree bit for bit. `dispatch` asserts the same in every debug
    /// build; this pins it on odd sizes and several radii.
    #[test]
    fn blur_dispatch_is_bit_identical() {
        let (h, w, c) = (37usize, 53usize, 3usize);
        let u8s: Vec<u8> = (0..h * w * c).map(|i| ((i * 7919) % 251) as u8).collect();
        let f32s: Vec<f32> = u8s.iter().map(|&v| f32::from(v) / 7.0 - 11.0).collect();
        let u8_buf = ViewBuffer::from_vec_with_shape(u8s, vec![h, w, c]);
        let f32_buf = ViewBuffer::from_vec_with_shape(f32s, vec![h, w, c]);
        for sigma in [0.8f32, 2.0, 5.5] {
            let (a, b) = (
                separable_gaussian_blur_typed::<u8, f32>(&u8_buf, sigma),
                separable_gaussian_blur_body::<u8, f32>(&u8_buf, sigma),
            );
            assert_eq!(a.as_slice::<u8>(), b.as_slice::<u8>(), "u8 sigma {sigma}");
            let (a, b) = (
                separable_gaussian_blur_typed::<f32, f32>(&f32_buf, sigma),
                separable_gaussian_blur_body::<f32, f32>(&f32_buf, sigma),
            );
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(a.as_slice::<f32>()),
                bits(b.as_slice::<f32>()),
                "f32 sigma {sigma}"
            );
        }
    }
}

#[inline(always)]
fn separable_gaussian_blur_body<T, F>(rows_buf: &ViewBuffer, sigma: f32) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + Copy + CastFrom<F>,
    F: BlurAcc + CastFrom<T>,
{
    let shape = rows_buf.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape.get(2).copied().unwrap_or(1);
    let n = h * w * c;
    let wc = w * c;

    let kernel: Vec<F> = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;
    // Read in place, at the view's own row stride (a crop, a vertical flip).
    let src_rows: Vec<&[T]> = rows_buf
        .dense_rows::<T>()
        .expect("the blur is handed packed rows (`to_dense_rows`)");

    // The scratch slab is taken out of the thread-local for the duration of
    // the call rather than used inside a `with` closure: the passes must be in
    // this function's own body to be compiled with its target features (a
    // closure is a separate function and does not inherit them).
    let mut slab = F::take_slab();
    if slab.len() < n {
        slab.resize(n, F::zero());
    }
    let result = {
        let horiz = &mut slab[..n];
        // One input row in F, converted as the horizontal pass reaches it:
        // converting the whole image first allocated and wrote 4 bytes per
        // element that were read once.
        let mut row_in = vec![F::zero(); wc];

        // ── Horizontal pass (T → F row → F) ─────────────────────────────
        for y in 0..h {
            for (dst, &v) in row_in.iter_mut().zip(src_rows[y]) {
                *dst = F::cast_from(v);
            }
            let row_out = &mut horiz[y * wc..(y + 1) * wc];

            if w <= 2 * radius {
                // Image narrower than the kernel: clamped gather everywhere.
                for x in 0..w {
                    for ch in 0..c {
                        let mut sum = F::zero();
                        for (ki, &kw) in kernel.iter().enumerate() {
                            let sx = (x as i64 + ki as i64 - radius as i64).clamp(0, w as i64 - 1)
                                as usize;
                            sum += kw * row_in[sx * c + ch];
                        }
                        row_out[x * c + ch] = sum;
                    }
                }
                continue;
            }

            // Interior: per-tap shifted-slice multiply-add over contiguous
            // memory — the inner zip vectorizes.
            let lo = radius * c;
            let hi = (w - radius) * c;
            row_out[lo..hi].fill(F::zero());
            for (ki, &kw) in kernel.iter().enumerate() {
                let shift = (ki as i64 - radius as i64) * c as i64;
                let src_start = (lo as i64 + shift) as usize;
                let src_slice = &row_in[src_start..src_start + (hi - lo)];
                for (o, &v) in row_out[lo..hi].iter_mut().zip(src_slice) {
                    *o += kw * v;
                }
            }
            // Borders: clamped gather for `radius` columns on each side.
            for x in (0..radius).chain(w - radius..w) {
                for ch in 0..c {
                    let mut sum = F::zero();
                    for (ki, &kw) in kernel.iter().enumerate() {
                        let sx =
                            (x as i64 + ki as i64 - radius as i64).clamp(0, w as i64 - 1) as usize;
                        sum += kw * row_in[sx * c + ch];
                    }
                    row_out[x * c + ch] = sum;
                }
            }
        }

        // ── Vertical pass (F → F row accumulation) → T ───────────────────
        // Stored through M5: round-then-saturate to an integer, as is to f32.
        let mut acc_row = vec![F::zero(); wc];
        let mut out: Vec<T> = Vec::with_capacity(n);
        for y in 0..h {
            acc_row.fill(F::zero());
            for (ki, &kw) in kernel.iter().enumerate() {
                let sy = (y as i64 + ki as i64 - radius as i64).clamp(0, h as i64 - 1) as usize;
                let src_row = &horiz[sy * wc..(sy + 1) * wc];
                for (o, &v) in acc_row.iter_mut().zip(src_row) {
                    *o += kw * v;
                }
            }
            out.extend(acc_row.iter().map(|&v| T::cast_from(v)));
        }

        ViewBuffer::from_vec_with_shape(out, shape.to_vec())
    };
    F::give_slab(slab);
    result
}

#[cfg(test)]
mod blur_radius_tests {
    //! The blur op declares a `Neighborhood` radius that a spatial-window
    //! reorder would dilate a crop by; if it understated the true kernel reach
    //! the halo would be too small and the reorder would corrupt border pixels.
    //! The declared radius (`ops/image.rs`) and the executed kernel
    //! (`gaussian_kernel_1d`) each compute `ceil(3σ)` independently, so this
    //! cross-checks the declaration against the kernel actually applied — the
    //! second authority `SpatialDependency` otherwise lacks for the radius.

    use super::gaussian_kernel_1d;
    use crate::ops::image::{ImageOp, ImageOpKind};
    use crate::ops::spatial_rule::SpatialDependency;
    use crate::ops::traits::Op;

    fn declared_radius(sigma: f32) -> usize {
        let op: ImageOp = ImageOp {
            kind: ImageOpKind::Blur { sigma },
        };
        match op.spatial_dependency() {
            SpatialDependency::Neighborhood(support) => support
                .radius
                .known()
                .expect("an executed blur's radius is known"),
            other => panic!("blur must be a Neighborhood dependency, got {other:?}"),
        }
    }

    #[test]
    fn declared_radius_matches_the_executed_kernel() {
        for sigma in [0.5f32, 1.0, 1.5, 2.0, 3.3, 5.0] {
            let kernel_radius = gaussian_kernel_1d::<f32>(sigma).len() / 2;
            assert_eq!(
                declared_radius(sigma),
                kernel_radius,
                "blur σ={sigma}: declared neighborhood radius must equal the \
                 executed 1-D kernel's half-width"
            );
        }
    }
}
