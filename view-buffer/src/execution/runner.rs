//! Pure operation runners — apply one op to one full buffer.
//!
//! These functions have no tiling or strategy logic; that lives in
//! [`execution::tiling`] and [`execution::plan`].  Everything here is
//! `pub(crate)` so both the full-image path and the tiling path can call the
//! same implementations.

use crate::core::buffer::ViewBuffer;
#[cfg(feature = "image_interop")]
use crate::core::convert::CastFrom;
#[cfg(feature = "image_interop")]
use crate::core::cut::{AgainstCut, Cut};
#[cfg(feature = "image_interop")]
use crate::core::dispatch::{dispatch, SimdKernel};
use crate::core::dtype::DType;
#[cfg(feature = "image_interop")]
use crate::core::dtype::{with_dtype, ViewType};
#[cfg(feature = "image_interop")]
use crate::core::map::{map_new, map_owned, map_pixels, ElementMap, ElementMapInPlace, PixelMap};
use crate::expr::ViewExpr;
use crate::ops::dto::ViewDto;
#[cfg(feature = "image_interop")]
use crate::ops::traits::Op;
use crate::ops::{ComputeOp, ImageOp, ViewOp};
#[cfg(feature = "image_interop")]
use std::marker::PhantomData;
#[cfg(feature = "image_interop")]
use std::mem::MaybeUninit;

#[cfg(feature = "image_interop")]
use crate::ops::{FilterType, ImageOpKind};

#[cfg(feature = "image_interop")]
use fast_image_resize as fir;

/// High-level entry point to execute a plan described by a sequence of ViewDto operations.
/// This acts as the bridge between the serialized plan (e.g. from Python) and the
/// execution engine.
pub fn execute_plan(source: ViewBuffer, ops: Vec<ViewDto>) -> ViewBuffer {
    let mut expr = ViewExpr::new_source(source);
    for op in ops {
        expr = expr.apply_op(op);
    }
    // plan() performs optimization (fusion, etc) before execution
    expr.plan().execute()
}

/// Applies a view operation to a buffer.
pub fn apply_view(buf: ViewBuffer, op: ViewOp) -> ViewBuffer {
    if let Some((start, end)) = op.window() {
        return buf.slice(&start, &end);
    }
    match op {
        ViewOp::Transpose { .. } => buf.permute(&op.axes()),
        // The planner packs a reshape's input (`MemoryEffect::ViewOfContiguous`);
        // `ViewBuffer::reshape` refuses a strided one in every build.
        ViewOp::Reshape { shape } => buf.reshape(
            shape
                .iter()
                .map(|&d| d as usize)
                .collect::<crate::core::layout::Dims>(),
        ),
        ViewOp::Flip { .. } => buf.flip(&op.axes()),
        ViewOp::Crop { .. } | ViewOp::Slice { .. } => unreachable!("windows are sliced above"),
        ViewOp::Rotate90 => {
            // Rotate90: transpose [1,0] then flip axis 1 (width)
            // For HWC layout: transpose swaps H and W, then flip W
            let shape = buf.shape();
            if shape.len() < 2 {
                return buf; // Can't rotate 1D or 0D
            }
            let perm = if shape.len() == 2 {
                vec![1, 0] // [H, W] -> [W, H]
            } else {
                vec![1, 0, 2] // [H, W, C] -> [W, H, C]
            };
            let transposed = buf.permute(&perm);
            transposed.flip(&[1]) // Flip width axis
        }
        ViewOp::Rotate180 => {
            // Rotate180: flip both height (axis 0) and width (axis 1)
            buf.flip(&[0, 1])
        }
        ViewOp::Rotate270 => {
            // Rotate270: transpose [1,0] then flip axis 0 (height)
            // For HWC layout: transpose swaps H and W, then flip H
            let shape = buf.shape();
            if shape.len() < 2 {
                return buf; // Can't rotate 1D or 0D
            }
            let perm = if shape.len() == 2 {
                vec![1, 0] // [H, W] -> [W, H]
            } else {
                vec![1, 0, 2] // [H, W, C] -> [W, H, C]
            };
            let transposed = buf.permute(&perm);
            transposed.flip(&[0]) // Flip height axis
        }
        ViewOp::ChannelSelect { index } => {
            let index = index as usize;
            let shape = buf.shape();
            if shape.len() != 3 {
                return buf;
            }
            let h = shape[0];
            let w = shape[1];
            // Slice to [H, W, 1] then materialize (slice is non-contiguous in HWC)
            let sliced = buf.slice(&[0, 0, index], &[h, w, index + 1]);
            sliced.to_contiguous().reshape(vec![h, w])
        }
    }
}

/// Applies a compute operation to a buffer.
#[inline]
pub(crate) fn apply_compute_inner(buf: ViewBuffer, op: ComputeOp) -> ViewBuffer {
    match op {
        ComputeOp::Cast { dtype } => buf.cast(dtype),
        ComputeOp::Affine(params) => super::warp::apply_affine_warp(buf, params),
        ComputeOp::RotateAffine {
            angle_deg,
            expand,
            interpolation,
            border_value,
        } => super::warp::rotate(buf, angle_deg, expand, interpolation, border_value),
        // Every per-value op: the scalar family, scale, relu, clamp, invert,
        // gamma, contrast, normalize and fused chains.
        ref per_value => crate::ops::elementwise::apply(buf, per_value),
    }
}

/// Reorder the channels of an `[H, W, C]` buffer: output channel `i` is
/// input channel `order[i]`. Pure data movement, in the input's own dtype —
/// the one channel reorder (`channel_swap`, and the colour conversions'
/// RGB <-> BGR).
pub fn apply_channel_swap(buf: &ViewBuffer, order: &[usize]) -> ViewBuffer {
    let shape = buf.shape();
    assert!(shape.len() == 3, "ChannelSwap requires 3D [H, W, C] input");
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    assert!(
        order.len() == c,
        "ChannelSwap order length {} must match channel count {}",
        order.len(),
        c
    );
    let contig = buf.to_contiguous();
    crate::core::dtype::with_dtype!(buf.dtype(), T => {
        let src = contig.as_slice::<T>();
        let mut out: Vec<T> = Vec::with_capacity(src.len());
        for pixel in src.chunks_exact(c) {
            out.extend(order.iter().map(|&i| pixel[i]));
        }
        ViewBuffer::from_vec_with_shape(out, vec![h, w, c])
    })
}

/// Whether [`apply_channel_merge`] can merge buffers of these shapes and dtypes
/// (CR-34): at least one input, every input `[H, W]` with the same H and W,
/// and all of one dtype (each is read as the first input's element type).
///
/// Over what is known of them (the planner's call; execution passes known
/// ones): two H (or W) sizes differ only when both are known, and two dtypes
/// only when both are, so an error is a verdict on a known fact.
pub fn validate_channel_merge(
    shapes: &[&[crate::ops::Dim]],
    dtypes: &[crate::PlannedDType],
) -> Result<(), crate::ops::validation::ValidationError> {
    use crate::ops::validation::ValidationError;
    use crate::PlannedDType;
    if shapes.is_empty() {
        return Err(ValidationError::InsufficientInputs {
            expected: 1,
            got: 0,
        });
    }
    // The first input that is not [H, W], or whose known H or W differs from
    // one known earlier.
    let mut hw: [Option<usize>; 2] = [None, None];
    for shape in shapes {
        let clash = shape.len() != 2
            || shape
                .iter()
                .zip(&mut hw)
                .any(|(d, seen)| match (d.known(), *seen) {
                    (Some(n), Some(m)) => n != m,
                    (Some(n), None) => {
                        *seen = Some(n);
                        false
                    }
                    (None, _) => false,
                });
        if clash {
            return Err(ValidationError::ShapeRequirement {
                requirement: "every channel_merge input [H, W] with the same H and W",
                got: shape.to_vec(),
            });
        }
    }
    let known: Vec<DType> = dtypes
        .iter()
        .filter_map(|d| match d {
            PlannedDType::Known(d) => Some(*d),
            PlannedDType::SomeFloat | PlannedDType::Unknown => None,
        })
        .collect();
    if known.iter().any(|d| *d != known[0]) {
        return Err(ValidationError::Generic {
            message: format!("channel_merge inputs must share one dtype, got {known:?}"),
        });
    }
    Ok(())
}

/// Merge multiple single-channel [H, W] buffers into a [H, W, C] buffer.
pub fn apply_channel_merge(buffers: &[&ViewBuffer]) -> ViewBuffer {
    assert!(
        !buffers.is_empty(),
        "ChannelMerge requires at least one input"
    );
    let h = buffers[0].shape()[0];
    let w = buffers[0].shape()[1];
    let c = buffers.len();

    // All buffers must be 2D [H, W] with matching dimensions
    for (i, buf) in buffers.iter().enumerate() {
        let s = buf.shape();
        assert!(
            s.len() == 2 && s[0] == h && s[1] == w,
            "ChannelMerge input {} has shape {:?}, expected [{}, {}]",
            i,
            s,
            h,
            w
        );
    }

    let contigs: Vec<ViewBuffer> = buffers.iter().map(|b| b.to_contiguous()).collect();
    // Pure data movement, in the inputs' own dtype (they share one:
    // `validate_channel_merge`).
    crate::core::dtype::with_dtype!(buffers[0].dtype(), T => {
        let planes: Vec<&[T]> = contigs.iter().map(|b| b.as_slice::<T>()).collect();
        let mut out: Vec<T> = Vec::with_capacity(h * w * c);
        for pixel in 0..h * w {
            out.extend(planes.iter().map(|plane| plane[pixel]));
        }
        ViewBuffer::from_vec_with_shape(out, vec![h, w, c])
    })
}

/// Convert a buffer to U8 for image operations.
///
/// This handles dtype promotion for image operations:
/// - F32/F64 in [0.0, 1.0] range: scale to [0, 255]
/// - F32/F64 outside range: clamp then scale
/// - Other integer types: cast directly
/// - U8: pass through
#[cfg(feature = "image_interop")]
fn convert_to_u8_for_image(buf: ViewBuffer) -> ViewBuffer {
    if buf.dtype() == DType::U8 {
        return buf;
    }

    let contig = buf.to_contiguous();
    let shape = crate::core::layout::Dims::from_slice(contig.shape());

    match contig.dtype() {
        DType::F32 => {
            let src = contig.as_slice::<f32>();
            // Scale from [0.0, 1.0] to [0, 255], clamping values outside range
            let new_data: Vec<u8> = src
                .iter()
                .map(|&x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::F64 => {
            let src = contig.as_slice::<f64>();
            let new_data: Vec<u8> = src
                .iter()
                .map(|&x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::U16 => {
            let src = contig.as_slice::<u16>();
            // Scale from [0, 65535] to [0, 255]
            let new_data: Vec<u8> = src.iter().map(|&x| (x >> 8) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::I16 => {
            let src = contig.as_slice::<i16>();
            let new_data: Vec<u8> = src.iter().map(|&x| x.clamp(0, 255) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::U32 => {
            let src = contig.as_slice::<u32>();
            let new_data: Vec<u8> = src.iter().map(|&x| (x.min(255)) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::I32 => {
            let src = contig.as_slice::<i32>();
            let new_data: Vec<u8> = src.iter().map(|&x| x.clamp(0, 255) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::I8 => {
            let src = contig.as_slice::<i8>();
            let new_data: Vec<u8> = src.iter().map(|&x| x.max(0) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        _ => {
            // For other types, use the cast method
            contig.cast(DType::U8)
        }
    }
}

/// Map our filter types to fast_image_resize algorithm types.
#[cfg(feature = "image_interop")]
#[inline]
fn to_fir_algorithm(filter: &FilterType) -> fir::ResizeAlg {
    match filter {
        // fir steps through source positions in floating point and breaks an
        // exact pixel-centre tie either way; nearest never reaches it.
        FilterType::Nearest => unreachable!(
            "nearest is resampled exactly by `execution::resample::nearest` (resize_strided)"
        ),
        FilterType::Triangle => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        FilterType::CatmullRom => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
        FilterType::Gaussian => fir::ResizeAlg::Convolution(fir::FilterType::Gaussian),
        FilterType::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
    }
}

/// Resize using fast_image_resize with SIMD optimization.
///
/// Nearest, of any dtype, is an exact gather (`execution::resample::nearest`):
/// fir's floating-point source positions break exact pixel-centre ties
/// either way. For the convolution filters, U8, U16 and F32 with 1–4
/// channels are fast_image_resize's own pixel types ([`resize_pixels`]). i8
/// and i16 are resized as F32 (which holds them exactly) and cast back; the
/// 32/64-bit integers and f64 are resampled in f64 by `execution::resample`.
/// The output keeps the input's dtype and rank.
#[cfg(feature = "image_interop")]
fn resize_strided(
    buf: ViewBuffer,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> ViewBuffer {
    use fir::pixels::{F32x2, F32x3, F32x4, U16x2, U16x3, U16x4, U8x2, U8x3, U8x4, F32, U16, U8};
    let (w, h) = (target_width, target_height);
    if filter == FilterType::Nearest {
        return super::resample::nearest(&buf, w as usize, h as usize);
    }
    let channels = buf.shape().get(2).copied().unwrap_or(1);
    match (buf.dtype(), channels) {
        (DType::U8, 1) => resize_pixels::<U8>(buf, w, h, filter),
        (DType::U8, 2) => resize_pixels::<U8x2>(buf, w, h, filter),
        (DType::U8, 3) => resize_pixels::<U8x3>(buf, w, h, filter),
        (DType::U8, 4) => resize_pixels::<U8x4>(buf, w, h, filter),
        (DType::U16, 1) => resize_pixels::<U16>(buf, w, h, filter),
        (DType::U16, 2) => resize_pixels::<U16x2>(buf, w, h, filter),
        (DType::U16, 3) => resize_pixels::<U16x3>(buf, w, h, filter),
        (DType::U16, 4) => resize_pixels::<U16x4>(buf, w, h, filter),
        (DType::F32, 1) => resize_pixels::<F32>(buf, w, h, filter),
        (DType::F32, 2) => resize_pixels::<F32x2>(buf, w, h, filter),
        (DType::F32, 3) => resize_pixels::<F32x3>(buf, w, h, filter),
        (DType::F32, 4) => resize_pixels::<F32x4>(buf, w, h, filter),
        (dtype @ (DType::U8 | DType::U16 | DType::F32), _) => unreachable!(
            "resize's contract (`ImageOp::validate`) refuses more than 4 channels, \
             so {dtype:?} with {channels} cannot reach the resampler"
        ),
        // i8/i16 resample exactly as f32 (`DType::accumulator`); the
        // 32/64-bit integers and f64 have no fir pixel type and resample in
        // f64 (`execution::resample`).
        (other, _) if other.accumulator() == DType::F32 => {
            resize_strided(buf.cast(DType::F32), w, h, filter).cast(other)
        }
        _ => super::resample::resample(&buf, w as usize, h as usize, filter),
    }
}

/// Resize an image of fast_image_resize pixel type `P` into a new buffer of
/// the same dtype and rank.
///
/// A view whose pixels are packed within each row (contiguous, a crop, a
/// vertical flip) is read where it lies ([`FirViewAdapter`]); any other layout
/// is packed first.
#[cfg(feature = "image_interop")]
fn resize_pixels<P>(
    buf: ViewBuffer,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> ViewBuffer
where
    P: fir::PixelTrait,
    P::Component: crate::core::dtype::ViewType + Default,
{
    use crate::interop::fir::{as_pixels_mut, FirViewAdapter};
    use crate::interop::ExternalView;

    let packed;
    let src = match FirViewAdapter::<P>::try_view(&buf) {
        Ok(view) => view,
        Err(crate::core::buffer::BufferError::IncompatibleLayout { .. }) => {
            packed = buf.to_contiguous();
            FirViewAdapter::<P>::try_view(&packed).expect("a packed buffer has packed rows")
        }
        Err(e) => panic!("resize: {e}"),
    };

    let mut out_shape = crate::core::layout::Dims::from_slice(buf.shape());
    out_shape[0] = target_height as usize;
    out_shape[1] = target_width as usize;
    let mut out = vec![P::Component::default(); out_shape.iter().product()];
    let mut dst = fir::images::TypedImage::<P>::from_pixels_slice(
        target_width,
        target_height,
        as_pixels_mut::<P>(&mut out),
    )
    .expect("the output holds target_width * target_height pixels");
    // Premultiplied exactly when the declaration says the last channel is
    // alpha (`ops::color::has_alpha`), not by fir's default for its 2/4
    // channel pixel types.
    let channels = buf.shape().get(2).copied().unwrap_or(1);
    let options = fir::ResizeOptions::new()
        .resize_alg(to_fir_algorithm(&filter))
        .use_alpha(crate::ops::color::has_alpha(channels));
    FIR_RESIZER.with(|cell| {
        cell.borrow_mut()
            .resize_typed(&src, &mut dst, &options)
            .expect("Resize failed");
    });
    ViewBuffer::from_vec_with_shape(out, out_shape)
}

#[cfg(feature = "image_interop")]
thread_local! {
    /// Reused across resize calls on the same worker thread to avoid
    /// reallocating fast_image_resize's internal scratch buffers on every row.
    /// `fir::Resizer` adapts to the pixel type and dimensions on each `resize`
    /// call, so a single instance safely serves U8/U16/F32 at any size. It is
    /// thread-local, so streaming morsel workers never share one.
    static FIR_RESIZER: std::cell::RefCell<fir::Resizer> =
        std::cell::RefCell::new(fir::Resizer::new());

    /// Scratch buffer for the horizontal pass of separable Gaussian blur.
    /// Holds f32 values for the current image (h × w × channels). Grows to
    /// fit the largest image seen per thread and is never shrunk — amortised
    /// O(1) allocation like TILE_EXTRACT_BUF in tiling.rs.
    static BLUR_HORIZ_BUF: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// The same slab for an f64-accumulating blur (`BlurAcc for f64`).
    static BLUR_HORIZ_BUF_F64: std::cell::RefCell<Vec<f64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Grayscale of any layout: BT.601, `Y = 0.299R + 0.587G + 0.114B` (u8 in
/// fixed point, `Y = (77R + 150G + 29B + 128) >> 8`), or a gray + alpha
/// image's gray channel, as a per-pixel map (`core::map`): any view is read
/// where it lies. More than four channels are read as the first three (BT.601
/// reads no others), through a view of them.
#[cfg(feature = "image_interop")]
fn grayscale_strided(buf: ViewBuffer) -> ViewBuffer {
    use crate::ops::color::Luma;
    let shape = buf.shape();
    let channels = shape.get(2).copied().unwrap_or(1);
    if channels == 1 {
        // Already gray. Packed, since every image op's output is planned
        // contiguous (a no-op when it already is).
        return buf.to_contiguous();
    }
    if channels > 4 {
        let (h, w) = (shape[0], shape[1]);
        return grayscale_strided(buf.slice(&[0, 0, 0], &[h, w, 3]));
    }
    fn gray<T: Luma>(buf: &ViewBuffer, channels: usize) -> ViewBuffer {
        match channels {
            2 => map_pixels::<T, T, 2, _>(buf, &Grayscale),
            3 => map_pixels::<T, T, 3, _>(buf, &Grayscale),
            4 => map_pixels::<T, T, 4, _>(buf, &Grayscale),
            other => unreachable!("grayscale maps 2-4 channels, not {other}"),
        }
    }
    with_dtype!(buf.dtype(), T => gray::<T>(&buf, channels))
}

/// Grayscale as a pixel map: the luma of a colour pixel (its first three
/// channels), the gray channel of a gray + alpha one (`color_channels(C) == 1`).
#[cfg(feature = "image_interop")]
struct Grayscale;

// SAFETY: `map_into` writes one value per pixel of `src`, into `dst`'s
// matching slot.
#[cfg(feature = "image_interop")]
unsafe impl<T: crate::ops::color::Luma, const C: usize> PixelMap<T, T, C> for Grayscale {
    #[inline(always)]
    fn map_into(&self, src: &[[T; C]], dst: &mut [MaybeUninit<T>]) {
        for (d, p) in dst.iter_mut().zip(src) {
            // `C` is a constant, so each instance keeps one arm. The blue
            // index is written `C.min(3) - 1` (2 for every `C >= 3`) only so
            // the discarded arm of gray + alpha stays in bounds.
            d.write(if crate::ops::color::color_channels(C) == 1 {
                p[0]
            } else {
                T::luma(p[0], p[1], p[C.min(3) - 1])
            });
        }
    }
}

/// Applies an image operation to a buffer.
///
/// The conversion strategy depends on the operation's ``working_dtype()``:
///
/// - ``Some(DType::U8)``: the operation requires U8 data (blur).
///   Float inputs in [0.0, 1.0] are scaled to [0, 255].
/// - ``None``: the operation works on the input's native dtype
///   (resize, rotate, grayscale, threshold).
///   The buffer is passed through unchanged.
///
/// Applies an image operation to a buffer.
///
/// Handles dtype promotion (converting to the op's working dtype before
/// dispatch) and validates the output dtype contract in debug builds.
#[cfg(feature = "image_interop")]
#[inline]
pub(crate) fn apply_image_inner(buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    let input_dtype = buf.dtype();

    // Only convert to U8 when the operation requires it.
    let work_buf = match op.working_dtype() {
        Some(DType::U8) => convert_to_u8_for_image(buf),
        Some(target) => buf.cast(target),
        None => buf, // Resize: use the input's native dtype
    };

    let result = apply_image_dispatch(work_buf, op.clone());

    debug_assert!(
        op.validate_output_dtype(input_dtype, result.dtype())
            .is_ok(),
        "ImageOp contract violation: {}",
        op.validate_output_dtype(input_dtype, result.dtype())
            .unwrap_err()
    );

    result
}

/// Threshold: `255` where an element exceeds `thresh`, else `0`, as a u8
/// element map (`core::map`): a u8 buffer only this op reads is written in
/// place, and any view is read where it lies.
///
/// A u8 input with the threshold inside the u8 range compares in u8 (`p > t`
/// for an integer `p` is `p > floor(t)`); any other integer compares exactly
/// exactly ([`Cut`]), and a float in `f64`.
#[cfg(feature = "image_interop")]
fn threshold_generic(buf: ViewBuffer, thresh: f64) -> ViewBuffer {
    let shape = buf.shape();

    // Validate: threshold only works on single-channel data
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "Threshold requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() first to convert multi-channel images to grayscale."
        );
    }

    if buf.dtype() == DType::U8 && (0.0..=255.0).contains(&thresh) {
        return map_owned::<u8, _>(buf, &ThresholdU8(thresh as u8));
    }
    with_dtype!(buf.dtype(), T => map_new(&buf, &Threshold::<T>::new(thresh)))
}

/// u8 threshold compared in u8.
#[cfg(feature = "image_interop")]
struct ThresholdU8(u8);

// SAFETY: `map_into` writes every element of `dst`.
#[cfg(feature = "image_interop")]
unsafe impl ElementMap<u8, u8> for ThresholdU8 {
    #[inline(always)]
    fn map_into(&self, src: &[u8], dst: &mut [MaybeUninit<u8>], _at: usize) {
        let t = self.0;
        for (d, &p) in dst.iter_mut().zip(src) {
            d.write(if p > t { 255 } else { 0 });
        }
    }
}

#[cfg(feature = "image_interop")]
impl ElementMapInPlace<u8> for ThresholdU8 {
    #[inline(always)]
    fn map_in_place(&self, data: &mut [u8]) {
        let t = self.0;
        for p in data.iter_mut() {
            *p = if *p > t { 255 } else { 0 };
        }
    }
}

/// Threshold of any dtype, compared exactly against the threshold
/// (`core::cut`, the one pixel-against-boundary comparison).
#[cfg(feature = "image_interop")]
struct Threshold<T>(Cut, PhantomData<fn(T)>);

#[cfg(feature = "image_interop")]
impl<T> Threshold<T> {
    fn new(thresh: f64) -> Self {
        Threshold(Cut::of(thresh), PhantomData)
    }
}

// SAFETY: `map_into` writes every element of `dst`.
#[cfg(feature = "image_interop")]
unsafe impl<T: ViewType + AgainstCut> ElementMap<T, u8> for Threshold<T> {
    #[inline(always)]
    fn map_into(&self, src: &[T], dst: &mut [MaybeUninit<u8>], _at: usize) {
        let cut = &self.0;
        for (d, &x) in dst.iter_mut().zip(src) {
            d.write(if x.above(cut) { 255 } else { 0 });
        }
    }
}

/// Check if a shape represents a single-channel image.
///
/// Valid single-channel shapes:
/// - `[H, W]` - 2D array
/// - `[H, W, 1]` - 3D with 1 channel
///
/// Invalid (multi-channel):
/// - `[H, W, C]` where C > 1
#[cfg(feature = "image_interop")]
#[inline]
fn is_single_channel(shape: &[usize]) -> bool {
    match shape.len() {
        2 => true,          // [H, W] - 2D is single channel
        3 => shape[2] == 1, // [H, W, 1] - explicit single channel
        _ => false,         // Other ranks not supported
    }
}

/// Get the number of channels from a shape.
#[cfg(feature = "image_interop")]
#[inline]
fn get_channel_count(shape: &[usize]) -> usize {
    match shape.len() {
        2 => 1,        // [H, W] - implicit single channel
        3 => shape[2], // [H, W, C]
        _ => 0,        // Invalid
    }
}

/// Grayscale and threshold against independent scalar references, over every
/// layout the kernels read differently: contiguous, cropped (dense rows with
/// a row stride), vertically flipped (negative row stride), horizontally
/// flipped and transposed (not dense rows, so materialised or walked per
/// pixel). Odd widths put a remainder after every vector loop.
#[cfg(all(test, feature = "image_interop"))]
mod grayscale_threshold_parity_tests {
    use super::apply_image_inner;
    use crate::core::buffer::ViewBuffer;
    use crate::core::dtype::DType;
    use crate::ops::{ImageOp, ImageOpKind};

    // (48, 70): more pixels than one walk scratch holds (8 KiB), so a
    // channel-first view is handed out in several runs.
    const SIZES: [(usize, usize); 6] = [(1, 1), (3, 5), (7, 33), (16, 17), (9, 64), (48, 70)];

    /// Pseudo-random bytes (an LCG's high bits). An arithmetic sequence such
    /// as `(i * 7919) % 256` correlates neighbouring channels so strongly
    /// that no pixel of it lands on a rounding boundary of the luma sum.
    fn pattern(len: usize) -> Vec<u8> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 56) as u8
            })
            .collect()
    }

    /// The u16 fixed-point luma equals the u32 formula on every RGB triple.
    #[test]
    fn luma_u8_matches_the_u32_formula_on_every_pixel() {
        for r in 0..=255u8 {
            for g in 0..=255u8 {
                for b in 0..=255u8 {
                    let (r32, g32, b32) = (u32::from(r), u32::from(g), u32::from(b));
                    let expected = ((77 * r32 + 150 * g32 + 29 * b32 + 128) >> 8) as u8;
                    assert_eq!(
                        <u8 as crate::ops::color::Luma>::luma(r, g, b),
                        expected,
                        "({r}, {g}, {b})"
                    );
                }
            }
        }
    }

    /// The layouts a `[h, w, c]` (or `[h, w]`) buffer reaches a kernel in,
    /// each a view over a larger or reordered parent.
    fn layouts(parent: &ViewBuffer) -> Vec<(&'static str, ViewBuffer)> {
        let shape = crate::core::layout::Dims::from_slice(parent.shape());
        let (h, w) = (shape[0], shape[1]);
        let mut out = vec![
            ("contiguous", parent.clone()),
            ("flip_v", parent.flip(&[0])),
            ("flip_h", parent.flip(&[1])),
        ];
        if h > 2 && w > 2 {
            let mut start = vec![1, 1];
            let mut end = vec![h - 1, w - 1];
            if shape.len() == 3 {
                start.push(0);
                end.push(shape[2]);
            }
            out.push(("crop", parent.slice(&start, &end)));
        }
        let mut perm: Vec<usize> = vec![1, 0];
        if shape.len() == 3 {
            perm.push(2);
        }
        out.push(("transpose", parent.permute(&perm)));
        if shape.len() == 3 && shape[2] > 1 {
            // Stored channel-first, viewed channels-last: channels
            // `h · w` elements apart, so the walk packs single elements and
            // a scratch-sized run need not end on a pixel boundary.
            let c = shape[2];
            let packed = parent.to_contiguous();
            out.push((
                "channel_first",
                crate::core::dtype::with_dtype!(parent.dtype(), T => {
                    let src = packed.as_slice::<T>();
                    let chw: Vec<T> = (0..c)
                        .flat_map(|k| src.iter().skip(k).step_by(c).copied())
                        .collect();
                    ViewBuffer::from_vec_with_shape(chw, vec![c, h, w]).permute(&[1, 2, 0])
                }),
            ));
        }
        out
    }

    fn run(buf: ViewBuffer, kind: ImageOpKind) -> ViewBuffer {
        apply_image_inner(buf, ImageOp { kind })
    }

    fn reference_luma_u8(p: &[u8]) -> u8 {
        if p.len() == 2 {
            return p[0];
        }
        let (r, g, b) = (u32::from(p[0]), u32::from(p[1]), u32::from(p[2]));
        ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8
    }

    #[test]
    fn u8_grayscale_matches_the_fixed_point_reference() {
        for (h, w) in SIZES {
            for c in [2usize, 3, 4] {
                let parent = ViewBuffer::from_vec_with_shape(pattern(h * w * c), vec![h, w, c]);
                for (layout, view) in layouts(&parent) {
                    let packed = view.to_contiguous();
                    let expected: Vec<u8> = packed
                        .as_slice::<u8>()
                        .chunks_exact(c)
                        .map(reference_luma_u8)
                        .collect();
                    let got = run(view.clone(), ImageOpKind::Grayscale);
                    let (vh, vw) = (view.shape()[0], view.shape()[1]);
                    assert_eq!(got.shape(), &[vh, vw, 1], "{layout} {h}x{w}x{c}");
                    assert_eq!(
                        got.to_contiguous().as_slice::<u8>(),
                        &expected[..],
                        "{layout} {h}x{w}x{c}"
                    );
                }
            }
        }
    }

    /// Non-u8 grayscale computes BT.601 in f64 and converts back, rounding
    /// and clamping integer targets.
    #[test]
    fn typed_grayscale_matches_the_f64_reference() {
        for (h, w) in SIZES {
            for c in [3usize, 4] {
                let src: Vec<f32> = pattern(h * w * c)
                    .iter()
                    .map(|&v| f32::from(v) * 1.37 - 20.25)
                    .collect();
                let parent = ViewBuffer::from_vec_with_shape(src, vec![h, w, c]);
                for (layout, view) in layouts(&parent) {
                    let packed = view.to_contiguous();
                    let expected: Vec<f32> = packed
                        .as_slice::<f32>()
                        .chunks_exact(c)
                        .map(|p| {
                            (0.299 * f64::from(p[0])
                                + 0.587 * f64::from(p[1])
                                + 0.114 * f64::from(p[2])) as f32
                        })
                        .collect();
                    let got = run(view, ImageOpKind::Grayscale);
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(got.to_contiguous().as_slice::<f32>()),
                        bits(&expected),
                        "{layout} {h}x{w}x{c}"
                    );
                }
                let u16s: Vec<u16> = pattern(h * w * c)
                    .iter()
                    .map(|&v| u16::from(v) * 257)
                    .collect();
                let parent = ViewBuffer::from_vec_with_shape(u16s.clone(), vec![h, w, c]);
                let expected: Vec<u16> = u16s
                    .chunks_exact(c)
                    .map(|p| {
                        let l = 0.299 * f64::from(p[0])
                            + 0.587 * f64::from(p[1])
                            + 0.114 * f64::from(p[2]);
                        l.round().clamp(0.0, 65535.0) as u16
                    })
                    .collect();
                let got = run(parent, ImageOpKind::Grayscale);
                assert_eq!(got.as_slice::<u16>(), &expected[..], "u16 {h}x{w}x{c}");
            }
        }
    }

    /// A gray + alpha image's grayscale is its gray channel, for every dtype:
    /// the u8 kernel always did this, while the typed one computed BT.601
    /// over (gray, alpha, alpha), mixing opacity into intensity.
    #[test]
    fn gray_alpha_grayscale_is_the_gray_channel_for_every_dtype() {
        let gray_alpha: Vec<u16> = vec![1000, 65535, 20, 0, 40000, 7];
        let buf = ViewBuffer::from_vec_with_shape(gray_alpha, vec![1, 3, 2]);
        let got = run(buf, ImageOpKind::Grayscale);
        assert_eq!(got.as_slice::<u16>(), &[1000, 20, 40000]);
        let buf = ViewBuffer::from_vec_with_shape(vec![0.25f32, 1.0, 0.5, 0.0], vec![2, 1, 2]);
        let got = run(buf, ImageOpKind::Grayscale);
        assert_eq!(got.as_slice::<f32>(), &[0.25, 0.5]);
    }

    #[test]
    fn threshold_matches_the_comparison_reference() {
        for (h, w) in SIZES {
            for rank3 in [false, true] {
                let shape = if rank3 { vec![h, w, 1] } else { vec![h, w] };
                let parent = ViewBuffer::from_vec_with_shape(pattern(h * w), shape);
                for (layout, view) in layouts(&parent) {
                    let packed = view.to_contiguous();
                    for t in [-1.0f64, 0.0, 127.5, 128.0, 254.9, 255.0, 300.0] {
                        let expected: Vec<u8> = packed
                            .as_slice::<u8>()
                            .iter()
                            .map(|&v| if f64::from(v) > t { 255 } else { 0 })
                            .collect();
                        let got = run(view.clone(), ImageOpKind::Threshold { value: t });
                        assert_eq!(got.shape(), view.shape(), "{layout} rank3={rank3}");
                        assert_eq!(got.dtype(), DType::U8);
                        assert_eq!(
                            got.to_contiguous().as_slice::<u8>(),
                            &expected[..],
                            "{layout} {h}x{w} rank3={rank3} t={t}"
                        );
                    }
                }
            }
        }
    }

    /// An integer pixel is compared with the threshold exactly (`p > t` iff
    /// `p > floor(t)`), not rounded to f64 first: above 2**53 neighbouring
    /// pixels share an f64, so the rounded comparison answered for the
    /// neighbour.
    #[test]
    fn integer_threshold_is_exact_beyond_f64_precision() {
        // 18442240474082181119 as f64 is ...120: the rounded comparison put
        // ...119 above it.
        let u = ViewBuffer::from_vec_with_shape(
            vec![
                18442240474082181119u64,
                18442240474082181120,
                18442240474082183169,
                u64::MAX,
            ],
            vec![1, 4],
        );
        let i = ViewBuffer::from_vec_with_shape(
            vec![3797082577976980i64, 3797082577976981, i64::MAX, i64::MIN],
            vec![1, 4],
        );
        // 2**53 + 1 rounds to 2**53 as f64, so the rounded comparison said it
        // does not exceed 2**53.
        let w = ViewBuffer::from_vec_with_shape(
            vec![
                9007199254740991i64,
                9007199254740992,
                9007199254740993,
                9007199254740995,
            ],
            vec![1, 4],
        );
        let cases: [(&ViewBuffer, f64, [u8; 4]); 12] = [
            (&w, 9007199254740992.0, [0, 0, 255, 255]),
            (&w, 9007199254740994.0, [0, 0, 0, 255]),
            (&u, 18442240474082181119u64 as f64, [0, 0, 255, 255]),
            (&u, 0.0, [255; 4]),
            (&u, -1e300, [255; 4]),
            (&u, 1e300, [0; 4]),
            (&u, f64::NAN, [0; 4]),
            (&i, 3797082577976980.5, [0, 255, 255, 0]),
            (&i, -0.5, [255, 255, 255, 0]),
            (&i, 9.3e18, [0; 4]),
            (&i, -9.3e18, [255; 4]),
            (&i, f64::NAN, [0; 4]),
        ];
        for (buf, t, expected) in cases {
            let got = run(buf.clone(), ImageOpKind::Threshold { value: t });
            assert_eq!(
                got.as_slice::<u8>(),
                &expected[..],
                "{:?} t={t}",
                buf.dtype()
            );
        }
    }

    /// Any other dtype compares the element's value, and still writes u8, for
    /// every layout.
    #[test]
    fn typed_threshold_matches_the_comparison_reference() {
        for (h, w) in SIZES {
            let values: Vec<f64> = pattern(h * w)
                .iter()
                .map(|&v| f64::from(v) * 3.5 - 200.0)
                .collect();
            let parents = [
                ViewBuffer::from_vec_with_shape(
                    values.iter().map(|&v| v as i16).collect::<Vec<_>>(),
                    vec![h, w, 1],
                ),
                ViewBuffer::from_vec_with_shape(
                    values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                    vec![h, w],
                ),
                ViewBuffer::from_vec_with_shape(
                    values
                        .iter()
                        .map(|&v| (v + 200.0) as u64)
                        .collect::<Vec<_>>(),
                    vec![h, w, 1],
                ),
            ];
            for parent in parents {
                for (layout, view) in layouts(&parent) {
                    let as_f64: Vec<f64> = crate::core::dtype::with_dtype!(view.dtype(), T => {
                        view.to_contiguous().as_slice::<T>().iter().map(|x| num_traits::ToPrimitive::to_f64(x).unwrap()).collect()
                    });
                    for t in [-50.0f64, 0.0, 99.5, 300.0] {
                        let expected: Vec<u8> = as_f64
                            .iter()
                            .map(|&v| if v > t { 255 } else { 0 })
                            .collect();
                        let got = run(view.clone(), ImageOpKind::Threshold { value: t });
                        assert_eq!(got.dtype(), DType::U8);
                        assert_eq!(
                            got.to_contiguous().as_slice::<u8>(),
                            &expected[..],
                            "{:?} {layout} {h}x{w} t={t}",
                            view.dtype()
                        );
                    }
                }
            }
        }
    }
}

/// Clamp an `f64` value to the representable range of the given integer
/// `DType`: the store the pre-M5 kernels used, kept for the verbatim oracles
/// of the warp and blur tests. Every kernel now stores by `CastFrom`.
#[cfg(test)]
pub(super) fn clamp_for_dtype(v: f64, dtype: DType) -> f64 {
    match dtype {
        // Round before clamping so that e.g. 127.9999 → 128, not 127.
        // NumCast::from truncates (floor) for positive floats, so without rounding
        // Gaussian convolution on a solid-128 image gives 127.
        DType::U8 => v.round().clamp(0.0, u8::MAX as f64),
        DType::I8 => v.round().clamp(i8::MIN as f64, i8::MAX as f64),
        DType::U16 => v.round().clamp(0.0, u16::MAX as f64),
        DType::I16 => v.round().clamp(i16::MIN as f64, i16::MAX as f64),
        DType::U32 => v.round().clamp(0.0, u32::MAX as f64),
        DType::I32 => v.round().clamp(i32::MIN as f64, i32::MAX as f64),
        DType::U64 => v.round().clamp(0.0, u64::MAX as f64),
        DType::I64 => v.round().clamp(i64::MIN as f64, i64::MAX as f64),
        DType::F32 | DType::F64 => v,
    }
}

/// Inner implementation of image operations (without tiling logic).
#[cfg(feature = "image_interop")]
#[inline]
fn apply_image_dispatch(work_buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    match op.kind {
        ImageOpKind::Threshold { value } => threshold_generic(work_buf, value),
        ImageOpKind::Grayscale => grayscale_strided(work_buf),
        ImageOpKind::Resize {
            width,
            height,
            filter,
        } => resize_strided(work_buf, width, height, filter),
        ImageOpKind::Canny {
            low_threshold,
            high_threshold,
        } => apply_canny(work_buf, low_threshold, high_threshold),
        ImageOpKind::HistogramEqualize => apply_histogram_equalize(work_buf),
        ImageOpKind::Erode { ksize, iterations } => apply_erode(work_buf, ksize, iterations),
        ImageOpKind::Dilate { ksize, iterations } => apply_dilate(work_buf, ksize, iterations),
        ImageOpKind::MorphGradient { ksize } => apply_morph_gradient(work_buf, ksize),
        ImageOpKind::Blur { sigma } => {
            let rows_buf = work_buf.to_dense_rows();
            // Every dtype in its own element type, accumulating in its
            // `DType::accumulator`.
            let dtype = rows_buf.dtype();
            with_dtype!(dtype, T => match dtype.accumulator() {
                DType::F64 => separable_gaussian_blur_typed::<T, f64>(&rows_buf, sigma),
                _ => separable_gaussian_blur_typed::<T, f32>(&rows_buf, sigma),
            })
        }
        // Deferred resizes: dimensions come from the kind's shape — the same
        // authority the planner reads — then the shared resize kernel runs.
        ref kind @ (ImageOpKind::ResizeScale { .. }
        | ImageOpKind::ResizeToHeight { .. }
        | ImageOpKind::ResizeToWidth { .. }
        | ImageOpKind::ResizeMax { .. }
        | ImageOpKind::ResizeMin { .. }) => {
            let shape = work_buf.shape();
            let [h, w] = kind.shape().concrete(&[&shape[..2]])[..] else {
                unreachable!("an H/W shape over a rank-2 input is rank 2")
            };
            let filter = match kind {
                ImageOpKind::ResizeScale { filter, .. }
                | ImageOpKind::ResizeToHeight { filter, .. }
                | ImageOpKind::ResizeToWidth { filter, .. }
                | ImageOpKind::ResizeMax { filter, .. }
                | ImageOpKind::ResizeMin { filter, .. } => *filter,
                _ => unreachable!(),
            };
            resize_strided(work_buf, w as u32, h as u32, filter)
        }
        ImageOpKind::Pad {
            top,
            bottom,
            left,
            right,
            value,
            mode,
        } => crate::ops::pad::pad(&work_buf, top, bottom, left, right, value, mode),
        ImageOpKind::PadToSize {
            height,
            width,
            position,
            value,
        } => crate::ops::pad::pad_to_size(&work_buf, height, width, position, value),
        ImageOpKind::Letterbox {
            height,
            width,
            value,
            filter,
        } => {
            let shape = work_buf.shape();
            let (fit_h, fit_w) =
                crate::ops::image::letterbox_fit(shape[0], shape[1], height, width);
            let resized = resize_strided(work_buf, fit_w as u32, fit_h as u32, filter);
            crate::ops::pad::pad_to_size(
                &resized,
                height,
                width,
                crate::ops::pad::PadPosition::Center,
                value,
            )
        }
        ImageOpKind::ChannelSwap { ref order } => {
            let order: Vec<usize> = order.iter().map(|&i| i as usize).collect();
            apply_channel_swap(&work_buf, &order)
        }
    }
}

/// Builds a normalised 1-D Gaussian kernel of radius `ceil(3σ)`.
/// The radius formula matches the halo declaration in `ops/image.rs`.
#[cfg(feature = "image_interop")]
fn gaussian_kernel_1d<F: BlurAcc>(sigma: f32) -> Vec<F> {
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
#[cfg(feature = "image_interop")]
trait BlurAcc:
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

#[cfg(feature = "image_interop")]
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

#[cfg(feature = "image_interop")]
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
/// the behaviour of `morph_minmax_typed` in the same file.
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
#[cfg(feature = "image_interop")]
/// The separable Gaussian blur, run through [`dispatch`] so the whole body
/// (conversion, both passes, the output clamp) has an AVX2 build (CR-35:
/// ~1.4x over the wheels' SSE2 baseline; dispatching only the row axpy
/// measured slower).
fn separable_gaussian_blur_typed<T, F>(contig_buf: &ViewBuffer, sigma: f32) -> ViewBuffer
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

#[cfg(feature = "image_interop")]
struct SeparableBlur<'a, T, F> {
    buf: &'a ViewBuffer,
    sigma: f32,
    _elem: std::marker::PhantomData<fn() -> (T, F)>,
}

#[cfg(feature = "image_interop")]
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

#[cfg(feature = "image_interop")]
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

#[cfg(feature = "image_interop")]
#[cfg(test)]
mod blur_dispatch_tests {
    use super::{separable_gaussian_blur_body, separable_gaussian_blur_typed};
    use crate::core::buffer::ViewBuffer;

    use super::{clamp_for_dtype, gaussian_kernel_1d, DType, ImageOp, ImageOpKind, BLUR_HORIZ_BUF};

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
                    let got = super::apply_image_inner(
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

#[cfg(feature = "image_interop")]
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

#[cfg(not(feature = "image_interop"))]
pub(crate) fn apply_image_inner(_buf: ViewBuffer, _op: ImageOp) -> ViewBuffer {
    panic!("Image operations require the 'image_interop' feature");
}

// ============================================================
// Morphological Operations (Erode / Dilate / Gradient)
// ============================================================

/// Morphological erosion: output = local minimum over ksize×ksize neighborhood.
///
/// Uses replicate border handling (edge pixels are replicated).
/// Operates on single-channel data only; panics for multi-channel input.
/// Dtype-generic: dispatches per element type.
#[cfg(feature = "image_interop")]
fn apply_erode(buf: ViewBuffer, ksize: u32, iterations: u32) -> ViewBuffer {
    let shape = buf.shape();
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "Erode requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() or .threshold() first."
        );
    }
    morph_iterated(buf, ksize, iterations, MorphKind::Min)
}

/// Morphological dilation: output = local maximum over ksize×ksize neighborhood.
///
/// Uses replicate border handling (edge pixels are replicated).
/// Operates on single-channel data only; panics for multi-channel input.
/// Dtype-generic: dispatches per element type.
#[cfg(feature = "image_interop")]
fn apply_dilate(buf: ViewBuffer, ksize: u32, iterations: u32) -> ViewBuffer {
    let shape = buf.shape();
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "Dilate requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() or .threshold() first."
        );
    }
    morph_iterated(buf, ksize, iterations, MorphKind::Max)
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
/// (`tests/morph_ref.rs`, NaN and signed-zero cases).
#[cfg(feature = "image_interop")]
fn morph_iterated(buf: ViewBuffer, ksize: u32, iterations: u32, kind: MorphKind) -> ViewBuffer {
    let radius = (ksize / 2) as usize;
    if matches!(buf.dtype(), DType::F32 | DType::F64) {
        let mut result = buf;
        for _ in 0..iterations {
            result = morph_minmax_pass(&result, radius, kind);
        }
        return result;
    }
    morph_minmax_pass(&buf, radius * iterations as usize, kind)
}

/// Morphological gradient: dilate(input) − erode(input), clamped to valid range.
///
/// Produces an edge outline by computing the difference between dilation and
/// erosion. Output dtype matches input.
#[cfg(feature = "image_interop")]
fn apply_morph_gradient(buf: ViewBuffer, ksize: u32) -> ViewBuffer {
    let shape = buf.shape();
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "MorphGradient requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() or .threshold() first."
        );
    }
    let radius = (ksize / 2) as usize;
    let dilated = morph_minmax_pass(&buf, radius, MorphKind::Max);
    let eroded = morph_minmax_pass(&buf, radius, MorphKind::Min);

    morph_subtract(&dilated, &eroded)
}

#[cfg(feature = "image_interop")]
#[derive(Clone, Copy)]
enum MorphKind {
    Min,
    Max,
}

/// Single-pass min or max filter over a `(2 * radius + 1)`-square
/// rectangular structuring element.
///
/// Uses a separable approach (row pass then column pass) for efficiency.
/// Border handling: replicate edge pixels.
#[cfg(feature = "image_interop")]
fn morph_minmax_pass(buf: &ViewBuffer, radius: usize, kind: MorphKind) -> ViewBuffer {
    if radius == 0 {
        return buf.clone();
    }
    let dtype = buf.dtype();
    match dtype {
        DType::U8 => morph_dispatch::<u8>(buf, radius, kind),
        DType::I8 => morph_dispatch::<i8>(buf, radius, kind),
        DType::U16 => morph_dispatch::<u16>(buf, radius, kind),
        DType::I16 => morph_dispatch::<i16>(buf, radius, kind),
        DType::U32 => morph_dispatch::<u32>(buf, radius, kind),
        DType::I32 => morph_dispatch::<i32>(buf, radius, kind),
        DType::F32 => morph_dispatch::<f32>(buf, radius, kind),
        DType::F64 => morph_dispatch::<f64>(buf, radius, kind),
        DType::U64 => morph_dispatch::<u64>(buf, radius, kind),
        DType::I64 => morph_dispatch::<i64>(buf, radius, kind),
    }
}

/// Select the const-generic Min/Max instantiation for one element type.
#[cfg(feature = "image_interop")]
fn morph_dispatch<T>(buf: &ViewBuffer, radius: usize, kind: MorphKind) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + Copy + PartialOrd,
{
    match kind {
        MorphKind::Min => morph_minmax_typed::<T, true>(buf, radius),
        MorphKind::Max => morph_minmax_typed::<T, false>(buf, radius),
    }
}

/// One Min/Max fold step with the comparison kind monomorphized out of the
/// loop.
///
/// The window's running extreme: [`crate::ops::util`]'s NaN rule, so a NaN
/// anywhere in the window is the result (it used to be kept as the centre
/// but skipped as a neighbour). A tie keeps the incumbent `val`.
#[cfg(feature = "image_interop")]
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
/// Bit-exact vs the pre-split implementation (see tests/morph_ref.rs).
#[cfg(feature = "image_interop")]
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
        .expect("to_dense_rows packs the rows");

    // ── Row pass ─────────────────────────────────────────────────────────
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

/// Element-wise saturating subtraction: result = a − b, clamped to valid range.
#[cfg(feature = "image_interop")]
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
#[cfg(feature = "image_interop")]
trait SubFloorZero: Copy {
    fn sub_floor_zero(self, other: Self) -> Self;
}

#[cfg(feature = "image_interop")]
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
#[cfg(feature = "image_interop")]
sub_floor_zero_int!(u8, i8, u16, i16, u32, i32, u64, i64);

#[cfg(feature = "image_interop")]
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
#[cfg(feature = "image_interop")]
sub_floor_zero_float!(f32, f64);

// ============================================================
// Canny Edge Detection
// ============================================================

/// Canny edge detection, computed as `cv2.Canny(image, low, high)` computes it
/// (aperture 3, L1 gradient) — `tests/reference/test_canny_ref.py` holds the
/// two to the same edge map, pixel for pixel:
///
/// 1. 3×3 Sobel `dx`, `dy` with a replicated border, and no pre-blur (blur
///    first with `.blur()`, as with OpenCV). A multi-channel input takes, per
///    pixel, the colour channel whose `|dx| + |dy|` is largest (the first on a
///    tie); an alpha channel ([`color_channels`]) is not an input.
/// 2. Non-maximum suppression over four directions, chosen by OpenCV's
///    fixed-point tangent test (`|dy|·2¹⁵` against `|dx|·TG22`), with its
///    asymmetric comparisons (`>` towards one neighbour, `>=` towards the
///    other; `>` both ways on a diagonal). Magnitude outside the image is 0.
/// 3. Double threshold (`m > low` is a candidate, `m > high` a seed; a low
///    threshold above the high one is swapped) and 8-connected hysteresis.
///
/// A u8 image runs in `i32`, OpenCV's own arithmetic, and so matches it bit
/// for bit; other dtypes run the same definition in `f64` ([`canny_core`]).
/// Output is U8 `[H, W, 1]`, 0 or 255.
#[cfg(feature = "image_interop")]
fn apply_canny(buf: ViewBuffer, low_threshold: f32, high_threshold: f32) -> ViewBuffer {
    let shape = crate::core::layout::Dims::from_slice(buf.shape());
    let (h, w) = (shape[0], shape[1]);
    let channels = shape.get(2).copied().unwrap_or(1);
    let used = crate::ops::color::color_channels(channels);
    let (low, high) = {
        let (a, b) = (f64::from(low_threshold), f64::from(high_threshold));
        if a > b {
            (b, a)
        } else {
            (a, b)
        }
    };
    let contig = buf.to_contiguous();
    let edges = if contig.dtype() == DType::U8 {
        // OpenCV's own arithmetic: exact integers. For an integer magnitude,
        // `m > t` and OpenCV's `m > floor(t)` agree for every threshold t.
        let src = contig.as_slice::<u8>();
        canny_core::<i32>(h, w, channels, used, |i| i32::from(src[i]), low, high)
    } else {
        let plane = contig.cast(DType::F64);
        let src = plane.as_slice::<f64>();
        canny_core::<f64>(h, w, channels, used, |i| src[i], low, high)
    };
    ViewBuffer::from_vec_with_shape(edges, vec![h, w, 1])
}

/// The arithmetic Canny needs: `i32` for u8 input (OpenCV's own), `f64` for
/// every other dtype. Both are exact on integer input.
#[cfg(feature = "image_interop")]
trait CannyNum:
    Copy
    + Default
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    const TWO: Self;
    /// OpenCV's `TG22 = (int)(tan(22.5°) · 2¹⁵ + 0.5)`.
    const TG22: Self;
    const SHIFT: Self; // 2¹⁵
    fn abs(self) -> Self;
    fn above(self, threshold: f64) -> bool;
}

#[cfg(feature = "image_interop")]
impl CannyNum for i32 {
    const TWO: Self = 2;
    const TG22: Self = 13573;
    const SHIFT: Self = 1 << 15;
    fn abs(self) -> Self {
        i32::abs(self)
    }
    fn above(self, threshold: f64) -> bool {
        f64::from(self) > threshold
    }
}

#[cfg(feature = "image_interop")]
impl CannyNum for f64 {
    const TWO: Self = 2.0;
    const TG22: Self = 13573.0;
    const SHIFT: Self = 32768.0;
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn above(self, threshold: f64) -> bool {
        self > threshold
    }
}

/// Canny over an `[h, w, channels]` image read through `at(flat index)`; see
/// [`apply_canny`]. Every plane is padded by one pixel — the source by
/// replication (Sobel's border), the magnitude by zeros and the edge map by
/// "not an edge" (OpenCV's) — so no inner loop tests a bound.
#[cfg(feature = "image_interop")]
fn canny_core<T: CannyNum>(
    h: usize,
    w: usize,
    channels: usize,
    used: usize,
    at: impl Fn(usize) -> T,
    low: f64,
    high: f64,
) -> Vec<u8> {
    if h == 0 || w == 0 {
        return Vec::new();
    }
    let pw = w + 2;
    let padded = |y: usize, x: usize| (y + 1) * pw + (x + 1);

    // 1. Sobel per colour channel on a replicate-padded plane; keep, per
    //    pixel, the channel with the largest |dx| + |dy| (the first on a tie).
    let mut src = vec![T::default(); (h + 2) * pw];
    let mut dx = vec![T::default(); (h + 2) * pw];
    let mut dy = vec![T::default(); (h + 2) * pw];
    let mut mag = vec![T::default(); (h + 2) * pw]; // zero border
    for k in 0..used {
        for py in 0..h + 2 {
            let y = py.saturating_sub(1).min(h - 1);
            for px in 0..pw {
                let x = px.saturating_sub(1).min(w - 1);
                src[py * pw + px] = at((y * w + x) * channels + k);
            }
        }
        for y in 0..h {
            let (up, row, down) = (y * pw, (y + 1) * pw, (y + 2) * pw);
            for x in 0..w {
                let c = x + 1;
                let gx = (src[up + c + 1] + T::TWO * src[row + c + 1] + src[down + c + 1])
                    - (src[up + c - 1] + T::TWO * src[row + c - 1] + src[down + c - 1]);
                let gy = (src[down + c - 1] + T::TWO * src[down + c] + src[down + c + 1])
                    - (src[up + c - 1] + T::TWO * src[up + c] + src[up + c + 1]);
                let m = gx.abs() + gy.abs();
                let i = row + c;
                if k == 0 || m > mag[i] {
                    (dx[i], dy[i], mag[i]) = (gx, gy, m);
                }
            }
        }
    }

    // 2 + 3. Suppression and classification. 0 = candidate, 1 = none, 2 = edge.
    const CANDIDATE: u8 = 0;
    const NONE: u8 = 1;
    const EDGE: u8 = 2;
    let mut map = vec![NONE; (h + 2) * pw];
    let mut stack: Vec<usize> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = padded(y, x);
            let m = mag[i];
            if !m.above(low) {
                continue;
            }
            let (ax, ay) = (dx[i].abs(), dy[i].abs() * T::SHIFT);
            let tg22x = ax * T::TG22;
            let is_max = if ay < tg22x {
                m > mag[i - 1] && m >= mag[i + 1]
            } else if ay > tg22x + ax * T::TWO * T::SHIFT {
                m > mag[i - pw] && m >= mag[i + pw]
            } else if (dx[i] < T::default()) != (dy[i] < T::default()) {
                m > mag[i - pw + 1] && m > mag[i + pw - 1]
            } else {
                m > mag[i - pw - 1] && m > mag[i + pw + 1]
            };
            if is_max {
                if m.above(high) {
                    map[i] = EDGE;
                    stack.push(i);
                } else {
                    map[i] = CANDIDATE;
                }
            }
        }
    }

    // Hysteresis: grow every edge through 8-connected candidates. The border
    // is NONE, so a neighbour index never leaves the padded plane.
    while let Some(i) = stack.pop() {
        for n in [
            i - pw - 1,
            i - pw,
            i - pw + 1,
            i - 1,
            i + 1,
            i + pw - 1,
            i + pw,
            i + pw + 1,
        ] {
            if map[n] == CANDIDATE {
                map[n] = EDGE;
                stack.push(n);
            }
        }
    }

    let mut edges = Vec::with_capacity(h * w);
    for y in 0..h {
        let row = &map[padded(y, 0)..padded(y, 0) + w];
        edges.extend(row.iter().map(|&e| if e == EDGE { 255u8 } else { 0 }));
    }
    edges
}

// ============================================================
// Histogram Equalization
// ============================================================

/// Histogram equalization for contrast enhancement.
///
/// Computes the histogram, cumulative distribution, then maps each pixel
/// through the normalized CDF. Operates per-channel for multi-channel images.
/// Input must be U8 (enforced by `working_dtype`). Output is U8.
#[cfg(feature = "image_interop")]
fn apply_histogram_equalize(buf: ViewBuffer) -> ViewBuffer {
    let contig = buf.to_contiguous();
    let shape = contig.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape.get(2).copied().unwrap_or(1);
    let count = contig.layout.num_elements();
    let src = contig.as_slice::<u8>();

    let total_pixels = h * w;
    let mut output = vec![0u8; count];

    for ch in 0..c {
        // Compute histogram for this channel
        let mut hist = [0u32; 256];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * c + ch;
                hist[src[idx] as usize] += 1;
            }
        }

        // Compute CDF
        let mut cdf = [0u32; 256];
        cdf[0] = hist[0];
        for i in 1..256 {
            cdf[i] = cdf[i - 1] + hist[i];
        }

        // Find minimum non-zero CDF value
        let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);

        // Map pixels through equalized CDF
        let denominator = (total_pixels as f64 - cdf_min as f64).max(1.0);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * c + ch;
                let val = src[idx] as usize;
                let equalized =
                    ((cdf[val] as f64 - cdf_min as f64) / denominator * 255.0).round() as u8;
                output[idx] = equalized;
            }
        }
    }

    if c == 1 && shape.len() == 2 {
        ViewBuffer::from_vec_with_shape(output, vec![h, w])
    } else {
        ViewBuffer::from_vec_with_shape(output, vec![h, w, c])
    }
}

// ============================================================
// Perceptual Hash Operations
// ============================================================

#[cfg(feature = "perceptual_hash")]
use crate::ops::phash::PerceptualHashOp;

/// Applies a perceptual hash operation to a buffer.
///
/// Perceptual hashing requires the buffer to be in image format.
/// The output is a 1D u8 buffer containing the hash bytes.
#[cfg(feature = "perceptual_hash")]
pub fn apply_perceptual_hash(buf: ViewBuffer, op: PerceptualHashOp) -> ViewBuffer {
    // Convert to U8 if needed (perceptual hash expects image format)
    let work_buf = convert_to_u8_for_image(buf);

    // Execute the perceptual hash operation
    op.execute(&work_buf)
}

#[cfg(not(feature = "perceptual_hash"))]
pub fn apply_perceptual_hash(
    _buf: ViewBuffer,
    _op: crate::ops::phash::PerceptualHashOp,
) -> ViewBuffer {
    panic!("Perceptual hash operations require the 'perceptual_hash' feature");
}

#[cfg(all(test, feature = "image_interop"))]
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
