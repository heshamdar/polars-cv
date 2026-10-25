//! Resize with fast_image_resize, and the dispatch to the exact resamplers
//! (`kernels::resample`) for what it cannot hold.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use crate::ops::FilterType;

use fast_image_resize as fir;

/// Map our filter types to fast_image_resize algorithm types.
#[inline]
fn to_fir_algorithm(filter: &FilterType) -> fir::ResizeAlg {
    match filter {
        // fir steps through source positions in floating point and breaks an
        // exact pixel-centre tie either way; nearest never reaches it.
        FilterType::Nearest => unreachable!(
            "nearest is resampled exactly by `kernels::resample::nearest` (resize_strided)"
        ),
        FilterType::Triangle => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        FilterType::CatmullRom => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
        FilterType::Gaussian => fir::ResizeAlg::Convolution(fir::FilterType::Gaussian),
        FilterType::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
        FilterType::Area => unreachable!("area resamples in f64 (`kernels::resample`)"),
    }
}

/// Resize using fast_image_resize with SIMD optimization.
///
/// Nearest, of any dtype, is an exact gather (`kernels::resample::nearest`):
/// fir's floating-point source positions break exact pixel-centre ties
/// either way. For the convolution filters, U8, U16 and F32 with 1–4
/// channels are fast_image_resize's own pixel types ([`resize_pixels`]). i8
/// and i16 are resized as F32 (which holds them exactly) and cast back; the
/// 32/64-bit integers and f64 are resampled in f64 by `kernels::resample`.
/// The output keeps the input's dtype and rank.
pub(crate) fn resize_strided(
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
    // fir has no area resampler (its `Box` is a centre-sampled box, which
    // includes or drops a pixel on a rounding tie): every dtype takes the
    // exact overlap weights in f64.
    if filter == FilterType::Area {
        return super::resample::resample(&buf, w as usize, h as usize, filter);
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
        // f64 (`kernels::resample`).
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

thread_local! {
    /// Reused across resize calls on the same worker thread to avoid
    /// reallocating fast_image_resize's internal scratch buffers on every row.
    /// `fir::Resizer` adapts to the pixel type and dimensions on each `resize`
    /// call, so a single instance safely serves U8/U16/F32 at any size. It is
    /// thread-local, so streaming morsel workers never share one.
    static FIR_RESIZER: std::cell::RefCell<fir::Resizer> =
        std::cell::RefCell::new(fir::Resizer::new());
}
