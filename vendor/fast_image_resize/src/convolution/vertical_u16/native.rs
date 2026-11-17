use crate::convolution::optimisations::Normalizer32;
use crate::pixels::InnerPixel;
use crate::{ImageView, ImageViewMut};

#[inline(always)]
pub(crate) fn vert_convolution<T>(
    src_view: &impl ImageView<Pixel = T>,
    dst_view: &mut impl ImageViewMut<Pixel = T>,
    offset: u32,
    normalizer: &Normalizer32,
) where
    T: InnerPixel<Component = u16>,
{
    let coefficients_chunks = normalizer.chunks();
    let precision = normalizer.precision();
    let initial: i64 = 1 << (precision - 1);
    let src_x_initial = offset as usize * T::count_of_components();

    let dst_rows = dst_view.iter_rows_mut(0);
    let coeffs_chunks_iter = coefficients_chunks.iter();
    for (coeffs_chunk, dst_row) in coeffs_chunks_iter.zip(dst_rows) {
        let first_y_src = coeffs_chunk.start;
        let ks = coeffs_chunk.values();
        let dst_components = T::components_mut(dst_row);
        let mut x_src = src_x_initial;

        const CHUNK_SIZE: usize = 16;
        let mut dst_chunks = dst_components.chunks_exact_mut(CHUNK_SIZE);
        for dst_chunk in &mut dst_chunks {
            x_src = convolution_one_chunk::<T, CHUNK_SIZE>(
                src_view,
                normalizer,
                initial,
                dst_chunk,
                x_src,
                first_y_src,
                ks,
            );
        }

        let tail = dst_chunks.into_remainder();
        if !tail.is_empty() {
            convolution_by_u16(src_view, normalizer, initial, tail, x_src, first_y_src, ks);
        }
    }
}

#[inline(always)]
fn convolution_one_chunk<T, const CHUNK_SIZE: usize>(
    src_view: &impl ImageView<Pixel = T>,
    normalizer: &Normalizer32,
    initial: i64,
    dst_chunk: &mut [u16],
    mut x_src: usize,
    first_y_src: u32,
    ks: &[i32],
) -> usize
where
    T: InnerPixel<Component = u16>,
{
    let mut ss = [initial; CHUNK_SIZE];
    let src_rows = src_view.iter_rows(first_y_src);
    let x_end = x_src + CHUNK_SIZE;

    for (k, src_row) in ks.iter().copied().zip(src_rows) {
        let src_components = T::components(src_row);
        let src_chunk: &[u16] = &src_components[x_src..x_end];
        for (s, &c) in ss.iter_mut().zip(src_chunk) {
            *s += c as i64 * (k as i64);
        }
    }

    for (i, s) in ss.iter().copied().enumerate() {
        dst_chunk[i] = normalizer.clip(s);
    }
    x_src += CHUNK_SIZE;
    x_src
}

#[inline(always)]
pub(crate) fn convolution_by_u16<T: InnerPixel<Component = u16>>(
    src_view: &impl ImageView<Pixel = T>,
    normalizer: &Normalizer32,
    initial: i64,
    dst_components: &mut [u16],
    mut x_src: usize,
    first_y_src: u32,
    ks: &[i32],
) -> usize {
    for dst_component in dst_components.iter_mut() {
        let mut ss = initial;
        let src_rows = src_view.iter_rows(first_y_src);
        for (&k, src_row) in ks.iter().zip(src_rows) {
            let src_components = T::components(src_row);
            let src_component = src_components[x_src];
            ss += src_component as i64 * (k as i64);
        }
        *dst_component = normalizer.clip(ss);
        x_src += 1
    }
    x_src
}
