use core::arch::x86_64::*;

use super::native;
use crate::convolution::optimisations::{CoefficientsI16Chunk, Normalizer16};
use crate::pixels::InnerPixel;
use crate::{simd_utils, ImageView, ImageViewMut};

#[inline]
pub(crate) fn vert_convolution<T>(
    src_view: &impl ImageView<Pixel = T>,
    dst_view: &mut impl ImageViewMut<Pixel = T>,
    offset: u32,
    normalizer: &Normalizer16,
) where
    T: InnerPixel<Component = u8>,
{
    let precision = normalizer.precision();

    macro_rules! call {
        ($imm8:expr) => {{
            vert_convolution_p::<T, $imm8>(src_view, dst_view, offset, normalizer);
        }};
    }
    constify_imm8!(precision, call);
}

fn vert_convolution_p<T, const PRECISION: i32>(
    src_view: &impl ImageView<Pixel = T>,
    dst_view: &mut impl ImageViewMut<Pixel = T>,
    offset: u32,
    normalizer: &Normalizer16,
) where
    T: InnerPixel<Component = u8>,
{
    let coefficients_chunks = normalizer.chunks();
    let src_x = offset as usize * T::count_of_components();
    let dst_rows = dst_view.iter_rows_mut(0);
    let dst_row_and_coefs = dst_rows.zip(coefficients_chunks);

    for (dst_row, coeffs_chunk) in dst_row_and_coefs {
        unsafe {
            vert_convolution_into_one_row::<T, PRECISION>(
                src_view,
                dst_row,
                src_x,
                coeffs_chunk,
                normalizer,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn vert_convolution_into_one_row<T, const PRECISION: i32>(
    src_view: &impl ImageView<Pixel = T>,
    dst_row: &mut [T],
    mut src_x: usize,
    coeffs_chunk: &CoefficientsI16Chunk,
    normalizer: &Normalizer16,
) where
    T: InnerPixel<Component = u8>,
{
    let y_start = coeffs_chunk.start;
    let coeffs = coeffs_chunk.values();
    let max_rows = coeffs.len() as u32;
    let y_last = (y_start + max_rows).max(1) - 1;
    let mut dst_u8 = T::components_mut(dst_row);

    let initial = _mm_set1_epi32(1 << (PRECISION - 1));

    let mut dst_chunks_32 = dst_u8.chunks_exact_mut(32);
    for dst_chunk in &mut dst_chunks_32 {
        let mut sss0 = [initial; 4];
        let mut sss1 = [initial; 4];

        let coeffs_chunks = coeffs.chunks_exact(2);
        let coeffs_reminder = coeffs_chunks.remainder();

        for (src_rows, two_coeffs) in src_view.iter_2_rows(y_start, max_rows).zip(coeffs_chunks) {
            let components1 = T::components(src_rows[0]);
            let components2 = T::components(src_rows[1]);

            // Load two coefficients at once
            let mmk = simd_utils::mm_load_and_clone_i16x2(two_coeffs);

            sss0 = conv_16_components_two_rows(components1, components2, src_x, sss0, mmk);
            sss1 = conv_16_components_two_rows(components1, components2, src_x + 16, sss1, mmk);
        }

        if let Some(&k) = coeffs_reminder.first() {
            if let Some(s_row) = src_view.iter_rows(y_last).next() {
                let components = T::components(s_row);
                let mmk = _mm_set1_epi32(k as i32);

                sss0 = convolution_16_components_one_row(components, src_x, sss0, mmk);
                sss1 = convolution_16_components_one_row(components, src_x + 16, sss1, mmk);
            }
        }

        let (dst0, dst1) = dst_chunk.split_at_mut(16);
        for (mut sss, dst) in [(sss0, dst0), (sss1, dst1)] {
            sss = sss.map(|v| _mm_srai_epi32::<PRECISION>(v));
            let half0 = _mm_packs_epi32(sss[0], sss[1]);
            let half1 = _mm_packs_epi32(sss[2], sss[3]);
            let components = _mm_packus_epi16(half0, half1);
            let dst_ptr = dst.as_mut_ptr() as *mut __m128i;
            _mm_storeu_si128(dst_ptr, components);
        }

        src_x += 32;
    }

    dst_u8 = dst_chunks_32.into_remainder();
    conv_less_that_32_components::<T, PRECISION>(src_view, dst_u8, src_x, coeffs_chunk, normalizer);
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn conv_less_that_32_components<T, const PRECISION: i32>(
    src_view: &impl ImageView<Pixel = T>,
    mut dst_u8: &mut [u8],
    mut src_x: usize,
    coeffs_chunk: &CoefficientsI16Chunk,
    normalizer: &Normalizer16,
) where
    T: InnerPixel<Component = u8>,
{
    let y_start = coeffs_chunk.start;
    let coeffs = coeffs_chunk.values();
    let max_rows = coeffs.len() as u32;
    let y_last = (y_start + max_rows).max(1) - 1;
    let initial = _mm_set1_epi32(1 << (PRECISION - 1));
    let zero_128 = _mm_setzero_si128();

    let mut dst_chunks_8 = dst_u8.chunks_exact_mut(8);
    for dst_chunk in &mut dst_chunks_8 {
        let mut sss = [initial; 2];

        let coeffs_chunks = coeffs.chunks_exact(2);
        let coeffs_reminder = coeffs_chunks.remainder();

        for (src_rows, two_coeffs) in src_view.iter_2_rows(y_start, max_rows).zip(coeffs_chunks) {
            let components1 = T::components(src_rows[0]);
            let components2 = T::components(src_rows[1]);
            // Load two coefficients at once
            let mmk = simd_utils::mm_load_and_clone_i16x2(two_coeffs);

            let source1 = simd_utils::loadl_epi64(components1, src_x); // top line
            let source2 = simd_utils::loadl_epi64(components2, src_x); // bottom line
            sss = conf_8_components_two_rows(source1, source2, zero_128, mmk, sss);
        }

        if let Some(&k) = coeffs_reminder.first() {
            if let Some(s_row) = src_view.iter_rows(y_last).next() {
                let components = T::components(s_row);
                let mmk = _mm_set1_epi32(k as i32);

                let source1 = simd_utils::loadl_epi64(components, src_x); // top line
                sss = conf_8_components_two_rows(source1, zero_128, zero_128, mmk, sss);
            }
        }

        sss = sss.map(|v| _mm_srai_epi32::<PRECISION>(v));
        let v_i16x8 = _mm_packs_epi32(sss[0], sss[1]);
        let v_u8x16 = _mm_packus_epi16(v_i16x8, v_i16x8);
        let dst_ptr = dst_chunk.as_mut_ptr() as *mut __m128i;
        _mm_storel_epi64(dst_ptr, v_u8x16);

        src_x += 8;
    }

    dst_u8 = dst_chunks_8.into_remainder();
    let mut dst_chunks_4 = dst_u8.chunks_exact_mut(4);
    if let Some(dst_chunk) = dst_chunks_4.next() {
        let mut sss = initial;

        let coeffs_chunks = coeffs.chunks_exact(2);
        let coeffs_reminder = coeffs_chunks.remainder();

        for (src_rows, two_coeffs) in src_view.iter_2_rows(y_start, max_rows).zip(coeffs_chunks) {
            let components1 = T::components(src_rows[0]);
            let components2 = T::components(src_rows[1]);
            // Load two coefficients at once
            let mmk = simd_utils::mm_load_and_clone_i16x2(two_coeffs);

            let source1 = simd_utils::mm_cvtsi32_si128_from_u8(components1, src_x); // top line
            let source2 = simd_utils::mm_cvtsi32_si128_from_u8(components2, src_x); // bottom line

            let source = _mm_unpacklo_epi8(source1, source2);
            let pix = _mm_unpacklo_epi8(source, zero_128);
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }

        if let Some(&k) = coeffs_reminder.first() {
            if let Some(s_row) = src_view.iter_rows(y_last).next() {
                let components = T::components(s_row);
                let pix = simd_utils::mm_cvtepu8_epi32_from_u8(components, src_x);
                let mmk = _mm_set1_epi32(k as i32);
                sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
            }
        }

        sss = _mm_srai_epi32::<PRECISION>(sss);
        sss = _mm_packs_epi32(sss, sss);
        let dst_ptr = dst_chunk.as_mut_ptr() as *mut i32;
        dst_ptr.write_unaligned(_mm_cvtsi128_si32(_mm_packus_epi16(sss, sss)));
        src_x += 4;
    }

    dst_u8 = dst_chunks_4.into_remainder();
    if !dst_u8.is_empty() {
        native::convolution_by_u8(
            src_view,
            normalizer,
            1 << (PRECISION - 1),
            dst_u8,
            src_x,
            y_start,
            coeffs,
        );
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn conv_16_components_two_rows(
    components1: &[u8],
    components2: &[u8],
    src_x: usize,
    sss: [__m128i; 4],
    mmk: __m128i,
) -> [__m128i; 4] {
    let zero_128 = _mm_setzero_si128();
    let source1 = simd_utils::loadu_si128(components1, src_x); // top line
    let source2 = simd_utils::loadu_si128(components2, src_x); // bottom line
    conv_loaded_16_components(source1, source2, sss, mmk, zero_128)
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolution_16_components_one_row(
    components1: &[u8],
    src_x: usize,
    sss: [__m128i; 4],
    mmk: __m128i,
) -> [__m128i; 4] {
    let zero_128 = _mm_setzero_si128();
    let source1 = simd_utils::loadu_si128(components1, src_x); // top line
    conv_loaded_16_components(source1, zero_128, sss, mmk, zero_128)
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn conv_loaded_16_components(
    source1: __m128i,
    source2: __m128i,
    sss: [__m128i; 4],
    mmk: __m128i,
    zero: __m128i,
) -> [__m128i; 4] {
    let r0 = conf_8_components_two_rows(source1, source2, zero, mmk, [sss[0], sss[1]]);

    let source = _mm_unpackhi_epi8(source1, source2);
    let pix = _mm_unpacklo_epi8(source, zero);
    let r3 = _mm_add_epi32(sss[2], _mm_madd_epi16(pix, mmk));
    let pix = _mm_unpackhi_epi8(source, zero);
    let r4 = _mm_add_epi32(sss[3], _mm_madd_epi16(pix, mmk));

    [r0[0], r0[1], r3, r4]
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn conf_8_components_two_rows(
    source1: __m128i,
    source2: __m128i,
    zero: __m128i,
    mmk: __m128i,
    mut sss: [__m128i; 2],
) -> [__m128i; 2] {
    let source = _mm_unpacklo_epi8(source1, source2);
    let pix = _mm_unpacklo_epi8(source, zero);
    sss[0] = _mm_add_epi32(sss[0], _mm_madd_epi16(pix, mmk));
    let pix = _mm_unpackhi_epi8(source, zero);
    sss[1] = _mm_add_epi32(sss[1], _mm_madd_epi16(pix, mmk));
    sss
}
