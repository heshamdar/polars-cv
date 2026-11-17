use core::arch::x86_64::*;

use super::sse4;
use crate::convolution::optimisations::{CoefficientsI16Chunk, Normalizer16};
use crate::image_view::ImageViewMut;
use crate::pixels::InnerPixel;
use crate::{simd_utils, ImageView};

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
#[target_feature(enable = "avx2")]
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

    let initial_256 = _mm256_set1_epi32(1 << (PRECISION as u8 - 1));

    let mut dst_chunks_64 = dst_u8.chunks_exact_mut(64);
    for dst_chunk in &mut dst_chunks_64 {
        let mut sss0 = [initial_256; 4];
        let mut sss1 = [initial_256; 4];

        let coeffs_chunks = coeffs.chunks_exact(2);
        let coeffs_reminder = coeffs_chunks.remainder();

        for (src_rows, two_coeffs) in src_view.iter_2_rows(y_start, max_rows).zip(coeffs_chunks) {
            let components1 = T::components(src_rows[0]);
            let components2 = T::components(src_rows[1]);

            // Load two coefficients at once
            let mmk = simd_utils::mm256_load_and_clone_i16x2(two_coeffs);

            conv_32_components_two_rows(components1, components2, src_x, &mut sss0, mmk);
            conv_32_components_two_rows(components1, components2, src_x + 32, &mut sss1, mmk);
        }

        if let Some(&k) = coeffs_reminder.first() {
            if let Some(s_row) = src_view.iter_rows(y_last).next() {
                let components = T::components(s_row);
                let mmk = _mm256_set1_epi32(k as i32);

                conv_32_components_one_row(components, src_x, &mut sss0, mmk);
                conv_32_components_one_row(components, src_x + 32, &mut sss1, mmk);
            }
        }

        let (dst0, dst1) = dst_chunk.split_at_mut(32);
        for (mut sss, dst) in [(sss0, dst0), (sss1, dst1)] {
            sss = sss.map(|v| _mm256_srai_epi32::<PRECISION>(v));
            let half0 = _mm256_packs_epi32(sss[0], sss[1]);
            let half1 = _mm256_packs_epi32(sss[2], sss[3]);
            let components = _mm256_packus_epi16(half0, half1);
            let dst_ptr = dst.as_mut_ptr() as *mut __m256i;
            _mm256_storeu_si256(dst_ptr, components);
        }

        src_x += 64;
    }

    // 32 components in one register
    dst_u8 = dst_chunks_64.into_remainder();
    let mut dst_chunks_32 = dst_u8.chunks_exact_mut(32);
    for dst_chunk in &mut dst_chunks_32 {
        let mut sss = [initial_256; 4];

        let coeffs_chunks = coeffs.chunks_exact(2);
        let coeffs_reminder = coeffs_chunks.remainder();

        for (src_rows, two_coeffs) in src_view.iter_2_rows(y_start, max_rows).zip(coeffs_chunks) {
            let components1 = T::components(src_rows[0]); // top line
            let components2 = T::components(src_rows[1]); // bottom line

            // Load two coefficients at once
            let mmk = simd_utils::mm256_load_and_clone_i16x2(two_coeffs);

            conv_32_components_two_rows(components1, components2, src_x, &mut sss, mmk);
        }

        if let Some(&k) = coeffs_reminder.first() {
            if let Some(s_row) = src_view.iter_rows(y_last).next() {
                let components = T::components(s_row);
                let mmk = _mm256_set1_epi32(k as i32);

                conv_32_components_one_row(components, src_x, &mut sss, mmk);
            }
        }

        let dst_ptr = dst_chunk.as_mut_ptr() as *mut __m256i;
        sss = sss.map(|v| _mm256_srai_epi32::<PRECISION>(v));
        let half0 = _mm256_packs_epi32(sss[0], sss[1]);
        let half1 = _mm256_packs_epi32(sss[2], sss[3]);
        let dst = _mm256_packus_epi16(half0, half1);
        _mm256_storeu_si256(dst_ptr, dst);

        src_x += 32;
    }

    dst_u8 = dst_chunks_32.into_remainder();
    sse4::conv_less_that_32_components::<T, PRECISION>(
        src_view,
        dst_u8,
        src_x,
        coeffs_chunk,
        normalizer,
    );
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn conv_32_components_two_rows(
    components1: &[u8],
    components2: &[u8],
    src_x: usize,
    sss: &mut [__m256i; 4],
    mmk: __m256i,
) {
    let zero_256 = _mm256_setzero_si256();
    let source1 = simd_utils::loadu_si256(components1, src_x); // top line
    let source2 = simd_utils::loadu_si256(components2, src_x); // bottom line
    conv_loaded_32_components(source1, source2, sss, mmk, zero_256);
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn conv_32_components_one_row(
    components1: &[u8],
    src_x: usize,
    sss: &mut [__m256i; 4],
    mmk: __m256i,
) {
    let zero_256 = _mm256_setzero_si256();
    let source1 = simd_utils::loadu_si256(components1, src_x); // top line
    conv_loaded_32_components(source1, zero_256, sss, mmk, zero_256);
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn conv_loaded_32_components(
    source1: __m256i,
    source2: __m256i,
    sss: &mut [__m256i; 4],
    mmk: __m256i,
    zero_256: __m256i,
) {
    let source = _mm256_unpacklo_epi8(source1, source2);
    let pix = _mm256_unpacklo_epi8(source, zero_256);
    sss[0] = _mm256_add_epi32(sss[0], _mm256_madd_epi16(pix, mmk));
    let pix = _mm256_unpackhi_epi8(source, zero_256);
    sss[1] = _mm256_add_epi32(sss[1], _mm256_madd_epi16(pix, mmk));

    let source = _mm256_unpackhi_epi8(source1, source2);
    let pix = _mm256_unpacklo_epi8(source, zero_256);
    sss[2] = _mm256_add_epi32(sss[2], _mm256_madd_epi16(pix, mmk));
    let pix = _mm256_unpackhi_epi8(source, zero_256);
    sss[3] = _mm256_add_epi32(sss[3], _mm256_madd_epi16(pix, mmk));
}
