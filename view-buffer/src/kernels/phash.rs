//! Standard pHash: the DCT-based perceptual hash.

use crate::core::buffer::ViewBuffer;

/// Standard pHash, the steps of `imagehash.phash`: BT.601 grayscale, a
/// Lanczos3 resize to a `4 * side` square, a 2-D DCT-II, its `side x side`
/// lowest frequencies thresholded at their median, and the bits row-major,
/// most significant first, in `n_bytes` bytes. The grayscale and the resize
/// are the engine's own (`grayscale` and `resize` ops), so the hash sees the
/// pixels a pipeline would.
///
/// It replaced image_hasher's `Mean` + `preproc_dct`, which thresholds a
/// 16x16-input DCT at its *mean*: the DC coefficient dominates that mean, so
/// most bits agreed between unrelated images (distinct COCO images ~18 bits
/// apart instead of ~32).
pub(crate) fn phash(buffer: &ViewBuffer, side: usize, n_bytes: usize) -> Vec<u8> {
    use crate::kernels::pixel::grayscale_strided;
    use crate::kernels::resize::resize_strided;
    use crate::ops::FilterType;

    let n = 4 * side;
    let gray = if buffer.shape().len() == 2 {
        let (h, w) = (buffer.shape()[0], buffer.shape()[1]);
        buffer.to_contiguous().reshape(vec![h, w, 1])
    } else {
        grayscale_strided(buffer.clone())
    };
    let small = resize_strided(gray, n as u32, n as u32, FilterType::Lanczos3).to_contiguous();
    let pixels: Vec<f64> = small
        .as_slice::<u8>()
        .iter()
        .map(|&v| f64::from(v))
        .collect();

    // DCT-II along axis 0 then axis 1, keeping the `side` lowest
    // frequencies of each. The constant factor is dropped: it scales every
    // coefficient alike, so the median threshold is unchanged.
    let basis: Vec<f64> = (0..side * n)
        .map(|i| {
            let (k, x) = (i / n, i % n);
            (std::f64::consts::PI * k as f64 * (2 * x + 1) as f64 / (2 * n) as f64).cos()
        })
        .collect();
    let mut cols = vec![0.0f64; side * n]; // [k][x]
    for k in 0..side {
        for y in 0..n {
            let b = basis[k * n + y];
            for x in 0..n {
                cols[k * n + x] += b * pixels[y * n + x];
            }
        }
    }
    let mut coeffs = vec![0.0f64; side * side]; // [k][l], row-major
    for k in 0..side {
        for l in 0..side {
            coeffs[k * side + l] = (0..n).map(|x| basis[l * n + x] * cols[k * n + x]).sum();
        }
    }

    // A coefficient that is mathematically 0 (e.g. every vertical frequency
    // of an image constant down its columns) comes out of the direct sums as
    // +-1e-13 rounding noise, and its sign would decide its bit; scipy's FFT
    // DCT, which imagehash uses, returns those as exact zeros. Snap them, so
    // they compare as not above the median.
    let scale = coeffs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for v in &mut coeffs {
        if v.abs() <= scale * 1e-9 {
            *v = 0.0;
        }
    }
    let mut sorted = coeffs.clone();
    sorted.sort_by(f64::total_cmp);
    let m = sorted.len();
    let median = if m % 2 == 1 {
        sorted[m / 2]
    } else {
        (sorted[m / 2 - 1] + sorted[m / 2]) / 2.0
    };

    let mut out = vec![0u8; n_bytes];
    for (i, &v) in coeffs.iter().enumerate() {
        if v > median {
            out[i / 8] |= 0x80 >> (i % 8);
        }
    }
    out
}
