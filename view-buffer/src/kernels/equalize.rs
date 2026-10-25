//! Histogram equalization.

use crate::core::buffer::ViewBuffer;

// ============================================================
// Histogram Equalization
// ============================================================

/// Histogram equalization for contrast enhancement.
///
/// Computes the histogram, cumulative distribution, then maps each pixel
/// through the normalized CDF. Operates per-channel for multi-channel images.
/// Input must be U8 (enforced by `working_dtype`). Output is U8.
pub(crate) fn apply_histogram_equalize(buf: ViewBuffer) -> ViewBuffer {
    // Read in place, at the view's own row stride (a crop, a vertical flip).
    let rows_buf = buf.to_dense_rows();
    let shape = rows_buf.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape.get(2).copied().unwrap_or(1);
    let count = rows_buf.layout.num_elements();
    let rows = rows_buf
        .dense_rows::<u8>()
        .expect("a rank-2 or rank-3 buffer has packed rows after to_dense_rows");

    let total_pixels = h * w;
    let mut output = vec![0u8; count];

    for ch in 0..c {
        // Compute histogram for this channel
        let mut hist = [0u32; 256];
        for row in &rows {
            for x in 0..w {
                hist[row[x * c + ch] as usize] += 1;
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
        for (y, row) in rows.iter().enumerate() {
            for x in 0..w {
                let idx = (y * w + x) * c + ch;
                let val = row[x * c + ch] as usize;
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
