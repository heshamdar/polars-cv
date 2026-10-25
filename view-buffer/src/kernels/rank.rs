//! The rank filter: each pixel's window percentile.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;

/// Each pixel's `height` x `width` window `q`-th percentile, per channel
/// (`reduction::percentile_of`, the one interpolation), outside the image
/// read by `border` as `convolve2d` reads it; the result in the input's
/// dtype. `check` keeps both sides odd. A u8 image takes a sliding
/// histogram ([`rank_filter_u8`]), every other dtype a selection per window
/// ([`rank_filter_values`]); the two give the same bits.
pub(crate) fn apply_rank_filter(
    buf: &ViewBuffer,
    height: usize,
    width: usize,
    q: f64,
    border: crate::ops::filter::BorderMode,
) -> ViewBuffer {
    let shape = buf.shape();
    let out = if buf.dtype() == DType::U8 {
        rank_filter_u8(buf, height, width, q, border)
    } else {
        rank_filter_values(buf, height, width, q, border)
    };
    ViewBuffer::from_vec_with_shape(out, shape.to_vec()).cast(buf.dtype())
}

/// The in-image index an out-of-image one `i` reads on an axis of `n`, or
/// `None` for a zero (`BorderMode::Zero`).
fn rank_border(border: crate::ops::filter::BorderMode, i: isize, n: usize) -> Option<usize> {
    use crate::ops::filter::BorderMode;
    let n_i = n as isize;
    match border {
        BorderMode::Zero => (0..n_i).contains(&i).then_some(i as usize),
        BorderMode::Replicate => Some(i.clamp(0, n_i - 1) as usize),
        // `dcba|abcd|dcba`, repeated for a window wider than the image.
        BorderMode::Reflect => {
            let period = 2 * n_i;
            let m = i.rem_euclid(period);
            Some(if m < n_i { m } else { period - 1 - m } as usize)
        }
    }
}

/// The rank filter by a selection over each window's values (any dtype).
fn rank_filter_values(
    buf: &ViewBuffer,
    height: usize,
    width: usize,
    q: f64,
    border: crate::ops::filter::BorderMode,
) -> Vec<f64> {
    let shape = buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let c = shape.get(2).copied().unwrap_or(1);
    let values = buf.cast_to(DType::F64).to_contiguous();
    let values = values.as_slice::<f64>();
    let (ry, rx) = ((height / 2) as isize, (width / 2) as isize);
    let mut out = Vec::with_capacity(h * w * c);
    let mut window = Vec::with_capacity(height * width);
    for y in 0..h as isize {
        for x in 0..w as isize {
            for ch in 0..c {
                window.clear();
                for dy in -ry..=ry {
                    for dx in -rx..=rx {
                        window.push(
                            match (
                                rank_border(border, y + dy, h),
                                rank_border(border, x + dx, w),
                            ) {
                                (Some(sy), Some(sx)) => values[(sy * w + sx) * c + ch],
                                _ => 0.0,
                            },
                        );
                    }
                }
                out.push(crate::ops::reduction::percentile_of(&mut window, q));
            }
        }
    }
    out
}

/// The rank filter of a u8 image by a sliding histogram (Huang): along each
/// row the window's 256-bin histogram drops its leftmost column and gains a
/// new rightmost one, so a pixel costs O(height + 256) rather than a
/// selection over `height * width` values. The percentile reads the two
/// order statistics it interpolates off the cumulative counts, by
/// `percentile_of`'s formula: the same bits.
fn rank_filter_u8(
    buf: &ViewBuffer,
    height: usize,
    width: usize,
    q: f64,
    border: crate::ops::filter::BorderMode,
) -> Vec<f64> {
    let shape = buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let c = shape.get(2).copied().unwrap_or(1);
    if h == 0 || w == 0 {
        // No pixel, and no window to seed.
        return Vec::new();
    }
    let contig = buf.to_contiguous();
    let data = contig.as_slice::<u8>();
    let (ry, rx) = ((height / 2) as isize, (width / 2) as isize);
    let n = height * width;
    // The order statistics a percentile interpolates, as `percentile_of`.
    let pos = q / 100.0 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    let mut out = vec![0.0; h * w * c];
    let mut hist = [0u32; 256];
    for ch in 0..c {
        for y in 0..h as isize {
            let rows: Vec<Option<usize>> = (-ry..=ry)
                .map(|dy| rank_border(border, y + dy, h))
                .collect();
            let column = |x: isize, hist: &mut [u32; 256], add: bool| {
                let sx = rank_border(border, x, w);
                for &sy in &rows {
                    let v = match (sy, sx) {
                        (Some(sy), Some(sx)) => data[(sy * w + sx) * c + ch],
                        _ => 0,
                    };
                    if add {
                        hist[v as usize] += 1;
                    } else {
                        hist[v as usize] -= 1;
                    }
                }
            };
            hist.fill(0);
            for dx in -rx..=rx {
                column(dx, &mut hist, true);
            }
            for x in 0..w as isize {
                if x > 0 {
                    column(x - 1 - rx, &mut hist, false);
                    column(x + rx, &mut hist, true);
                }
                // The `lo`-th and `lo + 1`-th values in order.
                let (mut seen, mut low, mut high) = (0usize, None, None);
                for (value, &count) in hist.iter().enumerate() {
                    seen += count as usize;
                    if low.is_none() && seen > lo {
                        low = Some(value as f64);
                    }
                    if seen > lo + 1 {
                        high = Some(value as f64);
                        break;
                    }
                }
                let low = low.expect("a window holds lo + 1 values");
                let value = match high.filter(|_| lo + 1 < n) {
                    Some(hi) => low * (1.0 - frac) + hi * frac,
                    None => low,
                };
                out[(y as usize * w + x as usize) * c + ch] = value;
            }
        }
    }
    out
}

#[cfg(test)]
mod rank_filter_tests {
    use super::*;
    use crate::ops::filter::BorderMode;

    /// The sliding histogram is the selection, bit for bit: random u8
    /// images, every border, windows narrower and wider than the image,
    /// percentiles with and without interpolation, one and three channels.
    #[test]
    fn the_u8_histogram_is_the_selection() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // An empty image too: there is no window to seed.
        for (h, w, c) in [
            (1, 1, 1),
            (5, 7, 1),
            (9, 4, 3),
            (3, 12, 1),
            (5, 0, 1),
            (0, 4, 3),
        ] {
            let data: Vec<u8> = (0..h * w * c).map(|_| (next() % 7 * 37) as u8).collect();
            let buf = ViewBuffer::from_vec_with_shape(data, vec![h, w, c]);
            for (wh, ww) in [(1, 1), (3, 3), (1, 5), (5, 1), (7, 9), (15, 3)] {
                for border in [BorderMode::Zero, BorderMode::Replicate, BorderMode::Reflect] {
                    for q in [0.0, 12.5, 33.333, 50.0, 90.0, 100.0] {
                        let fast = rank_filter_u8(&buf, wh, ww, q, border);
                        let slow = rank_filter_values(&buf, wh, ww, q, border);
                        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                        assert_eq!(
                            bits(&fast),
                            bits(&slow),
                            "{h}x{w}x{c} window {wh}x{ww} {border:?} q={q}"
                        );
                    }
                }
            }
        }
    }
}
