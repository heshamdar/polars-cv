//! The exact Euclidean distance transform (`distance_transform`, CR-93).
//!
//! Felzenszwalb & Huttenlocher's separable algorithm ("Distance Transforms of
//! Sampled Functions", 2012): the squared distance to the nearest site along
//! one axis, then the lower envelope of parabolas rooted at those values
//! along the other. Every squared distance is an integer, held exactly in
//! f64, so the result is the true Euclidean distance rounded once (by the
//! final `sqrt` and the store to f32): what `scipy.ndimage.
//! distance_transform_edt` and `cv2.distanceTransform(DIST_L2,
//! DIST_MASK_PRECISE)` compute.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{with_dtype, ViewType};

/// The distance transform of a single-channel `[H, W]` or `[H, W, 1]` buffer,
/// as an f32 buffer of the same shape.
///
/// A pixel is foreground when its value is non-zero in its own dtype (so NaN
/// and ±inf are foreground, as in NumPy's truthiness). Unsigned: a foreground
/// pixel's distance to the nearest background pixel, centre to centre, and 0
/// on the background. Signed: `d - 0.5` outside and `-(d - 0.5)` inside, `d`
/// the centre distance to the nearest pixel of the other class, which is the
/// distance to the pixel-edge boundary `extract_contours` traces. A class with
/// no pixel is infinitely far: `+inf` (or `-inf` inside, signed).
pub(crate) fn distance_transform(buf: &ViewBuffer, signed: bool) -> ViewBuffer {
    let shape = buf.shape().to_vec();
    // `validate` refuses anything else before the row runs.
    assert!(
        shape.len() == 2 || (shape.len() == 3 && shape[2] == 1),
        "DistanceTransform requires single-channel input, got shape {shape:?}"
    );
    let (h, w) = (shape[0], shape[1]);
    let inside = foreground(buf);
    let values: Vec<f32> = if signed {
        let d_in = squared_distance_to(&inside, h, w, false);
        let d_out = squared_distance_to(&inside, h, w, true);
        inside
            .iter()
            .zip(d_in.iter().zip(&d_out))
            .map(|(&fg, (&di, &dout))| {
                if fg {
                    -(di.sqrt() - 0.5) as f32
                } else {
                    (dout.sqrt() - 0.5) as f32
                }
            })
            .collect()
    } else {
        squared_distance_to(&inside, h, w, false)
            .into_iter()
            .map(|d| d.sqrt() as f32)
            .collect()
    };
    ViewBuffer::from_vec_with_shape(values, shape)
}

/// Whether each pixel, row-major, is non-zero.
fn foreground(buf: &ViewBuffer) -> Vec<bool> {
    let rows_buf = buf.to_dense_rows();
    with_dtype!(rows_buf.dtype(), T => foreground_typed::<T>(&rows_buf))
}

fn foreground_typed<T: ViewType + PartialEq + Default + Copy>(buf: &ViewBuffer) -> Vec<bool> {
    let zero = T::default();
    buf.dense_rows::<T>()
        .expect("a single-channel rank-2 or rank-3 buffer has packed rows after to_dense_rows")
        .into_iter()
        .flat_map(|row| row.iter().map(move |v| *v != zero))
        .collect()
}

/// The squared distance from every pixel to the nearest *site*: the pixels
/// whose foreground flag equals `sites_are` (`false` = the background).
/// `+inf` everywhere when there is no site.
fn squared_distance_to(inside: &[bool], h: usize, w: usize, sites_are: bool) -> Vec<f64> {
    let mut grid: Vec<f64> = inside
        .iter()
        .map(|&fg| if fg == sites_are { 0.0 } else { f64::INFINITY })
        .collect();
    let mut line = Vec::with_capacity(h.max(w));
    let mut out = vec![0.0; h.max(w)];
    let mut envelope = Envelope::with_capacity(h.max(w));
    // Columns, then rows: each pass is the 1-D transform of the previous.
    for x in 0..w {
        line.clear();
        line.extend((0..h).map(|y| grid[y * w + x]));
        envelope.transform(&line, &mut out[..h]);
        for y in 0..h {
            grid[y * w + x] = out[y];
        }
    }
    for y in 0..h {
        let row = &mut grid[y * w..(y + 1) * w];
        envelope.transform(row, &mut out[..w]);
        row.copy_from_slice(&out[..w]);
    }
    grid
}

/// The lower envelope of the parabolas `(q - p)^2 + f[p]`, reused across
/// lines so a transform allocates once.
struct Envelope {
    /// The parabolas' roots, left to right.
    roots: Vec<usize>,
    /// Where each parabola takes over from the previous one.
    starts: Vec<f64>,
}

impl Envelope {
    fn with_capacity(n: usize) -> Self {
        Envelope {
            roots: Vec::with_capacity(n),
            starts: Vec::with_capacity(n + 1),
        }
    }

    /// `out[q] = min_p (q - p)^2 + f[p]`, over the finite `f[p]`; `+inf`
    /// when none is finite. A non-finite `f[p]` roots no parabola (the
    /// textbook's intersection formula would compute `inf - inf`).
    fn transform(&mut self, f: &[f64], out: &mut [f64]) {
        self.roots.clear();
        self.starts.clear();
        for (q, &fq) in f.iter().enumerate() {
            if !fq.is_finite() {
                continue;
            }
            let mut start = f64::NEG_INFINITY;
            while let Some(&p) = self.roots.last() {
                // Where the parabola rooted at q overtakes the one at p.
                let (qf, pf) = (q as f64, p as f64);
                let s = ((fq + qf * qf) - (f[p] + pf * pf)) / (2.0 * (qf - pf));
                if s <= *self.starts.last().expect("one start per root") {
                    self.roots.pop();
                    self.starts.pop();
                } else {
                    start = s;
                    break;
                }
            }
            self.roots.push(q);
            self.starts.push(start);
        }
        if self.roots.is_empty() {
            out.fill(f64::INFINITY);
            return;
        }
        let mut k = 0;
        for (q, slot) in out.iter_mut().enumerate() {
            let qf = q as f64;
            while k + 1 < self.roots.len() && self.starts[k + 1] < qf {
                k += 1;
            }
            let p = self.roots[k];
            let d = qf - p as f64;
            *slot = d * d + f[p];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dtype::DType;

    fn run(values: &[u8], h: usize, w: usize, signed: bool) -> Vec<f32> {
        let buf = ViewBuffer::from_vec_with_shape(values.to_vec(), vec![h, w]);
        let out = distance_transform(&buf, signed);
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.shape(), &[h, w]);
        out.to_contiguous().as_slice::<f32>().to_vec()
    }

    /// The brute-force definition: the nearest site by exhaustive search.
    fn brute(values: &[u8], h: usize, w: usize, sites_are: bool) -> Vec<f64> {
        (0..h * w)
            .map(|i| {
                let (y, x) = ((i / w) as f64, (i % w) as f64);
                (0..h * w)
                    .filter(|&j| (values[j] != 0) == sites_are)
                    .map(|j| {
                        let (sy, sx) = ((j / w) as f64, (j % w) as f64);
                        ((y - sy).powi(2) + (x - sx).powi(2)).sqrt()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect()
    }

    #[test]
    fn a_row_measures_to_its_nearest_zero() {
        assert_eq!(
            run(&[0, 5, 7, 1, 1, 0, 3], 1, 7, false),
            vec![0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn a_single_zero_radiates_euclidean_distances() {
        let mut v = vec![1u8; 5 * 5];
        v[2 * 5 + 2] = 0;
        let got = run(&v, 5, 5, false);
        assert_eq!(got[2 * 5 + 2], 0.0);
        assert_eq!(got[0], (8.0f64).sqrt() as f32);
        assert_eq!(got[2 * 5 + 4], 2.0);
        assert_eq!(got[4 * 5 + 3], (5.0f64).sqrt() as f32);
    }

    #[test]
    fn unsigned_matches_the_brute_force_definition() {
        // A deterministic scatter with irregular gaps, rows and columns of
        // every parity.
        for (h, w) in [(1, 1), (1, 9), (9, 1), (7, 11), (16, 13)] {
            let v: Vec<u8> = (0..h * w)
                .map(|i| u8::from((i * 7 + i / 3) % 5 != 0))
                .collect();
            let expected: Vec<f32> = brute(&v, h, w, false).iter().map(|d| *d as f32).collect();
            assert_eq!(run(&v, h, w, false), expected, "{h}x{w}");
        }
    }

    #[test]
    fn signed_is_the_edge_distance_negative_inside() {
        let (h, w) = (7, 11);
        let v: Vec<u8> = (0..h * w)
            .map(|i| u8::from((i * 5 + i / 4) % 3 == 0))
            .collect();
        let d_in = brute(&v, h, w, false);
        let d_out = brute(&v, h, w, true);
        let expected: Vec<f32> = (0..h * w)
            .map(|i| {
                if v[i] != 0 {
                    -(d_in[i] - 0.5) as f32
                } else {
                    (d_out[i] - 0.5) as f32
                }
            })
            .collect();
        assert_eq!(run(&v, h, w, true), expected);
    }

    #[test]
    fn a_missing_class_is_infinitely_far() {
        assert!(run(&[3; 6], 2, 3, false)
            .iter()
            .all(|d| *d == f32::INFINITY));
        assert!(run(&[3; 6], 2, 3, true)
            .iter()
            .all(|d| *d == f32::NEG_INFINITY));
        assert!(run(&[0; 6], 2, 3, false).iter().all(|d| *d == 0.0));
        assert!(run(&[0; 6], 2, 3, true).iter().all(|d| *d == f32::INFINITY));
    }

    #[test]
    fn an_empty_image_keeps_its_shape() {
        assert!(run(&[], 0, 4, false).is_empty());
        assert!(run(&[], 3, 0, true).is_empty());
    }

    #[test]
    fn foreground_is_non_zero_in_the_input_dtype_nan_included() {
        let buf = ViewBuffer::from_vec_with_shape(vec![f32::NAN, 0.0, -2.0], vec![1, 3, 1]);
        let out = distance_transform(&buf, false);
        assert_eq!(out.shape(), &[1, 3, 1]);
        assert_eq!(out.to_contiguous().as_slice::<f32>(), &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn a_strided_view_reads_its_own_pixels() {
        // A vertical flip and a crop: rows read where they lie.
        let v: Vec<u8> = (0..6 * 5).map(|i| u8::from(i % 4 != 0)).collect();
        let full = ViewBuffer::from_vec_with_shape(v.clone(), vec![6, 5]);
        let flipped = full.flip(&[0]);
        let packed = ViewBuffer::from_vec_with_shape(
            flipped.to_contiguous().as_slice::<u8>().to_vec(),
            vec![6, 5],
        );
        assert_eq!(
            distance_transform(&flipped, true)
                .to_contiguous()
                .as_slice::<f32>(),
            distance_transform(&packed, true)
                .to_contiguous()
                .as_slice::<f32>(),
        );
    }
}
