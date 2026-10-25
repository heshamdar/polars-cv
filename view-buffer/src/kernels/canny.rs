//! Canny edge detection.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;

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
pub(crate) fn apply_canny(buf: ViewBuffer, low_threshold: f32, high_threshold: f32) -> ViewBuffer {
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
