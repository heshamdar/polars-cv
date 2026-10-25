//! The pixel-centre coordinate grid.

use crate::core::buffer::ViewBuffer;

/// The f32 `[H, W, 2]` grid of pixel centres `(x + 0.5, y + 0.5)` over `buf`'s
/// height and width, divided by `W` and `H` when `normalized`; no value of
/// `buf` is read. Each coordinate is computed in f64 and stored rounded to
/// the nearest f32.
pub(crate) fn coordinate_grid(buf: &ViewBuffer, normalized: bool) -> ViewBuffer {
    let shape = buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let (sx, sy) = if normalized {
        (w as f64, h as f64)
    } else {
        (1.0, 1.0)
    };
    let mut data = Vec::with_capacity(h * w * 2);
    for y in 0..h {
        let cy = ((y as f64 + 0.5) / sy) as f32;
        for x in 0..w {
            data.push(((x as f64 + 0.5) / sx) as f32);
            data.push(cy);
        }
    }
    ViewBuffer::from_vec_with_shape(data, vec![h, w, 2])
}
