//! Padding kernels: constant, edge, reflect and symmetric modes, in the
//! buffer's own dtype (all ten, so `pad` preserves the input dtype).

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::ViewType;
use crate::ops::pad::PadMode;

/// Convert the user-facing `f32` fill value into the element type.
///
/// Float→int `as` casts saturate in Rust, matching the old explicit
/// `clamp(0.0, 255.0)` behavior for u8 and extending it to every dtype.
pub(crate) trait FillValue: ViewType + Copy {
    fn from_f32(v: f32) -> Self;
}

macro_rules! impl_fill_value {
    ($($t:ty),+ $(,)?) => {
        $(impl FillValue for $t {
            #[inline]
            fn from_f32(v: f32) -> Self {
                v as $t
            }
        })+
    };
}
impl_fill_value!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

pub(crate) fn pad_generic<T: FillValue>(
    buffer: &ViewBuffer,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    value: f32,
    mode: PadMode,
) -> ViewBuffer {
    let shape = buffer.shape();
    assert!(
        shape.len() == 2 || shape.len() == 3,
        "pad requires a [H, W] or [H, W, C] buffer, got shape {shape:?}"
    );
    let input_h = shape[0];
    let input_w = shape[1];
    let channels = if shape.len() > 2 { shape[2] } else { 1 };
    let (top, bottom, left, right) = (top as usize, bottom as usize, left as usize, right as usize);
    let output_h = input_h + top + bottom;
    let output_w = input_w + left + right;
    let input_row = input_w * channels;
    let output_row = output_w * channels;

    // Read in place, at the view's own row stride (a crop, a vertical flip).
    let rows_buf = buffer.to_dense_rows();
    let rows: Vec<&[T]> = rows_buf
        .dense_rows::<T>()
        .expect("a rank-2 or rank-3 buffer has packed rows after to_dense_rows");
    let fill = T::from_f32(value);
    let mut output = vec![fill; output_h * output_w * channels];
    let out_shape = if shape.len() == 2 {
        vec![output_h, output_w]
    } else {
        vec![output_h, output_w, channels]
    };
    // An empty input is extended only by "constant" (the op's contract), so
    // under any other mode its output is empty too: nothing to read or write.
    if input_h * input_row == 0 && mode != PadMode::Constant {
        debug_assert!(
            output.is_empty(),
            "the contract refuses extending an empty axis"
        );
        return ViewBuffer::from_vec_with_shape(output, out_shape);
    }

    match mode {
        PadMode::Constant => {
            // Interior rows are memcpy'd; the fill initialization covers borders.
            for (y, row) in rows.iter().enumerate() {
                let dst = (y + top) * output_row + left * channels;
                output[dst..dst + input_row].copy_from_slice(row);
            }
        }
        PadMode::Edge => {
            for dst_y in 0..output_h {
                let src_y = dst_y.saturating_sub(top).min(input_h - 1);
                let src_row = rows[src_y];
                let dst_row_start = dst_y * output_row;
                // Left border: replicate the first pixel.
                let first = &src_row[..channels];
                for i in 0..left {
                    let d = dst_row_start + i * channels;
                    output[d..d + channels].copy_from_slice(first);
                }
                // Interior: copy the source row.
                let d = dst_row_start + left * channels;
                output[d..d + input_row].copy_from_slice(src_row);
                // Right border: replicate the last pixel.
                let last = &src_row[(input_w - 1) * channels..];
                for i in 0..right {
                    let d = dst_row_start + (left + input_w + i) * channels;
                    output[d..d + channels].copy_from_slice(last);
                }
            }
        }
        PadMode::Reflect | PadMode::Symmetric => {
            let symmetric = mode == PadMode::Symmetric;
            for dst_y in 0..output_h {
                let src_y = reflect_index(dst_y as isize - top as isize, input_h, symmetric);
                for dst_x in 0..output_w {
                    let src_x = reflect_index(dst_x as isize - left as isize, input_w, symmetric);
                    let s = src_x * channels;
                    let d = (dst_y * output_w + dst_x) * channels;
                    output[d..d + channels].copy_from_slice(&rows[src_y][s..s + channels]);
                }
            }
        }
    }

    ViewBuffer::from_vec_with_shape(output, out_shape)
}

/// Compute the reflected source index for a given output coordinate.
///
/// `symmetric=false` (reflect, NumPy `mode="reflect"`): mirror without edge
/// repetition. Input `[a, b, c, d]` → `[d, c, b, | a, b, c, d | c, b, a]`
///
/// `symmetric=true` (NumPy `mode="symmetric"`): mirror with edge repetition.
/// Input `[a, b, c, d]` → `[c, b, a, | a, b, c, d | d, c, b]`
#[inline]
fn reflect_index(idx: isize, len: usize, symmetric: bool) -> usize {
    if len <= 1 {
        return 0;
    }
    let n = len as isize;
    let period = if symmetric { 2 * n } else { 2 * (n - 1) };
    let mut i = ((idx % period) + period) % period;
    if i >= n {
        i = if symmetric { 2 * n - 1 - i } else { period - i };
    }
    i.clamp(0, n - 1) as usize
}
