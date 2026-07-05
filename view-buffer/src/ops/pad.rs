//! Padding kernels: constant, edge, reflect, and symmetric modes.
//!
//! All padding math lives here in the engine (it was previously implemented
//! inside the polars-cv plugin's graph executor). One dtype-generic
//! implementation covers every element type, so padding **preserves the
//! input dtype for all ten dtypes** — the old plugin implementation silently
//! cast non-u8/f32/f64 inputs to f32, violating the declared `PreserveInput`
//! dtype rule.
//!
//! Rank is preserved: a 2-D `[H, W]` input pads to 2-D, a 3-D `[H, W, C]`
//! input to 3-D (matching the `PreserveRank` contract).

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, ViewType};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Padding mode for Pad operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PadMode {
    /// Fill with constant value.
    Constant,
    /// Replicate edge values.
    Edge,
    /// Reflect without edge (NumPy `mode="reflect"`): `[a,b,c,d]` → `dcb|abcd|cba`.
    Reflect,
    /// Reflect with edge (NumPy `mode="symmetric"`): `[a,b,c,d]` → `cba|abcd|dcb`.
    Symmetric,
}

/// Position for PadToSize operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PadPosition {
    /// Center content in padded area.
    Center,
    /// Place at top-left corner.
    TopLeft,
    /// Place at bottom-right corner.
    BottomRight,
}

crate::naming::named_variants!(PadMode {
    "constant" => Constant,
    "edge" => Edge,
    "reflect" => Reflect,
    "symmetric" => Symmetric,
});

crate::naming::named_variants!(PadPosition {
    "center" => Center,
    "top-left" => TopLeft,
    "bottom-right" => BottomRight,
});

/// Convert the user-facing `f32` fill value into the element type.
///
/// Float→int `as` casts saturate in Rust, matching the old explicit
/// `clamp(0.0, 255.0)` behavior for u8 and extending it to every dtype.
trait FillValue: ViewType + Copy {
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

/// Dispatch a generic padding function over the buffer's dtype.
macro_rules! dispatch_dtype {
    ($buffer:expr, $call:ident($($arg:expr),*)) => {
        match $buffer.dtype() {
            DType::U8 => $call::<u8>($($arg),*),
            DType::I8 => $call::<i8>($($arg),*),
            DType::U16 => $call::<u16>($($arg),*),
            DType::I16 => $call::<i16>($($arg),*),
            DType::U32 => $call::<u32>($($arg),*),
            DType::I32 => $call::<i32>($($arg),*),
            DType::U64 => $call::<u64>($($arg),*),
            DType::I64 => $call::<i64>($($arg),*),
            DType::F32 => $call::<f32>($($arg),*),
            DType::F64 => $call::<f64>($($arg),*),
        }
    };
}

/// Pad a `[H, W]` or `[H, W, C]` buffer.
///
/// COST: full data copy — O(output_H × output_W × C); always allocates.
pub fn pad(
    buffer: &ViewBuffer,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    value: f32,
    mode: PadMode,
) -> ViewBuffer {
    dispatch_dtype!(
        buffer,
        pad_generic(buffer, top, bottom, left, right, value, mode)
    )
}

/// The `(top, bottom, left, right)` amounts that pad an `in_h × in_w` buffer
/// to `height × width` at the given position.
///
/// Shared by [`pad_to_size`] and plan-time shape math so placement cannot
/// diverge between planning and execution. Inputs larger than the target
/// saturate to zero padding on that axis.
pub fn pad_to_size_offsets(
    in_h: usize,
    in_w: usize,
    height: u32,
    width: u32,
    position: PadPosition,
) -> (u32, u32, u32, u32) {
    let pad_h = (height as usize).saturating_sub(in_h) as u32;
    let pad_w = (width as usize).saturating_sub(in_w) as u32;
    match position {
        PadPosition::Center => {
            let t = pad_h / 2;
            let l = pad_w / 2;
            (t, pad_h - t, l, pad_w - l)
        }
        PadPosition::TopLeft => (0, pad_h, 0, pad_w),
        PadPosition::BottomRight => (pad_h, 0, pad_w, 0),
    }
}

/// Constant-pad a buffer to an exact `height × width` at the given position.
pub fn pad_to_size(
    buffer: &ViewBuffer,
    height: u32,
    width: u32,
    position: PadPosition,
    value: f32,
) -> ViewBuffer {
    let shape = buffer.shape();
    let (top, bottom, left, right) =
        pad_to_size_offsets(shape[0], shape[1], height, width, position);
    pad(buffer, top, bottom, left, right, value, PadMode::Constant)
}

fn pad_generic<T: FillValue>(
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

    let contig = buffer.to_contiguous();
    let input = contig.as_slice::<T>();
    let fill = T::from_f32(value);
    let mut output = vec![fill; output_h * output_w * channels];

    match mode {
        PadMode::Constant => {
            // Interior rows are memcpy'd; the fill initialization covers borders.
            for y in 0..input_h {
                let src = y * input_row;
                let dst = (y + top) * output_row + left * channels;
                output[dst..dst + input_row].copy_from_slice(&input[src..src + input_row]);
            }
        }
        PadMode::Edge => {
            for dst_y in 0..output_h {
                let src_y = dst_y.saturating_sub(top).min(input_h - 1);
                let src_row_start = src_y * input_row;
                let dst_row_start = dst_y * output_row;
                // Left border: replicate the first pixel.
                let first = &input[src_row_start..src_row_start + channels];
                for i in 0..left {
                    let d = dst_row_start + i * channels;
                    output[d..d + channels].copy_from_slice(first);
                }
                // Interior: copy the source row.
                let d = dst_row_start + left * channels;
                output[d..d + input_row]
                    .copy_from_slice(&input[src_row_start..src_row_start + input_row]);
                // Right border: replicate the last pixel.
                let last_start = src_row_start + (input_w - 1) * channels;
                let last = &input[last_start..last_start + channels];
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
                    let s = (src_y * input_w + src_x) * channels;
                    let d = (dst_y * output_w + dst_x) * channels;
                    output[d..d + channels].copy_from_slice(&input[s..s + channels]);
                }
            }
        }
    }

    let out_shape = if shape.len() == 2 {
        vec![output_h, output_w]
    } else {
        vec![output_h, output_w, channels]
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn buf_u8(data: Vec<u8>, shape: Vec<usize>) -> ViewBuffer {
        ViewBuffer::from_vec_with_shape(data, shape)
    }

    /// Padding preserves the element dtype for EVERY dtype — the contract the
    /// old f32-casting implementation violated for e.g. u16.
    #[test]
    fn pad_preserves_dtype_for_all_dtypes() {
        macro_rules! check {
            ($t:ty, $dt:expr) => {{
                let buf = ViewBuffer::from_vec_with_shape(vec![1 as $t; 4], vec![2, 2, 1]);
                let padded = pad(&buf, 1, 1, 1, 1, 7.0, PadMode::Constant);
                assert_eq!(padded.dtype(), $dt, "dtype must be preserved");
                assert_eq!(padded.shape(), &[4, 4, 1]);
                let data = padded.as_slice::<$t>();
                assert_eq!(data[0], 7 as $t, "border carries the fill value");
                assert_eq!(data[1 * 4 + 1], 1 as $t, "interior preserved");
            }};
        }
        check!(u8, DType::U8);
        check!(i8, DType::I8);
        check!(u16, DType::U16);
        check!(i16, DType::I16);
        check!(u32, DType::U32);
        check!(i32, DType::I32);
        check!(u64, DType::U64);
        check!(i64, DType::I64);
        check!(f32, DType::F32);
        check!(f64, DType::F64);
    }

    #[test]
    fn pad_preserves_rank_for_2d_input() {
        let buf = buf_u8(vec![1, 2, 3, 4], vec![2, 2]);
        let padded = pad(&buf, 1, 0, 0, 0, 0.0, PadMode::Constant);
        assert_eq!(padded.shape(), &[3, 2], "2-D input must stay 2-D");
    }

    /// NumPy semantics for reflect/symmetric (doc examples, 1 row).
    #[test]
    fn pad_reflect_symmetric_match_numpy_semantics() {
        let buf = buf_u8(vec![1, 2, 3, 4], vec![1, 4, 1]);
        // np.pad([1,2,3,4], 3, mode="reflect") -> [4,3,2, 1,2,3,4, 3,2,1]
        let reflect = pad(&buf, 0, 0, 3, 3, 0.0, PadMode::Reflect);
        assert_eq!(reflect.as_slice::<u8>(), &[4, 3, 2, 1, 2, 3, 4, 3, 2, 1]);
        // np.pad([1,2,3,4], 3, mode="symmetric") -> [3,2,1, 1,2,3,4, 4,3,2]
        let symmetric = pad(&buf, 0, 0, 3, 3, 0.0, PadMode::Symmetric);
        assert_eq!(symmetric.as_slice::<u8>(), &[3, 2, 1, 1, 2, 3, 4, 4, 3, 2]);
    }

    #[test]
    fn pad_edge_replicates_borders() {
        let buf = buf_u8(vec![1, 2, 3, 4], vec![2, 2, 1]);
        let padded = pad(&buf, 1, 1, 1, 1, 0.0, PadMode::Edge);
        assert_eq!(padded.shape(), &[4, 4, 1]);
        let d = padded.as_slice::<u8>();
        assert_eq!(d[0], 1, "top-left corner replicates [0,0]");
        assert_eq!(d[3], 2, "top-right corner replicates [0,1]");
        assert_eq!(d[12], 3, "bottom-left corner replicates [1,0]");
        assert_eq!(d[15], 4, "bottom-right corner replicates [1,1]");
    }

    #[test]
    fn pad_to_size_positions() {
        let buf = buf_u8(vec![9; 4], vec![2, 2, 1]);
        let center = pad_to_size(&buf, 4, 4, PadPosition::Center, 0.0);
        assert_eq!(center.shape(), &[4, 4, 1]);
        assert_eq!(center.as_slice::<u8>()[4 + 1], 9, "content centered");

        let tl = pad_to_size(&buf, 4, 4, PadPosition::TopLeft, 0.0);
        assert_eq!(tl.as_slice::<u8>()[0], 9, "content at top-left");

        let br = pad_to_size(&buf, 4, 4, PadPosition::BottomRight, 0.0);
        assert_eq!(br.as_slice::<u8>()[15], 9, "content at bottom-right");

        // Input larger than the target: no padding on that axis.
        let same = pad_to_size(&buf, 1, 1, PadPosition::Center, 0.0);
        assert_eq!(same.shape(), &[2, 2, 1]);
    }

    #[test]
    fn pad_to_size_offsets_are_shared_math() {
        assert_eq!(
            pad_to_size_offsets(2, 2, 5, 4, PadPosition::Center),
            (1, 2, 1, 1)
        );
        assert_eq!(
            pad_to_size_offsets(2, 2, 4, 4, PadPosition::TopLeft),
            (0, 2, 0, 2)
        );
        assert_eq!(
            pad_to_size_offsets(2, 2, 4, 4, PadPosition::BottomRight),
            (2, 0, 2, 0)
        );
    }
}
