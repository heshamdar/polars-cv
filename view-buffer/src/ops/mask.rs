//! Mask application: weighted blending of a buffer with a mask.
//!
//! Moved from the polars-cv plugin's graph executor so all buffer math lives
//! in the engine; the graph layer only resolves which node provides the mask.

use std::mem::MaybeUninit;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, ViewType};
use crate::core::map::{map_new, ElementMap};
use crate::ops::binary::BinaryOp;

/// Apply a mask to a buffer via normalized blending (`pixel * mask`).
///
/// Mask semantics depend on the buffer dtype:
/// - integer buffers: mask values in `[0, 255]`; 255 keeps the pixel, 0 hides
///   it, intermediate values blend proportionally.
/// - float buffers: mask values in `[0.0, 1.0]`.
///
/// A 2-D `[H, W]` mask applied to a 3-D `[H, W, C]` buffer is expanded across
/// the channel dimension. `invert` flips the mask (`255 - m` / `1.0 - m`).
/// Whether [`apply_mask`] can combine a buffer of `buffer_shape` with a mask of
/// `mask_shape` (CR-34).
///
/// Reads the same two facts `apply_mask` acts on: a 2-D mask over a 3-D buffer
/// is first expanded to the buffer's channel count, and the result is then
/// blended with the buffer, so it must broadcast against it. Stated through
/// `BinaryOp::Blend`'s own `validate` so the two cannot drift. Over what is
/// known of the shapes (the planner's call; execution passes known ones), so
/// an error is a verdict on a known size.
pub fn validate_mask(
    buffer_shape: &[crate::ops::Dim],
    mask_shape: &[crate::ops::Dim],
) -> Result<(), crate::ops::validation::ValidationError> {
    use crate::ops::traits::Op as _;
    let effective: Vec<crate::ops::Dim> = match (mask_shape, buffer_shape) {
        ([h, w], [_, _, c]) => vec![*h, *w, *c],
        _ => mask_shape.to_vec(),
    };
    let f32 = crate::PlannedDType::Known(DType::F32);
    BinaryOp::Blend.validate(&[buffer_shape, &effective], &[f32, f32])
}

pub fn apply_mask(buffer: &ViewBuffer, mask: &ViewBuffer, invert: bool) -> ViewBuffer {
    let buf_shape = buffer.shape();
    let mask_shape = mask.shape();

    // Cast mask to match buffer dtype for proper blending
    let mask_dtype = buffer.dtype();
    let is_float = matches!(mask_dtype, DType::F32 | DType::F64);

    let effective_mask = if mask_shape.len() == 2 && buf_shape.len() == 3 {
        let h = mask_shape[0];
        let w = mask_shape[1];
        let c = buf_shape[2];
        // The mask broadcast across the channels (a stride-0 axis) and
        // packed, inverted on the way when asked, straight from the view.
        if is_float {
            expand::<f32>(&mask.cast_to(DType::F32), (h, w, c), invert)
        } else {
            expand::<u8>(&mask.cast_to(DType::U8), (h, w, c), invert)
        }
    } else if invert {
        // Written once, straight from the mask where it lies.
        if is_float {
            map_new::<f32, f32, _>(&mask.cast_to(DType::F32), &InvertMask)
        } else {
            map_new::<u8, u8, _>(&mask.cast_to(DType::U8), &InvertMask)
        }
    } else {
        mask.clone()
    };
    BinaryOp::Blend.execute(buffer, &effective_mask)
}

/// A `[h, w]` mask broadcast to `[h, w, c]` and packed (inverted on the way
/// when `invert`): a stride-0 channel axis, read through its walk.
fn expand<T: ViewType>(
    mask: &ViewBuffer,
    (h, w, c): (usize, usize, usize),
    invert: bool,
) -> ViewBuffer
where
    InvertMask: ElementMap<T, T>,
{
    let broadcast = mask.broadcast_to(&[c, h, w]).permute(&[1, 2, 0]);
    if invert {
        map_new::<T, T, _>(&broadcast, &InvertMask)
    } else {
        broadcast.to_contiguous()
    }
}

/// A mask inverted: `255 - m` for a u8 mask, `1 - m` for an f32 one.
struct InvertMask;

// SAFETY (both): `map_into` writes every slot of `dst`.
unsafe impl ElementMap<u8, u8> for InvertMask {
    #[inline(always)]
    fn map_into(&self, src: &[u8], dst: &mut [MaybeUninit<u8>], _at: usize) {
        for (d, &m) in dst.iter_mut().zip(src) {
            d.write(255 - m);
        }
    }
}

unsafe impl ElementMap<f32, f32> for InvertMask {
    #[inline(always)]
    fn map_into(&self, src: &[f32], dst: &mut [MaybeUninit<f32>], _at: usize) {
        for (d, &m) in dst.iter_mut().zip(src) {
            d.write(1.0 - m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_mask_expands_2d_mask_and_inverts() {
        // u8 path: 2x1 image with 2 channels, mask keeps row 0 and hides row 1.
        let buffer = ViewBuffer::from_vec_with_shape(vec![10u8, 20, 30, 40], vec![2, 1, 2]);
        let mask = ViewBuffer::from_vec_with_shape(vec![255u8, 0], vec![2, 1]);

        let masked = apply_mask(&buffer, &mask, false);
        assert_eq!(masked.as_slice::<u8>(), &[10, 20, 0, 0]);

        let inverted = apply_mask(&buffer, &mask, true);
        assert_eq!(inverted.as_slice::<u8>(), &[0, 0, 30, 40]);
    }

    #[test]
    fn apply_mask_float_path() {
        // f32 path: mask values in [0, 1].
        let buffer = ViewBuffer::from_vec_with_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 1, 2]);
        let mask = ViewBuffer::from_vec_with_shape(vec![1.0f32, 0.5], vec![2, 1]);

        let masked = apply_mask(&buffer, &mask, false);
        assert_eq!(masked.as_slice::<f32>(), &[1.0, 2.0, 1.5, 2.0]);

        let inverted = apply_mask(&buffer, &mask, true);
        assert_eq!(inverted.as_slice::<f32>(), &[0.0, 0.0, 1.5, 2.0]);
    }
}
