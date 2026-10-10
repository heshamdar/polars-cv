//! Mask application: weighted blending of a buffer with a mask.
//!
//! Moved from the polars-cv plugin's graph executor so all buffer math lives
//! in the engine; the graph layer only resolves which node provides the mask.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
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
        // Expanded straight from the mask where it lies (its runs, in
        // logical order): packing a mask view first was a second mask-sized
        // copy.
        fn expand<T: crate::core::dtype::ViewType>(
            mask: &ViewBuffer,
            (h, w, c): (usize, usize, usize),
            f: impl Fn(T) -> T,
        ) -> ViewBuffer {
            let mut expanded: Vec<T> = Vec::with_capacity(h * w * c);
            crate::core::map::for_each_run::<T>(mask, |run| {
                for &raw in run {
                    expanded.extend(std::iter::repeat_n(f(raw), c));
                }
            });
            ViewBuffer::from_vec_with_shape(expanded, vec![h, w, c])
        }
        if is_float {
            // For float buffers, mask values should be in [0, 1]
            expand(&mask.cast_to(DType::F32), (h, w, c), |v: f32| {
                if invert {
                    1.0 - v
                } else {
                    v
                }
            })
        } else {
            // For U8 buffers, mask values in [0, 255]
            expand(&mask.cast_to(DType::U8), (h, w, c), |v: u8| {
                if invert {
                    255 - v
                } else {
                    v
                }
            })
        }
    } else if invert {
        // Written once, straight from the mask where it lies (`append_to`
        // walks a view), then inverted in place: packing the view first was
        // a second mask-sized copy.
        if is_float {
            let mut inverted: Vec<f32> = Vec::new();
            mask.cast_to(DType::F32).append_to(&mut inverted);
            inverted.iter_mut().for_each(|v| *v = 1.0 - *v);
            ViewBuffer::from_vec_with_shape(inverted, mask_shape.to_vec())
        } else {
            let mut inverted: Vec<u8> = Vec::new();
            mask.cast_to(DType::U8).append_to(&mut inverted);
            inverted.iter_mut().for_each(|v| *v = 255 - *v);
            ViewBuffer::from_vec_with_shape(inverted, mask_shape.to_vec())
        }
    } else {
        mask.clone()
    };
    BinaryOp::Blend.execute(buffer, &effective_mask)
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
