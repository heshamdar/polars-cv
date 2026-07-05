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
        if is_float {
            // For float buffers, mask values should be in [0, 1]
            let mask_f32 = mask.cast_to(DType::F32).to_contiguous();
            let mask_data = mask_f32.as_slice::<f32>();
            let mut expanded: Vec<f32> = Vec::with_capacity(h * w * c);
            for y in 0..h {
                for x in 0..w {
                    let raw_val = mask_data[y * w + x];
                    let mask_val = if invert { 1.0 - raw_val } else { raw_val };
                    for _ in 0..c {
                        expanded.push(mask_val);
                    }
                }
            }
            ViewBuffer::from_vec_with_shape(expanded, vec![h, w, c])
        } else {
            // For U8 buffers, mask values in [0, 255]
            let mask_contig = mask.cast_to(DType::U8).to_contiguous();
            let mask_data = mask_contig.as_slice::<u8>();
            let mut expanded: Vec<u8> = Vec::with_capacity(h * w * c);
            for y in 0..h {
                for x in 0..w {
                    let raw_val = mask_data[y * w + x];
                    let mask_val = if invert { 255 - raw_val } else { raw_val };
                    for _ in 0..c {
                        expanded.push(mask_val);
                    }
                }
            }
            ViewBuffer::from_vec_with_shape(expanded, vec![h, w, c])
        }
    } else if invert {
        if is_float {
            let mask_f32 = mask.cast_to(DType::F32).to_contiguous();
            let mask_data = mask_f32.as_slice::<f32>();
            let inverted: Vec<f32> = mask_data.iter().map(|&v| 1.0 - v).collect();
            ViewBuffer::from_vec_with_shape(inverted, mask_shape.to_vec())
        } else {
            let mask_contig = mask.cast_to(DType::U8).to_contiguous();
            let mask_data = mask_contig.as_slice::<u8>();
            let inverted: Vec<u8> = mask_data.iter().map(|&v| 255 - v).collect();
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
