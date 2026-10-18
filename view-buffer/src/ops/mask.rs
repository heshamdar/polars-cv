//! Mask application: weighted blending of a buffer with a mask.
//!
//! Moved from the polars-cv plugin's graph executor so all buffer math lives
//! in the engine; the graph layer only resolves which node provides the mask.

use std::mem::MaybeUninit;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, PlannedDType, ViewType};
use crate::core::map::{map_new, ElementMap};
use crate::ops::binary::BinaryOp;

/// Whether [`apply_mask`] can combine a buffer of `buffer_shape` with a mask of
/// `mask_shape` and `mask_dtype` (CR-34, CR-110).
///
/// The mask's dtype says its scale, so only a u8 mask (`[0, 255]`) or a float
/// one (`[0, 1]`) has one; any other is refused, never read on a guessed
/// scale. Its shape: a single-channel mask over a 3-D buffer is first
/// expanded to the buffer's channel count, and the result is then weighed
/// against the buffer, so it must have the buffer's shape. Stated through
/// `BinaryOp::Blend`'s own `validate` so the two cannot drift. Over what is
/// known (the planner's call; execution passes known facts), so an error is a
/// verdict on a known size or dtype.
pub fn validate_mask(
    buffer_shape: &[crate::ops::Dim],
    mask_shape: &[crate::ops::Dim],
    mask_dtype: PlannedDType,
) -> Result<(), crate::ops::validation::ValidationError> {
    use crate::ops::traits::Op as _;
    if let PlannedDType::Known(dtype) = mask_dtype {
        if !matches!(dtype, DType::U8 | DType::F32 | DType::F64) {
            return Err(crate::ops::validation::ValidationError::Generic {
                message: format!(
                    "apply_mask: a mask is u8 (0 to 255) or float (0 to 1), got {}; \
                     cast or scale it first",
                    dtype.short_name()
                ),
            });
        }
    }
    let effective: Vec<crate::ops::Dim> = match (mask_shape, buffer_shape) {
        ([h, w], [_, _, c]) | ([h, w, crate::ops::Dim::Known(1)], [_, _, c]) => vec![*h, *w, *c],
        // Over a 2-D buffer a mask of one channel, or of a count only rows
        // know, is read as `[H, W]` (each row then checked on its own).
        ([h, w, c], [_, _]) if c.known().is_none_or(|c| c == 1) => vec![*h, *w],
        _ => mask_shape.to_vec(),
    };
    let f32 = crate::PlannedDType::Known(DType::F32);
    BinaryOp::Blend.validate(&[buffer_shape, &effective], &[f32, f32])
}

/// Weigh each pixel by its mask value; the result keeps the buffer's dtype.
///
/// The mask's own dtype says its scale, whatever the buffer's: a u8 mask
/// weighs `m / 255` (255 keeps the pixel, 0 hides it), a float mask `m` (in
/// `[0, 1]`). An integer result is rounded half away from zero and saturated
/// (the conversion rule); a u8 buffer under a u8 mask is `round(a·m / 255)`
/// in integers, as `BinaryOp::Blend` computes it.
///
/// A single-channel mask (`[H, W]` or `[H, W, 1]`) applied to a 3-D
/// `[H, W, C]` buffer is expanded across the channels, a `[H, W, 1]` one over
/// a `[H, W]` buffer is read as `[H, W]`; any other mask has the buffer's
/// shape. `invert` flips the mask (`255 - m` / `1.0 - m`). Masks of
/// other dtypes are refused by [`validate_mask`] before this runs.
pub fn apply_mask(buffer: &ViewBuffer, mask: &ViewBuffer, invert: bool) -> ViewBuffer {
    let buf_shape = buffer.shape();
    // A `[H, W, 1]` mask over channels, or over a `[H, W]` buffer, is read as
    // the `[H, W]` one it holds.
    let single =
        matches!(buf_shape.len(), 2 | 3) && mask.shape().len() == 3 && mask.shape()[2] == 1;
    let mask = &if single {
        mask.remove_axis(2)
    } else {
        mask.clone()
    };
    let mask_shape = mask.shape();

    // The mask stays on its own scale: u8, or float as f32.
    let is_float = matches!(mask.dtype(), DType::F32 | DType::F64);

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
    if buffer.dtype() == DType::U8 && !is_float {
        return BinaryOp::Blend.execute(buffer, &effective_mask);
    }
    let shape = buffer.shape().to_vec();
    let weighed = if is_float {
        crate::ops::binary::zip_with::<f64, _>(buffer, &effective_mask, &shape, |a, m| a * m)
    } else {
        crate::ops::binary::zip_with::<f64, _>(buffer, &effective_mask, &shape, |a, m| {
            a * m / 255.0
        })
    };
    weighed.cast_to(buffer.dtype())
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

    /// A `[H, W, 1]` mask (a rasterized one) spreads over the channels as
    /// the `[H, W]` mask it holds; it is `apply_mask`'s to spread, as the
    /// binary ops broadcast nothing.
    #[test]
    fn a_single_channel_mask_spreads_like_a_2d_one() {
        let buffer = ViewBuffer::from_vec_with_shape(vec![10u8, 20, 30, 40, 50, 60], vec![2, 1, 3]);
        let flat = ViewBuffer::from_vec_with_shape(vec![255u8, 0], vec![2, 1]);
        let single = ViewBuffer::from_vec_with_shape(vec![255u8, 0], vec![2, 1, 1]);
        let masked = |m: &ViewBuffer, invert| apply_mask(&buffer, m, invert);
        for invert in [false, true] {
            let got = masked(&single, invert);
            assert_eq!(got.shape(), &[2, 1, 3]);
            assert_eq!(got.as_slice::<u8>(), masked(&flat, invert).as_slice::<u8>());
        }
        assert_eq!(
            masked(&single, false).as_slice::<u8>(),
            &[10, 20, 30, 0, 0, 0]
        );
        let dims = |s: &[usize]| {
            s.iter()
                .map(|&n| crate::ops::Dim::Known(n))
                .collect::<Vec<_>>()
        };
        let u8 = PlannedDType::Known(DType::U8);
        assert!(validate_mask(&dims(&[2, 1, 3]), &dims(&[2, 1, 1]), u8).is_ok());
        assert!(validate_mask(&dims(&[2, 1, 3]), &dims(&[2, 1, 2]), u8).is_err());
        // Over a [H, W] buffer a [H, W, 1] mask is read as [H, W].
        assert!(validate_mask(&dims(&[2, 1]), &dims(&[2, 1, 1]), u8).is_ok());
        let flat = ViewBuffer::from_vec_with_shape(vec![10u8, 20], vec![2, 1]);
        let got = apply_mask(&flat, &single, false);
        assert_eq!(
            (got.shape(), got.as_slice::<u8>()),
            (&[2, 1][..], &[10, 0][..])
        );
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

    /// The mask's dtype says its scale, whatever the buffer's (CR-110): a
    /// u8 255 keeps a float pixel, a u8 mask weighs a u16 one by `m / 255`,
    /// a float mask weighs a u8 one by `m`, and no other mask dtype passes.
    #[test]
    fn the_mask_dtype_sets_its_scale() {
        let mask = ViewBuffer::from_vec_with_shape(vec![255u8, 51], vec![2, 1]);
        let f = ViewBuffer::from_vec_with_shape(vec![2.0f32, 10.0], vec![2, 1]);
        assert_eq!(apply_mask(&f, &mask, false).as_slice::<f32>(), &[2.0, 2.0]);
        let w = ViewBuffer::from_vec_with_shape(vec![1000u16, 1000], vec![2, 1]);
        assert_eq!(apply_mask(&w, &mask, false).as_slice::<u16>(), &[1000, 200]);
        let half = ViewBuffer::from_vec_with_shape(vec![0.5f32, 0.25], vec![2, 1]);
        let b = ViewBuffer::from_vec_with_shape(vec![101u8, 100], vec![2, 1]);
        assert_eq!(apply_mask(&b, &half, false).as_slice::<u8>(), &[51, 25]);
        let dims = [crate::ops::Dim::Known(2), crate::ops::Dim::Known(1)];
        let refused = validate_mask(&dims, &dims, PlannedDType::Known(DType::U16));
        assert!(refused.unwrap_err().to_string().contains("got u16"));
        assert!(validate_mask(&dims, &dims, PlannedDType::Unknown).is_ok());
    }
}
