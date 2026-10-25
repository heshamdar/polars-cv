//! Mask application kernels: a mask broadcast across channels, inverted.

use std::mem::MaybeUninit;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::ViewType;
use crate::core::map::{map_new, ElementMap};

/// A `[h, w]` mask broadcast to `[h, w, c]` and packed (inverted on the way
/// when `invert`): a stride-0 channel axis, read through its walk.
pub(crate) fn expand<T: ViewType>(
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
pub(crate) struct InvertMask;

// SAFETY: `map_into` writes every slot of `dst` (one `write` per `src` element).
unsafe impl ElementMap<u8, u8> for InvertMask {
    #[inline(always)]
    fn map_into(&self, src: &[u8], dst: &mut [MaybeUninit<u8>], _at: usize) {
        for (d, &m) in dst.iter_mut().zip(src) {
            d.write(255 - m);
        }
    }
}

// SAFETY: as above, for f32.
unsafe impl ElementMap<f32, f32> for InvertMask {
    #[inline(always)]
    fn map_into(&self, src: &[f32], dst: &mut [MaybeUninit<f32>], _at: usize) {
        for (d, &m) in dst.iter_mut().zip(src) {
            d.write(1.0 - m);
        }
    }
}
