//! Kernels: the loops that compute an op's output, one module per kind of computation.
//!
//! **The rule** (`view-buffer/AGENTS.md`, "Where code goes"): `ops/<family>.rs`
//! owns an op's contract (shape, dtype, validation), its parameters and its
//! `execute` entry; the kernel bodies the entry calls live here, in the module
//! of their kind. `execution/runner.rs` runs plans, and `ViewDto::execute`
//! is the one dispatch from a plan step to its family's entry.

#[cfg(test)]
use crate::core::dtype::DType;

pub(crate) mod binary;
pub(crate) mod blur;
pub(crate) mod canny;
pub(crate) mod channel;
pub(crate) mod color;
pub(crate) mod distance;
pub(crate) mod elementwise;
pub(crate) mod equalize;
pub(crate) mod filter;
pub(crate) mod grid;
pub(crate) mod histogram;
pub(crate) mod lut;
pub(crate) mod mask;
pub(crate) mod morph;
pub(crate) mod pad;
pub(crate) mod phash;
pub(crate) mod pixel;
pub(crate) mod rank;
pub(crate) mod reduction;
pub(crate) mod resample;
pub(crate) mod resize;
pub(crate) mod warp;

/// Check if a shape represents a single-channel image.
///
/// Valid single-channel shapes:
/// - `[H, W]` - 2D array
/// - `[H, W, 1]` - 3D with 1 channel
///
/// Invalid (multi-channel):
/// - `[H, W, C]` where C > 1
#[inline]
fn is_single_channel(shape: &[usize]) -> bool {
    match shape.len() {
        2 => true,          // [H, W] - 2D is single channel
        3 => shape[2] == 1, // [H, W, 1] - explicit single channel
        _ => false,         // Other ranks not supported
    }
}

/// Get the number of channels from a shape.
#[inline]
fn get_channel_count(shape: &[usize]) -> usize {
    match shape.len() {
        2 => 1,        // [H, W] - implicit single channel
        3 => shape[2], // [H, W, C]
        _ => 0,        // Invalid
    }
}

/// Clamp an `f64` value to the representable range of the given integer
/// `DType`: the store the pre-M5 kernels used, kept for the verbatim oracles
/// of the warp and blur tests. Every kernel now stores by `CastFrom`.
#[cfg(test)]
pub(super) fn clamp_for_dtype(v: f64, dtype: DType) -> f64 {
    match dtype {
        // Round before clamping so that e.g. 127.9999 → 128, not 127.
        // NumCast::from truncates (floor) for positive floats, so without rounding
        // Gaussian convolution on a solid-128 image gives 127.
        DType::U8 => v.round().clamp(0.0, u8::MAX as f64),
        DType::I8 => v.round().clamp(i8::MIN as f64, i8::MAX as f64),
        DType::U16 => v.round().clamp(0.0, u16::MAX as f64),
        DType::I16 => v.round().clamp(i16::MIN as f64, i16::MAX as f64),
        DType::U32 => v.round().clamp(0.0, u32::MAX as f64),
        DType::I32 => v.round().clamp(i32::MIN as f64, i32::MAX as f64),
        DType::U64 => v.round().clamp(0.0, u64::MAX as f64),
        DType::I64 => v.round().clamp(i64::MIN as f64, i64::MAX as f64),
        DType::F32 | DType::F64 => v,
    }
}
