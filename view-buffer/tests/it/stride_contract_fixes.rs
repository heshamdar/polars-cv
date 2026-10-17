//! Plan-time stride contracts must match what kernels actually produce.
//!
//! Ops whose kernels always materialize a fresh contiguous buffer must
//! declare contiguous output strides at plan time. Declaring the input's
//! strides instead (a) propagates non-contiguous strides the real output
//! doesn't have, making a following `reshape` panic spuriously at build
//! time, and (b) mis-declares byte strides entirely when the op changes
//! element size (threshold: any input -> u8 mask; gamma: integer -> f32).

use std::sync::Arc;
use view_buffer::{DType, ViewBuffer, ViewExpr};

/// 4x4 u8 ramp 0..16.
fn u8_4x4() -> ViewBuffer {
    ViewBuffer::from_vec_with_shape((0u8..16).collect::<Vec<_>>(), vec![4, 4])
}

/// 4x4 f32 ramp 0..16.
fn f32_4x4() -> ViewBuffer {
    ViewBuffer::from_vec_with_shape((0..16).map(|v| v as f32).collect::<Vec<_>>(), vec![4, 4])
}

/// Interior 2x2 crop => values [[5, 6], [9, 10]], non-contiguous view.
fn cropped(buf: ViewBuffer) -> Arc<ViewExpr> {
    ViewExpr::new_source(buf).crop(vec![1, 1], vec![3, 3])
}

#[cfg(feature = "image_interop")]
#[test]
fn threshold_after_crop_then_reshape() {
    let out = cropped(u8_4x4())
        .threshold(9.0)
        .reshape(vec![4])
        .plan()
        .execute();
    assert_eq!(out.dtype(), DType::U8);
    assert_eq!(out.shape(), &[4]);
    assert_eq!(out.to_contiguous().as_slice::<u8>(), &[0, 0, 0, 255]);
}

#[cfg(feature = "image_interop")]
#[test]
fn threshold_on_f32_input_then_reshape() {
    let buf = ViewBuffer::from_vec_with_shape(vec![0.1f32, 0.6, 0.4, 0.9, 0.2, 0.8], vec![2, 3]);
    let out = ViewExpr::new_source(buf)
        .threshold(0.5)
        .reshape(vec![6])
        .plan()
        .execute();
    assert_eq!(out.dtype(), DType::U8);
    assert_eq!(
        out.to_contiguous().as_slice::<u8>(),
        &[0, 255, 0, 255, 0, 255]
    );
}

#[cfg(feature = "image_interop")]
#[test]
fn threshold_output_is_contiguous() {
    let out = cropped(u8_4x4()).threshold(9.0).plan().execute();
    assert_eq!(out.shape(), &[2, 2]);
    // The kernel materializes a fresh contiguous u8 buffer.
    assert_eq!(out.strides_bytes(), &[2, 1]);
}

#[test]
fn adjust_gamma_u8_then_reshape() {
    // Gamma promotes u8 -> f32; the element size changes, so input byte
    // strides can never describe the output.
    let vals: Vec<u8> = vec![0, 51, 102, 153, 204, 255];
    let buf = ViewBuffer::from_vec_with_shape(vals.clone(), vec![2, 3]);
    let out = ViewExpr::new_source(buf)
        .adjust_gamma(1.0)
        .reshape(vec![6])
        .plan()
        .execute();
    assert_eq!(out.dtype(), DType::F32);
    let got = out.to_contiguous().as_slice::<f32>().to_vec();
    for (g, x) in got.iter().zip(vals) {
        assert!((g - x as f32).abs() < 1e-3, "got {g}, want {x}");
    }
}

#[test]
fn adjust_gamma_after_crop_then_reshape() {
    let out = cropped(f32_4x4())
        .adjust_gamma(2.0)
        .reshape(vec![4])
        .plan()
        .execute();
    assert_eq!(out.dtype(), DType::F32);
    // f32 path clamps to [0, 1] then squares: 5,6,9,10 all clamp to 1.0.
    assert_eq!(out.to_contiguous().as_slice::<f32>(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn invert_after_crop_then_reshape() {
    let out = cropped(u8_4x4()).invert().reshape(vec![4]).plan().execute();
    assert_eq!(out.dtype(), DType::U8);
    assert_eq!(out.to_contiguous().as_slice::<u8>(), &[250, 249, 246, 245]);
}

#[test]
fn clamp_after_crop_then_reshape() {
    let out = cropped(f32_4x4())
        .clamp(0.0, 8.0)
        .reshape(vec![4])
        .plan()
        .execute();
    assert_eq!(out.to_contiguous().as_slice::<f32>(), &[5.0, 6.0, 8.0, 8.0]);
}

#[test]
fn scale_after_crop_then_reshape() {
    let out = cropped(f32_4x4())
        .scale(2.0)
        .reshape(vec![4])
        .plan()
        .execute();
    assert_eq!(
        out.to_contiguous().as_slice::<f32>(),
        &[10.0, 12.0, 18.0, 20.0]
    );
}

#[test]
fn relu_after_crop_then_reshape() {
    let out = cropped(f32_4x4()).relu().reshape(vec![4]).plan().execute();
    assert_eq!(
        out.to_contiguous().as_slice::<f32>(),
        &[5.0, 6.0, 9.0, 10.0]
    );
}

#[test]
fn fused_chain_after_crop_then_reshape() {
    // scale + relu fuse into one kernel; the fused kernel also writes a
    // fresh contiguous buffer.
    let out = cropped(f32_4x4())
        .scale(2.0)
        .relu()
        .reshape(vec![4])
        .plan()
        .execute();
    assert_eq!(
        out.to_contiguous().as_slice::<f32>(),
        &[10.0, 12.0, 18.0, 20.0]
    );
}
