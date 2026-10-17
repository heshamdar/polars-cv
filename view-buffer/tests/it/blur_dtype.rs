//! Blur must operate on the input's native dtype (u8/u16/f32) and preserve it,
//! rather than force-downconverting to u8 (which silently lost precision and
//! changed the output dtype for f32/u16 images).
#![cfg(feature = "image_interop")]

use view_buffer::{DType, ViewBuffer, ViewExpr};

fn blur(buf: &ViewBuffer, sigma: f32) -> ViewBuffer {
    ViewExpr::new_source(buf.clone())
        .blur(sigma)
        .plan()
        .execute()
}

/// Interior pixel indices of an 8x8 single-channel image (avoids border-handling
/// ambiguity at the very edges).
fn interior_8x8() -> Vec<usize> {
    (2..6)
        .flat_map(|r| (2..6).map(move |c| r * 8 + c))
        .collect()
}

#[test]
fn blur_preserves_u16_dtype_and_range() {
    // 8x8 single-channel u16 with values far above the u8 range.
    let data: Vec<u16> = (0..64)
        .map(|i| {
            if (i / 8 + i % 8) % 2 == 0 {
                4000u16
            } else {
                1000
            }
        })
        .collect();
    let input = ViewBuffer::from_vec(data).reshape(vec![8, 8, 1]);

    let out = blur(&input, 1.5);

    assert_eq!(out.dtype(), DType::U16, "blur must preserve u16 dtype");
    let out = out.to_contiguous();
    let max = out.as_slice::<u16>().iter().copied().max().unwrap();
    assert!(
        max > 255,
        "u16 magnitudes must survive blur (max={max}); the old u8 path collapsed them to <=255"
    );
}

#[test]
fn blur_preserves_f32_dtype_and_subunit_precision() {
    // f32 image in [0,1]. The old path scaled to [0,255] and emitted u8,
    // destroying sub-unit precision; native f32 blur keeps values in [0,1].
    let data: Vec<f32> = (0..64)
        .map(|i| {
            if (i / 8 + i % 8) % 2 == 0 {
                0.8f32
            } else {
                0.2
            }
        })
        .collect();
    let input = ViewBuffer::from_vec(data).reshape(vec![8, 8, 1]);

    let out = blur(&input, 1.5);

    assert_eq!(out.dtype(), DType::F32, "blur must preserve f32 dtype");
    let out = out.to_contiguous();
    let vals = out.as_slice::<f32>();
    let max = vals.iter().copied().fold(f32::MIN, f32::max);
    assert!(
        max <= 1.0 + 1e-4,
        "f32 blur must stay in [0,1] (max={max}); the old path produced [0,255]"
    );
    assert!(
        vals.iter().any(|&v| v > 0.25 && v < 0.75),
        "blur should produce smoothed intermediate f32 values"
    );
}

#[test]
fn blur_of_solid_is_noop_per_dtype() {
    let interior = interior_8x8();

    // u16 constant.
    let out = blur(
        &ViewBuffer::from_vec(vec![1234u16; 64]).reshape(vec![8, 8, 1]),
        2.0,
    );
    assert_eq!(out.dtype(), DType::U16);
    let out = out.to_contiguous();
    let vals = out.as_slice::<u16>();
    for &i in &interior {
        assert_eq!(vals[i], 1234, "u16 solid blur changed interior pixel {i}");
    }

    // f32 constant.
    let out = blur(
        &ViewBuffer::from_vec(vec![0.37f32; 64]).reshape(vec![8, 8, 1]),
        2.0,
    );
    assert_eq!(out.dtype(), DType::F32);
    let out = out.to_contiguous();
    let vals = out.as_slice::<f32>();
    for &i in &interior {
        assert!(
            (vals[i] - 0.37).abs() < 1e-3,
            "f32 solid blur changed interior pixel {i}: {}",
            vals[i]
        );
    }

    // u8 constant (existing behavior preserved).
    let out = blur(
        &ViewBuffer::from_vec(vec![200u8; 64]).reshape(vec![8, 8, 1]),
        2.0,
    );
    assert_eq!(out.dtype(), DType::U8);
    let out = out.to_contiguous();
    let vals = out.as_slice::<u8>();
    for &i in &interior {
        assert_eq!(vals[i], 200, "u8 solid blur changed interior pixel {i}");
    }
}
