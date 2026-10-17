//! There is one luma: `grayscale()` and `convert_color(rgb -> gray)` give the
//! same pixel on every dtype.
//!
//! BT.601 used to be written out in five places. `convert_color` computed it
//! in f32 for i8/u16/i16/f32, where `grayscale` computes in f64 and rounds
//! once, so the two disagreed on the same input (u16 by one unit, f32 in the
//! last bit).

#![cfg(feature = "image_interop")]

use view_buffer::ops::color::{ColorConvertOp, ColorSpace};
use view_buffer::{DType, ImageOp, ImageOpKind, ViewBuffer, ViewDto, ViewExpr};

fn run(buf: &ViewBuffer, dto: ViewDto) -> ViewBuffer {
    ViewExpr::new_source(buf.clone())
        .apply_op(dto)
        .plan()
        .execute()
        .to_contiguous()
}

#[test]
fn convert_color_to_gray_is_grayscale_on_every_dtype() {
    let (h, w) = (9, 11);
    // Values spread over each dtype's range: odd, fractional once scaled,
    // so a rounding difference shows.
    let pattern: Vec<f64> = (0..h * w * 3)
        .map(|i| ((i * 7919) % 1000) as f64 / 1000.0)
        .collect();
    let mut differ = Vec::new();
    for &dtype in DType::ALL {
        let (lo, hi) = match dtype {
            DType::F32 | DType::F64 => (-3.7, 1234.567),
            // Up to just under each integer dtype's top (f64 holds them all
            // to 2^53; the pattern needs only the spread, not the extremes).
            d => (0.0, d.norm_range_max_f32().min(9.0e15) as f64),
        };
        let values: Vec<f64> = pattern.iter().map(|v| lo + v * (hi - lo)).collect();
        let buf = ViewBuffer::from_vec_with_shape(values, vec![h, w, 3]).cast(dtype);
        let gray = run(
            &buf,
            ViewDto::Image(ImageOp {
                kind: ImageOpKind::Grayscale,
            }),
        );
        let cvt = run(
            &buf,
            ViewDto::Color(ColorConvertOp {
                from_space: ColorSpace::Rgb,
                to_space: ColorSpace::Gray,
            }),
        );
        assert_eq!(gray.dtype(), cvt.dtype(), "{dtype:?}");
        let bits = |b: &ViewBuffer| -> Vec<u64> {
            let f = b.cast(DType::F64);
            f.as_slice::<f64>().iter().map(|v| v.to_bits()).collect()
        };
        if bits(&gray) != bits(&cvt) {
            differ.push(format!("{dtype:?}"));
        }
    }
    assert!(
        differ.is_empty(),
        "convert_color(rgb -> gray) != grayscale() on {differ:?}"
    );
}
