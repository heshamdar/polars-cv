//! Byte-parity for the image-ops batch (step 5c).
//!
//! Same template as the earlier parity test files. Single-channel inputs
//! for erode/dilate/morph_gradient (which require it); RGB for grayscale-
//! adjacent ops where shape isn't restricted.

use pcv_graph::op::{ExecCtx, OpInputs};
use pcv_graph::params::ParamMap;
use pcv_graph::registry::find_op;
use serde_json::json;
use view_buffer::expr::ViewExpr;
use view_buffer::ops::{ImageOp, ImageOpKind, NodeOutput, ViewDto};
use view_buffer::ViewBuffer;

fn make_gray_8x8() -> ViewBuffer {
    // Single-channel ramp 0..64 reshaped to 8x8
    let data: Vec<u8> = (0u8..64u8).collect();
    ViewBuffer::from_vec_with_shape(data, vec![8, 8])
}

fn make_rgb_8x8() -> ViewBuffer {
    // 8x8x3 RGB, varying channel intensities
    let mut data = Vec::with_capacity(8 * 8 * 3);
    for y in 0..8u8 {
        for x in 0..8u8 {
            data.push(x * 32);          // R
            data.push(y * 32);          // G
            data.push(((x + y) * 16).min(255));  // B
        }
    }
    ViewBuffer::from_vec_with_shape(data, vec![8, 8, 3])
}

fn v1_apply(input: ViewBuffer, dto: ViewDto) -> ViewBuffer {
    ViewExpr::new_source(input).apply_op(dto).plan().execute()
}

fn v2_apply(op_name: &str, params: ParamMap, input: ViewBuffer) -> ViewBuffer {
    let reg = find_op(op_name).unwrap();
    let op = (reg.factory)(&params).expect("factory");
    let input_node = NodeOutput::from_buffer(input);
    let out = op
        .execute(&ExecCtx::new(0), &OpInputs::single(&input_node))
        .expect("execute");
    (**out.as_buffer().unwrap()).clone()
}

fn buffers_eq(a: &ViewBuffer, b: &ViewBuffer) -> bool {
    let ac = a.to_contiguous();
    let bc = b.to_contiguous();
    let (pa, sa, _, da) = ac.as_raw_parts();
    let (pb, sb, _, db) = bc.as_raw_parts();
    if sa != sb || da != db {
        eprintln!("shape/dtype: v1={sa:?} {da:?}, v2={sb:?} {db:?}");
        return false;
    }
    let total: usize = sa.iter().product::<usize>() * da.size_of();
    unsafe {
        std::slice::from_raw_parts(pa, total) == std::slice::from_raw_parts(pb, total)
    }
}

#[test]
fn threshold_byte_parity() {
    let v1 = v1_apply(
        make_gray_8x8(),
        ViewDto::Image(ImageOp { kind: ImageOpKind::Threshold(32.0) }),
    );
    let mut params = ParamMap::new();
    params.insert("value".into(), json!(32.0));
    let v2 = v2_apply("threshold", params, make_gray_8x8());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn blur_byte_parity() {
    let v1 = v1_apply(
        make_rgb_8x8(),
        ViewDto::Image(ImageOp { kind: ImageOpKind::Blur { sigma: 1.5 } }),
    );
    let mut params = ParamMap::new();
    params.insert("sigma".into(), json!(1.5));
    let v2 = v2_apply("blur", params, make_rgb_8x8());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn canny_byte_parity() {
    let v1 = v1_apply(
        make_rgb_8x8(),
        ViewDto::Image(ImageOp {
            kind: ImageOpKind::Canny { low_threshold: 50.0, high_threshold: 150.0 },
        }),
    );
    let mut params = ParamMap::new();
    params.insert("low_threshold".into(), json!(50.0));
    params.insert("high_threshold".into(), json!(150.0));
    let v2 = v2_apply("canny", params, make_rgb_8x8());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn erode_byte_parity() {
    let v1 = v1_apply(
        make_gray_8x8(),
        ViewDto::Image(ImageOp {
            kind: ImageOpKind::Erode { ksize: 3, iterations: 1 },
        }),
    );
    let mut params = ParamMap::new();
    params.insert("ksize".into(), json!(3));
    let v2 = v2_apply("erode", params, make_gray_8x8());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn dilate_byte_parity() {
    let v1 = v1_apply(
        make_gray_8x8(),
        ViewDto::Image(ImageOp {
            kind: ImageOpKind::Dilate { ksize: 3, iterations: 1 },
        }),
    );
    let mut params = ParamMap::new();
    params.insert("ksize".into(), json!(3));
    let v2 = v2_apply("dilate", params, make_gray_8x8());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn morphology_gradient_byte_parity() {
    let v1 = v1_apply(
        make_gray_8x8(),
        ViewDto::Image(ImageOp { kind: ImageOpKind::MorphGradient { ksize: 3 } }),
    );
    let mut params = ParamMap::new();
    params.insert("ksize".into(), json!(3));
    let v2 = v2_apply("morphology_gradient", params, make_gray_8x8());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn equalize_histogram_byte_parity() {
    let v1 = v1_apply(
        make_gray_8x8(),
        ViewDto::Image(ImageOp { kind: ImageOpKind::HistogramEqualize }),
    );
    let v2 = v2_apply("equalize_histogram", ParamMap::new(), make_gray_8x8());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn all_image_ops_registered() {
    for name in [
        "threshold",
        "blur",
        "canny",
        "erode",
        "dilate",
        "morphology_gradient",
        "equalize_histogram",
    ] {
        assert!(find_op(name).is_some(), "{name} should be registered");
    }
}
