//! An op that reads its input one packed row at a time (`RequiresDenseRows`)
//! gives a crop or a flip exactly what it gives the same view packed.
//!
//! The planner hands such an op a crop or a vertical flip where it lies, at its
//! own row stride, gap or negative stride included, so the kernel's row
//! arithmetic is what this holds. A horizontal flip and a transpose are packed
//! by the planner first and are here as the control.

#![cfg(feature = "image_interop")]

use view_buffer::ops::color::{ColorConvertOp, ColorSpace};
use view_buffer::ops::filter::{BorderMode, ConvolveOp};
use view_buffer::ops::pad::{PadMode, PadPosition};
use view_buffer::ops::{InputLayout, Op};
use view_buffer::{DType, ImageOp, ImageOpKind, ViewBuffer, ViewDto, ViewExpr};

/// A patterned `[h, w, c]` image of `T`.
fn image<T: view_buffer::core::dtype::ViewType + num_traits::FromPrimitive>(
    h: usize,
    w: usize,
    c: usize,
) -> ViewBuffer {
    let data: Vec<T> = (0..h * w * c)
        .map(|i| T::from_usize(i * 37 % 241).unwrap())
        .collect();
    ViewBuffer::from_vec_with_shape(data, vec![h, w, c])
}

/// The views a dense-rows op is handed in place, and the packed controls.
fn views(base: &ViewBuffer) -> Vec<(&'static str, ViewBuffer)> {
    let [h, w, c] = base.shape()[..] else {
        unreachable!()
    };
    vec![
        ("crop", base.slice(&[3, 2, 0], &[h - 2, w - 3, c])),
        ("vflip", base.flip(&[0])),
        (
            "crop of vflip",
            base.flip(&[0]).slice(&[1, 4, 0], &[h - 5, w - 1, c]),
        ),
        ("hflip", base.flip(&[1])),
        ("transpose", base.permute(&[1, 0, 2])),
    ]
}

fn run(buf: ViewBuffer, op: &ViewDto) -> ViewBuffer {
    ViewExpr::new_source(buf)
        .apply_op(op.clone())
        .plan()
        .execute()
}

fn image_op(kind: ImageOpKind) -> ViewDto {
    ViewDto::Image(ImageOp { kind })
}

/// The dense-rows ops, as they are planned, for a `c`-channel `dtype` image.
fn ops(c: usize, dtype: DType) -> Vec<(&'static str, ViewDto)> {
    let mut ops = vec![
        ("blur", image_op(ImageOpKind::Blur { sigma: 1.3 })),
        (
            "pad",
            image_op(ImageOpKind::Pad {
                top: 2,
                bottom: 1,
                left: 3,
                right: 0,
                value: 7.0,
                mode: PadMode::Reflect,
            }),
        ),
        (
            "pad_to_size",
            image_op(ImageOpKind::PadToSize {
                height: 40,
                width: 33,
                position: PadPosition::Center,
                value: 5.0,
            }),
        ),
    ];
    if c == 1 {
        ops.extend([
            (
                "erode",
                image_op(ImageOpKind::Erode {
                    ksize: 3,
                    iterations: 2,
                }),
            ),
            (
                "dilate",
                image_op(ImageOpKind::Dilate {
                    ksize: 5,
                    iterations: 1,
                }),
            ),
            (
                "morph_gradient",
                image_op(ImageOpKind::MorphGradient { ksize: 3 }),
            ),
        ]);
    }
    if c == 3 {
        ops.push((
            "channel_swap",
            image_op(ImageOpKind::ChannelSwap {
                order: vec![2, 0, 1],
            }),
        ));
    }
    if dtype == DType::U8 {
        ops.push((
            "equalize_histogram",
            image_op(ImageOpKind::HistogramEqualize),
        ));
    }
    ops
}

/// Ops that read any layout through the cast their arithmetic starts with
/// (`StridePreserving`): a widening cast walks the view once.
fn cast_reading_ops(c: usize) -> Vec<(&'static str, ViewDto)> {
    let mut ops = vec![(
        "convolve2d",
        ViewDto::Filter(ConvolveOp {
            kernel: vec![1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0],
            normalize: false,
            border: BorderMode::Reflect,
        }),
    )];
    if c == 3 {
        ops.push((
            "cvt_color",
            ViewDto::Color(ColorConvertOp {
                from_space: ColorSpace::Rgb,
                to_space: ColorSpace::Hsv,
            }),
        ));
    }
    ops
}

fn layout_of(op: &ViewDto) -> InputLayout {
    match op {
        ViewDto::Image(op) => op.memory_effect().input_layout(),
        ViewDto::Filter(op) => op.memory_effect().input_layout(),
        ViewDto::Color(op) => op.memory_effect().input_layout(),
        other => panic!("not covered here: {other:?}"),
    }
}

/// `op` on each view equals `op` on the view's packed copy, byte for byte.
fn assert_reads_views_as_packed(name: &str, op: &ViewDto, base: &ViewBuffer) {
    for (layout, view) in views(base) {
        let got = run(view.clone(), op);
        let want = run(view.to_contiguous(), op);
        assert_eq!(got.shape(), want.shape(), "{name} on {layout}");
        assert_eq!(
            got.to_blob(),
            want.to_blob(),
            "{name} on {layout}, {:?} {:?}",
            base.dtype(),
            base.shape()
        );
    }
}

#[test]
fn a_dense_rows_op_reads_a_view_as_its_packed_copy() {
    for c in [1, 3] {
        for base in [
            image::<u8>(29, 23, c),
            image::<f32>(29, 23, c),
            image::<u16>(29, 23, c),
        ] {
            for (name, op) in ops(c, base.dtype()) {
                assert_eq!(layout_of(&op), InputLayout::DenseRows, "{name}");
                assert_reads_views_as_packed(name, &op, &base);
            }
        }
    }
}

#[test]
fn a_cast_reading_op_reads_a_view_as_its_packed_copy() {
    for c in [1, 3] {
        for base in [
            image::<u8>(29, 23, c),
            image::<f32>(29, 23, c),
            image::<u16>(29, 23, c),
        ] {
            for (name, op) in cast_reading_ops(c) {
                assert_eq!(layout_of(&op), InputLayout::Any, "{name}");
                assert_reads_views_as_packed(name, &op, &base);
            }
        }
    }
}

/// The planner hands a dense-rows op a crop where it lies, and a cast-reading
/// op any view.
#[test]
fn the_planner_packs_none_of_these_views() {
    let base = image::<u8>(29, 23, 3);
    let gray = image::<u8>(29, 23, 1);
    let cases = ops(3, DType::U8)
        .into_iter()
        .map(|(n, op)| (n, op, base.slice(&[3, 2, 0], &[26, 20, 3])))
        .chain(
            ops(1, DType::U8)
                .into_iter()
                .map(|(n, op)| (n, op, gray.flip(&[0]))),
        )
        .chain(
            cast_reading_ops(3)
                .into_iter()
                .map(|(n, op)| (n, op, base.permute(&[1, 0, 2]))),
        );
    for (name, op, view) in cases {
        let plan = ViewExpr::new_source(view).apply_op(op).plan();
        assert!(
            !format!("{:?}", plan.steps).contains("Materialize"),
            "{name}: the planner packs a view the op reads in place: {:?}",
            plan.steps
        );
    }
}
