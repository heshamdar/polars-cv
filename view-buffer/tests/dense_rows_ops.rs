//! An op that reads its input one packed row at a time (`RequiresDenseRows`)
//! gives a crop or a flip exactly what it gives the same view packed.
//!
//! The planner hands such an op a crop or a vertical flip where it lies, at its
//! own row stride, gap or negative stride included, so the kernel's row
//! arithmetic is what this holds. A horizontal flip and a transpose are packed
//! by the planner first and are here as the control.

#![cfg(feature = "image_interop")]

use view_buffer::ops::pad::{PadMode, PadPosition};
use view_buffer::ops::{InputLayout, Op};
use view_buffer::{ImageOp, ImageOpKind, ViewBuffer, ViewDto, ViewExpr};

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

fn run(buf: ViewBuffer, op: &ImageOp) -> ViewBuffer {
    ViewExpr::new_source(buf)
        .apply_op(ViewDto::Image(op.clone()))
        .plan()
        .execute()
}

/// Every op this file covers, as it is planned.
fn ops(c: usize) -> Vec<(&'static str, ImageOp)> {
    let mut ops = vec![
        ("blur", ImageOpKind::Blur { sigma: 1.3 }),
        (
            "pad",
            ImageOpKind::Pad {
                top: 2,
                bottom: 1,
                left: 3,
                right: 0,
                value: 7.0,
                mode: PadMode::Reflect,
            },
        ),
        (
            "pad_to_size",
            ImageOpKind::PadToSize {
                height: 40,
                width: 33,
                position: PadPosition::Center,
                value: 5.0,
            },
        ),
    ];
    if c == 1 {
        ops.extend([
            (
                "erode",
                ImageOpKind::Erode {
                    ksize: 3,
                    iterations: 2,
                },
            ),
            (
                "dilate",
                ImageOpKind::Dilate {
                    ksize: 5,
                    iterations: 1,
                },
            ),
            ("morph_gradient", ImageOpKind::MorphGradient { ksize: 3 }),
        ]);
    }
    ops.into_iter()
        .map(|(n, kind)| (n, ImageOp { kind }))
        .collect()
}

#[test]
fn a_dense_rows_op_reads_a_view_as_its_packed_copy() {
    for c in [1, 3] {
        for base in [
            image::<u8>(29, 23, c),
            image::<f32>(29, 23, c),
            image::<u16>(29, 23, c),
        ] {
            for (name, op) in ops(c) {
                assert_eq!(
                    op.memory_effect().input_layout(),
                    InputLayout::DenseRows,
                    "{name}: this file covers the dense-rows ops"
                );
                for (layout, view) in views(&base) {
                    let got = run(view.clone(), &op);
                    let want = run(view.to_contiguous(), &op);
                    assert_eq!(got.shape(), want.shape(), "{name} on {layout}, c={c}");
                    assert_eq!(
                        got.to_blob(),
                        want.to_blob(),
                        "{name} on {layout}, {:?} c={c}",
                        base.dtype()
                    );
                }
            }
        }
    }
}

/// The kernels read the view's own memory: no copy of a crop's rows is made.
#[test]
fn the_planner_hands_a_dense_rows_op_a_crop_where_it_lies() {
    let base = image::<u8>(29, 23, 1);
    let crop = base.slice(&[3, 2, 0], &[26, 20, 1]);
    for (name, op) in ops(1) {
        let plan = ViewExpr::new_source(crop.clone())
            .apply_op(ViewDto::Image(op))
            .plan();
        assert!(
            !format!("{:?}", plan.steps).contains("MaterializeContiguous"),
            "{name}: the planner packs a crop it could read in place: {:?}",
            plan.steps
        );
    }
}
