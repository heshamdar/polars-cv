//! `ViewExpr::apply_op` must execute EVERY `ViewDto` variant.
//!
//! `ViewDto` is defined as "exactly what the engine can run": no panic arms,
//! no silent no-ops. This test executes one probe per variant end-to-end
//! (`apply_op` → `plan()` → `execute()`) and checks the result against the
//! variant's own declared contracts (`infer_shape` via the backing `Op`, and
//! `output_dtype_rule`). The exhaustive match in `assert_probed` makes adding
//! a `ViewDto` variant a compile error until a probe exists here.

#![cfg(feature = "image_interop")]

use view_buffer::ops::color::{ColorConvertOp, ColorSpace};
use view_buffer::ops::filter::{BorderMode, ConvolveOp};
use view_buffer::{
    ComputeOp, DType, FilterType, ImageOp, ImageOpKind, ViewBuffer, ViewDto, ViewExpr, ViewOp,
};

/// One probe instance per `ViewDto` variant.
fn view_dto_probes() -> Vec<ViewDto> {
    vec![
        ViewDto::View(ViewOp::Transpose(vec![1, 0, 2])),
        ViewDto::Compute(ComputeOp::Scale(2.0)),
        ViewDto::Image(ImageOp {
            kind: ImageOpKind::Resize {
                width: 2,
                height: 2,
                filter: FilterType::Nearest,
            },
        }),
        ViewDto::Color(ColorConvertOp {
            from: ColorSpace::Rgb,
            to: ColorSpace::Gray,
        }),
        ViewDto::Filter(ConvolveOp {
            kernel: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ksize: 3,
            normalize: false,
            border: BorderMode::Replicate,
        }),
    ]
}

/// Completeness guard: adding a `ViewDto` variant fails to compile here until
/// it is acknowledged — add a matching probe to `view_dto_probes()` too.
fn assert_probed(dto: &ViewDto) {
    match dto {
        ViewDto::View(_)
        | ViewDto::Compute(_)
        | ViewDto::Image(_)
        | ViewDto::Color(_)
        | ViewDto::Filter(_) => (),
    }
}

#[test]
fn apply_op_executes_every_view_dto_variant() {
    for dto in view_dto_probes() {
        assert_probed(&dto);
        let name = dto.name();

        let source = ViewBuffer::from_vec_with_shape(vec![7u8; 4 * 4 * 3], vec![4, 4, 3]);
        let expected_shape = dto.as_op().infer_shape(&[&[4, 4, 3]]);
        let expected_dtype = dto.output_dtype_rule().resolve(DType::U8, None);

        let expr = ViewExpr::new_source(source).apply_op(dto);
        let result = expr.plan().execute();

        assert_eq!(
            result.shape(),
            expected_shape.as_slice(),
            "{name}: executed shape must match the Op contract's infer_shape"
        );
        assert_eq!(
            result.dtype(),
            expected_dtype,
            "{name}: executed dtype must match output_dtype_rule"
        );
    }
}
