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

/// The variant a value belongs to.
///
/// Exhaustive, so adding a `ViewDto` variant fails to compile here. That alone
/// only forced the *match* to grow, though — `view_dto_probes()` was a separate
/// list, and adding an arm without a probe compiled and passed. The test below
/// closes that by reading this function's own arms back out of the source and
/// requiring the probes to cover every one.
fn variant_name(dto: &ViewDto) -> &'static str {
    match dto {
        ViewDto::View(_) => "View",
        ViewDto::Compute(_) => "Compute",
        ViewDto::Image(_) => "Image",
        ViewDto::Color(_) => "Color",
        ViewDto::Filter(_) => "Filter",
    }
}

/// The variant names `variant_name` acknowledges, parsed from this file.
///
/// Source-scanning is the weaker technique and is used here for exactly the
/// part the type system cannot answer: Rust offers no way to enumerate an
/// enum's variants without a derive or a second list, and a second list is what
/// this is replacing. The parse asserts it found a plausible match rather than
/// silently matching nothing — the failure mode a scan has to be protected
/// from. Same shape as `resolve_op_arms_are_all_known_ops` in the plugin crate.
fn acknowledged_variants() -> Vec<String> {
    let src = include_str!("apply_op_coverage.rs");
    let body = src
        .split("fn variant_name(dto: &ViewDto) -> &'static str {")
        .nth(1)
        .expect("variant_name's definition moved — this scan reads nothing");
    let body = body
        .split("\n}")
        .next()
        .expect("variant_name's body has no closing brace");
    let names: Vec<String> = body
        .lines()
        .filter_map(|line| line.trim().strip_prefix("ViewDto::"))
        .filter_map(|rest| rest.split('(').next())
        .map(str::to_string)
        .collect();
    assert!(
        names.len() >= 5,
        "parsed {} variant arms from variant_name; the scan is out of date",
        names.len()
    );
    names
}

#[test]
fn every_view_dto_variant_has_a_probe() {
    let probed: std::collections::BTreeSet<&str> =
        view_dto_probes().iter().map(variant_name).collect();
    let acknowledged: std::collections::BTreeSet<String> =
        acknowledged_variants().into_iter().collect();
    let missing: Vec<&String> = acknowledged
        .iter()
        .filter(|name| !probed.contains(name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "these ViewDto variants are acknowledged but never probed: {missing:?}. \
         Acknowledging a variant and executing it must be the same act."
    );
}

#[test]
fn apply_op_executes_every_view_dto_variant() {
    for dto in view_dto_probes() {
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
