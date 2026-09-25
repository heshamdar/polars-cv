//! `IdentityRule` consistency guard.
//!
//! An op may only claim it is a removable no-op in a way its *other* contracts
//! corroborate:
//! - a [`WhenShapePreserved`] op is a no-op exactly when its shape
//!   (`OpShape::preserves`) proves every element stays in place — pinned here
//!   per probe, on a known input;
//! - a [`WhenDtypePreserved`] op (the same-dtype `cast`) changes only the dtype,
//!   so it must move no shape (rank + channels preserved).
//!
//! Value-semantics beyond that (a shape-preserving `flip` still flips) are
//! guarded by the differential-equivalence test in the plugin's Python suite
//! (`test_optimize_equivalence.py`).
//!
//! The probe list doubles as a classification fixture: each op is pinned to the
//! rule it must report, so a silent reclassification fails here. Rules never
//! depend on parameter values; `preserves` does.
//!
//! [`WhenDtypePreserved`]: view_buffer::IdentityRule::WhenDtypePreserved
//! [`WhenShapePreserved`]: view_buffer::IdentityRule::WhenShapePreserved

use view_buffer::ops::pad::{PadMode, PadPosition};
use view_buffer::ops::{Dim, OpShape};
use view_buffer::{ComputeOp, DType, IdentityRule, ImageOp, ImageOpKind, ViewDto, ViewOp};

fn pad(top: u32) -> ViewDto {
    ViewDto::Image(ImageOp {
        kind: ImageOpKind::Pad {
            top,
            bottom: 0,
            left: 0,
            right: 0,
            value: 0.0,
            mode: PadMode::Constant,
        },
    })
}

/// One probe per interesting outcome: the op, the rule it must report, and
/// whether it is a no-op on a known `[4, 4, 3]` input. Covers every op that
/// claims a non-`Never` rule, plus a non-identity counterexample for each
/// mechanism (a non-zero pad, an offset crop, a scale, a flip).
fn probes() -> Vec<(ViewDto, IdentityRule, bool)> {
    use IdentityRule::{Never, WhenDtypePreserved, WhenShapePreserved};
    vec![
        (pad(0), WhenShapePreserved, true),
        (pad(1), WhenShapePreserved, false),
        (
            ViewDto::View(ViewOp::Crop {
                start: vec![0, 0, 0],
                end: vec![4, 4, 3],
            }),
            WhenShapePreserved,
            true,
        ),
        // An offset crop can only keep the input's extent by running past the
        // edge, where the engine clamps it to a smaller window: never a no-op.
        (
            ViewDto::View(ViewOp::Crop {
                start: vec![2, 2, 0],
                end: vec![6, 6, usize::MAX],
            }),
            WhenShapePreserved,
            false,
        ),
        (
            ViewDto::View(ViewOp::Reshape(vec![4, 4, 3])),
            WhenShapePreserved,
            true,
        ),
        (
            ViewDto::View(ViewOp::Reshape(vec![16, 3])),
            WhenShapePreserved,
            false,
        ),
        (
            ViewDto::Image(ImageOp {
                kind: ImageOpKind::PadToSize {
                    height: 4,
                    width: 4,
                    position: PadPosition::TopLeft,
                    value: 0.0,
                },
            }),
            WhenShapePreserved,
            true,
        ),
        // A cast copies when the target equals the input (decided by dtype).
        (
            ViewDto::Compute(ComputeOp::Cast(DType::U8)),
            WhenDtypePreserved,
            false,
        ),
        // Representative computing / pixel-moving ops.
        (ViewDto::Compute(ComputeOp::Scale(2.0)), Never, false),
        (ViewDto::View(ViewOp::Flip(vec![0])), Never, false),
    ]
}

#[test]
fn identity_rule_classification_is_pinned() {
    let input = [Dim::Known(4), Dim::Known(4), Dim::Known(3)];
    for (dto, rule, no_op) in probes() {
        assert_eq!(
            dto.identity_rule(),
            rule,
            "identity_rule for {} changed — update the pass and this fixture together",
            dto.name()
        );
        if rule == IdentityRule::WhenShapePreserved {
            assert_eq!(
                dto.as_op().shape().preserves(Some(&input)),
                no_op,
                "{}: preserves() on [4, 4, 3]",
                dto.name()
            );
        }
    }
}

#[test]
fn when_dtype_preserved_moves_no_shape() {
    for (dto, rule, _) in probes() {
        if rule != IdentityRule::WhenDtypePreserved {
            continue;
        }
        assert_eq!(
            dto.as_op().shape(),
            OpShape::Preserve,
            "{}: WhenDtypePreserved must preserve the whole shape",
            dto.name()
        );
    }
}
