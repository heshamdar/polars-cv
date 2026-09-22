//! `IdentityRule` consistency guard.
//!
//! An op may only claim it is a removable no-op in a way its *other* contracts
//! corroborate:
//! - an [`Always`] op must change nothing — dtype, rank, channels and shape all
//!   pass through, for any input;
//! - a [`WhenDtypePreserved`] op (the same-dtype `cast`) changes only the dtype,
//!   so it must move no shape (rank + channels preserved).
//!
//! These are the machine-checkable halves of the identity contract.
//! [`WhenShapePreserved`] is value-semantic — a shape-preserving `flip` still
//! flips — so it is guarded by the differential-equivalence test in the plugin's
//! Python suite (`test_optimize_equivalence.py`), not here.
//!
//! The probe list doubles as a classification fixture: each op is pinned to the
//! rule it must report, so a silent reclassification fails here.
//!
//! [`Always`]: view_buffer::IdentityRule::Always
//! [`WhenDtypePreserved`]: view_buffer::IdentityRule::WhenDtypePreserved
//! [`WhenShapePreserved`]: view_buffer::IdentityRule::WhenShapePreserved

use view_buffer::ops::pad::{PadMode, PadPosition};
use view_buffer::{
    ComputeOp, DType, IdentityRule, ImageOp, ImageOpKind, OutputChannelRule, OutputRankRule,
    ViewDto, ViewOp,
};

/// One probe per interesting `identity_rule` outcome, each pinned to the rule
/// it must report. Covers every op that claims a non-`Never` rule, plus a
/// non-identity counterexample for each mechanism (a non-zero pad, a scale, a
/// flip).
fn probes() -> Vec<(ViewDto, IdentityRule)> {
    vec![
        // Always: zero padding on every side copies the input.
        (
            ViewDto::Image(ImageOp {
                kind: ImageOpKind::Pad {
                    top: 0,
                    bottom: 0,
                    left: 0,
                    right: 0,
                    value: 0.0,
                    mode: PadMode::Constant,
                },
            }),
            IdentityRule::Always {
                deciding_params: &["top", "bottom", "left", "right"],
            },
        ),
        // A non-zero pad is not a no-op.
        (
            ViewDto::Image(ImageOp {
                kind: ImageOpKind::Pad {
                    top: 1,
                    bottom: 0,
                    left: 0,
                    right: 0,
                    value: 0.0,
                    mode: PadMode::Constant,
                },
            }),
            IdentityRule::Never,
        ),
        // WhenShapePreserved: pure views and a size-preserving pad.
        (
            ViewDto::View(ViewOp::Crop {
                start: vec![0, 0, 0],
                end: vec![4, 4, 3],
            }),
            IdentityRule::WhenShapePreserved {
                deciding_params: &["top", "left"],
            },
        ),
        // A crop with a non-zero origin is never a no-op, even when its extent
        // equals the input's: it can only preserve shape by running past the
        // edge, where the engine clamps it to a smaller window.
        (
            ViewDto::View(ViewOp::Crop {
                start: vec![2, 2, 0],
                end: vec![6, 6, usize::MAX],
            }),
            IdentityRule::Never,
        ),
        (
            ViewDto::View(ViewOp::Reshape(vec![4, 4, 3])),
            IdentityRule::WhenShapePreserved {
                deciding_params: &[],
            },
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
            IdentityRule::WhenShapePreserved {
                deciding_params: &[],
            },
        ),
        // WhenDtypePreserved: a cast copies when the target equals the input.
        (
            ViewDto::Compute(ComputeOp::Cast(DType::U8)),
            IdentityRule::WhenDtypePreserved,
        ),
        // Never: representative computing / pixel-moving ops.
        (ViewDto::Compute(ComputeOp::Scale(2.0)), IdentityRule::Never),
        (ViewDto::View(ViewOp::Flip(vec![0])), IdentityRule::Never),
    ]
}

#[test]
fn identity_rule_classification_is_pinned() {
    for (dto, expected) in probes() {
        assert_eq!(
            dto.identity_rule(),
            expected,
            "identity_rule for {} changed — update the pass and this fixture together",
            dto.name()
        );
    }
}

#[test]
fn always_preserves_all_structure() {
    for (dto, rule) in probes() {
        if !matches!(rule, IdentityRule::Always { .. }) {
            continue;
        }
        let op = dto.as_op();
        for dt in [DType::U8, DType::F32, DType::I32] {
            assert_eq!(
                op.output_dtype_rule().resolve(dt),
                dt,
                "{}: Always must preserve dtype (got a promotion for {dt:?})",
                dto.name()
            );
        }
        assert_eq!(
            op.output_rank_rule(),
            OutputRankRule::PreserveRank,
            "{}: Always must preserve rank",
            dto.name()
        );
        assert_eq!(
            op.output_channel_rule(),
            OutputChannelRule::PreserveChannels,
            "{}: Always must preserve channels",
            dto.name()
        );
        let shape = [4usize, 4, 3];
        assert_eq!(
            op.infer_shape(&[&shape[..]]),
            shape.to_vec(),
            "{}: Always must preserve shape",
            dto.name()
        );
    }
}

#[test]
fn when_dtype_preserved_moves_no_shape() {
    for (dto, rule) in probes() {
        if rule != IdentityRule::WhenDtypePreserved {
            continue;
        }
        let op = dto.as_op();
        assert_eq!(
            op.output_rank_rule(),
            OutputRankRule::PreserveRank,
            "{}: WhenDtypePreserved must preserve rank",
            dto.name()
        );
        assert_eq!(
            op.output_channel_rule(),
            OutputChannelRule::PreserveChannels,
            "{}: WhenDtypePreserved must preserve channels",
            dto.name()
        );
    }
}
