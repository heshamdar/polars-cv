//! Declarative rules for how an operation transforms buffer *structure*
//! (rank and channel count).
//!
//! These are the plan-time-inspectable, op-coupled counterparts to
//! [`infer_shape`](crate::ops::Op::infer_shape), in the same spirit as
//! [`OutputDTypeRule`](crate::core::dtype::OutputDTypeRule) is for dtype.
//!
//! A shape transform decomposes into two parts:
//! - **Structural** — how rank changes and what happens to the channel
//!   dimension. This is declarable up front (these rules) and is exactly what
//!   plan-time schema inference needs when concrete dimensions are unknown.
//! - **Geometric** — the actual `H`/`W` values produced (e.g. a resize target).
//!   These depend on operation parameters and stay in `infer_shape`.
//!
//! `infer_shape` remains the concrete authority; these rules declare the
//! structural effect. A parity test (`tests`/`shape_rule_parity`) binds the two
//! so the declaration can never silently diverge from what `infer_shape`
//! actually produces.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// How an operation transforms the *rank* (number of dimensions) of its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OutputRankRule {
    /// Output rank equals input rank.
    PreserveRank,
    /// Output rank is input rank minus one (axis or channel-dimension drop).
    ///
    /// Mirrors `infer_shape`'s clamp behaviour: reducing the rank of a rank-1
    /// input still yields rank 1 (a single scalar slot), never rank 0.
    ReduceByOne,
    /// Output is always exactly this rank, regardless of input.
    Fixed(usize),
    /// Rank is not knowable at plan time from the rule alone (e.g. an op
    /// whose output rank depends on runtime data rather than its parameters).
    Unknown,
}

impl OutputRankRule {
    /// Predict the output rank for a given input rank.
    ///
    /// Returns `None` for [`OutputRankRule::Unknown`], signalling that the
    /// caller must fall back to a concrete shape (or leave the rank unknown).
    pub fn apply(&self, input_rank: usize) -> Option<usize> {
        match self {
            OutputRankRule::PreserveRank => Some(input_rank),
            OutputRankRule::ReduceByOne => Some(input_rank.saturating_sub(1).max(1)),
            OutputRankRule::Fixed(n) => Some(*n),
            OutputRankRule::Unknown => None,
        }
    }
}

/// How an operation transforms the *channel count* — the trailing dimension of
/// an `[H, W, C]` buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OutputChannelRule {
    /// Channel count is unchanged.
    PreserveChannels,
    /// Channel count is always exactly `n` (e.g. `grayscale`/`canny` → 1).
    Fixed(usize),
    /// Color channels become `color_channels`; an input alpha channel (an input
    /// channel count of 2 or 4) is preserved and added back on top.
    ///
    /// This is the alpha-aware "strip, process, restore" behaviour of color
    /// conversions: `RGBA`→gray yields `GrayA` (2ch), `RGB`→gray yields 1ch.
    StripProcessRestore { color_channels: usize },
    /// The operation does not produce an `[H, W, C]` image buffer, so a channel
    /// count is not meaningful (reductions, geometry measures, rank drops).
    NotApplicable,
    /// Channel count is not knowable at plan time from the rule alone
    /// (e.g. `transpose`, which can move the channel axis).
    Unknown,
}

impl OutputChannelRule {
    /// Predict the output channel count given the input channel count.
    ///
    /// `input_channels` is `None` when the input's channel count is unknown at
    /// plan time. Returns `None` when the result is not determinable
    /// ([`NotApplicable`](OutputChannelRule::NotApplicable),
    /// [`Unknown`](OutputChannelRule::Unknown), or an unknown input feeding a
    /// channel-dependent rule).
    pub fn apply(&self, input_channels: Option<usize>) -> Option<usize> {
        match self {
            OutputChannelRule::PreserveChannels => input_channels,
            OutputChannelRule::Fixed(n) => Some(*n),
            OutputChannelRule::StripProcessRestore { color_channels } => input_channels.map(|c| {
                let has_alpha = matches!(c, 2 | 4);
                color_channels + usize::from(has_alpha)
            }),
            OutputChannelRule::NotApplicable | OutputChannelRule::Unknown => None,
        }
    }
}

#[cfg(test)]
mod parity_tests {
    //! Bind the declarative rules to `infer_shape`: for every operation, the
    //! rank/channel a rule *predicts* must equal what `infer_shape` actually
    //! produces. This is the drift guard that makes the rules a faithful, single
    //! authority for plan-time structural inference rather than a parallel copy.

    use crate::geometry::ops::{ApproxMethod, ExtractMode, GeometryOp, ScaleOrigin};
    use crate::ops::binary::BinaryOp;
    use crate::ops::color::{ColorConvertOp, ColorSpace};
    use crate::ops::compute::ComputeOp;
    use crate::ops::filter::{BorderMode, ConvolveOp};
    use crate::ops::histogram::{HistogramOp, HistogramOutput};
    use crate::ops::image::{FilterType, ImageOp, ImageOpKind};
    use crate::ops::phash::{HashAlgorithm, PerceptualHashOp};
    use crate::ops::reduction::ReductionOp;
    use crate::ops::traits::Op;
    use crate::ops::view::ViewOp;

    /// Assert that `op`'s declared rank/channel rules agree with `infer_shape`
    /// on `probe`. `Unknown`/`NotApplicable` rules are intentionally skipped —
    /// they declare "not knowable", so there is nothing to bind.
    fn check(op: &dyn Op, probe: &[usize]) {
        // Duplicate the probe so multi-input ops (binary, pairwise geometry)
        // also have a second input; single-input ops ignore the extra.
        let out = op.infer_shape(&[probe, probe]);

        if let Some(expected_rank) = op.output_rank_rule().apply(probe.len()) {
            assert_eq!(
                expected_rank,
                out.len(),
                "{}: rank rule {:?} predicted rank {} but infer_shape gave {:?}",
                op.name(),
                op.output_rank_rule(),
                expected_rank,
                out,
            );
        }

        // The channel dimension only exists for a rank-3 [H, W, C] buffer.
        if probe.len() == 3 {
            if let Some(expected_c) = op.output_channel_rule().apply(Some(probe[2])) {
                assert_eq!(
                    out.len(),
                    3,
                    "{}: channel rule {:?} implies a rank-3 output but infer_shape gave {:?}",
                    op.name(),
                    op.output_channel_rule(),
                    out,
                );
                assert_eq!(
                    expected_c,
                    out[2],
                    "{}: channel rule {:?} predicted {} channels but infer_shape gave {:?}",
                    op.name(),
                    op.output_channel_rule(),
                    expected_c,
                    out,
                );
            }
        }
    }

    /// One probe instance per `ImageOpKind` variant.
    ///
    /// The exhaustive match in `image_kind_name` makes adding a variant a
    /// compile error until it is acknowledged there, and
    /// `every_image_kind_has_a_probe` reads that match back out of this file
    /// and requires the probes to cover it. Acknowledging a kind and probing
    /// it are one act; when they were two, an arm added without a probe
    /// compiled and passed.
    fn image_kind_probes() -> Vec<ImageOpKind> {
        use FilterType::Nearest;
        vec![
            ImageOpKind::Threshold(128.0),
            ImageOpKind::Resize {
                width: 8,
                height: 8,
                filter: Nearest,
            },
            ImageOpKind::Blur { sigma: 1.0 },
            ImageOpKind::Grayscale,
            ImageOpKind::Canny {
                low_threshold: 50.0,
                high_threshold: 150.0,
            },
            ImageOpKind::HistogramEqualize,
            // Morphological ops preserve channels — the case where the old
            // Python contract (drop→1ch) disagreed with execution.
            ImageOpKind::Erode {
                ksize: 3,
                iterations: 1,
            },
            ImageOpKind::Dilate {
                ksize: 3,
                iterations: 1,
            },
            ImageOpKind::MorphGradient { ksize: 3 },
            ImageOpKind::ResizeScale {
                scale_x: 0.5,
                scale_y: 2.0,
                filter: Nearest,
            },
            ImageOpKind::ResizeToHeight {
                height: 8,
                filter: Nearest,
            },
            ImageOpKind::ResizeToWidth {
                width: 3,
                filter: Nearest,
            },
            ImageOpKind::ResizeMax {
                max_size: 12,
                filter: Nearest,
            },
            ImageOpKind::ResizeMin {
                min_size: 2,
                filter: Nearest,
            },
            ImageOpKind::Pad {
                top: 1,
                bottom: 2,
                left: 3,
                right: 4,
                value: 0.0,
                mode: crate::ops::pad::PadMode::Constant,
            },
            ImageOpKind::PadToSize {
                height: 8,
                width: 8,
                position: crate::ops::pad::PadPosition::Center,
                value: 0.0,
            },
            ImageOpKind::Letterbox {
                height: 8,
                width: 8,
                value: 0.0,
                filter: Nearest,
            },
            ImageOpKind::ChannelSwap {
                order: vec![2, 1, 0],
            },
        ]
    }

    /// The variant a kind belongs to.
    ///
    /// Exhaustive, so adding an `ImageOpKind` fails to compile here. That alone
    /// only forced *this match* to grow: `image_kind_probes()` is a separate
    /// list, and adding an arm without a probe compiled and passed, which is
    /// the gap `every_image_kind_has_a_probe` below closes.
    fn image_kind_name(kind: &ImageOpKind) -> &'static str {
        match kind {
            ImageOpKind::Threshold(_) => "Threshold",
            ImageOpKind::Resize { .. } => "Resize",
            ImageOpKind::Blur { .. } => "Blur",
            ImageOpKind::Grayscale => "Grayscale",
            ImageOpKind::Canny { .. } => "Canny",
            ImageOpKind::HistogramEqualize => "HistogramEqualize",
            ImageOpKind::Erode { .. } => "Erode",
            ImageOpKind::Dilate { .. } => "Dilate",
            ImageOpKind::MorphGradient { .. } => "MorphGradient",
            ImageOpKind::ResizeScale { .. } => "ResizeScale",
            ImageOpKind::ResizeToHeight { .. } => "ResizeToHeight",
            ImageOpKind::ResizeToWidth { .. } => "ResizeToWidth",
            ImageOpKind::ResizeMax { .. } => "ResizeMax",
            ImageOpKind::ResizeMin { .. } => "ResizeMin",
            ImageOpKind::Pad { .. } => "Pad",
            ImageOpKind::PadToSize { .. } => "PadToSize",
            ImageOpKind::Letterbox { .. } => "Letterbox",
            ImageOpKind::ChannelSwap { .. } => "ChannelSwap",
        }
    }

    /// The variants `image_kind_name` acknowledges, read back from this file.
    ///
    /// Rust cannot enumerate an enum's variants without a derive or a second
    /// list, and a second list is what this replaces. The parse asserts it
    /// found a plausible match rather than silently matching nothing.
    fn acknowledged_image_kinds() -> Vec<String> {
        let src = include_str!("shape_rule.rs");
        let body = src
            .split("fn image_kind_name(kind: &ImageOpKind) -> &'static str {")
            .nth(1)
            .expect("image_kind_name's definition moved — this scan reads nothing");
        let body = body
            .split("\n    }")
            .next()
            .expect("image_kind_name's body has no closing brace");
        let names: Vec<String> = body
            .lines()
            .filter_map(|line| line.trim().strip_prefix("ImageOpKind::"))
            .filter_map(|rest| rest.split([' ', '(']).next())
            .map(str::to_string)
            .collect();
        assert!(
            names.len() >= 18,
            "parsed {} arms from image_kind_name; the scan is out of date",
            names.len()
        );
        names
    }

    #[test]
    fn every_image_kind_has_a_probe() {
        let probed: std::collections::BTreeSet<&str> =
            image_kind_probes().iter().map(image_kind_name).collect();
        let missing: Vec<String> = acknowledged_image_kinds()
            .into_iter()
            .filter(|name| !probed.contains(name.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "these ImageOpKind variants are acknowledged but never probed \
             against infer_shape: {missing:?}"
        );
    }

    #[test]
    fn image_ops_match_infer_shape() {
        // A non-square probe so H/W math errors cannot cancel out.
        let probe = [4usize, 6, 3];
        for kind in image_kind_probes() {
            check(&ImageOp { kind }, &probe);
        }
    }

    #[test]
    fn view_ops_match_infer_shape() {
        let probe = [4usize, 4, 3];
        check(&ViewOp::Transpose(vec![2, 1, 0]), &probe);
        check(&ViewOp::Reshape(vec![48]), &probe);
        check(&ViewOp::Flip(vec![0]), &probe);
        check(
            &ViewOp::Crop {
                start: vec![0, 0, 0],
                end: vec![2, 2, 3],
            },
            &probe,
        );
        check(&ViewOp::Rotate90, &probe);
        check(&ViewOp::Rotate180, &probe);
        check(&ViewOp::Rotate270, &probe);
        check(&ViewOp::ChannelSelect { index: 0 }, &probe);
    }

    #[test]
    fn compute_and_filter_ops_match_infer_shape() {
        let probe = [4usize, 4, 3];
        check(&ComputeOp::Scale(2.0), &probe);
        check(&ComputeOp::Relu, &probe);
        check(&ComputeOp::Invert, &probe);
        check(&ComputeOp::Clamp { min: 0.0, max: 1.0 }, &probe);
        check(
            &ConvolveOp {
                kernel: vec![0.0; 9],
                ksize: 3,
                normalize: false,
                border: BorderMode::Replicate,
            },
            &probe,
        );
    }

    #[test]
    fn binary_ops_match_infer_shape() {
        let probe = [4usize, 4, 3];
        check(&BinaryOp::Add, &probe);
        check(&BinaryOp::Multiply, &probe);
    }

    #[test]
    fn reduction_ops_match_infer_shape() {
        let probe = [4usize, 4, 3];
        check(&ReductionOp::Sum { axis: None }, &probe);
        check(&ReductionOp::Sum { axis: Some(0) }, &probe);
        check(&ReductionOp::Mean { axis: Some(2) }, &probe);
        check(&ReductionOp::ArgMax { axis: 1 }, &probe);
        check(&ReductionOp::PopCount, &probe);
        check(&ReductionOp::Percentile { q: 50.0 }, &probe);
    }

    #[test]
    fn histogram_ops_match_infer_shape() {
        let probe = [4usize, 4, 3];
        for output in [
            HistogramOutput::Counts,
            HistogramOutput::Normalized,
            HistogramOutput::Edges,
            HistogramOutput::Buckets,
            HistogramOutput::Quantized,
        ] {
            check(&HistogramOp::new(8).with_output(output), &probe);
        }
    }

    #[test]
    fn phash_matches_infer_shape() {
        let probe = [4usize, 4, 3];
        check(&PerceptualHashOp::new(HashAlgorithm::Perceptual), &probe);
    }

    #[test]
    fn geometry_ops_match_infer_shape() {
        let contour = [10usize, 2];
        check(&GeometryOp::Area { signed: false }, &contour);
        check(&GeometryOp::Perimeter, &contour);
        check(&GeometryOp::Centroid, &contour);
        check(&GeometryOp::BoundingBox, &contour);
        check(&GeometryOp::Translate { dx: 1.0, dy: 2.0 }, &contour);
        check(
            &GeometryOp::Scale {
                sx: 2.0,
                sy: 2.0,
                origin: ScaleOrigin::Centroid,
            },
            &contour,
        );
        check(&GeometryOp::Simplify { tolerance: 0.5 }, &contour);
        check(&GeometryOp::ConvexHull, &contour);
        check(
            &GeometryOp::ExtractContours {
                mode: ExtractMode::External,
                method: ApproxMethod::Simple,
                min_area: None,
            },
            &contour,
        );
        // Rasterize emits [H, W, 1]; probe rank-3 so the channel rule is bound too.
        check(
            &GeometryOp::Rasterize {
                width: 8,
                height: 8,
                fill_value: 255,
                background: 0,
            },
            &[4usize, 4, 3],
        );
    }

    #[test]
    fn color_convert_matches_infer_shape() {
        // ColorConvertOp has an inherent (single-input) infer_shape, so check it
        // directly against its declared rules.
        let cases = [
            (ColorSpace::Rgb, ColorSpace::Gray),
            (ColorSpace::Rgb, ColorSpace::Hsv),
            (ColorSpace::Bgr, ColorSpace::Rgb),
        ];
        for (from, to) in cases {
            let op = ColorConvertOp { from, to };
            for probe in [vec![4usize, 4, 3], vec![4, 4, 4]] {
                let out = op.infer_shape(&probe);
                assert_eq!(op.output_rank_rule().apply(probe.len()), Some(out.len()));
                if let Some(expected_c) = op.output_channel_rule().apply(Some(probe[2])) {
                    assert_eq!(expected_c, out[2], "{from:?}->{to:?} channel mismatch");
                }
            }
        }
    }
}
