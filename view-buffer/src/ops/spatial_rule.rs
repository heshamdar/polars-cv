//! Declarative rule for how an operation's output depends on the *spatial*
//! extent of its input — the plan-time authority for whether a spatial window
//! (a crop / ROI) may commute with an op.
//!
//! This is the spatial counterpart to the rank/channel rules in
//! [`shape_rule`](crate::ops::shape_rule) and the dtype rule in
//! [`OutputDTypeRule`](crate::core::dtype::OutputDTypeRule): a structural fact
//! about an op that the planner can read without any concrete input, declared
//! once on the [`Op`](crate::ops::Op) trait and surfaced over FFI to the Python
//! planner.
//!
//! # The four dependencies (closed, unambiguous)
//!
//! Each op's output at a buffer location `(y, x)` depends on its input in
//! exactly one of these ways:
//!
//! - [`Pointwise`](SpatialDependency::Pointwise) — depends only on the input at
//!   the *same* `(y, x)` (it may read/mix all channels there). Radius 0.
//!   Commutes with any spatial crop/window **exactly**.
//! - [`Neighborhood`](SpatialDependency::Neighborhood) — depends on input within
//!   a bounded radius of `(y, x)`, in the *same* coordinate system (no
//!   resampling). Commutes with a crop only if the crop is dilated by `radius`
//!   (a halo).
//! - [`Global`](SpatialDependency::Global) — depends on a statistic over *all*
//!   input pixels. A hard reorder barrier. Also the **conservative fallback**:
//!   an op whose dependence cannot be reasoned about (including data-dependent
//!   dependence, e.g. connected-component propagation) declares `Global`, which
//!   is always correct because it permits no reorder. There is deliberately no
//!   separate `Unknown` variant — keeping the set closed is what makes the
//!   taxonomy unambiguous.
//! - [`Geometric`](SpatialDependency::Geometric) — changes the coordinate
//!   system / resamples; the input→output pixel mapping is a coordinate
//!   transform. Commuting a crop requires transforming the window through the
//!   op's (inverse) map (and a halo for resampling filters).
//!
//! # No independent authority — how this is kept honest
//!
//! Unlike the rank/channel/dtype rules, a spatial dependency has *no* second
//! source of truth (no `shape` analog) to parity-check against: it is a
//! new primary declaration. Its correctness is therefore pinned by (1) the
//! compiler — [`Op::spatial_dependency`](crate::ops::Op::spatial_dependency) is
//! required with no default, *and every impl matches its enum exhaustively
//! rather than returning a blanket constant*, so a new op — or a new variant of
//! an existing op — cannot silently inherit a classification; it forces *a*
//! declaration, though not a correct one); (2) the expected-value coverage tests
//! in this module, which pin the declared dependency of each op they enumerate;
//! and (3) downstream, the
//! differential equivalence tests of any optimization that consumes it (a pass
//! gated on this rule must produce byte-identical output), which is what turns a
//! *wrong-but-compiling* classification into a visible wrong result rather than
//! a passing plan-time check. The spatial-window pushdown pass is the first such
//! consumer, and its equivalence sweep is (3) for the `Pointwise` arm.
//!
//! # Extensibility
//!
//! The enum is intentionally closed (four variants) so every op must handle it
//! and the compiler enforces completeness. Richer information lives in the
//! variant *payload* structs ([`NeighborhoodSupport`], [`GeometricEffect`]),
//! which can gain fields (separable/anisotropic support, an explicit coordinate
//! remap) without changing the enum — match arms reading `Neighborhood(_)` or
//! `Geometric(_)` keep compiling.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The bounded spatial support of a [`Neighborhood`](SpatialDependency::Neighborhood)
/// op.
///
/// The output at `(y, x)` depends on input pixels within `radius` (Chebyshev
/// distance) of `(y, x)`, in the input's own coordinate system. A crop of the
/// output therefore corresponds to a crop of the input dilated by `radius`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeighborhoodSupport {
    /// Half-extent of the dependency window, in input pixels (a `ksize×ksize`
    /// kernel has `radius = ksize / 2`).
    pub radius: usize,
    // Future enrichment — separable/anisotropic radii — extends this struct,
    // not the `SpatialDependency` enum.
}

impl NeighborhoodSupport {
    /// A symmetric neighborhood of the given half-extent.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
}

/// The coordinate-system effect of a [`Geometric`](SpatialDependency::Geometric)
/// op.
///
/// A marker today: it records only that the op remaps coordinates. A later
/// enrichment (an explicit, invertible coordinate-remap descriptor that a crop
/// window can be transformed through) extends this struct; the enum and every
/// `Geometric(_)` match arm are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GeometricEffect {
    // Intentionally empty for now — see the struct docs.
}

impl GeometricEffect {
    /// The (currently information-free) geometric effect marker.
    pub fn new() -> Self {
        Self::default()
    }
}

/// How an operation's output depends on the spatial extent of its input.
///
/// See the [module docs](self) for the precise, closed definition of each
/// variant and why there is no `Unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SpatialDependency {
    /// Output at `(y, x)` depends only on input at `(y, x)`. Radius 0.
    Pointwise,
    /// Output at `(y, x)` depends on input within a bounded radius of `(y, x)`.
    Neighborhood(NeighborhoodSupport),
    /// Output depends on a statistic over all input pixels; also the
    /// conservative, reorder-blocking fallback.
    Global,
    /// Output is produced by a coordinate transform of the input.
    Geometric(GeometricEffect),
}

impl SpatialDependency {
    /// A symmetric neighborhood dependency of the given radius — the common
    /// constructor for kernel ops (`radius = ksize / 2`).
    pub fn neighborhood(radius: usize) -> Self {
        SpatialDependency::Neighborhood(NeighborhoodSupport::new(radius))
    }

    /// A coordinate-transform dependency.
    pub fn geometric() -> Self {
        SpatialDependency::Geometric(GeometricEffect::new())
    }
}

#[cfg(test)]
mod tests {
    //! Expected-value coverage: pin the declared spatial dependency of each op
    //! enumerated here. There is no `shape`-style authority to
    //! parity-check against (see the module docs), so these hand-written
    //! expectations — together with the compiler's requiredness and the
    //! now-exhaustive matches in every `spatial_dependency` impl (no blanket
    //! arm silently absorbs a new variant) — are the guard for the declarations
    //! themselves. What the compiler still cannot judge is whether a forced
    //! declaration is *correct*; a wrong-but-compiling classification is caught
    //! by the first consumer's differential-equivalence tests (the
    //! spatial-window pushdown pass). The blur radius, the one non-trivial
    //! `Neighborhood` value, is additionally cross-checked against the executed
    //! kernel in `execution::runner`'s `blur_radius_tests`.

    use super::*;
    use crate::ops::binary::BinaryOp;
    use crate::ops::color::{ColorConvertOp, ColorSpace};
    use crate::ops::compute::ComputeOp;
    use crate::ops::filter::{BorderMode, ConvolveOp};
    use crate::ops::histogram::HistogramOp;
    use crate::ops::image::{FilterType, ImageOp, ImageOpKind};
    use crate::ops::pad::{PadMode, PadPosition};
    use crate::ops::phash::{HashAlgorithm, PerceptualHashOp};
    use crate::ops::reduction::ReductionOp;
    use crate::ops::scalar::ScalarOp;
    use crate::ops::traits::Op;
    use crate::ops::view::ViewOp;

    fn img(kind: ImageOpKind) -> ImageOp {
        ImageOp { kind }
    }

    #[test]
    fn pointwise_ops() {
        let pw = SpatialDependency::Pointwise;
        assert_eq!(
            ComputeOp::Cast {
                dtype: crate::core::dtype::DType::F32
            }
            .spatial_dependency(),
            pw
        );
        assert_eq!(ComputeOp::Scale { factor: 2.0 }.spatial_dependency(), pw);
        assert_eq!(ComputeOp::Relu.spatial_dependency(), pw);
        assert_eq!(ComputeOp::Invert.spatial_dependency(), pw);
        assert_eq!(
            ComputeOp::Clamp { min: 0.0, max: 1.0 }.spatial_dependency(),
            pw
        );
        assert_eq!(
            ComputeOp::AdjustGamma { gamma: 2.2 }.spatial_dependency(),
            pw
        );
        assert_eq!(ComputeOp::Scalar(ScalarOp::Relu).spatial_dependency(), pw);
        assert_eq!(
            ColorConvertOp {
                from_space: ColorSpace::Rgb,
                to_space: ColorSpace::Hsv
            }
            .spatial_dependency(),
            pw
        );
        assert_eq!(BinaryOp::Add.spatial_dependency(), pw);
        assert_eq!(BinaryOp::Multiply.spatial_dependency(), pw);
        // Spatially pointwise: picks a channel at the same (y, x).
        assert_eq!(ViewOp::ChannelSelect { index: 0 }.spatial_dependency(), pw);
        // Image-domain pointwise.
        assert_eq!(
            img(ImageOpKind::Threshold { value: 128.0 }).spatial_dependency(),
            pw
        );
        assert_eq!(img(ImageOpKind::Grayscale).spatial_dependency(), pw);
        assert_eq!(
            img(ImageOpKind::ChannelSwap {
                order: vec![2, 1, 0]
            })
            .spatial_dependency(),
            pw
        );
    }

    #[test]
    fn global_ops() {
        let g = SpatialDependency::Global;
        assert_eq!(
            ComputeOp::Normalize {
                method: crate::ops::compute::NormalizeMethod::MinMax,
                mean: None,
                std: None,
                out_dtype: None,
            }
            .spatial_dependency(),
            g
        );
        assert_eq!(
            ComputeOp::AdjustContrast { factor: 1.5 }.spatial_dependency(),
            g
        );
        assert_eq!(ReductionOp::Sum.spatial_dependency(), g);
        assert_eq!(HistogramOp::new(8).spatial_dependency(), g);
        assert_eq!(
            PerceptualHashOp::new(HashAlgorithm::Perceptual).spatial_dependency(),
            g
        );
        // Hysteresis links edges via connectivity that can span the whole
        // image, so Canny is NOT a bounded neighborhood.
        assert_eq!(
            img(ImageOpKind::Canny {
                low_threshold: 50.0,
                high_threshold: 150.0
            })
            .spatial_dependency(),
            g
        );
        assert_eq!(img(ImageOpKind::HistogramEqualize).spatial_dependency(), g);
    }

    #[test]
    fn neighborhood_ops() {
        // Convolve: radius = ksize / 2.
        assert_eq!(
            ConvolveOp {
                kernel: vec![0.0; 9],
                ksize: 3,
                normalize: false,
                border: BorderMode::Replicate
            }
            .spatial_dependency(),
            SpatialDependency::neighborhood(1)
        );
        // Blur: radius = ceil(3σ) (matches gaussian_kernel_1d in the runner).
        assert_eq!(
            img(ImageOpKind::Blur { sigma: 1.0 }).spatial_dependency(),
            SpatialDependency::neighborhood(3)
        );
        assert_eq!(
            img(ImageOpKind::Blur { sigma: 2.0 }).spatial_dependency(),
            SpatialDependency::neighborhood(6)
        );
        // Erode/Dilate: radius = (ksize / 2) * iterations.
        assert_eq!(
            img(ImageOpKind::Erode {
                ksize: 5,
                iterations: 2
            })
            .spatial_dependency(),
            SpatialDependency::neighborhood(4)
        );
        assert_eq!(
            img(ImageOpKind::Dilate {
                ksize: 3,
                iterations: 1
            })
            .spatial_dependency(),
            SpatialDependency::neighborhood(1)
        );
        // MorphGradient = dilate − erode, each one iteration: radius = ksize / 2.
        assert_eq!(
            img(ImageOpKind::MorphGradient { ksize: 7 }).spatial_dependency(),
            SpatialDependency::neighborhood(3)
        );
    }

    #[test]
    fn geometric_ops() {
        let geo = SpatialDependency::geometric();
        assert_eq!(
            ComputeOp::RotateAffine {
                angle_deg: 30.0,
                expand: false,
                interpolation: crate::ops::affine::InterpolationType::Bilinear,
                border_value: 0.0
            }
            .spatial_dependency(),
            geo
        );
        assert_eq!(ViewOp::transpose(&[1, 0, 2]).spatial_dependency(), geo);
        assert_eq!(
            ViewOp::Reshape { shape: vec![48] }.spatial_dependency(),
            geo
        );
        assert_eq!(ViewOp::flip(&[0]).spatial_dependency(), geo);
        assert_eq!(
            ViewOp::Slice {
                start: vec![0, 0, 0],
                end: vec![2, 2, 3]
            }
            .spatial_dependency(),
            geo
        );
        assert_eq!(ViewOp::Rotate90.spatial_dependency(), geo);
        assert_eq!(ViewOp::Rotate180.spatial_dependency(), geo);
        assert_eq!(ViewOp::Rotate270.spatial_dependency(), geo);
        assert_eq!(
            img(ImageOpKind::Resize {
                width: 8,
                height: 8,
                filter: FilterType::Nearest
            })
            .spatial_dependency(),
            geo
        );
        assert_eq!(
            img(ImageOpKind::Pad {
                top: 1,
                bottom: 2,
                left: 3,
                right: 4,
                value: 0.0,
                mode: PadMode::Constant
            })
            .spatial_dependency(),
            geo
        );
        assert_eq!(
            img(ImageOpKind::PadToSize {
                height: 8,
                width: 8,
                position: PadPosition::Center,
                value: 0.0
            })
            .spatial_dependency(),
            geo
        );
        assert_eq!(
            img(ImageOpKind::Letterbox {
                height: 8,
                width: 8,
                value: 0.0,
                filter: FilterType::Nearest
            })
            .spatial_dependency(),
            geo
        );
    }

    #[test]
    fn spatial_window_is_only_an_hw_crop() {
        // The `crop` builder emits `[top, left, 0] .. [_, _, usize::MAX]`: the
        // channel axis is left at full extent, so it is a hoistable H/W window.
        assert!(ViewOp::Slice {
            start: vec![1, 1, 0],
            end: vec![5, 5, usize::MAX],
        }
        .is_spatial_window());

        // A crop that slices the channel axis (start != 0, or a bounded channel
        // end) is not H/W-only and must not be hoistable.
        assert!(!ViewOp::Slice {
            start: vec![0, 0, 1],
            end: vec![5, 5, usize::MAX],
        }
        .is_spatial_window());
        assert!(!ViewOp::Slice {
            start: vec![0, 0, 0],
            end: vec![5, 5, 2],
        }
        .is_spatial_window());

        // Nothing else is a window: geometric neighbours, pointwise ops, reduces.
        assert!(!ViewOp::flip(&[0]).is_spatial_window());
        assert!(!ViewOp::Reshape { shape: vec![48] }.is_spatial_window());
        assert!(!img(ImageOpKind::Grayscale).is_spatial_window());
        assert!(!img(ImageOpKind::Resize {
            width: 8,
            height: 8,
            filter: FilterType::Nearest
        })
        .is_spatial_window());
        assert!(!ReductionOp::Sum.is_spatial_window());
    }
}
