//! [`OpShape`]: how an operation's output shape follows from its inputs — the
//! one authority for shape arithmetic, rank and channel count alike.
//!
//! Execution evaluates it on known sizes ([`OpShape::concrete`]); the planner
//! symbolically ([`OpShape::dims`], [`Dim`], [`Sym`]), with a per-row
//! parameter as [`Sym::PerRow`]. The output rank is the length of that shape
//! ([`OpShape::rank`]) and the channel count its axis 2, so neither is declared
//! a second time beside it.

/// One dimension of a shape at plan time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    /// A size known before execution.
    Known(usize),
    /// The unknown size of input axis `k`, carried through unchanged: an output
    /// axis reading `Input(k)` is provably that input axis's size, which is
    /// what lets the planner prove an op preserves an unknown shape.
    Input(usize),
    /// Not knowable before execution: it rests on data, a per-row parameter, or
    /// arithmetic on an unknown size.
    Unknown,
}

impl Dim {
    /// The size, when known.
    pub fn known(self) -> Option<usize> {
        match self {
            Dim::Known(n) => Some(n),
            Dim::Input(_) | Dim::Unknown => None,
        }
    }

    /// `f` of a known size; anything else is unknown.
    fn map(self, f: impl FnOnce(usize) -> usize) -> Dim {
        self.known().map_or(Dim::Unknown, |n| Dim::Known(f(n)))
    }

    /// This size plus `amount`. Adding a known zero keeps an `Input` symbol, so
    /// a zero pad is provably shape-preserving over an unknown shape.
    fn plus(self, amount: Sym<usize>) -> Dim {
        match amount {
            Sym::Known(0) => self,
            Sym::Known(a) => self.map(|n| n + a),
            Sym::PerRow => Dim::Unknown,
        }
    }
}

/// A shape-determining parameter at plan time: its value, or `PerRow` when a
/// per-row expression supplies it (known only once a row executes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sym<T> {
    Known(T),
    PerRow,
}

impl<T> Sym<T> {
    /// The value transformed, or still per-row.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Sym<U> {
        match self {
            Sym::Known(v) => Sym::Known(f(v)),
            Sym::PerRow => Sym::PerRow,
        }
    }

    /// The value, when known.
    pub fn known(self) -> Option<T> {
        match self {
            Sym::Known(v) => Some(v),
            Sym::PerRow => None,
        }
    }
}

impl Sym<usize> {
    fn dim(self) -> Dim {
        self.known().map_or(Dim::Unknown, Dim::Known)
    }
}

/// How an operation's output shape follows from its input shapes — the one
/// authority for shape arithmetic. Execution evaluates it on known sizes
/// ([`OpShape::concrete`]); the planner on symbolic ones ([`OpShape::dims`]),
/// with a per-row parameter as [`Sym::PerRow`] rather than a placeholder value.
///
/// "H/W" are dimensions 0 and 1; the geometric variants leave a rank below 2
/// unchanged and carry every axis past the second through.
#[derive(Debug, Clone, PartialEq)]
pub enum OpShape {
    /// The input shape, unchanged.
    Preserve,
    /// Exactly this shape, whatever the input (a hash, a histogram, a canvas,
    /// a reshape target).
    Fixed(Vec<Sym<usize>>),
    /// Not knowable before execution (a data-dependent length).
    Dynamic,
    /// `[H, W, C]` → `[H, W, 1]`; `[H, W]` unchanged. No other rank has an
    /// output (the ops refuse it).
    SingleChannel,
    /// A colour conversion to `channels` colour channels, plus one when the
    /// input carries alpha (2 or 4 channels). A rank-2 input gains a channel
    /// axis unless the target is gray. No rank but 2 or 3 has an output.
    ColorChannels { channels: usize, to_gray: bool },
    /// `[H, W, C]` → `[H, W]`; `[H, W]` unchanged. No other rank has an
    /// output.
    DropChannelAxis,
    /// H and W swap.
    SwapHw,
    /// H and W swap on some rows and not others (a per-row lattice rotation):
    /// known only for a square input.
    MaybeSwapHw,
    /// The bounding box of the input rotated by `angle` degrees.
    RotateExpand(Sym<f32>),
    /// H/W set to `(h, w)`.
    SetHw { h: Sym<usize>, w: Sym<usize> },
    /// H/W scaled by `(sy, sx)`, rounded.
    ScaleHw { sy: Sym<f32>, sx: Sym<f32> },
    /// H set to `h`, W following the aspect ratio.
    HeightTo(Sym<usize>),
    /// W set to `w`, H following the aspect ratio.
    WidthTo(Sym<usize>),
    /// The long side scaled to `n`, the other following the aspect ratio.
    LongSideTo(Sym<usize>),
    /// The short side scaled to `n`, the other following the aspect ratio.
    ShortSideTo(Sym<usize>),
    /// H grown by `top + bottom`, W by `left + right`.
    Pad {
        top: Sym<usize>,
        bottom: Sym<usize>,
        left: Sym<usize>,
        right: Sym<usize>,
    },
    /// H/W grown to at least `(h, w)`.
    AtLeastHw { h: Sym<usize>, w: Sym<usize> },
    /// Axis `i` of the output is input axis `perm[i]`.
    Transpose(Vec<usize>),
    /// Axis `i` starts at `start[i]` and keeps `len[i]` elements, or runs to
    /// the end of the axis when `len[i]` is `None`; later axes are kept whole.
    Crop {
        start: Vec<Sym<usize>>,
        len: Vec<Option<Sym<usize>>>,
    },
    /// Axis `axis` removed (a global reduction when `None`); never below rank 1.
    Reduce { axis: Option<usize> },
    /// The two inputs broadcast together.
    Broadcast,
    /// `[H, W]` inputs stacked along a new channel axis: `[H, W, n]`. No
    /// other rank has an output.
    StackChannels(usize),
    /// The input's rank as a 1-D vector: `[rank]` (reading the dimensions).
    InputRank,
}

impl OpShape {
    /// The output shape for `inputs`; `None` when it is not knowable before
    /// execution, or when there is no output — an input of a rank the shape
    /// takes none of (a channel op over a rank-4 buffer). Total: never a
    /// panic. A rank with no output is never stood in for, so a planner
    /// reasoning over every rank an input may have skips it.
    pub fn dims(&self, inputs: &[&[Dim]]) -> Option<Vec<Dim>> {
        let input: &[Dim] = inputs.first().copied().unwrap_or(&[]);
        let hw = |h: Dim, w: Dim| {
            let mut out = input.to_vec();
            if out.len() >= 2 {
                out[0] = h;
                out[1] = w;
            }
            out
        };
        let (in_h, in_w) = match input {
            [h, w, ..] => (*h, *w),
            _ => (Dim::Unknown, Dim::Unknown),
        };
        // Both sizes of an aspect-ratio computation, when known.
        let known_hw = in_h.known().zip(in_w.known());
        let aspect = |f: &dyn Fn(f32, f32) -> (f32, f32)| match known_hw {
            Some((h, w)) => {
                let (oh, ow) = f(h as f32, w as f32);
                hw(
                    Dim::Known(oh.round() as usize),
                    Dim::Known(ow.round() as usize),
                )
            }
            None => hw(Dim::Unknown, Dim::Unknown),
        };
        Some(match self {
            OpShape::Preserve => input.to_vec(),
            OpShape::Fixed(shape) => shape.iter().map(|s| s.dim()).collect(),
            OpShape::Dynamic => return None,
            OpShape::SingleChannel => match input {
                [h, w, _] => vec![*h, *w, Dim::Known(1)],
                [_, _] => input.to_vec(),
                _ => return None,
            },
            OpShape::ColorChannels { channels, to_gray } => match input {
                [_, _] if *to_gray => input.to_vec(),
                [h, w] => vec![*h, *w, Dim::Known(*channels)],
                [h, w, c] => {
                    let alpha = |c: usize| usize::from(matches!(c, 2 | 4));
                    vec![*h, *w, c.map(|c| channels + alpha(c))]
                }
                _ => return None,
            },
            OpShape::DropChannelAxis => match input {
                [h, w, _] => vec![*h, *w],
                [_, _] => input.to_vec(),
                _ => return None,
            },
            OpShape::SwapHw => hw(in_w, in_h),
            OpShape::MaybeSwapHw => match known_hw {
                Some((h, w)) if h == w => input.to_vec(),
                _ => hw(Dim::Unknown, Dim::Unknown),
            },
            OpShape::RotateExpand(angle) => match (known_hw, angle.known()) {
                (Some((h, w)), Some(angle)) => {
                    let rad = (angle as f64) * std::f64::consts::PI / 180.0;
                    let (cos, sin) = (rad.cos().abs(), rad.sin().abs());
                    let (h, w) = (h as f64, w as f64);
                    hw(
                        Dim::Known((h * cos + w * sin).round() as usize),
                        Dim::Known((w * cos + h * sin).round() as usize),
                    )
                }
                _ => hw(Dim::Unknown, Dim::Unknown),
            },
            OpShape::SetHw { h, w } => hw(h.dim(), w.dim()),
            OpShape::ScaleHw { sy, sx } => {
                let by = |d: Dim, s: Sym<f32>| match s {
                    Sym::Known(s) => d.map(|n| (n as f32 * s).round() as usize),
                    Sym::PerRow => Dim::Unknown,
                };
                hw(by(in_h, *sy), by(in_w, *sx))
            }
            OpShape::HeightTo(h) => match h.known() {
                Some(t) => aspect(&|ih, iw| (t as f32, t as f32 * (iw / ih))),
                None => hw(Dim::Unknown, Dim::Unknown),
            }
            .into_iter()
            .enumerate()
            .map(|(i, d)| if i == 0 { h.dim() } else { d })
            .collect(),
            OpShape::WidthTo(w) => match w.known() {
                Some(t) => aspect(&|ih, iw| (t as f32 * (ih / iw), t as f32)),
                None => hw(Dim::Unknown, Dim::Unknown),
            }
            .into_iter()
            .enumerate()
            .map(|(i, d)| if i == 1 { w.dim() } else { d })
            .collect(),
            OpShape::LongSideTo(n) | OpShape::ShortSideTo(n) => match n.known() {
                Some(n) => {
                    let long = matches!(self, OpShape::LongSideTo(_));
                    aspect(&|ih, iw| {
                        let side = if long { ih.max(iw) } else { ih.min(iw) };
                        let scale = n as f32 / side;
                        (ih * scale, iw * scale)
                    })
                }
                None => hw(Dim::Unknown, Dim::Unknown),
            },
            OpShape::Pad {
                top,
                bottom,
                left,
                right,
            } => hw(in_h.plus(*top).plus(*bottom), in_w.plus(*left).plus(*right)),
            OpShape::AtLeastHw { h, w } => {
                let at_least = |d: Dim, t: Sym<usize>| match (d.known(), t.known()) {
                    (Some(n), Some(t)) => Dim::Known(n.max(t)),
                    _ => Dim::Unknown,
                };
                hw(at_least(in_h, *h), at_least(in_w, *w))
            }
            OpShape::Transpose(perm) => perm
                .iter()
                .map(|&axis| input.get(axis).copied().unwrap_or(Dim::Unknown))
                .collect(),
            OpShape::Crop { start, len } => input
                .iter()
                .enumerate()
                .map(|(axis, &d)| {
                    let from = start.get(axis).copied().unwrap_or(Sym::Known(0));
                    match len.get(axis).copied().flatten() {
                        Some(len) => len.dim(),
                        None => match from {
                            Sym::Known(0) => d,
                            Sym::Known(s) => d.map(|n| n.saturating_sub(s)),
                            Sym::PerRow => Dim::Unknown,
                        },
                    }
                })
                .collect(),
            OpShape::Reduce { axis: None } => vec![Dim::Known(1)],
            OpShape::Reduce { axis: Some(axis) } => {
                let mut out = input.to_vec();
                if *axis < out.len() {
                    out.remove(*axis);
                }
                if out.is_empty() {
                    out.push(Dim::Known(1));
                }
                out
            }
            OpShape::StackChannels(n) => match input {
                [h, w] => vec![*h, *w, Dim::Known(*n)],
                _ => return None,
            },
            OpShape::InputRank => vec![Dim::Known(input.len())],
            OpShape::Broadcast => match inputs {
                [a, b] => {
                    let known = |s: &[Dim]| s.iter().map(|d| d.known()).collect::<Option<Vec<_>>>();
                    match known(a).zip(known(b)) {
                        Some((a, b)) => crate::ops::binary::broadcast_shapes(&a, &b)
                            .unwrap_or(a)
                            .into_iter()
                            .map(Dim::Known)
                            .collect(),
                        None => vec![Dim::Unknown; a.len().max(b.len())],
                    }
                }
                _ => input.to_vec(),
            },
        })
    }

    /// The output rank over input ranks (`None` for an input of unknown
    /// rank): the length of [`dims`](Self::dims) over symbolic inputs of those
    /// ranks, so it is not a second declaration. Over an input of unknown
    /// rank only a shape whose length ignores its input has one.
    pub fn rank(&self, inputs: &[Option<usize>]) -> Option<usize> {
        let symbolic: Option<Vec<Vec<Dim>>> = inputs
            .iter()
            .map(|rank| rank.map(|n| (0..n).map(Dim::Input).collect()))
            .collect();
        match symbolic {
            Some(shapes) => {
                let refs: Vec<&[Dim]> = shapes.iter().map(Vec::as_slice).collect();
                self.dims(&refs).map(|out| out.len())
            }
            None => match self {
                OpShape::Fixed(shape) => Some(shape.len()),
                OpShape::Transpose(perm) => Some(perm.len()),
                OpShape::Reduce { axis: None } => Some(1),
                OpShape::StackChannels(_) => Some(3),
                OpShape::InputRank => Some(1),
                _ => None,
            },
        }
    }

    /// The highest input rank any variant's [`dims`](Self::dims) names in a
    /// pattern (`[H, W, C]`): past it every variant carries an extra axis
    /// through, drops it by position, or has no output, so evaluating up to
    /// it covers every rank. Pinned by `a_shape_is_rank_stable_past_its_patterns`.
    pub const DISTINGUISHED_RANK: usize = 3;

    /// The output sizes over an input of **unknown rank** whose leading axes'
    /// sizes are `leading` (`None` unknown): an axis's size is known only
    /// where every rank the input may have (and the shape has an output for)
    /// agrees on it. Evaluated over ranks
    /// `1..=max(leading.len(), DISTINGUISHED_RANK)`, which is every rank the
    /// answer can differ at. Trailing unknowns are trimmed.
    pub fn dims_over_unknown_rank(&self, leading: &[Option<usize>]) -> Vec<Option<usize>> {
        let top = leading.len().max(Self::DISTINGUISHED_RANK);
        let outs: Vec<Vec<Dim>> = (1..=top)
            .filter_map(|rank| {
                let input: Vec<Dim> = (0..rank)
                    .map(|axis| match leading.get(axis).copied().flatten() {
                        Some(size) => Dim::Known(size),
                        None => Dim::Input(axis),
                    })
                    .collect();
                self.dims(&[&input])
            })
            .collect();
        let width = outs.iter().map(Vec::len).max().unwrap_or(0);
        let mut sizes: Vec<Option<usize>> = (0..width)
            .map(|axis| {
                let mut claims = outs.iter().filter_map(|out| out.get(axis).copied());
                let first = claims.next()?.known()?;
                claims.all(|d| d.known() == Some(first)).then_some(first)
            })
            .collect();
        while sizes.last() == Some(&None) {
            sizes.pop();
        }
        sizes
    }

    /// The output shape for known input shapes: [`dims`](Self::dims) on known
    /// sizes, which yields known sizes. Empty when not knowable before
    /// execution (a data-dependent length).
    pub fn concrete(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let known: Vec<Vec<Dim>> = inputs
            .iter()
            .map(|s| s.iter().map(|&n| Dim::Known(n)).collect())
            .collect();
        let refs: Vec<&[Dim]> = known.iter().map(Vec::as_slice).collect();
        self.dims(&refs)
            .unwrap_or_default()
            .into_iter()
            .map(|d| {
                d.known()
                    .expect("an op's shape over known sizes and known parameters is known")
            })
            .collect()
    }

    /// Whether the op provably hands every element through where it was —
    /// the output is the input's shape *and* nothing was offset — over
    /// `input`, or over any input at all when the rank is unknown (`None`).
    ///
    /// A crop away from a known-zero origin never preserves: it can keep the
    /// input's shape only by running past the edge. Any per-row parameter
    /// ([`Sym::PerRow`]) leaves its axis unknown, so it never proves anything.
    pub fn preserves(&self, input: Option<&[Dim]>) -> bool {
        let zero = |s: &Sym<usize>| *s == Sym::Known(0);
        let for_any_input = match self {
            OpShape::Preserve => true,
            OpShape::Pad {
                top,
                bottom,
                left,
                right,
            } => [top, bottom, left, right].into_iter().all(zero),
            OpShape::Crop { start, len } => {
                if !start.iter().all(zero) {
                    return false;
                }
                len.iter().all(Option::is_none)
            }
            _ => false,
        };
        for_any_input || input.is_some_and(|input| self.dims(&[input]).as_deref() == Some(input))
    }
}

#[cfg(test)]
mod symbolic_tests {
    //! `OpShape::dims` over symbolic sizes: what the planner reads instead of
    //! probing the op with placeholder values.

    use super::{Dim, OpShape, Sym};
    use Dim::{Input, Known, Unknown};

    fn dims(shape: OpShape, input: &[Dim]) -> Vec<Dim> {
        shape.dims(&[input]).expect("inferable")
    }

    const IMAGE: [Dim; 3] = [Input(0), Input(1), Known(3)];

    #[test]
    fn a_per_row_parameter_leaves_only_its_own_axis_unknown() {
        let resize = |h| OpShape::SetHw {
            h,
            w: Sym::Known(100),
        };
        assert_eq!(
            dims(resize(Sym::Known(224)), &IMAGE),
            [Known(224), Known(100), Known(3)]
        );
        assert_eq!(
            dims(resize(Sym::PerRow), &IMAGE),
            [Unknown, Known(100), Known(3)]
        );
        // An aspect-ratio resize over an unknown input knows only its target.
        assert_eq!(
            dims(OpShape::HeightTo(Sym::Known(8)), &IMAGE),
            [Known(8), Unknown, Known(3)]
        );
    }

    #[test]
    fn an_unknown_input_axis_is_carried_as_itself() {
        assert_eq!(dims(OpShape::Preserve, &IMAGE), IMAGE);
        assert_eq!(
            dims(OpShape::SwapHw, &IMAGE),
            [Input(1), Input(0), Known(3)]
        );
        assert_eq!(
            dims(OpShape::Transpose(vec![2, 0, 1]), &IMAGE),
            [Known(3), Input(0), Input(1)]
        );
        let zero = Sym::Known(0);
        let pad = |top| OpShape::Pad {
            top,
            bottom: zero,
            left: zero,
            right: zero,
        };
        // A zero pad provably preserves an unknown shape; a real one does not.
        assert_eq!(dims(pad(zero), &IMAGE), IMAGE);
        assert!(pad(zero).preserves(None));
        assert_eq!(dims(pad(Sym::Known(2)), &IMAGE)[0], Unknown);
        assert_eq!(
            dims(pad(Sym::Known(2)), &[Known(10), Known(10)]),
            [Known(12), Known(10)]
        );
        assert!(!pad(Sym::PerRow).preserves(Some(&IMAGE)));
    }

    #[test]
    fn a_crop_to_the_end_from_the_origin_is_the_input() {
        let crop = |top, height| OpShape::Crop {
            start: vec![top, Sym::Known(0)],
            len: vec![height, None],
        };
        assert_eq!(dims(crop(Sym::Known(0), None), &IMAGE), IMAGE);
        assert!(crop(Sym::Known(0), None).preserves(None));
        // A full-extent window at a known-zero origin preserves a known shape;
        // at any other origin it would run past the edge, so never.
        let full = [Known(4), Known(6)];
        assert!(crop(Sym::Known(0), Some(Sym::Known(4))).preserves(Some(&full)));
        assert!(!crop(Sym::Known(2), Some(Sym::Known(4))).preserves(Some(&full)));
        assert!(!crop(Sym::PerRow, Some(Sym::Known(4))).preserves(Some(&full)));
        assert_eq!(
            dims(crop(Sym::Known(2), None), &[Known(10), Known(6)]),
            [Known(8), Known(6)]
        );
        assert_eq!(dims(crop(Sym::PerRow, None), &IMAGE)[0], Unknown);
        assert_eq!(
            dims(crop(Sym::PerRow, Some(Sym::Known(4))), &IMAGE)[0],
            Known(4)
        );
    }

    #[test]
    fn a_per_row_rotation_is_known_only_for_a_square() {
        assert_eq!(
            dims(OpShape::MaybeSwapHw, &[Known(5), Known(5), Known(3)]),
            [Known(5), Known(5), Known(3)]
        );
        assert_eq!(
            dims(OpShape::MaybeSwapHw, &[Known(5), Known(4), Known(3)]),
            [Unknown, Unknown, Known(3)]
        );
        assert_eq!(
            dims(OpShape::RotateExpand(Sym::PerRow), &[Known(5), Known(4)]),
            [Unknown, Unknown]
        );
        assert_eq!(
            dims(
                OpShape::RotateExpand(Sym::Known(90.0)),
                &[Known(5), Known(4)]
            ),
            [Known(4), Known(5)]
        );
    }

    #[test]
    fn a_data_dependent_shape_is_not_inferable() {
        assert_eq!(OpShape::Dynamic.dims(&[&IMAGE]), None);
        assert!(OpShape::Dynamic.concrete(&[&[4, 4]]).is_empty());
    }

    #[test]
    fn the_rank_is_the_length_of_the_shape() {
        let k = |n| Sym::Known(n);
        let cases: Vec<(OpShape, Option<usize>, Option<usize>)> = vec![
            (OpShape::Preserve, Some(3), Some(3)),
            (OpShape::SingleChannel, Some(3), Some(3)),
            (OpShape::DropChannelAxis, Some(3), Some(2)),
            (
                OpShape::ColorChannels {
                    channels: 3,
                    to_gray: false,
                },
                Some(2),
                Some(3),
            ),
            (OpShape::Reduce { axis: Some(0) }, Some(3), Some(2)),
            (OpShape::Reduce { axis: Some(0) }, Some(1), Some(1)),
            (OpShape::Reduce { axis: None }, Some(3), Some(1)),
            (OpShape::Fixed(vec![k(4), k(2)]), Some(3), Some(2)),
            (OpShape::Transpose(vec![1, 0, 2]), Some(3), Some(3)),
            (OpShape::StackChannels(3), Some(2), Some(3)),
            (OpShape::InputRank, Some(3), Some(1)),
            (OpShape::Dynamic, Some(3), None),
            // An unknown input rank leaves every input-following shape unknown.
            (OpShape::Preserve, None, None),
            (OpShape::Reduce { axis: Some(0) }, None, None),
        ];
        for (shape, input, expected) in cases {
            assert_eq!(shape.rank(&[input]), expected, "{shape:?} over {input:?}");
        }
    }

    /// One sample of every variant (the match makes a new variant a compile
    /// error here until it has one).
    fn every_variant() -> Vec<OpShape> {
        let k = |n| Sym::Known(n);
        let samples = vec![
            OpShape::Preserve,
            OpShape::Fixed(vec![k(4), Sym::PerRow]),
            OpShape::Dynamic,
            OpShape::SingleChannel,
            OpShape::ColorChannels {
                channels: 3,
                to_gray: false,
            },
            OpShape::ColorChannels {
                channels: 1,
                to_gray: true,
            },
            OpShape::DropChannelAxis,
            OpShape::SwapHw,
            OpShape::MaybeSwapHw,
            OpShape::RotateExpand(Sym::Known(30.0)),
            OpShape::SetHw { h: k(5), w: k(6) },
            OpShape::ScaleHw {
                sy: Sym::Known(2.0),
                sx: Sym::Known(0.5),
            },
            OpShape::HeightTo(k(8)),
            OpShape::WidthTo(k(8)),
            OpShape::LongSideTo(k(8)),
            OpShape::ShortSideTo(k(8)),
            OpShape::Pad {
                top: k(1),
                bottom: k(2),
                left: k(3),
                right: k(0),
            },
            OpShape::AtLeastHw { h: k(9), w: k(9) },
            OpShape::Transpose(vec![1, 0, 2]),
            OpShape::Crop {
                start: vec![k(1), k(0)],
                len: vec![Some(k(2)), None],
            },
            OpShape::Reduce { axis: None },
            OpShape::Reduce { axis: Some(0) },
            OpShape::Reduce { axis: Some(3) },
            OpShape::Broadcast,
            OpShape::StackChannels(2),
            OpShape::InputRank,
        ];
        for shape in &samples {
            match shape {
                OpShape::Preserve
                | OpShape::Fixed(_)
                | OpShape::Dynamic
                | OpShape::SingleChannel
                | OpShape::ColorChannels { .. }
                | OpShape::DropChannelAxis
                | OpShape::SwapHw
                | OpShape::MaybeSwapHw
                | OpShape::RotateExpand(_)
                | OpShape::SetHw { .. }
                | OpShape::ScaleHw { .. }
                | OpShape::HeightTo(_)
                | OpShape::WidthTo(_)
                | OpShape::LongSideTo(_)
                | OpShape::ShortSideTo(_)
                | OpShape::Pad { .. }
                | OpShape::AtLeastHw { .. }
                | OpShape::Transpose(_)
                | OpShape::Crop { .. }
                | OpShape::Reduce { .. }
                | OpShape::Broadcast
                | OpShape::StackChannels(_)
                | OpShape::InputRank => {}
            }
        }
        samples
    }

    /// `dims_over_unknown_rank` evaluates ranks only up to
    /// `DISTINGUISHED_RANK` (or the leading sizes' length); that is sound only
    /// if no variant tells a higher rank apart. Every size it claims must hold
    /// at every higher rank too, or the axis be absent there.
    #[test]
    fn a_shape_is_rank_stable_past_its_patterns() {
        for leading in [vec![], vec![Some(7)], vec![Some(7), Some(9), Some(3)]] {
            let top = leading.len().max(OpShape::DISTINGUISHED_RANK);
            for shape in every_variant() {
                let claimed = shape.dims_over_unknown_rank(&leading);
                for rank in top + 1..=top + 3 {
                    let input: Vec<Dim> = (0..rank)
                        .map(|axis| match leading.get(axis).copied().flatten() {
                            Some(size) => Known(size),
                            None => Input(axis),
                        })
                        .collect();
                    // No output at this rank: no claim can be false there.
                    let Some(out) = shape.dims(&[&input]) else {
                        continue;
                    };
                    for (axis, size) in claimed.iter().enumerate() {
                        let Some(size) = size else { continue };
                        assert!(
                            out.get(axis).is_none_or(|d| d.known() == Some(*size)),
                            "{shape:?} (leading {leading:?}): claims axis {axis} is {size}, \
                             but over rank {rank} it is {:?}",
                            out.get(axis)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn an_unknown_rank_knows_only_what_every_rank_agrees_on() {
        // A resize's shape sets H and W from rank 2 up and leaves a rank-1
        // input as it is, so only W (which a rank-1 input lacks) is known. A
        // grayscale has an output only at ranks 2 and 3, and keeps a known H.
        assert_eq!(
            OpShape::SetHw {
                h: Sym::Known(4),
                w: Sym::Known(5)
            }
            .dims_over_unknown_rank(&[]),
            [None, Some(5)]
        );
        assert_eq!(
            OpShape::SingleChannel.dims_over_unknown_rank(&[Some(8)]),
            [Some(8), None, Some(1)]
        );
        // A declared size past the three the keywords name is carried too.
        assert_eq!(
            OpShape::Preserve.dims_over_unknown_rank(&[None, None, None, None, Some(6)]),
            [None, None, None, None, Some(6)]
        );
    }

    /// The shapes `rank` answers for over an unknown input rank really do
    /// ignore the input: the same length over every rank that has an output.
    #[test]
    fn a_rank_known_without_the_input_holds_over_every_input() {
        let shapes = [
            OpShape::Fixed(vec![Sym::Known(4), Sym::PerRow]),
            OpShape::Transpose(vec![2, 0, 1]),
            OpShape::Reduce { axis: None },
            OpShape::StackChannels(2),
            OpShape::InputRank,
        ];
        for shape in shapes {
            let claimed = shape.rank(&[None]).expect("answers without the input");
            for n in 1..=4 {
                let at = shape.rank(&[Some(n)]);
                assert!(
                    at.is_none_or(|at| at == claimed),
                    "{shape:?} over rank {n}: {at:?}"
                );
            }
        }
    }
}
