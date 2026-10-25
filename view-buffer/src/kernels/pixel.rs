//! Per-pixel maps: the u8 conversion image ops work in, grayscale and
//! threshold. Each output pixel depends on its input pixel alone.

use super::{get_channel_count, is_single_channel};
use crate::core::buffer::ViewBuffer;
use crate::core::cut::{AgainstCut, Cut};
use crate::core::dtype::DType;
use crate::core::dtype::{with_dtype, ViewType};
use crate::core::map::{map_new, map_owned, map_pixels, ElementMap, ElementMapInPlace, PixelMap};
use std::marker::PhantomData;
use std::mem::MaybeUninit;

/// Convert a buffer to U8 for image operations.
///
/// This handles dtype promotion for image operations:
/// - F32/F64 in [0.0, 1.0] range: scale to [0, 255]
/// - F32/F64 outside range: clamp then scale
/// - U16: scale its range (`>> 8`)
/// - Other integer types: saturate into `[0, 255]`
/// - U8: pass through
pub(crate) fn convert_to_u8_for_image(buf: ViewBuffer) -> ViewBuffer {
    if buf.dtype() == DType::U8 {
        return buf;
    }

    with_dtype!(buf.dtype(), S => map_new::<S, u8, _>(&buf, &ImageU8))
}

/// The u8 an image op working in u8 reads an element of each dtype as:
/// floats in `[0, 1]` scaled to `[0, 255]` (clamped first), u16 scaled by
/// its range (`>> 8`), every other integer saturated into `[0, 255]`.
trait ToImageU8: crate::core::dtype::ViewType {
    fn to_image_u8(self) -> u8;
}

macro_rules! to_image_u8 {
    ($($t:ty => |$x:ident| $body:expr),+ $(,)?) => {$(
        impl ToImageU8 for $t {
            #[inline(always)]
            fn to_image_u8(self) -> u8 {
                let $x = self;
                $body
            }
        }
    )+};
}

to_image_u8!(
    u8 => |x| x,
    f32 => |x| (x.clamp(0.0, 1.0) * 255.0).round() as u8,
    f64 => |x| (x.clamp(0.0, 1.0) * 255.0).round() as u8,
    u16 => |x| (x >> 8) as u8,
    i8 => |x| x.max(0) as u8,
    i16 => |x| x.clamp(0, 255) as u8,
    i32 => |x| x.clamp(0, 255) as u8,
    i64 => |x| x.clamp(0, 255) as u8,
    u32 => |x| x.min(255) as u8,
    u64 => |x| x.min(255) as u8,
);

/// [`ToImageU8`] as an element map (`core::map`): dispatched, reading any
/// view where it lies.
struct ImageU8;

// SAFETY: `map_into` writes every slot of `dst`.
unsafe impl<S: ToImageU8> ElementMap<S, u8> for ImageU8 {
    #[inline(always)]
    fn map_into(&self, src: &[S], dst: &mut [MaybeUninit<u8>], _at: usize) {
        for (d, &x) in dst.iter_mut().zip(src) {
            d.write(x.to_image_u8());
        }
    }
}

/// Grayscale of any layout: BT.601, `Y = 0.299R + 0.587G + 0.114B` (u8 in
/// fixed point, `Y = (77R + 150G + 29B + 128) >> 8`), or a gray + alpha
/// image's gray channel, as a per-pixel map (`core::map`): any view is read
/// where it lies. More than four channels are read as the first three (BT.601
/// reads no others), through a view of them.
pub(crate) fn grayscale_strided(buf: ViewBuffer) -> ViewBuffer {
    use crate::kernels::color::Luma;
    let shape = buf.shape();
    let channels = shape.get(2).copied().unwrap_or(1);
    if channels == 1 {
        // Already gray. Packed, since every image op's output is planned
        // contiguous (a no-op when it already is).
        return buf.to_contiguous();
    }
    if channels > 4 {
        let (h, w) = (shape[0], shape[1]);
        return grayscale_strided(buf.slice(&[0, 0, 0], &[h, w, 3]));
    }
    fn gray<T: Luma>(buf: &ViewBuffer, channels: usize) -> ViewBuffer {
        match channels {
            2 => map_pixels::<T, T, 2, 1, _>(buf, &Grayscale),
            3 => map_pixels::<T, T, 3, 1, _>(buf, &Grayscale),
            4 => map_pixels::<T, T, 4, 1, _>(buf, &Grayscale),
            other => unreachable!("grayscale maps 2-4 channels, not {other}"),
        }
    }
    with_dtype!(buf.dtype(), T => gray::<T>(&buf, channels))
}

/// Grayscale as a pixel map: the luma of a colour pixel (its first three
/// channels), the gray channel of a gray + alpha one (`color_channels(C) == 1`).
struct Grayscale;

// SAFETY: `map_into` writes one value per pixel of `src`, into `dst`'s
// matching slot.
unsafe impl<T: crate::kernels::color::Luma, const C: usize> PixelMap<T, T, C, 1> for Grayscale {
    #[inline(always)]
    fn map_into(&self, src: &[[T; C]], dst: &mut [MaybeUninit<T>]) {
        for (d, p) in dst.iter_mut().zip(src) {
            // `C` is a constant, so each instance keeps one arm. The blue
            // index is written `C.min(3) - 1` (2 for every `C >= 3`) only so
            // the discarded arm of gray + alpha stays in bounds.
            d.write(if crate::ops::color::color_channels(C) == 1 {
                p[0]
            } else {
                T::luma(p[0], p[1], p[C.min(3) - 1])
            });
        }
    }
}

/// Threshold: `255` where an element exceeds `thresh`, else `0`, as a u8
/// element map (`core::map`): a u8 buffer only this op reads is written in
/// place, and any view is read where it lies.
///
/// A u8 input with the threshold inside the u8 range compares in u8 (`p > t`
/// for an integer `p` is `p > floor(t)`); any other integer compares exactly
/// exactly ([`Cut`]), and a float in `f64`.
pub(crate) fn threshold_generic(buf: ViewBuffer, thresh: f64) -> ViewBuffer {
    let shape = buf.shape();

    // Validate: threshold only works on single-channel data
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "Threshold requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() first to convert multi-channel images to grayscale."
        );
    }

    if buf.dtype() == DType::U8 && (0.0..=255.0).contains(&thresh) {
        return map_owned::<u8, _>(buf, &ThresholdU8(thresh as u8));
    }
    with_dtype!(buf.dtype(), T => map_new(&buf, &Threshold::<T>::new(thresh)))
}

/// u8 threshold compared in u8.
struct ThresholdU8(u8);

// SAFETY: `map_into` writes every element of `dst`.
unsafe impl ElementMap<u8, u8> for ThresholdU8 {
    #[inline(always)]
    fn map_into(&self, src: &[u8], dst: &mut [MaybeUninit<u8>], _at: usize) {
        let t = self.0;
        for (d, &p) in dst.iter_mut().zip(src) {
            d.write(if p > t { 255 } else { 0 });
        }
    }
}

impl ElementMapInPlace<u8> for ThresholdU8 {
    #[inline(always)]
    fn map_in_place(&self, data: &mut [u8]) {
        let t = self.0;
        for p in data.iter_mut() {
            *p = if *p > t { 255 } else { 0 };
        }
    }
}

/// Threshold of any dtype, compared exactly against the threshold
/// (`core::cut`, the one pixel-against-boundary comparison).
struct Threshold<T>(Cut, PhantomData<fn(T)>);

impl<T> Threshold<T> {
    fn new(thresh: f64) -> Self {
        Threshold(Cut::of(thresh), PhantomData)
    }
}

// SAFETY: `map_into` writes every element of `dst`.
unsafe impl<T: ViewType + AgainstCut> ElementMap<T, u8> for Threshold<T> {
    #[inline(always)]
    fn map_into(&self, src: &[T], dst: &mut [MaybeUninit<u8>], _at: usize) {
        let cut = &self.0;
        for (d, &x) in dst.iter_mut().zip(src) {
            d.write(if x.above(cut) { 255 } else { 0 });
        }
    }
}

/// Grayscale and threshold against independent scalar references, over every
/// layout the kernels read differently: contiguous, cropped (dense rows with
/// a row stride), vertically flipped (negative row stride), horizontally
/// flipped and transposed (not dense rows, so materialised or walked per
/// pixel). Odd widths put a remainder after every vector loop.
#[cfg(test)]
mod grayscale_threshold_parity_tests {
    use crate::core::buffer::ViewBuffer;
    use crate::core::dtype::DType;
    use crate::ops::image::apply_image_inner;
    use crate::ops::{ImageOp, ImageOpKind};

    // (48, 70): more pixels than one walk scratch holds (8 KiB), so a
    // channel-first view is handed out in several runs.
    const SIZES: [(usize, usize); 6] = [(1, 1), (3, 5), (7, 33), (16, 17), (9, 64), (48, 70)];

    /// Pseudo-random bytes (an LCG's high bits). An arithmetic sequence such
    /// as `(i * 7919) % 256` correlates neighbouring channels so strongly
    /// that no pixel of it lands on a rounding boundary of the luma sum.
    fn pattern(len: usize) -> Vec<u8> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 56) as u8
            })
            .collect()
    }

    /// The u16 fixed-point luma equals the u32 formula on every RGB triple.
    #[test]
    fn luma_u8_matches_the_u32_formula_on_every_pixel() {
        for r in 0..=255u8 {
            for g in 0..=255u8 {
                for b in 0..=255u8 {
                    let (r32, g32, b32) = (u32::from(r), u32::from(g), u32::from(b));
                    let expected = ((77 * r32 + 150 * g32 + 29 * b32 + 128) >> 8) as u8;
                    assert_eq!(
                        <u8 as crate::kernels::color::Luma>::luma(r, g, b),
                        expected,
                        "({r}, {g}, {b})"
                    );
                }
            }
        }
    }

    /// The layouts a `[h, w, c]` (or `[h, w]`) buffer reaches a kernel in,
    /// each a view over a larger or reordered parent.
    fn layouts(parent: &ViewBuffer) -> Vec<(&'static str, ViewBuffer)> {
        let shape = crate::core::layout::Dims::from_slice(parent.shape());
        let (h, w) = (shape[0], shape[1]);
        let mut out = vec![
            ("contiguous", parent.clone()),
            ("flip_v", parent.flip(&[0])),
            ("flip_h", parent.flip(&[1])),
        ];
        if h > 2 && w > 2 {
            let mut start = vec![1, 1];
            let mut end = vec![h - 1, w - 1];
            if shape.len() == 3 {
                start.push(0);
                end.push(shape[2]);
            }
            out.push(("crop", parent.slice(&start, &end)));
        }
        let mut perm: Vec<usize> = vec![1, 0];
        if shape.len() == 3 {
            perm.push(2);
        }
        out.push(("transpose", parent.permute(&perm)));
        if shape.len() == 3 && shape[2] > 1 {
            // Stored channel-first, viewed channels-last: channels
            // `h · w` elements apart, so the walk packs single elements and
            // a scratch-sized run need not end on a pixel boundary.
            let c = shape[2];
            let packed = parent.to_contiguous();
            out.push((
                "channel_first",
                crate::core::dtype::with_dtype!(parent.dtype(), T => {
                    let src = packed.as_slice::<T>();
                    let chw: Vec<T> = (0..c)
                        .flat_map(|k| src.iter().skip(k).step_by(c).copied())
                        .collect();
                    ViewBuffer::from_vec_with_shape(chw, vec![c, h, w]).permute(&[1, 2, 0])
                }),
            ));
        }
        out
    }

    fn run(buf: ViewBuffer, kind: ImageOpKind) -> ViewBuffer {
        apply_image_inner(buf, ImageOp { kind })
    }

    fn reference_luma_u8(p: &[u8]) -> u8 {
        if p.len() == 2 {
            return p[0];
        }
        let (r, g, b) = (u32::from(p[0]), u32::from(p[1]), u32::from(p[2]));
        ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8
    }

    #[test]
    fn u8_grayscale_matches_the_fixed_point_reference() {
        for (h, w) in SIZES {
            for c in [2usize, 3, 4] {
                let parent = ViewBuffer::from_vec_with_shape(pattern(h * w * c), vec![h, w, c]);
                for (layout, view) in layouts(&parent) {
                    let packed = view.to_contiguous();
                    let expected: Vec<u8> = packed
                        .as_slice::<u8>()
                        .chunks_exact(c)
                        .map(reference_luma_u8)
                        .collect();
                    let got = run(view.clone(), ImageOpKind::Grayscale { keepdims: true });
                    let (vh, vw) = (view.shape()[0], view.shape()[1]);
                    assert_eq!(got.shape(), &[vh, vw, 1], "{layout} {h}x{w}x{c}");
                    assert_eq!(
                        got.to_contiguous().as_slice::<u8>(),
                        &expected[..],
                        "{layout} {h}x{w}x{c}"
                    );
                }
            }
        }
    }

    /// Non-u8 grayscale computes BT.601 in f64 and converts back, rounding
    /// and clamping integer targets.
    #[test]
    fn typed_grayscale_matches_the_f64_reference() {
        for (h, w) in SIZES {
            for c in [3usize, 4] {
                let src: Vec<f32> = pattern(h * w * c)
                    .iter()
                    .map(|&v| f32::from(v) * 1.37 - 20.25)
                    .collect();
                let parent = ViewBuffer::from_vec_with_shape(src, vec![h, w, c]);
                for (layout, view) in layouts(&parent) {
                    let packed = view.to_contiguous();
                    let expected: Vec<f32> = packed
                        .as_slice::<f32>()
                        .chunks_exact(c)
                        .map(|p| {
                            (0.299 * f64::from(p[0])
                                + 0.587 * f64::from(p[1])
                                + 0.114 * f64::from(p[2])) as f32
                        })
                        .collect();
                    let got = run(view, ImageOpKind::Grayscale { keepdims: true });
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(got.to_contiguous().as_slice::<f32>()),
                        bits(&expected),
                        "{layout} {h}x{w}x{c}"
                    );
                }
                let u16s: Vec<u16> = pattern(h * w * c)
                    .iter()
                    .map(|&v| u16::from(v) * 257)
                    .collect();
                let parent = ViewBuffer::from_vec_with_shape(u16s.clone(), vec![h, w, c]);
                let expected: Vec<u16> = u16s
                    .chunks_exact(c)
                    .map(|p| {
                        let l = 0.299 * f64::from(p[0])
                            + 0.587 * f64::from(p[1])
                            + 0.114 * f64::from(p[2]);
                        l.round().clamp(0.0, 65535.0) as u16
                    })
                    .collect();
                let got = run(parent, ImageOpKind::Grayscale { keepdims: true });
                assert_eq!(got.as_slice::<u16>(), &expected[..], "u16 {h}x{w}x{c}");
            }
        }
    }

    /// A gray + alpha image's grayscale is its gray channel, for every dtype:
    /// the u8 kernel always did this, while the typed one computed BT.601
    /// over (gray, alpha, alpha), mixing opacity into intensity.
    #[test]
    fn gray_alpha_grayscale_is_the_gray_channel_for_every_dtype() {
        let gray_alpha: Vec<u16> = vec![1000, 65535, 20, 0, 40000, 7];
        let buf = ViewBuffer::from_vec_with_shape(gray_alpha, vec![1, 3, 2]);
        let got = run(buf, ImageOpKind::Grayscale { keepdims: true });
        assert_eq!(got.as_slice::<u16>(), &[1000, 20, 40000]);
        let buf = ViewBuffer::from_vec_with_shape(vec![0.25f32, 1.0, 0.5, 0.0], vec![2, 1, 2]);
        let got = run(buf, ImageOpKind::Grayscale { keepdims: true });
        assert_eq!(got.as_slice::<f32>(), &[0.25, 0.5]);
    }

    #[test]
    fn threshold_matches_the_comparison_reference() {
        for (h, w) in SIZES {
            for rank3 in [false, true] {
                let shape = if rank3 { vec![h, w, 1] } else { vec![h, w] };
                let parent = ViewBuffer::from_vec_with_shape(pattern(h * w), shape);
                for (layout, view) in layouts(&parent) {
                    let packed = view.to_contiguous();
                    for t in [-1.0f64, 0.0, 127.5, 128.0, 254.9, 255.0, 300.0] {
                        let expected: Vec<u8> = packed
                            .as_slice::<u8>()
                            .iter()
                            .map(|&v| if f64::from(v) > t { 255 } else { 0 })
                            .collect();
                        let got = run(view.clone(), ImageOpKind::Threshold { value: t });
                        assert_eq!(got.shape(), view.shape(), "{layout} rank3={rank3}");
                        assert_eq!(got.dtype(), DType::U8);
                        assert_eq!(
                            got.to_contiguous().as_slice::<u8>(),
                            &expected[..],
                            "{layout} {h}x{w} rank3={rank3} t={t}"
                        );
                    }
                }
            }
        }
    }

    /// An integer pixel is compared with the threshold exactly (`p > t` iff
    /// `p > floor(t)`), not rounded to f64 first: above 2**53 neighbouring
    /// pixels share an f64, so the rounded comparison answered for the
    /// neighbour.
    #[test]
    fn integer_threshold_is_exact_beyond_f64_precision() {
        // 18442240474082181119 as f64 is ...120: the rounded comparison put
        // ...119 above it.
        let u = ViewBuffer::from_vec_with_shape(
            vec![
                18442240474082181119u64,
                18442240474082181120,
                18442240474082183169,
                u64::MAX,
            ],
            vec![1, 4],
        );
        let i = ViewBuffer::from_vec_with_shape(
            vec![3797082577976980i64, 3797082577976981, i64::MAX, i64::MIN],
            vec![1, 4],
        );
        // 2**53 + 1 rounds to 2**53 as f64, so the rounded comparison said it
        // does not exceed 2**53.
        let w = ViewBuffer::from_vec_with_shape(
            vec![
                9007199254740991i64,
                9007199254740992,
                9007199254740993,
                9007199254740995,
            ],
            vec![1, 4],
        );
        let cases: [(&ViewBuffer, f64, [u8; 4]); 12] = [
            (&w, 9007199254740992.0, [0, 0, 255, 255]),
            (&w, 9007199254740994.0, [0, 0, 0, 255]),
            (&u, 18442240474082181119u64 as f64, [0, 0, 255, 255]),
            (&u, 0.0, [255; 4]),
            (&u, -1e300, [255; 4]),
            (&u, 1e300, [0; 4]),
            (&u, f64::NAN, [0; 4]),
            (&i, 3797082577976980.5, [0, 255, 255, 0]),
            (&i, -0.5, [255, 255, 255, 0]),
            (&i, 9.3e18, [0; 4]),
            (&i, -9.3e18, [255; 4]),
            (&i, f64::NAN, [0; 4]),
        ];
        for (buf, t, expected) in cases {
            let got = run(buf.clone(), ImageOpKind::Threshold { value: t });
            assert_eq!(
                got.as_slice::<u8>(),
                &expected[..],
                "{:?} t={t}",
                buf.dtype()
            );
        }
    }

    /// Any other dtype compares the element's value, and still writes u8, for
    /// every layout.
    #[test]
    fn typed_threshold_matches_the_comparison_reference() {
        for (h, w) in SIZES {
            let values: Vec<f64> = pattern(h * w)
                .iter()
                .map(|&v| f64::from(v) * 3.5 - 200.0)
                .collect();
            let parents = [
                ViewBuffer::from_vec_with_shape(
                    values.iter().map(|&v| v as i16).collect::<Vec<_>>(),
                    vec![h, w, 1],
                ),
                ViewBuffer::from_vec_with_shape(
                    values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                    vec![h, w],
                ),
                ViewBuffer::from_vec_with_shape(
                    values
                        .iter()
                        .map(|&v| (v + 200.0) as u64)
                        .collect::<Vec<_>>(),
                    vec![h, w, 1],
                ),
            ];
            for parent in parents {
                for (layout, view) in layouts(&parent) {
                    let as_f64: Vec<f64> = crate::core::dtype::with_dtype!(view.dtype(), T => {
                        view.to_contiguous().as_slice::<T>().iter().map(|x| num_traits::ToPrimitive::to_f64(x).unwrap()).collect()
                    });
                    for t in [-50.0f64, 0.0, 99.5, 300.0] {
                        let expected: Vec<u8> = as_f64
                            .iter()
                            .map(|&v| if v > t { 255 } else { 0 })
                            .collect();
                        let got = run(view.clone(), ImageOpKind::Threshold { value: t });
                        assert_eq!(got.dtype(), DType::U8);
                        assert_eq!(
                            got.to_contiguous().as_slice::<u8>(),
                            &expected[..],
                            "{:?} {layout} {h}x{w} t={t}",
                            view.dtype()
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod image_u8_tests {
    use super::convert_to_u8_for_image;
    use crate::core::buffer::ViewBuffer;
    use crate::core::dtype::DType;

    /// Every integer dtype wider than u8 but u16 (which scales its range,
    /// `>> 8`) saturates into u8, the 64-bit ones as the 32-bit ones do: a
    /// 300 is 255 and a -5 is 0, not wrapped to 44 and 251.
    #[test]
    fn integers_saturate_into_u8_at_every_width() {
        let values = [-5i64, 0, 100, 255, 256, 300, 70_000];
        let want: Vec<u8> = values.iter().map(|&v| v.clamp(0, 255) as u8).collect();
        for dtype in [DType::I16, DType::I32, DType::I64] {
            let buf = ViewBuffer::from_vec_with_shape(values.to_vec(), vec![1, 7, 1]).cast(dtype);
            let got = convert_to_u8_for_image(buf);
            assert_eq!(got.as_slice::<u8>(), &want[..], "{dtype:?}");
        }
        let unsigned = [0u64, 100, 255, 256, 300, 70_000];
        let want: Vec<u8> = unsigned.iter().map(|&v| v.min(255) as u8).collect();
        for dtype in [DType::U32, DType::U64] {
            let buf = ViewBuffer::from_vec_with_shape(unsigned.to_vec(), vec![1, 6, 1]).cast(dtype);
            let got = convert_to_u8_for_image(buf);
            assert_eq!(got.as_slice::<u8>(), &want[..], "{dtype:?}");
        }
    }
}
