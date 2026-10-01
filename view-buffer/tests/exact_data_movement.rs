//! Ops that only move data are exact on every dtype.
//!
//! A channel reorder, a merge of planes, a gray-to-RGB replication and the
//! morphological gradient (a max, a min and their difference) compute no new
//! values, so no dtype may lose one. Each case uses values f32 cannot hold
//! (2**53 + 1, `u64::MAX`, 0.1 as f64): these ops used to convert every dtype
//! but u8/u16/f32 through f32 (`u32 16777217 -> 16777216`).

#![cfg(feature = "image_interop")]

use view_buffer::core::dtype::ViewType;
use view_buffer::ops::color::{ColorConvertOp, ColorSpace};
use view_buffer::{apply_channel_merge, ImageOp, ImageOpKind, ViewBuffer, ViewDto, ViewExpr};

fn run(buf: ViewBuffer, dto: ViewDto) -> ViewBuffer {
    ViewExpr::new_source(buf).apply_op(dto).plan().execute()
}

/// Four wide values per dtype, none of which survives a round trip through
/// f32 (for the 32/64-bit dtypes and f64).
fn check_all(check: &dyn Fn(Box<dyn Case>)) {
    check(Box::new(Wide([16_777_217u32, 16_777_219, u32::MAX, 3])));
    check(Box::new(Wide([-16_777_217i32, 16_777_219, i32::MIN, 3])));
    check(Box::new(Wide([
        9_007_199_254_740_993u64,
        u64::MAX,
        2,
        1 << 60,
    ])));
    check(Box::new(Wide([
        -9_007_199_254_740_993i64,
        i64::MAX,
        i64::MIN,
        5,
    ])));
    check(Box::new(Wide([0.1f64, 1.0 / 3.0, -2.5e-300, 1e300])));
}

trait Case {
    fn name(&self) -> String;
    /// `[1, 4, 3]`: pixel `p` is `(v[p], v[p+1], v[p+2])` cyclically.
    fn rgb(&self) -> (ViewBuffer, Vec<String>);
    /// `[1, 4]`: the four values.
    fn plane(&self, shift: usize) -> (ViewBuffer, Vec<String>);
    fn read(&self, buf: &ViewBuffer) -> Vec<String>;
}

struct Wide<T>([T; 4]);

impl<T: ViewType + std::fmt::Debug> Case for Wide<T> {
    fn name(&self) -> String {
        format!("{:?}", T::DTYPE)
    }
    fn rgb(&self) -> (ViewBuffer, Vec<String>) {
        let v = &self.0;
        let data: Vec<T> = (0..4)
            .flat_map(|p| [v[p], v[(p + 1) % 4], v[(p + 2) % 4]])
            .collect();
        let text = data.iter().map(|x| format!("{x:?}")).collect();
        (ViewBuffer::from_vec_with_shape(data, vec![1, 4, 3]), text)
    }
    fn plane(&self, shift: usize) -> (ViewBuffer, Vec<String>) {
        let data: Vec<T> = (0..4).map(|p| self.0[(p + shift) % 4]).collect();
        let text = data.iter().map(|x| format!("{x:?}")).collect();
        (ViewBuffer::from_vec_with_shape(data, vec![1, 4]), text)
    }
    fn read(&self, buf: &ViewBuffer) -> Vec<String> {
        assert_eq!(buf.dtype(), T::DTYPE, "{}: dtype changed", self.name());
        buf.to_contiguous()
            .as_slice::<T>()
            .iter()
            .map(|x| format!("{x:?}"))
            .collect()
    }
}

/// Reorder each pixel's three values (`v[p][order[i]]`).
fn reorder(values: &[String], order: [usize; 3]) -> Vec<String> {
    values
        .chunks(3)
        .flat_map(|px| order.map(|i| px[i].clone()))
        .collect()
}

#[test]
fn channel_swap_is_exact() {
    check_all(&|case| {
        let (buf, values) = case.rgb();
        let swap = ViewDto::Image(ImageOp {
            kind: ImageOpKind::ChannelSwap {
                order: vec![2, 0, 1],
            },
        });
        let out = case.read(&run(buf, swap));
        assert_eq!(out, reorder(&values, [2, 0, 1]), "{}", case.name());
    });
}

#[test]
fn rgb_to_bgr_is_exact() {
    check_all(&|case| {
        let (buf, values) = case.rgb();
        let to_bgr = ViewDto::Color(ColorConvertOp {
            from_space: ColorSpace::Rgb,
            to_space: ColorSpace::Bgr,
        });
        let out = case.read(&run(buf, to_bgr));
        assert_eq!(out, reorder(&values, [2, 1, 0]), "{}", case.name());
    });
}

#[test]
fn gray_to_rgb_is_exact() {
    check_all(&|case| {
        let (plane, values) = case.plane(0);
        let gray = plane.reshape(vec![1, 4, 1]);
        let to_rgb = ViewDto::Color(ColorConvertOp {
            from_space: ColorSpace::Gray,
            to_space: ColorSpace::Rgb,
        });
        let out = case.read(&run(gray, to_rgb));
        let expected: Vec<String> = values
            .iter()
            .flat_map(|v| [v.clone(), v.clone(), v.clone()])
            .collect();
        assert_eq!(out, expected, "{}", case.name());
    });
}

#[test]
fn channel_merge_is_exact() {
    check_all(&|case| {
        let planes: Vec<(ViewBuffer, Vec<String>)> = (0..3).map(|s| case.plane(s)).collect();
        let refs: Vec<&ViewBuffer> = planes.iter().map(|(b, _)| b).collect();
        let out = case.read(&apply_channel_merge(&refs));
        let expected: Vec<String> = (0..4)
            .flat_map(|p| planes.iter().map(move |(_, v)| v[p].clone()))
            .collect();
        assert_eq!(out, expected, "{}", case.name());
    });
}

/// Two values a step apart that f32 rounds to one: their gradient is the
/// step, not zero.
#[test]
fn morph_gradient_is_exact() {
    let gradient = ViewDto::Image(ImageOp {
        kind: ImageOpKind::MorphGradient { ksize: 3 },
    });
    let u = ViewBuffer::from_vec_with_shape(
        vec![9_007_199_254_740_993u64, 9_007_199_254_740_995],
        vec![1, 2, 1],
    );
    let out = run(u, gradient.clone());
    assert_eq!(out.to_contiguous().as_slice::<u64>(), &[2, 2]);
    let u = ViewBuffer::from_vec_with_shape(vec![16_777_217u32, 16_777_218], vec![1, 2, 1]);
    let out = run(u, gradient);
    assert_eq!(out.to_contiguous().as_slice::<u32>(), &[1, 1]);
}
