//! Resize reads a view whose pixels are packed within each row — a crop, a
//! vertical flip, both — where it lies, and must produce exactly the bytes
//! fast_image_resize produces from the same pixels packed into one slice and
//! read by its own `ImageRef`, independently of the engine's row adapter.
//!
//! Every native fast_image_resize pixel type (u8/u16/f32 × 1–4 channels) and a
//! rank-2 image are covered, each through downscale and upscale with the
//! filters that read one, two and many rows per output row, so a row read from
//! the wrong place, in the wrong order or at the wrong length changes the
//! output.

#![cfg(feature = "image_interop")]

use std::sync::Arc;

use fast_image_resize as fir;
use view_buffer::core::dtype::ViewType;
use view_buffer::execution::ExecutionPlan;
use view_buffer::{DType, FilterType, ViewBuffer, ViewExpr};

const H: usize = 37;
const W: usize = 29;

/// A patterned `[H, W, c]` image of `T` (rank 2 when `c` is `None`).
fn pattern<T: ViewType>(channels: Option<usize>, value: impl Fn(usize) -> T) -> ViewBuffer {
    let c = channels.unwrap_or(1);
    let data: Vec<T> = (0..H * W * c).map(value).collect();
    let shape = match channels {
        Some(c) => vec![H, W, c],
        None => vec![H, W],
    };
    ViewBuffer::from_vec_with_shape(data, shape)
}

fn image(dtype: DType, channels: Option<usize>) -> ViewBuffer {
    match dtype {
        DType::U8 => pattern(channels, |i| (i * 37 % 251) as u8),
        DType::U16 => pattern(channels, |i| (i * 7919 % 65_521) as u16),
        DType::F32 => pattern(channels, |i| (i * 37 % 251) as f32 * 0.731 - 20.0),
        other => unreachable!("no native resize for {other:?}"),
    }
}

/// The views under test: each has packed rows but is not contiguous.
fn views(buf: &ViewBuffer) -> Vec<(&'static str, ViewBuffer)> {
    let rank = buf.shape().len();
    let crop = |b: &ViewBuffer| {
        let mut start = vec![3, 5];
        let mut end = vec![H - 4, W - 6];
        if rank == 3 {
            start.push(0);
            end.push(buf.shape()[2]);
        }
        b.slice(&start, &end)
    };
    vec![
        ("crop", crop(buf)),
        ("flip_v", buf.flip(&[0])),
        ("crop of flip_v", crop(&buf.flip(&[0]))),
    ]
}

fn resize(buf: ViewBuffer, h: u32, w: u32, filter: FilterType) -> ViewBuffer {
    let build = |e: &Arc<ViewExpr>| e.resize(w, h, filter);
    let steps = build(&ViewExpr::new_source(buf.clone())).plan().steps;
    ExecutionPlan { source: buf, steps }.execute()
}

/// A contiguous buffer's elements as native-endian bytes.
fn bytes(buf: &ViewBuffer) -> Vec<u8> {
    match buf.dtype() {
        DType::U8 => buf.as_slice::<u8>().to_vec(),
        DType::U16 => buf
            .as_slice::<u16>()
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect(),
        DType::F32 => buf
            .as_slice::<f32>()
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect(),
        other => unreachable!("no native resize for {other:?}"),
    }
}

/// fast_image_resize run directly on `packed`'s pixels: the oracle.
fn fir_reference(packed: &ViewBuffer, h: u32, w: u32, filter: FilterType) -> Vec<u8> {
    use fir::PixelType as PT;
    let channels = packed.shape().get(2).copied().unwrap_or(1);
    let pixel_type = match (packed.dtype(), channels) {
        (DType::U8, 1) => PT::U8,
        (DType::U8, 2) => PT::U8x2,
        (DType::U8, 3) => PT::U8x3,
        (DType::U8, 4) => PT::U8x4,
        (DType::U16, 1) => PT::U16,
        (DType::U16, 2) => PT::U16x2,
        (DType::U16, 3) => PT::U16x3,
        (DType::U16, 4) => PT::U16x4,
        (DType::F32, 1) => PT::F32,
        (DType::F32, 2) => PT::F32x2,
        (DType::F32, 3) => PT::F32x3,
        (DType::F32, 4) => PT::F32x4,
        other => unreachable!("no native resize for {other:?}"),
    };
    let algorithm = match filter {
        FilterType::Nearest => fir::ResizeAlg::Nearest,
        FilterType::Triangle => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        FilterType::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
        other => unreachable!("not exercised: {other:?}"),
    };
    let size = packed.shape().iter().product::<usize>() * packed.dtype().size_of();
    // SAFETY: `packed` is contiguous, so its `size` bytes from the first
    // element are its pixels, aligned for its dtype.
    let src_bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr::<u8>(), size) };
    let (src_h, src_w) = (packed.shape()[0] as u32, packed.shape()[1] as u32);
    let src = fir::images::ImageRef::new(src_w, src_h, src_bytes, pixel_type).unwrap();
    let mut dst = fir::images::Image::new(w, h, pixel_type);
    fir::Resizer::new()
        .resize(
            &src,
            &mut dst,
            &fir::ResizeOptions::new().resize_alg(algorithm),
        )
        .unwrap();
    dst.into_vec()
}

#[test]
fn resizing_a_view_matches_resizing_it_packed() {
    let channel_counts = [None, Some(1), Some(2), Some(3), Some(4)];
    let sizes = [(11, 13), (53, 41)];
    let filters = [
        FilterType::Nearest,
        FilterType::Triangle,
        FilterType::Lanczos3,
    ];
    for dtype in [DType::U8, DType::U16, DType::F32] {
        for channels in channel_counts {
            let buf = image(dtype, channels);
            for (view_name, view) in views(&buf) {
                assert!(
                    !view.layout_facts().is_contiguous(),
                    "{view_name} is not a view"
                );
                let packed = view.to_contiguous();
                for (h, w) in sizes {
                    for filter in filters {
                        let label = format!(
                            "{dtype:?} channels={channels:?} {view_name} -> {h}x{w} {filter:?}"
                        );
                        let got = resize(view.clone(), h, w, filter);
                        let mut shape = view.shape().to_vec();
                        shape[..2].copy_from_slice(&[h as usize, w as usize]);
                        assert_eq!(got.shape(), shape, "{label}: shape");
                        assert_eq!(got.dtype(), dtype, "{label}: dtype");
                        assert!(
                            bytes(&got) == fir_reference(&packed, h, w, filter),
                            "{label}: pixels differ"
                        );
                    }
                }
            }
        }
    }
}
