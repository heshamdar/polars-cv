//! Copies of a buffer's elements, counted.
//!
//! Image encoding reads the buffer's pixels in place, and appending a buffer
//! to a flat values vector copies each element once.
//!
//! A copy of the pixels is one allocation the size of the image, so the
//! number of image-sized allocations made while encoding a flat (highly
//! compressible) image is the observable. A codec may need one of its own:
//! PNG stores 16-bit samples big-endian, so its encoder byte-swaps them into
//! a new buffer. Anything beyond the codec's own is a copy of ours. Tracked
//! per thread, so tests running concurrently in this binary do not see each
//! other's allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use view_buffer::interop::image::ImageAdapter;
use view_buffer::ViewBuffer;

struct Tracking;

thread_local! {
    /// Allocations of at least this many bytes are counted; 0 is off.
    static THRESHOLD: Cell<usize> = const { Cell::new(0) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

fn note(size: usize) {
    // `try_with`: the allocator also runs during thread teardown.
    let _ = THRESHOLD.try_with(|t| {
        if t.get() != 0 && size >= t.get() {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Tracking = Tracking;

/// How many allocations of at least `threshold` bytes `f` makes on this
/// thread.
fn large_allocations<R>(threshold: usize, f: impl FnOnce() -> R) -> (R, usize) {
    COUNT.with(|c| c.set(0));
    THRESHOLD.with(|t| t.set(threshold));
    let out = f();
    THRESHOLD.with(|t| t.set(0));
    (out, COUNT.with(Cell::get))
}

const H: usize = 512;
const W: usize = 512;

fn flat_u8(channels: usize) -> ViewBuffer {
    ViewBuffer::from_vec(vec![7u8; H * W * channels]).reshape(vec![H, W, channels])
}

fn flat_u16(channels: usize) -> ViewBuffer {
    ViewBuffer::from_vec(vec![7u16; H * W * channels]).reshape(vec![H, W, channels])
}

/// Encoding `buf` makes no image-sized allocation beyond the `codec_own`
/// the codec itself needs.
fn assert_no_pixel_copy(
    label: &str,
    buf: &ViewBuffer,
    codec_own: usize,
    encode: impl FnOnce(&ViewBuffer) -> Vec<u8>,
) {
    let image_bytes = buf.shape().iter().product::<usize>() * buf.dtype().size_of();
    let (encoded, count) = large_allocations(image_bytes, || encode(buf));
    assert!(!encoded.is_empty(), "{label}: nothing encoded");
    assert!(
        count <= codec_own,
        "{label}: {count} image-sized allocations while encoding, the codec needs \
         {codec_own} — the pixels were copied"
    );
}

#[test]
fn png_encodes_every_native_layout_in_place() {
    for c in 1..=4 {
        assert_no_pixel_copy(&format!("png u8 x{c}"), &flat_u8(c), 0, |b| {
            ImageAdapter::encode(b, image::ImageFormat::Png).unwrap()
        });
        // The big-endian swap.
        assert_no_pixel_copy(&format!("png u16 x{c}"), &flat_u16(c), 1, |b| {
            ImageAdapter::encode(b, image::ImageFormat::Png).unwrap()
        });
    }
}

#[test]
fn jpeg_encodes_its_native_layouts_in_place() {
    for c in [1, 3] {
        assert_no_pixel_copy(&format!("jpeg u8 x{c}"), &flat_u8(c), 0, |b| {
            ImageAdapter::encode_jpeg(b, 90).unwrap()
        });
    }
}

#[test]
fn a_rank_2_buffer_encodes_in_place() {
    let buf = ViewBuffer::from_vec(vec![7u8; H * W]).reshape(vec![H, W]);
    assert_no_pixel_copy("png u8 [H, W]", &buf, 0, |b| {
        ImageAdapter::encode(b, image::ImageFormat::Png).unwrap()
    });
}

/// JPEG has no alpha: the encoder refuses a `GrayA`/`RGBA` buffer as it is,
/// and the encode converts through the image crate instead (dropping alpha).
#[test]
fn jpeg_with_alpha_still_encodes_by_conversion() {
    for (c, decoded_channels) in [(2usize, 1u8), (4, 3)] {
        let bytes = ImageAdapter::encode_jpeg(&flat_u8(c), 90).unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!(decoded.color().channel_count(), decoded_channels, "x{c}");
        assert_eq!((decoded.width(), decoded.height()), (W as u32, H as u32));
    }
}

/// Every layout a view can have: contiguous, permuted, flipped (negative
/// strides) and a sliced window with an offset.
fn layouts() -> Vec<(&'static str, ViewBuffer)> {
    let base =
        ViewBuffer::from_vec((0..(6 * 5 * 3) as u16).collect::<Vec<_>>()).reshape(vec![6, 5, 3]);
    vec![
        ("contiguous", base.clone()),
        ("permuted", base.permute(&[1, 0, 2])),
        ("inner permuted", base.permute(&[2, 1, 0])),
        ("flipped", base.flip(&[0, 1])),
        ("sliced", base.slice(&[1, 1, 0], &[5, 4, 2])),
    ]
}

#[test]
fn append_to_appends_the_row_major_elements() {
    for (label, buf) in layouts() {
        let mut out: Vec<u16> = vec![9, 9];
        buf.append_to(&mut out);
        let expected = buf.to_contiguous();
        assert_eq!(&out[..2], &[9, 9], "{label}: existing elements kept");
        assert_eq!(&out[2..], expected.as_slice::<u16>(), "{label}");
    }
}

#[test]
fn append_to_copies_each_element_once() {
    for (label, buf) in layouts() {
        let n = buf.shape().iter().product::<usize>();
        let mut out: Vec<u16> = Vec::with_capacity(n);
        // Anything the size of the view is an intermediate copy: the
        // destination is already reserved.
        let ((), count) = large_allocations(n * 2, || buf.append_to(&mut out));
        assert_eq!(count, 0, "{label}: an intermediate copy");
        assert_eq!(out.len(), n, "{label}");
    }
}

#[test]
#[should_panic(expected = "dtype")]
fn append_to_refuses_another_element_type() {
    let buf = ViewBuffer::from_vec(vec![1u8, 2, 3]);
    buf.append_to(&mut Vec::<u16>::new());
}

// --- Element-wise ops: which allocations a solely owned buffer pays ---

use std::sync::Arc;
use view_buffer::execution::ExecutionPlan;
use view_buffer::ops::scalar::{FusedKernel, ScalarOp};
use view_buffer::{DType, FilterType, Normalization, ViewExpr};

/// Run `build` on `buf` the way the plugin's executor does: plan from a
/// clone, drop the plan's copy, and execute with the source moved in, so the
/// buffer reaches the op as its sole owner.
fn run_owned(buf: ViewBuffer, build: impl Fn(&Arc<ViewExpr>) -> Arc<ViewExpr>) -> ViewBuffer {
    let steps = build(&ViewExpr::new_source(buf.clone())).plan().steps;
    ExecutionPlan { source: buf, steps }.execute()
}

/// A patterned `[H, W, c]` u8 image (not flat, so a wrong in-place result
/// would show in the values).
fn pattern_u8(channels: usize) -> ViewBuffer {
    let data: Vec<u8> = (0..H * W * channels)
        .map(|i| (i * 31 % 251) as u8)
        .collect();
    ViewBuffer::from_vec(data).reshape(vec![H, W, channels])
}

/// How many allocations the size of `buf`'s u8 pixels running `build` on a
/// solely owned `buf` makes, with the result.
fn owned_u8_allocations(
    buf: ViewBuffer,
    build: impl Fn(&Arc<ViewExpr>) -> Arc<ViewExpr>,
) -> (ViewBuffer, usize) {
    let image_bytes = buf.shape().iter().product::<usize>();
    large_allocations(image_bytes, || run_owned(buf, build))
}

#[test]
fn a_solely_owned_u8_invert_writes_in_place() {
    let (out, count) = owned_u8_allocations(pattern_u8(3), |e| e.invert());
    assert_eq!(out.dtype(), DType::U8);
    assert_eq!(out.as_slice::<u8>()[..4], [255, 224, 193, 162]);
    assert_eq!(count, 0, "invert allocated a buffer only it reads");
}

#[test]
fn a_shared_u8_invert_allocates_its_output_once() {
    let buf = pattern_u8(3);
    let keep = buf.clone();
    let image_bytes = buf.shape().iter().product::<usize>();
    let (out, count) = large_allocations(image_bytes, || run_owned(buf, |e| e.invert()));
    assert_eq!(count, 1, "a shared input must be copied once, not written");
    assert_eq!(
        keep.as_slice::<u8>()[..2],
        [0, 31],
        "the shared input was written"
    );
    assert_eq!(out.as_slice::<u8>()[..2], [255, 224]);
}

#[test]
fn a_solely_owned_u8_to_u8_fused_chain_writes_in_place() {
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Mul(1.2));
    kernel.push(ScalarOp::Add(-10.0));
    kernel.push(ScalarOp::Clamp(0.0, 255.0));
    kernel.out_dtype = DType::U8;
    let (out, count) = owned_u8_allocations(pattern_u8(3), |e| e.fused(kernel.clone()));
    assert_eq!(out.dtype(), DType::U8);
    // 31 * 1.2 - 10 = 27.2 -> 27; 62 * 1.2 - 10 = 64.4 -> 64.
    assert_eq!(out.as_slice::<u8>()[..3], [0, 27, 64]);
    assert_eq!(
        count, 0,
        "a u8 -> u8 chain allocated a buffer only it reads"
    );
}

#[test]
fn a_solely_owned_u8_threshold_writes_in_place() {
    let (out, count) = owned_u8_allocations(pattern_u8(1), |e| e.threshold(100.0));
    assert_eq!(out.as_slice::<u8>()[..5], [0, 0, 0, 0, 255]);
    assert_eq!(count, 0, "threshold allocated a buffer only it reads");
}

/// u8 -> f32 needs exactly one new buffer: the f32 output. The old path
/// cast to f32 first and then mapped into a second f32 buffer.
#[test]
fn a_u8_preset_normalize_allocates_only_its_f32_output() {
    let preset = Normalization::Preset {
        mean: vec![123.7, 116.3, 103.5],
        std: vec![58.4, 57.1, 57.4],
    };
    let (out, count) =
        owned_u8_allocations(pattern_u8(3), |e| e.normalize(preset.clone(), DType::F32));
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.as_slice::<f32>()[0], (0.0 - 123.7f32) / 58.4);
    assert_eq!(
        count, 1,
        "{count} image-sized allocations, the f32 output is the only one needed"
    );
}

/// Unchanged behaviour, pinned beside the new cases: an f32 scale on a
/// solely owned buffer writes in place.
#[test]
fn a_solely_owned_f32_scale_writes_in_place() {
    let data: Vec<f32> = (0..H * W).map(|i| i as f32).collect();
    let buf = ViewBuffer::from_vec(data).reshape(vec![H, W]);
    let (out, count) = large_allocations(H * W * 4, || run_owned(buf, |e| e.scale(2.0)));
    assert_eq!(out.as_slice::<f32>()[..3], [0.0, 2.0, 4.0]);
    assert_eq!(count, 0);
}

/// Resize reads a crop or a vertical flip where it lies: the output is the
/// only buffer the size of the view's pixels. Packing the view first, as
/// resize once did, is a second one. The output is made larger than the view
/// so it is counted; fast_image_resize's own scratch is warmed up first (it
/// keeps it per thread, across calls).
#[test]
fn resizing_a_view_with_packed_rows_allocates_only_its_output() {
    let views = [
        ("crop", pattern_u8(3).slice(&[64, 32, 0], &[448, 416, 3])),
        (
            "flip_v",
            pattern_u8(3).slice(&[0, 0, 0], &[384, 384, 3]).flip(&[0]),
        ),
    ];
    for (label, view) in views {
        let (h, w) = (view.shape()[0] as u32 + 16, view.shape()[1] as u32 + 16);
        run_owned(view.clone(), |e| e.resize(w, h, FilterType::Triangle));
        let view_bytes = view.shape().iter().product::<usize>();
        let (out, count) = large_allocations(view_bytes, || {
            run_owned(view, |e| e.resize(w, h, FilterType::Triangle))
        });
        assert_eq!(out.shape(), [h as usize, w as usize, 3], "{label}");
        assert_eq!(
            count, 1,
            "{label}: {count} view-sized allocations, the output is the only one needed"
        );
    }
}
