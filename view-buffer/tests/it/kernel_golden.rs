//! Golden digests of the hot kernels' outputs, recorded before the kernels
//! were optimised: every byte each case produces, hashed, against
//! `kernel_golden.txt`.
//!
//! The Python golden corpus pins every op once, on one 100×200 image. This
//! adds what a fast path is likeliest to get wrong — 1×1, odd widths, rows
//! that no SIMD width divides, views (crop, flips, transpose) and the
//! common chains — so any byte of drift in those fails here.
//!
//! The digests are exact, so they are compared only on x86_64 Linux, where
//! they were recorded (another target's libm may round a transcendental
//! differently). Record afresh with `VIEW_BUFFER_BLESS=1` — only for a change
//! that is meant to move outputs, and say so in its commit message.

use std::fmt::Write as _;
use std::sync::Arc;
use view_buffer::execution::ExecutionPlan;
use view_buffer::ops::scalar::{FusedKernel, ScalarOp};
use view_buffer::{
    ColorConvertOp, ColorSpace, DType, FilterType, NormalizeMethod, ViewBuffer, ViewDto, ViewExpr,
};

const FIXTURE: &str = "tests/it/kernel_golden.txt";

fn exec(buf: ViewBuffer, build: impl Fn(&Arc<ViewExpr>) -> Arc<ViewExpr>) -> ViewBuffer {
    let steps = build(&ViewExpr::new_source(buf.clone())).plan().steps;
    ExecutionPlan { source: buf, steps }.execute()
}

/// FNV-1a over the dtype, shape and logical bytes.
fn digest(buf: &ViewBuffer) -> String {
    let dense = buf.to_contiguous();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    eat(format!("{:?}{:?}", dense.dtype(), dense.shape()).as_bytes());
    macro_rules! bytes {
        ($t:ty) => {{
            let s = dense.as_slice::<$t>();
            // SAFETY: a slice of plain numbers viewed as its bytes.
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
        }};
    }
    match dense.dtype() {
        DType::U8 => eat(bytes!(u8)),
        DType::U16 => eat(bytes!(u16)),
        DType::F32 => eat(bytes!(f32)),
        DType::F64 => eat(bytes!(f64)),
        other => panic!("no golden case produces {other:?}"),
    }
    format!("{h:016x}")
}

/// A deterministic gradient-plus-noise `[h, w, c]` u8 image.
fn image(h: usize, w: usize, c: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (h * 131 + w * 7 + c) as u64;
    let mut out = Vec::with_capacity(h * w * c);
    for y in 0..h {
        for x in 0..w {
            for k in 0..c {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = (state >> 58) as usize;
                out.push(((x * 255 / w.max(1) + y * 97 + k * 53 + noise) % 256) as u8);
            }
        }
    }
    out
}

fn source(h: usize, w: usize, c: usize, dtype: DType, view: &str) -> ViewBuffer {
    let data = image(h, w, c);
    let buf = match dtype {
        DType::U8 => ViewBuffer::from_vec_with_shape(data, vec![h, w, c]),
        DType::U16 => ViewBuffer::from_vec_with_shape(
            data.iter().map(|&v| u16::from(v) * 257).collect::<Vec<_>>(),
            vec![h, w, c],
        ),
        DType::F32 => ViewBuffer::from_vec_with_shape(
            data.iter()
                .map(|&v| f32::from(v) * 1.003 - 0.4)
                .collect::<Vec<_>>(),
            vec![h, w, c],
        ),
        other => panic!("no golden source is {other:?}"),
    };
    match view {
        "crop" if h > 2 && w > 2 => buf.slice(&[1, 1, 0], &[h - 1, w - 1, c]),
        "flip_h" => buf.flip(&[1]),
        "flip_v" => buf.flip(&[0]),
        "transpose" => buf.permute(&[1, 0, 2]),
        _ => buf,
    }
}

type Build = fn(&Arc<ViewExpr>) -> Arc<ViewExpr>;

fn color(to: ColorSpace) -> ViewDto {
    ViewDto::Color(ColorConvertOp {
        from_space: ColorSpace::Rgb,
        to_space: to,
    })
}

/// `(name, channels it takes, dtypes, build)`.
fn ops() -> Vec<(&'static str, &'static [usize], &'static [DType], Build)> {
    const ANY: &[usize] = &[1, 3, 4];
    const RGB: &[usize] = &[3];
    const ALL: &[DType] = &[DType::U8, DType::U16, DType::F32];
    const U8F32: &[DType] = &[DType::U8, DType::F32];
    vec![
        ("normalize_minmax", ANY, ALL, |e| {
            e.normalize(NormalizeMethod::MinMax, DType::F32)
        }),
        ("normalize_zscore", ANY, ALL, |e| {
            e.normalize(NormalizeMethod::ZScore, DType::F32)
        }),
        ("normalize_preset", RGB, U8F32, |e| {
            e.normalize_preset(
                vec![123.7, 116.3, 103.5],
                vec![58.4, 57.1, 57.4],
                DType::F32,
            )
        }),
        ("fused_chain_to_u8", ANY, U8F32, |e| {
            let mut k = FusedKernel::new();
            k.push(ScalarOp::Mul(1.2));
            k.push(ScalarOp::Add(-10.0));
            k.push(ScalarOp::Clamp(0.0, 255.0));
            k.out_dtype = DType::U8;
            e.fused(k)
        }),
        ("blur_1_5", ANY, ALL, |e| e.blur(1.5)),
        ("grayscale", &[3, 4], ALL, |e| e.grayscale()),
        ("resize_lanczos3", ANY, U8F32, |e| {
            e.resize(29, 17, FilterType::Lanczos3)
        }),
        ("resize_lanczos_normalize", RGB, &[DType::U8], |e| {
            e.resize(29, 17, FilterType::Lanczos3)
                .normalize(NormalizeMethod::MinMax, DType::F32)
        }),
        ("gray_blur_normalize", RGB, &[DType::U8], |e| {
            e.grayscale()
                .blur(1.5)
                .normalize(NormalizeMethod::MinMax, DType::F32)
        }),
        ("to_hsv", RGB, U8F32, |e| e.apply_op(color(ColorSpace::Hsv))),
        ("to_lab", RGB, U8F32, |e| e.apply_op(color(ColorSpace::Lab))),
        ("to_ycbcr", RGB, U8F32, |e| {
            e.apply_op(color(ColorSpace::YCbCr))
        }),
    ]
}

const SHAPES: &[(usize, usize)] = &[(1, 1), (1, 9), (7, 13), (33, 65), (64, 64), (101, 37)];
const VIEWS: &[&str] = &["none", "crop", "flip_h", "flip_v", "transpose"];

fn current() -> String {
    let mut out = String::new();
    for (name, channels, dtypes, build) in ops() {
        for &dtype in dtypes {
            for &c in channels {
                for &(h, w) in SHAPES {
                    for view in VIEWS {
                        let buf = source(h, w, c, dtype, view);
                        let result = exec(buf, build);
                        writeln!(
                            out,
                            "{name} {dtype:?} {h}x{w}x{c} {view} {}",
                            digest(&result)
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
    out
}

#[test]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn hot_kernels_match_their_golden_digests() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let got = current();
    if std::env::var_os("VIEW_BUFFER_BLESS").is_some() {
        std::fs::write(&path, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).expect("the golden fixture exists");
    let mismatched: Vec<String> = got
        .lines()
        .zip(want.lines())
        .filter(|(g, w)| g != w)
        .map(|(g, w)| format!("  got  {g}\n  want {w}"))
        .collect();
    assert_eq!(
        got.lines().count(),
        want.lines().count(),
        "the case list changed; re-bless deliberately"
    );
    assert!(
        mismatched.is_empty(),
        "{} of {} golden kernel outputs drifted:\n{}",
        mismatched.len(),
        want.lines().count(),
        mismatched.join("\n")
    );
}
