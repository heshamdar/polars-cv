//! Kernel micro-benchmarks: the engine's hot per-image kernels, timed through
//! the public API the plugin itself uses (`ViewExpr` → plan → execute, and
//! `ImageAdapter` for the codecs).
//!
//! These are the baseline and the gate for the performance plan
//! (`PERFORMANCE_PLAN.md`): each phase compares its kernels here, base vs
//! head, before the end-to-end regression harness runs. Run it twice to see
//! what runtime CPU dispatch buys over the wheels' baseline target:
//!
//! ```text
//! scripts/with-pyo3-env.sh cargo bench -p view-buffer --all-features --bench kernels
//! RUSTFLAGS="-C target-cpu=x86-64" scripts/with-pyo3-env.sh cargo bench ...   # wheel baseline
//! ```
//!
//! (`.cargo/config.toml` builds locally for `x86-64-v3`; the published wheels
//! clear that, so the plain `x86-64` run is what users get.)
//!
//! Every input is freshly owned per iteration (`iter_batched`), as a decoded
//! image reaching the engine is, so in-place paths are measured as they run.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::hint::black_box;
use std::time::Duration;
use view_buffer::execution::ExecutionPlan;
use view_buffer::ops::scalar::{FusedKernel, ScalarOp};
use view_buffer::{
    ComputeOp, DType, FilterType, ImageAdapter, InterpolationType, Normalization, ViewBuffer,
    ViewDto, ViewExpr,
};

const SIZES: [usize; 3] = [256, 512, 1024];

/// A deterministic `[size, size, channels]` u8 image: a gradient with noise,
/// so codecs see realistic (neither trivial nor incompressible) content.
fn image_u8(size: usize, channels: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut out = Vec::with_capacity(size * size * channels);
    for y in 0..size {
        for x in 0..size {
            for c in 0..channels {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = ((state >> 59) as usize) & 0xF;
                out.push((((x + y + 37 * c) * 255) / (2 * size) + noise).min(255) as u8);
            }
        }
    }
    out
}

fn owned_u8(data: &[u8], size: usize, channels: usize) -> ViewBuffer {
    ViewBuffer::from_vec_with_shape(data.to_vec(), vec![size, size, channels])
}

fn owned_f32(data: &[u8], size: usize, channels: usize) -> ViewBuffer {
    let v: Vec<f32> = data.iter().map(|&x| f32::from(x) * 1.003 - 0.4).collect();
    ViewBuffer::from_vec_with_shape(v, vec![size, size, channels])
}

/// Time `run` on a fresh input from `make` at every size.
fn bench_sizes(
    c: &mut Criterion,
    name: &str,
    make: impl Fn(usize) -> ViewBuffer,
    run: impl Fn(ViewBuffer) -> ViewBuffer,
) {
    let mut group = c.benchmark_group(name);
    for size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let input = make(size);
            b.iter_batched(
                || input.clone().to_contiguous_owned(),
                |buf| black_box(run(buf)),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// A deep copy, so each iteration's input has a single owner.
trait OwnedCopy {
    fn to_contiguous_owned(self) -> ViewBuffer;
}

impl OwnedCopy for ViewBuffer {
    fn to_contiguous_owned(self) -> ViewBuffer {
        macro_rules! copy {
            ($t:ty) => {
                ViewBuffer::from_vec_with_shape(
                    self.as_slice::<$t>().to_vec(),
                    self.shape().to_vec(),
                )
            };
        }
        match self.dtype() {
            DType::U8 => copy!(u8),
            DType::F32 => copy!(f32),
            other => panic!("bench inputs are u8 or f32, not {other:?}"),
        }
    }
}

/// Run `build` on `buf` as the plugin's executor does: plan from a clone,
/// drop it, and execute with the source moved in, so a kernel that may write
/// its input in place (the buffer's sole owner) does.
fn exec(
    buf: ViewBuffer,
    build: impl Fn(&std::sync::Arc<ViewExpr>) -> std::sync::Arc<ViewExpr>,
) -> ViewBuffer {
    let steps = build(&ViewExpr::new_source(buf.clone())).plan().steps;
    ExecutionPlan { source: buf, steps }.execute()
}

fn color_kernels(c: &mut Criterion) {
    let rgb = |s| owned_u8(&image_u8(s, 3), s, 3);
    let gray = |s| owned_u8(&image_u8(s, 1), s, 1);

    bench_sizes(c, "grayscale_u8_rgb", rgb, |b| exec(b, |e| e.grayscale()));
    bench_sizes(
        c,
        "grayscale_f32_rgb",
        |s| owned_f32(&image_u8(s, 3), s, 3),
        |b| exec(b, |e| e.grayscale()),
    );
    bench_sizes(c, "threshold_u8", gray, |b| exec(b, |e| e.threshold(128.0)));
    // A strided (cropped) input: the non-contiguous threshold path.
    bench_sizes(c, "threshold_u8_cropped", gray, |b| {
        let s = b.shape()[0];
        exec(b, |e| {
            e.crop(vec![1, 1, 0], vec![s - 1, s - 1, 1])
                .threshold(128.0)
        })
    });
}

fn value_kernels(c: &mut Criterion) {
    let rgb = |s| owned_u8(&image_u8(s, 3), s, 3);
    let rgb_f32 = |s| owned_f32(&image_u8(s, 3), s, 3);

    bench_sizes(c, "cast_f32_to_u8", rgb_f32, |b| {
        exec(b, |e| e.cast(DType::U8))
    });
    bench_sizes(c, "cast_u8_to_f32", rgb, |b| {
        exec(b, |e| e.cast(DType::F32))
    });
    bench_sizes(c, "invert_u8", rgb, |b| exec(b, |e| e.invert()));
    bench_sizes(c, "adjust_gamma_u8", rgb, |b| {
        exec(b, |e| e.adjust_gamma(0.7))
    });
    bench_sizes(c, "normalize_preset_u8_to_f32", rgb, |b| {
        exec(b, |e| {
            e.normalize(
                Normalization::Preset {
                    mean: vec![123.7, 116.3, 103.5],
                    std: vec![58.4, 57.1, 57.4],
                },
                DType::F32,
            )
        })
    });
    bench_sizes(c, "normalize_zscore_u8", rgb, |b| {
        exec(b, |e| e.normalize(Normalization::ZScore, DType::F32))
    });
    bench_sizes(c, "adjust_contrast_u8", rgb, |b| {
        exec(b, |e| e.adjust_contrast(1.4))
    });
    // A standalone float-promoting op: u8 in, f32 out.
    bench_sizes(c, "scale_u8", rgb, |b| exec(b, |e| e.scale(0.5)));
    // The per-channel float path (no lookup table for f32 input).
    bench_sizes(c, "normalize_preset_f32", rgb_f32, |b| {
        exec(b, |e| {
            e.normalize(
                Normalization::Preset {
                    mean: vec![123.7, 116.3, 103.5],
                    std: vec![58.4, 57.1, 57.4],
                },
                DType::F32,
            )
        })
    });
    // cast → scale → clamp → cast back: the fused-kernel path with a
    // float→int output conversion.
    bench_sizes(c, "fused_chain_u8", rgb, |b| {
        let mut k = FusedKernel::new();
        k.push(ScalarOp::Mul(1.2));
        k.push(ScalarOp::Add(-10.0));
        k.push(ScalarOp::Clamp(0.0, 255.0));
        k.out_dtype = DType::U8;
        exec(b, |e| e.fused(k.clone()))
    });
}

fn layout_kernels(c: &mut Criterion) {
    let rgb = |s| owned_u8(&image_u8(s, 3), s, 3);

    // View ops are free until something materialises them; these time the
    // materialisation every sink and non-strided kernel pays.
    bench_sizes(c, "materialize_flip_h_u8", rgb, |b| {
        b.flip(&[1]).to_contiguous()
    });
    bench_sizes(c, "materialize_flip_v_u8", rgb, |b| {
        b.flip(&[0]).to_contiguous()
    });
    bench_sizes(c, "materialize_transpose_u8", rgb, |b| {
        b.permute(&[1, 0, 2]).to_contiguous()
    });
    // A cast of a view: converted straight from its runs, no packed copy.
    bench_sizes(c, "cast_u8_to_f32_flip_h", rgb, |b| {
        exec(b, |e| e.flip(vec![1]).cast(DType::F32))
    });
    // A per-value op on a transposed float view: the engine's strided read.
    bench_sizes(
        c,
        "scale_f32_transposed",
        |s| owned_f32(&image_u8(s, 3), s, 3),
        |b| exec(b, |e| e.transpose(vec![1, 0, 2]).scale(0.5)),
    );
    bench_sizes(c, "grayscale_u8_flip_h", rgb, |b| {
        exec(b, |e| e.flip(vec![1]).grayscale())
    });
    bench_sizes(c, "resize_224_u8", rgb, |b| {
        exec(b, |e| e.resize(224, 224, FilterType::Triangle))
    });
    bench_sizes(c, "crop_then_resize_224_u8", rgb, |b| {
        let s = b.shape()[0];
        exec(b, |e| {
            e.crop(vec![s / 8, s / 8, 0], vec![s - s / 8, s - s / 8, 3])
                .resize(224, 224, FilterType::Triangle)
        })
    });
    // The same crop handed to resize already packed: the difference from
    // `crop_then_resize_224_u8` is what packing the view costs.
    bench_sizes(
        c,
        "packed_crop_resize_224_u8",
        |s| {
            rgb(s)
                .slice(&[s / 8, s / 8, 0], &[s - s / 8, s - s / 8, 3])
                .to_contiguous()
        },
        |b| exec(b, |e| e.resize(224, 224, FilterType::Triangle)),
    );
    // A vertical flip reaching resize: against `resize_224_u8`, the pack.
    bench_sizes(c, "flip_v_then_resize_224_u8", rgb, |b| {
        exec(b, |e| {
            e.flip(vec![0]).resize(224, 224, FilterType::Triangle)
        })
    });
}

fn spatial_kernels(c: &mut Criterion) {
    let rgb = |s| owned_u8(&image_u8(s, 3), s, 3);
    let gray = |s| owned_u8(&image_u8(s, 1), s, 1);

    bench_sizes(c, "blur_sigma2_u8_rgb", rgb, |b| exec(b, |e| e.blur(2.0)));
    bench_sizes(c, "erode_k3_x3_u8", gray, |b| exec(b, |e| e.erode(3, 3)));
    bench_sizes(c, "rotate_30_bilinear_u8_rgb", rgb, |b| {
        exec(b, |e| {
            e.apply_op(ViewDto::Compute(ComputeOp::Rotate {
                angle: 30.0,
                expand: false,
                interpolation: InterpolationType::Bilinear,
                border_value: 0.0,
            }))
        })
    });
}

fn codec_kernels(c: &mut Criterion) {
    for (name, quality) in [
        ("encode_jpeg_q90_rgb", Some(90u8)),
        ("encode_png_rgb", None),
    ] {
        let mut group = c.benchmark_group(name);
        for size in SIZES {
            let buf = owned_u8(&image_u8(size, 3), size, 3);
            group.bench_with_input(BenchmarkId::from_parameter(size), &buf, |b, buf| {
                b.iter(|| match quality {
                    Some(q) => black_box(ImageAdapter::encode_jpeg(buf, q).unwrap()),
                    None => black_box(ImageAdapter::encode(buf, image::ImageFormat::Png).unwrap()),
                });
            });
        }
        group.finish();
    }
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = config();
    targets = color_kernels, value_kernels, layout_kernels, spatial_kernels, codec_kernels
}
criterion_main!(benches);
