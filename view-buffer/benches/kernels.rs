//! Kernel micro-benchmarks: the engine's hot per-image kernels, timed through
//! the public API the plugin itself uses (`ViewExpr` → plan → execute, and
//! `ImageAdapter` for the codecs).
//!
//! These are the gate for a kernel change: compare its kernels here, base vs
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
use view_buffer::ops::filter::{BorderMode, ConvolveOp};
use view_buffer::ops::scalar::{FusedKernel, ScalarOp};
use view_buffer::{
    BinaryOp, ComputeOp, DType, FilterType, ImageAdapter, InterpolationType, Normalization,
    ViewBuffer, ViewDto, ViewExpr,
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
            DType::U16 => copy!(u16),
            DType::F32 => copy!(f32),
            other => panic!("bench inputs are u8, u16 or f32, not {other:?}"),
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
    // The same chain over a solely owned f32 image, written in place.
    bench_sizes(c, "fused_chain_f32", rgb_f32, |b| {
        let mut k = FusedKernel::new();
        k.push(ScalarOp::Mul(1.2));
        k.push(ScalarOp::Add(-10.0));
        k.push(ScalarOp::Clamp(0.0, 255.0));
        exec(b, |e| e.fused(k.clone()))
    });
    bench_sizes(c, "normalize_zscore_f32", rgb_f32, |b| {
        exec(b, |e| e.normalize(Normalization::ZScore, DType::F32))
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
    // A vertical flip reaching grayscale: read where it lies (CR-57).
    bench_sizes(c, "grayscale_u8_flip_v", rgb, |b| {
        exec(b, |e| e.flip(vec![0]).grayscale())
    });
    bench_sizes(c, "grayscale_u8_transpose", rgb, |b| {
        exec(b, |e| e.transpose(vec![1, 0, 2]).grayscale())
    });
    bench_sizes(
        c,
        "grayscale_u16_flip_v",
        |s| {
            let v: Vec<u16> = image_u8(s, 3).iter().map(|&x| u16::from(x) * 257).collect();
            ViewBuffer::from_vec_with_shape(v, vec![s, s, 3])
        },
        |b| exec(b, |e| e.flip(vec![0]).grayscale()),
    );
    // Per-value ops on views: an integer map and a table, into a new buffer.
    bench_sizes(c, "invert_u8_flip_h", rgb, |b| {
        exec(b, |e| e.flip(vec![1]).invert())
    });
    bench_sizes(c, "adjust_gamma_u8_flip_v", rgb, |b| {
        exec(b, |e| e.flip(vec![0]).adjust_gamma(0.7))
    });
    bench_sizes(
        c,
        "threshold_u8_flip_h",
        |s| owned_u8(&image_u8(s, 1), s, 1),
        |b| exec(b, |e| e.flip(vec![1]).threshold(128.0)),
    );
    bench_sizes(c, "resize_224_u8", rgb, |b| {
        exec(b, |e| e.resize(224, 224, FilterType::Triangle))
    });
    // Nearest is the engine's own exact gather, not fir's: down and up.
    bench_sizes(c, "resize_nearest_224_u8", rgb, |b| {
        exec(b, |e| e.resize(224, 224, FilterType::Nearest))
    });
    bench_sizes(c, "resize_nearest_2x_u8", rgb, |b| {
        let s = b.shape()[0] as u32;
        exec(b, |e| e.resize(2 * s, 2 * s, FilterType::Nearest))
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

    let gray_f32 = |s| owned_f32(&image_u8(s, 1), s, 1);
    let rgb_f32 = |s| owned_f32(&image_u8(s, 3), s, 3);
    bench_sizes(c, "blur_sigma2_u8_rgb", rgb, |b| exec(b, |e| e.blur(2.0)));
    bench_sizes(c, "blur_sigma2_u8_gray", gray, |b| exec(b, |e| e.blur(2.0)));
    bench_sizes(c, "blur_sigma2_f32_rgb", rgb_f32, |b| {
        exec(b, |e| e.blur(2.0))
    });
    bench_sizes(c, "erode_k3_x1_u8", gray, |b| exec(b, |e| e.erode(3, 1)));
    bench_sizes(c, "erode_k3_x3_u8", gray, |b| exec(b, |e| e.erode(3, 3)));
    bench_sizes(c, "dilate_k5_x4_u8", gray, |b| exec(b, |e| e.dilate(5, 4)));
    bench_sizes(c, "erode_k3_x3_f32", gray_f32, |b| {
        exec(b, |e| e.erode(3, 3))
    });
    let rotate = |angle: f32, interpolation| {
        move |b| {
            exec(b, |e| {
                e.apply_op(ViewDto::Compute(ComputeOp::Rotate {
                    angle,
                    expand: false,
                    interpolation,
                    border_value: 0.0,
                }))
            })
        }
    };
    let rgba = |s| owned_u8(&image_u8(s, 4), s, 4);
    let bilinear = InterpolationType::Bilinear;
    bench_sizes(c, "rotate_30_bilinear_u8_rgb", rgb, rotate(30.0, bilinear));
    bench_sizes(
        c,
        "rotate_30_bilinear_u8_gray",
        gray,
        rotate(30.0, bilinear),
    );
    bench_sizes(
        c,
        "rotate_30_bilinear_u8_rgba",
        rgba,
        rotate(30.0, bilinear),
    );
    bench_sizes(
        c,
        "rotate_30_bilinear_f32_rgb",
        rgb_f32,
        rotate(30.0, bilinear),
    );
    bench_sizes(
        c,
        "rotate_30_nearest_u8_rgb",
        rgb,
        rotate(30.0, InterpolationType::Nearest),
    );
    bench_sizes(c, "rotate_0_u8_rgb", rgb, rotate(0.0, bilinear));

    // convolve2d accumulates in f32 for these inputs. An f32 input is read at
    // the view's own strides: rows in place where they are packed (crop,
    // vertical flip), a transpose or horizontal flip a gathered row at a time.
    let convolve = |side: usize| {
        ViewDto::Filter(ConvolveOp {
            kernel: (0..side * side).map(|i| (i % 7) as f32 - 3.0).collect(),
            normalize: false,
            border: BorderMode::Reflect,
        })
    };
    for side in [3, 5, 9] {
        bench_sizes(c, &format!("convolve2d_k{side}_f32_rgb"), rgb_f32, |b| {
            exec(b, |e| e.apply_op(convolve(side)))
        });
    }
    bench_sizes(c, "convolve2d_k3_f32_rgb_crop", rgb_f32, |b| {
        let s = b.shape()[0];
        exec(b, |e| {
            e.crop(vec![8, 8, 0], vec![s - 8, s - 8, 3])
                .apply_op(convolve(3))
        })
    });
    bench_sizes(c, "convolve2d_k3_f32_rgb_flip_v", rgb_f32, |b| {
        exec(b, |e| e.flip(vec![0]).apply_op(convolve(3)))
    });
    bench_sizes(c, "convolve2d_k3_f32_rgb_flip_h", rgb_f32, |b| {
        exec(b, |e| e.flip(vec![1]).apply_op(convolve(3)))
    });
    bench_sizes(c, "convolve2d_k3_f32_rgb_transpose", rgb_f32, |b| {
        exec(b, |e| e.transpose(vec![1, 0, 2]).apply_op(convolve(3)))
    });
    // A u8 input is converted to f32 first, which reads any view once.
    bench_sizes(c, "convolve2d_k3_u8_rgb", rgb, |b| {
        exec(b, |e| e.apply_op(convolve(3)))
    });
    bench_sizes(c, "convolve2d_k3_u8_rgb_transpose", rgb, |b| {
        exec(b, |e| e.transpose(vec![1, 0, 2]).apply_op(convolve(3)))
    });
}

/// Time `op` on the operand pair `make` builds, at every size. Binary ops
/// borrow their operands, so one pair serves every iteration.
fn bench_pairs(
    c: &mut Criterion,
    name: &str,
    op: BinaryOp,
    make: impl Fn(usize) -> (ViewBuffer, ViewBuffer),
) {
    let mut group = c.benchmark_group(name);
    for size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let (x, y) = make(size);
            b.iter(|| black_box(op.execute(&x, &y)));
        });
    }
    group.finish();
}

fn owned_u16(data: &[u8], size: usize, channels: usize) -> ViewBuffer {
    let v: Vec<u16> = data.iter().map(|&x| u16::from(x) * 257).collect();
    ViewBuffer::from_vec_with_shape(v, vec![size, size, channels])
}

/// Binary ops (`zip_with`): two operands of one dtype and shape are zipped
/// slice against slice or row against row; otherwise each is read a block
/// at a time, a packed run at once. A transpose or horizontal flip is
/// packed first.
fn binary_kernels(c: &mut Criterion) {
    let rgb = |s| owned_u8(&image_u8(s, 3), s, 3);
    let rgb_u16 = |s| owned_u16(&image_u8(s, 3), s, 3);
    let rgb_f32 = |s| owned_f32(&image_u8(s, 3), s, 3);
    let add = BinaryOp::Add;
    let blend = BinaryOp::Blend;

    bench_pairs(c, "add_u8_rgb", add, |s| (rgb(s), rgb(s)));
    bench_pairs(c, "add_u8_rgb_crop", add, |s| {
        let big = rgb(s + 16);
        let crop = big.slice(&[8, 8, 0], &[s + 8, s + 8, 3]);
        (crop.clone(), crop)
    });
    bench_pairs(c, "add_u8_u16_rgb", add, |s| (rgb(s), rgb_u16(s)));
    bench_pairs(c, "add_u8_flip_v_u16_rgb", add, |s| {
        (rgb(s).flip(&[0]), rgb_u16(s))
    });
    bench_pairs(c, "add_u8_transpose_u16_rgb", add, |s| {
        (rgb(s).permute(&[1, 0, 2]), rgb_u16(s))
    });
    // Broadcasts: a [h, w, 1] mask reads one element per run, a [3] channel
    // vector and a [h, 1, 3] column one pixel per run.
    bench_pairs(c, "blend_f32_rgb_mask", blend, |s| {
        (rgb_f32(s), owned_f32(&image_u8(s, 1), s, 1))
    });
    bench_pairs(c, "blend_f32_rgb_mask_crop", blend, |s| {
        let image = rgb_f32(s + 16).slice(&[8, 8, 0], &[s + 8, s + 8, 3]);
        let mask = owned_f32(&image_u8(s + 16, 1), s + 16, 1);
        (image, mask.slice(&[8, 8, 0], &[s + 8, s + 8, 1]))
    });
    bench_pairs(c, "blend_f32_rgb_channels", blend, |s| {
        let channels = ViewBuffer::from_vec_with_shape(vec![0.2f32, 0.5, 0.9], vec![3]);
        (rgb_f32(s), channels)
    });
    bench_pairs(c, "blend_f32_rgb_column", blend, |s| {
        let column = rgb_f32(s).slice(&[0, 0, 0], &[s, 1, 3]).to_contiguous();
        (rgb_f32(s), column)
    });
    // Contiguous operands of rank 1 and 4: read in place, never packed.
    bench_pairs(c, "add_u8_rank1", add, |s| {
        let v = ViewBuffer::from_vec(image_u8(s, 3));
        (v.clone(), v)
    });
    bench_pairs(c, "add_f32_rank4", add, |s| {
        // Four [s/2, s/2, 3] images: as many elements as one [s, s, 3].
        let h = s / 2;
        let image: Vec<f32> = image_u8(h, 3).iter().map(|&v| f32::from(v)).collect();
        let batch = ViewBuffer::from_vec_with_shape(image.repeat(4), vec![4, h, h, 3]);
        (batch.clone(), batch)
    });
    // A tall, narrow view: a block spans many short rows.
    let mut group = c.benchmark_group("add_u8_flip_v_u16_tall");
    group.bench_function("65536x4", |b| {
        let data = image_u8(512, 3);
        let x = ViewBuffer::from_vec_with_shape(data.clone(), vec![65536, 4, 3]).flip(&[0]);
        let y = ViewBuffer::from_vec_with_shape(
            data.iter().map(|&v| u16::from(v) * 257).collect::<Vec<_>>(),
            vec![65536, 4, 3],
        );
        b.iter(|| black_box(add.execute(&x, &y)));
    });
    group.finish();
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

/// The half-precision tensor sink's conversion (`sink("numpy", dtype="f16")`).
/// `f16_per_element_*` is the plugin's former row conversion, verbatim (cast
/// to f32, pack, then `f16::from_f32` and a 2-byte `extend_from_slice` per
/// element); `f16_bits_*` is `ViewBuffer::to_f16_bits`, which replaced it.
fn sink_kernels(c: &mut Criterion) {
    fn per_element(buffer: ViewBuffer) -> Vec<u8> {
        let f32_buf = buffer.cast(DType::F32).to_contiguous();
        let f32_slice = f32_buf.as_slice::<f32>();
        let mut bytes: Vec<u8> = Vec::with_capacity(f32_slice.len() * 2);
        for &v in f32_slice {
            bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        bytes
    }
    let rgb = |s| owned_u8(&image_u8(s, 3), s, 3);
    let rgb_f32 = |s| owned_f32(&image_u8(s, 3), s, 3);
    for (name, make) in [
        ("f32", &rgb_f32 as &dyn Fn(usize) -> ViewBuffer),
        ("u8", &rgb),
    ] {
        bench_sizes(c, &format!("f16_per_element_{name}"), make, |b| {
            ViewBuffer::from_vec(per_element(b))
        });
        bench_sizes(c, &format!("f16_bits_{name}"), make, |b| b.to_f16_bits());
    }
    bench_sizes(c, "f16_per_element_f32_transposed", rgb_f32, |b| {
        ViewBuffer::from_vec(per_element(b.permute(&[1, 0, 2])))
    });
    bench_sizes(c, "f16_bits_f32_transposed", rgb_f32, |b| {
        b.permute(&[1, 0, 2]).to_f16_bits()
    });
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
    targets = color_kernels, value_kernels, layout_kernels, spatial_kernels, binary_kernels, codec_kernels, sink_kernels
}
criterion_main!(benches);
