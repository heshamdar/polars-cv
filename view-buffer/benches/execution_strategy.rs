//! Criterion benchmarks comparing FullImage vs Tiled execution strategies.
//!
//! Tests the hypothesis that segment-level outer-loop tiling improves throughput
//! for large images by keeping tiles cache-resident across all ops in a segment,
//! compared to the full-image path which flushes the entire buffer to DRAM between
//! every operation.
//!
//! Run with:
//!   cargo bench -p view-buffer --bench execution_strategy --all-features
//!   cargo bench -p view-buffer --bench execution_strategy --all-features -- "1024x1024"

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use view_buffer::{with_execution_strategy, ExecutionStrategy, ViewBuffer, ViewExpr};

// ── Image generation ──────────────────────────────────────────────────────────

/// Build a deterministic RGB u8 image of the given dimensions.
fn make_rgb(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<u8> = (0..h * w * 3)
        .map(|i| ((i.wrapping_mul(2654435769)) >> 24) as u8)
        .collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w, 3])
}

/// Build a deterministic f32 image in [0, 1].
fn make_f32(h: usize, w: usize) -> ViewBuffer {
    let data: Vec<f32> = (0..h * w)
        .map(|i| ((i.wrapping_mul(2654435769) >> 24) as f32) / 255.0)
        .collect();
    ViewBuffer::from_vec(data).reshape(vec![h, w])
}

// ── Strategies ────────────────────────────────────────────────────────────────

fn full_image() -> ExecutionStrategy {
    ExecutionStrategy::FullImage
}

/// Tiling always active (threshold_bytes = 0 ⇒ tile every image regardless of size).
fn always_tiled() -> ExecutionStrategy {
    ExecutionStrategy::Tiled {
        tile_size: 256,
        threshold_bytes: 0,
    }
}

// ── Pipeline runners ──────────────────────────────────────────────────────────

fn run_scale(buf: ViewBuffer, strategy: ExecutionStrategy) -> ViewBuffer {
    with_execution_strategy(strategy, || {
        ViewExpr::new_source(buf).scale(2.0).plan().execute()
    })
}

fn run_scale_relu_clamp(buf: ViewBuffer, strategy: ExecutionStrategy) -> ViewBuffer {
    with_execution_strategy(strategy, || {
        ViewExpr::new_source(buf)
            .scale(0.5)
            .relu()
            .clamp(0.0, 1.0)
            .plan()
            .execute()
    })
}

fn run_grayscale(buf: ViewBuffer, strategy: ExecutionStrategy) -> ViewBuffer {
    with_execution_strategy(strategy, || {
        ViewExpr::new_source(buf).grayscale().plan().execute()
    })
}

fn run_blur(buf: ViewBuffer, strategy: ExecutionStrategy) -> ViewBuffer {
    with_execution_strategy(strategy, || {
        ViewExpr::new_source(buf).blur(2.0).plan().execute()
    })
}

fn run_mixed_segment(buf: ViewBuffer, strategy: ExecutionStrategy) -> ViewBuffer {
    // All ops are tileable → one segment: grayscale + blur + threshold
    with_execution_strategy(strategy, || {
        ViewExpr::new_source(buf)
            .grayscale()
            .blur(2.0)
            .threshold(128.0)
            .plan()
            .execute()
    })
}

fn run_global_barrier(buf: ViewBuffer, strategy: ExecutionStrategy) -> ViewBuffer {
    // resize is Global → breaks the segment: [grayscale] | resize | [scale]
    with_execution_strategy(strategy, || {
        ViewExpr::new_source(buf)
            .grayscale()
            .resize(256, 256, view_buffer::FilterType::Nearest)
            .scale(0.5)
            .plan()
            .execute()
    })
}

fn run_long_pipeline(buf: ViewBuffer, strategy: ExecutionStrategy) -> ViewBuffer {
    // Long tileable segment: scale → relu → grayscale → blur → clamp
    with_execution_strategy(strategy, || {
        ViewExpr::new_source(buf)
            .scale(0.5)
            .relu()
            .grayscale()
            .blur(1.5)
            .clamp(0.0, 1.0)
            .plan()
            .execute()
    })
}

// ── Benchmark groups ──────────────────────────────────────────────────────────

fn bench_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_single_op");

    for &(h, w) in &[(128usize, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let label = format!("{h}x{w}");
        let bytes = (h * w) as u64; // f32 bench uses 1 channel
        group.throughput(Throughput::Bytes(bytes * 4)); // f32 = 4 bytes

        let buf = make_f32(h, w);

        group.bench_with_input(
            BenchmarkId::new("full_image", &label),
            &label,
            |b, _| b.iter(|| run_scale(black_box(buf.clone()), full_image())),
        );
        group.bench_with_input(
            BenchmarkId::new("tiled", &label),
            &label,
            |b, _| b.iter(|| run_scale(black_box(buf.clone()), always_tiled())),
        );
    }
    group.finish();
}

fn bench_scale_relu_clamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_relu_clamp_3ops");

    for &(h, w) in &[(128usize, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let label = format!("{h}x{w}");
        group.throughput(Throughput::Bytes((h * w * 4) as u64));

        let buf = make_f32(h, w);

        group.bench_with_input(
            BenchmarkId::new("full_image", &label),
            &label,
            |b, _| b.iter(|| run_scale_relu_clamp(black_box(buf.clone()), full_image())),
        );
        group.bench_with_input(
            BenchmarkId::new("tiled", &label),
            &label,
            |b, _| b.iter(|| run_scale_relu_clamp(black_box(buf.clone()), always_tiled())),
        );
    }
    group.finish();
}

fn bench_grayscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("grayscale");

    for &(h, w) in &[(128usize, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let label = format!("{h}x{w}");
        group.throughput(Throughput::Bytes((h * w * 3) as u64));

        let buf = make_rgb(h, w);

        group.bench_with_input(
            BenchmarkId::new("full_image", &label),
            &label,
            |b, _| b.iter(|| run_grayscale(black_box(buf.clone()), full_image())),
        );
        group.bench_with_input(
            BenchmarkId::new("tiled", &label),
            &label,
            |b, _| b.iter(|| run_grayscale(black_box(buf.clone()), always_tiled())),
        );
    }
    group.finish();
}

fn bench_blur(c: &mut Criterion) {
    let mut group = c.benchmark_group("blur_sigma2");
    // Blur has halo=6 — largest per-op overhead for tiling, but biggest cache benefit.

    for &(h, w) in &[(128usize, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let label = format!("{h}x{w}");
        group.throughput(Throughput::Bytes((h * w * 3) as u64));

        let buf = make_rgb(h, w);

        group.bench_with_input(
            BenchmarkId::new("full_image", &label),
            &label,
            |b, _| b.iter(|| run_blur(black_box(buf.clone()), full_image())),
        );
        group.bench_with_input(
            BenchmarkId::new("tiled", &label),
            &label,
            |b, _| b.iter(|| run_blur(black_box(buf.clone()), always_tiled())),
        );
    }
    group.finish();
}

fn bench_mixed_segment(c: &mut Criterion) {
    let mut group = c.benchmark_group("grayscale_blur_threshold_segment");
    // The key benchmark: all three ops form one tileable segment.
    // Tiling should show its biggest advantage here at large sizes.

    for &(h, w) in &[(128usize, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let label = format!("{h}x{w}");
        group.throughput(Throughput::Bytes((h * w * 3) as u64));

        let buf = make_rgb(h, w);

        group.bench_with_input(
            BenchmarkId::new("full_image", &label),
            &label,
            |b, _| b.iter(|| run_mixed_segment(black_box(buf.clone()), full_image())),
        );
        group.bench_with_input(
            BenchmarkId::new("tiled", &label),
            &label,
            |b, _| b.iter(|| run_mixed_segment(black_box(buf.clone()), always_tiled())),
        );
    }
    group.finish();
}

fn bench_global_barrier(c: &mut Criterion) {
    let mut group = c.benchmark_group("global_barrier_pipeline");
    // Resize breaks the segment — only the grayscale and scale portions tile.
    // Tiling benefit is smaller here because resize dominates.

    for &(h, w) in &[(256usize, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let label = format!("{h}x{w}");
        group.throughput(Throughput::Bytes((h * w * 3) as u64));

        let buf = make_rgb(h, w);

        group.bench_with_input(
            BenchmarkId::new("full_image", &label),
            &label,
            |b, _| b.iter(|| run_global_barrier(black_box(buf.clone()), full_image())),
        );
        group.bench_with_input(
            BenchmarkId::new("tiled", &label),
            &label,
            |b, _| b.iter(|| run_global_barrier(black_box(buf.clone()), always_tiled())),
        );
    }
    group.finish();
}

fn bench_long_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("long_pipeline_5ops");
    // scale → relu → grayscale → blur → clamp: one big tileable segment.

    for &(h, w) in &[(128usize, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let label = format!("{h}x{w}");
        group.throughput(Throughput::Bytes((h * w * 3) as u64));

        let buf = make_rgb(h, w);

        group.bench_with_input(
            BenchmarkId::new("full_image", &label),
            &label,
            |b, _| b.iter(|| run_long_pipeline(black_box(buf.clone()), full_image())),
        );
        group.bench_with_input(
            BenchmarkId::new("tiled", &label),
            &label,
            |b, _| b.iter(|| run_long_pipeline(black_box(buf.clone()), always_tiled())),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_scale,
    bench_scale_relu_clamp,
    bench_grayscale,
    bench_blur,
    bench_mixed_segment,
    bench_global_barrier,
    bench_long_pipeline,
);
criterion_main!(benches);
