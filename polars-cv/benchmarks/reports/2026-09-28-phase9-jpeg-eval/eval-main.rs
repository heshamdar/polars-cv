//! JPEG-encoder evaluation (performance plan, phase 9).
//!
//! Compares `image::codecs::jpeg::JpegEncoder` (production, through
//! `ImageAdapter::encode_jpeg`) with `jpeg-encoder` (`simd`) on a
//! synthesized corpus: gradient (smooth, like decoded photos' large areas)
//! and noise, at 256²/512²/1024², L8/Rgb8, quality 75 and 90.
//!
//! Gate (the owner's decision): adopt only if the geomean speedup is at least
//! 1.5x with decode-back PSNR against the source within 0.5 dB of the
//! production encoder's in every cell. jpeg-encoder runs 4:4:4, the
//! production encoder's sampling, so outputs are like for like; its default
//! 4:2:0 is timed for information only.
//!
//! Run: `cargo run --release` (RUSTFLAGS="-C target-cpu=x86-64" for the
//! wheels' baseline target).

use image::ImageEncoder;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Pattern {
    Gradient,
    Noise,
}

#[derive(Clone, Copy, PartialEq)]
enum Color {
    L8,
    Rgb8,
}

fn synth(size: usize, pattern: Pattern, color: Color) -> Vec<u8> {
    let channels = if color == Color::Rgb8 { 3 } else { 1 };
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut data = Vec::with_capacity(size * size * channels);
    for y in 0..size {
        for x in 0..size {
            for c in 0..channels {
                data.push(match pattern {
                    // A gradient with mild noise, per channel offset.
                    Pattern::Gradient => {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        let noise = ((state >> 59) & 0x7) as usize;
                        (((x + y + 37 * c) * 255) / (2 * size) + noise).min(255) as u8
                    }
                    Pattern::Noise => {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        (state >> 56) as u8
                    }
                });
            }
        }
    }
    data
}

fn encode_image(data: &[u8], size: usize, color: Color, q: u8) -> Vec<u8> {
    let mut out = Vec::new();
    let ct = match color {
        Color::L8 => image::ExtendedColorType::L8,
        Color::Rgb8 => image::ExtendedColorType::Rgb8,
    };
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, q)
        .write_image(data, size as u32, size as u32, ct)
        .unwrap();
    out
}

fn encode_jpeg_encoder(
    data: &[u8],
    size: usize,
    color: Color,
    q: u8,
    sampling: jpeg_encoder::SamplingFactor,
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = jpeg_encoder::Encoder::new(&mut out, q);
    enc.set_sampling_factor(sampling);
    let ct = match color {
        Color::L8 => jpeg_encoder::ColorType::Luma,
        Color::Rgb8 => jpeg_encoder::ColorType::Rgb,
    };
    enc.encode(data, size as u16, size as u16, ct).unwrap();
    out
}

fn psnr(source: &[u8], jpeg: &[u8]) -> f64 {
    let decoded = image::load_from_memory(jpeg).unwrap().into_bytes();
    assert_eq!(decoded.len(), source.len());
    let mse = source
        .iter()
        .zip(&decoded)
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum::<f64>()
        / source.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn time_ms(n: usize, mut f: impl FnMut() -> Vec<u8>) -> f64 {
    // One warm-up, then the median of `n`.
    std::hint::black_box(f());
    let mut times: Vec<f64> = (0..n)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[n / 2]
}

fn main() {
    println!(
        "| pattern | color | size | q | image ms | jpeg-encoder 4:4:4 ms | speedup | 4:2:0 ms | PSNR image | PSNR 4:4:4 | Δ dB | bytes image | bytes 4:4:4 |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let mut speedups = Vec::new();
    let mut worst_delta = 0.0f64;
    for pattern in [Pattern::Gradient, Pattern::Noise] {
        for color in [Color::L8, Color::Rgb8] {
            for size in [256usize, 512, 1024] {
                let data = synth(size, pattern, color);
                for q in [75u8, 90] {
                    let n = if size >= 1024 { 15 } else { 41 };
                    let a = encode_image(&data, size, color, q);
                    let b = encode_jpeg_encoder(
                        &data,
                        size,
                        color,
                        q,
                        jpeg_encoder::SamplingFactor::F_1_1,
                    );
                    let (pa, pb) = (psnr(&data, &a), psnr(&data, &b));
                    let delta = (pa - pb).abs();
                    worst_delta = worst_delta.max(delta);
                    let ta = time_ms(n, || encode_image(&data, size, color, q));
                    let tb = time_ms(n, || {
                        encode_jpeg_encoder(
                            &data,
                            size,
                            color,
                            q,
                            jpeg_encoder::SamplingFactor::F_1_1,
                        )
                    });
                    let tc = time_ms(n, || {
                        encode_jpeg_encoder(
                            &data,
                            size,
                            color,
                            q,
                            jpeg_encoder::SamplingFactor::F_2_2,
                        )
                    });
                    speedups.push(ta / tb);
                    println!(
                        "| {} | {} | {size} | {q} | {ta:.2} | {tb:.2} | {:.2}x | {tc:.2} | {pa:.2} | {pb:.2} | {delta:.2} | {} | {} |",
                        match pattern {
                            Pattern::Gradient => "gradient",
                            Pattern::Noise => "noise",
                        },
                        match color {
                            Color::L8 => "L8",
                            Color::Rgb8 => "Rgb8",
                        },
                        ta / tb,
                        a.len(),
                        b.len(),
                    );
                }
            }
        }
    }
    let geomean = (speedups.iter().map(|s| s.ln()).sum::<f64>() / speedups.len() as f64).exp();
    println!(
        "\ngeomean speedup (4:4:4, like for like): {geomean:.2}x; gate >= 1.50x. \
         worst |ΔPSNR|: {worst_delta:.2} dB; gate <= 0.50 dB."
    );
    println!(
        "verdict: {}",
        if geomean >= 1.5 && worst_delta <= 0.5 {
            "ADOPT"
        } else {
            "DO NOT ADOPT"
        }
    );
}
